"""站点端：把一个站点上已完成的算例逐个「导出 → 回传 → 清理」的常驻循环。

租期有限、租用服务器磁盘又紧，所以导出不能等到整批跑完：每个算例一 `End` 就把它
的末态压成 VTK 原始数据包，回传集群 `~/dataset_export/`，校验字节数一致后再把
**该算例**的 `processor*`、中间时刻与 `postProcessing` 删掉腾空间。

顺序与幂等
----------
每个算例在状态文件里记三个阶段，任何一步失败只重做那一步，不会重复导出或重复删：

    exported   导出包已生成且文件齐全（internal.vtu / hull.vtp / case_manifest.json）
    returned   导出包已 rsync 到集群，且远端逐文件字节数与本地一致
    pruned     processor* / 中间时刻 / postProcessing 已删除（只在 returned 之后）

清理只删三样东西，其余一律保留：`log.interFoam`、`constant/polyMesh`、`system`、
`0.orig`、`0`、末态时刻目录、`logs/`、`case_manifest.json`、导出包本身。

用法
----
租用服务器（在站点上跑，回传集群）：

    python3 export_return_loop.py --site rented-b --root /data/cases \
        --export-root ~/dataset_export --return-to cluster \
        --openfoam-bashrc /usr/lib/openfoam/openfoam2506/etc/bashrc

集群（只导出不搬运，--return-to 留空）：

    python3 export_return_loop.py --site cluster --root <数据集仓库> --export-root ~/dataset_export

默认一直循环（每 --poll 秒重扫一次，新完成的算例自动进入队列）；`--once` 只扫一轮，
`--dry-run` 只打印将要做什么。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 导出包里必须齐全的文件；`<hull>_Hull.stl.gz` 与 `kcs_reference.json` 只作文档与复核，
# 不参与组装，缺了不算失败
REQUIRED = ("internal.vtu", "hull.vtp", "case_manifest.json")
MIN_BYTES = {"internal.vtu": 10_000_000, "hull.vtp": 100_000, "case_manifest.json": 500}

# 清理时保留的时刻目录：初始时刻与末态时刻（末态由 log.interFoam 的最后一个 Time 给出）
KEEP_TIME = "0"


def tail_text(path: Path, limit: int = 8192) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - limit))
        return handle.read().decode("utf-8", "replace")


def case_state(case: Path) -> str:
    """done / failed / pending。判据与 run_site_queue.py 一致：尾部有 `End` 为完成，有 FATAL 为失败。"""
    log = case / "log.interFoam"
    if not log.is_file():
        return "pending"
    text = tail_text(log)
    if any(line.strip() == "End" for line in text.splitlines()):
        return "done"
    if "FATAL" in text or "signal 8" in text:
        return "failed"
    return "pending"


def final_time(case: Path) -> str | None:
    """log.interFoam 里最后一个 `Time = <t>`，即末态时刻目录名。"""
    log = case / "log.interFoam"
    if not log.is_file():
        return None
    last = None
    for line in tail_text(log, 200_000).splitlines():
        if line.startswith("Time = "):
            last = line.split("=", 1)[1].strip()
    return last


def tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            file = Path(base) / name
            try:
                total += file.lstat().st_size
            except OSError:
                pass
    return total


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def ledger(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a") as handle:
        if new:
            handle.write("\t".join(("time", "site", "case", "internal_vtu", "hull_vtp",
                                    "manifest", "stl", "returned_bytes", "freed_bytes",
                                    "stage", "note")) + "\n")
        handle.write("\t".join(str(row.get(key, "")) for key in
                               ("time", "site", "case", "internal_vtu", "hull_vtp",
                                "manifest", "stl", "returned_bytes", "freed_bytes",
                                "stage", "note")) + "\n")


def package_sizes(pkg: Path) -> dict:
    sizes = {}
    for name in REQUIRED:
        file = pkg / name
        sizes[name] = file.stat().st_size if file.is_file() else 0
    stl = sorted(pkg.glob("*_Hull.stl.gz"))
    sizes["stl"] = stl[0].stat().st_size if stl else 0
    return sizes


def verify_package(pkg: Path) -> tuple[dict, str | None]:
    """包内必备文件齐全且大小合理；返回 (字节数, 问题描述)。"""
    if not pkg.is_dir():
        return {}, f"导出包不存在: {pkg}"
    sizes = package_sizes(pkg)
    problems = [f"{name} 缺失或过小（{sizes[name]} B）"
                for name, floor in MIN_BYTES.items() if sizes[name] < floor]
    if not sizes["stl"]:
        problems.append("包内没有 *_Hull.stl.gz（只作文档复核，继续）")
    return sizes, ("；".join(problems) if problems else None)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def export_case(script: Path, case: Path, export_root: Path, bashrc: str,
                dry_run: bool) -> tuple[bool, str]:
    env = dict(os.environ, OPENFOAM_BASHRC=bashrc)
    if dry_run:
        print(f"[dry-run] {script} {case} {export_root}")
        return True, "dry-run"
    result = run(["bash", str(script), str(case), str(export_root)], env=env)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()[-500:]
    return True, ""


def remote_sizes(host: str, remote_dir: str, names: list[str]) -> dict | None:
    listing = " ".join(f'"{name}"' for name in names)
    script = (f'cd {remote_dir} 2>/dev/null || exit 2; '
              f'stat -c "%n %s" {listing} 2>/dev/null')
    result = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, script])
    if result.returncode != 0:
        return None
    sizes = {}
    for line in result.stdout.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            sizes[Path(parts[0]).name] = int(parts[1])
    return sizes


def return_package(pkg: Path, host: str, remote_root: str, dry_run: bool) -> tuple[bool, str]:
    """rsync 导出包到集群，再逐文件比对字节数。VTK/ 是 internal.vtu 的副本，不回传。"""
    dest = f"{host}:{remote_root}/{pkg.name}/"
    cmd = ["rsync", "-a", "--partial", "--exclude", "VTK/", "--exclude", ".foamtovtk-*/",
           f"{pkg}/", dest]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return True, "dry-run"
    result = run(cmd)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()[-500:]
    local = {name: (pkg / name).stat().st_size for name in REQUIRED if (pkg / name).is_file()}
    stl = sorted(pkg.glob("*_Hull.stl.gz"))
    if stl:
        local[stl[0].name] = stl[0].stat().st_size
    remote = remote_sizes(host, f"{remote_root}/{pkg.name}", list(local))
    if remote is None:
        return False, "回传后取不到远端文件清单"
    mismatch = [f"{name}: 本地 {size} B / 远端 {remote.get(name)} B"
                for name, size in local.items() if remote.get(name) != size]
    if mismatch:
        return False, "字节数不一致：" + "；".join(mismatch)
    return True, ""


def prune_case(case: Path, dry_run: bool) -> tuple[int, list[str]]:
    """删 processor*、中间时刻与 postProcessing；返回 (释放字节, 删除清单)。

    保留：初始时刻 `0`、`0.orig`、末态时刻目录、`constant/`（含 polyMesh）、`system/`、
    `logs/`、各 `log.*`、`case_manifest.json`。
    """
    keep_times = {KEEP_TIME, final_time(case)}
    removed, freed = [], 0
    targets = sorted(p for p in case.iterdir() if p.is_dir() and p.name.startswith("processor"))
    targets += sorted(p for p in case.iterdir()
                      if p.is_dir() and p.name.isdigit() and p.name not in keep_times)
    targets.append(case / "postProcessing")
    for path in targets:
        if not path.exists():
            continue
        size = tree_bytes(path)
        removed.append(f"{path.name} ({size / 1e9:.2f} GB)")
        freed += size
        if not dry_run:
            shutil.rmtree(path, ignore_errors=True)
    return freed, removed


def prune_package_vtk(pkg: Path, dry_run: bool) -> tuple[int, list[str]]:
    vtk = pkg / "VTK"
    if not vtk.is_dir():
        return 0, []
    size = tree_bytes(vtk)
    if not dry_run:
        shutil.rmtree(vtk, ignore_errors=True)
    return size, [f"{pkg.name}/VTK ({size / 1e9:.2f} GB)"]


def case_dirs(root: Path, only: set[str] | None = None) -> list[Path]:
    cases = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith(("_", ".")):
            continue
        if not (path / "case_manifest.json").is_file():
            continue
        if only and path.name not in only:
            continue
        cases.append(path)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", required=True, help="站点名，只用于状态文件与台账")
    parser.add_argument("--root", required=True, help="算例根目录（站点上的 cases 目录）")
    parser.add_argument("--export-root", required=True, help="导出包目录")
    parser.add_argument("--return-to", default=None,
                        help="回传目标主机（ssh 别名，如 cluster）；留空表示只导出不搬运")
    parser.add_argument("--remote-export-root", default="/path/to/cluster-home/dataset_export")
    parser.add_argument("--export-script", default=str(Path(__file__).with_name("export_case_vtk.sh")))
    parser.add_argument("--openfoam-bashrc", default="/usr/lib/openfoam/openfoam2506/etc/bashrc")
    parser.add_argument("--state-dir", default=None, help="状态与台账目录（缺省工作目录下 logs/）")
    parser.add_argument("--cases", nargs="*", default=None, help="只处理这些算例（缺省全部）")
    parser.add_argument("--max-cases", type=int, default=0, help="本轮最多处理几个算例，0 表示不限")
    parser.add_argument("--poll", type=int, default=600, help="队列空时的重扫间隔秒数")
    parser.add_argument("--gap", type=int, default=20, help="两个算例之间暂停秒数，避免占满 I/O")
    parser.add_argument("--no-prune", action="store_true", help="只导出回传，不清理")
    parser.add_argument("--once", action="store_true", help="只扫一轮就退出")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    export_root = Path(args.export_root).expanduser().resolve()
    state_dir = Path(args.state_dir).expanduser() if args.state_dir else Path.cwd() / "logs"
    state_path = state_dir / f"export_state_{args.site}.json"
    ledger_path = state_dir / f"export-ledger-{args.site}.tsv"
    export_script = Path(args.export_script).expanduser().resolve()
    if not export_script.is_file():
        raise SystemExit(f"找不到导出脚本 {export_script}")
    if args.return_to is None and args.site != "cluster":
        raise SystemExit("站点回传目标为空时只会导出不会搬运，确认用 --site cluster 或给 --return-to")

    state = load_state(state_path)
    only = set(args.cases) if args.cases else None
    print(f"站点 {args.site}：root={root} export_root={export_root} "
          f"return_to={args.return_to or '—'} state={state_path}", flush=True)

    while True:
        done = 0
        queue = []
        for case in case_dirs(root, only):
            entry = state.get(case.name, {})
            if entry.get("stage") == "pruned" or (entry.get("stage") == "returned"
                                                  and args.no_prune):
                continue
            if case_state(case) != "done":
                continue
            queue.append(case)
        # 已导出的先回传（它们已经占了本地磁盘），再导出新完成的
        queue.sort(key=lambda case: (state.get(case.name, {}).get("stage") != "exported",
                                     case.name))

        for case in queue:
            entry = state.setdefault(case.name, {"stage": None, "attempts": 0})
            pkg = export_root / case.name
            stamp = time.strftime("%F %T")
            print(f"\n== {stamp} {case.name}（stage={entry['stage']}）", flush=True)

            if entry["stage"] is None:
                # 之前手工导出的包（老流程）直接采用，不重新跑一遍 foamToVTK
                adopted = verify_package(pkg)[0] if pkg.is_dir() else {}
                if adopted and all(adopted[name] >= floor for name, floor in MIN_BYTES.items()):
                    ok, note, sizes, problem = (True, "已有导出包，直接采用",
                                                package_sizes(pkg), "")
                else:
                    ok, note = export_case(export_script, case, export_root,
                                           args.openfoam_bashrc, args.dry_run)
                    if args.dry_run:
                        print("   [dry-run] 跳过导出校验、回传与清理", flush=True)
                        done += 1
                        continue
                    sizes, problem = verify_package(pkg) if ok else ({}, note)
                sizes = sizes or package_sizes(pkg)
                if not ok or (problem and "缺失或过小" in problem):
                    print(f"   导出失败：{note or problem}", flush=True)
                    entry["attempts"] += 1
                    ledger(ledger_path, {"time": stamp, "site": args.site, "case": case.name,
                                         "stage": "export-failed", "note": (note or problem)[:200]})
                    save_state(state_path, state)
                    continue
                entry.update({"stage": "exported", "sizes": sizes,
                              "exported_at": stamp, "export_note": note})
                save_state(state_path, state)
                print(f"   导出完成{'（' + note + '）' if note else ''} "
                      f"internal.vtu={sizes['internal.vtu'] / 1e6:.1f} MB "
                      f"hull.vtp={sizes['hull.vtp'] / 1e6:.2f} MB "
                      f"stl={sizes['stl'] / 1e6:.1f} MB{('  ' + problem) if problem else ''}",
                      flush=True)
                ledger(ledger_path, {"time": stamp, "site": args.site, "case": case.name,
                                     "internal_vtu": sizes["internal.vtu"],
                                     "hull_vtp": sizes["hull.vtp"],
                                     "manifest": sizes["case_manifest.json"],
                                     "stl": sizes["stl"], "stage": "exported",
                                     "note": problem or ""})

            if entry["stage"] == "exported" and args.return_to:
                ok, note = return_package(pkg, args.return_to, args.remote_export_root,
                                          args.dry_run)
                if args.dry_run:
                    print("   [dry-run] 不写状态、不清理", flush=True)
                    done += 1
                    continue
                if not ok:
                    print(f"   回传失败：{note}", flush=True)
                    ledger(ledger_path, {"time": stamp, "site": args.site, "case": case.name,
                                         "stage": "return-failed", "note": note[:200]})
                    continue
                entry.update({"stage": "returned", "returned_at": time.strftime("%F %T")})
                save_state(state_path, state)
                print(f"   回传完成 → {args.return_to}:{args.remote_export_root}/{case.name} "
                      f"（逐文件字节数一致）", flush=True)
                ledger(ledger_path, {"time": stamp, "site": args.site, "case": case.name,
                                     "internal_vtu": entry["sizes"]["internal.vtu"],
                                     "hull_vtp": entry["sizes"]["hull.vtp"],
                                     "manifest": entry["sizes"]["case_manifest.json"],
                                     "stl": entry["sizes"]["stl"],
                                     "returned_bytes": sum(entry["sizes"].values()),
                                     "stage": "returned"})
            elif entry["stage"] == "exported" and not args.return_to:
                entry["stage"] = "returned"   # 集群上导出即到位，没有搬运步骤
                entry["returned_at"] = stamp
                save_state(state_path, state)

            if entry["stage"] == "returned" and not args.no_prune:
                freed, removed = prune_case(case, args.dry_run)
                freed_pkg, removed_pkg = prune_package_vtk(pkg, args.dry_run)
                freed += freed_pkg
                removed += removed_pkg
                if args.dry_run:
                    print(f"   [dry-run] 可释放 {freed / 1e9:.2f} GB："
                          f"{len(removed)} 项（不删不做记录）", flush=True)
                    done += 1
                    continue
                entry.update({"stage": "pruned", "pruned_at": time.strftime("%F %T"),
                              "freed_bytes": freed})
                save_state(state_path, state)
                print(f"   清理完成，释放 {freed / 1e9:.2f} GB："
                      f"{', '.join(removed) if removed else '无'}", flush=True)
                ledger(ledger_path, {"time": stamp, "site": args.site, "case": case.name,
                                     "internal_vtu": entry["sizes"]["internal.vtu"],
                                     "hull_vtp": entry["sizes"]["hull.vtp"],
                                     "manifest": entry["sizes"]["case_manifest.json"],
                                     "stl": entry["sizes"]["stl"],
                                     "returned_bytes": sum(entry["sizes"].values()),
                                     "freed_bytes": freed, "stage": "pruned",
                                     "note": "; ".join(removed)})

            done += 1
            if args.max_cases and done >= args.max_cases:
                break
            time.sleep(args.gap)

        if args.once or args.dry_run:
            print(f"\n本轮处理 {done} 个算例，退出", flush=True)
            break
        print(f"{time.strftime('%F %T')} 本轮处理 {done} 个算例，{args.poll}s 后重扫", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
