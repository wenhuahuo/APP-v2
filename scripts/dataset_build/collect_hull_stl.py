"""把 11 个船型的船体 STL 收拢到 dataset/hull_stl/ 并生成 manifest.json。

用法:
    python3 scripts/dataset_build/collect_hull_stl.py

船型来源是 NurbsShipGen 的 classic_hulls。多数 STL 是毫米量级、还带多余几何，
与数据集的坐标约定（米制、x∈[0,Lpp]、整船 y 双向、龙骨 z=0）不一致，因此逐个
用 bbox 与设计条目 meta.json 的 Lpp/T 对账，并与 dataset/.build_cache/ 里该算例
实际使用的 `<hull>_Hull.stl.gz` 比对 bbox：一致就复制 NurbsShipGen 版本，不一致
以 .build_cache 版本为准。
"""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]

NURBS_ROOT = Path("/path/to/NurbsShipGen/datasets/classic_hulls")

# 数据集船型名 -> classic_hulls 下的子目录与 STL 文件名（目录名有拼写差异）
NURBS_STL = {
    "kcs": ("KCS", "kcs.stl"),
    "dtc": ("DTC", "canonical.stl"),
    "containership": ("containership", "containership.stl"),
    "series60": ("series60", "series60.stl"),
    "s175": ("s_175_u_water", "s-175-u-water.stl"),
    "kvlcc2": ("KVLCC2", "kvlcc2.stl"),
    "frigate": ("firgate", "firgate.stl"),
    "wigley": ("wigley_hull", "wigley hull.stl"),
    "workboat": ("work_boat", "work boat.stl"),
    "npl_round_bilge": ("npl_round_bilge_full_scale", "NPL Round BilGe Full Scale.stl"),
    "npl_round_bilge_4a": ("npl_round_bilge_4a_model", "NPL Round Bilge 4a Model.stl"),
}

# 数据集船型名 -> 设计条目 id（Lpp 与 T 的来源）
DESIGN_ENTRY = {
    "kcs": "kcs_fr026_normal_mesh",
    "dtc": "dtc_fr022_normal_mesh",
    "containership": "containership_fr022_normal_mesh",
    "series60": "series60_fr022_normal_mesh",
    "s175": "s175_fr029_normal_mesh",
    "kvlcc2": "kvlcc2_fr014_normal_mesh",
    "frigate": "frigate_fr030_normal_mesh",
    "wigley": "wigley_fr030_normal_mesh",
    "workboat": "workboat_fr030_normal_mesh",
    "npl_round_bilge": "npl_round_bilge_fr040_normal_mesh",
    "npl_round_bilge_4a": "npl_round_bilge_4a_fr040_normal_mesh",
}

# 已在数据集侧核实过一致的船型：跳过与 .build_cache 的比对
VERIFIED_HULLS = {"kcs": "NurbsShipGen 的 kcs.stl 与 .build_cache 版本逐面一致"}

LPP_TOL = 0.10  # x 跨度与 Lpp 的相对差上限
BBOX_TOL = 0.01  # 与 .build_cache 版本逐维 bbox 的相对差上限


def read_stl(path):
    """读 STL（支持 .gz），返回顶点数、面数与 bbox。"""
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            mesh = trimesh.load(handle, file_type="stl", force="mesh")
    else:
        mesh = trimesh.load(str(path), force="mesh")
    return {
        "path": str(path),
        "faces": int(len(mesh.faces)),
        "bounds": np.asarray(mesh.bounds, dtype=float),
    }


def span_check(info, lpp, draft):
    """x 跨度与 Lpp 相对差、z 跨度是否盖过吃水；返回 (通过, 说明)。"""
    low, high = info["bounds"]
    dx, dz = high[0] - low[0], high[2] - low[2]
    rel = abs(dx - lpp) / lpp
    ok = rel < LPP_TOL and dz > draft
    return ok, f"dx/Lpp-1={rel:+.3f} dz/T={dz / draft:.2f}"


def bbox_match(info_a, info_b):
    """两版 STL 的 bbox 逐维相对差是否都在 1% 以内；返回 (通过, 三维相对差)。"""
    span_a = info_a["bounds"][1] - info_a["bounds"][0]
    span_b = info_b["bounds"][1] - info_b["bounds"][0]
    rel = np.abs(span_a - span_b) / np.maximum(span_b, 1e-12)
    return bool(np.all(rel < BBOX_TOL)), rel


def build_cache_stl(build_cache, source_case):
    """算例目录里的船体 STL（`<hull>_Hull.stl.gz`，KCS 写作 KCS_Hull）。"""
    case_dir = build_cache / source_case
    found = sorted(case_dir.glob("*.stl.gz"))
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(f"{case_dir} 下有多个船体 STL：{[p.name for p in found]}")
    return found[0]


def collect(dataset_root, out_dir, build_cache, nurbs_root, skip_copy=False):
    records = {}
    rows = []
    for hull, (subdir, name) in NURBS_STL.items():
        meta = json.loads((dataset_root / DESIGN_ENTRY[hull] / "meta.json").read_text())
        lpp, draft = float(meta["Lpp"]), float(meta["T"])
        nurbs = read_stl(nurbs_root / subdir / name)
        nurbs_ok, nurbs_note = span_check(nurbs, lpp, draft)

        cache_path = build_cache_stl(build_cache, meta["source_case"])
        cache = read_stl(cache_path) if cache_path is not None else None
        if hull in VERIFIED_HULLS:
            match, rel = None, "已验证一致"
        elif cache is None:
            match, rel = None, "无 .build_cache 版本"
        else:
            match, rel = bbox_match(nurbs, cache)
            rel = f"{rel.max():.3f}"  # 三个方向里最大的相对差

        # 量级对不上（毫米制、多几何）或以 .build_cache 版本不一致时，用后者
        use_cache = cache is not None and hull not in VERIFIED_HULLS and not (nurbs_ok and match)
        chosen = cache if use_cache else nurbs
        if not span_check(chosen, lpp, draft)[0]:
            raise ValueError(f"{hull} 的选源 STL 与 meta.json 的 Lpp/T 量级不符："
                             f"{chosen['path']} {span_check(chosen, lpp, draft)[1]}")

        target = out_dir / f"{hull}.stl"
        if not skip_copy:
            out_dir.mkdir(parents=True, exist_ok=True)
            if chosen["path"].endswith(".gz"):
                with gzip.open(chosen["path"], "rb") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                shutil.copyfile(chosen["path"], target)
            copied = read_stl(target)
            if not np.allclose(copied["bounds"], chosen["bounds"]) or copied["faces"] != chosen["faces"]:
                raise ValueError(f"{hull} 复制后的 STL 与源不一致：{target}")

        records[hull] = {
            "stl": target.name,
            "source": chosen["path"],
            "Lpp": lpp,
            "T": draft,
            "design_entry": DESIGN_ENTRY[hull],
            "source_case": meta["source_case"],
            "faces": chosen["faces"],
            "bounds": chosen["bounds"].tolist(),
            "nurbs_stl": nurbs["path"],
            "nurbs_scale_ok": bool(nurbs_ok),
            "cache_bbox_match": None if match is None else bool(match),
        }
        rows.append((hull, lpp, draft, nurbs["faces"], chosen["faces"], nurbs_note, rel,
                     "build_cache" if use_cache else "nurbs"))

    print(f"{'船型':<20}{'Lpp':>8}{'T':>8}{'Nurbs面':>10}{'选源面':>9}  "
          f"{'Nurbs量级检查':<26}{'bbox最大相对差':<14}选源")
    for hull, lpp, draft, n0, n1, note, rel, src in rows:
        print(f"{hull:<20}{lpp:>8.3f}{draft:>8.4f}{n0:>10d}{n1:>9d}  {note:<26}{rel:<14}{src}")
    for hull, note in VERIFIED_HULLS.items():
        print(f"※ {hull}: {note}")
    return records


def main():
    parser = argparse.ArgumentParser(description="收集 11 个船型的船体 STL")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--nurbs-root", default=str(NURBS_ROOT))
    parser.add_argument("--skip-copy", action="store_true", help="只打印验证表，不复制")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_dir = dataset_root / "hull_stl"
    records = collect(dataset_root, out_dir, dataset_root / ".build_cache",
                      Path(args.nurbs_root), args.skip_copy)
    if args.skip_copy:
        return
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{len(records)} 个船型写入 {out_dir}，清单 {manifest.name}")
    counts = {}
    for record in records.values():
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    for source, n in sorted(counts.items()):
        print(f"  来源 {source}: {n}")


if __name__ == "__main__":
    sys.exit(main())
