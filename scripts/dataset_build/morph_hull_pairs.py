"""批量融合 11 个经典船型的全部两两组合（55 对 × 3 权重 = 165 个融合船型）。

船型取自 `dataset/classic_hulls/manifest.json`，按字母序组合成 55 对，每对默认生成
w = 0.25 / 0.50 / 0.75 三个权重（w = 0 / 1 就是母船本身，不生成）。产物

    dataset/morphed_hulls/<a>_<b>/<a>_<b>_w<三位权重>.stl
    dataset/morphed_hulls/<a>_<b>/morph_meta.json

由 `src/dataset/hull_morph.py` 的 `write_pair` 写出，与单对入口
`scripts/dataset_build/morph_hulls.py` 是同一份逻辑、同一份 schema。

幂等与增量：

* 默认**不覆盖**已存在的 STL：先回读它并现场重算同一权重做对照（顶点数、面数、
  包围盒），一致就跳过、不一致就报错停下。
* 缺的权重按对补齐（例如 dtc_frigate 只需补 w025 / w075），`morph_meta.json` 里
  已有权重的条目原样保留。
* `--overwrite` 用于几何代码改动后重建全部产物：**先打印将被覆盖的文件清单**，
  再逐个重写（不做重算对照，因为几何本来就要变）。

用法（在仓库根目录）:

    python3 scripts/dataset_build/morph_hull_pairs.py --dry-run      # 只打印计划
    python3 scripts/dataset_build/morph_hull_pairs.py                # 全部 55 对
    python3 scripts/dataset_build/morph_hull_pairs.py --hulls kcs,s175   # 只跑这一对
    python3 scripts/dataset_build/morph_hull_pairs.py --hulls dtc frigate --weights 0.25 0.75
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

from hull_morph import morph_hulls, weight_tag, write_pair  # noqa: E402

# 本轮融合的 11 个经典船型（`dtmb5415` 因基础算例波高越界不进多船型集，不在 manifest 里）
EXPECTED_HULLS = 11
DEFAULT_WEIGHTS = (0.25, 0.50, 0.75)
# 重算对照的包围盒容差（相对 L_ref）
BOUNDS_TOLERANCE = 1e-6


def load_manifest(dataset_root):
    path = Path(dataset_root) / "classic_hulls" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，先跑 scripts/dataset_build/collect_hull_stl.py")
    return json.loads(path.read_text())


def select_hulls(values, available):
    """`--hulls a,b c` -> 按 manifest 顺序排列的子集（缺省是全部）。"""
    if not values:
        return list(available)
    wanted = set()
    for item in values:
        for name in item.split(","):
            name = name.strip()
            if not name:
                continue
            if name not in available:
                raise KeyError(f"{name} 不在 classic_hulls/manifest.json 里，可选 {list(available)}")
            wanted.add(name)
    return [name for name in available if name in wanted]


def verify_existing(stl_path, result):
    """重算结果与已有 STL 的几何对照：顶点数、面数、包围盒。

    STL 是 float32 的三角形汤且顶点顺序在重新导出后可能不同，所以不比逐点坐标；
    顶点数/面数与包围盒一致即认定同一份代码产物。返回 None 表示一致，否则返回
    不一致的原因。
    """
    mesh = trimesh.load(str(stl_path), force="mesh")
    points, faces = result["points"], result["faces"]
    if len(mesh.vertices) != len(points) or len(mesh.faces) != len(faces):
        return (f"顶点/面数 {len(mesh.vertices)}/{len(mesh.faces)} 与重算的 "
                f"{len(points)}/{len(faces)} 不一致")
    scale = max(abs(float(result["meta"]["L_ref_m"])), 1.0)
    diff = float(np.abs(np.asarray(mesh.bounds) - np.asarray(
        [points.min(axis=0), points.max(axis=0)])).max())
    if diff > BOUNDS_TOLERANCE * scale:
        return f"包围盒差 {diff:.3e} m 超过 {BOUNDS_TOLERANCE}·L_ref"
    return None


def main():
    parser = argparse.ArgumentParser(description="11 个经典船型的全部两两融合（55 对 × 3 权重）")
    parser.add_argument("--hulls", nargs="+", default=None,
                        help="船型子集（逗号或空格分隔），默认全部 11 个；只给两个即单对重跑")
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--n-u", type=int, default=160, help="公共站位网格的 u 格点数")
    parser.add_argument("--n-v", type=int, default=60, help="段 1（龙骨→水线）的格点数")
    parser.add_argument("--n-v2", type=int, default=20, help="段 2（水线→甲板中线）的格点数")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--output", default=str(ROOT / "dataset" / "morphed_hulls"))
    parser.add_argument("--no-verify-existing", dest="verify_existing",
                        action="store_false", help="跳过已有 STL 的重算对照（默认做）")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的 STL（先打印清单；用于几何代码改动后重建）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写任何文件")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    manifest = load_manifest(dataset)
    available = sorted(manifest)
    if len(available) != EXPECTED_HULLS:
        print(f"注意：manifest 里有 {len(available)} 个船型（期望 {EXPECTED_HULLS}）：{available}")

    hulls = select_hulls(args.hulls, available)
    if len(hulls) < 2:
        raise SystemExit(f"至少需要两个船型，当前选中 {hulls}")
    pairs = list(itertools.combinations(hulls, 2))
    total_cases = len(pairs) * len(args.weights)
    print(f"船型 {len(hulls)} 个、{len(pairs)} 对、权重 {args.weights} -> 最多 {total_cases} 个船型")

    plans, n_new, n_existing = [], 0, 0
    for name_a, name_b in pairs:
        pair = f"{name_a}_{name_b}"
        out_dir = output / pair
        meta_path = out_dir / "morph_meta.json"
        previous = json.loads(meta_path.read_text()) if meta_path.exists() else None
        cases = (previous or {}).get("cases", {})
        todo, keep = [], []
        for weight in args.weights:
            stl_name = f"{pair}_w{weight_tag(weight)}.stl"
            path = out_dir / stl_name
            if path.exists():
                if stl_name not in cases and not args.overwrite:
                    raise SystemExit(f"{path} 已存在，但 morph_meta.json 里没有"
                                     f"条目（{meta_path}），先人工确认再跑")
                if args.overwrite:
                    todo.append(weight)      # 覆盖：清单在下面打印
                    continue
                keep.append(weight)
            else:
                todo.append(weight)
        n_new += len(todo)
        n_existing += len(keep)
        plans.append({"pair": pair, "hull_a": name_a, "hull_b": name_b,
                      "out_dir": out_dir, "previous": previous,
                      "todo": todo, "keep": keep})

    print(f"已存在 {n_existing} 个（不覆盖{'，先做重算对照' if args.verify_existing and n_existing else ''}），"
          f"待生成 {n_new} 个")
    if args.overwrite:
        replaced = [plan["out_dir"] / f"{plan['pair']}_w{weight_tag(w)}.stl"
                    for plan in plans for w in plan["todo"]
                    if (plan["out_dir"] / f"{plan['pair']}_w{weight_tag(w)}.stl").exists()]
        print(f"\n--overwrite：将覆盖 {len(replaced)} 个已有文件（清单）：")
        for path in replaced:
            print(f"  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        if not replaced:
            print("  （无）")
    if args.dry_run:
        for plan in plans:
            if not plan["todo"]:
                continue
            print(f"  {plan['pair']:32s} 待生成 {plan['todo']}  保留 {plan['keep']}")
        print("（dry-run：未写入任何文件）")
        return 0

    if args.verify_existing and not args.overwrite:
        for plan in plans:
            if not plan["keep"]:
                continue
            hull_a, hull_b = manifest[plan["hull_a"]], manifest[plan["hull_b"]]
            for weight in plan["keep"]:
                stl_name = f"{plan['pair']}_w{weight_tag(weight)}.stl"
                result = morph_hulls(
                    dataset / "classic_hulls" / hull_a["stl"],
                    dataset / "classic_hulls" / hull_b["stl"],
                    hull_a["Lpp"], hull_b["Lpp"], hull_a["T"], hull_b["T"], weight,
                    n_u=args.n_u, n_v=args.n_v, n_v2=args.n_v2)
                problem = verify_existing(plan["out_dir"] / stl_name, result)
                if problem:
                    raise SystemExit(
                        f"{plan['out_dir'] / stl_name} 与当前 hull_morph 的重算结果不一致："
                        f"{problem}。该文件是已有算例/条目的溯源文件，不覆盖；先人工确认")
                print(f"  {stl_name}: 与重算一致，跳过")

    written = 0
    for index, plan in enumerate(plans, 1):
        if not plan["todo"]:
            continue
        hull_a, hull_b = manifest[plan["hull_a"]], manifest[plan["hull_b"]]
        print(f"[{index}/{len(plans)}] {plan['pair']}")
        write_pair(dataset / "classic_hulls" / hull_a["stl"],
                   dataset / "classic_hulls" / hull_b["stl"],
                   hull_a["Lpp"], hull_b["Lpp"], hull_a["T"], hull_b["T"],
                   plan["todo"], plan["out_dir"], pair=plan["pair"],
                   hull_a=plan["hull_a"], hull_b=plan["hull_b"],
                   n_u=args.n_u, n_v=args.n_v, n_v2=args.n_v2,
                   previous=plan["previous"])
        written += len(plan["todo"])

    print(f"完成：新生成 {written} 个，保留 {n_existing} 个，共 {written + n_existing} 个船型")
    return 0


if __name__ == "__main__":
    sys.exit(main())
