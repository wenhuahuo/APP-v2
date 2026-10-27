"""按权重融合两个船型的半船湿表面，输出 STL 与元信息。

用法:
    python3 scripts/dataset_build/morph_hulls.py --hull-a kcs --hull-b s175 \
        --weights 0.25 0.5 0.75
    python3 scripts/dataset_build/morph_hulls.py --hull-a kvlcc2 --hull-b wigley \
        --weights 0.5 --n-u 200 --n-v 80

产物写 `output/ship_morphing/<hull-a>_<hull-b>/`：每个权重一个
`morphed_w<三位整数weight>.stl`（如 `morphed_w050.stl`），元信息汇总在
`morph_meta.json`。
"""

import argparse
import json
import sys
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

from hull_morph import morph_hulls  # noqa: E402


def load_manifest(dataset_root):
    path = dataset_root / "hull_stl" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，先跑 scripts/dataset_build/collect_hull_stl.py")
    return json.loads(path.read_text())


def weight_tag(weight):
    """权重写成三位整数，例 0.25 -> "025"，STL 文件名里带它。"""
    return f"{round(weight * 100):03d}"


def main():
    parser = argparse.ArgumentParser(description="两个船型的半船湿表面融合")
    parser.add_argument("--hull-a", required=True)
    parser.add_argument("--hull-b", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5])
    parser.add_argument("--n-u", type=int, default=160, help="公共 UV 网格的 u 格点数")
    parser.add_argument("--n-v", type=int, default=60, help="公共 UV 网格的 v 格点数")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--output", default=str(ROOT / "output" / "ship_morphing"))
    args = parser.parse_args()

    manifest = load_manifest(Path(args.dataset))
    for hull in (args.hull_a, args.hull_b):
        if hull not in manifest:
            raise KeyError(f"{hull} 不在 {args.dataset}/hull_stl/manifest.json 里")
    hull_a, hull_b = manifest[args.hull_a], manifest[args.hull_b]
    stl_a = Path(args.dataset) / "hull_stl" / hull_a["stl"]
    stl_b = Path(args.dataset) / "hull_stl" / hull_b["stl"]

    out_dir = Path(args.output) / f"{args.hull_a}_{args.hull_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for weight in args.weights:
        result = morph_hulls(stl_a, stl_b, hull_a["Lpp"], hull_b["Lpp"],
                             hull_a["T"], hull_b["T"], weight,
                             n_u=args.n_u, n_v=args.n_v)
        meta = result["meta"]
        stl_name = f"morphed_w{weight_tag(weight)}.stl"
        trimesh.Trimesh(result["points"], result["faces"], process=False).export(
            out_dir / stl_name)
        meta["stl"] = stl_name
        results[stl_name] = meta
        print(f"{stl_name}: {meta['n_points']} 点 {meta['n_faces']} 面  "
              f"Lpp={meta['Lpp_new']:.4f} T={meta['T_new']:.4f} "
              f"体积={meta['volume']:.4f}  "
              f"归一化插值={meta['volume_interp_norm']:.4f} "
              f"({meta['volume_rel_diff_norm'] * 100:+.2f}%)  "
              f"物理体积插值={meta['volume_interp']:.4f} "
              f"({meta['volume_rel_diff'] * 100:+.2f}%)")

    summary = {
        "hull_a": args.hull_a,
        "hull_b": args.hull_b,
        "stl_a": str(stl_a),
        "stl_b": str(stl_b),
        "n_u": args.n_u,
        "n_v": args.n_v,
        "weights": list(args.weights),
        "cases": results,
    }
    (out_dir / "morph_meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"写出 {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
