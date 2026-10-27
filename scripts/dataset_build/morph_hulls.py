"""按权重融合两个船型，输出水密全船 STL 与元信息。

用法:
    python3 scripts/dataset_build/morph_hulls.py --hull-a kcs --hull-b s175 \
        --weights 0.25 0.5 0.75
    python3 scripts/dataset_build/morph_hulls.py --hull-a kvlcc2 --hull-b wigley \
        --weights 0.5 --n-u 200 --n-v 80 --n-v2 26

产物写 `dataset/morphed_hulls/<hull-a>_<hull-b>/`：每个权重一个
`<hull-a>_<hull-b>_w<三位整数weight>.stl`（如 `kvlcc2_wigley_w050.stl`），元信息
汇总在 `morph_meta.json`（同一 `<a>_<b>/` 下的一份，按 STL 文件名索引各权重）。
母船 STL 取自 `dataset/classic_hulls/manifest.json`。

输出是**封闭全船**（镜像 y → −y + 首尾封口），可直接给 snappyHexMesh 建算例。
每个权重的元信息含 CFD 建算例要用的 `L_ref_m`、`T_m`、`half_beam_m`、`depth_m`、
`n_triangles`、`displacement_m3` 与 `watertight`，以及复现用的网格参数与诊断计数。
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
    path = dataset_root / "classic_hulls" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，先跑 scripts/dataset_build/collect_hull_stl.py")
    return json.loads(path.read_text())


def weight_tag(weight):
    """权重写成三位整数，例 0.25 -> "025"，STL 文件名里带它。"""
    return f"{round(weight * 100):03d}"


def main():
    parser = argparse.ArgumentParser(description="两个船型的全船融合")
    parser.add_argument("--hull-a", required=True)
    parser.add_argument("--hull-b", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5])
    parser.add_argument("--n-u", type=int, default=160, help="公共站位网格的 u 格点数")
    parser.add_argument("--n-v", type=int, default=60, help="段 1（龙骨→水线）的格点数")
    parser.add_argument("--n-v2", type=int, default=20, help="段 2（水线→甲板中线）的格点数")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--output", default=str(ROOT / "dataset" / "morphed_hulls"))
    args = parser.parse_args()

    manifest = load_manifest(Path(args.dataset))
    for hull in (args.hull_a, args.hull_b):
        if hull not in manifest:
            raise KeyError(f"{hull} 不在 {args.dataset}/classic_hulls/manifest.json 里")
    hull_a, hull_b = manifest[args.hull_a], manifest[args.hull_b]
    stl_a = Path(args.dataset) / "classic_hulls" / hull_a["stl"]
    stl_b = Path(args.dataset) / "classic_hulls" / hull_b["stl"]

    pair = f"{args.hull_a}_{args.hull_b}"
    out_dir = Path(args.output) / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for weight in args.weights:
        result = morph_hulls(stl_a, stl_b, hull_a["Lpp"], hull_b["Lpp"],
                             hull_a["T"], hull_b["T"], weight,
                             n_u=args.n_u, n_v=args.n_v, n_v2=args.n_v2)
        meta = result["meta"]
        stl_name = f"{pair}_w{weight_tag(weight)}.stl"
        path = out_dir / stl_name
        trimesh.Trimesh(result["points"], result["faces"], process=False).export(path)
        # 回读校验：STL 没有顶点共享信息，必须让 trimesh 合并顶点后再判水密
        reloaded = trimesh.load(str(path), force="mesh")
        if not reloaded.is_watertight:
            raise RuntimeError(f"{path} 回读后不水密，检查站位网格与封口")
        meta["stl"] = stl_name
        meta["stl_watertight_on_reload"] = bool(reloaded.is_watertight)
        results[stl_name] = meta
        print(f"{stl_name}: {meta['n_points']} 点 {meta['n_triangles']} 面  "
              f"L_ref={meta['L_ref_m']:.4f} T={meta['T_m']:.4f} B/2={meta['half_beam_m']:.4f} "
              f"D={meta['depth_m']:.4f} 排水量={meta['displacement_m3']:.4f} "
              f"(归一化插值 {meta['displacement_interp_norm_m3']:.4f}, "
              f"{meta['displacement_rel_diff_norm'] * 100:+.2f}%) 水密={meta['watertight']}")

    summary = {
        "hull_a": args.hull_a,
        "hull_b": args.hull_b,
        "stl_a": str(stl_a),
        "stl_b": str(stl_b),
        "Lpp_a": hull_a["Lpp"],
        "Lpp_b": hull_b["Lpp"],
        "T_a": hull_a["T"],
        "T_b": hull_b["T"],
        "n_u": args.n_u,
        "n_v": args.n_v,
        "n_v2": args.n_v2,
        "weights": list(args.weights),
        "cases": results,
    }
    (out_dir / "morph_meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"写出 {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
