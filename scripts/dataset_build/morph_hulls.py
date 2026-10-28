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

写出逻辑（建元信息 dict + 写 STL + 写 `morph_meta.json`）在
`src/dataset/hull_morph.py` 的 `write_pair`，批量入口 `morph_hull_pairs.py` 用同一个
函数，元信息 schema 只有一处定义。已有的 `morph_meta.json` 若存在，其权重条目原样
保留（增量补权重不丢溯源信息）。

输出是**封闭全船**（镜像 y → −y + 首尾封口），可直接给 snappyHexMesh 建算例。
每个权重的元信息含 CFD 建算例要用的 `L_ref_m`、`T_m`、`half_beam_m`、`depth_m`、
`n_triangles`、`displacement_m3` 与 `watertight`，以及复现用的网格参数与诊断计数。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

from hull_morph import write_pair  # noqa: E402


def load_manifest(dataset_root):
    path = dataset_root / "classic_hulls" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，先跑 scripts/dataset_build/collect_hull_stl.py")
    return json.loads(path.read_text())


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
    meta_path = out_dir / "morph_meta.json"
    previous = json.loads(meta_path.read_text()) if meta_path.exists() else None

    write_pair(stl_a, stl_b, hull_a["Lpp"], hull_b["Lpp"], hull_a["T"], hull_b["T"],
               args.weights, out_dir, pair=pair, hull_a=args.hull_a, hull_b=args.hull_b,
               n_u=args.n_u, n_v=args.n_v, n_v2=args.n_v2, previous=previous)


if __name__ == "__main__":
    sys.exit(main())
