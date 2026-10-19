"""在训练集条目上拟合归一化统计量，写 dataset/stats.json。

用法:
    python3 scripts/datapipe/build_stats.py --dataset dataset \
        --entries kcs_fr026_normal_mesh
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "datapipe"))

from normalize import Normalizer  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--entries", nargs="*", default=None,
                        help="训练集条目 id，默认取 index.json 的全部条目")
    parser.add_argument("--output", default=None, help="默认 <dataset>/stats.json")
    args = parser.parse_args()

    dataset = ShipDataset(args.dataset, entry_ids=args.entries)
    normalizer = Normalizer.fit(dataset)
    output = Path(args.output) if args.output else Path(args.dataset) / "stats.json"
    normalizer.save(output)

    stats = normalizer.to_dict()
    print(f"{len(dataset)} 个条目 -> {output}")
    for group, values in stats.items():
        if any(values["constant"]):
            print(f"  警告：{group} 有常数通道，标准化后恒为 0", file=sys.stderr)
    print(f"  flow      mean={[round(v, 4) for v in stats['flow']['mean']]}"
          f" std={[round(v, 4) for v in stats['flow']['std']]}")
    print(f"  hull      mean={[round(v, 4) for v in stats['hull']['mean']]}"
          f" std={[round(v, 4) for v in stats['hull']['std']]}")
    print(f"  condition mean={[round(v, 4) for v in stats['condition']['mean']]}"
          f" std={[round(v, 4) for v in stats['condition']['std']]}")


if __name__ == "__main__":
    main()
