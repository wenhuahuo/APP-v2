"""计算三个划分验证集的恒零预测基线，写逐条目物理量指标。

用法:
    python3 scripts/eval/evaluate_zero_baseline.py
    python3 scripts/eval/evaluate_zero_baseline.py --splits random_seed0

所有验证点的 eta、速度、液面 Cp 与船面 Cp 预测均设为零；结果写
`output/zero_baseline_<划分>/metrics.json`，不经过模型，也不修改模型评测文件。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ("datapipe", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from evaluate import evaluate_zero_entry  # noqa: E402
from report import summary_lines  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402

STRIDES = (1, 2, 4, 8)
SPLITS = ("loto_kvlcc2", "lofo_fr_design", "random_seed0")


def zero_density_sweep(dataset_root, entries, strides):
    """按查询密度计算恒零预测指标，结构对齐模型评测的 val 部分。"""
    results = {}
    for stride in strides:
        dataset = ShipDataset(dataset_root, entries, stride=stride)
        results[stride] = {entry: evaluate_zero_entry(sample)
                           for entry, sample in zip(dataset.entry_ids, dataset)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--strides", type=int, nargs="+", default=list(STRIDES))
    args = parser.parse_args()

    split_config = json.loads((ROOT / "configs/splits_multihull.json").read_text())
    for split in args.splits:
        entries = split_config[split]["val"]
        blocks = zero_density_sweep(ROOT / "dataset/shipbench", entries, args.strides)
        results = {
            "baseline": "zero_prediction",
            "tasks": {"val": entries},
            "strides": {"val": blocks},
        }
        output_dir = ROOT / "output" / f"zero_baseline_{split}"
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"zero_prediction，{split}，指标写入 {metrics_path}")
        for line in summary_lines({"val": entries}, {"val": blocks}, args.strides,
                                  quantities=("rmse",)):
            print(line)


if __name__ == "__main__":
    main()
