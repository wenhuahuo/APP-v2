"""汇总三个划分下 7 个模型的验证指标。

用法:
    python3 scripts/eval/summarize_splits.py [--splits loto_kvlcc2 lofo_fr_design random_seed0]
                                             [--checkpoint best|last]

每个 (模型, 划分) 从 `output/<配置名>/metrics_<config>.json` 读逐条目的 rmse / r2，
按格点数加权汇总成该划分上的总 RMSE（mm），这样不同船型点数不同也不会互相淹没。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("appsolver_v2", "appsolver_oformer", "oformer", "gnot", "upt", "transolver", "appsolver_v1")
SPLITS = ("loto_kvlcc2", "lofo_fr_design", "random_seed0")
STRIDES = ("1", "2", "4", "8")
ETA = 0
CP = 0


def pooled(entries, group, channel):
    """按点数加权的总 RMSE；R² 取逐条目平均。"""
    sse = 0.0
    n_total = 0
    r2_values = []
    for record in entries.values():
        block = record[group]
        n = block["n_points"]
        sse += block["rmse"][channel] ** 2 * n
        n_total += n
        r2_values.append(block["r2"][channel])
    return (sse / n_total) ** 0.5, sum(r2_values) / len(r2_values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    parser.add_argument("--checkpoint", default="best", choices=("best", "last"))
    args = parser.parse_args()

    missing = []
    for split in args.splits:
        first = ROOT / "output" / f"appsolver_v2_{split}" / f"metrics_{args.checkpoint}.json"
        if not first.is_file():
            missing.append(str(first))
            continue
        val_entries = json.loads(first.read_text())["tasks"]["val"]
        print(f"\n划分 {split}（val {len(val_entries)} 个条目，checkpoint {args.checkpoint}）")
        header = "  {:<20s}".format("模型") + "".join(f"{'η' + s:>9s}" for s in STRIDES)
        header += f"{'η R²':>9s}{'Cp':>9s}{'Cp R²':>8s}"
        print(header)
        for model in MODELS:
            path = ROOT / "output" / f"{model}_{split}" / f"metrics_{args.checkpoint}.json"
            if not path.is_file():
                print(f"  {model:<20s}{'（缺 metrics）':>20s}")
                continue
            metrics = json.loads(path.read_text())
            row = f"  {model:<20s}"
            eta_r2 = cp = cp_r2 = 0.0
            for stride in STRIDES:
                eta_rmse, eta_r2 = pooled(metrics["strides"]["val"][stride], "flow", ETA)
                row += f"{eta_rmse * 1000:9.2f}"
            _, eta_r2 = pooled(metrics["strides"]["val"]["1"], "flow", ETA)
            cp, cp_r2 = pooled(metrics["strides"]["val"]["1"], "hull", CP)
            row += f"{eta_r2:9.2f}{cp:9.3f}{cp_r2:8.3f}"
            print(row)
    if missing:
        print("\n缺 metrics 的划分：", *missing, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
