"""汇总 APP+OFormer 两项全量消融（固定 GFLOPs 容量、random_seed0 划分、三种训练种子）。

口径与主实验一致：

- 指标来自各 run 的 `output/<配置名>/metrics_{best,last}.json` 的 `strides.val.stride1`
  （34 条 `random_seed0` 留出集、全密度查询），用 `src/eval/aggregate.py` 的同一条池化
  公式算整体流场 RMSE、整体 R² 与 η 通道池化 RMSE（mm）；三种子取均值与样本标准差
  （`ddof=1`）；
- 参数量、stride 1 主干 token 数、训练 token 总量与前向 GFLOPs 由
  `scripts/eval/profile_model_flops.py --json` 统计后以 `--flops` 传入，本脚本只做汇总；
- 两项消融各有一个 `--kind`：`partition` 对照均匀 / 学习 / 自适应流场分词，
  `conditioning` 对照六种工况融合方式；`adaptive` 与 `film` 组直接复用
  `appsolver_oformer_random_seed0[_train_seed1/2]_isoflops` 的既有结果。

    python3 scripts/eval/summarize_app_oformer_ablations.py --kind partition \
        --flops output/app_oformer_partition_ablation_flops.json \
        --json output/app_oformer_partition_ablation_summary.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ("eval",):
    sys.path.insert(0, str(ROOT / "src" / name))

from aggregate import eta_rmse_mm, overall  # noqa: E402

SPLIT = "random_seed0"
TRAIN_SEEDS = (0, 1, 2)
STRIDE = "1"
CHECKPOINTS = ("best", "last")
# 主表用 best.pt；last.pt 只做稳定性核对
PRIMARY = "best"
BASE = f"appsolver_oformer_{SPLIT}_isoflops"
# (变体名, 配置名前缀)；前缀 + `_random_seed0[_train_seed<S>]_isoflops.yaml`
VARIANTS = {
    "partition": (
        ("uniform", "appsolver_oformer_uniform_partition"),
        ("learned", "appsolver_oformer_learned_partition"),
        ("adaptive", "appsolver_oformer"),
    ),
    "conditioning": (
        ("film", "appsolver_oformer"),
        ("none", "appsolver_oformer_conditioning_none"),
        ("adaln_zero", "appsolver_oformer_conditioning_adaln_zero"),
        ("fourier_add", "appsolver_oformer_conditioning_fourier_add"),
        ("cond_token", "appsolver_oformer_conditioning_cond_token"),
        ("cond_xattn", "appsolver_oformer_conditioning_cond_xattn"),
    ),
}
LABELS = {
    "partition": "流场分词",
    "conditioning": "工况融合",
}


def run_name(prefix, train_seed):
    suffix = "" if train_seed == 0 else f"_train_seed{train_seed}"
    return f"{prefix}_{SPLIT}{suffix}_isoflops"


def metrics_path(run, checkpoint):
    return ROOT / "output" / run / f"metrics_{checkpoint}.json"


def seed_metrics(run, checkpoint):
    """单个 run 的整体流场 RMSE、整体 R² 与 η RMSE（mm）。"""
    metrics = json.loads(metrics_path(run, checkpoint).read_text())
    records = metrics["strides"]["val"][STRIDE]
    summary = overall(records)
    return {"checkpoint": f"{checkpoint}.pt", "checkpoint_epoch": metrics["checkpoint_epoch"],
            "parameters": metrics["parameters"], "n_points": summary["n_points"],
            "overall_rmse": summary["overall_rmse"], "overall_r2": summary["overall_r2"],
            "eta_rmse_mm": eta_rmse_mm(records)}


def mean_std(values):
    """三种子的均值与样本标准差（`ddof=1`）。"""
    return {"mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def flops_row(rows, run):
    for row in rows:
        if row["config"] == run:
            return row
    raise KeyError(f"缺少 {run} 的计算量统计")


def variant_row(prefix, rows):
    """一个变体的三种子指标与计算成本；成本取 seed 0 配置的结构统计。"""
    profile = flops_row(rows, run_name(prefix, 0))
    seeds = {seed: seed_metrics(run_name(prefix, seed), PRIMARY) for seed in TRAIN_SEEDS}
    row = {
        "config_prefix": prefix,
        "configs": [f"configs/appsolver_oformer/{run_name(prefix, seed)}.yaml"
                    for seed in TRAIN_SEEDS],
        "parameters": profile["parameters"],
        "tokens_stride1": profile["tokens"],
        "token_kind": profile["token_kind"],
        "training_tokens": profile["training_tokens"],
        "gflops": profile["gflops"],
        "train_seeds": list(TRAIN_SEEDS),
        "overall_rmse": mean_std([seeds[seed]["overall_rmse"] for seed in TRAIN_SEEDS]),
        "overall_r2": mean_std([seeds[seed]["overall_r2"] for seed in TRAIN_SEEDS]),
        "eta_rmse_mm": mean_std([seeds[seed]["eta_rmse_mm"] for seed in TRAIN_SEEDS]),
        "seeds": [dict(run=run_name(prefix, seed), **seeds[seed]) for seed in TRAIN_SEEDS],
    }
    last = [seed_metrics(run_name(prefix, seed), "last") for seed in TRAIN_SEEDS]
    row["last"] = {"overall_rmse": mean_std([item["overall_rmse"] for item in last]),
                   "eta_rmse_mm": mean_std([item["eta_rmse_mm"] for item in last])}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--flops", required=True,
                        help="profile_model_flops.py --json 写出的计算量统计")
    parser.add_argument("--json", required=True, help="把汇总写进这个 json")
    args = parser.parse_args()

    rows = json.loads(Path(args.flops).read_text())
    variants = [variant_row(prefix, rows) for _, prefix in VARIANTS[args.kind]]
    summary = {
        "ablation": args.kind,
        "source": f"output/<配置名>/metrics_{PRIMARY}.json 的 strides.val.stride{STRIDE}"
                  f"（34 条 {SPLIT} 留出集）",
        "aggregation": "src/eval/aggregate.py：逐条目点数加权池化 → 五通道 NRMSE → "
                       "整体 RMSE / 整体 R²",
        "statistics": "ddof=1 的样本均值与样本标准差，n=3 个训练种子",
        "cost_source": args.flops,
        "checkpoint_rule": "best.pt 为主表，last.pt 用于稳定性核对",
        "reused_run": BASE,
        "variants": variants,
    }
    path = ROOT / args.json
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    header = (f"{'变体':14s}{'参数(M)':>9s}{'GFLOPs':>9s}{'token':>7s}{'训练token(M)':>13s}"
              f"{'整体RMSE%':>16s}{'整体R²':>16s}{'ηRMSE(mm)':>16s}")
    print(f"APP+OFormer {LABELS[args.kind]}消融（{SPLIT} val，stride 1，n=3 种子）")
    print(header)
    for (name, _), row in zip(VARIANTS[args.kind], variants):
        print(f"{name:14s}{row['parameters'] / 1e6:9.3f}{row['gflops']:9.2f}"
              f"{row['tokens_stride1']:7d}{row['training_tokens'] / 1e6:13.1f}"
              f"{row['overall_rmse']['mean']:10.2f}±{row['overall_rmse']['std']:.2f}"
              f"{row['overall_r2']['mean']:10.4f}±{row['overall_r2']['std']:.4f}"
              f"{row['eta_rmse_mm']['mean']:10.3f}±{row['eta_rmse_mm']['std']:.3f}")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
