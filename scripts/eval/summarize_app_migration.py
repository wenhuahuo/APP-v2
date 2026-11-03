"""汇总 APP 分词前端迁移结果：8 个模型（4 原生 + 4 APP）× 3 训练种子的整体指标与成本。

口径与主实验一致：

- 指标来自各 run 的 `output/<配置名>/metrics_best.json` 的 `strides.val.stride1`
  （34 条 `random_seed0` 留出集、全密度查询），用 `src/eval/aggregate.py` 的同一条池化
  公式算整体 RMSE、整体 R² 与 η 通道池化 RMSE；三种子取均值与样本标准差（`ddof=1`）；
- 参数量、stride 1 主干 token 数、训练 token 总量与前向 GFLOPs 由
  `scripts/eval/profile_model_flops.py` 统计后以 `--flops` 传入，本脚本只做汇总；
- 延迟取 `seed=0` 的 `best.pt` 在单张 A6000 上的 stride 1 单条目前向均值
  （`scripts/eval/measure_inference_time.py` 的 `inference_time.json`），4 个原生模型与
  APP+OFormer 复用主实验同一口径的 `output/inference_time_18runs.json`。

    python3 scripts/eval/summarize_app_migration.py \
        --flops output/app_migration_flops.json \
        --latency output/inference_time_18runs.json output/inference_time_app_migration.json \
        --json output/app_migration_summary.json
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
PAIRS = (("oformer", "appsolver_oformer"), ("transolver", "appsolver_transolver"),
         ("upt", "appsolver_upt"), ("gnot", "appsolver_gnot"))
CHECKPOINT = "best.pt"
METRICS = "best"
STRIDE = "1"


def run_name(model, train_seed):
    return (f"{model}_{SPLIT}" if train_seed == 0
            else f"{model}_{SPLIT}_train_seed{train_seed}")


def config_path(model, train_seed):
    return ROOT / "configs" / model / f"{run_name(model, train_seed)}.yaml"


def metrics_path(model, train_seed, checkpoint):
    return ROOT / "output" / run_name(model, train_seed) / f"metrics_{checkpoint}.json"


def seed_metrics(model, train_seed):
    """单个 run 的整体 RMSE、整体 R² 与 η RMSE（mm）。"""
    metrics = json.loads(metrics_path(model, train_seed, METRICS).read_text())
    records = metrics["strides"]["val"][STRIDE]
    summary = overall(records)
    return {"train_seed": train_seed, "run": run_name(model, train_seed),
            "checkpoint_epoch": metrics["checkpoint_epoch"],
            "n_points": summary["n_points"],
            "overall_rmse": summary["overall_rmse"], "overall_r2": summary["overall_r2"],
            "eta_rmse_mm": eta_rmse_mm(records)}


def mean_std(values):
    """三种子的均值与样本标准差（`ddof=1`）。"""
    return {"mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def flops_row(rows, model):
    stem = f"{model}_{SPLIT}"
    for row in rows:
        if row["config"] == stem:
            return row
    raise KeyError(f"缺少 {stem} 的计算量统计")


def latency_ms(records, model):
    """`seed=0` 的 `best.pt` 在 stride 1 上的单条目前向均值（秒 → 毫秒）。"""
    path = str(config_path(model, 0).relative_to(ROOT))
    for record in records:
        if record["config"] == path:
            if record["checkpoint"] != CHECKPOINT:
                raise ValueError(f"{path} 的延迟不是 {CHECKPOINT}")
            return 1000.0 * record["strides"][STRIDE]["mean"]
    raise KeyError(f"缺少 {path} 的延迟")


def model_summary(model, rows, records):
    seeds = [seed_metrics(model, seed) for seed in TRAIN_SEEDS]
    profile = flops_row(rows, model)
    return {
        "model": model,
        "config": str(config_path(model, 0).relative_to(ROOT)),
        "parameters": profile["parameters"],
        "tokens_stride1": profile["tokens"],
        "token_kind": profile["token_kind"],
        "training_tokens": profile["training_tokens"],
        "gflops": profile["gflops"],
        "latency_stride1_ms": latency_ms(records, model),
        "overall_rmse": mean_std([seed["overall_rmse"] for seed in seeds]),
        "overall_r2": mean_std([seed["overall_r2"] for seed in seeds]),
        "eta_rmse_mm": mean_std([seed["eta_rmse_mm"] for seed in seeds]),
        "seeds": seeds,
    }


def migration_row(native, app):
    return {
        "native": native["model"], "app": app["model"],
        "overall_r2_change": app["overall_r2"]["mean"] - native["overall_r2"]["mean"],
        "overall_rmse_change": app["overall_rmse"]["mean"] - native["overall_rmse"]["mean"],
        "eta_rmse_change_mm": app["eta_rmse_mm"]["mean"] - native["eta_rmse_mm"]["mean"],
        "token_compression": native["tokens_stride1"] / app["tokens_stride1"],
        "latency_ratio": app["latency_stride1_ms"] / native["latency_stride1_ms"],
    }


def checkpoint_check(model, train_seed):
    """best / last 两份指标是否都已写出（训练完成的证据）。"""
    return {checkpoint: metrics_path(model, train_seed, checkpoint).is_file()
            for checkpoint in ("best", "last")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flops", default="output/app_migration_flops.json")
    parser.add_argument("--latency", nargs="+",
                        default=["output/inference_time_18runs.json",
                                 "output/inference_time_app_migration.json"])
    parser.add_argument("--json", default="output/app_migration_summary.json")
    args = parser.parse_args()

    rows = json.loads((ROOT / args.flops).read_text())
    records = [record for path in args.latency
               for record in json.loads((ROOT / path).read_text())]
    models = [model for pair in PAIRS for model in pair]
    summaries = {model: model_summary(model, rows, records) for model in models}
    summary = {
        "source": f"output/<配置名>/metrics_{METRICS}.json 的 strides.val.stride{STRIDE}"
                  f"（34 条 {SPLIT} 留出集）",
        "aggregation": "src/eval/aggregate.py：逐条目点数加权池化 → 五通道 NRMSE → "
                       "整体 RMSE / 整体 R²",
        "statistics": "ddof=1 的样本均值与样本标准差，n=3 个训练种子",
        "cost_source": {
            "parameters_tokens_gflops": args.flops,
            "latency": args.latency,
        },
        "latency_protocol": "单条目 kcs_fr026_normal_mesh、batch 1、float32、eval + no_grad、"
                            "单张 A6000、warmup 5 + repeat 20 × 5 轮取均值最小轮、stride 1",
        "train_seeds": list(TRAIN_SEEDS),
        "val_entries": 34,
        "pairs": [{"native": native, "app": app} for native, app in PAIRS],
        "models": [summaries[model] for model in models],
        "migration": [migration_row(summaries[native], summaries[app]) for native, app in PAIRS],
        "new_runs": [{"model": model, "train_seed": seed,
                      "checkpoints": checkpoint_check(model, seed)}
                     for model in ("appsolver_transolver", "appsolver_upt", "appsolver_gnot")
                     for seed in TRAIN_SEEDS],
    }
    path = ROOT / args.json
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    header = (f"{'模型':22s}{'参数(M)':>9s}{'token':>8s}{'训练token(M)':>13s}"
              f"{'GFLOPs':>9s}{'RMSE%':>9s}{'R²':>9s}{'ηRMSE(mm)':>12s}{'延迟(ms)':>10s}")
    print(header)
    for model in models:
        row = summaries[model]
        print(f"{model:22s}{row['parameters'] / 1e6:9.3f}{row['tokens_stride1']:8d}"
              f"{row['training_tokens'] / 1e6:13.1f}{row['gflops']:9.2f}"
              f"{row['overall_rmse']['mean']:9.2f}{row['overall_r2']['mean']:9.4f}"
              f"{row['eta_rmse_mm']['mean']:12.2f}{row['latency_stride1_ms']:10.2f}")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
