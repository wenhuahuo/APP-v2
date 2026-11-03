"""汇总 APP+OFormer 船体融合机制消融的三组结果（固定 GFLOPs 容量、random_seed0 划分）。

口径与主实验一致：

- 指标来自各 run 的 `output/<配置名>/metrics_{best,last}.json` 的 `strides.val.stride1`
  （34 条 `random_seed0` 留出集、全密度查询），用 `src/eval/aggregate.py` 的同一条池化
  公式算整体 RMSE、整体 R²、η 通道池化 RMSE（mm）；联合模型另给船面 Cp 通道的池化
  RMSE 与 R²；
- 参数量、stride 1 主干 token 数、训练 token 总量与前向 GFLOPs 由
  `scripts/eval/profile_model_flops.py --json` 统计后以 `--flops` 传入，本脚本只做汇总；
- `flow_with_hull` 组直接复用 `output/appsolver_oformer_random_seed0_isoflops`
  （同协议全量 seed0 的既有结果），`flow_only` 与 `joint_flow_hull` 由
  `scripts/train/train_app_oformer_hull_fusion.sbatch` 训练。

    python3 scripts/eval/summarize_hull_fusion.py \
        --flops output/app_oformer_hull_fusion_flops.json \
        --json output/app_oformer_hull_fusion_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ("eval",):
    sys.path.insert(0, str(ROOT / "src" / name))

from aggregate import eta_rmse_mm, overall, pooled_channel  # noqa: E402

STRIDE = "1"
CHECKPOINTS = ("best", "last")
# (配置名, 输入域, 输出域, lambda_hull)；flow_with_hull 组是既有基础配置。
CONFIGS = (
    ("appsolver_oformer_flow_only_random_seed0_isoflops", ("flow",), ("flow",), 0.0),
    ("appsolver_oformer_random_seed0_isoflops", ("flow", "hull"), ("flow",), 0.0),
    ("appsolver_oformer_joint_flow_hull_random_seed0_isoflops",
     ("flow", "hull"), ("flow", "hull"), 1.0),
)


def metrics_path(run, checkpoint):
    return ROOT / "output" / run / f"metrics_{checkpoint}.json"


def checkpoint_summary(run, checkpoint):
    """单个 checkpoint 的整体 RMSE、整体 R²、η RMSE（mm）与船面 Cp 指标。"""
    metrics = json.loads(metrics_path(run, checkpoint).read_text())
    records = metrics["strides"]["val"][STRIDE]
    summary = overall(records)
    row = {
        "checkpoint": f"{checkpoint}.pt",
        "checkpoint_epoch": metrics["checkpoint_epoch"],
        "parameters": metrics["parameters"],
        "n_points": summary["n_points"],
        "overall_rmse": summary["overall_rmse"],
        "overall_r2": summary["overall_r2"],
        "eta_rmse_mm": eta_rmse_mm(records),
        "channel_rmse": dict(zip(("eta", "u", "v", "w", "p_rgh"), summary["rmse"])),
        "channel_nrmse": dict(zip(("eta", "u", "v", "w", "p_rgh"), summary["nrmse"])),
    }
    if all("hull" in record for record in records.values()):
        rmse, sigma, count = pooled_channel(records, 0, group="hull")
        row["hull"] = {"n_points": count, "cp_rmse": rmse,
                       "cp_nrmse": rmse / sigma, "cp_r2": 1.0 - (rmse / sigma) ** 2}
    return row


def flops_row(rows, run):
    for row in rows:
        if row["config"] == run:
            return row
    raise KeyError(f"缺少 {run} 的计算量统计")


def model_row(run, input_domains, output_domains, lambda_hull, rows):
    profile = flops_row(rows, run)
    return {
        "config": run,
        "role": {"input_domains": list(input_domains),
                 "output_domains": list(output_domains),
                 "lambda_hull": lambda_hull},
        "parameters": profile["parameters"],
        "tokens_stride1": profile["tokens"],
        "token_kind": profile["token_kind"],
        "training_tokens": profile["training_tokens"],
        "gflops": profile["gflops"],
        "checkpoints": {name: checkpoint_summary(run, name) for name in CHECKPOINTS},
    }


def relative_change(new, base):
    return 100.0 * (new - base) / base


def comparisons(rows):
    """两个单因素对照：加船体输入（flow_only → flow_with_hull）、加船面监督。"""
    by_run = {row["config"]: row for row in rows}
    flow_only = by_run[CONFIGS[0][0]]
    with_hull = by_run[CONFIGS[1][0]]
    joint = by_run[CONFIGS[2][0]]
    channels = ("eta", "u", "v", "w", "p_rgh")

    def per_channel(base, new):
        return {channel: relative_change(new["channel_rmse"][channel],
                                         base["channel_rmse"][channel])
                for channel in channels}

    base_best = flow_only["checkpoints"]["best"]
    with_hull_best = with_hull["checkpoints"]["best"]
    joint_best = joint["checkpoints"]["best"]
    return {
        "hull_input_effect": {
            "base": flow_only["config"], "new": with_hull["config"], "checkpoint": "best.pt",
            "overall_rmse_change_percent": relative_change(
                with_hull_best["overall_rmse"], base_best["overall_rmse"]),
            "overall_r2_change": with_hull_best["overall_r2"] - base_best["overall_r2"],
            "eta_rmse_change_percent": relative_change(
                with_hull_best["eta_rmse_mm"], base_best["eta_rmse_mm"]),
            "channel_rmse_change_percent": per_channel(base_best, with_hull_best),
        },
        "hull_supervision_effect": {
            "base": with_hull["config"], "new": joint["config"], "checkpoint": "best.pt",
            "overall_rmse_change_percent": relative_change(
                joint_best["overall_rmse"], with_hull_best["overall_rmse"]),
            "overall_r2_change": joint_best["overall_r2"] - with_hull_best["overall_r2"],
            "eta_rmse_change_percent": relative_change(
                joint_best["eta_rmse_mm"], with_hull_best["eta_rmse_mm"]),
            "channel_rmse_change_percent": per_channel(with_hull_best, joint_best),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flops", required=True,
                        help="profile_model_flops.py --json 写出的计算量统计")
    parser.add_argument("--json", default=None, help="把汇总写进这个 json")
    args = parser.parse_args()

    rows = [model_row(run, input_domains, output_domains, lambda_hull,
                      json.loads(Path(args.flops).read_text()))
            for run, input_domains, output_domains, lambda_hull in CONFIGS]
    summary = {"checkpoint_rule": "best.pt 为主表，last.pt 用于稳定性核对",
               "eval_split": "random_seed0 val（34 条）",
               "query_stride": int(STRIDE),
               "models": rows,
               "comparisons": comparisons(rows)}

    header = (f"{'配置':52s} {'参数(M)':>8s} {'GFLOPs':>8s} {'token':>7s} "
              f"{'整体RMSE%':>9s} {'整体R²':>7s} {'etaRMSE(mm)':>11s}")
    print(header)
    for row in rows:
        best = row["checkpoints"]["best"]
        print(f"{row['config']:52s} {row['parameters'] / 1e6:8.3f} {row['gflops']:8.3f} "
              f"{row['tokens_stride1']:7d} {best['overall_rmse']:9.3f} "
              f"{best['overall_r2']:7.3f} {best['eta_rmse_mm']:11.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"\n写入 {args.json}")


if __name__ == "__main__":
    main()
