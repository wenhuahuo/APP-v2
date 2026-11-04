"""汇总泛化实验（留一船型 / 留出设计 Fr、固定 GFLOPs 容量、三种训练种子）。

口径与主实验一致：

- 指标来自各 run 的 `output/<配置名>/metrics_best.json` 的 `strides.val.stride1`
  （留出集全密度查询），用 `src/eval/aggregate.py` 的同一条池化公式算整体流场 RMSE、
  整体 R² 与 η 通道池化 RMSE（mm）；
- 三种子 `train_seed=0/1/2` 取样本均值与样本标准差（`ddof=1`）；
- 报告 checkpoint 固定 `best.pt`、查询密度固定 `stride=1`。

    python3 scripts/eval/summarize_generalization.py --json output/generalization_summary.json
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

# (划分名, 训练条目数, 留出条目数)
TASKS = (("loto_kvlcc2", 218, 6), ("lofo_fr_design", 55, 169))
# 本轮 5 个模型（模型目录, 报告名）
MODELS = (("appsolver_oformer", "APP+OFormer"), ("transolver", "Transolver"),
          ("appsolver_v2", "APPSolver-v2"), ("gnot", "GNOT"), ("oformer", "OFormer"))
TRAIN_SEEDS = (0, 1, 2)
STRIDE = "1"
PRIMARY = "best"


def run_name(model, split, train_seed):
    suffix = "" if train_seed == 0 else f"_train_seed{train_seed}"
    return f"{model}_{split}{suffix}"


def seed_metrics(run):
    """单个 run 的整体流场 RMSE、整体 R² 与 η RMSE（mm）。"""
    path = ROOT / "output" / run / f"metrics_{PRIMARY}.json"
    metrics = json.loads(path.read_text())
    records = metrics["strides"]["val"][STRIDE]
    summary = overall(records)
    return {"run": run, "checkpoint": f"{PRIMARY}.pt",
            "checkpoint_epoch": metrics["checkpoint_epoch"],
            "parameters": metrics["parameters"], "n_points": summary["n_points"],
            "overall_rmse": summary["overall_rmse"], "overall_r2": summary["overall_r2"],
            "eta_rmse_mm": eta_rmse_mm(records)}


def mean_std(values):
    """三种子的均值与样本标准差（`ddof=1`）。"""
    return {"mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def model_row(model, label, split):
    seeds = [seed_metrics(run_name(model, split, train_seed)) for train_seed in TRAIN_SEEDS]
    points = {seed["n_points"] for seed in seeds}
    if len(points) != 1:
        raise RuntimeError(f"{split}/{model} 的留出集点数不一致：{sorted(points)}")
    return {
        "model": model,
        "label": label,
        "configs": [f"configs/{model}/{seed['run']}.yaml" for seed in seeds],
        "parameters": seeds[0]["parameters"],
        "n_points": seeds[0]["n_points"],
        "train_seeds": list(TRAIN_SEEDS),
        "overall_rmse": mean_std([seed["overall_rmse"] for seed in seeds]),
        "overall_r2": mean_std([seed["overall_r2"] for seed in seeds]),
        "eta_rmse_mm": mean_std([seed["eta_rmse_mm"] for seed in seeds]),
        "seeds": seeds,
    }


def ranks(rows, split):
    """逐种子的整体 R² 名次（1 最好）与三种子平均名次。"""
    order = {row["model"]: [] for row in rows}
    for index in range(len(TRAIN_SEEDS)):
        ranked = sorted(rows, key=lambda row: row["seeds"][index]["overall_r2"], reverse=True)
        for place, row in enumerate(ranked, start=1):
            order[row["model"]].append(place)
    return {model: {"per_seed": places, "mean": statistics.fmean(places)}
            for model, places in order.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="把汇总写进这个 json")
    args = parser.parse_args()

    tasks = {}
    for split, n_train, n_val in TASKS:
        rows = [model_row(model, label, split) for model, label in MODELS]
        first = ROOT / "output" / run_name(MODELS[0][0], split, 0) / f"metrics_{PRIMARY}.json"
        metrics = json.loads(first.read_text())
        if len(metrics["tasks"]["val"]) != n_val or len(metrics["tasks"]["train"]) != n_train:
            raise RuntimeError(f"{split} 的条目数与划分定义不符")
        ranks_by_model = ranks(rows, split)
        for row in rows:
            row["rank"] = ranks_by_model[row["model"]]
        tasks[split] = {"train_entries": n_train, "val_entries": n_val,
                        "n_points": rows[0]["n_points"], "models": rows}

    summary = {
        "source": f"output/<配置名>/metrics_{PRIMARY}.json 的 strides.val.stride{STRIDE}",
        "aggregation": "src/eval/aggregate.py：逐条目点数加权池化 → 五通道 NRMSE → "
                       "整体 RMSE / 整体 R²",
        "statistics": "ddof=1 的样本均值与样本标准差，n=3 个训练种子",
        "checkpoint_rule": f"{PRIMARY}.pt；stride {STRIDE}",
        "rank_rule": "逐种子按整体 R² 排名（1 最好），再取三种子平均名次",
        "tasks": tasks,
    }
    path = ROOT / args.json
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    for split, task in tasks.items():
        print(f"\n{split}（train {task['train_entries']} / val {task['val_entries']}，"
              f"stride {STRIDE}，best.pt，n=3 种子）")
        print(f"{'模型':14s}{'参数(M)':>9s}{'ηRMSE(mm)':>18s}{'整体RMSE%':>18s}"
              f"{'整体R²':>20s}{'名次':>8s}")
        for row in task["models"]:
            print(f"{row['label']:14s}{row['parameters'] / 1e6:9.3f}"
                  f"{row['eta_rmse_mm']['mean']:10.3f}±{row['eta_rmse_mm']['std']:.3f}"
                  f"{row['overall_rmse']['mean']:10.2f}±{row['overall_rmse']['std']:.2f}"
                  f"{row['overall_r2']['mean']:11.4f}±{row['overall_r2']['std']:.4f}"
                  f"{row['rank']['mean']:8.2f}")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
