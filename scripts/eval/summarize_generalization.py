"""汇总三划分泛化实验（随机 15% / 留一船型 / 留出设计 Fr、固定 GFLOPs 容量、三种子）。

口径与主实验一致：

- 指标来自各 run 的 `output/<配置名>/metrics_best.json` 的 `strides.val.stride1`
  （留出集全密度查询），用 `src/eval/aggregate.py` 的同一条池化公式算整体流场 RMSE、
  整体 R² 与 η 通道池化 RMSE（mm）；
- 三种子 `train_seed=0/1/2` 取样本均值、样本方差与样本标准差（`ddof=1`）；
- 报告 checkpoint 固定 `best.pt`、查询密度固定 `stride=1`；
- 5 个模型共用固定 GFLOPs（iso-FLOPs）容量，GFLOPs 见
  `docs/isoflops_model_complexity.md`（单条目 `kcs_fr026_normal_mesh`、batch 1、stride 1）。

    python3 scripts/eval/summarize_generalization.py --json output/three_split_summary.json
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
for name in ("eval",):
    sys.path.insert(0, str(ROOT / "src" / name))

from aggregate import eta_rmse_mm, overall  # noqa: E402

SPLITS_FILE = ROOT / "configs" / "splits_multihull.json"
# (划分名, 训练条目数, 留出条目数, 留出语义)
TASKS = (
    ("random_seed0", 190, 34, "RandomState(0) 随机留出 15%（分布内插值）"),
    ("loto_kvlcc2", 218, 6, "留出完整 kvlcc2 船型（6 档 Fr）"),
    ("lofo_fr_design", 213, 11, "留出 11 个经典船型的 Fr 设计工况（fr_scale = 1.0）"),
)
# (模型目录, 报告名, 实测前向 GFLOPs)
MODELS = (
    ("appsolver_oformer", "APP+OFormer", 215.95),
    ("transolver", "Transolver", 218.99),
    ("appsolver_v2", "APPSolver-v2", 214.27),
    ("gnot", "GNOT", 210.88),
    ("oformer", "OFormer", 218.30),
)
TRAIN_SEEDS = (0, 1, 2)
STRIDE = "1"
PRIMARY = "best"
CAPACITY = "固定 GFLOPs（iso-FLOPs，锚点 APPSolver-v2 214.27 GFLOPs）"


def run_name(model, split, train_seed):
    suffix = "" if train_seed == 0 else f"_train_seed{train_seed}"
    return f"{model}_{split}{suffix}_isoflops"


def config_path(model, split, train_seed):
    return ROOT / "configs" / model / f"{run_name(model, split, train_seed)}.yaml"


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def seed_metrics(run):
    """单个 run 的整体流场 RMSE、整体 R² 与 η RMSE（mm）。"""
    path = ROOT / "output" / run / f"metrics_{PRIMARY}.json"
    metrics = json.loads(path.read_text())
    records = metrics["strides"]["val"][STRIDE]
    summary = overall(records)
    eta = eta_rmse_mm(records)
    for name, value in (("overall_rmse", summary["overall_rmse"]),
                        ("overall_r2", summary["overall_r2"]), ("eta_rmse_mm", eta)):
        if not finite(value):
            raise RuntimeError(f"{run} 的 {name} 非有限值：{value}")
    return {"run": run, "checkpoint": f"{PRIMARY}.pt",
            "checkpoint_epoch": metrics["checkpoint_epoch"],
            "parameters": metrics["parameters"], "n_points": summary["n_points"],
            "n_entries": len(records),
            "overall_rmse": summary["overall_rmse"], "overall_r2": summary["overall_r2"],
            "eta_rmse_mm": eta, "val_entries": sorted(records)}


def mean_std(values):
    """三种子的样本均值、样本方差与样本标准差（`ddof=1`）。"""
    return {"mean": statistics.fmean(values),
            "variance": statistics.variance(values) if len(values) > 1 else 0.0,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
            "range": max(values) - min(values), "values": values}


def check_config(model, split, train_seed, n_train, n_val, stats_name, val_entries):
    """配置与划分定义逐项对齐：条目、留出集、统计量。"""
    path = config_path(model, split, train_seed)
    if not path.is_file():
        raise RuntimeError(f"缺少配置 {path}")
    config = yaml.safe_load(path.read_text())
    entries = config["train"]["entries"]
    val = config["train"]["val_entries"]
    if len(entries) != n_train or len(val) != n_val:
        raise RuntimeError(f"{path.name} 条目数不符：train {len(entries)} / val {len(val)}")
    if sorted(val) != sorted(val_entries):
        raise RuntimeError(f"{path.name} 的留出集与 splits_multihull.json 不一致")
    if set(entries) & set(val):
        raise RuntimeError(f"{path.name} 的 train/val 有重叠")
    if config["train"].get("seed", 0) != train_seed:
        raise RuntimeError(f"{path.name} 的 train.seed 与文件名不符")
    if Path(config["stats"]).name != stats_name:
        raise RuntimeError(f"{path.name} 的 stats 指向 {config['stats']}")
    return path.relative_to(ROOT).as_posix()


def model_row(model, label, gflops, split, n_train, n_val, stats_name, val_entries):
    seeds = [seed_metrics(run_name(model, split, train_seed)) for train_seed in TRAIN_SEEDS]
    points = {seed["n_points"] for seed in seeds}
    if len(points) != 1:
        raise RuntimeError(f"{split}/{model} 的留出集点数不一致：{sorted(points)}")
    counts = {seed["n_entries"] for seed in seeds}
    if counts != {n_val}:
        raise RuntimeError(f"{split}/{model} 的留出条目数不符：{sorted(counts)}")
    params = {seed["parameters"] for seed in seeds}
    if len(params) != 1:
        raise RuntimeError(f"{split}/{model} 的参数量不一致：{sorted(params)}")
    configs = [check_config(model, split, train_seed, n_train, n_val, stats_name, val_entries)
               for train_seed in TRAIN_SEEDS]
    for seed in seeds:
        seed.pop("val_entries")
    return {
        "model": model,
        "label": label,
        "capacity": CAPACITY,
        "gflops": gflops,
        "configs": configs,
        "parameters": seeds[0]["parameters"],
        "n_points": seeds[0]["n_points"],
        "train_seeds": list(TRAIN_SEEDS),
        "overall_rmse": mean_std([seed["overall_rmse"] for seed in seeds]),
        "overall_r2": mean_std([seed["overall_r2"] for seed in seeds]),
        "eta_rmse_mm": mean_std([seed["eta_rmse_mm"] for seed in seeds]),
        "seeds": seeds,
    }


def ranks(rows):
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

    definitions = json.loads(SPLITS_FILE.read_text())
    tasks = {}
    for split, n_train, n_val, holdout in TASKS:
        definition = definitions[split]
        if definition["counts"] != {"train": n_train, "val": n_val}:
            raise RuntimeError(f"{split} 的划分定义条目数与汇总口径不符")
        val_entries = definition["val"]
        stats_name = f"stats_{split}.json"
        rows = [model_row(model, label, gflops, split, n_train, n_val, stats_name, val_entries)
                for model, label, gflops in MODELS]
        ranks_by_model = ranks(rows)
        for row in rows:
            row["rank"] = ranks_by_model[row["model"]]
        rows.sort(key=lambda row: row["overall_r2"]["mean"], reverse=True)
        tasks[split] = {
            "protocol": definition["protocol"],
            "holdout": definition["holdout"],
            "holdout_semantics": holdout,
            "train_entries": n_train,
            "val_entries": n_val,
            "stats": f"dataset/shipbench/{stats_name}",
            "n_points": rows[0]["n_points"],
            "split_source": "configs/splits_multihull.json",
            "models": rows,
        }

    summary = {
        "snapshot": "ShipBenchv2 全量 224 条 normal_mesh 快照（169 个船型）",
        "capacity": CAPACITY,
        "gflops_source": "docs/isoflops_model_complexity.md：单条目 kcs_fr026_normal_mesh、"
                         "batch 1、stride 1 前向",
        "run_count": len(TASKS) * len(MODELS) * len(TRAIN_SEEDS),
        "source": f"output/<配置名>/metrics_{PRIMARY}.json 的 strides.val.stride{STRIDE}",
        "aggregation": "src/eval/aggregate.py：逐条目点数加权池化 → 五通道 NRMSE → "
                       "整体 RMSE / 整体 R²",
        "statistics": "ddof=1 的样本均值、样本方差与样本标准差，n=3 个训练种子",
        "checkpoint_rule": f"{PRIMARY}.pt；stride {STRIDE}",
        "rank_rule": "逐种子按整体 R² 排名（1 最好），再取三种子平均名次；表内按整体 R² 均值降序",
        "checks": "每个 run 核对配置条目数 / 留出集 / stats / train.seed、留出条目数与"
                  "总点数、参数量一致性与三个指标的有限性",
        "tasks": tasks,
    }
    path = ROOT / args.json
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    for split, task in tasks.items():
        print(f"\n{split}（train {task['train_entries']} / val {task['val_entries']}，"
              f"stride {STRIDE}，best.pt，n=3 种子，{task['n_points']} 点）")
        print(f"{'模型':14s}{'参数(M)':>9s}{'GFLOPs':>9s}{'ηRMSE(mm)':>19s}"
              f"{'整体RMSE%':>19s}{'整体R²':>21s}{'名次':>8s}")
        for row in task["models"]:
            print(f"{row['label']:14s}{row['parameters'] / 1e6:9.3f}{row['gflops']:9.2f}"
                  f"{row['eta_rmse_mm']['mean']:10.3f}±{row['eta_rmse_mm']['std']:.3f}"
                  f"{row['overall_rmse']['mean']:10.2f}±{row['overall_rmse']['std']:.2f}"
                  f"{row['overall_r2']['mean']:+11.4f}±{row['overall_r2']['std']:.4f}"
                  f"{row['rank']['mean']:8.2f}")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
