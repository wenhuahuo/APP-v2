"""汇总主实验（7 模型 × 3 种子）与三划分泛化实验（5 模型 × 3 划分 × 3 种子）。

口径：`random_seed0` 划分固定为随机留出 15%，`loto_kvlcc2` 留出整个 kvlcc2 船型的 6 档
Fr，`lofo_fr_design` 留出 11 个经典船型的 Fr 设计工况（`fr_scale = 1.0`，同船型内插）。
全部 run 为固定计算量容量（锚点 APPSolver-v2 214.27 GFLOPs）、100 epoch × 每条训练条目
一步（`train_strides=[1]`，主实验 19 000 步）、`eval.strides=[1]`；指标取 `best.pt` 的
stride 1 全密度查询，按 `src/eval/aggregate.py` 的逐条目点数加权池化汇总，三种子取
`ddof=1` 的均值与样本标准差。

    # 主实验：7 模型 × 3 种子 → output/main_experiment_summary.json
    python3 scripts/eval/summarize_generalization.py --task main \
        --json output/main_experiment_summary.json
    # 泛化：5 模型 × 3 划分 × 3 种子 → output/three_split_summary.json
    python3 scripts/eval/summarize_generalization.py --task splits \
        --json output/three_split_summary.json

`--models` 可以只汇总部分模型（诊断用），默认给出该任务的完整模型清单。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ("eval",):
    sys.path.insert(0, str(ROOT / "src" / name))

import summary as runs  # noqa: E402

SPLITS_FILE = Path("configs") / "splits_multihull.json"
# 主实验 7 个模型：模型目录名（同时是 run 名前缀）→ 报告名
MODEL_LABELS = {
    "appsolver_v2": "APPSolver-v2",
    "appsolver_oformer": "APP+OFormer",
    "appsolver_v1": "APPSolver-v1",
    "gnot": "GNOT",
    "upt": "UPT",
    "transolver": "Transolver",
    "oformer": "OFormer",
}
MAIN_MODELS = ("appsolver_v2", "appsolver_oformer", "appsolver_v1", "gnot", "upt",
               "transolver", "oformer")
# 泛化实验的 5 个模型（变化 v1 / UPT 只训练主实验划分）
GENERALIZATION_MODELS = ("appsolver_oformer", "oformer", "appsolver_v2", "transolver", "gnot")
TRAIN_SEEDS = (0, 1, 2)
MAIN_SPLIT = "random_seed0"
# (划分名, 留出语义)
SPLITS = (
    (MAIN_SPLIT, "RandomState(0) 随机留出 15%（分布内插值）"),
    ("loto_kvlcc2", "留出完整 kvlcc2 船型（6 档 Fr）；训练集含 30 条含 kvlcc2 几何的融合船型"),
    ("lofo_fr_design", "留出 11 个经典船型的 Fr 设计工况（fr_scale = 1.0，同船型内插，无外推档位）"),
)
CAPACITY = "固定前向计算量（iso-FLOPs，锚点 APPSolver-v2 214.27 GFLOPs）"


def run_name(model, split, train_seed):
    suffix = "" if train_seed == 0 else f"_train_seed{train_seed}"
    return f"{model}_{split}{suffix}_isoflops"


def split_definition(root, split):
    definitions = json.loads((Path(root) / SPLITS_FILE).read_text())
    if split not in definitions:
        raise KeyError(f"{SPLITS_FILE.as_posix()} 缺少划分 {split}")
    return definitions[split]


def task_rows(root, models, split, table):
    """一个划分下若干模型的行；留出条目数与留出集逐项对齐划分定义。"""
    definition = split_definition(root, split)
    rows = runs.model_rows(
        root, [MODEL_LABELS[model] for model in models],
        lambda label: [run_name(model, split, seed)
                       for model in models if MODEL_LABELS[model] == label
                       for seed in TRAIN_SEEDS],
        table, definition["counts"]["train"])
    expected = sorted(definition["val"])
    for row in rows:
        if row["n_entries"] != definition["counts"]["val"]:
            raise RuntimeError(f"{row['runs'][0]} 的留出条目数为 {row['n_entries']}，"
                               f"划分定义是 {definition['counts']['val']}")
        metrics = (Path(root) / row["seeds"][0]["run_dir"] / "metrics_best.json").read_text()
        if sorted(json.loads(metrics)["tasks"]["val"]) != expected:
            raise RuntimeError(f"{row['runs'][0]} 的留出集与 {SPLITS_FILE.as_posix()} 不一致")
    places = runs.ranks(rows)
    for row in rows:
        row["rank"] = places[row["label"]]
    return runs.sort_by_mean(rows), definition


def task_block(root, models, split, table, semantics):
    rows, definition = task_rows(root, models, split, table)
    return {
        "split": split,
        "protocol": definition["protocol"],
        "holdout": definition["holdout"],
        "holdout_semantics": semantics,
        "train_entries": definition["counts"]["train"],
        "val_entries": definition["counts"]["val"],
        "n_points": rows[0]["n_points"],
        "stats": f"dataset/shipbench/stats_{split}.json",
        "split_source": SPLITS_FILE.as_posix(),
        "models": rows,
    }


def main_task(root, models, table):
    _, definition = task_rows(root, models, MAIN_SPLIT, table)
    return {
        "experiment": "main",
        "task": task_block(root, models, MAIN_SPLIT, table, dict(SPLITS)[MAIN_SPLIT]),
        "run_count": len(models) * len(TRAIN_SEEDS),
        "capacity": CAPACITY,
        "train_seeds": list(TRAIN_SEEDS),
        "epochs": runs.TRAIN_STEPS_PER_ENTRY,
        "train_steps": definition["counts"]["train"] * runs.TRAIN_STEPS_PER_ENTRY,
        "rank_rule": "逐种子按整体 R² 排名（1 最好），再取三种子平均名次；"
                     "表内按整体 R² 均值降序",
        **runs.baseline_metadata(root),
    }


def splits_task(root, models, table):
    tasks = {split: task_block(root, models, split, table, semantics)
             for split, semantics in SPLITS}
    cross = {MODEL_LABELS[model]: {} for model in models}
    for split, task in tasks.items():
        for row in task["models"]:
            cross[row["label"]][split] = {"overall_r2": row["overall_r2"],
                                          "eta_rmse_mm": row["eta_rmse_mm"],
                                          "rank": row["rank"]}
    return {
        "experiment": "generalization",
        "capacity": CAPACITY,
        "train_seeds": list(TRAIN_SEEDS),
        "run_count": len(models) * len(tasks) * len(TRAIN_SEEDS),
        "tasks": tasks,
        "cross_split": cross,
        "rank_rule": "逐种子按整体 R² 排名（1 最好），再取三种子平均名次；"
                     "表内按整体 R² 均值降序",
        **runs.baseline_metadata(root),
    }


def print_models(title, rows, train_entries):
    print(f"\n{title}（train {train_entries}，best.pt，stride 1，n=3 种子，"
          f"{rows[0]['n_points']} 点，{rows[0]['n_entries']} 条留出）")
    print(f"{'模型':14s}{'参数(M)':>9s}{'GFLOPs':>8s}{'token':>8s}"
          f"{'训练token(M)':>12s}{'ηRMSE(mm)':>19s}{'整体RMSE%':>19s}{'整体R²':>21s}"
          f"{'名次':>7s}")
    for row in rows:
        gflops = row["gflops"]
        print(f"{row['label']:14s}{row['parameters'] / 1e6:9.3f}"
              f"{'-' if gflops is None else format(gflops, '8.2f')}"
              f"{row['tokens_stride1']:8d}{row['training_tokens'] / 1e6:12.3f}"
              f"{row['eta_rmse_mm']['mean']:10.3f}±{row['eta_rmse_mm']['std']:.3f}"
              f"{row['overall_rmse']['mean']:10.2f}±{row['overall_rmse']['std']:.2f}"
              f"{row['overall_r2']['mean']:+11.4f}±{row['overall_r2']['std']:.4f}"
              f"{row['rank']['mean']:7.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("main", "splits"))
    parser.add_argument("--json", required=True, help="把汇总写进这个 json")
    parser.add_argument("--models", nargs="*", default=None,
                        help="只汇总这些模型（诊断用，默认取该任务的完整清单）")
    parser.add_argument("--root", default=str(ROOT), help="仓库根目录")
    args = parser.parse_args()

    root = Path(args.root)
    table = runs.load_complexity(root)
    models = tuple(args.models) if args.models else (
        MAIN_MODELS if args.task == "main" else GENERALIZATION_MODELS)
    unknown = [model for model in models if model not in MODEL_LABELS]
    if unknown:
        parser.error(f"未知模型 {unknown}，可选 {sorted(MODEL_LABELS)}")

    summary = main_task(root, models, table) if args.task == "main" \
        else splits_task(root, models, table)
    path = root / args.json
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if args.task == "main":
        task = summary["task"]
        print_models(f"主实验 {MAIN_SPLIT}（{len(models)} 模型 × 3 种子）",
                     task["models"], task["train_entries"])
    else:
        for split, task in summary["tasks"].items():
            print_models(f"{split}：{task['holdout_semantics']}",
                         task["models"], task["train_entries"])
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
