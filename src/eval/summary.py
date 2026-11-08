"""跨 run 汇总评测指标：主实验、泛化实验与消融共用同一条读取与聚合路径。

口径固定为：

- 指标来自各 run 的 `metrics_{best,last}.json`（`scripts/eval/evaluate_model.py` 写出）
  的 `strides.val.stride1`，即留出集全密度查询；
- 整体流场 RMSE、整体 R² 与 η 通道池化 RMSE（mm）走 `src/eval/aggregate.py` 的同一条
  点数加权池化公式；
- 多个训练种子取样本均值、样本方差与样本标准差（`ddof=1`）；
- 前向 GFLOPs、主干 token 数与 token 口径来自 `scripts/eval/model_complexity.json`
  （标定值见 `docs/isoflops_model_complexity.md`），参数量取各 run 的 `parameters` 字段。

`output/archive/stale_after_collision/` 是被 `scripts/eval/archive_outputs.py` 移走的旧
路径目录：其中 4 个 `random_seed0` 训练的 seed 0 run 因 NFS 同名目录命中旧位置而写进了
归档位置，指标仍属本轮报告口径，因此 `resolve_run` 只在找不到 `output/<run>` 时回退到
这个子目录，不搜索其它归档子目录（其余归档是旧协议或重复副本）。
"""

import json
import math
import re
import statistics
from pathlib import Path

from aggregate import eta_rmse_mm, overall, pooled_channel

# 与 `src/eval/metrics.py` 的 FLOW_CHANNELS 顺序一致
CHANNELS = ("eta", "u", "v", "w", "p_rgh")
OUTPUT_DIR = Path("output")
ARCHIVE_DIR = OUTPUT_DIR / "archive" / "stale_after_collision"
COMPLEXITY_FILE = Path("scripts") / "eval" / "model_complexity.json"
# 三个划分共用同一份模型结构与容量，标定表按 random_seed0 的 run 名登记
SPLIT_TOKENS = ("loto_kvlcc2", "lofo_fr_design", "random_seed0")
TRAIN_SEED_TOKEN = re.compile(r"_train_seed\d+(?=_isoflops$|$)")
PRIMARY = "best"
STRIDE = "1"
TRAIN_STEPS_PER_ENTRY = 100


def resolve_run(root, run):
    """run 目录：`output/<run>` 优先，其次 `output/archive/stale_after_collision/<run>`。"""
    root = Path(root)
    for base in (OUTPUT_DIR, ARCHIVE_DIR):
        candidate = root / base / run
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"缺少 run 目录 output/{run}"
                            f"（也不在 {ARCHIVE_DIR.as_posix()}/ 下）")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def check_protocol(metrics, run):
    """本轮统一协议：`train_strides=[1]`、只评测 stride 1、checkpoint 为 `best.pt`。"""
    if list(metrics["train_strides"]) != [1]:
        raise RuntimeError(f"{run} 的 train_strides={metrics['train_strides']}，期望 [1]")
    if sorted(metrics["strides"]["val"]) != [STRIDE]:
        raise RuntimeError(f"{run} 的评测密度为 {sorted(metrics['strides']['val'])}，"
                           f"期望只有 stride {STRIDE}")
    if metrics["checkpoint"] != f"{PRIMARY}.pt":
        raise RuntimeError(f"{run} 的 checkpoint 为 {metrics['checkpoint']}")


def run_metrics(root, run, checkpoint=PRIMARY, stride=STRIDE):
    """单个 run 的整体指标；联合输出模型另给船面 Cp 的池化 RMSE 与 R²。"""
    directory = resolve_run(root, run)
    metrics = json.loads((directory / f"metrics_{checkpoint}.json").read_text())
    check_protocol(metrics, run)
    records = metrics["strides"]["val"][stride]
    summary = overall(records)
    row = {
        "run": run,
        "run_dir": directory.relative_to(Path(root)).as_posix(),
        "checkpoint": f"{checkpoint}.pt",
        "checkpoint_epoch": metrics["checkpoint_epoch"],
        "parameters": metrics["parameters"],
        "n_points": summary["n_points"],
        "n_entries": len(records),
        "overall_rmse": summary["overall_rmse"],
        "overall_r2": summary["overall_r2"],
        "eta_rmse_mm": eta_rmse_mm(records),
    }
    for name in ("overall_rmse", "overall_r2", "eta_rmse_mm"):
        if not finite(row[name]):
            raise RuntimeError(f"{run} 的 {name} 非有限值：{row[name]}")
    if all("hull" in record for record in records.values()):
        rmse, sigma, count = pooled_channel(records, 0, group="hull")
        row["hull"] = {"n_points": count, "cp_rmse": rmse, "cp_nrmse": rmse / sigma,
                       "cp_r2": 1.0 - (rmse / sigma) ** 2}
    return {"config": metrics["config"], **row}


def mean_std(values):
    """样本均值、样本方差与样本标准差（`ddof=1`），另给极值。"""
    return {
        "mean": statistics.fmean(values),
        "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "values": list(values),
    }


def load_complexity(root):
    """`scripts/eval/model_complexity.json`：标定的 GFLOPs、主干 token 数与 token 口径。"""
    return json.loads((Path(root) / COMPLEXITY_FILE).read_text())


def complexity_of(table, run):
    """按 run 名取标定值；三个划分与训练种子共用同一条目，缺项时报错，不静默填 0。"""
    entries = table["entries"]
    stem = TRAIN_SEED_TOKEN.sub("", run)
    candidates = [run, stem]
    for split in SPLIT_TOKENS[:-1]:
        if f"_{split}" in stem:
            candidates.append(stem.replace(f"_{split}", f"_{SPLIT_TOKENS[-1]}"))
    for candidate in candidates:
        if candidate in entries:
            return entries[candidate]
    raise KeyError(f"{COMPLEXITY_FILE.as_posix()} 缺少 {run} 的标定值")


def training_tokens(tokens, n_entries):
    """训练 token 总量 = token@stride1 × 训练条目数 × 100 epoch。"""
    return int(tokens) * int(n_entries) * TRAIN_STEPS_PER_ENTRY


def seed_rows(root, runs):
    """逐种子指标行（`runs` 按训练种子顺序给出）。"""
    return [run_metrics(root, run) for run in runs]


def model_row(root, label, runs, table, n_train, reference=None):
    """一个模型（或一个消融变体）的三种子行与成本字段。

    `runs` 按训练种子 0/1/2 顺序；`reference` 给出对照行的整体 R² 均值与 η RMSE 均值时，
    额外写出相对变化（`contrast`）。
    """
    seeds = seed_rows(root, runs)
    base = complexity_of(table, runs[0])
    points = {seed["n_points"] for seed in seeds}
    if len(points) != 1:
        raise RuntimeError(f"{runs[0]} 的种子间留出点数不一致：{sorted(points)}")
    counts = {seed["n_entries"] for seed in seeds}
    if len(counts) != 1:
        raise RuntimeError(f"{runs[0]} 的种子间留出条目数不一致：{sorted(counts)}")
    parameters = [seed["parameters"] for seed in seeds]
    row = {
        "label": label,
        "runs": list(runs),
        "configs": [seed["config"] for seed in seeds],
        "parameters": parameters[0],
        "parameters_per_seed": parameters,
        "parameters_identical": len(set(parameters)) == 1,
        "tokens_stride1": base["tokens"],
        "token_kind": base["token_kind"],
        "gflops": base["gflops"],
        "gflops_status": base.get("gflops_status"),
        "training_tokens": training_tokens(base["tokens"], n_train),
        "train_entries": n_train,
        "n_points": seeds[0]["n_points"],
        "n_entries": seeds[0]["n_entries"],
        "overall_rmse": mean_std([seed["overall_rmse"] for seed in seeds]),
        "overall_r2": mean_std([seed["overall_r2"] for seed in seeds]),
        "eta_rmse_mm": mean_std([seed["eta_rmse_mm"] for seed in seeds]),
        "seeds": seeds,
    }
    if all("hull" in seed for seed in seeds):
        row["hull"] = {
            "cp_rmse": mean_std([seed["hull"]["cp_rmse"] for seed in seeds]),
            "cp_r2": mean_std([seed["hull"]["cp_r2"] for seed in seeds]),
            "n_points": seeds[0]["hull"]["n_points"],
        }
    if reference is not None:
        row["contrast"] = {
            "reference": reference["label"],
            "overall_r2_change": row["overall_r2"]["mean"] - reference["overall_r2"]["mean"],
            "overall_rmse_change": (row["overall_rmse"]["mean"]
                                    - reference["overall_rmse"]["mean"]),
            "eta_rmse_change_mm": (row["eta_rmse_mm"]["mean"]
                                   - reference["eta_rmse_mm"]["mean"]),
        }
    return row


def model_rows(root, labels, runs_of, table, n_train, reference_label=None):
    """按 `labels` 顺序建行；`reference_label` 指定对照（对照行自身不带 `contrast`）。"""
    rows = [model_row(root, label, runs_of(label), table, n_train)
            for label in labels]
    by_label = {row["label"]: row for row in rows}
    if reference_label is not None:
        reference = by_label[reference_label]
        for row in rows:
            if row is reference:
                continue
            row["contrast"] = {
                "reference": reference["label"],
                "overall_r2_change": (row["overall_r2"]["mean"]
                                      - reference["overall_r2"]["mean"]),
                "overall_rmse_change": (row["overall_rmse"]["mean"]
                                        - reference["overall_rmse"]["mean"]),
                "eta_rmse_change_mm": (row["eta_rmse_mm"]["mean"]
                                       - reference["eta_rmse_mm"]["mean"]),
            }
    return rows


def ranks(rows, key="overall_r2"):
    """逐训练种子的整体 R² 名次（1 最好）与三种子平均名次。"""
    seeds = len(rows[0]["seeds"])
    places = {row["label"]: [] for row in rows}
    for index in range(seeds):
        ranked = sorted(rows, key=lambda row: row["seeds"][index][key], reverse=True)
        for place, row in enumerate(ranked, start=1):
            places[row["label"]].append(place)
    return {label: {"per_seed": values, "mean": statistics.fmean(values)}
            for label, values in places.items()}


def sort_by_mean(rows, key="overall_r2"):
    return sorted(rows, key=lambda row: row[key]["mean"], reverse=True)


def baseline_metadata(root):
    """本轮统一报告口径的公共字段（写进每份汇总 json）。"""
    return {
        "source": f"output/<配置名>/metrics_{PRIMARY}.json 的 strides.val.stride{STRIDE}"
                  f"（留出集全密度查询）",
        "aggregation": "src/eval/aggregate.py：逐条目点数加权池化 → 五通道 NRMSE → "
                       "整体 RMSE / 整体 R²；η RMSE 取 η 通道池化 RMSE（mm）",
        "statistics": "ddof=1 的样本均值、样本方差与样本标准差，n=3 个训练种子",
        "checkpoint_rule": f"{PRIMARY}.pt；查询密度固定 stride {STRIDE}",
        "protocol_checks": "每个 run 校验 train_strides=[1]、评测密度只有 stride 1、"
                           "checkpoint=best.pt，留出点数与条目数在三种子间一致",
        "complexity_source": f"{COMPLEXITY_FILE.as_posix()}：GFLOPs 与主干 token 数取 "
                             f"docs/isoflops_model_complexity.md 的标定值，参数量取各 run "
                             f"的 metrics 字段",
        "training_token_protocol": f"训练 token 总量 = token@stride1 × 训练条目数 × "
                                   f"{TRAIN_STEPS_PER_ENTRY} epoch",
    }
