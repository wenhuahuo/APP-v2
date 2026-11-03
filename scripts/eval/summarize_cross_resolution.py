"""汇总两种容量口径下的跨分辨率精度与单条目推理延迟。

数据来源（全部为已落盘的评测产物）：

- `output/<配置名>/metrics_best.json`：训练后同步抽样评测，输入与查询同时按 stride
  1/2/4/8 抽稀，18 个 run 全部有；
- `output/<配置名>/metrics_best_encode1.json`：输入编码固定 stride=1、只改查询密度的
  评测（`scripts/eval/evaluate_model.py --encode-stride 1 --splits val`），6 个独立
  查询模型 × 2 种容量共 12 个 run；
- `output/<配置名>/inference_time.json`：
  `scripts/eval/measure_inference_time.py` 的单条目前向延迟。

精度按 `src/eval/aggregate.py` 的池化公式汇总成整体 R²/整体 RMSE，与主表同一口径。
脚本只读产物、不改动它们；缺文件时明确报错。默认打印 markdown 表，`--json` 另存一份
汇总结果。

用法:
    python3 scripts/eval/summarize_cross_resolution.py
    python3 scripts/eval/summarize_cross_resolution.py --json /tmp/cross_resolution.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))

from aggregate import eta_rmse_mm, overall  # noqa: E402

MODELS = ("appsolver_v2", "appsolver_oformer", "appsolver_v1", "gnot", "upt",
          "transolver", "oformer", "gino", "coral")
CAPACITIES = (("固定参数量", ""), ("固定计算量", "_isoflops"))
STRIDES = (1, 2, 4, 8)
ENCODE_MODELS = ("appsolver_v2", "appsolver_oformer", "oformer", "upt", "gino", "coral")
HEADER_STRIDES = tuple(f"stride {stride}" for stride in STRIDES)


def stem(model, suffix):
    return f"{model}_random_seed0{suffix}"


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少评测产物：{path}")
    return json.loads(path.read_text())


def metrics_blocks(stem_name, filename):
    """读一份评测产物的 val 逐条目指标，按 stride 汇总成整体指标。"""
    metrics = read_json(ROOT / "output" / stem_name / filename)
    records = metrics["strides"]["val"]
    return {stride: {"overall": overall(records[str(stride)]),
                     "eta_mm": eta_rmse_mm(records[str(stride)])}
            for stride in STRIDES}


def sync_row(model, label, suffix):
    blocks = metrics_blocks(stem(model, suffix), "metrics_best.json")
    return {"capacity": label, "model": model,
            "r2": {stride: blocks[stride]["overall"]["overall_r2"] for stride in STRIDES},
            "rmse": {stride: blocks[stride]["overall"]["overall_rmse"] for stride in STRIDES},
            "eta_mm": {stride: blocks[stride]["eta_mm"] for stride in STRIDES},
            "n_points": {stride: blocks[stride]["overall"]["n_points"]
                         for stride in STRIDES}}


def encode_row(model, label, suffix):
    name = stem(model, suffix)
    metrics = read_json(ROOT / "output" / name / "metrics_best_encode1.json")
    if metrics["encode_stride"] != 1:
        raise ValueError(f"{name} 的 encode_stride 不是 1：{metrics['encode_stride']}")
    blocks = metrics_blocks(name, "metrics_best_encode1.json")
    return {"capacity": label, "model": model,
            "r2": {stride: blocks[stride]["overall"]["overall_r2"] for stride in STRIDES},
            "rmse": {stride: blocks[stride]["overall"]["overall_rmse"] for stride in STRIDES},
            "n_points": {stride: blocks[stride]["overall"]["n_points"]
                         for stride in STRIDES}}


def latency_row(model, label, suffix):
    record = read_json(ROOT / "output" / stem(model, suffix) / "inference_time.json")
    return {"capacity": label, "model": model,
            "ms": {stride: 1000.0 * record["strides"][str(stride)]["mean"]
                   for stride in STRIDES},
            "std_ms": {stride: 1000.0 * record["strides"][str(stride)]["std"]
                       for stride in STRIDES},
            "query_points": {stride: record["strides"][str(stride)]["query_points"]
                             for stride in STRIDES},
            "sweep_means": {stride: [1000.0 * value for value in
                                      record["strides"][str(stride)]["sweep_means"]]
                            for stride in STRIDES},
            "device": record["device"], "torch": record["torch"],
            "cuda": record["cuda"], "warmup": record["warmup"],
            "repeat": record["repeat"], "sweeps": record["sweeps"],
            "entry": record["entry"]}


def rank(rows):
    return sorted(rows, key=lambda row: row["r2"][1], reverse=True)


def table(rows, header, cells):
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(cells(row)) + " |" for row in rows)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="把汇总结果写进这个 json")
    args = parser.parse_args()

    precision = []
    encode = []
    latency = []
    for label, suffix in CAPACITIES:
        for model in MODELS:
            precision.append(sync_row(model, label, suffix))
            if model in ENCODE_MODELS:
                encode.append(encode_row(model, label, suffix))
            latency.append(latency_row(model, label, suffix))

    for label, _ in CAPACITIES:
        rows = rank([row for row in precision if row["capacity"] == label])
        print(f"\n### 同步抽样整体 R²（{label}）\n")
        print(table(rows, ("模型",) + HEADER_STRIDES,
                    lambda row: [row["model"]]
                    + [f"{row['r2'][stride]:+.4f}" for stride in STRIDES]))
        print(f"\n### 同步抽样整体 RMSE %（{label}）\n")
        print(table(rows, ("模型",) + HEADER_STRIDES,
                    lambda row: [row["model"]]
                    + [f"{row['rmse'][stride]:.2f}" for stride in STRIDES]))

    for label, _ in CAPACITIES:
        rows = rank([row for row in encode if row["capacity"] == label])
        print(f"\n### 固定输入编码 stride=1，整体 R²（{label}）\n")
        print(table(rows, ("模型",) + HEADER_STRIDES,
                    lambda row: [row["model"]]
                    + [f"{row['r2'][stride]:+.4f}" for stride in STRIDES]))

    print("\n### 单条目推理延迟（GPU，ms）\n")
    query_line = "、".join(f"stride {stride}：{latency[0]['query_points'][stride]}"
                           for stride in STRIDES)
    print(f"查询点数 {query_line}；均值，逐次样本见各 `inference_time.json`。")
    for label, _ in CAPACITIES:
        order = {row["model"]: index
                 for index, row in enumerate(rank([row for row in precision
                                                   if row["capacity"] == label]))}
        rows = sorted([row for row in latency if row["capacity"] == label],
                      key=lambda row: order[row["model"]])
        print(f"\n{label}：\n")
        print(table(rows, ("模型",) + HEADER_STRIDES,
                    lambda row: [row["model"]]
                    + [f"{row['ms'][stride]:.2f}" for stride in STRIDES]))

    checks = []
    for row in encode:
        sync = next(item for item in precision
                    if item["capacity"] == row["capacity"] and item["model"] == row["model"])
        checks.append({
            "config": stem(row["model"], "" if row["capacity"] == "固定参数量" else "_isoflops"),
            "stride1_r2_delta": abs(row["r2"][1] - sync["r2"][1]),
            "query_points_match": all(row["n_points"][stride] == sync["n_points"][stride]
                                      for stride in STRIDES),
        })
    print("\n### 核验\n")
    print(f"固定输入编码在 stride 1 与同步抽样使用同一输入、同一查询：12 个 run 的整体 R² "
          f"最大差 {max(check['stride1_r2_delta'] for check in checks):.2e}；各 stride 的查询"
          f"点数全部一致：{all(check['query_points_match'] for check in checks)}。")
    devices = sorted({row["device"] for row in latency})
    deviations = []
    for row in latency:
        for stride in STRIDES:
            means = row["sweep_means"][stride]
            centre = statistics.median(means)
            deviations.append(statistics.median([abs(value - centre) for value in means])
                              / centre)
    print(f"延迟设备 {'；'.join(devices)}；PyTorch {latency[0]['torch']}、CUDA "
          f"{latency[0]['cuda']}；warmup {latency[0]['warmup']}、repeat "
          f"{latency[0]['repeat']}、sweeps {latency[0]['sweeps']}；条目 "
          f"{latency[0]['entry']}；同一格逐轮均值的中位绝对偏差最大 "
          f"{100 * max(deviations):.2f}%。")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"precision": precision, "encode": encode, "latency": latency,
             "checks": checks}, indent=2, ensure_ascii=False) + "\n")
        print(f"\n汇总写入 {args.json}")


if __name__ == "__main__":
    main()
