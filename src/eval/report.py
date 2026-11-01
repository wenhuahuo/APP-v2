"""把逐条目指标汇总成控制台表格：每行一个 `stride`，单元格是条目间的 mean±std。

R² 以真值的加权方差为基准、上界 1，越高越好；RMSE 与物理量同单位、越低越好。
条目数很少、各条目点数差异大，这里不按点数加权，直接报条目间的总体标准差
（除以 n，而不是样本标准差）。
"""

import statistics

from metrics import FLOW_CHANNELS

# (表头, RMSE 的量纲缩放, 单位)：eta 换成 mm 便于阅读，速度用 m/s，压力系数无量纲。
COLUMNS = (
    ("eta", 1000.0, "mm"),
    ("u", 1.0, "m/s"),
    ("v", 1.0, "m/s"),
    ("w", 1.0, "m/s"),
    ("Cp", 1.0, "-"),
    ("hull Cp", 1.0, "-"),
)
# 前 FLOW_COLUMNS 列对应流场通道（p_rgh 即液面 Cp），最后一列是船面 Cp。
FLOW_COLUMNS = len(FLOW_CHANNELS)
CELL = 16


def mean_std(values):
    """条目间均值与总体标准差。"""
    return statistics.fmean(values), statistics.pstdev(values)


def channel_values(records, column, quantity):
    """`records` 是 {条目: 指标}；前 5 列取流场通道，最后一列取船面 Cp。"""
    index, group = (column, "flow") if column < FLOW_COLUMNS else (0, "hull")
    return [record[group][quantity][index] for record in records.values()]


def _heading(label, unit, quantity):
    return label if quantity == "r2" else f"{label}[{unit}]"


def _cell(values, scale, quantity):
    mean, std = mean_std(values)
    if quantity == "r2":
        return f"{mean:+.4f}±{std:.4f}"
    return f"{mean * scale:.4f}±{std * scale:.4f}"


def table_lines(split, entries, blocks, strides, quantity):
    """一个划分、一种指标（`r2` 或 `rmse`）的表；`blocks` 是 {stride: {条目: 指标}}。"""
    title = "R²↑（越高越好）" if quantity == "r2" else "RMSE↓（越低越好）"
    header = f"{'stride':>8s}" + "".join(
        f"{_heading(label, unit, quantity):>{CELL}s}" for label, _, unit in COLUMNS)
    lines = [f"\n{split}（{len(entries)} 个条目）  {title}", header]
    for stride in strides:
        values = [channel_values(blocks[stride], column, quantity)
                  for column in range(len(COLUMNS))]
        line = f"{stride:^8d}" + "".join(
            f"{_cell(value, scale, quantity):>{CELL}s}"
            for value, (_, scale, _) in zip(values, COLUMNS))
        lines.append(line)
    return lines


def summary_lines(splits, blocks, strides, quantities=("r2", "rmse")):
    """`splits` 是 {划分: 条目列表}，`blocks` 是 {划分: {stride: {条目: 指标}}}。"""
    lines = []
    for split, entries in splits.items():
        for quantity in quantities:
            lines.extend(table_lines(split, entries, blocks[split], strides, quantity))
    return lines
