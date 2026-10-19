"""APP 分区：把点集划成物理尺寸受控的 patch。

停止判据是 patch 的**物理边长** `max_side × scale`，与查询点集的密度解耦：点集
加密时每个 patch 的点数变多、patch 边界不变。v1 的"每 patch 点数上限"判据随查询
密度漂移，已废弃。

参数域本身就是物理长度时（流场域按 Lpp 归一化）`scale = 1`，判据严格只依赖几何。
参数域被拉伸时（船体 UV 域，船艏被展宽）`scale = None`，由 patch 的 `ΣΔ` 反推
局地尺度，船艏因此被细分得更细。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Partition:
    order: np.ndarray      # (N,) 点索引，按 patch 连续排列
    starts: np.ndarray     # (P+1,) 每个 patch 在 order 中的起止
    centres: np.ndarray    # (P,2) Δ 加权形心
    areas: np.ndarray      # (P,) ΣΔ，即 patch 覆盖的物理面积
    bounds: np.ndarray     # (P,4) 参数域包围盒 x0, y0, x1, y1
    levels: np.ndarray     # (P,) 二分次数


def tiered_tolerance(key, tiers):
    """按分段常数把 key 映射成允许的 patch 边长；tiers 形如 [[key_max, tol], ..., [None, tol]]。"""
    tolerance = np.full(key.shape, tiers[-1][1], dtype=np.float64)
    for key_max, value in reversed(tiers[:-1]):
        tolerance = np.where(key <= key_max, value, tolerance)
    return tolerance


def build_partition(points, weights, extent, tolerance, scale=1.0, max_level=12):
    """按物理边长把点集划成 patch。

    points (N,2) 参数域坐标，weights (N,) 求积权重（物理面积），
    extent (x0, y0, x1, y1) 参数域范围，tolerance (N,) 每点允许的 patch 物理边长，
    scale 为参数域单位对应的物理长度（`None` 表示由 `ΣΔ` 反推）。
    `max_level` 是二分次数的安全上限。
    """
    x0, y0, x1, y1 = extent
    resolution = 2 ** max_level
    column = np.clip(((points[:, 0] - x0) / (x1 - x0) * resolution).astype(np.int64),
                     0, resolution - 1)
    row = np.clip(((points[:, 1] - y0) / (y1 - y0) * resolution).astype(np.int64),
                  0, resolution - 1)
    span_x = (x1 - x0) / resolution
    span_y = (y1 - y0) / resolution

    leaves = []
    stack = [(np.arange(len(points)), 0, 0, resolution, 0, resolution)]
    while stack:
        indices, level, column_lo, column_hi, row_lo, row_hi = stack.pop()
        if len(indices) == 0:
            continue
        side = max((column_hi - column_lo) * span_x, (row_hi - row_lo) * span_y)
        if scale is None:
            area = weights[indices].sum()
            side *= np.sqrt(area / ((column_hi - column_lo) * span_x
                                    * (row_hi - row_lo) * span_y))
        else:
            side *= scale
        if side <= tolerance[indices].min() or level >= max_level:
            leaves.append((indices, level, column_lo, column_hi, row_lo, row_hi))
            continue
        if (column_hi - column_lo) * span_x >= (row_hi - row_lo) * span_y:
            middle = (column_lo + column_hi) // 2
            left = indices[column[indices] < middle]
            stack.append((left, level + 1, column_lo, middle, row_lo, row_hi))
            stack.append((indices[column[indices] >= middle], level + 1, middle, column_hi,
                          row_lo, row_hi))
        else:
            middle = (row_lo + row_hi) // 2
            stack.append((indices[row[indices] < middle], level + 1, column_lo, column_hi,
                          row_lo, middle))
            stack.append((indices[row[indices] >= middle], level + 1, column_lo, column_hi,
                          middle, row_hi))

    order = np.concatenate([leaf[0] for leaf in leaves])
    starts = np.zeros(len(leaves) + 1, dtype=np.int64)
    np.cumsum([len(leaf[0]) for leaf in leaves], out=starts[1:])
    areas = np.add.reduceat(weights[order], starts[:-1])
    centres = np.column_stack([
        np.add.reduceat(points[order, axis] * weights[order], starts[:-1]) / areas
        for axis in range(2)
    ])
    bounds = np.array([[x0 + span_x * leaf[2], y0 + span_y * leaf[4],
                        x0 + span_x * leaf[3], y0 + span_y * leaf[5]] for leaf in leaves])
    return Partition(
        order=order,
        starts=starts,
        centres=centres,
        areas=areas,
        bounds=bounds,
        levels=np.array([leaf[1] for leaf in leaves], dtype=np.int64),
    )
