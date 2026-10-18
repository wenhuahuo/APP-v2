"""自由液面格点到船体水线投影轮廓的符号距离。

二维流场域的几何通道。船体投影轮廓取吃水以下船体表面点的最外缘 y(x)，与对称面
y = 0 组成封闭区域：轮廓之内（船体正下方）没有自由液面，符号取负，该处的格点
在组装数据集时被掩除。
"""

import numpy as np
from scipy.spatial import cKDTree


def footprint_profile(points, draft, lpp):
    """吃水以下船体点的最外缘 y(x) 折线，返回 (x, y)。"""
    submerged = points[points[:, 2] <= draft]
    x = submerged[:, 0]
    y = submerged[:, 1]

    n_bins = max(int(round((x.max() - x.min()) / (0.002 * lpp))), 1)
    edge = np.linspace(x.min(), x.max(), n_bins + 1)
    index = np.clip(np.digitize(x, edge) - 1, 0, n_bins - 1)
    outer = np.full(n_bins, np.inf)
    np.minimum.at(outer, index, y)
    filled = np.bincount(index, minlength=n_bins) > 0

    centre = 0.5 * (edge[:-1] + edge[1:])
    return centre[filled], outer[filled]


def _resample(x, y, spacing):
    """按弧长等距重采样折线，供最近邻查询使用。"""
    segment = np.hypot(np.diff(x), np.diff(y))
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    count = max(int(round(arc[-1] / spacing)), 2)
    target = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(target, arc, x), np.interp(target, arc, y)])


def signed_distance(points, profile, lpp):
    """格点 (x,y) 到船体投影轮廓的距离，船体投影内部为负。"""
    outline = _resample(*profile, spacing=0.0005 * lpp)
    distance, _ = cKDTree(outline).query(points)

    x, y = profile
    inside = (points[:, 0] > x.min()) & (points[:, 0] < x.max())
    inside &= points[:, 1] > np.interp(points[:, 0], x, y)
    return np.where(inside, -distance, distance)
