"""自由变形（Free-Form Deformation）与 KCS 几何变体族。

Sederberg & Parry (1986) 的 FFD：把几何嵌入一个长方体控制格架，移动控制点后，
格架内任意点的位移由三元张量积 Bernstein 多项式插值给出

    X' = X + Σ_i Σ_j Σ_k B_i(s) B_j(t) B_k(u) · D_ijk

s、t、u 是点在格架内的归一化坐标，D_ijk 是控制点位移。位移场在格架内处处光滑，
三角形连接关系不变的曲面变形后仍然封闭。

控制点位移由 `fit_control_displacements` 反解：变体先给出船体表面要达到的目标位移，
再用 Bernstein 基做最小二乘求出控制点位移。直接手写控制点位移会被 Bernstein 平滑
削弱到设计值的三成左右（局部变体更少），反解让变体幅度可控。

KCS 变体族的约定：

- 格架取船体 STL 的包围盒，整船都在格架内，干舷与水线一起随动，几何处处连续。
- 控制点数 `KCS_LATTICE = (9, 5, 6)`：沿 x 8 阶可分辨艏、舯、艉与球鼻，
  沿 y 4 阶（中纵剖面与两舷），沿 z 5 阶（船底、舭部、水线附近、甲板）。
- 沿 y 的位移按 `sign(y)` 施加，左右舷对称；中纵剖面上的点不动。
- 目标位移取 20–60 mm（船长的 0.3%–0.8%），湿面积变化 1%–5%，
  snappyHexMesh 仍能贴合。
"""

from math import comb

import numpy as np

KCS_LATTICE = (9, 5, 6)


def bernstein_matrix(t, count):
    """次数 count-1 的 Bernstein 基：输入 (N,) 的归一化坐标，返回 (N, count)。"""
    degree = count - 1
    index = np.arange(count, dtype=np.float64)
    coefficient = np.array([comb(degree, k) for k in range(count)], dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[:, None]
    return coefficient * t ** index * (1.0 - t) ** (degree - index)


def control_lattice(bounds, shape=KCS_LATTICE):
    """控制点坐标，形状 (shape[0], shape[1], shape[2], 3)。"""
    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, shape)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def normalized_coordinates(points, bounds):
    """点在格架内的归一化坐标 (N,3)；格架外的分量会落在 [0,1] 之外。"""
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack([
        (points[:, axis] - lo) / (hi - lo) for axis, (lo, hi) in enumerate(bounds)])


def fit_control_displacements(points, bounds, target, shape=KCS_LATTICE):
    """由采样点上的目标位移反解控制点位移，返回 (shape[0], shape[1], shape[2], 3)。"""
    local = normalized_coordinates(points, bounds)
    basis = [bernstein_matrix(local[:, axis], shape[axis]) for axis in range(3)]
    design = np.einsum("ni,nj,nk->nijk", *basis).reshape(len(local), -1)
    solution, *_ = np.linalg.lstsq(design, np.asarray(target, dtype=np.float64), rcond=None)
    return solution.reshape(tuple(shape) + (3,))


def apply_ffd(points, bounds, displacements):
    """格架内的点按 Bernstein 位移场变形；格架外的点原样返回。"""
    points = np.asarray(points, dtype=np.float64)
    local = normalized_coordinates(points, bounds)
    inside = np.all((local >= 0.0) & (local <= 1.0), axis=1)
    shifted = points.copy()
    if inside.any():
        basis = [bernstein_matrix(local[inside, axis], displacements.shape[axis])
                 for axis in range(3)]
        shifted[inside] += np.einsum("ni,nj,nk,ijkc->nc", *basis, displacements)
    return shifted


def hull_bounds(points):
    """船体 STL 的包围盒，作为控制格架范围。"""
    points = np.asarray(points, dtype=np.float64)
    return tuple((float(points[:, axis].min()), float(points[:, axis].max()))
                 for axis in range(3))


def _normalized(points, bounds):
    x0, x1 = bounds[0]
    z0, z1 = bounds[2]
    return (points[:, 0] - x0) / (x1 - x0), (points[:, 2] - z0) / (z1 - z0)


def _bell(value, center, width):
    """以 center 为中心、width 为宽度的平滑包络，用于把位移限制在局部。"""
    return np.exp(-((value - center) / width) ** 2)


def _centerline_taper(points, bounds, width=0.25):
    """中纵剖面附近的横向位移衰减。

    艏柱与艉柱处左右舷的 |y| 趋近于 0，若横向位移不减到 0，两舷会互相穿透、
    三角形翻转。用平方衰减保证 y = 0 处位移与一阶导数都为 0，Bernstein 基拟合
    不会在艏艉过冲。
    """
    half_beam = (bounds[1][1] - bounds[1][0]) / 2.0
    return np.clip((np.abs(points[:, 1]) / (width * half_beam)) ** 2, 0.0, 1.0)


def wide_midship(points, bounds, amount=0.030):
    """全局：舯部半宽外扩 30 mm。"""
    sx, _ = _normalized(points, bounds)
    target = np.zeros_like(points)
    target[:, 1] = np.sign(points[:, 1]) * amount * _bell(sx, 0.5, 0.28) \
        * _centerline_taper(points, bounds)
    return target


def narrow_entrance(points, bounds, amount=0.032):
    """全局：艏部进流段半宽内收 32 mm，从舯部平滑过渡到艏柱。"""
    sx, _ = _normalized(points, bounds)
    target = np.zeros_like(points)
    target[:, 1] = -np.sign(points[:, 1]) * amount * _bell(sx, 0.86, 0.22) \
        * _centerline_taper(points, bounds)
    return target


def full_stern(points, bounds, amount=0.028):
    """全局：艉部半宽外扩 28 mm，改善艉部伴流。"""
    sx, _ = _normalized(points, bounds)
    target = np.zeros_like(points)
    target[:, 1] = np.sign(points[:, 1]) * amount * _bell(sx, 0.12, 0.20) \
        * _centerline_taper(points, bounds)
    return target


def shallow_body(points, bounds, amount=0.025):
    """全局：船底整体上抬 25 mm，减小方形系数。"""
    sx, sz = _normalized(points, bounds)
    target = np.zeros_like(points)
    target[:, 2] = amount * _bell(sz, 0.0, 0.32) * (0.4 + 0.6 * _bell(sx, 0.5, 0.45))
    return target


def deep_forefoot(points, bounds, amount=0.022):
    """全局：艏部前踵下沉 22 mm，增大艏部排水量。"""
    sx, sz = _normalized(points, bounds)
    target = np.zeros_like(points)
    target[:, 2] = -amount * _bell(sx, 0.88, 0.20) * _bell(sz, 0.10, 0.28)
    return target


def bow_bulb_protrusion(points, bounds, amount=0.060):
    """局部：球鼻艏沿 +x 前伸 60 mm。"""
    sx, sz = _normalized(points, bounds)
    target = np.zeros_like(points)
    target[:, 0] = amount * _bell(sx, 1.0, 0.16) * _bell(sz, 0.16, 0.20)
    return target


def bilge_rounding(points, bounds, amount=0.022):
    """局部：舭部外扩 22 mm 并上抬 13 mm，增大舭部圆角。"""
    sx, sz = _normalized(points, bounds)
    weight = _bell(sx, 0.55, 0.24) * _bell(sz, 0.10, 0.20)
    target = np.zeros_like(points)
    target[:, 1] = np.sign(points[:, 1]) * amount * weight * _centerline_taper(points, bounds)
    target[:, 2] = 0.6 * amount * weight
    return target


def stern_dent(points, bounds, amount=0.020):
    """局部：艉部舭部区内凹 20 mm。"""
    sx, sz = _normalized(points, bounds)
    target = np.zeros_like(points)
    target[:, 1] = -np.sign(points[:, 1]) * amount * _bell(sx, 0.10, 0.14) * _bell(sz, 0.25, 0.22) \
        * _centerline_taper(points, bounds)
    return target


def bow_flare_underwater(points, bounds, amount=0.025):
    """局部：艏部水线附近水下外飘 25 mm。"""
    sx, sz = _normalized(points, bounds)
    target = np.zeros_like(points)
    target[:, 1] = np.sign(points[:, 1]) * amount * _bell(sx, 0.88, 0.18) * _bell(sz, 0.42, 0.18) \
        * _centerline_taper(points, bounds)
    return target


# 变体名 → (目标位移函数, 中文说明)。9 个变体：5 个全局、4 个局部
KCS_VARIANTS = {
    "wide_midship": (wide_midship, "全局：舯部半宽外扩 30 mm"),
    "narrow_entrance": (narrow_entrance, "全局：艏部进流段内收 32 mm"),
    "full_stern": (full_stern, "全局：艉部半宽外扩 28 mm"),
    "shallow_body": (shallow_body, "全局：船底上抬 25 mm"),
    "deep_forefoot": (deep_forefoot, "全局：艏部前踵下沉 22 mm"),
    "bow_bulb_protrusion": (bow_bulb_protrusion, "局部：球鼻艏前伸 60 mm"),
    "bilge_rounding": (bilge_rounding, "局部：舭部外扩上抬 22 mm"),
    "stern_dent": (stern_dent, "局部：艉部舭部区内凹 20 mm"),
    "bow_flare_underwater": (bow_flare_underwater, "局部：艏部水下外飘 25 mm"),
}


def kcs_displacements(variant, points, bounds):
    """变体的控制点位移 (shape[0], shape[1], shape[2], 3)，由船体表面目标位移反解。"""
    return fit_control_displacements(points, bounds, KCS_VARIANTS[variant][0](points, bounds))
