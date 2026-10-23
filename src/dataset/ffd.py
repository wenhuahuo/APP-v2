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
- 变体是**全局参数级**变形：半宽缩放（按局部半宽比例 `|y|/half_beam`）、吃水平移
  （整体 z 平移）与舯部中体形状（钟形包络内的半宽缩放）三类，位移 40–130 mm
  （船长的 0.5%–1.8%），湿面积变化 ±12%~±16%。

为什么取 10% 量级的湿面积变化：KCS 周长约 2(T+B/2) = 1.703 m，湿面积 8.35 m²。
湿面积变 X% 对应自由面波高（Fr=0.26 约 0.11 m）同量级的 X% 位移信号，即约 X% × 波高。
位移 20–60 mm、湿面积 0.2%–2.7% 的第一批变体，eta 场 RMS 差只有 1.0–1.6 mm，
与 20s/25s 截断误差（0.97 mm）同量级，模型无法区分；湿面积 ≥10% 才让信号
（≥10 mm）高出截断误差一个量级。

吃水变化用 z 平移实现（水面 z = `KCS_DRAFT` 不动、船体整体下沉/上浮），这样水面位置
不变，`build_entry.py` 的湿表面判断（z ≤ 0.3418）与 setFields 的 boxToCell 上限都不用改。
"""

from math import comb

import numpy as np

KCS_LATTICE = (9, 5, 6)

# KCS 设计吃水（水面 z 坐标，龙骨在 z = 0）。吃水变化用 z 平移，水面本身不动。
KCS_DRAFT = 0.3418


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


def _smoothstep(edge0, edge1, value):
    """C1 连续的一次过渡：value ≤ edge0 取 0，≥ edge1 取 1。"""
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _beam_shift(points, bounds, ratio, envelope=None):
    """半宽按局部半宽比例缩放 ratio 的横向位移（左右舷对称，艏艉中线衰减）。

    位移取 `ratio × |y|`：船体表面每点的横向外扩/内收量正比于该点的局部半宽，
    于是横剖面形状保持相似，半宽整体缩放 ratio。`envelope` 给出沿 x 的包络，
    用来把变形限制在中体（改方形系数）。`_centerline_taper` 在艏柱/艉柱处把位移
    衰减到 0，否则左右舷互相穿透、三角形翻转。
    """
    weight = np.abs(points[:, 1])
    if envelope is not None:
        weight = weight * envelope(_normalized(points, bounds)[0])
    return np.sign(points[:, 1]) * ratio * weight * _centerline_taper(points, bounds)


def _midship_envelope(sx):
    """中体包络：sx ∈ [0.25, 0.75] 取 1，两端 smoothstep 过渡到 0（sx ≤ 0.05、≥ 0.95）。"""
    return _smoothstep(0.05, 0.25, sx) * (1.0 - _smoothstep(0.75, 0.95, sx))


def _dry_taper(points):
    """水线以上把横向位移平滑衰减到 0（水线处为 1，z ≥ 水线 + 0.1 m 处为 0）。

    STL 的甲板平面是从艉部一点扇形展开到甲板边线的一批大长条三角形（边长达 5 m、
    宽度只有 15 mm）。任何沿 x 变化的横向位移都会让长条的两端错动而翻面——这是粗
    甲板三角化的产物，不是光滑映射的折叠，却会让 snappyHexMesh 在干舷侧出现伪
    相交。中体变形的横向位移只在湿面以下起作用，水线以上保留原宽度即可。
    """
    return 1.0 - _smoothstep(KCS_DRAFT, KCS_DRAFT + 0.10, points[:, 2])


def _draft_shift(points, ratio):
    """吃水变化 ratio（相对设计吃水）的 z 平移：吃水增加 = 船体下沉 = z 减小。

    水面（z = `KCS_DRAFT`）不动、船体整体平移，均匀位移用 Bernstein 基可以精确表示
    （基函数和为 1），拟合残差为 0。
    """
    return np.full(len(points), -ratio * KCS_DRAFT, dtype=np.float64)


def _global_form(points, bounds, beam=0.0, draft=0.0, midship=0.0):
    """全局参数级变形的合成：整体半宽缩放 + 吃水平移 + 中体半宽缩放。"""
    target = np.zeros_like(points)
    if beam:
        target[:, 1] += _beam_shift(points, bounds, beam)
    if midship:
        target[:, 1] += _beam_shift(points, bounds, midship,
                                    envelope=_midship_envelope) * _dry_taper(points)
    if draft:
        target[:, 2] += _draft_shift(points, draft)
    return target


def fat_hull(points, bounds, beam=0.12, draft=0.12):
    """全局：半宽 +12% + 吃水 +12%，湿面积 +15.6%。"""
    return _global_form(points, bounds, beam=beam, draft=draft)


def thin_hull(points, bounds, beam=-0.12, draft=-0.12):
    """全局：半宽 −12% + 吃水 −12%，湿面积 −12.9%。"""
    return _global_form(points, bounds, beam=beam, draft=draft)


def wide_beam(points, bounds, beam=0.20):
    """全局：半宽 +20%，湿面积 +12.9%。"""
    return _global_form(points, bounds, beam=beam)


def narrow_beam(points, bounds, beam=-0.20):
    """全局：半宽 −20%，湿面积 −12.4%。"""
    return _global_form(points, bounds, beam=beam)


def deep_draft(points, bounds, draft=0.24):
    """全局：吃水 +24%（船体下沉 82 mm），湿面积 +15.6%。"""
    return _global_form(points, bounds, draft=draft)


def shallow_draft(points, bounds, draft=-0.24):
    """全局：吃水 −24%（船体上浮 82 mm），湿面积 −13.6%。"""
    return _global_form(points, bounds, draft=draft)


def high_cb(points, bounds, midship=0.25):
    """全局：中体半宽 +25%（sx ∈ [0.25, 0.75] 平台包络，水线以上衰减到 0），湿面积 +13.9%。"""
    return _global_form(points, bounds, midship=midship)


def low_cb(points, bounds, midship=-0.25):
    """全局：中体半宽 −25%（平台包络，水线以上衰减到 0），湿面积 −13.3%。"""
    return _global_form(points, bounds, midship=midship)


def full_form(points, bounds, beam=0.08, draft=0.08, midship=0.08):
    """全局：半宽 +8% + 吃水 +8% + 中体 +8%，湿面积 +13.8%。"""
    return _global_form(points, bounds, beam=beam, draft=draft, midship=midship)


# 变体名 → (目标位移函数, 中文说明, 吃水变化 ΔT m)。ΔT > 0 表示吃水加深（船体下沉）；
# 没有吃水变化的变体取 0。湿面积变化是相对原船 8.3525 m² 的实测值。
KCS_VARIANTS = {
    "fat_hull": (fat_hull, "全局：半宽 +12% + 吃水 +12%（湿面积 +15.6%）",
                 0.12 * KCS_DRAFT),
    "thin_hull": (thin_hull, "全局：半宽 −12% + 吃水 −12%（湿面积 −12.9%）",
                  -0.12 * KCS_DRAFT),
    "wide_beam": (wide_beam, "全局：半宽 +20%（湿面积 +12.9%）", 0.0),
    "narrow_beam": (narrow_beam, "全局：半宽 −20%（湿面积 −12.4%）", 0.0),
    "deep_draft": (deep_draft, "全局：吃水 +24%（湿面积 +15.6%）", 0.24 * KCS_DRAFT),
    "shallow_draft": (shallow_draft, "全局：吃水 −24%（湿面积 −13.6%）", -0.24 * KCS_DRAFT),
    "high_cb": (high_cb, "全局：中体半宽 +25%（湿面积 +13.9%）", 0.0),
    "low_cb": (low_cb, "全局：中体半宽 −25%（湿面积 −13.3%）", 0.0),
    "full_form": (full_form, "全局：半宽 +8% + 吃水 +8% + 中体 +8%（湿面积 +13.3%）",
                  0.08 * KCS_DRAFT),
}


def wetted_area(points, faces, draft=KCS_DRAFT):
    """设计吃水以下三角形的面积和，用于核对变体的湿面积变化。"""
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    wetted = (triangles[:, :, 2] <= draft).all(axis=1)
    cross = np.cross(triangles[wetted, 1] - triangles[wetted, 0],
                     triangles[wetted, 2] - triangles[wetted, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def flipped_triangles(points, moved, faces):
    """变形后法向反转的三角形数量（几何非法：曲面自交/折叠）。"""
    original = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    deformed = np.asarray(moved, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    before = np.cross(original[:, 1] - original[:, 0], original[:, 2] - original[:, 0])
    after = np.cross(deformed[:, 1] - deformed[:, 0], deformed[:, 2] - deformed[:, 0])
    return int(np.count_nonzero(np.einsum("nc,nc->n", before, after) <= 0.0))


def kcs_displacements(variant, points, bounds):
    """变体的控制点位移 (shape[0], shape[1], shape[2], 3)，由船体表面目标位移反解。"""
    return fit_control_displacements(points, bounds, KCS_VARIANTS[variant][0](points, bounds))
