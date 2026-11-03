"""自由变形（Free-Form Deformation）与 KCS 艏部小幅几何变体。

FFD 核心是 Sederberg & Parry (1986) 的标准构造：把几何嵌入一个长方体控制格架，
移动控制点后，格架内任意点的位移由三元张量积 Bernstein 多项式插值给出

    X' = X + Σ_i Σ_j Σ_k B_i(s) B_j(t) B_k(u) · D_ijk

s、t、u 是点在格架内的归一化坐标，D_ijk 是控制点位移。位移场在格架内处处光滑，
三角形连接关系不变的曲面变形后仍然封闭。控制点位移由 `fit_control_displacements`
反解：变体先给出船体表面要达到的目标位移，再用 Bernstein 基做最小二乘求控制点位移。
直接手写控制点位移会被 Bernstein 平滑削弱到设计值的三成左右，反解让幅度可控。

变体依据（不是自拟的变形）
--------------------------

A. V. Pechenyuk, «Оптимизация судовых обводов для снижения сопротивления
движению», Компьютерные исследования и моделирование 9(1):57–65, 2017
（DOI 10.20537/2076-7633-2017-9-57-65）§2 与图 1、图 2：

- 变体定义在**无量纲舷弧面积曲线**（строевая по шпангоутам，ω = A(x)/A_mid）上，
  艏部按长度分成三段，每段给出**有限体积增量** δV ≈ 0.4% 排水体积（正负各一个
  模型，共 6 个修改 + 原形「модель 0」）；
- 剖面重建时**保持甲板形状不变**，只重新分配剖面面积沿高度的分布；
- 论文的船长尺度：x/L 的 0 点在艏垂线，球鼻艏伸出到 x/L ≈ −0.03，图 1 的曲线
  正好画到 −0.03；图 2 是同一批修改在理论图上的型线表达（各站横剖面叠加，黑实线
  是原形，彩色虚线是修改形，水线 КВЛ 标注在图上）。

从图 1 读出的三段的纵向范围（x/Lpp，按像素标定轴 0.50 → −0.05）：第 1 段
≈ [−0.03, 0.10]（正增量在艏尖约 +0.024ω、平台在 x/L ≈ −0.01…0.06），第 2 段
≈ [0.10, 0.25]，第 3 段 ≈ [0.25, 0.45]。本文件只做**第 1 段**，也就是论文里
最靠艏、阻力变化最大的一对修改：

| 变体 | 论文模型 | 舷弧面积 | 几何方向 |
| --- | --- | --- | --- |
| `bow_outward` | модель 1（δV > 0） | 第 1 段 +0.4% 排水体积 | 艏部水下剖面外扩 |
| `bow_inward` | модель 2（δV < 0） | 第 1 段 −0.4% 排水体积 | 艏部水下剖面内收 |

坐标系换算：论文的 x/L 以艏垂线为 0、向艉为正（球鼻艏在 −0.03）；本地 STL 以 x = 0
在艉、x = Lpp 在艏垂线，所以 s = 1 − x/Lpp（`paper_x_over_lpp`）。第 1 段因此落在
x/Lpp ∈ [0.90, 1.03]，与 KCS 的球鼻艏（x/Lpp 到 1.034）完全吻合。

两个变体对同一段、同一形状界面取相反符号，因此横向位移严格互为相反数，其余
（船长、基准站、艉部、干舷甲板）完全一致 —— 这是论文 §5 里比较 δR(δV) 的做法，
也保证后续 CFD 的阻力差只来自「艏部第 1 段的体积转移」。

实现约定
--------

- 目标位移只沿 y：舷弧面积在第 1 段上的增量由**局部半宽按比例缩放**实现，
  `Δy = A · W(s) · g(z) · |y|`（s 是论文的 x/Lpp），A 是窗峰处的相对缩放量
  （由 |δV|/V = 0.4% 反标定），W 是第 1 段的升余弦窗，g 是沿高度的权
  （`sin²(π z / D)`）。
- x、z 不动：船长、站号位置、龙骨与甲板平面严格保持（论文只改舷弧面积、固定甲板）。
- 左右舷严格镜像：位移场在折叠坐标 (x, |y|, z) 上求值、乘 `sign(y)` 施加，
  中纵剖面（|y| = 0）上位移**恰好为 0**，不会出现左右舷互相穿透或中线错动。
- 竖直权 `g(z) = sin²(π z / D)`（D = 主甲板高 0.5443 m）：龙骨与甲板处为 0、
  水线以上最大，所以甲板形状与龙骨线不变；增量剖面的形状按 |y| 成比例，等价于
  「保持甲板形状、把剖面面积沿高度重新分配」，与图 2 里「各曲线在甲板边收敛、
  差异集中在水线下方」的表达一致。
- 纵向包络 `W`：第 1 段 `[−0.030, 0.100]` 上是升余弦窗（Hann）——论文图 1 的增量
  是「一段平台 + 两端圆角」，但带平台的窗在 Bernstein 格架上要 10 阶以上才不振铃
  （实测控制点位移放大到几百米），所以取同一作用区间上的光滑版本：峰值 1、两端
  一阶导为 0。δω 峰值因此比图上读数高约 30%（0.034 vs 0.024），∫δω（即 δV）相同。
  球鼻艏最前端 (`x/L = −0.034`) 落在窗的上升沿上，与图 1 里 x/L ≈ −0.02 处 δω 仍
  未归零一致；那里剖面面积趋 0，位移自然也趋 0。
- 控制格架只罩住艏部第 1 段（x 恰好是窗的作用区间，y 用折叠区间 `[0, B/2]`、
  z 用 `[0, 主甲板]`）。格架外的点按 FFD 定义原样保留；窗在格架边界上恰好为 0，
  所以边界上没有台阶（实测边界位移 0）。

幅度校准：`BOW_AMPLITUDE` 是窗峰处的相对半宽缩放量（0.551），取使两个变体的
|δV|/V 都等于 `BOW_VOLUME_FRACTION`（0.4%，论文 §2 的「约 0.4%」）的值；实测
排水体积变化由 `scripts/dataset_build/build_kcs_ffd_variants.py` 用
`hull_morph.mesh_displacement`（与 ShipBenchv2 排水量同一口径）复核。
"""

from math import comb

import numpy as np

# 控制格架点数：x 8 阶、y 1 阶、z 8 阶（次数 = 点数 − 1）。沿 y 取 1 阶是因为目标场
# 严格正比于 |y|（线性基精确表示，同时保证「y 按比例缩放」这个保定向映射不被拟合破坏，
# 任何三角形都不会翻转）；沿 x、z 各 8 阶能把升余弦窗与 sin² 权拟合到 6e-9 m（表面
# 位移 0.03 m 的 2e-7），偶数阶的 Bernstein 系数比奇数阶小（1.41 vs 2.35）。再往上加
# 控制点，条件数迅速恶化、控制格架位移放大到几米，数值上不可用。
KCS_LATTICE = (9, 2, 9)

# KCS 设计吃水（水面 z 坐标，龙骨在 z = 0）与主甲板高度。竖直权 g(z) 在 z = 0 与
# z = deck 处为 0：龙骨与甲板平面在变形前后逐点相同（论文「保持甲板形状」）。
# KCS 的 STL 在主甲板之上还有一段艏楼（z 到 0.744 m），它整段落在 g 的零区之外，
# 所以也保持不动。
KCS_DRAFT = 0.3418
KCS_DECK = 0.5443208

# 论文图 1 第 1 段，坐标系是论文的 x/L：0 在艏垂线、向艉为正，球鼻艏伸出到 −0.03。
# 有限增量的作用区间；区间内用升余弦窗（见 `bow_window`）。
BOW_SEGMENT = (-0.030, 0.100)
# 每次修改的体积增量（相对排水体积），论文 §2：「примерно на 0.4 %」。
BOW_VOLUME_FRACTION = 0.004
# 平台处的相对半宽缩放量，由 |δV|/V = 0.4% 反标定（见模块说明）。
BOW_AMPLITUDE = 0.551


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


def paper_x_over_lpp(points_x, lpp):
    """船体 x → 论文的 x/Lpp（0 在艏垂线、向艉为正）。

    本地 STL 的约定是 x = 0 在艉、x = Lpp 在艏垂线（`hull_uv` 的 u = x/Lpp 也是这个
    方向），KCS 的球鼻艏还会伸出到 x > Lpp，所以 s = 1 − x/Lpp 在艏尖附近取负值——
    与论文图 1 的横轴（0.50 → −0.05，曲线在 −0.03 收尾）逐点对得上。
    """
    return 1.0 - np.asarray(points_x, dtype=np.float64) / lpp


def bow_window(paper_x):
    """论文图 1 第 1 段的纵向包络：作用区间 [−0.030, 0.100] 上的升余弦窗。

    论文的增量在图上是一段「平台 + 圆角」的有限增量；FFD 只能在格架上用 Bernstein
    基表示，带平台的窗需要 10 阶以上才能无振铃地拟合（实测控制点位移放大到几百米、
    条件数 1e8），所以取同一作用区间上的光滑版本（Hann 窗）：峰值 1、两端一阶导为 0。
    两种窗的体积增量相同（都按 0.4% 反标定），差别是平台宽度——见
    `vis/kcs_ffd_var/README.md` 里与论文图 1 的逐点对照。
    """
    paper_x = np.asarray(paper_x, dtype=np.float64)
    span = BOW_SEGMENT[1] - BOW_SEGMENT[0]
    u = np.clip((paper_x - BOW_SEGMENT[0]) / span, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * u))


def bow_vertical_weight(z, deck):
    """沿高度的权：龙骨（z = 0）与主甲板（z = D）处为 0，z = D/2 = 0.27 m 处最大。

    甲板平面、甲板边线与龙骨线因此在变形前后逐点相同（论文「保持甲板形状」）。
    """
    z = np.clip(np.asarray(z, dtype=np.float64), 0.0, deck)
    return np.sin(np.pi * z / deck) ** 2


def fold_y(points):
    """把右舷折到左舷：y → |y|，其余坐标不变（位移场只需在半个格架上定义）。"""
    points = np.asarray(points, dtype=np.float64)
    folded = points.copy()
    folded[:, 1] = np.abs(folded[:, 1])
    return folded


def bow_lattice_bounds(points, lpp, deck):
    """艏部第 1 段的控制格架范围：x 是窗的作用区间，y 折叠成 [0, B/2]，z 为 [0, D]。

    x 不加余量：升余弦窗在区间两端本来就取 0，格架贴住作用区间时窗在格架里占满，
    同阶数下拟合最好；格架外（x 更靠艉/更靠艏）的点按 FFD 定义原样保留，而那里的
    目标位移本来就是 0，所以边界上没有台阶。
    """
    folded = fold_y(points)
    half_beam = float(folded[:, 1].max())
    return ((lpp * (1.0 - BOW_SEGMENT[1]), lpp * (1.0 - BOW_SEGMENT[0])),
            (0.0, half_beam),
            (0.0, float(deck)))


def bow_target(points, lpp, deck, amplitude):
    """第 1 段的目标横向位移（米），输入是折叠后的点，输出只改 y 的 (N,3) 位移。"""
    folded = fold_y(points)
    scale = (amplitude * bow_window(paper_x_over_lpp(folded[:, 0], lpp))
             * bow_vertical_weight(folded[:, 2], deck) * folded[:, 1])
    target = np.zeros_like(folded)
    target[:, 1] = scale
    return target


def bow_displacements(variant, points, lpp, deck):
    """变体的控制点位移 (shape[0], shape[1], shape[2], 3)。

    目标位移对 |y| 严格线性（Δy = |y| · S(x, z)），所以只反解斜率场 S 的 (x, z)
    二维控制系数，y 向系数固定成 |y| 的精确 Bernstein 表示 `[0, B/2]`。这样变形
    严格等价于「每个 (x, z) 处把半宽乘上 1 + S」，是一个定向保持的一一映射：
    三角形不可能翻转，中纵剖面（|y| = 0）上的位移也严格为 0（不是拟合残差量级的
    近似）。若把三维目标一次性最小二乘，y 向低阶系数会残留一个非零常数项，|y| → 0
    的细长面片仍会翻转（实测 inward 84 个）。
    """
    folded = fold_y(points)
    bounds = bow_lattice_bounds(points, lpp, deck)
    half_beam = bounds[1][1]
    local = normalized_coordinates(folded, bounds)
    inside = np.all((local >= 0.0) & (local <= 1.0), axis=1)
    basis = [bernstein_matrix(local[inside, axis], KCS_LATTICE[axis]) for axis in (0, 2)]
    design = np.einsum("ni,nk->nik", *basis).reshape(int(inside.sum()), -1)
    slope = variant_amplitude(variant) * bow_window(
        paper_x_over_lpp(folded[inside, 0], lpp)) * bow_vertical_weight(
        folded[inside, 2], deck)
    coefficient, *_ = np.linalg.lstsq(design, slope, rcond=None)
    lattice = np.zeros(tuple(KCS_LATTICE) + (3,))
    lattice[:, 1, :, 1] = half_beam * coefficient.reshape(KCS_LATTICE[0], KCS_LATTICE[2])
    return lattice


def bow_fit_residual(points, displacements, lpp, deck, amplitude):
    """变形后的表面位移与目标位移之差的最大绝对值（米），用于报告拟合保真度。"""
    points = np.asarray(points, dtype=np.float64)
    target = np.sign(points[:, 1]) * bow_target(points, lpp, deck, amplitude)[:, 1]
    applied = deform_bow(points, displacements, lpp, deck)[:, 1] - points[:, 1]
    return float(np.abs(applied - target).max())


def deform_bow(points, displacements, lpp, deck):
    """把控制点位移用到整船顶点上：只改 y，左右舷严格镜像，中纵剖面不动。

    x、z 严格保持原值（位移场只含 y 分量），所以船长、站号位置、龙骨与甲板平面
    在变形前后逐点相同。
    """
    points = np.asarray(points, dtype=np.float64)
    folded = fold_y(points)
    bounds = bow_lattice_bounds(points, lpp, deck)
    moved = apply_ffd(folded, bounds, displacements)
    shifted = points.copy()
    shifted[:, 1] = np.sign(points[:, 1]) * moved[:, 1]
    return shifted


def variant_amplitude(variant):
    """变体名 → 带符号的相对半宽缩放量（`bow_outward` 为正、`bow_inward` 为负）。"""
    sign, _ = KCS_BOW_VARIANTS[variant]
    return sign * BOW_AMPLITUDE


def flipped_triangles(points, moved, faces):
    """变形后法向反转的三角形数量（几何非法：曲面自交/折叠）。"""
    original = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    deformed = np.asarray(moved, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    before = np.cross(original[:, 1] - original[:, 0], original[:, 2] - original[:, 0])
    after = np.cross(deformed[:, 1] - deformed[:, 0], deformed[:, 2] - deformed[:, 0])
    return int(np.count_nonzero(np.einsum("nc,nc->n", before, after) <= 0.0))


def wetted_area(points, faces, draft=KCS_DRAFT):
    """设计吃水以下三角形的面积和，用于核对变体的湿面积变化。"""
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    wetted = (triangles[:, :, 2] <= draft).all(axis=1)
    cross = np.cross(triangles[wetted, 1] - triangles[wetted, 0],
                     triangles[wetted, 2] - triangles[wetted, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


# 变体名 → (符号, 说明)。符号乘 `BOW_AMPLITUDE` 得到相对半宽缩放量；说明里的
# 「图 1 模型号」是论文里对应的 3D 模型编号（модель 1 / модель 2）。
KCS_BOW_VARIANTS = {
    "bow_outward": (+1, "艏部第 1 段舷弧面积 +0.4% 排水体积：水下横剖面外扩（图 1 模型 1）"),
    "bow_inward": (-1, "艏部第 1 段舷弧面积 −0.4% 排水体积：水下横剖面内收（图 1 模型 2）"),
}
