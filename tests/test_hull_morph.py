"""hull morphing 的单元测试。

两个解析全船（Wigley 型抛物截面 + 垂直舷侧 + 平甲板）：半船剖面取
 y = −(B/2)(1−ζ²)(2η−η²)，ζ = 2x/L − 1、η = z/T，即抛物线剖面、最大宽在设计水线
上、龙骨是一条 y = 0 的线；干舷段取垂直舷侧 y = −(B/2)(1−ζ²)，甲板平在 z = D。
两个母船只差 B 与 T（D 按比例跟着变），水线以下解析排水体积 V = (4/9)·B·L·T。
全船 STL 由「半船剖面 + 甲板 + 镜像」绕成管状闭合曲面，`half_ship_surface` 再切回
y ≤ 0。

另外两条回归测试针对**球鼻艏 + 凹槽**：

* `bulbous_hull_stl` 是一个解析船型，它的艏柱线 x_max(z) 在中上部凹进去（下方球鼻
  外凸、上方干舷外飘）。x = const 的横切平面在凹槽处会把剖面切成两条链，老实现
  桥接后生成一堵竖直墙；测试断言重建后的 x_max(z) 跟着解析艏柱线走。
* dtc 母船 STL 的 w = 0 回代偏差（读 `dataset/classic_hulls/dtc.stl`，缺席时 skip）。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

import trimesh  # noqa: E402

from hull_morph import (  # noqa: E402
    BOW_FAN_START, WELD_FRACTION, full_ship_mesh, half_ship_surface,
    hull_station_grid, local_section, morph_hulls, station_frames,
)
from trimesh.intersections import mesh_plane  # noqa: E402

N_STATION = 81
N_GIRTH = 41
N_TOPSIDE = 9
N_DECK = 13
HULL_A = {"Lpp": 1.0, "T": 0.10, "B": 0.20, "D": 0.16}
HULL_B = {"Lpp": 1.0, "T": 0.12, "B": 0.32, "D": 0.18}
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_U, N_V, N_V2 = 160, 60, 20
# 站位的 x ≈ [0.002, 0.998]·L；再跳过两端的尖端站位看形状
KEEP_STATIONS = slice(2, N_U - 2)
# STL 只存 float32，甲板高度、吃水这类整数量级会掉到 1e-8 量级，比较时留出余量
FLOAT32_TOL = 1e-6
# 带球鼻艏 + 凹槽的解析船型（z 向放样的双端船）：span 是 x 全长，艏柱线在中上部
# 凹进去 9 cm（球鼻外凸、干舷外飘），凹槽整段在吃水以上（T = 0.10 < 0.13）
HULL_BULB = {"span": 2.0, "T": 0.10, "D": 0.16, "B": 0.30,
             "groove_depth": 0.09, "groove_z": 0.13, "groove_width": 0.022}
N_BULB_T = 181
N_BULB_Z = 161
# 凹槽检测用的取高（一半在水线下、一半在凹槽里）
GROOVE_SAMPLES = (0.02, 0.06, 0.095, 0.115, 0.128, 0.14, 0.152)
# dtc 母船数据（dataset/ 不入 git，缺席时 skip）
DTC_STL = ROOT / "dataset" / "classic_hulls" / "dtc.stl"
DTC_LPP, DTC_T = 5.976, 0.244


def analytic_half_breadth(x, z, hull):
    """解析半船在 (x, z) 处的半宽（y ≤ 0）：水线以下是抛物剖面，以上是垂直舷侧。"""
    zeta = 2.0 * np.asarray(x) / hull["Lpp"] - 1.0
    eta = np.clip(np.asarray(z) / hull["T"], 0.0, 1.0)
    underwater = np.asarray(z) <= hull["T"]
    draft_shape = np.where(underwater, 2.0 * eta - eta ** 2, 1.0)
    return -0.5 * hull["B"] * (1.0 - zeta ** 2) * draft_shape


def half_profile(hull):
    """半个剖面折线 (y, z)：龙骨 (0, 0) → 水线 → 舷边 (y_edge, D)。"""
    z = np.r_[np.linspace(0.0, hull["T"], N_GIRTH),
              np.linspace(hull["T"], hull["D"], N_TOPSIDE)[1:]]
    return z


def full_ship_stl(path, hull, hole=None):
    """把解析全船三角化成 STL（半船剖面 + 甲板 + 镜像成整船管面）。

    hole = (x0, x1, eta0, eta1) 时挖掉这块面片，让中段站位的剖面断成两条链，
    用来检查多环拼接（不允许静默丢环）。
    """
    stations = np.linspace(0.0, hull["Lpp"], N_STATION)
    z_half = half_profile(hull)
    deck = np.linspace(0.0, 1.0, N_DECK)
    loops = []
    for x in stations:
        half = np.column_stack([analytic_half_breadth(x, z_half, hull), z_half])
        edge = half[-1, 0]
        deck_line = np.column_stack([edge * (1.0 - deck[1:-1]),
                                     np.full(N_DECK - 2, hull["D"])])
        starboard = half[::-1] * np.array([-1.0, 1.0])  # 镜像（含中线两点）
        loops.append(np.vstack([half, deck_line, starboard]))
    n_loop = len(loops[0])
    points = np.array([[x, y, z] for x, loop in zip(stations, loops) for y, z in loop])
    faces = []
    for i in range(N_STATION - 1):
        for j in range(n_loop - 1):
            if hole is not None:
                x0, x1, e0, e1 = hole
                low = e0 * hull["T"]
                high = e1 * hull["T"]
                inside_x = x0 <= stations[i] and stations[i + 1] <= x1
                inside_z = low <= loops[i][j][1] and loops[i][j + 1][1] <= high
                if inside_x and inside_z:
                    continue
            corner = (i * n_loop + j, (i + 1) * n_loop + j,
                      (i + 1) * n_loop + j + 1, i * n_loop + j + 1)
            faces.append((corner[0], corner[1], corner[2]))
            faces.append((corner[0], corner[2], corner[3]))
    trimesh.Trimesh(points, np.asarray(faces), process=False).export(path)
    return path


def analytic_volume(hull):
    return 4.0 / 9.0 * hull["B"] * hull["Lpp"] * hull["T"]


def bulb_stem_x(z, hull):
    """解析艏柱线 x_max(z)：中上部挖一个高斯坑（球鼻上方内凹 = 凹槽）。"""
    z = np.asarray(z, dtype=np.float64)
    return (0.5 * hull["span"]
            - hull["groove_depth"]
            * np.exp(-((z - hull["groove_z"]) / hull["groove_width"]) ** 2))


def bulb_draft_shape(z, hull):
    """吃水以下的剖面形状（2η−η²）与吃水以上的垂直舷侧；留一层 8% 宽的平底。"""
    eta = np.clip(np.asarray(z, dtype=np.float64) / hull["T"], 0.0, 1.0)
    return np.maximum(2.0 * eta - eta ** 2, 0.08)


def bulbous_hull_stl(path, hull):
    """带球鼻艏 + 凹槽的解析船型三角化成 STL（z 向放样，艏艉对称的双端船）。

    水面线（每个高度 z）是半轴 X(z) × (B/2)·d(z) 的椭圆，因此 x_max(z) = X(z)，
    而 X(z) 在中上部凹进去。于量 x 落在 (X_min, X_max] 的横切平面剖面被切成两条链：
    下方球鼻、上方干舷外飘、中间是真实凹槽——与 dtc / containership / kcs / kvlcc2
    的艏部形态相同（那里 x = const 的横切平面会把剖面切成「干舷外飘 + 球鼻前缘」）。
    """
    z = np.linspace(0.0, hull["D"], N_BULB_Z)
    angle = np.linspace(0.0, 2.0 * np.pi, N_BULB_T, endpoint=False)
    loops = []
    for height in z:
        loops.append(np.column_stack([
            bulb_stem_x(height, hull) * np.cos(angle),
            0.5 * hull["B"] * bulb_draft_shape(height, hull) * np.sin(angle),
            np.full_like(angle, height)]))
    points = np.vstack(loops + [
        np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, hull["D"]]])])
    bottom, top = len(z) * N_BULB_T, len(z) * N_BULB_T + 1

    faces = []
    for level in range(len(z) - 1):
        for index in range(N_BULB_T):
            a = level * N_BULB_T + index
            b = level * N_BULB_T + (index + 1) % N_BULB_T
            faces.append((a, b, b + N_BULB_T))
            faces.append((a, b + N_BULB_T, a + N_BULB_T))
    last = (len(z) - 1) * N_BULB_T
    for index in range(N_BULB_T):  # 甲板与平底两个封口
        head, tail = index, (index + 1) % N_BULB_T
        faces.append((top, last + head, last + tail))
        faces.append((bottom, tail, head))
    trimesh.Trimesh(points, np.asarray(faces), process=False).export(path)
    return path


def in_plane_deviation(stl_path, lpp, draft, n_u=N_U, n_v=N_V, n_v2=N_V2):
    """每站自身平面内的回代偏差：重建格点到母船 STL 该平面交线折线的最小距离。

    返回 (n_u,) 的偏差数组与站位参数 u。这就是「w = 0 回代偏差」的口径（融合在
    w = 0 时输出的格点就是母船自己的剖面点）。
    """
    surface = hull_station_grid(stl_path, lpp, draft, n_u=n_u, n_v=n_v, n_v2=n_v2)
    half_points, half_faces = half_ship_surface(stl_path, lpp)
    mesh = trimesh.Trimesh(half_points, half_faces, process=False)
    frames = station_frames(*surface["x_range"], n_u)
    grid = surface["points"] * lpp
    deviation = np.full(n_u, np.nan)
    for index in range(n_u):
        if not surface["stations"][index]:
            continue
        section = local_section(
            mesh_plane(mesh, tuple(frames["normals"][index]),
                       tuple(frames["origins"][index])),
            frames["origins"][index], frames["u_hat"][index])
        section = np.asarray(section, dtype=np.float64).reshape(-1, 2, 2)
        section = section[np.linalg.norm(section[:, 1] - section[:, 0], axis=1)
                          > WELD_FRACTION * lpp]
        if len(section) == 0:
            continue
        local = np.column_stack([
            (grid[index] - frames["origins"][index]) @ frames["u_hat"][index],
            grid[index][:, 2]])
        offset = local[:, None, :] - section[None, :, 0]
        edge = section[None, :, 1] - section[None, :, 0]
        ratio = np.clip((offset * edge).sum(-1)
                        / np.maximum((edge * edge).sum(-1), np.finfo(float).tiny),
                        0.0, 1.0)
        # 每行取到所有交线段的最小距离，再取该站位的最大值
        deviation[index] = np.linalg.norm(
            offset - ratio[..., None] * edge, axis=-1).min(axis=1).max()
    return deviation, surface["u"]


def fold_amplitude(grid, lpp, low=0.12, high=0.88):
    """相邻三站位同一行的二阶差分（折返带的量度），返回船中段的最大值 / 折减长度。

    同一行格点沿 u 应该连续：`|P[i+1] − 2P[i] + P[i−1]|` 只反映曲率（量级 O(Δu²)）。
    剖面被三角化切成多条链、选链逐站翻转时，这一项会跳到若干个 % L_ref。端部 12%
    是艉封板/艏柱，剖面下端本来就要抬离龙骨线，不计入。

    u 取站位序号（艏区是旋转平面，站位的 x 不再逐站递增，不能拿 x 当 u）。
    """
    second = np.linalg.norm(grid[2:] - 2.0 * grid[1:-1] + grid[:-2], axis=2)
    u = np.linspace(0.0, 1.0, len(grid))
    body = (u[1:-1] > low) & (u[1:-1] < high)
    return float(second[body].max() / lpp)


class HullMorphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(__file__).resolve().parent / ".hull_morph_tmp"
        cls.tmp.mkdir(exist_ok=True)
        cls.stl_a = full_ship_stl(cls.tmp / "hull_a.stl", HULL_A)
        cls.stl_b = full_ship_stl(cls.tmp / "hull_b.stl", HULL_B)
        cls.morphed = {
            weight: morph_hulls(cls.stl_a, cls.stl_b, HULL_A["Lpp"], HULL_B["Lpp"],
                                HULL_A["T"], HULL_B["T"], weight,
                                n_u=N_U, n_v=N_V, n_v2=N_V2)
            for weight in WEIGHTS
        }

    @classmethod
    def tearDownClass(cls):
        for path in cls.tmp.glob("*"):
            path.unlink()
        cls.tmp.rmdir()

    def test_endpoints_reproduce_parents(self):
        """w = 0 / w = 1 分别还原两个母船的解析曲面。

        v 是站位剖面的弧长，w = 0 的格点就是母船自己的剖面折线点，残差只剩三角化
        与弧长重采样的离散化误差。
        """
        for weight, hull in ((0.0, HULL_A), (1.0, HULL_B)):
            result = self.morphed[weight]
            grid = result["grid"][KEEP_STATIONS].reshape(-1, 3)
            # 甲板行是水平面，半宽公式（z 的函数）对它不成立，单独断言
            side = grid[:, 2] < hull["D"] - FLOAT32_TOL
            residual = grid[side, 1] - analytic_half_breadth(grid[side, 0],
                                                             grid[side, 2], hull)
            self.assertLess(np.abs(residual).max() / hull["B"], 2e-3,
                            msg=f"w={weight} 半宽残差 {np.abs(residual).max():.2e}")
            deck = grid[~side]
            self.assertGreater(len(deck), 0)
            np.testing.assert_allclose(deck[:, 2], hull["D"], atol=FLOAT32_TOL)
            edge = analytic_half_breadth(deck[:, 0], hull["D"], hull)
            self.assertTrue(np.all(deck[:, 1] >= edge - 1e-9))
            self.assertTrue(np.all(deck[:, 1] <= 1e-9))
            # T_new 是融合面 v = 1 一行的 z 中位数；水线端就是 z = T 的等值点
            self.assertAlmostEqual(result["meta"]["T_m"], hull["T"], delta=FLOAT32_TOL)
            # 甲板中线行落在 y = 0、z = D 上
            self.assertLess(np.abs(result["grid"][:, -1, 1]).max(), 1e-12)
            self.assertAlmostEqual(float(np.median(result["grid"][:, -1, 2])),
                                   hull["D"], delta=FLOAT32_TOL)

    def test_midpoint_is_average_of_parents(self):
        """同一 (u, v) 格点上，w = 0.5 的坐标等于两母船的中点。"""
        half = self.morphed[0.5]["grid"]
        average = 0.5 * (self.morphed[0.0]["grid"] + self.morphed[1.0]["grid"])
        self.assertEqual(half.shape, average.shape)
        np.testing.assert_allclose(half, average, atol=1e-12)

    def test_waterline_row_is_at_interpolated_draft(self):
        """v = 1 一行（第 n_v−1 行）严格落在插值吃水上。"""
        for weight, result in self.morphed.items():
            meta, grid = result["meta"], result["grid"]
            water = grid[:, N_V - 1, 2]
            np.testing.assert_allclose(water, meta["T_m"], atol=1e-12)
            self.assertAlmostEqual(meta["T_m"], meta["T_interp_norm_m"], delta=1e-12)
            # 归一化口径的插值吃水：(1−w)·T_a/L_a + w·T_b/L_b，再乘 Lpp_new
            lpp_new = meta["Lpp_new"]
            expect = ((1.0 - weight) * HULL_A["T"] / HULL_A["Lpp"]
                      + weight * HULL_B["T"] / HULL_B["Lpp"]) * lpp_new
            self.assertAlmostEqual(meta["T_m"], expect, delta=FLOAT32_TOL)

    def test_full_ship_is_watertight(self):
        """镜像 + 首尾封口后是全船水密封闭网格，回读 STL（合并顶点）仍水密。"""
        for weight, result in self.morphed.items():
            mesh = trimesh.Trimesh(result["points"], result["faces"], process=False)
            self.assertTrue(mesh.is_watertight, msg=f"w={weight} 不水密")
            self.assertTrue(mesh.is_winding_consistent, msg=f"w={weight} 绕向不一致")
            self.assertTrue(mesh.is_volume, msg=f"w={weight} 不是体积网格")
            self.assertTrue(result["meta"]["watertight"])
            # 全船对称、z 从龙骨 0 到甲板 D、x 是全长
            points = result["points"]
            self.assertLess(abs(points[:, 1].max() + points[:, 1].min()), 1e-9)
            self.assertGreaterEqual(points[:, 2].min(), -1e-9)
            self.assertAlmostEqual(points[:, 2].max(), result["meta"]["depth_m"],
                                   delta=FLOAT32_TOL)
            # 回读：STL 没有顶点共享信息，合并顶点后才是真实拓扑
            path = self.tmp / f"round_trip_w{int(round(weight * 100))}.stl"
            mesh.export(path)
            reloaded = trimesh.load(str(path), force="mesh")
            self.assertTrue(reloaded.is_watertight, msg=f"w={weight} 回读不水密")
            path.unlink()

    def test_half_breadth_and_draft_move_monotonically(self):
        """B 与 T 都随 w 增大，因此半宽与吃水随 w 单调。"""
        draft = [self.morphed[w]["meta"]["T_m"] for w in WEIGHTS]
        half_breadth = [np.abs(self.morphed[w]["grid"][KEEP_STATIONS, :, 1]).max()
                        for w in WEIGHTS]
        self.assertTrue(np.all(np.diff(draft) > 0), draft)
        self.assertTrue(np.all(np.diff(half_breadth) > 0), half_breadth)

    def test_rows_are_continuous_between_stations(self):
        """相邻站位同一行格点的二阶差分 < 1% L_ref。

        这是「选链逐站翻转」的回归防线：母船剖面在粗三角化平底上会被切成「中线碎链 +
        主剖面」两段，碎片链若被接到路径尾部、或起链位置被平板节点的 z 浮点噪声决定，
        同一行格点就会在相邻站位之间来回跳（HEAD 代码算 dtc×s175 的折返带是 63%
        L_ref）。现在选链按邻站连续性决定，这一项回到曲率量级。
        """
        for weight, result in self.morphed.items():
            lpp = result["meta"]["L_ref_m"]
            amplitude = fold_amplitude(result["grid"], lpp)
            self.assertLess(amplitude, 1e-2,
                            msg=f"w={weight} 折返幅度 {amplitude:.3e} L_ref")

    def test_displacement_is_monotone_and_matches_analytic(self):
        """排水体积（z ≤ T_new 站位积分）随 w 单调，且与解析体积对账。"""
        volumes = [self.morphed[w]["meta"]["displacement_m3"] for w in WEIGHTS]
        self.assertTrue(all(v is not None for v in volumes))
        self.assertTrue(np.all(np.diff(volumes) > 0), volumes)
        # 残差来自 n_v 个点的折线内接抛物剖面（少量偏小）；实测 0.5% 量级
        for weight, hull in ((0.0, HULL_A), (1.0, HULL_B)):
            index = 0 if weight == 0.0 else -1
            self.assertAlmostEqual(volumes[index] / analytic_volume(hull), 1.0,
                                   delta=5e-3)
        # 坐标逐点线性组合，截面积不是坐标的线性函数，融合体积因此系统性略低于
        # 两母船体积的线性插值（两船剖面不同，点对点平均会把截面削薄）
        interpolation = [(1 - w) * volumes[0] + w * volumes[-1] for w in WEIGHTS]
        np.testing.assert_allclose(volumes, interpolation, rtol=3e-2)

    def test_spliced_section_keeps_whole_profile(self):
        """剖面被挖断成两条链时，按端点拼接，水线端仍然在 z = T 上。

        STL 的洞让中段站位的剖面断成两段（真实船型里对应球鼻艏过渡区、带附体的
        艉段）。实现不允许静默丢环：两段都保留、按端点拼接，v = 1 仍落在水线上。
        """
        stl = full_ship_stl(self.tmp / "hull_hole.stl", HULL_A,
                            hole=(0.40, 0.60, 0.45, 0.75))
        grid = hull_station_grid(stl, HULL_A["Lpp"], HULL_A["T"],
                                 n_u=160, n_v=60, n_v2=20)
        self.assertGreater(grid["n_spliced_stations"], 0)
        # 洞的两侧边缘会切出毫米级碎片，丢弃它们并计数（不静默丢整环）
        self.assertLessEqual(grid["n_dropped_fragments"], 2)
        self.assertEqual(grid["n_invalid_stations"], 0)
        points = grid["points"][grid["stations"]] * HULL_A["Lpp"]
        # 每个站位（含被挖断的）的水线行都在吃水上
        self.assertAlmostEqual(float(np.median(points[:, N_V - 1, 2])), HULL_A["T"],
                               delta=FLOAT32_TOL)
        self.assertLess(np.abs(points[:, -1, 1]).max(), 1e-12)  # 甲板中线在 y = 0
        # 跨越洞的那几行是直线插值（真实船型里同样如此），残差按 2%·B 放
        side = points[:, :, 2] < HULL_A["D"] - FLOAT32_TOL
        residual = points[:, :, 1] - analytic_half_breadth(
            points[:, :, 0], points[:, :, 2], HULL_A)
        self.assertLess(np.abs(residual[side]).max() / HULL_A["B"], 2e-2)
        # 洞的两侧边缘必须是「直线跨过」而不是折返：除洞所在的那一段 x，其余站位的
        # 同行二阶差分仍在 1% L_ref 以内
        second = np.linalg.norm(points[2:] - 2.0 * points[1:-1] + points[:-2], axis=2)
        x = points[:, 0, 0] / HULL_A["Lpp"]
        outside = (x[1:-1] < 0.35) | (x[1:-1] > 0.65)
        self.assertLess(second[outside].max() / HULL_A["Lpp"], 1e-2)

    def test_bulbous_bow_has_no_vertical_wall(self):
        """带球鼻艏 + 凹槽的解析船型：x_max(z) 必须跟着解析艏柱线走。

        `x = const` 的横切平面在凹槽处（下方球鼻、上方干舷外飘）会把剖面切成两条
        链，用平面内直线桥接后在艏部生成一堵竖直墙——dtc 的 w = 0 回代就是这样偏
        2.35% L（x_max 偏差 2.8% L）。老实现在这里给出的 x_max(z) 是一条竖直线
        （相对解析艏柱线的最大偏差 3.9% span），新实现跟到 0.8% span 以内。
        """
        hull = HULL_BULB
        span = hull["span"]
        stl = bulbous_hull_stl(self.tmp / "bulb.stl", hull)
        grid = hull_station_grid(stl, span, hull["T"], n_u=N_U, n_v=N_V, n_v2=N_V2)
        self.assertEqual(grid["n_invalid_stations"], 0)
        # 艏区确实是旋转平面族（u ≥ BOW_FAN_START 的站位不再是 x = const）
        self.assertGreaterEqual(grid["n_bow_fan_stations"], 10)
        self.assertGreater(float(np.degrees(grid["theta"]).max()), 80.0)
        points, faces = full_ship_mesh(grid["points"][grid["stations"]] * span)
        mesh = trimesh.Trimesh(points, faces, process=False)
        self.assertTrue(mesh.is_watertight)

        for height in GROOVE_SAMPLES:
            lines = mesh_plane(mesh, (0.0, 0.0, 1.0), (0.0, 0.0, float(height)))
            self.assertGreater(len(lines), 0, msg=f"z = {height} 处切不出剖面")
            rebuilt = float(np.asarray(lines)[:, :, 0].max())
            expected = float(bulb_stem_x(height, hull))
            self.assertLess(abs(rebuilt - expected) / span, 2e-2,
                            msg=f"z = {height}: 重建 x_max {rebuilt:.4f} vs 解析 "
                                f"{expected:.4f}（凹槽变成竖直墙）")

    def test_dtc_back_substitution_in_the_bow(self):
        """dtc 母船在艏区的 w = 0 回代偏差 < 0.5% L（数据缺席时 skip）。

        凹槽整段在水线以上（球鼻顶 z = 0.232、T = 0.244），所以水下球鼻尺寸本来就
        准；偏差全部来自 x/L ≥ 0.97 的剖面双链。改成艏部旋转平面后艏区最大偏差从
        2.35% L 降到 0.000% L。艏区以远的残差是艉部螺旋桨/艉轴包的**真实分离结
        构**（u ≈ 0.05，被按端点拼接成一段直线，0.89% L），与本次改动无关。
        """
        if not DTC_STL.exists():
            self.skipTest(f"缺少 {DTC_STL}")
        deviation, u = in_plane_deviation(DTC_STL, DTC_LPP, DTC_T)
        known = np.isfinite(deviation)
        bow = known & (u >= BOW_FAN_START)
        self.assertTrue(bow.any())
        self.assertLess(float(deviation[bow].max()) / DTC_LPP, 5e-3,
                        msg=f"艏区最大回代偏差 {deviation[bow].max() / DTC_LPP * 100:.3f}% L")


if __name__ == "__main__":
    unittest.main()
