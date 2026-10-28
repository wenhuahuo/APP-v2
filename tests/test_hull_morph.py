"""hull morphing 的单元测试：两个解析全船（Wigley 型抛物截面 + 垂直舷侧 + 平甲板）。

半船剖面取 y = −(B/2)(1−ζ²)(2η−η²)，ζ = 2x/L − 1、η = z/T，即抛物线剖面、最大
宽在设计水线上、龙骨是一条 y = 0 的线；干舷段取垂直舷侧 y = −(B/2)(1−ζ²)，
甲板平在 z = D。两个母船只差 B 与 T（D 按比例跟着变），水线以下解析排水体积
V = (4/9)·B·L·T。

全船 STL 由「半船剖面 + 甲板 + 镜像」绕成管状闭合曲面，`half_ship_surface` 再切回
y ≤ 0。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

import trimesh  # noqa: E402

from hull_morph import hull_station_grid, morph_hulls  # noqa: E402

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


def fold_amplitude(grid, lpp, low=0.12, high=0.88):
    """相邻三站位同一行的二阶差分（折返带的量度），返回船中段的最大值 / 折减长度。

    同一行格点沿 u 应该连续：`|P[i+1] − 2P[i] + P[i−1]|` 只反映曲率（量级 O(Δu²)）。
    剖面被三角化切成多条链、选链逐站翻转时，这一项会跳到若干个 % L_ref。端部 12%
    是艉封板/艏柱，剖面下端本来就要抬离龙骨线，不计入。
    """
    second = np.linalg.norm(grid[2:] - 2.0 * grid[1:-1] + grid[:-2], axis=2)
    u = (grid[:, 0, 0] - grid[0, 0, 0]) / (grid[-1, 0, 0] - grid[0, 0, 0])
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


if __name__ == "__main__":
    unittest.main()
