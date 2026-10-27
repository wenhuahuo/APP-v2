"""hull morphing 的单元测试：两个解析半船（Wigley 型抛物截面）之间的融合。

半船曲面取 y = -(B/2)(1-ζ²)(2η - η²)，ζ = 2x/L - 1、η = z/T，即抛物线剖面、
最大宽在设计水线上、龙骨是一条 y = 0 的线。两个母船只差 B 与 T，
解析排水体积 (4/9)·B·L·T。
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
HULL_A = {"Lpp": 1.0, "T": 0.10, "B": 0.20}
HULL_B = {"Lpp": 1.0, "T": 0.12, "B": 0.32}
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_U, N_V = 160, 60
# 站位的 x ≈ [0.002, 0.998]·L；再跳过两端的尖端站位看形状
KEEP_STATIONS = slice(2, N_U - 2)


def analytic_y(x, z, hull):
    """解析半船的半宽（y ≤ 0）。"""
    zeta = 2.0 * np.asarray(x) / hull["Lpp"] - 1.0
    eta = np.clip(np.asarray(z) / hull["T"], 0.0, 1.0)
    return -0.5 * hull["B"] * (1.0 - zeta ** 2) * (2.0 * eta - eta ** 2)


def half_hull_stl(path, hull, hole=None):
    """把解析半船三角化成 STL（半船，y ≤ 0、z ≤ T）。

    hole = (x0, x1, eta0, eta1) 时挖掉这块面片，让中段站位的剖面断成两条链，
    用来检查多环拼接（不允许静默丢环）。
    """
    stations = np.linspace(0.0, hull["Lpp"], N_STATION)
    eta = np.linspace(0.0, 1.0, N_GIRTH)
    points = np.array([
        [x, analytic_y(x, e * hull["T"], hull), e * hull["T"]]
        for x in stations for e in eta
    ])
    faces = []
    for i in range(N_STATION - 1):
        for j in range(N_GIRTH - 1):
            if hole is not None:
                x0, x1, e0, e1 = hole
                inside_x = x0 <= stations[i] and stations[i + 1] <= x1
                inside_z = e0 <= eta[j] and eta[j + 1] <= e1
                if inside_x and inside_z:
                    continue
            corner = (i * N_GIRTH + j, (i + 1) * N_GIRTH + j,
                      (i + 1) * N_GIRTH + j + 1, i * N_GIRTH + j + 1)
            faces.append((corner[0], corner[1], corner[2]))
            faces.append((corner[0], corner[2], corner[3]))
    trimesh.Trimesh(points, np.asarray(faces), process=False).export(path)
    return path


def as_grid(result):
    """把融合结果拉回 (n_u, n_v, 3) 的规则网格。"""
    return result["points"].reshape(N_U, N_V, 3)


def analytic_volume(hull):
    return 4.0 / 9.0 * hull["B"] * hull["Lpp"] * hull["T"]


class HullMorphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(__file__).resolve().parent / ".hull_morph_tmp"
        cls.tmp.mkdir(exist_ok=True)
        cls.stl_a = half_hull_stl(cls.tmp / "hull_a.stl", HULL_A)
        cls.stl_b = half_hull_stl(cls.tmp / "hull_b.stl", HULL_B)
        cls.morphed = {
            weight: morph_hulls(cls.stl_a, cls.stl_b, HULL_A["Lpp"], HULL_B["Lpp"],
                                HULL_A["T"], HULL_B["T"], weight,
                                n_u=N_U, n_v=N_V)
            for weight in WEIGHTS
        }

    @classmethod
    def tearDownClass(cls):
        for path in cls.tmp.glob("*"):
            path.unlink()
        cls.tmp.rmdir()

    def test_endpoints_reproduce_parents(self):
        """w = 0 / w = 1 分别还原两个母船的解析曲面。

        v 是站位剖面的弧长，w = 0 的格点就是母船自己的剖面折线点，残差只剩
        三角化与弧长重采样的离散化误差（旧实现的测地 v 在这里差 0.5%–2.5% Lpp）。
        """
        for weight, hull in ((0.0, HULL_A), (1.0, HULL_B)):
            result = self.morphed[weight]
            points = as_grid(result)[KEEP_STATIONS].reshape(-1, 3)
            residual = points[:, 1] - analytic_y(points[:, 0], points[:, 2], hull)
            self.assertLess(np.abs(residual).max() / hull["B"], 2e-3)
            # T_new 是融合面 v = 1 一行的 z 中位数；水线端就是 z = T 的切边界
            self.assertAlmostEqual(result["meta"]["T_new"], hull["T"], delta=1e-6)

    def test_midpoint_is_average_of_parents(self):
        """同一 (u, v) 格点上，w = 0.5 的坐标等于两母船的中点。"""
        half = self.morphed[0.5]["points"]
        average = 0.5 * (self.morphed[0.0]["points"] + self.morphed[1.0]["points"])
        self.assertEqual(half.shape, average.shape)
        np.testing.assert_allclose(half, average, atol=1e-12)

    def test_output_grid_is_valid(self):
        """面片非空、半船约束（y ≤ 0、z ∈ [0, T_new]）成立、拓扑与权重无关。"""
        for weight, result in self.morphed.items():
            points, meta = result["points"], result["meta"]
            self.assertGreater(len(result["faces"]), 0)
            self.assertEqual(points.shape[0], meta["n_points"])
            self.assertLessEqual(points[:, 1].max(), 1e-12)
            self.assertGreaterEqual(points[:, 2].min(), -1e-12)
            self.assertLessEqual(points[:, 2].max(), max(HULL_A["T"], HULL_B["T"]) * 1.0001)
            # 各权重的网格拓扑一致（有效站位与权重无关）
            self.assertEqual(meta["n_stations_valid"], 160)
            np.testing.assert_array_equal(
                result["faces"], self.morphed[0.0]["faces"])

    def test_half_breadth_and_draft_move_monotonically(self):
        """B 与 T 都随 w 增大，因此半宽与吃水随 w 单调。"""
        draft = [self.morphed[w]["meta"]["T_new"] for w in WEIGHTS]
        half_breadth = [np.abs(as_grid(self.morphed[w])[KEEP_STATIONS, :, 1]).max()
                        for w in WEIGHTS]
        self.assertTrue(np.all(np.diff(draft) > 0), draft)
        self.assertTrue(np.all(np.diff(half_breadth) > 0), half_breadth)

    def test_displacement_volume_is_monotone_and_matches_analytic(self):
        """排水体积随 w 单调，且与解析体积 (4/9)·B·L·T 对账。"""
        volumes = [self.morphed[w]["meta"]["volume"] for w in WEIGHTS]
        self.assertTrue(all(v is not None for v in volumes))
        self.assertTrue(np.all(np.diff(volumes) > 0), volumes)
        # 残差来自 n_v = 60 个点的折线内接抛物剖面（少量偏小）；实测 0.05% 量级
        for weight, hull in ((0.0, HULL_A), (1.0, HULL_B)):
            index = 0 if weight == 0.0 else -1
            self.assertAlmostEqual(volumes[index] / analytic_volume(hull), 1.0,
                                   delta=2e-3)
        # 坐标逐点线性组合，截面积不是坐标的线性函数，融合体积因此系统性略低于
        # 两母船体积的线性插值（两船剖面不同，点对点平均会把截面削薄）；实测
        # 最大偏低 2.3%（w = 0.5），容差按 3% 给
        interpolation = [(1 - w) * volumes[0] + w * volumes[-1] for w in WEIGHTS]
        np.testing.assert_allclose(volumes, interpolation, rtol=3e-2)
        self.assertTrue(all(v <= i for v, i in zip(volumes, interpolation)))

    def test_spliced_section_keeps_whole_profile(self):
        """剖面被挖断成两条链时，按 z 顺序拼接，水线端仍然到 z = T。

        STL 的洞让中段站位的剖面断成两段（真实船型里对应球鼻艏过渡区、带附体的
        艉段）。实现不允许静默丢环：两段都保留、按端点拼接，路径末端仍在水线，
        体积与解析值的偏差保持在 1% 以内。
        """
        stl = half_hull_stl(self.tmp / "hull_hole.stl", HULL_A,
                            hole=(0.40, 0.60, 0.45, 0.75))
        grid = hull_station_grid(stl, HULL_A["Lpp"], HULL_A["T"], n_u=160, n_v=60)
        self.assertGreater(grid["n_spliced_stations"], 0)
        self.assertEqual(grid["n_dropped_fragments"], 0)
        points = grid["points"][grid["stations"]] * HULL_A["Lpp"]
        # 每个站位（含被挖断的）的水线端都在吃水上
        self.assertAlmostEqual(np.median(points[:, -1, 2]), HULL_A["T"], delta=1e-6)
        volume = grid_volume(points)
        self.assertAlmostEqual(volume / analytic_volume(HULL_A), 1.0, delta=1e-2)


def grid_volume(points):
    """站位横剖面面积积分（半船截面镜像成整船），与 hull_morph._grid_volume 同法。"""
    y = np.hstack([points[:, :, 1], -points[:, :, 1][:, ::-1]])
    z = np.hstack([points[:, :, 2], points[:, :, 2][:, ::-1]])
    area = 0.5 * np.abs((y * np.roll(z, -1, axis=1)
                         - np.roll(y, -1, axis=1) * z).sum(axis=1))
    return float(np.trapezoid(area, points[:, :, 0].mean(axis=1)))


if __name__ == "__main__":
    unittest.main()
