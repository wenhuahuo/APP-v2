"""hull morphing 的单元测试：两个解析半船（Wigley 型抛物截面）之间的融合。

半船曲面取 y = -(B/2)(1-ζ²)(2η - η²)，ζ = 2x/L - 1、η = z/T，即 Wigley 型的抛物线
剖面、最大宽在 design waterline 上：龙骨是一条 y = 0 的线（`half_hull_uv` 的源点），
水线 v = 1 落在 z = T 上。两个母船只差 B 与 T。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

from hull_morph import morph_hulls  # noqa: E402

N_STATION = 41
N_GIRTH = 21
HULL_A = {"Lpp": 1.0, "T": 0.10, "B": 0.20}
HULL_B = {"Lpp": 1.0, "T": 0.12, "B": 0.32}


def analytic_y(x, z, hull):
    """解析半船的半宽（y ≤ 0）。"""
    zeta = 2.0 * np.asarray(x) / hull["Lpp"] - 1.0
    eta = np.asarray(z) / hull["T"]
    return -0.5 * hull["B"] * (1.0 - zeta ** 2) * (2.0 * eta - eta ** 2)


def half_hull_stl(path, hull):
    """把解析半船三角化成 STL（半船，y ≤ 0、z ≤ T）。"""
    import trimesh

    stations = np.linspace(0.0, hull["Lpp"], N_STATION)
    eta = np.linspace(0.0, 1.0, N_GIRTH)
    points = np.array([
        [x, analytic_y(x, e * hull["T"], hull), e * hull["T"]]
        for x in stations for e in eta
    ])
    faces = []
    for i in range(N_STATION - 1):
        for j in range(N_GIRTH - 1):
            corner = (i * N_GIRTH + j, (i + 1) * N_GIRTH + j,
                      (i + 1) * N_GIRTH + j + 1, i * N_GIRTH + j + 1)
            faces.append((corner[0], corner[1], corner[2]))
            faces.append((corner[0], corner[2], corner[3]))
    trimesh.Trimesh(points, np.asarray(faces), process=False).export(path)
    return path


def hull_kwargs(hull):
    return {"lpp_a": hull["Lpp"], "lpp_b": hull["Lpp"],
            "draft_a": hull["T"], "draft_b": hull["T"]}


class HullMorphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(__file__).resolve().parent / ".hull_morph_tmp"
        cls.tmp.mkdir(exist_ok=True)
        cls.stl_a = half_hull_stl(cls.tmp / "hull_a.stl", HULL_A)
        cls.stl_b = half_hull_stl(cls.tmp / "hull_b.stl", HULL_B)
        cls.morphed = {
            weight: morph_hulls(cls.stl_a, cls.stl_b, HULL_A["Lpp"], HULL_B["Lpp"],
                                HULL_A["T"], HULL_B["T"], weight, n_u=48, n_v=24)
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0)
        }

    @classmethod
    def tearDownClass(cls):
        for path in cls.tmp.glob("*"):
            path.unlink()
        cls.tmp.rmdir()

    def test_endpoints_reproduce_parents(self):
        """w = 0 / w = 1 分别还原两个母船的解析曲面。"""
        for weight, hull in ((0.0, HULL_A), (1.0, HULL_B)):
            points = self.morphed[weight]["points"]
            residual = points[:, 1] - analytic_y(points[:, 0], points[:, 2], hull)
            # 水线附近 2η - η² 的导数小，坐标误差影响被放大，按 B 归一后仍有界
            self.assertLess(np.abs(residual).max() / hull["B"], 0.02)
            # T_new 是融合面 v = 1 一行 z 的中位数；`half_hull_uv` 的 v 按站位弧长
            # 归一，实测 v 到 1 时 z 比吃水低 3% 左右，故容差按 5% 给
            self.assertAlmostEqual(self.morphed[weight]["meta"]["T_new"], hull["T"],
                                   delta=0.05 * hull["T"])

    def test_midpoint_is_average_of_parents(self):
        """同一 (u, v) 格点上，w = 0.5 的坐标等于两母船插值的中点。"""
        half = self.morphed[0.5]["points"]
        average = 0.5 * (self.morphed[0.0]["points"] + self.morphed[1.0]["points"])
        self.assertEqual(half.shape, average.shape)
        np.testing.assert_allclose(half, average, atol=1e-9)

    def test_output_grid_is_valid(self):
        """面片非空、半船约束（y ≤ 0、z ∈ [0, T_new]）成立。"""
        for result in self.morphed.values():
            points, meta = result["points"], result["meta"]
            self.assertGreater(len(result["faces"]), 0)
            self.assertEqual(points.shape[0], meta["n_points"])
            self.assertLessEqual(points[:, 1].max(), 1e-9)
            self.assertGreaterEqual(points[:, 2].min(), -1e-9)
            # 半船湿表面的 z 上界是两母船的较大吃水；T_new 取的是水线行的中位数
            self.assertLessEqual(points[:, 2].max(), max(HULL_A["T"], HULL_B["T"]) * 1.02)
            self.assertTrue(0.0 <= meta["u_range"][0] < meta["u_range"][1] <= 1.0)

    def test_displacement_volume_is_monotone_in_weight(self):
        """B 与 T 都随 w 增大，排水体积必须单调上升。"""
        volumes = [self.morphed[w]["meta"]["volume"] for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertTrue(all(v is not None for v in volumes))
        self.assertTrue(np.all(np.diff(volumes) > 0), volumes)
        # 与解析体积 (4/9) B L T 对账。系统偏差有两处：v = 1 落在略低于吃水的位置
        # （约 2%），以及横剖面用 n_v 点折线内接真实曲线（约 2%），两者都偏小
        for weight, hull in ((0.0, HULL_A), (1.0, HULL_B)):
            exact = 4.0 / 9.0 * hull["B"] * hull["Lpp"] * hull["T"]
            self.assertAlmostEqual(volumes[0 if weight == 0.0 else -1] / exact, 1.0, delta=0.08)


if __name__ == "__main__":
    unittest.main()
