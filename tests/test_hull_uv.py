"""半船 UV 参数化的单元测试：合成四分之一圆剖面网格上的解析解。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dataset"))

from hull_uv import half_hull_uv  # noqa: E402

RADIUS = 0.5
N_STATION = 6
N_GIRTH = 13


def quarter_shell():
    """x∈[0,1]、θ∈[0,π/2] 的四分之一圆剖面壳体：(x, -R sinθ, R (1 - cosθ))。"""
    stations = np.linspace(0.0, 1.0, N_STATION)
    theta = np.linspace(0.0, 0.5 * np.pi, N_GIRTH)
    grid = np.array([
        [[x, -RADIUS * np.sin(t), RADIUS * (1.0 - np.cos(t))] for t in theta]
        for x in stations
    ])
    points = grid.reshape(-1, 3)
    polygons = []
    for i in range(N_STATION - 1):
        for j in range(N_GIRTH - 1):
            corner = [i * N_GIRTH + j, (i + 1) * N_GIRTH + j,
                      (i + 1) * N_GIRTH + j + 1, i * N_GIRTH + j + 1]
            polygons.append(corner)
    offsets = np.arange(0, 4 * len(polygons) + 1, 4, dtype=np.int64)
    return points, offsets, np.asarray(polygons, dtype=np.int64).ravel()


class HalfHullUvTest(unittest.TestCase):
    def test_v_is_relative_section_arc_length(self):
        points, offsets, connectivity = quarter_shell()
        result = half_hull_uv(points, offsets, connectivity, lpp=1.0, draft=RADIUS)
        uv = result["uv"]

        expected_v = np.tile(np.linspace(0.0, 1.0, N_GIRTH), N_STATION)
        np.testing.assert_allclose(uv[:, 1], expected_v, atol=1e-9)
        np.testing.assert_allclose(uv[:, 0], np.repeat(np.linspace(0.0, 1.0, N_STATION),
                                                       N_GIRTH), atol=1e-12)

    def test_vertex_area_sums_to_wetted_area(self):
        points, offsets, connectivity = quarter_shell()
        result = half_hull_uv(points, offsets, connectivity, lpp=1.0, draft=RADIUS)
        # 三角化把圆弧换成弦，面积略小于解析值
        wetted = 1.0 * (0.5 * np.pi * RADIUS)
        np.testing.assert_allclose(result["vertex_area"].sum(), wetted, rtol=1e-3)

    def test_normals_point_outward(self):
        points, offsets, connectivity = quarter_shell()
        result = half_hull_uv(points, offsets, connectivity, lpp=1.0, draft=RADIUS)
        normals = result["normal"]
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
        # 剖面圆心在 (y, z) = (0, R)：外法向与「顶点减圆心」同向
        radial = points - np.array([0.0, 0.0, RADIUS])
        radial[:, 0] = 0.0
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        self.assertGreater((np.sum(normals * radial, axis=1) > 0.5).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
