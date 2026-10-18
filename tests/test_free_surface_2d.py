"""规则格点与散点插值到格点的单元测试。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dataset"))

from free_surface_2d import flow_lattice, interpolate_to_lattice  # noqa: E402

BOX = {"x_over_Lpp": (-1.0, 1.0), "y_over_Lpp": (-0.5, 0.0), "spacing_over_Lpp": 0.1}


class FlowLatticeTest(unittest.TestCase):
    def test_geometry_follows_box(self):
        lattice = flow_lattice(10.0, BOX)
        self.assertEqual((lattice.nx, lattice.ny), (21, 6))
        self.assertAlmostEqual(lattice.spacing, 1.0)
        self.assertEqual(lattice.points.shape, (21 * 6, 2))
        np.testing.assert_allclose(lattice.points[:, 0].min(), -10.0)
        np.testing.assert_allclose(lattice.points[:, 0].max(), 10.0)
        np.testing.assert_allclose(lattice.points[:, 1].min(), -5.0)
        np.testing.assert_allclose(lattice.points[:, 1].max(), 0.0)

    @staticmethod
    def enclosing_sample(lattice):
        """比格点域外扩一格的粗采样点：避开线性插值在凸包边界的 NaN。"""
        margin = 2.0 * lattice.spacing
        x = np.arange(lattice.origin[0] - margin,
                      lattice.origin[0] + lattice.spacing * (lattice.nx - 1) + margin,
                      lattice.spacing)
        y = np.arange(lattice.origin[1] - margin,
                      lattice.origin[1] + lattice.spacing * (lattice.ny - 1) + margin,
                      lattice.spacing)
        grid_x, grid_y = np.meshgrid(x, y)
        return np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def test_linear_field_is_reproduced(self):
        lattice = flow_lattice(10.0, BOX)
        sample = self.enclosing_sample(lattice)
        values = 2.0 * sample[:, 0] - 3.0 * sample[:, 1] + 1.0
        grid = interpolate_to_lattice(sample, values, lattice)
        expected = 2.0 * lattice.points[:, 0] - 3.0 * lattice.points[:, 1] + 1.0
        self.assertFalse(np.isnan(grid).any())
        np.testing.assert_allclose(grid, expected, atol=1e-9)

    def test_multiple_channels_are_stacked(self):
        lattice = flow_lattice(10.0, BOX)
        sample = self.enclosing_sample(lattice)
        values = np.column_stack([sample[:, 0], sample[:, 1], np.ones(len(sample))])
        grid = interpolate_to_lattice(sample, values, lattice)
        self.assertEqual(grid.shape, (21 * 6, 3))
        np.testing.assert_allclose(grid[:, 2], 1.0, atol=1e-9)

    def test_points_outside_sample_hull_are_nan(self):
        lattice = flow_lattice(10.0, BOX)
        sample = np.array([[-5.0, -2.5], [5.0, -2.5], [0.0, -1.0]])
        grid = interpolate_to_lattice(sample, np.ones(len(sample)), lattice)
        far_corner = np.argmin(np.hypot(lattice.points[:, 0] + 10.0, lattice.points[:, 1] + 5.0))
        self.assertTrue(np.isnan(grid[far_corner]))


if __name__ == "__main__":
    unittest.main()
