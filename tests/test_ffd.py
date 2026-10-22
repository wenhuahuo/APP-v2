import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/dataset"))

from ffd import (KCS_VARIANTS, apply_ffd, bernstein_matrix, control_lattice,
                 fit_control_displacements, kcs_displacements)


class BernsteinTest(unittest.TestCase):
    def test_partition_of_unity(self):
        t = np.linspace(0.0, 1.0, 21)
        for count in (2, 4, 7):
            basis = bernstein_matrix(t, count)
            np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)

    def test_interpolates_end_control_points(self):
        basis = bernstein_matrix(np.array([0.0, 1.0]), 5)
        np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(basis[1], [0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-12)


class FreeFormDeformationTest(unittest.TestCase):
    bounds = ((0.0, 7.0), (-0.5, 0.5), (0.0, 0.75))

    def setUp(self):
        rng = np.random.default_rng(0)
        self.points = np.column_stack([
            rng.uniform(*self.bounds[0], 400),
            rng.uniform(*self.bounds[1], 400),
            rng.uniform(*self.bounds[2], 400),
        ])

    def test_zero_displacement_is_identity(self):
        zero = np.zeros((9, 5, 6, 3))
        np.testing.assert_allclose(apply_ffd(self.points, self.bounds, zero), self.points)

    def test_points_outside_lattice_stay_fixed(self):
        outside = np.array([[8.0, 0.2, 0.3], [3.0, 0.9, 0.3], [3.0, 0.2, 1.2]])
        displacement = kcs_displacements("wide_midship", self.points, self.bounds)
        np.testing.assert_allclose(apply_ffd(outside, self.bounds, displacement), outside)

    def test_fitted_displacement_reproduces_target(self):
        target = KCS_VARIANTS["wide_midship"][0](self.points, self.bounds)
        fitted = fit_control_displacements(self.points, self.bounds, target)
        moved = apply_ffd(self.points, self.bounds, fitted)
        residual = np.abs((moved - self.points) - target).max()
        self.assertLess(residual, 0.35 * np.abs(target).max())

    def test_centerline_target_has_no_lateral_shift(self):
        centerline = np.column_stack([np.linspace(0.1, 6.9, 50), np.zeros(50),
                                      np.linspace(0.0, 0.7, 50)])
        for name in KCS_VARIANTS:
            target = KCS_VARIANTS[name][0](centerline, self.bounds)
            np.testing.assert_allclose(target[:, 1], 0.0, atol=1e-15, err_msg=name)

    def test_lateral_target_is_mirror_symmetric(self):
        mirrored = self.points * np.array([1.0, -1.0, 1.0])
        for name in ("wide_midship", "bow_flare_underwater", "bilge_rounding"):
            forward = KCS_VARIANTS[name][0](self.points, self.bounds)
            backward = KCS_VARIANTS[name][0](mirrored, self.bounds)
            np.testing.assert_allclose(forward[:, 1], -backward[:, 1], atol=1e-12)

    def test_variant_amplitude_stays_small(self):
        length = self.bounds[0][1] - self.bounds[0][0]
        for name in KCS_VARIANTS:
            shift = apply_ffd(self.points, self.bounds,
                              kcs_displacements(name, self.points, self.bounds))
            maximum = np.abs(shift - self.points).max()
            self.assertGreater(maximum, 1e-4, f"{name} 没有任何位移")
            self.assertLess(maximum, 0.02 * length, f"{name} 位移 {maximum:.3f} m 过大")


if __name__ == "__main__":
    unittest.main()
