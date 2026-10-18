"""流场域符号距离的单元测试：合成抛物线水线。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dataset"))

from sdf import footprint_profile, signed_distance  # noqa: E402

LPP = 1.0
HALF_BEAM = 0.5


def parabolic_hull(offset=0.0):
    """半船的吃水以下表面点：吃水线 y = -half_beam sin(π x)。"""
    x = np.linspace(0.0, 1.0, 201)
    y = -HALF_BEAM * np.sin(np.pi * x)
    return np.column_stack([x, y, np.full_like(x, offset)])


class SignedDistanceTest(unittest.TestCase):
    def setUp(self):
        points = parabolic_hull()
        self.profile = footprint_profile(points, draft=0.2, lpp=LPP)

    def test_distance_outside_is_euclidean(self):
        query = np.array([[0.5, -1.0], [0.5, -0.5 - 0.25]])
        distance = signed_distance(query, self.profile, LPP)
        # 中线处水线深度为半个型宽，纵向距离近似横向距离
        np.testing.assert_allclose(distance, [0.5, 0.25], atol=0.01)

    def test_inside_footprint_is_negative(self):
        query = np.array([[0.5, -0.1], [0.3, -0.05]])
        distance = signed_distance(query, self.profile, LPP)
        self.assertTrue((distance < 0).all())
        # 幅值等于到水线的欧氏距离：轮廓内凹，最近点不一定在正下方，用密集采样独立核对
        dense = np.column_stack([np.linspace(0.0, 1.0, 20001),
                                 -HALF_BEAM * np.sin(np.pi * np.linspace(0.0, 1.0, 20001))])
        expected = np.linalg.norm(query[:, None, :] - dense[None, :, :], axis=2).min(axis=1)
        np.testing.assert_allclose(-distance, expected, atol=2e-3)

    def test_zero_level_follows_waterline(self):
        x = np.linspace(0.1, 0.9, 9)
        query = np.column_stack([x, -HALF_BEAM * np.sin(np.pi * x)])
        np.testing.assert_allclose(signed_distance(query, self.profile, LPP), 0.0, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
