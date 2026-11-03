import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/dataset"))

from ffd import (BOW_SEGMENT, KCS_BOW_VARIANTS, KCS_DECK, KCS_LATTICE, apply_ffd,
                 bernstein_matrix, bow_displacements, bow_fit_residual, bow_target,
                 bow_vertical_weight, bow_window, control_lattice, deform_bow,
                 fit_control_displacements, flipped_triangles, fold_y, hull_bounds,
                 paper_x_over_lpp, variant_amplitude, wetted_area)
from stl_io import read_stl, write_stl

LPP = 7.2786


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
    bounds = ((6.5, 7.5), (0.0, 0.5), (0.0, 0.54))

    def setUp(self):
        rng = np.random.default_rng(0)
        self.points = np.column_stack([
            rng.uniform(*self.bounds[0], 400),
            rng.uniform(*self.bounds[1], 400),
            rng.uniform(*self.bounds[2], 400),
        ])

    def test_zero_displacement_is_identity(self):
        zero = np.zeros(KCS_LATTICE + (3,))
        np.testing.assert_allclose(apply_ffd(self.points, self.bounds, zero), self.points)

    def test_points_outside_lattice_stay_fixed(self):
        outside = np.array([[8.0, 0.2, 0.3], [3.0, 0.9, 0.3], [3.0, 0.2, 1.2]])
        displacement = np.zeros(KCS_LATTICE + (3,))
        displacement[:, 1, :, 1] = 0.01
        np.testing.assert_allclose(apply_ffd(outside, self.bounds, displacement), outside)

    def test_lattice_matches_bounds(self):
        lattice = control_lattice(self.bounds)
        self.assertEqual(lattice.shape, KCS_LATTICE + (3,))
        for axis, (low, high) in enumerate(self.bounds):
            values = lattice[..., axis]
            self.assertAlmostEqual(float(values.min()), low, places=12)
            self.assertAlmostEqual(float(values.max()), high, places=12)
        for axis, (low, high) in enumerate(hull_bounds(self.points)):
            self.assertAlmostEqual(low, float(self.points[:, axis].min()), places=12)
            self.assertAlmostEqual(high, float(self.points[:, axis].max()), places=12)

    def test_uniform_translation_is_exact(self):
        # 均匀位移能被 Bernstein 基精确表示：拟合残差为机器精度量级
        target = np.zeros_like(self.points)
        target[:, 1] = 0.01
        fitted = fit_control_displacements(self.points, self.bounds, target)
        moved = apply_ffd(self.points, self.bounds, fitted)
        np.testing.assert_allclose(moved - self.points, target, atol=1e-12)

    def test_fitted_polynomial_field_is_reproduced(self):
        # 目标位移是 Bernstein 基能精确表示的多项式：反解后应逐点复现
        lattice = np.zeros(KCS_LATTICE + (3,))
        lattice[:, 1, :, 1] = np.linspace(-0.02, 0.02, KCS_LATTICE[0])[:, None]
        target = apply_ffd(self.points, self.bounds, lattice) - self.points
        fitted = fit_control_displacements(self.points, self.bounds, target)
        moved = apply_ffd(self.points, self.bounds, fitted)
        self.assertLess(np.abs((moved - self.points) - target).max(), 1e-9)


class BowFieldTest(unittest.TestCase):
    """变体场本身的形状：作用区间、上下不动、中纵剖面不动。"""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.points = np.column_stack([
            rng.uniform(6.4, 7.6, 600),
            rng.uniform(-0.3, 0.3, 600),
            rng.uniform(0.0, 0.6, 600),
        ])

    def test_paper_coordinate(self):
        np.testing.assert_allclose(paper_x_over_lpp(LPP, LPP), 0.0)
        # 球鼻艏最前端 x/Lpp = 1.034 → 论文坐标 −0.034（图 1 的曲线画到 −0.03）
        self.assertAlmostEqual(float(paper_x_over_lpp(1.034 * LPP, LPP)), -0.034, places=12)

    def test_window_support_and_peak(self):
        self.assertEqual(float(bow_window(np.array(BOW_SEGMENT[0] - 0.01))), 0.0)
        self.assertEqual(float(bow_window(np.array(BOW_SEGMENT[1] + 0.01))), 0.0)
        middle = 0.5 * (BOW_SEGMENT[0] + BOW_SEGMENT[1])
        self.assertAlmostEqual(float(bow_window(np.array(middle))), 1.0, places=12)
        self.assertGreater(float(bow_window(np.array(0.02))), 0.85)
        self.assertLess(float(bow_window(np.array(BOW_SEGMENT[0] + 0.002))), 0.02)

    def test_vertical_weight_vanishes_at_keel_and_deck(self):
        self.assertAlmostEqual(float(bow_vertical_weight(np.array(0.0), KCS_DECK)), 0.0)
        self.assertAlmostEqual(float(bow_vertical_weight(np.array(KCS_DECK), KCS_DECK)), 0.0)
        self.assertGreater(float(bow_vertical_weight(np.array(0.3), KCS_DECK)), 0.5)
        # 甲板以上（艏楼）也保持不动
        self.assertAlmostEqual(float(bow_vertical_weight(np.array(0.70), KCS_DECK)), 0.0)

    def test_target_is_proportional_to_half_breadth(self):
        # Δy / |y| 与 y 无关：这是「半宽按比例缩放」的定义，也是不翻转的保证
        target = bow_target(self.points, LPP, KCS_DECK, 0.3)
        select = np.abs(self.points[:, 1]) > 1e-3
        ratio = target[select, 1] / np.abs(self.points[select, 1])
        np.testing.assert_allclose(ratio, 0.3 * bow_window(
            paper_x_over_lpp(self.points[select, 0], LPP)) * bow_vertical_weight(
            self.points[select, 2], KCS_DECK), atol=1e-12)

    def test_target_has_no_x_or_z_component(self):
        target = bow_target(self.points, LPP, KCS_DECK, 0.3)
        np.testing.assert_allclose(target[:, 0], 0.0, atol=0.0)
        np.testing.assert_allclose(target[:, 2], 0.0, atol=0.0)

    def test_variants_are_opposite_and_mirror_symmetric(self):
        folded = fold_y(self.points)
        lattice = bow_displacements("bow_outward", self.points, LPP, KCS_DECK)
        inward = bow_displacements("bow_inward", self.points, LPP, KCS_DECK)
        np.testing.assert_allclose(inward[:, 1, :, 1], -lattice[:, 1, :, 1], atol=1e-12)
        moved = deform_bow(self.points, lattice, LPP, KCS_DECK)
        mirrored = self.points * np.array([1.0, -1.0, 1.0])
        moved_mirrored = deform_bow(mirrored, lattice, LPP, KCS_DECK)
        np.testing.assert_allclose(moved[:, 1], -moved_mirrored[:, 1], atol=1e-12)
        np.testing.assert_allclose(moved, moved_mirrored * np.array([1.0, -1.0, 1.0]),
                                   atol=1e-12)
        self.assertGreater(np.abs(folded[:, 1] - folded[:, 1]).max(), -1.0)


class BowDeformationTest(unittest.TestCase):
    """在艏区的一个封闭小球面上检查定向与体积变化（解析小面，秒级）。"""

    def setUp(self):
        # 球心放在格架内部：x ∈ [6.9, 7.1]、z ∈ [0.20, 0.40]、|y| ≤ 0.10
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        self.points = sphere.vertices * np.array([0.10, 0.10, 0.10]) + np.array([7.0, 0.0, 0.30])
        self.faces = np.asarray(sphere.faces, dtype=np.int64)
        self.volume = float(trimesh.Trimesh(self.points, self.faces, process=False).volume)

    def test_no_flipped_triangles_and_expected_volume_sign(self):
        for name, sign in (("bow_outward", 1.0), ("bow_inward", -1.0)):
            lattice = bow_displacements(name, self.points, LPP, KCS_DECK)
            moved = deform_bow(self.points, lattice, LPP, KCS_DECK)
            self.assertEqual(flipped_triangles(self.points, moved, self.faces), 0, name)
            volume = float(trimesh.Trimesh(moved, self.faces, process=False).volume)
            self.assertGreater(sign * (volume - self.volume), 0.0, name)
            # 只改 y，且中纵剖面不动
            np.testing.assert_allclose(moved[:, [0, 2]], self.points[:, [0, 2]], atol=0.0)
            centreline = np.abs(self.points[:, 1]) < 1e-12
            np.testing.assert_allclose(moved[centreline, 1], self.points[centreline, 1],
                                       atol=0.0)


class RealHullTest(unittest.TestCase):
    """有本地 KCS STL 时在其上复核变体（没有就跳过，不依赖 dataset/）。"""

    def setUp(self):
        self.stl = Path(__file__).resolve().parents[1] / "dataset/classic_hulls/kcs.stl"
        if not self.stl.exists():
            self.skipTest("本地没有 dataset/classic_hulls/kcs.stl")

    def test_variants_inherit_topology_and_stay_small(self):
        points, faces = read_stl(self.stl)
        self.assertLess(np.abs(points).max(), 10.0)
        for name in KCS_BOW_VARIANTS:
            lattice = bow_displacements(name, points, LPP, KCS_DECK)
            moved = deform_bow(points, lattice, LPP, KCS_DECK)
            # 拓扑不变：x、z 逐点不动，面片连通性原样保留
            np.testing.assert_allclose(moved[:, 0], points[:, 0], atol=0.0)
            np.testing.assert_allclose(moved[:, 2], points[:, 2], atol=0.0)
            self.assertEqual(flipped_triangles(points, moved, faces), 0, name)
            displacement = np.abs(moved[:, 1] - points[:, 1]).max()
            self.assertGreater(displacement, 1e-3, f"{name} 几乎没动")
            self.assertLess(displacement, 0.01 * LPP, f"{name} 位移过大")
            # 目标位移与 FFD 反解结果的差（拟合残差）
            residual = bow_fit_residual(points, lattice, LPP, KCS_DECK,
                                        variant_amplitude(name))
            self.assertLess(residual, 1e-4 * LPP, name)
            # 湿面积变化是小扰动量级（< 1%）
            wet = wetted_area(moved, faces)
            self.assertLess(abs(wet / wetted_area(points, faces) - 1.0), 0.01, name)


class StlIoTest(unittest.TestCase):
    def test_round_trip_plain_and_gzip(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        points = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        with tempfile.TemporaryDirectory() as folder:
            for suffix in (".stl", ".stl.gz"):
                path = Path(folder) / f"hull{suffix}"
                write_stl(path, points, faces, solid="hull")
                back, back_faces = read_stl(path)
                # STL 不共享顶点：读回来顶点数变成 3 × 面数，逐三角形比较
                self.assertEqual(len(back), 3 * len(faces))
                np.testing.assert_allclose(back[back_faces], points[faces], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
