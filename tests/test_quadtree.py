"""APP 分区：覆盖性、尺寸判据与查询密度的稳定性。"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))
sys.path.insert(0, str(ROOT / "src" / "datapipe"))

from quadtree import build_partition, tiered_tolerance  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402

DATASET = ROOT / "dataset"
ENTRY = "kcs_fr026_normal_mesh"
SPACING = 0.01
TIERS = [[0.5, 0.02], [0.9, 0.05], [None, 0.15]]
DATASET_TIERS = [[0.05, 0.02], [0.2, 0.05], [None, 0.15]]


def lattice(nx=120, ny=40):
    """带一列作废点的规则格点，用来验证掩除位置不产生空 patch。"""
    xs = SPACING * np.arange(nx)
    ys = SPACING * np.arange(ny)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    keep = points[:, 0] > 0.3
    return points[keep], np.full(keep.sum(), SPACING ** 2)


class PartitionTest(unittest.TestCase):
    def setUp(self):
        self.points, self.weights = lattice()
        self.extent = (0.0, 0.0, SPACING * 119, SPACING * 39)
        self.tolerance = tiered_tolerance(self.points[:, 0], TIERS)
        self.partition = build_partition(self.points, self.weights, self.extent,
                                         self.tolerance)

    def test_every_point_belongs_to_exactly_one_patch(self):
        order = self.partition.order
        np.testing.assert_array_equal(np.sort(order), np.arange(len(self.points)))
        self.assertEqual(self.partition.starts[0], 0)
        self.assertEqual(self.partition.starts[-1], len(self.points))

    def test_patch_side_respects_tolerance(self):
        bounds = self.partition.bounds
        side = np.maximum(bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1])
        for index, low, high in zip(range(len(side)), self.partition.starts[:-1],
                                    self.partition.starts[1:]):
            self.assertLessEqual(side[index], self.tolerance[order_slice(self.partition, low, high)].min()
                                 + SPACING)

    def test_patch_area_matches_weights(self):
        np.testing.assert_allclose(self.partition.areas.sum(), self.weights.sum())

    def test_fine_tier_produces_smaller_patches(self):
        near = np.sqrt(self.partition.areas[self.partition.centres[:, 0] < 0.4])
        far = np.sqrt(self.partition.areas[self.partition.centres[:, 0] > 1.0])
        self.assertLess(near.max(), far.min())


def order_slice(partition, low, high):
    return partition.order[low:high]


class DistortedDomainTest(unittest.TestCase):
    def test_local_scale_refines_the_stretched_region(self):
        """参数域被拉伸处（同样的参数面积覆盖更大物理面积）patch 更小。"""
        xs = np.linspace(0.0, 1.0, 81)
        u, v = np.meshgrid(xs, xs, indexing="ij")
        points = np.column_stack([u.ravel(), v.ravel()])
        stretch = np.where(points[:, 0] > 0.5, 4.0, 1.0)
        weights = (1.0 / 81) ** 2 * stretch
        partition = build_partition(points, weights, (0.0, 0.0, 1.0, 1.0),
                                    np.full(len(points), 0.1), scale=None)
        left = partition.levels[partition.centres[:, 0] < 0.4].max()
        right = partition.levels[partition.centres[:, 0] > 0.6].min()
        self.assertLess(left, right)


@unittest.skipUnless((DATASET / ENTRY / "meta.json").exists(), "数据集条目不存在")
class QueryDensityTest(unittest.TestCase):
    def test_partition_is_stable_under_stride(self):
        bounds = {}
        for stride in (1, 2):
            sample = ShipDataset(DATASET, [ENTRY], stride=stride)[0]
            mask = sample["flow_mask"].numpy()
            points = sample["flow_coords"].numpy()[mask]
            weights = sample["flow_weight"].numpy()[mask]
            tolerance = tiered_tolerance(sample["flow_sdf"].numpy()[mask], DATASET_TIERS)
            partition = build_partition(points, weights,
                                        tuple(sample["flow_extent"].tolist()), tolerance)
            bounds[stride] = set(map(tuple, np.round(partition.bounds, 9)))
        overlap = len(bounds[1] & bounds[2]) / len(bounds[2])
        self.assertGreater(overlap, 0.95)


if __name__ == "__main__":
    unittest.main()
