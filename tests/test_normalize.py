"""归一化：加权统计量与标准化。"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "datapipe"))

from normalize import Normalizer  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402

DATASET = ROOT / "dataset" / "shipbench"
ENTRY = "kcs_fr026_normal_mesh"


@unittest.skipUnless((DATASET / ENTRY / "meta.json").exists(), "数据集条目不存在")
class NormalizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ShipDataset(DATASET, [ENTRY])
        cls.sample = cls.dataset[0]
        cls.normalizer = Normalizer.fit(cls.dataset)

    def _weighted_moments(self, values, weights):
        weight = weights[:, None]
        return ((values * weight).sum(axis=0) / weight.sum()).tolist(), \
               ((values ** 2 * weight).sum(axis=0) / weight.sum()).tolist()

    def test_flow_target_is_standardized(self):
        mask = self.sample["flow_mask"]
        values = self.normalizer.flow(self.sample["flow_target"])[mask]
        mean, square = self._weighted_moments(values, self.sample["flow_weight"][mask])
        for m, s in zip(mean, square):
            self.assertAlmostEqual(m, 0.0, places=5)
            self.assertAlmostEqual(s, 1.0, places=3)

    def test_hull_pressure_is_standardized(self):
        values = self.normalizer.hull(self.sample["hull_pressure"])[:, None]
        mean, square = self._weighted_moments(values, self.sample["hull_weight"])
        self.assertAlmostEqual(mean[0], 0.0, places=5)
        self.assertAlmostEqual(square[0], 1.0, places=3)

    def test_constant_condition_maps_to_zero(self):
        self.assertTrue(torch.all(self.normalizer.condition(self.sample["condition"]) == 0))


if __name__ == "__main__":
    unittest.main()
