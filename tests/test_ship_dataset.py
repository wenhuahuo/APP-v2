"""数据集加载：格点回填、无量纲化与 stride 抽样。"""

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "datapipe"))

from ship_dataset import ShipDataset  # noqa: E402

DATASET = ROOT / "dataset"
ENTRY = "kcs_fr026_normal_mesh"


@unittest.skipUnless((DATASET / ENTRY / "meta.json").exists(), "数据集条目不存在")
class ShipDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = json.loads((DATASET / ENTRY / "meta.json").read_text())
        cls.sample = ShipDataset(DATASET, [ENTRY])[0]

    def test_shapes_and_dtypes(self):
        nx, ny = self.sample["flow_coords"].shape[:2]
        self.assertEqual((nx, ny), tuple(self.meta["flow_field"]["domain_over_Lpp"]["n_grid"]))
        self.assertEqual(self.sample["flow_target"].shape, (nx, ny, 5))
        self.assertEqual(self.sample["flow_weight"].shape, (nx, ny))
        self.assertEqual(self.sample["flow_target"].dtype, torch.float32)
        self.assertEqual(self.sample["flow_mask"].dtype, torch.bool)
        self.assertEqual(self.sample["condition"].shape, (2,))
        self.assertEqual(self.sample["scale"].shape, (3,))

    def test_scatter_covers_every_entry_point_once(self):
        self.assertEqual(int(self.sample["flow_mask"].sum()), self.meta["flow_field"]["n_points"])

    def test_masked_points_carry_no_data(self):
        mask = self.sample["flow_mask"]
        self.assertTrue(torch.all(self.sample["flow_weight"][~mask] == 0))
        self.assertTrue(torch.all(self.sample["flow_sdf"][mask] >= 0))

    def test_weights_are_lattice_cells(self):
        spacing = self.meta["flow_field"]["domain_over_Lpp"]["spacing"]
        mask = self.sample["flow_mask"]
        self.assertAlmostEqual(float(self.sample["flow_weight"][mask].max()), spacing ** 2, places=9)

    def test_quantities_are_dimensionless(self):
        lpp = self.meta["Lpp"]
        target = self.sample["flow_target"]
        mask = self.sample["flow_mask"]
        # eta/Lpp 远小于 1；远场 u/U_inf 在 1 附近（本算例流向沿 -x）
        self.assertLess(float(target[..., 0][mask].abs().max()), 0.05)
        self.assertAlmostEqual(float(target[..., 1][mask].median()), -1.0, delta=0.3)


@unittest.skipUnless((DATASET / ENTRY / "meta.json").exists(), "数据集条目不存在")
class StrideTest(unittest.TestCase):
    def setUp(self):
        self.fine = ShipDataset(DATASET, [ENTRY])[0]
        self.coarse = ShipDataset(DATASET, [ENTRY], stride=2)[0]

    def test_weight_grows_with_cell_area(self):
        self.assertAlmostEqual(float(self.coarse["flow_weight"].max()),
                               4.0 * float(self.fine["flow_weight"].max()), places=9)

    def test_quadrature_preserves_the_integral(self):
        integral = lambda s: float((s["flow_weight"] * s["flow_target"][..., 0]).sum())
        self.assertAlmostEqual(integral(self.coarse) / integral(self.fine), 1.0, delta=0.05)

    def test_panels_share_the_coarse_mask(self):
        self.assertEqual(tuple(self.coarse["flow_mask"].shape),
                         tuple(self.fine["flow_mask"][::2, ::2].shape))


if __name__ == "__main__":
    unittest.main()
