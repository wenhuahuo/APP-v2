"""批处理：流场面板堆叠与船面点集 padding。"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "datapipe"))

from collate import collate_samples  # noqa: E402


def fake_sample(hull_points):
    return {
        "flow_coords": torch.zeros(3, 2, 2),
        "flow_extent": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "flow_sdf": torch.zeros(3, 2),
        "flow_target": torch.zeros(3, 2, 5),
        "flow_weight": torch.ones(3, 2),
        "flow_mask": torch.ones(3, 2, dtype=torch.bool),
        "hull_points": torch.ones(hull_points, 3),
        "hull_uv": torch.ones(hull_points, 2),
        "hull_normals": torch.ones(hull_points, 3),
        "hull_pressure": torch.ones(hull_points),
        "hull_weight": torch.ones(hull_points),
        "hull_mask": torch.ones(hull_points, dtype=torch.bool),
        "condition": torch.zeros(2),
        "scale": torch.zeros(3),
    }


class CollateTest(unittest.TestCase):
    def test_hull_points_are_padded_and_masked(self):
        batch = collate_samples([fake_sample(4), fake_sample(2)])
        self.assertEqual(tuple(batch["flow_target"].shape), (2, 3, 2, 5))
        self.assertEqual(tuple(batch["hull_uv"].shape), (2, 4, 2))
        self.assertEqual(batch["hull_mask"].sum(axis=1).tolist(), [4, 2])
        self.assertTrue(torch.all(batch["hull_weight"][1, 2:] == 0))
        self.assertTrue(torch.all(batch["hull_points"][1, 2:] == 0))

    def test_mixed_query_density_is_rejected(self):
        samples = [fake_sample(2), fake_sample(2)]
        samples[1]["flow_sdf"] = torch.zeros(6, 4)
        with self.assertRaises(ValueError):
            collate_samples(samples)


if __name__ == "__main__":
    unittest.main()
