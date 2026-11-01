"""恒零预测基线在物理量空间计算加权 RMSE，并保持模型评测的数据结构。"""

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from evaluate import evaluate_zero_entry  # noqa: E402


class ZeroBaselineTest(unittest.TestCase):
    def test_weighted_physical_rmse_and_schema(self):
        flow_target = torch.zeros(1, 2, 5)
        flow_target[0, 0, :] = torch.tensor([0.001, 0.002, 0.003, 0.004, 0.005])
        flow_target[0, 1, :] = torch.tensor([0.003, 0.004, 0.005, 0.006, 0.007])
        sample = {
            "flow_mask": torch.ones(1, 2, dtype=torch.bool),
            "flow_target": flow_target,
            "flow_weight": torch.tensor([[1.0, 3.0]]),
            "hull_pressure": torch.tensor([1.0, 3.0]),
            "hull_weight": torch.tensor([1.0, 3.0]),
            "scale": torch.tensor([2.0, 4.0, 8.0]),
        }
        result = evaluate_zero_entry(sample)

        self.assertEqual(result["flow"]["n_points"], 2)
        self.assertEqual(result["hull"]["n_points"], 2)
        self.assertEqual(result["flow"]["channels"], ["eta", "u", "v", "w", "p_rgh"])
        self.assertEqual(result["hull"]["channels"], ["Cp"])
        scales = (2.0, 4.0, 4.0, 4.0, 1.0)
        for channel, scale in enumerate(scales):
            target = flow_target.reshape(-1, 5)[:, channel] * scale
            expected = math.sqrt(float((torch.tensor([1.0, 3.0]) * target ** 2).sum() / 4))
            self.assertAlmostEqual(result["flow"]["rmse"][channel], expected)
        self.assertAlmostEqual(result["hull"]["rmse"][0], math.sqrt(7.0), places=6)


if __name__ == "__main__":
    unittest.main()
