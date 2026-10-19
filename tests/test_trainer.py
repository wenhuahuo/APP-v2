"""训练侧的查询点采样：必须覆盖整个格点域，而不是其中一条边。"""

import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "train"):
    sys.path.insert(0, str(ROOT / "src" / name))

from trainer import Trainer  # noqa: E402


class SampleQueriesTest(unittest.TestCase):
    def test_queries_span_both_axes(self):
        dataset = types.SimpleNamespace(flow_coords=torch.zeros(701, 151, 2))
        index = Trainer._sample_queries(4096, torch.Generator().manual_seed(0), dataset)
        self.assertEqual(tuple(index.shape), (4096, 2))
        self.assertGreaterEqual(int(index[:, 0].max()), 690)
        self.assertGreaterEqual(int(index[:, 1].max()), 140)
        self.assertGreaterEqual(int(index[:, 0].min()), 0)
        self.assertLessEqual(int(index[:, 1].max()), 150)


if __name__ == "__main__":
    unittest.main()
