"""APPSolver-v2 端到端：输出形状、两种位置编码与梯度。"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "model", "train"):
    sys.path.insert(0, str(ROOT / "src" / name))

from appsolver_v2 import APPSolverV2  # noqa: E402
from collate import collate_samples  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402
from tokenizer import flow_tokens  # noqa: E402

DATASET = ROOT / "dataset" / "shipbench"
ENTRY = "kcs_fr026_normal_mesh"
QUERY_COUNT = 256


def build_model(**overrides):
    config = dict(flow_size_tiers=[[0.05, 0.02], [0.2, 0.05], [None, 0.15]],
                  hull_size=0.05, d_model=32, nhead=4, encoder_layers=1, decoder_layers=1,
                  fourier_features=8)
    config.update(overrides)
    return APPSolverV2(**config)


@unittest.skipUnless((DATASET / ENTRY / "meta.json").exists(), "数据集条目不存在")
class APPSolverV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample = ShipDataset(DATASET, [ENTRY])[0]
        cls.batch = collate_samples([sample])
        nx, ny = sample["flow_coords"].shape[:2]
        flat = torch.arange(nx * ny).reshape(nx, ny)[::max(nx // 16, 1), ::max(ny // 16, 1)]
        cls.index = torch.stack([flat // ny, flat % ny], dim=-1).reshape(-1, 2)[:QUERY_COUNT]
        cls.queries = sample["flow_coords"][cls.index[:, 0], cls.index[:, 1]][None]

    def test_forward_shapes(self):
        model = build_model()
        output = model(self.batch, self.queries)
        self.assertEqual(tuple(output["flow"].shape), (1, QUERY_COUNT, 5))
        self.assertEqual(tuple(output["hull"].shape),
                         (1, self.batch["hull_uv"].shape[1], 1))

    def test_rope_positional_and_fourier_query_basis(self):
        model = build_model(positional="rope", query_basis="fourier")
        output = model(self.batch, self.queries)
        self.assertEqual(tuple(output["flow"].shape), (1, QUERY_COUNT, 5))

    def test_gradients_reach_every_parameter(self):
        model = build_model()
        output = model(self.batch, self.queries)
        loss = output["flow"].sum() + output["hull"].sum()
        loss.backward()
        missing = [name for name, parameter in model.named_parameters()
                   if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
        self.assertEqual(missing, [])

    def test_flow_only(self):
        model = build_model(input_domains=("flow",), output_domains=("flow",))
        output = model(self.batch, self.queries)
        self.assertEqual(set(output), {"flow"})
        self.assertEqual(tuple(output["flow"].shape), (1, QUERY_COUNT, 5))
        self.assertIsNone(model.hull_head)

    def test_hull_only_does_not_require_flow_queries(self):
        model = build_model(input_domains=("hull",), output_domains=("hull",))
        output = model(self.batch)
        self.assertEqual(set(output), {"hull"})
        self.assertEqual(tuple(output["hull"].shape),
                         (1, self.batch["hull_uv"].shape[1], 1))
        self.assertIsNone(model.flow_head)

    def test_hull_input_with_flow_only_output(self):
        model = build_model(output_domains=("flow",))
        output = model(self.batch, self.queries)
        self.assertEqual(set(output), {"flow"})
        self.assertEqual(tuple(output["flow"].shape), (1, QUERY_COUNT, 5))
        self.assertIsNotNone(model.hull_input)
        self.assertIsNone(model.hull_head)

    def test_flow_tokens_contain_only_coordinates(self):
        tiers = [[0.05, 0.02], [0.2, 0.05], [None, 0.15]]
        tokens, _, _ = flow_tokens(self.batch, 0, tiers)
        self.assertEqual(tokens.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
