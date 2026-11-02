"""前向计算量统计口径：CPU 融合注意力必须被计数，token 口径要覆盖 9 个主实验模型。

判别点：

- `nn.MultiheadAttention` 在 CPU 上走 `_scaled_dot_product_flash_attention_for_cpu`，
  `FlopCounterMode` 未登记该算子，不补齐时注意力一项完全不计数（UPT / CORAL / GINO 的
  perceiver、APPSolver-v1 的 TransformerEncoderLayer 都会被低估）。这里用解析值逐项
  核对：投影矩阵乘 + q@kᵀ + scores@v 之和必须与实测一致；
- 计数与重复次数无关（同一模型连续两次统计给出同一个 GFLOPs）；
- `tokens_from_batch` 对 9 个模型都给出明确口径，且感性 token 数与主算子序列长度一致。
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from flops import _attention_flops, forward_flops  # noqa: E402
from profile_model_flops import tokens_from_batch  # noqa: E402

CONFIG_DIR = ROOT / "configs"
MODELS = ("appsolver_v2", "appsolver_oformer", "appsolver_v1", "gnot", "upt",
          "transolver", "oformer", "gino", "coral")
NX, NY, HULL_POINTS = 17, 9, 40


def load(model):
    return yaml.safe_load((CONFIG_DIR / model / f"{model}_multihull_normal.yaml").read_text())


def synthetic_batch():
    """与 `collate_samples` 契约一致的合成批：17×9 自由液面格点 + 40 个船面点。"""
    x = torch.linspace(-1.0, 0.6, NX)
    y = torch.linspace(-0.4, 0.0, NY)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    generator = torch.Generator().manual_seed(0)
    return {
        "flow_coords": torch.stack([grid_x, grid_y], dim=-1)[None],
        "flow_extent": torch.tensor([[-1.0, -0.4, 0.6, 0.0]]),
        "flow_sdf": (grid_x ** 2 + grid_y ** 2).sqrt()[None],
        "flow_target": torch.randn(1, NX, NY, 5, generator=generator),
        "flow_weight": torch.ones(1, NX, NY),
        "flow_mask": torch.ones(1, NX, NY, dtype=torch.bool),
        "hull_uv": torch.rand(1, HULL_POINTS, 2, generator=generator),
        "hull_points": torch.rand(1, HULL_POINTS, 3, generator=generator) * 0.1,
        "hull_normals": torch.randn(1, HULL_POINTS, 3, generator=generator),
        "hull_pressure": torch.randn(1, HULL_POINTS, generator=generator),
        "hull_weight": torch.ones(1, HULL_POINTS),
        "hull_mask": torch.ones(1, HULL_POINTS, dtype=torch.bool),
        "condition": torch.tensor([[0.26, 8.0]]),
    }


class AttentionFlopFormulaTest(unittest.TestCase):
    def test_attention_formula_counts_both_matrices(self):
        """q@kᵀ 与 scores@v 各一次批量矩阵乘，形状 (B,H,Sq,D)×(B,H,Sk,D)。"""
        flops = _attention_flops((2, 4, 8, 16), (2, 4, 10, 16), (2, 4, 10, 16))
        self.assertEqual(flops, 2 * (2 * 4) * 8 * 10 * 16 * 2)

    def test_multihead_attention_is_counted(self):
        """实测必须等于解析值：投影 + 两个注意力矩阵乘 + 输出投影。"""
        length, width, heads = 7, 32, 4

        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(width, heads, dropout=0.0,
                                                       batch_first=True)

            def forward(self, batch, queries):
                return self.attention(queries, queries, queries, need_weights=False)[0]

        model = Wrapped().train()
        queries = torch.randn(1, length, width)
        _, gflops = forward_flops(model, {}, queries)
        head_dim = width // heads
        expected = (2 * length * width * 3 * width  # in_proj
                    + 4 * heads * length * length * head_dim  # q@kᵀ 与 scores@v
                    + 2 * length * width * width)  # out_proj
        self.assertAlmostEqual(gflops, expected / 1e9, places=6)

    def test_repeated_measurement_is_identical(self):
        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 5))

            def forward(self, batch, queries):
                return self.mlp(queries)

        model = Wrapped().train()
        queries = torch.randn(1, 11, 16)
        first = forward_flops(model, {}, queries)[1]
        second = forward_flops(model, {}, queries)[1]
        self.assertGreater(first, 0.0)
        self.assertEqual(first, second)


class TokenDefinitionTest(unittest.TestCase):
    """9 个模型的主干 token 数与口径：patch / point / latent / latent_grid。"""

    def setUp(self):
        self.batch = synthetic_batch()
        self.points = NX * NY + HULL_POINTS

    def tokens(self, model):
        return tokens_from_batch(load(model), self.batch)

    def test_patch_models_count_adaptive_patches(self):
        for model in ("appsolver_v2", "appsolver_oformer"):
            count, kind = self.tokens(model)
            self.assertEqual(kind, "patch")
            self.assertLess(count, self.points)
            self.assertGreater(count, 0)

    def test_appsolver_v1_counts_point_budget_patches(self):
        count, kind = self.tokens("appsolver_v1")
        self.assertEqual(kind, "patch")
        self.assertLess(count, self.points)
        self.assertGreater(count, 1)

    def test_point_models_count_every_input_point(self):
        for model in ("gnot", "oformer", "transolver"):
            count, kind = self.tokens(model)
            self.assertEqual((count, kind), (self.points, "point"))

    def test_latent_models_report_bottleneck(self):
        for model, expected in (("upt", 128), ("coral", 128), ("gino", 32 * 16 * 8)):
            count, kind = self.tokens(model)
            self.assertEqual(count, expected)
            self.assertEqual(kind, "latent" if model != "gino" else "latent_grid")

    def test_every_main_model_has_a_defined_budget(self):
        for model in MODELS:
            count, kind = self.tokens(model)
            self.assertIn(kind, ("patch", "point", "latent", "latent_grid"))
            self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
