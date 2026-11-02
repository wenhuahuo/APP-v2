"""GINO 的结构与数值测试：GNO 核积分、谱卷积模态截断、潜网格路线与任意查询解码。

判别点：

- GNO 层必须等于「半径邻域内 `k(x,y)·f(y)` 的均值」，用稠密参考实现逐元素对照，
  这条测试能区分真正的核积分与同名 MLP 替代；
- 谱卷积只保留截断模态：把高频分量加进输入，输出不应改变；
- 输入几何 → 潜网格 → FNO → 输出查询的路线中，潜网格形状由配置决定、与输入点数和
  查询点数都无关；同一坐标的预测不随查询点集变化；
- 船体几何、工况与潜网格配置必须真实影响输出。
"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from gino import (GINOBaseline, GNOBlock, SinusoidalEmbedding, SpectralConv3d,  # noqa: E402
                  radius_neighbors)

NX, NY = 17, 9
HULL_POINTS = 40
LATENT = (6, 5, 4)


def fake_batch(condition=None):
    x = torch.linspace(-1.0, 0.6, NX)
    y = torch.linspace(-0.4, 0.0, NY)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1)
    generator = torch.Generator().manual_seed(0)
    return {
        "flow_coords": coords[None],
        "flow_extent": torch.tensor([[-1.0, -0.4, 0.6, 0.0]]),
        "flow_sdf": (grid_x ** 2 + grid_y ** 2).sqrt()[None],
        "condition": torch.zeros(1, 2) if condition is None else condition,
        "hull_uv": torch.rand(1, HULL_POINTS, 2, generator=generator),
        "hull_points": torch.rand(1, HULL_POINTS, 3, generator=generator) * 0.1,
        "hull_normals": torch.randn(1, HULL_POINTS, 3, generator=generator),
        "hull_mask": torch.ones(1, HULL_POINTS, dtype=torch.bool),
        "flow_mask": torch.ones(1, NX, NY, dtype=torch.bool),
        "flow_weight": torch.ones(1, NX, NY),
        "hull_weight": torch.full((1, HULL_POINTS), 1.0 / HULL_POINTS),
    }


def small_gino(**kwargs):
    settings = dict(latent_shape=LATENT, n_modes=(3, 2, 2), hidden_channels=16, fno_layers=2,
                    embed_channels=8, gno_embed_channels=6, gno_kernel_hidden=16,
                    in_radius=0.15, out_radius=0.25, gno_chunk=64, lifting_channels=16,
                    projection_channels=16)
    settings.update(kwargs)
    return GINOBaseline(**settings)


class SinusoidalEmbeddingTest(unittest.TestCase):
    def test_matches_transformer_formula(self):
        embedding = SinusoidalEmbedding(in_channels=3, num_frequencies=4, max_positions=10000)
        coords = torch.tensor([[0.0, 0.3, 0.7]])
        args = coords.unsqueeze(-1) * (10000.0 ** (-torch.arange(4) / 3.0))
        expected = torch.stack([args.sin(), args.cos()], dim=-1).flatten(-3)
        torch.testing.assert_close(embedding(coords), expected)
        self.assertEqual(embedding(coords).shape, (1, 2 * 4 * 3))


class RadiusNeighborTest(unittest.TestCase):
    def test_neighbor_sets_and_empty_fallback(self):
        data = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]])
        queries = torch.tensor([[0.05, 0.0, 0.0], [5.0, 5.0, 5.0]])
        index, counts = radius_neighbors(data, queries, 0.2)
        self.assertEqual(counts.tolist(), [2, 1])
        self.assertEqual(sorted(index[:2].tolist()), [0, 1])
        # 半径内没有邻居时退化为最近邻，不产生空邻域
        self.assertEqual(index[2].item(), 2)


class GNOIntegralTest(unittest.TestCase):
    def test_matches_dense_kernel_average(self):
        torch.manual_seed(0)
        block = GNOBlock(in_channels=6, out_channels=6, coord_dim=3, radius=0.45,
                         pos_embed_channels=4, kernel_hidden=8, chunk=2)
        y = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.4, 0.4, 0.0]])
        # 两个查询点都在半径内至少有一个邻居；空邻域退化由 RadiusNeighborTest 覆盖
        x = torch.tensor([[0.05, 0.0, 0.0], [0.4, 0.4, 0.0]])
        f_y = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6) / 10.0
        output = block(y, x, f_y)

        embedded_y = block.pos_embedding(y)
        embedded_x = block.pos_embedding(x)
        expected = []
        for row, point in enumerate(x):
            distances = torch.linalg.norm(y - point, dim=-1)
            members = torch.nonzero(distances <= 0.45).squeeze(-1)
            kernel = block.kernel(torch.cat([embedded_y[members],
                                             embedded_x[row].expand(len(members), -1)],
                                            dim=-1))
            expected.append((kernel * f_y[0][members]).mean(dim=0))
        torch.testing.assert_close(output[0], torch.stack(expected), rtol=1e-6, atol=1e-6)


class SpectralConvTest(unittest.TestCase):
    def test_truncated_modes_ignore_high_frequencies(self):
        torch.manual_seed(0)
        conv = SpectralConv3d(channels=2, n_modes=(2, 2, 2), spatial_shape=(8, 8, 4))
        base = torch.randn(1, 2, 8, 8, 4)
        axis = torch.arange(8).float()
        # 沿第一维的 3 周高频分量：截断到 2 个模态后应完全不可见
        high = torch.sin(2 * torch.pi * axis * 3.0)[None, None, :, None, None]
        first = conv(base)
        second = conv(base + 5.0 * high.expand(1, 2, 8, 8, 4))
        torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-4)

    def test_rejects_modes_beyond_the_grid(self):
        with self.assertRaises(ValueError):
            SpectralConv3d(channels=2, n_modes=(16, 2, 2), spatial_shape=(8, 8, 4))


class GINORouteTest(unittest.TestCase):
    def setUp(self):
        self.batch = fake_batch()
        self.model = small_gino()
        generator = torch.Generator().manual_seed(5)
        self.queries = torch.stack([torch.rand(7, generator=generator) * 1.6 - 1.0,
                                    torch.rand(7, generator=generator) * 0.4 - 0.4],
                                   dim=-1)[None]

    def test_latent_grid_shape_comes_from_configuration(self):
        state = self.model.encode(self.batch)
        self.assertEqual(tuple(state["latent"].shape), (1, 16, *LATENT))
        self.assertEqual(tuple(self.model.latent_coords.shape), (LATENT[0] * LATENT[1] * LATENT[2], 3))

    def test_queries_are_decoded_independently(self):
        output = self.model(self.batch, self.queries)
        self.assertEqual(tuple(output["flow"].shape), (1, 7, 5))
        subset = self.model(self.batch, self.queries[:, :3])
        torch.testing.assert_close(subset["flow"], output["flow"][:, :3])
        order = torch.tensor([3, 0, 6, 1, 5, 2, 4])
        permuted = self.model(self.batch, self.queries[:, order])
        torch.testing.assert_close(permuted["flow"], output["flow"][:, order])

    def test_decode_matches_forward(self):
        state = self.model.encode(self.batch)
        torch.testing.assert_close(self.model.decode(state, self.queries),
                                   self.model(self.batch, self.queries)["flow"])

    def test_hull_geometry_reaches_the_flow_field(self):
        baseline = self.model(self.batch, self.queries)["flow"]
        batch = fake_batch()
        batch["hull_points"] = batch["hull_points"] + 0.05
        moved = self.model(batch, self.queries)["flow"]
        self.assertGreater(float((moved - baseline).abs().max()), 0.0)

    def test_conditioning_and_latent_configuration_are_effective(self):
        baseline = self.model(self.batch, self.queries)["flow"]
        conditioned = self.model(fake_batch(torch.tensor([[0.3, 9.0]])), self.queries)["flow"]
        self.assertGreater(float((conditioned - baseline).abs().max()), 0.0)

        larger = small_gino(latent_shape=(8, 6, 4))
        self.assertEqual(tuple(larger.encode(self.batch)["latent"].shape), (1, 16, 8, 6, 4))

    def test_gradients_reach_every_parameter(self):
        output = self.model(self.batch, self.queries)
        output["flow"].sum().backward()
        missing = [name for name, parameter in self.model.named_parameters()
                   if parameter.grad is None]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
