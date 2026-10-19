"""patch 积分 token 与求积权重注意力。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("app", "model"):
    sys.path.insert(0, str(ROOT / "src" / name))

from patch_integrate import integrate  # noqa: E402
from quad_attention import QuadAttention  # noqa: E402
from quadtree import build_partition, tiered_tolerance  # noqa: E402


class IntegrateTest(unittest.TestCase):
    def test_constant_field_reproduces_the_constant(self):
        features = torch.column_stack([torch.zeros(6, 2), torch.full((6, 1), 3.0)])
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        tokens, areas, _ = integrate(features, weights, torch.tensor([0, 2, 6]))
        np.testing.assert_allclose(tokens[:, 2].numpy(), [3.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(areas.numpy(), [3.0, 18.0], rtol=1e-6)

    def test_patch_centroid_is_weighted_mean(self):
        features = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        weights = torch.tensor([1.0, 1.0, 1.0])
        _, _, centres = integrate(features, weights, torch.tensor([0, 3]))
        np.testing.assert_allclose(centres.numpy()[0], [1 / 3, 1 / 3], rtol=1e-6)

    def test_token_converges_under_point_refinement(self):
        """同一物理 patch 在两个密度的点集上积分，token 相差远小于场的变化幅度。"""
        spacing = 0.01
        xs = spacing * np.arange(120)
        grid_x, grid_y = np.meshgrid(xs, xs, indexing="ij")
        fine = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        keep = ((np.arange(len(fine)) // 120) % 2 == 0) & ((np.arange(len(fine)) % 120) % 2 == 0)
        extent = (0.0, 0.0, xs[-1], xs[-1])

        def field(points):
            return np.sin(6 * points[:, 0]) + np.cos(5 * points[:, 1])

        tokens = {}
        for name, points, weights in (("fine", fine, np.full(len(fine), spacing ** 2)),
                                      ("coarse", fine[keep], np.full(keep.sum(), 4 * spacing ** 2))):
            partition = build_partition(points, weights, extent,
                                        np.full(len(points), 0.16))
            patch_values = field(points)[partition.order] * weights[partition.order]
            means = np.add.reduceat(patch_values, partition.starts[:-1]) / partition.areas
            tokens[name] = {tuple(np.round(bound, 9)): float(mean)
                            for bound, mean in zip(partition.bounds, means)}

        shared = set(tokens["coarse"]) & set(tokens["fine"])
        self.assertGreater(len(shared), 0.9 * len(tokens["coarse"]))
        differences = [abs(tokens["coarse"][bound] - tokens["fine"][bound]) for bound in shared]
        # 场幅值为 2，要求 token 变化在半个百分点以内
        self.assertLess(max(differences), 1e-2)


class QuadAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attention = QuadAttention(d_model=16, nhead=4)
        self.reference = torch.nn.MultiheadAttention(16, 4, batch_first=True)
        with torch.no_grad():
            self.reference.in_proj_weight.copy_(torch.cat([
                self.attention.query.weight, self.attention.key.weight,
                self.attention.value.weight]))
            self.reference.in_proj_bias.copy_(torch.cat([
                self.attention.query.bias, self.attention.key.bias,
                self.attention.value.bias]))
            self.reference.out_proj.weight.copy_(self.attention.output.weight)
            self.reference.out_proj.bias.copy_(self.attention.output.bias)

    def test_equal_weights_reduce_to_standard_attention(self):
        tokens = torch.randn(2, 7, 16)
        ours = self.attention(tokens, tokens, weight=torch.ones(2, 7))
        reference, _ = self.reference(tokens, tokens, tokens, need_weights=False)
        np.testing.assert_allclose(ours.detach().numpy(), reference.detach().numpy(), atol=1e-6)

    def test_zero_weight_masks_the_token(self):
        tokens = torch.randn(1, 4, 16)
        masked = self.attention(tokens, tokens, weight=torch.tensor([[1.0, 1.0, 1.0, 0.0]]))
        reference = self.attention(tokens[:, :3], tokens[:, :3], weight=torch.ones(1, 3))
        np.testing.assert_allclose(masked[:, :3].detach().numpy(),
                                   reference.detach().numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
