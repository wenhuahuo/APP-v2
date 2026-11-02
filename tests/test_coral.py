"""CORAL 的结构与数值测试：latent 调制的 SIREN 坐标神经场与任意查询解码。

判别点：

- `ModulatedSiren` 必须等于官方 `modulated_forward`：隐藏层在正弦激活之前做
  `scale·(Wx+b) + shift` 调制，最后一层不做调制；用固定权重逐层对照，这条测试能区分
  真正的 SIREN + latent 调制与普通 MLP；
- SIREN 权重按 w0 缩放初始化（首层 1/dim、隐藏层 sqrt(6/dim)/w0），`w0` 真实生效；
- latent 码经 `LatentToModulation` 决定每层调制，换 latent 就换场；
- 查询坐标独立于输入点集，同一坐标的预测不随查询点集变化。
"""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from coral import CORALBaseline, LatentToModulation, SirenLayer, ModulatedSiren  # noqa: E402

NX, NY = 17, 9
HULL_POINTS = 40


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
        "hull_points": torch.rand(1, HULL_POINTS, 3, generator=generator),
        "hull_normals": torch.randn(1, HULL_POINTS, 3, generator=generator),
        "hull_mask": torch.ones(1, HULL_POINTS, dtype=torch.bool),
        "flow_mask": torch.ones(1, NX, NY, dtype=torch.bool),
        "flow_weight": torch.ones(1, NX, NY),
        "hull_weight": torch.full((1, HULL_POINTS), 1.0 / HULL_POINTS),
    }


def small_coral(**kwargs):
    settings = dict(d_model=32, nhead=4, encoder_layers=2, latents=8, mlp_ratio=2,
                    siren_hidden=16, siren_layers=3, modulation_hidden=16,
                    modulation_layers=2)
    settings.update(kwargs)
    return CORALBaseline(**settings)


def reference_field(field, coords, modulations):
    """官方 `modulated_forward` 的逐步参考实现（`modulate_shift` / `modulate_scale`）。"""
    hidden = field.hidden_features
    middle = field.num_modulations // 2 if (field.modulate_scale and field.modulate_shift) else 0
    features = coords
    index = 0
    for layer in field.net:
        linear = F.linear(features, layer.linear.weight, layer.linear.bias)
        shift = (modulations[:, middle + index:middle + index + hidden].unsqueeze(1)
                 if field.modulate_shift else 0.0)
        scale = (modulations[:, index:index + hidden].unsqueeze(1) + 1.0
                 if field.modulate_scale else 1.0)
        features = torch.sin(layer.activation.w0 * (linear * scale + shift))
        index += hidden
    return F.linear(features, field.last_layer.linear.weight, field.last_layer.linear.bias)


class SirenLayerTest(unittest.TestCase):
    def test_initialisation_follows_siren_scaling(self):
        first = SirenLayer(2, 16, w0=30.0, is_first=True)
        hidden = SirenLayer(16, 16, w0=30.0)
        self.assertLessEqual(float(first.linear.weight.abs().max()), 1.0 / 2)
        self.assertLessEqual(float(hidden.linear.weight.abs().max()),
                             math.sqrt(6.0 / 16) / 30.0 + 1e-7)
        self.assertAlmostEqual(first.activation.w0, 30.0)

    def test_last_layer_is_not_activated(self):
        layer = SirenLayer(4, 3, is_last=True)
        inputs = torch.randn(5, 4)
        torch.testing.assert_close(layer(inputs), layer.linear(inputs))


class ModulatedSirenTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.field = ModulatedSiren(in_features=2, hidden_features=8, out_features=5,
                                    num_layers=4, w0=30.0, w0_initial=30.0)
        self.coords = torch.rand(1, 6, 2) * 2 - 1
        self.modulations = torch.randn(1, self.field.num_modulations)

    def test_matches_official_modulated_forward(self):
        output = self.field(self.coords, self.modulations)
        torch.testing.assert_close(output, reference_field(self.field, self.coords,
                                                          self.modulations))

    def test_last_layer_is_unmodulated(self):
        """只有隐藏层吃调制：改 shift 后最后一层的输入变化，但最后一层权重不变。"""
        zero = torch.zeros_like(self.modulations)
        nonzero = zero.clone()
        nonzero[:, :8] = 1.5
        first = reference_field(self.field, self.coords, zero)
        second = reference_field(self.field, self.coords, nonzero)
        self.assertGreater(float((first - second).abs().max()), 0.0)
        # 最后一层没有对应的调制量：输出通道数乘上隐藏宽度后调制量刚好用完
        self.assertEqual(self.field.num_modulations, 8 * 3)

    def test_scale_modulation_multiplies_before_activation(self):
        field = ModulatedSiren(in_features=2, hidden_features=8, out_features=5,
                              num_layers=4, modulate_scale=True)
        modulations = torch.randn(1, field.num_modulations)
        torch.testing.assert_close(field(self.coords, modulations),
                                   reference_field(field, self.coords, modulations))

    def test_latent_changes_the_field(self):
        generator = torch.Generator().manual_seed(1)
        first = self.field(self.coords, torch.randn(1, self.field.num_modulations,
                                                    generator=generator))
        second = self.field(self.coords, torch.randn(1, self.field.num_modulations,
                                                    generator=generator))
        self.assertGreater(float((first - second).abs().max()), 0.0)


class LatentToModulationTest(unittest.TestCase):
    def test_maps_latent_to_the_requested_number_of_modulations(self):
        module = LatentToModulation(latent_dim=32, num_modulations=48, hidden_dim=16,
                                    num_layers=2)
        self.assertEqual(tuple(module(torch.randn(3, 32)).shape), (3, 48))


class CORALRouteTest(unittest.TestCase):
    def setUp(self):
        self.batch = fake_batch()
        self.model = small_coral()
        generator = torch.Generator().manual_seed(5)
        self.queries = torch.stack([torch.rand(7, generator=generator) * 1.6 - 1.0,
                                    torch.rand(7, generator=generator) * 0.4 - 0.4],
                                   dim=-1)[None]

    def test_queries_are_decoded_independently(self):
        output = self.model(self.batch, self.queries)
        self.assertEqual(tuple(output["flow"].shape), (1, 7, 5))
        subset = self.model(self.batch, self.queries[:, :3])
        torch.testing.assert_close(subset["flow"], output["flow"][:, :3])
        order = torch.tensor([2, 6, 0, 4, 1, 5, 3])
        permuted = self.model(self.batch, self.queries[:, order])
        torch.testing.assert_close(permuted["flow"], output["flow"][:, order])

    def test_decode_matches_forward(self):
        state = self.model.encode(self.batch)
        torch.testing.assert_close(self.model.decode(state, self.queries),
                                   self.model(self.batch, self.queries)["flow"])

    def test_latent_modulation_drives_the_flow_field(self):
        state = self.model.encode(self.batch)
        baseline = self.model.decode(state, self.queries)
        moved = dict(state)
        moved["latent"] = state["latent"] + 0.5
        self.assertGreater(float((self.model.decode(moved, self.queries)
                                 - baseline).abs().max()), 0.0)

    def test_hull_geometry_and_condition_reach_the_flow_field(self):
        baseline = self.model(self.batch, self.queries)["flow"]
        batch = fake_batch()
        batch["hull_points"] = batch["hull_points"] + 0.1
        moved = self.model(batch, self.queries)["flow"]
        self.assertGreater(float((moved - baseline).abs().max()), 0.0)

        conditioned = self.model(fake_batch(torch.tensor([[0.3, 9.0]])),
                                 self.queries)["flow"]
        self.assertGreater(float((conditioned - baseline).abs().max()), 0.0)

    def test_condition_modulation_starts_at_zero(self):
        """解码器的工况增量零初始化：初始 SIREN 调制完全来自 latent。"""
        self.assertEqual(float(self.model.condition_modulation.weight.abs().max()), 0.0)
        self.assertEqual(float(self.model.condition_modulation.bias.abs().max()), 0.0)

    def test_without_conditioning_there_is_no_condition_branch(self):
        model = small_coral(conditioning="none")
        self.assertIsNone(model.condition_modulation)
        self.assertEqual(tuple(model(self.batch, self.queries)["flow"].shape), (1, 7, 5))

    def test_gradients_reach_every_parameter(self):
        self.model(self.batch, self.queries)["flow"].sum().backward()
        missing = [name for name, parameter in self.model.named_parameters()
                   if parameter.grad is None]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
