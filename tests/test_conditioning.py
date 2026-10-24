"""工况调制：`none` / `film` / `adaln` / `adaln_zero` 四种配置的语义、切换与非法值。

用合成的小批次与小块尺寸，不依赖数据集条目。
"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("app", "baseline", "model"):
    sys.path.insert(0, str(ROOT / "src" / name))

from appsolver_v2 import APPSolverV2  # noqa: E402
from build import build_model  # noqa: E402
from conditioning import CONDITIONINGS, AdaNorm, FiLM, NoConditioning  # noqa: E402
from encoder import Encoder, EncoderBlock  # noqa: E402
from query_decoder import DecoderBlock, QueryDecoder  # noqa: E402

BATCH = 2
NX, NY = 10, 6
HULL_POINTS = 24
QUERIES = 7
D_MODEL = 16
CONDITION = torch.tensor([[0.26, 9.0], [0.31, 8.8]])

MODEL = dict(flow_size_tiers=[[0.5, 0.4], [None, 0.6]], hull_size=0.6, d_model=D_MODEL,
             nhead=2, encoder_layers=2, decoder_layers=2, fourier_features=4)


def synthetic_batch():
    """一条合成批次：10×6 规则格点 + 24 个船面点，覆盖分词与两域形状。"""
    generator = torch.Generator().manual_seed(0)
    x = torch.linspace(0.0, 1.0, NX)
    y = torch.linspace(-0.5, 0.0, NY)
    coords = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)[None]
    coords = coords.expand(BATCH, -1, -1, -1)
    sdf = (-coords[..., 1]).clamp_min(0.0) / 0.5
    return {
        "flow_coords": coords.clone(),
        "flow_extent": torch.tensor([0.0, -0.5, 1.0, 0.0]).expand(BATCH, 4).clone(),
        "flow_sdf": sdf.clone(),
        "flow_weight": torch.full((BATCH, NX, NY), (1.0 / NX) * (0.5 / NY)),
        "flow_mask": torch.ones((BATCH, NX, NY), dtype=torch.bool),
        "hull_uv": torch.rand(BATCH, HULL_POINTS, 2, generator=generator),
        "hull_points": torch.rand(BATCH, HULL_POINTS, 3, generator=generator),
        "hull_normals": torch.randn(BATCH, HULL_POINTS, 3, generator=generator),
        "hull_weight": torch.full((BATCH, HULL_POINTS), 1.0 / HULL_POINTS),
        "hull_mask": torch.ones((BATCH, HULL_POINTS), dtype=torch.bool),
        "condition": CONDITION.clone(),
    }


def model_config(conditioning=None):
    """`build_model` 的配置字典；不给 `conditioning` 时走默认值。"""
    model = dict(MODEL)
    if conditioning is not None:
        model["conditioning"] = conditioning
    return {"model_type": "appsolver_v2", "model": model}


def block_inputs():
    tokens = torch.randn(BATCH, 5, 8)
    weight = torch.rand(BATCH, 5) + 0.1
    coords = torch.rand(BATCH, 5, 2)
    return tokens, weight, coords


class ModulationShapesTest(unittest.TestCase):
    def test_film_returns_two_channelwise_tensors(self):
        gamma, beta = FiLM(2, 8)(CONDITION)
        self.assertEqual(tuple(gamma.shape), (BATCH, 8))
        self.assertEqual(tuple(beta.shape), (BATCH, 8))

    def test_adanorm_returns_two_branches(self):
        for zero_gate, per_branch in ((False, 2), (True, 3)):
            with self.subTest(zero_gate=zero_gate):
                branches = AdaNorm(2, 8, zero_gate=zero_gate)(CONDITION)
                self.assertEqual(len(branches), 2)
                for branch in branches:
                    self.assertEqual(len(branch), per_branch)
                    for tensor in branch:
                        self.assertEqual(tuple(tensor.shape), (BATCH, 8))

    def test_adaln_zero_gate_and_projection_start_at_zero(self):
        modulation = AdaNorm(2, 8, zero_gate=True)
        for branch in modulation(CONDITION):
            for tensor in branch:
                self.assertEqual(float(tensor.abs().max()), 0.0)
        self.assertEqual(float(modulation.projection.weight.abs().max()), 0.0)
        self.assertEqual(float(modulation.projection.bias.abs().max()), 0.0)


class FiLMEquivalenceTest(unittest.TestCase):
    """film 必须保持重构前的公式：入口 `x·(1+γ)+β`，随后原注意力与 MLP。"""

    def test_encoder_block_matches_reference_formula(self):
        block = EncoderBlock(8, 2, conditioning="film")
        tokens, weight, coords = block_inputs()
        output = block(tokens, weight, coords, CONDITION)

        gamma, beta = block.modulation.projection(CONDITION).chunk(2, dim=-1)
        expected = tokens * (1.0 + gamma[:, None]) + beta[:, None]
        expected = expected + block.attention(block.norm_attention(expected),
                                             block.norm_attention(expected),
                                             weight=weight, coords_query=coords,
                                             coords_key=coords)
        expected = expected + block.mlp(block.norm_mlp(expected))
        torch.testing.assert_close(output, expected)

    def test_decoder_block_matches_reference_formula(self):
        block = DecoderBlock(8, 2, conditioning="film")
        queries = torch.randn(BATCH, 4, 8)
        tokens, weight, coords_key = block_inputs()
        coords_query = torch.rand(BATCH, 4, 2)
        output = block(queries, tokens, weight, coords_query, coords_key, CONDITION)

        gamma, beta = block.modulation.projection(CONDITION).chunk(2, dim=-1)
        expected = queries * (1.0 + gamma[:, None]) + beta[:, None]
        expected = expected + block.cross(block.norm_cross(expected), tokens, weight=weight,
                                         coords_query=coords_query, coords_key=coords_key)
        expected = expected + block.mlp(block.norm_mlp(expected))
        torch.testing.assert_close(output, expected)


class AdaNormSemanticsTest(unittest.TestCase):
    def test_adaln_modulates_after_layernorm_without_entrance_film(self):
        for kind in ("adaln", "adaln_zero"):
            with self.subTest(conditioning=kind):
                block = EncoderBlock(8, 2, conditioning=kind)
                tokens, weight, coords = block_inputs()
                seen = []
                handle = block.attention.register_forward_pre_hook(
                    lambda module, args: seen.append(args[0]))
                block(tokens, weight, coords, CONDITION)
                handle.remove()

                shift, scale = block.modulation(CONDITION)[0][:2]
                expected = block.norm_attention(tokens) * (1.0 + scale[:, None])
                expected = expected + shift[:, None]
                torch.testing.assert_close(seen[0], expected)

    def test_adaln_zero_gates_each_residual_branch_to_zero(self):
        block = EncoderBlock(8, 2, conditioning="adaln_zero")
        tokens, weight, coords = block_inputs()
        branch = []
        handle = block.attention.register_forward_hook(
            lambda module, args, output: branch.append(output))
        output = block(tokens, weight, coords, CONDITION)
        handle.remove()

        # 分支本身算出了非零结果，但初始门控为 0，block 是恒等路径。
        self.assertGreater(float(branch[0].abs().max()), 0.0)
        torch.testing.assert_close(output, tokens)

    def test_adaln_zero_decoder_block_is_identity_at_init(self):
        block = DecoderBlock(8, 2, conditioning="adaln_zero")
        queries = torch.randn(BATCH, 4, 8)
        tokens, weight, coords_key = block_inputs()
        coords_query = torch.rand(BATCH, 4, 2)
        output = block(queries, tokens, weight, coords_query, coords_key, CONDITION)
        torch.testing.assert_close(output, queries)

    def test_adaln_without_gate_is_not_identity_at_init(self):
        block = EncoderBlock(8, 2, conditioning="adaln")
        tokens, weight, coords = block_inputs()
        output = block(tokens, weight, coords, CONDITION)
        self.assertGreater(float((output - tokens).abs().max()), 0.0)

    def test_cross_attention_memory_weight_and_coords_unchanged(self):
        for kind in CONDITIONINGS:
            with self.subTest(conditioning=kind):
                block = DecoderBlock(8, 2, conditioning=kind)
                queries = torch.randn(BATCH, 4, 8)
                tokens, weight, coords_key = block_inputs()
                coords_query = torch.rand(BATCH, 4, 2)
                seen = {}
                handle = block.cross.register_forward_pre_hook(
                    lambda module, args, kwargs: seen.update(
                        query=args[0], memory=args[1], kwargs=kwargs),
                    with_kwargs=True)
                block(queries, tokens, weight, coords_query, coords_key, CONDITION)
                handle.remove()

                self.assertIs(seen["memory"], tokens)
                self.assertIs(seen["kwargs"]["weight"], weight)
                self.assertIs(seen["kwargs"]["coords_query"], coords_query)
                self.assertIs(seen["kwargs"]["coords_key"], coords_key)
                if kind == "none":
                    expected = block.norm_cross(queries)
                else:
                    modulation = block.modulation(CONDITION)
                    if kind == "film":
                        gamma, beta = modulation
                        expected = queries * (1.0 + gamma[:, None]) + beta[:, None]
                        expected = block.norm_cross(expected)
                    else:
                        shift, scale = modulation[0][:2]
                        expected = block.norm_cross(queries) * (1.0 + scale[:, None])
                        expected = expected + shift[:, None]
                torch.testing.assert_close(seen["query"], expected)


class NoConditioningTest(unittest.TestCase):
    """`none` 必须真正不使用条件，而不是把条件乘零后留下无效参数。"""

    @classmethod
    def setUpClass(cls):
        cls.batch = synthetic_batch()
        cls.queries = torch.rand(BATCH, QUERIES, 3)

    def test_none_adds_no_conditioning_parameters(self):
        model = build_model(model_config("none"))
        names = [name for name, _ in model.named_parameters()]
        self.assertEqual([name for name in names if "modulation" in name], [])
        modules = ([block.modulation for block in model.encoder.blocks]
                   + [block.modulation for block in model.decoder.blocks])
        for module in modules:
            self.assertIsInstance(module, NoConditioning)
            self.assertEqual(list(module.parameters()), [])
        film = build_model(model_config("film"))
        self.assertLess(sum(p.numel() for p in model.parameters()),
                        sum(p.numel() for p in film.parameters()))

    def test_output_is_invariant_to_condition_values(self):
        model = build_model(model_config("none"))
        model.eval()
        reference = model(self.batch, self.queries)
        for condition in (torch.zeros_like(CONDITION),
                          CONDITION * 10.0,
                          torch.full_like(CONDITION, float("nan"))):
            with self.subTest(condition=condition.tolist()):
                batch = dict(self.batch, condition=condition)
                output = model(batch, self.queries)
                torch.testing.assert_close(output["flow"], reference["flow"])
                torch.testing.assert_close(output["hull"], reference["hull"])

    def test_encoder_block_is_plain_residual_without_modulation(self):
        block = EncoderBlock(8, 2, conditioning="none")
        tokens, weight, coords = block_inputs()
        output = block(tokens, weight, coords, CONDITION)
        expected = tokens + block.attention(block.norm_attention(tokens),
                                           block.norm_attention(tokens),
                                           weight=weight, coords_query=coords,
                                           coords_key=coords)
        expected = expected + block.mlp(block.norm_mlp(expected))
        torch.testing.assert_close(output, expected)

    def test_gradients_are_finite(self):
        model = build_model(model_config("none"))
        output = model(self.batch, self.queries)
        (output["flow"].sum() + output["hull"].sum()).backward()
        bad = [name for name, parameter in model.named_parameters()
               if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
        self.assertEqual(bad, [])


class ConditioningConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = synthetic_batch()
        cls.queries = torch.rand(BATCH, QUERIES, 3)

    def test_default_is_film(self):
        model = build_model(model_config())
        self.assertEqual(model.conditioning, "film")
        for blocks in (model.encoder.blocks, model.decoder.blocks):
            self.assertIsInstance(blocks[0].modulation, FiLM)

    def test_config_switches_encoder_and_decoder_modulation(self):
        for kind in CONDITIONINGS:
            with self.subTest(conditioning=kind):
                model = build_model(model_config(kind))
                self.assertEqual(model.conditioning, kind)
                modules = ([block.modulation for block in model.encoder.blocks]
                           + [block.modulation for block in model.decoder.blocks])
                expected = {"none": NoConditioning, "film": FiLM,
                            "adaln": AdaNorm, "adaln_zero": AdaNorm}[kind]
                for module in modules:
                    self.assertIsInstance(module, expected)
                if expected is AdaNorm:
                    self.assertEqual({module.zero_gate for module in modules},
                                     {kind == "adaln_zero"})
                # 逐层独立，不跨层共享参数。
                self.assertEqual(len({id(module) for module in modules}), len(modules))

    def test_forward_shapes_for_each_mode(self):
        for kind in CONDITIONINGS:
            with self.subTest(conditioning=kind):
                output = build_model(model_config(kind))(self.batch, self.queries)
                self.assertEqual(tuple(output["flow"].shape), (BATCH, QUERIES, 5))
                self.assertEqual(tuple(output["hull"].shape), (BATCH, HULL_POINTS, 1))

    def test_gradients_are_finite_for_each_mode(self):
        for kind in CONDITIONINGS:
            with self.subTest(conditioning=kind):
                model = build_model(model_config(kind))
                output = model(self.batch, self.queries)
                (output["flow"].sum() + output["hull"].sum()).backward()
                bad = [name for name, parameter in model.named_parameters()
                       if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
                self.assertEqual(bad, [])

    def test_unknown_conditioning_raises(self):
        with self.assertRaises(ValueError):
            build_model(model_config("soft_prompt"))
        with self.assertRaises(ValueError):
            Encoder(D_MODEL, 2, 1, conditioning="adaln-x")
        with self.assertRaises(ValueError):
            QueryDecoder(D_MODEL, 2, 1, conditioning="")


if __name__ == "__main__":
    unittest.main()
