"""APP+OFormer 全量消融（固定 GFLOPs 容量）：分词与工况融合的配置派生与真实生效。

判别点：

- 配置：均匀/学习分块各三份、五种工况融合各三份，均由生成器确定性派生；除目标字段
  与 `train.seed` 外与 `appsolver_oformer_random_seed0_isoflops.yaml` 逐项相同，自适应
  分块与 `film` 组直接复用该配置；
- 分词：`uniform` / `learned` / `adaptive` 三种分词真的换掉流场 token（数量与预测都变），
  学习分块的分配参数参与反传，船体 token 与查询解码不变；
- 工况融合：`film` 与重构前公式逐项一致；`none` 不引入任何调制参数、预测与工况取值
  无关；`adaln_zero` 初始为恒等路径；`cond_xattn` 初始与原块等价、开闸后随工况变化；
  `fourier_add` 初始为零注入；`cond_token` 在 token 序列头部加一个 token。
"""

import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from build import build_model  # noqa: E402
from collate import collate_samples  # noqa: E402
from conditioning import ConditionCrossAttention, FiLM, NoConditioning  # noqa: E402
from oformer import EncoderBlock  # noqa: E402
import make_conditioning_ablation_configs as conditioning_configs  # noqa: E402
import make_partition_ablation_configs as partition_configs  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "appsolver_oformer"
BASE = "appsolver_oformer_random_seed0_isoflops.yaml"
CONDITIONINGS = ("none", "film", "adaln_zero", "fourier_add", "cond_token", "cond_xattn")
NX, NY, HULL_POINTS, QUERIES = 17, 9, 40, 7
CONDITION = torch.tensor([[0.26, 9.0]])
BLOCK = dict(d_model=8, nhead=2, mlp_ratio=4.0, min_freq=0.25, scale=16.0, dropout=0.0)


def fake_batch(seed=0):
    """一条合成的批处理张量：17×9 规则格点 + 40 个船面点。"""
    x = torch.linspace(-1.0, 0.6, NX)
    y = torch.linspace(-0.4, 0.0, NY)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    generator = torch.Generator().manual_seed(seed)
    return collate_samples([{
        "flow_coords": torch.stack([grid_x, grid_y], dim=-1),
        "flow_extent": torch.tensor([-1.0, -0.4, 0.6, 0.0]),
        "flow_sdf": (grid_x ** 2 + grid_y ** 2).sqrt(),
        "flow_target": torch.randn(NX, NY, 5, generator=generator),
        "flow_weight": torch.ones(NX, NY),
        "flow_mask": torch.ones(NX, NY, dtype=torch.bool),
        "hull_uv": torch.rand(HULL_POINTS, 2, generator=generator),
        "hull_points": torch.rand(HULL_POINTS, 3, generator=generator) * 0.1,
        "hull_normals": torch.randn(HULL_POINTS, 3, generator=generator),
        "hull_pressure": torch.randn(HULL_POINTS, generator=generator),
        "hull_weight": torch.full((HULL_POINTS,), 1.0 / HULL_POINTS),
        "hull_mask": torch.ones(HULL_POINTS, dtype=torch.bool),
        "condition": torch.zeros(2),
        "scale": torch.ones(3),
    }])


def load(filename):
    return yaml.safe_load((CONFIG_DIR / filename).read_text())


def build(filename, **overrides):
    config = load(filename)
    return build_model({"model_type": config["model_type"],
                        "model": dict(config["model"], **overrides)})


def derived_name(filename, train_seed):
    if not train_seed:
        return filename
    return filename.replace("_random_seed0_isoflops",
                            f"_random_seed0_train_seed{train_seed}_isoflops")


def block(kind):
    return EncoderBlock(conditioning=kind, **BLOCK)


class PartitionConfigTest(unittest.TestCase):
    def test_variants_match_the_generator(self):
        for filename, seed_texts in partition_configs.oformer_variants().items():
            for train_seed, text in seed_texts.items():
                name = derived_name(filename, train_seed)
                with self.subTest(config=name):
                    self.assertEqual((CONFIG_DIR / name).read_text(), text)

    def test_variants_only_change_the_flow_tokenizer(self):
        base = load(BASE)
        self.assertNotIn("flow_tokenizer", base["model"])
        self.assertEqual(base["model"]["input_domains"], ["flow", "hull"])
        expected = {"uniform": {"flow_tokenizer": "uniform", "flow_uniform_size": 0.1},
                    "learned": {"flow_tokenizer": "learned", "flow_learned_tokens": 512}}
        for name, settings in expected.items():
            for train_seed in (0, 1, 2):
                filename = derived_name(
                    f"appsolver_oformer_{name}_partition_random_seed0_isoflops.yaml",
                    train_seed)
                with self.subTest(config=filename):
                    config = load(filename)
                    self.assertEqual({key: config["model"][key] for key in settings}, settings)
                    self.assertEqual(config["train"]["seed"], train_seed)
                    model = dict(config["model"])
                    for key in settings:
                        model.pop(key)
                    self.assertEqual(model, dict(base["model"]))
                    self.assertEqual(dict(config["train"], seed=base["train"]["seed"]),
                                     base["train"])


class ConditioningConfigTest(unittest.TestCase):
    def test_variants_match_the_generator(self):
        for filename, seed_texts in conditioning_configs.oformer_configs().items():
            for train_seed, text in seed_texts.items():
                name = derived_name(filename, train_seed)
                with self.subTest(config=name):
                    self.assertEqual((CONFIG_DIR / name).read_text(), text)

    def test_variants_only_change_the_conditioning(self):
        base = load(BASE)
        self.assertEqual(base["model"]["conditioning"], "film")
        self.assertEqual(set(conditioning_configs.OFORMER_METHODS),
                         set(CONDITIONINGS) - {"film"})
        for method in CONDITIONINGS:
            if method == "film":
                continue
            for train_seed in (0, 1, 2):
                filename = derived_name(
                    f"appsolver_oformer_conditioning_{method}_random_seed0_isoflops.yaml",
                    train_seed)
                with self.subTest(config=filename):
                    config = load(filename)
                    self.assertEqual(config["model"]["conditioning"], method)
                    self.assertEqual(config["train"]["seed"], train_seed)
                    self.assertEqual(dict(config["model"], conditioning="film"), base["model"])
                    self.assertEqual(dict(config["train"], seed=base["train"]["seed"]),
                                     base["train"])


class AblationSbatchTest(unittest.TestCase):
    """两个消融数组：分词 6 个元素、工况融合 15 个元素，元素与配置一一对应。"""

    def test_partition_array_covers_six_runs(self):
        text = (ROOT / "scripts" / "train" /
                "train_app_oformer_partition_ablation.sbatch").read_text()
        self.assertIn("--array=0-5%1", text)
        self.assertIn("--job-name=app-oformer-partition-ablation", text)
        block = text.split("CONFIGS=(", 1)[1].split(")", 1)[0]
        configs = tuple(block.split())
        self.assertEqual(len(configs), 6)
        for path in configs:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_conditioning_array_covers_fifteen_runs(self):
        text = (ROOT / "scripts" / "train" /
                "train_app_oformer_conditioning_ablation.sbatch").read_text()
        self.assertIn("--array=0-14%1", text)
        self.assertIn("--job-name=app-oformer-conditioning-ablation", text)
        block = text.split("CONFIGS=(", 1)[1].split(")", 1)[0]
        configs = tuple(block.split())
        self.assertEqual(len(configs), 15)
        for path in configs:
            self.assertTrue((ROOT / path).is_file(), path)


class PartitionForwardTest(unittest.TestCase):
    def setUp(self):
        self.batch = fake_batch()
        self.queries = self.batch["flow_coords"].reshape(-1, 2)[:QUERIES][None]

    def token_counts(self, model):
        flow, hull = model._patches(self.batch)
        return len(flow[0]), len(hull[0])

    def test_uniform_and_learned_really_change_the_flow_tokens(self):
        adaptive = build(BASE).eval()
        uniform = build(BASE, flow_tokenizer="uniform", flow_uniform_size=0.2).eval()
        learned = build(BASE, flow_tokenizer="learned", flow_learned_tokens=16).eval()
        adaptive_count = self.token_counts(adaptive)
        uniform_count = self.token_counts(uniform)
        learned_count = self.token_counts(learned)
        # 船体 UV patch 不随流场分词方法改变
        self.assertEqual(adaptive_count[1], uniform_count[1])
        self.assertEqual(adaptive_count[1], learned_count[1])
        self.assertNotEqual(uniform_count[0], adaptive_count[0])
        self.assertEqual(learned_count[0], 16)
        with torch.no_grad():
            outputs = [model(self.batch, self.queries)["flow"]
                       for model in (adaptive, uniform, learned)]
        self.assertGreater(float((outputs[0] - outputs[1]).abs().max()), 0.0)
        self.assertGreater(float((outputs[0] - outputs[2]).abs().max()), 0.0)

    def test_learned_assignment_parameters_are_trained(self):
        model = build(BASE, flow_tokenizer="learned", flow_learned_tokens=16)
        model(self.batch, self.queries)["flow"].sum().backward()
        for name in ("assignment_key.weight", "token_embed"):
            self.assertIsNotNone(model.learned_flow.get_parameter(name).grad, name)
        self.assertIsNone(build(BASE).learned_flow)

    def test_unknown_tokenizer_raises(self):
        with self.assertRaises(ValueError):
            build(BASE, flow_tokenizer="quadtree")


class ConditioningForwardTest(unittest.TestCase):
    def setUp(self):
        self.batch = fake_batch()
        self.queries = self.batch["flow_coords"].reshape(-1, 2)[:QUERIES][None]

    def test_film_matches_the_reference_formula(self):
        """重构后的 `film` 必须保持入口 `x·(1+γ)+β`，随后原注意力与 MLP。"""
        module = block("film")
        self.assertIsInstance(module.modulation, FiLM)
        tokens = torch.randn(1, 5, BLOCK["d_model"])
        coordinates = torch.rand(1, 5, 2)
        output = module(tokens, coordinates, CONDITION)

        gamma, beta = module.modulation.projection(CONDITION).chunk(2, dim=-1)
        expected = tokens * (1.0 + gamma[:, None]) + beta[:, None]
        expected = expected + module.attention(module.norm_attention(expected), coordinates)
        expected = expected + module.mlp(module.norm_mlp(expected))
        torch.testing.assert_close(output, expected)

    def test_none_has_no_modulation_parameters_and_ignores_the_condition(self):
        model = build(BASE, conditioning="none").eval()
        names = [name for name, _ in model.named_parameters() if "modulation" in name]
        self.assertEqual(names, [])
        self.assertIsInstance(model.encoder[0].modulation, NoConditioning)
        with torch.no_grad():
            reference = model(self.batch, self.queries)["flow"]
            shifted = model(dict(self.batch, condition=torch.full_like(
                self.batch["condition"], 3.0)), self.queries)["flow"]
        torch.testing.assert_close(reference, shifted)

    def test_adaln_zero_starts_as_identity_and_adaln_does_not(self):
        tokens = torch.randn(1, 5, BLOCK["d_model"])
        coordinates = torch.rand(1, 5, 2)
        torch.testing.assert_close(block("adaln_zero")(tokens, coordinates, CONDITION), tokens)
        changed = block("adaln")(tokens, coordinates, CONDITION) - tokens
        self.assertGreater(float(changed.abs().max()), 0.0)

    def test_condition_cross_attention_is_closed_at_init_and_opens_with_the_gate(self):
        module = block("cond_xattn")
        plain = block("none")
        self.assertIsInstance(module.condition_cross, ConditionCrossAttention)
        for name in ("attention", "mlp", "norm_attention", "norm_mlp"):
            getattr(plain, name).load_state_dict(getattr(module, name).state_dict())
        tokens = torch.randn(1, 5, BLOCK["d_model"])
        coordinates = torch.rand(1, 5, 2)
        torch.testing.assert_close(module(tokens, coordinates, CONDITION),
                                   plain(tokens, coordinates, CONDITION))
        module.condition_cross.gate.data.fill_(0.5)
        self.assertFalse(torch.allclose(module(tokens, coordinates, CONDITION),
                                        module(tokens, coordinates, CONDITION * 2)))

    def test_fourier_add_starts_as_zero_injection(self):
        self.assertEqual(float(block("fourier_add").modulation(CONDITION).abs().max()), 0.0)

    def test_condition_token_is_prepended(self):
        model = build(BASE, conditioning="cond_token").eval()
        width = model.flow_input.out_features
        tokens = torch.randn(5, width)
        centres = torch.rand(5, 2)
        combined, combined_centres = model._add_condition_token(tokens, centres, CONDITION)
        self.assertEqual(tuple(combined.shape), (6, width))
        torch.testing.assert_close(combined[1:], tokens)
        torch.testing.assert_close(combined_centres[0], torch.zeros(2))
        self.assertIsNotNone(model.condition_token)
        self.assertIsNone(build(BASE).condition_token)

    def test_every_mode_runs_and_backpropagates(self):
        for method in CONDITIONINGS:
            with self.subTest(conditioning=method):
                model = build(BASE, conditioning=method)
                output = model(self.batch, self.queries)["flow"]
                self.assertEqual(tuple(output.shape), (1, QUERIES, 5))
                output.sum().backward()
                bad = [name for name, parameter in model.named_parameters()
                       if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
                self.assertEqual(bad, [])

    def test_encoder_and_decoder_share_the_same_mode(self):
        for method in CONDITIONINGS:
            with self.subTest(conditioning=method):
                model = build(BASE, conditioning=method)
                self.assertEqual({item.kind for item in model.encoder}, {method})
                self.assertEqual({item.kind for item in model.decoder}, {method})

    def test_capacity_order(self):
        """调制参数随机制不同：`none` < `film` < `adaln_zero` < `cond_xattn`。"""
        sizes = {method: sum(parameter.numel()
                             for parameter in build(BASE, conditioning=method).parameters())
                 for method in CONDITIONINGS}
        self.assertLess(sizes["none"], sizes["film"])
        self.assertLess(sizes["film"], sizes["adaln_zero"])
        self.assertLess(sizes["adaln_zero"], sizes["cond_xattn"])


if __name__ == "__main__":
    unittest.main()
