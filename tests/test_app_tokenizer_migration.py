"""APP 分词前端迁移（Transolver / UPT / GNOT）：patch 输入真实进入主干、查询独立、形状与梯度。

三个新模型共用 `PatchFrontend`（APP 双域分词），差别只在接收方主干——Transolver 的物理
切片注意力、UPT 的 perceiver latent、GNOT 的几何-物理异构注意力——与解码器（APP 坐标
查询解码 / UPT 原生解码）。这里只校验能区分这三种前端行为的不变量：

- patch 输入真实生效：主干 token 数等于分词器给出的 patch 数、远少于输入点数，换分档
  粒度后 token 数与预测都改变；
- 查询独立：同一坐标在更短的查询集里取值相同，解码器不做查询间自注意力；
- 形状与梯度：主表配置构造的模型输出 `(1,Q,5)`，五通道全有限，全部参数有梯度；
- 配置：9 份配置由生成器确定性派生，`train` 段与对应基线逐项相同（只换训练种子）。
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
import make_app_migration_configs as configs  # noqa: E402

MODELS = ("transolver", "upt", "gnot")
# 本轮实验只训练 APP+Transolver 与 APP+UPT（6 个 run）；APP+GNOT 保留配置不跑。
SBATCH_MODELS = ("appsolver_transolver", "appsolver_upt")
SBATCH = ROOT / "scripts" / "train" / "train_app_tokenizer_migration.sbatch"
SPLIT = "random_seed0"
NX, NY, HULL_POINTS, QUERIES = 17, 9, 40, 7
FINE_TIERS = [[None, 0.2]]
COARSE_TIERS = [[None, 0.4]]


def fake_batch():
    """一条合成的批处理张量：17×9 规则格点 + 40 个船面点。"""
    x = torch.linspace(-1.0, 0.6, NX)
    y = torch.linspace(-0.4, 0.0, NY)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    generator = torch.Generator().manual_seed(0)
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


def load(model, name=None):
    name = f"appsolver_{model}_{SPLIT}" if name is None else name
    return yaml.safe_load((ROOT / "configs" / f"appsolver_{model}" / f"{name}.yaml").read_text())


class MigrationConfigTest(unittest.TestCase):
    def test_configs_match_the_generator_and_the_baseline_protocol(self):
        for model in MODELS:
            path = configs.output_path(model)
            with self.subTest(model=model):
                self.assertEqual(path.read_text(), configs.derive(model))
                config = yaml.safe_load(path.read_text())
                self.assertEqual(config["model_type"], f"appsolver_{model}")
                baseline = yaml.safe_load(configs.baseline_path(model).read_text())
                self.assertEqual(config["stats"], baseline["stats"])
                for key, value in baseline["train"].items():
                    self.assertEqual(config["train"][key], value, key)
                # 接收方主干结构与宽度逐项取基线配置
                for key in configs.STRUCT_KEYS[model]:
                    self.assertEqual(config["model"][key], baseline["model"][key], key)
                self.assertEqual(config["model"]["flow_size_tiers"],
                                 configs.tokenizer_settings()["flow_size_tiers"])

    def test_receiver_backbone_comes_from_the_isoflops_baseline(self):
        """接收方主干宽度取固定 GFLOPs 基线：与它逐项相同，而不是固定参数量口径。"""
        for model in MODELS:
            with self.subTest(model=model):
                self.assertEqual(configs.baseline_path(model).name,
                                 f"{model}_{SPLIT}_isoflops.yaml")
                config = yaml.safe_load(configs.output_path(model).read_text())
                baseline = yaml.safe_load(configs.baseline_path(model).read_text())
                for key in configs.STRUCT_KEYS[model]:
                    self.assertEqual(config["model"][key], baseline["model"][key], key)

    def test_configs_use_the_single_resolution_protocol(self):
        for model in MODELS:
            with self.subTest(model=model):
                config = yaml.safe_load(configs.output_path(model).read_text())
                self.assertEqual(config["train"]["train_strides"], [1])
                self.assertEqual(config["eval"]["strides"], [1])
                self.assertNotIn("steps_per_epoch", config["train"])

    def test_sbatch_array_covers_two_models_and_three_seeds(self):
        """数组 0-5：先种子、后模型（APP+Transolver / APP+UPT），与配置一一对应。"""
        text = SBATCH.read_text()
        self.assertIn("--array=0-5%2", text)
        block = text.split("MODELS=(", 1)[1].split(")", 1)[0]
        self.assertEqual(tuple(block.split()), SBATCH_MODELS)
        self.assertIn('CONFIG="configs/${MODEL}/${MODEL}_${SPLIT}.yaml"', text)
        self.assertIn('_${SPLIT}_train_seed${TRAIN_SEED}.yaml', text)
        for model in SBATCH_MODELS:
            for train_seed in (0, 1, 2):
                suffix = "" if not train_seed else f"_train_seed{train_seed}"
                path = ROOT / "configs" / model / f"{model}_{SPLIT}{suffix}.yaml"
                self.assertTrue(path.is_file(), path)

    def test_train_seed_configs_only_change_the_seed(self):
        for model in MODELS:
            base = load(model)
            for train_seed in (1, 2):
                with self.subTest(model=model, seed=train_seed):
                    config = load(model, f"appsolver_{model}_{SPLIT}_train_seed{train_seed}")
                    self.assertEqual(config["train"]["seed"], train_seed)
                    self.assertEqual(config["model"], base["model"])
                    for key, value in base["train"].items():
                        if key != "seed":
                            self.assertEqual(config["train"][key], value, key)


class MigrationForwardTest(unittest.TestCase):
    def setUp(self):
        self.batch = fake_batch()
        self.queries = self.batch["flow_coords"].reshape(-1, 2)[:QUERIES][None]
        self.models = {model: build_model(load(model)).eval() for model in MODELS}

    def test_patch_input_drives_the_backbone(self):
        """换分档粒度同时改变 patch token 数与预测：主干输入就是 patch token。"""
        points = NX * NY + HULL_POINTS
        for model, module in self.models.items():
            with self.subTest(model=model):
                module.frontend.flow_size_tiers = FINE_TIERS
                fine_tokens = module.frontend(self.batch)[0].shape[1]
                with torch.no_grad():
                    fine = module.decode(module.encode(self.batch), self.queries[..., :2])
                module.frontend.flow_size_tiers = COARSE_TIERS
                coarse_tokens = module.frontend(self.batch)[0].shape[1]
                with torch.no_grad():
                    coarse = module.decode(module.encode(self.batch), self.queries[..., :2])
                self.assertLess(fine_tokens, points)
                self.assertLess(coarse_tokens, fine_tokens)
                self.assertGreater(float((fine - coarse).abs().max()), 0.0)

    def test_queries_are_independent(self):
        """同一坐标在更短的查询集里取值相同：解码器不做查询间自注意力。"""
        for model, module in self.models.items():
            with self.subTest(model=model):
                module.frontend.flow_size_tiers = FINE_TIERS
                with torch.no_grad():
                    state = module.encode(self.batch)
                    short = module.decode(state, self.queries[:, :4])
                    long = module.decode(state, self.queries)
                self.assertLess(float((short - long[:, :4]).abs().max()), 1e-6)

    def test_output_shape_and_gradients(self):
        for model, module in self.models.items():
            with self.subTest(model=model):
                module.frontend.flow_size_tiers = FINE_TIERS
                output = module(self.batch, self.queries)
                self.assertEqual(tuple(output["flow"].shape), (1, QUERIES, 5))
                self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1))
                self.assertTrue(bool(torch.isfinite(output["flow"]).all()))
                # 迁移模型沿用主实验口径（流场 + 船面双域输出），两个域一起反传
                (output["flow"].sum() + output["hull"].sum()).backward()
                missing = [name for name, parameter in module.named_parameters()
                           if parameter.grad is None]
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
