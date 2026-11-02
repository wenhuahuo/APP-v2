"""主实验公平配置：9 个模型在同一任务口径下的配置与模型行为。

判别点：

- 三个主表划分（随机 15%、留一船型 kvlcc2、留一设计 Fr）下 9 个模型的 `stats`、
  训练/验证条目和训练超参完全一致，只有模型自身结构不同；
- 每个主表配置都真实生效：模型只输出流场、不构造船面输出头、船体几何与工况都改变
  流场预测、损失只含流场项（`lambda_hull=0.0` 且模型没有 hull 输出域）；
- 历史 KCS / 消融配置不被默认值静默改写：仍保留双域输出与 `lambda_hull=1.0`。
"""

import json
import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from build import build_model  # noqa: E402
from collate import collate_samples  # noqa: E402
from normalize import Normalizer  # noqa: E402
from trainer import Trainer  # noqa: E402

CONFIG_DIR = ROOT / "configs"
MODELS = ("appsolver_v2", "appsolver_oformer", "appsolver_v1", "gnot", "upt",
          "transolver", "oformer", "gino", "coral")
SPLITS = ("random_seed0", "loto_kvlcc2", "lofo_fr_design")
SHARED_TRAIN_KEYS = ("epochs", "steps_per_epoch", "learning_rate", "weight_decay",
                     "batch_queries", "val_queries", "train_strides", "seed", "lambda_hull",
                     "device", "dataset_root")
NX, NY = 17, 9
HULL_POINTS = 40


def load(model, split):
    return yaml.safe_load((CONFIG_DIR / model / f"{model}_{split}.yaml").read_text())


def synthetic_sample():
    """与 `collate_samples` 契约一致的合成样本。"""
    x = torch.linspace(-1.0, 0.6, NX)
    y = torch.linspace(-0.4, 0.0, NY)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    generator = torch.Generator().manual_seed(0)
    return {
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
    }


def unit_normalizer():
    return Normalizer(flow={"mean": [0.0] * 5, "std": [1.0] * 5, "constant": [False] * 5},
                      hull={"mean": [0.0], "std": [1.0], "constant": [False]},
                      condition={"mean": [0.0, 0.0], "std": [1.0, 1.0]})


def queries_from(batch):
    return batch["flow_coords"].reshape(-1, 2)[:7][None]


def perturbed(batch, key, value):
    changed = dict(batch)
    changed[key] = value
    return changed


class MainConfigTest(unittest.TestCase):
    def test_every_model_has_the_three_main_splits(self):
        for split in SPLITS:
            for model in MODELS:
                with self.subTest(split=split, model=model):
                    config = load(model, split)
                    self.assertEqual(config["model_type"], model)

    def test_splits_share_data_statistics_and_training_budget(self):
        for split in SPLITS:
            reference = load(MODELS[0], split)
            for model in MODELS[1:]:
                with self.subTest(split=split, model=model):
                    config = load(model, split)
                    self.assertEqual(config["stats"], reference["stats"])
                    self.assertEqual(config["train"]["entries"],
                                     reference["train"]["entries"])
                    self.assertEqual(config["train"]["val_entries"],
                                     reference["train"]["val_entries"])
                    for key in SHARED_TRAIN_KEYS:
                        self.assertEqual(config["train"][key], reference["train"][key], key)

    def test_entries_match_the_shared_split_file(self):
        splits = {key: value for key, value
                  in json.loads((CONFIG_DIR / "splits_multihull.json").read_text()).items()
                  if not key.startswith("_")}
        for split, definition in splits.items():
            config = load(MODELS[0], split)
            self.assertEqual(config["train"]["entries"], definition["train"])
            self.assertEqual(config["train"]["val_entries"], definition["val"])
            self.assertEqual(config["stats"],
                             f"dataset/shipbench/stats_{split}.json")

    def test_task_protocol_is_unified(self):
        for split in SPLITS:
            for model in MODELS:
                with self.subTest(split=split, model=model):
                    model_config = load(model, split)["model"]
                    self.assertEqual(model_config["output_domains"], ["flow"])
                    self.assertEqual(model_config["conditioning"], "film")
                    self.assertIn("hull", model_config["input_domains"])
                    self.assertEqual(load(model, split)["train"]["lambda_hull"], 0.0)

    def test_historical_configs_keep_legacy_semantics(self):
        """KCS 与消融配置保留双域输出与 lambda_hull=1.0，不被默认值改写。"""
        for path in ("configs/gnot/gnot_kcs_normal.yaml",
                     "configs/upt/upt_kcs_fr_sweep.yaml",
                     "configs/oformer/oformer_kcs_normal.yaml",
                     "configs/appsolver_v1/appsolver_v1_kcs_normal.yaml",
                     "configs/appsolver_oformer/appsolver_oformer_kcs_normal.yaml"):
            with self.subTest(config=path):
                config = yaml.safe_load((ROOT / path).read_text())
                self.assertEqual(config["train"]["lambda_hull"], 1.0)
                self.assertNotIn("output_domains", config["model"])

        joint = yaml.safe_load(
            (CONFIG_DIR / "appsolver_v2/appsolver_v2_joint_flow_hull_random_seed0.yaml").read_text())
        self.assertEqual(joint["model"]["output_domains"], ["flow", "hull"])
        self.assertEqual(joint["train"]["lambda_hull"], 1.0)


class UnifiedForwardTest(unittest.TestCase):
    """用主表配置构造模型，验证任务口径在代码里真实生效（不只是 YAML 字段）。"""

    @classmethod
    def setUpClass(cls):
        cls.configs = {model: load(model, "random_seed0") for model in MODELS}
        cls.models = {model: build_model(config).eval()
                      for model, config in cls.configs.items()}
        cls.sample = synthetic_sample()
        cls.batch = collate_samples([cls.sample])

    def test_only_the_flow_domain_is_constructed_and_returned(self):
        queries = queries_from(self.batch)
        for model, module in self.models.items():
            with self.subTest(model=model):
                output = module(self.batch, queries)
                self.assertEqual(set(output), {"flow"})
                self.assertEqual(tuple(output["flow"].shape), (1, 7, 5))
                self.assertEqual(tuple(module.output_domains), ("flow",))
                # 船面输入投影保留（双域输入），船面输出头不应被构造
                hull_parameters = [name for name, _ in module.named_parameters()
                                   if name.startswith(("hull_head", "hull_patches",
                                                       "hull_extra"))]
                self.assertEqual(hull_parameters, [])

    def test_hull_geometry_and_condition_change_the_flow_prediction(self):
        queries = queries_from(self.batch)
        for model, module in self.models.items():
            with self.subTest(model=model):
                baseline = module(self.batch, queries)["flow"]
                moved = perturbed(self.batch, "hull_points",
                                  self.batch["hull_points"] + 0.05)
                with torch.no_grad():
                    shifted = module(moved, queries)["flow"]
                self.assertGreater(float((shifted - baseline).abs().max()), 0.0)

                conditioned = perturbed(self.batch, "condition",
                                        torch.tensor([[0.32, 9.1]]))
                with torch.no_grad():
                    changed = module(conditioned, queries)["flow"]
                self.assertGreater(float((changed - baseline).abs().max()), 0.0)

    def test_flow_only_loss_has_no_hull_term(self):
        for model, module in self.models.items():
            with self.subTest(model=model):
                trainer = object.__new__(Trainer)
                trainer.model = module
                trainer.normalizer = unit_normalizer()
                trainer.device = torch.device("cpu")
                trainer.domains = tuple(module.output_domains)
                trainer.config = {"lambda_hull": 0.0}
                index = torch.stack([torch.arange(7), torch.zeros(7, dtype=torch.long)],
                                    dim=-1)
                _, flow, hull = trainer._forward(self.sample, index)
                self.assertIsNone(hull)
                self.assertIsNotNone(flow)

    def test_historical_configs_still_predict_both_domains(self):
        """历史 KCS / 联合训练配置保持双域输出（模型构造与转发的回归）。"""
        queries = queries_from(self.batch)
        for path in ("configs/gnot/gnot_kcs_normal.yaml",
                     "configs/upt/upt_kcs_fr_sweep.yaml",
                     "configs/appsolver_v2/appsolver_v2_joint_flow_hull_random_seed0.yaml"):
            with self.subTest(config=path):
                config = yaml.safe_load((ROOT / path).read_text())
                module = build_model(config).eval()
                self.assertEqual(tuple(module.output_domains), ("flow", "hull"))
                with torch.no_grad():
                    output = module(self.batch, queries)
                self.assertEqual(set(output), {"flow", "hull"})
                self.assertEqual(tuple(output["flow"].shape), (1, 7, 5))
                self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1))

    def test_gradients_reach_every_parameter_of_the_flow_task(self):
        queries = queries_from(self.batch)
        for model, module in self.models.items():
            with self.subTest(model=model):
                module.zero_grad(set_to_none=True)
                module(self.batch, queries)["flow"].sum().backward()
                missing = [name for name, parameter in module.named_parameters()
                           if parameter.grad is None]
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
