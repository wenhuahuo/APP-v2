"""船体融合机制消融（APP+OFormer 固定 GFLOPs 容量）：配置派生与域门控。

判别点：

- 三份配置由 `make_framework_ablation_configs.py` 确定性派生：`flow_only` 与
  `flow_with_hull` 与主实验配置（`appsolver_oformer_random_seed0_isoflops`，即
  `joint_flow_hull` 组）只差输入域、输出域与 `lambda_hull`，其余模型/训练字段逐项相同；
- `flow_only` 只拼流场 token：扰动船体输入（船面点、法向、UV、压力）后预测不变，
  且船体投影不参与反传；
- `flow_with_hull` 与 `joint_flow_hull` 的预测随船体输入变化；只有联合模型构造船面输出头；
- 三组容量与主干结构相同：`flow_only` 与 `flow_with_hull` 参数量相等，联合模型只多一个
  船面输出头，符合「只改任务口径」的消融前提；
- 评测路径给联合模型输出船面 Cp 指标（该模型没有独立 `encode`，靠整片前向取 `hull`）。
"""

import sys
import types
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
from evaluate import predict_entry  # noqa: E402
import make_framework_ablation_configs as configs  # noqa: E402

OFORMER_DIR = ROOT / "configs" / "appsolver_oformer"
FLOW_ONLY = "appsolver_oformer_flow_only_random_seed0_isoflops.yaml"
FLOW_WITH_HULL = "appsolver_oformer_flow_with_hull_random_seed0_isoflops.yaml"
# 联合预测就是主实验默认口径，直接复用主实验配置（不另写一份 joint 配置）
JOINT = "appsolver_oformer_random_seed0_isoflops.yaml"
NX, NY, HULL_POINTS, QUERIES = 17, 9, 40, 7


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
    return yaml.safe_load((OFORMER_DIR / filename).read_text())


def perturb_hull(batch, scale=3.0):
    """船体输入整体扰动（点、法向、UV），流场与工况不动。"""
    disturbed = dict(batch)
    disturbed["hull_points"] = batch["hull_points"] + scale
    disturbed["hull_normals"] = batch["hull_normals"] * scale
    disturbed["hull_uv"] = torch.remainder(batch["hull_uv"] + 0.3, 1.0)
    return disturbed


class HullFusionConfigTest(unittest.TestCase):
    def test_derived_configs_match_the_generator(self):
        derived = configs.oformer_variants()
        self.assertEqual(set(derived), {FLOW_ONLY, FLOW_WITH_HULL})
        for filename, seed_texts in derived.items():
            self.assertEqual(set(seed_texts), {0, 1, 2})
            for train_seed, text in seed_texts.items():
                name = (filename if not train_seed else filename.replace(
                    "_random_seed0_isoflops",
                    f"_random_seed0_train_seed{train_seed}_isoflops"))
                with self.subTest(config=name):
                    self.assertEqual((OFORMER_DIR / name).read_text(), text)

    def test_variants_only_change_the_task_protocol(self):
        base = load(JOINT)
        self.assertEqual(base["model"]["input_domains"], ["flow", "hull"])
        self.assertEqual(base["model"]["output_domains"], ["flow", "hull"])
        self.assertEqual(base["train"]["lambda_hull"], 1.0)
        expected = {"flow_only": (["flow"], ["flow"], 0.0),
                    "flow_with_hull": (["flow", "hull"], ["flow"], 0.0)}
        for group, (inputs, outputs, lambda_hull) in expected.items():
            for train_seed in (0, 1, 2):
                filename = (f"appsolver_oformer_{group}_random_seed0_isoflops.yaml"
                            if not train_seed else
                            f"appsolver_oformer_{group}_random_seed0_train_seed"
                            f"{train_seed}_isoflops.yaml")
                with self.subTest(config=filename):
                    config = load(filename)
                    self.assertEqual(config["model_type"], base["model_type"])
                    self.assertEqual(config["train"]["seed"], train_seed)
                    self.assertEqual(config["model"]["input_domains"], inputs)
                    self.assertEqual(config["model"]["output_domains"], outputs)
                    self.assertEqual(config["train"]["lambda_hull"], lambda_hull)
                    self.assertEqual(config["stats"], base["stats"])
                    model = dict(config["model"])
                    base_model = dict(base["model"])
                    for key in ("input_domains", "output_domains"):
                        model.pop(key)
                        base_model.pop(key)
                    self.assertEqual(model, base_model)
                    train = dict(config["train"])
                    base_train = dict(base["train"])
                    train.pop("lambda_hull")
                    base_train.pop("lambda_hull")
                    train.pop("seed")
                    base_train.pop("seed")
                    self.assertEqual(train, base_train)


class HullFusionSbatchTest(unittest.TestCase):
    """数组 0-5：flow_only / flow_with_hull 各三个训练种子，与配置一一对应。"""

    def test_array_covers_six_runs(self):
        text = (ROOT / "scripts" / "train" /
                "train_app_oformer_hull_fusion.sbatch").read_text()
        self.assertIn("--array=0-5%1", text)
        self.assertIn("--job-name=app-oformer-hull-fusion", text)
        for group in ("flow_only", "flow_with_hull"):
            for train_seed in (0, 1, 2):
                suffix = "" if not train_seed else f"_train_seed{train_seed}"
                path = (OFORMER_DIR /
                        f"appsolver_oformer_{group}_random_seed0{suffix}_isoflops.yaml")
                self.assertTrue(path.is_file(), path)


class JointHullMetricTest(unittest.TestCase):
    """联合模型的评测必须给出船面 Cp，不能因为缺独立 `encode` 被静默跳过。"""

    def test_predict_entry_returns_the_hull_prediction(self):
        model = build_model(load(JOINT)).eval()
        batch = fake_batch()
        sample = {key: (value[0] if isinstance(value, torch.Tensor) and value.dim()
                        and value.shape[0] == 1 else value)
                  for key, value in batch.items()}
        sample = dict(sample, flow_coords=sample["flow_coords"].reshape(NX, NY, 2))
        normalizer = types.SimpleNamespace(flow_mean=torch.zeros(5), flow_std=torch.ones(5),
                                           hull_mean=torch.zeros(1), hull_std=torch.ones(1))
        flow, hull, index = predict_entry(model, normalizer, sample, torch.device("cpu"))
        self.assertIsNotNone(flow)
        self.assertIsNotNone(hull)
        self.assertEqual(tuple(hull.shape), (HULL_POINTS,))
        self.assertEqual(tuple(flow.shape), (int(sample["flow_mask"].sum()), 5))
        self.assertEqual(len(index), tuple(flow.shape)[0])


class HullFusionForwardTest(unittest.TestCase):
    def setUp(self):
        self.batch = fake_batch()
        self.queries = self.batch["flow_coords"].reshape(-1, 2)[:QUERIES][None]
        self.models = {filename: build_model(load(filename)).eval()
                       for filename in (FLOW_ONLY, FLOW_WITH_HULL, JOINT)}

    def test_only_the_joint_model_has_a_hull_head(self):
        for filename, model in self.models.items():
            with self.subTest(config=filename):
                output = model(self.batch, self.queries)
                if filename == JOINT:
                    self.assertEqual(set(output), {"flow", "hull"})
                    self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1))
                else:
                    self.assertEqual(set(output), {"flow"})
                self.assertEqual(tuple(output["flow"].shape), (1, QUERIES, 5))

    def test_flow_only_ignores_the_hull_input(self):
        model = self.models[FLOW_ONLY]
        with torch.no_grad():
            reference = model(self.batch, self.queries)["flow"]
            disturbed = model(perturb_hull(self.batch), self.queries)["flow"]
        self.assertEqual(float((reference - disturbed).abs().max()), 0.0)

    def test_hull_input_changes_the_flow_prediction(self):
        for filename in (FLOW_WITH_HULL, JOINT):
            with self.subTest(config=filename):
                model = self.models[filename]
                with torch.no_grad():
                    reference = model(self.batch, self.queries)["flow"]
                    disturbed = model(perturb_hull(self.batch), self.queries)["flow"]
                self.assertGreater(float((reference - disturbed).abs().max()), 0.0)

    def test_hull_projection_only_trains_with_hull_input(self):
        for filename in (FLOW_ONLY, FLOW_WITH_HULL):
            with self.subTest(config=filename):
                model = self.models[filename]
                model.zero_grad()
                model(self.batch, self.queries)["flow"].sum().backward()
                self.assertEqual(model.hull_input.weight.grad is None,
                                 filename == FLOW_ONLY)

    def test_capacity_is_shared_across_the_three_configs(self):
        counts = {filename: sum(p.numel() for p in model.parameters())
                  for filename, model in self.models.items()}
        self.assertEqual(counts[FLOW_ONLY], counts[FLOW_WITH_HULL])
        hull_head = sum(p.numel() for p in self.models[JOINT].hull_head.parameters())
        self.assertEqual(counts[JOINT], counts[FLOW_WITH_HULL] + hull_head)


if __name__ == "__main__":
    unittest.main()
