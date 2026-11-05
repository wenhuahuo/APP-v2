"""等计算量（iso-FLOPs）配置：9 个模型在共同前向 GFLOPs 下的容量标定。

判别点：

- 每个主实验模型都有 `configs/<模型>/<模型>_multihull_isoflops.yaml`，且与
  `*_multihull_normal.yaml` 的差别只在主干容量（宽度/层数）：分词或潜网格、latent 数、
  任务口径与训练协议（epoch、步数、优化器）完全一致；例外是 CORAL 与 APPSolver-v1 的
  学习率降为 1e-4（lr=1e-3 在这两个模型上塌缩为常数场），单独断言；
- token 预算与宽度解耦：标定后的主干 token 数与基础配置相同；
- 配置可构造、可前向（合成批），形状契约与主实验一致；
- 真实输入上的前向 GFLOPs 落在 APPSolver-v2 主线实测值的 ±5% 内（历史 iso-FLOPs 对照
  同口径）；该断言用 `kcs_fr026_normal_mesh` 的真实格点，缺数据集时跳过。测量在子进程里
  逐个模型执行：`FlopCounterMode` 需要 autograd 图，单次前向的激活已占数 GB，同一进程
  连续测量 9 个模型会把内存耗尽。
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from build import build_model  # noqa: E402
from profile_model_flops import tokens_from_batch  # noqa: E402

CONFIG_DIR = ROOT / "configs"
DATASET = ROOT / "dataset" / "shipbench"
ENTRY = "kcs_fr026_normal_mesh"
MODELS = ("appsolver_v2", "appsolver_oformer", "appsolver_v1", "gnot", "upt",
          "transolver", "oformer", "gino", "coral")
ANCHOR = "appsolver_v2"
TOLERANCE = 0.05
# 每个模型被标定的容量键；其余键必须与基础配置逐项相同。
CAPACITY_KEYS = {
    "appsolver_v2": set(),
    "appsolver_oformer": {"d_model", "propagator_layers"},
    "appsolver_v1": {"d_model"},
    "gnot": {"d_model"},
    "upt": {"d_model"},
    "transolver": {"d_model"},
    "oformer": {"d_model", "propagator_layers"},
    "gino": {"gno_kernel_hidden"},
    "coral": {"siren_hidden", "modulation_hidden"},
}
# 学习率降到 1e-4 的两个模型：固定计算量容量下 lr=1e-3 会塌缩为常数场
LOW_LR_MODELS = ("coral", "appsolver_v1")
NX, NY, HULL_POINTS = 17, 9, 40


def load(model, suffix):
    return yaml.safe_load((CONFIG_DIR / model / f"{model}_{suffix}.yaml").read_text())


def synthetic_batch():
    generator = torch.Generator().manual_seed(0)
    grid = torch.stack(torch.meshgrid(torch.linspace(-1.0, 0.6, NX),
                                      torch.linspace(-0.4, 0.0, NY), indexing="ij"), dim=-1)
    return {
        "flow_coords": grid[None],
        "flow_extent": torch.tensor([[-1.0, -0.4, 0.6, 0.0]]),
        "flow_sdf": grid[..., 0].abs()[None],
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


class IsoflopsConfigTest(unittest.TestCase):
    def test_every_model_has_a_config(self):
        for model in MODELS:
            path = CONFIG_DIR / model / f"{model}_multihull_isoflops.yaml"
            self.assertTrue(path.exists(), f"缺少 {path.relative_to(ROOT)}")

    def test_only_capacity_keys_differ(self):
        for model in MODELS:
            base, calibrated = load(model, "multihull_normal"), load(model, "multihull_isoflops")
            self.assertEqual(base["model_type"], calibrated["model_type"])
            # 学习率单独断言：CORAL / APPSolver-v1 的等计算量配置降为 1e-4，其余训练项相同
            self.assertEqual({key: value for key, value in base["train"].items()
                              if key != "learning_rate"},
                             {key: value for key, value in calibrated["train"].items()
                              if key != "learning_rate"})
            self.assertEqual(calibrated["train"]["learning_rate"],
                             1e-4 if model in LOW_LR_MODELS else base["train"]["learning_rate"])
            self.assertEqual(base["stats"], calibrated["stats"])
            changed = {key for key in set(base["model"]) | set(calibrated["model"])
                       if base["model"].get(key) != calibrated["model"].get(key)}
            self.assertLessEqual(changed, CAPACITY_KEYS[model], f"{model} 改了 {changed}")

    def test_token_budget_is_unchanged(self):
        batch = synthetic_batch()
        for model in MODELS:
            self.assertEqual(tokens_from_batch(load(model, "multihull_normal"), batch),
                             tokens_from_batch(load(model, "multihull_isoflops"), batch),
                             f"{model} 的 token 预算被容量标定改动了")

    def test_configs_build_and_run(self):
        batch = synthetic_batch()
        queries = batch["flow_coords"].reshape(1, -1, 2)
        for model in MODELS:
            config = load(model, "multihull_isoflops")
            output = build_model(config).eval()(batch, queries)
            self.assertEqual(tuple(output["flow"].shape), (1, NX * NY, 5))


@unittest.skipUnless((DATASET / ENTRY / "meta.json").exists(), "数据集条目不存在")
class IsoflopsMeasurementTest(unittest.TestCase):
    """真实输入上的前向 GFLOPs：9 个模型都落在锚点的 ±5% 内。"""

    @staticmethod
    def measure(config_path):
        """在独立进程里跑 `profile_model_flops.py`，返回 (参数量, 前向 GFLOPs)。"""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "flops.json"
            subprocess.run([sys.executable, str(ROOT / "scripts" / "eval" /
                                               "profile_model_flops.py"),
                            "--configs", str(config_path), "--dataset", str(DATASET),
                            "--entry", ENTRY, "--json", str(target)],
                           capture_output=True, text=True, check=True)
            row = json.loads(Path(target).read_text())[0]
        return row["parameters"], row["gflops"]

    @classmethod
    def setUpClass(cls):
        cls.target = cls.measure(CONFIG_DIR / ANCHOR / f"{ANCHOR}_multihull_normal.yaml")[1]
        cls.measured = {model: cls.measure(CONFIG_DIR / model /
                                           f"{model}_multihull_isoflops.yaml")[1]
                        for model in MODELS}

    def test_forward_gflops_match_the_anchor(self):
        for model in MODELS:
            error = abs(self.measured[model] - self.target) / self.target
            self.assertLessEqual(error, TOLERANCE,
                                 f"{model} 的前向 GFLOPs {self.measured[model]:.1f} 偏离锚点 "
                                 f"{self.target:.1f} 超过 {TOLERANCE:.0%}")


if __name__ == "__main__":
    unittest.main()
