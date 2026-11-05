"""多分辨率训练验证实验：配置派生、训练/评测档位与数组作业覆盖。

判别点：

- 9 个模型 × 训练种子 0/1/2 共 27 份配置由 `make_multires_configs.py` 从主实验
  `random_seed0_isoflops` 配置确定性派生，只有两处不同：`train_strides: [1, 2]`
  （每条条目在 1/2 中随机取一档）与 `eval.strides: [1, 2, 4, 8]`；
- 数据划分、统计量、容量、优化超参与训练轮数逐项沿用基准配置；训练种子变体只改
  `train.seed` 与头注释；
- 数组作业 27 个元素（先种子、后模型）、`%2`，元素与配置一一对应；拥有 `encode` /
  `decode` 接口的 6 个模型额外做固定输入编码（`encode_stride = 1`）评测。
"""

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import make_multires_configs as configs  # noqa: E402
import make_split_configs  # noqa: E402

MODELS = make_split_configs.MODELS
TRAIN_SEEDS = (0, 1, 2)
SBATCH = ROOT / "scripts" / "train" / "train_multires_experiment.sbatch"
ENCODE_MODELS = ("appsolver_v2", "appsolver_oformer", "oformer", "upt", "gino", "coral")


def load(path):
    return yaml.safe_load(path.read_text())


class MultiresConfigTest(unittest.TestCase):
    def test_configs_match_the_generator(self):
        for model in MODELS:
            text = configs.derive(model)
            for train_seed in TRAIN_SEEDS:
                path = configs.output_path(model, train_seed)
                with self.subTest(model=model, seed=train_seed):
                    suffix = "" if not train_seed else f"_train_seed{train_seed}"
                    self.assertEqual(
                        path.name,
                        f"{model}_multires_{configs.SPLIT}{suffix}_isoflops.yaml")
                    self.assertEqual(path.read_text(), text if not train_seed
                                     else make_split_configs.derive_train_seed_from_text(
                                         text, configs.base_path(model).name, train_seed,
                                         configs.SPLIT))

    def test_only_the_resolution_protocol_changes(self):
        for model in MODELS:
            base = load(configs.base_path(model))
            for train_seed in TRAIN_SEEDS:
                with self.subTest(model=model, seed=train_seed):
                    config = load(configs.output_path(model, train_seed))
                    self.assertEqual(config["train"]["train_strides"], [1, 2])
                    self.assertEqual(config["eval"]["strides"], [1, 2, 4, 8])
                    self.assertNotIn("encode_stride", config["eval"])
                    self.assertNotIn("steps_per_epoch", config["train"])
                    self.assertEqual(config["stats"], base["stats"])
                    self.assertEqual(config["model"], base["model"])
                    self.assertEqual(config["train"]["seed"],
                                     base["train"]["seed"] if not train_seed else train_seed)
                    # 条目、轮数与优化超参逐项沿用基准配置
                    self.assertEqual(config["train"]["entries"], base["train"]["entries"])
                    self.assertEqual(config["train"]["val_entries"],
                                     base["train"]["val_entries"])
                    for key in ("epochs", "learning_rate", "weight_decay", "batch_queries",
                                "val_queries", "lambda_hull", "device", "dataset_root"):
                        self.assertEqual(config["train"][key], base["train"][key], key)

    def test_stats_follow_the_split(self):
        for model in MODELS:
            config = load(configs.output_path(model))
            self.assertEqual(config["stats"],
                             f"dataset/shipbench/stats_{configs.SPLIT}.json")


class MultiresSbatchTest(unittest.TestCase):
    def test_array_covers_nine_models_and_three_seeds(self):
        text = SBATCH.read_text()
        self.assertIn("--array=0-26%2", text)
        self.assertIn("--gres=gpu:a6000:1", text)
        self.assertIn("--partition=gpu", text)
        self.assertIn("--chdir=/path/to/cluster-home/Documents/APPSolver-v2", text)
        self.assertIn("--output=logs/slurm/%x-%A_%a.out", text)
        self.assertIn("--job-name=multires-isoflops-multiseed", text)
        models_block = text.split("MODELS=(", 1)[1].split(")", 1)[0]
        self.assertEqual(tuple(models_block.split()), MODELS)
        seeds_block = text.split("TRAIN_SEEDS=(", 1)[1].split(")", 1)[0]
        self.assertEqual(tuple(int(seed) for seed in seeds_block.split()), TRAIN_SEEDS)
        encode_block = text.split("ENCODE_MODELS=(", 1)[1].split(")", 1)[0]
        self.assertEqual(tuple(encode_block.split()), ENCODE_MODELS)
        self.assertIn("--encode-stride 1 --splits val", text)

    def test_every_referenced_config_exists(self):
        for model in MODELS:
            for train_seed in TRAIN_SEEDS:
                suffix = ("" if not train_seed
                          else f"_train_seed{train_seed}")
                path = ROOT / "configs" / model / \
                    f"{model}_multires_{configs.SPLIT}{suffix}_isoflops.yaml"
                self.assertTrue(path.is_file(), path)


if __name__ == "__main__":
    unittest.main()
