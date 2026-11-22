"""泛化实验（留一船型 / 留出设计 Fr）的配置派生与数组作业映射。

判别点：

- 两个留出划分各 5 个模型 × 训练种子 0/1/2：seed 0 复用既有的 `*_<划分>_isoflops.yaml`，
  seed 1/2 由 `make_split_configs.py` 从 seed 0 确定性派生，文件名区分划分与训练种子；
- 派生配置只改 `train.seed` 与头注释：划分条目、`stats`、容量与训练预算逐项沿用 seed 0；
- 划分语义取 `configs/splits_multihull.json` 的既定定义（218/6 与 213/11），统计量指向
  同一划分训练集拟合的 `stats_<划分>.json`，三个种子共用；
- 两个数组作业各 15 个元素、`%2`，元素与配置一一对应（每个模型三个训练种子连续排列）;
- 两个划分的配置都只训练与评测 stride 1（`train_strides: [1]`、`eval.strides: [1]`）。
"""

import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import make_split_configs  # noqa: E402

CONFIG_DIR = ROOT / "configs"
SPLITS = ("loto_kvlcc2", "lofo_fr_design")
MODELS = ("appsolver_oformer", "transolver", "appsolver_v2", "gnot", "oformer")
TRAIN_SEEDS = (0, 1, 2)
SBATCH = {"loto_kvlcc2": ROOT / "scripts" / "train" / "train_generalization_leave_hull.sbatch",
          "lofo_fr_design": ROOT / "scripts" / "train" / "train_generalization_leave_fr.sbatch"}
JOB_NAMES = {"loto_kvlcc2": "generalization-leave-hull",
             "lofo_fr_design": "generalization-leave-fr"}


def config_path(model, split, train_seed):
    """`seed 0` 复用既有的划分配置，`seed 1/2` 用生成器派生的多种子配置。"""
    base_path = CONFIG_DIR / model / f"{model}_{split}_isoflops.yaml"
    if not train_seed:
        return base_path
    return base_path.parent / make_split_configs.train_seed_output_name(
        base_path, train_seed, split)


def splits_definition():
    return {key: value for key, value
            in json.loads((CONFIG_DIR / "splits_multihull.json").read_text()).items()
            if not key.startswith("_")}


class GeneralizationConfigTest(unittest.TestCase):
    def test_configs_only_change_the_training_seed(self):
        """30 份配置：划分、统计量、容量与训练预算与各自的 seed 0 逐项一致。"""
        for split in SPLITS:
            for model in MODELS:
                base = yaml.safe_load(
                    (CONFIG_DIR / model / f"{model}_{split}_isoflops.yaml").read_text())
                for train_seed in TRAIN_SEEDS:
                    with self.subTest(split=split, model=model, seed=train_seed):
                        path = config_path(model, split, train_seed)
                        config = yaml.safe_load(path.read_text())
                        self.assertEqual(config["train"]["seed"], train_seed)
                        self.assertEqual(config["model_type"], model)
                        self.assertEqual(config["stats"], base["stats"])
                        self.assertEqual(config.get("model"), base.get("model"))
                        for key, value in base["train"].items():
                            if key != "seed":
                                self.assertEqual(config["train"][key], value, key)
                        if train_seed:
                            self.assertIn(f"训练种子 {train_seed}",
                                          path.read_text().splitlines()[0])

    def test_configs_match_the_generator(self):
        """派生幂等：seed 1/2 配置由 seed 0 配置确定生成，文件名标注划分与训练种子。"""
        for split in SPLITS:
            for model in MODELS:
                base_path = CONFIG_DIR / model / f"{model}_{split}_isoflops.yaml"
                for train_seed in (1, 2):
                    with self.subTest(split=split, model=model, seed=train_seed):
                        path = config_path(model, split, train_seed)
                        self.assertEqual(
                            path.name,
                            make_split_configs.train_seed_output_name(base_path, train_seed, split))
                        self.assertEqual(
                            path.read_text(),
                            make_split_configs.derive_train_seed(base_path, train_seed, split))

    def test_configs_follow_the_shared_split_definition(self):
        """条目与统计量取既定划分：218/6 留出 KVLCC2、213/11 只留出经典船型设计 Fr。"""
        definitions = splits_definition()
        for split, (n_train, n_val) in zip(SPLITS, ((218, 6), (213, 11))):
            definition = definitions[split]
            self.assertEqual(len(definition["train"]), n_train)
            self.assertEqual(len(definition["val"]), n_val)
            with self.subTest(split=split):
                self.assertTrue(all(entry.startswith("kvlcc2_") for entry in definition["val"])
                                == (split == "loto_kvlcc2"))
                for model in MODELS:
                    config = yaml.safe_load(
                        (CONFIG_DIR / model / f"{model}_{split}_isoflops.yaml").read_text())
                    self.assertEqual(config["train"]["entries"], definition["train"])
                    self.assertEqual(config["train"]["val_entries"], definition["val"])
                    self.assertEqual(config["stats"],
                                     f"dataset/shipbench/stats_{split}.json")

    def test_generalization_runs_single_resolution(self):
        """泛化实验沿用主实验协议：训练与评测都只跑 stride 1，没有 steps_per_epoch。"""
        for split in SPLITS:
            for model in MODELS:
                for train_seed in TRAIN_SEEDS:
                    with self.subTest(split=split, model=model, seed=train_seed):
                        config = yaml.safe_load(config_path(model, split, train_seed).read_text())
                        self.assertEqual(config["train"]["train_strides"], [1])
                        self.assertEqual(config["eval"]["strides"], [1])
                        self.assertNotIn("steps_per_epoch", config["train"])

    def test_sbatch_arrays_cover_exactly_the_runs(self):
        """两个数组各 15 个元素、%2，元素按「模型 × 训练种子」排列且与配置一一对应。"""
        if not all(job.is_file() for job in SBATCH.values()):
            self.skipTest("Local Slurm job scripts are absent")
        for split in SPLITS:
            text = SBATCH[split].read_text()
            with self.subTest(split=split):
                self.assertIn("--array=0-14%2", text)
                self.assertIn(f"--job-name={JOB_NAMES[split]}", text)
                block = text.split("CONFIGS=(", 1)[1].split(")", 1)[0]
                configs = tuple(block.split())
                expected = tuple(f"configs/{model}/{config_path(model, split, seed).name}"
                                 for model in MODELS for seed in TRAIN_SEEDS)
                self.assertEqual(configs, expected)
                for path in configs:
                    self.assertTrue((ROOT / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
