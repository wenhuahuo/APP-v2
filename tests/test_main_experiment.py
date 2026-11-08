"""主实验公平配置：7 个模型在同一任务口径下的配置与模型行为。

主实验范围是 `random_seed0` 随机划分：7 个模型 × 2 种容量口径（固定参数量
`_multihull_isoflops`）= 14 份配置。留一船型与留一
设计 Fr 的配置由同一生成器派生、保留给后续泛化实验，本轮主实验不训练它们。

判别点：

- 主表 14 份配置的 `stats`、训练/验证条目和训练超参完全一致，同一容量口径下只有模型
  自身结构不同（例外：APPSolver-v1 的固定计算量配置用 lr 1e-4，因为 lr=1e-3 会塌缩为
  常数场）；
- 派生配置由 `make_split_configs.py` 按划分定义确定性生成：只替换 `stats`、
  `train.entries`、`train.val_entries` 与头注释，容量超参与基础配置逐项相同（三个划分
  × 两种容量共 54 份都走同一条生成路径，含保留的泛化实验配置）；
- 全部配置共用单一种子 `seed=0`；多种子复跑从 `random_seed0` 派生 28 份
  `<模型>_random_seed0_train_seed{1,2}[_isoflops].yaml`，只改 `train.seed` 与头注释，
  数据划分、统计量、容量与训练预算逐项不变（文件名同时标注划分与训练种子）；
- 训练数组脚本本轮只补训练种子 1、2：`TRAIN_SEEDS=(1 2)`、`--array=0-13%2`，数组按
  先种子、后模型展开；种子 0 的 7 个 run 已在上一轮完成，不重跑；
- 主表配置都真实生效：模型同时输出流场与船面 Cp（双域输出头都构造）、船体几何与
  工况都改变流场预测、损失同时含流场项与 `lambda_hull=1.0` 的船面项；
- 历史 KCS 配置不被默认值静默改写：仍保留双域输出与 `lambda_hull=1.0`。
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
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from build import build_model  # noqa: E402
from collate import collate_samples  # noqa: E402
from conditioning import FourierAdd  # noqa: E402
from losses import appsolver_loss  # noqa: E402
from normalize import Normalizer  # noqa: E402
from trainer import Trainer  # noqa: E402
import make_split_configs  # noqa: E402

CONFIG_DIR = ROOT / "configs"
# 主实验的 7 个模型；GINO 与 CORAL 已因训练成本与数值塌缩移出实验计划（实现与
# `configs/{gino,coral}/*` 保留在仓库里，但不再训练与报告）
MODELS = ("appsolver_v2", "appsolver_oformer", "appsolver_v1", "gnot", "upt",
          "transolver", "oformer")
# 主实验只跑随机划分；留一船型和留一设计 Fr 的配置保留给后续泛化实验
MAIN_SPLIT = "random_seed0"
RETAINED_SPLITS = ("loto_kvlcc2", "lofo_fr_design")
SPLITS = (MAIN_SPLIT,) + RETAINED_SPLITS
# 容量口径：目录里固定的基础配置后缀 -> 该口径派生配置的附加后缀
CAPACITIES = (("_multihull_normal", ""), ("_multihull_isoflops", "_isoflops"))
SBATCH = ROOT / "scripts" / "train" / "train_main_experiment.sbatch"
SHARED_TRAIN_KEYS = ("epochs", "weight_decay", "batch_queries",
                     "val_queries", "train_strides", "seed", "lambda_hull", "device",
                     "dataset_root")
# 学习率降到 1e-4 的模型（固定计算量容量下 lr=1e-3 会塌缩为常数场）：CORAL 已移出实验
# 计划，其配置保留在仓库里，因此仍列出
LOW_LR_MODELS = ("coral", "appsolver_v1")
# 正式实验的训练协议：每个 epoch 遍历全部训练条目，默认只训练与评测 stride 1。
DEFAULT_TRAIN_STRIDES = [1]
DEFAULT_EVAL_STRIDES = [1]
SPLIT_KEYS = ("entries", "val_entries")
NX, NY = 17, 9
HULL_POINTS = 40


def load(model, split, capacity_suffix=""):
    return yaml.safe_load((CONFIG_DIR / model /
                           f"{model}_{split}{capacity_suffix}.yaml").read_text())


# 多种子复跑的训练种子（数据划分固定为 MAIN_SPLIT）
TRAIN_SEEDS = (1, 2)


def seed0_path(model, capacity_suffix=""):
    return CONFIG_DIR / model / f"{model}_{MAIN_SPLIT}{capacity_suffix}.yaml"


def multiseed_path(model, train_seed, capacity_suffix=""):
    return (CONFIG_DIR / model /
            f"{model}_{MAIN_SPLIT}_train_seed{train_seed}{capacity_suffix}.yaml")


def splits_definition():
    return {key: value for key, value
            in json.loads((CONFIG_DIR / "splits_multihull.json").read_text()).items()
            if not key.startswith("_")}


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
    def test_main_table_is_seven_models_in_two_capacities(self):
        """主表恰好 14 份配置：7 模型 × 固定参数量 / 固定计算量，划分都是 random_seed0。"""
        for _, capacity_suffix in CAPACITIES:
            for model in MODELS:
                with self.subTest(model=model, capacity=capacity_suffix):
                    config = load(model, MAIN_SPLIT, capacity_suffix)
                    self.assertEqual(config["model_type"], model)
                    self.assertEqual(config["stats"],
                                     f"dataset/shipbench/stats_{MAIN_SPLIT}.json")

    def test_retained_split_configs_still_exist(self):
        """留一船型 / 留一设计 Fr 的配置保留给后续泛化实验，不参与本轮主实验。"""
        for _, capacity_suffix in CAPACITIES:
            for split in RETAINED_SPLITS:
                for model in MODELS:
                    with self.subTest(split=split, model=model, capacity=capacity_suffix):
                        self.assertEqual(load(model, split, capacity_suffix)["model_type"], model)

    def test_main_table_shares_data_statistics_and_training_budget(self):
        for _, capacity_suffix in CAPACITIES:
            reference = load(MODELS[0], MAIN_SPLIT, capacity_suffix)
            for model in MODELS[1:]:
                with self.subTest(model=model, capacity=capacity_suffix):
                    config = load(model, MAIN_SPLIT, capacity_suffix)
                    self.assertEqual(config["stats"], reference["stats"])
                    self.assertEqual(config["train"]["entries"],
                                     reference["train"]["entries"])
                    self.assertEqual(config["train"]["val_entries"],
                                     reference["train"]["val_entries"])
                    for key in SHARED_TRAIN_KEYS:
                        self.assertEqual(config["train"][key], reference["train"][key], key)

    def test_low_learning_rate_only_for_appsolver_v1(self):
        """APPSolver-v1 用 lr 1e-4（lr=1e-3 会塌缩为常数场），其余主实验模型仍为 1e-3。"""
        for model in MODELS:
            with self.subTest(model=model):
                config = load(model, MAIN_SPLIT, "_isoflops")
                expected = 1e-4 if model in LOW_LR_MODELS else 1e-3
                self.assertEqual(config["train"]["learning_rate"], expected)

    def test_entries_match_the_shared_split_file(self):
        for split, definition in splits_definition().items():
            self.assertIn(split, SPLITS)
            for _, capacity_suffix in CAPACITIES:
                config = load(MODELS[0], split, capacity_suffix)
                with self.subTest(split=split, capacity=capacity_suffix):
                    self.assertEqual(config["train"]["entries"], definition["train"])
                    self.assertEqual(config["train"]["val_entries"], definition["val"])
                    self.assertEqual(config["stats"],
                                     f"dataset/shipbench/stats_{split}.json")

    def test_single_seed_zero_for_every_config(self):
        for _, capacity_suffix in CAPACITIES:
            for split in SPLITS:
                for model in MODELS:
                    with self.subTest(split=split, model=model, capacity=capacity_suffix):
                        self.assertEqual(load(model, split, capacity_suffix)["train"]["seed"], 0)

    def test_committed_configs_match_the_generator(self):
        """派生幂等：重新生成必须逐字节等于提交的划分派生配置。"""
        for base_suffix, capacity_suffix in CAPACITIES:
            for model in MODELS:
                base = CONFIG_DIR / model / f"{model}{base_suffix}.yaml"
                for split, definition in splits_definition().items():
                    with self.subTest(split=split, model=model, capacity=capacity_suffix):
                        expected = make_split_configs.derive(base, split, definition)
                        committed = (CONFIG_DIR / model /
                                     make_split_configs.output_name(base, split)).read_text()
                        self.assertEqual(committed, expected)

    def test_derived_configs_only_change_split_fields(self):
        """容量与模型超参来自基础配置：派生配置除 stats/条目/头注释外逐项相同。"""
        for base_suffix, capacity_suffix in CAPACITIES:
            for model in MODELS:
                base = yaml.safe_load(
                    (CONFIG_DIR / model / f"{model}{base_suffix}.yaml").read_text())
                for split in SPLITS:
                    derived = load(model, split, capacity_suffix)
                    definition = splits_definition()[split]
                    with self.subTest(split=split, model=model, capacity=capacity_suffix):
                        self.assertEqual(derived["train"]["entries"], definition["train"])
                        self.assertEqual(derived["train"]["val_entries"], definition["val"])
                        for key in ("model", "eval"):
                            self.assertEqual(derived.get(key), base.get(key))
                        for key, value in base["train"].items():
                            if key in SPLIT_KEYS:
                                continue
                            self.assertEqual(derived["train"][key], value, key)

    def test_task_protocol_is_unified(self):
        """主表口径：双域输入、流场 + 船面双域输出、逐层 Fourier add、lambda_hull=1.0。"""
        for _, capacity_suffix in CAPACITIES:
            for model in MODELS:
                with self.subTest(model=model, capacity=capacity_suffix):
                    config = load(model, MAIN_SPLIT, capacity_suffix)
                    self.assertEqual(config["model"]["input_domains"], ["flow", "hull"])
                    self.assertEqual(config["model"]["output_domains"], ["flow", "hull"])
                    self.assertEqual(config["model"]["conditioning"], "fourier_add")
                    self.assertEqual(config["train"]["lambda_hull"], 1.0)

    def test_sbatch_covers_exactly_the_main_table(self):
        """数组任务与主表一一对应：7 个模型 × 训练种子 1/2、固定计算量容量（种子 0 已完成）。"""
        text = SBATCH.read_text()
        self.assertIn("--array=0-13%2", text)
        self.assertIn("--job-name=main-isoflops-seed12", text)
        self.assertIn(f'SPLIT="{MAIN_SPLIT}"', text)
        self.assertNotIn("CAPACITY", text)
        models_block = text.split("MODELS=(", 1)[1].split(")", 1)[0]
        self.assertEqual(tuple(models_block.split()), MODELS)
        seeds_block = text.split("TRAIN_SEEDS=(", 1)[1].split(")", 1)[0]
        # 本轮只提交训练种子 1、2；种子 0 复用上一轮的 run，不重跑
        self.assertEqual(tuple(int(seed) for seed in seeds_block.split()), TRAIN_SEEDS)
        self.assertIn("_isoflops.yaml", text)
        self.assertIn("_train_seed${TRAIN_SEED}_isoflops.yaml", text)

    def test_multiseed_configs_only_change_the_training_seed(self):
        """28 份多种子配置：划分、统计量、容量与训练预算与 seed0 逐项一致，只改 train.seed。"""
        for _, capacity_suffix in CAPACITIES:
            for model in MODELS:
                base = yaml.safe_load(seed0_path(model, capacity_suffix).read_text())
                for train_seed in TRAIN_SEEDS:
                    with self.subTest(model=model, capacity=capacity_suffix, seed=train_seed):
                        path = multiseed_path(model, train_seed, capacity_suffix)
                        config = yaml.safe_load(path.read_text())
                        self.assertEqual(config["train"]["seed"], train_seed)
                        self.assertEqual(config["stats"], base["stats"])
                        self.assertEqual(config.get("model"), base.get("model"))
                        for key, value in base["train"].items():
                            if key != "seed":
                                self.assertEqual(config["train"][key], value, key)
                        self.assertIn(f"训练种子 {train_seed}", path.read_text().splitlines()[0])

    def test_multiseed_configs_match_the_generator(self):
        """派生幂等：多种子配置由 seed0 配置确定生成，文件名区分划分与训练种子。"""
        for _, capacity_suffix in CAPACITIES:
            for model in MODELS:
                base_path = seed0_path(model, capacity_suffix)
                for train_seed in TRAIN_SEEDS:
                    with self.subTest(model=model, capacity=capacity_suffix, seed=train_seed):
                        path = multiseed_path(model, train_seed, capacity_suffix)
                        self.assertEqual(
                            path.name,
                            make_split_configs.train_seed_output_name(base_path, train_seed))
                        self.assertEqual(path.read_text(),
                                         make_split_configs.derive_train_seed(base_path, train_seed))

    def test_main_protocol_is_single_resolution(self):
        """默认实验只训练与评测 stride 1，且不再有 steps_per_epoch（每轮遍历全部条目）。"""
        for model in MODELS:
            with self.subTest(model=model):
                config = load(model, MAIN_SPLIT, "_isoflops")
                self.assertEqual(config["train"]["train_strides"], DEFAULT_TRAIN_STRIDES)
                self.assertEqual(config["eval"]["strides"], DEFAULT_EVAL_STRIDES)
                self.assertNotIn("steps_per_epoch", config["train"])

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


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才能校验 CPU 分词 → GPU 模型的搬运")
class CudaForwardTest(unittest.TestCase):
    """7 个主实验模型在 GPU 上跑通：分词在 CPU 完成，参数与形心必须搬到模型所在设备。"""

    def test_all_main_models_forward_on_cuda(self):
        sample = synthetic_sample()
        batch = {key: value.cuda() for key, value in collate_samples([sample]).items()}
        queries = queries_from(batch)
        for model in MODELS:
            with self.subTest(model=model):
                module = build_model(load(model, MAIN_SPLIT)).cuda().eval()
                with torch.no_grad():
                    output = module(batch, queries)
                self.assertEqual(tuple(output["flow"].shape), (1, 7, 5))
                self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1))
                self.assertEqual(output["flow"].device.type, "cuda")
                del module
                torch.cuda.empty_cache()


class UnifiedForwardTest(unittest.TestCase):
    """用主表配置构造模型，验证任务口径在代码里真实生效（不只是 YAML 字段）。"""

    @classmethod
    def setUpClass(cls):
        cls.configs = {model: load(model, "random_seed0") for model in MODELS}
        cls.models = {model: build_model(config).eval()
                      for model, config in cls.configs.items()}
        cls.sample = synthetic_sample()
        cls.batch = collate_samples([cls.sample])

    def test_both_domains_are_constructed_and_returned(self):
        queries = queries_from(self.batch)
        for model, module in self.models.items():
            with self.subTest(model=model):
                output = module(self.batch, queries)
                self.assertEqual(set(output), {"flow", "hull"})
                self.assertEqual(tuple(output["flow"].shape), (1, 7, 5))
                self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1))
                self.assertEqual(tuple(module.output_domains), ("flow", "hull"))
                # 船面输出头必须真实构造（主实验同时预测流场与表面压力）
                hull_parameters = [name for name, _ in module.named_parameters()
                                   if name.startswith(("hull_head", "hull_patches",
                                                       "hull_extra"))]
                self.assertNotEqual(hull_parameters, [])

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

                # `fourier_add` 的投影零初始化，训练前工况对输出无影响；这里把投影改成
                # 非零权重，模拟训练后的状态，检验逐层加性注入确实接到前向上。
                fourier = [item for item in module.modules() if isinstance(item, FourierAdd)]
                self.assertNotEqual(fourier, [])
                for item in fourier:
                    torch.nn.init.normal_(item.projection.weight, std=0.1)
                baseline = module(self.batch, queries)["flow"]
                conditioned = perturbed(self.batch, "condition",
                                        torch.tensor([[0.32, 9.1]]))
                with torch.no_grad():
                    changed = module(conditioned, queries)["flow"]
                self.assertGreater(float((changed - baseline).abs().max()), 0.0)

    def test_joint_loss_includes_the_hull_term(self):
        for model, module in self.models.items():
            with self.subTest(model=model):
                trainer = object.__new__(Trainer)
                trainer.model = module
                trainer.normalizer = unit_normalizer()
                trainer.device = torch.device("cpu")
                trainer.domains = tuple(module.output_domains)
                trainer.config = {"lambda_hull": 1.0}
                index = torch.stack([torch.arange(7), torch.zeros(7, dtype=torch.long)],
                                    dim=-1)
                _, flow, hull = trainer._forward(self.sample, index)
                self.assertIsNotNone(flow)
                self.assertIsNotNone(hull)
                losses = appsolver_loss(flow, hull, trainer.config["lambda_hull"])
                self.assertAlmostEqual(float(losses["total"]),
                                       float(losses["flow"] + losses["hull"]), places=6)

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

    def test_gradients_reach_every_parameter_of_the_joint_task(self):
        queries = queries_from(self.batch)
        for model, module in self.models.items():
            with self.subTest(model=model):
                module.zero_grad(set_to_none=True)
                output = module(self.batch, queries)
                (output["flow"].sum() + output["hull"].sum()).backward()
                missing = [name for name, parameter in module.named_parameters()
                           if parameter.grad is None]
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
