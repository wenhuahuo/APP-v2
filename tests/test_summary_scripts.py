"""汇总脚本（`src/eval/summary.py` 与三份 summarize_*.py）的口径与产物 schema。

判别点：

- `resolve_run` 只回退到 `output/archive/stale_after_collision/`，不搜索其它归档子目录；
- 协议校验拒绝 `train_strides != [1]`、缺 stride 1 或 checkpoint 不是 `best.pt` 的 run；
- 整体 RMSE / R² / η RMSE 与 `src/eval/aggregate.py` 的池化公式一致，三种子统计为
  `ddof=1` 的样本均值与样本标准差；
- 三份脚本的命令行能端到端跑通，写出的 json 记录 run 目录、`checkpoint_epoch`、逐种子
  值与均值/标准差。
"""

import json
import re
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("eval",):
    sys.path.insert(0, str(ROOT / "src" / name))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

import aggregate  # noqa: E402
import summary as runs  # noqa: E402
import summarize_app_migration as migration  # noqa: E402
import summarize_app_oformer_ablations as ablations  # noqa: E402
import summarize_generalization as generalization  # noqa: E402

SEEDS = (0, 1, 2)
# 小划分定义：脚本只校验留出条目数与留出集，训练条目数只影响训练 token 总量
SPLITS = {
    "_comment": "测试用划分定义",
    "random_seed0": {"protocol": "random_split", "holdout": "RandomState(0) 抽 15%",
                     "counts": {"train": 3, "val": 2},
                     "train": ["t0", "t1", "t2"], "val": ["v0", "v1"]},
    "loto_kvlcc2": {"protocol": "leave_one_hull_out", "holdout": "kvlcc2",
                    "counts": {"train": 4, "val": 1},
                    "train": ["t0", "t1", "t2", "t3"], "val": ["v0"]},
    "lofo_fr_design": {"protocol": "leave_one_fr_scale_out", "holdout": "fr_scale=1.0",
                       "counts": {"train": 4, "val": 3},
                       "train": ["t0", "t1", "t2", "t3"], "val": ["v0", "v1", "v2"]},
}
JOINT = ablations.run_name("appsolver_oformer_joint_flow_hull", 0)


def channel_values(seed, entry_index):
    """逐条目、逐通道的 RMSE 与 R²：互不相同，池化结果才可核对。"""
    rmse = [0.10 * (entry_index + 1) + 0.01 * channel + 0.001 * seed for channel in range(5)]
    r2 = [0.5 - 0.05 * entry_index] * 5
    return rmse, r2


def metrics_text(run, seed, entries, with_hull=False, train_strides=(1,), parameters=None,
                 checkpoint="best.pt", strides=("1",)):
    records = {}
    for index, entry in enumerate(entries):
        rmse, r2 = channel_values(seed, index)
        record = {"flow": {"channels": ["eta", "u", "v", "w", "p_rgh"], "units": ["-"] * 5,
                           "rmse": rmse, "r2": r2,
                           "n_points": 100 + 10 * index}}
        if with_hull:
            record["hull"] = {"channels": ["cp"], "units": ["-"], "rmse": [0.02 + 0.001 * seed],
                              "r2": [0.8], "n_points": 40}
        records[entry] = record
    payload = {
        "config": f"configs/x/{run}.yaml",
        "model_type": "appsolver_oformer",
        "parameters": 1_000 + seed if parameters is None else parameters,
        "checkpoint": checkpoint,
        "checkpoint_epoch": 10 + seed,
        "train_strides": list(train_strides),
        "encode_stride": None,
        "tasks": {"train": ["t0"], "val": list(entries)},
        "strides": {"train": {"1": []},
                    "val": {stride: (records if stride == strides[0] else {})
                            for stride in strides}},
    }
    return json.dumps(payload, ensure_ascii=False)


class Fixture:
    """在临时目录里造一份最小可跑的仓库视图：划分、标定表与 output/<run>。"""

    def __init__(self, root: Path):
        self.root = root
        (root / "configs").mkdir(parents=True)
        (root / "scripts" / "eval").mkdir(parents=True)
        (root / "configs" / "splits_multihull.json").write_text(
            json.dumps(SPLITS, ensure_ascii=False))
        self.entries = {}
        self.complexity = {"entries": {}}

    def add(self, run, split, seed, with_hull=False, gflops=100.0, tokens=10,
            parameters=None, subdir="output", **kwargs):
        directory = self.root / subdir / run
        directory.mkdir(parents=True, exist_ok=True)
        entries = SPLITS[split]["val"]
        (directory / "metrics_best.json").write_text(
            metrics_text(run, seed, entries, with_hull=with_hull, parameters=parameters,
                         **kwargs))
        # 标定表按 random_seed0 的 run 名登记，三个划分与训练种子共用同一条目
        base = re.sub(r"_train_seed\d+", "", run)
        for split in ("loto_kvlcc2", "lofo_fr_design"):
            base = base.replace(split, "random_seed0")
        if base not in self.complexity["entries"]:
            self.complexity["entries"][base] = {"tokens": tokens, "token_kind": "patch",
                                                "gflops": gflops}
        self.entries[run] = directory

    def write_complexity(self):
        (self.root / "scripts" / "eval" / "model_complexity.json").write_text(
            json.dumps(self.complexity, ensure_ascii=False))


def build_fixture(root):
    fixture = Fixture(root)
    for model in ("appsolver_v2", "appsolver_oformer", "oformer"):
        for seed in SEEDS:
            for split in ("random_seed0", "loto_kvlcc2", "lofo_fr_design"):
                run = generalization.run_name(model, split, seed)
                fixture.add(run, split, seed)
    for native, app, _, _ in migration.PAIRS:
        for seed in SEEDS:
            fixture.add(migration.native_run(native, seed), "random_seed0", seed)
            fixture.add(migration.app_run(app, seed), "random_seed0", seed, gflops=None)
    for kind, variants in ablations.VARIANTS.items():
        for _, prefix, _, _ in variants:
            for seed in SEEDS:
                run = ablations.run_name(prefix, seed)
                fixture.add(run, "random_seed0", seed, with_hull="joint_flow_hull" in run,
                            parameters=1_000)
    return fixture


class RunResolutionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prefers_output_then_the_collision_archive(self):
        live = self.root / "output" / "r"
        live.mkdir(parents=True)
        self.assertEqual(runs.resolve_run(self.root, "r"), live)
        archived = self.root / "output" / "archive" / "stale_after_collision" / "r"
        archived.mkdir(parents=True)
        self.assertEqual(runs.resolve_run(self.root, "r"), live)
        other = self.root / "output" / "archive" / "pre_sampling_fix" / "r2"
        other.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            runs.resolve_run(self.root, "r2")

    def test_protocol_check_rejects_strided_training(self):
        fixture = Fixture(self.root)
        fixture.add("run", "random_seed0", 0, train_strides=(1, 2))
        fixture.write_complexity()
        with self.assertRaises(RuntimeError):
            runs.run_metrics(self.root, "run")
        fixture.add("run", "random_seed0", 0, checkpoint="last.pt")
        with self.assertRaises(RuntimeError):
            runs.run_metrics(self.root, "run")
        fixture.add("run", "random_seed0", 0, strides=("2",))
        with self.assertRaises(RuntimeError):
            runs.run_metrics(self.root, "run")


class AggregationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fixture = build_fixture(self.root)
        self.fixture.write_complexity()

    def tearDown(self):
        self.tmp.cleanup()

    def test_seed_metrics_match_the_pooling_formula(self):
        run = generalization.run_name("appsolver_oformer", "random_seed0", 0)
        row = runs.run_metrics(self.root, run)
        records = json.loads((self.root / "output" / run / "metrics_best.json")
                             .read_text())["strides"]["val"]["1"]
        expected = aggregate.overall(records)
        self.assertAlmostEqual(row["overall_rmse"], expected["overall_rmse"])
        self.assertAlmostEqual(row["overall_r2"], 1.0 - (expected["overall_rmse"] / 100) ** 2)
        self.assertAlmostEqual(row["eta_rmse_mm"], aggregate.eta_rmse_mm(records))
        self.assertEqual(row["n_points"], expected["n_points"])
        self.assertEqual(row["n_entries"], 2)
        self.assertEqual(row["checkpoint_epoch"], 10)
        self.assertEqual(row["run_dir"], f"output/{run}")

    def test_mean_std_uses_sample_statistics(self):
        values = [0.5, 0.6, 0.7]
        row = runs.mean_std(values)
        self.assertAlmostEqual(row["mean"], 0.6)
        self.assertAlmostEqual(row["std"], statistics.stdev(values))
        self.assertAlmostEqual(row["variance"], statistics.variance(values))
        self.assertEqual(runs.mean_std([0.4])["std"], 0.0)

    def test_training_tokens_follow_the_step_count(self):
        self.assertEqual(runs.training_tokens(625, 190), 625 * 19_000)
        row = runs.model_row(
            self.root, "X", [generalization.run_name("appsolver_v2", "random_seed0", seed)
                             for seed in SEEDS],
            runs.load_complexity(self.root), SPLITS["random_seed0"]["counts"]["train"])
        self.assertEqual(row["training_tokens"], 10 * SPLITS["random_seed0"]["counts"]["train"]
                         * runs.TRAIN_STEPS_PER_ENTRY)

    def test_ranks_order_by_overall_r2_per_seed(self):
        labels = ["appsolver_oformer", "appsolver_v2"]
        rows = runs.model_rows(
            self.root, labels,
            lambda label: [generalization.run_name(label, "random_seed0", seed)
                           for seed in SEEDS],
            runs.load_complexity(self.root), SPLITS["random_seed0"]["counts"]["train"])
        places = runs.ranks(rows)
        for index in range(len(SEEDS)):
            best = max(rows, key=lambda row: row["seeds"][index]["overall_r2"])
            self.assertEqual(places[best["label"]]["per_seed"][index], 1)
        self.assertEqual(set(places), set(labels))

    def test_missing_calibration_raises(self):
        with self.assertRaises(KeyError):
            runs.complexity_of({"entries": {}}, "unknown_run")

    def test_calibration_is_shared_across_splits_and_seeds(self):
        table = runs.load_complexity(self.root)
        base = runs.complexity_of(table, "appsolver_oformer_random_seed0_isoflops")
        for run in ("appsolver_oformer_loto_kvlcc2_isoflops",
                    "appsolver_oformer_lofo_fr_design_train_seed2_isoflops",
                    "appsolver_oformer_random_seed0_train_seed1_isoflops"):
            with self.subTest(run=run):
                self.assertEqual(runs.complexity_of(table, run), base)

    def test_hull_channel_summary_only_for_joint_output(self):
        joint = runs.run_metrics(self.root, JOINT)
        self.assertIn("hull", joint)
        self.assertGreater(joint["hull"]["cp_r2"], 0.0)
        plain = runs.run_metrics(
            self.root, ablations.run_name("appsolver_oformer", 0))
        self.assertNotIn("hull", plain)


class CommandLineTest(unittest.TestCase):
    """三份脚本的端到端运行：命令行走通并写出要求的 schema。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture = build_fixture(self.root)
        fixture.write_complexity()

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, script, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "eval" / script),
             "--root", str(self.root), *args],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def load(self, name):
        return json.loads((self.root / name).read_text())

    def test_main_experiment_summary(self):
        self.run_script("summarize_generalization.py", "--task", "main",
                        "--models", "appsolver_v2", "appsolver_oformer",
                        "--json", "main.json")
        payload = self.load("main.json")
        self.assertEqual(payload["experiment"], "main")
        self.assertEqual(payload["run_count"], 6)
        self.assertEqual({row["label"] for row in payload["task"]["models"]},
                         {"APP+OFormer", "APPSolver-v2"})
        for row in payload["task"]["models"]:
            self.assertEqual(len(row["seeds"]), 3)
            for seed in row["seeds"]:
                self.assertTrue(seed["run_dir"].startswith("output/"))
                self.assertIn("checkpoint_epoch", seed)
            mean = statistics.fmean([seed["overall_r2"] for seed in row["seeds"]])
            self.assertAlmostEqual(row["overall_r2"]["mean"], mean)
            self.assertEqual(len(row["rank"]["per_seed"]), 3)

    def test_three_split_summary(self):
        self.run_script("summarize_generalization.py", "--task", "splits",
                        "--models", "appsolver_oformer", "oformer",
                        "--json", "splits.json")
        payload = self.load("splits.json")
        self.assertEqual(set(payload["tasks"]),
                         {"random_seed0", "loto_kvlcc2", "lofo_fr_design"})
        self.assertEqual(payload["run_count"], 2 * 3 * 3)
        self.assertEqual(payload["tasks"]["loto_kvlcc2"]["val_entries"], 1)
        self.assertEqual(set(payload["cross_split"]), {"APP+OFormer", "OFormer"})
        self.assertIn("kvlcc2", payload["tasks"]["loto_kvlcc2"]["holdout_semantics"])

    def test_app_migration_summary(self):
        self.run_script("summarize_app_migration.py", "--json", "mig.json")
        payload = self.load("mig.json")
        self.assertEqual([row["label"] for row in payload["models"]],
                         ["Transolver", "APP+Transolver", "UPT", "APP+UPT"])
        for row in payload["migration"]:
            self.assertAlmostEqual(row["token_compression"], 1.0)
        app = [row for row in payload["models"] if row["label"].startswith("APP")]
        for row in app:
            self.assertIsNone(row["gflops"])
            self.assertIn("gflops_status", row)
        native = [row for row in payload["models"] if not row["label"].startswith("APP")]
        for row in native:
            self.assertEqual(row["gflops"], 100.0)

    def test_ablation_summaries(self):
        for kind, json_name, count in (("hull_fusion", "hull.json", 3),
                                       ("partition", "partition.json", 3),
                                       ("conditioning", "conditioning.json", 6)):
            with self.subTest(kind=kind):
                self.run_script("summarize_app_oformer_ablations.py", "--kind", kind,
                                "--json", json_name)
                payload = self.load(json_name)
                self.assertEqual(payload["ablation"], kind)
                self.assertEqual(len(payload["variants"]), count)
                self.assertEqual(payload["run_count"], count * 3)
                reference = payload["reference_variant"]
                for row in payload["variants"]:
                    self.assertEqual(len(row["seeds"]), 3)
                    if row["variant"] != reference:
                        self.assertIn("contrast", row)
                        self.assertEqual(row["contrast"]["reference"],
                                         {"hull_fusion": "flow_with_hull",
                                          "partition": "adaptive",
                                          "conditioning": "film"}[kind])
        joint = [row for row in self.load("hull.json")["variants"]
                 if row["variant"] == "joint_flow_hull"][0]
        self.assertIn("hull", joint)
        self.assertAlmostEqual(
            joint["hull"]["cp_rmse"]["mean"],
            statistics.fmean([0.02, 0.021, 0.022]))


if __name__ == "__main__":
    unittest.main()
