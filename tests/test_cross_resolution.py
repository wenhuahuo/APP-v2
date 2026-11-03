"""跨分辨率汇总与单条目前向延迟：聚合公式、评测文件名、计时口径。

判别点：

- 池化公式与主表一致：同一通道内各条目 R² 相同时，整体 R² 就等于该 R²；点数加权
  的池化 RMSE 落在两个条目之间、靠近点数多的那个；
- 固定输入编码的评测产物文件名带 `_encode<stride>`，不会覆盖同步抽样的
  `metrics_best.json`；
- 计时只跑 warmup + repeat 次前向、样本数等于 repeat、输入张量不被改动、前向在
  `no_grad` 下执行。
"""

import math
import statistics
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from aggregate import eta_rmse_mm, overall, pooled_channel  # noqa: E402
from latency import fixed_inputs, summary, time_forward  # noqa: E402
from report import metrics_filename  # noqa: E402
import summarize_cross_resolution as summarize  # noqa: E402

DATASET = ROOT / "dataset" / "shipbench"
ENTRY = "kcs_fr026_normal_mesh"


def record(rmse, r2, n_points):
    return {"flow": {"rmse": list(rmse), "r2": list(r2), "n_points": n_points}}


class AggregateTest(unittest.TestCase):
    def test_identical_r2_per_channel_reproduces_that_r2(self):
        """池化方差由逐条目 RMSE 与 R² 还原：同 R² 时整体 R² 等于该 R²。"""
        records = {"a": record([1.0] * 5, [0.75] * 5, 100),
                   "b": record([2.0] * 5, [0.75] * 5, 300)}
        result = overall(records)
        self.assertAlmostEqual(result["overall_r2"], 0.75, places=12)
        self.assertAlmostEqual(result["overall_rmse"], 50.0, places=12)
        self.assertEqual(result["n_points"], 400)

    def test_pooled_rmse_is_weighted_towards_the_larger_entry(self):
        records = {"a": record([1.0] * 5, [0.5] * 5, 100),
                   "b": record([2.0] * 5, [0.5] * 5, 300)}
        rmse, sigma, total = pooled_channel(records, 0)
        self.assertAlmostEqual(rmse, math.sqrt((100 * 1.0 + 300 * 4.0) / 400), places=12)
        # 同 R² 下池化方差 = rmse²/(1−R²)，因此 NRMSE 恒为 sqrt(0.5)。
        self.assertAlmostEqual(rmse / sigma, math.sqrt(0.5), places=12)
        self.assertEqual(total, 400)
        self.assertAlmostEqual(overall(records)["overall_r2"], 0.5, places=12)

    def test_overall_r2_equals_mean_of_channel_r2(self):
        """整体 R² = mean_c (1 − NRMSE(c)²)；等方差时等于逐通道 R² 的均值。"""
        records = {"a": record([1.0, 2.0, 3.0, 4.0, 5.0], [0.9, 0.5, 0.1, 0.0, 0.8], 10)}
        result = overall(records)
        self.assertAlmostEqual(result["overall_r2"],
                               statistics.fmean([0.9, 0.5, 0.1, 0.0, 0.8]), places=12)

    def test_eta_rmse_mm_uses_the_eta_channel(self):
        records = {"a": record([0.005, 1.0, 1.0, 1.0, 1.0], [0.5] * 5, 10)}
        self.assertAlmostEqual(eta_rmse_mm(records), 5.0, places=12)

    def test_empty_records_is_an_error(self):
        with self.assertRaises(ValueError):
            overall({})


class MetricsFilenameTest(unittest.TestCase):
    def test_sync_and_fixed_encode_results_never_share_a_file(self):
        self.assertEqual(metrics_filename("best.pt"), "metrics_best.json")
        self.assertEqual(metrics_filename("last.pt"), "metrics_last.json")
        self.assertEqual(metrics_filename("best.pt", 1), "metrics_best_encode1.json")
        self.assertEqual(metrics_filename("best.pt", 2), "metrics_best_encode2.json")
        self.assertNotEqual(metrics_filename("best.pt", 1), metrics_filename("best.pt"))


class CountingModel(torch.nn.Module):
    """记录前向次数、`no_grad` 状态，并返回与查询坐标相关的输出。"""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = 0
        self.grad_enabled = []

    def forward(self, batch, queries):
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        return {"flow": queries[..., :1] * self.weight}


class LatencyTest(unittest.TestCase):
    def setUp(self):
        self.model = CountingModel()
        self.batch = {"flow_coords": torch.zeros(1, 4, 3, 2)}
        self.queries = torch.zeros(1, 6, 2)
        self.batch_before = {key: value.clone() for key, value in self.batch.items()}
        self.queries_before = self.queries.clone()

    def test_warmup_is_not_counted_and_repeat_sets_the_sample_count(self):
        samples = time_forward(self.model, self.batch, self.queries, warmup=3, repeat=7)
        self.assertEqual(self.model.calls, 10)
        self.assertEqual(len(samples), 7)
        self.assertTrue(all(value >= 0.0 for value in samples))

    def test_forward_runs_without_grad_and_keeps_inputs(self):
        time_forward(self.model, self.batch, self.queries, warmup=1, repeat=2)
        self.assertFalse(any(self.model.grad_enabled))
        self.assertIsNone(self.model.weight.grad)
        torch.testing.assert_close(self.queries, self.queries_before)
        for key, value in self.batch.items():
            torch.testing.assert_close(value, self.batch_before[key])

    def test_summary_reports_mean_and_sample_std(self):
        result = summary([0.1, 0.3])
        self.assertAlmostEqual(result["mean"], 0.2, places=12)
        self.assertAlmostEqual(result["std"], statistics.stdev([0.1, 0.3]), places=12)
        self.assertAlmostEqual(result["median"], 0.2, places=12)
        self.assertAlmostEqual(result["min"], 0.1, places=12)
        self.assertEqual(summary([0.5])["std"], 0.0)

    def test_invalid_repeat_or_warmup_is_rejected(self):
        for warmup, repeat in ((-1, 2), (1, 0)):
            with self.subTest(warmup=warmup, repeat=repeat):
                with self.assertRaises(ValueError):
                    time_forward(self.model, self.batch, self.queries, warmup, repeat)
        with self.assertRaises(ValueError):
            summary([])


@unittest.skipUnless((DATASET / ENTRY / "meta.json").exists(), "数据集条目不存在")
class FixedInputsTest(unittest.TestCase):
    def test_inputs_are_float32_and_queries_cover_the_strided_lattice(self):
        batch, queries = fixed_inputs(DATASET, ENTRY, 2, torch.device("cpu"))
        for key, value in batch.items():
            if value.is_floating_point():
                self.assertEqual(value.dtype, torch.float32, key)
        nx, ny = batch["flow_coords"].shape[1:3]
        self.assertEqual(tuple(queries.shape), (1, nx * ny, 2))
        # stride 2 的格点数是全密度的四分之一上下（行、列各隔一个）。
        _, fine = fixed_inputs(DATASET, ENTRY, 1, torch.device("cpu"))
        self.assertGreater(fine.shape[1], queries.shape[1])


class SummarizeFormatTest(unittest.TestCase):
    def test_rank_sorts_by_stride1_r2_and_table_is_markdown(self):
        rows = [{"model": "a", "r2": {1: 0.1}}, {"model": "b", "r2": {1: 0.4}}]
        self.assertEqual([row["model"] for row in summarize.rank(rows)], ["b", "a"])
        rendered = summarize.table([{"model": "b", "r2": {1: 0.4}}], ("模型", "stride 1"),
                                   lambda row: [row["model"], f"{row['r2'][1]:+.4f}"])
        self.assertEqual(rendered.splitlines()[0], "| 模型 | stride 1 |")
        self.assertEqual(rendered.splitlines()[1], "| --- | --- |")
        self.assertEqual(rendered.splitlines()[2], "| b | +0.4000 |")

    def test_missing_artifact_names_the_path(self):
        with self.assertRaises(FileNotFoundError) as context:
            summarize.read_json(ROOT / "output" / "definitely-missing" / "metrics_best.json")
        self.assertIn("metrics_best.json", str(context.exception))


if __name__ == "__main__":
    unittest.main()
