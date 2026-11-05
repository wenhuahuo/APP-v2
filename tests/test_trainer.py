"""训练侧的控制流：查询点采样必须覆盖整个格点域；验证间隔与 checkpoint 写入时机。"""

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "train"):
    sys.path.insert(0, str(ROOT / "src" / name))

from trainer import Trainer  # noqa: E402
from losses import appsolver_loss  # noqa: E402


class SampleQueriesTest(unittest.TestCase):
    def test_queries_span_both_axes(self):
        dataset = types.SimpleNamespace(flow_coords=torch.zeros(701, 151, 2))
        index = Trainer._sample_queries(4096, torch.Generator().manual_seed(0), dataset)
        self.assertEqual(tuple(index.shape), (4096, 2))
        self.assertGreaterEqual(int(index[:, 0].max()), 690)
        self.assertGreaterEqual(int(index[:, 1].max()), 140)
        self.assertGreaterEqual(int(index[:, 0].min()), 0)
        self.assertLessEqual(int(index[:, 1].max()), 150)


class SingleDomainLossTest(unittest.TestCase):
    def test_hull_loss_is_disabled_by_default(self):
        prediction = torch.tensor([[1.0], [2.0]])
        target = torch.zeros_like(prediction)
        weight = torch.ones(2)
        losses = appsolver_loss(None, (prediction, target, weight))
        self.assertEqual(float(losses["total"]), 0.0)
        self.assertGreater(float(losses["hull"]), 0.0)

    def test_only_active_domain_contributes_to_total(self):
        prediction = torch.tensor([[1.0], [2.0]])
        target = torch.zeros_like(prediction)
        weight = torch.ones(2)
        flow = appsolver_loss((prediction, target, weight), None, lambda_hull=0.5)
        hull = appsolver_loss(None, (prediction, target, weight), lambda_hull=0.5)
        self.assertEqual(set(flow), {"flow", "total"})
        self.assertEqual(set(hull), {"hull", "total"})
        self.assertAlmostEqual(float(flow["total"]), 2.5)
        self.assertAlmostEqual(float(hull["total"]), 1.25)


class EpochTraversalTest(unittest.TestCase):
    """每个 epoch 遍历全部训练条目各一次，每条的查询密度档位取自 `train_strides`。"""

    ENTRIES = ("a", "b", "c", "d", "e")

    class TraceDataset:
        """记录被取到的条目名的假数据集，只实现 `__len__` / `__getitem__`。"""

        def __init__(self, stride, trace):
            self.stride = stride
            self.trace = trace

        def __len__(self):
            return len(EpochTraversalTest.ENTRIES)

        def __getitem__(self, i):
            self.trace.append((self.stride, EpochTraversalTest.ENTRIES[i]))
            return {}

    def _trainer(self, train_strides):
        """绕过 `__init__`：假数据集 + 单参数模型，只驱动 `train_epoch` 的抽样控制流。"""
        trainer = object.__new__(Trainer)
        parameter = torch.nn.Parameter(torch.zeros(1))
        self.trace = []
        trainer.model = types.SimpleNamespace(train=lambda: None)
        trainer.config = {"train_strides": list(train_strides), "batch_queries": 4,
                          "lambda_hull": 0.0}
        trainer.domains = ("flow",)
        trainer.generator = torch.Generator().manual_seed(0)
        trainer.optimizer = torch.optim.SGD([parameter], lr=0.0)
        trainer.datasets = {stride: self.TraceDataset(stride, self.trace)
                            for stride in train_strides}
        trainer._sample_queries = lambda count, generator, dataset: None
        trainer._forward = lambda sample, index: (
            None, (parameter, torch.zeros(1), torch.ones(1)), None)
        return trainer, parameter

    def test_every_entry_is_visited_exactly_once_per_epoch(self):
        trainer, _ = self._trainer([1])
        trainer.train_epoch(0)
        self.assertEqual(sorted(entry for _, entry in self.trace), sorted(self.ENTRIES))
        # 一个 epoch 一条目一步：没有条目被重复访问，也没有条目被跳过
        self.assertEqual(len(self.trace), len(self.ENTRIES))

    def test_epoch_order_is_a_random_permutation_from_the_seeded_generator(self):
        trainer, _ = self._trainer([1])
        trainer.train_epoch(0)
        first = [entry for _, entry in self.trace]
        self.trace.clear()
        trainer.train_epoch(1)
        second = [entry for _, entry in self.trace]
        self.assertNotEqual(first, second)
        # 同种子重建：两轮的顺序序列可复现
        replay, _ = self._trainer([1])
        replay.train_epoch(0)
        self.assertEqual([entry for _, entry in self.trace], first)

    def test_single_stride_is_used_for_every_entry(self):
        trainer, _ = self._trainer([1])
        trainer.train_epoch(0)
        self.assertEqual({stride for stride, _ in self.trace}, {1})

    def test_strides_are_drawn_from_the_configured_levels(self):
        trainer, _ = self._trainer([1, 2])
        levels = []
        for epoch in range(20):
            self.trace.clear()
            trainer.train_epoch(epoch)
            self.assertEqual(len(self.trace), len(self.ENTRIES))
            levels.extend(stride for stride, _ in self.trace)
            self.assertEqual(sorted(entry for _, entry in self.trace), sorted(self.ENTRIES))
        self.assertEqual(set(levels), {1, 2})


class EvaluateTest(unittest.TestCase):
    def test_validation_only_returns_active_domain(self):
        for domain in ("flow", "hull"):
            with self.subTest(domain=domain):
                trainer = object.__new__(Trainer)
                trainer.model = types.SimpleNamespace(eval=lambda: None)
                trainer.config = {"train_strides": [1], "lambda_hull": 0.5}
                trainer.domains = (domain,)
                trainer.validation = {1: [None]}
                trainer.val_queries = None
                tensors = (torch.ones(1, 1), torch.zeros(1, 1), torch.ones(1))
                trainer._forward = lambda sample, index: (
                    None, tensors if domain == "flow" else None,
                    tensors if domain == "hull" else None)
                metrics = trainer.evaluate()
                self.assertEqual(set(metrics), {domain})
                self.assertEqual(float(metrics[domain]), 1.0)


class RunTest(unittest.TestCase):
    """绕过 `__init__`（不建数据集与模型），只驱动 `run()` 的控制流。"""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output_dir = Path(directory.name)
        self.trained = []
        self.evaluated = []
        self.epoch = None

    def _run(self, epochs, validation_losses):
        """`validation_losses` 按验证次序给出 `flow` 损失，`hull` 固定为 0。"""
        trainer = object.__new__(Trainer)
        trainer.config = {"epochs": epochs, "lambda_hull": 0.5}
        trainer.full_config = {}
        trainer.model = types.SimpleNamespace(state_dict=lambda: {})
        trainer.output_dir = self.output_dir
        losses = iter(validation_losses)

        def train_epoch(epoch):
            self.epoch = epoch
            self.trained.append(epoch)
            return float(epoch)

        def evaluate():
            self.evaluated.append(self.epoch)
            return {"flow": next(losses), "hull": 0.0}

        trainer.train_epoch = train_epoch
        trainer.evaluate = evaluate
        saves = []
        real_save = torch.save

        def save(checkpoint, path):
            saves.append((checkpoint["epoch"], Path(path).name))
            real_save(checkpoint, path)

        quiet = contextlib.redirect_stdout(io.StringIO())
        with mock.patch("torch.save", side_effect=save), quiet:
            history = trainer.run()
        return history, saves

    def test_validation_every_five_epochs_and_on_last(self):
        history, _ = self._run(12, [2.0, 1.0, 0.5])
        # 第 5、10 轮（0 起算的 4、9）按间隔验证，最后一轮 12 即使不在间隔上也验证。
        self.assertEqual(self.trained, list(range(12)))
        self.assertEqual(self.evaluated, [4, 9, 11])
        self.assertEqual([record["epoch"] for record in history], list(range(12)))

    def test_hundred_epochs_end_on_a_validation_round(self):
        _, saves = self._run(100, [float(n) for n in range(20)])
        self.assertEqual(self.evaluated, list(range(4, 100, 5)))
        self.assertEqual(self.evaluated[-1], 99)
        self.assertEqual([epoch for epoch, name in saves if name == "last.pt"],
                         list(range(100)))

    def test_history_has_no_fabricated_validation(self):
        history, _ = self._run(12, [2.0, 1.0, 0.5])
        validated = {4: 2.0, 9: 1.0, 11: 0.5}
        for record in history:
            self.assertEqual(set(record) - {"epoch", "train"},
                             {"validation", "flow", "hull"} if record["epoch"] in validated
                             else set())
            self.assertEqual(record["train"], float(record["epoch"]))
            if record["epoch"] in validated:
                self.assertEqual(record["flow"], validated[record["epoch"]])
                self.assertEqual(record["validation"], validated[record["epoch"]])

    def test_last_saved_every_epoch_and_best_only_when_validation_improves(self):
        # 验证损失在第一次验证最低，之后变差：best.pt 停在 epoch 4，last.pt 走到 epoch 11。
        _, saves = self._run(12, [1.0, 3.0, 2.0])
        self.assertEqual([epoch for epoch, name in saves if name == "last.pt"],
                         list(range(12)))
        self.assertEqual([epoch for epoch, name in saves if name == "best.pt"], [4])
        last = torch.load(self.output_dir / "last.pt", weights_only=False)
        best = torch.load(self.output_dir / "best.pt", weights_only=False)
        self.assertEqual(last["epoch"], 11)
        self.assertEqual(best["epoch"], 4)


if __name__ == "__main__":
    unittest.main()
