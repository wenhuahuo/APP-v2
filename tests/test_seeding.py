"""训练种子：`train.seed` 同时固定 Python / NumPy / PyTorch / CUDA 的全局 RNG。

判别点：

- 同一种子重跑得到同一串 Python、NumPy、PyTorch 随机数，不同种子得到不同串；
- `Trainer` 构造时按 `config["seed"]` 重新固定全局 RNG，因此训练/验证查询采样与调用方
  是否预先设过种子无关；
- 训练集条目按确定性下标顺序读取（无 DataLoader、无 shuffle），查询子集的唯一随机源是
  Trainer 里按 `seed` 初始化的 generator。
"""

import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "train"):
    sys.path.insert(0, str(ROOT / "src" / name))

from seeding import set_seed  # noqa: E402
from trainer import Trainer  # noqa: E402


class SetSeedTest(unittest.TestCase):
    def test_reproduces_python_numpy_and_torch_streams(self):
        for seed in (0, 1, 7):
            with self.subTest(seed=seed):
                set_seed(seed)
                first = (random.random(), float(np.random.rand(3).sum()),
                         float(torch.rand(4).sum()))
                set_seed(seed)
                second = (random.random(), float(np.random.rand(3).sum()),
                          float(torch.rand(4).sum()))
                self.assertEqual(first[0], second[0])
                self.assertEqual(first[1], second[1])
                self.assertEqual(first[2], second[2])

    def test_different_seeds_give_different_streams(self):
        set_seed(0)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        set_seed(1)
        second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertNotEqual(first, second)

    def test_returns_the_integer_seed(self):
        self.assertEqual(set_seed("0"), 0)
        self.assertIsInstance(set_seed("0"), int)

    @unittest.skipUnless(torch.cuda.is_available(), "没有可用的 CUDA 设备")
    def test_cuda_generators_are_seeded(self):
        for seed in (0, 5):
            with self.subTest(seed=seed):
                set_seed(seed)
                first = torch.rand(4, device="cuda")
                set_seed(seed)
                second = torch.rand(4, device="cuda")
                self.assertTrue(torch.equal(first, second))


class _DummyModel(torch.nn.Module):
    """只有可训练参数与流场输出域，够 `Trainer.__init__` 构造优化器与验证查询。"""

    output_domains = ("flow",)

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class TrainerSeedTest(unittest.TestCase):
    def test_trainer_reseeds_the_global_rng_from_config(self):
        dataset = types.SimpleNamespace(flow_coords=torch.zeros(8, 5, 2))
        train = {"device": "cpu", "seed": 3, "train_strides": [1],
                 "dataset_root": "unused", "entries": ["a"], "val_entries": ["b"],
                 "val_queries": 4, "learning_rate": 1e-3, "weight_decay": 1e-4}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("trainer.ShipDataset", return_value=dataset):
            trainer = Trainer(_DummyModel(), None, {"train": train}, Path(directory))
            # 构造后再取随机数：序列必须与刚设置过 seed=3 的状态一致
            after = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        set_seed(3)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(trainer.seed, 3)
        self.assertEqual(after, expected)

    def test_query_sampling_follows_the_config_seed(self):
        dataset = types.SimpleNamespace(flow_coords=torch.zeros(8, 5, 2))
        train = {"device": "cpu", "seed": 11, "train_strides": [1],
                 "dataset_root": "unused", "entries": ["a"], "val_entries": ["b"],
                 "val_queries": 4, "learning_rate": 1e-3, "weight_decay": 1e-4,
                 "batch_queries": 6}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("trainer.ShipDataset", return_value=dataset):
            trainer = Trainer(_DummyModel(), None, {"train": train}, Path(directory))
            first = trainer._sample_queries(6, trainer.generator, dataset)
            second = Trainer._sample_queries(
                6, torch.Generator().manual_seed(11), dataset)
        self.assertTrue(torch.equal(first, second))
        # 验证查询点集同样由 config seed 决定
        self.assertTrue(torch.equal(
            trainer.val_queries,
            Trainer._sample_queries(4, torch.Generator().manual_seed(11), dataset)))


if __name__ == "__main__":
    unittest.main()
