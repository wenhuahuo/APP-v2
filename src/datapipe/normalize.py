"""带求积权重的逐通道标准化。

流场目标按格点求积权重统计，船面压力按顶点面积权重统计，条件输入按条目等权统计。
统计量只在训练集上拟合并存成 `dataset/stats.json`，验证与测试复用同一份。
目标在 `ShipDataset` 里已无量纲化，这里只做平移缩放。
"""

import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "dataset" / "stats.json"


def _weighted_moments(values, weights, mask):
    """按最后一维逐通道累加加权一阶矩与二阶矩，返回 (权重和, 加权和, 加权平方和)。"""
    w = weights[mask].astype(np.float64)
    x = values[mask].astype(np.float64)
    return w.sum(), (w[:, None] * x).sum(axis=0), (w[:, None] * x ** 2).sum(axis=0)


def _standardize_params(group):
    """标准化参数；常数通道的标准差取 1，标准化后恒为 0。"""
    mean = np.asarray(group["mean"], dtype=np.float64)
    std = np.asarray(group["std"], dtype=np.float64)
    constant = np.asarray(group.get("constant", std == 0.0), dtype=bool)
    return mean, np.where(constant, 1.0, std), constant


class Normalizer:
    """流场目标、船面压力、条件输入各自的逐通道平移缩放。"""

    def __init__(self, flow, hull, condition):
        self.flow_mean, self.flow_std, self.flow_constant = _standardize_params(flow)
        self.hull_mean, self.hull_std, self.hull_constant = _standardize_params(hull)
        self.condition_mean, self.condition_std, self.condition_constant = (
            _standardize_params(condition))
        self.flow_mean = torch.tensor(self.flow_mean, dtype=torch.float32)
        self.flow_std = torch.tensor(self.flow_std, dtype=torch.float32)
        self.hull_mean = torch.tensor(self.hull_mean, dtype=torch.float32)
        self.hull_std = torch.tensor(self.hull_std, dtype=torch.float32)
        self.condition_mean = torch.tensor(self.condition_mean, dtype=torch.float32)
        self.condition_std = torch.tensor(self.condition_std, dtype=torch.float32)

    @classmethod
    def fit(cls, dataset):
        flow_sum = np.zeros(5)
        flow_square = np.zeros(5)
        flow_weight = 0.0
        hull_sum = np.zeros(1)
        hull_square = np.zeros(1)
        hull_weight = 0.0
        conditions = []
        for i in range(len(dataset)):
            sample = dataset[i]
            w, s1, s2 = _weighted_moments(sample["flow_target"].numpy(),
                                          sample["flow_weight"].numpy(),
                                          sample["flow_mask"].numpy())
            flow_weight += w
            flow_sum += s1
            flow_square += s2
            w, s1, s2 = _weighted_moments(sample["hull_pressure"].numpy()[:, None],
                                          sample["hull_weight"].numpy(),
                                          sample["hull_mask"].numpy())
            hull_weight += w
            hull_sum += s1
            hull_square += s2
            conditions.append(sample["condition"].numpy())

        conditions = np.asarray(conditions, dtype=np.float64)
        return cls(
            flow=cls._moments(flow_weight, flow_sum, flow_square),
            hull=cls._moments(hull_weight, hull_sum, hull_square),
            condition={"mean": conditions.mean(axis=0).tolist(),
                       "std": conditions.std(axis=0).tolist()},
        )

    @staticmethod
    def _moments(weight, total, square):
        mean = total / weight
        variance = np.maximum(square / weight - mean ** 2, 0.0)
        return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}

    def flow(self, target):
        return (target - self.flow_mean) / self.flow_std

    def hull(self, pressure):
        return (pressure - self.hull_mean) / self.hull_std

    def condition(self, value):
        return (value - self.condition_mean) / self.condition_std

    def to_dict(self):
        return {
            "flow": {"mean": self.flow_mean.tolist(), "std": self.flow_std.tolist(),
                     "constant": self.flow_constant.tolist()},
            "hull": {"mean": self.hull_mean.tolist(), "std": self.hull_std.tolist(),
                     "constant": self.hull_constant.tolist()},
            "condition": {"mean": self.condition_mean.tolist(),
                          "std": self.condition_std.tolist(),
                          "constant": self.condition_constant.tolist()},
        }

    def save(self, path=DEFAULT_PATH):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        stats = json.loads(Path(path).read_text())
        return cls(stats["flow"], stats["hull"], stats["condition"])
