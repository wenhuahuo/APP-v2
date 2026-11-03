"""单条目前向延迟：batch 1、float32、`eval` + `no_grad`，数据预先驻留设备。

计时只覆盖一次完整前向 `model(batch, queries)`：模型自带的编码、分词与邻域搜索都在
前向内部，因此计入；模型加载、归一化统计量与主机到设备的数据传输都在计时之外。每次
前向前后各做一次 `torch.cuda.synchronize()`，warmup 的耗时不计入统计。

输入由 `ShipDataset` 与 `collate_samples` 构造，与评测（`src/eval/evaluate.py`）同一
条数据路径；查询坐标取该 stride 下的整片自由液面格点，计时期间张量不再改动。
"""

import statistics
import time

import torch

from collate import collate_samples
from ship_dataset import ShipDataset

DTYPE = torch.float32


def synchronize(reference):
    """与 `reference` 同设备的显式同步；CPU 计时不需要。"""
    if reference.device.type == "cuda":
        torch.cuda.synchronize(reference.device)


def fixed_inputs(dataset_root, entry, stride, device):
    """返回 (模型输入批次, 查询坐标 `(1,Q,2)`)；两者都在目标设备上、float32。"""
    sample = ShipDataset(dataset_root, [entry], stride=stride)[0]
    batch = {key: value.to(device) for key, value in collate_samples([sample]).items()}
    for key, value in batch.items():
        if value.is_floating_point() and value.dtype is not DTYPE:
            raise ValueError(f"{key} 不是 float32：{value.dtype}")
    return batch, batch["flow_coords"].reshape(1, -1, 2)


@torch.no_grad()
def time_forward(model, batch, queries, warmup=5, repeat=20):
    """返回每次前向的秒数列表；模型需已 `eval()` 且参数在查询坐标所在设备上。"""
    if warmup < 0:
        raise ValueError(f"warmup 不能为负：{warmup}")
    if repeat < 1:
        raise ValueError(f"repeat 至少为 1：{repeat}")
    model.eval()
    for _ in range(warmup):
        model(batch, queries)
    synchronize(queries)
    samples = []
    for _ in range(repeat):
        synchronize(queries)
        start = time.perf_counter()
        model(batch, queries)
        synchronize(queries)
        samples.append(time.perf_counter() - start)
    return samples


def summary(samples):
    """均值、样本标准差（ddof=1）、最小值与中位数，单位与输入一致。"""
    if not samples:
        raise ValueError("没有计时样本")
    return {
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "median": statistics.median(samples),
    }
