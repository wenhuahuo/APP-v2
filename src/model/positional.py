"""坐标位置编码：Gaussian Fourier features 与坐标 RoPE，模型里二选一。

两种编码都只依赖物理坐标（patch 形心、查询点坐标），不含任何按序号的参数，
因此对查询点集的密度与排列不敏感。
"""

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """γ(c) = [sin(2πBc), cos(2πBc)]，B 为固定的高斯随机矩阵。"""

    def __init__(self, n_dims, n_features, scale=4.0):
        super().__init__()
        self.register_buffer("matrix", torch.randn(n_dims, n_features) * scale)

    def forward(self, coords):
        projection = 2.0 * math.pi * coords @ self.matrix
        return torch.cat([projection.sin(), projection.cos()], dim=-1)


def apply_rope(features, coords, base=10000.0):
    """按坐标旋转特征（旋转位置编码）。

    features (..., D)，coords (..., C)，D 按坐标轴等分成 C 块，每块内部按
    (2j, 2j+1) 配对做旋转，频率为 base^(-2j/块宽)。
    """
    *lead, dim = features.shape
    coords = coords.expand(*lead, coords.shape[-1])
    n_dims = coords.shape[-1]
    chunk = dim // n_dims
    if dim % n_dims or chunk % 2:
        raise ValueError(f"特征维度 {dim} 无法按坐标维数 {n_dims} 分块做旋转编码")

    frequency = base ** (-torch.arange(0, chunk, 2, dtype=features.dtype,
                                       device=features.device) / chunk)
    angle = (coords[..., None] * frequency).reshape(*lead, dim // 2)
    cosine, sine = angle.cos(), angle.sin()

    paired = features.reshape(*lead, dim // 2, 2)
    even, odd = paired[..., 0], paired[..., 1]
    rotated = torch.stack([even * cosine - odd * sine, even * sine + odd * cosine], dim=-1)
    return rotated.reshape(*lead, dim)
