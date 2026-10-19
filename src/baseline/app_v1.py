"""APPSolver-v1 基线：按点数分 patch，patch 特征展平后过 transformer，逐点输出。

v1 的分词判据是 patch 内点数，patch 顺序带有可学习的位置嵌入，输出端把 patch 预测
展平回点。这三处都依赖训练时的点集密度：点数变化时 patch 数、位置嵌入含义与展平
维度都不再对应，因此换查询密度不随之调整。
"""

import numpy as np
import torch
import torch.nn as nn

from point_cloud import ConditionBias, flow_cloud, hull_cloud, lattice_index


def point_patches(coords, valid, patch_size):
    """按点数切分（v1 判据）：沿较长的坐标轴从中位数二分，直到 patch 内点数 ≤ patch_size。

    返回 (P, K) 的点下标与 (P, K) 的有效掩码，K = patch_size；展平维度因此固定。
    """
    coords = np.asarray(coords, dtype=np.float64)
    stack = [np.nonzero(valid)[0]]
    groups = []
    while stack:
        points = stack.pop()
        if len(points) <= patch_size:
            groups.append(points)
            continue
        axis = int(np.argmax(np.ptp(coords[points], axis=0)))
        order = np.argsort(coords[points, axis], kind="stable")
        middle = len(points) // 2
        stack.append(points[order[:middle]])
        stack.append(points[order[middle:]])
    groups = [group for group in groups if len(group)]
    groups.sort(key=lambda group: int(group[0]))

    index = np.zeros((len(groups), patch_size), dtype=np.int64)
    mask = np.zeros((len(groups), patch_size), dtype=bool)
    for patch, group in enumerate(groups):
        index[patch, :len(group)] = group
        mask[patch, :len(group)] = True
    return index, mask


class PatchTransformer(nn.Module):
    """patch 特征展平 → transformer → 展平回点。"""

    def __init__(self, in_features, out_features, patch_size, d_model, nhead, layers,
                 mlp_ratio, max_patches, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.in_features = in_features
        self.out_features = out_features
        self.input = nn.Linear(patch_size * in_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, int(mlp_ratio * d_model), dropout,
                                           batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, patch_size * out_features)

    def forward(self, patches, bias):
        tokens = self.input(patches) + self.position[:, :patches.shape[1]] + bias
        tokens = self.norm(self.blocks(tokens))
        out = self.output(tokens)
        return out.reshape(patches.shape[0], patches.shape[1], self.patch_size,
                           self.out_features)


class APPSolverV1Baseline(nn.Module):
    def __init__(self, patch_size=64, d_model=128, nhead=4, layers=3, mlp_ratio=4,
                 max_patches=2048, dropout=0.0, condition_dim=2, flow_channels=5,
                 hull_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.condition = ConditionBias(condition_dim, d_model)
        self.flow_patches = PatchTransformer(3, flow_channels, patch_size, d_model, nhead,
                                             layers, mlp_ratio, max_patches, dropout)
        self.hull_patches = PatchTransformer(8, hull_channels, patch_size, d_model, nhead,
                                             layers, mlp_ratio, max_patches, dropout)

    def _points(self, features, patches, bias):
        coords = features[..., :2].numpy()[0]
        index, mask = point_patches(coords, np.ones(len(coords), dtype=bool), self.patch_size)
        index = torch.from_numpy(index).to(features.device)
        mask = torch.from_numpy(mask).to(features.device)
        patch_features = features[0][index].reshape(index.shape[0], -1)
        predicted = patches(patch_features[None], bias)
        values = predicted.new_zeros(len(coords), predicted.shape[-1])
        values[index[mask]] = predicted[0][mask]
        return values

    def forward(self, batch, flow_queries):
        """输入整片点云，输出查询点处的场值；批大小固定为 1。"""
        bias = self.condition(batch["condition"])
        flow = self._points(flow_cloud(batch), self.flow_patches, bias)
        flow = flow.index_select(0, lattice_index(batch, flow_queries[0]))[None]
        hull = self._points(hull_cloud(batch), self.hull_patches, bias)[None]
        return {"flow": flow, "hull": hull}
