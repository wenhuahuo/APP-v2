"""APPSolver-v1 基线：按点数分 patch，patch 特征展平后过 transformer，逐点输出。

v1 的分词判据是 patch 内点数，patch 顺序带有可学习的位置嵌入，输出端把 patch 预测
展平回点。这三处都依赖训练时的点集密度：点数变化时 patch 数、位置嵌入含义与展平
维度都不再对应，因此换查询密度不随之调整。

给出 `input_domains` 时流场与船面点云先用共享的双域嵌入投影，再拼成同一片点云按点数
分 patch（船体几何因此进入流场预测），`conditioning: fourier_add`（主实验口径）/ `film`
在每层 transformer 之前做逐层工况注入（`point_cloud.token_modulation`）；缺省时保持历史
口径的独立编码与工况加性偏置。
"""

import numpy as np
import torch
import torch.nn as nn

from point_cloud import (FLOW_FEATURES, CloudEmbedding, ConditionBias, apply_modulation,
                         flow_cloud, hull_cloud, lattice_index, token_modulation)


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
    """patch 特征展平 → transformer → 展平回点；每层之前可做逐层 FiLM。"""

    def __init__(self, in_features, out_features, patch_size, d_model, nhead, layers,
                 mlp_ratio, max_patches, dropout, conditioning=None, condition_dim=2):
        super().__init__()
        self.patch_size = patch_size
        self.in_features = in_features
        self.out_features = out_features
        self.input = nn.Linear(patch_size * in_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, int(mlp_ratio * d_model), dropout,
                                       batch_first=True) for _ in range(layers)])
        self.modulations = nn.ModuleList([
            token_modulation(conditioning, condition_dim, d_model) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, patch_size * out_features)

    def forward(self, patches, bias, condition=None):
        tokens = self.input(patches) + self.position[:, :patches.shape[1]] + bias
        for block, modulation in zip(self.blocks, self.modulations):
            tokens = apply_modulation(tokens, modulation, condition)
            tokens = block(tokens)
        out = self.output(self.norm(tokens))
        return out.reshape(patches.shape[0], patches.shape[1], self.patch_size,
                           self.out_features)


class APPSolverV1Baseline(nn.Module):
    def __init__(self, patch_size=64, d_model=128, nhead=4, layers=3, mlp_ratio=4,
                 max_patches=2048, dropout=0.0, condition_dim=2, flow_channels=5,
                 hull_channels=1, input_domains=None, output_domains=None, conditioning=None):
        super().__init__()
        self.input_domains = tuple(input_domains) if input_domains else ("flow", "hull")
        self.output_domains = tuple(output_domains) if output_domains else ("flow", "hull")
        self.joint = input_domains is not None
        self.patch_size = patch_size
        self.condition = ConditionBias(condition_dim, d_model) if conditioning is None else None
        if self.joint:
            self.cloud = CloudEmbedding(d_model)
            features = d_model
        else:
            features = len(FLOW_FEATURES)
        if "flow" in self.output_domains:
            self.flow_patches = PatchTransformer(
                features, flow_channels, patch_size, d_model, nhead, layers, mlp_ratio,
                max_patches, dropout, conditioning, condition_dim)
        if self.joint:
            self.hull_head = (nn.Linear(flow_channels, hull_channels)
                              if "hull" in self.output_domains else None)
        elif "hull" in self.output_domains:
            self.hull_patches = PatchTransformer(
                8, hull_channels, patch_size, d_model, nhead, layers, mlp_ratio,
                max_patches, dropout, conditioning, condition_dim)

    def _points(self, features, coords, patches, bias, condition=None):
        points = coords[0].detach().cpu().numpy()
        index, mask = point_patches(points, np.ones(len(points), dtype=bool), self.patch_size)
        index = torch.from_numpy(index).to(features.device)
        mask = torch.from_numpy(mask).to(features.device)
        patch_features = features[0][index].reshape(index.shape[0], -1)
        predicted = patches(patch_features[None], bias, condition)
        values = predicted.new_zeros(len(points), predicted.shape[-1])
        values[index[mask]] = predicted[0][mask]
        return values

    def _joint_points(self, batch):
        """双域点云 → 联合 patch transformer → 逐点流场值（流场点在前）。"""
        tokens, coords, count = self.cloud(batch)
        return self._points(tokens, coords, self.flow_patches, 0.0,
                            batch["condition"]), count

    def forward(self, batch, flow_queries):
        """输入整片点云，输出查询点处的场值；批大小固定为 1。"""
        condition = batch["condition"]
        bias = self.condition(condition) if self.condition is not None else 0.0
        output = {}
        values = count = None
        if self.joint:
            values, count = self._joint_points(batch)
        if "flow" in self.output_domains:
            if self.joint:
                flow = values[:count]
            else:
                features = flow_cloud(batch)
                flow = self._points(features, features[..., :2], self.flow_patches, bias,
                                    condition)
            output["flow"] = flow.index_select(
                0, lattice_index(batch, flow_queries[0]))[None]
        if "hull" in self.output_domains:
            if self.joint:
                output["hull"] = self.hull_head(values[count:])[None]
            else:
                features = hull_cloud(batch)
                output["hull"] = self._points(features, features[..., :2], self.hull_patches,
                                              bias, condition)[None]
        return output
