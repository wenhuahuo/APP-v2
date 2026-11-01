"""APP 分词器：把一片点集按物理尺寸切成 patch，每个 patch 积分成一个 token。

流场域与船体 UV 域共用。`scale` 为参数域单位对应的物理长度：流场域坐标已按 Lpp
归一化，取 1；船体 UV 域被拉伸，取 `None` 由 patch 的 `ΣΔ` 反推局地物理尺度。
分词结果与点集密度无关，只依赖几何与容差。
"""

import numpy as np
import torch

from patch_integrate import integrate
from quadtree import build_partition, tiered_tolerance

HULL_EXTENT = (0.0, 0.0, 1.0, 1.0)


def tokenize(features, weights, extent, tolerance, scale=1.0, max_level=12):
    """features (N,C) 前两列为参数域坐标，weights (N,) 为求积权重。

    返回 token 特征 (P,C)、patch 面积 (P,)、Δ 加权形心 (P,2)。
    """
    partition = build_partition(features[:, :2].numpy(), weights.numpy(), extent, tolerance,
                               scale=scale, max_level=max_level)
    order = torch.from_numpy(partition.order)
    starts = torch.from_numpy(partition.starts)
    tokens, _, centres = integrate(features[order], weights[order], starts)
    return tokens, torch.from_numpy(partition.areas), centres


def flow_tokens(batch, index, size_tiers, max_level=12, use_sdf=True):
    """流场域：SDF 始终用于自适应分档，但是否写入 token 由 use_sdf 控制。"""
    mask = batch["flow_mask"][index].cpu().reshape(-1)
    coords = batch["flow_coords"][index].cpu().reshape(-1, 2)[mask]
    sdf = batch["flow_sdf"][index].cpu().reshape(-1)[mask]
    weights = batch["flow_weight"][index].cpu().reshape(-1)[mask]
    features = torch.column_stack([coords, sdf]) if use_sdf else coords
    tolerance = tiered_tolerance(sdf.numpy(), size_tiers)
    return tokenize(features, weights, tuple(batch["flow_extent"][index].tolist()), tolerance,
                    1.0, max_level)


def hull_tokens(batch, index, hull_size, max_level=12):
    """船体域：半船 UV 参数域上分词，容差用固定物理弧长。"""
    mask = batch["hull_mask"][index].cpu()
    features = torch.cat([batch["hull_uv"][index].cpu(), batch["hull_points"][index].cpu(),
                          batch["hull_normals"][index].cpu()], dim=-1)[mask]
    weights = batch["hull_weight"][index].cpu()[mask]
    tolerance = np.full(len(weights), hull_size)
    return tokenize(features, weights, HULL_EXTENT, tolerance, None, max_level)
