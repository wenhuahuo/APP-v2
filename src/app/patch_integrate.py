"""patch 积分 token：token = Σ Δ f / Σ Δ。

patch 由 `build_partition` 给出，点按 patch 连续排列，因此分段求和即可。
`features` 的前两列必须是参数域坐标（流场域 xy、船体域 uv），用于求 patch 形心
——形心是后续位置编码的唯一依据，不存在按 patch 序号的可学习嵌入。
"""

import torch


def segment_sum(values, starts):
    """沿第 0 维按 `starts` 分段求和，values (N,C) -> (P,C)。"""
    cumulative = torch.cat([torch.zeros_like(values[:1]), values.cumsum(0)])
    return cumulative[starts[1:]] - cumulative[starts[:-1]]


def integrate(features, weights, starts):
    """返回 (tokens, areas, centres)。

    features (N,F) 点特征，weights (N,) 求积权重，starts (P+1,) patch 边界。
    """
    weighted = features * weights[:, None]
    areas = segment_sum(weights[:, None], starts)[:, 0]
    tokens = segment_sum(weighted, starts) / areas[:, None]
    centres = segment_sum(weighted[:, :2], starts) / areas[:, None]
    return tokens, areas, centres
