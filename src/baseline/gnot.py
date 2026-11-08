"""GNOT 基线：MOE 门控 transformer，在原始点云上逐点输出。

参考 Neural-Solver-Library 的 GNOT（ICML 2023）：几何分支与物理分支交替做线性
注意力交叉，MOE-MLP 由点坐标门控。预测点集与输入点集相同，没有物理尺寸分词，
换查询密度时结构不随之调整。

给出 `input_domains` 时流场与船面点云拼成同一片点云联合编码（船体几何因此进入流场
预测），`conditioning: fourier_add`（主实验口径）/ `film` 在 block 入口做逐层工况注入
（`point_cloud.token_modulation`，与 APPSolver-v2 同一套公式）；缺省时保持历史口径的
独立编码与工况加性偏置。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from point_cloud import (FLOW_FEATURES, CloudEmbedding, ConditionBias, apply_modulation,
                         flow_cloud, hull_cloud, lattice_index, token_modulation)


class LinearAttention(nn.Module):
    def __init__(self, dim, nhead, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.dim_head = dim // nhead
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None):
        y = x if y is None else y
        batch, n_query, dim = x.shape
        n_key = y.shape[1]
        shape = (batch, n_query, self.nhead, self.dim_head)
        q = self.q_proj(x).view(shape).transpose(1, 2).softmax(dim=-1)
        k = self.k_proj(y).view(batch, n_key, self.nhead, self.dim_head).transpose(1, 2)
        v = self.v_proj(y).view(batch, n_key, self.nhead, self.dim_head).transpose(1, 2)
        k = k.softmax(dim=-1)
        normalizer = 1.0 / (q * k.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True)
        context = k.transpose(-2, -1) @ v
        out = (q @ context) * normalizer + q
        return self.proj(self.dropout(out.transpose(1, 2).reshape(batch, n_query, dim)))


class GNOTBlock(nn.Module):
    """几何-物理交替 + 位置门控 MOE。"""

    def __init__(self, d_model, nhead, mlp_ratio=4, n_experts=3, dropout=0.0, space_dim=2,
                 conditioning=None, condition_dim=2):
        super().__init__()
        self.modulation = token_modulation(conditioning, condition_dim, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(5)])
        self.cross = LinearAttention(d_model, nhead, dropout)
        self.self_attention = LinearAttention(d_model, nhead, dropout)
        self.experts_physics = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
                          nn.Linear(int(mlp_ratio * d_model), d_model))
            for _ in range(n_experts)])
        self.experts_geometry = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
                          nn.Linear(int(mlp_ratio * d_model), d_model))
            for _ in range(n_experts)])
        self.gate = nn.Sequential(
            nn.Linear(space_dim, int(mlp_ratio * d_model)), nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), int(mlp_ratio * d_model)), nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), n_experts))

    @staticmethod
    def _moe(experts, gate, tokens):
        out = torch.zeros_like(tokens)
        for i, expert in enumerate(experts):
            out = out + gate[..., i, None] * expert(tokens)
        return out

    def forward(self, geometry, physics, coords, condition=None):
        gate = F.softmax(self.gate(coords), dim=-1)
        physics = apply_modulation(physics, self.modulation, condition)
        physics = physics + self.cross(self.norms[0](physics), self.norms[1](geometry))
        physics = physics + self.norms[2](self._moe(self.experts_physics, gate, physics))
        physics = physics + self.self_attention(self.norms[3](physics))
        physics = physics + self.norms[4](self._moe(self.experts_geometry, gate, physics))
        return physics


class GNOTBaseline(nn.Module):
    def __init__(self, d_model=96, nhead=6, layers=2, mlp_ratio=2, n_experts=3,
                 dropout=0.0, condition_dim=2, flow_channels=5, hull_channels=1,
                 input_domains=None, output_domains=None, conditioning=None):
        super().__init__()
        self.input_domains = tuple(input_domains) if input_domains else ("flow", "hull")
        self.output_domains = tuple(output_domains) if output_domains else ("flow", "hull")
        self.joint = input_domains is not None
        if self.joint:
            self.cloud = CloudEmbedding(d_model)
        else:
            self.flow_input = nn.Linear(len(FLOW_FEATURES), d_model)
            self.hull_input = nn.Linear(8, d_model)
        self.condition = ConditionBias(condition_dim, d_model) if conditioning is None else None
        self.blocks = nn.ModuleList([
            GNOTBlock(d_model, nhead, mlp_ratio, n_experts, dropout, conditioning=conditioning,
                      condition_dim=condition_dim) for _ in range(layers)])
        self.placeholder = nn.Parameter(torch.rand(d_model) / d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU())
        self.flow_head = (nn.Linear(2 * d_model, flow_channels)
                          if "flow" in self.output_domains else None)
        self.hull_head = (nn.Linear(2 * d_model, hull_channels)
                          if "hull" in self.output_domains else None)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # `zero_init` 的 Linear（`conditioning.FourierAdd` 的投影）保持零初始化，
        # 使工况加性注入在初始时不改变网络。
        if isinstance(module, nn.Linear) and not getattr(module, "zero_init", False):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def _points(self, geometry, coords, bias, condition=None):
        physics = geometry + bias
        for block in self.blocks:
            physics = block(geometry, physics, coords, condition)
        return self.head(physics)

    def forward(self, batch, flow_queries):
        """输入整片点云，输出查询点处的场值；批大小固定为 1。"""
        condition = batch["condition"]
        bias = self.condition(condition) if self.condition is not None else 0.0
        output = {}
        values = None
        if self.joint:
            tokens, coords, count = self.cloud(batch)
            values = self._points(tokens + self.placeholder, coords, bias, condition)[0]
        if self.flow_head is not None:
            if self.joint:
                flow = self.flow_head(values[:count])[None]
            else:
                features = flow_cloud(batch)
                geometry = self.flow_input(features) + self.placeholder
                flow = self.flow_head(self._points(geometry, features[..., :2], bias, condition))
            output["flow"] = flow[0].index_select(
                0, lattice_index(batch, flow_queries[0]))[None]
        if self.hull_head is not None:
            if self.joint:
                output["hull"] = self.hull_head(values[count:])[None]
            else:
                features = hull_cloud(batch)
                geometry = self.hull_input(features) + self.placeholder
                output["hull"] = self.hull_head(
                    self._points(geometry, features[..., :2], bias, condition))
        return output
