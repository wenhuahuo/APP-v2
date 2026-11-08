"""Transolver 基线：物理注意力（learned slice）在原始点云上逐点输出。

参考 Transolver（ICLR 2024）：每层把 N 个点软分配到固定数量的 slice 上，在 slice
之间做注意力，再按同一权重投影回点。slice 数固定，输入点数变化时表征容量不变，
因此换查询密度不随之调整。

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


class PhysicsAttention(nn.Module):
    def __init__(self, d_model, nhead, slices=32, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.dim_head = d_model // nhead
        self.slices = slices
        self.temperature = nn.Parameter(torch.ones(1, nhead, 1, 1) * 0.5)
        self.project_x = nn.Linear(d_model, d_model)
        self.project_fx = nn.Linear(d_model, d_model)
        self.project_slice = nn.Linear(self.dim_head, slices)
        nn.init.orthogonal_(self.project_slice.weight)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def forward(self, tokens):
        batch, n_point, _ = tokens.shape
        shape = (batch, n_point, self.nhead, self.dim_head)
        geometry = self.project_x(tokens).view(shape).permute(0, 2, 1, 3)
        physics = self.project_fx(tokens).view(shape).permute(0, 2, 1, 3)
        weight = F.softmax(self.project_slice(geometry) / self.temperature, dim=-1)
        norm = weight.sum(2)[..., None]
        slice_token = torch.einsum("bhnc,bhng->bhgc", physics, weight) / (norm + 1e-5)
        query, key, value = self.to_q(slice_token), self.to_k(slice_token), self.to_v(slice_token)
        attention = F.softmax(query @ key.transpose(-1, -2) * self.dim_head ** -0.5, dim=-1)
        out = torch.einsum("bhgc,bhng->bhnc", attention @ value, weight)
        return self.to_out(out.permute(0, 2, 1, 3).reshape(batch, n_point, -1))


class TransolverBlock(nn.Module):
    def __init__(self, d_model, nhead, slices, mlp_ratio=4, dropout=0.0, conditioning=None,
                 condition_dim=2):
        super().__init__()
        self.modulation = token_modulation(conditioning, condition_dim, d_model)
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = PhysicsAttention(d_model, nhead, slices, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
                                 nn.Linear(int(mlp_ratio * d_model), d_model))

    def forward(self, tokens, condition=None):
        tokens = apply_modulation(tokens, self.modulation, condition)
        tokens = tokens + self.attention(self.norm_attention(tokens))
        return tokens + self.mlp(self.norm_mlp(tokens))


class TransolverBaseline(nn.Module):
    def __init__(self, d_model=160, nhead=8, layers=3, slices=32, mlp_ratio=2,
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
            TransolverBlock(d_model, nhead, slices, mlp_ratio, dropout, conditioning,
                            condition_dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.flow_head = (nn.Linear(d_model, flow_channels)
                          if "flow" in self.output_domains else None)
        self.hull_head = (nn.Linear(d_model, hull_channels)
                          if "hull" in self.output_domains else None)

    def _points(self, tokens, bias, condition=None):
        tokens = tokens + bias
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.norm(tokens)

    def forward(self, batch, flow_queries):
        """输入整片点云，输出查询点处的场值；批大小固定为 1。"""
        condition = batch["condition"]
        bias = self.condition(condition) if self.condition is not None else 0.0
        output = {}
        values = None
        if self.joint:
            tokens, _, count = self.cloud(batch)
            values = self._points(tokens, bias, condition)[0]
        if self.flow_head is not None:
            if self.joint:
                flow = self.flow_head(values[:count])
            else:
                flow = self.flow_head(
                    self._points(self.flow_input(flow_cloud(batch)), bias, condition)[0])
            output["flow"] = flow.index_select(
                0, lattice_index(batch, flow_queries[0]))[None]
        if self.hull_head is not None:
            if self.joint:
                output["hull"] = self.hull_head(values[count:])[None]
            else:
                output["hull"] = self.hull_head(
                    self._points(self.hull_input(hull_cloud(batch)), bias, condition))
        return output
