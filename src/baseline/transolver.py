"""Transolver 基线：物理注意力（learned slice）在原始点云上逐点输出。

参考 Transolver（ICLR 2024）：每层把 N 个点软分配到固定数量的 slice 上，在 slice
之间做注意力，再按同一权重投影回点。slice 数固定，输入点数变化时表征容量不变，
因此换查询密度不随之调整。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from point_cloud import ConditionBias, flow_cloud, hull_cloud, lattice_index


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
    def __init__(self, d_model, nhead, slices, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = PhysicsAttention(d_model, nhead, slices, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
                                 nn.Linear(int(mlp_ratio * d_model), d_model))

    def forward(self, tokens):
        tokens = tokens + self.attention(self.norm_attention(tokens))
        return tokens + self.mlp(self.norm_mlp(tokens))


class TransolverBaseline(nn.Module):
    def __init__(self, d_model=160, nhead=8, layers=3, slices=32, mlp_ratio=2,
                 dropout=0.0, condition_dim=2, flow_channels=5, hull_channels=1):
        super().__init__()
        self.flow_input = nn.Linear(3, d_model)
        self.hull_input = nn.Linear(8, d_model)
        self.condition = ConditionBias(condition_dim, d_model)
        self.blocks = nn.ModuleList([
            TransolverBlock(d_model, nhead, slices, mlp_ratio, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.flow_head = nn.Linear(d_model, flow_channels)
        self.hull_head = nn.Linear(d_model, hull_channels)

    def _points(self, features, projection, bias):
        tokens = projection(features) + bias
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, batch, flow_queries):
        """输入整片点云，输出查询点处的场值；批大小固定为 1。"""
        bias = self.condition(batch["condition"])
        flow = self.flow_head(self._points(flow_cloud(batch), self.flow_input, bias))
        flow = flow[0].index_select(0, lattice_index(batch, flow_queries[0]))[None]
        hull = self.hull_head(self._points(hull_cloud(batch), self.hull_input, bias))
        return {"flow": flow, "hull": hull}
