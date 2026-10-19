"""UPT 基线：perceiver 编码到固定数量 latent，再解码回原始点云。

参考 Universal Physics Transformers（NeurIPS 2024）：输入点云经交叉注意力压到固定
数量的 latent token，在 latent 上做自注意力，再用查询坐标交叉注意力解码回点。
latent 数量固定，输入点数变化时表征容量不变，因此换查询密度不随之调整。
"""

import math

import torch
import torch.nn as nn

from point_cloud import ConditionBias, flow_cloud, hull_cloud, lattice_index


class SincosEmbed(nn.Module):
    """坐标的连续正弦位置编码，输出 d_model 维。"""

    def __init__(self, d_model, space_dim=2):
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.exp(-math.log(10000.0) * torch.arange(d_model // (2 * space_dim))
                      / (d_model // (2 * space_dim))), requires_grad=False)

    def forward(self, coords):
        args = coords.unsqueeze(-1) * self.frequencies
        return torch.cat([args.sin(), args.cos()], dim=-1).flatten(-2)


class PerceiverBlock(nn.Module):
    """latent 交叉注意力 + latent 自注意力，各带逐层 MLP。"""

    def __init__(self, d_model, nhead, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_key = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                    batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
                                 nn.Linear(int(mlp_ratio * d_model), d_model))

    def forward(self, latents, points):
        latents = latents + self.cross(self.norm_cross(latents), self.norm_key(points),
                                       self.norm_key(points), need_weights=False)[0]
        latents = latents + self.self_attention(self.norm_self(latents), self.norm_self(latents),
                                                self.norm_self(latents), need_weights=False)[0]
        return latents + self.mlp(self.norm_mlp(latents))


class UPTBaseline(nn.Module):
    def __init__(self, d_model=128, nhead=8, layers=2, latents=128, mlp_ratio=4,
                 dropout=0.0, condition_dim=2, flow_channels=5, hull_channels=1):
        super().__init__()
        self.position = SincosEmbed(d_model)
        self.flow_input = nn.Linear(3, d_model)
        self.hull_input = nn.Linear(8, d_model)
        self.condition = ConditionBias(condition_dim, d_model)
        self.latent = nn.Parameter(torch.randn(1, latents, d_model) * d_model ** -0.5)
        self.blocks = nn.ModuleList([
            PerceiverBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.decode = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.decode_key = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU())
        self.flow_head = nn.Linear(int(mlp_ratio * d_model), flow_channels)
        self.hull_head = nn.Linear(int(mlp_ratio * d_model), hull_channels)

    def _points(self, features, projection, bias):
        # 查询与编码共用同一个逐点嵌入：解码端因此能拿到该点自身的坐标与几何特征，
        # 与参考实现 `RansPerceiver_Decoder` 把 query_fx 拼进 query 的做法一致。
        coords = features[..., :2]
        points = self.position(coords) + projection(features) + bias
        latents = self.latent.expand(points.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, points)
        latents = self.norm(latents)
        decoded = self.decode(points, self.decode_key(latents), self.decode_key(latents),
                              need_weights=False)[0]
        return self.head(decoded + points)

    def forward(self, batch, flow_queries):
        """输入整片点云，输出查询点处的场值；批大小固定为 1。"""
        bias = self.condition(batch["condition"])
        flow = self.flow_head(self._points(flow_cloud(batch), self.flow_input, bias))
        flow = flow[0].index_select(0, lattice_index(batch, flow_queries[0]))[None]
        hull = self.hull_head(self._points(hull_cloud(batch), self.hull_input, bias))
        return {"flow": flow, "hull": hull}
