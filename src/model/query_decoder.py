"""交叉注意力查询解码器，查询之间不做自注意力，保持分辨率无关性。

工况融合与编码器使用同一配置与同一个融合骨架（`conditioning.fused_block`）。
`cond_xattn` 在几何交叉注意力之后增加零门控的工况交叉注意力；`fourier_add` 在 block
入口加入工况编码；`cond_token` 只通过编码器输出的 memory 融入，不向查询序列添加
token 或查询间注意力。
"""

import torch.nn as nn

from conditioning import ConditionedBlock, fused_block
from positional import FourierFeatures
from quad_attention import QuadAttention


class QueryBasis(nn.Module):
    """查询点坐标的坐标基：`siren` / `fourier` / `linear`，输出 d_model 维。"""

    def __init__(self, n_dims, d_model, kind="siren", fourier_scale=2.0, hidden=2):
        super().__init__()
        if kind == "siren":
            layers = [nn.Linear(n_dims, d_model), nn.SiLU()]
            layers += [layer for _ in range(hidden)
                       for layer in (nn.Linear(d_model, d_model), nn.SiLU())]
            self.net = nn.Sequential(*layers)
        elif kind == "fourier":
            self.net = nn.Sequential(
                FourierFeatures(n_dims, d_model // 2, scale=fourier_scale),
                nn.Linear(2 * (d_model // 2), d_model),
            )
        elif kind == "linear":
            self.net = nn.Linear(n_dims, d_model)
        else:
            raise ValueError(f"未知的坐标基类型: {kind}")

    def forward(self, coords):
        return self.net(coords)


class DecoderBlock(ConditionedBlock):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0, rope=False,
                 conditioning="film", condition_dim=2):
        super().__init__(conditioning, condition_dim, d_model, nhead, dropout)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross = QuadAttention(d_model, nhead, rope=rope, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
        )

    def forward(self, queries, tokens, weight, coords_query, coords_key, condition):
        return fused_block(
            queries, condition, self, self.norm_cross,
            lambda value: self.cross(value, tokens, weight=weight,
                                     coords_query=coords_query, coords_key=coords_key),
            self.norm_mlp, self.mlp)


class QueryDecoder(nn.Module):
    def __init__(self, d_model, nhead, layers, mlp_ratio=4.0, dropout=0.0, rope=False,
                 conditioning="film", condition_dim=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, nhead, mlp_ratio, dropout, rope, conditioning, condition_dim)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, tokens, weight, coords_query, coords_key, condition):
        for block in self.blocks:
            queries = block(queries, tokens, weight, coords_query, coords_key, condition)
        return self.norm(queries)
