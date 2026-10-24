"""交叉注意力查询解码器。

查询点只是输出函数的一种离散化：查询之间不做自注意力，否则某个点的取值会依赖
同批其它查询点的选取，破坏分辨率无关性。因此解码代价是关于查询点数的线性开销。

工况调制与编码器用同一种配置（none / FiLM / adaLN / adaLN-Zero），作用在查询特征上；
`none` 不加任何调制，也不引入参数。交叉注意力的 memory、求积权重与坐标语义不变。
"""

import torch.nn as nn

from conditioning import gate_residual, make_conditioning, modulate
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


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0, rope=False,
                 conditioning="film", condition_dim=2):
        super().__init__()
        self.kind = conditioning
        self.modulation = make_conditioning(conditioning, condition_dim, d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross = QuadAttention(d_model, nhead, rope=rope, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
        )

    def forward(self, queries, tokens, weight, coords_query, coords_key, condition):
        if self.kind == "none":
            queries = queries + self.cross(self.norm_cross(queries), tokens, weight=weight,
                                          coords_query=coords_query, coords_key=coords_key)
            queries = queries + self.mlp(self.norm_mlp(queries))
            return queries

        modulation = self.modulation(condition)
        if self.kind == "film":
            gamma, beta = modulation
            queries = queries * (1.0 + gamma[:, None]) + beta[:, None]
            queries = queries + self.cross(self.norm_cross(queries), tokens, weight=weight,
                                          coords_query=coords_query, coords_key=coords_key)
            queries = queries + self.mlp(self.norm_mlp(queries))
            return queries

        cross_modulation, mlp_modulation = modulation
        cross = self.cross(modulate(self.norm_cross(queries), cross_modulation), tokens,
                           weight=weight, coords_query=coords_query, coords_key=coords_key)
        queries = queries + gate_residual(cross, cross_modulation)
        mlp = self.mlp(modulate(self.norm_mlp(queries), mlp_modulation))
        queries = queries + gate_residual(mlp, mlp_modulation)
        return queries


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
