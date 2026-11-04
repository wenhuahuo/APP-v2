"""预归一化 transformer 块，支持逐层工况调制、加性编码与条件交叉注意力。

块内融合骨架与调制模块都在 `conditioning.py`（`fused_block` / `ConditionedBlock`），
APPSolver-v2 与 OFormer 主干共用；本文件只负责 geometric 注意力与 MLP 的组装。
`film` / `adaln` / `adaln_zero` 保持原有公式；`fourier_add` 在 block 入口加零初始化
的工况编码；`cond_xattn` 在原注意力之后插入零门控条件交叉注意力；`none` 与
`cond_token` 不在 block 内调制，后者由整模型向序列加入条件 token。
"""

import torch.nn as nn

from conditioning import ConditionedBlock, fused_block
from quad_attention import QuadAttention


class EncoderBlock(ConditionedBlock):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0, rope=False,
                 conditioning="film", condition_dim=2):
        super().__init__(conditioning, condition_dim, d_model, nhead, dropout)
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = QuadAttention(d_model, nhead, rope=rope, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
        )

    def forward(self, tokens, weight, coords, condition):
        return fused_block(
            tokens, condition, self, self.norm_attention,
            lambda value: self.attention(value, value, weight=weight,
                                        coords_query=coords, coords_key=coords),
            self.norm_mlp, self.mlp)


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, layers, mlp_ratio=4.0, dropout=0.0, rope=False,
                 conditioning="film", condition_dim=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, nhead, mlp_ratio, dropout, rope, conditioning, condition_dim)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, weight, coords, condition):
        for block in self.blocks:
            tokens = block(tokens, weight, coords, condition)
        return self.norm(tokens)
