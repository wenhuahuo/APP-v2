"""预归一化 transformer 块，支持逐层工况调制、加性编码与条件交叉注意力。

`film` / `adaln` / `adaln_zero` 保持原有公式；`fourier_add` 在 block 入口加零初始化
的工况编码；`cond_xattn` 在原注意力之后插入零门控条件交叉注意力；`none` 与
`cond_token` 不在 block 内调制，后者由整模型向序列加入条件 token。
"""

import torch.nn as nn

from conditioning import ConditionCrossAttention, gate_residual, make_conditioning, modulate
from quad_attention import QuadAttention


class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0, rope=False,
                 conditioning="film", condition_dim=2):
        super().__init__()
        self.kind = conditioning
        self.modulation = make_conditioning(conditioning, condition_dim, d_model)
        self.condition_cross = (ConditionCrossAttention(condition_dim, d_model, nhead, dropout)
                                if conditioning == "cond_xattn" else None)
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = QuadAttention(d_model, nhead, rope=rope, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
        )

    def forward(self, tokens, weight, coords, condition):
        if self.kind in {"none", "cond_token", "cond_xattn", "fourier_add"}:
            if self.kind == "fourier_add":
                tokens = tokens + self.modulation(condition)
            tokens = tokens + self.attention(self.norm_attention(tokens),
                                            self.norm_attention(tokens),
                                            weight=weight, coords_query=coords, coords_key=coords)
            if self.condition_cross is not None:
                tokens = self.condition_cross(tokens, condition)
            tokens = tokens + self.mlp(self.norm_mlp(tokens))
            return tokens

        modulation = self.modulation(condition)
        if self.kind == "film":
            gamma, beta = modulation
            tokens = tokens * (1.0 + gamma[:, None]) + beta[:, None]
            tokens = tokens + self.attention(self.norm_attention(tokens),
                                            self.norm_attention(tokens),
                                            weight=weight, coords_query=coords, coords_key=coords)
            tokens = tokens + self.mlp(self.norm_mlp(tokens))
            return tokens

        attention_modulation, mlp_modulation = modulation
        normed = modulate(self.norm_attention(tokens), attention_modulation)
        attention = self.attention(normed, normed, weight=weight,
                                   coords_query=coords, coords_key=coords)
        tokens = tokens + gate_residual(attention, attention_modulation)
        mlp = self.mlp(modulate(self.norm_mlp(tokens), mlp_modulation))
        tokens = tokens + gate_residual(mlp, mlp_modulation)
        return tokens


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
