"""预归一化 transformer 块，工况参数由配置选择 none / FiLM / adaLN / adaLN-Zero 调制。

工况（Fr、log10 Re）不拼成 token 前缀：v1 已证明该做法拿不到条件增益。这里由
条件 MLP 生成逐层的调制参数：`film` 在 block 入口做 `x·(1+γ)+β`；`adaln` /
`adaln_zero` 把 LayerNorm 后的特征按残差分支（注意力 / MLP）分别调制，
`adaln_zero` 再加逐分支门控。`none` 不构造调制模块、不读取工况值，block 只保留
原来的注意力与 MLP。参数只与工况维度有关，与 token 数无关。
"""

import torch.nn as nn

from conditioning import gate_residual, make_conditioning, modulate
from quad_attention import QuadAttention


class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0, rope=False,
                 conditioning="film", condition_dim=2):
        super().__init__()
        self.kind = conditioning
        self.modulation = make_conditioning(conditioning, condition_dim, d_model)
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = QuadAttention(d_model, nhead, rope=rope, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
        )

    def forward(self, tokens, weight, coords, condition):
        if self.kind == "none":
            tokens = tokens + self.attention(self.norm_attention(tokens),
                                            self.norm_attention(tokens),
                                            weight=weight, coords_query=coords, coords_key=coords)
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
