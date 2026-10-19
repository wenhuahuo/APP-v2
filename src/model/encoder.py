"""预归一化 transformer 块，工况参数用 FiLM 调制。

工况（Fr、log10 Re）不拼成 token 前缀：v1 已证明该做法拿不到条件增益。这里由
条件 MLP 生成逐层的 (γ, β)，对 token 特征做 `x·(1+γ) + β`，作用在每一层，
参数只与工况维度有关，与 token 数无关。
"""

import torch.nn as nn

from quad_attention import QuadAttention


class FiLM(nn.Module):
    def __init__(self, condition_dim, d_model):
        super().__init__()
        self.projection = nn.Linear(condition_dim, 2 * d_model)

    def forward(self, condition):
        gamma, beta = self.projection(condition).chunk(2, dim=-1)
        return gamma, beta


class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0, rope=False):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = QuadAttention(d_model, nhead, rope=rope, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
        )

    def forward(self, tokens, weight, coords, modulation):
        gamma, beta = modulation
        tokens = tokens * (1.0 + gamma[:, None]) + beta[:, None]
        tokens = tokens + self.attention(self.norm_attention(tokens), self.norm_attention(tokens),
                                        weight=weight, coords_query=coords, coords_key=coords)
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, layers, mlp_ratio=4.0, dropout=0.0, rope=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, nhead, mlp_ratio, dropout, rope) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, weight, coords, modulations):
        for block, modulation in zip(self.blocks, modulations):
            tokens = block(tokens, weight, coords, modulation)
        return self.norm(tokens)
