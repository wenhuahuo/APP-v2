"""APP 分词 + Transolver slice 主干：分词器与查询解码器保持 APPSolver-v2 不变。

Transolver 的物理注意力把 N 个 token 软分配到固定数量的 slice、在 slice 之间做注意力
再按同一组权重投影回 token，是纯 token→token 映射，与 token 数量无关，因此可以直接
替换 v2 主干里的求积权重注意力。分词器、patch 形心位置编码、逐层 FiLM、查询解码器与
双头全部沿用 v2，两者的差别只有主干注意力这一项。

物理注意力不吃坐标（对 token 置换不变），位置必须由 token 自身的特征给出，因此
`positional` 只能是 `fourier`；`rope` 把位置放在注意力内部，切片注意力拿不到。
"""

import torch.nn as nn

from appsolver_v2 import APPSolverV2
from transolver import TransolverBlock


class SliceEncoderBlock(nn.Module):
    """逐层 FiLM 调制 + Transolver 的 slice 物理注意力。"""

    def __init__(self, d_model, nhead, slices, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.block = TransolverBlock(d_model, nhead, slices, mlp_ratio, dropout)

    def forward(self, tokens, weight, coords, modulation):
        gamma, beta = modulation
        return self.block(tokens * (1.0 + gamma[:, None]) + beta[:, None])


class SliceEncoder(nn.Module):
    """与 `encoder.Encoder` 同签名；求积权重与坐标不再进入注意力。"""

    def __init__(self, d_model, nhead, layers, slices, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SliceEncoderBlock(d_model, nhead, slices, mlp_ratio, dropout) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, weight, coords, modulations):
        for block, modulation in zip(self.blocks, modulations):
            tokens = block(tokens, weight, coords, modulation)
        return self.norm(tokens)


class APPSolverTransolver(APPSolverV2):
    def __init__(self, slices=32, **kwargs):
        super().__init__(**kwargs)
        if self.positional != "fourier":
            raise ValueError("slice 注意力不吃坐标，位置只能由 patch 形心编码给出"
                             "（positional: fourier）")
        self.encoder = SliceEncoder(
            kwargs["d_model"], kwargs["nhead"], kwargs["encoder_layers"], slices,
            kwargs.get("mlp_ratio", 4.0), kwargs.get("dropout", 0.0))
