"""工况条件融合：逐层调制、Fourier 加性编码与门控条件交叉注意力。

`none` / `film` / `adaln` / `adaln_zero` 为已有调制路线；`fourier_add` 在每个 block
入口加入零初始化的工况 Fourier 编码投影；`cond_token` 与 `cond_xattn` 在 block 内
不做调制，分别由整模型添加上下文 token、由本模块的条件交叉注意力逐层融合。

`adaLN` 与 `adaLN-Zero` 的语义取自 DiT（Peebles & Xie, 2023）：
`y = x + α₁ ⊙ MSA((1+γ₁)⊙LN(x) + β₁)`，`x = y + α₂ ⊙ MLP((1+γ₂)⊙LN(y) + β₂)`。
`fourier_add` 参考 DDPM 将时间编码加入每个残差块；`cond_xattn` 使用 Flamingo 式
零初始化 tanh 门控，使新增条件分支在初始时不改变原网络。
"""

import torch
import torch.nn as nn

from positional import FourierFeatures

CONDITIONINGS = ("none", "film", "adaln", "adaln_zero", "fourier_add",
                 "cond_token", "cond_xattn")


class NoConditioning(nn.Module):
    """显式不使用条件：不含任何参数，也不产生调制张量。"""

    def forward(self, condition):
        return None


class FiLM(nn.Module):
    """block 入口调制：返回逐通道 `(γ, β)`，作用为 `x·(1+γ) + β`。"""

    def __init__(self, condition_dim, d_model):
        super().__init__()
        self.projection = nn.Linear(condition_dim, 2 * d_model)

    def forward(self, condition):
        return self.projection(condition).chunk(2, dim=-1)


class AdaNorm(nn.Module):
    """adaLN 调制：返回两个残差分支各自的 `(shift, scale)` 或 `(shift, scale, gate)`。

    每个 block 一个实例，参数逐层独立生成。
    """

    def __init__(self, condition_dim, d_model, zero_gate=False):
        super().__init__()
        self.zero_gate = zero_gate
        self.projection = nn.Linear(condition_dim, 6 * d_model if zero_gate else 4 * d_model)
        if zero_gate:
            # 权重与偏置都置零，初始 gate 严格为 0（偏置若保留随机初始化就不是零门控）。
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, condition):
        channels = self.projection(condition)
        if self.zero_gate:
            shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = channels.chunk(6, dim=-1)
            return (shift, scale, gate), (shift_mlp, scale_mlp, gate_mlp)
        shift, scale, shift_mlp, scale_mlp = channels.chunk(4, dim=-1)
        return (shift, scale), (shift_mlp, scale_mlp)


class FourierAdd(nn.Module):
    """固定 Fourier 工况编码经零初始化线性投影，逐层加性注入。"""

    def __init__(self, condition_dim, d_model, n_features=16):
        super().__init__()
        self.encoding = FourierFeatures(condition_dim, n_features, scale=1.0)
        self.projection = nn.Linear(2 * n_features, d_model)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, condition):
        return self.projection(self.encoding(condition))[:, None]


class ConditionCrossAttention(nn.Module):
    """对单个工况嵌入做交叉注意力，以零初始化标量门控接入残差路径。"""

    def __init__(self, condition_dim, d_model, nhead, dropout=0.0):
        super().__init__()
        self.condition = nn.Linear(condition_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, condition):
        context = self.condition(condition)[:, None]
        attended = self.attention(self.norm(tokens), context, context,
                                  need_weights=False)[0]
        return tokens + torch.tanh(self.gate) * attended


def make_conditioning(kind, condition_dim, d_model):
    """按配置名构造调制模块；未知名字直接报错，不做静默回退。"""
    if kind in {"none", "cond_token", "cond_xattn"}:
        return NoConditioning()
    if kind == "film":
        return FiLM(condition_dim, d_model)
    if kind == "fourier_add":
        return FourierAdd(condition_dim, d_model)
    if kind == "adaln":
        return AdaNorm(condition_dim, d_model)
    if kind == "adaln_zero":
        return AdaNorm(condition_dim, d_model, zero_gate=True)
    raise ValueError(f"未知的工况调制方式: {kind}（可选 {' / '.join(CONDITIONINGS)}）")


def modulate(x, branch):
    """`(1+scale)·x + shift`，逐通道、按样本广播。"""
    shift, scale = branch[0], branch[1]
    return x * (1.0 + scale[:, None]) + shift[:, None]


def gate_residual(residual, branch):
    """adaLN-Zero 的逐分支门控 `α ⊙ residual`；`adaln` 无门控，原样返回。"""
    return residual if len(branch) == 2 else residual * branch[2][:, None]
