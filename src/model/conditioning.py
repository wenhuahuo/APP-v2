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
        # 通用的逐模块权重初始化（如 GNOT 的 `self.apply(_init_weights)`）会把投影当普通
        # Linear 重置；打上标记让这些初始化函数跳过，保住「初始加性项为零」。
        self.projection.zero_init = True

    def forward(self, condition):
        return self.projection(self.encoding(condition))[:, None]


class ConditionCrossAttention(nn.Module):
    """对单个工况嵌入做交叉注意力，以零初始化标量门控接入残差路径。

    token 可以是 `(B, N, d)`，也可以是不带 batch 维的 `(N, d)`（APP+OFormer 的
    patch token 序列就是后一种）；后者按 `B = 1` 处理并原样返回形状。
    """

    def __init__(self, condition_dim, d_model, nhead, dropout=0.0):
        super().__init__()
        self.condition = nn.Linear(condition_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, condition):
        unbatched = tokens.dim() == 2
        tokens = tokens[None] if unbatched else tokens
        context = self.condition(condition)[:, None]
        attended = self.attention(self.norm(tokens), context, context,
                                  need_weights=False)[0]
        tokens = tokens + torch.tanh(self.gate) * attended
        return tokens[0] if unbatched else tokens


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


class ConditionedBlock(nn.Module):
    """块内工况融合的公共部件：调制模块与（可选的）条件交叉注意力。

    `conditioning` 为 `None` 时保持历史口径的「块内不调制」（由整模型的工况加性偏置
    负责），因此 `None` 与 `none` 在块内行为相同，但两者是否带偏置由模型决定。
    """

    def __init__(self, conditioning, condition_dim, d_model, nhead=None, dropout=0.0):
        super().__init__()
        self.kind = conditioning
        self.modulation = make_conditioning("none" if conditioning is None else conditioning,
                                            condition_dim, d_model)
        self.condition_cross = (ConditionCrossAttention(condition_dim, d_model, nhead, dropout)
                                if conditioning == "cond_xattn" else None)


def fused_block(x, condition, block, norm_attention, attend, norm_mlp, mlp):
    """带工况融合的预归一化残差块，返回 `x + 注意力分支 + MLP 分支`。

    `block` 给出 `kind` / `modulation` / `condition_cross`（即 `ConditionedBlock`），
    `attend` 与 `mlp` 接收归一化（并按需调制）后的张量并返回残差分支输出，由调用方
    绑定各自的注意力与 MLP 模块。APPSolver-v2 与 OFormer 主干共用这一套公式。
    """
    if block.kind in {None, "none", "cond_token", "cond_xattn", "fourier_add"}:
        if block.kind == "fourier_add":
            x = x + block.modulation(condition)
        x = x + attend(norm_attention(x))
        if block.condition_cross is not None:
            x = block.condition_cross(x, condition)
        return x + mlp(norm_mlp(x))

    branch = block.modulation(condition)
    if block.kind == "film":
        gamma, beta = branch
        x = x * (1.0 + gamma[:, None]) + beta[:, None]
        x = x + attend(norm_attention(x))
        return x + mlp(norm_mlp(x))

    attention_branch, mlp_branch = branch
    residual = attend(modulate(norm_attention(x), attention_branch))
    x = x + gate_residual(residual, attention_branch)
    residual = mlp(modulate(norm_mlp(x), mlp_branch))
    return x + gate_residual(residual, mlp_branch)
