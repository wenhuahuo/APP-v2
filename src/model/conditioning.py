"""工况条件调制：`none` / `film` / `adaln` / `adaln_zero` 四种，由配置选择。

带条件的三种都从工况向量（`Fr`、`log10 Re`）生成逐层、逐通道的调制参数，参数量只
与工况维度和 `d_model` 有关，与 token 数、查询点数无关；工况不占 token 位。

- `none`：不使用条件编码。没有任何条件调制参数，block 内只有原来的注意力与 MLP；
  工况张量只作为批次接口保留，不参与任何计算。
- `film`：每个 block **入口**做 `x·(1+γ) + β`，随后是原来的注意力与 MLP。
- `adaln`：DiT 的 adaLN。self/cross-attention 与 MLP 两个残差分支各自在 LayerNorm
  之后做 `x·(1+scale) + shift` 再进入分支，block 入口不再叠加 FiLM。
- `adaln_zero`：adaLN 再加逐分支逐通道门控 `α`，残差写成 `x + α ⊙ branch(...)`；
  调制投影的最后一层零初始化，因此初始 `α = 0`、各残差分支被零门控、block 是恒等路径。

`adaLN` 与 `adaLN-Zero` 的语义取自 DiT（Peebles & Xie, 2023）：
`y = x + α₁ ⊙ MSA((1+γ₁)⊙LN(x) + β₁)`，`x = y + α₂ ⊙ MLP((1+γ₂)⊙LN(y) + β₂)`。
"""

import torch.nn as nn

CONDITIONINGS = ("none", "film", "adaln", "adaln_zero")


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


def make_conditioning(kind, condition_dim, d_model):
    """按配置名构造调制模块；未知名字直接报错，不做静默回退。"""
    if kind == "none":
        return NoConditioning()
    if kind == "film":
        return FiLM(condition_dim, d_model)
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
