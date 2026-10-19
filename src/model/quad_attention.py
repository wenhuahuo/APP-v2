"""带求积权重的注意力（NNs-to-NOs 式 13）。

    Attn(q_i) = Σ_j Δ_j exp(s_ij) v_j / Σ_j Δ_j exp(s_ij),  s_ij = q_i·k_j/√d

等价于在 logits 上加 `log Δ_j`：Δ 全相等时退化为标准 softmax 注意力，
Δ_j = 0（padding token）时该位置被自动屏蔽。`weight` 的绝对尺度无影响。
"""

import torch
import torch.nn as nn

from positional import apply_rope


class QuadAttention(nn.Module):
    def __init__(self, d_model, nhead, rope=False, dropout=0.0):
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model {d_model} 不能被头数 {nhead} 整除")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.rope = rope
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value=None, weight=None, coords_query=None, coords_key=None):
        """query (B,Nq,D)，key (B,Nk,D)，weight (B,Nk) 为 key 的求积权重。"""
        value = key if value is None else value
        batch, n_query, _ = query.shape

        q = self._split_heads(self.query(query))
        k = self._split_heads(self.key(key))
        v = self._split_heads(self.value(value))
        if self.rope:
            if coords_query is None or coords_key is None:
                raise ValueError("启用 RoPE 时必须提供查询点与 key 的坐标")
            q = apply_rope(q, coords_query)
            k = apply_rope(k, coords_key)

        logits = q @ k.transpose(-1, -2) * self.scale
        if weight is not None:
            logits = logits + weight.log()[:, None, None, :]
        attention = torch.softmax(logits, dim=-1)
        output = self.dropout(attention) @ v
        output = output.transpose(1, 2).reshape(batch, n_query, self.nhead * self.head_dim)
        return self.output(output)
