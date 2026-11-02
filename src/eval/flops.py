"""前向计算量统计：`FlopCounterMode` 加补齐未登记算子，供等计算量（iso-FLOPs）对照使用。

`torch.utils.flop_counter` 只登记了 CUDA 侧的 SDPA 融合算子
（`_scaled_dot_product_efficient_attention` / `_flash_attention_forward` /
`_cudnn_attention`）。CPU 上 `nn.MultiheadAttention` 走的是
`_scaled_dot_product_flash_attention_for_cpu`，未登记时整段注意力不计数：
UPT / CORAL / GINO 的 perceiver 编码器与 APPSolver-v1 的 `TransformerEncoderLayer`
都会因此被低估（APPSolver-v1 在 A100 级配置上从 ~12 G 掉到 4.8 G，注意力一项全丢）。
这里按官方 `sdpa_flop_count` 的口径补上 CPU 融合算子：`q @ kᵀ` 与 `scores @ v`
两个批量矩阵乘，因果掩码按一半计。

统计口径（与 `scripts/eval/profile_model_flops.py` 一致）：只计 matmul / bmm / conv
类算子，不含 softmax、LayerNorm 等逐元素算子；结果单位 GFLOPs，一次前向。
"""

import torch
from torch.utils.flop_counter import FlopCounterMode, flop_registry, register_flop_formula

# CPU 侧的 SDPA 融合算子名随 torch 版本变化（2.5 为 for_cpu 变体），逐个探测后再登记，
# 避免 torch 自己登记过时重复注册报错。
_CPU_SDPA_PACKETS = ("_scaled_dot_product_flash_attention_for_cpu",)


def _attention_flops(query_shape, key_shape, value_shape):
    """q @ kᵀ + scores @ v 的矩阵乘 FLOPs。"""
    *batch_heads, q_len, head_dim = query_shape
    k_len = key_shape[-2]
    count = 1
    for size in batch_heads:
        count *= size
    return 2 * count * q_len * k_len * head_dim * 2


def _cpu_sdpa_flop(query_shape, key_shape, value_shape, dropout_p=0.0, is_causal=False,
                   scale=None, out_shape=None, **kwargs):
    """CPU 融合注意力：形状为 `(batch, head, len, dim)`，因果掩码按一半计。"""
    flops = _attention_flops(query_shape, key_shape, value_shape)
    return flops // 2 if is_causal else flops


def register_missing_flop_formulas():
    """补齐 `FlopCounterMode` 未登记的 CPU 融合注意力算子，返回本次登记的名字。"""
    registered = []
    for name in _CPU_SDPA_PACKETS:
        packet = getattr(torch.ops.aten, name, None)
        if packet is None or packet in flop_registry:
            continue
        register_flop_formula([packet])(_cpu_sdpa_flop)
        registered.append(name)
    return registered


register_missing_flop_formulas()


def forward_flops(model, batch, queries):
    """返回 `(参数量, 前向 GFLOPs)`。

    ``model`` 需已置于 ``train()``：`FlopCounterMode` 依赖 autograd 的 `grad_fn` 才能
    观察到算子，eval 模式下 `nn.TransformerEncoderLayer` 会走融合内核。所有配置的
    dropout 都为 0，两种模式的前向路径与数值一致。
    """
    counter = FlopCounterMode(display=False)
    with counter:
        output = model(batch, queries)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    total = counter.get_total_flops() / 1e9
    del output
    return parameters, total
