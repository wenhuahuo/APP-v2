"""基线模型共用的点云组装、双域联合嵌入、工况融合与坐标查询工具。

任务口径有两种，由模型构造参数显式区分：

- 历史口径（不给 `input_domains` / `output_domains` / `conditioning`）：流场与船面
  各自独立编码，输出流场与船面两个域，工况以加性偏置加进输入嵌入；
- 主实验口径（`input_domains: [flow, hull]`、`output_domains: [flow]`、
  `conditioning: film`）：两个输入域拼成同一片点云联合编码，船体几何因此真实影响
  流场预测；工况在没有自注意力的块入口做逐层 FiLM（与 APPSolver-v2 同一公式）。

批大小固定为 1：训练每步只用一条数据集条目（`collate_samples([sample])`）。
需要分辨率无关查询的模型（UPT / GINO / CORAL / OFormer）从查询坐标解码，不经过
格点下标；把输出留在输入点上的模型（GNOT / Transolver / APPSolver-v1）用
`lattice_index` 从整片预测里取值。

流场输入特征只有自由液面坐标 `(x, y)`：到船体水线的符号距离只用于 APP 分词器的
分档判据（`src/app/tokenizer.py`），不进任何模型的输入。
"""

import math

import torch
import torch.nn as nn

from conditioning import FiLM, NoConditioning

FLOW_FEATURES = ("x", "y")
HULL_FEATURES = ("u", "v", "x", "y", "z", "nx", "ny", "nz")


def flow_cloud(batch, index=0):
    """当前条目流场格点的输入特征 (1, nx*ny, 2)，只有坐标；掩除位只由权重屏蔽。"""
    coords = batch["flow_coords"][index].reshape(-1, 2)
    return coords[None]


def hull_cloud(batch, index=0):
    """船面点云的输入特征 (1, M, 8)（UV、坐标、法向），padding 位不参与编码。"""
    features = torch.cat([batch["hull_uv"][index], batch["hull_points"][index],
                          batch["hull_normals"][index]], dim=-1)
    if "hull_mask" in batch:
        features = features[batch["hull_mask"][index]]
    return features[None]


def hull_coordinates(batch, index=0, space="physical"):
    """船面点的二维位置：`physical` 取 (x, y) 与流场同处一个坐标系，`uv` 取参数域。"""
    features = hull_cloud(batch, index)[0]
    return features[:, :2] if space == "uv" else features[:, 2:4]


def joint_cloud(batch, index=0):
    """流场与船面点云拼成同一片点云：特征 (1, N+M, ·)、二维位置 (1, N+M, 2)、流场点数。

    两个域的特征维数不同，模型用各自的线性投影后再拼接，因此这里只返回裁剪好的
    原始特征与位置，不做投影。
    """
    flow = flow_cloud(batch, index)[0]
    hull = hull_cloud(batch, index)[0]
    coords = torch.cat([flow, hull[:, 2:4]])
    return flow, hull, coords, len(flow)


def lattice_index(batch, queries, index=0):
    """查询点坐标 → 格点展平下标。查询点由格点抽样得到，整数换算没有舍入误差。"""
    nx, ny = batch["flow_coords"].shape[1:3]
    x0, y0, x1, y1 = batch["flow_extent"][index].tolist()
    step_x, step_y = (x1 - x0) / (nx - 1), (y1 - y0) / (ny - 1)
    ix = torch.round((queries[..., 0] - x0) / step_x).long()
    iy = torch.round((queries[..., 1] - y0) / step_y).long()
    return ix * ny + iy


def flow_frame(batch, index=0):
    """流场参数域的 `(中心, 半宽)`，用于把查询坐标映射到 [-1, 1]。"""
    x0, y0, x1, y1 = batch["flow_extent"][index].tolist()
    centre = torch.tensor([0.5 * (x0 + x1), 0.5 * (y0 + y1)],
                          dtype=batch["flow_extent"].dtype, device=batch["flow_extent"].device)
    half = torch.tensor([0.5 * (x1 - x0), 0.5 * (y1 - y0)],
                        dtype=batch["flow_extent"].dtype, device=batch["flow_extent"].device)
    return centre, half


def normalize_within(coords, low, span):
    """把坐标按包围盒映射到 [0, 1]；`low` / `span` 的形状广播到 `coords` 的最后一维。"""
    return (coords - low) / span.clamp_min(torch.finfo(coords.dtype).eps)


def token_modulation(kind, condition_dim, d_model):
    """块入口的工况调制：`film` 为逐层独立参数的 FiLM，其余（含 `None`）不调制。"""
    return FiLM(condition_dim, d_model) if kind == "film" else NoConditioning()


def apply_modulation(tokens, modulation, condition):
    """FiLM 的块入口形式 `x·(1+γ)+β`；无调制时原样返回。"""
    if isinstance(modulation, NoConditioning):
        return tokens
    gamma, beta = modulation(condition)
    return tokens * (1.0 + gamma[:, None]) + beta[:, None]


class ConditionBias(nn.Module):
    """历史口径：工况参数作为逐点偏置加到输入嵌入上，不做逐层 FiLM。"""

    def __init__(self, condition_dim, d_model):
        super().__init__()
        self.projection = nn.Linear(condition_dim, d_model)

    def forward(self, condition):
        return self.projection(condition)[:, None, :]


class CloudEmbedding(nn.Module):
    """双域点云的逐点嵌入：两个域各自的线性投影 + 域标识，拼成同一片 token 序列。

    流场点在前、船面点在后，调用方用返回的 `flow_count` 切开两个域。
    """

    def __init__(self, d_model):
        super().__init__()
        self.flow_input = nn.Linear(len(FLOW_FEATURES), d_model)
        self.hull_input = nn.Linear(len(HULL_FEATURES), d_model)
        self.domain = nn.Embedding(2, d_model)

    def forward(self, batch, index=0):
        flow, hull, coords, count = joint_cloud(batch, index)
        tokens = torch.cat([self.flow_input(flow) + self.domain.weight[0],
                            self.hull_input(hull) + self.domain.weight[1]])
        return tokens[None], coords[None], count


class PerceiverBlock(nn.Module):
    """latent 交叉注意力 + latent 自注意力，各带逐层 MLP 与可选的块入口 FiLM。"""

    def __init__(self, d_model, nhead, mlp_ratio=4, dropout=0.0, conditioning=None,
                 condition_dim=2):
        super().__init__()
        self.modulation = token_modulation(conditioning, condition_dim, d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_key = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                    batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
                                 nn.Linear(int(mlp_ratio * d_model), d_model))

    def forward(self, latents, points, condition=None):
        if condition is not None:
            latents = apply_modulation(latents, self.modulation, condition)
        latents = latents + self.cross(self.norm_cross(latents), self.norm_key(points),
                                       self.norm_key(points), need_weights=False)[0]
        latents = latents + self.self_attention(self.norm_self(latents), self.norm_self(latents),
                                                self.norm_self(latents), need_weights=False)[0]
        return latents + self.mlp(self.norm_mlp(latents))


class PerceiverEncoder(nn.Module):
    """固定数量 latent 对输入点云做交叉注意力，再在 latent 上做自注意力。

    latent 数量固定，输入点数变化时表征容量不变；编码结果只取决于输入点云与工况，
    与查询点无关，因此可以先编码一次、在任意查询点集上多次解码。
    """

    def __init__(self, d_model, nhead, layers, latents, mlp_ratio=4, dropout=0.0,
                 conditioning=None, condition_dim=2):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, latents, d_model) * d_model ** -0.5)
        self.blocks = nn.ModuleList([
            PerceiverBlock(d_model, nhead, mlp_ratio, dropout, conditioning, condition_dim)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, points, condition=None):
        latents = self.latent.expand(points.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, points, condition)
        return self.norm(latents)


def sincos_embedding(coords, d_model):
    """ContinuousSincosEmbed：把 `coords (..., C)` 编码成 `d_model` 维正弦位置编码。

    每个坐标轴分到 `d_model // C` 维，内部按 `freq_i = 1 / 10000^(i / half)` 取频率
    并交错存放 sin / cos。
    """
    n_dims = coords.shape[-1]
    per_dim = d_model // n_dims
    if per_dim % 2:
        raise ValueError(f"位置编码维度 {d_model} 无法按坐标维数 {n_dims} 均分")
    half = per_dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=coords.dtype,
                                                             device=coords.device) / half)
    args = coords.unsqueeze(-1) * frequencies
    return torch.cat([args.sin(), args.cos()], dim=-1).flatten(-2)
