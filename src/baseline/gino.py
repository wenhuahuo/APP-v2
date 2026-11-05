"""GINO 基线：输入几何点云 → GNO 到潜规则网格 → FNO → GNO 解码到任意输出查询。

参考 Geometry-Informed Neural Operator（NeurIPS 2023）与官方实现
`neuralop.models.GINO` / `neuralop.layers.gno_block` / `integral_transform` /
`fno_block`：

1. 输入 GNO：以半径为邻域把定义在不规则点云上的输入函数积分到规则潜网格
   `f(x) = mean_{y∈B_r(x)} k(x, y) · f(y)`，核 `k` 为坐标正弦编码上的逐点 MLP；
2. FNO：在潜网格上做谱卷积（截断模态的 rfft）+ 线性跳连 + 逐点通道 MLP；
3. 输出 GNO：把潜网格特征按同一核积分到任意输出查询点，再逐点投影到通道。

潜网格覆盖输入几何的包围盒并归一化到 [0,1]³，与官方「latent queries 位于单位立方体」
的约定一致；`radius` 也在归一化坐标里度量。潜网格大小、模态数与半径都是配置项。

与官方实现的差异（均为必要适配，不改变路线）：

- 官方 `fno_norm='ada_in'` 的工况注入换成项目统一的逐层 FiLM（每个 FNO block 入口
  `x·(1+γ)+β`），以便和 APPSolver-v2 及其它基线使用同一工况融合口径；
- 邻域搜索用 SciPy 的 `cKDTree`（CPU、离散索引无梯度），核宽为配置项，官方
  输入/输出核 MLP 宽度分别为 [80,80,80] 与 [512,256]，这里作为计算量旋钮；
- 半径内没有邻居的查询点退化为最近邻，避免潜网格空邻域产生 NaN。

没有引入 `neuraloperator` 依赖（其附带 wandb / tensorly / configmypy 等重量级包），
上述结构是按官方源码的最小忠实复现。
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from point_cloud import (FLOW_FEATURES, apply_modulation, flow_cloud, hull_cloud,
                         normalize_within, token_modulation)


def radius_neighbors(data, queries, radius):
    """半径邻域搜索：返回展平邻居下标与每个查询的邻居数（空邻域退化为最近邻）。"""
    tree = cKDTree(np.ascontiguousarray(data, dtype=np.float64))
    groups = tree.query_ball_point(np.ascontiguousarray(queries, dtype=np.float64), radius,
                                   return_sorted=True)
    counts = np.fromiter((len(group) for group in groups), dtype=np.int64, count=len(groups))
    empty = np.nonzero(counts == 0)[0]
    if len(empty):
        nearest = tree.query(np.ascontiguousarray(queries[empty], dtype=np.float64), k=1)[1]
        for query, point in zip(empty, np.atleast_1d(nearest)):
            groups[query] = [int(point)]
        counts[empty] = 1
    index = np.fromiter((point for group in groups for point in group), dtype=np.int64)
    return index, counts


class SinusoidalEmbedding(nn.Module):
    """`transformer` 风格的正弦位置编码，输出 `2 · num_frequencies · in_channels` 维。"""

    def __init__(self, in_channels, num_frequencies, max_positions=10000):
        super().__init__()
        frequency = torch.arange(num_frequencies) / in_channels
        self.register_buffer("frequencies", (1.0 / max_positions) ** frequency)
        self.in_channels = in_channels
        self.out_channels = 2 * num_frequencies * in_channels

    def forward(self, coords):
        args = coords.unsqueeze(-1) * self.frequencies
        return torch.stack([args.sin(), args.cos()], dim=-1).flatten(-3)


class ChannelMLP(nn.Module):
    """逐点 MLP（等价于 1×1 卷积），与 `neuralop.layers.channel_mlp.ChannelMLP` 同构。"""

    def __init__(self, in_channels, hidden_channels, out_channels=None, n_layers=2,
                 dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        dims = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, features):
        for i, layer in enumerate(self.layers):
            features = layer(features)
            if i < len(self.layers) - 1:
                features = self.dropout(F.gelu(features))
        return features


class GNOBlock(nn.Module):
    """GNO 层：坐标正弦编码上的半径核积分（`linear` 变换，按邻居数取平均）。"""

    def __init__(self, in_channels, out_channels, coord_dim=3, radius=0.02,
                 pos_embed_channels=32, kernel_hidden=64, chunk=4096, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.chunk = chunk
        self.pos_embedding = SinusoidalEmbedding(coord_dim, pos_embed_channels)
        self.kernel = ChannelMLP(2 * self.pos_embedding.out_channels, kernel_hidden,
                                 out_channels, n_layers=2, dropout=dropout)

    def forward(self, y, x, f_y):
        """`y (N,3)` 积分点、`x (M,3)` 输出点、`f_y (B,N,C)` → `(B,M,out)`。"""
        embedded_y = self.pos_embedding(y)
        tree_data = y.detach().cpu()
        outputs = []
        for start in range(0, len(x), self.chunk):
            part = x[start:start + self.chunk]
            index, counts = radius_neighbors(tree_data, part.detach().cpu(), self.radius)
            index = torch.from_numpy(index).to(y.device)
            counts = torch.from_numpy(counts).to(y.device)
            query_index = torch.repeat_interleave(
                torch.arange(len(part), device=y.device), counts)
            kernel = self.kernel(torch.cat([embedded_y[index],
                                            self.pos_embedding(part)[query_index]], dim=-1))
            summed = f_y.new_zeros((f_y.shape[0], len(part), self.out_channels))
            summed.index_add_(1, query_index, kernel[None] * f_y[:, index])
            outputs.append(summed / counts[:, None])
        return torch.cat(outputs, dim=1)


class SpectralConv3d(nn.Module):
    """三维谱卷积：rfftn 截断模态后做通道线性收缩，再 irfftn 回原分辨率。"""

    def __init__(self, channels, n_modes, spatial_shape):
        super().__init__()
        self.n_modes = list(n_modes)
        self.spatial_shape = tuple(spatial_shape)
        for i, (mode, size) in enumerate(zip(self.n_modes, self.spatial_shape)):
            limit = size // 2 + 1 if i == len(self.n_modes) - 1 else size
            if not 0 < mode <= limit:
                raise ValueError(f"n_modes={n_modes} 超出潜网格 {spatial_shape} 的可用模态")
        self.weight = nn.Parameter(
            torch.randn(channels, channels, *self.n_modes, dtype=torch.cfloat))
        scale = 1.0 / (channels * math.prod(self.n_modes))
        with torch.no_grad():
            self.weight.real.uniform_(-scale, scale)
            self.weight.imag.uniform_(-scale, scale)

    def forward(self, features):
        batch, _, *shape = features.shape
        modes = self.n_modes
        fft = torch.fft.rfftn(features, dim=(-3, -2, -1), norm="forward")
        truncated = fft[..., :modes[0], :modes[1], :modes[2]]
        out_fft = torch.zeros((batch, self.weight.shape[1], *fft.shape[2:]),
                              dtype=torch.cfloat, device=features.device)
        out_fft[..., :modes[0], :modes[1], :modes[2]] = torch.einsum(
            "bixyz,ioxyz->boxyz", truncated, self.weight)
        return torch.fft.irfftn(out_fft, s=shape, dim=(-3, -2, -1), norm="forward")


class FNOBlock(nn.Module):
    """FNO block：块入口 FiLM → 谱卷积 + 线性跳连 → 通道 MLP（soft-gating 跳连）。"""

    def __init__(self, channels, n_modes, spatial_shape, expansion=0.5, condition_dim=2,
                 conditioning="film", dropout=0.0, last=False):
        super().__init__()
        self.last = last
        self.modulation = token_modulation(conditioning, condition_dim, channels)
        self.conv = SpectralConv3d(channels, n_modes, spatial_shape)
        self.skip = nn.Conv3d(channels, channels, 1)
        self.channel_mlp = ChannelMLP(channels, max(1, round(channels * expansion)), channels,
                                      dropout=dropout)
        self.channel_skip = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def _modulate(self, features, condition):
        """把 (B,C,D,H,W) 展平成 token 序列后做块入口 FiLM，再还原。"""
        batch, channels, *shape = features.shape
        tokens = features.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
        tokens = apply_modulation(tokens, self.modulation, condition)
        return tokens.reshape(batch, *shape, channels).permute(0, 4, 1, 2, 3)

    def forward(self, features, condition):
        features = self._modulate(features, condition)
        spectral = self.conv(features) + self.skip(features)
        features = spectral if self.last else F.gelu(spectral)
        tokens = features.permute(0, 2, 3, 4, 1).reshape(features.shape[0], -1,
                                                         features.shape[1])
        tokens = self.channel_mlp(tokens).reshape(features.shape[0], *features.shape[2:],
                                                  features.shape[1])
        features = tokens.permute(0, 4, 1, 2, 3) + features * self.channel_skip
        return features if self.last else F.gelu(features)


class GINOBaseline(nn.Module):
    supports_query_decoder = True

    def __init__(self, latent_shape=(24, 12, 6), n_modes=(12, 6, 3), hidden_channels=64,
                 fno_layers=3, embed_channels=32, gno_embed_channels=24,
                 gno_kernel_hidden=64, in_radius=0.02, out_radius=0.06, gno_chunk=4096,
                 lifting_channels=128, projection_channels=128, expansion=0.5,
                 condition_dim=2, conditioning="film", dropout=0.0, flow_channels=5,
                 hull_channels=1, input_domains=("flow", "hull"),
                 output_domains=("flow",)):
        super().__init__()
        self.input_domains = tuple(input_domains)
        self.output_domains = tuple(output_domains)
        if "flow" not in self.input_domains:
            raise ValueError("GINO 的输入几何必须包含自由液面点云")
        self.latent_shape = tuple(int(size) for size in latent_shape)
        self.hidden_channels = hidden_channels
        self.flow_input = nn.Linear(len(FLOW_FEATURES), embed_channels)
        self.hull_input = nn.Linear(8, embed_channels) if "hull" in self.input_domains else None
        self.domain = nn.Embedding(2, embed_channels)
        # 线性核积分的核输出通道必须与输入特征通道一致（官方 IntegralTransform）
        self.gno_in = GNOBlock(embed_channels, embed_channels, radius=in_radius,
                               pos_embed_channels=gno_embed_channels,
                               kernel_hidden=gno_kernel_hidden, chunk=gno_chunk,
                               dropout=dropout)
        self.lifting = ChannelMLP(embed_channels, lifting_channels, hidden_channels, n_layers=3,
                                  dropout=dropout)

        grid = torch.stack(torch.meshgrid(
            *[torch.linspace(0.0, 1.0, size) for size in self.latent_shape], indexing="ij"),
            dim=-1).reshape(-1, 3)
        self.register_buffer("latent_coords", grid)
        self.fno_blocks = nn.ModuleList([
            FNOBlock(hidden_channels, n_modes, self.latent_shape, expansion, condition_dim,
                     conditioning, dropout, last=index == fno_layers - 1)
            for index in range(fno_layers)])
        self.gno_out = GNOBlock(hidden_channels, hidden_channels, radius=out_radius,
                                pos_embed_channels=gno_embed_channels,
                                kernel_hidden=gno_kernel_hidden, chunk=gno_chunk,
                                dropout=dropout)
        self.flow_head = (ChannelMLP(hidden_channels, projection_channels, flow_channels,
                                     n_layers=2, dropout=dropout)
                          if "flow" in self.output_domains else None)
        self.hull_head = (ChannelMLP(hidden_channels, projection_channels, hull_channels,
                                     n_layers=2, dropout=dropout)
                          if "hull" in self.output_domains else None)

    def _cloud(self, batch):
        """输入点云的三维坐标与逐点特征：自由液面位于 z=0，船面保留三维形状。"""
        flow = flow_cloud(batch)[0]
        flow_geometry = torch.cat([flow[:, :2], flow.new_zeros(len(flow), 1)], dim=-1)
        features = self.flow_input(flow) + self.domain.weight[0]
        geometry = [flow_geometry]
        if self.hull_input is not None:
            hull = hull_cloud(batch)[0]
            geometry.append(hull[:, 2:5])
            features = torch.cat([features, self.hull_input(hull) + self.domain.weight[1]])
        return torch.cat(geometry)[None], features[None]

    def _frame(self, geometry):
        """输入几何包围盒：把几何与查询坐标一起映射到 [0,1]³。"""
        low = geometry.amin(dim=0)
        span = (geometry.amax(dim=0) - low).clamp_min(1e-6)
        return low, span

    def encode(self, batch):
        """输入几何 → 潜网格特征（与查询点集无关）。"""
        geometry, features = self._cloud(batch)
        low, span = self._frame(geometry[0])
        normalised = normalize_within(geometry, low, span)
        latent = self.gno_in(normalised[0], self.latent_coords, features)
        latent = self.lifting(latent).view(1, *self.latent_shape, self.hidden_channels)
        latent = latent.permute(0, 4, 1, 2, 3)
        for block in self.fno_blocks:
            latent = block(latent, batch["condition"])
        return {"latent": latent, "low": low, "span": span, "condition": batch["condition"]}

    def query(self, state, points):
        """任意查询点的 GNO 输出隐特征；`points (1,Q,3)`。"""
        normalised = normalize_within(points, state["low"], state["span"])
        latent = state["latent"].permute(0, 2, 3, 4, 1).reshape(points.shape[0], -1,
                                                               self.hidden_channels)
        return self.gno_out(self.latent_coords, normalised[0], latent)

    def decode(self, state, query_coords):
        """在任意自由液面查询坐标 `(1,Q,2)` 处解码五个流场通道。"""
        points = torch.cat([query_coords[..., :2],
                            query_coords.new_zeros(*query_coords.shape[:-1], 1)], dim=-1)
        return self.flow_head(self.query(state, points))

    def forward(self, batch, flow_queries):
        """流场在任意查询点上取值，船面取整条湿表面；批大小固定为 1。"""
        state = self.encode(batch)
        output = {}
        if self.flow_head is not None:
            output["flow"] = self.decode(state, flow_queries)
        if self.hull_head is not None:
            hull = hull_cloud(batch)[0]
            output["hull"] = self.hull_head(self.query(state, hull[:, 2:5][None]))
        return output
