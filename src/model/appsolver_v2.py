"""APPSolver-v2：APP 分词 + 求积权重注意力主干 + 查询解码器，两个输出域。

流程（每个样本）：

1. 流场域与船体域各自按物理尺寸划 patch，token = patch 内 Δ 加权平均；
   流场域 patch 的容差按到船体水线的距离分档（薄 patch 只出现在船体附近）。
2. 两个域的 token 拼成一条序列，加域标记与 patch 形心的位置编码，过一个 transformer。
3. 查询点（自由液面任意坐标 / 船体湿表面任意 UV）经坐标基与交叉注意力取回该处取值，
   分别过流场头与船面压力头。

工况参数（Fr、log10 Re）通过逐层 FiLM 注入，不占 token 位。
"""

import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder, FiLM
from patch_integrate import integrate
from positional import FourierFeatures
from quadtree import build_partition, tiered_tolerance
from query_decoder import QueryBasis, QueryDecoder

FLOW_FEATURES = ("x", "y", "sdf")
HULL_FEATURES = ("u", "v", "x", "y", "z", "nx", "ny", "nz")
HULL_EXTENT = (0.0, 0.0, 1.0, 1.0)


class APPSolverV2(nn.Module):
    def __init__(self, flow_size_tiers, hull_size, d_model=64, nhead=4,
                 encoder_layers=2, decoder_layers=2, mlp_ratio=4.0, dropout=0.0,
                 positional="fourier", query_basis="siren", fourier_features=16,
                 fourier_scale=4.0, condition_dim=2, flow_channels=5, hull_channels=1,
                 max_level=12):
        super().__init__()
        self.flow_size_tiers = flow_size_tiers
        self.hull_size = hull_size
        self.max_level = max_level
        self.positional = positional

        self.flow_input = nn.Linear(len(FLOW_FEATURES), d_model)
        self.hull_input = nn.Linear(len(HULL_FEATURES), d_model)
        self.domain = nn.Embedding(2, d_model)
        self.position = (nn.Linear(2 * fourier_features, d_model)
                         if positional == "fourier" else None)
        self.fourier = (FourierFeatures(2, fourier_features, fourier_scale)
                        if positional == "fourier" else None)

        rope = positional == "rope"
        self.encoder = Encoder(d_model, nhead, encoder_layers, mlp_ratio, dropout, rope)
        self.basis = QueryBasis(2, d_model, query_basis, fourier_scale)
        self.flow_extra = nn.Linear(len(FLOW_FEATURES) - 2, d_model)
        self.hull_extra = nn.Linear(len(HULL_FEATURES) - 2, d_model)
        self.decoder = QueryDecoder(d_model, nhead, decoder_layers, mlp_ratio, dropout, rope)
        self.flow_head = nn.Linear(d_model, flow_channels)
        self.hull_head = nn.Linear(d_model, hull_channels)
        self.film = nn.ModuleList([
            FiLM(condition_dim, d_model) for _ in range(encoder_layers + decoder_layers)
        ])

    def forward(self, batch, flow_queries):
        """batch 为 `collate_samples` 的输出；flow_queries (B,Q,3) 为 [x, y, sdf]。"""
        condition = batch["condition"]
        modulations = [film(condition) for film in self.film]
        split = len(self.encoder.blocks)

        tokens, weight, coords = self._pad_tokens(
            [self._tokenize(batch, i) for i in range(len(condition))])
        memory = self.encoder(tokens, weight, coords, modulations[:split])

        flow = self.flow_head(self._decode(flow_queries, self.flow_extra, memory, weight,
                                          coords, modulations[split:]))
        hull = self.hull_head(self._decode(self._hull_queries(batch), self.hull_extra, memory,
                                          weight, coords, modulations[split:]))
        return {"flow": flow, "hull": hull}

    def _decode(self, queries, extra, memory, weight, coords, modulations):
        embedding = self.basis(queries[..., :2]) + extra(queries[..., 2:])
        return self.decoder(embedding, memory, weight, queries[..., :2], coords, modulations)

    def _position(self, centres):
        if self.position is None:
            return 0.0
        return self.position(self.fourier(centres))

    def _domain_token(self, features, weights, extent, tolerance, projection, domain_id, scale):
        partition = build_partition(features[:, :2].numpy(), weights.numpy(), extent, tolerance,
                                    scale=scale, max_level=self.max_level)
        order = torch.from_numpy(partition.order)
        starts = torch.from_numpy(partition.starts)
        tokens, _, centres = integrate(features[order], weights[order], starts)
        device = projection.weight.device
        tokens = projection(tokens.to(device))
        centres = centres.to(device)
        marker = torch.full((len(tokens),), domain_id, dtype=torch.long, device=device)
        return (tokens + self.domain(marker) + self._position(centres),
                torch.from_numpy(partition.areas).to(device), centres)

    def _tokenize(self, batch, i):
        """分词在 CPU 上用 numpy 做（分区与积分），token 再搬到参数所在设备。"""
        flow_mask = batch["flow_mask"][i].cpu()
        flow_coords = batch["flow_coords"][i].cpu()[flow_mask]
        flow_weights = batch["flow_weight"][i].cpu()[flow_mask]
        flow_sdf = batch["flow_sdf"][i].cpu()[flow_mask]
        flow_features = torch.column_stack([flow_coords, flow_sdf])
        flow = self._domain_token(
            flow_features, flow_weights, tuple(batch["flow_extent"][i].tolist()),
            tiered_tolerance(flow_sdf.numpy(), self.flow_size_tiers),
            self.flow_input, 0, 1.0)

        hull_mask = batch["hull_mask"][i].cpu()
        hull_weights = batch["hull_weight"][i].cpu()[hull_mask]
        hull_features = torch.column_stack([
            batch["hull_uv"][i].cpu()[hull_mask], batch["hull_points"][i].cpu()[hull_mask],
            batch["hull_normals"][i].cpu()[hull_mask],
        ])
        hull = self._domain_token(
            hull_features, hull_weights, HULL_EXTENT,
            np.full(len(hull_weights), self.hull_size), self.hull_input, 1, None)
        return tuple(torch.cat([flow[k], hull[k]]) for k in range(3))

    @staticmethod
    def _pad_tokens(per_sample):
        batch = len(per_sample)
        length = max(len(tokens) for tokens, _, _ in per_sample)
        dim = per_sample[0][0].shape[-1]
        tokens = per_sample[0][0].new_zeros((batch, length, dim))
        weight = per_sample[0][1].new_zeros((batch, length))
        coords = per_sample[0][2].new_zeros((batch, length, 2))
        for i, (sample_tokens, sample_weight, sample_coords) in enumerate(per_sample):
            count = len(sample_tokens)
            tokens[i, :count] = sample_tokens
            weight[i, :count] = sample_weight
            coords[i, :count] = sample_coords
        return tokens, weight, coords

    @staticmethod
    def _hull_queries(batch):
        return torch.cat([batch["hull_uv"], batch["hull_points"], batch["hull_normals"]], dim=-1)
