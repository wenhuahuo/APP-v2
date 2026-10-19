"""APP 分词 + OFormer 主干与查询解码器：分词器即插即用的证据。

OFormer 基线保留每个输入点做线性注意力；这里只把输入换成 APP patch token
（token = patch 内 Δ 加权平均，位置取 patch 形心），编码器、解码器、propagator
与 OFormer 基线完全一致。分词不引入参数，因此两者参数量相同、训练超参相同，
差别只有「逐点」与「APP 分词」这一项。
"""

import torch
import torch.nn as nn

from oformer import DecoderBlock, EncoderBlock, GaussianFourierFeatures, Propagator
from point_cloud import ConditionBias
from tokenizer import flow_tokens, hull_tokens


class APPSolverOFormer(nn.Module):
    def __init__(self, flow_size_tiers, hull_size, d_model=128, nhead=8, encoder_layers=2,
                 decoder_layers=2, mlp_ratio=4.0, propagator_layers=3, fourier_scale=8.0,
                 rotary_min_freq=0.25, rotary_scale=16.0, dropout=0.0, condition_dim=2,
                 flow_channels=5, hull_channels=1, max_level=12):
        super().__init__()
        self.flow_size_tiers = flow_size_tiers
        self.hull_size = hull_size
        self.max_level = max_level
        self.flow_input = nn.Linear(3, d_model)
        self.hull_input = nn.Linear(8, d_model)
        self.condition = ConditionBias(condition_dim, d_model)
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, nhead, mlp_ratio, rotary_min_freq, rotary_scale, dropout)
            for _ in range(encoder_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.coordinates = nn.Sequential(
            GaussianFourierFeatures(2, d_model // 2, fourier_scale), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, nhead, mlp_ratio, rotary_min_freq, rotary_scale, dropout)
            for _ in range(decoder_layers)])
        self.propagator = nn.ModuleList([
            Propagator(d_model, mlp_ratio) for _ in range(propagator_layers)])
        self.flow_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                       nn.GELU(), nn.Linear(d_model // 2, flow_channels))
        self.hull_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                       nn.GELU(), nn.Linear(d_model // 2, hull_channels))

    def _points(self, patches, projection, bias, query_coords):
        """APP 分词 → 编码器（位置用 patch 形心）→ 在查询坐标处解码。"""
        tokens, _, centres = patches
        device = projection.weight.device
        latents = projection(tokens.to(device)) + bias
        coordinates = centres.to(device)[None]
        for block in self.encoder:
            latents = block(latents, coordinates)
        latents = self.norm(latents)

        queries = self.coordinates(query_coords)
        for block in self.decoder:
            queries = block(queries, latents, query_coords, coordinates)
        for layer in self.propagator:
            queries = layer(queries, query_coords)
        return queries

    def forward(self, batch, flow_queries):
        """批大小固定为 1；流场在任意查询点上取值，船面取整条湿表面。"""
        bias = self.condition(batch["condition"])
        flow_patches = flow_tokens(batch, 0, self.flow_size_tiers, self.max_level)
        flow = self.flow_head(self._points(flow_patches, self.flow_input, bias,
                                          flow_queries[..., :2]))
        hull_patches = hull_tokens(batch, 0, self.hull_size, self.max_level)
        hull_query = batch["hull_uv"][0][batch["hull_mask"][0]].to(flow.device)[None]
        hull = self.hull_head(self._points(hull_patches, self.hull_input, bias, hull_query))
        return {"flow": flow, "hull": hull}
