"""APPSolver-v2：自适应 Patch 分词、求积权重注意力与坐标查询解码。

默认输入自由液面坐标和船体表面编码，仅预测五个自由液面流场通道；船体压力输出
可通过 `output_domains` 显式启用。流场分词支持自适应硬分块、均匀硬分块和学习软分块。

工况参数（Fr、log10 Re）默认通过逐层 FiLM 融合。`conditioning` 同时控制编码器与
查询解码器，可选 `none` / `film` / `adaln` / `adaln_zero` / `fourier_add` /
`cond_token` / `cond_xattn`，用于工况融合消融。
"""

import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from learned_tokenizer import LearnedTokenizer
from positional import FourierFeatures
from query_decoder import QueryBasis, QueryDecoder
from tokenizer import flow_tokens, hull_tokens

FLOW_FEATURES = ("x", "y")
HULL_FEATURES = ("u", "v", "x", "y", "z", "nx", "ny", "nz")


class APPSolverV2(nn.Module):
    def __init__(self, flow_size_tiers, hull_size, d_model=64, nhead=4,
                 encoder_layers=2, decoder_layers=2, mlp_ratio=4.0, dropout=0.0,
                 positional="fourier", query_basis="siren", fourier_features=16,
                 fourier_scale=4.0, condition_dim=2, conditioning="film",
                 flow_channels=5, hull_channels=1, max_level=12,
                 input_domains=("flow", "hull"), output_domains=("flow",),
                 flow_tokenizer="adaptive", flow_uniform_size=0.1, flow_learned_tokens=512):
        super().__init__()
        self.input_domains = tuple(input_domains)
        self.output_domains = tuple(output_domains)
        for domains in (self.input_domains, self.output_domains):
            if not domains or set(domains) - {"flow", "hull"}:
                raise ValueError(f"未知或空的域配置: {domains}")
        self.flow_size_tiers = flow_size_tiers
        self.hull_size = hull_size
        self.max_level = max_level
        self.positional = positional
        self.flow_tokenizer = flow_tokenizer
        self.flow_uniform_size = flow_uniform_size
        if flow_tokenizer not in {"adaptive", "uniform", "learned"}:
            raise ValueError(f"未知流场分词方法: {flow_tokenizer}")

        self.flow_input = (nn.Linear(len(FLOW_FEATURES), d_model)
                           if "flow" in self.input_domains else None)
        self.hull_input = (nn.Linear(len(HULL_FEATURES), d_model)
                           if "hull" in self.input_domains else None)
        self.learned_flow = (LearnedTokenizer(len(FLOW_FEATURES), d_model, flow_learned_tokens)
                             if "flow" in self.input_domains and flow_tokenizer == "learned"
                             else None)
        self.domain = nn.Embedding(2, d_model)
        self.condition_token = (nn.Linear(condition_dim, d_model)
                                if conditioning == "cond_token" else None)
        self.condition_domain = (nn.Parameter(torch.zeros(d_model))
                                 if conditioning == "cond_token" else None)
        self.position = (nn.Linear(2 * fourier_features, d_model)
                         if positional == "fourier" else None)
        self.fourier = (FourierFeatures(2, fourier_features, fourier_scale)
                        if positional == "fourier" else None)

        rope = positional == "rope"
        self.conditioning = conditioning
        self.encoder = Encoder(d_model, nhead, encoder_layers, mlp_ratio, dropout, rope,
                               conditioning, condition_dim)
        self.basis = QueryBasis(2, d_model, query_basis, fourier_scale)
        self.hull_extra = (nn.Linear(len(HULL_FEATURES) - 2, d_model)
                           if "hull" in self.output_domains else None)
        self.decoder = QueryDecoder(d_model, nhead, decoder_layers, mlp_ratio, dropout, rope,
                                    conditioning, condition_dim)
        self.flow_head = (nn.Linear(d_model, flow_channels)
                          if "flow" in self.output_domains else None)
        self.hull_head = (nn.Linear(d_model, hull_channels)
                          if "hull" in self.output_domains else None)

    def forward(self, batch, flow_queries=None):
        """flow_queries (B,Q,2) 为 [x, y]；仅返回启用的输出域。"""
        condition = batch["condition"]

        tokens, weight, coords = self._pad_tokens(
            [self._tokenize(batch, i) for i in range(len(condition))])
        if self.condition_token is not None:
            tokens, weight, coords = self._add_condition_token(
                tokens, weight, coords, condition)
        memory = self.encoder(tokens, weight, coords, condition)

        output = {}
        if "flow" in self.output_domains:
            output["flow"] = self.flow_head(self._decode(
                flow_queries, None, memory, weight, coords, condition))
        if "hull" in self.output_domains:
            output["hull"] = self.hull_head(self._decode(
                self._hull_queries(batch), self.hull_extra, memory, weight, coords, condition))
        return output

    def _decode(self, queries, extra, memory, weight, coords, condition):
        embedding = self.basis(queries[..., :2])
        if extra is not None:
            embedding = embedding + extra(queries[..., 2:])
        return self.decoder(embedding, memory, weight, queries[..., :2], coords, condition)

    def _position(self, centres):
        if self.position is None:
            return 0.0
        return self.position(self.fourier(centres))

    def _domain_token(self, patches, projection, domain_id):
        tokens, areas, centres = patches
        device = projection.weight.device
        tokens = projection(tokens.to(device))
        centres = centres.to(device)
        marker = torch.full((len(tokens),), domain_id, dtype=torch.long, device=device)
        return (tokens + self.domain(marker) + self._position(centres),
                areas.to(device), centres)

    def _tokenize(self, batch, i):
        """硬分词在 CPU 上完成；学习分词在参数所在设备上进行软分配。"""
        patches = []
        if "flow" in self.input_domains:
            if self.flow_tokenizer == "learned":
                flow = self._learned_flow_tokens(batch, i)
            else:
                tiers = (self.flow_size_tiers if self.flow_tokenizer == "adaptive"
                         else [[None, self.flow_uniform_size]])
                flow = flow_tokens(batch, i, tiers, self.max_level)
            patches.append(self._domain_token(flow, self.flow_input, 0))
        if "hull" in self.input_domains:
            patches.append(self._domain_token(
                hull_tokens(batch, i, self.hull_size, self.max_level), self.hull_input, 1))
        if len(patches) == 1:
            return patches[0]
        return tuple(torch.cat([patch[k] for patch in patches]) for k in range(3))

    def _learned_flow_tokens(self, batch, i):
        device = self.learned_flow.token_embed.device
        mask = batch["flow_mask"][i].to(device).reshape(-1)
        features = batch["flow_coords"][i].to(device).reshape(-1, 2)[mask]
        weights = batch["flow_weight"][i].to(device).reshape(-1)[mask]
        return self.learned_flow(features, weights)

    def _add_condition_token(self, tokens, weight, coords, condition):
        token = (self.condition_token(condition) + self.condition_domain)[:, None]
        leading_weight = weight.new_ones((len(weight), 1))
        leading_coords = coords.new_zeros((len(coords), 1, 2))
        return (torch.cat([token, tokens], dim=1),
                torch.cat([leading_weight, weight], dim=1),
                torch.cat([leading_coords, coords], dim=1))

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
