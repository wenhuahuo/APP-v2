"""APPSolver-v2：APP 分词 + 求积权重注意力主干 + 查询解码器，两个输出域。

流程（每个样本）：

1. 流场域与船体域各自按物理尺寸划 patch，token = patch 内 Δ 加权平均；
   流场域 patch 的容差按到船体水线的距离分档（薄 patch 只出现在船体附近）。
2. 两个域的 token 拼成一条序列，加域标记与 patch 形心的位置编码，过一个 transformer。
3. 查询点（自由液面任意坐标 / 船体湿表面任意 UV）经坐标基与交叉注意力取回该处取值，
   分别过流场头与船面压力头。

工况参数（Fr、log10 Re）按配置 `conditioning` 逐层注入，不占 token 位：
`none`（不使用条件编码）/ `film`（默认，block 入口 `x·(1+γ)+β`）/ `adaln` /
`adaln_zero`（DiT 式，见 `conditioning.py`）。编码器与查询解码器用同一种配置；
`none` 时两处都没有条件调制参数，工况值不参与前向计算。
"""

import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
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
                 input_domains=("flow", "hull"), output_domains=("flow", "hull")):
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

        self.flow_input = (nn.Linear(len(FLOW_FEATURES), d_model)
                           if "flow" in self.input_domains else None)
        self.hull_input = (nn.Linear(len(HULL_FEATURES), d_model)
                           if "hull" in self.input_domains else None)
        self.domain = nn.Embedding(2, d_model)
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
        """分词在 CPU 上用 numpy 做（分区与积分），token 再搬到参数所在设备。"""
        patches = []
        if "flow" in self.input_domains:
            patches.append(self._domain_token(
                flow_tokens(batch, i, self.flow_size_tiers, self.max_level), self.flow_input, 0))
        if "hull" in self.input_domains:
            patches.append(self._domain_token(
                hull_tokens(batch, i, self.hull_size, self.max_level), self.hull_input, 1))
        if len(patches) == 1:
            return patches[0]
        return tuple(torch.cat([patch[k] for patch in patches]) for k in range(3))

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
