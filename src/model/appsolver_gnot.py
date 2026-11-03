"""APP 分词 + GNOT 异构注意力主干与坐标查询解码。

GNOT 基线在原始点云上交替做几何分支与物理分支的线性注意力，MOE-MLP 由点坐标门控，
预测点集与输入点集相同。这里把输入换成 APP patch token（patch 内求积加权平均，位置取
形心），`GNOTBlock` 的几何-物理交替、线性注意力、位置门控 MOE 与 `placeholder` 参数
沿用 `gnot.py` 的实现，接收方主干宽度与层数取基线配置，门控坐标用 patch 形心。

主干之后接 APP 的坐标查询解码器：查询点坐标经 `QueryBasis` 嵌入，与 patch token 做
带求积权重（patch 面积）的交叉注意力，查询之间不做自注意力，输出经 GNOT 的
`d_model → 2·d_model` 前馈头给出五个流场通道。
"""

import torch
import torch.nn as nn

from gnot import GNOTBlock
from patch_frontend import PatchFrontend
from point_cloud import ConditionBias, hull_cloud
from query_decoder import QueryBasis, QueryDecoder


class APPSolverGNOT(nn.Module):
    supports_query_decoder = True

    def __init__(self, flow_size_tiers, hull_size, d_model=108, nhead=4, layers=2, mlp_ratio=3,
                 n_experts=3, dropout=0.0, condition_dim=2, flow_channels=5, hull_channels=1,
                 max_level=12, fourier_features=16, fourier_scale=4.0, query_basis="siren",
                 decoder_layers=2, input_domains=None, output_domains=None, conditioning=None):
        super().__init__()
        self.input_domains = tuple(input_domains) if input_domains else ("flow", "hull")
        self.output_domains = tuple(output_domains) if output_domains else ("flow", "hull")
        self.frontend = PatchFrontend(flow_size_tiers, hull_size, d_model, fourier_features,
                                      fourier_scale, max_level, self.input_domains)
        self.condition = ConditionBias(condition_dim, d_model) if conditioning is None else None
        self.blocks = nn.ModuleList([
            GNOTBlock(d_model, nhead, mlp_ratio, n_experts, dropout, space_dim=2,
                      conditioning=conditioning, condition_dim=condition_dim)
            for _ in range(layers)])
        self.placeholder = nn.Parameter(torch.rand(d_model) / d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU())
        self.basis = QueryBasis(2, d_model, query_basis, fourier_scale)
        self.hull_extra = (nn.Linear(6, d_model) if "hull" in self.output_domains else None)
        self.decoder = QueryDecoder(d_model, nhead, decoder_layers, mlp_ratio, dropout, False,
                                    conditioning, condition_dim)
        self.flow_head = (nn.Linear(2 * d_model, flow_channels)
                          if "flow" in self.output_domains else None)
        self.hull_head = (nn.Linear(2 * d_model, hull_channels)
                          if "hull" in self.output_domains else None)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def encode(self, batch):
        """patch token 序列 → 异构注意力主干编码（与查询点集无关）。"""
        tokens, weight, centres = self.frontend(batch)
        condition = batch["condition"]
        bias = self.condition(condition) if self.condition is not None else 0.0
        geometry = tokens + self.placeholder
        physics = geometry + bias
        for block in self.blocks:
            physics = block(geometry, physics, centres, condition)
        return {"tokens": physics, "weight": weight, "centres": centres,
                "condition": condition}

    def _features(self, state, queries, extra=None):
        embedding = self.basis(queries[..., :2])
        if extra is not None:
            embedding = embedding + extra(queries[..., 2:])
        return self.head(self.decoder(embedding, state["tokens"], state["weight"],
                                      queries[..., :2], state["centres"], state["condition"]))

    def decode(self, state, query_x):
        """在任意查询坐标处解码五个流场通道。"""
        return self.flow_head(self._features(state, query_x))

    def forward(self, batch, flow_queries):
        """批大小固定为 1；流场在任意查询点上取值，船面取整条湿表面。"""
        state = self.encode(batch)
        output = {}
        if self.flow_head is not None:
            output["flow"] = self.decode(state, flow_queries[..., :2])
        if self.hull_head is not None:
            features = hull_cloud(batch)
            output["hull"] = self.hull_head(
                self._features(state, features, self.hull_extra))
        return output
