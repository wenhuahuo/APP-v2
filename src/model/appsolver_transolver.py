"""APP 分词 + Transolver 物理切片主干与坐标查询解码。

Transolver 基线在原始点云上把 N 个点软分配到固定数量的 slice、在 slice 之间做注意力，
再按同一权重投影回点。这里把输入换成 APP patch token（patch 内求积加权平均，位置取
形心），`PhysicsAttention` 的 slice 主干（slice 数、温度、逐层 MLP、逐层 FiLM）与基线
逐项一致，接收方主干宽度与层数取基线配置，差别只有序列长度从逐点变成 patch。

切片主干之后接 APP 的坐标查询解码器：查询点坐标经 `QueryBasis` 嵌入，与 patch token
做带求积权重（patch 面积）的交叉注意力，查询之间不做自注意力，因此换查询密度不改变
结构。`encode` / `decode` 分开暴露，评测可固定输入编码、只改查询密度。
"""

import torch.nn as nn

from patch_frontend import PatchFrontend
from point_cloud import ConditionBias, hull_cloud
from query_decoder import QueryBasis, QueryDecoder
from transolver import TransolverBlock


class APPSolverTransolver(nn.Module):
    supports_query_decoder = True

    def __init__(self, flow_size_tiers, hull_size, d_model=208, nhead=8, layers=3, slices=32,
                 mlp_ratio=3, dropout=0.0, condition_dim=2, flow_channels=5, hull_channels=1,
                 max_level=12, fourier_features=16, fourier_scale=4.0, query_basis="siren",
                 decoder_layers=2, input_domains=None, output_domains=None, conditioning=None):
        super().__init__()
        self.input_domains = tuple(input_domains) if input_domains else ("flow", "hull")
        self.output_domains = tuple(output_domains) if output_domains else ("flow", "hull")
        self.frontend = PatchFrontend(flow_size_tiers, hull_size, d_model, fourier_features,
                                      fourier_scale, max_level, self.input_domains)
        self.condition = ConditionBias(condition_dim, d_model) if conditioning is None else None
        self.blocks = nn.ModuleList([
            TransolverBlock(d_model, nhead, slices, mlp_ratio, dropout, conditioning,
                            condition_dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.basis = QueryBasis(2, d_model, query_basis, fourier_scale)
        self.hull_extra = (nn.Linear(6, d_model) if "hull" in self.output_domains else None)
        self.decoder = QueryDecoder(d_model, nhead, decoder_layers, mlp_ratio, dropout, False,
                                    conditioning, condition_dim)
        self.flow_head = (nn.Linear(d_model, flow_channels)
                          if "flow" in self.output_domains else None)
        self.hull_head = (nn.Linear(d_model, hull_channels)
                          if "hull" in self.output_domains else None)

    def _tokens(self, batch):
        tokens, weight, centres = self.frontend(batch)
        bias = self.condition(batch["condition"]) if self.condition is not None else 0.0
        tokens = tokens + bias
        for block in self.blocks:
            tokens = block(tokens, batch["condition"])
        return {"tokens": self.norm(tokens), "weight": weight, "centres": centres,
                "condition": batch["condition"]}

    def encode(self, batch):
        """patch token 序列 → 切片主干编码（与查询点集无关）。"""
        return self._tokens(batch)

    def _features(self, state, queries, extra=None):
        embedding = self.basis(queries[..., :2])
        if extra is not None:
            embedding = embedding + extra(queries[..., 2:])
        return self.decoder(embedding, state["tokens"], state["weight"], queries[..., :2],
                            state["centres"], state["condition"])

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
