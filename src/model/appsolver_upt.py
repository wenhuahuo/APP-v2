"""APP 分词 + UPT perceiver 编码与原生任意坐标查询解码。

UPT 基线把整片输入点云的坐标正弦编码过 MLP 后压到固定数量的 latent，再按查询坐标
交叉注意力到 latent 上解码。这里把编码器的输入换成 APP patch token（patch 内求积加权
平均，位置取形心），perceiver 编码器（latent 交叉注意力 + latent 自注意力 + 逐层 FiLM）
与解码器（查询坐标正弦编码 + MLP，逐查询交叉注意力到 latent）沿用 `upt.py` 与
`point_cloud.py` 的实现，latent token 数量与接收方主干宽度取基线配置，未被替换。

`encode` / `decode` 分开暴露：编码只取决于 patch token 与工况，解码只取决于查询坐标，
因此可以在固定输入密度上编码一次、在任意密度与位置的查询点集上解码。
"""

import torch.nn as nn

from patch_frontend import PatchFrontend
from point_cloud import ConditionBias, PerceiverEncoder, hull_cloud, sincos_embedding
from upt import QueryCrossBlock


class APPSolverUPT(nn.Module):
    supports_query_decoder = True

    def __init__(self, flow_size_tiers, hull_size, d_model=168, nhead=8, layers=2, latents=128,
                 decoder_layers=1, mlp_ratio=4, dropout=0.0, condition_dim=2, flow_channels=5,
                 hull_channels=1, max_level=12, fourier_features=16, fourier_scale=4.0,
                 input_domains=None, output_domains=None, conditioning=None):
        super().__init__()
        self.d_model = d_model
        self.input_domains = tuple(input_domains) if input_domains else ("flow", "hull")
        self.output_domains = tuple(output_domains) if output_domains else ("flow", "hull")
        self.frontend = PatchFrontend(flow_size_tiers, hull_size, d_model, fourier_features,
                                      fourier_scale, max_level, self.input_domains)
        self.condition = ConditionBias(condition_dim, d_model) if conditioning is None else None
        self.encoder = PerceiverEncoder(d_model, nhead, layers, latents, mlp_ratio, dropout,
                                        conditioning, condition_dim)
        self.query_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        self.decoder = nn.ModuleList([
            QueryCrossBlock(d_model, nhead, mlp_ratio, dropout, conditioning, condition_dim)
            for _ in range(decoder_layers)])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.query_input = nn.Linear(8, d_model) if "hull" in self.output_domains else None
        self.flow_head = (nn.Linear(d_model, flow_channels)
                          if "flow" in self.output_domains else None)
        self.hull_head = (nn.Linear(d_model, hull_channels)
                          if "hull" in self.output_domains else None)

    def encode(self, batch):
        """patch token 序列 → 固定数量 latent（与查询点集无关）。"""
        tokens, _, _ = self.frontend(batch)
        condition = batch["condition"]
        if self.condition is not None:
            tokens = tokens + self.condition(condition)
        return {"latent": self.encoder(tokens, condition), "condition": condition}

    def _features(self, state, query_x, query_fx=None):
        queries = self.query_mlp(sincos_embedding(query_x[..., :2], self.d_model))
        if query_fx is not None:
            queries = queries + self.query_input(query_fx)
        for block in self.decoder:
            queries = block(queries, state["latent"], state["condition"])
        return self.decoder_norm(queries)

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
                self._features(state, features, features))
        return output
