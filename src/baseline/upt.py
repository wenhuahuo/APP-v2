"""UPT 基线：perceiver 编码到固定数量 latent，再按查询坐标解码到任意点。

参考 Universal Physics Transformers（NeurIPS 2024）的编码-传播-解码三段结构
（官方实现 `ml-jku/UPT` 的 `rans_perceiver` 编码器与 `RansPerceiver` 解码器）：

- 编码器把整片输入点云的坐标正弦编码过 MLP 后压到固定数量的 latent
  （latent 交叉注意力 + latent 自注意力）；
- 解码器为 `decoder(x, query_x, query_fx)`：查询点先做坐标正弦编码与 MLP，
  有逐点输入特征时再叠加该特征的投影，然后交叉注意力到 latent 上输出场值。

latent 数量固定，输入点数变化时表征容量不变；解码只依赖查询坐标，因此可以在固定
输入密度上编码一次、在任意密度与位置的查询点集上解码。`encode` / `decode` 分开
暴露，评测据此保证换查询密度时不重跑编码器。

工况默认走历史口径（加性偏置）；配置给出 `conditioning: film` 时编码器块与解码器块
按 APPSolver-v2 的公式做逐层 FiLM。给出 `input_domains: [flow, hull]` 时两个输入域
联合编码，否则保持历史口径的独立编码。
"""

import torch.nn as nn

from point_cloud import (CloudEmbedding, ConditionBias, PerceiverEncoder, apply_modulation,
                         flow_cloud, hull_cloud, sincos_embedding, token_modulation)


class QueryCrossBlock(nn.Module):
    """查询到 latent 的交叉注意力块；查询之间不做自注意力，保持查询独立性。"""

    def __init__(self, d_model, nhead, mlp_ratio=4, dropout=0.0, conditioning=None,
                 condition_dim=2):
        super().__init__()
        self.modulation = token_modulation(conditioning, condition_dim, d_model)
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_key = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
                                 nn.Linear(int(mlp_ratio * d_model), d_model))

    def forward(self, queries, latents, condition):
        queries = apply_modulation(queries, self.modulation, condition)
        queries = queries + self.cross(self.norm_query(queries), self.norm_key(latents),
                                       self.norm_key(latents), need_weights=False)[0]
        return queries + self.mlp(self.norm_mlp(queries))


class UPTBaseline(nn.Module):
    supports_query_decoder = True

    def __init__(self, d_model=128, nhead=8, layers=2, latents=128, decoder_layers=1,
                 mlp_ratio=4, dropout=0.0, condition_dim=2, flow_channels=5,
                 hull_channels=1, input_domains=None, output_domains=None, conditioning=None):
        super().__init__()
        self.d_model = d_model
        self.input_domains = tuple(input_domains) if input_domains else ("flow", "hull")
        self.output_domains = tuple(output_domains) if output_domains else ("flow", "hull")
        self.joint = input_domains is not None
        self.conditioning = conditioning

        if self.joint:
            self.cloud = CloudEmbedding(d_model)
        else:
            self.flow_input = nn.Linear(3, d_model)
            self.hull_input = nn.Linear(8, d_model)
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

    def _point_tokens(self, features, projection, condition):
        """逐点输入嵌入：坐标正弦编码 + 特征投影，历史口径再加工况偏置。"""
        tokens = sincos_embedding(features[..., :2], self.d_model) + projection(features)
        return tokens + self.condition(condition) if self.condition is not None else tokens

    def _latent(self, tokens, condition):
        return self.encoder(tokens, condition)

    def encode(self, batch):
        """整片输入点云 → 固定数量 latent（与查询点集无关）。"""
        condition = batch["condition"]
        if self.joint:
            tokens, coords, _ = self.cloud(batch)
            tokens = tokens + sincos_embedding(coords, self.d_model)
        else:
            tokens = self._point_tokens(flow_cloud(batch)[0], self.flow_input, condition)
        return {"latent": self._latent(tokens, condition), "condition": condition}

    def _features(self, state, query_x, query_fx=None):
        """解码器输出的逐查询特征；`query_fx` 为可选的查询点逐点输入特征。"""
        queries = self.query_mlp(sincos_embedding(query_x[..., :2], self.d_model))
        if query_fx is not None:
            if self.query_input is None:
                raise ValueError("该配置没有船面输出，解码器未构造查询特征支路")
            queries = queries + self.query_input(query_fx)
        for block in self.decoder:
            queries = block(queries, state["latent"], state["condition"])
        return self.decoder_norm(queries)

    def decode(self, state, query_x):
        """`decoder(x, query_x)`：在任意查询坐标处解码五个流场通道。"""
        return self.flow_head(self._features(state, query_x))

    def forward(self, batch, flow_queries):
        """流场在任意查询点上取值，船面取整条湿表面；批大小固定为 1。"""
        state = self.encode(batch)
        output = {}
        if self.flow_head is not None:
            output["flow"] = self.decode(state, flow_queries[..., :2])
        if self.hull_head is not None:
            features = hull_cloud(batch)
            if not self.joint:
                state = {"latent": self._latent(
                    self._point_tokens(features, self.hull_input, state["condition"]),
                    state["condition"]), "condition": state["condition"]}
            output["hull"] = self.hull_head(
                self._features(state, features[..., :2], features))
        return output
