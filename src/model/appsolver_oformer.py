"""APP 分词 + OFormer 主干与查询解码器：分词器即插即用的证据。

OFormer 基线保留每个输入点做线性注意力；这里只把输入换成 APP patch token
（token = patch 内 Δ 加权平均，位置取 patch 形心），编码器、解码器、propagator
与 OFormer 基线完全一致。分词不引入参数，因此两者参数量相同、训练超参相同，
差别只有「逐点」与「APP 分词」这一项。

给出 `input_domains` 时流场与船面的 patch token 用各自的线性投影加域标识后拼进同一
个编码器（与 APPSolver-v2 的双域口径一致，船体几何因此进入流场预测）。`conditioning`
可以是 `none` / `film` / `adaln` / `adaln_zero` / `fourier_add` / `cond_token` /
`cond_xattn`：块内融合由 `src/model/conditioning.py` 的同一套公式完成，`cond_token`
在拼接后的 token 序列头部加一个零位置的工况上下文 token；缺省时保持历史口径的
独立编码与工况加性偏置。`input_domains` 只给出一个域时（船体融合消融的 `flow_only`）
只拼该域 token，另一域的分词与投影不参与前向。

流场分词方法由 `flow_tokenizer` 选择：`adaptive` 用 `flow_size_tiers` 的 SDF 分档，
`uniform` 用固定物理边长 `flow_uniform_size`，`learned` 用 `LearnedTokenizer` 的
Transolver 式软分配（每样本 `flow_learned_tokens` 个 token）。两者与 APPSolver-v2
共用同一实现，船体分词与查询解码不随分词方法改变。
"""

import torch
import torch.nn as nn

from learned_tokenizer import LearnedTokenizer
from oformer import DecoderBlock, EncoderBlock, GaussianFourierFeatures, Propagator
from point_cloud import ConditionBias
from tokenizer import flow_tokens, hull_tokens

FLOW_TOKENIZERS = ("adaptive", "uniform", "learned")


class APPSolverOFormer(nn.Module):
    supports_query_decoder = True

    def __init__(self, flow_size_tiers, hull_size, d_model=128, nhead=8, encoder_layers=2,
                 decoder_layers=2, mlp_ratio=4.0, propagator_layers=3, fourier_scale=8.0,
                 rotary_min_freq=0.25, rotary_scale=16.0, dropout=0.0, condition_dim=2,
                 flow_channels=5, hull_channels=1, max_level=12, input_domains=None,
                 output_domains=None, conditioning=None, flow_tokenizer="adaptive",
                 flow_uniform_size=0.1, flow_learned_tokens=512):
        super().__init__()
        self.input_domains = tuple(input_domains) if input_domains else ("flow", "hull")
        self.output_domains = tuple(output_domains) if output_domains else ("flow", "hull")
        self.joint = input_domains is not None
        self.flow_size_tiers = flow_size_tiers
        self.hull_size = hull_size
        self.max_level = max_level
        self.flow_tokenizer = flow_tokenizer
        self.flow_uniform_size = flow_uniform_size
        if flow_tokenizer not in FLOW_TOKENIZERS:
            raise ValueError(f"未知流场分词方法: {flow_tokenizer}（可选 "
                             f"{' / '.join(FLOW_TOKENIZERS)}）")
        self.flow_input = nn.Linear(2, d_model)
        self.hull_input = nn.Linear(8, d_model)
        self.learned_flow = (LearnedTokenizer(2, d_model, flow_learned_tokens)
                             if "flow" in self.input_domains and flow_tokenizer == "learned"
                             else None)
        self.domain = nn.Embedding(2, d_model) if self.joint else None
        self.condition_token = (nn.Linear(condition_dim, d_model)
                                if conditioning == "cond_token" else None)
        self.condition_domain = (nn.Parameter(torch.zeros(d_model))
                                 if conditioning == "cond_token" else None)
        self.condition = ConditionBias(condition_dim, d_model) if conditioning is None else None
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, nhead, mlp_ratio, rotary_min_freq, rotary_scale, dropout,
                         conditioning, condition_dim) for _ in range(encoder_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.coordinates = nn.Sequential(
            GaussianFourierFeatures(2, d_model // 2, fourier_scale), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, nhead, mlp_ratio, rotary_min_freq, rotary_scale, dropout,
                         conditioning, condition_dim) for _ in range(decoder_layers)])
        self.propagator = nn.ModuleList([
            Propagator(d_model, mlp_ratio) for _ in range(propagator_layers)])
        self.flow_head = (nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                        nn.GELU(), nn.Linear(d_model // 2, flow_channels))
                          if "flow" in self.output_domains else None)
        self.hull_head = (nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                        nn.GELU(), nn.Linear(d_model // 2, hull_channels))
                          if "hull" in self.output_domains else None)

    def _encode(self, tokens, centres, bias, condition):
        for block in self.encoder:
            tokens = block(tokens, centres, condition)
        return self.norm(tokens + bias)

    def _decode(self, latents, centres, query_coords, condition):
        queries = self.coordinates(query_coords)
        for block in self.decoder:
            queries = block(queries, latents, query_coords, centres, condition)
        for layer in self.propagator:
            queries = layer(queries, query_coords)
        return queries

    def _patches(self, batch, index=0):
        """按 `input_domains` 分词；未启用的输入域不构造 token。

        硬分词在 CPU 上完成（几何分区不参与梯度）；学习分词在参数所在设备上做
        软分配，与 APPSolver-v2 的 `_learned_flow_tokens` 同一口径。
        """
        flow = None
        if "flow" in self.input_domains:
            if self.flow_tokenizer == "learned":
                flow = self._learned_flow_tokens(batch, index)
            else:
                tiers = (self.flow_size_tiers if self.flow_tokenizer == "adaptive"
                         else [[None, self.flow_uniform_size]])
                flow = flow_tokens(batch, index, tiers, self.max_level)
        hull = (hull_tokens(batch, index, self.hull_size, self.max_level)
                if "hull" in self.input_domains else None)
        return flow, hull

    def _learned_flow_tokens(self, batch, index):
        device = self.learned_flow.token_embed.device
        mask = batch["flow_mask"][index].to(device).reshape(-1)
        features = batch["flow_coords"][index].to(device).reshape(-1, 2)[mask]
        weights = batch["flow_weight"][index].to(device).reshape(-1)[mask]
        return self.learned_flow(features, weights)

    def _add_condition_token(self, tokens, centres, condition):
        """`cond_token`：在 token 序列头部加一个零位置的工况上下文 token。

        `tokens (P, d)` 与 `centres (P, 2)` 都是单条目形状（批大小固定为 1），
        与 `_joint` 的输出一致。
        """
        token = self.condition_token(condition) + self.condition_domain
        return (torch.cat([token, tokens]),
                torch.cat([centres.new_zeros((1, 2)), centres]))

    def _joint(self, flow, hull):
        """启用的输入域 patch token 各自投影 + 域标识后拼接，流场 token 在前。

        逗号分词在 CPU 上完成（几何分区不参与梯度），投影前必须搬回参数所在设备；
        形心同样要跟着搬，否则位置编码会与 latents 设备不一致。船体融合消融的
        `flow_only` 只拼流场 token，`flow_with_hull` / `joint_flow_hull` 拼两域。
        """
        device = self.flow_input.weight.device
        blocks = []
        if flow is not None:
            blocks.append(self.flow_input(flow[0].to(device)) + self.domain.weight[0])
        if hull is not None:
            blocks.append(self.hull_input(hull[0].to(device)) + self.domain.weight[1])
        tokens = torch.cat(blocks)
        # 学习分词的形心已在参数设备上，硬分词的形心还在 CPU，拼接前逐域搬过来。
        centres = torch.cat([part[2].to(device) for part in (flow, hull)
                             if part is not None])
        return tokens, centres

    def forward(self, batch, flow_queries):
        """批大小固定为 1；流场在任意查询点上取值，船面取整条湿表面。"""
        condition = batch["condition"]
        bias = self.condition(condition) if self.condition is not None else 0.0
        flow, hull = self._patches(batch)
        hull_query = batch["hull_uv"][0][batch["hull_mask"][0]].to(
            self.flow_input.weight.device)[None]
        output = {}
        if self.joint:
            tokens, centres = self._joint(flow, hull)
            if self.condition_token is not None:
                tokens, centres = self._add_condition_token(tokens, centres, condition)
            latents = self._encode(tokens, centres[None], bias, condition)
            if self.flow_head is not None:
                output["flow"] = self.flow_head(
                    self._decode(latents, centres[None], flow_queries[..., :2], condition))
            if self.hull_head is not None:
                output["hull"] = self.hull_head(
                    self._decode(latents, centres[None], hull_query, condition))
            return output

        for name, patches in (("flow", flow), ("hull", hull)):
            head = self.flow_head if name == "flow" else self.hull_head
            if head is None or patches is None:
                continue
            projection = self.flow_input if name == "flow" else self.hull_input
            tokens = projection(patches[0].to(projection.weight.device))
            centres = patches[2].to(projection.weight.device)
            if self.condition_token is not None:
                tokens, centres = self._add_condition_token(tokens, centres, condition)
            latents = self._encode(tokens, centres[None], bias, condition)
            query = flow_queries[..., :2] if name == "flow" else hull_query
            output[name] = head(self._decode(latents, centres[None], query, condition))
        return output
