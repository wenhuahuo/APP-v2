"""APP 双域分词前端：把流场与船面点集积分成 patch token 序列。

三个 APP 分词前端迁移模型（Transolver / UPT / GNOT 主干）共用这一层。分词器直接调用
`src/app/tokenizer.py` 的 `flow_tokens` / `hull_tokens`，不复制分词实现：流场按到船体
水线的距离分档决定 patch 物理边长，船面在 UV 参数域按固定物理弧长切分；每个 patch
的 token 是 patch 内求积加权平均，位置取 patch 形心。

两个域各自做线性投影、加域标识与形心 Fourier 位置，再拼成同一序列（流场 token 在前，
船面 token 在后）。返回的 `weight` 为 patch 积分面积，作为求积权重交给下游的加权注意力
与坐标查询解码器；`centres` 为 patch 形心坐标，供位置相关的主干使用。
"""

import torch
import torch.nn as nn

from positional import FourierFeatures
from tokenizer import flow_tokens, hull_tokens


class PatchFrontend(nn.Module):
    """`(batch) → (tokens (1,P,d), weight (1,P), centres (1,P,2))`。"""

    def __init__(self, flow_size_tiers, hull_size, d_model, fourier_features=16,
                 fourier_scale=4.0, max_level=12, input_domains=("flow", "hull")):
        super().__init__()
        self.input_domains = tuple(input_domains)
        self.flow_size_tiers = flow_size_tiers
        self.hull_size = hull_size
        self.max_level = max_level
        self.flow_input = (nn.Linear(2, d_model) if "flow" in self.input_domains else None)
        self.hull_input = (nn.Linear(8, d_model) if "hull" in self.input_domains else None)
        self.domain = nn.Embedding(2, d_model)
        self.fourier = FourierFeatures(2, fourier_features, fourier_scale)
        self.position = nn.Linear(2 * fourier_features, d_model)

    def _domain_token(self, patches, projection, domain_id):
        tokens, areas, centres = patches
        device = projection.weight.device
        centres = centres.to(device)
        marker = torch.full((len(tokens),), domain_id, dtype=torch.long, device=device)
        return (projection(tokens.to(device)) + self.domain(marker)
                + self.position(self.fourier(centres)), areas.to(device), centres)

    def forward(self, batch, index=0):
        """分词在 CPU 上完成（几何分区不参与梯度），投影前搬到参数所在设备。"""
        patches = []
        if "flow" in self.input_domains:
            patches.append(self._domain_token(
                flow_tokens(batch, index, self.flow_size_tiers, self.max_level),
                self.flow_input, 0))
        if "hull" in self.input_domains:
            patches.append(self._domain_token(
                hull_tokens(batch, index, self.hull_size, self.max_level),
                self.hull_input, 1))
        if len(patches) == 1:
            tokens, weight, centres = patches[0]
        else:
            tokens = torch.cat([patch[0] for patch in patches])
            weight = torch.cat([patch[1] for patch in patches])
            centres = torch.cat([patch[2] for patch in patches])
        return tokens[None], weight[None], centres[None]
