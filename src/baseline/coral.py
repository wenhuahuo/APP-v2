"""CORAL 基线：几何点集编码到 latent，再由受 latent 调制的 SIREN 神经场解码任意查询点。

参考 CORAL（NeurIPS 2023，Operator Learning with Neural Fields）与官方实现
`LouisSerrano/coral` 的 `coral/siren.py::ModulatedSiren` 与
`coral/mfn.py::LatentToModulation`：

1. 点集编码器把输入点云（自由液面 + 船面，带域标识与坐标正弦编码）压成固定数量的
   latent token，池化成每个样本一个 latent 码 `z`（论文两阶段流程里的潜码）；编码器
   与 UPT 共用同一套 perceiver 结构（点集 → 固定 latent），CORAL 的判别性在 latent
   调制的坐标神经场；
2. `LatentToModulation` 把 `z` 映射成 SIREN 每个隐藏层的逐特征 shift（可选 scale）；
3. 坐标神经场在任意查询坐标处输出通道值：`x ← sin(w0·(Wx·s + b))`，隐藏层在
   激活前做 `scale·(Ws) + shift` 调制，最后一层不做调制（官方 `modulated_forward`）。

与官方两阶段流程的差异：论文先用 INR 拟合每个样本得到潜码、再训练模型预测潜码，
这里按项目单阶段训练目标把点集编码器与 SIREN 端到端联合训练；解码器语义（SIREN +
latent 逐层调制）保持不变，不用普通 MLP 替代。

工况走项目统一的逐层 FiLM：编码器各 block 用与 APPSolver-v2 相同的块入口 FiLM；
解码器另外把 `condition` 经零初始化线性层映射为每层 shift/scale 增量、叠加在 latent
调制量上，因此 SIREN 的调制在初始时完全来自 latent。

查询坐标按流场参数域包围盒归一化到 [-1, 1]（SIREN 的常规输入范围）；解码只依赖
查询坐标与潜码，`encode` / `decode` 分开暴露，换查询密度时不重跑编码器。
"""

import math

import torch
import torch.nn as nn

from point_cloud import (CloudEmbedding, PerceiverEncoder, flow_frame, normalize_within,
                         sincos_embedding)


class Sine(nn.Module):
    """SIREN 的正弦激活，`w0` 为 SIREN 论文的 ω₀。"""

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, inputs):
        return torch.sin(self.w0 * inputs)


class SirenLayer(nn.Module):
    """单个 SIREN 层，权重按 SIREN 论文初始化（首层与隐藏层不同）。"""

    def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.linear = nn.Linear(dim_in, dim_out)
        bound = 1.0 / dim_in if is_first else math.sqrt(c / dim_in) / w0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)
        self.activation = Sine(w0)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        return outputs if self.is_last else self.activation(outputs)


class LatentToModulation(nn.Module):
    """官方 `LatentToModulation`：latent 码 → 一组逐层调制量。"""

    def __init__(self, latent_dim, num_modulations, hidden_dim, num_layers):
        super().__init__()
        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            layers += [nn.Linear(hidden_dim, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


class ModulatedSiren(nn.Module):
    """受 latent 调制的 SIREN 坐标神经场（官方 `ModulatedSiren.modulated_forward`）。"""

    def __init__(self, in_features, hidden_features, out_features, num_layers, w0=30.0,
                 w0_initial=30.0, modulate_scale=False, modulate_shift=True):
        super().__init__()
        self.hidden_features = hidden_features
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        layers = [SirenLayer(in_features if index == 0 else hidden_features, hidden_features,
                             w0_initial if index == 0 else w0, is_first=index == 0)
                  for index in range(num_layers - 1)]
        self.net = nn.Sequential(*layers)
        self.last_layer = SirenLayer(hidden_features, out_features, w0=w0, is_last=True)
        # 只调制隐藏层，最后一层不调制（官方实现）
        self.num_modulations = hidden_features * (num_layers - 1)
        if modulate_scale and modulate_shift:
            self.num_modulations *= 2

    def forward(self, coords, modulations):
        middle = self.num_modulations // 2 if (self.modulate_scale and self.modulate_shift) else 0
        index = 0
        features = coords
        for layer in self.net:
            scale = (modulations[:, index:index + self.hidden_features].unsqueeze(1) + 1.0
                     if self.modulate_scale else 1.0)
            shift = (modulations[:, middle + index:middle + index + self.hidden_features]
                     .unsqueeze(1) if self.modulate_shift else 0.0)
            features = layer.activation(layer.linear(features) * scale + shift)
            index += self.hidden_features
        return self.last_layer(features)


class CORALBaseline(nn.Module):
    supports_query_decoder = True

    def __init__(self, d_model=128, nhead=8, encoder_layers=2, latents=64, mlp_ratio=4,
                 dropout=0.0, siren_hidden=128, siren_layers=4, w0=30.0, w0_initial=30.0,
                 modulate_scale=False, modulate_shift=True, modulation_hidden=128,
                 modulation_layers=2, condition_dim=2, conditioning="film", flow_channels=5,
                 hull_channels=1, input_domains=("flow", "hull"), output_domains=("flow",)):
        super().__init__()
        self.input_domains = tuple(input_domains)
        self.output_domains = tuple(output_domains)
        if "flow" not in self.input_domains:
            raise ValueError("CORAL 的输入点集必须包含自由液面点云")
        self.d_model = d_model
        self.conditioning = conditioning
        self.cloud = CloudEmbedding(d_model)
        self.encoder = PerceiverEncoder(d_model, nhead, encoder_layers, latents, mlp_ratio,
                                        dropout, conditioning, condition_dim)
        self.latent_norm = nn.LayerNorm(d_model)
        self.field = ModulatedSiren(2, siren_hidden, flow_channels, siren_layers, w0, w0_initial,
                                   modulate_scale, modulate_shift)
        self.modulation = LatentToModulation(d_model, self.field.num_modulations,
                                             modulation_hidden, modulation_layers)
        # 工况的逐层 FiLM 增量，零初始化使初始模型等价于纯 CORAL
        self.condition_modulation = (nn.Linear(condition_dim, self.field.num_modulations)
                                     if conditioning == "film" else None)
        if self.condition_modulation is not None:
            nn.init.zeros_(self.condition_modulation.weight)
            nn.init.zeros_(self.condition_modulation.bias)
        self.hull_head = nn.Linear(flow_channels, hull_channels) \
            if "hull" in self.output_domains else None

    def encode(self, batch):
        """输入点集与工况 → latent 码（与查询点集无关）。"""
        tokens, coords, _ = self.cloud(batch)
        tokens = tokens + sincos_embedding(coords, self.d_model)
        latents = self.encoder(tokens, batch["condition"])
        centre, half = flow_frame(batch)
        return {"latent": self.latent_norm(latents.mean(dim=1)),
                "condition": batch["condition"], "centre": centre, "half": half}

    def decode(self, state, query_coords):
        """在任意查询坐标 `(1,Q,2)` 处解码；坐标先按流场参数域归一化到 [-1, 1]。"""
        modulations = self.modulation(state["latent"])
        if self.condition_modulation is not None:
            modulations = modulations + self.condition_modulation(state["condition"])
        return self.field(normalize_within(query_coords[..., :2], state["centre"],
                                          state["half"]), modulations)

    def forward(self, batch, flow_queries):
        """流场在任意查询点上取值；批大小固定为 1。"""
        state = self.encode(batch)
        output = {"flow": self.decode(state, flow_queries)}
        if self.hull_head is not None:
            hull = batch["hull_points"][0][batch["hull_mask"][0]]
            output["hull"] = self.hull_head(self.decode(state, hull[..., :2][None]))
        return output
