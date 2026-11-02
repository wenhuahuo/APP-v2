"""OFormer 基线：几何编码器 + 查询解码器，线性注意力带二维旋转相对位置。

参考 BaratiLab/OFormer 的 `IrregSpatialEncoder2D` + `PointWiseDecoder2D`：

- 编码器在整片输入点云上做 galerkin 线性注意力（Key/Value 逐点归一化），
  位置只以二维旋转相对位置进入注意力，没有 patch 或 latent 压缩；
- 解码器把查询坐标做 Gaussian Fourier 特征，与编码特征做同一种线性交叉注意力
  （查询与潜特征各用自己的坐标旋转），再过位置感知的 propagator 输出通道值。

因此 OFormer 与 APP 一样是分辨率无关的查询算子，两者的差别在于分词方式：
APP 按物理尺寸把点集压成 patch token，OFormer 保留每个输入点做线性注意力。
与原始仓库的差异只有维度取整（交叉注意力在 d_model 而不是 d_model/2）与激活函数，
机制保持一致。输入特征与其它基线相同（流场 `(x, y, sdf)`、船面八维特征），
位置取特征的前两维（流场 `(x, y)`、船面 `(u, v)`）。
"""

import math

import torch
import torch.nn as nn

from point_cloud import ConditionBias, flow_cloud, hull_cloud


class RotaryEmbedding(nn.Module):
    """一维旋转频率表；二维坐标按轴分别生成。`scale / min_freq` 设定波长下界。"""

    def __init__(self, dim, min_freq=1 / 4, scale=16.0):
        super().__init__()
        self.scale = scale
        self.min_freq = min_freq
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, coordinates):
        freqs = torch.einsum("...i,j->...ij", coordinates * (self.scale / self.min_freq),
                             self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def _rotate_half(tensor):
    pair = tensor.reshape(*tensor.shape[:-1], 2, -1)
    first, second = pair.unbind(dim=-2)
    return torch.cat([-second, first], dim=-1)


def _rotate(tensor, frequencies):
    return tensor * frequencies.cos() + _rotate_half(tensor) * frequencies.sin()


class PointPositions(nn.Module):
    """把二维坐标转成各头的旋转频率（`dim_head` 维）。"""

    def __init__(self, dim_head, min_freq, scale):
        super().__init__()
        self.embedding = RotaryEmbedding(dim_head // 2, min_freq, scale)

    def forward(self, coordinates, heads):
        freqs_x = self.embedding(coordinates[..., 0])
        freqs_y = self.embedding(coordinates[..., 1])
        return (freqs_x.unsqueeze(1).expand(-1, heads, -1, -1),
                freqs_y.unsqueeze(1).expand(-1, heads, -1, -1))


def _apply_2d(tensor, freqs_x, freqs_y):
    """张量 (B,H,N,d) 前后两半分别按 x、y 坐标旋转。"""
    half = tensor.shape[-1] // 2
    return torch.cat([_rotate(tensor[..., :half], freqs_x),
                      _rotate(tensor[..., half:], freqs_y)], dim=-1)


class _Heads(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.dim_head = d_model // nhead

    def _split(self, tensor):
        return tensor.reshape(*tensor.shape[:-1], self.nhead, self.dim_head).transpose(-3, -2)

    def _merge(self, tensor):
        tensor = tensor.transpose(-3, -2)
        return tensor.reshape(*tensor.shape[:-2], self.nhead * self.dim_head)


class GalerkinAttention(_Heads):
    """编码器自注意力：galerkin 线性注意力，Key/Value 逐点归一化后按坐标旋转。"""

    def __init__(self, d_model, nhead, min_freq, scale, dropout=0.0):
        super().__init__(d_model, nhead)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.key_norm = nn.LayerNorm(self.dim_head)
        self.value_norm = nn.LayerNorm(self.dim_head)
        self.positions = PointPositions(self.dim_head, min_freq, scale)
        self.project = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, coordinates):
        query, key, value = self.qkv(tokens).chunk(3, dim=-1)
        query, key, value = self._split(query), self._split(key), self._split(value)
        key = self.key_norm(key)
        value = self.value_norm(value)
        freqs_x, freqs_y = self.positions(coordinates, self.nhead)
        query = _apply_2d(query, freqs_x, freqs_y)
        key = _apply_2d(key, freqs_x, freqs_y)
        context = key.transpose(-1, -2) @ value
        out = query @ context / key.shape[-2]
        return self.dropout(self.project(self._merge(out)))


class CrossGalerkinAttention(_Heads):
    """解码器交叉注意力：查询来自坐标嵌入，Key/Value 来自编码特征。"""

    def __init__(self, d_model, nhead, min_freq, scale, dropout=0.0):
        super().__init__(d_model, nhead)
        self.to_query = nn.Linear(d_model, d_model, bias=False)
        self.to_key_value = nn.Linear(d_model, 2 * d_model, bias=False)
        self.key_norm = nn.LayerNorm(self.dim_head)
        self.value_norm = nn.LayerNorm(self.dim_head)
        self.positions = PointPositions(self.dim_head, min_freq, scale)
        self.project = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, latents, query_coords, latent_coords):
        query = self._split(self.to_query(queries))
        key, value = self.to_key_value(latents).chunk(2, dim=-1)
        key, value = self._split(key), self._split(value)
        key = self.key_norm(key)
        value = self.value_norm(value)
        query_freqs = self.positions(query_coords, self.nhead)
        latent_freqs = self.positions(latent_coords, self.nhead)
        query = _apply_2d(query, *query_freqs)
        key = _apply_2d(key, *latent_freqs)
        context = key.transpose(-1, -2) @ value
        out = query @ context / key.shape[-2]
        return self.dropout(self.project(self._merge(out)))


class FeedForward(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(int(mlp_ratio * d_model), d_model))

    def forward(self, tokens):
        return self.net(tokens)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio, min_freq, scale, dropout):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = GalerkinAttention(d_model, nhead, min_freq, scale, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, tokens, coordinates):
        tokens = tokens + self.attention(self.norm_attention(tokens), coordinates)
        return tokens + self.mlp(self.norm_mlp(tokens))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio, min_freq, scale, dropout):
        super().__init__()
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross = CrossGalerkinAttention(d_model, nhead, min_freq, scale, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, queries, latents, query_coords, latent_coords):
        queries = queries + self.cross(self.norm_cross(queries), latents, query_coords,
                                       latent_coords)
        return queries + self.mlp(self.norm_mlp(queries))


class Propagator(nn.Module):
    """位置感知残差层：归一化后把坐标拼回特征再过一个 MLP。"""

    def __init__(self, d_model, mlp_ratio=4.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model + 2, int(mlp_ratio * d_model)), nn.GELU(),
                                 nn.Linear(int(mlp_ratio * d_model), d_model))

    def forward(self, queries, coordinates):
        return self.mlp(torch.cat([self.norm(queries), coordinates], dim=-1)) + queries


class GaussianFourierFeatures(nn.Module):
    """查询坐标的 Gaussian 随机 Fourier 特征。"""

    def __init__(self, space_dim, mapping_size, scale):
        super().__init__()
        self.register_buffer("frequencies", torch.randn(space_dim, mapping_size) * scale)

    def forward(self, coordinates):
        projected = 2 * math.pi * (coordinates @ self.frequencies)
        return torch.cat([projected.sin(), projected.cos()], dim=-1)


class OFormerBaseline(nn.Module):
    supports_query_decoder = True

    def __init__(self, d_model=128, nhead=8, encoder_layers=2, decoder_layers=2,
                 mlp_ratio=4.0, propagator_layers=2, fourier_scale=8.0,
                 rotary_min_freq=0.25, rotary_scale=16.0, dropout=0.0, condition_dim=2,
                 flow_channels=5, hull_channels=1):
        super().__init__()
        self.flow_input = nn.Linear(3, d_model)
        self.hull_input = nn.Linear(8, d_model)
        self.condition = ConditionBias(condition_dim, d_model)
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, nhead, mlp_ratio, rotary_min_freq, rotary_scale, dropout)
            for _ in range(encoder_layers)])
        self.norm = nn.LayerNorm(d_model)

        # 查询侧：坐标 → Fourier 特征 → 交叉注意力 → 位置感知 propagator → 输出头
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

    def _points(self, features, projection, bias, query_coords):
        """整片点云编码一次，再在查询坐标处解码。"""
        coordinates = features[..., :2]
        latents = projection(features) + bias
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
        flow = self.flow_head(self._points(flow_cloud(batch), self.flow_input, bias,
                                          flow_queries[..., :2]))
        hull_features = hull_cloud(batch)
        hull = self.hull_head(self._points(hull_features, self.hull_input, bias,
                                          hull_features[..., :2]))
        return {"flow": flow, "hull": hull}
