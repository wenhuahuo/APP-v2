"""基线模型共用的点云组装。

基线不含 APP 的物理尺寸分词，也不含独立的查询解码器：它们在整片输入点云上直接
输出同一批点的场值（GNOT、Transolver、APPSolver-v1），或先压成固定容量 latent 再
解码回同一批点（UPT）。因此查询点必然取自当前密度的格点，这里用格点下标从整片
预测里取值。

批大小固定为 1：训练每步只用一条数据集条目（`collate_samples([sample])`）。
"""

import torch
import torch.nn as nn

FLOW_FEATURES = ("x", "y", "sdf")
HULL_FEATURES = ("u", "v", "x", "y", "z", "nx", "ny", "nz")


def flow_cloud(batch, index=0):
    """当前条目流场格点的输入特征 (1, nx*ny, 3)；格点全部参与，掩除位只由权重屏蔽。"""
    coords = batch["flow_coords"][index].reshape(-1, 2)
    sdf = batch["flow_sdf"][index].reshape(-1)
    return torch.column_stack([coords, sdf])[None]


def hull_cloud(batch, index=0):
    """船面点云的输入特征 (1, M, 8)（UV、坐标、法向）。"""
    features = torch.cat([batch["hull_uv"][index], batch["hull_points"][index],
                          batch["hull_normals"][index]], dim=-1)
    return features[None]


def lattice_index(batch, queries, index=0):
    """查询点坐标 → 格点展平下标。查询点由格点抽样得到，整数换算没有舍入误差。"""
    nx, ny = batch["flow_coords"].shape[1:3]
    x0, y0, x1, y1 = batch["flow_extent"][index].tolist()
    step_x, step_y = (x1 - x0) / (nx - 1), (y1 - y0) / (ny - 1)
    ix = torch.round((queries[..., 0] - x0) / step_x).long()
    iy = torch.round((queries[..., 1] - y0) / step_y).long()
    return ix * ny + iy


class ConditionBias(nn.Module):
    """工况参数作为逐点偏置加到输入嵌入上；基线不引入 APP 的逐层 FiLM。"""

    def __init__(self, condition_dim, d_model):
        super().__init__()
        self.projection = nn.Linear(condition_dim, d_model)

    def forward(self, condition):
        return self.projection(condition)[:, None, :]
