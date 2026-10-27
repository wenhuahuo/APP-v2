"""把二维自由液面数据集条目加载成 torch 张量。

流场域位于按 Lpp 归一化的规则格点上，所有条目共用同一张 (`nx`, `ny`) 索引网格：
条目文件存的是掩除后的散点，加载时按整数下标回填成稠密张量，掩除位置由
`flow_mask` 标记。批处理因此不需要点级 padding，查询点集密度（`stride` 抽样）
也退化成索引视图。

量纲：坐标与 SDF 除以 Lpp；目标无量纲化为 `eta/Lpp`、`u,v,w/U_inf`、
`p_rgh/(0.5 rho U_inf^2)`；条件输入为 `(Fr, log10 Re)`；求积权重除以 `Lpp^2`。
`scale` 返回 `(Lpp, U_inf, 0.5 rho U_inf^2)`，供评测把预测换算回物理量。
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "shipbench"


def _tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


class ShipDataset(Dataset):
    def __init__(self, dataset_root=DEFAULT_ROOT, entry_ids=None, stride=1):
        self.root = Path(dataset_root)
        index = json.loads((self.root / "index.json").read_text())
        self.entry_ids = sorted(index["entries"]) if entry_ids is None else list(entry_ids)
        if not self.entry_ids:
            raise ValueError("没有可选的数据集条目")
        self.stride = stride
        self.metas = [self._read_meta(entry_id) for entry_id in self.entry_ids]

        domain = self.metas[0]["flow_field"]["domain_over_Lpp"]
        self.nx, self.ny = domain["n_grid"]
        self.origin = (domain["x"][0], domain["y"][0])
        self.spacing = domain["spacing"]
        for meta in self.metas[1:]:
            if meta["flow_field"]["domain_over_Lpp"] != domain:
                raise ValueError(f"条目 {meta['entry_id']} 的流场格点域与其他条目不一致")

        xs = self.origin[0] + self.spacing * np.arange(self.nx)
        ys = self.origin[1] + self.spacing * np.arange(self.ny)
        # 格点域范围与 stride 无关，APP 分区要用它做参数域包围盒
        self.extent = (xs[0], ys[0], xs[-1], ys[-1])
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        self.flow_coords = _tensor(np.stack([grid_x, grid_y], axis=-1)[::stride, ::stride])

    def _read_meta(self, entry_id):
        return json.loads((self.root / entry_id / "meta.json").read_text())

    def _load(self, entry_id, name):
        return np.load(self.root / entry_id / f"{name}.npy")

    def __len__(self):
        return len(self.entry_ids)

    def __getitem__(self, i):
        entry_id = self.entry_ids[i]
        meta = self.metas[i]
        lpp = meta["Lpp"]
        u_inf = meta["U_inf"]
        pressure_scale = 0.5 * meta["rho"] * u_inf ** 2
        stride = self.stride

        shape = (self.nx, self.ny)
        sdf = np.zeros(shape)
        weight = np.zeros(shape)
        target = np.zeros(shape + (5,))

        points = self._load(entry_id, "flow_points")
        column, row = np.rint((points / lpp - self.origin) / self.spacing).astype(np.int64).T
        if column.min() < 0 or column.max() >= self.nx or row.min() < 0 or row.max() >= self.ny:
            raise ValueError(f"{entry_id} 的流场点落在归一化格点域之外")
        if len(np.unique(column * self.ny + row)) != len(points):
            raise ValueError(f"{entry_id} 的流场点回填到格点时出现重复下标")

        values = self._load(entry_id, "flow_target")
        sdf[column, row] = self._load(entry_id, "flow_sdf") / lpp
        weight[column, row] = self._load(entry_id, "flow_quad_weights") / lpp ** 2
        target[column, row] = np.column_stack([
            values[:, 0] / lpp,
            values[:, 1] / u_inf,
            values[:, 2] / u_inf,
            values[:, 3] / u_inf,
            values[:, 4] / pressure_scale,
        ])

        hull_uv = self._load(entry_id, "hull_uv")
        mask = weight > 0.0
        return {
            "flow_coords": self.flow_coords,
            "flow_extent": _tensor(self.extent),
            "flow_sdf": _tensor(sdf[::stride, ::stride]),
            "flow_target": _tensor(target[::stride, ::stride]),
            "flow_weight": _tensor(weight[::stride, ::stride] * stride ** 2),
            "flow_mask": torch.from_numpy(mask[::stride, ::stride]),
            "hull_points": _tensor(self._load(entry_id, "hull_points") / lpp),
            "hull_uv": _tensor(hull_uv),
            "hull_normals": _tensor(self._load(entry_id, "hull_normals")),
            "hull_pressure": _tensor(self._load(entry_id, "hull_pressure") / pressure_scale),
            "hull_weight": _tensor(self._load(entry_id, "hull_quad_weights") / lpp ** 2),
            "hull_mask": torch.ones(len(hull_uv), dtype=torch.bool),
            "condition": _tensor([meta["Fr"], np.log10(meta["Re"])]),
            "scale": _tensor([lpp, u_inf, pressure_scale]),
        }
