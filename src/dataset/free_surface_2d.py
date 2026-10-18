"""把 interFoam 末态三维场压缩成自由液面二维场（按 Lpp 归一化的规则格点）。

二维场定义在设计水线平面上、按 Lpp 归一化的规则格点上：

- 自由液面取 `alpha.water = 0.5` 等值面，其高度插值到格点得到波高 `eta`（相对
  设计吃水）。等值面在船体投影内部（船体正上方的空腔）也会生成伪面片，插值前按
  符号距离剔除。
- 液面速度与动压：在 `(x, y, eta)` 处对单元场做单元内插值，不做跨单元平均，
  避免界面附近的速度被空气相抹平。
- 落在船体投影内部的格点没有自由液面，按符号距离掩除。

不采用竖直柱积分 `∫α dz`：网格相邻单元的投影有约 1 mm 的交叠（分块网格的偏斜
单元），按投影凸包逐单元累加会在交叠带重复计数，粗网格层的重复量可达单层厚度。
"""

from typing import NamedTuple

import numpy as np
from scipy.interpolate import griddata

FLOW_CHANNELS = ("eta", "u", "v", "w", "p_rgh")


class Lattice(NamedTuple):
    """水平规则格点：除坐标外还要带原点与格距，供可视化和反查格点下标。"""

    points: np.ndarray   # (N,2) 物理坐标 m
    origin: tuple        # (x0, y0)
    spacing: float       # 格距 m
    nx: int
    ny: int


def flow_lattice(lpp, box):
    """按归一化域定义生成规则格点；box 给出 x/Lpp、y/Lpp 范围与格距/Lpp。"""
    x0, x1 = (b * lpp for b in box["x_over_Lpp"])
    y0, y1 = (b * lpp for b in box["y_over_Lpp"])
    spacing = box["spacing_over_Lpp"] * lpp
    nx = int(round((x1 - x0) / spacing)) + 1
    ny = int(round((y1 - y0) / spacing)) + 1
    xs = x0 + spacing * np.arange(nx)
    ys = y0 + spacing * np.arange(ny)
    gx, gy = np.meshgrid(xs, ys)
    return Lattice(np.column_stack([gx.ravel(), gy.ravel()]), (x0, y0), spacing, nx, ny)


def interpolate_to_lattice(sample_xy, sample_value, lattice):
    """把散点值线性插值到格点，返回 (N,) 或 (N,C)；散点凸包之外的格点为 NaN。"""
    gx = lattice.origin[0] + lattice.spacing * np.arange(lattice.nx)
    gy = lattice.origin[1] + lattice.spacing * np.arange(lattice.ny)
    grid_x, grid_y = np.meshgrid(gx, gy)

    sample_value = np.asarray(sample_value, dtype=np.float64)
    single = sample_value.ndim == 1
    columns = sample_value[:, None] if single else sample_value
    values = np.empty((lattice.nx * lattice.ny, columns.shape[1]), dtype=np.float64)
    for k in range(columns.shape[1]):
        grid = griddata(sample_xy, columns[:, k], (grid_x, grid_y), method="linear")
        values[:, k] = grid.ravel()
    return values[:, 0] if single else values
