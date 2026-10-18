"""读取 OpenFOAM 二进制标量场文件的 internalField。"""

import re
from pathlib import Path

import numpy as np

_UNIFORM = re.compile(rb"internalField\s+uniform\s+([-0-9.eE+]+)")
_NONUNIFORM = re.compile(rb"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\(")


def read_scalar_field(path, n_cells):
    """返回 (n_cells,) float32；支持 uniform 与非均匀二进制两种写法。"""
    raw = Path(path).read_bytes()

    uniform = _UNIFORM.search(raw)
    if uniform is not None:
        return np.full(n_cells, float(uniform.group(1)), dtype=np.float32)

    match = _NONUNIFORM.search(raw)
    if match is None:
        raise ValueError(f"{path} 的 internalField 既不是 uniform 也不是二进制非均匀标量场")
    count = int(match.group(1))
    if count != n_cells:
        raise ValueError(f"{path} 记录 {count} 个值，网格有 {n_cells} 个单元")
    values = np.frombuffer(raw, dtype="<f8", count=count, offset=match.end())
    return values.astype(np.float32)
