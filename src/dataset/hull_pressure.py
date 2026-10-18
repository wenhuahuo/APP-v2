"""把面心标量插值到网格顶点。"""

import numpy as np


def face_to_vertex_area_weighted(points, offsets, connectivity, face_values):
    """按相邻面的面积加权平均，把面心标量插值到顶点。

    面积用 Newell 公式从多边形顶点计算，不依赖面心位置。
    """
    weighted = np.zeros(len(points), dtype=np.float64)
    weights = np.zeros(len(points), dtype=np.float64)

    for i, value in enumerate(np.asarray(face_values, dtype=np.float64)):
        corner = connectivity[offsets[i]:offsets[i + 1]]
        polygon = points[corner].astype(np.float64)
        vector_area = 0.5 * np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
        area = np.linalg.norm(vector_area)
        weighted[corner] += area * value
        weights[corner] += area

    missing = int((weights == 0.0).sum())
    if missing:
        raise RuntimeError(f"有 {missing} 个顶点没有相邻面，无法插值")
    return (weighted / weights).astype(np.float32)
