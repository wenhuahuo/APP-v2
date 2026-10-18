"""船体表面几何量：面面积、顶点求积权重、顶点法向。"""

import numpy as np


def face_area(points, offsets, connectivity):
    """多边形面积，用 Newell 公式从顶点计算，不依赖面心位置。"""
    area = np.empty(len(offsets) - 1, dtype=np.float64)
    for i in range(len(area)):
        polygon = points[connectivity[offsets[i]:offsets[i + 1]]]
        vector = 0.5 * np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
        area[i] = np.linalg.norm(vector)
    return area


def vertex_area(points, offsets, connectivity):
    """顶点求积权重：每个面的面积按角点数均摊到其顶点。"""
    weight = np.zeros(len(points), dtype=np.float64)
    for i, area in enumerate(face_area(points, offsets, connectivity)):
        corner = connectivity[offsets[i]:offsets[i + 1]]
        weight[corner] += area / len(corner)
    return weight


def vertex_normals(points, offsets, connectivity):
    """面积加权的顶点法向，方向与面片绕向一致。"""
    normals = np.zeros((len(points), 3), dtype=np.float64)
    for i in range(len(offsets) - 1):
        corner = connectivity[offsets[i]:offsets[i + 1]]
        polygon = points[corner]
        vector = 0.5 * np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
        normals[corner] += vector / len(corner)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / length
