"""船体 STL 网格的读写（ASCII，可选 gzip），供 FFD 几何变体使用。

只处理三角形面片：读入顶点与面索引，变形后按原三角形顺序写回。不做顶点合并，
FFD 位移是点坐标的连续函数，重复顶点得到相同位移，写回后曲面仍然封闭。

容器按后缀决定：`.gz` 走 gzip（集群算例的 `KCS_Hull.stl.gz` 约定），其余写普通
文本（`dataset/` 下的变体与母船 `classic_hulls/*.stl` 都是普通文本）。
"""

import gzip
from pathlib import Path

import numpy as np


def _open(path, mode):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def read_stl(path):
    """读 STL（普通或 gzip），返回 (顶点 (V,3) float64, 面 (F,3) int64)。"""
    import trimesh

    with _open(path, "rb") as handle:
        mesh = trimesh.load(file_obj=handle, file_type="stl", process=False)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def write_stl(path, points, faces, solid="hull"):
    """按三角形顺序写 ASCII STL（普通或 gzip），法向由顶点叉积给出。"""
    triangles = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-30
    with _open(path, "wt") as handle:
        handle.write(f"solid {solid}\n")
        for normal, triangle in zip(normals, triangles):
            handle.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            handle.write("    outer loop\n")
            for vertex in triangle:
                handle.write(f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}\n")
            handle.write("    endloop\n  endfacet\n")
        handle.write(f"endsolid {solid}\n")
