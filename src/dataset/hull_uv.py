"""半船湿表面 UV 参数化：u 为相对站位 x/Lpp，v 为自龙骨到设计水线的相对弧长。

半船曲面（对称面 y = 0 处截断）的每个横剖面都是开放曲线，湿表面即其中 z ≤ T 的
一段，因此 (u, v) 无需处理周向接缝，v = 1 恰好落在设计水线上。

v 由曲面测地距离定义：以龙骨中线为源点在三角形网格上做最短路，距离除以该站位
湿表面部分的剖面弧长。测地路径受曲面拓扑约束，只能沿舷侧上行，故该距离即剖面弧长。
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from hull_geometry import vertex_area, vertex_normals


def _triangles(offsets, connectivity):
    tris = []
    for i in range(len(offsets) - 1):
        corner = connectivity[offsets[i]:offsets[i + 1]]
        for k in range(1, len(corner) - 1):
            tris.append((corner[0], corner[k], corner[k + 1]))
    return np.asarray(tris, dtype=np.int64)


def _edge_graph(points, tris):
    edge = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edge = np.unique(np.sort(edge, axis=1), axis=0)
    length = np.linalg.norm(points[edge[:, 0]] - points[edge[:, 1]], axis=1)
    n_points = len(points)
    rows = np.concatenate([edge[:, 0], edge[:, 1]])
    cols = np.concatenate([edge[:, 1], edge[:, 0]])
    graph = coo_matrix((np.concatenate([length, length]), (rows, cols)),
                      shape=(n_points, n_points))
    return graph.tocsr()


def _girth_profile(u, distance, n_bins=400):
    """各站位湿表面的剖面弧长（龙骨到设计水线），空站位插值补全后做滑动中值平滑。"""
    bins = np.linspace(u.min(), u.max(), n_bins + 1)
    index = np.clip(np.digitize(u, bins) - 1, 0, n_bins - 1)
    girth = np.zeros(n_bins)
    np.maximum.at(girth, index, distance)
    filled = np.bincount(index, minlength=n_bins) > 0

    centre = 0.5 * (bins[:-1] + bins[1:])
    profile = np.interp(centre, centre[filled], girth[filled])
    window = 5
    padded = np.pad(profile, window // 2, mode="edge")
    profile = np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
    return centre, profile


def half_hull_uv(points, offsets, connectivity, lpp, draft):
    """返回 dict：uv (M,2)、vertex_area (M,) 求积权重、normal (M,3)。

    v 按设计水线归一，z > draft 的顶点（干舷与甲板）v 会超过 1，组装数据集时剔除。
    """
    tris = _triangles(offsets, connectivity)
    graph = _edge_graph(points, tris)

    z_min = points[:, 2].min()
    keel = (np.abs(points[:, 1]) < 1e-4 * lpp) & (points[:, 2] < z_min + 0.01 * lpp)
    if keel.sum() < 2:
        raise ValueError("半船网格上找不到龙骨中线（y≈0 且位于船底）的顶点")

    distance = dijkstra(graph, directed=False, indices=np.flatnonzero(keel)).min(axis=0)
    if not np.isfinite(distance).all():
        raise ValueError("船体网格不连通，测地距离存在不可达顶点")

    u = points[:, 0] / lpp
    wetted = points[:, 2] <= draft
    centre, profile = _girth_profile(u[wetted], distance[wetted])
    v = distance / np.interp(u, centre, profile)

    return {
        "uv": np.column_stack([u, np.clip(v, 0.0, 1.0)]),
        "vertex_area": vertex_area(points, offsets, connectivity),
        "normal": vertex_normals(points, offsets, connectivity),
    }
