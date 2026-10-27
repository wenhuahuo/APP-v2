"""两船型半船湿表面在 UV 参数域上的融合（hull morphing）。

做法：两船各自切出半船湿表面（y ≤ 0、z ≤ T），用 `half_hull_uv` 得到 (u, v)；
在公共 UV 网格上分别插值出坐标场 x、y、z（已除 Lpp），按权重 w 做坐标线性组合，
再乘以融合船的 Lpp 回到物理尺度。因此 w = 0 还原船 A、w = 1 还原船 B。

CAD 导出的 STL 有几十万到上百万面，直接在其上求测地距离（`half_hull_uv` 的
dijkstra）会慢到不可用，而且切片会在接缝处留下重复顶点。参数化前先按 Lpp 的
固定分数做顶点聚类降采样，再按半格距焊接一次，最后只保留最大的连通片。
"""

import gzip

import numpy as np
import trimesh
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree
from trimesh.intersections import slice_mesh_plane

from hull_uv import half_hull_uv

# 参数化网格的顶点间距取 Lpp / 该值。与公共 UV 网格（默认 n_u = 160）同量级，
# 既保留船体形状，又让测地距离求解留在秒级
PARAM_CELL_DIV = 300
# 聚类后再按该倍数格距焊接：CAD 接缝上坐标重合的顶点会被分进不同格子，需要并起来
WELD_CELL_FRACTION = 0.5
# 对称面吸附容差（相对 Lpp）。STL 的三角化在龙骨附近没有恰好落在 y = 0 上的顶点，
# 而 `half_hull_uv` 以 y ≈ 0 的龙骨中线作测地距离源点，不吸附会让 v 混入沿龙骨的假距离
PLANE_SNAP_FRACTION = 1e-3


def _compact(points, tris):
    """去掉退化三角形与重复面，并删除未被引用的顶点。"""
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[ok]
    _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    tris = tris[np.sort(first)]
    used, tris = np.unique(tris, return_inverse=True)
    return points[used], tris.reshape(-1, 3)


def _group_representative(indices, labels):
    """按 labels 分组，返回每个条目所属组里 raw 下标最小的条目。"""
    order = np.argsort(labels, kind="stable")  # 稳定排序保证组内按 raw 下标升序
    sorted_labels = labels[order]
    first = np.r_[True, sorted_labels[1:] != sorted_labels[:-1]]
    group_of = np.empty(len(labels), dtype=np.int64)
    group_of[order] = np.cumsum(first) - 1
    representative = indices[order][first]
    return representative[group_of]


def _weld_map(points, representative, tol):
    """把距离小于 tol 的代表点并成一组，返回合并后的代表（仍是 raw 下标）。"""
    unique, _ = np.unique(representative, return_inverse=True)
    pairs = cKDTree(points[unique]).query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return representative
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(unique), len(unique)))
    n_group, group = connected_components(graph, directed=False)
    group_min = np.full(n_group, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(group_min, group, unique)
    merged = np.searchsorted(unique, group_min)[group]
    # representative 的值都在 unique 里，用位置映射回合并后的代表
    position = np.searchsorted(unique, representative)
    return unique[merged[position]]


def _decimate(points, tris, cell):
    """顶点聚类降采样并焊接 CAD 接缝，返回降采样后的网格。

    对称面 y = 0 上的顶点不参与聚类与焊接：`half_hull_uv` 以龙骨中线为源点求测地
    距离，源点被聚稀会让 v 明显失真（v 混入沿龙骨方向的假距离）。
    """
    on_plane = points[:, 1] == 0.0
    representative = np.arange(len(points), dtype=np.int64)
    if (~on_plane).any():
        free = np.flatnonzero(~on_plane)
        key = np.floor(points[free] / cell).astype(np.int64)
        _, cluster = np.unique(key, axis=0, return_inverse=True)
        representative[free] = _group_representative(free, cluster)
        representative[free] = _weld_map(points, representative[free],
                                         WELD_CELL_FRACTION * cell)
    if on_plane.any():
        # 对称面上的顶点按细格焊接，消掉吸附后几乎重合的顶点，但保留整条龙骨
        plane = np.flatnonzero(on_plane)
        _, same = np.unique(np.round(points[plane] / (0.05 * cell), 0), axis=0,
                            return_inverse=True)
        representative[plane] = _group_representative(plane, same)
    kept = np.unique(representative)
    index = np.searchsorted(kept, representative)
    return _compact(points[kept], index[tris])


def _densify_plane_trace(points, tris, max_edge):
    """把对称面上的边界边补到不超过 max_edge 一段。

    `half_hull_uv` 以 y ≈ 0 的龙骨中线为测地距离源点。粗网格（例如平底船的底板）
    在龙骨上常常只有几个顶点，源点之间隔着大半米，v 会混进沿龙骨方向的假距离，
    水线附近因此被裁到 1，融合面就塌了。
    """
    directed = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    opposite = np.concatenate([tris[:, 2], tris[:, 0], tris[:, 1]])
    unique, inverse, count = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    on_plane = points[:, 1] == 0.0
    candidate = ((count[inverse] == 1)
                 & on_plane[directed[:, 0]] & on_plane[directed[:, 1]])
    length = np.linalg.norm(points[directed[:, 1]] - points[directed[:, 0]], axis=1)
    split = np.flatnonzero(candidate & (length > max_edge))
    if len(split) == 0:
        return points, tris

    added = []
    fans = []
    dropped = np.unique(split % len(tris))
    for i in split:
        a, b = directed[i]
        corner = opposite[i]
        pieces = int(np.ceil(length[i] / max_edge))
        fractions = (np.arange(1, pieces) / pieces)[:, None]
        first = len(points) + len(added)
        added.extend(points[a] + fractions * (points[b] - points[a]))
        chain = [int(a), *range(first, first + pieces - 1), int(b)]
        fans.extend((u, v, int(corner)) for u, v in zip(chain[:-1], chain[1:]))
    keep = np.ones(len(tris), dtype=bool)
    keep[dropped] = False
    return (np.vstack([points, np.asarray(added)]),
            np.vstack([tris[keep], np.asarray(fans, dtype=np.int64)]))


def _largest_component(points, tris):
    """只留最大的连通片：切片会在边界上留下孤立面片，无法做测地距离。"""
    graph = coo_matrix(
        (np.ones(3 * len(tris)),
         (tris.ravel(), np.roll(tris, -1, axis=1).ravel())),
        shape=(len(points), len(points)),
    )
    n_group, group = connected_components(graph, directed=False)
    if n_group == 1:
        return points, tris
    tag = group[tris[:, 0]]
    keep = tag == np.argmax(np.bincount(tag, minlength=n_group))
    return _compact(points, tris[keep])


def half_wet_surface(stl_path, lpp, draft):
    """STL -> 半船湿表面网格（y ≤ 0、z ≤ draft），面为 (M,3) 三角面。"""
    if str(stl_path).endswith(".gz"):
        with gzip.open(stl_path, "rb") as handle:
            mesh = trimesh.load(handle, file_type="stl", force="mesh")
    else:
        mesh = trimesh.load(str(stl_path), force="mesh")
    # slice_mesh_plane 保留法向正侧：先切 y ≤ 0（对称面），再切 z ≤ draft（设计水线）
    half = slice_mesh_plane(mesh, (0.0, -1.0, 0.0), (0.0, 0.0, 0.0))
    wet = slice_mesh_plane(half, (0.0, 0.0, -1.0), (0.0, 0.0, draft))
    points = np.asarray(wet.vertices, dtype=np.float64)
    tris = np.asarray(wet.faces, dtype=np.int64)
    # 把贴近对称面的顶点吸附到 y = 0，让半船曲面有真正的对称面边界
    points[np.abs(points[:, 1]) < PLANE_SNAP_FRACTION * lpp, 1] = 0.0
    cell = lpp / PARAM_CELL_DIV
    points, tris = _largest_component(*_decimate(points, tris, cell))
    return _densify_plane_trace(points, tris, cell)


def _uv_fields(uv, points):
    """(u, v) -> 三个坐标的散点插值器，共用同一个 Delaunay 三角剖分。

    用分片线性而不是 CloughTocher：水线附近大量顶点被裁到 v = 1，UV 点云里有成片
    共线点，CloughTocher 的梯度估计会过冲（实测 z 达 2 倍吃水），分片线性不会跑出
    数据范围。
    """
    # 水线附近顶点被裁到 v = 1，曲面上多个点可以落在同一个 (u, v) 上；
    # 三角剖分不接受重复点，按 (u, v) 去重保留首个
    _, unique = np.unique(np.round(uv, 9), axis=0, return_index=True)
    unique = np.sort(unique)
    triangulation = Delaunay(uv[unique])
    return [LinearNDInterpolator(triangulation, points[unique, axis])
            for axis in range(3)]


def _hull_surface(stl_path, lpp, draft):
    """船体 STL -> 半船湿表面的 UV 参数化（坐标为 Lpp 归一化后的值）。"""
    points, faces = half_wet_surface(stl_path, lpp, draft)
    uv = half_hull_uv(points, 3 * np.arange(len(faces) + 1, dtype=np.int64),
                      faces.ravel(), lpp, draft)["uv"]
    scaled = points / lpp
    return {"points": scaled, "faces": faces, "uv": uv,
            "fields": _uv_fields(uv, scaled), "lpp": float(lpp)}


def _sample_grid(fields, u_range, n_u, n_v):
    """在规则 UV 网格上插值出坐标场，返回 (坐标 (n_u,n_v,3), 有效掩码)。"""
    u = np.linspace(u_range[0], u_range[1], n_u)
    v = np.linspace(0.0, 1.0, n_v)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    query = np.column_stack([grid_u.ravel(), grid_v.ravel()])
    values = np.column_stack([interpolator(query) for interpolator in fields])
    values = values.reshape(n_u, n_v, 3)
    return u, values, np.isfinite(values).all(axis=2)


def _grid_faces(valid):
    """规则网格三角化：任一角点被掩除即跳过该四边形。绕向使法向朝船外。"""
    index = np.full(valid.size, -1, dtype=np.int64)
    index[valid.ravel()] = np.arange(int(valid.sum()))
    cell = index.reshape(valid.shape)
    quad = np.stack([cell[:-1, :-1], cell[1:, :-1], cell[1:, 1:], cell[:-1, 1:]], axis=-1)
    quad = quad.reshape(-1, 4)
    quad = quad[(quad >= 0).all(axis=1)]
    return np.vstack([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])


def _grid_volume(values, valid):
    """按站位横剖面面积积分求排水体积（半船截面镜像成整船）。

    网格点按 v 从龙骨排到水线，站位多边形 = 舷侧曲线 + 镜像的舷侧曲线，面积用
    shoelace 公式算；体积对 u 做梯形积分。只取整列都有效的站位。
    """
    complete = valid.all(axis=1)
    if complete.sum() < 2:
        return None
    section = values[complete]
    y, z = section[:, :, 1], section[:, :, 2]
    y = np.hstack([y, -y[:, ::-1]])
    z = np.hstack([z, z[:, ::-1]])
    # shoelace：0.5 |∮ y dz - z dy|
    area = 0.5 * np.abs(
        (y * np.roll(z, -1, axis=1) - np.roll(y, -1, axis=1) * z).sum(axis=1))
    x = section[:, :, 0].mean(axis=1)
    return float(np.trapezoid(area, x))


def morph_hulls(stl_a, stl_b, lpp_a, lpp_b, draft_a, draft_b, weight, n_u=160, n_v=60):
    """按权重 weight 融合两个船型的半船湿表面。

    返回 dict：`points`（物理尺度网格点）、`faces`（三角面）、`meta`（元信息）。
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight 应在 [0, 1] 内，收到 {weight}")
    weight = float(weight)

    surfaces = [_hull_surface(stl_a, lpp_a, draft_a), _hull_surface(stl_b, lpp_b, draft_b)]

    # 公共 UV 网格：u 取两船 u 范围的交集，v 固定 [0, 1]
    u_range = (max(surface["uv"][:, 0].min() for surface in surfaces),
               min(surface["uv"][:, 0].max() for surface in surfaces))
    if u_range[1] <= u_range[0]:
        raise ValueError(f"两船 u 范围无交集：[{u_range[0]:.3f}, {u_range[1]:.3f}]")

    sampled = [_sample_grid(surface["fields"], u_range, n_u, n_v) for surface in surfaces]
    valid = sampled[0][2] & sampled[1][2]
    if valid.sum() < 3:
        raise ValueError(f"公共 UV 网格上有效格点只有 {int(valid.sum())} 个")

    # 归一化坐标下的融合点，再乘融合船的 Lpp 回到物理尺度
    values = ((1.0 - weight) * sampled[0][1] + weight * sampled[1][1]) * \
        ((1.0 - weight) * lpp_a + weight * lpp_b)
    points = values[valid]
    faces = _grid_faces(valid)

    # 新吃水取融合面水线的 z 中位数，v = 1 那一行必须有效
    if not valid[:, -1].any():
        raise ValueError("v = 1 的水线在公共 UV 网格上无数值，无法确定新吃水")
    draft_new = float(np.median(values[valid[:, -1], -1, 2]))

    # 排水体积：融合船与被融合的两母船用同一套网格积分，保证可比
    volume = _grid_volume(values, valid)
    # 母船的体积在同一套 UV 网格上积分，再乘 Lpp³ 换回物理体积
    mother_volume = []
    for surface, (_, grid, mask) in zip(surfaces, sampled):
        mother = _grid_volume(grid, mask)
        mother_volume.append(None if mother is None else mother * surface["lpp"] ** 3)
    volume_interp = (1.0 - weight) * mother_volume[0] + weight * mother_volume[1] \
        if all(v is not None for v in mother_volume) else None
    volume_rel_diff = (volume - volume_interp) / volume_interp \
        if volume is not None and volume_interp else None

    meta = {
        "hull_a": str(stl_a),
        "hull_b": str(stl_b),
        "weight": weight,
        "n_u": n_u,
        "n_v": n_v,
        "Lpp_a": float(lpp_a),
        "Lpp_b": float(lpp_b),
        "T_a": float(draft_a),
        "T_b": float(draft_b),
        "Lpp_new": float((1.0 - weight) * lpp_a + weight * lpp_b),
        "T_new": draft_new,
        "u_range": [float(u_range[0]), float(u_range[1])],
        "n_points": int(valid.sum()),
        "n_faces": int(len(faces)),
        "volume": None if volume is None else float(volume),
        "volume_hull_a": None if mother_volume[0] is None else float(mother_volume[0]),
        "volume_hull_b": None if mother_volume[1] is None else float(mother_volume[1]),
        "volume_interp": None if volume_interp is None else float(volume_interp),
        "volume_rel_diff": None if volume_rel_diff is None else float(volume_rel_diff),
        "volume_method": "uv_grid_section_integration",
    }
    return {"points": points, "faces": faces, "meta": meta}
