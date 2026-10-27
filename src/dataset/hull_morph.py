"""两船型半船湿表面的站位剖面融合（hull morphing）。

算法（站位剖面参数化）：

1. STL → 半船湿表面三角网格（y ≤ 0、z ≤ T 两个平面切片）；
2. u ∈ [0, 1] 按**每船自身湿表面 x 范围**归一，共 n_u 个站位，端点内缩
   `STATION_EPS`。球鼻艏伸出 FP 的部分与艉突出都算进来，两船因此是艏尖对艏尖、
   艉端对艉端的对齐；
3. 每个站位用平面-网格交线取剖面折线 (y, z)，整理成**从船底端到水线端**的有序
   路径（端点用最近邻行走串起来）。出现多个分离环时（球鼻艏过渡区、带附体的艉段）
   按环的最低 z 排序首尾拼接，环数 > 1 的站位数记在 meta 的 `n_spliced_stations`；
4. v = 沿剖面路径的累积弧长归一化到 [0, 1]，在公共 v 网格（n_v 点）上线性插值
   重采样 (y, z)。每船得到规则 (n_u × n_v) 网格，x 就是站位坐标；
5. 融合：逐格点 P = (1-w)·PA + w·PB，坐标先各自除 Lpp、融合后乘
   Lpp_new = (1-w)Lpp_A + w·Lpp_B。某个站位只要有一船缺失就整列掩除，
   因此各权重的网格拓扑完全一致。w = 0 精确还原船 A、w = 1 精确还原船 B。

v 取站位剖面的弧长，而不是全局测地距离（旧实现的 `half_hull_uv`）：测地距离在
粗三角化的平底船（kvlcc2 的底板）上会混入沿龙骨方向的假距离，实测让 kvlcc2 ×
wigley 的融合体积只有插值值的 54%；弧长只依赖单条剖面折线，对三角化粗细不敏感。
"""

import gzip

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from trimesh.intersections import mesh_plane, slice_mesh_plane

# 站位端点内缩量（相对湿表面 x 范围）：尖端剖面退化成一个点，弧长为 0 无法归一
STATION_EPS = 2e-3
# 剖面端点焊接容差（相对 Lpp）
WELD_FRACTION = 1e-6
# 剖面折线最短弧长（相对 Lpp），更短的视为空截面
MIN_ARC_FRACTION = 1e-6
# 剖面碎段阈值（相对该站位路径总弧长）：三角化在切面近乎相切处会切出若干毫米级的
# 假碎段，实测都落在 y ≈ 0、z ≈ 0 的底板细片上，不丢会把路径末端拉到 z = 0
MIN_PIECE_FRACTION = 1e-2
# 对称面吸附容差（相对 Lpp）：切片后边界点的 y 是浮点零，吸附掉便于判断
PLANE_SNAP_FRACTION = 1e-8
# 碎连通片面积阈值（相对最大连通片）：CAD 导出的 STL 常带几十个退化细片，
# 实测 kcs 有 84 个、合计面积 1.4e-7 m²，而主船体 21.4 m²
AREA_FRACTION = 1e-6


def _compact(points, tris):
    """去掉退化三角形与重复面，并删除未被引用的顶点。"""
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    tris = tris[ok]
    _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    tris = tris[np.sort(first)]
    used, tris = np.unique(tris, return_inverse=True)
    return points[used], tris.reshape(-1, 3)


def _drop_debris(points, tris):
    """只留面积不可忽略的连通片。

    STL 里常有一堆退化细片（集中在某个 x 上、y ≈ 0、面积 ~1e-9 m²），它们会在
    站位平面上给出假交线，把剖面路径带歪。
    """
    graph = coo_matrix(
        (np.ones(3 * len(tris)),
         (tris.ravel(), np.roll(tris, -1, axis=1).ravel())),
        shape=(len(points), len(points)),
    )
    n_group, group = connected_components(graph, directed=False)
    if n_group == 1:
        return points, tris
    edge = points[tris[:, 1]] - points[tris[:, 0]]
    other = points[tris[:, 2]] - points[tris[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(edge, other), axis=1)
    total = np.bincount(group[tris[:, 0]], weights=area, minlength=n_group)
    keep = total >= AREA_FRACTION * total.max()
    if keep.all():
        return points, tris
    return _compact(points, tris[keep[group[tris[:, 0]]]])


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
    points, tris = _drop_debris(*_compact(points, tris))
    # 切片给出的是浮点零（1e-21 量级），吸附到真正的对称面
    points[np.abs(points[:, 1]) < PLANE_SNAP_FRACTION * lpp, 1] = 0.0
    return points, tris


def _weld(points, tol):
    """把间距 ≤ tol 的点并成同一节点，返回 (节点坐标, 每个点的节点下标)。"""
    count = len(points)
    if count == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(count, count))
    n_node, label = connected_components(graph, directed=False)
    first = np.full(n_node, count, dtype=np.int64)
    np.minimum.at(first, label, np.arange(count))
    return points[first], label


def _chains(nodes, start, end):
    """把交线段串成若干条有序链（每条链从自己的最低 z 端点起）。

    交线段在平面上的连通分量是一条折线；但 STL 三角化在切面近乎相切的地方会把
    同一条剖面切成几段（细长的艏柱区），或者出现分叉（叠了重复线段），所以按分量
    逐段行走、遇到走不通的节点就地再起一段。
    """
    count = len(nodes)
    graph = coo_matrix((np.ones(2 * len(start)), (np.r_[start, end], np.r_[end, start])),
                       shape=(count, count))
    n_ring, ring = connected_components(graph, directed=False)
    edge = np.r_[start, end]
    neighbour = [np.r_[end, start][edge == i] for i in range(count)]

    # 起点候选顺序：先按 z 升序，再按 |y| 升序（平底船从龙骨中线一侧起）
    rank = np.lexsort((np.abs(nodes[:, 0]), nodes[:, 1]))
    used = np.zeros(count, dtype=bool)
    chains = []
    for c in sorted(range(n_ring), key=lambda c: nodes[ring == c, 1].min()):
        for seed in rank:
            if ring[seed] != c or used[seed]:
                continue
            used[seed] = True
            piece = [seed]
            current = seed
            while True:
                candidate = neighbour[current][~used[neighbour[current]]]
                if len(candidate) == 0:
                    break
                step = nodes[candidate] - nodes[current]
                current = candidate[np.argmin(np.einsum("ij,ij->i", step, step))]
                used[current] = True
                piece.append(current)
            chains.append(nodes[piece])
    return chains


def _splice(chains, min_piece):
    """把若干条链拼成一条路径：丢碎段，其余按端点最近原则接到路径的两端。

    起点取「最低 z（同 z 取 |y| 最小）」端点所在的链，然后向两端生长：每次把所有
    剩余链端点里离当前路径某端点最近的链接上去（链可以整体反向）。只往尾巴上接
    会在「主剖面 + 底部碎段」这种站位上把碎段接到水线端，路径末端就落回龙骨。

    碎段是三角化噪声（实测弧长占比 ≤ 1% 的碎段都在 y ≈ 0、z ≈ 0 的底板细片上），
    丢弃并计数，不静默丢整环。
    """
    length = np.array([np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in chains])
    total = length.sum()
    keep = length >= min_piece * total
    dropped = int((~keep).sum())
    chains = [chains[i] for i in np.flatnonzero(keep)]
    if not chains:
        return None, dropped

    # 起点：最低 z / 最小 |y| 的端点所在链（各链已从自己的该端点起）
    first = min(range(len(chains)),
                key=lambda i: (chains[i][0, 1], abs(chains[i][0, 0])))
    path = chains[first]
    remaining = [i for i in range(len(chains)) if i != first]
    while remaining:
        best = None
        for i in remaining:
            for at_start in (False, True):  # 接到路径的尾端 / 首端
                end = path[0] if at_start else path[-1]
                for reverse in (False, True):
                    tip = chains[i][-1] if reverse else chains[i][0]
                    distance = np.linalg.norm(tip - end)
                    if best is None or distance < best[0]:
                        best = (distance, i, at_start, reverse)
        _, i, at_start, reverse = best
        piece = chains[i][::-1] if reverse else chains[i]
        path = np.vstack([piece, path]) if at_start else np.vstack([path, piece])
        remaining.remove(i)
    return path, dropped


def _section_path(segments, tol, min_arc):
    """站位剖面折线 -> 从船底端到水线端的有序路径。

    返回 (路径 (m,2) 的 (y,z), 信息 dict)；断面为空时返回 (None, 信息)。
    """
    info = {"n_chains": 0, "n_dropped": 0, "n_clipped": 0}
    if len(segments) == 0:
        return None, info
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    # 丢掉退化交线（细长碎片面与平面近乎相切）
    seg = seg[np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1) > tol]
    if len(seg) == 0:
        return None, info

    nodes, label = _weld(seg.reshape(-1, 2), tol)
    start, end = label[0::2], label[1::2]
    ok = start != end
    start, end = start[ok], end[ok]
    if len(start) == 0:
        return None, info

    chains = _chains(nodes, start, end)
    info["n_chains"] = len(chains)
    path, info["n_dropped"] = _splice(chains, MIN_PIECE_FRACTION)
    if path is None or path.shape[0] < 2:
        return None, info
    # 兜底：路径必须从船底端走到水线端
    if path[0, 1] > path[-1, 1]:
        path = path[::-1]
        info["n_reversed"] = 1
    # 闭合环或底部多挂一段的站位（例如艉部带内壳的船型）：路径末端不在最高点，
    # 在最高点截断，保证「从船底端走到水线端」，截断计数记进 meta
    top = int(np.argmax(path[:, 1]))
    if top < path.shape[0] - 1:
        path = path[:top + 1]
        info["n_clipped"] = 1
        if path.shape[0] < 2:
            return None, info
    if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < min_arc:
        return None, info
    return path, info


def _resample(path, n_v):
    """按累积弧长把剖面折线重采样成 n_v 个 (y, z)，弧长归一化即 v。"""
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    # 弧长单调不减：重复值（零长段）会让 np.interp 的 xp 非严格递增
    unique = np.r_[True, np.diff(s) > 0]
    s = s[unique] / s[-1]
    v = np.linspace(0.0, 1.0, n_v)
    return np.column_stack([np.interp(v, s, path[unique, 0]),
                            np.interp(v, s, path[unique, 1])])


def hull_station_grid(stl_path, lpp, draft, n_u=160, n_v=60, eps=STATION_EPS):
    """单船的站位剖面网格。

    返回 dict：`points`（(n_u, n_v, 3)，坐标已除 Lpp）、`stations`（(n_u,) 站位掩码）、
    `valid`（(n_u, n_v) 格点掩码）、
    `x_range`（湿表面的物理 x 范围）、`n_spliced_stations`（多环拼接的站位数）、
    `n_invalid_stations`（空截面的站位数）。
    """
    points, faces = half_wet_surface(stl_path, lpp, draft)
    mesh = trimesh.Trimesh(points, faces, process=False)
    x_lo, x_hi = float(points[:, 0].min()), float(points[:, 0].max())
    span = x_hi - x_lo
    stations = x_lo + (eps + np.linspace(0.0, 1.0, n_u) * (1.0 - 2.0 * eps)) * span

    grid = np.zeros((n_u, n_v, 3), dtype=np.float64)
    valid = np.zeros(n_u, dtype=bool)
    spliced = 0
    dropped = 0
    clipped = 0
    for index, station in enumerate(stations):
        lines = mesh_plane(mesh, (1.0, 0.0, 0.0), (station, 0.0, 0.0))
        path, info = _section_path(lines[:, :, (1, 2)], WELD_FRACTION * lpp,
                                  MIN_ARC_FRACTION * lpp)
        if path is None:
            continue
        profile = _resample(path, n_v)
        grid[index, :, 0] = station
        grid[index, :, 1] = profile[:, 0]
        grid[index, :, 2] = profile[:, 1]
        valid[index] = True
        spliced += int(info["n_chains"] - info["n_dropped"] > 1)
        dropped += info["n_dropped"]
        clipped += info["n_clipped"]

    # valid 展开成 (n_u, n_v)：有效站位的每一行都有效，无效站位整列掩除
    return {"points": grid / lpp, "valid": np.repeat(valid[:, None], n_v, axis=1),
            "stations": valid, "x_range": (x_lo, x_hi),
            "n_spliced_stations": int(spliced),
            "n_dropped_fragments": int(dropped),
            "n_clipped_loops": int(clipped),
            "n_invalid_stations": int((~valid).sum())}


def grid_faces(valid):
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

    格点按 v 从龙骨排到水线，站位多边形 = 舷侧曲线 + 镜像的舷侧曲线，面积用
    shoelace 公式算；体积对 x 做梯形积分。只取整列都有效的站位。
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

    surfaces = [hull_station_grid(stl_a, lpp_a, draft_a, n_u, n_v),
                hull_station_grid(stl_b, lpp_b, draft_b, n_u, n_v)]
    valid = surfaces[0]["valid"] & surfaces[1]["valid"]
    station_ok = valid.all(axis=1)
    if station_ok.sum() < 3:
        raise ValueError(f"两船都有效的站位只有 {int(station_ok.sum())} 个")
    if not station_ok[-1]:
        raise ValueError("v = 1 的水线端在最后一个站位上缺失，无法确定新吃水")

    # 归一化坐标下逐格点融合，再乘融合船的 Lpp 回到物理尺度
    lpp_new = (1.0 - weight) * float(lpp_a) + weight * float(lpp_b)
    values = ((1.0 - weight) * surfaces[0]["points"] + weight * surfaces[1]["points"])
    points = values[station_ok].reshape(-1, 3) * lpp_new
    faces = grid_faces(valid)

    # 新吃水取融合面水线（v = 1 一行）的 z 中位数
    draft_new = float(np.median(values[station_ok][:, -1, 2])) * lpp_new

    # 排水体积：融合船与两母船用同一套站位网格积分，保证可比
    volume = _grid_volume(values, valid)
    volume = None if volume is None else volume * lpp_new ** 3
    mother_volume = []
    for surface, lpp in zip(surfaces, (lpp_a, lpp_b)):
        mother = _grid_volume(surface["points"], surface["valid"])
        mother_volume.append(None if mother is None else mother * float(lpp) ** 3)
    volume_interp = (1.0 - weight) * mother_volume[0] + weight * mother_volume[1] \
        if all(v is not None for v in mother_volume) else None
    volume_rel_diff = (volume - volume_interp) / volume_interp \
        if volume is not None and volume_interp else None
    # 融合在 Lpp 归一化坐标里做，体积按 Lpp_new³ 缩放，所以「同一预期的体积」应为
    # 归一化体积的线性插值再缩放（两船 Lpp 差得多时，与物理体积线性插值差别很大）
    volume_interp_norm = None
    if all(v is not None for v in mother_volume):
        volume_interp_norm = (1.0 - weight) * mother_volume[0] * (lpp_new / float(lpp_a)) ** 3 \
            + weight * mother_volume[1] * (lpp_new / float(lpp_b)) ** 3
    volume_rel_diff_norm = (volume - volume_interp_norm) / volume_interp_norm \
        if volume is not None and volume_interp_norm else None

    meta = {
        "hull_a": str(stl_a),
        "hull_b": str(stl_b),
        "weight": weight,
        "n_u": n_u,
        "n_v": n_v,
        "station_eps": STATION_EPS,
        "Lpp_a": float(lpp_a),
        "Lpp_b": float(lpp_b),
        "T_a": float(draft_a),
        "T_b": float(draft_b),
        "Lpp_new": lpp_new,
        "T_new": draft_new,
        "x_range_a": [float(v) for v in surfaces[0]["x_range"]],
        "x_range_b": [float(v) for v in surfaces[1]["x_range"]],
        "n_stations_valid": int(station_ok.sum()),
        "n_stations_invalid_a": surfaces[0]["n_invalid_stations"],
        "n_stations_invalid_b": surfaces[1]["n_invalid_stations"],
        "n_spliced_stations_a": surfaces[0]["n_spliced_stations"],
        "n_spliced_stations_b": surfaces[1]["n_spliced_stations"],
        "n_dropped_fragments_a": surfaces[0]["n_dropped_fragments"],
        "n_dropped_fragments_b": surfaces[1]["n_dropped_fragments"],
        "n_clipped_loops_a": surfaces[0]["n_clipped_loops"],
        "n_clipped_loops_b": surfaces[1]["n_clipped_loops"],
        "n_points": int(station_ok.sum() * n_v),
        "n_faces": int(len(faces)),
        "volume": None if volume is None else float(volume),
        "volume_hull_a": None if mother_volume[0] is None else float(mother_volume[0]),
        "volume_hull_b": None if mother_volume[1] is None else float(mother_volume[1]),
        "volume_interp": None if volume_interp is None else float(volume_interp),
        "volume_rel_diff": None if volume_rel_diff is None else float(volume_rel_diff),
        "volume_interp_norm": None if volume_interp_norm is None
        else float(volume_interp_norm),
        "volume_rel_diff_norm": None if volume_rel_diff_norm is None
        else float(volume_rel_diff_norm),
        "volume_method": "station_section_integration",
        "parameterization": "station_section_arc_length",
    }
    return {"points": points, "faces": faces, "meta": meta}
