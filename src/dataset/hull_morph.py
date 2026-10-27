"""两船型全船（含干舷与甲板）的站位剖面融合（hull morphing）。

OpenFOAM 的 snappyHexMesh 需要完整封闭船体，所以融合船的输出是**水密全船**：
半船剖面从龙骨沿舷侧上行、经水线、过甲板边缘、横过半甲板到中线，镜像后逐站位
首尾封口。

算法（站位剖面参数化）：

1. STL → 半船曲面三角网格（只切 y = 0 对称面，**不再切设计水线**）；
2. u ∈ [0, 1] 按**每船自身曲面 x 范围**归一，共 n_u 个站位，端点内缩
   `STATION_EPS`。球鼻艏伸出 FP 的部分与艉突出都算进来，两船因此是艏尖对艏尖、
   艉端对艉端的对齐；
3. 每个站位用平面-网格交线取剖面折线 (y, z)，整理成**从龙骨到甲板中线**的有序
   路径：端点用最近邻行走串起来，多段剖面（平底船的底部分段、球鼻艏过渡区、
   带附体的艉段）按「离路径尾端最近的端点」接到尾部，超过 `MAX_JOIN_FRACTION`
   的接不上就丢并计数；路径末端若回落到最高点以下（闭合环、挂在尾部的多余分叉）
   在最后一个最高点截断；
4. v 分两段，各按段内弧长归一：段 1 龙骨 → 水线 v ∈ [0, 1]，段 2 水线 → 甲板中线
   v ∈ [1, 2]。水线位置由 z = T 与剖面折线的交点插值给出，因此 v = 1 一行严格在
   设计吃水上。公共网格 v 取 `n_v + n_v2 - 1` 行（默认 60 + 20 - 1 = 79 行）；
5. 融合：逐格点 P = (1-w)·PA + w·PB，坐标先各自除 Lpp、融合后乘
   Lpp_new = (1-w)Lpp_A + w·Lpp_B。某个站位只要有一船缺失就整列掩除，
   因此各权重的网格拓扑完全一致。w = 0 精确还原船 A、w = 1 精确还原船 B；
6. 全船：半船网格镜像 y → −y（中线两行共用顶点，不重复造点），首尾两个站位的
   剖面环用「拉链」三角化封口，得到 `is_watertight` 的封闭船体。

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

# 站位端点内缩量（相对曲面 x 范围）：尖端剖面退化成一个点，弧长为 0 无法归一
STATION_EPS = 2e-3
# 剖面端点焊接容差（相对 Lpp）
WELD_FRACTION = 1e-6
# 剖面折线最短弧长（相对 Lpp），更短的视为空截面
MIN_ARC_FRACTION = 1e-6
# 剖面碎段阈值（相对该站位路径总弧长）：三角化在切面近乎相切处会切出若干毫米级的
# 假碎段，实测都落在 y ≈ 0、z ≈ 0 的底板细片上，不丢会把路径末端拉到 z = 0
MIN_PIECE_FRACTION = 1e-2
# 分段剖面拼接的最大端距（相对 Lpp）：超过它说明不是同一条剖面，丢掉并计数
MAX_JOIN_FRACTION = 0.10
# 对称面吸附容差（相对 Lpp）：切片后边界点的 y 是浮点零，吸附掉便于判断
PLANE_SNAP_FRACTION = 1e-8
# 碎连通片面积阈值（相对最大连通片）：CAD 导出的 STL 常带几十个退化细片，
# 实测 kcs 有 84 个、合计面积 1.4e-7 m²，而主船体 21.4 m²
AREA_FRACTION = 1e-6
# 顶端截断容差（相对 Lpp）：判断「路径末端是否回落到最高点以下」
TOP_TOL_FRACTION = 1e-9
# 拼接跳变落在中线时的推开量（相对 Lpp）：人造跳变在 y = 0 上会形成零宽度竖墙，
# 镜像后两半重合、顶点合并后那些边被 4 个面片共用（is_watertight 不通过），所以把
# 内部行推到 −6e-6·Lpp（6 m 船上约 36 μm），几何上仍是零宽度墙
SEAM_OFFSET_FRACTION = 6e-6
# 镜像算子
MIRROR = np.array([1.0, -1.0, 1.0])


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


def _load(stl_path):
    if str(stl_path).endswith(".gz"):
        with gzip.open(stl_path, "rb") as handle:
            return trimesh.load(handle, file_type="stl", force="mesh")
    return trimesh.load(str(stl_path), force="mesh")


def _slice(points, faces, normal, origin, lpp):
    """平面切片 + 清理（去退化面、去碎连通片、吸附对称面）。"""
    mesh = trimesh.Trimesh(points, faces, process=False)
    cut = slice_mesh_plane(mesh, tuple(normal), tuple(origin))
    points = np.asarray(cut.vertices, dtype=np.float64)
    tris = np.asarray(cut.faces, dtype=np.int64)
    points, tris = _drop_debris(*_compact(points, tris))
    # 切片给出的是浮点零（1e-21 量级），吸附到真正的对称面
    points[np.abs(points[:, 1]) < PLANE_SNAP_FRACTION * lpp, 1] = 0.0
    return points, tris


def half_ship_surface(stl_path, lpp):
    """STL → 半船曲面网格（y ≤ 0，含干舷与甲板），面为 (M,3) 三角面。"""
    mesh = _load(stl_path)
    return _slice(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64),
                  (0.0, -1.0, 0.0), (0.0, 0.0, 0.0), lpp)


def half_wet_surface(stl_path, lpp, draft):
    """STL → 半船**湿**表面网格（y ≤ 0、z ≤ draft），用于表面积对账与体检。"""
    points, tris = half_ship_surface(stl_path, lpp)
    return _slice(points, tris, (0.0, 0.0, -1.0), (0.0, 0.0, draft), lpp)


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
    同一条剖面切成几段（平底船底板上的粗长三角形、细长的艏柱区），或者出现分叉
    （叠了重复线段），所以按分量逐段行走、遇到走不通的节点就地再起一段。
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


def _splice(chains, min_piece, max_join):
    """把若干条链拼成一条路径：丢碎段，其余接到路径**尾端**。

    起点取「最低 z（同 z 取 |y| 最小）」端点所在的链，从它出发只往尾部生长：每次
    在剩余链的端点里挑离当前路径尾端最近的一个接上去（链可以整体反向）。只往尾部
    长是关键——底板分段给出的那截「中线碎链」会落到文件末尾，随后被顶端截断丢掉，
    若允许往首端接（旧实现的做法）就会把它接到龙骨前面，整条剖面被挤到一端。

    碎段是三角化噪声（实测弧长占比 ≤ 1% 的碎段都在 y ≈ 0、z ≈ 0 的底板细片上），
    丢弃并计数；端距超过 `max_join` 的链不拼，计入 `detached`，不静默丢环。
    """
    length = np.array([np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in chains])
    total = length.sum()
    keep = length >= min_piece * total
    dropped = int((~keep).sum())
    chains = [chains[i] for i in np.flatnonzero(keep)]
    if not chains:
        return None, dropped, 0

    first = min(range(len(chains)), key=lambda i: (chains[i][0, 1], abs(chains[i][0, 0])))
    path = chains[first]
    remaining = [i for i in range(len(chains)) if i != first]
    detached = 0
    while remaining:
        best = None
        for i in remaining:
            for reverse in (False, True):
                tip = chains[i][-1] if reverse else chains[i][0]
                distance = np.linalg.norm(tip - path[-1])
                if best is None or distance < best[0]:
                    best = (distance, i, reverse)
        distance, i, reverse = best
        if distance > max_join:
            detached = len(remaining)
            break
        piece = chains[i][::-1] if reverse else chains[i]
        path = np.vstack([path, piece])
        remaining.remove(i)
    return path, dropped, detached


def _section_path(segments, tol, min_arc, max_join, top_tol):
    """站位剖面交线 -> 从龙骨端到甲板中线的有序路径。

    终止点有两个候选规则，优先用「中线」那一条：

    * 半船曲面在 y = 0 上有边界（龙骨线、艏柱、甲板中线），路径的终点必然落在这条
      边界上。取**中线节点里 z 最高的那个**作终点：正常船型就是甲板中线；舷墙/
      舷弧比甲板高的船型（frigate、kvlcc2）也不会因为最高点落在舷边而把甲板切掉。
    * 剖面上没有中线节点（整个环落在 y < 0 的附体）时退回「最后一个最高点」。

    返回 (路径 (m,2) 的 (y,z), 信息 dict)；剖面为空时返回 (None, 信息)。
    """
    info = {"n_chains": 0, "n_dropped": 0, "n_clipped": 0, "n_detached": 0}
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
    path, info["n_dropped"], info["n_detached"] = _splice(
        chains, MIN_PIECE_FRACTION, max_join)
    if path is None or path.shape[0] < 2:
        return None, info
    # 终点：中线节点里 z 最高的那个（起点是龙骨，不参与候选）
    seam = np.flatnonzero(np.abs(path[:, 0]) <= tol)
    seam = seam[seam > 0]
    if len(seam) > 0:
        stop = int(seam[np.argmax(path[seam, 1])])
    else:
        stop = int(np.flatnonzero(path[:, 1] >= path[:, 1].max() - top_tol)[-1])
    if stop < path.shape[0] - 1:
        path = path[:stop + 1]
        info["n_clipped"] = 1
        if path.shape[0] < 2:
            return None, info
    if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < min_arc:
        return None, info
    return path, info


def _waterline_fraction(path, draft):
    """剖面折线与水线 z = draft 的第一个交点的弧长占比；不相交时返回 None。

    站位的剖面可能是「整条都在水线以上」（艏柱前倾的尖端）或「整条都在水线以下」
    （球鼻艏的尖端），这时 v = 1 没有几何意义，由 `hull_station_grid` 用邻站的占比
    补上，保证沿 u 的 v 映射连续、并且不产生退化的重复行。
    """
    z = path[:, 1]
    above = z >= draft
    if not above.any():
        return None
    index = int(np.argmax(above))
    if index == 0:
        return None
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    if s[-1] <= 0:
        return None
    z0, z1 = z[index - 1], z[index]
    ratio = (draft - z0) / (z1 - z0) if z1 > z0 else 0.0
    return float((s[index - 1] + ratio * step[index - 1]) / s[-1])


def _split_by_fraction(path, fraction):
    """按弧长占比把剖面折线切成两段，交点用线性插值给出。"""
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    if s[-1] <= 0:
        return None
    target = float(np.clip(fraction, 0.0, 1.0)) * s[-1]
    index = int(np.clip(np.searchsorted(s, target, "right"), 1, len(path) - 1))
    span = s[index] - s[index - 1]
    ratio = (target - s[index - 1]) / span if span > 0 else 0.0
    crossing = path[index - 1] + ratio * (path[index] - path[index - 1])
    return (np.vstack([path[:index], crossing]), np.vstack([crossing, path[index:]]))


def _resample(path, n):
    """按累积弧长把剖面折线重采样成 n 个 (y, z)，段内弧长归一化即该段的 v。"""
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    # 弧长单调不减：重复值（零长段）会让 np.interp 的 xp 非严格递增
    unique = np.r_[True, np.diff(s) > 0]
    s = s[unique] / s[-1]
    v = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(v, s, path[unique, 0]),
                            np.interp(v, s, path[unique, 1])])


def hull_station_grid(stl_path, lpp, draft, n_u=160, n_v=60, n_v2=20, eps=STATION_EPS):
    """单船的站位剖面网格（半船，含干舷与甲板）。

    返回 dict：`points`（(n_u, n_row, 3)，坐标已除 Lpp，n_row = n_v + n_v2 - 1）、
    `stations`（(n_u,) 站位掩码）、`valid`（(n_u, n_row) 格点掩码）、`x_range`
    （半船曲面的物理 x 范围）、`n_v`/`n_v2`（水线行是第 n_v - 1 行）、
    `n_waterline_stations`（剖面真的穿过设计水线的站位数）以及多环拼接、截断、
    丢弃的站位数等诊断计数。

    两遍扫描：第一遍取各站位剖面与 v = 1 的弧长占比，第二遍按占比重采样。占比要
    单独算，是因为艏艉尖端的剖面可能整条落在水线以上或以下（frigate 艏柱、kvlcc2
    球鼻艏），这些站位没有水线交点，用邻站的占比补上，v 映射才沿 u 连续。
    """
    points_, faces = half_ship_surface(stl_path, lpp)
    mesh = trimesh.Trimesh(points_, faces, process=False)
    x_lo, x_hi = float(points_[:, 0].min()), float(points_[:, 0].max())
    span = x_hi - x_lo
    stations = x_lo + (eps + np.linspace(0.0, 1.0, n_u) * (1.0 - 2.0 * eps)) * span
    n_row = n_v + n_v2 - 1
    min_arc = MIN_ARC_FRACTION * lpp

    paths, fractions, counts = [None] * n_u, np.full(n_u, np.nan), []
    for index, station in enumerate(stations):
        lines = mesh_plane(mesh, (1.0, 0.0, 0.0), (station, 0.0, 0.0))
        path, info = _section_path(lines[:, :, (1, 2)], WELD_FRACTION * lpp,
                                   min_arc, MAX_JOIN_FRACTION * lpp,
                                   TOP_TOL_FRACTION * lpp)
        counts.append(info)
        if path is None:
            continue
        if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < min_arc:
            continue
        paths[index] = path
        fraction = _waterline_fraction(path, draft)
        if fraction is not None:
            fractions[index] = fraction

    known = np.flatnonzero(~np.isnan(fractions))
    if len(known) == 0:
        raise ValueError(f"{stl_path} 的剖面都不穿过设计水线 z = {draft}")
    # 缺口（水线以上/以下的尖端站位）用最近的有效站位线性补，两侧都没有就取端点
    fractions = np.interp(np.arange(n_u), known, fractions[known])

    grid = np.zeros((n_u, n_row, 3), dtype=np.float64)
    valid = np.zeros(n_u, dtype=bool)
    for index, station in enumerate(stations):
        if paths[index] is None:
            continue
        split = _split_by_fraction(paths[index], fractions[index])
        if split is None:
            continue
        first, second = split
        if (np.linalg.norm(np.diff(first, axis=0), axis=1).sum() < min_arc
                or np.linalg.norm(np.diff(second, axis=0), axis=1).sum() < min_arc):
            continue
        profile = np.vstack([_resample(first, n_v), _resample(second, n_v2)[1:]])
        # 内部行（龙骨行与甲板中线行除外）贴在中线上的推开一点，避免镜像重合
        inner = slice(1, len(profile) - 1)
        too_close = np.abs(profile[inner, 0]) < SEAM_OFFSET_FRACTION
        profile[inner, 0] = np.where(too_close, -SEAM_OFFSET_FRACTION, profile[inner, 0])
        grid[index, :, 0] = station
        grid[index, :, 1:] = profile
        valid[index] = True

    infos = [info for info, path in zip(counts, paths) if path is not None]
    return {"points": grid / lpp, "valid": np.repeat(valid[:, None], n_row, axis=1),
            "stations": valid, "x_range": (x_lo, x_hi), "n_v": n_v, "n_v2": n_v2,
            "n_row": n_row,
            "n_waterline_stations": int(len(known)),
            "n_spliced_stations": int(sum(i["n_chains"] - i["n_dropped"] > 1
                                          for i in infos)),
            "n_dropped_fragments": int(sum(i["n_dropped"] for i in infos)),
            "n_clipped_loops": int(sum(i["n_clipped"] for i in infos)),
            "n_detached_stations": int(sum(i["n_detached"] for i in infos)),
            "n_invalid_stations": int((~valid).sum())}


def valid_run(valid):
    """返回 `valid`（(n_u,) 站位掩码）里最长的一段连续 True 的切片。

    全船 STL 要水密，站位必须是连续一段：中间夹一个无效站位就会在船体上开一道
    口子，首尾封口也封不住。
    """
    best, start = None, None
    for index in range(len(valid) + 1):
        inside = index < len(valid) and bool(valid[index])
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            span = slice(start, index)
            if best is None or span.stop - span.start > best.stop - best.start:
                best = span
            start = None
    return best


def _cap_rows(n_row):
    """封口用的「拉链」三角化：把每个站位的剖面环按中线对折成四边形条带。

    剖面环 = 半个剖面 + 它的镜像，两半只在中线的两行（龙骨、甲板中线）相接。把
    第 j 行与它的镜像行 mm(j) = (2(n_row-1) - j) mod 2(n_row-1) 连成四边形，
    j 从 0 走到 n_row-2 就铺满整个环，永远不会自交（不需要耳切，也就不会退化）。
    """
    n_loop = 2 * n_row - 2
    quads = []
    for j in range(n_row - 1):
        mirror_j = (n_loop - j) % n_loop
        mirror_next = (n_loop - (j + 1)) % n_loop
        quad = [j, j + 1, mirror_next, mirror_j]
        unique = []
        for item in quad:
            if item not in unique:
                unique.append(item)
        quads.append(unique)
    return quads, n_loop


def full_ship_mesh(grid):
    """规则站位网格 (n_u, n_row, 3) -> 水密全船三角网格，返回 (points, faces)。

    镜像 y → −y 时中线两行共用顶点（不重复造点），侧表面沿剖面环首尾相接，首尾
    两个站位的截面用 `_cap_rows` 的拉链条带封口，因此得到的网格主线（每条边恰好
    两个面片）闭合。
    """
    grid = np.asarray(grid, dtype=np.float64)
    n_u, n_row = grid.shape[0], grid.shape[1]
    rows, n_loop = _cap_rows(n_row)
    loop = np.concatenate([grid, grid[:, -2:0:-1] * MIRROR], axis=1)
    points = loop.reshape(-1, 3).copy()

    station = np.arange(n_u - 1)
    j = np.arange(n_loop)
    next_j = (j + 1) % n_loop
    a = (station[:, None] * n_loop + j[None, :]).ravel()
    b = ((station + 1)[:, None] * n_loop + j[None, :]).ravel()
    c = ((station + 1)[:, None] * n_loop + next_j[None, :]).ravel()
    d = (station[:, None] * n_loop + next_j[None, :]).ravel()
    faces = [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]

    # 首尾封口：剖面环在 (y, z) 平面上是顺时针走向，右手法则给出 −x 法向，所以
    # 艉端（外法向朝 −x）用原序、艏端（朝 +x）逆序
    for offset, reverse in ((0, False), ((n_u - 1) * n_loop, True)):
        for quad in rows:
            order = quad[::-1] if reverse else quad
            for k in range(1, len(order) - 1):
                faces.append(np.array([[offset + order[0], offset + order[k],
                                        offset + order[k + 1]]]))
    faces = np.vstack(faces)

    # 绕向整体校验：水密网格的有向体积必须为正，否则把全部面片翻过来
    tri = points[faces]
    volume = np.einsum("ij,ij->i", tri[:, 0],
                       np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    if volume < 0:
        faces = faces[:, ::-1]
    return points, faces


def _profile_area(profile, draft):
    """一条剖面折线（(m,2) 的 (y,z)）在 z ≤ draft 以下围成的面积（半船镜像后）。

    折线从龙骨走到甲板中线，水线以下的截面积分只要取 z ≤ draft 的那一段，端点用
    线性插值补出准确的水线点。
    """
    z = profile[:, 1]
    above = np.flatnonzero(z > draft)
    if len(above) == 0:
        cut = profile
    else:
        index = int(above[0])
        if index == 0:
            return 0.0
        z0, z1 = z[index - 1], z[index]
        ratio = (draft - z0) / (z1 - z0) if z1 > z0 else 0.0
        crossing = profile[index - 1] + ratio * (profile[index] - profile[index - 1])
        cut = np.vstack([profile[:index], crossing])
    y = np.r_[cut[:, 0], -cut[:, 0][::-1]]
    zz = np.r_[cut[:, 1], cut[:, 1][::-1]]
    return float(0.5 * np.abs(
        (y * np.roll(zz, -1) - np.roll(y, -1) * zz).sum()))


def grid_displacement(grid, draft):
    """按站位横剖面面积积分求水线以下的排水体积（站位积分法）。

    `grid` 是 (n_u, n_row, 3) 的物理尺度网格，只含连续有效站位；每个站位的截面面积
    取 z ≤ draft 的部分（含镜像），对 x 做梯形积分。
    """
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape[0] < 2:
        return None
    areas = np.array([_profile_area(station, draft) for station in grid[:, :, 1:]])
    x = grid[:, :, 0]
    if not np.allclose(x, x[:, :1]):
        raise ValueError("站位网格的 x 不是逐行常数")
    return float(np.trapezoid(areas, x[:, 0]))


def morph_hulls(stl_a, stl_b, lpp_a, lpp_b, draft_a, draft_b, weight,
                n_u=160, n_v=60, n_v2=20):
    """按权重 weight 融合两个船型，输出水密全船网格。

    返回 dict：`points`（物理尺度全船顶点）、`faces`（三角面）、
    `grid`（(n_valid, n_row, 3) 半船站位网格，物理尺度）、`stations`（站位的物理
    x 坐标）、`meta`（元信息）。
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight 应在 [0, 1] 内，收到 {weight}")
    weight = float(weight)

    surfaces = [hull_station_grid(stl_a, lpp_a, draft_a, n_u, n_v, n_v2),
                hull_station_grid(stl_b, lpp_b, draft_b, n_u, n_v, n_v2)]
    common = surfaces[0]["stations"] & surfaces[1]["stations"]
    run = valid_run(common)
    if run is None or run.stop - run.start < 3:
        raise ValueError("两船都有效的连续站位少于 3 个")
    n_row = surfaces[0]["n_row"]
    station_ok = np.zeros(n_u, dtype=bool)
    station_ok[run] = True
    n_valid = int(station_ok.sum())

    # 归一化坐标下逐格点融合，再乘融合船的 Lpp 回到物理尺度
    lpp_new = (1.0 - weight) * float(lpp_a) + weight * float(lpp_b)
    values = ((1.0 - weight) * surfaces[0]["points"] + weight * surfaces[1]["points"])
    grid = values[station_ok].reshape(n_valid, n_row, 3) * lpp_new
    points, faces = full_ship_mesh(grid)
    mesh = trimesh.Trimesh(points, faces, process=False)

    # 新吃水取融合面水线（v = 1 一行）的 z 中位数
    water_row = n_v - 1
    draft_new = float(np.median(grid[:, water_row, 2]))
    # 归一化口径的插值吃水：两船各自把吃水按 Lpp 归一后再融合、再乘 Lpp_new
    draft_interp = ((1.0 - weight) * float(draft_a) / float(lpp_a)
                    + weight * float(draft_b) / float(lpp_b)) * lpp_new

    # 排水体积：融合船与两母船用同一套站位网格积分，保证可比
    displacement = grid_displacement(grid, draft_new)
    mother_displacement = []
    for surface, lpp, draft in zip(surfaces, (lpp_a, lpp_b), (draft_a, draft_b)):
        mask = surface["stations"]
        mother_grid = surface["points"][mask].reshape(int(mask.sum()), n_row, 3) * float(lpp)
        mother_displacement.append(grid_displacement(mother_grid, float(draft)))
    interp_norm = None
    if all(value is not None for value in mother_displacement):
        interp_norm = ((1.0 - weight) * mother_displacement[0] * (lpp_new / float(lpp_a)) ** 3
                       + weight * mother_displacement[1] * (lpp_new / float(lpp_b)) ** 3)
    interp_phys = None
    if all(value is not None for value in mother_displacement):
        interp_phys = ((1.0 - weight) * mother_displacement[0]
                       + weight * mother_displacement[1])

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    meta = {
        "hull_a": str(stl_a),
        "hull_b": str(stl_b),
        "weight": weight,
        # —— CFD 建算例要用的几何量 ——
        "L_ref_m": float(bounds[1, 0] - bounds[0, 0]),
        "T_m": draft_new,
        "half_beam_m": float(np.abs(points[:, 1]).max()),
        "depth_m": float(bounds[1, 2]),
        "n_triangles": int(len(faces)),
        "displacement_m3": displacement,
        "parents": {"a": str(stl_a), "b": str(stl_b)},
        # —— 复现与对账用的中间量 ——
        "n_u": n_u,
        "n_v": n_v,
        "n_v2": n_v2,
        "n_row": n_row,
        "station_eps": STATION_EPS,
        "Lpp_a": float(lpp_a),
        "Lpp_b": float(lpp_b),
        "Lpp_new": lpp_new,
        "T_a": float(draft_a),
        "T_b": float(draft_b),
        "T_interp_norm_m": float(draft_interp),
        "T_rel_diff_norm": float((draft_new - draft_interp) / draft_interp),
        "x_range_a": [float(v) for v in surfaces[0]["x_range"]],
        "x_range_b": [float(v) for v in surfaces[1]["x_range"]],
        "x_range_new": [float(bounds[0, 0]), float(bounds[1, 0])],
        "n_stations_u": n_u,
        "n_stations_valid": n_valid,
        "n_stations_invalid_a": surfaces[0]["n_invalid_stations"],
        "n_stations_invalid_b": surfaces[1]["n_invalid_stations"],
        "n_spliced_stations_a": surfaces[0]["n_spliced_stations"],
        "n_spliced_stations_b": surfaces[1]["n_spliced_stations"],
        "n_dropped_fragments_a": surfaces[0]["n_dropped_fragments"],
        "n_dropped_fragments_b": surfaces[1]["n_dropped_fragments"],
        "n_clipped_loops_a": surfaces[0]["n_clipped_loops"],
        "n_clipped_loops_b": surfaces[1]["n_clipped_loops"],
        "n_detached_stations_a": surfaces[0]["n_detached_stations"],
        "n_detached_stations_b": surfaces[1]["n_detached_stations"],
        "n_points": int(len(points)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "enclosed_volume_m3": float(mesh.volume) if mesh.is_volume else None,
        "displacement_hull_a_m3": mother_displacement[0],
        "displacement_hull_b_m3": mother_displacement[1],
        "displacement_interp_norm_m3": interp_norm,
        "displacement_rel_diff_norm": None if not interp_norm else
        float((displacement - interp_norm) / interp_norm),
        "displacement_interp_phys_m3": interp_phys,
        "displacement_rel_diff_phys": None if not interp_phys else
        float((displacement - interp_phys) / interp_phys),
        "displacement_method": "station_section_integration_z_le_T",
        "parameterization": "station_section_arc_length_two_segment_v",
        "symmetry": "full_ship_mirrored_y",
    }
    return {"points": points, "faces": faces, "grid": grid,
            "stations": grid[:, 0, 0], "meta": meta}
