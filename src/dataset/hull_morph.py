"""两船型全船（含干舷与甲板）的站位剖面融合（hull morphing）。

OpenFOAM 的 snappyHexMesh 需要完整封闭船体，所以融合船的输出是**水密全船**：
半船剖面从龙骨沿舷侧上行、经水线、过甲板边缘、横过半甲板到中线，镜像后逐站位
首尾封口。

算法（站位剖面参数化）：

1. STL → 半船曲面三角网格（只切 y = 0 对称面，**不再切设计水线**）；
2. u ∈ [0, 1] 按**每船自身曲面 x 范围**归一，共 n_u 个站位，端点内缩
   `STATION_EPS`。球鼻艏伸出 FP 的部分与艉突出都算进来，两船因此是艏尖对艏尖、
   艉端对艉端的对齐；
3. 每个站位用平面-网格交线取剖面折线（平面局部坐标 (r, z)），整理成**从龙骨到
   甲板中线**的有序路径：端点用最近邻行走串起来，多段剖面（平底船的底部分段、
   带附体的艉段）按「离路径尾端最近的端点」接到尾部，超过 `MAX_JOIN_FRACTION`
   的接不上就丢并计数；路径末端若回落到最高点以下（闭合环、挂在尾部的多余分叉）
   在最后一个最高点截断。

   **艏区改用竖直平面族**（`BOW_FAN_START`，照 ShipHullGAN 的艏部旋转剖面）：
   `x = const` 的横切平面在带球鼻艏的船型上会把 x/L ≥ 0.97 的剖面切成「干舷外飘」
   与「球鼻前缘」两条链，中间是真实凹槽，桥接后在艏部生成一堵竖直墙（dtc 的
   w = 0 回代误差 2.8% L）。因此 u ≥ `BOW_FAN_START` 的站位换成一族**都过同一根
   竖直枢轴** `(x_b, 0)` 的竖直平面：平面含 `e_z` 与水平方向
   `u_hat(θ) = (−sin θ, cos θ, 0)`，θ 从 0（就是 `x = x_b`，与前一站同一条剖面）
   按 `sin(πt/2)` 光滑加宽到 `90° − BOW_FAN_END_DEG`。这样每个站位的剖面都整体
   落在艏区内（`x ≥ x_b`、`y ≤ 0`，一个点对应一个 θ），剖面不再双重连通；截面
   仍是「u = 剖面序号、v = 剖面内弧长」的规则网格，只是格点的 x 不再固定。非艏区
   `u_hat = e_y` 就退化成原来的 `x = const` 横切平面。
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

写文件的入口是 `write_pair`：一对船型的若干权重 → 每个权重一个 STL，外加同目录
一份按 STL 文件名索引的 `morph_meta.json`。两个命令行入口
（`scripts/dataset_build/morph_hulls.py` 单对、`morph_hull_pairs.py` 批量）都只
调它，元信息 schema 只有这一处定义。
"""

import gzip
import json
from pathlib import Path

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
# 艏区起点（相对半船曲面 x 跨度）：u ≥ 该值的站位改用绕竖直枢轴旋转的竖直平面族，
# 其余站位仍是 x = const 的横切平面。取 0.9 是因为球鼻艏与干舷外飘的凹槽落在艏部
# 10% 船长里（dtc 的剖面从 x/L ≈ 0.98 起被切成两条链）
BOW_FAN_START = 0.9
# 艏区末站平面与中心面的夹角（度）。0 表示一直转到中心面：末站的平面就是 y = 0，
# 剖面即艏柱轮廓（见 `stem_profile`），x_max 是真实艏柱、L_ref 与母船一致。
# 上一轮取 3°，末站平面停在与中心面成 3° 的位置、并用平封口盖上，把
# `|y| < tan 3°·(x − x_b)` 的艏尖楔形切掉了：L_ref 比母船短 0.15%（dtc）–2.23%
# （wigley），艏柱线 x_max(z) 在艏尖亏欠 0.5%–8.2% L，排水量偏小 0.08%–0.72%。
# 往小调角度不是出路：1° 时 series60 的末站剖面窄到 6 mm，`_weld_points` 把甲板
# 中线并点后出现一条被 4 个面片共用的边（不水密）。所以末站不再平封口，改由
# 「倒数第二站 → 艏柱轮廓」的 loft 收成三角楔；实测 11 个母船全部水密、
# 单连通、L_ref 相对母船包围盒 −0.02%…+0.02%。
BOW_FAN_END_DEG = 0.0
# 艏柱轮廓的横切层数：末站的平面恰好是 y = 0，而尖艏的曲面在那条平面上与平面
# 相切（半宽在艏柱处趋于 0），`mesh_plane` 切不出交线。改成按 z = const 横切
# 母船曲面、每层取最靠前的 x——按对称性那个点就落在中心面上，也就是艏柱。
# 层心取在两端锚点之间（避开与母船底板/甲板平面重合的退化层），层间线性插值，
# 艏柱轮廓在艏区内光滑，插值误差是二阶的，20 层已远小于网格行距的离散误差。
STEM_LEVELS = 20


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
    """STL → 半船曲面网格（y ≤ 0，含干舷与甲板），面为 (M,3) 三角面。

    切在**模型自己的中心面**上：偏移取顶点到中心面的最小距离，正常船型（龙骨与甲板
    中线有落在 y = 0 上的顶点）该值就是 0，与直接切 y = 0 完全一样。series60 的 STL
    是两块壳拼的、中线留了一道 0.42 mm 的缝，按 y = 0 切什么也切不掉——半船曲面其实
    是整条闭合壳，两块船壳都在里面，艏区平面族里每张近中心面的平面都会切到那道缝的
    内壁（θ = 89.5° 时缝壁在离枢轴 0.18 m 处），选出那条假链后艏部出现 V 形缺口
    （艏区折返带 16.7% L）。切在缝壁上、再把切面平移回 y = 0，半船曲面就是干净的
    单侧壳，水平位移量等于缝的半宽（series60 0.21 mm，占半宽的 0.04%）。
    """
    mesh = _load(stl_path)
    points = np.asarray(mesh.vertices, dtype=np.float64)
    offset = float(np.abs(points[:, 1]).min())
    cut = _slice(points, np.asarray(mesh.faces, np.int64),
                 (0.0, -1.0, 0.0), (0.0, -offset, 0.0), lpp)
    cut[0][:, 1] += offset
    return cut


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


# 站位剖面拼接的候选枚举上限：链数超过它就退回贪心拼接（避免状态爆炸）
DP_MAX_CHAINS = 7
# 每个站位保留的候选路径数上限
MAX_CANDIDATES = 256
# 起点判定的 z 带宽（相对 Lpp）：低于「最低点 + 带宽」的节点都算船底，取其中 |y| 最小的
# 作起点。平底船的底板节点 z 是 1e-11~1e-5 量级的噪声/薄片，严格比大小会让起链在
# 「中线碎链」与「主剖面链」之间逐站翻转
START_BAND_FRACTION = 1e-3
# 起点必须落在「实质链」上：弧长不足该站位链总弧长 1% 的碎链不能当起点（否则艉部
# 断面上几十微米的碎链会因为 z 更低而劫持起点，整条路径只剩那一小簇点）
START_MIN_PIECE = 1e-2
# 候选路径的弧长下限（相对该站位最长候选）：更短的候选是碎片的退化路径
MIN_CANDIDATE_RATIO = 2e-2
# 端点在跨站连续性距离里的权重（相当于多少行的权重）：端点是「甲板中线」锚点，
# 只按行平均时它只占 1/79，路径沿中线缝回走、末端停在龙骨附近这种错位会被稀释
ENDPOINT_WEIGHT = 10.0


def _chain_end(chains, index, side):
    """链 `index` 的端点：side = 0 取首点，1 取末点。"""
    return chains[index][0] if side == 0 else chains[index][-1]


def _oriented(chains, index, entry):
    """按「从端点 entry 进入」的方向取链。"""
    return chains[index] if entry == 0 else chains[index][::-1]


def _start_of(chains, band, pool=None):
    """起链：船底带（最低 z 加 `band`）里 |y| 最小的端点所在链，确定性 tie-break。

    起点必须贴在中线上（龙骨线 / 平底的中线端），但不是「z 严格最小」——底板节点
    的 z 是浮点噪声或薄片厚度，严格比大小会逐站翻转起链。先取最低 z，再把
    「最低 z + band」以内的节点都当船底，在其中取 |y| 最小的端点。
    """
    allowed = set(range(len(chains))) if pool is None else set(pool)
    ends = [(index, side) for index in range(len(chains)) for side in (0, 1)
            if index in allowed]
    floor = min(_chain_end(chains, *item)[1] for item in ends)
    band_ends = [item for item in ends if _chain_end(chains, *item)[1] <= floor + band]
    return min(band_ends, key=lambda item: (abs(_chain_end(chains, *item)[0]),
                                            item[0], item[1]))


def _substantial_chains(chains):
    """弧长占比 ≥ `START_MIN_PIECE` 的链下标；全都太短时退回全部链。"""
    lengths = np.array([np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in chains])
    total = float(lengths.sum())
    if total <= 0.0:
        return list(range(len(chains)))
    keep = [index for index in range(len(chains))
            if lengths[index] >= START_MIN_PIECE * total]
    return keep or list(range(len(chains)))


def _greedy_chains(chains, tol, max_join, band):
    """贪心拼接（链数超过 `DP_MAX_CHAINS` 时的退路）：每次接离尾端最近的链。"""
    start_index, start_side = _start_of(chains, band, _substantial_chains(chains))
    path = _oriented(chains, start_index, start_side)
    used, max_gap = (start_index,), 0.0
    remaining = [index for index in range(len(chains)) if index != start_index]
    while remaining:
        best = None
        for index in remaining:
            for entry in (0, 1):
                distance = float(np.linalg.norm(_chain_end(chains, index, entry) - path[-1]))
                if best is None or distance < best[0]:
                    best = (distance, index, entry)
        distance, index, entry = best
        if distance > max_join:
            break
        path = np.vstack([path, _oriented(chains, index, entry)])
        used += (index,)
        max_gap = max(max_gap, distance)
        remaining.remove(index)
    return [{"points": path, "used": used, "max_gap": max_gap, "total_gap": max_gap}]


def _assemble_chains(chains, tol, max_join, band):
    """链集合 -> 候选拼接路径（起点固定为「最低 (z, |y|) 端」，只向尾端生长）。

    起点取所有链端点里**量化后的 z 最小、再 |y| 最小**的那个。z 必须量化：平底船的
    底板节点 z 都是 ~1e-11 量级的不同噪声值，直接比大小会让起链在「中线小碎链」与
    「主剖面链」之间逐站翻转，路径随之在「正常 / 底板走两遍 / 上部被截断」之间跳变，
    融合船的同一行格点于是跨站不连续（dtc 的折返带就是这么来的）。

    之后在「已用链集合 × 当前所在链 × 当前端点」上做 DP，状态里保留拼接缺口
    （先比最大缺口、再比总缺口）最小的那条路径。枚举子集是必需的：分段平底给出的
    重叠碎链有时应当整条丢掉，而不是硬接上去。

    返回按 (最大缺口, 总缺口, 用链数, 用链顺序) 升序的候选列表，每个是 dict：
    `points`（(m,2) 的 (y, z)）、`used`（用到的链下标）、`max_gap`、`total_gap`。
    顺序只由输入决定，不含浮点噪声驱动的分支，因此可复现。
    """
    if not chains:
        return []
    tol = max(float(tol), np.finfo(float).tiny)
    gap_limit = max_join
    if len(chains) > DP_MAX_CHAINS:
        return _greedy_chains(chains, tol, gap_limit, band)

    start_index, start_side = _start_of(chains, band, _substantial_chains(chains))
    states = {1 << start_index: {
        (start_index, 1 - start_side): {
            "max_gap": 0.0, "total_gap": 0.0,
            "points": _oriented(chains, start_index, start_side),
            "used": (start_index,)}}}

    count = len(chains)
    for size in range(1, count):
        for mask in [m for m in states if bin(m).count("1") == size]:
            for (last, end_bit), item in list(states[mask].items()):
                tip = item["points"][-1]
                for index in range(count):
                    if mask >> index & 1:
                        continue
                    for entry in (0, 1):
                        gap = float(np.linalg.norm(_chain_end(chains, index, entry) - tip))
                        if gap > gap_limit:
                            continue
                        max_gap = max(item["max_gap"], gap)
                        total_gap = item["total_gap"] + gap
                        key = (mask | 1 << index, index, 1 - entry)
                        slot = states.setdefault(key[0], {})
                        old = slot.get(key[1:])
                        if old is not None and (
                                (old["max_gap"], old["total_gap"]) <= (max_gap, total_gap)):
                            continue
                        slot[key[1:]] = {
                            "max_gap": max_gap, "total_gap": total_gap,
                            "points": np.vstack([item["points"],
                                                 _oriented(chains, index, entry)]),
                            "used": item["used"] + (index,)}

    candidates = [item for slot in states.values() for item in slot.values()]
    candidates.sort(key=lambda item: (item["max_gap"], item["total_gap"],
                                      len(item["used"]), item["used"]))
    return candidates[:MAX_CANDIDATES]


def _truncate_path(path, mode, tol, top_tol):
    """按终止规则截断路径。

    * `seam`：截到「中线节点里 z 最高的那个」——正常船型就是甲板中线；舷墙/舷弧比
      甲板高的船型（frigate、kvlcc2）也不会因为最高点落在舷边而把甲板切掉。
    * `top`：截到最后一个最高点（剖面上没有中线节点时）。
    * `deck`：先按 `seam` 截断，若末端不在中线上，再用一条水平直线补到中线（甲板面
      缺失时用，和 kvlcc2「剖面止于舷边、由镜像条带补薄甲板」是同一套处理）。
    * `as_is`：不截断。
    """
    if mode == "deck":
        base = _truncate_path(path, "seam", tol, top_tol)
        if base.shape[0] >= 2 and abs(base[-1, 0]) > tol and base[-1, 1] > base[0, 1]:
            return np.vstack([base, np.array([[0.0, base[-1, 1]]])])
        return base
    if mode == "as_is":
        return path
    if mode == "seam":
        seam = np.flatnonzero(np.abs(path[:, 0]) <= tol)
        seam = seam[seam > 0]
        if len(seam) > 0:
            stop = int(seam[np.argmax(path[seam, 1])])
            return path[:stop + 1]
        mode = "top"
    stop = int(np.flatnonzero(path[:, 1] >= path[:, 1].max() - top_tol)[-1])
    return path[:stop + 1]


def _strip_seam(path, tol):
    """去掉路径内部贴在中线上（|y| ≤ tol）的点，首尾保留。

    半船剖面只应在**两端**碰到中线（龙骨点与甲板中线）。中段还出现一整段 y ≈ 0 的
    竖壁，是两半船体拼接留下的缝（series60 的 0.4 mm 缝），或者某几个站位独有的
    退化碎面；它会把弧长吃掉一大截，让同一行格点在相邻站位之间跳变。这里把这种
    内部竖壁从路径里摘掉（几何上等价于「从龙骨直接连到离开中线的那一点」），是否
    采用交给跨站连续性判断：真正的中心线呆木（skeg）在邻站同样存在，会一致地选回
    带竖壁的那条。
    """
    if path.shape[0] < 3:
        return path
    interior = np.abs(path[1:-1, 0]) <= tol
    if not interior.any():
        return path
    keep = np.r_[True, ~interior, True]
    return path[keep]


def _straddles(points, draft):
    """折线或线段组是否跨越 z = draft。

    输入是 (m,2) 的剖面路径或 (n,2,2) 的交线段组，取第 1 列（z）判断相邻点是否
    落在 draft 两侧。
    """
    z = np.asarray(points, dtype=np.float64)[..., 1]
    return bool(np.any(np.diff(z >= draft, axis=-1) != 0))


def _station_paths(segments, tol, min_arc, max_join, top_tol, draft=None, band=None):
    """一个站位的切片交线 -> 候选路径列表（确定性顺序）+ 链数诊断。

    `draft` 给定时做**水线过滤**：切片本身穿过 z = draft 时，只保留真的含有该交点
    的候选 —— 禁止把水线所在的段落截断掉，v = 1 必须来自剖面与 z = draft 的真实
    交点。

    每个候选是 dict：`points`、`mode`（终止规则）、`used`、`max_gap`、`total_gap`、
    `n_dropped`、`n_detached`、`n_clipped`。剖面为空时返回 ([], 诊断)。
    """
    info = {"n_chains": 0, "n_dropped": 0, "n_clipped": 0, "n_detached": 0}
    if len(segments) == 0:
        return [], info
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    # 丢掉退化交线（细长碎片面与平面近乎相切）
    seg = seg[np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1) > tol]
    if len(seg) == 0:
        return [], info

    nodes, label = _weld(seg.reshape(-1, 2), tol)
    start, end = label[0::2], label[1::2]
    ok = start != end
    start, end = start[ok], end[ok]
    if len(start) == 0:
        return [], info

    chains = _chains(nodes, start, end)
    info["n_chains"] = len(chains)
    if not chains:
        return [], info

    if band is None:
        band = START_BAND_FRACTION * tol / WELD_FRACTION
    section_crosses = draft is not None and _straddles(seg, draft)
    candidates, seen = [], set()
    threshold = min_arc
    for item in _assemble_chains(chains, tol, max_join, band):
        for mode in ("seam", "deck", "top", "as_is"):
            path = _truncate_path(item["points"], mode, tol, top_tol)
            if path.shape[0] < 2:
                continue
            if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < min_arc:
                continue
            if section_crosses and not _straddles(path, draft):
                continue
            missing = [index for index in range(len(chains)) if index not in item["used"]]
            detached = 0
            for index in missing:
                near = min(float(np.linalg.norm(_chain_end(chains, index, side) - endpoint))
                           for side in (0, 1) for endpoint in (path[0], path[-1]))
                detached += int(near > max_join)
            variants = [(mode, path)]
            stripped = _strip_seam(path, band)
            if stripped.shape[0] != path.shape[0]:
                variants.append((mode + "+seamstrip", stripped))
            for variant_mode, variant_path in variants:
                key = variant_path.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "points": variant_path, "mode": variant_mode, "used": item["used"],
                    "max_gap": item["max_gap"], "total_gap": item["total_gap"],
                    "n_dropped": len(missing), "n_detached": detached,
                    "n_clipped": int(variant_path.shape[0] < item["points"].shape[0])})
    if candidates:
        best = max(np.linalg.norm(np.diff(c["points"], axis=0), axis=1).sum()
                   for c in candidates)
        threshold = max(min_arc, MIN_CANDIDATE_RATIO * best)
        candidates = [c for c in candidates
                      if np.linalg.norm(np.diff(c["points"], axis=0), axis=1).sum()
                      >= threshold]
    return candidates, info


def _path_rows(path, draft, n_v, n_v2):
    """剖面路径 -> 与站位网格同一口径的行点（(n_v + n_v2 - 1, 2)）。

    行号就是两段弧长（龙骨 → 水线 → 甲板中线）的等分位，也就是「同一行格点」的
    定义。剖面整条落在水线一侧（艏艉尖端）时退回单段弧长均分。
    """
    n_row = n_v + n_v2 - 1
    fraction = _waterline_fraction(path, draft)
    split = None if fraction is None else _split_by_fraction(path, fraction)
    if split is None:
        return _resample(path, n_row)
    first, second = split
    return np.vstack([_resample(first, n_v), _resample(second, n_v2)[1:]])


def _profile_distance(first, second, weight=ENDPOINT_WEIGHT):
    """两份剖面摘要（行点 + 端点）的加权 L2 距离。

    用**行口径**而不是等弧长口径比较：折返带的判据本来就是「同一行相邻站位的位移」，
    而甲板、舷侧这些短弧长段落按等弧长比较会被稀释（一条 5% 弧长的甲板错位在 201 个
    等弧长采样点里只占十几点），按行比较则每行等权。端点再按 `weight` 行的权重加进去。
    """
    rows = np.sum((first["rows"] - second["rows"]) ** 2, axis=1)
    count = len(rows)
    return float((rows.sum() + weight * np.sum((first["end"] - second["end"]) ** 2))
                 / (count + weight))


def _bridge_cost(candidate, arc):
    """拼接缺口带来的偏离曲面代价（与 `_profile_distance` 同量纲的均方长度）。

    带缺口的候选在缺口处是两点直连，那一小段（约占弧长 gap/arc）离真实曲面
    gap/2 远。不罚它的话，行距只差百分之几的缺口候选会被选中：workboat 的
    θ = 85.6° 站有个 0.25 m 的缺口候选，行距比无缺口的小 4%，选中后艏部回代偏差
    从 0.06% 涨到 0.90% L。小的缺口（拼接噪声量级）罚项可以忽略，不影响原来的选链。
    """
    gap = float(candidate["max_gap"])
    if gap <= 0.0 or arc <= 0.0:
        return 0.0
    return 0.25 * gap * gap * min(gap / arc, 1.0)


def _select_paths(sections, draft, n_v, n_v2):
    """按邻站连续性给每个站位选一条候选路径。

    种子取「单链（没有拼接歧义）且弧长最长」的站位（正常船型是船中），再向两端逐站
    推进：每站从候选里选与已定邻站**行点**距离最小的一条（同距离时取弧长更长者、
    再取候选序号，保证确定性）。同一行的格点因此沿 u 连续，不会在「正常 / 底板走
    两遍 / 上部被截断」之间跳变。
    """
    n = len(sections)

    def arc(candidate):
        return float(np.linalg.norm(np.diff(candidate["points"], axis=0), axis=1).sum())

    def summary(candidate):
        if "summary" not in candidate:
            candidate["summary"] = {
                "rows": _path_rows(candidate["points"], draft, n_v, n_v2),
                "end": candidate["points"][-1]}
        return candidate["summary"]

    chosen = [None] * n
    # 种子只从「切片只有一条链」的站位里选：多链站的候选里必有一条超长的（底板走两遍
    # 之类），用它当参照会把整段剖面带偏
    pool = [index for index, item in enumerate(sections)
            if item["candidates"] and item["info"]["n_chains"] == 1]
    if not pool:
        pool = [index for index, item in enumerate(sections) if item["candidates"]]
        if not pool:
            return chosen
        fewest = min(sections[index]["info"]["n_chains"] for index in pool)
        pool = [index for index in pool if sections[index]["info"]["n_chains"] == fewest]

    seed = max(pool, key=lambda index: (max(arc(c) for c in sections[index]["candidates"]),
                                        -index))
    # 种子的候选优先取「终止在中线（甲板中线）」的那类：剖面闭合环在某些船型上会多走
    # 一段中线缝回来，弧长更长，若按最长来挑种子，整套网格都会以「末端停在龙骨附近」
    # 为参照，端部行号随之跳变
    candidates = sections[seed]["candidates"]
    canonical = [k for k, item in enumerate(candidates) if item["mode"] in ("seam", "deck")]
    best = max(canonical or range(len(candidates)),
               key=lambda k: (arc(candidates[k]), -k))
    chosen[seed] = best
    for step in (1, -1):
        reference = summary(sections[seed]["candidates"][best])
        for index in range(seed + step, n if step > 0 else -1, step):
            candidates = sections[index]["candidates"]
            if not candidates:
                continue
            pick = min(range(len(candidates)),
                       key=lambda k: (_profile_distance(summary(candidates[k]), reference)
                                      + _bridge_cost(candidates[k], arc(candidates[k])),
                                      -arc(candidates[k]), k))
            chosen[index] = pick
            reference = summary(candidates[pick])
    return chosen


def _section_path(segments, tol, min_arc, max_join, top_tol, band=None):
    """站位剖面交线 -> 从龙骨端到甲板中线的有序路径（单站简化入口）。

    不做跨站连续性选择（那要用整条船的站位序列，见 `hull_station_grid`）：取 DP 里
    「用到的链最多、最大缺口最小」的那条，并优先 `seam` 终止规则。给
    `check_hull_stl.py` 的单站检查与诊断用。

    返回 (路径 (m,2) 的 (y,z), 信息 dict)；剖面为空时返回 (None, 信息)。
    """
    candidates, info = _station_paths(segments, tol, min_arc, max_join, top_tol,
                                      band=band)
    if not candidates:
        return None, info
    chosen = max(candidates, key=lambda item: (len(item["used"]), item["mode"] == "seam",
                                               -item["max_gap"], -item["total_gap"]))
    info["n_dropped"] = chosen["n_dropped"]
    info["n_detached"] = chosen["n_detached"]
    info["n_clipped"] = chosen["n_clipped"]
    return chosen["points"], info


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
    if s[-1] <= 0.0:
        return np.tile(np.asarray(path, dtype=np.float64)[0], (n, 1))
    # 弧长单调不减：重复值（零长段）会让 np.interp 的 xp 非严格递增
    unique = np.r_[True, np.diff(s) > 0]
    s = s[unique] / s[-1]
    v = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(v, s, path[unique, 0]),
                            np.interp(v, s, path[unique, 1])])


def _resample_knot(path, n, apex=None):
    """按弧长把剖面折线重采样成 n 个点；`apex`（顶点下标）给定时它必定是一行。

    弧长均分的行距在剖面折线的**尖角**处会把尖角削掉最多半行：艏柱轮廓的最前点
    （艏柱尖）就是这样一个尖角，而它就是 `L_ref` 的来源。给 `apex` 时把折线在
    该顶点处切成两段、各自等分，（近似）保持原来的行距。
    """
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    if apex is None or not 0 < apex < len(path) - 1 or s[-1] <= 0.0:
        return _resample(path, n)
    keep = int(np.clip(round(s[apex] / s[-1] * (n - 1)), 1, n - 2))
    return np.vstack([_resample(path[:apex + 1], keep + 1),
                      _resample(path[apex:], n - keep)[1:]])


def station_frames(x_lo, x_hi, n_u, eps=STATION_EPS):
    """每站的剖面平面：`origin` + 平面内水平方向 `u_hat`（含 `e_z`）+ 旋转角 `theta`。

    非艏区（u < `BOW_FAN_START`）是 `x = const` 的横切平面：`origin = (x, 0, 0)`、
    `u_hat = e_y`，点 = `origin + r·u_hat + z·e_z`，局部坐标 (r, z) 就是原来的
    (y, z)。艏区是一族**都过同一根竖直枢轴** `(x_b, 0)` 的竖直平面：
    `u_hat(θ) = (−sin θ, cos θ, 0)`，点 = `(x_b + |r| sin θ, r cos θ, z)`（r ≤ 0），
    因此整个剖面都落在 `x ≥ x_b`、`y ≤ 0` 的艏区里，一个点对应一个 θ，剖面不会
    双连通。θ = 0 的第一站就是 `x = x_b`，与前一站共用同一条剖面。

    θ 按 `sin(πt/2)` 从 0 加到 `90° − BOW_FAN_END_DEG`：艏部甲板边线是往艏尖收敛的
    曲线，等弧长分点要求越靠近艏尖站位越密（验算 dtc / kcs / containership 的
    甲板边线，这条排布给出的甲板边 |y| 步长几乎是常数）。
    """
    u = np.linspace(0.0, 1.0, n_u)
    x = x_lo + (eps + u * (1.0 - 2.0 * eps)) * (x_hi - x_lo)
    origins = np.column_stack([x, np.zeros(n_u), np.zeros(n_u)])
    u_hat = np.tile(np.array([0.0, 1.0, 0.0]), (n_u, 1))
    theta = np.zeros(n_u)
    hit = np.flatnonzero(u >= BOW_FAN_START)
    fan_from = int(hit[0]) if len(hit) else n_u
    if n_u - fan_from >= 2:
        angle_max = np.deg2rad(90.0 - BOW_FAN_END_DEG)
        t = np.linspace(0.0, 1.0, n_u - fan_from)
        theta[fan_from:] = angle_max * np.sin(0.5 * np.pi * t)
        origins[fan_from:, 0] = x[fan_from]
        u_hat[fan_from:, 0] = -np.sin(theta[fan_from:])
        u_hat[fan_from:, 1] = np.cos(theta[fan_from:])
    elif fan_from < n_u:  # 只剩一站：无法形成平面族，留在横切平面上
        fan_from = n_u
    # cos(π/2) 是 6.1e-17 而不是 0：末站一旦转到中心面，就把这个残量清零，那一站
    # 整站落在 y = 0 上（`on_midline` 靠这一行成立）
    u_hat[np.abs(u_hat) < 1e-15] = 0.0
    on_midline = u_hat[:, 1] == 0.0
    midline = int(np.flatnonzero(on_midline)[0]) if on_midline.any() else None
    return {"u": u, "x": x, "origins": origins, "u_hat": u_hat,
            "normals": np.column_stack([u_hat[:, 1], -u_hat[:, 0],
                                        np.zeros(n_u)]),
            "theta": theta, "fan_from": fan_from, "midline_station": midline,
            "on_midline": on_midline,
            "x_bow_fan": float(origins[fan_from, 0]) if fan_from < n_u else None}


def local_section(lines, origin, direction):
    """平面-网格交线 -> 平面局部坐标 (r, z) 的线段组，形状 (n, 2, 2)。

    `r` 是沿 `direction` 的坐标（非艏区就是 y ≤ 0 那一半），`z` 是高度。
    """
    if len(lines) == 0:
        return np.zeros((0, 2, 2))
    points = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    r = (points - np.asarray(origin, dtype=np.float64)) @ np.asarray(direction)
    return np.column_stack([r, points[:, 2]]).reshape(-1, 2, 2)


def stem_profile(bow_mesh, x_b, lower, upper, levels=STEM_LEVELS):
    """母船艏区网格 -> 艏区末站的剖面（中心面 y = 0 上的艏柱轮廓）。

    末站的平面就是 y = 0，剖面是中心面剖线：从龙骨锚点沿艏部底面（龙骨线）向前到
    前踵，再沿艏柱上行到艏尖，最后沿甲板中线回到甲板锚点。`lower` / `upper` 是艏区
    首站剖面的两端（龙骨点与甲板中线点的局部坐标 (r, z)）——它们就是竖直枢轴与曲面
    的两个交点，所有艏区剖面都过这两点。

    这条线不能用 `mesh_plane` 切 `y = 0` 得到：尖艏的半宽在艏柱处趋于 0，曲面与那张
    平面相切，切不出交线。两条支各用自己合适的横切方向：

    * **艏柱**用 z = const 横切，每层取最靠前的 x——按对称性那个点落在中心面上；
    * **龙骨线**用 x = const 横切，每层取最低的 z。z 层在平板底板与甲板平面上切片
      退化，反过来 x 层在艏柱尖端退化，所以不能只用一种方向。

    返回局部坐标 (m, 2) 的折线，r = x_b − x ≤ 0（θ = 90° 时 u_hat = (−1, 0, 0)，
    点 = (x_b − r, 0, z)）。
    """
    vertices = np.asarray(bow_mesh.vertices, dtype=np.float64)
    # 最前点：x 最大，x 相同的取 z 最高（竖直艏柱的整条艏柱线都是最前点，取顶端那个）
    index = int(np.lexsort((-vertices[:, 2], -vertices[:, 0]))[0])
    apex_x, apex_z = float(vertices[index, 0]), float(vertices[index, 2])
    z_lo, z_hi = float(lower[1]), float(upper[1])
    # 层心铺在「龙骨锚点 → 上界」之间，上界取甲板锚点与母船最前点高度的较大者：
    # 艏柱后倾、最高点即最前点的船型（frigate、workboat）甲板中线那一段因此是一段
    # 回落的直线。底层不取（与母船底板平面重合，切片退化），顶层取（艏柱上端就在
    # 那里，不取等于把艏柱顶削掉：wigley 的竖直艏柱丢 4.6% L）；最前点单独插进去，
    # 它就是 L_ref 的来源
    z_top = max(z_hi, apex_z)
    heights = np.unique(np.r_[np.linspace(z_lo, z_top, levels + 2)[1:], apex_z])
    stem = []
    for height in heights:
        if height == apex_z:  # 艏柱尖那一层与甲板/底板相切，切片退化，直接用顶点
            stem.append([x_b - apex_x, apex_z])
            continue
        lines = mesh_plane(bow_mesh, (0.0, 0.0, 1.0), (0.0, 0.0, float(height)))
        if len(lines) == 0:
            continue
        forward = float(np.asarray(lines, dtype=np.float64)[:, :, 0].max())
        stem.append([x_b - forward, float(height)])

    # 龙骨线：从枢轴 x_b 到前踵（最低那层艏柱点的 x），逐层取该 x 处最低的 z
    forefoot = x_b - stem[0][0] if stem else x_b
    keel = []
    for x in np.linspace(x_b, forefoot, levels + 2)[1:-1]:
        lines = mesh_plane(bow_mesh, (1.0, 0.0, 0.0), (float(x), 0.0, 0.0))
        if len(lines) == 0:
            continue
        keel.append([x_b - float(x),
                     float(np.asarray(lines, dtype=np.float64)[:, :, 2].min())])

    ends = np.array([lower, upper], dtype=np.float64)
    middle = np.array(keel + stem, dtype=np.float64).reshape(-1, 2)
    return np.vstack([ends[:1], middle, ends[1:]])


def hull_station_grid(stl_path, lpp, draft, n_u=160, n_v=60, n_v2=20, eps=STATION_EPS):
    """单船的站位剖面网格（半船，含干舷与甲板）。

    返回 dict：`points`（(n_u, n_row, 3)，全三维、格点的 x 在艏区不再固定，坐标已除
    Lpp，n_row = n_v + n_v2 - 1）、`u`（(n_u,) 站位参数 0→艉、1→艏）、`theta`
    （(n_u,) 站平面绕竖直枢轴的旋转角，非艏区为 0）、`stations`（(n_u,) 站位掩码）、
    `valid`（(n_u, n_row) 格点掩码）、`x_range`（半船曲面的物理 x 范围）、`x_bow_fan`
    （艏区起点 x，无艏区时 None）、`n_v`/`n_v2`（水线行是第 n_v - 1 行）、
    `n_waterline_stations`（剖面真的穿过设计水线的站位数）以及拼接、截断、丢弃的
    站位数等诊断计数。

    三遍：

    1. 逐站求交线、拆链，并**枚举候选路径**（不同的选链组合 × 三种终止规则，见
       `_station_paths`）。穿过水线的切片只保留真的含该交点的候选，水线段不会被截掉。
    2. **按邻站连续性选路径**（`_select_paths`）：从船中那条单链站位出发向两端推进，
       每站取与已定邻站路径 L2 距离最小的候选。剖面可能被三角化切成多条链、碎链
       有时该丢，逐站独立决策会让同一行格点跨站跳变（折返带）；用邻站做参照后，
       同一行的格点沿 u 连续。
    3. 取各站位剖面与 v = 1 的弧长占比，按占比重采样成公共行网格。占比要单独算，
       是因为艏艉尖端的剖面可能整条落在水线以上或以下（frigate 艏柱、kvlcc2 球鼻
       艏），这些站位没有水线交点，用邻站的占比补上，v 映射才沿 u 连续。

    站平面由 `station_frames` 给出：艏区（u ≥ `BOW_FAN_START`）是绕竖直枢轴旋转的
    竖直平面族，其余是 `x = const` 的横切平面。两种情况的剖面都在平面局部坐标
    (r, z) 里提取与重采样，再还原成三维点，因此 `_station_paths` / `_chains` /
    `_select_paths` / `_resample` 与邻站连续性选链的行为完全一致。

    平面族末站的平面是中心面 `y = 0`（`BOW_FAN_END_DEG = 0`），它的剖面不是切片
    得到的而是 `stem_profile` 单独构的艏柱轮廓（见那里的说明）；那一站整站落在
    `y = 0` 上，镜像时与中线行共用顶点，所以不做 `SEAM_OFFSET_FRACTION` 的推开。
    """
    points_, faces = half_ship_surface(stl_path, lpp)
    mesh = trimesh.Trimesh(points_, faces, process=False)
    x_lo, x_hi = float(points_[:, 0].min()), float(points_[:, 0].max())
    n_row = n_v + n_v2 - 1
    min_arc = MIN_ARC_FRACTION * lpp
    tol = WELD_FRACTION * lpp
    frames = station_frames(x_lo, x_hi, n_u, eps)
    midline = frames["midline_station"]
    on_midline = frames["on_midline"]
    # 艏柱轮廓要按 z 逐层横切的只是艏区那一小块网格：整条船的三角面里 x ≥ x_b 的
    # 只占几个百分点，先筛出来再切，`STEM_LEVELS` 层切片才不至于拖慢单船网格
    bows = None
    if midline is not None:
        forward = points_[faces][:, :, 0].max(axis=1)
        bows = trimesh.Trimesh(points_, faces[forward >= frames["x_bow_fan"]],
                               process=False)

    sections = []
    for index in range(n_u):
        if index == midline:
            # 末站平面与中心面重合，切片退化，由下面的 `stem_profile` 单独给剖面
            sections.append({"candidates": [],
                             "info": {"n_chains": 1, "n_dropped": 0,
                                      "n_clipped": 0, "n_detached": 0}})
            continue
        lines = mesh_plane(mesh, tuple(frames["normals"][index]),
                           tuple(frames["origins"][index]))
        segments = local_section(lines, frames["origins"][index],
                                 frames["u_hat"][index])
        candidates, info = _station_paths(segments, tol, min_arc,
                                          MAX_JOIN_FRACTION * lpp,
                                          TOP_TOL_FRACTION * lpp, draft,
                                          START_BAND_FRACTION * lpp)
        sections.append({"candidates": candidates, "info": info})

    choice = _select_paths(sections, draft, n_v, n_v2)

    paths, fractions, counts = [None] * n_u, np.full(n_u, np.nan), []
    for index, section in enumerate(sections):
        candidates, info = section["candidates"], dict(section["info"])
        if choice[index] is None:
            counts.append(info)
            continue
        candidate = candidates[choice[index]]
        info.update({key: candidate[key]
                     for key in ("n_dropped", "n_detached", "n_clipped")})
        counts.append(info)
        path = candidate["points"]
        if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < min_arc:
            continue
        paths[index] = path
        fraction = _waterline_fraction(path, draft)
        if fraction is not None:
            fractions[index] = fraction

    # 末站的艏柱轮廓：两端锚点取艏区首站剖面的两端（所有艏区剖面都过那两点）
    if midline is not None:
        anchor = paths[frames["fan_from"]]
        if anchor is not None:
            paths[midline] = stem_profile(bows, frames["x_bow_fan"], anchor[0], anchor[-1])
            fraction = _waterline_fraction(paths[midline], draft)
            if fraction is not None:
                fractions[midline] = fraction

    known = np.flatnonzero(~np.isnan(fractions))
    if len(known) == 0:
        raise ValueError(f"{stl_path} 的剖面都不穿过设计水线 z = {draft}")
    # 缺口（水线以上/以下的尖端站位）用最近的有效站位线性补，两侧都没有就取端点
    fractions = np.interp(np.arange(n_u), known, fractions[known])

    grid = np.zeros((n_u, n_row, 3), dtype=np.float64)
    valid = np.zeros(n_u, dtype=bool)
    for index in range(n_u):
        if paths[index] is None:
            continue
        split = _split_by_fraction(paths[index], fractions[index])
        if split is None:
            continue
        first, second = split
        if (np.linalg.norm(np.diff(first, axis=0), axis=1).sum() < min_arc
                or np.linalg.norm(np.diff(second, axis=0), axis=1).sum() < min_arc):
            continue
        # 末站（艏柱轮廓）的最前点必须在行上，否则弧长均分的行距会把艏尖削掉
        # 最多半行（实测 kcs 0.17%、npl_round_bilge 0.28% L_ref）
        apex = int(np.argmin(paths[index][:, 0])) if index == midline else None
        cut = len(first) - 1
        profile = np.vstack([
            _resample_knot(first, n_v, apex if apex is not None and apex < cut else None),
            _resample_knot(second, n_v2,
                           None if apex is None or apex < cut else apex - cut + 1)[1:]])
        # 内部行（龙骨行与甲板中线行除外）贴在中线上的推开一点，避免镜像重合。
        # 中心面那一站（末站）整站就在 y = 0 上，镜像时本来就与中线行共用顶点，
        # 推开反而会在艏尖造出一薄壁
        if not on_midline[index]:
            inner = slice(1, len(profile) - 1)
            too_close = np.abs(profile[inner, 0]) < SEAM_OFFSET_FRACTION
            profile[inner, 0] = np.where(too_close, -SEAM_OFFSET_FRACTION, profile[inner, 0])
        # 平面局部坐标 (r, z) -> 三维：origin + r·u_hat + z·e_z（origin 的 y、z 都是 0）
        origin, direction = frames["origins"][index], frames["u_hat"][index]
        grid[index, :, 0] = origin[0] + profile[:, 0] * direction[0]
        grid[index, :, 1] = profile[:, 0] * direction[1]
        grid[index, :, 2] = profile[:, 1]
        valid[index] = True

    # 艏区平面族的所有剖面都过同一根竖直枢轴，所以每站的龙骨行与甲板中线行按构造
    # 就是枢轴与曲面的那两个交点——它们必须**无条件**对齐到艏区首站（θ = 0，就是
    # x = x_b 的横切剖面）的那两行。求交的浮点噪声是纳米级，但起链点稍稍偏开枢轴
    # （series60 的 θ = 89.5° 站偏 1.4 mm）会让这一行的折线出现一个小尖刺，镜像后
    # 那个顶点上出现被 4 个面片共用的边（不水密）
    fan = np.flatnonzero(frames["theta"] > 0.0)
    if len(fan) and valid[frames["fan_from"]]:
        for row, reference in zip((0, -1), grid[frames["fan_from"], [0, -1]]):
            grid[fan, row] = reference

    infos = [info for info, path in zip(counts, paths) if path is not None]
    return {"points": grid / lpp, "valid": np.repeat(valid[:, None], n_row, axis=1),
            "stations": valid, "x_range": (x_lo, x_hi), "n_v": n_v, "n_v2": n_v2,
            "n_row": n_row, "u": frames["u"], "theta": frames["theta"],
            "x": frames["x"], "x_bow_fan": frames["x_bow_fan"],
            "n_bow_fan_stations": int(n_u - frames["fan_from"]),
            "n_stem_stations": int(midline is not None),
            "bow_fan_end_deg": BOW_FAN_END_DEG,
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


def _weld_points(points, faces, tol):
    """把距离 ≤ tol 的顶点并成一个，并删掉退化/重复面与孤立顶点。

    艏区平面族的所有剖面都过同一根竖直枢轴，因此龙骨行与甲板中线行在所有艏区站位
    上是同一个点（浮点噪声量级）。这使相邻两站的那两条四边形退化成零面积三角形；
    不并点的话 STL 回读（会自己 merge vertices）会撞上重复顶点，零面积面也留了下来。
    """
    count = len(points)
    if count == 0:
        return points, faces
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return points, faces
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(count, count))
    n_node, label = connected_components(graph, directed=False)
    first = np.full(n_node, count, dtype=np.int64)
    np.minimum.at(first, label, np.arange(count))
    return _compact(points[first], label[faces])


def full_ship_mesh(grid):
    """规则站位网格 (n_u, n_row, 3) -> 水密全船三角网格，返回 (points, faces)。

    镜像 y → −y 时中线两行共用顶点（不重复造点），侧表面沿剖面环首尾相接，首尾
    两个站位的截面用 `_cap_rows` 的拉链条带封口，因此得到的网格主线（每条边恰好
    两个面片）闭合。艏区剖面族的枢轴行会重合，最后由 `_weld_points` 并点并清掉
    退化成零面积的三角形。
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
    points, faces = _weld_points(points, faces, WELD_FRACTION * 1e-3 * _grid_scale(grid))

    # 绕向整体校验：水密网格的有向体积必须为正，否则把全部面片翻过来
    tri = points[faces]
    volume = np.einsum("ij,ij->i", tri[:, 0],
                       np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    if volume < 0:
        faces = faces[:, ::-1]
    return points, faces


def _grid_scale(grid):
    """网格的空间尺度（用于把比例容差变成几何容差）。"""
    return float(np.abs(grid).max())


def mesh_displacement(points, faces, draft):
    """封闭全船网格在 z ≤ draft 下的体积（平面裁剪 + 散度定理）。

    艏区改成旋转平面族后，站位不再垂直于 x，「纵剖面面积 × dx」不再成立（截面是
    斜的、且一行格点的 x 不固定）。这里回到体积的定义：把封闭网格用 z = draft 平面
    裁开，对剩下的开口曲面做散度定理积分，原点取在裁剪平面上（`q = (0, 0, draft)`）
    —— 平面的封闭盖的贡献恰好为 0，所以不需要三角化引擎就能得到准确的
    水下体积。融合船与母船只用同一套网格（各自的站位重建）+ 同一个函数，可比性不变。
    """
    if len(faces) == 0:
        return None
    mesh = trimesh.Trimesh(np.asarray(points, dtype=np.float64),
                           np.asarray(faces, dtype=np.int64), process=False)
    cut = trimesh.intersections.slice_mesh_plane(
        mesh, (0.0, 0.0, -1.0), (0.0, 0.0, float(draft)), cap=False)
    vertices = np.asarray(cut.vertices, dtype=np.float64)
    triangles = np.asarray(cut.faces, dtype=np.int64)
    if len(triangles) == 0:
        return 0.0
    q = np.array([0.0, 0.0, float(draft)])
    a = vertices[triangles[:, 0]] - q
    b = vertices[triangles[:, 1]] - q
    c = vertices[triangles[:, 2]] - q
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def grid_displacement(grid, draft):
    """站位网格 -> 全船封闭网格 -> 水线以下体积（z ≤ draft）。

    `grid` 是 (n_u, n_row, 3) 的物理尺度网格，只含连续有效站位。时刻注意：艏区的
    格点不在 x = const 的平面上，所以不再能按 `面积 × dx` 积分，改由
    `full_ship_mesh` 的封闭网格做平面裁剪求体积（见 `mesh_displacement`）。
    """
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape[0] < 2:
        return None
    points, faces = full_ship_mesh(grid)
    return mesh_displacement(points, faces, draft)


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

    # 排水体积：融合船与两母船用同一套站位网格（各自的站位重建）+ 同一个函数
    # （封闭网格在 z ≤ T 下的裁剪体积），保证可比
    displacement = mesh_displacement(points, faces, draft_new)
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
        "n_bow_fan_stations": surfaces[0]["n_bow_fan_stations"],
        "bow_fan_start": BOW_FAN_START,
        "bow_fan_end_deg": BOW_FAN_END_DEG,
        "x_bow_fan_a": surfaces[0]["x_bow_fan"],
        "x_bow_fan_b": surfaces[1]["x_bow_fan"],
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
        "displacement_method": "closed_mesh_volume_z_le_T",
        "parameterization": "station_section_arc_length_two_segment_v",
        "symmetry": "full_ship_mirrored_y",
    }
    return {"points": points, "faces": faces, "grid": grid,
            "u": surfaces[0]["u"][station_ok], "meta": meta}


def weight_tag(weight):
    """权重写成三位整数，例 0.25 -> "025"，STL 文件名里带它。"""
    return f"{round(weight * 100):03d}"


def _number(value, spec=".4f"):
    """元信息里的量可能是 None（母船有效站位不足时排水量算不出来），打印时给 n/a。"""
    return "n/a" if value is None else format(value, spec)


def write_pair(stl_a, stl_b, lpp_a, lpp_b, draft_a, draft_b, weights, out_dir,
               pair=None, hull_a=None, hull_b=None, n_u=160, n_v=60, n_v2=20,
               previous=None, log=print):
    """融合一对船型的若干权重，写 STL 与 `morph_meta.json`，返回汇总 dict。

    `out_dir` 是该对船的目录（`dataset/morphed_hulls/<a>_<b>/`）：每个权重一个
    `<pair>_w<三位权重>.stl`，元信息按 STL 文件名索引在同一份 `morph_meta.json` 里。
    `hull_a` / `hull_b` 是船型名（默认取 STL 文件名），`pair` 是目录名（默认
    `<hull_a>_<hull_b>`）。

    `previous` 是该对船已有的 `morph_meta.json` 内容，用于增量补权重：里面已有的
    权重条目**原样保留**（它们是已有算例与数据集条目的溯源信息），只覆盖本次真的
    重算的权重。回读 STL 后不水密直接抛错，不留半成品。
    """
    hull_a = str(hull_a if hull_a is not None else Path(stl_a).stem)
    hull_b = str(hull_b if hull_b is not None else Path(stl_b).stem)
    pair = str(pair if pair is not None else f"{hull_a}_{hull_b}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = dict(previous.get("cases", {})) if previous else {}
    for weight in weights:
        result = morph_hulls(stl_a, stl_b, lpp_a, lpp_b, draft_a, draft_b, weight,
                             n_u=n_u, n_v=n_v, n_v2=n_v2)
        meta = result["meta"]
        stl_name = f"{pair}_w{weight_tag(weight)}.stl"
        path = out_dir / stl_name
        trimesh.Trimesh(result["points"], result["faces"], process=False).export(path)
        # 回读校验：STL 没有顶点共享信息，必须让 trimesh 合并顶点后再判水密
        reloaded = trimesh.load(str(path), force="mesh")
        if not reloaded.is_watertight:
            raise RuntimeError(f"{path} 回读后不水密，检查站位网格与封口")
        meta["stl"] = stl_name
        meta["stl_watertight_on_reload"] = bool(reloaded.is_watertight)
        cases[stl_name] = meta
        relative = meta["displacement_rel_diff_norm"]
        log(f"{stl_name}: {meta['n_points']} 点 {meta['n_triangles']} 面  "
            f"L_ref={_number(meta['L_ref_m'])} T={_number(meta['T_m'])} "
            f"B/2={_number(meta['half_beam_m'])} D={_number(meta['depth_m'])} "
            f"排水量={_number(meta['displacement_m3'])} "
            f"(归一化插值 {_number(meta['displacement_interp_norm_m3'])}, "
            f"{_number(None if relative is None else relative * 100, '+.2f')}%) "
            f"水密={meta['watertight']}")

    done = list(weights)
    summary = {
        "hull_a": hull_a,
        "hull_b": hull_b,
        "stl_a": str(stl_a),
        "stl_b": str(stl_b),
        "Lpp_a": lpp_a,
        "Lpp_b": lpp_b,
        "T_a": draft_a,
        "T_b": draft_b,
        "n_u": n_u,
        "n_v": n_v,
        "n_v2": n_v2,
        "weights": sorted(set(previous.get("weights", [])) | set(done)) if previous else done,
        "cases": cases,
    }
    (out_dir / "morph_meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    log(f"写出 {out_dir}")
    return summary
