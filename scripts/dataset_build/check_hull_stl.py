"""船体 STL 健康检查：连通性、水密性、内壳（隐藏面片）、站位剖面环数。

船型融合（`src/dataset/hull_morph.py`）要求母船 STL 是「外表面单壁」的：
面片全部从外部可见、每个站位剖面是单条开环折线。CAD 导出的 STL 常见两种病：
内壳（外表面里又套一层回折面，剖面变成同心双环）、附体/碎面（剖面被切成多条
互不相连的链）。这个脚本把两类问题都量化出来，可对任意 STL 重复运行。

用法:
    python3 scripts/dataset_build/check_hull_stl.py                    # 11 个经典船型
    python3 scripts/dataset_build/check_hull_stl.py --stl a.stl b.stl  # 指定文件
    python3 scripts/dataset_build/check_hull_stl.py --json out.json    # 附 JSON

检查项：

1. 拓扑：顶点/面数（顶点按位置合并后）、`is_watertight`、绕向一致、连通分量数、
   欧拉数、表面积、封闭体积。
2. 内壳：对每个采样面片，从「面心沿外法向偏移 1e-6·L」的点向 +x 发射线，数射线与
   网格的交点个数；交点数为奇数说明该点在实体内部，即这片是从外部看不见的**内壳
   面片**。射线方向取 +x 而不是法向：法向射线在近壁面处容易擦到相邻面片，+x 射线
   只需一个 (y, z) 二维网格做粗筛，判定稳定且与面片朝向无关（沿法向发射时，面片
   自身就在射线上，起点偏移量必须精确）。每片用 3 个确定性抖动各投一次、取多数。
3. 站位剖面：在 x 方向取 `--stations` 个站位，用 x = const 平面切片，端点焊接后
   数连通环数，并在 y ≤ 0 半船上按 z 水平线数交点数。单壁船型每个站位是 1 条环线
   （艏部球鼻、艉部轴支架等真实分离结构处才会 > 1），且同一 z 高度只应交 1 次；
   **同一 z 高度交 2 次以上 = 同心双壁（内壳）**，这是内壳的判定口径。
4. 湿表面：套用融合前的 y ≤ 0、z ≤ T 切片，统计多链站位（融合时会被拼接）。
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from trimesh.intersections import mesh_plane

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

from hull_morph import (  # noqa: E402
    MAX_JOIN_FRACTION, MIN_ARC_FRACTION, TOP_TOL_FRACTION, WELD_FRACTION,
    _section_path, _weld, half_wet_surface,
)

# 面心沿外法向的偏移量（相对 Lpp）：既要跳出自身面片，又不能跨过薄壁
NORMAL_OFFSET_FRACTION = 1e-6
# 射线抖动量（相对 Lpp）：射线穿过两个面片的公共棱时奇偶会翻，抖动几下取多数
RAY_JITTER_FRACTION = 2e-7
N_JITTER = 3
# 内壳采样面片数上限（大网格按等步长抽样）
MAX_RAY_FACES = 3000
# 站位剖面水平线的层数
N_LEVELS = 120


def load_mesh(path):
    """读 STL（含 .gz），按位置合并顶点（STL 是三角形汤，不合并就没有拓扑）。"""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            mesh = trimesh.load(handle, file_type="stl", force="mesh")
    else:
        mesh = trimesh.load(str(path), force="mesh")
    return mesh


class RayGrid:
    """射线求交的粗筛结构：三角形沿射线轴的垂直平面铺一张均匀网格。

    只支持沿 **单根坐标轴** 的射线，查询点到网格单元的映射因此是一次查表：每个
    单元里存所有垂直包围盒覆盖该单元的三角形。单元边长按三角形数自适应。
    """

    def __init__(self, triangles, axis=0, n_cell=192):
        self.triangles = triangles
        self.axis = axis
        # 垂直平面上的两个坐标轴
        self.plane = [i for i in range(3) if i != axis]
        flat = triangles[:, :, self.plane].reshape(-1, 2)
        low = flat.min(axis=0)
        high = flat.max(axis=0)
        self.low = low
        self.cell = max(float((high - low).max()) / n_cell, 1e-9)
        self.count = np.maximum(np.ceil((high - low) / self.cell).astype(np.int64) + 1, 1)
        corner = np.floor((triangles[:, :, self.plane] - low) / self.cell).astype(np.int64)
        i0 = np.clip(corner.min(axis=1), 0, self.count - 1)
        i1 = np.clip(corner.max(axis=1), 0, self.count - 1)
        span = i1 - i0 + 1
        repeat = span[:, 0] * span[:, 1]
        owner = np.repeat(np.arange(len(triangles)), repeat)
        start = np.r_[0, np.cumsum(repeat)[:-1]]
        offset = np.arange(int(repeat.sum())) - np.repeat(start, repeat)
        width = span[owner, 1]
        key = ((i0[owner, 0] + offset // width) * self.count[1]
               + i0[owner, 1] + offset % width)
        self.key = np.sort(key)
        self.value = owner[np.argsort(key)]

    def candidates(self, uv):
        first = int(min(max((uv[0] - self.low[0]) / self.cell, 0.0), self.count[0] - 1))
        second = int(min(max((uv[1] - self.low[1]) / self.cell, 0.0), self.count[1] - 1))
        key = first * self.count[1] + second
        lo = np.searchsorted(self.key, key, "left")
        hi = np.searchsorted(self.key, key, "right")
        return self.triangles[self.value[lo:hi]]

    def parity(self, points, lpp):
        """points (n,3) -> 沿 +axis 的射线交点数的奇偶（True = 在实体内）。

        射线穿过两个面片的公共棱时奇偶会翻，所以每个点用 N_JITTER 个确定性抖动
        各投一次票，取多数。
        """
        votes = np.zeros(len(points), dtype=np.int64)
        for jitter in range(N_JITTER):
            offset = RAY_JITTER_FRACTION * lpp * ((jitter % 2) - 0.5)
            side = RAY_JITTER_FRACTION * lpp * (jitter - (N_JITTER - 1) / 2.0)
            for index, point in enumerate(points):
                uv = (point[self.plane[0]] + offset, point[self.plane[1]] + side)
                tri = self.candidates(uv)
                if len(tri) and _crossings(tri, point, self.axis, self.plane) % 2 == 1:
                    votes[index] += 1
        return votes * 2 > N_JITTER


def _crossings(tri, point, axis, plane):
    """沿 +axis 的射线与三角形组求交，返回交点个数（只数沿射线方向更远的）。

    三角形投影到垂直平面做二维重心坐标判定，再用重心权重插值出交点沿射线轴的坐标。
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    pa, pb, pc = a[:, plane], b[:, plane], c[:, plane]
    v0, v1 = pb - pa, pc - pa
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(det) > 0
    det = np.where(ok, det, 1.0)
    rel = np.array([point[plane[0]], point[plane[1]]]) - pa
    u = (rel[:, 0] * v1[:, 1] - rel[:, 1] * v1[:, 0]) / det
    v = (v0[:, 0] * rel[:, 1] - v0[:, 1] * rel[:, 0]) / det
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
    if not hit.any():
        return 0
    along = a[hit, axis] + u[hit] * (b[hit, axis] - a[hit, axis]) \
        + v[hit] * (c[hit, axis] - a[hit, axis])
    return int((along > point[axis]).sum())


def hidden_faces(mesh, lpp, max_faces=MAX_RAY_FACES):
    """抽样面片里「两面都是实体」的片数，即被包在实体内部的内壳面片。

    判定口径与面片绕向无关（母船 STL 的绕向并不统一：kvlcc2 整体朝内、wigley 与
    npl 两个船型绕向不一致，见汇报里的健康检查表）：把面心沿法向两侧各推开一点点，
    两边的点都落在实体内，这片就是内部面片；只要有一侧在外面，它就是外表面。
    实体判据是沿坐标轴发射线的交点数奇偶（奇 = 在实体内），三个轴各投一票、取多数
    ——单根射线擦着面片走时奇偶会翻，单方向判定不稳。

    输出 STL 不是封闭时（kcs、containership 等带孔），内外本身没有定义，这一列只能
    当参考。
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    stride = max(1, len(triangles) // max_faces)
    sample = np.arange(0, len(triangles), stride)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    centre = triangles.mean(axis=1)
    offset = NORMAL_OFFSET_FRACTION * lpp * normals[sample]
    votes = np.zeros(len(sample), dtype=np.int64)
    for axis in range(3):
        grid = RayGrid(triangles, axis)
        votes += (grid.parity(centre[sample] + offset, lpp)
                  & grid.parity(centre[sample] - offset, lpp))
    inside = votes >= 2
    return {"n_sampled": int(len(sample)), "n_hidden": int(inside.sum()),
            "hidden_index": sample[inside]}


def station_rings(mesh, stations, lpp, draft):
    """每个站位剖面：环数、最大同高度交点数（>1 即同心双壁）、y ≤ 0 与 z ≤ T 后的链数。"""
    rings, crossings, wet_chains = [], [], []
    for station in stations:
        lines = mesh_plane(mesh, (1.0, 0.0, 0.0), (float(station), 0.0, 0.0))
        if len(lines) == 0:
            rings.append(0)
            crossings.append(0)
            wet_chains.append(0)
            continue
        seg = lines[:, :, (1, 2)]
        seg = seg[np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1) > WELD_FRACTION * lpp]
        wet = seg[(seg[:, :, 0].max(axis=1) <= WELD_FRACTION * lpp)
                  & (seg[:, :, 1].min(axis=1) <= draft)]
        nodes, label = _weld(seg.reshape(-1, 2), WELD_FRACTION * lpp)
        start, end = label[0::2], label[1::2]
        keep = start != end
        n_node = len(nodes)
        graph = coo_matrix((np.ones(2 * int(keep.sum())),
                            (np.r_[start[keep], end[keep]], np.r_[end[keep], start[keep]])),
                           shape=(n_node, n_node))
        rings.append(int(connected_components(graph, directed=False)[0]))

        # y ≤ 0 半船上的同高度交点数：单壁船型每个 z 只交 1 次
        half = seg[seg[:, :, 0].mean(axis=1) <= WELD_FRACTION * lpp]
        if len(half) == 0:
            crossings.append(0)
        else:
            low, high = half[:, :, 1].min(), min(half[:, :, 1].max(), draft)
            levels = np.linspace(low, high, N_LEVELS + 2)[1:-1]
            z0, z1 = half[:, 0, 1], half[:, 1, 1]
            count = [int(((np.minimum(z0, z1) <= level)
                          & (level <= np.maximum(z0, z1))).sum()) for level in levels]
            crossings.append(int(max(count)) if count else 0)

        # 湿表面（y ≤ 0、z ≤ T）的链数：融合时多链会被拼接，是光顺性的隐患
        path, info = _section_path(wet, WELD_FRACTION * lpp, MIN_ARC_FRACTION * lpp,
                                   MAX_JOIN_FRACTION * lpp, TOP_TOL_FRACTION * lpp)
        wet_chains.append(0 if path is None else int(info["n_chains"]))
    return {"rings": rings, "crossings": crossings, "wet_chains": wet_chains}


def check_one(path, lpp=None, draft=None, n_stations=40):
    """单个 STL 的全部健康指标。"""
    mesh = load_mesh(path)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if lpp is None:
        lpp = float(bounds[1, 0] - bounds[0, 0])
    if draft is None:
        draft = float(bounds[0, 2] + 0.5 * (bounds[1, 2] - bounds[0, 2]))
    n_vertex = len(mesh.vertices)
    parts = mesh.split(only_watertight=False)

    hidden = hidden_faces(mesh, lpp)
    x_lo, x_hi = float(bounds[0, 0]), float(bounds[1, 0])
    stations = x_lo + np.linspace(0.02, 0.98, n_stations) * (x_hi - x_lo)
    sections = station_rings(mesh, stations, lpp, draft)

    wet_points, wet_faces = half_wet_surface(path, lpp, draft)
    wet = trimesh.Trimesh(wet_points, wet_faces, process=False)
    return {
        "stl": str(path),
        "L_ref_m": float(bounds[1, 0] - bounds[0, 0]),
        "T_m": float(draft),
        "n_vertices": int(n_vertex),
        "n_faces": int(len(mesh.faces)),
        "n_components": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "area_m2": float(mesh.area),
        "volume_m3": float(mesh.volume) if mesh.is_volume else None,
        "n_hidden_sampled": hidden["n_hidden"],
        "n_hidden_sampled_of": hidden["n_sampled"],
        "hidden_fraction": hidden["n_hidden"] / max(hidden["n_sampled"], 1),
        "n_stations": int(n_stations),
        "station_rings_max": int(max(sections["rings"])),
        "station_rings_mean": float(np.mean(sections["rings"])),
        "stations_multi_ring": int(sum(r > 1 for r in sections["rings"])),
        "station_crossings_max": int(max(sections["crossings"])),
        "stations_nested": int(sum(c > 1 for c in sections["crossings"])),
        "wet_area_m2": float(wet.area),
        "wet_stations_multi_chain": int(sum(c > 1 for c in sections["wet_chains"])),
    }


def format_table(rows):
    """把结果排成等宽表（终端直接看）。"""
    columns = (("hull", 20), ("faces", 8), ("comp", 5), ("water", 6), ("wind", 6),
               ("euler", 7), ("hidden", 14), ("rings", 8), ("nested", 8),
               ("wet≤T m²", 10), ("wet链", 6))
    lines = [" ".join(f"{name:<{width}}" if index == 0 else f"{name:>{width}}"
                      for index, (name, width) in enumerate(columns)),
             "-" * (sum(width for _, width in columns) + len(columns) - 1)]
    for row in rows:
        euler = row["euler_number"]
        values = [
            row["name"],
            str(row["n_faces"]),
            str(row["n_components"]),
            "yes" if row["watertight"] else "NO",
            "yes" if row["winding_consistent"] else "NO",
            "nan" if euler != euler else str(int(euler)),
            f"{row['n_hidden_sampled']}/{row['n_hidden_sampled_of']}"
            f"({row['hidden_fraction'] * 100:.1f}%)",
            f"{row['station_rings_max']}/{row['station_rings_mean']:.2f}",
            f"{row['stations_nested']}/{row['stations_multi_ring']}",
            f"{row['wet_area_m2']:.4f}",
            str(row["wet_stations_multi_chain"]),
        ]
        lines.append(" ".join(
            f"{value:<{width}}" if index == 0 else f"{value:>{width}}"
            for index, (value, (_, width)) in enumerate(zip(values, columns))))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="船体 STL 健康检查")
    parser.add_argument("--stl", nargs="+", default=None, help="待检查的 STL；默认 11 个经典船型")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--stations", type=int, default=40, help="x 方向抽样站位数")
    parser.add_argument("--json", default=None, help="结果另存为 JSON")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "classic_hulls" / "manifest.json").read_text())
    if args.stl:
        jobs = [(Path(p).stem, Path(p), None, None) for p in args.stl]
    else:
        jobs = [(name, dataset / "classic_hulls" / entry["stl"], entry["Lpp"], entry["T"])
                for name, entry in manifest.items()]

    rows = []
    for name, path, lpp, draft in jobs:
        row = check_one(path, lpp, draft, n_stations=args.stations)
        row["name"] = name
        rows.append(row)
        print(f"  checked {name} ({row['n_faces']} faces)")
    print()
    print(format_table(rows))
    print()
    print("列说明：faces=面片数，comp=连通分量数，water=是否水密，wind=绕向是否一致，"
          "euler=欧拉数，hidden=内壳面片（抽样，两侧都在实体内的片数），"
          "rings=站位剖面环数（最大/均值），nested=同心双壁站位/多环站位，"
          "wet=湿表面(y≤0,z≤T)面积，wet链=湿表面剖面多链站位数")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"stations": args.stations, "hulls": rows}, indent=2,
                       ensure_ascii=False) + "\n")
        print(f"写出 {args.json}")


if __name__ == "__main__":
    sys.exit(main())
