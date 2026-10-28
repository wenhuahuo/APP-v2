"""母船 STL 的站位剖面重建回代检查（融合权重 w = 0 的口径）。

`hull_station_grid` 把一艏船化成「u = 站位序号、v = 剖面内弧长」的规则网格，
morph 在 w = 0 时输出的格点就是母船自己的剖面点。这个脚本逐个母船核对：

1. **回代偏差**：在**每站自己的平面**里（艏区是绕竖直枢轴旋转的平面、非艏区是
   `x = const`），把重建格点与母船 STL 的该平面交线折线求最小距离，取
   max / p99 / 中位数（除以 L_ref）。这是「重建曲面 = 母船曲面」的直接度量。
   艏区（u ≥ `BOW_FAN_START`）单独给一列：`x = const` 的横切平面在带球鼻艏的船型
   上会把艏部剖面切成「干舷外飘 + 球鼻前缘」两条链，桥接后生成一堵竖直墙，偏差
   就是从这里来的。
2. **剖面链数**：每站平面上的交线按焊接后连通分量数，1 条为正常；端部 5% 内的
   真实分离结构（螺旋桨轴包、呆木、艉轴支架）允许 >1，列出来备查。
3. **水线行**：v = 1 一行（第 n_v − 1 行）的 z 与设计吃水的偏差，以及有多少站位的
   水线占比是用邻站补的（剖面整条落在水线一侧的艏艉尖端）。
4. **折返带**：同一行格点沿站位的二阶差分 `|P[i+1] − 2P[i] + P[i−1]|`，分船中段
   `u ∈ (0.12, 0.88)`（告警口径）与艏区（旋转平面，允许更大）两个数。
5. **艏柱线 `x_max(z)`**：末站的平面恰好是中心面 `y = 0`，尖艏的曲面在那条平面上
   与平面相切、切不出交线，所以末站不进第 1 项（回代口径对它没有定义）。改用另一
   个口径覆盖艏尖：按 z 逐层横切母船曲面与重建网格，取每层最靠前的 x，两者的差
   除以 L_ref。这同时给出全长：`x_max` 在所有层上的最大值就是 `L_ref`。
6. **排水量**：重建网格在 z ≤ T 下的体积，与母船 STL 同口径的体积对账（STL 本身
   不水密的船型只能给 n/a，见 `check_hull_stl.py` 的体检表）。
7. **耗时**：单船 `hull_station_grid`。

用法:

    python3 scripts/dataset_build/check_station_reconstruction.py
    python3 scripts/dataset_build/check_station_reconstruction.py --hulls dtc kcs
    python3 scripts/dataset_build/check_station_reconstruction.py --json /tmp/station.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from trimesh.intersections import mesh_plane

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

from hull_morph import (  # noqa: E402
    BOW_FAN_START, WELD_FRACTION, _weld, full_ship_mesh, half_ship_surface,
    hull_station_grid, local_section, mesh_displacement, station_frames, valid_run,
)

# 折返带：船中段（告警口径）与艏区的统计区间
FOLD_BODY = (0.12, 0.88)
# 端部范围（相对船长）：这里允许出现真实分离结构（艉轴、呆木、艏球鼻）
END_FRACTION = 0.05
# 回代偏差的分块上限（点数 × 线段数），避免一次算整个矩阵
DISTANCE_BLOCK = 4_000_000
# 艏柱线口径的 z 分层数：层心取 z = const（层边界上母船的底板/甲板恰好与平面重合，
# 切不出交线），40 层在 0.02–2 m 的型深上给 5–50 mm 的分辨率
STEM_LEVELS = 40


def point_segment_distance(points, seg_a, seg_b):
    """(n,2) 的点到 (m,2)-(m,2) 线段组的最小距离。"""
    out = np.full(len(points), np.inf)
    step = max(1, int(DISTANCE_BLOCK // max(len(seg_a), 1)))
    for start in range(0, len(points), step):
        block = points[start:start + step][:, None, :]
        edge = (seg_b - seg_a)[None]
        offset = block - seg_a[None]
        ratio = np.clip((offset * edge).sum(-1)
                        / np.maximum((edge * edge).sum(-1), np.finfo(float).tiny),
                        0.0, 1.0)
        out[start:start + step] = np.linalg.norm(
            offset - ratio[..., None] * edge, axis=-1).min(axis=1)
    return out


def section_components(section, lpp):
    """站平面交线（(n,2,2) 的局部坐标线段）焊接后有几条链。"""
    segment = np.asarray(section, dtype=np.float64).reshape(-1, 2, 2)
    segment = segment[np.linalg.norm(segment[:, 1] - segment[:, 0], axis=1)
                      > WELD_FRACTION * lpp]
    if len(segment) == 0:
        return 0
    nodes, label = _weld(segment.reshape(-1, 2), WELD_FRACTION * lpp)
    start, end = label[0::2], label[1::2]
    keep = start != end
    if not keep.any():
        return 0
    graph = coo_matrix((np.ones(2 * int(keep.sum())),
                        (np.r_[start[keep], end[keep]], np.r_[end[keep], start[keep]])),
                       shape=(len(nodes), len(nodes)))
    return int(connected_components(graph, directed=False)[0])


def forward_envelope(mesh, z_low, z_high, levels=STEM_LEVELS):
    """逐层 z = const 横切，返回每层最靠前的 x 与层心高度。

    艏柱线 `x_max(z)` 的同一个口径用在母船曲面与重建网格上：对称船的每层最靠前
    点都落在中心面上，所以这就是艏柱（含球鼻、艏柱前倾）。切片层心避开母船的
    底板与甲板平面（那两个平面上曲面与平面重合，切不出交线）。
    """
    z = np.linspace(z_low, z_high, levels + 1)
    centre = 0.5 * (z[1:] + z[:-1])
    forward = np.full(levels, np.nan)
    for index, height in enumerate(centre):
        lines = mesh_plane(mesh, (0.0, 0.0, 1.0), (0.0, 0.0, float(height)))
        if len(lines) == 0:
            continue
        forward[index] = float(np.asarray(lines, dtype=np.float64)[:, :, 0].max())
    return centre, forward


def stem_deviation(mother, rebuilt, lpp, draft):
    """母船曲面与重建网格的艏柱线对账（逐层 `x_max`，单位 m 与 L_ref）。"""
    z_low = float(min(mother.bounds[0, 2], rebuilt.bounds[0, 2]))
    z_high = float(max(mother.bounds[1, 2], rebuilt.bounds[1, 2]))
    centre, reference = forward_envelope(mother, z_low, z_high)
    _, measured = forward_envelope(rebuilt, z_low, z_high)
    known = np.isfinite(reference) & np.isfinite(measured)
    if not known.any():
        return {"z": [], "mother_m": [], "rebuilt_m": [], "delta_over_L_ref": [],
                "max_over_L_ref": None, "max_z": None, "max_wet_over_L_ref": None,
                "max_wet_z": None}
    delta = np.where(known, measured - reference, np.nan)
    wet = known & (centre <= draft)

    def worst(mask):
        if not mask.any():
            return None, None
        index = int(np.nanargmax(np.where(mask, np.abs(delta), -np.inf)))
        return float(delta[index] / lpp), float(centre[index])

    value, where = worst(known)
    wet_value, wet_where = worst(wet)
    return {
        "z": [float(v) for v in centre],
        "mother_m": [None if not np.isfinite(v) else float(v) for v in reference],
        "rebuilt_m": [None if not np.isfinite(v) else float(v) for v in measured],
        "delta_over_L_ref": [None if not np.isfinite(v) else float(v / lpp) for v in delta],
        "max_over_L_ref": value, "max_z": where,
        "max_wet_over_L_ref": wet_value, "max_wet_z": wet_where,
    }


def fold_amplitude(grid, u, low, high):
    """同一行格点沿站位的二阶差分：区间内最大值与所在 (u, 行号)。"""
    second = np.linalg.norm(grid[2:] - 2.0 * grid[1:-1] + grid[:-2], axis=2)
    inside = (u[1:-1] > low) & (u[1:-1] < high)
    if not inside.any():
        return None
    sub = second[inside]
    station, row = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return {"max": float(sub.max()), "u": float(u[1:-1][inside][station]),
            "row": int(row)}


def check_hull(stl_path, lpp, draft, n_u, n_v, n_v2):
    """单船的全部检查项，返回结果 dict。"""
    started = time.perf_counter()
    surface = hull_station_grid(stl_path, lpp, draft, n_u=n_u, n_v=n_v, n_v2=n_v2)
    elapsed = time.perf_counter() - started

    half_points, half_faces = half_ship_surface(stl_path, lpp)
    mesh = trimesh.Trimesh(half_points, half_faces, process=False)
    frames = station_frames(*surface["x_range"], n_u, eps=surface.get("station_eps",
                                                                    2e-3))
    u = surface["u"]
    grid = surface["points"] * lpp
    valid = surface["stations"]
    n_row = surface["n_row"]

    distance = np.full(n_u, np.nan)
    chains = np.zeros(n_u, dtype=np.int64)
    for index in range(n_u):
        section = local_section(
            mesh_plane(mesh, tuple(frames["normals"][index]),
                       tuple(frames["origins"][index])),
            frames["origins"][index], frames["u_hat"][index])
        section = np.asarray(section, dtype=np.float64).reshape(-1, 2, 2)
        section = section[np.linalg.norm(section[:, 1] - section[:, 0], axis=1)
                          > WELD_FRACTION * lpp]
        # 末站平面就是中心面：切片是退化的（曲面与平面相切，切出来的是甲板/龙骨
        # 中线那一圈边界边），链数与回代都没有定义，由下面的艏柱线口径覆盖
        if frames["u_hat"][index, 1] == 0.0:
            continue
        chains[index] = section_components(section, lpp)
        if not valid[index] or len(section) == 0:
            continue
        r = (grid[index] - frames["origins"][index]) @ frames["u_hat"][index]
        local = np.column_stack([r, grid[index][:, 2]])
        distance[index] = point_segment_distance(local, section[:, 0], section[:, 1]).max()

    run = valid_run(valid)
    if run is None:
        raise ValueError(f"{stl_path} 没有有效站位")
    known = np.isfinite(distance)
    bow = known & (u >= BOW_FAN_START)
    worst = int(np.argmax(np.where(known, distance, -np.inf)))
    worst_bow = int(np.argmax(np.where(bow, distance, -np.inf))) if bow.any() else None

    # 水线行：v = 1 是第 n_v − 1 行；占比用邻站补的站位不在 z = T 上
    water = grid[:, n_v - 1, 2]
    real = np.abs(water - draft) <= 1e-12
    water_error = float(np.abs(water - draft).max())

    # 排水量：重建网格 vs 母船 STL（同一口径：封闭网格在 z ≤ T 下的体积）
    points, faces = full_ship_mesh(grid[valid])
    rebuilt = mesh_displacement(points, faces, draft)
    mother_mesh = trimesh.load(str(stl_path), force="mesh")
    mother_mesh.merge_vertices()
    mother = None
    if mother_mesh.is_watertight and mother_mesh.is_volume:
        mother = mesh_displacement(np.asarray(mother_mesh.vertices),
                                   np.asarray(mother_mesh.faces), draft)

    rebuilt_mesh = trimesh.Trimesh(points, faces, process=False)
    stem = stem_deviation(mother_mesh, rebuilt_mesh, lpp, draft)
    length = {
        "mother_bbox_m": float(np.ptp(np.asarray(mother_mesh.vertices)[:, 0])),
        "rebuilt_m": float(np.ptp(points[:, 0])),
    }
    length["relative"] = float((length["rebuilt_m"] - length["mother_bbox_m"])
                               / length["mother_bbox_m"])

    result = {
        "hull": Path(stl_path).stem,
        "Lpp": float(lpp), "T": float(draft),
        "length": length,
        "stem_x": stem,
        "n_u": n_u, "n_stations_valid": int(valid.sum()),
        "n_bow_fan_stations": int(surface["n_bow_fan_stations"]),
        "x_bow_fan": surface["x_bow_fan"],
        "time_s": elapsed,
        "back_substitution": {
            "n_stations_scored": int(known.sum()),
            "max_over_L_ref": float(distance[worst] / lpp),
            "max_u": float(u[worst]),
            "p99_over_L_ref": float(np.percentile(distance[known], 99) / lpp),
            "median_over_L_ref": float(np.median(distance[known]) / lpp),
            "bow_max_over_L_ref": None if worst_bow is None else
            float(distance[worst_bow] / lpp),
            "bow_max_u": None if worst_bow is None else float(u[worst_bow]),
            "stations_over_half_percent": [float(u[i]) for i in np.flatnonzero(
                known & (distance / lpp > 5e-3))],
        },
        "chains": {
            "max": int(chains.max()), "n_multi": int((chains > 1).sum()),
            "multi_u": [float(u[i]) for i in np.flatnonzero(chains > 1)],
            "multi_u_body": [float(u[i]) for i in np.flatnonzero(chains > 1)
                             if END_FRACTION < u[i] < 1.0 - END_FRACTION],
            # 多链站分两类：拼接桥离真实曲面 > 0.5% L 的是**真实分离结构**（艉轴包、
            # 螺旋桨）；其余是母船 STL 的微米级缝（甲板与舷侧的接缝、series60 的中线
            # 双壁），拼接几乎不吃距离
            "multi_u_off_surface": [float(u[i]) for i in np.flatnonzero(
                (chains > 1) & (distance / lpp > 5e-3))],
            "max_deviation_seam_multi": _seam_deviation(distance / lpp, chains),
        },
        "waterline": {
            "max_abs_z_minus_T": water_error,
            "n_stations_on_draft": int(real.sum()),
            "n_waterline_interpolated": int(n_u - surface["n_waterline_stations"]),
        },
        "fold": {"body": fold_amplitude(grid[run], u[run], *FOLD_BODY),
                 "bow": fold_amplitude(grid[run], u[run], BOW_FAN_START, 1.0)},
        "displacement": {
            "rebuilt_m3": rebuilt, "mother_stl_m3": mother,
            "relative": None if not mother else float((rebuilt - mother) / mother),
        },
        "diagnostics": {key: surface[key] for key in
                        ("n_spliced_stations", "n_dropped_fragments",
                         "n_clipped_loops", "n_detached_stations",
                         "n_invalid_stations")},
    }
    return result


def format_table(rows):
    head = (f"{'hull':18s} {'回代max%':>9s} {'艏区max%':>9s} {'p99%':>7s} {'中位%':>7s} "
            f"{'链max':>6s} {'多链站':>7s} {'水线Δz':>9s} {'折返中段%':>10s} "
            f"{'折返艏区%':>10s} {'排水量m³':>9s} {'STL m³':>8s} {'相对':>8s} "
            f"{'全长%':>7s} {'艏柱线%':>8s} {'s':>6s}")
    lines = [head, "-" * len(head)]
    for row in rows:
        back = row["back_substitution"]
        fold = row["fold"]
        disp = row["displacement"]
        stem_max = None if row["stem_x"]["max_over_L_ref"] is None else abs(
            row["stem_x"]["max_over_L_ref"])
        lines.append(
            f"{row['hull']:18s} {back['max_over_L_ref'] * 100:9.3f} "
            f"{_pct(back['bow_max_over_L_ref']):>9s} {back['p99_over_L_ref'] * 100:7.3f} "
            f"{back['median_over_L_ref'] * 100:7.4f} "
            f"{row['chains']['max']:6d} {row['chains']['n_multi']:7d} "
            f"{row['waterline']['max_abs_z_minus_T']:9.2e} "
            f"{_pct(None if fold['body'] is None else fold['body']['max'] / row['Lpp']):>10s} "
            f"{_pct(None if fold['bow'] is None else fold['bow']['max'] / row['Lpp']):>10s} "
            f"{_num(disp['rebuilt_m3'], '.4f'):>9s} "
            f"{_num(disp['mother_stl_m3'], '.4f'):>8s} "
            f"{_num(None if disp['relative'] is None else disp['relative'] * 100, '+.2f'):>8s} "
            f"{row['length']['relative'] * 100:+7.2f} "
            f"{_pct(stem_max):>8s} "
            f"{row['time_s']:6.2f}")
    return "\n".join(lines)


def _seam_deviation(distance, chains):
    """多链站里「缝」那一类的最大回代偏差（剔除离面 > 0.5% L 的真实分离结构）。"""
    inside = (chains > 1) & np.isfinite(distance) & (distance <= 5e-3)
    if not inside.any():
        return None
    return float(distance[inside].max())


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.3f}"


def _num(value, spec):
    return "n/a" if value is None else format(value, spec)


def main():
    parser = argparse.ArgumentParser(description="母船 STL 的站位剖面重建回代检查")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--hulls", nargs="+", default=None, help="船型子集，默认 manifest 全部")
    parser.add_argument("--n-u", type=int, default=160)
    parser.add_argument("--n-v", type=int, default=60)
    parser.add_argument("--n-v2", type=int, default=20)
    parser.add_argument("--json", default=None, help="把逐船结果写成 JSON")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "classic_hulls" / "manifest.json").read_text())
    names = args.hulls or sorted(manifest)
    rows = []
    for name in names:
        entry = manifest[name]
        stl = dataset / "classic_hulls" / entry["stl"]
        result = check_hull(stl, float(entry["Lpp"]), float(entry["T"]),
                            args.n_u, args.n_v, args.n_v2)
        rows.append(result)
        print(f"  {name} 完成（{result['time_s']:.2f} s）", flush=True)

    print()
    print(format_table(rows))
    print()
    for row in rows:
        back = row["back_substitution"]
        note = []
        if back["max_u"] < BOW_FAN_START:
            note.append(f"最大偏差站在 u = {back['max_u']:.3f}（非艏区）")
        if row["chains"]["multi_u_body"]:
            note.append("船中多链站 u = "
                        + ", ".join(f"{value:.3f}" for value in row["chains"]["multi_u_body"]))
        if row["waterline"]["n_waterline_interpolated"]:
            note.append(f"{row['waterline']['n_waterline_interpolated']} 个尖端站位的水线占比"
                        "由邻站插值（剖面整条落在水线一侧）")
        stem = row["stem_x"]
        if stem["max_over_L_ref"] is not None:
            wet = "n/a" if stem["max_wet_over_L_ref"] is None else (
                f"{stem['max_wet_over_L_ref'] * 100:+.3f}")
            note.append(f"艏柱线最差在 z = {stem['max_z']:.3f} m"
                        f"（{stem['max_over_L_ref'] * 100:+.3f}% L），"
                        f"水线以下最差 {wet}% L（z = {stem['max_wet_z']:.3f} m）")
        if note:
            print(f"  {row['hull']}: " + "；".join(note))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        print(f"\n写出 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
