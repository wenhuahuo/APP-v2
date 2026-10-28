"""融合船 STL 的几何质量门禁：封闭性、剖面单链、皱褶、尺度、排水量、权重单调、端点回代。

扫 `dataset/morphed_hulls/` 下全部 `<a>_<b>/<a>_<b>_w###.stl`（本轮规划 55 对 × 3 权重
= 165 个），逐项判定后写 `dataset/morphed_hulls/quality_report.json` 并打一张控制台表。

判定项（前两项与单调性为硬门禁，其余为 warn 提示）：

1. **封闭性**：回读 STL（顶点按位置合并）后 `is_watertight`、单连通分量、
   `is_volume` 且封闭体积为正。任一不满足即 `fail` —— 这种 STL 不能直接给
   snappyHexMesh。
2. **权重单调性**（按对判）：同一对船的 `L_ref_m`、`T_m`、`half_beam_m`、`depth_m`、
   `displacement_m3` 必须随 w 单调（同增或同减）。反向回退超过两端量级的 1% 即 `fail`
   （回退不到 1% 视为归一化融合的固有非线性，不算缺陷）。
3. **站位剖面单链**：对 y ≤ 0 半船沿 x 取 40 个等距站位切片，统计剖面折线链数，
   期望每站 1 条。出现 2 条以上记 `warn` 并给出站位数；dtc 类艉部轴支架、球鼻是真实
   分离结构，只允许出现在两端的 5% 之内，落在中段同样记 `warn`。
4. **折返带（皱褶）**：把融合船 STL 按与生成时相同的站位/行网格重采样
   （`hull_station_grid`），统计同一行格点的**二阶差分**
   `|P[i+1] − 2P[i] + P[i−1]|`（平滑曲面上只有曲率量级），船中段 `u ∈ (0.12, 0.88)`
   的最大值超过 `0.01·L_ref` 记 `warn` 并给出站位 u、行号与幅度；两端各 12% 是艉封板/
   艏柱，剖面下端本来就要抬离龙骨线，单独以宽口径 `u ∈ (0.05, 0.95)` 记录。
5. **相对尺度合理性**：`L/B`、`B/T`、`D/T` 与两母船的对应值比较，超出
   `[min, max]` 的 ±5% 区间记 `warn`（融合是凸组合，正常应落在区间内）。
6. **排水量一致性**：`|排水量 − 归一化插值排水量| / 归一化插值排水量 > 8%` 记
   `warn`。母船削薄效应是方法本身的偏差（kvlcc2 × wigley 约 −4.9%），不是实现错误。
7. **端点回代**：每对船用 w = 0 与 w = 1 各跑一次 `morph_hulls()`，与对应母船的站位
   剖面逐格点比较，p99 偏差超过 `0.005·L_ref` 记 `warn`。

已知缺陷只记录不修：`frigate` 水下艏部在 0.962L 截止；`kvlcc2` 剖面止于舷边（无甲板，
由镜像条带补出薄甲板）；`npl_round_bilge × npl_round_bilge_4a` 是近似同型对，融合船
接近母船本身（只贡献近重复样本），报告里单独列出。

用法:

    python3 scripts/dataset_build/check_morphed_hulls.py              # 全部
    python3 scripts/dataset_build/check_morphed_hulls.py --pairs kcs_s175 dtc_frigate
    python3 scripts/dataset_build/check_morphed_hulls.py --stations 60 --no-back-substitution
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from trimesh.intersections import mesh_plane

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

from hull_morph import (  # noqa: E402
    WELD_FRACTION, _weld, half_ship_surface, hull_station_grid, morph_hulls, valid_run,
)

# 站位切片数（与 check_hull_stl 的默认口径一致）
N_STATIONS = 40
# 剖面链数允许出现分离结构的端部范围（相对船长）
END_FRACTION = 0.05
# 折返带（相邻三站位同一行的二阶差分）的告警阈值（相对 L_ref）
FOLD_FRACTION = 1e-2
# 折返带统计的船中段范围：两端各 12% 是艉封板/艏柱与母船 STL 端部缺面区，剖面下端
# 本来就要抬离龙骨线（真实端部形状），单独用宽口径记录、不参与告警
FOLD_BODY = (0.12, 0.88)
# 记录用的宽口径（与上一轮同一区间）
FOLD_WIDE = (0.05, 0.95)
# 尺度比的允许外扩比例（相对母船区间）
RATIO_TOLERANCE = 0.05
# 权重单调性允许的**相对回退幅度**：沿总量变化的方向走，任何一步反向的量不超过
# 两端量级的 1% 即算单调。归一化坐标融合再乘 Lpp_new，让排水量（~Lpp³）、半宽、
# 型深都带进一个随 w 变化的 Lpp 因子，两船 Lpp 相差大时区间内会出现极值（实测回退
# 0.3%–1.0%），它不是网格缺陷：几何仍连续、水密、单链
MONOTONIC_TOLERANCE = 1e-2
# 排水量相对归一化插值的允许偏差
DISPLACEMENT_TOLERANCE = 0.08
# 端点回代的 p99 允许偏差（相对 L_ref）
BACK_SUB_FRACTION = 0.005
# 已知的近似同型对：融合船接近母船本身，只贡献近重复样本
NEAR_DUPLICATE_PAIRS = {("npl_round_bilge", "npl_round_bilge_4a"):
                        "两条母船本身接近，融合船近似重复样本"}


def load_manifest(dataset_root):
    path = Path(dataset_root) / "classic_hulls" / "manifest.json"
    return json.loads(path.read_text())


def mother_ratios(entry):
    """母船的 L/B、B/T、D/T：包围盒与设计吃水，与融合船的口径一致。"""
    low, high = entry["bounds"]
    length = float(high[0] - low[0])
    half_beam = float(max(abs(low[1]), abs(high[1])))
    depth = float(high[2])
    draft = float(entry["T"])
    return {"L_over_B": length / (2.0 * half_beam),
            "B_over_T": 2.0 * half_beam / draft,
            "D_over_T": depth / draft}


def check_closure(path):
    """封闭性：水密、单连通、封闭体积为正。"""
    mesh = trimesh.load(str(path), force="mesh")
    mesh.merge_vertices()
    volume = float(mesh.volume) if mesh.is_volume else None
    return {"watertight": bool(mesh.is_watertight),
            "n_components": int(mesh.body_count),
            "is_volume": bool(mesh.is_volume),
            "volume_m3": volume,
            "n_triangles": int(len(mesh.faces)),
            "n_vertices": int(len(mesh.vertices))}


def station_chains(path, lpp, n_stations=N_STATIONS):
    """y ≤ 0 半船沿 x 的等距站位上，剖面折线被切成几条链。"""
    points, faces = half_ship_surface(path, lpp)
    mesh = trimesh.Trimesh(points, faces, process=False)
    x_lo, x_hi = float(points[:, 0].min()), float(points[:, 0].max())
    u = np.linspace(0.02, 0.98, n_stations)
    chains, stations = [], []
    for value in u:
        station = x_lo + value * (x_hi - x_lo)
        lines = mesh_plane(mesh, (1.0, 0.0, 0.0), (station, 0.0, 0.0))
        if len(lines) == 0:
            chains.append(0)
            stations.append(float(value))
            continue
        segment = lines[:, :, (1, 2)]
        segment = segment[np.linalg.norm(segment[:, 1] - segment[:, 0], axis=1)
                          > WELD_FRACTION * lpp]
        if len(segment) == 0:
            chains.append(0)
            stations.append(float(value))
            continue
        nodes, label = _weld(segment.reshape(-1, 2), WELD_FRACTION * lpp)
        start, end = label[0::2], label[1::2]
        keep = start != end
        graph = coo_matrix((np.ones(2 * int(keep.sum())),
                            (np.r_[start[keep], end[keep]], np.r_[end[keep], start[keep]])),
                           shape=(len(nodes), len(nodes)))
        chains.append(int(connected_components(graph, directed=False)[0]))
        stations.append(float(value))
    chains = np.asarray(chains)
    multi = np.flatnonzero(chains > 1)
    middle = [float(stations[i]) for i in multi
              if END_FRACTION < stations[i] < 1.0 - END_FRACTION]
    return {"n_stations": int(n_stations),
            "n_empty": int((chains == 0).sum()),
            "max_chains": int(chains.max()) if len(chains) else 0,
            "n_multi_chain": int(len(multi)),
            "multi_chain_u": [round(float(stations[i]), 4) for i in multi],
            "multi_chain_u_middle": [round(value, 4) for value in middle]}


def station_smoothness(path, meta, n_u, n_v, n_v2):
    """同一行格点沿站位的连续性：相邻位移与二阶差分（折返尖点/皱褶）。

    网格按生成时的参数重采样（`hull_station_grid` 对融合船 STL 现场切片），行号
    即 v 的两段弧长行。

    **折返幅度**（相邻三站位的二阶差分 `|P[i+1] − 2P[i] + P[i−1]|`）是主判据：
    平滑曲面上它只有 O(Δu²) 的曲率量级，选链逐站翻转时会跳到几个 % L。统计分两个
    口径：船中段 `u ∈ (12%, 88%)`（告警用）与宽口径 `u ∈ (5%, 95%)`（记录用）——
    两端各 12% 是艉封板/艏柱，剖面下端本来就要抬离龙骨线，是真实端部形状。
    """
    lpp = float(meta["L_ref_m"])
    draft = float(meta["T_m"])
    surface = hull_station_grid(path, lpp, draft, n_u=n_u, n_v=n_v, n_v2=n_v2)
    grid = surface["points"] * lpp
    rows = surface["stations"]
    n_row = n_v + n_v2 - 1
    jump = np.full((n_u - 1, n_row), np.nan)   # 第 i 行 = 站位 i 与 i+1 之间
    for index in range(n_u - 1):
        if not (rows[index] and rows[index + 1]):
            continue
        jump[index] = np.linalg.norm(grid[index + 1] - grid[index], axis=1)
    if np.all(np.isnan(jump)):
        return {"max_jump_m": None, "max_jump_over_L_ref": None,
                "max_jump_u": None, "max_jump_row": None,
                "max_jump_mid_m": None, "max_jump_mid_over_L_ref": None,
                "max_jump_mid_u": None, "max_jump_mid_row": None,
                "max_fold_body_m": None, "max_fold_body_over_L_ref": None,
                "max_fold_body_u": None, "max_fold_body_row": None,
                "max_fold_wide_over_L_ref": None,
                "max_overall_fold_over_L_ref": None}

    width = grid[-1, 0, 0] - grid[0, 0, 0]
    u_of_station = (grid[:, 0, 0] - grid[0, 0, 0]) / width if width else np.zeros(n_u)
    middle = (u_of_station[:-1] > END_FRACTION) & (u_of_station[:-1] < 1.0 - END_FRACTION)
    assert len(middle) == len(jump)

    def locate(values, station_map):
        """values 可能是被掩码抽出来的子集，station_map 把它映回原站位下标。"""
        position, row_index = np.unravel_index(np.nanargmax(values), values.shape)
        station_index = int(station_map[position])
        span = grid[station_index + 1, 0, 0] - grid[station_index, 0, 0]
        maximum = float(values[position, row_index])
        return {"m": maximum, "over": maximum / lpp,
                "slope": maximum / float(span) if span else None,
                "u": float(u_of_station[station_index]), "row": int(row_index)}

    overall = locate(jump, np.arange(len(jump)))
    mid = locate(jump[middle], np.flatnonzero(middle))
    fold = np.linalg.norm(grid[2:] - 2.0 * grid[1:-1] + grid[:-2], axis=2)
    fold_u = u_of_station[1:-1]
    fold_body = fold[(fold_u > FOLD_BODY[0]) & (fold_u < FOLD_BODY[1])]
    fold_wide = fold[(fold_u > FOLD_WIDE[0]) & (fold_u < FOLD_WIDE[1])]
    fold_body_index = np.unravel_index(np.argmax(fold_body), fold_body.shape)
    fold_wide_index = np.unravel_index(np.argmax(fold_wide), fold_wide.shape)
    return {"max_jump_m": overall["m"],
            "max_jump_over_L_ref": overall["over"],
            "max_jump_slope": overall["slope"],
            "max_jump_u": overall["u"],
            "max_jump_row": overall["row"],
            "max_jump_mid_m": mid["m"],
            "max_jump_mid_over_L_ref": mid["over"],
            "max_jump_mid_u": mid["u"],
            "max_jump_mid_row": mid["row"],
            "max_fold_body_m": float(fold_body[fold_body_index]),
            "max_fold_body_over_L_ref": float(fold_body[fold_body_index] / lpp),
            "max_fold_body_u": float(fold_u[(fold_u > FOLD_BODY[0])
                                           & (fold_u < FOLD_BODY[1])][fold_body_index[0]]),
            "max_fold_body_row": int(fold_body_index[1]),
            "max_fold_wide_over_L_ref": float(fold_wide[fold_wide_index] / lpp),
            "max_overall_fold_over_L_ref": float(np.max(fold) / lpp)}


def check_ratios(meta, ratios_a, ratios_b):
    """L/B、B/T、D/T 相对母船区间（±5%）的位置。"""
    values = {"L_over_B": float(meta["L_ref_m"]) / (2.0 * float(meta["half_beam_m"])),
              "B_over_T": 2.0 * float(meta["half_beam_m"]) / float(meta["T_m"]),
              "D_over_T": float(meta["depth_m"]) / float(meta["T_m"])}
    bands, outside = {}, []
    for key, value in values.items():
        low, high = sorted((ratios_a[key], ratios_b[key]))
        lo, hi = low * (1.0 - RATIO_TOLERANCE), high * (1.0 + RATIO_TOLERANCE)
        bands[key] = [float(low), float(high)]
        if not lo <= value <= hi:
            outside.append(key)
    return {"values": values, "parent_bands": bands, "outside": outside}


def check_displacement(meta):
    value = meta["displacement_m3"]
    reference = meta["displacement_interp_norm_m3"]
    if value is None or reference is None:
        return {"value": value, "interp_norm": reference, "relative": None}
    return {"value": float(value), "interp_norm": float(reference),
            "relative": float((value - reference) / reference)}


def check_monotonic(rows):
    """同一对船的几何量随 w 单调（同增或同减）。返回 (是否通过, 明细)。

    「单调」按**相对回退幅度**判：先按首尾量级定出主方向（首尾差在 1e-6 量级以内
    则算平），再看有没有哪一步往反方向走超过 `MONOTONIC_TOLERANCE`×量级。三档权重
    的区间内出现极值是归一化融合的固有非线性，只要回退不到 1% 就不算缺陷；
    回退幅度写进明细供报警引用。
    """
    ordered = sorted(rows, key=lambda row: row["weight"])
    keys = ("L_ref_m", "T_m", "half_beam_m", "depth_m", "displacement_m3")
    detail, ok = {}, True
    for key in keys:
        series = [row["geometry"][key] for row in ordered]
        if any(value is None for value in series):
            detail[key] = {"series": series, "monotonic": None}
            continue
        values = np.asarray(series, dtype=np.float64)
        steps = np.diff(values)
        scale = max(abs(float(values[0])), abs(float(values[-1])), 1e-12)
        total = float(values[-1] - values[0])
        direction = 0 if abs(total) <= 1e-6 * scale else int(np.sign(total))
        if len(steps) == 0:
            reversal = 0.0
        elif direction == 0:
            reversal = float(np.abs(steps).max())
        else:
            reversal = float(np.maximum(-direction * steps, 0.0).max())
        relative = reversal / scale
        monotonic = relative <= MONOTONIC_TOLERANCE
        detail[key] = {"series": [float(value) for value in values],
                       "monotonic": monotonic,
                       "direction": {1: "increasing", -1: "decreasing",
                                     0: "flat"}[direction],
                       "max_reversal_m": reversal,
                       "max_reversal_relative": relative}
        ok = ok and monotonic
    return ok, detail


def back_substitution(dataset, manifest, name_a, name_b, n_u, n_v, n_v2):
    """w = 0 / w = 1 的融合网格与对应母船站位剖面的 p99 偏差（相对 L_ref）。"""
    stl_a = dataset / "classic_hulls" / manifest[name_a]["stl"]
    stl_b = dataset / "classic_hulls" / manifest[name_b]["stl"]
    hull_a, hull_b = manifest[name_a], manifest[name_b]
    grids = [hull_station_grid(stl_a, hull_a["Lpp"], hull_a["T"], n_u, n_v, n_v2),
             hull_station_grid(stl_b, hull_b["Lpp"], hull_b["T"], n_u, n_v, n_v2)]
    run = valid_run(grids[0]["stations"] & grids[1]["stations"])
    index = np.arange(n_u)[run]
    result = {}
    for weight, key, hull, grid in ((0.0, "w0_vs_a", hull_a, grids[0]),
                                    (1.0, "w1_vs_b", hull_b, grids[1])):
        fused = morph_hulls(stl_a, stl_b, hull_a["Lpp"], hull_b["Lpp"],
                            hull_a["T"], hull_b["T"], weight, n_u=n_u, n_v=n_v,
                            n_v2=n_v2)
        reference = grid["points"][index] * float(hull["Lpp"])
        distance = np.linalg.norm(fused["grid"] - reference, axis=2)
        lpp = float(fused["meta"]["L_ref_m"])
        result[key] = {"p99_m": float(np.percentile(distance, 99)),
                       "max_m": float(distance.max()),
                       "p99_over_L_ref": float(np.percentile(distance, 99) / lpp),
                       "max_over_L_ref": float(distance.max() / lpp)}
    return result


def collect_cases(output):
    """遍历 `<a>_<b>/morph_meta.json`，返回 [(pair, weight, stl 绝对路径, meta)]。"""
    cases = []
    for meta_path in sorted(output.glob("*/morph_meta.json")):
        pair = meta_path.parent.name
        summary = json.loads(meta_path.read_text())
        for stl_name, meta in sorted(summary.get("cases", {}).items()):
            path = meta_path.parent / stl_name
            if not path.exists():
                raise SystemExit(f"{meta_path} 里记了 {stl_name}，但文件不存在")
            cases.append({"pair": pair, "hull": Path(stl_name).stem,
                          "weight": float(meta["weight"]), "path": path,
                          "stl": f"{pair}/{stl_name}", "meta": meta,
                          "parents": (summary.get("hull_a"), summary.get("hull_b"))})
    return cases


def format_table(rows):
    columns = (("hull", 30), ("water", 6), ("comp", 5), ("vol+", 5), ("链max", 6),
               ("多链站", 7), ("折返/L", 8), ("宽口径", 8), ("尺度", 9), ("disp", 8),
               ("状态", 6))
    lines = [" ".join(f"{name:<{width}}" if index == 0 else f"{name:>{width}}"
                      for index, (name, width) in enumerate(columns)),
             "-" * (sum(width for _, width in columns) + len(columns) - 1)]
    for row in rows:
        fold = row["smoothness"].get("max_fold_body_over_L_ref")
        wide = row["smoothness"].get("max_fold_wide_over_L_ref")
        relative = row["displacement"]["relative"]
        lines.append(" ".join(
            f"{value:<{width}}" if index == 0 else f"{value:>{width}}"
            for index, (value, (_, width)) in enumerate(zip(
                (row["hull"],
                 "yes" if row["closure"]["watertight"] else "NO",
                 str(row["closure"]["n_components"]),
                 "yes" if (row["closure"]["volume_m3"] or 0.0) > 0 else "NO",
                 str(row["stations"]["max_chains"]),
                 str(row["stations"]["n_multi_chain"]),
                 "n/a" if fold is None else f"{fold * 100:.3f}%",
                 "n/a" if wide is None else f"{wide * 100:.2f}%",
                 "-" if not row["ratios"]["outside"] else ",".join(row["ratios"]["outside"]),
                 "n/a" if relative is None else f"{relative * 100:+.1f}%",
                 row["status"]), columns))))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="融合船 STL 的几何质量门禁")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--output", default=str(ROOT / "dataset" / "morphed_hulls"))
    parser.add_argument("--report", default=None, help="默认 <output>/quality_report.json")
    parser.add_argument("--pairs", nargs="+", default=None, help="只检查这些对（目录名）")
    parser.add_argument("--stations", type=int, default=N_STATIONS)
    parser.add_argument("--no-back-substitution", dest="back_substitution",
                        action="store_false", help="跳过端点回代（每对 2 次 morph_hulls）")
    parser.add_argument("--n-u", type=int, default=160, help="重采样的站位格点数")
    parser.add_argument("--n-v", type=int, default=60, help="段 1 的格点数")
    parser.add_argument("--n-v2", type=int, default=20, help="段 2 的格点数")
    args = parser.parse_args()

    dataset, output = Path(args.dataset), Path(args.output)
    report_path = Path(args.report) if args.report else output / "quality_report.json"
    manifest = load_manifest(dataset)

    cases = collect_cases(output)
    if args.pairs:
        wanted = set(args.pairs)
        cases = [case for case in cases if case["pair"] in wanted]
        if not cases:
            raise SystemExit(f"--pairs {sorted(wanted)} 没有匹配的产物")
    print(f"检查 {len(cases)} 个融合船 STL（共 {len({case['pair'] for case in cases})} 对）")

    rows = []
    for index, case in enumerate(cases, 1):
        meta = case["meta"]
        name_a, name_b = case["parents"]
        closure = check_closure(case["path"])
        stations = station_chains(case["path"], float(meta["L_ref_m"]), args.stations)
        smoothness = station_smoothness(case["path"], meta, args.n_u, args.n_v, args.n_v2)
        ratios = check_ratios(meta, mother_ratios(manifest[name_a]),
                              mother_ratios(manifest[name_b]))
        displacement = check_displacement(meta)

        reasons, warnings = [], []
        if not closure["watertight"]:
            reasons.append("回读后不水密")
        if closure["n_components"] != 1:
            reasons.append(f"连通分量 {closure['n_components']} 个")
        if not closure["is_volume"] or (closure["volume_m3"] or 0.0) <= 0:
            reasons.append("封闭体积不为正")
        if stations["n_multi_chain"]:
            where = ("端部" if not stations["multi_chain_u_middle"] else
                     f"含中段 {stations['multi_chain_u_middle']}")
            warnings.append(f"剖面多链 {stations['n_multi_chain']}/{stations['n_stations']}"
                            f" 站（最多 {stations['max_chains']} 条，{where}）")
        fold_body = smoothness.get("max_fold_body_over_L_ref")
        if fold_body is not None and fold_body > FOLD_FRACTION:
            warnings.append(
                f"船中段折返 {fold_body * 100:.2f}% L"
                f" @ u={smoothness['max_fold_body_u']:.3f} 行 {smoothness['max_fold_body_row']}"
                f"（宽口径 {smoothness['max_fold_wide_over_L_ref'] * 100:.2f}% L）")
        if ratios["outside"]:
            warnings.append("尺度超出母船区间：" + ", ".join(
                f"{key} {ratios['values'][key]:.2f} ∉ "
                f"[{ratios['parent_bands'][key][0]:.2f}, {ratios['parent_bands'][key][1]:.2f}]"
                for key in ratios["outside"]))
        if displacement["relative"] is not None and \
                abs(displacement["relative"]) > DISPLACEMENT_TOLERANCE:
            warnings.append(f"排水量偏离归一化插值 {displacement['relative'] * 100:+.1f}%")

        rows.append({"hull": case["hull"], "pair": case["pair"],
                     "weight": case["weight"], "stl": case["stl"],
                     "parents": {"a": name_a, "b": name_b},
                     "geometry": {key: meta[key] for key in
                                  ("L_ref_m", "T_m", "half_beam_m", "depth_m",
                                   "displacement_m3", "n_triangles", "n_points")},
                     "closure": closure, "stations": stations,
                     "smoothness": smoothness, "ratios": ratios,
                     "displacement": displacement,
                     "status": "fail" if reasons else "pass",
                     "reasons": reasons, "warnings": warnings})
        print(f"  [{index}/{len(cases)}] {case['hull']}: "
              f"{'fail' if reasons else 'pass'}"
              f"{'' if not warnings else '（' + str(len(warnings)) + ' warn）'}")

    # 权重单调性：按对判，违反即 fail
    pair_rows = []
    for pair in sorted({row["pair"] for row in rows}):
        group = [row for row in rows if row["pair"] == pair]
        if len(group) < 2:
            continue
        ok, detail = check_monotonic(group)
        if not ok:
            for row in group:
                row["status"] = "fail"
                row["reasons"].append("同对权重单调性违反：" + ", ".join(
                    f"{key} 回退 {detail[key]['max_reversal_relative'] * 100:.1f}%"
                    for key, item in detail.items() if item["monotonic"] is False))
        pair_rows.append({"pair": pair, "monotonic": ok, "detail": detail,
                          "weights": sorted(row["weight"] for row in group)})

    # 端点回代：每对一次（w = 0 与 w = 1 各跑一次 morph_hulls）
    back_rows = []
    if args.back_substitution:
        pairs = sorted({row["pair"] for row in rows})
        print(f"端点回代：{len(pairs)} 对 × 2 次 morph_hulls …")
        for index, pair in enumerate(pairs, 1):
            name_a, name_b = next((row["parents"]["a"], row["parents"]["b"])
                                  for row in rows if row["pair"] == pair)
            result = back_substitution(dataset, manifest, name_a, name_b,
                                       args.n_u, args.n_v, args.n_v2)
            warn = [key for key, item in result.items()
                    if item["p99_over_L_ref"] > BACK_SUB_FRACTION]
            back_rows.append({"pair": pair, "hull_a": name_a, "hull_b": name_b,
                              "max_p99_over_L_ref": max(item["p99_over_L_ref"]
                                                        for item in result.values()),
                              "warn": warn, **result})
            for row in rows:
                if row["pair"] != pair:
                    continue
                for key in warn:
                    mother = name_a if key.startswith("w0") else name_b
                    row["warnings"].append(
                        f"端点回代 w={key[1]} 对 {mother} 的 p99 "
                        f"{result[key]['p99_over_L_ref'] * 100:.2f}% L")
            print(f"  [{index}/{len(pairs)}] {pair}: p99 "
                  f"{back_rows[-1]['max_p99_over_L_ref'] * 100:.3f}% L"
                  f"{'  WARN' if warn else ''}")

    fails = [row for row in rows if row["status"] == "fail"]
    warns = [row for row in rows if row["warnings"]]
    print()
    print(format_table(rows))
    print()
    print(f"合计 {len(rows)} 个：fail {len(fails)}，有 warn {len(warns)}，"
          f"全通过 {len(rows) - len(warns)}")
    if fails:
        print("\nfail 船型（必须单独处理，不能进算例）：")
        for row in fails:
            print(f"  {row['hull']}: {'；'.join(row['reasons'])}")
    if warns:
        print(f"\nwarn 汇总（{len(warns)} 个，可继续生产）：")
        for row in warns:
            print(f"  {row['hull']}: {'；'.join(row['warnings'])}")
    near = [row for row in rows
            if (row["parents"]["a"], row["parents"]["b"]) in NEAR_DUPLICATE_PAIRS]
    if near:
        note = NEAR_DUPLICATE_PAIRS[(near[0]["parents"]["a"], near[0]["parents"]["b"])]
        print(f"\n近似同型对 {near[0]['pair']}（{note}）："
              f"{len(near)} 个船型只贡献近重复样本")

    report = {
        "generated_by": "scripts/dataset_build/check_morphed_hulls.py",
        "n_cases": len(rows),
        "n_pairs": len({row["pair"] for row in rows}),
        "thresholds": {
            "stations": args.stations,
            "end_fraction": END_FRACTION,
            "fold_fraction": FOLD_FRACTION,
            "fold_body_band": list(FOLD_BODY),
            "fold_wide_band": list(FOLD_WIDE),
            "ratio_tolerance": RATIO_TOLERANCE,
            "monotonic_tolerance": MONOTONIC_TOLERANCE,
            "displacement_tolerance": DISPLACEMENT_TOLERANCE,
            "back_substitution_fraction": BACK_SUB_FRACTION,
        },
        "redundant_pairs": [{"pair": pair[0] + "_" + pair[1], "note": note}
                            for pair, note in NEAR_DUPLICATE_PAIRS.items()],
        "summary": {"fail": len(fails), "warn": len(warns),
                    "pass_clean": len(rows) - len(warns),
                    "fail_hulls": [row["hull"] for row in fails],
                    "warn_hulls": [row["hull"] for row in warns]},
        "pairs": pair_rows,
        "back_substitution": back_rows,
        "cases": rows,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n写出 {report_path}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
