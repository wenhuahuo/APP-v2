"""构建 KCS 的两个艏部小幅 FFD 几何变体（论文 Pechenyuk 2017 的第 1 段 ±δV）。

用法:
    /opt/miniconda3/envs/mesh/bin/python3 \
        scripts/dataset_build/build_kcs_ffd_variants.py

产物（`dataset/` 不入 Git）：

| 文件 | 内容 |
| --- | --- |
| `dataset/kcs_var/kcs_bow_outward.stl` | 艏部第 1 段 +0.4% 排水体积（图 1 模型 1） |
| `dataset/kcs_var/kcs_bow_inward.stl` | 艏部第 1 段 −0.4% 排水体积（图 1 模型 2） |
| `dataset/kcs_var/<变体>.json` | 该变体的 FFD 参数、几何统计与检查结果 |
| `dataset/kcs_var/<变体>_ffd.npy` | 控制点位移 (9, 2, 9, 3)（米） |
| `dataset/kcs_var/index.json` | 母船与两个变体的汇总（基准、论文出处、共用参数） |

母船 `dataset/classic_hulls/kcs.stl` 只读，不覆盖；ShipBenchv2 条目不动。

检查项（每一对都跑，结果写进 `index.json` 的 `checks`）：

1. STL 可读回、顶点/面数不变、坐标有限；
2. 面片法向有限、面积有限且非负，三角形翻转数（变形后法向与变形前点积 ≤ 0）为 0；
3. 水密性与绕向：与母船同一口径（`trimesh` 直接读，母船本身不水密——85 个连通片，
   这是 STL 的既有状态，不是变体引入的）；
4. 排水体积（`hull_morph.mesh_displacement`，与 ShipBenchv2 排水量同一口径）、
   湿表面积（`hull_morph.half_wet_surface`）、半宽、型深，以及它们相对母船的变化；
5. 变形前后站位对应：两边都用 `hull_station_grid`（同一 160 站、同一 x 范围），
   逐站比较站位 x、水下剖面面积 A(x)，给出舷弧面积曲线 ω = A/A_mid 的差；
6. 最大位移（米与 %Lpp）、水线附近的最大横向位移（v = 1 那一行）、控制格架位移量级。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

import ffd  # noqa: E402
import hull_morph  # noqa: E402
from stl_io import read_stl, write_stl  # noqa: E402

# KCS 参照参数（`dataset/classic_hulls/manifest.json`）：Lpp 与设计吃水。
DEFAULT_LPP = 7.2786
DEFAULT_DRAFT = 0.3418
# 母船主甲板高度：竖直权 g(z) 在 z = 0 与 z = deck 处为 0（甲板形状与龙骨不动）。
# 取自 STL 的型深分布（x ∈ [0.1, 5.9] m 的顶边 z），艏楼（到 0.744 m）在 g 的零区外。
DEFAULT_DECK = ffd.KCS_DECK
# 论文出处（变体定义的唯一依据）。
PAPER = {
    "authors": "A. V. Pechenyuk",
    "title": "Optimization of a hull form for decrease ship resistance to movement",
    "journal": "Computer Research and Modeling, 2017, 9(1):57-65",
    "doi": "10.20537/2076-7633-2017-9-57-65",
    "basis": "§2 图 1：艏部三段舷弧面积曲线的有限体积增量，每段约 0.4% 排水体积，"
             "剖面重建保持甲板形状；图 2：对应的型线表达。本轮只做第 1 段（最靠艏）",
}


def dump(payload, path):
    """写 JSON：numpy 标量统一转成 Python 标量。"""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               default=lambda item: item.item()) + "\n")


def station_profile(stl, lpp, draft):
    """站位网格 → (站位 x 数组, 水下剖面面积 A(x), 水线半宽 b(x), 型深 z(x))。"""
    grid = hull_morph.hull_station_grid(stl, lpp, draft)
    points = np.asarray(grid["points"], dtype=np.float64) * lpp
    n_v = grid["n_v"]
    x = np.asarray(grid["x"], dtype=np.float64)
    y = -points[:, :n_v, 1]
    z = points[:, :n_v, 2]
    area = 2.0 * np.sum(0.5 * (y[:, 1:] + y[:, :-1]) * np.diff(z, axis=1), axis=1)
    return {
        "x": x,
        "area": area,
        "waterline_half_beam": y[:, n_v - 1],
        "z_max": points[:, :, 2].max(axis=1),
        "grid": grid,
    }


def stats(stl, lpp, draft):
    """一个 STL 的几何统计。"""
    points, faces = read_stl(stl)
    # STL 不共享顶点（一个三角形三个顶点），直接建 mesh 会把每个面当成一个连通片；
    # 先按位置合并顶点，水密性/连通片数才是曲面本身的属性
    mesh = trimesh.Trimesh(points, faces, process=False)
    mesh.merge_vertices()
    triangles = points[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    wet_points, wet_faces = hull_morph.half_wet_surface(stl, lpp, draft)
    wet = trimesh.Trimesh(wet_points, wet_faces, process=False)
    return {
        "file": str(Path(stl).name),
        "n_vertices": int(len(points)),
        "n_faces": int(len(faces)),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "l_ref_m": float(mesh.bounds[1][0] - mesh.bounds[0][0]),
        "half_beam_m": float(np.abs(points[:, 1]).max()),
        "depth_m": float(points[:, 2].max()),
        "displacement_m3": float(hull_morph.mesh_displacement(points, faces, draft)),
        "wetted_area_m2": float(wet.area),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "n_components": int(len(mesh.split(only_watertight=False))),
        "face_area_min_m2": float(area.min()),
        "face_area_max_m2": float(area.max()),
        "nonfinite_vertices": int(np.count_nonzero(~np.isfinite(points))),
        "nonfinite_faces": int(np.count_nonzero(~np.isfinite(faces))),
        "nonfinite_normals": int(np.count_nonzero(~np.isfinite(cross))),
    }


def build_variant(name, base_points, base_faces, base_profile, base_stats, lpp, draft, deck,
                  out_dir):
    """构建一个变体，写 STL 与 metadata，返回 (metadata, 检查结果)。"""
    displacements = ffd.bow_displacements(name, base_points, lpp, deck)
    moved = ffd.deform_bow(base_points, displacements, lpp, deck)
    dy = moved[:, 1] - base_points[:, 1]
    sign, description = ffd.KCS_BOW_VARIANTS[name]
    amplitude = ffd.variant_amplitude(name)
    residual = ffd.bow_fit_residual(base_points, displacements, lpp, deck, amplitude)

    stl = out_dir / f"{name}.stl"
    write_stl(stl, moved, base_faces, solid=name)
    np.save(out_dir / f"{name}_ffd.npy", displacements.astype(np.float64))

    # 读回来再量：STL 写-读往返是后续 CFD/数据集流程的真实入口
    reloaded, reloaded_faces = read_stl(stl)
    roundtrip = float(np.abs(reloaded - moved).max())
    checks = stats(stl, lpp, draft)
    profile = station_profile(stl, lpp, draft)

    if profile["x"].shape != base_profile["x"].shape:
        raise SystemExit(f"{name}: 站位数量与母船不一致")
    a_mid = float(base_profile["area"].max())
    delta_area = profile["area"] - base_profile["area"]
    omega = delta_area / a_mid
    waterline_dy = profile["waterline_half_beam"] - base_profile["waterline_half_beam"]
    # 水线半宽的相对变化只在母船有明显半宽（> 10% B/2）的站位上统计：艏尖附近的
    # 半宽只有几毫米，相对量会被小数放大
    wide = base_profile["waterline_half_beam"] > 0.1 * checks["half_beam_m"]
    relative_waterline = np.abs(waterline_dy[wide]) / base_profile["waterline_half_beam"][wide]
    waterline_peak = int(np.argmax(np.abs(waterline_dy)))

    cross_flip = ffd.flipped_triangles(base_points, moved, base_faces)
    wandering = int(np.count_nonzero(np.abs(moved[:, 0] - base_points[:, 0]) > 1e-12)
                    + np.count_nonzero(np.abs(moved[:, 2] - base_points[:, 2]) > 1e-12))
    outward = np.count_nonzero(np.abs(dy[np.abs(base_points[:, 1]) < 1e-9]))
    paper_axis = ffd.paper_x_over_lpp(profile["x"], lpp)
    peak = int(np.argmax(np.abs(omega)))

    metadata = {
        "variant": name,
        "description": description,
        "parent": {
            "name": "kcs",
            "stl": "dataset/classic_hulls/kcs.stl",
            "lpp_m": lpp,
            "draft_m": draft,
            "deck_m": deck,
            "displacement_m3": base_stats["displacement_m3"],
            "wetted_area_m2": base_stats["wetted_area_m2"],
        },
        "paper": dict(PAPER, model="модель 1" if sign > 0 else "модель 2",
                      segment_x_over_lpp=list(ffd.BOW_SEGMENT),
                      volume_fraction=ffd.BOW_VOLUME_FRACTION,
                      volume_sign="+" if sign > 0 else "-"),
        "ffd": {
            "lattice": list(ffd.KCS_LATTICE),
            "bounds": [list(axis) for axis in ffd.bow_lattice_bounds(base_points, lpp, deck)],
            "amplitude": float(amplitude),
            "field": "dy = amplitude · W(s) · sin²(pi z / deck) · |y|,"
                     " s = 1 − x/Lpp（σ = 论文图 1 第 1 段，升余弦窗）；x、z 不动",
            "control_displacement_file": f"{name}_ffd.npy",
            "max_control_displacement_m": float(np.abs(displacements).max()),
            "fit_residual_m": residual,
        },
        "geometry": {
            "file": stl.name,
            "n_vertices": checks["n_vertices"],
            "n_faces": checks["n_faces"],
            "max_abs_dy_m": float(np.abs(dy).max()),
            "max_abs_dy_over_lpp": float(np.abs(dy).max() / lpp),
            "rms_dy_m": float(np.sqrt(np.mean(dy ** 2))),
            "displacement_m3": checks["displacement_m3"],
            "volume_change": float(checks["displacement_m3"] / base_stats["displacement_m3"] - 1.0),
            "wetted_area_m2": checks["wetted_area_m2"],
            "wetted_area_change": float(checks["wetted_area_m2"] / base_stats["wetted_area_m2"] - 1.0),
            "waterline_max_abs_dy_m": float(np.abs(waterline_dy).max()),
            "waterline_max_at_x_over_lpp": float(1.0 - profile["x"][waterline_peak] / lpp),
            "waterline_max_relative_dy": float(relative_waterline.max()),
            "section_area_peak_over_midship": float(np.abs(omega).max()),
            "section_area_peak_at_x_over_lpp": float(paper_axis[peak]),
            "l_ref_m": checks["l_ref_m"],
            "half_beam_m": checks["half_beam_m"],
            "depth_m": checks["depth_m"],
        },
        "checks": {
            "roundtrip_max_coordinate_error_m": roundtrip,
            "flipped_triangles": cross_flip,
            "vertices_moved_along_x_or_z": wandering,
            "centerline_vertices_with_lateral_shift": outward,
            "watertight": checks["watertight"],
            "winding_consistent": checks["winding_consistent"],
            "n_components": checks["n_components"],
            "nonfinite_vertices": checks["nonfinite_vertices"],
            "nonfinite_faces": checks["nonfinite_faces"],
            "nonfinite_normals": checks["nonfinite_normals"],
            "face_area_min_m2": checks["face_area_min_m2"],
            "face_area_max_m2": checks["face_area_max_m2"],
            "station_x_max_error_m": float(np.abs(profile["x"] - base_profile["x"]).max()),
            "l_ref_unchanged": bool(abs(checks["l_ref_m"] - base_stats["l_ref_m"]) < 1e-9),
            "depth_unchanged": bool(abs(checks["depth_m"] - base_stats["depth_m"]) < 1e-9),
        },
    }
    # 舷弧面积曲线逐站明细留给可视化脚本用：这里只在 index.json 里存曲线本身
    profile_out = {
        "x_over_lpp": paper_axis.tolist(),
        "area_m2": profile["area"].tolist(),
        "waterline_half_beam_m": profile["waterline_half_beam"].tolist(),
    }
    dump(metadata, out_dir / f"{name}.json")
    return metadata, profile_out, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stl", default=str(ROOT / "dataset/classic_hulls/kcs.stl"))
    parser.add_argument("--out", default=str(ROOT / "dataset/kcs_var"))
    parser.add_argument("--lpp", type=float, default=DEFAULT_LPP)
    parser.add_argument("--draft", type=float, default=DEFAULT_DRAFT)
    parser.add_argument("--deck", type=float, default=DEFAULT_DECK)
    parser.add_argument("--variants", nargs="+", default=sorted(ffd.KCS_BOW_VARIANTS))
    args = parser.parse_args()

    source = Path(args.stl)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_points, base_faces = read_stl(source)
    base_stats = stats(source, args.lpp, args.draft)
    base_profile = station_profile(source, args.lpp, args.draft)
    base_stats.update({k: v for k, v in base_profile.items() if k != "grid"})
    print("母船 %s: %d 顶点 / %d 面, L_ref %.4f m, 排水量 %.5f m³, 湿面积 %.5f m²"
          % (source.name, base_stats["n_vertices"], base_stats["n_faces"],
             base_stats["l_ref_m"], base_stats["displacement_m3"],
             base_stats["wetted_area_m2"]))

    entries, profiles = {}, {}
    for name in args.variants:
        metadata, profile_out, _ = build_variant(
            name, base_points, base_faces, base_profile, base_stats,
            args.lpp, args.draft, args.deck, out_dir)
        entries[name] = metadata
        profiles[name] = profile_out
        geometry, checks = metadata["geometry"], metadata["checks"]
        print("%-12s 最大位移 %6.1f mm (%.3f%% Lpp)  排水量 %+.4f%%  湿面积 %+.4f%%  "
              "舷弧面积峰值 %.4f  翻转 %d  水线最大位移 %5.1f mm"
              % (name, 1e3 * geometry["max_abs_dy_m"], 100 * geometry["max_abs_dy_over_lpp"],
                 100 * geometry["volume_change"], 100 * geometry["wetted_area_change"],
                 geometry["section_area_peak_over_midship"], checks["flipped_triangles"],
                 1e3 * geometry["waterline_max_abs_dy_m"]))

    index = {
        "hull": "kcs",
        "parent_stl": str(source.relative_to(ROOT)),
        "parent": {
            "lpp_m": args.lpp,
            "draft_m": args.draft,
            "deck_m": args.deck,
            "n_vertices": base_stats["n_vertices"],
            "n_faces": base_stats["n_faces"],
            "l_ref_m": base_stats["l_ref_m"],
            "displacement_m3": base_stats["displacement_m3"],
            "wetted_area_m2": base_stats["wetted_area_m2"],
            "half_beam_m": base_stats["half_beam_m"],
            "depth_m": base_stats["depth_m"],
            "watertight": base_stats["watertight"],
            "n_components": base_stats["n_components"],
        },
        "paper": PAPER,
        "variants": {name: {
            "file": entry["geometry"]["file"],
            "metadata": f"{name}.json",
            "volume_fraction": entry["paper"]["volume_fraction"],
            "volume_sign": entry["paper"]["volume_sign"],
            "max_abs_dy_over_lpp": entry["geometry"]["max_abs_dy_over_lpp"],
            "volume_change": entry["geometry"]["volume_change"],
            "wetted_area_change": entry["geometry"]["wetted_area_change"],
        } for name, entry in entries.items()},
        "base_profile": {
            "x_over_lpp": ffd.paper_x_over_lpp(base_profile["x"], args.lpp).tolist(),
            "area_m2": base_profile["area"].tolist(),
            "waterline_half_beam_m": base_profile["waterline_half_beam"].tolist(),
        },
        "variant_profiles": profiles,
    }
    dump(index, out_dir / "index.json")
    print("写出 %s" % out_dir)


if __name__ == "__main__":
    main()
