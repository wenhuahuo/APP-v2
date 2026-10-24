"""把集群导出包组装成一条二维自由液面数据集条目。

用法:
    python3 src/dataset/build_entry.py --bundle dataset/.build_cache/<source-case> \
        --dataset dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np

from foam_vtk import probe_cell_fields, read_free_surface, read_internal, read_patch
from free_surface_2d import (
    FLOW_CHANNELS,
    flow_lattice,
    interpolate_to_lattice,
)
from hull_pressure import face_to_vertex_area_weighted
from hull_uv import half_hull_uv
from sdf import footprint_profile, signed_distance

FIELDS = ("U", "p_rgh", "alpha.water")

# 二维流场域：按 Lpp 归一化。波幅在 x/Lpp ∈ [-1.5, 2.0]、y/Lpp ∈ [-0.75, 0] 之外
# 已降到 0.01 m 量级，格距取 0.005 Lpp 可分辨近船波峰。
FLOW_BOX = {"x_over_Lpp": (-1.5, 2.0), "y_over_Lpp": (-0.75, 0.0), "spacing_over_Lpp": 0.005}


def case_parameters(bundle):
    """读出组装所需的船型、几何与流体参数。

    批量船型的 manifest 直接给 `geometry` 与 `condition`；KCS 的 manifest 只有工况，
    几何与流体在 `kcs_reference.json` 里（它同时是湿表面积 S0 的唯一来源）。
    manifest 的 `run` 一律写作 `<船型>[_<几何变体>]_fr<三位整数>_<网格档位>_<物理时长>`。
    FFD 变体的 manifest 带 `geometry_variant.name`，条目 id 因此多一段变体名。
    """
    manifest = json.loads((bundle / "case_manifest.json").read_text())
    if "geometry" in manifest:
        geometry, condition = manifest["geometry"], manifest["condition"]
        return {
            "run": manifest["run"],
            "hull": manifest["hull"],
            "variant": None,
            "mesh_level": manifest["mesh_level"],
            "Fr": condition["Fr"],
            "Re": condition["Re"],
            "U_inf": condition["U_m_s"],
            "Lpp": geometry["L_ref_m"],
            "T": geometry["T_m"],
            "S0": None,
            "rho": condition["rho"],
            "nu": condition["nu"],
            "scale": manifest["mesh_scale_to_kcs"],
            "end_time": manifest.get("end_time"),
        }
    reference = json.loads((bundle / "kcs_reference.json").read_text())
    parts = manifest["run"].split("_")
    # 派生算例的 manifest 直接给 mesh_level；早期算例名没有 Fr 段，靠位置切分
    mesh_level = manifest.get("mesh_level")
    if mesh_level is None:
        mesh_parts = parts[2:-1] if parts[1].startswith("fr") else parts[1:-1]
        mesh_level = "_".join(mesh_parts)
    return {
        "run": manifest["run"],
        "hull": parts[0],
        "variant": (manifest.get("geometry_variant") or {}).get("name"),
        "mesh_level": mesh_level,
        "Fr": manifest["froude"],
        "Re": manifest["speed"]["Re"],
        "U_inf": manifest["speed"]["U_m_s"],
        "Lpp": reference["geometry"]["Lpp_m"],
        "T": reference["geometry"]["T_m"],
        "S0": reference["geometry"]["S0_with_rudder_m2"],
        "rho": reference["fluid"]["rho_kg_m3"],
        "nu": reference["fluid"]["nu_m2_s"],
        "scale": reference["geometry"]["scale_lambda"],
        # 物理时长取自 manifest。早期算例目录名带 `_25s` 但已按 20 s 重新导出，
        # 目录名不再代表数据时刻
        "end_time": manifest.get("end_time"),
    }


def entry_identity(parameters):
    """条目 id：<船型>[_<几何变体>]_fr<三位整数>_<网格档位>。"""
    fr_code = round(parameters["Fr"] * 100)
    stem = parameters["hull"]
    if parameters["variant"]:
        stem = f"{stem}_{parameters['variant']}"
    return f"{stem}_fr{fr_code:03d}_{parameters['mesh_level']}"


def build_entry(bundle, dataset_root, box=FLOW_BOX):
    bundle = Path(bundle)
    dataset_root = Path(dataset_root)
    parameters = case_parameters(bundle)
    entry_id = entry_identity(parameters)
    out = dataset_root / entry_id
    out.mkdir(parents=True, exist_ok=True)

    internal = read_internal(bundle / "internal.vtu", FIELDS)
    hull_points, hull_offsets, hull_faces, patch_fields = read_patch(bundle / "hull.vtp", ("p_rgh",))

    lpp = parameters["Lpp"]
    draft = parameters["T"]
    # 批量船型没有参考 S0，用船体网格自身的水线以下顶点面积给出名义值
    s0 = parameters["S0"]

    alpha = internal["fields"]["alpha.water"].astype(np.float64)
    # MULES 的有界性有残差：实测高 Fr 粗网格算例（npl_round_bilge，Fr 0.40）末态
    # 有 0.2% 单元落在 1 + 2.3e-4，2.7% 单元落在 -2.9e-7。FFD 变形网格上残差更大：
    # `wide_beam`（半宽 +20%）有 3.9% 的迭代步出现 alpha > 1.001、末态最大 1.00112，
    # `fat_hull`（半宽/吃水 +12%）最大 1.00145。越界都在 1.0 附近，alpha = 0.5
    # 等值面不受影响，容差按实测放到 2e-3
    if alpha.min() < -1e-3 or alpha.max() > 1.0 + 2e-3:
        raise ValueError(f"{entry_id} 的 alpha.water 越界: [{alpha.min()}, {alpha.max()}]")
    velocity = internal["fields"]["U"].astype(np.float64)
    pressure = internal["fields"]["p_rgh"].astype(np.float64)
    if not (np.isfinite(velocity).all() and np.isfinite(pressure).all()):
        raise ValueError(f"{entry_id} 的 U 或 p_rgh 含非有限值")

    lattice = flow_lattice(lpp, box)
    profile = footprint_profile(hull_points, draft, lpp)
    sdf = signed_distance(lattice.points, profile, lpp)

    surface = read_free_surface(internal["grid"])
    # 船体投影内部（船体正上方的空腔）也会生成伪面片，插值前剔除
    outside = signed_distance(surface[:, :2], profile, lpp) > 0.0
    eta = interpolate_to_lattice(surface[outside, :2], surface[outside, 2], lattice) - draft
    valid = (sdf > 0.0) & np.isfinite(eta)

    probe_points = np.column_stack([lattice.points[valid], eta[valid] + draft])
    probed, found = probe_cell_fields(internal["grid"], probe_points, ("U", "p_rgh"))

    # vtkProbeFilter 对非结构网格的少数奇点（多面体单元与不共形界面）无法定位单元，
    # 这些格点剔出流场域
    keep = valid.copy()
    keep[valid] = found

    hull_uv = half_hull_uv(hull_points, hull_offsets, hull_faces, lpp, draft)
    hull_pressure = face_to_vertex_area_weighted(
        hull_points, hull_offsets, hull_faces, patch_fields["p_rgh"]
    )
    # 船体域取设计吃水以下的湿表面：干舷与甲板的 p_rgh 恒为零，是平凡目标
    wetted = hull_points[:, 2] <= draft

    arrays = {
        "flow_points": lattice.points[keep],
        "flow_sdf": sdf[keep],
        "flow_target": np.column_stack([eta[valid][found], probed["U"][found], probed["p_rgh"][found]]),
        "flow_quad_weights": np.full(int(keep.sum()), lattice.spacing ** 2),
        "hull_points": hull_points[wetted],
        "hull_uv": hull_uv["uv"][wetted],
        "hull_normals": hull_uv["normal"][wetted],
        "hull_quad_weights": hull_uv["vertex_area"][wetted],
        "hull_pressure": hull_pressure[wetted],
    }
    for name, array in arrays.items():
        np.save(out / f"{name}.npy", np.ascontiguousarray(array, dtype=np.float32))

    meta = {
        "entry_id": entry_id,
        "source_case": parameters["run"],
        "end_time": parameters["end_time"],
        "hull": parameters["hull"],
        "geometry_variant": parameters["variant"],
        "mesh_level": parameters["mesh_level"],
        "Fr": parameters["Fr"],
        "Re": parameters["Re"],
        "U_inf": parameters["U_inf"],
        "rho": parameters["rho"],
        "nu": parameters["nu"],
        "scale": parameters["scale"],
        "Lpp": lpp,
        "T": draft,
        "S0": float(2 * arrays["hull_quad_weights"].sum()) if s0 is None else s0,
        "S0_source": "mesh_wetted_area" if s0 is None else "reference",
        "attitude": {"mode": "fixed_design_draught", "sinkage_m": 0.0, "trim_deg": 0.0},
         "mesh": {
            "n_cells": int(len(alpha)),
            "n_hull_points": int(len(hull_points)),
            "n_hull_faces": int(len(hull_offsets) - 1),
            "n_hull_wetted_points": int(wetted.sum()),
        },
        "flow_field": {
            "domain_over_Lpp": {
                "x": list(box["x_over_Lpp"]),
                "y": list(box["y_over_Lpp"]),
                "spacing": box["spacing_over_Lpp"],
                "n_grid": [lattice.nx, lattice.ny],
            },
            "n_points": int(keep.sum()),
            "n_dropped_outside_mesh": int((~found).sum()),
            "channels": list(FLOW_CHANNELS),
            "eta": "alpha.water = 0.5 等值面高度插值到格点，相对设计吃水 T",
            "surface_velocity_pressure": "在 (x, y, eta + T) 处对 U、p_rgh 做单元内插值",
            "quadrature": "uniform h^2",
            "mask": "船体投影内部的格点（无自由液面）被掩除",
        },
        "hull_surface": {
            "domain": "设计吃水以下的湿表面（z ≤ T）",
            "uv": {
                "u": "x / Lpp",
                "v": "自龙骨（y≈0, z=z_min）沿剖面到设计水线的相对测地弧长，v = 1 即设计水线",
            },
            "channels": ["p_rgh"],
            "quadrature": "vertex_area",
            "pressure_map": "face_to_vertex_area_weighted_v1",
        },
        "input_channels": {"flow": ["sdf"], "condition": ["Fr", "Re", "U_inf"]},
        "symmetry": "starboard_half_y_le_0",
        "files": sorted([f"{name}.npy" for name in arrays] + ["meta.json"]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


def update_index(dataset_root, meta):
    path = Path(dataset_root) / "index.json"
    index = json.loads(path.read_text()) if path.exists() else {"entries": {}}
    index["entries"][meta["entry_id"]] = {
        "source_case": meta["source_case"],
        "hull": meta["hull"],
        "mesh_level": meta["mesh_level"],
        "Fr": meta["Fr"],
        "Re": meta["Re"],
        "n_cells": meta["mesh"]["n_cells"],
        "n_flow_points": meta["flow_field"]["n_points"],
        "n_hull_points": meta["mesh"]["n_hull_wetted_points"],
        "files": meta["files"],
    }
    index["n_entries"] = len(index["entries"])
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="集群导出包目录")
    parser.add_argument("--dataset", default="dataset", help="数据集根目录")
    args = parser.parse_args()

    meta = build_entry(args.bundle, args.dataset)
    update_index(args.dataset, meta)
    print(
        f"{meta['entry_id']}: n_cells={meta['mesh']['n_cells']} "
        f"flow_points={meta['flow_field']['n_points']} "
        f"hull_points={meta['mesh']['n_hull_points']}"
    )


if __name__ == "__main__":
    main()
