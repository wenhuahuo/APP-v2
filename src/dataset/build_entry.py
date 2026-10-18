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


def entry_identity(manifest):
    """从算例名与傅汝德数生成条目 id：<船型>_fr<三位整数>_<网格档位>。"""
    parts = manifest["run"].split("_")
    if len(parts) < 3:
        raise ValueError(f"算例名 {manifest['run']} 不含 <船型>_<网格档位>_<时长> 结构")
    hull = parts[0]
    mesh_level = "_".join(parts[1:-1])
    fr_code = round(manifest["froude"] * 100)
    return f"{hull}_fr{fr_code:03d}_{mesh_level}", hull, mesh_level


def build_entry(bundle, dataset_root, box=FLOW_BOX):
    bundle = Path(bundle)
    dataset_root = Path(dataset_root)
    manifest = json.loads((bundle / "case_manifest.json").read_text())
    reference = json.loads((bundle / "kcs_reference.json").read_text())

    entry_id, hull, mesh_level = entry_identity(manifest)
    out = dataset_root / entry_id
    out.mkdir(parents=True, exist_ok=True)

    internal = read_internal(bundle / "internal.vtu", FIELDS)
    hull_points, hull_offsets, hull_faces, patch_fields = read_patch(bundle / "hull.vtp", ("p_rgh",))

    geometry = reference["geometry"]
    fluid = reference["fluid"]
    speed = manifest["speed"]
    lpp = geometry["Lpp_m"]
    draft = geometry["T_m"]
    s0 = geometry["S0_with_rudder_m2"]

    alpha = internal["fields"]["alpha.water"].astype(np.float64)
    # interFoam 的 alpha 有界性允许 1e-6 级过冲，这里只拦明显越界
    if alpha.min() < -1e-4 or alpha.max() > 1.0 + 1e-4:
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
        "source_case": manifest["run"],
        "hull": hull,
        "mesh_level": mesh_level,
        "Fr": manifest["froude"],
        "Re": speed["Re"],
        "U_inf": speed["U_m_s"],
        "rho": fluid["rho_kg_m3"],
        "nu": fluid["nu_m2_s"],
        "scale": geometry["scale_lambda"],
        "Lpp": lpp,
        "T": draft,
        "S0": s0,
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
        "n_hull_points": meta["mesh"]["n_hull_points"],
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
