"""把集群导出包组装成一条数据集条目。

用法:
    python3 src/dataset/build_entry.py --bundle dataset/.build_cache/<source-case> \
        --dataset dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np

from foam_field import read_scalar_field
from foam_vtk import read_internal, read_patch
from hull_pressure import face_to_vertex_area_weighted

FIELDS = ("U", "p_rgh", "alpha.water")


def entry_identity(manifest):
    """从算例名与傅汝德数生成条目 id：<船型>_fr<三位整数>_<网格档位>。"""
    parts = manifest["run"].split("_")
    if len(parts) < 3:
        raise ValueError(f"算例名 {manifest['run']} 不含 <船型>_<网格档位>_<时长> 结构")
    hull = parts[0]
    mesh_level = "_".join(parts[1:-1])
    fr_code = round(manifest["froude"] * 100)
    return f"{hull}_fr{fr_code:03d}_{mesh_level}", hull, mesh_level


def build_entry(bundle, dataset_root):
    bundle = Path(bundle)
    dataset_root = Path(dataset_root)
    manifest = json.loads((bundle / "case_manifest.json").read_text())
    reference = json.loads((bundle / "kcs_reference.json").read_text())

    entry_id, hull, mesh_level = entry_identity(manifest)
    out = dataset_root / entry_id
    out.mkdir(parents=True, exist_ok=True)

    cell_points, cell_fields = read_internal(bundle / "internal.vtu", FIELDS)
    hull_points, hull_offsets, hull_faces, patch_fields = read_patch(bundle / "hull.vtp", ("p_rgh",))
    n_cells = len(cell_points)

    alpha_initial = read_scalar_field(bundle / "alpha.water.t0", n_cells)
    hull_pressure = face_to_vertex_area_weighted(
        hull_points, hull_offsets, hull_faces, patch_fields["p_rgh"]
    )

    alpha_final = cell_fields["alpha.water"]
    # interFoam 的 alpha 有界性允许 1e-6 级过冲，这里只拦明显越界
    if alpha_final.min() < -1e-4 or alpha_final.max() > 1.0 + 1e-4:
        raise ValueError(f"{entry_id} 的 alpha.water 越界: [{alpha_final.min()}, {alpha_final.max()}]")
    if not (np.isfinite(cell_fields["U"]).all() and np.isfinite(cell_fields["p_rgh"]).all()):
        raise ValueError(f"{entry_id} 的 U 或 p_rgh 含非有限值")

    geometry = reference["geometry"]
    fluid = reference["fluid"]
    speed = manifest["speed"]
    s0 = geometry["S0_with_rudder_m2"]

    arrays = {
        "hull_points": hull_points,
        "cell_points": cell_points,
        "alpha_water_initial": alpha_initial,
        "alpha_water_final": alpha_final,
        "U_final": cell_fields["U"],
        "p_rgh_final": cell_fields["p_rgh"],
        "hull_pressure": hull_pressure,
    }
    for name, array in arrays.items():
        dtype = np.int32 if array.dtype.kind in "iu" else np.float32
        np.save(out / f"{name}.npy", np.ascontiguousarray(array, dtype=dtype))

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
        "Lpp": geometry["Lpp_m"],
        "T": geometry["T_m"],
        "S0": s0,
        "attitude": {"mode": "fixed_design_draught", "sinkage_m": 0.0, "trim_deg": 0.0},
        "mesh": {
            "n_cells": n_cells,
            "n_hull_points": int(len(hull_points)),
            "n_hull_faces": int(len(hull_offsets) - 1),
        },
        "pressure_map": "face_to_vertex_area_weighted_v1",
        "initial_state": {
            "U_zero": True,
            "free_surface_z": geometry["T_m"],
            "source": "0/alpha.water（setFields 写入的原始场）",
        },
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
        f"hull_points={meta['mesh']['n_hull_points']} "
        f"hull_faces={meta['mesh']['n_hull_faces']}"
    )


if __name__ == "__main__":
    main()
