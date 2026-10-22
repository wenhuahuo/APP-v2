"""比较同一算例不同物理时刻的自由液面与船面压力，判断能否把物理时长截短。

用法:
    python3 scripts/dataset_build/compare_transient_times.py \
        --bundle-root ~/dataset_export_time_check --case-root ~/Documents/OpenFoamWorks/APPSolver-v2-Dataset \
        --cases kcs_normal_mesh_25s kcs_fr520_normal_mesh_25s \
        --times 15 20 25 --reference 25 --reference-json kcs_reference.json \
        --dataset dataset --out times.json

流场走与 `src/dataset/build_entry.py` 相同的重建路径（alpha.water = 0.5 等值面 +
单元内插值），船面压力走面心到顶点的面积加权插值。以最末时刻为参考，其余时刻与
参考的差异给三项度量：RMSE、按 Lpp（场）或动压（船面 Cp）归一化的相对量、R²。

截断误差是否影响学习，用相邻 Fr 工况的同名度量作尺度：时间截断的差异明显小于
相邻 Fr 的信号差时，它不构成数据集噪声。

对称面 y = 0 附近的两相界面重建不稳，比较时剔除 y/Lpp < 0.01 的中纵剖面带。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/dataset")]

from foam_vtk import probe_cell_fields, read_free_surface, read_internal, read_patch
from free_surface_2d import flow_lattice, interpolate_to_lattice
from hull_pressure import face_to_vertex_area_weighted
from sdf import footprint_profile, signed_distance

FIELDS = ("U", "p_rgh", "alpha.water")
FLOW_BOX = {"x_over_Lpp": (-1.5, 2.0), "y_over_Lpp": (-0.75, 0.0), "spacing_over_Lpp": 0.005}
FLOW_KEYS = ("eta", "u", "v", "w", "p_rgh")


def free_surface_of(internal, lpp, draft, profile, lattice, sdf):
    """重建一个时刻的自由液面二维场。

    返回逐通道的整格点数组，无效格点填 NaN：不同时刻等值面点数不同，只有落在
    共同有效格点上的值才能直接相减。
    """
    surface = read_free_surface(internal["grid"])
    outside = signed_distance(surface[:, :2], profile, lpp) > 0.0
    eta = interpolate_to_lattice(surface[outside, :2], surface[outside, 2], lattice) - draft
    valid = (sdf > 0.0) & np.isfinite(eta)
    probe_points = np.column_stack([lattice.points[valid], eta[valid] + draft])
    probed, found = probe_cell_fields(internal["grid"], probe_points, ("U", "p_rgh"))
    keep = valid.copy()
    keep[valid] = found
    values = {key: np.full(len(lattice.points), np.nan) for key in FLOW_KEYS}
    values["eta"][keep] = eta[valid][found]
    values["p_rgh"][keep] = probed["p_rgh"][found]
    for axis, name in enumerate(("u", "v", "w")):
        values[name][keep] = probed["U"][found, axis]
    return values


def metric(reference, other, scale):
    """RMSE、按尺度归一化的 RMSE、以参考为真值的 R²。"""
    diff = other - reference
    rmse = float(np.sqrt((diff ** 2).mean()))
    ss_res = float((diff ** 2).sum())
    ss_tot = float(((reference - reference.mean()) ** 2).sum())
    return {"rmse": rmse, "rmse_over_scale": rmse / scale, "r2_vs_reference": 1.0 - ss_res / ss_tot}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--case-root", required=True, help="算例根目录，用于读 U_m_s")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--times", nargs="+", type=int, required=True)
    parser.add_argument("--reference", type=int, required=True)
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--dataset", help="数据集根目录，用来量出相邻 Fr 的信号差")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ref = json.loads(Path(args.reference_json).read_text())
    geometry = ref["geometry"]
    lpp, draft = float(geometry["Lpp_m"]), float(geometry["T_m"])
    rho = float(ref["fluid"]["rho_kg_m3"])
    lattice = flow_lattice(lpp, FLOW_BOX)
    near_symmetry = np.abs(lattice.points[:, 1]) / lpp < 0.01

    report = {"times": args.times, "reference_time": args.reference, "cases": {}}
    for case in args.cases:
        bundle = Path(args.bundle_root).expanduser() / case
        manifest = json.loads((Path(args.case_root).expanduser() / case / "case_manifest.json").read_text())
        speed = float(manifest["speed"]["U_m_s"])
        dynamic = 0.5 * rho * speed ** 2

        # 船体几何三个时刻相同，读一次；船面压力逐时刻读
        points, offsets, faces, _ = read_patch(bundle / f"{args.reference}" / "hull.vtp", ("p_rgh",))
        profile = footprint_profile(points, draft, lpp)
        sdf = signed_distance(lattice.points, profile, lpp)
        wet = points[:, 2] <= draft

        fields = {}
        for t in args.times:
            internal = read_internal(bundle / f"{t}" / "internal.vtu", FIELDS)
            field = free_surface_of(internal, lpp, draft, profile, lattice, sdf)
            patch = read_patch(bundle / f"{t}" / "hull.vtp", ("p_rgh",))
            pressure = face_to_vertex_area_weighted(points, offsets, faces, patch[3]["p_rgh"])
            field["cp"] = pressure[wet] / dynamic
            fields[t] = field

        common = ~near_symmetry.copy()
        for t in args.times:
            common &= np.isfinite(fields[t]["eta"])
        compared = {}
        for t in args.times:
            if t == args.reference:
                continue
            row = {}
            for key in FLOW_KEYS + ("cp",):
                reference = fields[args.reference][key]
                other = fields[t][key]
                if key != "cp":
                    reference, other = reference[common], other[common]
                scale = (lpp if key == "eta" else dynamic if key == "cp"
                         else float(np.abs(reference).max()))
                row[key] = metric(reference, other, scale)
            compared[str(t)] = row
        report["cases"][case] = {
            "speed_m_s": speed, "dynamic_pressure_Pa": dynamic,
            "points_compared": int(common.sum()), "compared": compared,
        }

    if args.dataset:
        report["adjacent_fr_signal"] = adjacent_fr_signal(Path(args.dataset), lpp, rho)

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def adjacent_fr_signal(dataset, lpp, rho):
    """相邻 Fr 条目之间自由液面与船面 Cp 的 RMS 差，作为工况间的信号尺度。"""
    entries = sorted(p for p in dataset.iterdir()
                     if p.is_dir() and p.name.startswith("kcs_fr") and p.name.endswith("normal_mesh"))
    signal = {}
    for a, b in zip(entries, entries[1:]):
        ta, tb = np.load(a / "flow_target.npy"), np.load(b / "flow_target.npy")
        if ta.shape != tb.shape:
            continue
        meta_a = json.loads((a / "meta.json").read_text())
        meta_b = json.loads((b / "meta.json").read_text())
        cp_a = np.load(a / "hull_pressure.npy") / (0.5 * rho * meta_a["U_inf"] ** 2)
        cp_b = np.load(b / "hull_pressure.npy") / (0.5 * rho * meta_b["U_inf"] ** 2)
        eta = float(np.sqrt(((ta[:, 0] - tb[:, 0]) ** 2).mean()))
        signal[f"{a.name}__{b.name}"] = {
            "eta_rmse": eta,
            "eta_rmse_over_lpp": eta / lpp,
            "cp_rmse": float(np.sqrt(((cp_a - cp_b) ** 2).mean())) if cp_a.shape == cp_b.shape else None,
        }
    return signal


if __name__ == "__main__":
    main()
