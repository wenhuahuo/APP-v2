"""基准 KCS 与两个艏部 FFD 变体的 Michell 势流波阻（复用 ShipD 的 `ModMichell`）。

三个船型的 STL（`dataset/classic_hulls/kcs.stl`、`dataset/kcs_var/bow_outward.stl`、
`dataset/kcs_var/bow_inward.stl`）各自生成同一套 `Y(x, z)` 半宽格点，再直接调用
`ModifiedMichellCw.ModMichell(Y, U, X, Z, RHO, N)`，积分公式不改写、不做可视化。

几何口径
--------

- 坐标即 STL 坐标：x 从艉到艏、z 向上、龙骨在 z ≈ 0，设计吃水水线是 z = T = 0.3418 m，
  单位米；船体左右舷对称（y ∈ [−0.5095, 0.5095]），半宽按 `|y|` 取。
- `Y(x, z)` = 该 (x, z) 上船体表面 `|y|` 的最大值（外包半宽）。带球鼻艏 / 多段剖面
  的站位（艏部 x/Lpp ≳ 0.97、艉部）在同一 z 上取最外侧表面，不做任何几何替换。
- `X`：`linspace(0, L, Nx)`，`Nx` 取奇数；`L = 7.5779 m` 是三个 STL 水下纵向尺度
  （艉部水线最后端 x = −0.0776 到球鼻艏最前端 x = 7.5003），三个船型共用同一 `L`、
  同一站位（FFD 只动 y，x/z 不变，实测三个 STL 的水下 x/z 范围逐位相同）。
- `Z`：`linspace(-T, 0, Nz)`，与 ShipD `HullParameterization.gen_PC_for_Cw` 的返回
  顺序一致（该函数返回 `Z - Z[-1]`，即从 −draft 升序到 0），覆盖设计吃水到龙骨。

波阻口径
--------

`ModMichell` 的返回值直接作为全船波阻，**不再乘 2**。依据：

1. 薄船理论的经典 Michell 积分为（半宽 b、全船）
   `Rw = (4ρU²k0²/π) ∫₀^{π/2} sec³θ |∫∫ ∂b/∂x·exp(k0 z sec³… )|² dθ`，
   系数 `4 = 2²` 是左右舷两片源汇（σ = 2U ∂b/∂x）在 |·|² 里的镜像因子。
2. 代码里 `c = 4ρU²/π`、`k = k0 sec²θ`、`a = secθ`，其 θ 积分核的三项
   `k²(P²+Q²) + 2ka(Q·Pt − P·Qt) + a²(Pt²+Qt²)` 正是
   `a²·|∫∫ ∂Y/∂x·exp(k0 z sec²θ)·exp(i k0 x secθ) dx dz|²` 的分部积分展开，
   即 `R(θ) = (4ρU²k0²/π)·sec³θ·|A_x(θ)|²`，与上面第 1 条的经典式逐项相同。
3. ShipD 的调用方 `HullParameterization.gen_PC_for_Cw` 把 `halfBeam_MidBody(z)`
   （半宽）作为 `Y` 传入，不再对返回值做任何缩放；其数据集侧的 `Cw = Rw/(0.5ρU²L²)`
   也把返回值当作全船波阻。

脚本用解析算例复核这一口径（`checks.closed_form_*`）：取 `Y = s·x`（与 z 无关的
线性半宽），Michell 积分有闭式
`Rw = (4ρU²k0²/π)∫₀^{π/2} sec³θ |s(1−e^{−kT})/k·(e^{iK L}−1)/(iK)|²dθ`（k = k0 sec²θ、
K = k0 secθ），与 `ModMichell` 的返回值比对。

工况与系数：ρ = 999.5 kg/m³、U = 2.196 m/s、Lpp = 7.2786 m、S0 = 9.5531 m²、
T = 0.3418 m，`Cw = Rw/(0.5ρU²S0)`。

用法:
    python3 scripts/eval/eval_kcs_michell.py
    python3 scripts/eval/eval_kcs_michell.py --michell /path/to/ModifiedMichellCw.py

产物: `output/kcs_michell_resistance.json`
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from trimesh.intersections import mesh_plane

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "dataset"))

from stl_io import read_stl  # noqa: E402

MICHELL = Path(
    "/path/to/local-home/Documents/OpenFoamWorks/ShipMaster/baselines/shipd/"
    "upstream/ShipD/ModifiedMichellCw.py")
HULLS = {
    "base": ROOT / "dataset" / "classic_hulls" / "kcs.stl",
    "bow_outward": ROOT / "dataset" / "kcs_var" / "bow_outward.stl",
    "bow_inward": ROOT / "dataset" / "kcs_var" / "bow_inward.stl",
}
OUTPUT = ROOT / "output" / "kcs_michell_resistance.json"
TEST_JSON = ROOT / "output" / "kcs_optim_application_test.json"

CONDITION = {
    "Fr": 0.26,
    "Re": 12600000.0,
    "U_inf": 2.196,
    "rho": 999.5,
    "Lpp": 7.2786,
    "draft": 0.3418,
    "s0_reference_m2": 9.5531,
}
G = 9.81
# 收敛检查：(Nx, Nz, N)。最后一组是报告采用的结果。
CONVERGENCE = [(121, 41, 40), (241, 61, 80), (481, 81, 160), (961, 121, 320)]
TOL = 1e-12


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_modmichell(path):
    """importlib 直接加载参考实现，不改动该文件。"""
    spec = importlib.util.spec_from_file_location("modified_michell_cw", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submerged_extent(mesh, draft, n_levels=41):
    """水下部分（z ∈ [0, draft]）的 x 范围：逐层水平剖面的 x 极值并集。"""
    x_lo, x_hi = np.inf, -np.inf
    levels = np.linspace(0.0, draft, n_levels)
    for z in levels:
        segments = mesh_plane(mesh, (0.0, 0.0, 1.0), (0.0, 0.0, float(z)))
        if len(segments) == 0:
            continue
        x_lo = min(x_lo, float(segments[:, :, 0].min()))
        x_hi = max(x_hi, float(segments[:, :, 0].max()))
    return float(x_lo), float(x_hi)


def waterplane_extent(mesh, draft):
    segments = mesh_plane(mesh, (0.0, 0.0, 1.0), (0.0, 0.0, float(draft)))
    return float(segments[:, :, 0].min()), float(segments[:, :, 0].max())


def half_breadth(mesh, stations, waterlines):
    """每个 (x, z) 处船体表面 |y| 的最大值（外包半宽）。

    `stations` / `waterlines` 是 STL 坐标系下的值。返回 (Y (Nx, Nz), 逐站覆盖水线数)。
    """
    Y = np.zeros((len(stations), len(waterlines)))
    covered = np.zeros(len(stations), dtype=int)
    for i, x in enumerate(stations):
        segments = mesh_plane(mesh, (1.0, 0.0, 0.0), (float(x), 0.0, 0.0))
        if len(segments) == 0:
            continue
        y1, z1 = segments[:, 0, 1], segments[:, 0, 2]
        y2, z2 = segments[:, 1, 1], segments[:, 1, 2]
        lo = np.minimum(z1, z2)
        hi = np.maximum(z1, z2)
        for j, z in enumerate(waterlines):
            mask = (lo <= z + TOL) & (hi >= z - TOL)
            if not mask.any():
                continue
            dz = z2[mask] - z1[mask]
            ratio = np.where(np.abs(dz) > 1e-15,
                             (z - z1[mask]) / np.where(dz == 0.0, 1.0, dz), 0.0)
            values = np.concatenate([y1[mask] + ratio * (y2[mask] - y1[mask]),
                                     y1[mask], y2[mask]])
            Y[i, j] = float(np.abs(values).max())
            covered[i] += 1
    return Y, covered


def section_data(Y, stations, waterlines):
    """格点几何摘要：水线半宽、各站面积、水线面面积（waterlines 升序，末行是设计水线）。"""
    area = np.trapezoid(Y, waterlines, axis=1)            # 半船剖面面积
    waterline = Y[:, -1]
    return {
        "waterline_half_beam_max_m": float(waterline.max()),
        "half_beam_max_m": float(Y.max()),
        "half_beam_max_station_x_over_Lpp": float(
            stations[int(np.unravel_index(Y.argmax(), Y.shape)[0])] / CONDITION["Lpp"]),
        "section_area_max_m2": float(area.max()),
        "section_area_max_full_m2": float(2.0 * area.max()),
        "waterplane_area_full_m2": float(2.0 * np.trapezoid(waterline, stations)),
    }


def closed_form_linear(slope, stations, waterlines, U, rho):
    """Y = s·x（与 z 无关）时 Michell 积分的闭式值，用于核验常数与全船口径。"""
    k0 = G / U ** 2
    L = stations[-1]
    T = -waterlines[0]
    dz = waterlines[1] - waterlines[0]
    theta = np.linspace(0.0, np.pi / 2.0, 200001)[:-1]
    a = 1.0 / np.cos(theta)
    K = k0 * a
    k = k0 * a ** 2
    z_integral = (1.0 - np.exp(-k * T)) / k
    x_integral = (np.exp(1j * K * L) - 1.0) / (1j * K)
    amplitude = slope * z_integral * x_integral
    integrand = a ** 3 * np.abs(amplitude) ** 2
    # 与 ModMichell 相同的 Z 网格跨度（-T 到 0），保证 dz/T 一致
    return float((4.0 * rho * U ** 2 * k0 ** 2 / np.pi)
                 * np.trapezoid(integrand, theta)), float(T), float(L), float(dz)


def z_quadrature_check(U, draft, nz):
    """复核 ModMichell 的 z 向指数权重：Σ w·exp(k z)·dz 对解析积分 (1−e^{−kT})/k。"""
    k0 = G / U ** 2
    dz = draft / (nz - 1)
    Z = np.linspace(-draft, 0.0, nz)
    out = {}
    for k in (0.5, 2.0, 8.0):
        a = np.sqrt(k / k0)
        Kz = k0 * dz * a ** 2
        Kzsq = Kz ** 2
        w = np.full(nz, (np.exp(Kz) - 1.0 + np.exp(-Kz) - 1.0) / Kzsq)
        w[0] = (np.exp(Kz) - 1.0 - Kz) / Kzsq
        w[-1] = (np.exp(-Kz) - 1.0 + Kz) / Kzsq
        numeric = float(np.sum(w * np.exp(k * Z) * dz))
        analytic = float((1.0 - np.exp(-k * draft)) / k)
        out[f"k={k:g}"] = {"numeric": numeric, "analytic": analytic,
                           "rel_error": numeric / analytic - 1.0}
    return out


def run_config(module, meshes, extents, draft, config, U, rho):
    nx, nz, n = config
    x_lo, x_hi = extents["base"]
    length = x_hi - x_lo
    stations = np.linspace(0.0, length, nx)
    waterlines = np.linspace(-draft, 0.0, nz)
    grid = {
        "Nx": nx, "Nz": nz, "N": n,
        "nx_odd": nx % 2 == 1,
        "x_start_m": 0.0, "x_end_m": length, "dx_m": length / (nx - 1),
        "z_start_m": -draft, "z_end_m": 0.0, "dz_m": draft / (nz - 1),
        "x_offset_m": x_lo, "z_offset_m": draft,
        "L_integration_m": length, "L_over_Lpp": length / CONDITION["Lpp"],
        "station_geometry_x_m": [float(stations[0] + x_lo), float(stations[-1] + x_lo)],
        "waterline_geometry_z_m": [0.0, draft],
    }
    hulls = {}
    for name, mesh in meshes.items():
        Y, covered = half_breadth(mesh, stations + x_lo, draft + waterlines)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            rw = float(module.ModMichell(Y, U, stations, waterlines, rho, n))
        hulls[name] = {
            "rw_N": rw,
            "cw_S0": rw / (0.5 * rho * U ** 2 * CONDITION["s0_reference_m2"]),
            "stations_with_section": int((covered > 0).sum()),
            "waterlines_covered_min": int(covered.min()),
            "geometry": section_data(Y, stations, waterlines),
        }
    return grid, hulls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--michell", type=Path, default=MICHELL,
                        help="ModMichell 参考实现路径")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    module = load_modmichell(args.michell)
    U = CONDITION["U_inf"]
    rho = CONDITION["rho"]
    draft = CONDITION["draft"]

    meshes = {}
    extents = {}
    geometry = {}
    for name, path in HULLS.items():
        points, faces = read_stl(path)
        mesh = trimesh.Trimesh(points, faces, process=False)
        meshes[name] = mesh
        extents[name] = submerged_extent(mesh, draft)
        wp = waterplane_extent(mesh, draft)
        geometry[name] = {
            "stl": str(path.relative_to(ROOT)),
            "stl_md5": md5(path),
            "n_vertices": int(len(points)),
            "n_faces": int(len(faces)),
            "x_extent_submerged_m": list(extents[name]),
            "L_integration_m": extents[name][1] - extents[name][0],
            "waterplane_x_extent_m": list(wp),
            "waterplane_length_m": wp[1] - wp[0],
            "half_beam_m": float(np.abs(points[:, 1]).max()),
            "z_keel_m": float(points[:, 2].min()),
            "z_max_m": float(points[:, 2].max()),
        }

    base_extent = extents["base"]
    for name, extent in extents.items():
        if not np.allclose(extent, base_extent, atol=1e-9):
            raise SystemExit(f"{name} 的水下 x 范围 {extent} 与基准 {base_extent} 不一致")

    convergence = []
    for config in CONVERGENCE:
        grid, hulls = run_config(module, meshes, extents, draft, config, U, rho)
        convergence.append({"grid": grid, "hulls": hulls})
    reported = convergence[-1]
    for entry in convergence:
        for name in HULLS:
            entry["hulls"][name]["rel_diff_vs_reported"] = (
                entry["hulls"][name]["rw_N"] / reported["hulls"][name]["rw_N"] - 1.0)

    results = {}
    for name in HULLS:
        hull = reported["hulls"][name]
        results[name] = {
            "rw_N": hull["rw_N"],
            "cw_S0": hull["cw_S0"],
            "cw_definition": "Cw = Rw / (0.5 rho U^2 S0)，S0 = 9.5531 m^2",
            "rw_over_cfd_pressure_N": None,
            "geometry": hull["geometry"],
            "stations_with_section": hull["stations_with_section"],
            "waterlines_covered_min": hull["waterlines_covered_min"],
        }

    variants = {}
    for name in ("bow_outward", "bow_inward"):
        drw = results[name]["rw_N"] - results["base"]["rw_N"]
        variants[name] = {
            "michell_delta_N": drw,
            "michell_delta_percent": drw / results["base"]["rw_N"] * 100.0,
            "michell_delta_cw_S0": results[name]["cw_S0"] - results["base"]["cw_S0"],
        }

    comparison = {"source": str(TEST_JSON.relative_to(ROOT)), "cfd": {}, "model": {}}
    if TEST_JSON.exists():
        test = json.loads(TEST_JSON.read_text())
        for name in HULLS:
            case = test["cfd"]["cases"][name]
            pred = test["prediction"][name]
            comparison["cfd"][name] = {
                "resistance_total_N": case["resistance_total_N"],
                "resistance_pressure_N": case["resistance_pressure_N"],
                "cd_pressure_S0": case["cd_pressure_S0"],
            }
            comparison["model"][name] = {
                "resistance_full_ship_N": pred["resistance_full_ship_N"],
                "cd_pressure_S0": pred["cd_pressure_S0"],
            }
            results[name]["rw_over_cfd_pressure_N"] = (
                results[name]["rw_N"] / case["resistance_pressure_N"])
            results[name]["rw_over_cfd_total_N"] = (
                results[name]["rw_N"] / case["resistance_total_N"])
        for name in ("bow_outward", "bow_inward"):
            variants[name].update({
                "cfd_delta_pressure_N": test["variants"][name]["cfd_delta_N"],
                "cfd_delta_pressure_percent": test["variants"][name]["cfd_delta_percent"],
                "cfd_delta_total_N": test["variants"][name]["cfd_total_delta_N"],
                "cfd_delta_total_percent": test["variants"][name]["cfd_total_delta_percent"],
                "model_delta_pressure_N": test["variants"][name]["model_delta_N"],
                "model_delta_pressure_percent": test["variants"][name]["model_delta_percent"],
            })

    rank = {
        "michell": [name for name in sorted(HULLS, key=lambda k: results[k]["rw_N"])],
        "cfd_pressure": [name for name in sorted(
            HULLS, key=lambda k: comparison["cfd"][k]["resistance_pressure_N"])],
        "cfd_total": [name for name in sorted(
            HULLS, key=lambda k: comparison["cfd"][k]["resistance_total_N"])],
        "model": [name for name in sorted(
            HULLS, key=lambda k: comparison["model"][k]["resistance_full_ship_N"])],
    }

    slope = 0.05
    nx, nz, n = reported["grid"]["Nx"], reported["grid"]["Nz"], reported["grid"]["N"]
    stations = np.linspace(0.0, reported["grid"]["L_integration_m"], nx)
    waterlines = np.linspace(-draft, 0.0, nz)
    Y_linear = np.tile(slope * stations[:, None], (1, nz))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rw_linear = float(module.ModMichell(Y_linear, U, stations, waterlines, rho, n))
    rw_closed, T_closed, L_closed, dz_closed = closed_form_linear(
        slope, stations, waterlines, U, rho)

    payload = {
        "test": "kcs_michell_wave_resistance",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_implementation": {
            "file": str(args.michell),
            "function": "ModMichell(Y, U, X, Z, RHO, N)",
            "md5": md5(args.michell),
            "loaded_by": "importlib.util.spec_from_file_location（不改动参考文件）",
            "theory": "Michell 薄船理论势流波阻（线性自由面、深水、无粘、无航态变化）",
        },
        "condition": {**CONDITION,
                      "pressure_scale_0.5_rho_u2": 0.5 * rho * U ** 2,
                      "gravity_m_s2": G},
        "conventions": {
            "x_axis": "STL 坐标，x 由艉指向艏，单位米",
            "z_axis": "STL 坐标，z 向上，龙骨 z ≈ 0，设计吃水水线 z = 0.3418 m",
            "Z_input": "linspace(-draft, 0, Nz)，与 ShipD gen_PC_for_Cw 的升序返回一致",
            "y_definition": "每个 (x, z) 上船体表面 |y| 的最大值（左右舷对称的外包半宽，米）",
            "y_multi_section": "同一 z 上存在多段表面（球鼻艏、艉部）时取最外侧表面，"
                               "不替换几何",
            "full_ship_factor": "ModMichell 返回值为全船波阻，不再乘 2（见 checks）",
            "cw_definition": "Cw = Rw / (0.5 rho U^2 S0)",
        },
        "geometry": geometry,
        "reported_grid": reported["grid"],
        "results": results,
        "variants": variants,
        "ranking": rank,
        "comparison": comparison,
        "checks": {
            "z_quadrature_vs_analytic": z_quadrature_check(
                U, draft, reported["grid"]["Nz"]),
            "closed_form_linear_half_breadth": {
                "profile": "Y = 0.05 * x（与 z 无关）",
                "T_m": T_closed, "L_m": L_closed, "dz_m": dz_closed, "N": n,
                "modmichell_N": rw_linear,
                "closed_form_N": rw_closed,
                "rel_error": rw_linear / rw_closed - 1.0,
                "closed_form": "(4 rho U^2 k0^2 / pi) * int_0^{pi/2} sec^3(t) "
                               "|s(1-exp(-kT))/k * (exp(iKL)-1)/(iK)|^2 dt, "
                               "k = k0 sec^2(t), K = k0 sec(t)",
            },
            "theta_kernel_identity": "R(Theta) = (4 rho U^2 k0^2 / pi) * sec^3(Theta) * "
                                     "|int int dY/dx exp(k0 z sec^2) exp(i k0 x sec) dx dz|^2，"
                                     "代码的 k^2(P^2+Q^2) + 2ka(Q Pt - P Qt) + a^2(Pt^2+Qt^2) "
                                     "即该模平方的 a^2 倍",
            "all_zero_half_breadth_yields_zero": None,
        },
        "convergence": {
            "configs": convergence,
            "reported_config": [nx, nz, n],
            "max_abs_rel_diff_vs_reported": {
                name: max(abs(entry["hulls"][name]["rel_diff_vs_reported"])
                          for entry in convergence[:-1])
                for name in HULLS},
        },
    }
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        payload["checks"]["all_zero_half_breadth_yields_zero"] = float(
            module.ModMichell(np.zeros((nx, nz)), U, stations, waterlines, rho, n))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"grid Nx={nx} Nz={nz} N={n}  L={reported['grid']['L_integration_m']:.4f} m")
    for name in HULLS:
        print(f"  {name:12s} Rw = {results[name]['rw_N']:9.4f} N   "
              f"Cw = {results[name]['cw_S0']:.5e}")
    for name, delta in variants.items():
        print(f"  {name:12s} dRw = {delta['michell_delta_N']:+.4f} N  "
              f"({delta['michell_delta_percent']:+.2f} %)")
    print("closed-form check rel error = "
          f"{payload['checks']['closed_form_linear_half_breadth']['rel_error']:+.2e}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
