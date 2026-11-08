"""KCS 艏部 FFD 变体的下游验证：APP+OFormer 的船面 Cp 积分 vs 新全量 CFD。

用主实验口径训练的 `appsolver_oformer_random_seed0_isoflops`（双域输入/输出、逐层 Fourier add、
`lambda_hull=1.0`，含船面输出头）
在基准 KCS 与两个 FFD 变体上预测船面 Cp，沿半船湿表面做轴向压力积分，得到全船压差阻力，
与三个算例 `force.dat` 的 20–25 s 窗口压力项对照。几何对应、FFD 点变形、法向/求积权重更新、
变体水线 SDF 与积分口径全部复用 `probe_kcs_ffd_pressure.py`，本脚本只做本轮的 CFD 读数、
相对/绝对误差与最小积分核验。

CFD 口径（见数据集 `AGENT.md` 与 `calib/README.md`）
--------------------------------------------------

- 算例为半域（`midPlane` 为 `symmetryPlane`，y ≤ 0），`force.dat` 是半船力，
  `R = 2|Fx|`；来流沿 -x（`U internalField = (-2.196 0 0)`），阻力 Fx < 0。
- 时间窗口取 20–25 s 均值（数据集验收判据：`R_T = 2|Fx|` 在 20–25 s 窗口的均值）。
- 分量：`pressure_x` / `viscous_x` / `total_x` 逐列独立取窗均值，再乘 2 取绝对值。
- 基准算例的 `postProcessing/forces/0/` 只到 t=2.887 s（Co=1 段），续算段在
  `postProcessing/forces/2/`（t≥2.0 s），窗口内数据全部来自后者。

用法:
    python3 scripts/eval/verify_kcs_optim_application.py \
        --dataset-root /path/to/openfoam-cases

产物: `output/kcs_optim_application_test.json`、
      `dataset/kcs_var/cfd_force_summary.json`、`dataset/kcs_var/cfd_forces_window20to25.npz`

除阻力对照外，脚本还记录三个算例的收尾状态（末态时刻 / `Mesh OK.` / 末行 `End`）、
变体算例船体 STL 与仓库变体 STL 的 md5 一致性，以及模型在基准船面的 Cp RMSE。
"""

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from probe_kcs_ffd_pressure import (  # noqa: E402
    HULLS,
    ROOT,
    ShipDataset,
    Normalizer,
    axial_resistance,
    build_hull,
    entry_batch,
    hull_geometry,
    predict_cp,
)

from build import build_model  # noqa: E402

ENTRY = "kcs_fr026_normal_mesh"
MODEL_RUN = "appsolver_oformer_random_seed0_isoflops"
# 该 run 的产物在归档目录（同名活跃目录被移走后 NFS 缓存仍解析到旧位置）
MODEL_DIR = (ROOT / "output" / "archive" / "pre_dual_domain_fourier_add"
             / f"{MODEL_RUN}__archived_20261108_134104")
CHECKPOINT = MODEL_DIR / "best.pt"
STATS = ROOT / "dataset" / "shipbench" / "stats_random_seed0.json"
VARIANTS = ROOT / "dataset" / "kcs_var"
DEFAULT_DATA_ROOT = Path(
    "/path/to/openfoam-cases")
OUTPUT = ROOT / "output" / "kcs_optim_application_test.json"
FORCE_SUMMARY = VARIANTS / "cfd_force_summary.json"
FORCE_SERIES = VARIANTS / "cfd_forces_window20to25.npz"

WINDOW = (20.0, 25.0)
CASES = {
    "base": {"case": "kcs_normal_mesh_25s", "slurm_job": None},
    "bow_outward": {"case": "kcs_ffd_bow_outward_fr026_normal_mesh_25s", "slurm_job": 14803},
    "bow_inward": {"case": "kcs_ffd_bow_inward_fr026_normal_mesh_25s", "slurm_job": 14804},
}
# force.dat 列序：Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x ...
COLUMNS = {"time": 0, "total": 1, "pressure": 4, "viscous": 7}


def read_force_history(case_dir, window=WINDOW):
    """读该算例所有 `postProcessing/forces/<start>/force.dat`，返回窗口内的样本。"""
    files, blocks = [], []
    for path in sorted((case_dir / "postProcessing" / "forces").glob("*/force.dat")):
        table = np.loadtxt(path, comments="#", ndmin=2)
        files.append({"path": str(path.relative_to(case_dir)), "rows": int(len(table)),
                      "t_first": float(table[0, 0]), "t_last": float(table[-1, 0])})
        keep = (table[:, 0] >= window[0]) & (table[:, 0] <= window[1])
        if keep.any():
            blocks.append(table[keep])
    if not blocks:
        raise RuntimeError(f"{case_dir.name} 在 {window} s 窗口内没有 force.dat 样本")
    return files, np.vstack(blocks)


def log_summary(case_dir):
    """求解日志的末态时刻、末段执行时间、正常结束标记与网格门控。"""
    text = (case_dir / "log.interFoam").read_text(errors="ignore")
    times = [float(x) for x in re.findall(r"^Time = ([\d.eE+-]+)", text, re.M)]
    execution = [float(x) for x in re.findall(r"^ExecutionTime = ([\d.]+) s", text, re.M)]
    check_mesh = case_dir / "log.checkMesh"
    return {"last_time_s": times[-1] if times else None,
            "n_time_steps": len(times),
            "execution_time_s": execution[-1] if execution else None,
            "reached_end": bool(re.search(r"^End\s*$", text, re.M)),
            "n_procs_decomposed": len(list(case_dir.glob("processor*"))),
            "mesh_ok": bool(check_mesh.exists()
                            and re.search(r"^\s*Mesh OK\.",
                                          check_mesh.read_text(errors="ignore"), re.M))}


def digest(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def stl_identity(case_dir, variant_stl):
    """算例用的船体 STL（gz）与仓库内变体 STL 解压后逐字节一致。"""
    gz = sorted(case_dir.glob("*_Hull.stl.gz"))
    if not gz or variant_stl is None:
        return None
    with gzip.open(gz[0], "rb") as handle:
        case_md5 = hashlib.md5(handle.read()).hexdigest()
    repo_md5 = digest(variant_stl)
    return {"case_stl_gz": gz[0].name, "case_stl_md5": case_md5,
            "variant_stl": str(variant_stl.relative_to(ROOT)),
            "variant_stl_md5": repo_md5, "identical": case_md5 == repo_md5}


def cfd_resistance(case_dir, window=WINDOW):
    """半船 force.dat → 全船总阻力、压力项、粘性项（N）与窗口统计。"""
    files, table = read_force_history(case_dir, window)
    means = {key: float(table[:, index].mean()) for key, index in COLUMNS.items()}
    stds = {key: float(table[:, index].std(ddof=1)) for key, index in COLUMNS.items()}
    last = {key: float(table[-1, index]) for key, index in COLUMNS.items()}
    return {
        "case": case_dir.name,
        "case_path": str(case_dir),
        "force_files": files,
        "solution_log": log_summary(case_dir),
        "n_samples": int(len(table)),
        "t_first_s": float(table[0, COLUMNS["time"]]),
        "t_last_s": float(table[-1, COLUMNS["time"]]),
        "window_mean_half_ship_N": means,
        "window_std_half_ship_N": stds,
        "last_sample_half_ship_N": last,
        "resistance_total_N": -2.0 * means["total"],
        "resistance_pressure_N": -2.0 * means["pressure"],
        "resistance_viscous_N": -2.0 * means["viscous"],
        "resistance_total_std_N": 2.0 * stds["total"],
        "resistance_pressure_std_N": 2.0 * stds["pressure"],
        "resistance_pressure_at_last_time_N": -2.0 * last["pressure"],
        "pressure_fraction": (-means["pressure"]
                              / (-means["pressure"] - means["viscous"])),
    }, table


def load_checkpoint(device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint, config


def axis_factor(normals, weights):
    """轴向投影面积 Σ|n_x| A（半船），用于说明净阻力是正负项的抵消残差。"""
    return 2.0 * float((np.abs(normals[:, 0]) * weights).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="OpenFOAM 算例根目录")
    args = parser.parse_args()

    meta = json.loads((VARIANTS.parent / "shipbench" / ENTRY / "meta.json").read_text())
    lpp, draft = meta["Lpp"], meta["T"]
    rho, u_meta, s0 = meta["rho"], meta["U_inf"], meta["S0"]
    pressure_scale = 0.5 * rho * u_meta ** 2
    sample = ShipDataset(VARIANTS.parent / "shipbench", [ENTRY], stride=1)[0]
    normalizer = Normalizer.load(STATS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CFD：三个算例的窗口阻力
    cfd_results, window_series = {}, {}
    for hull, spec in CASES.items():
        case_dir = args.dataset_root / spec["case"]
        record, table = cfd_resistance(case_dir)
        record["slurm_job"] = spec["slurm_job"]
        variant_stl = (VARIANTS / f"{hull}.stl") if hull != "base" else None
        record["stl_identity"] = stl_identity(case_dir, variant_stl)
        record["cd_pressure_S0"] = record["resistance_pressure_N"] / (pressure_scale * s0)
        record["ct_total_S0"] = record["resistance_total_N"] / (pressure_scale * s0)
        cfd_results[hull] = record
        for key, index in COLUMNS.items():
            window_series[f"{hull}_{key}"] = table[:, index].astype(np.float64)

    # 模型：三个船型的船面 Cp 与轴向压力积分
    hulls = {hull: build_hull(sample, meta, hull) for hull in HULLS}
    geometries = {hull: hull_geometry(hulls[hull], meta, hull) for hull in HULLS}
    model, checkpoint, config = load_checkpoint(device)

    model_results = {}
    for hull in HULLS:
        surface = hulls[hull]
        cp = normalizer.hull_mean.item() + normalizer.hull_std.item() * predict_cp(
            model, entry_batch(sample, surface, lpp), device)
        pressure = cp * pressure_scale
        resistance = axial_resistance(pressure, surface["normals"], surface["weights"])
        item = {
            "resistance_full_ship_N": resistance,
            "resistance_half_ship_N": resistance / 2.0,
            "cd_pressure_S0": resistance / (pressure_scale * s0),
            "cd_pressure_lpp2": resistance / (pressure_scale * lpp ** 2),
            "cp_weighted_mean": float((cp * surface["weights"]).sum()
                                      / surface["weights"].sum()),
            "cp_min": float(cp.min()),
            "cp_max": float(cp.max()),
            "axial_pressure_l1_N": 2.0 * float(
                (np.abs(pressure * surface["normals"][:, 0]) * surface["weights"]).sum()),
        }
        if hull == "base":
            truth = sample["hull_pressure"].numpy().astype(np.float64)
            item["cp_rmse_vs_cfd_label"] = float(math.sqrt(
                ((cp - truth) ** 2 * surface["weights"]).sum()
                / surface["weights"].sum()))
            item["cp_mae_vs_cfd_label"] = float(
                (np.abs(cp - truth) * surface["weights"]).sum()
                / surface["weights"].sum())
        item["cfd_pressure_resistance_N"] = cfd_results[hull]["resistance_pressure_N"]
        item["cfd_pressure_cd_S0"] = cfd_results[hull]["cd_pressure_S0"]
        item["abs_error_N"] = item["resistance_full_ship_N"] - item["cfd_pressure_resistance_N"]
        item["rel_error"] = (item["abs_error_N"]
                             / cfd_results[hull]["resistance_pressure_N"])
        model_results[hull] = item

    base_cfd = cfd_results["base"]["resistance_pressure_N"]
    base_model = model_results["base"]["resistance_full_ship_N"]
    variants = {}
    for hull in HULLS[1:]:
        cfd_delta = cfd_results[hull]["resistance_pressure_N"] - base_cfd
        model_delta = model_results[hull]["resistance_full_ship_N"] - base_model
        variants[hull] = {
            "cfd_delta_N": cfd_delta,
            "cfd_delta_percent": 100.0 * cfd_delta / base_cfd,
            "cfd_total_delta_N": (cfd_results[hull]["resistance_total_N"]
                                  - cfd_results["base"]["resistance_total_N"]),
            "cfd_total_delta_percent": 100.0 * (
                cfd_results[hull]["resistance_total_N"]
                - cfd_results["base"]["resistance_total_N"]
            ) / cfd_results["base"]["resistance_total_N"],
            "model_delta_N": model_delta,
            "model_delta_percent": 100.0 * model_delta / base_model,
            "delta_sign_agrees": bool(np.sign(cfd_delta) == np.sign(model_delta)),
            "delta_abs_error_N": model_delta - cfd_delta,
        }
    order = {"cfd": [hull for hull in sorted(
        HULLS, key=lambda name: cfd_results[name]["resistance_pressure_N"])],
        "model": [hull for hull in sorted(
            HULLS, key=lambda name: model_results[name]["resistance_full_ship_N"])]}

    # 最小积分核验：常量压力、基准条目 CFD 标签的同一积分路径
    label_cp = sample["hull_pressure"].numpy().astype(np.float64)
    label_pressure = label_cp * pressure_scale
    label_resistance = axial_resistance(label_pressure, hulls["base"]["normals"],
                                        hulls["base"]["weights"])

    document = {
        "test": "kcs_optim_application_test",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry": ENTRY,
        "condition": {"Fr": meta["Fr"], "Re": meta["Re"], "U_inf": u_meta, "rho": rho,
                      "Lpp": lpp, "draft": draft, "s0_reference_m2": s0,
                      "pressure_scale_0.5_rho_u2": pressure_scale},
        "model": {
            "run": MODEL_RUN,
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_epoch": checkpoint["epoch"],
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "architecture": config["model_type"],
            "input_domains": list(model.input_domains),
            "output_domains": list(model.output_domains),
            "lambda_hull": config["train"].get("lambda_hull"),
            "train_strides": config["train"].get("train_strides"),
            "train_entries": len(config["train"]["entries"]),
            "stats_file": str(STATS.relative_to(ROOT)),
            "stats_hull_mean": normalizer.hull_mean.tolist(),
            "stats_hull_std": normalizer.hull_std.tolist(),
            "entry_in_training": ENTRY in config["train"]["entries"],
            "entry_in_validation": ENTRY in config["train"]["val_entries"],
        },
        "cfd": {
            "dataset_root": str(args.dataset_root),
            "time_window_s": list(WINDOW),
            "window_basis": "数据集 AGENT.md 验收判据 + calib/README.md：R_T = 2|Fx| 取 20–25 s 均值"
                            "（baseline 参考值 91.24 N）",
            "half_domain": "算例 y ≤ 0，midPlane 为 symmetryPlane；force.dat 为半船力，乘 2 得全船",
            "drag_sign": "来流沿 -x（U = (-2.196 0 0)），阻力 Fx < 0，R = -2 Fx",
            "decomposition": "total_x / pressure_x / viscous_x 逐列取窗均值后再乘 2",
            "cases": cfd_results,
            "files": {"summary": str(FORCE_SUMMARY.relative_to(ROOT)),
                      "window_series": str(FORCE_SERIES.relative_to(ROOT)),
                      "window_series_keys": sorted(window_series)},
        },
        "geometry": {
            "hulls": geometries,
            "base_cfd_label_resistance_full_ship_N": label_resistance,
            "base_sdf_reproduction_max_abs_error": hulls["base"]["base_sdf_max_error"],
            "uv": "变体无面片连接关系，船面 UV 按顶点下标沿用基准条目",
        },
        "prediction": model_results,
        "variants": variants,
        "rank_by_pressure_resistance": order,
        "checks": {
            "constant_pressure_1pa_resistance_full_ship_N": {
                hull: geometries[hull]["constant_pressure_1pa_resistance_full_ship_N"]
                for hull in HULLS},
            "base_cfd_label_integration_minus_cfd_force_N":
                label_resistance - cfd_results["base"]["resistance_pressure_N"],
            "base_cfd_label_integration_rel_error":
                (label_resistance - cfd_results["base"]["resistance_pressure_N"])
                / cfd_results["base"]["resistance_pressure_N"],
            "base_cfd_label_integration_vs_force_at_25s": {
                "label_N": label_resistance,
                "cfd_force_at_last_time_N":
                    cfd_results["base"]["resistance_pressure_at_last_time_N"],
                "rel_error": (label_resistance
                              - cfd_results["base"]["resistance_pressure_at_last_time_N"])
                             / cfd_results["base"]["resistance_pressure_at_last_time_N"]},
            "base_resistance_window_vs_calib_reference": {
                "computed_N": cfd_results["base"]["resistance_total_N"],
                "reference_N": 91.24,
                "rel_error": (cfd_results["base"]["resistance_total_N"] - 91.24) / 91.24},
            "axial_projected_area_half_ship_m2": {
                hull: axis_factor(hulls[hull]["normals"], hulls[hull]["weights"])
                for hull in HULLS},
        },
        "formula": {
            "pressure_integral": "D = -2 Σ_i p_i n_i,x A_i，p_i = Cp_i · 0.5 ρ U²，"
                                 "n 为条目存储的指向船体内部的法向（外法向取 -n）",
            "cfd_pressure_resistance": "R_p = -2 · mean_t(Fx_pressure)，t ∈ [20, 25] s",
            "coefficient": "C_Dp = D / (0.5 ρ U² S0)，S0 = 9.5531 m²（KCS 参考湿表面积）",
            "delta": "Δ = D(变体) − D(base)，百分比相对 D(base)",
        },
    }

    FORCE_SUMMARY.write_text(json.dumps({
        "note": "KCS 艏部 FFD 变体下游验证用的 CFD 力摘要；独立于 ShipBench 条目",
        "generated_at": document["generated_at"],
        "dataset_root": str(args.dataset_root),
        "time_window_s": list(WINDOW),
        "condition": document["condition"],
        "cases": cfd_results,
        "cfd_label_integration": {
            "base_cfd_label_resistance_full_ship_N": label_resistance,
            "cfd_force_pressure_window_N": cfd_results["base"]["resistance_pressure_N"],
            "cfd_force_pressure_at_last_time_N":
                cfd_results["base"]["resistance_pressure_at_last_time_N"]},
        "window_series_file": FORCE_SERIES.name,
    }, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(FORCE_SERIES, **window_series)
    OUTPUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")

    print(f"写入 {OUTPUT}")
    print(f"{'hull':12s} {'CFD Rp (N)':>11s} {'model D (N)':>12s} {'err (N)':>8s} "
          f"{'err %':>7s} {'CFD C_Dp':>10s} {'model C_Dp':>11s}")
    for hull in HULLS:
        item = model_results[hull]
        cfd = cfd_results[hull]
        print(f"{hull:12s} {cfd['resistance_pressure_N']:11.3f} "
              f"{item['resistance_full_ship_N']:12.3f} {item['abs_error_N']:8.3f} "
              f"{100 * item['rel_error']:7.2f} {cfd['cd_pressure_S0']:10.6f} "
              f"{item['cd_pressure_S0']:11.6f}")
    print(f"排序（压差阻力升序） CFD={order['cfd']} model={order['model']}")
    for name, item in variants.items():
        print(f"{name:12s} CFD Δ={item['cfd_delta_N']:+7.3f} N "
              f"({item['cfd_delta_percent']:+6.2f}%)  model Δ={item['model_delta_N']:+7.3f} N "
              f"({item['model_delta_percent']:+6.2f}%)")


if __name__ == "__main__":
    main()
