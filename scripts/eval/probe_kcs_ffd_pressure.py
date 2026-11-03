"""KCS 艏部 FFD 变体的船面压力预测与压差阻力积分探测。

用历史联合（流场 + 船面压力）监督的 3 个模型与唯一的纯船面模型，在基准
`kcs_fr026_normal_mesh` 和两个 FFD 变体上预测船面 Cp，沿半船湿表面做轴向压力积分，
给出牛顿值与无量纲压差阻力系数。结果写 `output/kcs_ffd_pressure_probe.json`。

用法:
    python3 scripts/eval/probe_kcs_ffd_pressure.py

几何与积分口径
--------------

- 变体几何不重建网格：`dataset/kcs_var/<变体>_ffd.npy` 是作用在母船 STL 上的控制点
  位移，按 `<变体>.json` 记录的同一格架作用到条目船面顶点上（顶点一一对应）。
  船面 UV 按顶点下标沿用基准条目：条目只有湿表面点、没有面片连接关系，`v` 的测地
  弧长无法按变体几何重算；变体只改 y、幅度 ≤ 0.53% Lpp。求解与积分只用坐标/法向/权重，
  UV 只作为模型输入。
- 法向与求积权重随变形更新：`n' ∝ J^{-T} n`、`A' = A · det(J) · ||J^{-T} n||`，
  `J` 是 FFD 映射的雅可比（中心差分；左舷取镜像 `M J M`）。
- 自由液面输入的 SDF 按各自水线投影轮廓重算（`footprint_profile` + `signed_distance`），
  掩除格点取「基准有效探测格点 ∩ 变体 SDF > 0」。
- 压力为动压 `p_rgh`，条目里存的是 `Cp = p_rgh / (0.5 ρ U²)`；预测值用该模型训练时的
  船面 normalizer 反标准化后乘 `0.5 ρ U²` 还原为 Pa。
- 船体为半船（starboard, y ≤ 0），条目存储法向指向船体内部；积分时外法向取 `-n`。
  压强合力对船体为 `-p n_out`，阻力取 +x 方向（船艏在 +x，流场 `u < 0`），于是

      D_half = Σ p (n_out · e_x) A = -Σ p n_x A,   D = 2 D_half

  乘 2 得全船。系数 `C_Dp = D / (0.5 ρ U² S0)`，`S0` 取条目参考湿表面积。
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for name in ("datapipe", "app", "baseline", "model", "train", "eval", "dataset"):
    sys.path.insert(0, str(ROOT / "src" / name))

from build import build_model  # noqa: E402
from ffd import apply_ffd, fold_y  # noqa: E402
from normalize import Normalizer  # noqa: E402
from sdf import footprint_profile, signed_distance  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402
from stl_io import read_stl  # noqa: E402

ENTRY = "kcs_fr026_normal_mesh"
DATASET = ROOT / "dataset" / "shipbench"
VARIANTS = ROOT / "dataset" / "kcs_var"
ARCHIVE = ROOT / "output" / "archive" / "pre_full_dataset"
OLD_STATS = ROOT / "archive" / "pre_full_dataset" / "stats_before_refit"
OUTPUT = ROOT / "output" / "kcs_ffd_pressure_probe.json"

# 统一历史验证集（pre-full-dataset random_seed0 的 31 条留出集）上，lambda_hull=1.0 且
# checkpoint 含船面输出头的联合模型；按 stride1 池化船面 Cp RMSE 取前 3 个可忠实加载的。
JOINT_CANDIDATES = ("appsolver_v1_random_seed0", "oformer_random_seed0",
                    "appsolver_v2_joint_flow_hull_random_seed0", "transolver_random_seed0",
                    "upt_random_seed0")
# 同名 run 的 checkpoint 已被主实验口径（双域输入、仅流场输出、lambda_hull=0）覆盖，
# 归档 metrics 与 best.pt 不再对应，不能用于本轮。
STALE_RUNS = {
    "gnot_random_seed0": "metrics 是 31 条留出集的双域结果，best.pt 已覆盖为 lambda_hull=0、"
                         "仅流场输出（embedded config val_entries=34），不能产生船面输出",
    "appsolver_oformer_random_seed0": "metrics 为 34 条留出集且无船面指标，best.pt 已覆盖为"
                                      "仅流场输出",
}
HULL_ONLY = "appsolver_v2_hull_only_random_seed0"
HULLS = ("base", "bow_outward", "bow_inward")


def load_config(checkpoint):
    """历史 checkpoint 的自带配置；旧键 `domains` 拆成输入/输出域，`flow_sdf` 已废弃。"""
    config = dict(checkpoint["config"])
    model = dict(config["model"])
    if "domains" in model:
        domains = tuple(model.pop("domains"))
        model.setdefault("input_domains", domains)
        model.setdefault("output_domains", domains)
    model.pop("flow_sdf", None)
    config["model"] = model
    return config


def load_model(name, device):
    checkpoint = torch.load(ARCHIVE / name / "best.pt", map_location="cpu", weights_only=False)
    config = load_config(checkpoint)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint, config


def pooled_hull_rmse(name):
    """历史 metrics_best.json 里 val / stride1 的船面 Cp 池化 RMSE（docs 的池化公式）。"""
    metrics = json.loads((ARCHIVE / name / "metrics_best.json").read_text())
    entries = metrics["strides"]["val"]["1"]
    hull = {entry: value["hull"] for entry, value in entries.items() if "hull" in value}
    square = sum(value["rmse"][0] ** 2 * value["n_points"] for value in hull.values())
    points = sum(value["n_points"] for value in hull.values())
    return {"pooled_rmse": math.sqrt(square / points), "n_entries": len(hull),
            "checkpoint_epoch": metrics["checkpoint_epoch"],
            "metric_config": metrics["config"], "entry_ids": sorted(hull)}


def fold_map(points, bounds, displacements):
    """FFD 在折叠半船（y → |y|）上的映射。"""
    return apply_ffd(fold_y(points), bounds, displacements)


def deform(points, bounds, displacements):
    """把 FFD 位移作用到船体顶点上（只改 y，中纵剖面不动）。"""
    moved = fold_map(points, bounds, displacements)
    shifted = points.copy()
    shifted[:, 1] = np.sign(points[:, 1]) * moved[:, 1]
    return shifted


def jacobian(points, bounds, displacements, step=1e-4):
    """FFD 映射在船体顶点处的雅可比；左舷由镜像 `M J M` 得到（M = diag(1, -1, 1)）。"""
    folded = fold_y(points)
    result = np.empty((len(points), 3, 3))
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        plus = apply_ffd(folded + offset, bounds, displacements)
        minus = apply_ffd(folded - offset, bounds, displacements)
        result[:, :, axis] = (plus - minus) / (2 * step)
    mirror = np.array([1.0, -1.0, 1.0])
    left = points[:, 1] < 0.0
    result[left] = mirror[:, None] * result[left] * mirror[None, :]
    return result


def transport(normals, jac):
    """余法向变换与面积拉伸因子；返回 (n', 面积因子, det J)。"""
    det = np.linalg.det(jac)
    transformed = np.linalg.solve(np.transpose(jac, (0, 2, 1)), normals[..., None])[..., 0]
    length = np.linalg.norm(transformed, axis=1)
    return transformed / length[:, None], det * length, det


def build_hull(sample, meta, variant):
    """基准/变体的船面几何与流场 SDF。"""
    lpp, draft = meta["Lpp"], meta["T"]
    points = sample["hull_points"].numpy().astype(np.float64) * lpp
    normals = sample["hull_normals"].numpy().astype(np.float64)
    weights = sample["hull_weight"].numpy().astype(np.float64) * lpp ** 2
    det = np.ones(len(points))
    if variant != "base":
        info = json.loads((VARIANTS / f"{variant}.json").read_text())
        bounds = [tuple(edge) for edge in info["ffd"]["bounds"]]
        displacements = np.load(VARIANTS / f"{variant}_ffd.npy").astype(np.float64)
        jac = jacobian(points, bounds, displacements)
        points = deform(points, bounds, displacements)
        normals, stretch, det = transport(normals, jac)
        weights = weights * stretch

    grid = sample["flow_coords"].numpy().astype(np.float64).reshape(-1, 2) * lpp
    base_points = sample["hull_points"].numpy().astype(np.float64) * lpp
    stored = sample["flow_mask"].numpy()
    sdf = (signed_distance(grid, footprint_profile(points, draft, lpp), lpp) / lpp
           ).reshape(stored.shape)
    base_sdf = (signed_distance(grid, footprint_profile(base_points, draft, lpp), lpp) / lpp
                ).reshape(stored.shape)
    # 基准条目里 sdf > 0 但被剔除的格点是探针失败的单元，变体沿用同一剔除集合
    dropped = (base_sdf > 0.0) & ~stored
    mask = (sdf > 0.0) & ~dropped
    weight = sample["flow_weight"].numpy().copy()
    weight[mask] = weight[stored].max()  # 新解除掩除的格点按同一均匀 h² 求积权重
    base_error = float(np.abs(base_sdf[stored] - sample["flow_sdf"].numpy()[stored]).max())
    return {"points": points, "normals": normals, "weights": weights,
            "sdf": np.where(mask, sdf, 0.0), "sdf_raw": sdf, "mask": mask,
            "flow_weight": weight, "det": det, "base_sdf_max_error": base_error,
            "added_cells": int((mask & ~stored).sum()),
            "removed_cells": int((stored & ~mask).sum()),
            "dy": points[:, 1] - base_points[:, 1]}


def entry_batch(sample, hull, lpp):
    """把基准条目样本替换成变体几何，得到批大小 1 的模型输入。"""
    batch = {key: value[None].clone() for key, value in sample.items()}
    batch["hull_points"] = torch.from_numpy((hull["points"] / lpp).astype(np.float32))[None]
    batch["hull_normals"] = torch.from_numpy(hull["normals"].astype(np.float32))[None]
    batch["hull_weight"] = torch.from_numpy((hull["weights"] / lpp ** 2).astype(np.float32))[None]
    batch["flow_sdf"] = torch.from_numpy(hull["sdf"].astype(np.float32))[None]
    batch["flow_weight"] = torch.from_numpy(hull["flow_weight"].astype(np.float32))[None]
    batch["flow_mask"] = torch.from_numpy(hull["mask"])[None]
    batch["hull_mask"] = torch.ones((1, len(hull["points"])), dtype=torch.bool)
    return batch


@torch.no_grad()
def predict_cp(model, batch, device):
    """船面 Cp（无量纲）；流场查询点只给一个，积分不需要流场输出。"""
    queries = batch["flow_coords"].reshape(1, -1, 2)[:, :1].to(device)
    output = model({key: value.to(device) for key, value in batch.items()}, queries)
    return output["hull"][0][:, 0].cpu().numpy().astype(np.float64)


def axial_resistance(pressure, normals, weights):
    """半船轴向压力合力 → 全船压差阻力（+x 为船前进方向，见模块说明）。"""
    return -2.0 * float((pressure * normals[:, 0] * weights).sum())


def hull_geometry(hull, meta, variant):
    """几何统计与积分口径的验证量。"""
    info = (json.loads((VARIANTS / f"{variant}.json").read_text())
            if variant != "base" else None)
    reference = (info["geometry"]["wetted_area_m2"] if info else
                 json.loads((VARIANTS / "index.json").read_text())["parent"]["wetted_area_m2"])
    lpp = meta["Lpp"]
    area = float(hull["weights"].sum())
    return {
        "n_points": int(len(hull["points"])),
        "half_wetted_area_m2": area,
        "full_wetted_area_m2": 2.0 * area,
        "stl_wetted_area_m2": float(reference),
        "wetted_area_ratio_vs_stl": 2.0 * area / reference,
        "jacobian_det_min": float(hull["det"].min()),
        "jacobian_det_max": float(hull["det"].max()),
        "flow_mask_cells": int(hull["mask"].sum()),
        "flow_mask_removed_vs_base": hull["removed_cells"],
        "flow_mask_added_vs_base": hull["added_cells"],
        "flow_sdf_min": float(hull["sdf_raw"].min()),
        "flow_sdf_max": float(hull["sdf_raw"].max()),
        "max_abs_dy_m": float(np.abs(hull["dy"]).max()),
        "max_abs_dy_over_lpp": float(np.abs(hull["dy"]).max() / lpp),
        "stl_max_abs_dy_m": info["geometry"]["max_abs_dy_m"] if info else 0.0,
        "constant_pressure_1pa_resistance_full_ship_N": axial_resistance(
            np.ones(len(hull["points"])), hull["normals"], hull["weights"]),
    }


def stl_checks():
    """变体 STL 与「按同一 FFD 作用到母船 STL」逐顶点对照。"""
    base_points, base_faces = read_stl(ROOT / "dataset" / "classic_hulls" / "kcs.stl")
    checks = {}
    for variant in HULLS[1:]:
        info = json.loads((VARIANTS / f"{variant}.json").read_text())
        bounds = [tuple(edge) for edge in info["ffd"]["bounds"]]
        displacements = np.load(VARIANTS / f"{variant}_ffd.npy").astype(np.float64)
        points, faces = read_stl(VARIANTS / f"{variant}.stl")
        checks[variant] = {
            "same_vertex_count_and_faces": bool(
                points.shape == base_points.shape and np.array_equal(faces, base_faces)),
            "max_abs_vertex_error_m": float(np.abs(
                deform(base_points, bounds, displacements) - points).max()),
            "max_abs_dy_m": float(np.abs(points[:, 1] - base_points[:, 1]).max()),
            "waterline_max_abs_dy_m": info["geometry"]["waterline_max_abs_dy_m"],
        }
    return checks


def model_selection(device):
    """按统一历史验证集选模型，并记录每个 checkpoint 的忠实加载结果。"""
    selection = {"validation": {
        "split": "pre-full-dataset random_seed0（历史留出集）", "stride": 1,
        "metric": "船面 Cp 池化 RMSE = sqrt(Σ n_e rmse_e² / Σ n_e)"},
        "candidates": [], "selected": [], "hull_only": None}
    reference_ids = None
    same_val_set = True
    for name in JOINT_CANDIDATES:
        record = {"model": name, **pooled_hull_rmse(name)}
        ids = record.pop("entry_ids")
        reference_ids = reference_ids or ids
        record["same_val_set"] = ids == reference_ids
        same_val_set &= record["same_val_set"]
        state = torch.load(ARCHIVE / name / "best.pt", map_location="cpu", weights_only=False)
        record["hull_output_params"] = sorted(
            {key.split(".")[0] for key in state["model"] if key.startswith("hull_head")}
            or {key.split(".")[0] for key in state["model"] if key.startswith("hull_patches")})
        record["lambda_hull"] = state["config"]["train"].get("lambda_hull")
        record["stats"] = state["config"].get("stats")
        try:
            model, checkpoint, _ = load_model(name, device)
            record.update({"load_strict": True, "checkpoint_epoch": checkpoint["epoch"],
                           "output_domains": list(model.output_domains), "error": None})
        except Exception as error:  # 架构已改写的 checkpoint 记为不可忠实加载
            legacy = any("blocks.layers." in key for key in state["model"])
            record.update({"load_strict": False, "output_domains": None,
                           "error": f"{type(error).__name__}: {error}"[:600],
                           "legacy_nesting": legacy,
                           "diagnosis": (
                               "checkpoint 的 PatchTransformer 用 nn.TransformerEncoder 包裹"
                               "（blocks.layers.N），当前代码在 f0fffcf 改为 ModuleList（blocks.N）；"
                               "参数与计算一致，但键名嵌套不同，严格加载失败" if legacy else
                               "模块结构已改写，严格加载失败（缺/多出的键见 error）")})
        selection["candidates"].append(record)
    selection["candidates"].sort(key=lambda item: item["pooled_rmse"])
    for rank, item in enumerate(selection["candidates"], start=1):
        item["rank"] = rank
    selection["validation"].update({"n_entries": len(reference_ids),
                                    "entry_ids": reference_ids,
                                    "same_val_set": bool(same_val_set)})
    selection["selected"] = [item["model"] for item in selection["candidates"]
                             if item["load_strict"]][:3]
    for item in selection["candidates"]:
        item["selected"] = item["model"] in selection["selected"]
    unloadable = [item["model"] for item in selection["candidates"] if not item["load_strict"]]
    selection["note"] = (
        "按池化船面 Cp RMSE 排名取前 3 个可由当前代码严格加载的模型；"
        f"不能忠实加载的排名模型：{unloadable}（原因见各自 error / legacy_nesting）")
    hull_only = pooled_hull_rmse(HULL_ONLY)
    selection["validation"]["same_val_set"] &= hull_only.pop("entry_ids") == reference_ids
    selection["hull_only"] = {"model": HULL_ONLY, "protocol": "仅船面输入与船面 Cp 输出",
                              "load_strict": True, **hull_only}
    selection["excluded_runs"] = [
        {"model": name, "reason": reason} for name, reason in STALE_RUNS.items()]
    selection["excluded_families"] = [
        {"family": "*_kcs_fr_sweep（KCS Fr 扫掠）",
         "reason": "验证集是单条 kcs_fr042（stride1），与本轮统一留出集不同；"
                   "且这些 checkpoint 的 flow_input 是 3 通道（含 SDF），当前代码为 2 通道"},
        {"family": "multihull / lofo / loto 历史 run 与 appsolver_* 迁移主干",
         "reason": "验证集不同，或架构在迁移后已改写，checkpoint 不能由当前代码严格加载"},
    ]
    return selection


def main():
    meta = json.loads((DATASET / ENTRY / "meta.json").read_text())
    lpp, draft = meta["Lpp"], meta["T"]
    rho, u_meta, s0 = meta["rho"], meta["U_inf"], meta["S0"]
    pressure_scale = 0.5 * rho * u_meta ** 2
    u_froude = meta["Fr"] * math.sqrt(9.81 * lpp)

    sample = ShipDataset(DATASET, [ENTRY], stride=1)[0]
    hulls = {variant: build_hull(sample, meta, variant) for variant in HULLS}
    normalizer = Normalizer.load(OLD_STATS / "stats_random_seed0.json")
    current = json.loads((DATASET / "stats_random_seed0.json").read_text())["hull"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    geometries = {variant: hull_geometry(hulls[variant], meta, variant) for variant in HULLS}
    selection = model_selection(device)
    if len(selection["selected"]) != 3:
        raise RuntimeError(f"可忠实加载的联合模型不足 3 个：{selection['selected']}")

    results = {}
    for name in selection["selected"] + [HULL_ONLY]:
        model, checkpoint, config = load_model(name, device)
        model_results = {}
        for variant in HULLS:
            hull = hulls[variant]
            cp = normalizer.hull_mean.item() + normalizer.hull_std.item() * predict_cp(
                model, entry_batch(sample, hull, lpp), device)
            pressure = cp * pressure_scale
            resistance = axial_resistance(pressure, hull["normals"], hull["weights"])
            item = {
                "resistance_full_ship_N": resistance,
                "resistance_half_ship_N": resistance / 2.0,
                "cd_pressure_S0": resistance / (pressure_scale * s0),
                "cd_pressure_lpp2": resistance / (pressure_scale * lpp ** 2),
                "cp_weighted_mean": float((cp * hull["weights"]).sum() / hull["weights"].sum()),
                "cp_min": float(cp.min()), "cp_max": float(cp.max()),
                # 轴向压力积分的 L1 尺度：净阻力是 ± 大项的抵消残差
                "axial_pressure_l1_N": 2.0 * float(
                    (np.abs(pressure * hull["normals"][:, 0]) * hull["weights"]).sum()),
            }
            if variant == "base":
                truth = sample["hull_pressure"].numpy().astype(np.float64)
                item["base_cp_rmse_vs_cfd_label"] = float(math.sqrt(
                    ((cp - truth) ** 2 * hull["weights"]).sum() / hull["weights"].sum()))
            model_results[variant] = item
        base = model_results["base"]
        for variant in HULLS[1:]:
            item = model_results[variant]
            item["delta_resistance_vs_base_N"] = (
                item["resistance_full_ship_N"] - base["resistance_full_ship_N"])
            item["delta_cd_pressure_vs_base"] = (
                item["cd_pressure_S0"] - base["cd_pressure_S0"])
        results[name] = {"architecture": config["model_type"],
                         "checkpoint_epoch": checkpoint["epoch"],
                         "lambda_hull": config["train"].get("lambda_hull"),
                         "hulls": model_results}

    truth = sample["hull_pressure"].numpy().astype(np.float64) * pressure_scale
    base = hulls["base"]
    reference_resistance = axial_resistance(truth, base["normals"], base["weights"])
    reference_l1 = 2.0 * float((np.abs(truth * base["normals"][:, 0])
                                * base["weights"]).sum())
    document = {
        "probe": "kcs_ffd_pressure",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry": ENTRY,
        "input_conditions": {"Fr": meta["Fr"], "Re": meta["Re"], "Lpp": lpp, "rho": rho,
                             "U_inf": u_meta, "draft": draft,
                             "pressure_scale_0.5_rho_u2": pressure_scale,
                             "s0_reference_m2": s0},
        "conventions": {
            "x_axis": "+x 指向艏部（条目船面 x/Lpp ∈ [-0.011, 1.030]，流场 u < 0）",
            "drag_sign": "阻力沿 +x 方向为正（与来流反向）：D = -2 Σ p n_x A，"
                         "n 为条目存储的指向船体内部的法向",
            "pressure": "p_rgh（动压），Cp = p_rgh / (0.5 ρ U²)",
            "half_hull": "条目为半船（y ≤ 0），轴向积分乘 2 得全船",
            "quadrature": "船面按顶点面积权重（变形后按 FFD 雅可比更新）",
            "coefficient": "C_Dp = D / (0.5 ρ U² S0)",
        },
        "model_selection": selection,
        "normalization": {
            "stats_file": str(OLD_STATS / "stats_random_seed0.json"),
            "note": "历史模型训练时的 random_seed0 统计量（refit 前归档）；"
                    "本轮全量重拟合的 dataset/shipbench/stats_random_seed0.json 不用于这些 checkpoint",
            "hull_mean": normalizer.hull_mean.tolist(),
            "hull_std": normalizer.hull_std.tolist(),
            "current_refit_hull_mean": current["mean"],
            "current_refit_hull_std": current["std"],
        },
        "geometry": {
            "hulls": geometries,
            "stl_roundtrip": stl_checks(),
            "base_sdf_reproduction_max_abs_error": hulls["base"]["base_sdf_max_error"],
            "uv": {
                "recomputed": False,
                "note": "条目无面片连接关系，按顶点下标沿用基准 UV；变体只改 y（≤ 0.53% Lpp）",
            },
        },
        "validation": {
            "constant_pressure_1pa_resistance_full_ship_N": {
                variant: geometries[variant][
                    "constant_pressure_1pa_resistance_full_ship_N"] for variant in HULLS},
            "base_cfd_label_resistance_full_ship_N": reference_resistance,
            "base_cfd_label_axial_pressure_l1_N": reference_l1,
            "base_cfd_label_cd_pressure_S0": reference_resistance / (pressure_scale * s0),
            "u_inf_from_froude_m_s": u_froude,
            "u_inf_relative_error": abs(u_froude - u_meta) / u_meta,
        },
        "results": results,
        "notes": [
            "预测压力来自历史模型，不是 CFD 真值；base_cfd_label_* 只是基准条目 CFD 标签的"
            "锤点，用于核对量级与法向/半船倍数约定。",
            "净阻力是 ± 百牛量级轴向压力项的抵消残差（见 axial_pressure_l1_N），"
            "船型间的差值逐模型给出（delta_resistance_vs_base_N），模型间方向不一致。",
            "变体检查点不重建 CFD 网格：船体点、法向、求积权重按 FFD 映射及其雅可比更新，"
            "船面 UV 按顶点下标沿用基准条目。",
        ],
    }
    OUTPUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"写入 {OUTPUT}")
    print(f"{'model':48s} {'hull':12s} {'D (N)':>9s} {'C_Dp (S0)':>11s} {'Cp mean':>9s}")
    for name, record in results.items():
        for variant, item in record["hulls"].items():
            print(f"{name:48s} {variant:12s} {item['resistance_full_ship_N']:9.3f} "
                  f"{item['cd_pressure_S0']:11.6f} {item['cp_weighted_mean']:9.5f}")


if __name__ == "__main__":
    main()
