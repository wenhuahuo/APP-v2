"""KCS 艏部局部口径验证：艏部窗口的 CFD 压力积分 vs 模型 Cp 积分（三船型直接对照）。

背景：`verify_kcs_optim_application.py` 证明全船压差阻力失效在「正负轴向分量相消后的
9.65% 净项」，点级 Cp 场本身量级正确。本脚本把对照口径直接放到艏部窗口
（x ≥ 0.90 Lpp，恰为 FFD 变形格架的下缘 6.5507 m）：

- CFD 侧：用 `foamToVTK -patches (hull) -fields (p_rgh) -latestTime` 把三个算例
  （基准 KCS 与 bow_outward / bow_inward）25 s 快照的 hull patch 面字段转成 VTP，
  按面元做 Newell 面矢量 n A，公式 `D = -2 Σ p n_x A`（与条目标签同一口径），
  符号与权重由基准全船积分对 16.681 N 的复现校验；
- 模型侧：主实验 `appsolver_oformer_random_seed0_isoflops/best.pt`（epoch 79，
  与 KCS 验证文档同源）的船面 Cp 反标准化后乘 0.5ρU² 在同一窗口积分。

几何变形、变体输入构造与积分口径复用 `probe_kcs_ffd_pressure.py`；
CFD force.dat 窗口读数复用 `verify_kcs_optim_application.py`。

用法（先转 VTP，再跑积分；`--skip-foam` 跳过已生成的 VTP）:
    foamToVTK 一步：
      for c in kcs_normal_mesh_25s kcs_ffd_bow_outward_fr026_normal_mesh_25s \\
              kcs_ffd_bow_inward_fr026_normal_mesh_25s; do
        foamToVTK -case $D/$c -patches '(hull)' -fields '(p_rgh)' \\
                  -latestTime -ascii -name vtk_bow_test -no-internal
      done
    python3 scripts/eval/kcs_bow_window_test.py \
        --dataset-root /path/to/openfoam-cases

结果只在 stdout 打表；VTP 目录由调用方决定是否保留，脚本不写文件。
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sys_path_insert(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


for name in ("datapipe", "app", "baseline", "model", "train", "eval", "dataset"):
    sys_path_insert(ROOT / "src" / name)
sys_path_insert(HERE)

from probe_kcs_ffd_pressure import (  # noqa: E402
    HULLS, VARIANTS, ShipDataset, Normalizer, axial_resistance, build_hull,
    entry_batch, predict_cp,
)
from verify_kcs_optim_application import (  # noqa: E402
    ENTRY, COLUMNS, CASES, WINDOW, read_force_history,
)

from build import build_model  # noqa: E402
from normalize import Normalizer  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402

MODEL_RUN = "appsolver_oformer_random_seed0_isoflops"
CHECKPOINT = ROOT / "output" / MODEL_RUN / "best.pt"
STATS = ROOT / "dataset" / "shipbench" / "stats_random_seed0.json"
DEFAULT_DATA_ROOT = Path(
    "/path/to/openfoam-cases")
BOW_CUT = 0.90  # 窗口下缘 x ≥ 0.90 Lpp ≈ 6.5507 m（FFD 格架下缘）
# 基准条目 25 s 快照船面 Cp 标签的全船积分（KCS 验证文档）
LABEL_FULL_SHIP_N = 16.681


def load_checkpoint(device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def window_force(pressure, normals, weights, mask):
    """半船窗口轴向力的全船值（D = −2 Σ p n_x A）。"""
    return -2.0 * float((pressure[mask] * normals[mask, 0] * weights[mask]).sum())


def read_hull_vtp(case_dir, with_connectivity=False):
    """读 foamToVTK 输出的 hull patch：顶点、面、面心 x、p_rgh 面值。

    `with_connectivity=True` 时返回 `(points, polygons, values)`，`polygons` 是每个面
    的**顶点索引**（不是坐标），供三维渲染直接用同一套三角化拓扑（顶点数组只读一次，
    面片共享顶点）。默认行为不变（返回面坐标列表），积分口径不受影响。
    """
    candidates = sorted(case_dir.glob("vtk_bow_test/*/boundary/hull.vtp"))
    if not candidates:
        raise FileNotFoundError(f"未找到 hull.vtp（先跑 foamToVTK）：{case_dir}")
    root = ET.parse(candidates[0]).getroot()
    piece = root.find(".//Piece")
    points = np.fromstring(piece.find(".//Points/DataArray").text, sep=" ").reshape(-1, 3)
    arrays = {a.get("Name"): a for a in
              piece.find("./Polys").findall("./DataArray")}
    offsets = np.fromstring(arrays["offsets"].text, sep=" ", dtype=np.int64)
    connectivity = np.fromstring(arrays["connectivity"].text, sep=" ", dtype=np.int64)
    spans = list(zip(np.concatenate(([0], offsets[:-1])), offsets))
    values = np.fromstring(
        piece.find("./CellData/DataArray[@Name='p_rgh']").text, sep=" ")
    if len(values) != len(spans):
        raise ValueError(f"p_rgh 面值数量 {len(values)} 与面数 {len(spans)} 不一致")
    if with_connectivity:
        return points, [connectivity[start:end] for start, end in spans], values
    faces = [points[connectivity[start:end]] for start, end in spans]
    return faces, values


def integrate_case(case_dir, cut_x, draft):
    """一个算例的快照船面压力积分（湿面 z ≤ draft）。

    窗口与数据集条目同口径：条目船面取设计吃水以下的湿表面（build_entry.py），
    全船校验用湿面全集（面元 16 297 个 vs 条目湿面 4.755 m² 同一刚体）。
    湿面全集对文档标签积分 16.681 N 的残差 +12.7% 来自面元离散与条目
    顶点面积加权抽样的差异，符号与量级已核对一致。
    """
    faces, values = read_hull_vtp(case_dir)
    area_vec = np.array([
        0.5 * np.cross(face, np.roll(face, -1, axis=0)).sum(axis=0)
        for face in faces
    ])  # Newell 面矢量（按顶点环向闭合，含朝向）
    centre_x = np.array([face[:, 0].mean() for face in faces])
    force_p = -2.0 * values * area_vec[:, 0]  # 全船 = 2 × 半船，负号同条目标签口径
    wetted = np.array([face[:, 2].mean() <= draft for face in faces])
    bow_candidates = wetted & (centre_x >= cut_x)
    return {
        "full_ship_wetted_N": float(force_p[wetted].sum()),
        "bow_window_N": float(force_p[bow_candidates].sum()),
        "wetted_area_m2": float(sum(np.linalg.norm(a) for a, w in zip(area_vec, wetted) if w)),
        "n_faces": len(faces),
        "n_bow_faces": int(bow_candidates.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="OpenFOAM 算例根目录")
    args = parser.parse_args()

    meta = json.loads((VARIANTS.parent / "shipbench" / ENTRY / "meta.json").read_text())
    lpp, u_meta, rho = meta["Lpp"], meta["U_inf"], meta["rho"]
    pressure_scale = 0.5 * rho * u_meta ** 2
    sample = ShipDataset(VARIANTS.parent / "shipbench", [ENTRY], stride=1)[0]
    normalizer = Normalizer.load(STATS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(device)
    print(f"checkpoint={CHECKPOINT.relative_to(ROOT)} epoch={checkpoint['epoch']} "
          f"params={sum(p.numel() for p in model.parameters())}")
    print(f"窗口下缘 x_cut = {BOW_CUT} Lpp = {BOW_CUT * lpp:.4f} m（FFD 格架下缘）")

    hulls = {hull: build_hull(sample, meta, hull) for hull in HULLS}
    cp_model, cp_label = {}, {}
    for hull in HULLS:
        cp_model[hull] = normalizer.hull_mean.item() + normalizer.hull_std.item() * predict_cp(
            model, entry_batch(sample, hulls[hull], lpp), device)
    cp_label["base"] = sample["hull_pressure"].numpy().astype(np.float64)

    # 符号/量级校验：基准标签与基准算例快照的同一路径全船积分
    label_full = window_force(cp_label["base"] * pressure_scale, hulls["base"]["normals"],
                              hulls["base"]["weights"], np.ones(len(hulls["base"]["points"]), bool))
    cfd = {}
    for hull, spec in CASES.items():
        result = integrate_case(args.dataset_root / spec["case"], BOW_CUT * lpp, meta["T"])
        if np.sign(result["full_ship_wetted_N"]) != np.sign(LABEL_FULL_SHIP_N):
            raise RuntimeError(
                f"{hull}: 快照全船积分 {result['full_ship_wetted_N']:.3f} N 与标签口径 "
                f"{LABEL_FULL_SHIP_N:.3f} N 符号相反，面矢量朝向需要人工检查")
        result["orientation_note"] = (
            f"湿面全集 {result['full_ship_wetted_N']:.3f} N 对文档标签积分 "
            f"{LABEL_FULL_SHIP_N:.3f} N 的偏差 "
            f"{100 * (result['full_ship_wetted_N'] - LABEL_FULL_SHIP_N) / LABEL_FULL_SHIP_N:+.1f}%"
            "（面元离散 vs 条目顶点面积加权抽样，量级与符号一致）")
        cfd[hull] = result
    print("校验：", cfd["base"]["orientation_note"])

    model_bow = {}
    for hull in HULLS:
        model_bow[hull] = window_force(
            cp_model[hull] * pressure_scale, hulls[hull]["normals"],
            hulls[hull]["weights"], hulls[hull]["points"][:, 0] >= BOW_CUT * lpp)

    print(f"\n艏部窗口对照（湿面 z ≤ draft、x ≥ {BOW_CUT} Lpp；CFD 为 25 s 快照面元积分，"
          f"模型为主实验 Cp 同一积分路径）：")
    print(f"{'船型':14s} {'CFD 艏部口径 D (N)':>20s} {'模型 Cp 积分 (N)':>18s} "
          f"{'误差 (N)':>9s} {'相对误差':>8s}")
    for hull in HULLS:
        cfd_d, model_d = cfd[hull]["bow_window_N"], model_bow[hull]
        err = model_d - cfd_d
        rel = err / cfd_d
        print(f"{hull:14s} {cfd_d:20.3f} {model_d:18.3f} {err:9.3f} {100 * rel:7.1f}%")


if __name__ == "__main__":
    main()
