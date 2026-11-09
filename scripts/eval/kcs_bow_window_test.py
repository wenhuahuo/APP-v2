"""KCS 艏部局部口径验证：艏部窗口的模型 Cp 轴向积分 vs CFD 变体压力阻力。

背景：`verify_kcs_optim_application.py` 证明全船压差阻力失效在「正负轴向分量相消后的
9.65% 净项」，点级 Cp 场本身量级正确。本脚本检验局部口径的可行性：把积分窗口截到
艏部 x ≥ x_cut（FFD 变形艏段在 0.90–1.03 Lpp），比较三船型的

- 模型窗口轴向力 D_bow（预测 Cp × 0.5ρU² 后在窗口内按同一路径积分），
- 基准条目 CFD 快照 Cp 标签的窗口积分（窗口占比 α 的来源），
- CFD 参照 = ΔR_p × α(x_cut)：假设变体 ΔR_p 主要发生在窗口内、窗口占比沿用基准标签。

复用 `probe_kcs_ffd_pressure.py` 的几何变形与积分口径、`verify_kcs_optim_application.py`
的 CFD force.dat 读数。检查点取主实验 `appsolver_oformer_random_seed0_isoflops/best.pt`
（epoch 79、1 509 638 参数，与 KCS 验证文档同源；文档里的归档路径已无权重，活跃目录
与文档核对过同一 epoch/参数量）。

用法:
    python3 scripts/eval/kcs_bow_window_test.py \
        --dataset-root /path/to/openfoam-cases

结果只在 stdout 打表，不落盘；复现直接重跑本脚本。
"""

import argparse
import json
import sys
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
# 艏部窗口扫描：以 Lpp 为单位的轴向截断位置；0.90 是 FFD 格架下缘
CUTS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00)


def load_checkpoint(device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def window_terms(pressure, hull, mask):
    """半船窗口轴向力的全船值（×2）、窗口 |p n_x| A 绝对和的全船值、净占比。"""
    n, a, p = hull["normals"][mask, 0], hull["weights"][mask], pressure[mask]
    net = -2.0 * float((p * n * a).sum())
    gross = 2.0 * float((np.abs(p * n * a)).sum())
    return net, gross, (net / gross if gross else 0.0)


def sign(value):
    return (value > 0.0) - (value < 0.0)


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

    hulls = {hull: build_hull(sample, meta, hull) for hull in HULLS}
    cp_model, cp_label = {}, {}
    for hull in HULLS:
        cp_model[hull] = normalizer.hull_mean.item() + normalizer.hull_std.item() * predict_cp(
            model, entry_batch(sample, hulls[hull], lpp), device)
    cp_label["base"] = sample["hull_pressure"].numpy().astype(np.float64)

    cfd_rp = {}
    for hull, spec in CASES.items():
        _, table = read_force_history(args.dataset_root / spec["case"], WINDOW)
        cfd_rp[hull] = -2.0 * float(table[:, COLUMNS["pressure"]].mean())
        print(f"cfd {hull:12s} R_p = {cfd_rp[hull]:7.3f} N （20–25 s 窗口均值）")

    full_mask = {hull: hulls[hull]["points"][:, 0] > -np.inf for hull in HULLS}
    print("\n全船口径复现（对照 KCS 验证文档 −1.390 / −5.991 / +3.315 N）：")
    for hull in HULLS:
        net, gross, share = window_terms(
            cp_model[hull] * pressure_scale, hulls[hull], full_mask[hull])
        print(f"  {hull:12s} D_model = {net:8.3f} N  |网|= {gross:8.1f} N  "
              f"净占比 {100 * share:6.2f}%")

    label_perf, _, _ = window_terms(
        cp_label["base"] * pressure_scale, hulls["base"], full_mask["base"])
    print(f"\n窗口扫描（x_cut × Lpp；基准标签全船 D = {label_perf:.3f} N；"
          f"CFD 参照 = ΔR_p × α(x_cut)）：")
    print(f"{'x_cut':>6s} {'α':>6s} | {'模型 base':>10s} {'out':>10s} {'in':>10s} | "
          f"{'CFD base':>10s} {'out':>10s} {'in':>10s} | {'Δout':>4s} {'Δin':>4s} "
          f"{'模型净占比%':>9s}")
    for cut in CUTS:
        mask = {hull: hulls[hull]["points"][:, 0] >= cut * lpp for hull in HULLS}
        model_d, label_bow, net_share = {}, None, None
        for hull in HULLS:
            model_d[hull], _, net_share = window_terms(
                cp_model[hull] * pressure_scale, hulls[hull], mask[hull])
        label_bow = window_terms(
            cp_label["base"] * pressure_scale, hulls["base"], mask["base"])[0]
        alpha = label_bow / label_perf
        check = {}
        for hull in HULLS[1:]:
            cfd_delta_ref = sign(cfd_rp[hull] - cfd_rp["base"])
            model_delta = sign(model_d[hull] - model_d["base"])
            check[hull] = "OK" if model_delta == cfd_delta_ref else "X"
        print(f"{cut:6.2f} {alpha:6.3f} | {model_d['base']:10.3f} "
              f"{model_d['bow_outward']:10.3f} {model_d['bow_inward']:10.3f} | "
              f"{cfd_rp['base'] * alpha:10.3f} "
              f"{(cfd_rp['bow_outward'] - cfd_rp['base']) * alpha:10.3f} "
              f"{(cfd_rp['bow_inward'] - cfd_rp['base']) * alpha:10.3f} | "
              f"{check['bow_outward']:>4s} {check['bow_inward']:>4s} "
              f"{100 * net_share:9.2f}")
    print("\n注：CFD 列是 ΔR_p×α 参照；α 取基准标签的窗口占比；"
          "最后一列为模型基准窗口的净/毛比，数值越大相消越少、该口径越可学。")


if __name__ == "__main__":
    main()
