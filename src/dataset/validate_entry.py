"""校验数据集条目的文件清单与数组一致性，并打印诊断量。

用法:
    python3 src/dataset/validate_entry.py [--dataset dataset/shipbench] [entry-id ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np

# 液面速度取液面下最近水相单元，其大小应与来流同量级
SPEED_RATIO_MAX = 2.0
# 船体点云在吃水以下的面积权重之和应与参考湿表面积 S0/2 对账
WETTED_AREA_TOLERANCE = 0.05


def validate(dataset_root, entry_id):
    """返回 (诊断字典, 问题列表)。"""
    entry = Path(dataset_root) / entry_id
    meta = json.loads((entry / "meta.json").read_text())
    arrays = {path.stem: np.load(path) for path in entry.glob("*.npy")}
    problems = []

    listed = sorted([f"{name}.npy" for name in arrays] + ["meta.json"])
    if listed != meta["files"]:
        problems.append(f"文件清单与 meta 不一致: {listed} != {meta['files']}")

    for name, array in arrays.items():
        if not np.isfinite(array).all():
            problems.append(f"{name} 含非有限值")
    for name in ("flow_points", "flow_sdf", "flow_target", "flow_quad_weights",
                 "hull_points", "hull_uv", "hull_normals", "hull_quad_weights", "hull_pressure"):
        if name not in arrays:
            problems.append(f"缺少 {name}.npy")

    flow = arrays["flow_target"]
    if flow.shape[1] != len(meta["flow_field"]["channels"]):
        problems.append(f"flow_target 通道数 {flow.shape[1]} 与 meta 不符")
    eta = flow[:, 0]
    speed = np.linalg.norm(flow[:, 1:4], axis=1)
    draft = meta["T"]
    lpp = meta["Lpp"]

    if np.abs(eta).max() > draft:
        problems.append(f"波高越界: max|eta| = {np.abs(eta).max():.4f} > T = {draft}")
    if speed.max() > SPEED_RATIO_MAX * meta["U_inf"]:
        problems.append(f"液面速度越界: max|U| = {speed.max():.3f}")

    sdf = arrays["flow_sdf"]
    if sdf.min() < -1e-6:
        problems.append(f"有效格点落在船体投影内部: min sdf = {sdf.min():.4f}")

    weight = arrays["flow_quad_weights"]
    if not np.allclose(weight, weight[0]):
        problems.append("flow_quad_weights 不均匀")

    uv = arrays["hull_uv"]
    if uv[:, 1].min() < -1e-6 or uv[:, 1].max() > 1.0 + 1e-6:
        problems.append(f"v 越出 [0,1]: [{uv[:, 1].min():.4f}, {uv[:, 1].max():.4f}]")

    hull_points = arrays["hull_points"]
    if hull_points[:, 2].max() > draft + 1e-6:
        problems.append(f"船体域含设计吃水以上的点: max z = {hull_points[:, 2].max():.4f} > T")
    wetted_area = arrays["hull_quad_weights"].sum()
    reference = 0.5 * meta["S0"]
    area_ok = abs(wetted_area / reference - 1.0) <= WETTED_AREA_TOLERANCE
    if meta["S0_source"] == "reference" and not area_ok:
        problems.append(f"湿表面积 {wetted_area:.4f} m² 与 S0/2 = {reference:.4f} m² 偏差超 "
                        f"{WETTED_AREA_TOLERANCE:.0%}")

    diagnostics = {
        "entry": entry_id,
        "n_flow": len(flow),
        "n_hull": len(hull_points),
        "eta": (eta.min(), eta.max()),
        "eta_over_T": (eta.min() / draft, eta.max() / draft),
        "speed_over_U_inf": (speed.min() / meta["U_inf"], speed.max() / meta["U_inf"]),
        "p_rgh": (flow[:, 4].min(), flow[:, 4].max()),
        "sdf": (sdf.min(), sdf.max()),
        "sdf_over_Lpp": (sdf.min() / lpp, sdf.max() / lpp),
        "u_range": (uv[:, 0].min(), uv[:, 0].max()),
        "wetted_area": (wetted_area, reference),
    }
    return diagnostics, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/shipbench")
    parser.add_argument("entries", nargs="*", help="条目 id，默认校验 index.json 中全部条目")
    args = parser.parse_args()

    if args.entries:
        entries = args.entries
    else:
        index = json.loads((Path(args.dataset) / "index.json").read_text())
        entries = sorted(index["entries"])

    failed = False
    for entry_id in entries:
        diagnostics, problems = validate(args.dataset, entry_id)
        print(f"[{entry_id}]")
        for key, value in diagnostics.items():
            if key == "entry":
                continue
            if isinstance(value, tuple):
                print(f"  {key:18s} {value[0]:+.4f} .. {value[1]:+.4f}")
            else:
                print(f"  {key:18s} {value}")
        if problems:
            failed = True
            for problem in problems:
                print(f"  问题: {problem}")
        else:
            print("  OK")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
