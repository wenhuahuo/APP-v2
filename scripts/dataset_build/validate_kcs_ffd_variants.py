"""核查 KCS 的 FFD 全局变体几何是否合法、湿面积信号是否够强。

用法:
    python3 scripts/dataset_build/validate_kcs_ffd_variants.py \
        --stl dataset/.build_cache/kcs_fr312_normal_mesh_25s/KCS_Hull.stl.gz

对每个变体给出：目标位移最大值、Bernstein 最小二乘拟合残差（全船与湿面）、
实际位移最大值、三角形翻转数、湿面积变化、变形后包围盒，并按三条判据判定：

- 三角形翻转数 = 0（曲面不自交折叠）
- 湿面积变化 |Δ| ≥ 10%（eta 信号要高出 20s/25s 截断误差一个量级）
- 变形后仍在 snappyHexMesh 最内层加密框内（|y| ≤ 0.65、z ≥ −0.15、x ∈ [−0.62, 7.72]）

任一变体不达标时以非零状态退出，便于在提交算例前拦截。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/dataset")]

from ffd import (KCS_DRAFT, KCS_VARIANTS, apply_ffd, flipped_triangles, hull_bounds,
                 kcs_displacements, wetted_area)
from stl_io import read_stl_gz

# snappyHexMesh 最内层加密框（KCS normal_mesh 的设置）
REFINEMENT_BOX = {"x": (-0.62, 7.72), "y": (-0.65, 0.65), "z": (-0.15, 0.65)}
MIN_WETTED_CHANGE = 0.10


def inspect(variant, points, faces, bounds, base_area):
    target = KCS_VARIANTS[variant][0](points, bounds)
    displacements = kcs_displacements(variant, points, bounds)
    moved = apply_ffd(points, bounds, displacements)
    shift = moved - points
    area = wetted_area(moved, faces, KCS_DRAFT)
    return {
        "variant": variant,
        "draft_change_m": KCS_VARIANTS[variant][2],
        "target_max_mm": 1e3 * float(np.abs(target).max()),
        "fit_residual_mm": 1e3 * float(np.abs(shift - target).max()),
        # 水线以上只有干舷与甲板，残差集中在那里；湿面残差才是影响绕流的量
        "fit_residual_wet_mm": 1e3 * float(np.abs(shift - target)[points[:, 2] <= KCS_DRAFT].max()),
        "actual_max_mm": 1e3 * float(np.abs(shift).max()),
        "flipped_triangles": flipped_triangles(points, moved, faces),
        "wetted_area_m2": area,
        "wetted_area_change": area / base_area - 1.0,
        "y_max": float(np.abs(moved[:, 1]).max()),
        "z_min": float(moved[:, 2].min()),
        "z_max": float(moved[:, 2].max()),
        "x_range": (float(moved[:, 0].min()), float(moved[:, 0].max())),
    }


def verdict(row):
    """返回不达标项列表；空列表表示通过。"""
    problems = []
    if row["flipped_triangles"]:
        problems.append(f"{row['flipped_triangles']} 个三角形翻转")
    if abs(row["wetted_area_change"]) < MIN_WETTED_CHANGE:
        problems.append(f"湿面积变化 {row['wetted_area_change']:+.2%} < 10%")
    if row["y_max"] > REFINEMENT_BOX["y"][1]:
        problems.append(f"y_max {row['y_max']:.4f} > {REFINEMENT_BOX['y'][1]}")
    if row["z_min"] < REFINEMENT_BOX["z"][0]:
        problems.append(f"z_min {row['z_min']:.4f} < {REFINEMENT_BOX['z'][0]}")
    if row["x_range"][0] < REFINEMENT_BOX["x"][0] or row["x_range"][1] > REFINEMENT_BOX["x"][1]:
        problems.append(f"x 范围 {row['x_range']} 超出加密框")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", default=str(
        ROOT / "dataset/.build_cache/kcs_fr312_normal_mesh_25s/KCS_Hull.stl.gz"),
        help="原船船体 STL（gzip）")
    parser.add_argument("--variants", nargs="+", default=sorted(KCS_VARIANTS),
                        choices=sorted(KCS_VARIANTS))
    args = parser.parse_args()

    points, faces = read_stl_gz(Path(args.stl).expanduser())
    bounds = hull_bounds(points)
    base_area = wetted_area(points, faces, KCS_DRAFT)
    print(f"原船：{len(points)} 点 {len(faces)} 面，湿面积 {base_area:.4f} m²，"
          f"包围盒 x={bounds[0]} y={bounds[1]} z={bounds[2]}")

    print(f"\n{'变体':<14}{'目标':>7}{'残差干':>8}{'残差湿':>8}{'实际':>7}{'翻转':>5}"
          f"{'湿面积':>10}{'变化':>9}{'y_max':>8}{'z_min':>8}  判定")
    failed = []
    for variant in args.variants:
        row = inspect(variant, points, faces, bounds, base_area)
        problems = verdict(row)
        print(f"{variant:<14}{row['target_max_mm']:>6.1f}m{row['fit_residual_mm']:>7.1f}m"
              f"{row['fit_residual_wet_mm']:>7.1f}m{row['actual_max_mm']:>6.1f}m"
              f"{row['flipped_triangles']:>5d}"
              f"{row['wetted_area_m2']:>10.4f}{row['wetted_area_change']:>+9.2%}"
              f"{row['y_max']:>8.4f}{row['z_min']:>8.4f}  "
              f"{'通过' if not problems else '；'.join(problems)}")
        if problems:
            failed.append((variant, problems))

    if failed:
        print(f"\n{len(failed)} 个变体不达标")
        return 1
    print(f"\n{len(args.variants)} 个变体全部达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
