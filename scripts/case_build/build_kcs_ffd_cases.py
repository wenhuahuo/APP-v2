"""生成 KCS 的 FFD 几何变体算例骨架，网格由各算例的 Allmesh 生成。

用法:
    python3 scripts/case_build/build_kcs_ffd_cases.py \
        --source ~/Documents/OpenFoamWorks/APPSolver-v2-Dataset/kcs_normal_mesh_25s \
        --variants wide_midship narrow_entrance full_stern shallow_body deep_forefoot \
                   bow_bulb_protrusion bilge_rounding stern_dent bow_flare_underwater

每个变体生成 `<root>/kcs_<variant>_normal_mesh_25s`：

- `KCS_Hull.stl.gz`：按 FFD 变形后的船体表面
- `system/`、`0.orig/`、`Allmesh`：复制源算例，网格由 Allmesh 重建
- `scripts/submit.sbatch`：建网格作业（blockMesh + 6 次 refineMesh + snappyHexMesh）
- `ffd_displacements.npy`：控制点位移 (9, 5, 6, 3)，几何标签
- `case_manifest.json`：变体名、目标位移、湿面积变化

派生 6 个 Fr 工况用上游的 build_kcs_fr_sweep.py --variant <name>。
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/dataset"))

from ffd import KCS_VARIANTS, apply_ffd, hull_bounds, kcs_displacements
from stl_io import read_stl_gz, write_stl_gz

MESH_SUBMIT = """#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=burst
#SBATCH --job-name={job}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --chdir={chdir}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load openfoam/v2506

printf 'job=%s name=%s host=%s start=%s cwd=%s\\n' \\
  "${{SLURM_JOB_ID}}" "${{SLURM_JOB_NAME}}" "$(hostname)" "$(date -Is)" "$(pwd)"

./Allmesh

printf 'job=%s finish=%s\\n' "${{SLURM_JOB_ID}}" "$(date -Is)"
"""


def wetted_area(points, faces, draft):
    """设计吃水以下三角形的面积和。"""
    triangles = points[faces]
    wetted = (triangles[:, :, 2] <= draft).all(axis=1)
    cross = np.cross(triangles[wetted, 1] - triangles[wetted, 0],
                     triangles[wetted, 2] - triangles[wetted, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--variants", nargs="+", required=True, choices=sorted(KCS_VARIANTS))
    parser.add_argument("--end-time", type=float, required=True, help="物理时长 s")
    parser.add_argument("--draft", type=float, default=0.3418)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    root = source.parent
    end_time = args.end_time
    source_stl = source / "KCS_Hull.stl.gz"
    points, faces = read_stl_gz(source_stl)
    bounds = hull_bounds(points)
    base_area = wetted_area(points, faces, args.draft)

    for variant in args.variants:
        name = f"kcs_{variant}_normal_mesh_{end_time:g}s"
        destination = root / name
        if destination.exists():
            raise SystemExit(f"{destination} 已存在，先确认是否要覆盖")
        destination.mkdir()
        for item in ("system", "0.orig"):
            shutil.copytree(source / item, destination / item)
        shutil.copytree(source / "constant", destination / "constant",
                        ignore=shutil.ignore_patterns("polyMesh", "triSurface",
                                                      "extendedFeatureEdgeMesh"))
        shutil.copy2(source / "Allmesh", destination / "Allmesh")
        control = destination / "system" / "controlDict"
        text = control.read_text()
        updated, count = re.subn(r"^endTime\s+[\d.]+;", f"endTime         {end_time:g};",
                                 text, count=1, flags=re.M)
        if count != 1:
            raise SystemExit(f"{control} 里找不到 endTime")
        control.write_text(updated)
        (destination / "logs").mkdir()
        (destination / "scripts").mkdir()
        (destination / "scripts" / "submit.sbatch").write_text(
            MESH_SUBMIT.format(job=name.replace("_", "-"), chdir=destination))

        displacements = kcs_displacements(variant, points, bounds)
        moved = apply_ffd(points, bounds, displacements)
        write_stl_gz(destination / "KCS_Hull.stl.gz", moved, faces, solid=name)
        np.save(destination / "ffd_displacements.npy", displacements)

        area = wetted_area(moved, faces, args.draft)
        manifest = json.loads((source / "case_manifest.json").read_text())
        manifest.update({
            "case": name,
            "end_time": end_time,
            "run": name,
            "derived_from": source.name,
            "stl_gz": str(destination / "KCS_Hull.stl.gz"),
            "geometry_variant": {
                "name": variant,
                "description": KCS_VARIANTS[variant][1],
                "method": "FFD：9×5×6 控制格架，三元 Bernstein 位移场，控制点位移由目标位移最小二乘反解",
                "control_lattice": [9, 5, 6],
                "lattice_bounds": [list(b) for b in bounds],
                "max_displacement_m": float(np.abs(moved - points).max()),
                "wetted_area_m2": area,
                "wetted_area_change": area / base_area - 1.0,
            },
            "modifications": [f"船体几何 FFD 变体 {variant}（{KCS_VARIANTS[variant][1]}）",
                              f"网格由 Allmesh 重建，加密框与边界层设置沿用 {source.name}"],
        })
        manifest.pop("mesh_reused_from", None)
        manifest["numerics"] = dict(manifest.get("numerics") or {})
        manifest["numerics"]["endTime"] = f"{end_time:g} s"
        (destination / "case_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"{name}: 最大位移 {manifest['geometry_variant']['max_displacement_m']*1000:.1f} mm, "
              f"湿面积 {area:.4f} m² ({area/base_area-1:+.2%})")


if __name__ == "__main__":
    main()
