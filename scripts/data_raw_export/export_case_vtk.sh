#!/bin/bash
# 集群端：把单个算例末态导出为 VTK 原始数据包，供本地组装数据集使用。
#
# 用法: export_case_vtk.sh <case-dir> <out-root>
# 产物: <out-root>/<case>/{internal.vtu,hull.vtp,alpha.water.t0,KCS_Hull.stl.gz,
#                          case_manifest.json,kcs_reference.json,log.foamToVTK}
set -eo pipefail

CASE=$(cd "$1" && pwd)
OUT=$2
NAME=$(basename "$CASE")
WORK=$OUT/$NAME

[[ -n "$NAME" && "$WORK" == "$OUT"/* ]] || { echo "输出路径异常: $WORK" >&2; exit 2; }

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load openfoam/v2506

# foamToVTK 需要可写的算例目录；用符号链接把只读的算例内容挂进工作目录，
# 使 VTK/ 落在导出区而不是算例目录
rm -rf "$WORK"
mkdir -p "$WORK"
ln -s "$CASE/constant" "$WORK/constant"
ln -s "$CASE/system"   "$WORK/system"
for t in "$CASE"/[0-9]*; do
  ln -s "$t" "$WORK/$(basename "$t")"
done

cd "$WORK"
# 单元中心用 OpenFOAM 自己写的 C 场：vtkCellCenters 对少数多面体单元给出错误中心
postProcess -func writeCellCentres -latestTime > log.writeCellCentres 2>&1
foamToVTK -latestTime -fields "(U p_rgh alpha.water C)" -patches hull -no-point-data \
  > log.foamToVTK 2>&1

VDIR=$(find VTK -mindepth 1 -maxdepth 1 -type d | head -1)
[[ -f "$VDIR/internal.vtu" && -f "$VDIR/boundary/hull.vtp" ]] || {
  echo "VTK 导出不完整，见 $WORK/log.foamToVTK" >&2; exit 3; }

cp "$VDIR/internal.vtu" "$WORK/internal.vtu"
cp "$VDIR/boundary/hull.vtp" "$WORK/hull.vtp"
cp "$CASE/case_manifest.json" "$WORK/case_manifest.json"
cp "$CASE/KCS_Hull.stl.gz" "$WORK/KCS_Hull.stl.gz"
# 初始自由面直接取 setFields 写入的原始场文件，避免按 setFieldsDict 重建的近似
cp "$CASE/0/alpha.water" "$WORK/alpha.water.t0"

REF=$HOME/Documents/ShipMaster/openfoam/calm_water/reference/kcs_tokyo2015_case2_1.json
[[ -f "$REF" ]] && cp "$REF" "$WORK/kcs_reference.json"

ls -la "$WORK" | awk 'NR>1 {printf "%12s  %s\n", $5, $9}'
