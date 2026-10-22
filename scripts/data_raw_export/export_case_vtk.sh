#!/bin/bash
# 集群端：把单个算例末态导出为 VTK 原始数据包，供本地组装数据集使用。
#
# 用法: export_case_vtk.sh <case-dir> <out-root>
# 产物: <out-root>/<case>/{internal.vtu,hull.vtp,KCS_Hull.stl.gz,
#                          case_manifest.json,kcs_reference.json,log.foamToVTK}
#
# 只导出末态体场与 hull patch：二维自由液面由 alpha.water = 0.5 等值面重建，
# 初始自由面在设计吃水下恒为平面，不需要导出 0/alpha.water。
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
foamToVTK -latestTime -fields "(U p_rgh alpha.water)" -patches hull -no-point-data \
  > log.foamToVTK 2>&1

VDIR=$(find VTK -mindepth 1 -maxdepth 1 -type d | head -1)
[[ -f "$VDIR/internal.vtu" && -f "$VDIR/boundary/hull.vtp" ]] || {
  echo "VTK 导出不完整，见 $WORK/log.foamToVTK" >&2; exit 3; }

cp "$VDIR/internal.vtu" "$WORK/internal.vtu"
cp "$VDIR/boundary/hull.vtp" "$WORK/hull.vtp"
cp "$CASE/case_manifest.json" "$WORK/case_manifest.json"

# 船体 STL 名字按船型命名（`KCS_Hull.stl.gz`、`dtc_Hull.stl.gz` …），从 manifest 读，
# 不硬编码；KCS manifest 用 `stl_gz`，批量船型用 `geometry.stl`。
# 批量船型给相对文件名，KCS 的 manifest 给绝对路径，两种都要支持
STL=$(python3 -c '
import json, sys
manifest = json.load(open(sys.argv[1]))
print(manifest.get("stl_gz") or manifest["geometry"]["stl"])
' "$CASE/case_manifest.json")
if [[ "$STL" = /* ]]; then
  cp "$STL" "$WORK/$(basename "$STL")"
else
  cp "$CASE/$STL" "$WORK/$STL"
fi

# 只有 KCS 有参考几何（Lpp/T/S0/rho/nu）；其余船型的这些量在 manifest 里
REF=$HOME/Documents/ShipMaster/openfoam/calm_water/reference/kcs_tokyo2015_case2_1.json
if [[ -f "$CASE/KCS_Hull.stl.gz" ]]; then
  cp "$REF" "$WORK/kcs_reference.json"
fi

ls -la "$WORK" | awk 'NR>1 {printf "%12s  %s\n", $5, $9}'
