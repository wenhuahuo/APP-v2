#!/bin/bash
# 集群端：把单个算例的指定时刻导出为 VTK 原始数据包，供瞬时时间敏感性分析使用。
#
# 用法: export_case_times_vtk.sh <case-dir> <out-root> "<time> <time> ..."
# 产物: <out-root>/<case>/<time>/{internal.vtu,hull.vtp}
#
# 与 export_case_vtk.sh 的区别：只导出给定时刻、不复制 STL 与 manifest。
# 求解时每个物理秒都写了场（processor*/ 保留 0..endTime），所以历史算例不必重跑。
#
# 每个时刻单独调用一次 foamToVTK：一次调用只产生一个 VTK 子目录，不必按时间步号
# 反查物理时间（v2506 的 foamToVTK 没有 `-useTimeName`）。符号链接放在独立的
# 工作目录里，避免与输出目录同名冲突。
set -eo pipefail

CASE=$(cd "$1" && pwd)
OUT=$2
TIMES=$3
NAME=$(basename "$CASE")
WORK=$OUT/.work/$NAME

[[ -n "$NAME" && "$WORK" == "$OUT"/* ]] || { echo "输出路径异常: $WORK" >&2; exit 2; }

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load openfoam/v2506

# 顶层只重建过末态；缺失的时刻先由 processor*/ 重建
for t in $TIMES; do
  if [[ ! -d "$CASE/$t" ]]; then
    echo "reconstructPar -time $t ($NAME)"
    (cd "$CASE" && reconstructPar -time "$t")
  fi
done

rm -rf "$WORK" "$OUT/$NAME"
mkdir -p "$OUT/$NAME"
for t in $TIMES; do
  STEP=$WORK/$t
  mkdir -p "$STEP"
  ln -s "$CASE/constant" "$STEP/constant"
  ln -s "$CASE/system"   "$STEP/system"
  ln -s "$CASE/$t"       "$STEP/$t"
  (cd "$STEP" && foamToVTK -time "$t" -fields "(U p_rgh alpha.water)" \
      -patches hull -no-point-data > log.foamToVTK 2>&1)
  VDIR=$(find "$STEP/VTK" -mindepth 1 -maxdepth 1 -type d | head -1)
  [[ -f "$VDIR/internal.vtu" && -f "$VDIR/boundary/hull.vtp" ]] || {
    echo "时刻 $t 的 VTK 导出不完整，见 $STEP/log.foamToVTK" >&2; exit 3; }
  mkdir -p "$OUT/$NAME/$t"
  cp "$VDIR/internal.vtu" "$OUT/$NAME/$t/internal.vtu"
  cp "$VDIR/boundary/hull.vtp" "$OUT/$NAME/$t/hull.vtp"
done
rm -rf "$WORK"

ls -la "$OUT/$NAME"/*/ | awk 'NF>4 {printf "%12s  %s\n", $5, $9}'
