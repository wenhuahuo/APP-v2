#!/bin/bash
# 站点端：把单个算例末态导出为 VTK 原始数据包，供本地组装数据集使用。
#
# 用法: export_case_vtk.sh <case-dir> <out-root> [time]
# 产物: <out-root>/<case>/{internal.vtu,hull.vtp,KCS_Hull.stl.gz,
#                          case_manifest.json,kcs_reference.json,log.foamToVTK}
#
# 只导出体场与 hull patch：二维自由液面由 alpha.water = 0.5 等值面重建，
# 初始自由面在设计吃水下恒为平面，不需要导出 0/alpha.water。
# 不给 time 时导出最新时刻；给 time 时导出该物理时刻（顶层缺该时刻要先 reconstructPar）。
#
# foamToVTK 写在 <out-root>/.foamtovtk-<case>/ 这个临时目录里，包目录只放上面几个
# 文件：VTK/ 里的 internal.vtu 与包里的完全相同，留在包里等于把回传体积翻一倍。
set -eo pipefail

CASE=$(cd "$1" && pwd)
OUT=$2
VTK_TIME=${3:+-time $3}
VTK_TIME=${VTK_TIME:--latestTime}
NAME=$(basename "$CASE")
WORK=$OUT/$NAME
TMP=$OUT/.foamtovtk-$NAME

[[ -n "$NAME" && "$WORK" == "$OUT"/* ]] || { echo "输出路径异常: $WORK" >&2; exit 2; }

# 集群走 module（非交互 shell 要先 source 一次 module 初始化），
# 租用服务器没有 module，由 OPENFOAM_BASHRC 指定 bashrc
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/openscow-modules.sh ]]; then
  source /etc/profile.d/openscow-modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module purge
  module load openfoam/v2506
elif [[ -n "${OPENFOAM_BASHRC:-}" ]]; then
  # OpenFOAM 的 setup 脚本末尾会返回非零，触发 set -e；结束后统一校验
  set +e
  source "$OPENFOAM_BASHRC"
  set -e
  [[ -n "${WM_PROJECT_DIR:-}" ]] || { echo "OpenFOAM 环境未就绪（$OPENFOAM_BASHRC）" >&2; exit 2; }
else
  echo "找不到 OpenFOAM 环境：既没有 module，也没有设 OPENFOAM_BASHRC" >&2
  exit 2
fi

# foamToVTK 需要可写的算例目录；用符号链接把只读的算例内容挂进临时工作目录，
# 使 VTK/ 落在临时目录而不是算例目录，也不占包目录空间
rm -rf "$TMP"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
ln -s "$CASE/constant" "$TMP/constant"
ln -s "$CASE/system"   "$TMP/system"
for t in "$CASE"/[0-9]*; do
  ln -s "$t" "$TMP/$(basename "$t")"
done

cd "$TMP"
foamToVTK $VTK_TIME -fields "(U p_rgh alpha.water)" -patches hull -no-point-data \
  > log.foamToVTK 2>&1

VDIR=$(find VTK -mindepth 1 -maxdepth 1 -type d | head -1)
[[ -f "$VDIR/internal.vtu" && -f "$VDIR/boundary/hull.vtp" ]] || {
  echo "VTK 导出不完整，见 $TMP/log.foamToVTK" >&2; exit 3; }

rm -rf "$WORK"
mkdir -p "$WORK"
cp "$VDIR/internal.vtu" "$WORK/internal.vtu"
cp "$VDIR/boundary/hull.vtp" "$WORK/hull.vtp"
cp "$CASE/case_manifest.json" "$WORK/case_manifest.json"
cp log.foamToVTK "$WORK/log.foamToVTK"

# 船体 STL 名字按船型命名（`KCS_Hull.stl.gz`、`dtc_w075_Hull.stl.gz` …），从 manifest 读，
# 不硬编码；KCS manifest 用 `stl_gz`，批量船型用 `geometry.stl`。manifest 里的 STL 有
# 三种写法：绝对路径、相对算例目录、相对仓库根目录
STL=$(python3 -c '
import json, sys
manifest = json.load(open(sys.argv[1]))
print(manifest.get("stl_gz") or manifest["geometry"]["stl"])
' "$CASE/case_manifest.json")
if [[ "$STL" = /* ]]; then
  cp "$STL" "$WORK/$(basename "$STL")"
elif [[ -f "$CASE/$STL" ]]; then
  cp "$CASE/$STL" "$WORK/$(basename "$STL")"
elif [[ -f "$(dirname "$CASE")/$STL" ]]; then
  cp "$(dirname "$CASE")/$STL" "$WORK/$(basename "$STL")"
else
  echo "找不到 STL：$STL（算例 $CASE）" >&2
  exit 3
fi

# 只有 KCS 有参考几何（Lpp/T/S0/rho/nu）；其余船型的这些量在 manifest 里
REF=$HOME/Documents/ShipMaster/openfoam/calm_water/reference/kcs_tokyo2015_case2_1.json
if [[ -f "$CASE/KCS_Hull.stl.gz" ]]; then
  cp "$REF" "$WORK/kcs_reference.json"
fi

ls -la "$WORK" | awk 'NR>1 {printf "%12s  %s\n", $5, $9}'
