#!/bin/bash
# 组装单个算例的数据集条目并刷新 index.json，组装完立刻校验。
#
# 数据集只在集群侧生成：导出包留在 ~/dataset_export/<source-case>/，不回传笔记本，
# 也不用本地 dataset/.build_cache/ 组装。本脚本在集群上跑，触发方式是本地
# `scripts/dataset_build/submit_pending_assembly.py` 扫到新包后，为每个算例提交一个
# `scripts/dataset_build/assemble_entry.sbatch`；集群上没有常驻轮询进程。
#
# 用法（集群，需要带 numpy/trimesh 的 mesh 环境）：
#   PYTHON=/path/to/cluster-home/.conda/envs/mesh/bin/python3 \
#       build_dataset.sh <source-case> [<source-case> ...]
#
# 环境变量：
#   PYTHON                 解释器，默认 python3
#   APPSOLVER_EXPORT_ROOT  导出包根目录，默认 ~/dataset_export
#   APPSOLVER_DATASET      数据集根目录，默认 <repo>/dataset/shipbench
set -eo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python3}
EXPORT_ROOT=${APPSOLVER_EXPORT_ROOT:-$HOME/dataset_export}
DATASET=${APPSOLVER_DATASET:-$ROOT/dataset/shipbench}

[[ $# -gt 0 ]] || { echo "用法: $0 <source-case> [<source-case> ...]" >&2; exit 2; }

for name in "$@"; do
  bundle="$EXPORT_ROOT/$name"
  for file in case_manifest.json internal.vtu hull.vtp; do
    [[ -s "$bundle/$file" ]] || { echo "导出包不完整: $bundle/$file" >&2; exit 3; }
  done
  # build_entry.py 首行输出 "<entry-id>: n_cells=... flow_points=... hull_points=..."
  out=$("$PYTHON" "$ROOT/src/dataset/build_entry.py" --bundle "$bundle" --dataset "$DATASET")
  echo "$out"
  entry=${out%%:*}
  "$PYTHON" "$ROOT/src/dataset/validate_entry.py" --dataset "$DATASET" "$entry"
  echo "组装并校验通过: $name -> $entry"
done
