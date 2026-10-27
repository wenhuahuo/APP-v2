#!/bin/bash
# 把 dataset/.build_cache/ 里的集群导出包组装成数据集条目并刷新 index.json
#
# 用法: build_dataset.sh <source-case> [<source-case> ...]
set -eo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[[ $# -gt 0 ]] || { echo "用法: $0 <source-case> [...]" >&2; exit 2; }

for name in "$@"; do
  python3 "$ROOT/src/dataset/build_entry.py" \
    --bundle "$ROOT/dataset/.build_cache/$name" \
    --dataset "$ROOT/dataset/shipbench"
done
