"""把 `output/<配置名>/` 归档到带时间戳的子目录。

训练与评测按配置文件名写 `output/<配置名>/`。归档时若沿用原名（`output/archive/<标签>/<配置名>/`），
NFS 的目录项缓存可能让后续运行依旧解析到刚被移走的旧目录，把新产物写进归档里——本轮实验
就出过这个事故。这里统一把归档目录命名为 `<配置名>__archived_<时间戳>`，归档名与活跃名
永不重复，旧目录「仍以同名存在」这个前提就消失了。

`<配置名>` 默认取 `output/<配置名>`；给带路径的形式（含 `/`）时按该路径取源目录，用于
归档已经在 `output/archive/<标签>/` 里的旧产物。同名前缀的归档已存在时自动加 `_2`、`_3` 后缀，
不覆盖任何已有产物。

用法:
    python3 scripts/eval/archive_outputs.py --label pre_sampling_fix \
        appsolver_v2_random_seed0_isoflops gnot_random_seed0_train_seed1_isoflops
    python3 scripts/eval/archive_outputs.py --label stale --dry-run appsolver_v2_random_seed0
"""

import argparse
import datetime
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"


def archive_one(spec, label, output_root, timestamp, dry_run=False):
    """归档单个 run 目录，返回 (源, 目标)；源目录不存在时报错。"""
    path = Path(spec)
    source = path if (path.is_absolute() or "/" in spec) else output_root / spec
    if not source.is_dir():
        raise FileNotFoundError(f"run 目录不存在: {source}")
    target_dir = output_root / "archive" / label
    target = target_dir / f"{source.name}__archived_{timestamp}"
    suffix = 2
    while target.exists():
        target = target_dir / f"{source.name}__archived_{timestamp}_{suffix}"
        suffix += 1
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    return source, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="配置名（`output/<配置名>`）或含 / 的目录路径")
    parser.add_argument("--label", required=True, help="归档标签，对应 `output/archive/<标签>/`")
    parser.add_argument("--timestamp", default=None,
                        help="归档时间戳，缺省取当前时间 `%%Y%%m%%d_%%H%%M%%S`")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不移动目录")
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for spec in args.runs:
        source, target = archive_one(spec, args.label, OUTPUT, timestamp, args.dry_run)
        print(f"{'mv' if not args.dry_run else 'mv (dry-run)'} {source} -> {target}")


if __name__ == "__main__":
    main()
