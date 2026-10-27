"""按 `configs/splits_multihull.json` 从 7 个多船型配置派生 3 个划分的配置。

    python3 scripts/train/make_split_configs.py

每个派生配置只改三处：`train.entries`、`train.val_entries`、`stats`（指向该划分
训练集拟合的统计量）。模型结构与优化超参与 `*_multihull_normal.yaml` 完全一致，
因此 7 个模型之间仍是 248 GFLOPs 等算力对照。
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
BASE_SUFFIX = "_multihull_normal"


def bullet(items, indent="  - "):
    return "".join(f"{indent}{item}\n" for item in items)


def replace_block(text, key, items):
    pattern = rf"(^  {key}:\n)(?:  - .*\n)+"
    new, hits = re.subn(pattern, lambda m: m.group(1) + bullet(items), text, count=1, flags=re.M)
    if hits != 1:
        raise RuntimeError(f"{key} 模板命中 {hits} 次")
    return new


def main():
    splits = {k: v for k, v in json.loads((CONFIG_DIR / "splits_multihull.json").read_text()).items()
              if not k.startswith("_")}
    bases = sorted(CONFIG_DIR.glob(f"*{BASE_SUFFIX}.yaml"))
    if len(bases) != 7:
        raise RuntimeError(f"期望 7 个基础配置，实际 {len(bases)}")
    for base in bases:
        model = base.stem[: -len(BASE_SUFFIX)]
        # 基础配置开头写着旧的 4+2 船型划分，派生配置换成自己的头注释
        text = "".join(line for line in base.read_text().splitlines(keepends=True)
                       if not line.startswith("#"))
        for name, split in splits.items():
            derived = text
            derived = derived.replace("stats: dataset/shipbench/stats.json",
                                      f"stats: dataset/shipbench/stats_{name}.json")
            derived = replace_block(derived, "entries", split["train"])
            derived = replace_block(derived, "val_entries", split["val"])
            header = (f"# 划分 {name}（{split['protocol']}，留出 {split['holdout']}）："
                      f"train {len(split['train'])} / val {len(split['val'])}；"
                      f"其余超参与 {base.name} 一致\n")
            out = CONFIG_DIR / f"{model}_{name}.yaml"
            out.write_text(header + derived)
            print(f"{out.name}: train {len(split['train'])} / val {len(split['val'])}")


if __name__ == "__main__":
    main()
