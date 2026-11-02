"""按 `configs/splits_multihull.json` 从主实验基础配置派生三个划分的配置。

    python3 scripts/train/make_split_configs.py

两种容量口径各派生 27 份，共 54 份：9 模型 × 3 划分

- 固定参数量：`<模型>_multihull_normal.yaml` → `<模型>_<划分>.yaml`
- 固定计算量：`<模型>_multihull_isoflops.yaml` → `<模型>_<划分>_isoflops.yaml`

每个派生配置只改三处：`train.entries`、`train.val_entries`、`stats`（指向该划分
训练集拟合的统计量），外加开头一行头注释。模型结构、容量超参、任务口径与优化超参与
基础配置完全一致（双域输入、仅流场输出、逐层 FiLM、`lambda_hull=0.0`），因此同一
容量口径下的 9 个模型是同口径对照，两种容量口径之间的差别只有主容量。
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
# (基础配置后缀, 派生配置的附加后缀)
CAPACITIES = (("_multihull_normal", ""), ("_multihull_isoflops", "_isoflops"))
CAPACITY_PROTOCOLS = {
    "_multihull_normal": "固定参数量",
    "_multihull_isoflops": "固定计算量（iso-FLOPs）",
}


def bullet(items, indent="  - "):
    return "".join(f"{indent}{item}\n" for item in items)


def replace_block(text, key, items):
    pattern = rf"(^  {key}:\n)(?:  - .*\n)+"
    new, hits = re.subn(pattern, lambda m: m.group(1) + bullet(items), text, count=1, flags=re.M)
    if hits != 1:
        raise RuntimeError(f"{key} 模板命中 {hits} 次")
    return new


def derive(base_path, split_name, split):
    """返回 `base_path` 在划分 `split` 下的派生配置全文（纯函数，便于幂等校验）。"""
    base_path = Path(base_path)
    base_suffix = next(suffix for suffix, _ in CAPACITIES if base_path.stem.endswith(suffix))
    # 基础配置开头写着旧的 4+2 船型划分，派生配置换成自己的头注释
    text = "".join(line for line in base_path.read_text().splitlines(keepends=True)
                   if not line.startswith("#"))
    text = text.replace("stats: dataset/shipbench/stats.json",
                        f"stats: dataset/shipbench/stats_{split_name}.json")
    text = replace_block(text, "entries", split["train"])
    text = replace_block(text, "val_entries", split["val"])
    header = (f"# 划分 {split_name}（{split['protocol']}，留出 {split['holdout']}）："
              f"train {len(split['train'])} / val {len(split['val'])}；"
              f"{CAPACITY_PROTOCOLS[base_suffix]}容量，其余超参与 {base_path.name} 一致\n")
    return header + text


def output_name(base_path, split_name):
    base_path = Path(base_path)
    base_suffix, extra = next((suffix, extra) for suffix, extra in CAPACITIES
                              if base_path.stem.endswith(suffix))
    model = base_path.stem[: -len(base_suffix)]
    return f"{model}_{split_name}{extra}.yaml"


def main():
    splits = {k: v for k, v in json.loads((CONFIG_DIR / "splits_multihull.json").read_text()).items()
              if not k.startswith("_")}
    for base_suffix, _ in CAPACITIES:
        bases = sorted(CONFIG_DIR.glob(f"*/*{base_suffix}.yaml"))
        if len(bases) != 9:
            raise RuntimeError(f"期望 9 个 {base_suffix} 基础配置，实际 {len(bases)}")
        for base in bases:
            for name, split in splits.items():
                out = base.parent / output_name(base, name)
                out.write_text(derive(base, name, split))
                print(f"{out.name}: train {len(split['train'])} / val {len(split['val'])}")


if __name__ == "__main__":
    main()
