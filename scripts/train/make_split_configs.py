"""按 `configs/splits_multihull.json` 从主实验基础配置派生三个划分的配置，
并从 `random_seed0` 派生多种子复跑配置。

    python3 scripts/train/make_split_configs.py

两种容量口径各派生 27 份，共 54 份：9 模型 × 3 划分

- 固定参数量：`<模型>_multihull_normal.yaml` → `<模型>_<划分>.yaml`
- 固定计算量：`<模型>_multihull_isoflops.yaml` → `<模型>_<划分>_isoflops.yaml`

每个派生配置只改三处：`train.entries`、`train.val_entries`、`stats`（指向该划分
训练集拟合的统计量），外加开头一行头注释。模型结构、容量超参、任务口径与优化超参与
基础配置完全一致（双域输入、流场与船面双域输出、逐层 Fourier add、`lambda_hull=1.0`），
因此同一容量口径下的模型互为同口径对照，两种容量口径之间的差别只有主容量。

多种子复跑以 `random_seed0` 的 14 份配置为基准再派生 28 份（7 模型 × 2 种容量 ×
训练种子 1、2）：

- 固定参数量：`<模型>_random_seed0.yaml` → `<模型>_random_seed0_train_seed<S>.yaml`
- 固定计算量：`<模型>_random_seed0_isoflops.yaml` →
  `<模型>_random_seed0_train_seed<S>_isoflops.yaml`

GINO 与 CORAL 已从实验计划中移除（训练成本与数值塌缩），源代码与
`configs/{gino,coral}/*` 保留，但不进入实验计划，也不再派生它们的多种子与多分辨率配置。

数据划分始终是 `random_seed0`（同一份 `stats_random_seed0.json`、同样的 190/34 条目），
文件名里 `random_seed0` 与 `train_seed<S>` 分别标注「数据划分」与「训练种子」；派生
配置只改 `train.seed` 与头注释，容量、训练预算与划分逐项不变。

泛化实验沿用同一条派生路径：两个留出划分的固定计算量配置再派生训练种子 1、2，共 20 份
（5 模型 × 2 划分 × 2 种子）：

- `<模型>_loto_kvlcc2_isoflops.yaml` → `<模型>_loto_kvlcc2_train_seed<S>_isoflops.yaml`
- `<模型>_lofo_fr_design_isoflops.yaml` →
  `<模型>_lofo_fr_design_train_seed<S>_isoflops.yaml`
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
# 主实验的 7 个模型（主表顺序，配置目录同名）
MODELS = ("appsolver_v2", "appsolver_oformer", "appsolver_v1", "gnot", "upt",
          "transolver", "oformer")
# 多种子复跑：数据划分固定为 MAIN_SPLIT，只换训练种子
MAIN_SPLIT = "random_seed0"
TRAIN_SEEDS = (1, 2)
# 派生配置的附加容量后缀 -> 容量口径描述
DERIVED_PROTOCOLS = {"": CAPACITY_PROTOCOLS["_multihull_normal"],
                     "_isoflops": CAPACITY_PROTOCOLS["_multihull_isoflops"]}
# 泛化实验：两个留出划分、5 个模型，只派生固定计算量容量
GENERALIZATION_SPLITS = ("loto_kvlcc2", "lofo_fr_design")
GENERALIZATION_MODELS = ("appsolver_oformer", "transolver", "appsolver_v2", "gnot", "oformer")


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


def split_capacity(base_name, split_name=MAIN_SPLIT):
    """返回 `<模型>_<划分>[_isoflops]` 的附加容量后缀，并校验文件名属于该划分。"""
    head, sep, extra = Path(base_name).stem.partition(f"_{split_name}")
    if not head or not sep or extra not in DERIVED_PROTOCOLS:
        raise RuntimeError(f"{base_name} 不是划分 {split_name} 的派生配置")
    return extra


def derive_train_seed(base_path, train_seed, split_name=MAIN_SPLIT):
    """返回 `base_path` 在训练种子 `train_seed` 下的派生配置全文（纯函数）。

    只改 `train.seed` 与头注释：数据划分、统计量、容量与训练预算逐项沿用基准配置。
    """
    base_path = Path(base_path)
    return derive_train_seed_from_text(base_path.read_text(), base_path.name, train_seed,
                                       split_name)


def derive_train_seed_from_text(base_text, base_name, train_seed, split_name=MAIN_SPLIT):
    """同 `derive_train_seed`，输入是配置全文与文件名，供尚未落盘的派生配置使用。"""
    train_seed = int(train_seed)
    lines = base_text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("#"):
        raise RuntimeError(f"{base_name} 缺少头注释")
    text, hits = re.subn(r"^  seed: \d+$", f"  seed: {train_seed}", "".join(lines[1:]),
                         count=1, flags=re.M)
    if hits != 1:
        raise RuntimeError(f"{base_name} 的 train.seed 命中 {hits} 次")
    protocol = DERIVED_PROTOCOLS[split_capacity(base_name, split_name)]
    # 头注释沿用基准配置的划分说明（划分与条目数不变），插入训练种子
    split_header = lines[0].rstrip("\n").split("；")[0]
    header = (f"{split_header}；训练种子 {train_seed}（数据划分仍为 {split_name}）；"
              f"{protocol}容量，其余超参与 {base_name} 一致\n")
    return header + text


def train_seed_output_name(base_path, train_seed, split_name=MAIN_SPLIT):
    """`<模型>_<划分>[_isoflops].yaml` → `<模型>_<划分>_train_seed<S>[_isoflops].yaml`。"""
    stem = Path(base_path).stem
    extra = split_capacity(base_path, split_name)
    model = stem[: -len(f"_{split_name}{extra}")]
    return f"{model}_{split_name}_train_seed{int(train_seed)}{extra}.yaml"


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
    # 多种子复跑：从 random_seed0 的 18 份配置派生，数据划分与统计量保持不变
    for capacity_suffix in ("", "_isoflops"):
        for model in MODELS:
            base = CONFIG_DIR / model / f"{model}_{MAIN_SPLIT}{capacity_suffix}.yaml"
            if not base.is_file():
                raise RuntimeError(f"缺少 {MAIN_SPLIT} 基准配置 {base}")
            for train_seed in TRAIN_SEEDS:
                out = base.parent / train_seed_output_name(base, train_seed)
                out.write_text(derive_train_seed(base, train_seed))
                print(f"{out.name}: train.seed {train_seed}，划分 {MAIN_SPLIT}")
    # 泛化实验：两个留出划分的固定计算量配置补训练种子 1、2，划分与统计量不变
    for split_name in GENERALIZATION_SPLITS:
        for model in GENERALIZATION_MODELS:
            base = CONFIG_DIR / model / f"{model}_{split_name}_isoflops.yaml"
            if not base.is_file():
                raise RuntimeError(f"缺少 {split_name} 基准配置 {base}")
            for train_seed in TRAIN_SEEDS:
                out = base.parent / train_seed_output_name(base, train_seed, split_name)
                out.write_text(derive_train_seed(base, train_seed, split_name))
                print(f"{out.name}: train.seed {train_seed}，划分 {split_name}")


if __name__ == "__main__":
    main()
