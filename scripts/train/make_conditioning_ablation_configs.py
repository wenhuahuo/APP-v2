"""生成工况融合消融配置。

两种协议各六种工况融合方法（`none` / `film` / `adaln_zero` / `fourier_add` /
`cond_token` / `cond_xattn`），只改 `conditioning`，其余逐项沿用对应基础配置：

- APPSolver-v2（`lofo_fr_classic` 划分）：基础配置
  `appsolver_v2_flow_with_hull_random_seed0.yaml`，验证集为 11 个经典船型的设计 Fr；
  训练集使用当前快照的其余条目（含全部融合船型）。训练/验证条目数随
  `configs/splits_multihull.json` 的全量快照变化，不再写死条目数。
  仓库中已提交的六份配置来自 207 条快照的历史结果，重新运行 `--target v2` 会用当前
  快照覆盖它们。
- APP+OFormer（`random_seed0` 划分，固定 GFLOPs 容量）：基础配置
  `appsolver_oformer_random_seed0_isoflops.yaml` 本身是 `fourier_add` 组（主实验默认口径，
  已有训练结果，生成器不重写它），只派生其余五种，各含训练种子 0/1/2。

    python3 scripts/train/make_conditioning_ablation_configs.py            # APPSolver-v2
    python3 scripts/train/make_conditioning_ablation_configs.py --target oformer
"""

import argparse
import json
import re
from pathlib import Path

import make_split_configs

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "appsolver_v2"
BASE = CONFIG_DIR / "appsolver_v2_flow_with_hull_random_seed0.yaml"
SPLITS = ROOT / "configs" / "splits_multihull.json"
METHODS = ("none", "film", "adaln_zero", "fourier_add", "cond_token", "cond_xattn")
# 经典船型个数是数据契约（`AGENT.md`：11 个经典船型），不随快照条目数变化。
CLASSIC_HULL_COUNT = 11
OFORMER_DIR = ROOT / "configs" / "appsolver_oformer"
OFORMER_BASE = OFORMER_DIR / "appsolver_oformer_random_seed0_isoflops.yaml"
# `fourier_add` 组就是基础配置本身（主实验默认口径），只派生其余五种。
OFORMER_METHODS = tuple(method for method in METHODS if method != "fourier_add")


def bullet_block(text, key, items):
    pattern = rf"(^  {key}:\n)(?:  - .*\n)+"
    replacement = lambda match: match.group(1) + "".join(f"  - {item}\n" for item in items)
    result, hits = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if hits != 1:
        raise RuntimeError(f"{key} 模板命中 {hits} 次")
    return result


def lofo_fr_classic_split():
    splits = json.loads(SPLITS.read_text())
    entries = sorted(splits["random_seed0"]["train"] + splits["random_seed0"]["val"])
    val = [entry for entry in splits["lofo_fr_design"]["val"]
           if not re.search(r"_w\d{3}_fr\d{3}_normal_mesh$", entry)]
    if len(val) != CLASSIC_HULL_COUNT or len(entries) <= len(val):
        raise RuntimeError(f"数据快照或经典船型验证集数量异常: {len(entries)} / {len(val)}")
    train = sorted(set(entries) - set(val))
    return train, sorted(val)


def v2_configs():
    """APPSolver-v2 的 `lofo_fr_classic` 六组配置：{文件名: 文本}。"""
    train, val = lofo_fr_classic_split()
    base = "".join(line for line in BASE.read_text().splitlines(keepends=True)
                   if not line.startswith("#"))
    base = base.replace("stats_random_seed0.json", "stats_lofo_fr_classic.json", 1)
    base = bullet_block(base, "entries", train)
    base = bullet_block(base, "val_entries", val)
    base = re.sub(r"^  conditioning: .*\n", "", base, count=1, flags=re.M)

    configs = {}
    for method in METHODS:
        text = base.replace("model:\n", f"model:\n  conditioning: {method}\n", 1)
        header = (f"# 工况融合消融 {method}：lofo_fr_classic，train {len(train)} / "
                  f"val {len(val)}；双域输入、仅流场监督；其余超参与主实验一致\n")
        configs[f"appsolver_v2_conditioning_{method}_lofo_fr_classic.yaml"] = header + text
    return configs


def oformer_configs():
    """APP+OFormer 固定 GFLOPs 容量的五组配置：{文件名: {种子: 文本}}。"""
    base_text = OFORMER_BASE.read_text()
    if "  conditioning: fourier_add\n" not in base_text:
        raise RuntimeError(f"{OFORMER_BASE.name} 不是 fourier_add 组")
    base = "".join(line for line in base_text.splitlines(keepends=True)
                   if not line.startswith("#"))
    configs = {}
    for method in OFORMER_METHODS:
        header = (f"# 划分 random_seed0（random_split，留出 RandomState(0) 抽 15%）："
                  f"train 190 / val 34；工况融合消融 {method}（固定 GFLOPs 容量），"
                  f"其余超参与 {OFORMER_BASE.name} 一致\n")
        seed_texts = {}
        for train_seed in (0, *make_split_configs.TRAIN_SEEDS):
            text = header + base.replace("  conditioning: fourier_add\n",
                                         f"  conditioning: {method}\n", 1)
            if train_seed:
                text = make_split_configs.derive_train_seed_from_text(
                    text, OFORMER_BASE.name, train_seed)
            seed_texts[train_seed] = text
        stem = OFORMER_BASE.stem.replace("_random_seed0_isoflops", "")
        configs[f"{stem}_conditioning_{method}_random_seed0_isoflops.yaml"] = seed_texts
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("v2", "oformer", "all"), default="v2",
                        help="生成哪一组配置；默认只生成 APPSolver-v2 的六组")
    args = parser.parse_args()

    if args.target in {"v2", "all"}:
        for filename, text in v2_configs().items():
            (CONFIG_DIR / filename).write_text(text)
            print(filename)

    if args.target in {"oformer", "all"}:
        for filename, seed_texts in oformer_configs().items():
            for train_seed, text in seed_texts.items():
                name = (filename if not train_seed
                        else make_split_configs.train_seed_output_name(filename, train_seed))
                (OFORMER_DIR / name).write_text(text)
                print(name)


if __name__ == "__main__":
    main()
