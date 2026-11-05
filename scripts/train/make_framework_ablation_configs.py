"""从随机划分默认配置派生船体融合机制消融配置。

两个模型各三组，只改变输入域、输出域和船面损失权重，其余超参逐项继承基础配置：

- APPSolver-v2（`random_seed0` 划分）：基础配置 `appsolver_v2_flow_with_hull_random_seed0.yaml`；
- APP+OFormer（`random_seed0` 划分，固定 GFLOPs 容量）：基础配置
  `appsolver_oformer_random_seed0_isoflops.yaml`，即 `flow_with_hull` 组本身（正式实验
  直接复用它的三个训练种子 run），生成器不重写它，只派生 `flow_only` 与
  `joint_flow_hull` 两组，每组训练种子 0/1/2。

    python3 scripts/train/make_framework_ablation_configs.py
"""

import re
from pathlib import Path

import yaml

import make_split_configs

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "appsolver_v2"
OFORMER_DIR = ROOT / "configs" / "appsolver_oformer"
BASE = CONFIG_DIR / "appsolver_v2_flow_with_hull_random_seed0.yaml"
# APP+OFormer 的 `flow_with_hull` 组就是固定 GFLOPs 的全量 random_seed0 配置，直接复用。
OFORMER_BASE = OFORMER_DIR / "appsolver_oformer_random_seed0_isoflops.yaml"
OFORMER_DERIVED = ("flow_only", "joint_flow_hull")
VARIANTS = {
    "flow_only": (("flow",), ("flow",)),
    "flow_with_hull": (("flow", "hull"), ("flow",)),
    "joint_flow_hull": (("flow", "hull"), ("flow", "hull")),
}


def replace_domains(text, key, domains):
    pattern = rf"(^  {key}:\n)(?:  - .*\n)+"
    block = f"  {key}:\n" + "".join(f"  - {domain}\n" for domain in domains)
    result, hits = re.subn(pattern, block, text, count=1, flags=re.M)
    if hits != 1:
        raise RuntimeError(f"{key} 模板命中 {hits} 次")
    return result


def derive(base_text, input_domains, output_domains):
    """把基础配置文本改成指定输入/输出域的变体文本（仅动域列表与船面损失权重）。"""
    base = "".join(line for line in base_text.splitlines(keepends=True)
                   if not line.startswith("#"))
    text = replace_domains(base, "input_domains", input_domains)
    text = replace_domains(text, "output_domains", output_domains)
    if "hull" in output_domains:
        text = text.replace("  lambda_hull: 0.0\n", "  lambda_hull: 1.0\n", 1)
    return text


def header(name):
    return (f"# 船体融合机制消融 {name}：random_seed0 划分；"
            "按变体设置输入域、输出域和船面损失权重，其余训练超参不变\n")


def oformer_variants():
    """APP+OFormer 固定 GFLOPs 容量的两组派生配置：{文件名: {种子: 文本}}。"""
    base_text = OFORMER_BASE.read_text()
    base = yaml.safe_load(base_text)
    model = base["model"]
    # flow_with_hull 组已作为基础配置存在，语义必须与它一致，否则派生结果不可比。
    if (tuple(model["input_domains"]), tuple(model["output_domains"]),
            base["train"]["lambda_hull"]) != (("flow", "hull"), ("flow",), 0.0):
        raise RuntimeError(f"{OFORMER_BASE.name} 不是 flow_with_hull 口径")
    derived = {}
    for name in OFORMER_DERIVED:
        input_domains, output_domains = VARIANTS[name]
        text = header(name) + derive(base_text, input_domains, output_domains)
        stem = OFORMER_BASE.stem.replace("_random_seed0_isoflops", "")
        filename = f"{stem}_{name}_random_seed0_isoflops.yaml"
        derived[filename] = {0: text}
        # 训练种子变体只改 `train.seed` 与头注释，复用主实验的派生规则。
        for train_seed in make_split_configs.TRAIN_SEEDS:
            derived[filename][train_seed] = make_split_configs.derive_train_seed_from_text(
                text, OFORMER_BASE.name, train_seed)
    return derived


def main():
    base = "".join(line for line in BASE.read_text().splitlines(keepends=True)
                   if not line.startswith("#"))
    for name, (input_domains, output_domains) in VARIANTS.items():
        text = replace_domains(base, "input_domains", input_domains)
        text = replace_domains(text, "output_domains", output_domains)
        if "hull" in output_domains:
            text = text.replace("  lambda_hull: 0.0\n", "  lambda_hull: 1.0\n", 1)
        output = CONFIG_DIR / f"appsolver_v2_{name}_random_seed0.yaml"
        output.write_text(header(name) + text)
        print(output.name)

    for filename, seed_texts in oformer_variants().items():
        for train_seed, text in seed_texts.items():
            output = OFORMER_DIR / filename
            if train_seed:
                output = OFORMER_DIR / make_split_configs.train_seed_output_name(
                    OFORMER_DIR / filename, train_seed)
            output.write_text(text)
            print(output.name)


if __name__ == "__main__":
    main()
