"""从随机划分主实验派生流场分词消融配置。

两个模型各三种流场分词方法，只改流场分词参数，船体分词、主干与查询解码逐项继承基础
配置（双域输入、仅流场输出、`lambda_hull=0.0`）：

- APPSolver-v2（`random_seed0` 划分）：基础配置
  `appsolver_v2_flow_with_hull_random_seed0.yaml`，派生均匀、学习、自适应三份；
- APP+OFormer（`random_seed0` 划分，固定 GFLOPs 容量）：基础配置
  `appsolver_oformer_random_seed0_isoflops.yaml` 本身就是自适应分词组（已有训练结果，
  生成器不重写它），只派生均匀与学习两组，各含训练种子 0/1/2。

    python3 scripts/train/make_partition_ablation_configs.py
"""

from pathlib import Path

import make_split_configs

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "appsolver_v2"
BASE = CONFIG_DIR / "appsolver_v2_flow_with_hull_random_seed0.yaml"
VARIANTS = {
    "uniform_partition": "  flow_tokenizer: uniform\n  flow_uniform_size: 0.1\n",
    "learned_partition": "  flow_tokenizer: learned\n  flow_learned_tokens: 512\n",
    "adaptive_partition": "  flow_tokenizer: adaptive\n",
}
OFORMER_DIR = ROOT / "configs" / "appsolver_oformer"
OFORMER_BASE = OFORMER_DIR / "appsolver_oformer_random_seed0_isoflops.yaml"
OFORMER_VARIANTS = {
    "uniform_partition": "  flow_tokenizer: uniform\n  flow_uniform_size: 0.1\n",
    "learned_partition": "  flow_tokenizer: learned\n  flow_learned_tokens: 512\n",
}


def strip_header(text):
    return "".join(line for line in text.splitlines(keepends=True)
                   if not line.startswith("#"))


def oformer_variants():
    """APP+OFormer 固定 GFLOPs 容量的两组派生配置：{文件名: (种子0文本, {种子: 文本})}。"""
    base_text = OFORMER_BASE.read_text()
    base = strip_header(base_text)
    if "flow_tokenizer" in base:
        raise RuntimeError(f"{OFORMER_BASE.name} 应使用默认的自适应分词")
    derived = {}
    for name, settings in OFORMER_VARIANTS.items():
        header = (f"# 划分 random_seed0（random_split，留出 RandomState(0) 抽 15%）："
                  f"train 190 / val 34；APP 分词消融 {name}（固定 GFLOPs 容量），"
                  f"其余超参与 {OFORMER_BASE.name} 一致\n")
        stem = OFORMER_BASE.stem.replace("_random_seed0_isoflops", "")
        seed_texts = {}
        for train_seed in (0, *make_split_configs.TRAIN_SEEDS):
            text = base.replace("model:\n", "model:\n" + settings, 1)
            text = header + text
            if train_seed == 0:
                seed_texts[0] = text
                continue
            # 训练种子变体只改 `train.seed` 与头注释，复用主实验的派生规则。
            seed_texts[train_seed] = make_split_configs.derive_train_seed_from_text(
                text, OFORMER_BASE.name, train_seed)
        derived[f"{stem}_{name}_random_seed0_isoflops.yaml"] = seed_texts
    return derived


def main():
    base = strip_header(BASE.read_text())
    for name, settings in VARIANTS.items():
        text = base.replace("model:\n", "model:\n" + settings, 1)
        header = (f"# APP 分词消融 {name}：random_seed0 划分；双域输入、仅流场监督；"
                  "仅改流场分词方法，其余训练超参不变\n")
        output = CONFIG_DIR / f"appsolver_v2_{name}_random_seed0.yaml"
        output.write_text(header + text)
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
