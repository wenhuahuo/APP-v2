"""从随机划分主实验派生三种流场分词消融配置。

    python3 scripts/train/make_partition_ablation_configs.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "appsolver_v2"
BASE = CONFIG_DIR / "appsolver_v2_flow_with_hull_random_seed0.yaml"
VARIANTS = {
    "uniform_partition": "  flow_tokenizer: uniform\n  flow_uniform_size: 0.1\n",
    "learned_partition": "  flow_tokenizer: learned\n  flow_learned_tokens: 512\n",
    "adaptive_partition": "  flow_tokenizer: adaptive\n",
}


def main():
    base = "".join(line for line in BASE.read_text().splitlines(keepends=True)
                   if not line.startswith("#"))
    for name, settings in VARIANTS.items():
        text = base.replace("model:\n", "model:\n" + settings, 1)
        header = (f"# APP 分词消融 {name}：random_seed0 划分；双域输入、仅流场监督；"
                  "仅改流场分词方法，其余训练超参不变\n")
        output = CONFIG_DIR / f"appsolver_v2_{name}_random_seed0.yaml"
        output.write_text(header + text)
        print(output.name)


if __name__ == "__main__":
    main()
