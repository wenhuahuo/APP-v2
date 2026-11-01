"""从随机划分主实验派生三种船体融合机制消融配置。

    python3 scripts/train/make_framework_ablation_configs.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
BASE = CONFIG_DIR / "appsolver_v2_random_seed0.yaml"
VARIANTS = {
    "flow_only": (("flow",), ("flow",)),
    "flow_with_hull": (("flow", "hull"), ("flow",)),
    "joint_flow_hull": (("flow", "hull"), ("flow", "hull")),
}


def domain_settings(input_domains, output_domains):
    inputs = "".join(f"  - {domain}\n" for domain in input_domains)
    outputs = "".join(f"  - {domain}\n" for domain in output_domains)
    return f"  input_domains:\n{inputs}  output_domains:\n{outputs}"


def main():
    base = BASE.read_text()
    for name, (input_domains, output_domains) in VARIANTS.items():
        settings = domain_settings(input_domains, output_domains)
        text = base.replace("model:\n", "model:\n" + settings, 1)
        header = (f"# 船体融合机制消融 {name}：random_seed0 划分；"
                  "仅改输入域和输出域，其余训练超参不变\n")
        output = CONFIG_DIR / f"appsolver_v2_{name}_random_seed0.yaml"
        output.write_text(header + text)
        print(output.name)


if __name__ == "__main__":
    main()
