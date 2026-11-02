"""从随机划分默认配置派生三种船体融合机制消融配置。

    python3 scripts/train/make_framework_ablation_configs.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "appsolver_v2"
BASE = CONFIG_DIR / "appsolver_v2_flow_with_hull_random_seed0.yaml"
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


def main():
    base = "".join(line for line in BASE.read_text().splitlines(keepends=True)
                   if not line.startswith("#"))
    for name, (input_domains, output_domains) in VARIANTS.items():
        text = replace_domains(base, "input_domains", input_domains)
        text = replace_domains(text, "output_domains", output_domains)
        if "hull" in output_domains:
            text = text.replace("  lambda_hull: 0.0\n", "  lambda_hull: 1.0\n", 1)
        header = (f"# 船体融合机制消融 {name}：random_seed0 划分；"
                  "按变体设置输入域、输出域和船面损失权重，其余训练超参不变\n")
        output = CONFIG_DIR / f"appsolver_v2_{name}_random_seed0.yaml"
        output.write_text(header + text)
        print(output.name)


if __name__ == "__main__":
    main()
