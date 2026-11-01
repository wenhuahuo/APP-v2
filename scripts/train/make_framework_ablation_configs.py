"""从随机划分主实验派生四种框架消融配置，只改预测域与 SDF 输入特征。

    python3 scripts/train/make_framework_ablation_configs.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
BASE = CONFIG_DIR / "appsolver_v2_random_seed0.yaml"
VARIANTS = {
    "flow_only": (("flow",), False),
    "hull_only": (("hull",), False),
    "joint_no_sdf": (("flow", "hull"), False),
    "joint_sdf": (("flow", "hull"), True),
}


def main():
    base = BASE.read_text()
    for name, (domains, flow_sdf) in VARIANTS.items():
        domain_lines = "".join(f"  - {domain}\n" for domain in domains)
        settings = f"  domains:\n{domain_lines}  flow_sdf: {str(flow_sdf).lower()}\n"
        text = base.replace("model:\n", "model:\n" + settings, 1)
        header = (f"# 框架消融 {name}：random_seed0 划分；仅改预测域与 SDF 输入特征，"
                  "自适应 patch 规则及训练超参不变\n")
        output = CONFIG_DIR / f"appsolver_v2_{name}_random_seed0.yaml"
        output.write_text(header + text)
        print(output.name)


if __name__ == "__main__":
    main()
