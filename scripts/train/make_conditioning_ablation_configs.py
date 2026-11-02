"""生成工况融合消融配置与纯经典船型留出 Fr 划分。

    python3 scripts/train/make_conditioning_ablation_configs.py

验证集为 11 个经典船型的设计 Fr；训练集使用当前快照的其余条目（含全部融合船型）。
除数据划分、归一化统计量和 conditioning 外，模型与训练超参沿用 random_seed0。

训练/验证条目数随 `configs/splits_multihull.json` 的全量快照变化，不再写死条目数。
仓库中已提交的六份配置来自 207 条快照的历史结果，重新运行本脚本会用当前快照覆盖它们。
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "appsolver_v2"
BASE = CONFIG_DIR / "appsolver_v2_flow_with_hull_random_seed0.yaml"
SPLITS = ROOT / "configs" / "splits_multihull.json"
METHODS = ("none", "film", "adaln_zero", "fourier_add", "cond_token", "cond_xattn")
# 经典船型个数是数据契约（`AGENT.md`：11 个经典船型），不随快照条目数变化。
CLASSIC_HULL_COUNT = 11


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


def main():
    train, val = lofo_fr_classic_split()
    base = "".join(line for line in BASE.read_text().splitlines(keepends=True)
                   if not line.startswith("#"))
    base = base.replace("stats_random_seed0.json", "stats_lofo_fr_classic.json", 1)
    base = bullet_block(base, "entries", train)
    base = bullet_block(base, "val_entries", val)
    base = re.sub(r"^  conditioning: .*\n", "", base, count=1, flags=re.M)

    for method in METHODS:
        text = base.replace("model:\n", f"model:\n  conditioning: {method}\n", 1)
        header = (f"# 工况融合消融 {method}：lofo_fr_classic，train {len(train)} / "
                  f"val {len(val)}；双域输入、仅流场监督；其余超参与主实验一致\n")
        output = CONFIG_DIR / f"appsolver_v2_conditioning_{method}_lofo_fr_classic.yaml"
        output.write_text(header + text)
        print(output.name)


if __name__ == "__main__":
    main()
