"""生成 `configs/splits_multihull.json`：三个评测划分的条目清单。

数据集仓库里两类算例合起来是 66 个条目 = 11 个船型 × 6 档 Fr：
  scripts/case_build/classic_fr_cases.json   10 个经典船型，`fr_scale` 直接给出
  kcs_*_normal_mesh_25s/case_manifest.json   KCS 的 6 档 Fr，`fr_scale = Fr / 0.26`

三个划分：
  loto_kvlcc2     留一个船型（kvlcc2）出验证集
  lofo_fr_design  留出设计 Fr（fr_scale = 1.0）那一档，11 个船型各一条
  random_seed0    15% 随机留出（RandomState(0) 抽样）

用法:
    python3 scripts/train/make_splits.py --dataset-repo ~/Documents/OpenFoamWorks/APPSolver-v2-Dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KCS_DESIGN_FR = 0.26
HOLDOUT_HULL = "kvlcc2"
RANDOM_FRACTION = 0.15
RANDOM_SEED = 0


def entry_id(hull, fr):
    return f"{hull}_fr{round(fr * 100):03d}_normal_mesh"


def load_rows(dataset_repo):
    """返回 [(entry, hull, fr_scale)]。"""
    rows = []
    classic = json.loads((dataset_repo / "scripts/case_build/classic_fr_cases.json").read_text())
    for case in classic["cases"]:
        rows.append((entry_id(case["hull"], case["Fr"]), case["hull"], case["fr_scale"]))
    for manifest_path in sorted(dataset_repo.glob("kcs*_normal_mesh_25s/case_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        fr = manifest["froude"]
        rows.append((entry_id("kcs", fr), "kcs", round(fr / KCS_DESIGN_FR, 4)))
    return sorted(rows)


def split_rows(rows):
    entries = [row[0] for row in rows]
    if len(set(entries)) != len(entries):
        raise ValueError("条目 id 有重复")
    holdout_hull = [e for e, hull, _ in rows if hull == HOLDOUT_HULL]
    holdout_fr = [e for e, _, scale in rows if scale == 1.0]
    n_random = round(RANDOM_FRACTION * len(rows))
    order = np.random.RandomState(RANDOM_SEED).permutation(len(rows))[:n_random]
    holdout_random = sorted(entries[i] for i in order)
    return {
        "loto_kvlcc2": {
            "protocol": "leave_one_hull_out",
            "holdout": HOLDOUT_HULL,
            "train": sorted(set(entries) - set(holdout_hull)),
            "val": sorted(holdout_hull),
        },
        "lofo_fr_design": {
            "protocol": "leave_one_fr_scale_out",
            "holdout": "fr_scale=1.0（设计 Fr）",
            "train": sorted(set(entries) - set(holdout_fr)),
            "val": sorted(holdout_fr),
        },
        "random_seed0": {
            "protocol": "random_split",
            "holdout": f"RandomState({RANDOM_SEED}) 抽 {RANDOM_FRACTION:.0%}",
            "train": sorted(set(entries) - set(holdout_random)),
            "val": holdout_random,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default="~/Documents/OpenFoamWorks/APPSolver-v2-Dataset")
    parser.add_argument("--output", default=str(ROOT / "configs/splits_multihull.json"))
    args = parser.parse_args()

    rows = load_rows(Path(args.dataset_repo).expanduser())
    splits = {
        "_comment": "三个评测划分。loto 留一个船型、lofo 留一档 Fr、random 随机 15%。"
                    "val 是留出集，train 是其余条目；66 个条目 = 11 船型 × 6 档 Fr。",
        **split_rows(rows),
    }
    Path(args.output).write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rows)} 个条目 -> {args.output}")
    for name, split in splits.items():
        if name.startswith("_"):
            continue
        print(f"  {name}: train {len(split['train'])} / val {len(split['val'])}")


if __name__ == "__main__":
    main()
