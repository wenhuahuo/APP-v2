"""生成 `configs/splits_multihull.json`：三个评测划分的条目清单。

条目以 `dataset/shipbench/index.json` 中已组装的 normal_mesh 数据为准，
因此数据集尚未构建完成时也能对当前快照做初步实验。经典船型的设计 Fr
来自上游 `classic_fr_cases.json`（KCS 为 0.26）；融合船型的设计 Fr 按两母船
设计 Fr 与融合权重插值，无需依赖仍在变化的算例目录。

三个划分：
  loto_kvlcc2     留一个船型（kvlcc2）出验证集
  lofo_fr_design  留出设计 Fr（fr_scale = 1.0）那一档
  random_seed0    15% 随机留出（RandomState(0) 抽样）

用法:
    python3 scripts/train/make_splits.py --dataset-repo ~/Documents/OpenFoamWorks/APPSolver-v2-Dataset
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KCS_DESIGN_FR = 0.26
HOLDOUT_HULL = "kvlcc2"
RANDOM_FRACTION = 0.15
RANDOM_SEED = 0


def load_design_frs(dataset_repo):
    """返回经典及全部两两融合船型的设计 Fr。"""
    classic = json.loads((dataset_repo / "scripts/case_build/classic_fr_cases.json").read_text())
    design_frs = {case["hull"]: case["Fr"] for case in classic["cases"]
                  if case["fr_scale"] == 1.0}
    design_frs["kcs"] = KCS_DESIGN_FR
    for (hull_a, fr_a), (hull_b, fr_b) in combinations(sorted(design_frs.items()), 2):
        for weight in (0.25, 0.50, 0.75):
            hull = f"{hull_a}_{hull_b}_w{round(weight * 100):03d}"
            design_frs[hull] = (1 - weight) * fr_a + weight * fr_b
    return design_frs


def load_rows(dataset_repo, dataset_root):
    """返回当前已组装 normal_mesh 条目的 [(entry, hull, fr_scale)]。"""
    design_frs = load_design_frs(dataset_repo)
    index = json.loads((dataset_root / "index.json").read_text())
    rows = []
    for entry, meta in sorted(index["entries"].items()):
        if meta["mesh_level"] != "normal_mesh":
            continue
        hull = meta["hull"]
        if hull not in design_frs:
            raise ValueError(f"无法确定条目 {entry} 的设计 Fr（船型 {hull}）")
        rows.append((entry, hull, round(meta["Fr"] / design_frs[hull], 4)))
    return rows


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
    parser.add_argument("--dataset", default=str(ROOT / "dataset" / "shipbench"),
                        help="已组装数据集根目录；只使用其中 normal_mesh 条目")
    parser.add_argument("--output", default=str(ROOT / "configs/splits_multihull.json"))
    args = parser.parse_args()

    rows = load_rows(Path(args.dataset_repo).expanduser(), Path(args.dataset))
    splits = {
        "_comment": "三个评测划分。loto 留一个船型、lofo 留一档 Fr、random 随机 15%。"
                    f"val 是留出集，train 是其余条目；当前快照共 {len(rows)} 个 normal_mesh 条目。",
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
