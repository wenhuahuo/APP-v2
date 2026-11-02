"""生成 `configs/splits_multihull.json`：三个主实验划分的条目清单。

条目以 ShipBenchv2 全量快照中的 `normal_mesh` 条目为准。生成器只读元数据
（`index.json` 与各条目的 `meta.json`），不读任何数组，因此任何持有该快照元数据的
机器都能复现同一份划分：

    python3 scripts/train/make_splits.py --dataset /path/to/dataset/shipbench

条目筛选与排除依据（数据契约，见 `AGENT.md`）：

- 只收 `mesh_level == normal_mesh`；`sparse_mesh` 仅用于标签保真度检查，不进正式划分。
- 收录条目必须同时出现在 `index.json` 与磁盘目录中，且 `meta.json` 的
  `entry_id / hull / Fr / mesh_level` 与索引一致；不一致直接报错，不静默丢弃。
- 船型集合为「11 个经典船型」与「经典船型两两融合船型」。融合船型按
  `<A>_<B>_w<WWW>` 命名，`WWW` 为乙母船权重百分数；其设计 Fr 由两母船设计 Fr 加权
  插值得到，因此 158 条融合条目给出 11 个经典设计 Fr 的超定方程组（最小二乘残差为
  机器精度），划分不依赖外部数据集仓库的算例配置。

三个划分（`val` 是留出集，同时用于选择 checkpoint 与报告指标）：

    random_seed0    RandomState(0) 抽 15% 作留出集，其余训练
    loto_kvlcc2     整船型留出 kvlcc2 的全部 Fr 工况
    lofo_fr_design  整档留出设计 Fr（`fr_scale = 1.0`）：经典船型设计工况 +
                    全部融合船型工况（融合船型只有其设计 Fr 一个工况）

每个划分都校验 train/val 互不重叠、二者恰好覆盖快照全部条目，并校验留出语义。
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 11 个经典船型；融合船型由它们两两按 0.25/0.50/0.75 权重插值生成。
CLASSIC_HULLS = (
    "containership",
    "dtc",
    "frigate",
    "kcs",
    "kvlcc2",
    "npl_round_bilge",
    "npl_round_bilge_4a",
    "s175",
    "series60",
    "wigley",
    "workboat",
)
FUSION_WEIGHTS = (25, 50, 75)  # 乙母船权重百分数
DESIGN_FR_SCALE = 1.0
HOLDOUT_HULL = "kvlcc2"
RANDOM_FRACTION = 0.15
RANDOM_SEED = 0
# 融合 Fr 由设计 Fr 线性插值得到，元数据保留 4 位小数，容差取 1e-4 即可判定一致。
FR_TOL = 1e-4


def load_entries(dataset_root):
    """返回全量快照的 normal_mesh 条目 [(entry_id, hull, Fr)]，按 entry_id 排序。"""
    dataset_root = Path(dataset_root)
    index = json.loads((dataset_root / "index.json").read_text())
    rows = []
    for entry, meta in sorted(index["entries"].items()):
        if meta["mesh_level"] != "normal_mesh":
            continue
        meta_path = dataset_root / entry / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"条目 {entry} 在索引中但缺少 {meta_path}")
        entry_meta = json.loads(meta_path.read_text())
        for key in ("entry_id", "hull", "Fr", "mesh_level"):
            expected = entry if key == "entry_id" else meta[key]
            if entry_meta[key] != expected:
                raise ValueError(
                    f"条目 {entry} 的 meta.{key}={entry_meta[key]!r} 与索引 {expected!r} 不一致")
        rows.append((entry, meta["hull"], meta["Fr"]))
    if not rows:
        raise ValueError(f"{dataset_root} 中没有 normal_mesh 条目")
    return rows


def parse_fusion_hull(hull):
    """把融合船型名拆成 (母船甲, 母船乙, 乙权重)；非融合船型返回 None。"""
    for first in sorted(CLASSIC_HULLS, key=len, reverse=True):
        if not hull.startswith(f"{first}_"):
            continue
        rest = hull[len(first) + 1:]
        for second in sorted(CLASSIC_HULLS, key=len, reverse=True):
            if not rest.startswith(f"{second}_w"):
                continue
            tag = rest[len(second) + 3:]
            if tag.isdigit() and int(tag) in FUSION_WEIGHTS:
                if first == second:
                    raise ValueError(f"融合船型 {hull} 的两个母船相同")
                return first, second, int(tag) / 100
    return None


def recover_design_frs(rows):
    """由融合船型 Fr 的加权插值反解经典船型设计 Fr，并核对经典设计工况确实存在。"""
    fusion_triples = []
    for entry, hull, fr in rows:
        if hull in CLASSIC_HULLS:
            continue
        parsed = parse_fusion_hull(hull)
        if parsed is None:
            raise ValueError(f"无法解析融合船型名 {hull}（条目 {entry}）")
        fusion_triples.append((entry, *parsed, fr))
    if not fusion_triples:
        raise ValueError("快照中没有融合船型条目，无法反解设计 Fr")

    design = np.zeros(len(CLASSIC_HULLS))
    matrix = np.zeros((len(fusion_triples), len(CLASSIC_HULLS)))
    target = np.zeros(len(fusion_triples))
    for row, (_, first, second, weight, fr) in enumerate(fusion_triples):
        matrix[row, CLASSIC_HULLS.index(first)] = 1.0 - weight
        matrix[row, CLASSIC_HULLS.index(second)] = weight
        target[row] = fr
    if np.linalg.matrix_rank(matrix) != len(CLASSIC_HULLS):
        raise ValueError("融合船型方程不足以唯一反解 11 个经典设计 Fr")
    design, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = float(np.abs(matrix @ design - target).max())
    if residual > FR_TOL:
        raise ValueError(f"融合船型 Fr 与设计 Fr 插值不自洽（最大残差 {residual:.3e}）")

    # 用经典船型自身的设计工况把反解值钉到实际条目 Fr 上，避免浮点尾数进入划分依据。
    classic_frs = {hull: [] for hull in CLASSIC_HULLS}
    for entry, hull, fr in rows:
        if hull in CLASSIC_HULLS:
            classic_frs[hull].append((entry, fr))
    design_frs = {}
    for hull, solved in zip(CLASSIC_HULLS, design):
        matches = [fr for _, fr in classic_frs[hull] if abs(fr - solved) <= FR_TOL]
        if len(matches) != 1:
            raise ValueError(
                f"船型 {hull} 反解设计 Fr {solved:.6f} 没有唯一对应的实际条目"
                f"（候选 {sorted(f for _, f in classic_frs[hull])}）")
        design_frs[hull] = matches[0]
    return design_frs


def fr_scales(rows, design_frs):
    """返回 {entry: fr_scale}；融合船型只有设计 Fr 一个工况，故其档位为 1.0。"""
    scales = {}
    for entry, hull, fr in rows:
        if hull in CLASSIC_HULLS:
            scales[entry] = round(fr / design_frs[hull], 4)
        else:
            first, second, weight = parse_fusion_hull(hull)
            expected = ((1 - weight) * design_frs[first] + weight * design_frs[second])
            if abs(fr - expected) > FR_TOL:
                raise ValueError(
                    f"融合条目 {entry} 的 Fr={fr} 不是设计 Fr 插值 {expected:.4f}")
            scales[entry] = DESIGN_FR_SCALE
    return scales


def validate_split(name, rows, scales, split):
    """校验无泄漏、完整覆盖与各划分的留出语义。"""
    entries = {entry for entry, _, _ in rows}
    hull_of = {entry: hull for entry, hull, _ in rows}
    train, val = set(split["train"]), set(split["val"])
    if train & val:
        raise ValueError(f"{name}: train 与 val 重叠 {sorted(train & val)[:5]}")
    if train | val != entries:
        missing = sorted(entries - (train | val))
        extra = sorted((train | val) - entries)
        raise ValueError(f"{name}: 覆盖不完整，缺失 {missing[:5]}，多余 {extra[:5]}")
    if split["train"] != sorted(train) or split["val"] != sorted(val):
        raise ValueError(f"{name}: 条目清单未排序")
    if not val:
        raise ValueError(f"{name}: 留出集为空")

    if name == "lofo_fr_design":
        if {scales[e] for e in val} != {DESIGN_FR_SCALE}:
            raise ValueError(f"{name}: 留出集不是整档设计 Fr")
        if DESIGN_FR_SCALE in {scales[e] for e in train}:
            raise ValueError(f"{name}: 训练集混入了设计 Fr 条目")
        if len({hull_of[e] for e in val}) < len(CLASSIC_HULLS):
            raise ValueError(f"{name}: 留出集未覆盖全部经典船型")
    elif name == "loto_kvlcc2":
        if {hull_of[e] for e in val} != {HOLDOUT_HULL}:
            raise ValueError(f"{name}: 留出集不止 {HOLDOUT_HULL} 船型")
        if HOLDOUT_HULL in {hull_of[e] for e in train}:
            raise ValueError(f"{name}: 训练集混入了留出船型 {HOLDOUT_HULL}")
    elif name == "random_seed0":
        expected = round(RANDOM_FRACTION * len(rows))
        if len(val) != expected:
            raise ValueError(f"{name}: 留出 {len(val)} 条，期望 {expected} 条")


def split_rows(rows):
    """按固定规则切分；同一批元数据必然得到同一结果。"""
    rows = sorted(rows)
    entries = [entry for entry, _, _ in rows]
    if len(set(entries)) != len(entries):
        raise ValueError("条目 id 有重复")
    design_frs = recover_design_frs(rows)
    scales = fr_scales(rows, design_frs)

    holdout_hull = [entry for entry, hull, _ in rows if hull == HOLDOUT_HULL]
    holdout_fr = [entry for entry, _, _ in rows
                  if scales[entry] == DESIGN_FR_SCALE]
    n_random = round(RANDOM_FRACTION * len(rows))
    order = np.random.RandomState(RANDOM_SEED).permutation(len(rows))[:n_random]
    holdout_random = sorted(entries[index] for index in order)

    splits = {
        "loto_kvlcc2": {
            "protocol": "leave_one_hull_out",
            "holdout": HOLDOUT_HULL,
            "train": sorted(set(entries) - set(holdout_hull)),
            "val": sorted(holdout_hull),
        },
        "lofo_fr_design": {
            "protocol": "leave_one_fr_scale_out",
            "holdout": f"fr_scale={DESIGN_FR_SCALE}（设计 Fr）",
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
    for name, split in splits.items():
        validate_split(name, rows, scales, split)
        split["counts"] = {"train": len(split["train"]), "val": len(split["val"])}
    return splits, design_frs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "dataset" / "shipbench"),
                        help="ShipBenchv2 快照根目录；只读取 index.json 与各条目 meta.json")
    parser.add_argument("--output", default=str(ROOT / "configs/splits_multihull.json"))
    args = parser.parse_args()

    rows = load_entries(args.dataset)
    splits, design_frs = split_rows(rows)
    document = {
        "_comment": "三个主实验划分（val 为留出集，同时用于选择 checkpoint 与报告指标）。"
                    "loto 留一个船型、lofo 整档留出设计 Fr、random 随机 15%。"
                    f"当前快照共 {len(rows)} 个 normal_mesh 条目、"
                    f"{len({hull for _, hull, _ in rows})} 个船型；"
                    f"train/val 互不重叠且覆盖全部条目。",
        **splits,
    }
    Path(args.output).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rows)} 个条目 -> {args.output}")
    print("设计 Fr：" + "，".join(f"{hull} {fr:g}" for hull, fr in design_frs.items()))
    for name, split in splits.items():
        print(f"  {name}: train {len(split['train'])} / val {len(split['val'])}"
              f"（val 覆盖 {len({h for e, h, _ in rows if e in set(split['val'])})} 个船型）")


if __name__ == "__main__":
    main()
