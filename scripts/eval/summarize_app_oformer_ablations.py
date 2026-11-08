"""汇总 APP+OFormer 三项全量消融：船体融合、流场分词、工况融合。

三项消融都跑在 `random_seed0` 全量划分（train 190 / val 34）、固定前向计算量容量上，
每种变体三个训练种子（100 epoch × 190 步 = 19 000 步、`train_strides=[1]`），指标取
`best.pt` 的 stride 1 全密度查询，按 `src/eval/aggregate.py` 的逐条目点数加权池化汇总，
三种子取 `ddof=1` 的均值与样本标准差。

    # 船体融合：flow_only / flow_with_hull / joint_flow_hull
    python3 scripts/eval/summarize_app_oformer_ablations.py --kind hull_fusion \
        --json output/app_oformer_hull_fusion_summary.json
    # 流场分词：adaptive / uniform / learned
    python3 scripts/eval/summarize_app_oformer_ablations.py --kind partition \
        --json output/app_oformer_partition_ablation_summary.json
    # 工况融合：film / none / adaln_zero / fourier_add / cond_token / cond_xattn
    python3 scripts/eval/summarize_app_oformer_ablations.py --kind conditioning \
        --json output/app_oformer_conditioning_ablation_summary.json

对照组（`joint_flow_hull` / `adaptive` / `fourier_add`）都是
`appsolver_oformer_random_seed0[_train_seed1/2]_isoflops`，与主实验的 APP+OFormer 是同一批
run，不重复训练。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ("eval",):
    sys.path.insert(0, str(ROOT / "src" / name))

import summary as runs  # noqa: E402

SPLIT = "random_seed0"
TRAIN_SEEDS = (0, 1, 2)
# (kind, 报告名, 配置名前缀, 是否对照)；对照组的 run 名与主实验 APP+OFormer 相同
VARIANTS = {
    "hull_fusion": (
        ("flow_only", "appsolver_oformer_flow_only", False,
         "只输入自由液面点云，输出流场"),
        ("flow_with_hull", "appsolver_oformer_flow_with_hull", False,
         "输入自由液面 + 船体湿表面，只输出流场"),
        ("joint_flow_hull", "appsolver_oformer", True,
         "输入双域，同时输出流场与船面 Cp，lambda_hull = 1.0（默认口径）"),
    ),
    "partition": (
        ("adaptive", "appsolver_oformer", True,
         "按到船体水线距离分档的自适应 Patch 分词"),
        ("uniform", "appsolver_oformer_uniform_partition", False,
         "物理边长统一的均匀 Patch 分词"),
        ("learned", "appsolver_oformer_learned_partition", False,
         "学习得到的 512 个潜 token"),
    ),
    "conditioning": (
        ("film", "appsolver_oformer_conditioning_film", False, "逐层 FiLM"),
        ("none", "appsolver_oformer_conditioning_none", False, "不输入工况"),
        ("adaln_zero", "appsolver_oformer_conditioning_adaln_zero", False, "adaLN-Zero 调制"),
        ("fourier_add", "appsolver_oformer", True,
         "Fourier 加性编码（默认口径）"),
        ("cond_token", "appsolver_oformer_conditioning_cond_token", False, "条件 token 拼接"),
        ("cond_xattn", "appsolver_oformer_conditioning_cond_xattn", False,
         "条件交叉注意力"),
    ),
}
TITLES = {
    "hull_fusion": "船体融合机制消融",
    "partition": "流场分词消融",
    "conditioning": "工况融合消融",
}


def run_name(prefix, train_seed):
    suffix = "" if train_seed == 0 else f"_train_seed{train_seed}"
    return f"{prefix}_{SPLIT}{suffix}_isoflops"


def variant_rows(root, kind, table, n_train):
    definitions = VARIANTS[kind]
    reference = next(name for name, _, is_reference, _ in definitions if is_reference)
    rows = runs.model_rows(
        root,
        [name for name, _, _, _ in definitions],
        lambda label: [run_name(prefix, seed)
                       for name, prefix, _, _ in definitions if name == label
                       for seed in TRAIN_SEEDS],
        table, n_train, reference_label=reference)
    for row, (_, prefix, _, description) in zip(rows, definitions):
        row["variant"] = row["label"]
        row["config_prefix"] = prefix
        row["description"] = description
        row["is_reference"] = row["label"] == reference
    return rows, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--json", required=True, help="把汇总写进这个 json")
    parser.add_argument("--root", default=str(ROOT), help="仓库根目录")
    args = parser.parse_args()

    root = Path(args.root)
    table = runs.load_complexity(root)
    split = json.loads((root / "configs" / "splits_multihull.json").read_text())[SPLIT]
    n_train = split["counts"]["train"]
    rows, reference = variant_rows(root, args.kind, table, n_train)
    summary = {
        "ablation": args.kind,
        "title": TITLES[args.kind],
        "split": SPLIT,
        "protocol": split["protocol"],
        "holdout": split["holdout"],
        "train_entries": n_train,
        "val_entries": split["counts"]["val"],
        "n_points": rows[0]["n_points"],
        "train_seeds": list(TRAIN_SEEDS),
        "epochs": runs.TRAIN_STEPS_PER_ENTRY,
        "train_steps": n_train * runs.TRAIN_STEPS_PER_ENTRY,
        "reference_variant": reference,
        "run_count": len(VARIANTS[args.kind]) * len(TRAIN_SEEDS),
        "variants": rows,
        **runs.baseline_metadata(root),
    }
    path = root / args.json
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"APP+OFormer {TITLES[args.kind]}（{SPLIT} val，stride 1，n=3 种子，"
          f"{rows[0]['n_points']} 点）")
    print(f"{'变体':16s}{'参数(M)':>9s}{'GFLOPs':>8s}{'token':>7s}"
          f"{'ηRMSE(mm)':>19s}{'整体RMSE%':>19s}{'整体R²':>21s}{'ΔR²':>10s}")
    for row in rows:
        change = row.get("contrast", {}).get("overall_r2_change")
        print(f"{row['variant']:16s}{row['parameters'] / 1e6:9.3f}"
              f"{('未标定' if row['gflops'] is None else format(row['gflops'], '.2f')):>9s}"
              f"{row['tokens_stride1']:7d}"
              f"{row['eta_rmse_mm']['mean']:10.3f}±{row['eta_rmse_mm']['std']:.3f}"
              f"{row['overall_rmse']['mean']:10.2f}±{row['overall_rmse']['std']:.2f}"
              f"{row['overall_r2']['mean']:+11.4f}±{row['overall_r2']['std']:.4f}"
              f"{('—' if change is None else format(change, '+.4f')):>10s}")
        if "hull" in row:
            print(f"{'':16s}船面 Cp：RMSE {row['hull']['cp_rmse']['mean']:.4f}"
                  f"±{row['hull']['cp_rmse']['std']:.4f}，"
                  f"R² {row['hull']['cp_r2']['mean']:+.4f}"
                  f"±{row['hull']['cp_r2']['std']:.4f}"
                  f"（{row['hull']['n_points']} 个船面点）")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
