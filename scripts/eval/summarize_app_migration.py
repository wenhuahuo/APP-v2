"""汇总 APP 分词前端迁移：APP+Transolver 与 APP+UPT 两组对照（各三种子）。

迁移配置由 `scripts/train/make_app_migration_configs.py` 从主实验固定计算量容量的基线
配置派生：接收方主干的结构与宽度（Transolver `d_model 176`、UPT `d_model 224`）逐项取自
`<模型>_random_seed0_isoflops.yaml`，只把点云/latent 前端换成 APP 双域 Patch 分词。
因此两组对照的原生一侧就是主实验的同样三种子 run，不重复训练。

成本口径：参数量取各 run 的 `parameters` 字段；stride 1 主干 token 数与 GFLOPs 取
`scripts/eval/model_complexity.json`（标定值见 `docs/isoflops_model_complexity.md`）。
APP+Transolver 与 APP+UPT 的 GFLOPs 本轮没有重标定，汇总里记为 `null` 并给出
`gflops_status`；token 压缩倍率 = 原生主干 token / APP 主干 token。

    python3 scripts/eval/summarize_app_migration.py \
        --json output/app_migration_summary.json
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
# (原生模型, 迁移模型, 原生主干宽度, 迁移说明)
PAIRS = (
    ("transolver", "appsolver_transolver", 176,
     "APP Patch token 作为 Transolver 物理切片注意力输入，slice 主干之后接坐标查询解码"),
    ("upt", "appsolver_upt", 224,
     "APP Patch token 作为 UPT perceiver 编码器输入，latent 与原生坐标查询解码不变"),
)
LABELS = {"transolver": "Transolver", "appsolver_transolver": "APP+Transolver",
          "upt": "UPT", "appsolver_upt": "APP+UPT"}


def native_run(model, train_seed):
    """原生一侧沿用主实验 run 名（`_isoflops` 后缀）。"""
    suffix = "" if train_seed == 0 else f"_train_seed{train_seed}"
    return f"{model}_{SPLIT}{suffix}_isoflops"


def app_run(model, train_seed):
    """迁移 run 的目录名没有 `_isoflops` 后缀。"""
    suffix = "" if train_seed == 0 else f"_train_seed{train_seed}"
    return f"{model}_{SPLIT}{suffix}"


def pair_rows(root, table, n_train):
    rows = {}
    for native, app, width, note in PAIRS:
        native_row = runs.model_row(
            root, LABELS[native],
            [native_run(native, seed) for seed in TRAIN_SEEDS], table, n_train)
        app_row = runs.model_row(
            root, LABELS[app],
            [app_run(app, seed) for seed in TRAIN_SEEDS], table, n_train)
        for row, model, width_used, description in (
                (native_row, native, width, "原生点云 / latent 前端"),
                (app_row, app, width, note)):
            row["model"] = model
            row["backbone_d_model"] = width_used
            row["frontend"] = description
            rows[model] = row
    return rows


def migration_row(native, app):
    """迁移前后对比：指标变化、token 压缩倍率与参数量变化。"""
    return {
        "native": native["model"],
        "app": app["model"],
        "native_label": native["label"],
        "app_label": app["label"],
        "overall_r2_change": app["overall_r2"]["mean"] - native["overall_r2"]["mean"],
        "overall_rmse_change": app["overall_rmse"]["mean"] - native["overall_rmse"]["mean"],
        "eta_rmse_change_mm": app["eta_rmse_mm"]["mean"] - native["eta_rmse_mm"]["mean"],
        "token_compression": native["tokens_stride1"] / app["tokens_stride1"],
        "parameters_change": app["parameters"] - native["parameters"],
        "gflops_change": (None if app["gflops"] is None or native["gflops"] is None
                          else app["gflops"] - native["gflops"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="output/app_migration_summary.json",
                        help="把汇总写进这个 json")
    parser.add_argument("--root", default=str(ROOT), help="仓库根目录")
    args = parser.parse_args()

    root = Path(args.root)
    table = runs.load_complexity(root)
    split = json.loads((root / "configs" / "splits_multihull.json").read_text())[SPLIT]
    rows = pair_rows(root, table, split["counts"]["train"])
    models = [rows[model] for native, app, _, _ in PAIRS for model in (native, app)]
    summary = {
        "experiment": "app_tokenizer_migration",
        "split": SPLIT,
        "protocol": split["protocol"],
        "holdout": split["holdout"],
        "train_entries": split["counts"]["train"],
        "val_entries": split["counts"]["val"],
        "n_points": models[0]["n_points"],
        "train_seeds": list(TRAIN_SEEDS),
        "epochs": runs.TRAIN_STEPS_PER_ENTRY,
        "train_steps": split["counts"]["train"] * runs.TRAIN_STEPS_PER_ENTRY,
        "run_count": len(PAIRS) * 2 * len(TRAIN_SEEDS),
        "pairs": [{"native": native, "app": app, "backbone_d_model": width,
                   "note": note} for native, app, width, note in PAIRS],
        "models": models,
        "migration": [migration_row(rows[native], rows[app]) for native, app, _, _ in PAIRS],
        "token_compression_rule": "原生主干 token / APP 主干 token；< 1 表示 APP 前端的 "
                                  "token 反而更多",
        **runs.baseline_metadata(root),
    }
    path = root / args.json
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"APP 分词迁移（{SPLIT} val，stride 1，n=3 种子，{models[0]['n_points']} 点）")
    print(f"{'模型':16s}{'d_model':>8s}{'参数(M)':>9s}{'GFLOPs':>8s}{'token':>8s}"
          f"{'ηRMSE(mm)':>19s}{'整体RMSE%':>19s}{'整体R²':>21s}")
    for row in models:
        print(f"{row['label']:16s}{row['backbone_d_model']:8d}{row['parameters'] / 1e6:9.3f}"
              f"{('未标定' if row['gflops'] is None else format(row['gflops'], '.2f')):>9s}"
              f"{row['tokens_stride1']:8d}"
              f"{row['eta_rmse_mm']['mean']:10.3f}±{row['eta_rmse_mm']['std']:.3f}"
              f"{row['overall_rmse']['mean']:10.2f}±{row['overall_rmse']['std']:.2f}"
              f"{row['overall_r2']['mean']:+11.4f}±{row['overall_r2']['std']:.4f}")
    print()
    for row in summary["migration"]:
        print(f"{row['native_label']} → {row['app_label']}："
              f"ΔR² {row['overall_r2_change']:+.4f}，"
              f"Δη {row['eta_rmse_change_mm']:+.3f} mm，"
              f"token 压缩 {row['token_compression']:.2f}×")
    print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
