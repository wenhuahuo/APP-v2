"""从固定 GFLOPs 容量的主实验 `random_seed0` 基线配置派生 APP 分词前端迁移配置。

三个新模型（APP+Transolver、APP+UPT、APP+GNOT）各自复用对应基线配置的训练协议：
`train` 段（190/34 条目、统计量、100 epoch、`train_strides=[1]`、`lambda_hull`、种子）
逐行照抄，因此数据划分、训练预算与任务口径与主实验完全一致。模型段只保留基线的接收方
主干结构与宽度（`d_model` / 层数 / slice 数 / latent 数 / 专家数等），再接上 APP 双域
分词前端与坐标查询解码；分词参数（`flow_size_tiers` / `hull_size`）取
`appsolver_oformer_random_seed0_isoflops.yaml`，与主实验的 APP 分词口径相同。

    python3 scripts/train/make_app_migration_configs.py

每种容量口径只派生 `random_seed0` 与训练种子 1、2，共 3 模型 × 3 种子 = 9 份配置；
留一船型、留一设计 Fr 与等计算量口径不派生。本轮实验只训练 APP+Transolver 与
APP+UPT（`train_app_tokenizer_migration.sbatch`），APP+GNOT 保留配置不训练。
"""

from pathlib import Path

import yaml

import make_split_configs

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
SPLIT = "random_seed0"
TRAIN_SEEDS = (1, 2)
# 基线配置目录名 = 模型名前缀；新模型名沿用 `appsolver_` 前缀
BASELINES = ("transolver", "upt", "gnot")
# 接收方主干的结构与宽度：逐项从基线配置取，保证「主干容量不变」
STRUCT_KEYS = {
    "transolver": ("d_model", "nhead", "layers", "slices", "mlp_ratio", "dropout"),
    "upt": ("d_model", "nhead", "layers", "latents", "mlp_ratio", "dropout"),
    "gnot": ("d_model", "nhead", "layers", "mlp_ratio", "n_experts", "dropout"),
}
# 坐标查询解码器：与 APPSolver-v2 同一套组件与默认值（UPT 沿用其原生解码器，只取前端位置参数）
QUERY_KEYS = {
    "transolver": {"fourier_features": 16, "fourier_scale": 4.0, "query_basis": "siren",
                   "decoder_layers": 2},
    "upt": {"fourier_features": 16, "fourier_scale": 4.0},
    "gnot": {"fourier_features": 16, "fourier_scale": 4.0, "query_basis": "siren",
             "decoder_layers": 2},
}
# APP 分词参数取自主实验的 APP+OFormer 配置（与 APPSolver-v2 相同）
TOKENIZER_SOURCE = CONFIG_DIR / "appsolver_oformer" / f"appsolver_oformer_{SPLIT}_isoflops.yaml"
HEADERS = {
    "transolver": "APP Patch token 作为 Transolver 物理切片注意力输入，slice 主干之后接坐标查询解码",
    "upt": "APP Patch token 作为 UPT perceiver 编码器输入，latent 与原生坐标查询解码不变",
    "gnot": "APP Patch token 作为 GNOT 几何-物理异构注意力输入，主干之后接坐标查询解码",
}


def baseline_path(model):
    """接收方基准：主实验固定 GFLOPs 容量口径，主干宽度与层数逐项取自它。"""
    return CONFIG_DIR / model / f"{model}_{SPLIT}_isoflops.yaml"


def new_model(model):
    return f"appsolver_{model}"


def tokenizer_settings():
    """APP 分词参数：与主实验 APP 分词口径相同的 `flow_size_tiers` / `hull_size`。"""
    source = yaml.safe_load(TOKENIZER_SOURCE.read_text())["model"]
    return {"flow_size_tiers": source["flow_size_tiers"], "hull_size": source["hull_size"]}


def model_section(model):
    """新模型段：任务口径与分词参数 + 基线主干结构与宽度 + 坐标查询解码。"""
    baseline = yaml.safe_load(baseline_path(model).read_text())["model"]
    section = {"input_domains": baseline["input_domains"],
               "output_domains": baseline["output_domains"],
               "conditioning": baseline["conditioning"],
               **tokenizer_settings()}
    section.update({key: baseline[key] for key in STRUCT_KEYS[model]})
    section.update(QUERY_KEYS[model])
    return section


def derive(model):
    """返回新模型 `random_seed0` 配置全文（纯函数，便于幂等校验）。"""
    baseline = baseline_path(model)
    text = baseline.read_text()
    _, train = text.split("\ntrain:\n", 1)
    head = {"stats": f"dataset/shipbench/stats_{SPLIT}.json",
            "model_type": new_model(model), "model": model_section(model)}
    header = (f"# 划分 {SPLIT}（random_split，留出 RandomState(0) 抽 15%）："
              f"train 190 / val 34；{HEADERS[model]}（主干容量与 {baseline.name} 一致），"
              f"其余训练协议与 {baseline.name} 一致\n")
    return (header
            + yaml.safe_dump(head, sort_keys=False, default_flow_style=False, allow_unicode=True)
            + "train:\n" + train)


def output_path(model):
    return CONFIG_DIR / new_model(model) / f"{new_model(model)}_{SPLIT}.yaml"


def main():
    for model in BASELINES:
        path = output_path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(derive(model))
        print(f"{path.relative_to(ROOT)}: {new_model(model)}")
        for train_seed in TRAIN_SEEDS:
            derived = path.parent / make_split_configs.train_seed_output_name(path, train_seed)
            derived.write_text(make_split_configs.derive_train_seed(path, train_seed))
            print(f"{derived.relative_to(ROOT)}: train.seed {train_seed}，划分 {SPLIT}")


if __name__ == "__main__":
    main()
