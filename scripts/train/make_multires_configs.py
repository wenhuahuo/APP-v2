"""从固定 GFLOPs 容量的主实验 `random_seed0` 配置派生多分辨率训练验证实验配置。

5 个模型（与留一船型泛化实验同批，按 `GENERALIZATION_MODELS` 顺序）× 训练种子 0/1/2
共 15 份配置。与基准配置的差别只有两处：

- `train.train_strides` 由 `[1]` 改为 `[1, 2]`：每个 epoch 遍历全部训练条目，
  每条条目在 1/2 两档里随机取一档查询密度（`src/train/trainer.py`）；
- `eval.strides` 由 `[1]` 改为 `[1, 2, 4, 8]`：训练后按同步抽样协议评测四个查询
  密度档位；拥有 `encode` / `decode` 接口的模型另做一次固定输入编码
  （`eval.encode_stride = 1`）的评测，由 `train_multires_experiment.sbatch` 用
  `--encode-stride 1` 触发，因此配置本身不写 `encode_stride`。

数据划分、统计量、容量、优化超参与训练轮数逐项沿用基准配置。

    python3 scripts/train/make_multires_configs.py
"""

from pathlib import Path

import make_split_configs

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
SPLIT = "random_seed0"
# 与留一船型泛化实验同一批模型（按 `GENERALIZATION_MODELS` 顺序）
MODELS = make_split_configs.GENERALIZATION_MODELS
TRAIN_SEEDS = (1, 2)
EVAL_STRIDES = (1, 2, 4, 8)
BASE_EVAL = "eval:\n  strides:\n  - 1\n"
MULTIRES_EVAL = "eval:\n  strides:\n" + "".join(f"  - {stride}\n" for stride in EVAL_STRIDES)


def base_path(model):
    return CONFIG_DIR / model / f"{model}_{SPLIT}_isoflops.yaml"


def output_path(model, train_seed=0):
    name = f"{model}_multires_{SPLIT}_isoflops.yaml"
    path = CONFIG_DIR / model / name
    if not train_seed:
        return path
    return path.parent / make_split_configs.train_seed_output_name(path, train_seed, SPLIT)


def derive(model):
    """返回多分辨率训练配置全文（纯函数，便于幂等校验）。"""
    base = base_path(model)
    lines = base.read_text().splitlines(keepends=True)
    if not lines or not lines[0].startswith("#"):
        raise RuntimeError(f"{base.name} 缺少头注释")
    body = "".join(lines[1:])
    if "  train_strides:\n  - 1\n" not in body or BASE_EVAL not in body:
        raise RuntimeError(f"{base.name} 不是默认的 stride 1 训练与评测口径")
    body = body.replace("  train_strides:\n  - 1\n", "  train_strides:\n  - 1\n  - 2\n", 1)
    body = body.replace(BASE_EVAL, MULTIRES_EVAL, 1)
    header = (f"# 多分辨率训练验证（划分 {SPLIT}）：每条训练条目在 stride 1/2 中随机取一档，"
              f"评测 stride 1/2/4/8；固定计算量（iso-FLOPs）容量，"
              f"其余超参与 {base.name} 一致\n")
    return header + body


def main():
    for model in MODELS:
        text = derive(model)
        for train_seed in (0, *TRAIN_SEEDS):
            path = output_path(model, train_seed)
            path.write_text(text if not train_seed
                            else make_split_configs.derive_train_seed_from_text(
                                text, base_path(model).name, train_seed, SPLIT))
            print(f"{path.name}: train_strides {[1, 2]}，评测 stride {list(EVAL_STRIDES)}")


if __name__ == "__main__":
    main()
