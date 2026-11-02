"""统计各模型的前向计算量（GFLOPs）、参数量与主干 token 数，用同一份输入做对照。

协议：batch size 1，单条目 `kcs_fr026_normal_mesh`，stride 1（全部自由液面格点同时
作为输入点与查询点），船面点云一并输入，与评测里的前向形状一致；所有模型都按
`(1, Q, 2)` 的查询坐标调用，避免基线各自猜测查询特征。FLOPs 由
`torch.utils.flop_counter.FlopCounterMode` 统计（matmul / bmm / conv 类算子，
不含 softmax、LayerNorm 等逐元素算子），CPU 侧融合注意力由 `src/eval/flops.py`
补齐，单位 GFLOPs。

主干 token 数按模型的主算子序列长度统计，口径见 `token_count`。

用法:
    python3 scripts/eval/profile_model_flops.py
    python3 scripts/eval/profile_model_flops.py --configs configs/appsolver_v2/appsolver_v2_multihull_normal.yaml
    python3 scripts/eval/profile_model_flops.py --target-gflops 214.270
"""

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from app_v1 import point_patches  # noqa: E402
from build import build_model  # noqa: E402
from collate import collate_samples  # noqa: E402
from flops import forward_flops  # noqa: E402
from point_cloud import joint_cloud  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402
from tokenizer import flow_tokens, hull_tokens  # noqa: E402

# 用主实验配置的模型结构做对照：主表 9 个模型的任务口径一致（双域输入、仅流场输出、
# 逐层 FiLM、lambda_hull=0.0），参数量与 GFLOPs 才可比。历史 KCS/消融配置保留原口径。
# 等计算量配置（`*_multihull_isoflops.yaml`）由 `scripts/eval/calibrate_isoflops.py`
# 写出，统计口径与这里完全相同。
DEFAULT_CONFIGS = (
    "configs/appsolver_v2/appsolver_v2_multihull_normal.yaml",
    "configs/oformer/oformer_multihull_normal.yaml",
    "configs/gnot/gnot_multihull_normal.yaml",
    "configs/upt/upt_multihull_normal.yaml",
    "configs/transolver/transolver_multihull_normal.yaml",
    "configs/appsolver_oformer/appsolver_oformer_multihull_normal.yaml",
    "configs/appsolver_v1/appsolver_v1_multihull_normal.yaml",
    "configs/gino/gino_multihull_normal.yaml",
    "configs/coral/coral_multihull_normal.yaml",
)
POINT_TOKEN_MODELS = ("gnot", "oformer", "transolver")


def load_config(config_path):
    return yaml.safe_load(Path(config_path).read_text())


def batch_of(dataset, entry, device, stride=1):
    sample = ShipDataset(dataset, [entry], stride=stride)[0]
    return {key: value.to(device) for key, value in collate_samples([sample]).items()}


def profile(config_path, dataset, entry, device, repeat=1):
    """返回 (参数量, 前向 GFLOPs, 重复测量标准差)。"""
    config = load_config(config_path)
    batch = batch_of(dataset, entry, device)
    queries = batch["flow_coords"].reshape(-1, 2)[None]
    measurements, parameters = [], None
    for _ in range(repeat):
        # 用 train 模式测量：eval 模式下 `nn.TransformerEncoderLayer` 会走融合内核，算子
        # 不可分解。dropout 全为 0，两种模式的前向路径一致。每轮重复重建模型，避免两次
        # 前向的激活同时驻留。
        model = build_model(config).to(device).train()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        measurements.append(forward_flops(model, batch, queries)[1])
        del model
        gc.collect()
    return parameters, measurements[0], statistics.pstdev(measurements)


def token_count(config_path, dataset, entry, index=0):
    """主干 token 数与其口径（stride 1 的输入密度）。"""
    return token_count_at(config_path, dataset, entry, 1, index)


def token_count_at(config_path, dataset, entry, stride=1, index=0):
    """给定输入点云密度下的主干 token 数与其口径，输入由 `ShipDataset` 构造。"""
    config = load_config(config_path)
    return tokens_from_batch(config, batch_of(dataset, entry, torch.device("cpu"), stride), index)


def tokens_from_batch(config, batch, index=0):
    """主干 token 数与其口径。

    - `patch`: APP 分词器的 patch token（自适应/均匀/学习分块），双域输入时含船体 UV
      域的 patch；APPSolver-v1 的判据是 patch 内点数，token 数随输入密度变化；
    - `point`: 逐点算子把每个输入点当一个 token（GNOT、OFormer、Transolver 的点云主干）；
    - `latent`: perceiver 编码器的固定 latent 数（UPT、CORAL 的编码瓶颈）；
    - `latent_grid`: GINO 的潜规则网格格点数（GNO 积分到的网格，FNO 在其上卷积）。
    """
    model = config["model"]
    model_type = config["model_type"]
    domains = tuple(model.get("input_domains", ("flow", "hull")))
    flow_points = int(batch["flow_coords"].shape[1] * batch["flow_coords"].shape[2])
    hull_points = int(batch["hull_mask"][index].sum()) if "hull_mask" in batch else 0

    if "flow_size_tiers" in model:
        method = model.get("flow_tokenizer", "adaptive")
        if method == "learned":
            count = int(model.get("flow_learned_tokens", 512))
        else:
            tiers = (model["flow_size_tiers"] if method == "adaptive"
                     else [[None, model.get("flow_uniform_size", 0.1)]])
            count = len(flow_tokens(batch, index, tiers, model.get("max_level", 12))[0])
        if "hull" in domains:
            count += len(hull_tokens(batch, index, model["hull_size"],
                                     model.get("max_level", 12))[0])
        return count, "patch"
    if model_type == "appsolver_v1":
        coords = joint_cloud(batch, index)[2].numpy()
        patches = point_patches(coords, [True] * len(coords), int(model["patch_size"]))[0]
        return len(patches), "patch"
    if "latent_shape" in model:
        return math.prod(int(size) for size in model["latent_shape"]), "latent_grid"
    if "latents" in model:
        return int(model["latents"]), "latent"
    if model_type in POINT_TOKEN_MODELS:
        return flow_points + hull_points, "point"
    raise ValueError(f"{model_type} 的 token 口径未定义")


def training_tokens(config_path, dataset, entry, steps=2400):
    """一个训练协议（100 epoch × 48 step）下的主干 token 总量。

    `train_strides=[1,2]` 按步交替，`steps` 为每档步数（2400 步 stride 1 + 2400 步
    stride 2）。APP 系列与逐点模型的 token 数随输入密度变化，latent / 潜网格模型的
    编码 token 数与输入密度无关。
    """
    fine = token_count_at(config_path, dataset, entry, 1)[0]
    coarse = token_count_at(config_path, dataset, entry, 2)[0]
    return steps * (fine + coarse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--dataset", default=str(ROOT / "dataset" / "shipbench"))
    parser.add_argument("--entry", default="kcs_fr026_normal_mesh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeat", type=int, default=1,
                        help="重复统计次数，用于确认 GFLOPs 计数确定性；每次重复重建模型，"
                             "大输入下单次前向的激活已占数 GB，重复会线性累加峰值内存")
    parser.add_argument("--target-gflops", type=float, default=None,
                        help="给出后打印每个模型相对该目标的前向计算量误差")
    parser.add_argument("--json", default=None, help="把结果写进这个 json")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    header = (f"{'配置':44s} {'模型':20s} {'参数(M)':>8s} {'GFLOPs':>8s} "
              f"{'误差%':>7s} {'token':>7s} {'口径':>12s} {'训练token(M)':>12s}")
    print(header)
    for config_path in args.configs:
        stem = Path(config_path).stem
        config = load_config(config_path)
        model_type = config["model_type"]
        parameters, flops, deviation = profile(config_path, args.dataset, args.entry, device,
                                               args.repeat)
        tokens, kind = token_count(config_path, args.dataset, args.entry)
        train_tokens = training_tokens(config_path, args.dataset, args.entry)
        error = (None if args.target_gflops is None
                 else 100.0 * (flops - args.target_gflops) / args.target_gflops)
        gc.collect()
        rows.append({"config": stem, "model_type": model_type, "parameters": parameters,
                     "gflops": flops, "gflops_std": deviation, "tokens": tokens,
                     "token_kind": kind, "training_tokens": train_tokens,
                     "gflops_error_percent": error})
        print(f"{stem:44s} {model_type:20s} {parameters / 1e6:8.3f} {flops:8.3f} "
              f"{'' if error is None else f'{error:7.2f}'} {tokens:7d} {kind:>12s} "
              f"{train_tokens / 1e6:12.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        print(f"\n写入 {args.json}")


if __name__ == "__main__":
    main()
