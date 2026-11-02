"""统计各模型的前向计算量（GFLOPs）与参数量，用同一份输入做对照。

协议：batch size 1，单条目 `kcs_fr026_normal_mesh`，stride 1（全部格点同时作为
输入点与查询点），船面点云一并输入，与评测里的前向形状一致；所有模型都按
`(1, Q, 2)` 的查询坐标调用，避免基线各自猜测查询特征。FLOPs 由
`torch.utils.flop_counter.FlopCounterMode` 统计（matmul / bmm / conv 类算子，
不含 softmax、LayerNorm 等逐元素算子），单位 GFLOPs。

用法:
    python3 scripts/eval/profile_model_flops.py
    python3 scripts/eval/profile_model_flops.py --configs configs/appsolver_v2/appsolver_v2_multihull_normal.yaml
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from build import build_model  # noqa: E402
from collate import collate_samples  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402
from tokenizer import flow_tokens, hull_tokens  # noqa: E402

# 用主实验配置的模型结构做对照：主表 9 个模型的任务口径一致（双域输入、仅流场输出、
# 逐层 FiLM、lambda_hull=0.0），参数量与 GFLOPs 才可比。历史 KCS/消融配置保留原口径。
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


def profile(config_path, dataset, entry, device):
    """返回 (参数量, 前向 GFLOPs)。"""
    config = yaml.safe_load(Path(config_path).read_text())
    # 用 train 模式测量：eval 模式下 PyTorch 会把 nn.TransformerEncoder 融合成单个算子
    # （`_transformer_encoder_layer_fwd`），统计结果漏掉九成注意力，APPSolver-v1 因此
    # 从 65 G 掉到 4.9 G。dropout 全为 0，两种模式的前向路径一致。
    model = build_model(config).to(device).train()
    sample = ShipDataset(dataset, [entry], stride=1)[0]
    batch = {key: value.to(device) for key, value in collate_samples([sample]).items()}
    queries = batch["flow_coords"].reshape(-1, 2)[None]

    # FlopCounterMode 依赖 autograd 的 grad_fn，因此不能包在 no_grad 里
    counter = FlopCounterMode(display=False)
    with counter:
        output = model(batch, queries)
    parameters = sum(p.numel() for p in model.parameters())
    total = counter.get_total_flops() / 1e9
    del output
    return parameters, total


def model_tokens(config_path, dataset, entry):
    """统计各模型的瓶颈 token 数：APP 为 Patch token，GINO 为潜网格，CORAL/UPT 为 latent。"""
    config = yaml.safe_load(Path(config_path).read_text())
    model = config["model"]
    if "flow_size_tiers" not in model:
        if "latent_shape" in model:
            return math.prod(int(size) for size in model["latent_shape"])
        return model.get("latents")
    sample = ShipDataset(dataset, [entry], stride=1)[0]
    batch = collate_samples([sample])
    method = model.get("flow_tokenizer", "adaptive")
    if method == "learned":
        count = model.get("flow_learned_tokens", 512)
    else:
        tiers = (model["flow_size_tiers"] if method == "adaptive"
                 else [[None, model.get("flow_uniform_size", 0.1)]])
        count = len(flow_tokens(batch, 0, tiers, model.get("max_level", 12))[0])
    if "hull" in model.get("input_domains", ("flow", "hull")):
        count += len(hull_tokens(batch, 0, model["hull_size"], model.get("max_level", 12))[0])
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--dataset", default=str(ROOT / "dataset" / "shipbench"))
    parser.add_argument("--entry", default="kcs_fr026_normal_mesh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None, help="把结果写进这个 json")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    print(f"{'配置':44s} {'模型':20s} {'参数(M)':>8s} {'GFLOPs':>8s} {'token':>6s}")
    for config_path in args.configs:
        stem = Path(config_path).stem
        model_type = yaml.safe_load(Path(config_path).read_text())["model_type"]
        parameters, flops = profile(config_path, args.dataset, args.entry, device)
        tokens = model_tokens(config_path, args.dataset, args.entry)
        rows.append({"config": stem, "model_type": model_type,
                     "parameters": parameters, "gflops": flops, "tokens": tokens})
        print(f"{stem:44s} {model_type:20s} {parameters / 1e6:8.3f} {flops:8.3f} "
              f"{'-' if tokens is None else tokens:>6}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        print(f"\n写入 {args.json}")


if __name__ == "__main__":
    main()
