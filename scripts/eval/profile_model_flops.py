"""统计各模型的前向计算量（GFLOPs）与参数量，用同一份输入做对照。

协议：batch size 1，单条目 `kcs_fr026_normal_mesh`，stride 1（全部格点做查询点），
查询点为全部自由液面格点 + 整条湿表面，与评测里的前向形状一致。FLOPs 由
`torch.utils.flop_counter.FlopCounterMode` 统计（matmul / bmm / conv 类算子，
不含 softmax、LayerNorm 等逐元素算子），单位 GFLOPs。

用法:
    python3 scripts/eval/profile_model_flops.py
    python3 scripts/eval/profile_model_flops.py --configs configs/appsolver_v2_kcs_fr_sweep.yaml
"""

import argparse
import json
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
from normalize import Normalizer  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402
from tokenizer import flow_tokens  # noqa: E402

DEFAULT_CONFIGS = (
    "configs/appsolver_v2_kcs_fr_sweep.yaml",
    "configs/oformer_kcs_fr_sweep.yaml",
    "configs/gnot_kcs_fr_sweep.yaml",
    "configs/upt_kcs_fr_sweep.yaml",
    "configs/transolver_kcs_fr_sweep.yaml",
    "configs/appsolver_oformer_kcs_fr_sweep.yaml",
    "configs/appsolver_v1_kcs_fr_sweep.yaml",
)


def profile(config_path, dataset, entry, device):
    """返回 (参数量, 前向 GFLOPs, APP token 数)。"""
    config = yaml.safe_load(Path(config_path).read_text())
    # 用 train 模式测量：eval 模式下 PyTorch 会把 nn.TransformerEncoder 融合成单个算子
    # （`_transformer_encoder_layer_fwd`），统计结果漏掉九成注意力，APPSolver-v1 因此
    # 从 65 G 掉到 4.9 G。dropout 全为 0，两种模式的前向路径一致。
    model = build_model(config).to(device).train()
    sample = ShipDataset(dataset, [entry], stride=1)[0]
    batch = {key: value.to(device) for key, value in collate_samples([sample]).items()}
    queries = torch.column_stack([batch["flow_coords"].reshape(-1, 2),
                                  batch["flow_sdf"].reshape(-1)])[None]

    # FlopCounterMode 依赖 autograd 的 grad_fn，因此不能包在 no_grad 里
    counter = FlopCounterMode(display=False)
    with counter:
        output = model(batch, queries)
    parameters = sum(p.numel() for p in model.parameters())
    total = counter.get_total_flops() / 1e9
    del output
    return parameters, total


def app_tokens(config_path, dataset, entry):
    """APP 类模型的 patch token 数（两域合计）；非 APP 模型返回 None。"""
    config = yaml.safe_load(Path(config_path).read_text())
    if "flow_size_tiers" not in config["model"]:
        return None
    sample = ShipDataset(dataset, [entry], stride=1)[0]
    batch = collate_samples([sample])
    flow = flow_tokens(batch, 0, config["model"]["flow_size_tiers"],
                       config["model"].get("max_level", 12))
    return len(flow[0])


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
        tokens = app_tokens(config_path, args.dataset, args.entry)
        rows.append({"config": stem, "model_type": model_type,
                     "parameters": parameters, "gflops": flops, "tokens": tokens})
        print(f"{stem:44s} {model_type:20s} {parameters / 1e6:8.3f} {flops:8.3f} "
              f"{'-' if tokens is None else tokens:>6}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        print(f"\n写入 {args.json}")


if __name__ == "__main__":
    main()
