"""按配置在若干查询密度上评测模型，写逐通道指标。

预测与真值都换算回物理量（波高 m、速度 m/s、液面与船面 Cp）；`stride` 依次给查询
点集密度，训练用过的档位是直接评估，其余是跨分辨率评估。配置里的
`eval.encode_stride`（或命令行 `--encode-stride`）把输入点云密度固定在某一档，查询
密度仍逐档变化，用于「固定输入编码、只改查询密度」的协议。指标写
`output/<配置文件名>/metrics_<checkpoint>.json`；固定输入编码的评测写
`metrics_<checkpoint>_encode<stride>.json`，与同步抽样结果分开保存；
`--splits` 限定只评测哪个划分（缺省两个都评测）。控制台只按划分与 `stride` 汇总
条目（mean±std），每个划分两张表：R²↑ 与 RMSE↓。

用法:
    python3 scripts/eval/evaluate_model.py --config configs/appsolver_v2/appsolver_v2_kcs_normal_wide.yaml
    python3 scripts/eval/evaluate_model.py --config configs/gino/gino_random_seed0.yaml \
        --encode-stride 1 --splits val
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from build import build_model  # noqa: E402
from evaluate import CHUNK  # noqa: E402
from normalize import Normalizer  # noqa: E402
from query_density import density_sweep  # noqa: E402
from report import metrics_filename, summary_lines  # noqa: E402

STRIDES = (1, 2, 4, 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--strides", type=int, nargs="+", default=list(STRIDES))
    parser.add_argument("--checkpoint", default="best.pt", help="best.pt 或 last.pt")
    parser.add_argument("--encode-stride", type=int, default=None,
                        help="固定输入点云密度（缺省用配置的 eval.encode_stride）")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"),
                        default=["train", "val"], help="只评测给定划分，缺省两个都评测")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output_dir = ROOT / "output" / config_path.stem
    device = torch.device(config["train"]["device"])
    checkpoint = torch.load(output_dir / args.checkpoint, weights_only=False, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = Normalizer.load(ROOT / config["stats"])

    train = config["train"]
    encode_stride = (args.encode_stride if args.encode_stride is not None
                     else config.get("eval", {}).get("encode_stride"))
    splits = {"train": train["entries"], "val": train["val_entries"]}
    splits = {split: entries for split, entries in splits.items() if split in args.splits}
    results = {
        "config": str(config_path),
        "model_type": config["model_type"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "train_strides": list(train["train_strides"]),
        "encode_stride": encode_stride,
        "tasks": splits,
        "strides": {split: density_sweep(model, normalizer, ROOT / train["dataset_root"],
                                         entries, args.strides, device, CHUNK, encode_stride)
                    for split, entries in splits.items()},
    }

    metrics_path = output_dir / metrics_filename(args.checkpoint, encode_stride)
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"{config['model_type']}，{args.checkpoint} epoch {checkpoint['epoch']}，"
          f"指标写入 {metrics_path}")
    for line in summary_lines(splits, results["strides"], args.strides):
        print(line)


if __name__ == "__main__":
    main()
