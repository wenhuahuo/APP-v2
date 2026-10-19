"""按配置在若干查询密度上评测模型，写逐通道指标。

预测与真值都换算回物理量（波高 m、速度 m/s、液面与船面 Cp）；`stride` 依次给查询
点集密度，训练用过的档位是直接评估，其余是跨分辨率评估。指标写
`output/<配置文件名>/metrics.json`。

用法:
    python3 scripts/eval/evaluate_model.py --config configs/appsolver_v2_kcs_normal_wide.yaml
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

STRIDES = (1, 2, 4, 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--strides", type=int, nargs="+", default=list(STRIDES))
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output_dir = ROOT / "output" / config_path.stem
    device = torch.device(config["train"]["device"])
    checkpoint = torch.load(output_dir / "best.pt", weights_only=False, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = Normalizer.load(ROOT / config["stats"])

    train = config["train"]
    results = {
        "config": str(config_path),
        "model_type": config["model_type"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": checkpoint["epoch"],
        "train_strides": list(train["train_strides"]),
        "strides": density_sweep(model, normalizer, ROOT / train["dataset_root"],
                                 train["entries"], args.strides, device, CHUNK),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"{config['model_type']}，best epoch {checkpoint['epoch']}，"
          f"指标写入 {output_dir / 'metrics.json'}")
    for entry_id in train["entries"]:
        print(f"\n{entry_id}")
        header = "  stride  " + "  ".join(f"{channel:>8s}" for channel in ("eta", "u", "v", "w", "Cp"))
        print(header + "   hull Cp")
        for stride in args.strides:
            block = results["strides"][stride][entry_id]
            values = "  ".join(f"{r2:+8.4f}" for r2 in block["flow"]["r2"])
            print(f"  {stride:^6d}  {values}   {block['hull']['r2'][0]:+.4f}")
        print("  波高 RMSE (mm) 与船面 Cp RMSE：")
        for stride in args.strides:
            block = results["strides"][stride][entry_id]
            print(f"    stride {stride}: eta {block['flow']['rmse'][0] * 1000:6.3f} mm，"
                  f"hull Cp {block['hull']['rmse'][0]:.4f}，"
                  f"流场点数 {block['flow']['n_points']}")


if __name__ == "__main__":
    main()
