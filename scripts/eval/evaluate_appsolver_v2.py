"""按配置评测 APPSolver-v2：完整标签格点与整条湿表面上的逐通道指标。

预测与真值都换算回物理量（波高 m、速度 m/s、液面与船面 Cp），指标写
`output/<配置文件名>/metrics.json`，同时打印成表。

用法:
    python3 scripts/eval/evaluate_appsolver_v2.py --config configs/appsolver_v2_kcs_normal_wide.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for name in ("datapipe", "app", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from appsolver_v2 import APPSolverV2  # noqa: E402
from collate import collate_samples  # noqa: E402
from metrics import (FLOW_CHANNELS, FLOW_UNITS, HULL_CHANNELS, HULL_UNITS,  # noqa: E402
                     channel_metrics)
from normalize import Normalizer  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402

CHUNK = 20000


def _physical_scale(scale):
    """无量纲化用的尺度：`eta/Lpp` 乘 Lpp，三个速度分量乘 U_inf，压力已是 Cp。"""
    lpp, u_inf = float(scale[0]), float(scale[1])
    return torch.tensor([lpp, u_inf, u_inf, u_inf, 1.0])


@torch.no_grad()
def evaluate_entry(model, normalizer, sample, device):
    batch = {key: value.to(device) for key, value in collate_samples([sample]).items()}
    mask = sample["flow_mask"]
    index = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    coords = sample["flow_coords"].reshape(-1, 2)
    sdf = sample["flow_sdf"].reshape(-1)
    prediction = []
    for start in range(0, len(index), CHUNK):
        part = index[start:start + CHUNK]
        queries = torch.column_stack([coords[part], sdf[part]])[None].to(device)
        output = model(batch, queries)
        prediction.append(output["flow"][0].cpu())
        hull_prediction = output["hull"][0][:, 0].cpu()
    prediction = torch.cat(prediction)

    # 预测在标准化空间，先还原到无量纲量；标签本身已是无量纲量，只乘尺度。
    scale = _physical_scale(sample["scale"])
    flow_prediction = (normalizer.flow_mean + normalizer.flow_std * prediction) * scale
    flow_target = sample["flow_target"].reshape(-1, 5)[index] * scale
    flow = channel_metrics(flow_prediction, flow_target, sample["flow_weight"].reshape(-1)[index])

    hull_prediction = normalizer.hull_mean + normalizer.hull_std * hull_prediction
    hull_target = sample["hull_pressure"]
    hull = channel_metrics(hull_prediction[:, None], hull_target[:, None], sample["hull_weight"])
    return {
        "flow": {"channels": list(FLOW_CHANNELS), "units": list(FLOW_UNITS), **flow,
                 "n_points": int(len(index))},
        "hull": {"channels": list(HULL_CHANNELS), "units": list(HULL_UNITS), **hull,
                 "n_points": int(len(hull_target))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output_dir = ROOT / "output" / config_path.stem
    checkpoint = torch.load(output_dir / "best.pt", weights_only=False)
    device = torch.device(config["train"]["device"])
    model = APPSolverV2(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = Normalizer.load(ROOT / config["stats"])

    train = config["train"]
    dataset = ShipDataset(ROOT / train["dataset_root"], train["entries"])
    results = {"config": str(config_path), "best_epoch": checkpoint["epoch"], "entries": {}}
    for entry_id, sample in zip(dataset.entry_ids, dataset):
        results["entries"][entry_id] = evaluate_entry(model, normalizer, sample, device)

    output = json.dumps(results, indent=2, ensure_ascii=False)
    (output_dir / "metrics.json").write_text(output + "\n")
    print(f"best epoch {checkpoint['epoch']}，指标写入 {output_dir / 'metrics.json'}")
    for entry_id, entry in results["entries"].items():
        print(f"\n{entry_id}")
        for domain in ("flow", "hull"):
            block = entry[domain]
            print(f"  {domain}（{block['n_points']} 点）")
            for name, unit, rmse, r2 in zip(block["channels"], block["units"],
                                            block["rmse"], block["r2"]):
                print(f"    {name:6s} RMSE {rmse:10.6f} {unit:4s}  R² {r2:+.4f}")


if __name__ == "__main__":
    main()
