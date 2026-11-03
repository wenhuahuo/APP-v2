"""测量主实验 18 个 run 的单条目前向延迟（stride 1/2/4/8），写 `inference_time.json`。

口径（与 `docs/main_experiment_protocol.md` 的推理时间章节一致）：

- 同一条目 `kcs_fr026_normal_mesh`、batch 1、`float32`、`eval()` + `torch.no_grad()`；
- 模型从各自的 `best.pt` 恢复，与主表同一 checkpoint；
- 数据（冻结场输入点云、船面点云、工况、查询坐标）预先搬到 GPU，计时只含一次完整
  前向 `model(batch, queries)`，模型自带的编码、分词与 GINO 的邻域搜索都在前向内部；
- 每次前向前后各 `torch.cuda.synchronize()`；warmup 与 repeat 由命令行给出（缺省 5 / 20），
  报告均值与样本标准差，逐次耗时一并写入 JSON。

每个配置写到 `output/<配置文件名>/inference_time.json`；`--json` 可另存一份合并结果。

用法:
    python3 scripts/eval/measure_inference_time.py
    python3 scripts/eval/measure_inference_time.py --configs configs/gino/gino_random_seed0.yaml
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
from latency import fixed_inputs, summary, time_forward  # noqa: E402

MODELS = ("appsolver_v2", "appsolver_oformer", "appsolver_v1", "gnot", "upt",
          "transolver", "oformer", "gino", "coral")
CAPACITY_SUFFIXES = ("", "_isoflops")
STRIDES = (1, 2, 4, 8)
ENTRY = "kcs_fr026_normal_mesh"
DEFAULT_CONFIGS = tuple(
    f"configs/{model}/{model}_random_seed0{suffix}.yaml"
    for model in MODELS for suffix in CAPACITY_SUFFIXES)


def device_name(device):
    if device.type != "cuda":
        return f"cpu（{torch.get_num_threads()} 线程）"
    properties = torch.cuda.get_device_properties(device)
    return f"{properties.name}（{properties.total_memory / 2**30:.0f} GiB）"


def measure(config_path, args):
    """单个配置：返回 (参数量, {stride: 计时结果}) 与写出的 JSON 内容。"""
    config = yaml.safe_load(config_path.read_text())
    output_dir = ROOT / "output" / config_path.stem
    device = torch.device(args.device)
    checkpoint = torch.load(output_dir / args.checkpoint, weights_only=False,
                            map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset_root = ROOT / config["train"]["dataset_root"]

    blocks = {}
    for stride in args.strides:
        batch, queries = fixed_inputs(dataset_root, args.entry, stride, device)
        samples = time_forward(model, batch, queries, warmup=args.warmup, repeat=args.repeat)
        blocks[str(stride)] = {"query_points": int(queries.shape[1]),
                               "input_points": int(batch["flow_coords"].shape[1]
                                                   * batch["flow_coords"].shape[2]),
                               **summary(samples),
                               "samples": samples}
    record = {
        "config": str(config_path),
        "model_type": config["model_type"],
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "entry": args.entry,
        "batch_size": 1,
        "dtype": "float32",
        "mode": "eval + torch.no_grad",
        "device": device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "sync": "torch.cuda.synchronize before and after every forward",
        "strides": blocks,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--strides", type=int, nargs="+", default=list(STRIDES))
    parser.add_argument("--entry", default=ENTRY)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--json", default=None, help="可选的合并结果路径")
    args = parser.parse_args()

    records = []
    header = f"{'配置':44s} {'模型':20s}" + "".join(f"{'s' + str(s):>12s}" for s in args.strides)
    print(header)
    for config in args.configs:
        path = Path(config)
        record = measure(path, args)
        records.append(record)
        row = f"{path.stem:44s} {record['model_type']:20s}"
        for stride in args.strides:
            block = record["strides"][str(stride)]
            row += f"{1000 * block['mean']:12.2f}"
        print(row)
        output_dir = ROOT / "output" / path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "inference_time.json"
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"  写入 {target}（{record['device']}，warmup {args.warmup}，"
              f"repeat {args.repeat}）")

    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        print(f"合并结果写入 {args.json}")


if __name__ == "__main__":
    main()
