"""把 9 个主实验模型的容量超参标定到共同的前向 GFLOPs，写出等计算量（iso-FLOPs）配置。

沿用 `*_kcs_fr_sweep_isoflops.yaml` 的做法：只调主干容量（宽度，必要时用层数补足），
分词 / 潜网格 / latent 数、任务口径与训练协议都不动；目标 GFLOPs 取 APPSolver-v2 主线
配置的实测前向计算量（论文分析实验的锚点模型）。统计口径与
`scripts/eval/profile_model_flops.py` 完全一致：batch 1、单条目
`kcs_fr026_normal_mesh`、stride 1、全部自由液面格点同时作为输入点与查询点。

每次测量都在独立子进程里重建模型：`FlopCounterMode` 需要 autograd 图，单次前向的
激活已占数 GB，同一进程反复测量会把内存耗尽（PyTorch 不把释放的显存/内存还给系统）。

用法:
    python3 scripts/eval/calibrate_isoflops.py                  # 只报告标定结果，不写文件
    python3 scripts/eval/calibrate_isoflops.py --write          # 写出 *_multihull_isoflops.yaml
    python3 scripts/eval/calibrate_isoflops.py --target-gflops 248.0 --write
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
for name in ("datapipe", "app", "baseline", "model", "train", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

import torch  # noqa: E402

from profile_model_flops import DEFAULT_CONFIGS, load_config, profile  # noqa: E402

# 每个模型的主干宽度旋钮：`appsolver_v2` 是锚点，不做改动。同一模型里成对出现的宽度
# （CORAL 的 SIREN 隐层与调制隐层）按官方实现等宽约定一起缩放。
ANCHOR = "appsolver_v2"
KNOBS = {
    "appsolver_v2": (),
    "appsolver_oformer": ("d_model",),
    "appsolver_v1": ("d_model",),
    "gnot": ("d_model",),
    "upt": ("d_model",),
    "transolver": ("d_model",),
    "oformer": ("d_model",),
    "gino": ("gno_kernel_hidden",),
    "coral": ("siren_hidden", "modulation_hidden"),
}
# GINO 的计算量由 GNO 核 MLP（每个查询点的邻域核）主导，FNO 通道数的影响小得多，
# 因此用核宽作为容量旋钮，`hidden_channels` 保持基础配置。
# 宽度步长：`d_model` 需能被 nhead 整除（本仓库的 nhead 为 4 或 8），OFormer 系列的
# 二维旋转位置编码还要求 `d_model / nhead` 被 4 整除（每轴再对半分），即 32 的倍数。
STEPS = {"appsolver_v1": 4, "gnot": 4, "oformer": 32, "appsolver_oformer": 32}
# 宽度取值网格太粗时用层数补足计算量（宽度 + 层数同时微调，不新增结构）。
SECONDARY = {
    "appsolver_oformer": "propagator_layers",
    "appsolver_v1": "layers",
    "gnot": "layers",
    "upt": "layers",
    "transolver": "layers",
    "oformer": "propagator_layers",
    "gino": "fno_layers",
    "coral": "siren_layers",
}
MAX_LAYERS = 12


def model_stem(config_path):
    return Path(config_path).stem[: -len("_multihull_normal")]


def target_isoflops_path(config_path):
    return Path(config_path).with_name(model_stem(config_path) + "_multihull_isoflops.yaml")


def variant(config_path, capacity):
    """把容量旋钮写进配置字典，构造待测模型。"""
    config = load_config(config_path)
    for key, value in capacity.items():
        config["model"][key] = int(value)
    return config


def measure(config_path, capacity, dataset, entry, device):
    """返回 (参数量, 前向 GFLOPs)：在独立子进程里按统一口径统计，避免内存累积。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(variant(config_path, capacity), handle, allow_unicode=True,
                       sort_keys=False)
        temporary = handle.name
    try:
        command = [sys.executable, str(Path(__file__).resolve()), "--measure-config", temporary,
                   "--dataset", dataset, "--entry", entry, "--device", str(device)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        row = json.loads(result.stdout.strip().splitlines()[-1])
        return row["parameters"], row["gflops"]
    finally:
        Path(temporary).unlink()


class Calibrator:
    """按宽度（必要时加层数）搜索到目标 FLOPs，测量结果按容量缓存。"""

    def __init__(self, config_path, target, dataset, entry, device, tolerance=0.05):
        self.config_path = config_path
        self.stem = model_stem(config_path)
        self.target = target
        self.dataset = dataset
        self.entry = entry
        self.device = device
        self.tolerance = tolerance
        self.base = load_config(config_path)["model"]
        self.knobs = [key for key in KNOBS[self.stem] if key in self.base]
        self.step = STEPS.get(self.stem, 8)
        self.secondary = SECONDARY.get(self.stem)
        self.cache = {}

    def error(self, flops):
        return 100.0 * (flops - self.target) / self.target

    def evaluate(self, capacity):
        key = tuple(sorted(capacity.items()))
        if key not in self.cache:
            parameters, flops = measure(self.config_path, capacity, self.dataset, self.entry,
                                        self.device)
            self.cache[key] = (dict(capacity), parameters, flops, self.error(flops))
        return self.cache[key]

    def width_only(self, width):
        return {key: width for key in self.knobs}

    def start(self):
        return {key: int(self.base[key]) for key in self.knobs}

    def accept(self, row):
        return abs(row[3]) <= 100 * self.tolerance

    def change(self, row):
        """相对基础配置的改动量：宽度按步长归一，层数按 1 层归一（同等误差时优先改动小的）。"""
        if not self.knobs:
            return 0.0
        width = row[0][self.knobs[0]]
        changed = abs(width - int(self.base[self.knobs[0]])) / self.step
        if self.secondary:
            layers = row[0].get(self.secondary, int(self.base[self.secondary]))
            changed += abs(layers - int(self.base[self.secondary]))
        return changed

    def search_width(self):
        """宽度二分：FLOPs 随宽度单调增，先按 `FLOPs ∝ 宽度²` 估初值。"""
        start = self.evaluate(self.start())
        if not self.knobs:
            return [start]
        guess = max(self.step, int(round(int(self.base[self.knobs[0]])
                                         * (self.target / start[2]) ** 0.5
                                         / self.step)) * self.step)
        low, high = self.step, guess
        while self.evaluate(self.width_only(high))[2] < self.target:
            if high >= 8 * int(self.base[self.knobs[0]]):
                break
            low, high = high, int(high * 1.5) // self.step * self.step
        rows = [self.evaluate(self.width_only(high))]
        while low <= high:
            middle = max(low + (high - low) // 2 // self.step * self.step, self.step)
            row = self.evaluate(self.width_only(middle))
            rows.append(row)
            if row[2] < self.target:
                low = middle + self.step
            else:
                high = middle - self.step
        return rows

    def refine_layers(self, rows):
        """宽度网格够不到目标时，用层数在两侧相邻宽度上补足计算量。"""
        if not self.secondary or self.secondary not in self.base:
            return rows
        layers = int(self.base[self.secondary])
        unique = {tuple(sorted(row[0].items())): row for row in rows}.values()
        straddling = sorted(unique, key=lambda row: abs(row[3]))[:2]
        for row in straddling:
            width = row[0][self.knobs[0]]
            capacity = self.width_only(width)
            capacity[self.secondary] = layers + 1
            slope = self.evaluate(capacity)[2] - row[2]
            if slope <= 0:
                continue
            needed = layers + round((self.target - row[2]) / slope)
            for candidate in (needed - 1, needed, needed + 1):
                if not 1 <= candidate <= MAX_LAYERS:
                    continue
                capacity = self.width_only(width)
                capacity[self.secondary] = candidate
                rows.append(self.evaluate(capacity))
        return rows

    def run(self):
        rows = self.search_width()
        if not any(self.accept(row) for row in rows):
            rows = self.refine_layers(rows)
        return min(rows, key=lambda row: (abs(row[3]), self.change(row)))


def train_block(text):
    """配置文本里的整段 `train:`（含其下所有缩进行）。"""
    lines = text.splitlines(keepends=True)
    start = next(index for index, line in enumerate(lines) if line.startswith("train:"))
    end = next((index for index in range(start + 1, len(lines))
                if lines[index] and not lines[index][0].isspace()), len(lines))
    return "".join(lines[start:end])


def write_config(config_path, capacity, parameters, flops):
    """按基础配置的行文写出等计算量配置：只改容量行与头注释。

    训练超参沿用既有等计算量配置：学习率等只针对该容量口径的覆盖项（APPSolver-v1 的
    `1e-4`）不属于容量，从 `*_multihull_normal.yaml` 重写会静默丢掉。条目、统计量等与
    基础配置的一致性由 `tests/test_isoflops.py` 逐项核对。
    """
    template = load_config(config_path)
    text = "".join(line for line in Path(config_path).read_text().splitlines(keepends=True)
                   if not line.startswith("#"))
    out = target_isoflops_path(config_path)
    if out.is_file():
        previous = "".join(line for line in out.read_text().splitlines(keepends=True)
                           if not line.startswith("#"))
        text = text.replace(train_block(text), train_block(previous), 1)
    for key, value in capacity.items():
        pattern = rf"(^  {key}: )(\S+)(.*)$"
        text, hits = re.subn(pattern, lambda match: f"{match.group(1)}{value}{match.group(3)}",
                             text, count=1, flags=re.M)
        if hits != 1:
            raise RuntimeError(f"{config_path} 的 {key} 命中 {hits} 次")
    changes = "、".join(f"{key} {template['model'][key]} → {value}"
                        for key, value in capacity.items()
                        if int(template["model"][key]) != int(value))
    header = (f"# {model_stem(config_path)}：等计算量（iso-FLOPs）对照，前向 {flops:.1f} GFLOPs、"
              f"{parameters / 1e6:.3f} M 参数\n"
              "# 主实验统一口径：自由液面 + 船体表面输入、流场与船面双域输出、逐层 Fourier add、\n"
              "# lambda_hull=1.0\n"
              f"# 与 {Path(config_path).name} 的唯一差别是主干容量\n"
              f"# （{changes or '锚点不动'}），分词/潜网格与训练协议不变\n")
    out.write_text(header + text)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--dataset", default=str(ROOT / "dataset" / "shipbench"))
    parser.add_argument("--entry", default="kcs_fr026_normal_mesh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-gflops", type=float, default=None,
                        help="共同前向 GFLOPs；缺省取 APPSolver-v2 主线配置的实测值")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="相对目标的可接受误差，默认 5%（与历史 iso-FLOPs 对照同口径）")
    parser.add_argument("--write", action="store_true", help="写出等计算量配置")
    parser.add_argument("--measure-config", default=None,
                        help="内部用：测量一个临时配置，结果以 JSON 写 stdout")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.measure_config:
        parameters, flops, _ = profile(args.measure_config, args.dataset, args.entry, device)
        print(json.dumps({"parameters": parameters, "gflops": flops}))
        return

    target = args.target_gflops
    if target is None:
        anchor = next(path for path in DEFAULT_CONFIGS if model_stem(path) == ANCHOR)
        anchor_parameters, target, _ = profile(anchor, args.dataset, args.entry, device)
        print(f"锚点 {model_stem(anchor)}：{anchor_parameters / 1e6:.3f} M 参数、"
              f"{target:.3f} GFLOPs\n")
    print(f"{'模型':20s} {'容量':>34s} {'参数(M)':>9s} {'GFLOPs':>9s} {'误差%':>7s}")
    for config_path in args.configs:
        capacity, parameters, flops, error = Calibrator(
            config_path, target, args.dataset, args.entry, device, args.tolerance).run()
        described = "、".join(f"{key}={value}" for key, value in capacity.items()) or "锚点不动"
        print(f"{model_stem(config_path):20s} {described:>34s} {parameters / 1e6:9.3f} "
              f"{flops:9.3f} {error:7.2f}", flush=True)
        if args.write:
            out = write_config(config_path, capacity, parameters, flops)
            print(f"  写入 {out}")


if __name__ == "__main__":
    main()
