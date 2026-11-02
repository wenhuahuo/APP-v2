"""按配置训练模型（APP 主线或基线）。

用法:
    python3 scripts/train/train_model.py --config configs/appsolver_v2/appsolver_v2_kcs_normal_wide.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
for name in ("datapipe", "app", "baseline", "model", "train"):
    sys.path.insert(0, str(ROOT / "src" / name))

from build import build_model  # noqa: E402
from normalize import Normalizer  # noqa: E402
from trainer import Trainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    model = build_model(config)
    normalizer = Normalizer.load(ROOT / config["stats"])

    trainer = Trainer(model, normalizer, config, ROOT / "output" / config_path.stem)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{config_path.stem}: {config['model_type']} {parameters / 1e6:.3f} M 参数，"
          f"条目 {config['train']['entries']}")
    trainer.run()


if __name__ == "__main__":
    main()
