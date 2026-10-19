"""训练循环：查询点集随机子采样、两域联合损失、按验证损失选 checkpoint。

自由液面域有十万量级查询点，每步随机抽一部分计算损失；查询是输出函数上的采样，
抽子集不改变模型的定义。验证集用固定查询点集，便于跨 epoch 比较。
`train_strides` 给出训练时的查询密度档位（1 = 标签点集原密度），每步随机取一档。
计算设备由配置的 `device` 显式给出（`cpu` / `cuda`），不做自动回退。
"""

from pathlib import Path

import torch

from collate import collate_samples
from losses import appsolver_loss
from ship_dataset import ShipDataset


class Trainer:
    def __init__(self, model, normalizer, config, output_dir):
        self.model = model
        self.normalizer = normalizer
        self.full_config = config
        self.config = config["train"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = self.config
        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self.datasets = {stride: ShipDataset(config["dataset_root"], config["entries"],
                                             stride=stride)
                         for stride in config["train_strides"]}
        self.generator = torch.Generator().manual_seed(config["seed"])
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config["weight_decay"])
        validation = self.datasets[config["train_strides"][0]]
        self.val_queries = self._sample_queries(config["val_queries"],
                                                torch.Generator().manual_seed(config["seed"]),
                                                validation)

    @staticmethod
    def _sample_queries(count, generator, dataset):
        """在给定查询密度档位的格点上随机抽查询点下标。"""
        nx, ny = dataset.flow_coords.shape[:2]
        flat = torch.randperm(nx * ny, generator=generator)[:count]
        return torch.stack([flat // ny, flat % ny], dim=-1)

    def _forward(self, sample, index):
        batch = {key: value.to(self.device) for key, value in collate_samples([sample]).items()}
        queries = torch.column_stack([
            sample["flow_coords"][index[:, 0], index[:, 1]],
            sample["flow_sdf"][index[:, 0], index[:, 1]],
        ])[None].to(self.device)
        output = self.model(batch, queries)
        flow_target = self.normalizer.flow(
            sample["flow_target"][index[:, 0], index[:, 1]])[None].to(self.device)
        flow_weight = sample["flow_weight"][index[:, 0], index[:, 1]][None].to(self.device)
        hull_target = self.normalizer.hull(sample["hull_pressure"])[:, None][None].to(self.device)
        return output, (output["flow"][0], flow_target[0], flow_weight[0]), \
            (output["hull"][0], hull_target[0], batch["hull_weight"][0])

    def train_epoch(self, epoch):
        self.model.train()
        strides = self.config["train_strides"]
        total = 0.0
        for step in range(self.config["steps_per_epoch"]):
            stride = strides[step % len(strides)]
            dataset = self.datasets[stride]
            sample = dataset[step % len(dataset)]
            index = self._sample_queries(self.config["batch_queries"], self.generator, dataset)
            _, flow, hull = self._forward(sample, index)
            losses = appsolver_loss(flow, hull, self.config["lambda_hull"])

            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()
            total += losses["total"].item()
        return total / self.config["steps_per_epoch"]

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        dataset = self.datasets[self.config["train_strides"][0]]
        totals = {"flow": 0.0, "hull": 0.0}
        for i in range(len(dataset)):
            _, flow, hull = self._forward(dataset[i], self.val_queries)
            losses = appsolver_loss(flow, hull, self.config["lambda_hull"])
            totals["flow"] += losses["flow"]
            totals["hull"] += losses["hull"]
        return {key: value / len(dataset) for key, value in totals.items()}

    def run(self):
        best = float("inf")
        history = []
        for epoch in range(self.config["epochs"]):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            validation = metrics["flow"] + self.config["lambda_hull"] * metrics["hull"]
            history.append({"epoch": epoch, "train": train_loss, "validation": validation,
                            **{key: float(value) for key, value in metrics.items()}})
            print(f"epoch {epoch:3d}  train {train_loss:.5f}  "
                  f"val flow {float(metrics['flow']):.5f}  val hull {float(metrics['hull']):.5f}")
            if validation < best:
                best = validation
                torch.save({"model": self.model.state_dict(), "config": self.full_config,
                            "epoch": epoch}, self.output_dir / "best.pt")
        return history
