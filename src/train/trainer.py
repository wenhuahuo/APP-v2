"""训练循环：查询点集随机子采样、按启用输出域计算损失、按验证损失选 checkpoint。

自由液面域有十万量级查询点，每步随机抽一部分计算损失；查询是输出函数上的采样，
抽子集不改变模型的定义。验证用的是独立条目（`train.val_entries`），固定查询点集，
便于跨 epoch 比较。
`train_strides` 给出训练时的查询密度档位（1 = 标签点集原密度），每步随机取一档。
计算设备由配置的 `device` 显式给出（`cpu` / `cuda`），不做自动回退。

验证要在整条验证集上前向，比一轮训练贵；`VALIDATION_INTERVAL` 给出验证间隔，
最后一轮无论间隔如何都验证一次，`best.pt` 只在实际验证的轮次上按验证损失更新。
"""

from pathlib import Path

import torch

from collate import collate_samples
from losses import appsolver_loss
from ship_dataset import ShipDataset

VALIDATION_INTERVAL = 5


class Trainer:
    def __init__(self, model, normalizer, config, output_dir):
        self.model = model
        self.normalizer = normalizer
        self.full_config = config
        self.config = config["train"]
        self.domains = getattr(model, "output_domains", ("flow", "hull"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = self.config
        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self.datasets = {stride: ShipDataset(config["dataset_root"], config["entries"],
                                             stride=stride)
                         for stride in config["train_strides"]}
        self.validation = {stride: ShipDataset(config["dataset_root"], config["val_entries"],
                                               stride=stride)
                           for stride in config["train_strides"]}
        self.generator = torch.Generator().manual_seed(config["seed"])
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config["weight_decay"])
        validation = self.validation[config["train_strides"][0]]
        self.val_queries = (self._sample_queries(
            config["val_queries"], torch.Generator().manual_seed(config["seed"]), validation)
            if "flow" in self.domains else None)

    @staticmethod
    def _sample_queries(count, generator, dataset):
        """在给定查询密度档位的格点上随机抽查询点下标。"""
        nx, ny = dataset.flow_coords.shape[:2]
        flat = torch.randperm(nx * ny, generator=generator)[:count]
        return torch.stack([flat // ny, flat % ny], dim=-1)

    def _forward(self, sample, index):
        batch = {key: value.to(self.device) for key, value in collate_samples([sample]).items()}
        queries = None
        flow = None
        hull = None
        if "flow" in self.domains:
            queries = sample["flow_coords"][index[:, 0], index[:, 1]][None].to(self.device)
        output = self.model(batch, queries)
        if "flow" in self.domains:
            flow_target = self.normalizer.flow(
                sample["flow_target"][index[:, 0], index[:, 1]])[None].to(self.device)
            flow_weight = sample["flow_weight"][index[:, 0], index[:, 1]][None].to(self.device)
            flow = (output["flow"][0], flow_target[0], flow_weight[0])
        if "hull" in self.domains:
            hull_target = self.normalizer.hull(sample["hull_pressure"])[:, None][None].to(self.device)
            hull = (output["hull"][0], hull_target[0], batch["hull_weight"][0])
        return output, flow, hull

    def train_epoch(self, epoch):
        self.model.train()
        strides = self.config["train_strides"]
        total = 0.0
        for step in range(self.config["steps_per_epoch"]):
            stride = strides[step % len(strides)]
            dataset = self.datasets[stride]
            sample = dataset[step % len(dataset)]
            index = (self._sample_queries(self.config["batch_queries"], self.generator, dataset)
                     if "flow" in self.domains else None)
            _, flow, hull = self._forward(sample, index)
            losses = appsolver_loss(flow, hull, self.config.get("lambda_hull", 0.0))

            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()
            total += losses["total"].item()
        return total / self.config["steps_per_epoch"]

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        dataset = self.validation[self.config["train_strides"][0]]
        totals = {domain: 0.0 for domain in self.domains}
        for i in range(len(dataset)):
            _, flow, hull = self._forward(dataset[i], self.val_queries)
            losses = appsolver_loss(flow, hull, self.config.get("lambda_hull", 0.0))
            for key in totals:
                totals[key] += losses[key]
        return {key: value / len(dataset) for key, value in totals.items()}

    def run(self):
        best = float("inf")
        history = []
        epochs = self.config["epochs"]
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            # 轮次从 1 起数（第 epoch+1 轮），因此 100 轮配置的第 100 轮正好落在间隔上；
            # 轮数不是 5 的倍数时由 `epoch == epochs - 1` 兜住最后一轮。
            validate = (epoch + 1) % VALIDATION_INTERVAL == 0 or epoch == epochs - 1
            record = {"epoch": epoch, "train": train_loss}
            # last.pt 每轮都写：训练被打断时也有最近状态的模型可用。
            checkpoint = {"model": self.model.state_dict(), "config": self.full_config,
                          "epoch": epoch}
            torch.save(checkpoint, self.output_dir / "last.pt")
            if validate:
                metrics = self.evaluate()
                validation = (metrics.get("flow", 0.0)
                              + self.config.get("lambda_hull", 0.0) * metrics.get("hull", 0.0))
                record["validation"] = validation
                record.update({key: float(value) for key, value in metrics.items()})
                details = "".join(f"  val {key} {float(value):.5f}"
                                  for key, value in metrics.items())
                print(f"epoch {epoch:3d}  train {train_loss:.5f}{details}")
                # 两条选择规则都留档：best.pt 按验证损失选（早停），last.pt 每轮覆盖。
                # 训练集条目很少时验证损失往往在最初几轮最低，只看 best.pt 会报成欠训练。
                if validation < best:
                    best = validation
                    torch.save(checkpoint, self.output_dir / "best.pt")
            else:
                print(f"epoch {epoch:3d}  train {train_loss:.5f}")
            history.append(record)
        return history
