"""训练随机种子的唯一入口：同时固定 Python、NumPy、PyTorch 与 CUDA 的全局 RNG。

配置里的 `train.seed` 只在这里生效一次入口，`train_model.py`（模型初始化前）与
`Trainer.__init__`（查询点采样前）都调用它，因此同一条配置重跑得到同样的权重初始化、
同样的训练查询子集和同样的验证查询集合。

训练循环按确定性下标顺序读取条目（`dataset[step % len(dataset)]`），本仓库没有
DataLoader、也没有 shuffle，所以不需要 DataLoader 级别的 generator；如果以后引入
DataLoader，必须用同一个 `seed` 构造 `torch.Generator(...).manual_seed(seed)`，
否则这里覆盖的随机性就不再完整。

用法:
    from seeding import set_seed
    set_seed(config["train"]["seed"])
"""

import random

import numpy as np
import torch


def set_seed(seed):
    """把全局随机源固定到 `seed`（Python / NumPy / PyTorch / 全部 CUDA 设备）。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
