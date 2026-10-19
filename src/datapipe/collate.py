"""把一批条目样本堆成张量。

流场域是固定的 (`nx`, `ny`) 索引网格，直接堆叠即可。船面域点数随船型与网格变化，
padding 到批内最大点数，padding 位置权重为 0 并由 `hull_mask` 标出。
同一批内查询点集密度必须一致，否则流场面板形状不同。
"""

import torch

STACKED_KEYS = (
    "flow_coords", "flow_extent", "flow_sdf", "flow_target", "flow_weight", "flow_mask",
    "condition", "scale",
)
PADDED_KEYS = ("hull_points", "hull_uv", "hull_normals", "hull_pressure", "hull_weight")


def collate_samples(samples):
    shapes = {tuple(s["flow_sdf"].shape) for s in samples}
    if len(shapes) != 1:
        raise ValueError(f"同一批内的查询点集密度不一致：流场面板形状 {sorted(shapes)}")

    batch = {key: torch.stack([s[key] for s in samples]) for key in STACKED_KEYS}

    max_points = max(len(s["hull_uv"]) for s in samples)
    for key in PADDED_KEYS:
        padded = samples[0][key].new_zeros((len(samples), max_points) + samples[0][key].shape[1:])
        for b, sample in enumerate(samples):
            padded[b, :len(sample[key])] = sample[key]
        batch[key] = padded

    hull_mask = torch.zeros((len(samples), max_points), dtype=torch.bool)
    for b, sample in enumerate(samples):
        hull_mask[b, :len(sample["hull_uv"])] = sample["hull_mask"]
    batch["hull_mask"] = hull_mask
    return batch
