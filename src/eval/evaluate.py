"""逐通道物理量指标：在给定查询密度的格点上整片预测，与真值比较。

预测在标准化空间，先还原到无量纲量；标签本身已是无量纲量，只乘尺度。
"""

import torch

from collate import collate_samples
from metrics import FLOW_CHANNELS, FLOW_UNITS, HULL_CHANNELS, HULL_UNITS, channel_metrics

CHUNK = 20000


def physical_flow_scale(scale):
    """无量纲化尺度：`eta/Lpp` 乘 Lpp，三个速度分量乘 U_inf，压力已是 Cp。"""
    lpp, u_inf = float(scale[0]), float(scale[1])
    return torch.tensor([lpp, u_inf, u_inf, u_inf, 1.0])


@torch.no_grad()
def predict_entry(model, normalizer, sample, device, chunk=CHUNK):
    """整片预测并换算回物理量，返回流场 (N, 5)、船面 Cp (M,) 与流场点的展平下标。

    `index` 取 `flow_mask.reshape(-1)` 的非零位置，与 `flow_points`、`flow_target`
    的散点顺序一致，可视化据此把预测铺回格点图像。
    """
    model.eval()
    batch = {key: value.to(device) for key, value in collate_samples([sample]).items()}
    index = torch.nonzero(sample["flow_mask"].reshape(-1), as_tuple=False).squeeze(-1)
    coords = sample["flow_coords"].reshape(-1, 2)
    sdf = sample["flow_sdf"].reshape(-1)
    prediction = []
    for start in range(0, len(index), chunk):
        part = index[start:start + chunk]
        queries = torch.column_stack([coords[part], sdf[part]])[None].to(device)
        output = model(batch, queries)
        prediction.append(output["flow"][0].cpu())
        hull_prediction = output["hull"][0][:, 0].cpu()
    prediction = torch.cat(prediction)

    scale = physical_flow_scale(sample["scale"])
    flow = (normalizer.flow_mean + normalizer.flow_std * prediction) * scale
    hull = normalizer.hull_mean + normalizer.hull_std * hull_prediction
    return flow, hull, index


@torch.no_grad()
def evaluate_entry(model, normalizer, sample, device, chunk=CHUNK):
    """在条目当前密度的格点（掩除位除外）与整条湿表面上算指标。"""
    flow_prediction, hull_prediction, index = predict_entry(
        model, normalizer, sample, device, chunk)
    scale = physical_flow_scale(sample["scale"])
    flow_target = sample["flow_target"].reshape(-1, 5)[index] * scale
    flow = channel_metrics(flow_prediction, flow_target, sample["flow_weight"].reshape(-1)[index])

    hull_target = sample["hull_pressure"]
    hull = channel_metrics(hull_prediction[:, None], hull_target[:, None], sample["hull_weight"])
    return {
        "flow": {"channels": list(FLOW_CHANNELS), "units": list(FLOW_UNITS), **flow,
                 "n_points": int(len(index))},
        "hull": {"channels": list(HULL_CHANNELS), "units": list(HULL_UNITS), **hull,
                 "n_points": int(len(hull_target))},
    }
