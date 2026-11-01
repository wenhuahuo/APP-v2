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
    domains = getattr(model, "output_domains", ("flow", "hull"))
    flow_prediction = None
    hull_prediction = None
    index = None
    if "flow" in domains:
        index = torch.nonzero(sample["flow_mask"].reshape(-1), as_tuple=False).squeeze(-1)
        coords = sample["flow_coords"].reshape(-1, 2)
        prediction = []
        for start in range(0, len(index), chunk):
            part = index[start:start + chunk]
            queries = coords[part]
            output = model(batch, queries[None].to(device))
            prediction.append(output["flow"][0].cpu())
            if "hull" in domains:
                hull_prediction = output["hull"][0][:, 0].cpu()
        flow_prediction = torch.cat(prediction)
    elif "hull" in domains:
        output = model(batch)
        hull_prediction = output["hull"][0][:, 0].cpu()

    if flow_prediction is not None:
        scale = physical_flow_scale(sample["scale"])
        flow_prediction = (normalizer.flow_mean + normalizer.flow_std * flow_prediction) * scale
    if hull_prediction is not None:
        hull_prediction = normalizer.hull_mean + normalizer.hull_std * hull_prediction
    return flow_prediction, hull_prediction, index


def _entry_metrics(flow_prediction, hull_prediction, sample, index):
    """在指定流场点与整条湿表面上比较预测和真值。"""
    result = {}
    if flow_prediction is not None:
        scale = physical_flow_scale(sample["scale"])
        flow_target = sample["flow_target"].reshape(-1, 5)[index] * scale
        flow = channel_metrics(flow_prediction, flow_target,
                               sample["flow_weight"].reshape(-1)[index])
        result["flow"] = {"channels": list(FLOW_CHANNELS), "units": list(FLOW_UNITS), **flow,
                          "n_points": int(len(index))}
    if hull_prediction is not None:
        hull_target = sample["hull_pressure"]
        hull = channel_metrics(hull_prediction[:, None], hull_target[:, None],
                               sample["hull_weight"])
        result["hull"] = {"channels": list(HULL_CHANNELS), "units": list(HULL_UNITS), **hull,
                          "n_points": int(len(hull_target))}
    return result


@torch.no_grad()
def evaluate_zero_entry(sample):
    """恒零预测基线：直接在物理量空间令所有输出为零。"""
    index = torch.nonzero(sample["flow_mask"].reshape(-1), as_tuple=False).squeeze(-1)
    flow_target = sample["flow_target"].reshape(-1, 5)[index]
    return _entry_metrics(torch.zeros_like(flow_target),
                          torch.zeros_like(sample["hull_pressure"]), sample, index)


@torch.no_grad()
def evaluate_entry(model, normalizer, sample, device, chunk=CHUNK):
    """在条目当前密度的格点（掩除位除外）与整条湿表面上算指标。"""
    flow_prediction, hull_prediction, index = predict_entry(
        model, normalizer, sample, device, chunk)
    return _entry_metrics(flow_prediction, hull_prediction, sample, index)
