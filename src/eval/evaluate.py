"""逐通道物理量指标：在给定查询密度的格点上整片预测，与真值比较。

预测在标准化空间，先还原到无量纲量；标签本身已是无量纲量，只乘尺度。

输入点云密度与查询密度可以解耦：`predict_entry` 的 `query_sample` 给出查询点集与
对应真值，`sample` 给出模型输入。支持独立查询的模型（`encode` / `decode` 分开）
只在输入点云上编码一次，随后分块解码，换查询密度不重跑编码器。
"""

import torch

from collate import collate_samples
from metrics import FLOW_CHANNELS, FLOW_UNITS, HULL_CHANNELS, HULL_UNITS, channel_metrics

CHUNK = 20000


def physical_flow_scale(scale):
    """无量纲化尺度：`eta/Lpp` 乘 Lpp，三个速度分量乘 U_inf，压力已是 Cp。"""
    lpp, u_inf = float(scale[0]), float(scale[1])
    return torch.tensor([lpp, u_inf, u_inf, u_inf, 1.0])


def _encode(model, batch):
    """支持独立查询的模型编码一次；其余模型在每次前向里自带编码。"""
    return model.encode(batch) if hasattr(model, "encode") else None


@torch.no_grad()
def encode_sample(model, sample, device):
    """先把输入点云编码一次并返回状态，供不同查询密度复用。"""
    model.eval()
    batch = {key: value.to(device) for key, value in collate_samples([sample]).items()}
    return _encode(model, batch)


def _flow_chunk(model, batch, state, queries):
    if state is not None:
        return model.decode(state, queries)
    return model(batch, queries)["flow"]


@torch.no_grad()
def predict_entry(model, normalizer, sample, device, chunk=CHUNK, query_sample=None,
                  encoded=None):
    """整片预测并换算回物理量，返回流场 (N, 5)、船面 Cp (M,) 与流场点的展平下标。

    `sample` 决定模型输入（点云密度），`query_sample` 决定查询点集与真值；缺省时
    两者相同。`encoded` 可传入预先算好的编码状态（`encode_sample`），从而在多个查询
    密度之间共享同一次编码。`index` 取查询点 `flow_mask.reshape(-1)` 的非零位置，与 `flow_points`、
    `flow_target` 的散点顺序一致，可视化据此把预测铺回格点图像。
    """
    model.eval()
    query_sample = sample if query_sample is None else query_sample
    batch = {key: value.to(device) for key, value in collate_samples([sample]).items()}
    domains = getattr(model, "output_domains", ("flow", "hull"))
    flow_prediction = None
    hull_prediction = None
    index = None
    if "flow" in domains:
        index = torch.nonzero(query_sample["flow_mask"].reshape(-1), as_tuple=False).squeeze(-1)
        coords = query_sample["flow_coords"].reshape(-1, 2)
        state = _encode(model, batch) if encoded is None else encoded
        prediction = []
        for start in range(0, len(index), chunk):
            part = index[start:start + chunk]
            queries = coords[part][None].to(device)
            prediction.append(_flow_chunk(model, batch, state, queries)[0].cpu())
        flow_prediction = torch.cat(prediction)
        if "hull" in domains and state is not None:
            # 船面输出与流场查询点无关，单独前向一次即可，不必在分块循环里重复
            hull_prediction = model(batch, coords[:1][None].to(device))["hull"][0][:, 0].cpu()
    elif "hull" in domains:
        output = model(batch)
        hull_prediction = output["hull"][0][:, 0].cpu()

    if flow_prediction is not None:
        scale = physical_flow_scale(query_sample["scale"])
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
def evaluate_entry(model, normalizer, sample, device, chunk=CHUNK, query_sample=None,
                   encoded=None):
    """在查询点集的格点（掩除位除外）与整条湿表面上算指标。"""
    query_sample = sample if query_sample is None else query_sample
    flow_prediction, hull_prediction, index = predict_entry(
        model, normalizer, sample, device, chunk, query_sample, encoded)
    return _entry_metrics(flow_prediction, hull_prediction, query_sample, index)
