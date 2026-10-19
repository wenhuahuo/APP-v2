"""带求积权重的逐通道误差指标，输入为同一物理单位的预测与真值。

RMSE 与预测/真值的量纲一致；R² 以真值的加权方差为基准，值域上界 1。
"""

FLOW_CHANNELS = ("eta", "u", "v", "w", "p_rgh")
FLOW_UNITS = ("m", "m/s", "m/s", "m/s", "Cp")
HULL_CHANNELS = ("Cp",)
HULL_UNITS = ("Cp",)


def channel_metrics(prediction, target, weight):
    """prediction/target (N, C)，weight (N,) 为该点的求积权重。"""
    w = weight[:, None]
    normalizer = weight.sum()
    residual = ((w * (prediction - target) ** 2).sum(dim=0) / normalizer).sqrt()
    mean = (w * target).sum(dim=0) / normalizer
    variance = (w * (target - mean) ** 2).sum(dim=0) / normalizer
    return {"rmse": residual.tolist(), "r2": (1.0 - residual ** 2 / variance).tolist()}
