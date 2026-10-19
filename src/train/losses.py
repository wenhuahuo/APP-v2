"""两域带求积权重的损失，所有量在标准化空间里比较。

权重为 0 的点（掩除的格点、padding 的船面点）不参与损失，不需要额外的 mask。
"""


def weighted_mse(prediction, target, weight):
    """逐通道的加权均方误差：prediction/target (..., C)，weight (...) 为该点的求积权重。

    分母同时除以通道数，因此数值上等于「平均每通道的 1 - R²」，λ 在两个域之间可比。
    """
    channels = prediction.shape[-1]
    return (weight[..., None] * (prediction - target) ** 2).sum() / (weight.sum() * channels)


def appsolver_loss(flow, hull, lambda_hull=1.0):
    """flow、hull 各为 (prediction, target, weight) 三元组。"""
    flow_loss = weighted_mse(*flow)
    hull_loss = weighted_mse(*hull)
    return {"flow": flow_loss, "hull": hull_loss,
            "total": flow_loss + lambda_hull * hull_loss}
