"""跨条目池化与整体指标：与主表 `docs/main_experiment_protocol.md` 完全相同的聚合公式。

逐条目指标按该条目查询点数 n_e 加权池化。记通道 c 的逐条目 RMSE 与 R² 为
rmse(e,c)、R²(e,c)：

- 池化 RMSE(c) = sqrt( Σ_e n_e · rmse(e,c)² / Σ_e n_e )
- 池化方差 σ(c)² = Σ_e n_e · rmse(e,c)² / (1 − R²(e,c)) / Σ_e n_e
- NRMSE(c) = 池化 RMSE(c) / σ(c)
- 整体 RMSE = sqrt( mean_c NRMSE(c)² ) × 100%
- 整体 R² = 1 − (整体 RMSE / 100)² = mean_c ( 1 − NRMSE(c)² )

五通道无量纲（eta/Lpp、速度/U_inf、Cp），因此整体 RMSE 直接按百分比读。`R² → 1` 的数值
情形按极大方差处理（NRMSE → 0），避免除零。
"""

import math

from metrics import FLOW_CHANNELS

MIN_VARIANCE_FRACTION = 1e-12


def pooled_channel(records, channel, group="flow"):
    """单通道的点数加权池化 RMSE、池化标准差与总点数。"""
    total = 0
    squared = 0.0
    variance = 0.0
    for record in records.values():
        block = record[group]
        count = block["n_points"]
        residual = block["rmse"][channel]
        total += count
        squared += count * residual * residual
        variance += count * residual * residual / max(1.0 - block["r2"][channel],
                                                      MIN_VARIANCE_FRACTION)
    if not total:
        raise ValueError("没有条目点数，无法池化")
    return math.sqrt(squared / total), math.sqrt(variance / total), total


def overall(records, channels=FLOW_CHANNELS):
    """整体 RMSE（%）与整体 R²，另给逐通道池化 RMSE、NRMSE 与总点数。"""
    pooled = [pooled_channel(records, channel) for channel in range(len(channels))]
    nrmse = [rmse / sigma for rmse, sigma, _ in pooled]
    overall_rmse = 100.0 * math.sqrt(sum(value * value for value in nrmse) / len(nrmse))
    return {
        "n_points": pooled[0][2],
        "rmse": [rmse for rmse, _, _ in pooled],
        "nrmse": nrmse,
        "overall_rmse": overall_rmse,
        "overall_r2": 1.0 - (overall_rmse / 100.0) ** 2,
    }


def eta_rmse_mm(records):
    """η 通道的池化 RMSE，单位 mm。"""
    return 1000.0 * pooled_channel(records, 0)[0]
