"""查询密度评测：同一模型在若干 `stride` 的格点上整片预测，报误差随密度的变化。

`stride` 对格点抽样，粗档真值由细格真值子采样得到、权重乘 `stride²`，因此每个档位
都是同一片物理域上不同密度的查询点集与对应真值。基线在各自密度的整片点云上直接
输出，APP 用同一密度的点云分词后在格点上查询，两者协议一致。
"""

from evaluate import evaluate_entry
from ship_dataset import ShipDataset


def density_sweep(model, normalizer, dataset_root, entry_ids, strides, device, chunk):
    """返回 {stride: {条目: 指标}}；`strides` 中的每档单独构造 Dataset。"""
    results = {}
    for stride in strides:
        dataset = ShipDataset(dataset_root, entry_ids, stride=stride)
        results[stride] = {entry_id: evaluate_entry(model, normalizer, sample, device, chunk)
                           for entry_id, sample in zip(dataset.entry_ids, dataset)}
    return results
