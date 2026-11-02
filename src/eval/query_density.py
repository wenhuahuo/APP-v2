"""查询密度评测：同一模型在若干 `stride` 的查询点集上整片预测，报误差随密度的变化。

`stride` 对标签格点抽样，粗档真值由细格真值子采样得到、权重乘 `stride²`，因此每个
档位都是同一片物理域上不同密度的查询点集与对应真值。

`encode_stride` 把输入点云密度与查询密度解耦：模型输入固定按 `encode_stride` 的格点
构造，每条条目只编码一次，随后在各档查询密度上共用这个编码结果，因此「固定输入编码
密度、改变查询密度」不会重跑编码器。只有真正按查询坐标解码的模型
（`supports_query_decoder`）支持这一协议；把输出留在输入点、再用格点下标取值的模型
在两种密度不一致时会静默取错点，这里直接报错。
"""

from evaluate import encode_sample, evaluate_entry
from ship_dataset import ShipDataset


def density_sweep(model, normalizer, dataset_root, entry_ids, strides, device, chunk,
                  encode_stride=None):
    """返回 {stride: {条目: 指标}}；`strides` 中的每档单独构造查询点集。"""
    if encode_stride is not None and not getattr(model, "supports_query_decoder", False):
        raise ValueError(f"{type(model).__name__} 不支持独立查询，不能固定输入密度评测")
    fixed = None
    states = {}
    if encode_stride is not None:
        fixed = ShipDataset(dataset_root, entry_ids, stride=encode_stride)
        states = {entry_id: encode_sample(model, sample, device)
                  for entry_id, sample in zip(fixed.entry_ids, fixed)}

    results = {}
    for stride in strides:
        queries = ShipDataset(dataset_root, entry_ids, stride=stride)
        inputs = fixed if fixed is not None else queries
        results[stride] = {
            entry_id: evaluate_entry(model, normalizer, sample, device, chunk,
                                     query_sample=query_sample,
                                     encoded=states.get(entry_id))
            for entry_id, sample, query_sample in zip(queries.entry_ids, inputs, queries)}
    return results
