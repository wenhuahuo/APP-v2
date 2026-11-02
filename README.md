# APPSolver-v2

APPSolver-v2 是面向船舶静水航行稳态自由液面流场的分辨率无关神经算子。方法按物理尺寸
动态划分 Patch，以求积积分形成 token，并通过坐标查询解码器在任意格点密度上预测流场。

## 数据集

ShipBenchv2 由 OpenFOAM 稳态算例构建，包含：

- 11 个经典船型的多 Fr 工况；
- 经典船型两两加权插值得到的融合船型单 Fr 工况；
- 自由液面五通道标签，以及用于几何编码和扩展预测的船体湿表面点云与压力标签。

数据集位于 `dataset/shipbench/`，字段约定见 `AGENT.md`。该目录不入 Git。

## 默认模型

默认配置：`configs/appsolver_v2/appsolver_v2_flow_with_hull_random_seed0.yaml`。

| 组件 | 设置 |
| --- | --- |
| 几何输入 | 自由液面坐标 + 船体表面 UV、坐标和法向 |
| 工况融合 | 逐层 FiLM，输入 `(Fr, log10 Re)` |
| 分词 | 距船分档的自适应 Patch |
| 输出 | 自由液面 `eta / u / v / w / p_rgh` |
| 船体压力监督 | 关闭，`lambda_hull: 0.0` |

船体压力预测保留为下游与消融功能，对应
`configs/appsolver_v2/appsolver_v2_joint_flow_hull_random_seed0.yaml`。

## 训练与评测

使用 `/opt/miniconda3/envs/mesh` 环境，并确保配置引用的归一化统计量已由训练集生成。

```bash
python3 scripts/train/train_model.py \
  --config configs/appsolver_v2/appsolver_v2_flow_with_hull_random_seed0.yaml

python3 scripts/eval/evaluate_model.py \
  --config configs/appsolver_v2/appsolver_v2_flow_with_hull_random_seed0.yaml \
  --checkpoint best.pt --strides 1 2 4 8

python3 scripts/eval/profile_model_flops.py
python3 -m pytest tests
```

产物写入 `output/<配置文件名>/`，包括 `best.pt`、`last.pt` 和对应指标。集群训练脚本位于
`scripts/train/*.sbatch`，配置路径按模型组织：

```text
configs/
├── appsolver_v2/
├── appsolver_oformer/
├── appsolver_v1/
├── gnot/
├── oformer/
├── transolver/
├── upt/
└── splits_multihull.json
```

## 文档

- `AGENT.md`：数据、模型和开发约定
- `TODO.md`：正式实验计划
- `docs/paper/paper_outline.md`：论文贡献与实验大纲
- `docs/three_split_evaluation.md`：多模型与多划分结果
- `docs/framework_ablation.md`：船体融合机制消融
- `docs/partition_ablation.md`：Patch 分词消融
- `docs/conditioning_ablation.md`：工况融合消融
