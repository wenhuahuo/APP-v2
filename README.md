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

## 主实验基线

主实验对比 9 个模型，口径完全一致（自由液面 + 船体表面输入、仅流场输出、逐层 FiLM、
`lambda_hull=0.0`）：APPSolver-v2、APP+OFormer、APPSolver-v1、GNOT、UPT、Transolver、
OFormer、GINO、CORAL。覆盖面 `random_seed0` / `loto_kvlcc2` / `lofo_fr_design` 三个
划分，配置由 `scripts/train/make_split_configs.py` 从各模型
`configs/<模型>/<模型>_multihull_isoflops.yaml` 派生（固定 GFLOPs 容量、训练种子 0/1/2）。
训练时每个 epoch 遍历全部训练条目各一次（随机顺序，一条目一步），`train_strides: [1]`
表示官方 run 只用全密度输入，评测也只在 stride 1 上做。

新基线按论文路线实现：GINO 为「输入几何点云 → GNO 半径核积分到潜规则网格 → FNO →
GNO 解码到任意查询点」，CORAL 为「点集编码到 latent → latent 调制的 SIREN 坐标神经场
解码任意查询点」，UPT 的 `decoder(x, query_x, query_fx)` 在查询坐标上独立解码。实现
依据与口径见 `docs/main_experiment_protocol.md`。

## 训练与评测

使用 `/opt/miniconda3/envs/mesh` 环境，并确保配置引用的归一化统计量已由训练集生成。

```bash
python3 scripts/train/train_model.py \
  --config configs/appsolver_v2/appsolver_v2_flow_with_hull_random_seed0.yaml

python3 scripts/eval/evaluate_model.py \
  --config configs/appsolver_v2/appsolver_v2_flow_with_hull_random_seed0.yaml \
  --checkpoint best.pt               # 查询密度缺省取配置的 eval.strides
# 多分辨率训练验证的配置（`*_multires_*`）在配置里给出 eval.strides: [1, 2, 4, 8]
# 固定输入编码密度、只改变查询密度（UPT / GINO / CORAL / OFormer / APP 系列）
python3 scripts/eval/evaluate_model.py \
  --config configs/gino/gino_random_seed0.yaml --encode-stride 2

python3 scripts/eval/profile_model_flops.py
python3 scripts/eval/measure_inference_time.py   # 18 个 run × stride 1/2/4/8 单条目延迟
python3 scripts/eval/summarize_cross_resolution.py
python3 -m pytest tests
```

产物写入 `output/<配置文件名>/`，包括 `best.pt`、`last.pt` 和对应指标。集群训练脚本位于
`scripts/train/*.sbatch`，配置路径按模型组织：

```text
configs/
├── appsolver_v2/
├── appsolver_oformer/
├── appsolver_v1/
├── coral/
├── gino/
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
- `docs/main_experiment_protocol.md`：主实验统一口径、基线实现依据、跨分辨率评测与推理时间
- `docs/isoflops_model_complexity.md`：固定 GFLOPs 下的参数量、token 量与计算量对照
- `docs/three_split_evaluation.md`：多模型与多划分结果
- `docs/framework_ablation.md`：船体融合机制消融
- `docs/partition_ablation.md`：Patch 分词消融
- `docs/conditioning_ablation.md`：工况融合消融
