# APPSolver-v2

船舶稳态自由液面流场与船体湿表面压力预测的分辨率无关神经算子。数据集来自 Slurm 集群上的
OpenFOAM 静水航行算例，按 DrivAerNet++ 的思路整理；字段定义与命名规范见 `AGENT.md`，
后续开发路线见 `TODO.md`。

输入 = 半船湿表面 UV 参数域 + 流场格点的符号距离 + 工况参数；
输出 = 二维自由液面场（波高、液面速度、液面动压）+ 船体湿表面压力。
数据集只存点云与格点，不含任何网格连接关系。

## 流程

```bash
# 1. 集群端：foamToVTK 导出算例末态单元场与 hull patch
ssh <cluster> 'bash export_case_vtk.sh ~/Documents/OpenFoamWorks/APPSolver-v2-Dataset/kcs_normal_mesh_25s ~/dataset_export'

# 2. 集群端：导出包留在 ~/dataset_export/，就地组装成条目（不回传笔记本）
PYTHON=/path/to/cluster-home/.conda/envs/mesh/bin/python3 \
  bash scripts/dataset_build/build_dataset.sh kcs_normal_mesh_25s kcs_sparse_mesh_25s

# 3. 校验条目（build_dataset.sh 已逐条校验，这里整批复核）
/path/to/cluster-home/.conda/envs/mesh/bin/python3 src/dataset/validate_entry.py

# 5. 拟合归一化统计量（只含训练集条目，写 dataset/shipbench/stats.json）
python3 scripts/datapipe/build_stats.py --entries kcs_fr026_normal_mesh

# 6. 训练（当前只有一条条目，训练集同时作验证集）
python3 scripts/train/train_appsolver_v2.py --config configs/appsolver_v2_kcs_normal.yaml

# 7. 可视化
python3 vis/kcs_flow_and_press/plot_flow_and_pressure.py
python3 vis/kcs_hull_uv/plot_hull_uv.py
python3 vis/kcs_flow_sdf/plot_flow_sdf.py
```

## 目录布局

`dataset/` 整个目录不入 git，分四块：

| 路径 | 内容 |
| --- | --- |
| `dataset/shipbench/` | 神经算子训练数据集：`index.json`、`stats*.json` 与 `<entry-id>/` 条目目录 |
| `dataset/classic_hulls/` | 11 个经典船型的船体 STL 与 `manifest.json`（船型融合的母船） |
| `dataset/morphed_hulls/` | 融合船型产物 `<a>_<b>/{<a>_<b>_w*.stl,morph_meta.json}`（水密全船 STL） |
| `dataset/.build_cache/` | 集群导出的中间 VTK，重新组装条目时才用 |

## 当前条目

| 条目 | 来源算例 | 单元数 | 流场格点 | 湿表面点 | 数据量 |
| --- | --- | ---: | ---: | ---: | ---: |
| `kcs_fr026_normal_mesh` | `kcs_normal_mesh_25s` | 1,261,936 | 103,079 | 17,390 | 11 MB |
| `kcs_fr026_sparse_mesh` | `kcs_sparse_mesh_25s` | 409,871 | 102,921 | 7,977 | 7 MB |

## 数据加载

`src/datapipe/` 把条目加载成 torch 张量：流场散点回填成 (`nx`, `ny`) 稠密张量，
目标无量纲化后按求积权重做逐通道标准化；查询点集密度用 `stride` 抽样，权重乘以
`stride²`。CFD 只有 normal 一档网格，分辨率只体现在查询点集密度上。

## 模型

`src/app/` 按物理尺寸把点集划成 patch，token 取 patch 内的求积加权平均；
`src/model/` 是共享编码器 + 查询解码器：位置编码可选 Fourier / RoPE，注意力带
求积权重，工况参数用逐层条件调制注入（`model.conditioning`：`none` 不使用条件 /
`film` 默认 / `adaln` / `adaln_zero`）。输出域两个：自由液面 5 通道、船面压力 1 通道，
查询点任意给。

## 依赖

使用 `/opt/miniconda3/envs/mesh` 环境：numpy、vtk、matplotlib、scipy、trimesh（见
`pyproject.toml`），模型阶段使用 torch。
