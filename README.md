# APPSolver-v2

船舶稳态流场与船体表面压力预测的分辨率无关神经算子。数据集来自 Slurm 集群上的
OpenFOAM 静水航行算例，按 DrivAerNet++ 的思路整理；字段定义与命名规范见 `AGENT.md`。

输入 = 船舶表面点云 + 流场单元中心 + 设计吃水初始自由面 + 工况；
输出 = 末态自由面、速度场、动压场与船体表面压力。数据集只存点云坐标，
不含任何网格连接关系。

## 流程

```bash
# 1. 集群端：foamToVTK 导出算例末态（VTK + 初始 alpha）
ssh <cluster> 'bash export_case_vtk.sh ~/Documents/OpenFoamWorks/APPSolver-v2-Dataset/kcs_normal_mesh_25s ~/dataset_export'

# 2. 本地：同步导出包到 dataset/.build_cache/
bash scripts/data_raw_export/sync_case.sh kcs_normal_mesh_25s

# 3. 组装数据集条目并刷新 dataset/index.json
bash scripts/dataset_build/build_dataset.sh kcs_normal_mesh_25s kcs_sparse_mesh_25s

# 4. 可视化
python3 vis/kcs_flow_and_press/plot_flow_and_pressure.py
```

## 当前条目

| 条目 | 来源算例 | 单元数 | 船体表面点 | 数据量 |
| --- | --- | ---: | ---: | ---: |
| `kcs_fr026_normal_mesh` | `kcs_normal_mesh_25s` | 1,261,936 | 37,184 | 44 MB |
| `kcs_fr026_sparse_mesh` | `kcs_sparse_mesh_25s` | 409,871 | 17,086 | 14 MB |

## 依赖

numpy、vtk、matplotlib、scipy、trimesh（`pyproject.toml`）。
