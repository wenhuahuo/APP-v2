# APPSolver-v2

船舶静水航行 CFD 深度学习数据集：把 Slurm 集群上的 OpenFOAM 算例整理成
DrivAerNet++ 形式的条目，并在此基础上开发代理模型。

输入 = 船舶表面点云 + 流场网格 + 设计吃水初始自由面 + 工况；
输出 = 末态自由面、速度场、动压场、船体表面压力与积分阻力。
字段定义与命名规范见 `AGENT.md`。

## 流程

```bash
# 1. 集群端：foamToVTK 导出算例末态（VTK + 初始 alpha + force.dat）
ssh <cluster> 'bash export_case_vtk.sh ~/Documents/OpenFoamWorks/APPSolver-v2-Dataset/kcs_normal_mesh_25s ~/dataset_export'

# 2. 本地：同步导出包到 dataset/.build_cache/
bash scripts/data_raw_export/sync_case.sh kcs_normal_mesh_25s

# 3. 组装数据集条目并刷新 dataset/index.json
bash scripts/dataset_build/build_dataset.sh kcs_normal_mesh_25s kcs_sparse_mesh_25s

# 4. 可视化
python3 vis/kcs_flow_and_press/plot_flow_and_pressure.py
```

## 当前条目

| 条目 | 来源算例 | 单元数 | R_T (N) | CT×1000 |
| --- | --- | ---: | ---: | ---: |
| `kcs_fr026_normal_mesh` | `kcs_normal_mesh_25s` | 1,261,936 | 91.25 | 3.963 |
| `kcs_fr026_sparse_mesh` | `kcs_sparse_mesh_25s` | 409,871 | 88.57 | 3.847 |

阻力为 15–25 s 窗口 `R_T = 2|Fx|` 的均值（半域算例，`force_scale = 2`）。
参考：EFD Tokyo 2015 Case 2.1 为 85.44 N（CT×1000 = 3.711，带舵）。

## 依赖

numpy、vtk、matplotlib、scipy、trimesh（`pyproject.toml`）。
