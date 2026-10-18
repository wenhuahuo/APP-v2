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

# 2. 本地：同步导出包到 dataset/.build_cache/
bash scripts/data_raw_export/sync_case.sh kcs_normal_mesh_25s

# 3. 组装数据集条目并刷新 dataset/index.json
bash scripts/dataset_build/build_dataset.sh kcs_normal_mesh_25s kcs_sparse_mesh_25s

# 4. 校验条目
python3 src/dataset/validate_entry.py

# 5. 可视化
python3 vis/kcs_flow_and_press/plot_flow_and_pressure.py
python3 vis/kcs_hull_uv/plot_hull_uv.py
python3 vis/kcs_flow_sdf/plot_flow_sdf.py
```

## 当前条目

| 条目 | 来源算例 | 单元数 | 流场格点 | 湿表面点 | 数据量 |
| --- | --- | ---: | ---: | ---: | ---: |
| `kcs_fr026_normal_mesh` | `kcs_normal_mesh_25s` | 1,261,936 | 103,079 | 17,390 | 11 MB |
| `kcs_fr026_sparse_mesh` | `kcs_sparse_mesh_25s` | 409,871 | 102,921 | 7,977 | 7 MB |

## 依赖

使用 `/opt/miniconda3/envs/mesh` 环境：numpy、vtk、matplotlib、scipy、trimesh（见
`pyproject.toml`），模型阶段使用 torch。
