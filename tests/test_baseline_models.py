"""基线模型的接口与分词不变量：整片点云进、同一批点出，查询点用格点下标取值。

APPSolver-v1 的可学习位置嵌入按 `max_patches` 建表，patch 数由输入点数决定，因此额外
校验表的容量覆盖全量 `normal_mesh` 快照在 `stride=1` 下的最大 patch 数（历史配置
`max_patches=2048` 在验证到 3132 个 patch 的条目时溢出）。
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from app_v1 import APPSolverV1Baseline, point_patches  # noqa: E402
from build import build_model  # noqa: E402
from collate import collate_samples  # noqa: E402
from gnot import GNOTBaseline  # noqa: E402
from oformer import OFormerBaseline  # noqa: E402
from point_cloud import joint_cloud, lattice_index  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402
from transolver import TransolverBaseline  # noqa: E402
from upt import UPTBaseline  # noqa: E402

FLOW_SHAPE = (17, 9)
HULL_POINTS = 40
QUERIES = 64
DATASET = ROOT / "dataset" / "shipbench"
CONFIG_DIR = ROOT / "configs" / "appsolver_v1"
SPLIT_CONFIGS = ("appsolver_v1_random_seed0", "appsolver_v1_random_seed0_isoflops")
PATCH_SIZE = 64


def fake_batch():
    """一条合成的批处理张量：17×9 规则格点 + 40 个船面点。"""
    nx, ny = FLOW_SHAPE
    x = torch.linspace(-1.0, 0.6, nx)
    y = torch.linspace(-0.4, 0.0, ny)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1)
    sdf = (grid_x ** 2 + grid_y ** 2).sqrt()
    return {
        "flow_coords": coords[None],
        "flow_sdf": sdf[None],
        "flow_extent": torch.tensor([[-1.0, -0.4, 0.6, 0.0]]),
        "condition": torch.zeros(1, 2),
        "hull_uv": torch.rand(1, HULL_POINTS, 2),
        "hull_points": torch.rand(1, HULL_POINTS, 3),
        "hull_normals": torch.rand(1, HULL_POINTS, 3),
        "hull_mask": torch.ones(1, HULL_POINTS, dtype=torch.bool),
        "flow_mask": torch.ones(1, nx, ny, dtype=torch.bool),
        "flow_weight": torch.full((1, nx, ny), (x[1] - x[0]) * (y[1] - y[0])),
        "hull_weight": torch.full((1, HULL_POINTS), 1.0 / HULL_POINTS),
    }


class LatticeIndexTest(unittest.TestCase):
    def test_round_trip_over_full_grid(self):
        batch = fake_batch()
        nx, ny = FLOW_SHAPE
        queries = batch["flow_coords"].reshape(-1, 2)
        index = lattice_index(batch, queries)
        self.assertEqual(tuple(index.shape), (nx * ny,))
        self.assertTrue(torch.equal(index, torch.arange(nx * ny)))


class PointPatchTest(unittest.TestCase):
    def test_every_point_in_exactly_one_patch(self):
        coords = np.random.RandomState(0).rand(500, 2)
        index, mask = point_patches(coords, np.ones(len(coords), dtype=bool), 64)
        used = index[mask]
        self.assertEqual(len(used), len(coords))
        self.assertEqual(len(np.unique(used)), len(coords))
        self.assertTrue(np.all(mask.sum(axis=1) <= 64))

    def test_patch_count_tracks_density(self):
        fine = np.random.RandomState(1).rand(1000, 2)
        coarse = fine[::4]
        _, fine_mask = point_patches(fine, np.ones(len(fine), dtype=bool), 64)
        _, coarse_mask = point_patches(coarse, np.ones(len(coarse), dtype=bool), 64)
        self.assertGreater(len(fine_mask), 4 * len(coarse_mask) - 10)


class PositionTableCapacityTest(unittest.TestCase):
    """位置嵌入表按配置容量建表，满载 patch（P == max_patches）时不越界、不截断。"""

    def test_position_table_covers_its_configured_capacity(self):
        max_patches, patch_size, d_model = 64, 8, 24
        model = APPSolverV1Baseline(patch_size=patch_size, d_model=d_model, nhead=4, layers=1,
                                    mlp_ratio=2, max_patches=max_patches,
                                    input_domains=("flow", "hull"),
                                    output_domains=("flow",), conditioning="film")
        self.assertEqual(tuple(model.flow_patches.position.shape), (1, max_patches, d_model))
        # 联合主干把 patch 内点数 × 每点特征（双域嵌入后的 d_model）展平后投影
        patches = torch.zeros(1, max_patches, patch_size * d_model)
        predicted = model.flow_patches(patches, 0.0, torch.zeros(1, 2))
        self.assertEqual(tuple(predicted.shape), (1, max_patches, patch_size, 5))

    def test_overflowing_patch_count_is_rejected_instead_of_truncated(self):
        """超出容量的 patch 数必须报错：不允许静默截断或动态补齐位置编码。"""
        model = APPSolverV1Baseline(patch_size=8, d_model=24, nhead=4, layers=1,
                                    mlp_ratio=2, max_patches=64,
                                    input_domains=("flow", "hull"),
                                    output_domains=("flow",), conditioning="film")
        patches = torch.zeros(1, 65, 8 * 24)
        with self.assertRaises(RuntimeError):
            model.flow_patches(patches, 0.0, torch.zeros(1, 2))


@unittest.skipUnless((DATASET / "kvlcc2_fr009_normal_mesh" / "meta.json").exists(),
                     "需要完整 ShipBenchv2 快照才能测量最大 patch 数")
class MaxPatchCoverageTest(unittest.TestCase):
    """全量 normal_mesh 在 stride 1 的最大 patch 数必须被位置嵌入覆盖。

    历史上 `max_patches=2048` 在验证 kvlcc2_fr009（3132 个 patch）时抛
    `RuntimeError: The size of tensor a (3132) must match ... (2048)`。
    """

    @classmethod
    def setUpClass(cls):
        index = json.loads((DATASET / "index.json").read_text())
        entries = sorted(entry for entry in index["entries"]
                         if entry.endswith("_normal_mesh"))
        if len(entries) != 224:
            raise unittest.SkipTest(f"期望 224 条 normal_mesh 条目，实际 {len(entries)}")
        worst = (0, None)
        for entry in entries:
            batch = collate_samples([ShipDataset(DATASET, [entry], stride=1)[0]])
            coords = joint_cloud(batch)[2].numpy()
            patches = len(point_patches(coords, [True] * len(coords), PATCH_SIZE)[0])
            if patches > worst[0]:
                worst = (patches, entry)
        cls.worst_patches, cls.worst_entry = worst

    def test_every_entry_fits_the_configured_capacity(self):
        for name in SPLIT_CONFIGS:
            config = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
            with self.subTest(config=name):
                self.assertGreaterEqual(
                    config["model"]["max_patches"], self.worst_patches,
                    f"{name} 的 max_patches 覆盖不了 {self.worst_entry} 的 "
                    f"{self.worst_patches} 个 patch")

    def test_worst_entry_forwards_at_stride_one(self):
        config = yaml.safe_load((CONFIG_DIR / "appsolver_v1_random_seed0.yaml").read_text())
        model = build_model(config).eval()
        sample = ShipDataset(DATASET, [self.worst_entry], stride=1)[0]
        batch = collate_samples([sample])
        queries = batch["flow_coords"].reshape(-1, 2)[:512][None]
        with torch.no_grad():
            output = model(batch, queries)
        self.assertEqual(tuple(output["flow"].shape), (1, 512, 5))
        self.assertTrue(bool(torch.isfinite(output["flow"]).all()))


class BaselineForwardTest(unittest.TestCase):
    def test_outputs_at_query_points(self):
        batch = fake_batch()
        queries = batch["flow_coords"].reshape(-1, 2)[:QUERIES]
        queries = torch.column_stack([queries, batch["flow_sdf"].reshape(-1)[:QUERIES]])[None]
        models = {
            "gnot": GNOTBaseline(d_model=24, nhead=4, layers=1, mlp_ratio=2, n_experts=2),
            "transolver": TransolverBaseline(d_model=24, nhead=4, layers=1, slices=8,
                                             mlp_ratio=2),
            "upt": UPTBaseline(d_model=24, nhead=4, layers=1, latents=16, mlp_ratio=2),
            "appsolver_v1": APPSolverV1Baseline(patch_size=8, d_model=24, nhead=4, layers=1,
                                                mlp_ratio=2, max_patches=64),
            "oformer": OFormerBaseline(d_model=32, nhead=4, encoder_layers=1, decoder_layers=1,
                                      mlp_ratio=2, propagator_layers=1),
        }
        for name, model in models.items():
            output = model(batch, queries)
            self.assertEqual(tuple(output["flow"].shape), (1, QUERIES, 5), name)
            self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1), name)
            output["flow"].sum().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()), name)


class HybridForwardTest(unittest.TestCase):
    """APP 分词 + OFormer 主干：输出与前缀查询点一致，参数全部有梯度。"""

    def test_forward_and_gradients(self):
        from appsolver_oformer import APPSolverOFormer

        batch = fake_batch()
        queries = batch["flow_coords"].reshape(-1, 2)[:QUERIES]
        queries = torch.column_stack([queries, batch["flow_sdf"].reshape(-1)[:QUERIES]])[None]
        model = APPSolverOFormer(flow_size_tiers=[[0.3, 0.2], [None, 0.4]], hull_size=0.3,
                                 d_model=32, nhead=4, encoder_layers=1, decoder_layers=1,
                                 mlp_ratio=2, propagator_layers=1)
        output = model(batch, queries)
        self.assertEqual(tuple(output["flow"].shape), (1, QUERIES, 5))
        self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1))
        (output["flow"].sum() + output["hull"].sum()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
