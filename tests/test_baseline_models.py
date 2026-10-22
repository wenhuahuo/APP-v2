"""基线模型的接口与分词不变量：整片点云进、同一批点出，查询点用格点下标取值。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from app_v1 import APPSolverV1Baseline, point_patches  # noqa: E402
from gnot import GNOTBaseline  # noqa: E402
from oformer import OFormerBaseline  # noqa: E402
from point_cloud import lattice_index  # noqa: E402
from transolver import TransolverBaseline  # noqa: E402
from upt import UPTBaseline  # noqa: E402

FLOW_SHAPE = (17, 9)
HULL_POINTS = 40
QUERIES = 64


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


class SliceHybridForwardTest(unittest.TestCase):
    """APP 分词 + Transolver slice 主干：形状与前缀查询点一致，参数全部有梯度。"""

    def test_forward_and_gradients(self):
        from appsolver_transolver import APPSolverTransolver

        batch = fake_batch()
        queries = batch["flow_coords"].reshape(-1, 2)[:QUERIES]
        queries = torch.column_stack([queries, batch["flow_sdf"].reshape(-1)[:QUERIES]])[None]
        model = APPSolverTransolver(flow_size_tiers=[[0.3, 0.2], [None, 0.4]], hull_size=0.3,
                                    d_model=32, nhead=4, encoder_layers=1, decoder_layers=1,
                                    mlp_ratio=2, slices=8)
        output = model(batch, queries)
        self.assertEqual(tuple(output["flow"].shape), (1, QUERIES, 5))
        self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1))
        (output["flow"].sum() + output["hull"].sum()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
