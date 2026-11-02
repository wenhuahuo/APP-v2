"""UPT 的独立查询解码，以及「固定输入密度 + 独立查询密度」评测协议。

判别点不是输出形状，而是：

- 查询坐标可以取在格点之外（任意实数坐标），输出随查询坐标变化；
- 查询点集变化（子集、置换、增减点数）只改变输出的行数，不改变同一坐标处的取值；
- 编码结果只由输入点云决定，评测换查询密度时编码器只跑一次；
- 把输出留在输入点上的模型不允许固定输入密度评测（否则会静默取错点）。
"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for name in ("datapipe", "app", "baseline", "model", "eval"):
    sys.path.insert(0, str(ROOT / "src" / name))

from gnot import GNOTBaseline  # noqa: E402
from normalize import Normalizer  # noqa: E402
from query_density import density_sweep  # noqa: E402
from ship_dataset import ShipDataset  # noqa: E402
from upt import UPTBaseline  # noqa: E402

DATASET = ROOT / "dataset" / "shipbench"
ENTRY = "kcs_fr026_normal_mesh"
NX, NY = 17, 9
HULL_POINTS = 40


def fake_batch(condition=None):
    """一条合成批次：17×9 规则格点 + 40 个船面点，工况可单独指定。"""
    x = torch.linspace(-1.0, 0.6, NX)
    y = torch.linspace(-0.4, 0.0, NY)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1)
    generator = torch.Generator().manual_seed(0)
    return {
        "flow_coords": coords[None],
        "flow_extent": torch.tensor([[-1.0, -0.4, 0.6, 0.0]]),
        "flow_sdf": (grid_x ** 2 + grid_y ** 2).sqrt()[None],
        "condition": torch.zeros(1, 2) if condition is None else condition,
        "hull_uv": torch.rand(1, HULL_POINTS, 2, generator=generator),
        "hull_points": torch.rand(1, HULL_POINTS, 3, generator=generator),
        "hull_normals": torch.randn(1, HULL_POINTS, 3, generator=generator),
        "hull_mask": torch.ones(1, HULL_POINTS, dtype=torch.bool),
        "flow_mask": torch.ones(1, NX, NY, dtype=torch.bool),
        "flow_weight": torch.ones(1, NX, NY),
        "hull_weight": torch.full((1, HULL_POINTS), 1.0 / HULL_POINTS),
    }


def irregular_queries(count, seed=3):
    """落在格点之间的任意实数坐标，格点下标取值无法复现这些位置。"""
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.rand(count, generator=generator) * 1.6 - 1.0,
                        torch.rand(count, generator=generator) * 0.4 - 0.4], dim=-1)[None]


def small_upt(**kwargs):
    return UPTBaseline(d_model=24, nhead=4, layers=1, latents=16, decoder_layers=1,
                       mlp_ratio=2, **kwargs)


class UPTQueryDecodingTest(unittest.TestCase):
    def setUp(self):
        self.batch = fake_batch()
        self.model = small_upt(input_domains=("flow", "hull"), output_domains=("flow",),
                               conditioning="film")

    def test_queries_lie_off_the_input_lattice(self):
        queries = irregular_queries(11)
        output = self.model(self.batch, queries)
        self.assertEqual(tuple(output["flow"].shape), (1, 11, 5))
        self.assertTrue(torch.isfinite(output["flow"]).all())

        # 同一坐标上的取值不随查询点集的规模变化：编码与解码都与查询数量无关。
        subset = self.model(self.batch, queries[:, :4])
        torch.testing.assert_close(subset["flow"], output["flow"][:, :4])

    def test_decode_is_independent_of_query_order(self):
        queries = irregular_queries(9)
        order = torch.tensor([4, 0, 8, 2, 6, 1, 7, 3, 5])
        direct = self.model(self.batch, queries)
        permuted = self.model(self.batch, queries[:, order])
        torch.testing.assert_close(permuted["flow"], direct["flow"][:, order])

    def test_forward_equals_encode_then_decode(self):
        queries = irregular_queries(6)
        state = self.model.encode(self.batch)
        decoded = self.model.decode(state, queries)
        torch.testing.assert_close(self.model(self.batch, queries)["flow"], decoded)

    def test_output_changes_with_query_coordinates(self):
        queries = irregular_queries(5)
        shifted = queries + torch.tensor([0.05, 0.0])
        difference = (self.model(self.batch, queries)["flow"]
                      - self.model(self.batch, shifted)["flow"]).abs().max()
        self.assertGreater(float(difference), 0.0)

    def test_condition_and_hull_geometry_reach_the_output(self):
        queries = irregular_queries(5)
        baseline = self.model(self.batch, queries)["flow"]
        conditioned = self.model(fake_batch(torch.tensor([[0.32, 9.1]])), queries)["flow"]
        self.assertGreater(float((conditioned - baseline).abs().max()), 0.0)

        batch = fake_batch()
        batch["hull_points"] = batch["hull_points"] + 0.1
        moved = self.model(batch, queries)["flow"]
        self.assertGreater(float((moved - baseline).abs().max()), 0.0)

    def test_latents_do_not_depend_on_queries(self):
        """编码只吃输入点云：两种查询点集下 latent 完全一致。"""
        state = self.model.encode(self.batch)
        self.model(self.batch, irregular_queries(3))
        again = self.model.encode(self.batch)
        torch.testing.assert_close(state["latent"], again["latent"])

    def test_legacy_hull_output_still_uses_query_features(self):
        legacy = small_upt()
        output = legacy(self.batch, irregular_queries(4))
        self.assertEqual(tuple(output["flow"].shape), (1, 4, 5))
        self.assertEqual(tuple(output["hull"].shape), (1, HULL_POINTS, 1))


class CountingModel(torch.nn.Module):
    """记录编码次数与每次编码看到的输入点云密度；解码返回零场。"""

    supports_query_decoder = True
    output_domains = ("flow",)

    def __init__(self):
        super().__init__()
        self.encodes = 0
        self.input_points = []
        self.query_points = []

    def encode(self, batch):
        self.encodes += 1
        self.input_points.append(batch["flow_coords"].shape[1:3])
        return batch["flow_coords"].reshape(1, -1, 2)

    def decode(self, state, queries):
        self.query_points.append(queries.shape[1])
        return torch.zeros(1, queries.shape[1], 5)


def normalizer():
    channels = {"mean": [0.0] * 5, "std": [1.0] * 5, "constant": [False] * 5}
    return Normalizer(flow=channels, hull={"mean": [0.0], "std": [1.0], "constant": [False]},
                      condition={"mean": [0.0, 0.0], "std": [1.0, 1.0]})


@unittest.skipUnless((DATASET / ENTRY / "meta.json").exists(), "数据集条目不存在")
class EncodeStrideProtocolTest(unittest.TestCase):
    def test_encoder_runs_once_per_entry_when_input_density_is_fixed(self):
        model = CountingModel()
        density_sweep(model, normalizer(), DATASET, [ENTRY], (1, 2, 4), "cpu", 4096,
                      encode_stride=2)
        # 一条条目、三个查询密度，但编码只发生一次，且输入始终是 stride=2 的格点。
        self.assertEqual(model.encodes, 1)
        coarse = ShipDataset(DATASET, [ENTRY], stride=2).flow_coords.shape[:2]
        self.assertEqual([tuple(shape) for shape in model.input_points], [tuple(coarse)])

    def test_query_density_follows_the_strides(self):
        model = CountingModel()
        density_sweep(model, normalizer(), DATASET, [ENTRY], (1, 4), "cpu", 4096,
                      encode_stride=2)
        fine, coarse = (ShipDataset(DATASET, [ENTRY], stride=stride)[0] for stride in (1, 4))
        self.assertEqual(sum(model.query_points),
                         int(fine["flow_mask"].sum()) + int(coarse["flow_mask"].sum()))

    def test_without_encode_stride_the_input_follows_the_query_density(self):
        model = CountingModel()
        density_sweep(model, normalizer(), DATASET, [ENTRY], (1, 2), "cpu", 4096)
        self.assertEqual(model.encodes, 2)
        expected = [tuple(ShipDataset(DATASET, [ENTRY], stride=stride).flow_coords.shape[:2])
                    for stride in (1, 2)]
        self.assertEqual([tuple(shape) for shape in model.input_points], expected)

    def test_lattice_models_reject_fixed_encode_density(self):
        model = GNOTBaseline(d_model=24, nhead=4, layers=1, mlp_ratio=2, n_experts=2)
        with self.assertRaises(ValueError):
            density_sweep(model, normalizer(), DATASET, [ENTRY], (1, 2), "cpu", 4096,
                          encode_stride=2)


if __name__ == "__main__":
    unittest.main()
