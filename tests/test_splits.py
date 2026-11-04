"""共享划分 `configs/splits_multihull.json` 的生成逻辑与提交产物校验。

判别点：

- 生成器只依赖当前快照的元数据（`index.json` + `meta.json`），同一批元数据必然给出
  同一份划分，且不依赖外部数据集仓库的算例配置；
- 三个划分都满足「train/val 互不重叠 + 恰好覆盖全部 normal_mesh 条目」；
- `lofo_fr_design` 只留出 11 个经典船型的设计 Fr（融合船型条目全部进训练集），
  `loto_kvlcc2` 整船型留出 kvlcc2，`random_seed0` 保持 `RandomState(0)` 抽 15% 的既定语义；
- 提交的划分文件与生成规则一致（含随机划分可从条目全集重算）。
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import make_splits  # noqa: E402
from make_splits import (CLASSIC_HULLS, DESIGN_FR_SCALE, HOLDOUT_HULL,  # noqa: E402
                         RANDOM_FRACTION, RANDOM_SEED, load_entries,
                         parse_fusion_hull, recover_design_frs, split_rows)
from make_conditioning_ablation_configs import lofo_fr_classic_split  # noqa: E402

SPLIT_NAMES = ("random_seed0", "loto_kvlcc2", "lofo_fr_design")
DESIGN_FRS = {hull: round(0.2 + 0.01 * index, 4) for index, hull in enumerate(CLASSIC_HULLS)}
CLASSIC_SCALES = (0.6, 0.8, DESIGN_FR_SCALE, 1.2, 1.4, 1.6)
fusion_hulls = {f"{first}_{second}_w{weight:03d}"
                for index, first in enumerate(CLASSIC_HULLS)
                for second in CLASSIC_HULLS[index + 1:]
                for weight in make_splits.FUSION_WEIGHTS}


def synthetic_rows(fusion_noise=0.0):
    """按数据契约造 11 经典船型 × 6 档 Fr + 全部两两融合船型设计工况。"""
    rows = [(f"{hull}_fr{round(DESIGN_FRS[hull] * scale, 4)}_normal_mesh", hull,
             round(DESIGN_FRS[hull] * scale, 4))
            for hull in CLASSIC_HULLS for scale in CLASSIC_SCALES]
    for first_index, first in enumerate(CLASSIC_HULLS):
        for second in CLASSIC_HULLS[first_index + 1:]:
            for weight in make_splits.FUSION_WEIGHTS:
                fraction = weight / 100
                fr = ((1 - fraction) * DESIGN_FRS[first] + fraction * DESIGN_FRS[second])
                hull = f"{first}_{second}_w{weight:03d}"
                rows.append((f"{hull}_fr{round(fr, 4):.3f}_normal_mesh", hull,
                             round(fr + fusion_noise, 4)))
    return sorted(rows)


class MetadataReaderTest(unittest.TestCase):
    def test_rejects_index_entries_without_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.json").write_text(
                '{"entries": {"missing_normal_mesh": {"hull": "kcs", "Fr": 0.26,'
                ' "mesh_level": "normal_mesh"}}}')
            with self.assertRaises(FileNotFoundError):
                load_entries(root)

    def test_rejects_meta_that_contradicts_the_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.json").write_text(
                '{"entries": {"entry_normal_mesh": {"hull": "kcs", "Fr": 0.26,'
                ' "mesh_level": "normal_mesh"}}}')
            (root / "entry_normal_mesh").mkdir()
            (root / "entry_normal_mesh" / "meta.json").write_text(
                '{"entry_id": "entry_normal_mesh", "hull": "dtc", "Fr": 0.26,'
                ' "mesh_level": "normal_mesh"}')
            with self.assertRaises(ValueError):
                load_entries(root)

    def test_skips_sparse_mesh_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.json").write_text(
                '{"entries": {"kcs_fr026_sparse_mesh": {"hull": "kcs", "Fr": 0.26,'
                ' "mesh_level": "sparse_mesh"},'
                ' "kcs_fr026_normal_mesh": {"hull": "kcs", "Fr": 0.26,'
                ' "mesh_level": "normal_mesh"}}}')
            (root / "kcs_fr026_normal_mesh").mkdir()
            (root / "kcs_fr026_normal_mesh" / "meta.json").write_text(
                '{"entry_id": "kcs_fr026_normal_mesh", "hull": "kcs", "Fr": 0.26,'
                ' "mesh_level": "normal_mesh"}')
            self.assertEqual(load_entries(root), [("kcs_fr026_normal_mesh", "kcs", 0.26)])


class FusionHullTest(unittest.TestCase):
    def test_parses_nested_classic_names(self):
        self.assertEqual(parse_fusion_hull("npl_round_bilge_npl_round_bilge_4a_w025"),
                         ("npl_round_bilge", "npl_round_bilge_4a", 0.25))
        self.assertEqual(parse_fusion_hull("containership_workboat_w075"),
                         ("containership", "workboat", 0.75))

    def test_classic_hull_is_not_a_fusion_hull(self):
        for hull in CLASSIC_HULLS:
            self.assertIsNone(parse_fusion_hull(hull))

    def test_rejects_unknown_weight(self):
        self.assertIsNone(parse_fusion_hull("kcs_dtc_w050x"))


class DesignFrRecoveryTest(unittest.TestCase):
    def test_recovers_design_frs_from_fusion_fr_interpolation(self):
        rows = synthetic_rows()
        self.assertEqual(recover_design_frs(rows), DESIGN_FRS)

    def test_rejects_inconsistent_fusion_fr(self):
        with self.assertRaises(ValueError):
            recover_design_frs(synthetic_rows(fusion_noise=0.01))


class SplitRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.splits, _ = split_rows(cls.rows)

    def test_is_reproducible(self):
        again, _ = split_rows(synthetic_rows())
        self.assertEqual(again, self.splits)

    def test_every_split_is_disjoint_and_covers_all_entries(self):
        universe = {entry for entry, _, _ in self.rows}
        for name in SPLIT_NAMES:
            train = set(self.splits[name]["train"])
            val = set(self.splits[name]["val"])
            with self.subTest(split=name):
                self.assertEqual(train & val, set())
                self.assertEqual(train | val, universe)
                self.assertEqual(self.splits[name]["train"], sorted(train))
                self.assertEqual(self.splits[name]["counts"],
                                 {"train": len(train), "val": len(val)})

    def test_lofo_holds_out_only_the_classic_design_fr_cases(self):
        split = self.splits["lofo_fr_design"]
        hull_of = {entry: hull for entry, hull, _ in self.rows}
        # 档位（Fr / 设计 Fr）才是留出的单位；不同船型的 Fr 数值可能碰巧相等，
        # 例如融合船型的设计 Fr 会落在某经典船型的非设计 Fr 上（真实快照同样存在）。
        scales = make_splits.fr_scales(self.rows, DESIGN_FRS)
        self.assertEqual({scales[entry] for entry in split["val"]}, {DESIGN_FR_SCALE})
        self.assertEqual({hull_of[entry] for entry in split["val"]}, set(CLASSIC_HULLS))
        self.assertEqual(len(split["val"]), len(CLASSIC_HULLS))
        # 训练集：经典船型的非设计 Fr + 全部融合船型（融合船型只有设计 Fr 一个工况）
        classic_train = [entry for entry in split["train"]
                         if hull_of[entry] in CLASSIC_HULLS]
        self.assertEqual({scales[entry] for entry in classic_train},
                         set(CLASSIC_SCALES) - {DESIGN_FR_SCALE})
        self.assertEqual({hull_of[entry] for entry in split["train"]} - set(CLASSIC_HULLS),
                         fusion_hulls)

    def test_loto_holds_out_one_whole_hull(self):
        split = self.splits["loto_kvlcc2"]
        hull_of = {entry: hull for entry, hull, _ in self.rows}
        self.assertEqual({hull_of[entry] for entry in split["val"]}, {HOLDOUT_HULL})
        self.assertNotIn(HOLDOUT_HULL, {hull_of[entry] for entry in split["train"]})
        self.assertEqual(len(split["val"]), len(CLASSIC_SCALES))

    def test_random_keeps_the_established_seed_and_fraction(self):
        split = self.splits["random_seed0"]
        entries = sorted(entry for entry, _, _ in self.rows)
        order = np.random.RandomState(RANDOM_SEED).permutation(len(entries))
        holdout = order[:round(RANDOM_FRACTION * len(entries))]
        self.assertEqual(split["val"], sorted(entries[index] for index in holdout))

    def test_rejects_val_that_breaks_the_holdout_semantics(self):
        split = dict(self.splits["loto_kvlcc2"])
        split["train"] = split["train"] + [split["val"][0]]
        with self.assertRaises(ValueError):
            make_splits.validate_split(
                "loto_kvlcc2", self.rows, {entry: 0.8 for entry, _, _ in self.rows}, split)


class CommittedSplitTest(unittest.TestCase):
    """提交的划分文件本身也必须满足同一套契约（无需数据集即可审计）。"""

    @classmethod
    def setUpClass(cls):
        import json
        cls.document = json.loads((ROOT / "configs" / "splits_multihull.json").read_text())
        cls.splits = {name: cls.document[name] for name in SPLIT_NAMES}
        cls.universe = set(cls.splits["random_seed0"]["train"]) | set(
            cls.splits["random_seed0"]["val"])

    def test_entries_are_disjoint_and_cover_one_universe(self):
        for name in SPLIT_NAMES:
            train = set(self.splits[name]["train"])
            val = set(self.splits[name]["val"])
            with self.subTest(split=name):
                self.assertEqual(train & val, set())
                self.assertEqual(train | val, self.universe)
                self.assertTrue(val)

    def test_counts_match_the_documented_snapshot(self):
        # 224 条 normal_mesh：66 条经典船型 × 6 档 Fr + 158 条融合船型设计工况
        self.assertEqual(len(self.universe), 224)
        self.assertEqual(self.splits["random_seed0"]["counts"], {"train": 190, "val": 34})
        self.assertEqual(self.splits["loto_kvlcc2"]["counts"], {"train": 218, "val": 6})
        self.assertEqual(self.splits["lofo_fr_design"]["counts"], {"train": 213, "val": 11})

    def test_random_split_is_the_documented_permutation(self):
        entries = sorted(self.universe)
        order = np.random.RandomState(RANDOM_SEED).permutation(len(entries))
        holdout = order[:round(RANDOM_FRACTION * len(entries))]
        self.assertEqual(self.splits["random_seed0"]["val"],
                         sorted(entries[index] for index in holdout))

    def test_loto_val_is_the_kvlcc2_hull(self):
        is_heldout_hull = re.compile(rf"^{HOLDOUT_HULL}_fr\d{{3}}_normal_mesh$")
        self.assertTrue(all(is_heldout_hull.match(entry)
                            for entry in self.splits["loto_kvlcc2"]["val"]))
        # 以 kvlcc2 为母船的融合船型属于其它船型，可以留在训练集
        self.assertFalse(any(is_heldout_hull.match(entry)
                             for entry in self.splits["loto_kvlcc2"]["train"]))

    def test_lofo_train_keeps_all_fusion_and_classic_non_design_cases(self):
        split = self.splits["lofo_fr_design"]
        fusion = re.compile(r"_w(?:025|050|075)_")
        # 留出集：11 条经典船型设计工况，不含任何融合船型
        self.assertEqual(len(split["val"]), len(CLASSIC_HULLS))
        self.assertEqual(len({entry.split("_fr")[0] for entry in split["val"]}),
                         len(CLASSIC_HULLS))
        self.assertFalse(any(fusion.search(entry) for entry in split["val"]))
        # 训练集：158 条融合船型 + 55 条经典船型非设计 Fr
        self.assertEqual(len(split["train"]), 213)
        self.assertEqual(sum(1 for entry in split["train"] if fusion.search(entry)), 158)
        self.assertEqual(sum(1 for entry in split["train"] if not fusion.search(entry)), 55)

    def test_conditioning_ablation_split_follows_the_shared_snapshot(self):
        train, val = lofo_fr_classic_split()
        self.assertEqual(len(val), len(CLASSIC_HULLS))
        self.assertEqual(len(train), len(self.universe) - len(val))
        self.assertEqual(set(train) & set(val), set())


if __name__ == "__main__":
    unittest.main()
