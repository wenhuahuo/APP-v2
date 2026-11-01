"""评测控制台汇总：条目间 mean±总体 std、六个量的单位与 R² 的符号。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))

from metrics import FLOW_CHANNELS, HULL_CHANNELS  # noqa: E402
from report import COLUMNS, FLOW_COLUMNS, summary_lines, table_lines  # noqa: E402

ENTRIES = ("kcs_a", "kcs_b")
STRIDES = (1, 2)


def block(flow_r2, flow_rmse, hull_r2, hull_rmse):
    return {"flow": {"r2": flow_r2, "rmse": flow_rmse},
            "hull": {"r2": [hull_r2], "rmse": [hull_rmse]}}


def blocks():
    """两个条目：均值取中间值，总体标准差为极差的一半。"""
    return {
        1: {
            "kcs_a": block([0.9, 0.5, 0.4, 0.3, 0.99], [0.001, 0.01, 0.02, 0.03, 0.05],
                           0.8, 0.01),
            "kcs_b": block([0.7, 0.1, 0.2, 0.1, 0.97], [0.003, 0.03, 0.04, 0.05, 0.07],
                           0.6, 0.03),
        },
        2: {
            "kcs_a": block([0.8, 0.4, 0.3, 0.2, 0.98], [0.002, 0.02, 0.03, 0.04, 0.06],
                           0.7, 0.02),
            "kcs_b": block([0.6, 0.0, 0.1, 0.0, 0.96], [0.004, 0.04, 0.05, 0.06, 0.08],
                           0.5, 0.04),
        },
    }


def cells(line):
    return line.split()


class TableTest(unittest.TestCase):
    def test_columns_cover_every_metric_channel(self):
        self.assertEqual(FLOW_COLUMNS, len(FLOW_CHANNELS))
        self.assertEqual(len(COLUMNS), len(FLOW_CHANNELS) + len(HULL_CHANNELS))

    def test_one_row_per_stride_in_given_order(self):
        lines = table_lines("val", ENTRIES, blocks(), (2, 1), "rmse")
        rows = lines[2:]
        self.assertEqual([cells(row)[0] for row in rows], ["2", "1"])
        self.assertEqual(len(rows), len(STRIDES))

    def test_r2_mean_keeps_sign_and_uses_population_std(self):
        lines = table_lines("val", ENTRIES, blocks(), STRIDES, "r2")
        self.assertIn("R²↑", lines[0])
        eta, u, v, w, cp, hull = cells(lines[2])[1:]
        self.assertEqual(eta, "+0.8000±0.1000")
        self.assertEqual(u, "+0.3000±0.2000")
        self.assertEqual(v, "+0.3000±0.1000")
        self.assertEqual(w, "+0.2000±0.1000")
        self.assertEqual(cp, "+0.9800±0.0100")
        self.assertEqual(hull, "+0.7000±0.1000")

    def test_rmse_reports_eta_in_mm_and_six_channels(self):
        lines = table_lines("val", ENTRIES, blocks(), STRIDES, "rmse")
        header = lines[1]
        for label in ("eta[mm]", "u[m/s]", "v[m/s]", "w[m/s]", "Cp[-]", "hull Cp[-]"):
            self.assertIn(label, header)
        eta, u, v, w, cp, hull = cells(lines[2])[1:]
        # 总体标准差：极差 0.002 m 的一半是 0.001 m，换 mm 后 1.0000。
        self.assertEqual(eta, "2.0000±1.0000")
        self.assertEqual(u, "0.0200±0.0100")
        self.assertEqual(v, "0.0300±0.0100")
        self.assertEqual(w, "0.0400±0.0100")
        self.assertEqual(cp, "0.0600±0.0100")
        self.assertEqual(hull, "0.0200±0.0100")

    def test_summary_prints_r2_then_rmse_per_split(self):
        lines = summary_lines({"train": ENTRIES, "val": ENTRIES},
                              {"train": blocks(), "val": blocks()}, STRIDES)
        titles = [line.strip() for line in lines if "↑" in line or "↓" in line]
        self.assertEqual(len(titles), 4)
        self.assertTrue(titles[0].startswith("train") and "R²↑" in titles[0])
        self.assertTrue(titles[1].startswith("train") and "RMSE↓" in titles[1])
        self.assertTrue(titles[2].startswith("val") and "R²↑" in titles[2])
        self.assertTrue(titles[3].startswith("val") and "RMSE↓" in titles[3])


if __name__ == "__main__":
    unittest.main()
