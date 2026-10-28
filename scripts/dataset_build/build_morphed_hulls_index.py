"""汇总全部融合船型的元信息成一张目录表 `dataset/morphed_hulls/morphed_hulls_index.csv`。

每一行是一个融合船型（55 对 × 3 权重 = 165 行）：

    hull,pair,w,L_ref_m,T_m,half_beam_m,depth_m,displacement_m3,design_Fr,U_m_s,Re

几何量直接取各对 `morph_meta.json` 的 `cases` 条目（`write_pair` 写出的口径），
设计工况由两母船的设计 Fr 按权重插值、再用融合船的 `L_ref_m` 反推：

    design_Fr = (1 - w)·Fr_a + w·Fr_b
    U_inf     = design_Fr·sqrt(g·L_ref_m)
    Re        = U_inf·L_ref_m / nu

母船的设计 Fr / nu 取自 `dataset/classic_hulls/manifest.json` 的 `design_entry`
（即该船型进数据集的条目）的 `meta.json`。11 个母船的运动黏度都相同（1.27e-6 m²/s），
nu 也按权重插值只是为了让这条公式对将来的混合介质仍然成立。这张表只是算例生产
的检索目录，不参与融合计算。

用法:

    python3 scripts/dataset_build/build_morphed_hulls_index.py
    python3 scripts/dataset_build/build_morphed_hulls_index.py --dry-run
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 重力加速度（与母船算例一致：U_inf = Fr·sqrt(g·Lpp)）
GRAVITY = 9.81
COLUMNS = ("hull", "pair", "w", "L_ref_m", "T_m", "half_beam_m", "depth_m",
           "displacement_m3", "design_Fr", "U_m_s", "Re")


def design_condition(manifest, dataset, name):
    """母船的设计 Fr 与运动黏度（来自它进数据集的那条条目）。"""
    entry = manifest[name]["design_entry"]
    meta = json.loads((dataset / "shipbench" / entry / "meta.json").read_text())
    return {"Fr": float(meta["Fr"]), "nu": float(meta["nu"])}


def build_rows(dataset, output):
    manifest = json.loads((dataset / "classic_hulls" / "manifest.json").read_text())
    condition = {name: design_condition(manifest, dataset, name) for name in manifest}
    rows = []
    for meta_path in sorted(output.glob("*/morph_meta.json")):
        pair = meta_path.parent.name
        summary = json.loads(meta_path.read_text())
        name_a, name_b = summary["hull_a"], summary["hull_b"]
        for stl_name, meta in summary.get("cases", {}).items():
            weight = float(meta["weight"])
            l_ref = float(meta["L_ref_m"])
            fr = ((1.0 - weight) * condition[name_a]["Fr"]
                  + weight * condition[name_b]["Fr"])
            speed = fr * math.sqrt(GRAVITY * l_ref)
            reynolds = (speed * l_ref
                        / ((1.0 - weight) * condition[name_a]["nu"]
                           + weight * condition[name_b]["nu"]))
            rows.append({"hull": Path(stl_name).stem, "pair": pair,
                         "w": weight, "L_ref_m": l_ref, "T_m": meta["T_m"],
                         "half_beam_m": meta["half_beam_m"],
                         "depth_m": meta["depth_m"],
                         "displacement_m3": meta["displacement_m3"],
                         "design_Fr": fr, "U_m_s": speed, "Re": round(reynolds)})
    rows.sort(key=lambda row: (row["pair"], row["w"]))
    return rows


def format_row(row):
    """数值用 6 位有效数字（与原表一致），Re 写成整数。"""
    out = {}
    for key in COLUMNS:
        value = row[key]
        out[key] = value if isinstance(value, str) else (
            f"{value:d}" if key == "Re" else f"{value:.6g}")
    return out


def main():
    parser = argparse.ArgumentParser(description="融合船型目录表（morphed_hulls_index.csv）")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--output", default=str(ROOT / "dataset" / "morphed_hulls"))
    parser.add_argument("--csv", default=None, help="默认 <output>/morphed_hulls_index.csv")
    parser.add_argument("--dry-run", action="store_true", help="只打印前几行")
    args = parser.parse_args()

    dataset, output = Path(args.dataset), Path(args.output)
    path = Path(args.csv) if args.csv else output / "morphed_hulls_index.csv"
    rows = build_rows(dataset, output)
    if len(rows) != 165:
        print(f"注意：目录表里有 {len(rows)} 行（规划 165 个融合船型）")
    if args.dry_run:
        for row in rows[:5]:
            print(format_row(row))
        print(f"（dry-run：{len(rows)} 行，未写文件）")
        return 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))
    print(f"写出 {path}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
