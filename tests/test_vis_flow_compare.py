"""两张论文图的判别点：共享色标范围、条目来自对应留出集、模型清单完整。

不跑推理（检查点在集群、数据集不入 git），只验证
- `row_limits` 由该行真值与预测合并计算，且误差列使用同一对 vmin/vmax；
- `plot_fields` 能为三行三列出图，返回的范围与 `row_limits` 逐行一致；
- 两个出图脚本的模型清单、配置与固定条目核对：7 个模型、5 个模型、条目在对应验证集里。
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vis"))

from common import flow_compare

SPLITS_FILE = ROOT / "configs" / "splits_multihull.json"
NX, NY, LPP = 24, 8, 7.0


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN = load_script(ROOT / "vis" / "main_experiment" / "plot_main_experiment.py")
THREE = load_script(ROOT / "vis" / "three_split_eval" / "plot_three_split_eval.py")

MAIN_MODELS = {"appsolver_oformer", "transolver", "appsolver_v2", "gnot", "upt",
               "oformer", "appsolver_v1"}
THREE_MODELS = {"appsolver_oformer", "transolver", "appsolver_v2", "gnot", "oformer"}


def synthetic_case(seed=0):
    """合成一张格点场：真值有正负波形，预测 = 真值 + 噪声。"""
    spacing = 0.05
    xs = -1.5 + spacing * np.arange(NX)
    ys = -0.75 + spacing * np.arange(NY)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1) * LPP
    meta = {
        "entry_id": "synthetic_entry",
        "Fr": 0.25,
        "Lpp": LPP,
        "flow_field": {"domain_over_Lpp": {"x": [-1.5, -1.5 + spacing * (NX - 1)],
                                           "y": [-0.75, -0.75 + spacing * (NY - 1)],
                                           "spacing": spacing, "n_grid": [NX, NY]}},
    }
    rng = np.random.default_rng(seed)
    target = np.zeros((len(points), 5))
    target[:, 0] = 0.03 * np.sin(6 * points[:, 0])
    target[:, 1] = 2.4 + 0.05 * np.cos(4 * points[:, 1])
    target[:, 2] = 0.01 * np.sin(3 * points[:, 0])
    target[:, 3] = 0.02 * np.cos(2 * points[:, 1])
    target[:, 4] = 0.08 * np.cos(5 * points[:, 0])
    prediction = target + rng.normal(scale=[2e-3, 0.02, 0.01, 0.01, 0.01], size=target.shape)
    return meta, points, target, prediction


def test_rows_cover_three_fields_with_units():
    assert [row.key for row in flow_compare.ROWS] == ["eta", "|U|", "Cp"]
    assert [row.short for row in flow_compare.ROWS] == ["$\\eta$ (m)", "$|U|$ (m/s)",
                                                        "$C_p$ (-)"]
    assert flow_compare.COLUMNS == ("CFD truth", "prediction", "prediction − truth")


def test_row_limits_symmetric_and_from_truth_plus_prediction():
    _, _, target, prediction = synthetic_case()
    panels = flow_compare.field_channels(target, prediction)
    for truth, guess in panels:
        vmin, vmax = flow_compare.row_limits(truth, guess)
        merged = np.concatenate([truth, guess])
        assert vmin == pytest.approx(-vmax) and vmax > 0
        if merged.min() < 0 < merged.max():
            # 跨零场：±P99，抑住个别极端点
            assert vmax == pytest.approx(np.percentile(np.abs(merged),
                                                       flow_compare.PERCENTILE))
        else:
            # 单侧为正的场（|U|）：±max 的完整包络，否则整行被压在色标顶端
            assert vmax == pytest.approx(np.abs(merged).max())
        # 合并顺序不影响范围，误差（guess - truth）不参与计算
        assert flow_compare.row_limits(guess, truth) == (vmin, vmax)


def test_error_value_cannot_rescale_the_row():
    """把预测推向极端只会通过真值+预测改变范围，误差自身没有独立的限幅口径。"""
    _, _, target, prediction = synthetic_case()
    vmin, vmax = flow_compare.row_limits(target[:, 0], prediction[:, 0])
    error = prediction[:, 0] - target[:, 0]
    assert error.max() < vmax and error.min() > vmin
    assert (vmin, vmax) == flow_compare.row_limits(target[:, 0], prediction[:, 0])


def test_assert_shared_limits_rejects_mismatch():
    limit = 0.5
    meshes = [types.SimpleNamespace(norm=types.SimpleNamespace(vmin=-limit, vmax=limit))
              for _ in range(3)]
    flow_compare.assert_shared_limits(meshes, -limit, limit)
    meshes[2].norm.vmax = 2 * limit
    with pytest.raises(AssertionError):
        flow_compare.assert_shared_limits(meshes, -limit, limit)


def test_group_limits_are_shared_across_the_model_set():
    """整组范围由真值+全部模型预测合并计算，与模型顺序无关；单模型时退化为 row_limits。"""
    _, _, target, prediction = synthetic_case(seed=1)
    flat = np.zeros_like(prediction)  # 退化模型（预测≈0，对应 APPSolver-v1 / CORAL）
    limits = flow_compare.group_limits(target, [prediction, flat])
    assert limits == flow_compare.group_limits(target, [flat, prediction])
    assert len(limits) == 3
    for vmin, vmax in limits:
        assert vmin == pytest.approx(-vmax) and vmax > 0
    panels = flow_compare.field_channels(target, prediction)
    single = tuple(flow_compare.row_limits(truth, guess) for truth, guess in panels)
    assert flow_compare.group_limits(target, [prediction]) == single
    # 退化模型把合并样本拉向 0：范围按 P99 收缩，但仍覆盖真值的主要幅值
    for (truth, _), (vmin, vmax) in zip(panels, limits):
        assert vmax >= np.percentile(np.abs(truth), 90)


def test_plot_fields_writes_three_columns_with_shared_scale(tmp_path):
    meta, points, target, prediction = synthetic_case()
    out_path = tmp_path / "synthetic_model.png"
    summary = flow_compare.plot_fields(
        meta, points, target, prediction, model_label="SMOKE", split="synthetic",
        epoch=0, entry=meta["entry_id"], out_path=out_path, dpi=80)
    assert out_path.is_file() and out_path.stat().st_size > 10_000
    assert len(summary["r2"]) == len(summary["rmse"]) == len(summary["limits"]) == 3
    panels = flow_compare.field_channels(target, prediction)
    for (truth, guess), (vmin, vmax) in zip(panels, summary["limits"]):
        assert (vmin, vmax) == flow_compare.row_limits(truth, guess)
        assert vmin == pytest.approx(-vmax)


def test_plot_fields_accepts_group_limits(tmp_path):
    """同一划分的整组模型必须真的画出同一套范围（而不是每张图各自重算）。"""
    meta, points, target, prediction = synthetic_case()
    limits = flow_compare.group_limits(target, [prediction])
    widened = tuple((vmin * 1.5, vmax * 1.5) for vmin, vmax in limits)
    summary = flow_compare.plot_fields(
        meta, points, target, prediction, model_label="SMOKE", split="synthetic",
        epoch=0, entry=meta["entry_id"], out_path=tmp_path / "group.png",
        limits=widened, dpi=80)
    assert tuple(summary["limits"]) == widened


def test_main_experiment_manifest():
    assert set(MAIN.MODELS) == MAIN_MODELS
    assert len(MAIN.MODELS) == 7 and len(set(MAIN.MODELS)) == 7
    splits = json.loads(SPLITS_FILE.read_text())
    assert MAIN.SPLIT == "random_seed0"
    assert MAIN.ENTRY in splits[MAIN.SPLIT]["val"]
    for model in MAIN.MODELS:
        config_path = ROOT / "configs" / model / f"{model}_{MAIN.SPLIT}_isoflops.yaml"
        assert config_path.is_file(), config_path
        config = yaml.safe_load(config_path.read_text())
        assert config["model_type"] == model
        assert config["stats"] == "dataset/shipbench/stats_random_seed0.json"
        assert MAIN.ENTRY in config["train"]["val_entries"]
        assert config["train"]["val_entries"] == splits[MAIN.SPLIT]["val"]
        assert MAIN.LABELS[model]


def test_three_split_manifest():
    assert set(THREE.MODELS) == THREE_MODELS
    assert len(THREE.MODELS) == 5 and len(set(THREE.MODELS)) == 5
    splits = json.loads(SPLITS_FILE.read_text())
    assert set(THREE.SPLITS) == {"loto_kvlcc2", "lofo_fr_design"}
    for split, entry in THREE.SPLITS.items():
        assert entry in splits[split]["val"], (split, entry)
        assert len(splits[split]["val"]) == (6 if split == "loto_kvlcc2" else 11)
        for model in THREE.MODELS:
            config_path = ROOT / "configs" / model / f"{model}_{split}_isoflops.yaml"
            assert config_path.is_file(), config_path
            config = yaml.safe_load(config_path.read_text())
            assert config["model_type"] == model
            assert config["stats"] == f"dataset/shipbench/stats_{split}.json"
            assert entry in config["train"]["val_entries"]
            assert config["train"]["val_entries"] == splits[split]["val"]
    # 留一船型那张图必须是 kvlcc2；留出设计 Fr 那张必须是 11 条设计工况之一
    assert THREE.SPLITS["loto_kvlcc2"].startswith("kvlcc2")
    assert THREE.SPLITS["lofo_fr_design"].endswith("_normal_mesh")
