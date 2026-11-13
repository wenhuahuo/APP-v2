"""Shared display helpers: hole masking and forwarding the requested hull line scale."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vis"))

from common.pinholes import fill_pinholes


def test_fill_pinholes_preserves_large_mask_and_input():
    image = np.arange(144, dtype=float).reshape(12, 12)
    image[1, 1] = np.nan
    image[5:9, 5:9] = np.nan
    original = image.copy()
    result = fill_pinholes(image, max_cells=4)

    assert np.isfinite(result[1, 1])
    assert result[1, 1] in original[[0, 1, 1, 2], [1, 0, 2, 1]]
    assert np.isnan(result[5:9, 5:9]).all()
    np.testing.assert_equal(image, original)
    valid = np.isfinite(original)
    np.testing.assert_equal(result[valid], original[valid])


@pytest.mark.parametrize("max_cells,filled", [(3, False), (4, True)])
def test_fill_pinholes_threshold(max_cells, filled):
    image = np.ones((8, 8))
    image[2:4, 2:4] = np.nan
    result = fill_pinholes(image, max_cells=max_cells)
    assert bool(np.isfinite(result[2:4, 2:4]).all()) is filled


def test_fill_pinholes_no_missing_values():
    image = np.arange(12, dtype=float).reshape(3, 4)
    np.testing.assert_equal(fill_pinholes(image), image)


@pytest.fixture(scope="module")
def hull_plot():
    path = ROOT / "vis/main_hull_comprison/plot_hull_comparison.py"
    spec = importlib.util.spec_from_file_location("vis_hull_helper_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scale", [0.35, 0.7, 1.4])
def test_hull_panel_forwards_line_scale(monkeypatch, hull_plot, scale):
    calls = []
    monkeypatch.setattr(hull_plot.hull3d, "draw", lambda *a, **kw: calls.append(kw))
    ax = SimpleNamespace(set_xlim=lambda *a: None, set_ylim=lambda *a: None,
                         set_aspect=lambda *a: None, set_axis_off=lambda: None)
    lines = object()
    hull_plot.draw_panel(ax, None, None, None, ((0, 1), (0, 1)),
                         lines=lines, lw_scale=scale)
    assert calls[0]["lw"] == scale
    assert calls[0]["lines"] is lines
    assert calls[0]["rasterize"] is True
