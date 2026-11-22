"""Regression: deck creases must be drawn in addition to the hull silhouette."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def morph():
    if not (ROOT / "vis/ship_morphing/plot_morphing.py").is_file():
        pytest.skip("Local visualization sources are absent")
    spec = importlib.util.spec_from_file_location(
        "vis_morphing_edge_test", ROOT / "vis/ship_morphing/plot_morphing.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def box_scene(morph):
    mesh = trimesh.creation.box(extents=[0.8, 0.2, 0.1])
    mesh.apply_translation([0.4, 0.0, 0.05])
    triangles = mesh.vertices[mesh.faces]
    proj = morph.plot3d.Projector(*morph.VIEW, mirror=True)
    front = morph.plot3d.face_normals(triangles) @ proj.c > 0
    return mesh, triangles, proj, front


def test_deck_edges_are_real_top_creases(morph):
    _, triangles, _, front = box_scene(morph)
    edges = morph.deck_edges(triangles, front, draught=0.04)
    assert len(edges) == 2
    np.testing.assert_allclose(edges[:, :, 2], 0.1)
    assert np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1).sum() == pytest.approx(1.0)
    assert not len(morph.deck_edges(triangles, np.zeros_like(front), draught=0.04))
    assert not len(morph.deck_edges(triangles, front, draught=0.2))


def test_rounded_deck_transition_is_not_dropped(morph):
    angle = np.deg2rad(20)
    points = np.array([[0, 0, 0.2], [1, 0, 0.2], [0, 1, 0.2],
                       [0, -np.cos(angle), 0.2 - np.sin(angle)]])
    triangles = points[[[0, 1, 2], [1, 0, 3]]]
    front = np.ones(2, dtype=bool)
    assert len(morph.deck_edges(triangles, front, draught=0.04)) == 1


def test_draw_hull_adds_internal_deck_stroke(morph, monkeypatch):
    mesh, _, proj, _ = box_scene(morph)
    surface = {"points": mesh.vertices, "faces": mesh.faces, "n_v": 1,
               "grid": np.array([[[0.0, 0.0, 0.04]]]),
               "vnormals": morph.hull3d.vertex_normals(mesh.vertices, mesh.faces)}
    monkeypatch.setattr(morph, "contour_mask", lambda *args: None)
    fig, ax = morph.plt.subplots()
    try:
        morph.draw_hull(ax, surface, proj)
        strokes = [c for c in ax.collections if hasattr(c, "get_segments")]
        assert len(strokes) == 1 and len(strokes[0].get_segments()) > 0
    finally:
        morph.plt.close(fig)


def test_deck_paths_handle_branches_and_closed_loops(morph):
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    segments = np.stack([square[:-1], square[1:]], axis=1)
    segments = np.concatenate([segments, [[[0, 0, 0], [-1, 0, 0]]]])
    curves = morph.deck_curves(segments)
    assert len(curves) == 2
    assert sum(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in curves) <= 5.0


def test_deck_smoothing_keeps_endpoints_and_small_changes(morph):
    points = np.column_stack([np.linspace(0.0, 0.01, 11),
                              0.0003 * (np.arange(11) % 2), np.full(11, 0.1)])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    curves = morph.deck_curves(segments)
    assert len(curves) == 1
    curve = curves[0]
    if curve[0, 0] > curve[-1, 0]:
        curve = curve[::-1]
    np.testing.assert_allclose(curve[[0, -1]], points[[0, -1]])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    raw = np.column_stack([np.interp(np.linspace(0, distance[-1], len(curve)), distance, a)
                           for a in points.T])
    assert np.linalg.norm(curve - raw, axis=1).max() <= 0.0005 + 1e-12
    assert curve[:, 1].max() < points[:, 1].max()
    assert morph.deck_curves(np.empty((0, 2, 3))) == []
