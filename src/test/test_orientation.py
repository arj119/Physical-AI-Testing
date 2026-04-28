"""Tests for yaw normalisation and bbox-based yaw estimation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from qa_cell_edge_agent.drivers.orientation import (
    OrientedDetection,
    estimate_yaw_in_bbox,
    minarearect_yaw,
    normalize_square_yaw,
)


_HAS_CV2 = importlib.util.find_spec("cv2") is not None


# ── normalize_square_yaw ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        (0.0, 0.0),
        (10.0, 10.0),
        (44.9, 44.9),
        (45.0, 45.0),
        (46.0, -44.0),
        (89.9, -0.1),
        (90.0, 0.0),
        (135.0, 45.0),
        (180.0, 0.0),
        (-30.0, -30.0),
        (-46.0, 44.0),
        (-90.0, 0.0),
        (270.0, 0.0),
        (315.0, 45.0),
        (359.0, -1.0),
    ],
)
def test_normalize_square_yaw(raw, expected):
    """Every angle collapses into (-45, 45] under 90° symmetry."""
    assert normalize_square_yaw(raw) == pytest.approx(expected, abs=1e-6)


def test_normalize_square_yaw_preserves_mod_90():
    """For all inputs, normalised value matches the input modulo 90°."""
    rng = np.random.default_rng(42)
    for raw in rng.uniform(-720, 720, size=200):
        norm = normalize_square_yaw(float(raw))
        diff = (raw - norm) % 90.0
        assert diff < 1e-6 or 90.0 - diff < 1e-6


# ── minarearect_yaw ──────────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_CV2, reason="cv2 not available")
def test_minarearect_yaw_returns_none_for_degenerate_contour():
    assert minarearect_yaw(None) is None
    assert minarearect_yaw(np.zeros((2, 1, 2), dtype=np.int32)) is None


@pytest.mark.skipif(not _HAS_CV2, reason="cv2 not available")
def test_minarearect_yaw_axis_aligned_square():
    """An axis-aligned square should report yaw within ±1° of 0."""
    contour = np.array(
        [[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]],
        dtype=np.int32,
    )
    det = minarearect_yaw(contour)
    assert det is not None
    assert isinstance(det, OrientedDetection)
    assert abs(det.yaw_deg) <= 1.0
    assert det.centre == pytest.approx((150.0, 150.0), abs=1.0)


@pytest.mark.skipif(not _HAS_CV2, reason="cv2 not available")
def test_minarearect_yaw_30_degree_rotation():
    """A square rotated 30° should report yaw near 30° (or -60° wrapped)."""
    import cv2

    pts = np.array([[-50, -50], [50, -50], [50, 50], [-50, 50]], dtype=np.float32)
    theta = np.deg2rad(30.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = (pts @ rot.T) + np.array([200, 200])
    contour = rotated.reshape(-1, 1, 2).astype(np.int32)

    det = minarearect_yaw(contour)
    assert det is not None
    # Under 90° symmetry, ±30° is the canonical value (and -60° wraps to 30°).
    assert abs(abs(det.yaw_deg) - 30.0) <= 2.0


# ── estimate_yaw_in_bbox ─────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_CV2, reason="cv2 not available")
def test_estimate_yaw_in_bbox_returns_none_on_empty_inputs():
    assert estimate_yaw_in_bbox(None, [0, 0, 10, 10]) is None
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert estimate_yaw_in_bbox(frame, []) is None
    assert estimate_yaw_in_bbox(frame, [0, 0, 0, 0]) is None


@pytest.mark.skipif(not _HAS_CV2, reason="cv2 not available")
def test_estimate_yaw_in_bbox_dark_square_on_light_background():
    """Dark cube on white background → yaw recovered within ±5°."""
    import cv2

    frame = np.full((300, 300, 3), 240, dtype=np.uint8)  # white background
    centre = (150, 150)
    size = (80, 80)
    box = cv2.boxPoints(((centre[0], centre[1]), size, 25.0))
    box = np.intp(box)
    cv2.fillPoly(frame, [box], (40, 40, 40))  # dark cube

    bbox_xywh = [110.0, 110.0, 80.0, 80.0]
    det = estimate_yaw_in_bbox(frame, bbox_xywh)
    assert det is not None
    assert abs(abs(det.yaw_deg) - 25.0) <= 5.0
    assert det.centre[0] == pytest.approx(centre[0], abs=5.0)
    assert det.centre[1] == pytest.approx(centre[1], abs=5.0)


@pytest.mark.skipif(not _HAS_CV2, reason="cv2 not available")
def test_estimate_yaw_in_bbox_light_square_on_dark_background():
    """Inverted contrast (light cube, dark mat) also works thanks to dual-mask."""
    import cv2

    frame = np.full((300, 300, 3), 20, dtype=np.uint8)
    centre = (150, 150)
    size = (80, 80)
    box = cv2.boxPoints(((centre[0], centre[1]), size, -15.0))
    box = np.intp(box)
    cv2.fillPoly(frame, [box], (220, 220, 220))

    det = estimate_yaw_in_bbox(frame, [110.0, 110.0, 80.0, 80.0])
    assert det is not None
    assert abs(abs(det.yaw_deg) - 15.0) <= 5.0
