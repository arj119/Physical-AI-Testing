"""Tests for PoseBuffer circular median + stability gate."""

from __future__ import annotations

import math
import time

import pytest

from qa_cell_edge_agent.drivers.pose_buffer import (
    PoseBuffer,
    _circular_median_90,
    _max_circular_spread_90,
)


# ── circular median primitives ───────────────────────────────────────


def test_circular_median_around_zero():
    assert _circular_median_90([0.0, 1.0, -1.0, 0.5, -0.5]) == pytest.approx(0.0, abs=0.5)


def test_circular_median_handles_45_wrap():
    """Samples straddling the ±45° boundary must not collapse to 0."""
    samples = [44.0, -44.0, 44.5, -44.0, 45.0]
    med = _circular_median_90(samples)
    # The true centre is the ±45° symmetry point.
    assert abs(abs(med) - 45.0) <= 1.5


def test_circular_median_30_degrees():
    samples = [29.0, 30.0, 31.0, 30.5, 29.5]
    assert _circular_median_90(samples) == pytest.approx(30.0, abs=0.5)


def test_circular_spread_zero_for_identical():
    assert _max_circular_spread_90([10.0, 10.0, 10.0], 10.0) == 0.0


def test_circular_spread_handles_wrap():
    """Spread of 44 and -44 around 45 should be small (≈ 1°), not ≈ 88°."""
    spread = _max_circular_spread_90([44.0, -44.0], 45.0)
    assert spread <= 2.0


# ── PoseBuffer add / median ──────────────────────────────────────────


def test_add_and_length():
    buf = PoseBuffer(n=3)
    assert len(buf) == 0
    buf.add(10.0, 20.0, 5.0)
    buf.add(10.5, 20.5, 5.5)
    assert len(buf) == 2
    buf.add(11.0, 19.0, 6.0)
    buf.add(11.5, 19.5, 5.5)
    assert len(buf) == 3, "buffer should cap at n=3"


def test_median_returns_none_when_empty():
    assert PoseBuffer(n=3).median() is None


def test_median_xy_and_yaw():
    buf = PoseBuffer(n=5)
    for x, y, a in [(100, 200, 10), (101, 199, 11), (100, 200, 10), (99, 201, 9), (100, 200, 10)]:
        buf.add(x, y, a)
    m = buf.median()
    assert m is not None
    assert m.x_mm == pytest.approx(100.0, abs=0.5)
    assert m.y_mm == pytest.approx(200.0, abs=0.5)
    assert m.yaw_deg == pytest.approx(10.0, abs=0.5)
    assert m.n_samples == 5


# ── stability gate ───────────────────────────────────────────────────


def test_not_stable_until_buffer_full():
    buf = PoseBuffer(n=5, xy_tol_mm=2.0, yaw_tol_deg=2.0)
    for _ in range(4):
        buf.add(0, 0, 0)
    assert not buf.stable
    buf.add(0, 0, 0)
    assert buf.stable


def test_stable_within_tolerances():
    buf = PoseBuffer(n=5, xy_tol_mm=3.0, yaw_tol_deg=3.0)
    for x, y, a in [(100, 200, 10), (101, 199, 11), (100, 200, 10), (99, 201, 9), (100, 200, 10)]:
        buf.add(x, y, a)
    assert buf.stable


def test_unstable_when_xy_exceeds_tol():
    buf = PoseBuffer(n=3, xy_tol_mm=1.0, yaw_tol_deg=10.0)
    buf.add(0, 0, 0)
    buf.add(0, 0, 0)
    buf.add(10, 0, 0)
    assert not buf.stable


def test_unstable_when_yaw_exceeds_tol():
    buf = PoseBuffer(n=3, xy_tol_mm=10.0, yaw_tol_deg=2.0)
    buf.add(0, 0, 0)
    buf.add(0, 0, 0)
    buf.add(0, 0, 30)
    assert not buf.stable


def test_stale_samples_dropped():
    buf = PoseBuffer(n=5, max_age_s=0.5)
    base = time.monotonic()
    buf.add(0, 0, 0, t=base - 1.0)   # too old
    buf.add(0, 0, 0, t=base - 0.6)   # too old
    buf.add(0, 0, 0, t=base - 0.1)
    buf.add(0, 0, 0, t=base)
    # Both stale samples evicted on the most recent add
    assert len(buf) == 2


def test_reset_clears():
    buf = PoseBuffer(n=3)
    buf.add(1, 2, 3)
    buf.reset()
    assert len(buf) == 0
    assert buf.median() is None
    assert not buf.stable


def test_circular_stability_around_45():
    """Yaw samples straddling ±45° should still register as stable."""
    buf = PoseBuffer(n=5, xy_tol_mm=3.0, yaw_tol_deg=3.0)
    for _ in range(5):
        buf.add(0, 0, 44.5)
    buf.add(0, 0, -44.5)  # this looks like a 89° jump but is ≈1° under symmetry
    assert buf.stable
    m = buf.median()
    assert m is not None
    assert abs(abs(m.yaw_deg) - 45.0) <= 1.5
