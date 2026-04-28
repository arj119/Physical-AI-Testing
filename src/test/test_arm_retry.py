"""Closed-loop pick retry orchestration tests.

Uses the mock Arm (no hardware required) and a stub Gripper that returns
configurable normalised_load values per call to verify:
  * first-attempt success skips redetect_cb,
  * sub-threshold load triggers redetect_cb + a second try_pick,
  * exhaustion returns PickOutcome(success=False, ...) with attempt count.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from qa_cell_edge_agent.drivers import arm as arm_module
from qa_cell_edge_agent.drivers.arm import Arm, PickOutcome
from qa_cell_edge_agent.drivers.transforms import PickTarget


@dataclass
class _GripData:
    normalized_load: float
    servo_load: float = 0.0
    grip_state: str = "CLOSED"
    object_detected: bool = False


class _StubGripper:
    """Records every call and returns scripted load values."""

    def __init__(self, loads):
        self._loads = list(loads)
        self.opens = 0
        self.closes = 0
        self.releases = 0
        self.reads = 0

    def open_gripper(self):
        self.opens += 1

    def close_gripper(self):
        self.closes += 1

    def release(self):
        self.releases += 1

    def read(self):
        self.reads += 1
        load = self._loads.pop(0) if self._loads else 0.0
        return _GripData(normalized_load=load)


def _make_target(x=100.0, y=100.0, reachable=True):
    return PickTarget(
        coords=[x, y, 50.0, 0.0, 0.0, 0.0],
        pixel_centre=(320, 240),
        reachable=reachable,
        distance_from_base=((x * x + y * y) ** 0.5),
    )


def _make_arm():
    """Mock Arm with no hardware. Uses default waypoints (SCOUT/BIN_PASS exist)."""
    return Arm(mock=True, via_table={})


# ── happy path: first-attempt success ────────────────────────────────


def test_first_attempt_success_no_redetect(monkeypatch):
    monkeypatch.setattr(arm_module, "GRIP_VERIFY_DELAY_S", 0.0)
    arm = _make_arm()
    grip = _StubGripper(loads=[0.40])
    redetect_called = {"n": 0}

    def cb():
        redetect_called["n"] += 1
        return None

    outcome = arm.pick_and_place(
        decision="PASS",
        gripper=grip,
        pick_target=_make_target(),
        rotation_angle=0.0,
        redetect_cb=cb,
    )
    assert outcome.success is True
    assert outcome.attempts == 1
    assert outcome.last_load == pytest.approx(0.40)
    assert outcome.bin_name == "BIN_PASS"
    assert redetect_called["n"] == 0, "redetect_cb must not be called on first-try success"
    assert grip.releases == 1
    assert grip.closes == 1


# ── retry: first attempt fails, second succeeds ──────────────────────


def test_retry_then_success_invokes_redetect_once(monkeypatch):
    monkeypatch.setattr(arm_module, "GRIP_VERIFY_DELAY_S", 0.0)
    monkeypatch.setattr(arm_module, "MAX_PICK_RETRIES", 2)
    arm = _make_arm()
    # Two reads (one per attempt): 0.05 (fail), 0.40 (success)
    grip = _StubGripper(loads=[0.05, 0.40])
    redetect_calls = []

    def cb():
        redetect_calls.append(True)
        # Return a slightly moved target to bypass the nudge logic
        return (_make_target(x=110.0, y=100.0), 12.0)

    outcome = arm.pick_and_place(
        decision="FAIL",
        gripper=grip,
        pick_target=_make_target(),
        rotation_angle=0.0,
        redetect_cb=cb,
    )
    assert outcome.success is True
    assert outcome.attempts == 2
    assert outcome.last_load == pytest.approx(0.40)
    assert outcome.bin_name == "BIN_FAIL"
    assert outcome.attempt_loads == [pytest.approx(0.05), pytest.approx(0.40)]
    assert len(redetect_calls) == 1


# ── retry exhaustion ─────────────────────────────────────────────────


def test_exhausted_retries_returns_failure(monkeypatch):
    monkeypatch.setattr(arm_module, "GRIP_VERIFY_DELAY_S", 0.0)
    monkeypatch.setattr(arm_module, "MAX_PICK_RETRIES", 2)
    arm = _make_arm()
    grip = _StubGripper(loads=[0.05, 0.04, 0.03])
    redetect_calls = []

    def cb():
        redetect_calls.append(True)
        return (_make_target(x=100.0 + 5 * len(redetect_calls), y=100.0), 0.0)

    outcome = arm.pick_and_place(
        decision="REVIEW",
        gripper=grip,
        pick_target=_make_target(),
        redetect_cb=cb,
    )
    assert outcome.success is False
    assert outcome.attempts == 3, "1 initial + 2 retries"
    assert outcome.last_load == pytest.approx(0.03)
    assert len(redetect_calls) == 2
    assert grip.releases == 0, "should not release into a bin on failure"


def test_redetect_returns_none_aborts_retries(monkeypatch):
    monkeypatch.setattr(arm_module, "GRIP_VERIFY_DELAY_S", 0.0)
    monkeypatch.setattr(arm_module, "MAX_PICK_RETRIES", 3)
    arm = _make_arm()
    grip = _StubGripper(loads=[0.02])

    def cb():
        return None  # detection lost

    outcome = arm.pick_and_place(
        decision="REVIEW",
        gripper=grip,
        pick_target=_make_target(),
        redetect_cb=cb,
    )
    assert outcome.success is False
    assert outcome.attempts == 1, "second attempt aborted because redetect returned None"


# ── bin override + colour mapping ────────────────────────────────────


def test_bin_override_takes_precedence(monkeypatch):
    monkeypatch.setattr(arm_module, "GRIP_VERIFY_DELAY_S", 0.0)
    arm = _make_arm()
    grip = _StubGripper(loads=[0.5])

    outcome = arm.pick_and_place(
        decision="PASS",
        gripper=grip,
        pick_target=_make_target(),
        bin_override="BIN_A",
    )
    assert outcome.success is True
    assert outcome.bin_name == "BIN_A"


# ── unreachable target ──────────────────────────────────────────────


def test_unreachable_target_falls_back_to_static_pick(monkeypatch):
    monkeypatch.setattr(arm_module, "GRIP_VERIFY_DELAY_S", 0.0)
    arm = _make_arm()
    grip = _StubGripper(loads=[])
    outcome = arm.pick_and_place(
        decision="PASS",
        gripper=grip,
        pick_target=_make_target(reachable=False),
    )
    # Static fallback path returns success without closed-loop verification
    assert outcome.success is True
    assert outcome.attempts == 1
    assert outcome.bin_name == "BIN_PASS"
