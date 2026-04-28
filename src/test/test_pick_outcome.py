"""Tests for the PickOutcome dataclass + serialisation."""

from __future__ import annotations

from qa_cell_edge_agent.drivers.arm import PickOutcome


def test_default_fields():
    outcome = PickOutcome(success=True, attempts=1)
    assert outcome.success is True
    assert outcome.attempts == 1
    assert outcome.last_load == 0.0
    assert outcome.final_pose is None
    assert outcome.total_time_s == 0.0
    assert outcome.bin_name is None
    assert outcome.attempt_loads == []


def test_as_dict_round_trip_keys():
    outcome = PickOutcome(
        success=False,
        attempts=3,
        last_load=0.043,
        final_pose=(120.5, -45.2, 30.0),
        total_time_s=12.3456,
        bin_name="BIN_REVIEW",
        attempt_loads=[0.05, 0.04, 0.043],
    )
    d = outcome.as_dict()
    assert set(d.keys()) == {
        "success", "attempts", "last_load",
        "final_pose", "total_time_s", "bin_name", "attempt_loads",
    }
    assert d["success"] is False
    assert d["attempts"] == 3
    assert d["last_load"] == 0.043
    assert d["final_pose"] == [120.5, -45.2, 30.0]
    assert d["total_time_s"] == 12.346
    assert d["bin_name"] == "BIN_REVIEW"
    assert d["attempt_loads"] == [0.05, 0.04, 0.043]


def test_as_dict_handles_none_final_pose():
    outcome = PickOutcome(success=True, attempts=1, final_pose=None)
    assert outcome.as_dict()["final_pose"] is None


def test_attempt_loads_is_independent_per_instance():
    """Default factory must give each instance its own list."""
    a = PickOutcome(success=True, attempts=1)
    b = PickOutcome(success=True, attempts=1)
    a.attempt_loads.append(0.1)
    assert b.attempt_loads == []
