"""Unit tests for the sensor fusion decision engine."""

import pytest
from qa_cell_edge_agent.fusion.engine import FusionEngine, FusionResult


@pytest.fixture
def engine():
    """Default engine with standard thresholds."""
    return FusionEngine(confidence_threshold=0.75, grip_tolerance=0.65)


# ---------------------------------------------------------------------------
# PASS cases — high confidence + low grip load
# ---------------------------------------------------------------------------

class TestPassDecisions:
    def test_clear_pass(self, engine):
        result = engine.decide("widget_good", 0.95, 0.20)
        assert result.decision == "PASS"
        assert result.vision_agrees is True

    def test_boundary_pass(self, engine):
        """Exactly at both thresholds → should still PASS."""
        result = engine.decide("widget_good", 0.75, 0.65)
        assert result.decision == "PASS"

    def test_pass_reason_contains_both_sensors(self, engine):
        result = engine.decide("widget_good", 0.90, 0.30)
        assert "vision_pass" in result.reason
        assert "grip_normal" in result.reason


# ---------------------------------------------------------------------------
# FAIL cases — both sensors agree something is wrong
# ---------------------------------------------------------------------------

class TestFailDecisions:
    def test_clear_fail_defect_class(self, engine):
        result = engine.decide("widget_defect", 0.95, 0.80)
        assert result.decision == "FAIL"
        assert result.vision_agrees is True

    def test_fail_low_confidence_high_load(self, engine):
        result = engine.decide("widget_good", 0.50, 0.80)
        assert result.decision == "FAIL"

    def test_fail_unknown_class_high_load(self, engine):
        result = engine.decide("widget_unknown", 0.60, 0.90)
        assert result.decision == "FAIL"


# ---------------------------------------------------------------------------
# REVIEW cases — sensors disagree
# ---------------------------------------------------------------------------

class TestReviewDecisions:
    def test_vision_pass_grip_anomaly(self, engine):
        """Vision says good but gripper load is high → REVIEW."""
        result = engine.decide("widget_good", 0.90, 0.80)
        assert result.decision == "REVIEW"
        assert result.vision_agrees is False

    def test_vision_fail_grip_normal(self, engine):
        """Vision says defect but gripper load is normal → REVIEW."""
        result = engine.decide("widget_defect", 0.90, 0.30)
        assert result.decision == "REVIEW"
        assert result.vision_agrees is False

    def test_low_confidence_good_class_normal_grip(self, engine):
        """Low confidence good + normal grip → sensors disagree → REVIEW."""
        result = engine.decide("widget_good", 0.50, 0.30)
        assert result.decision == "REVIEW"


# ---------------------------------------------------------------------------
# Threshold updates
# ---------------------------------------------------------------------------

class TestThresholdUpdates:
    def test_update_grip_tolerance(self, engine):
        # With default 0.65, load=0.70 triggers anomaly
        result_before = engine.decide("widget_good", 0.90, 0.70)
        assert result_before.decision == "REVIEW"

        # Raise tolerance to 0.80 — same reading now passes
        engine.update_thresholds(grip_tolerance=0.80)
        result_after = engine.decide("widget_good", 0.90, 0.70)
        assert result_after.decision == "PASS"

    def test_update_confidence_threshold(self, engine):
        # With default 0.75, confidence=0.70 is below threshold
        result_before = engine.decide("widget_good", 0.70, 0.30)
        assert result_before.decision == "REVIEW"

        # Lower threshold to 0.60 — same reading now passes
        engine.update_thresholds(confidence_threshold=0.60)
        result_after = engine.decide("widget_good", 0.70, 0.30)
        assert result_after.decision == "PASS"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_confidence(self, engine):
        result = engine.decide("widget_good", 0.0, 0.30)
        assert result.decision in ("FAIL", "REVIEW")

    def test_max_load(self, engine):
        result = engine.decide("widget_good", 0.90, 1.0)
        assert result.decision == "REVIEW"

    def test_none_grip_load_degrades_gracefully(self, engine):
        """When grip data is missing, fall back to vision-only decision."""
        result = engine.decide("widget_good", 0.90, None)
        assert result.decision == "REVIEW"
        assert "degraded" in result.reason.lower() or "grip" in result.reason.lower()
