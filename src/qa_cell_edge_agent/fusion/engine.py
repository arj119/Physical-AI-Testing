"""Sensor fusion decision engine.

Combines vision inference confidence with gripper load to produce a
PASS / FAIL / REVIEW decision.  Thresholds are dynamically updatable
at runtime (via UPDATE_TOLERANCE operator commands).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Immutable result of a single fusion decision."""

    decision: str          # PASS | FAIL | REVIEW
    reason: str            # human-readable explanation
    vision_agrees: bool    # True when both sensors agree


class FusionEngine:
    """Stateful fusion engine whose thresholds can be hot-updated."""

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        grip_tolerance: float = 0.65,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.grip_tolerance = grip_tolerance
        logger.info(
            "FusionEngine initialised — confidence_threshold=%.2f, grip_tolerance=%.2f",
            self.confidence_threshold,
            self.grip_tolerance,
        )

    # ── public API ────────────────────────────────────────────────────

    def decide(
        self,
        vision_class: str,
        confidence: float,
        normalized_load: float,
    ) -> FusionResult:
        """Run fusion logic and return a FusionResult.

        Parameters
        ----------
        vision_class : str
            Detected class from the vision model (e.g. ``widget_good``).
        confidence : float
            Model confidence in ``[0, 1]``.
        normalized_load : float
            Normalised gripper load in ``[0, 1]``.
        """
        vision_pass = (
            vision_class == "widget_good"
            and confidence >= self.confidence_threshold
        )
        grip_normal = normalized_load <= self.grip_tolerance

        if vision_pass and grip_normal:
            return FusionResult(
                decision="PASS",
                reason="vision_pass + grip_normal",
                vision_agrees=True,
            )
        elif not vision_pass and not grip_normal:
            return FusionResult(
                decision="FAIL",
                reason="vision_fail + grip_anomaly",
                vision_agrees=True,
            )
        else:
            v_label = "pass" if vision_pass else "fail"
            g_label = "normal" if grip_normal else "anomaly"
            return FusionResult(
                decision="REVIEW",
                reason=f"vision_{v_label} + grip_{g_label}",
                vision_agrees=False,
            )

    def update_thresholds(
        self,
        confidence_threshold: float | None = None,
        grip_tolerance: float | None = None,
    ) -> None:
        """Hot-update thresholds (called when an UPDATE_TOLERANCE command arrives)."""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if grip_tolerance is not None:
            self.grip_tolerance = grip_tolerance
        logger.info(
            "Thresholds updated — confidence_threshold=%.2f, grip_tolerance=%.2f",
            self.confidence_threshold,
            self.grip_tolerance,
        )
