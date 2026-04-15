"""myCobot 280 arm driver — waypoint-based pick-and-place motions.

Waypoints must be calibrated per physical setup using ``scripts/calibrate_arm.py``.
When ``mock=True`` the driver logs motions without moving hardware.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from pymycobot.mycobot import MyCobot
except ImportError:
    MyCobot = None  # type: ignore[assignment,misc]


@dataclass
class Waypoint:
    """A named joint-angle position for the arm."""

    name: str
    angles: List[float]      # 6 joint angles in degrees
    speed: int = 50          # 0-100


# ── Default waypoints (MUST be calibrated for your setup) ─────────
# These are placeholder values — run calibrate_arm.py to set real ones.
DEFAULT_WAYPOINTS: Dict[str, Waypoint] = {
    "HOME":       Waypoint("HOME",       [0, 0, 0, 0, 0, 0]),
    "PICK":       Waypoint("PICK",       [0, -30, -20, 0, 0, 0]),
    "BIN_PASS":   Waypoint("BIN_PASS",   [45, -30, -20, 0, 0, 0]),
    "BIN_FAIL":   Waypoint("BIN_FAIL",   [-45, -30, -20, 0, 0, 0]),
    "BIN_REVIEW": Waypoint("BIN_REVIEW", [90, -30, -20, 0, 0, 0]),
}

# Maps fusion decision to bin waypoint name
DECISION_TO_BIN = {
    "PASS": "BIN_PASS",
    "FAIL": "BIN_FAIL",
    "REVIEW": "BIN_REVIEW",
}


class Arm:
    """High-level arm controller for pick-and-place sorting."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        waypoints: Optional[Dict[str, Waypoint]] = None,
        mock: bool = False,
    ) -> None:
        self.mock = mock or MyCobot is None
        self.waypoints = waypoints or DEFAULT_WAYPOINTS
        self._mc: Optional[MyCobot] = None

        if not self.mock:
            try:
                self._mc = MyCobot(port, baud)
                time.sleep(0.5)
                logger.info("Arm connected on %s @ %d", port, baud)
            except Exception as exc:
                logger.error("Arm init failed: %s — falling back to mock", exc)
                self.mock = True
        if self.mock:
            logger.info("Arm running in MOCK mode")

    # ── public API ────────────────────────────────────────────────────

    def go_to(self, waypoint_name: str) -> None:
        """Move to a named waypoint."""
        wp = self.waypoints.get(waypoint_name)
        if wp is None:
            logger.error("Unknown waypoint: %s", waypoint_name)
            return
        if self.mock:
            logger.info("[MOCK] Arm → %s %s", wp.name, wp.angles)
            time.sleep(0.3)
            return
        self._mc.send_angles(wp.angles, wp.speed)
        self._wait_until_stopped()

    def pick_and_place(self, decision: str) -> None:
        """Execute a full pick → sort cycle for the given fusion decision.

        Sequence: HOME → PICK → (close gripper) → BIN → (open gripper) → HOME
        """
        bin_name = DECISION_TO_BIN.get(decision, "BIN_REVIEW")
        logger.info("Pick-and-place: decision=%s → bin=%s", decision, bin_name)
        self.go_to("PICK")
        # Gripper close/open is handled by the caller (defect_detection process)
        self.go_to(bin_name)
        # Gripper release is handled by the caller
        self.go_to("HOME")

    def safe_position(self) -> None:
        """Move to HOME and disable servos (E-STOP safe state)."""
        logger.warning("E-STOP: moving to safe position")
        self.go_to("HOME")
        if not self.mock and self._mc:
            self._mc.release_all_servos()

    # ── internals ─────────────────────────────────────────────────────

    def _wait_until_stopped(self, timeout: float = 5.0) -> None:
        """Block until the arm reports it has stopped moving."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self._mc.is_moving() == 0:
                    return
            except Exception:
                pass
            time.sleep(0.1)
        logger.warning("Arm motion timed out after %.1fs", timeout)
