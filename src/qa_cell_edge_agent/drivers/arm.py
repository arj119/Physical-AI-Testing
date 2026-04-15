"""myCobot 280 arm driver — waypoint-based pick-and-place motions.

Waypoints must be calibrated per physical setup using ``scripts/calibrate_arm.py``.
When ``mock=True`` the driver logs motions without moving hardware.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from qa_cell_edge_agent.drivers.connection import get_connection

if TYPE_CHECKING:
    from qa_cell_edge_agent.drivers.gripper import Gripper

logger = logging.getLogger(__name__)

WAYPOINTS_FILE = os.path.join(os.path.dirname(__file__), "waypoints.json")


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


def _load_waypoints_from_file() -> Optional[Dict[str, Waypoint]]:
    """Load calibrated waypoints from ``waypoints.json`` if it exists."""
    if not os.path.isfile(WAYPOINTS_FILE):
        return None
    try:
        with open(WAYPOINTS_FILE) as f:
            data = json.load(f)
        waypoints = {}
        for name, entry in data.items():
            waypoints[name] = Waypoint(name=name, angles=entry["angles"])
        logger.info("Loaded calibrated waypoints from %s", WAYPOINTS_FILE)
        return waypoints
    except Exception as exc:
        logger.warning("Failed to load waypoints from %s: %s", WAYPOINTS_FILE, exc)
        return None


class Arm:
    """High-level arm controller for pick-and-place sorting."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        waypoints: Optional[Dict[str, Waypoint]] = None,
        mock: bool = False,
    ) -> None:
        self._mc = get_connection(port, baud, mock=mock)
        self.mock = self._mc is None

        self.waypoints = waypoints or _load_waypoints_from_file() or DEFAULT_WAYPOINTS
        if waypoints:
            logger.info("Using caller-supplied waypoints")
        elif self.waypoints is not DEFAULT_WAYPOINTS:
            pass  # already logged in _load_waypoints_from_file
        else:
            logger.warning("Using DEFAULT placeholder waypoints — run calibrate_arm.py")

        if self.mock:
            logger.info("Arm running in MOCK mode")
        else:
            logger.info("Arm connected on %s @ %d", port, baud)

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

    def pick_and_place(self, decision: str, gripper: Gripper) -> None:
        """Execute a full pick → sort cycle for the given fusion decision.

        Sequence: HOME → PICK → close gripper → BIN → release gripper → HOME
        """
        bin_name = DECISION_TO_BIN.get(decision, "BIN_REVIEW")
        logger.info("Pick-and-place: decision=%s → bin=%s", decision, bin_name)
        self.go_to("HOME")
        self.go_to("PICK")
        gripper.close_gripper()
        self.go_to(bin_name)
        gripper.release()
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
