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
    from qa_cell_edge_agent.drivers.transforms import PickTarget

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
            try:
                self._mc.set_vision_mode(1)
                logger.info("Vision mode enabled (posture flip prevention)")
            except Exception:
                logger.debug("set_vision_mode not supported on this firmware")

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

    def go_to_coords(self, coords: List[float], speed: int = 50) -> None:
        """Move to a Cartesian position using linear interpolation.

        Parameters
        ----------
        coords : list
            ``[x, y, z, rx, ry, rz]`` in mm and degrees (base frame).
        speed : int
            Movement speed 1-100.
        """
        if self.mock:
            logger.info("[MOCK] Arm → coords %s", coords)
            time.sleep(0.3)
            return
        self._mc.send_coords(coords, speed, mode=1)  # mode=1 = linear
        self._wait_until_stopped()

    # Heights in mm — adjust for your setup
    APPROACH_HEIGHT_MM = 100.0  # above the surface (clear of cubes + camera safe)
    GRIP_HEIGHT_MM = 25.0       # half cube height (~50mm cube / 2)
    TRANSIT_HEIGHT_MM = 150.0   # safe height for moving between positions (below camera at 500mm)

    def pick_and_place(
        self,
        decision: str,
        gripper: Gripper,
        pick_target: Optional[PickTarget] = None,
    ) -> None:
        """Execute a full pick → sort cycle with approach/lift phases.

        Sequence:
          HOME
          → move above pick position (approach height)
          → lower to grip height
          → activate gripper/pump
          → lift to transit height
          → move to bin
          → release
          → HOME

        Parameters
        ----------
        decision : str
            Sorting decision: "PASS", "FAIL", or "REVIEW".
        gripper : Gripper
            Gripper driver instance.
        pick_target : PickTarget, optional
            If provided and reachable, uses Cartesian coords for the pick.
        """
        bin_name = DECISION_TO_BIN.get(decision, "BIN_REVIEW")
        logger.info("Pick-and-place: decision=%s → bin=%s", decision, bin_name)

        self.go_to("HOME")

        if pick_target and pick_target.reachable:
            coords = pick_target.coords
            logger.info(
                "Dynamic pick at (%.1f, %.1f) — %.1f mm from base",
                coords[0], coords[1], pick_target.distance_from_base,
            )
            rx, ry, rz = coords[3], coords[4], coords[5]

            # Approach above the cube
            approach = [coords[0], coords[1], self.APPROACH_HEIGHT_MM, rx, ry, rz]
            self.go_to_coords(approach)

            # Lower to grip height
            grip_pos = [coords[0], coords[1], self.GRIP_HEIGHT_MM, rx, ry, rz]
            self.go_to_coords(grip_pos, speed=30)

            # Activate gripper/pump
            gripper.close_gripper()

            # Lift to transit height
            lift_pos = [coords[0], coords[1], self.TRANSIT_HEIGHT_MM, rx, ry, rz]
            self.go_to_coords(lift_pos)
        else:
            if pick_target and not pick_target.reachable:
                logger.warning("Pick target unreachable — using fixed PICK waypoint")
            self.go_to("PICK")
            gripper.close_gripper()

        # Move to bin and release
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
