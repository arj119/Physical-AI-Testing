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

# Camera rotation offset: degrees to add to the detected angle to align
# with J6 zero. Measure once: place cube aligned with gripper at J6=0,
# note the angle the camera reports, set this to negative of that.
CAMERA_ROTATION_OFFSET = float(os.environ.get("CAMERA_ROTATION_OFFSET", "0"))


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

    MOTION_TIMEOUT = 15  # seconds to wait for arm to reach target

    def go_to(self, waypoint_name: str) -> None:
        """Move to a named waypoint. Blocks until the arm arrives."""
        wp = self.waypoints.get(waypoint_name)
        if wp is None:
            logger.error("Unknown waypoint: %s", waypoint_name)
            return
        if self.mock:
            logger.info("[MOCK] Arm → %s %s", wp.name, wp.angles)
            time.sleep(0.3)
            return
        logger.debug("Arm → %s (speed=%d)", wp.name, wp.speed)
        self._mc.sync_send_angles(wp.angles, wp.speed, timeout=self.MOTION_TIMEOUT)

    def go_to_coords(self, coords: List[float], speed: int = 50) -> None:
        """Move to a Cartesian position. Blocks until the arm arrives.

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
        logger.debug("Arm → coords [%.1f, %.1f, %.1f] speed=%d", coords[0], coords[1], coords[2], speed)
        self._mc.sync_send_coords(coords, speed, mode=1, timeout=self.MOTION_TIMEOUT)

    # Heights in mm — adjust for your setup
    # Z heights in robot base-frame coordinates (mm)
    # Adjust these to match your setup — check with mc.get_coords() at the surface
    GRIP_HEIGHT_MM = float(os.environ.get("GRIP_HEIGHT_MM", "88"))       # surface level where gripper grabs
    APPROACH_HEIGHT_MM = float(os.environ.get("APPROACH_HEIGHT_MM", "160"))  # above cube, clear for approach
    TRANSIT_HEIGHT_MM = float(os.environ.get("TRANSIT_HEIGHT_MM", "200"))    # safe travel height between positions

    def pick_and_place(
        self,
        decision: str,
        gripper: Gripper,
        pick_target: Optional[PickTarget] = None,
        rotation_angle: float = 0.0,
    ) -> None:
        """Execute a full pick → sort cycle with approach/lift/rotation phases.

        Sequence:
          HOME
          → open gripper
          → move above pick position (approach height)
          → rotate J6 to align with cube
          → lower to grip height (slow)
          → close gripper
          → lift to transit height
          → move to bin
          → open gripper
          → HOME

        Parameters
        ----------
        decision : str
            Sorting decision: "PASS", "FAIL", or "REVIEW".
        gripper : Gripper
            Gripper driver instance.
        pick_target : PickTarget, optional
            If provided and reachable, uses Cartesian coords for the pick.
        rotation_angle : float
            Detected cube rotation in degrees (from camera). Applied to J6
            with CAMERA_ROTATION_OFFSET correction.
        """
        bin_name = DECISION_TO_BIN.get(decision, "BIN_REVIEW")
        logger.info("Pick-and-place: decision=%s → bin=%s", decision, bin_name)

        self.go_to("HOME")
        gripper.open_gripper()

        # Compute gripper rotation: camera angle + offset
        grip_rz = rotation_angle + CAMERA_ROTATION_OFFSET

        if pick_target and pick_target.reachable:
            coords = pick_target.coords
            logger.info(
                "Dynamic pick at (%.1f, %.1f) rz=%.1f° — %.1f mm from base",
                coords[0], coords[1], grip_rz, pick_target.distance_from_base,
            )
            rx, ry = coords[3], coords[4]

            # Approach above the cube (with rotation already set)
            approach = [coords[0], coords[1], self.APPROACH_HEIGHT_MM, rx, ry, grip_rz]
            self.go_to_coords(approach)

            # Lower to grip height (slow for precision)
            grip_pos = [coords[0], coords[1], self.GRIP_HEIGHT_MM, rx, ry, grip_rz]
            self.go_to_coords(grip_pos, speed=30)

            # Close gripper
            gripper.close_gripper()

            # Lift to transit height
            lift_pos = [coords[0], coords[1], self.TRANSIT_HEIGHT_MM, rx, ry, grip_rz]
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
