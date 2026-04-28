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
def _get_camera_rotation_offset() -> float:
    return float(os.environ.get("CAMERA_ROTATION_OFFSET", "0"))


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

    MOTION_TIMEOUT = 300
    POSITION_TIMEOUT = 20  # seconds to wait for is_in_position

    # Z heights in robot base-frame mm — set via env or .env
    GRIP_HEIGHT_MM = float(os.environ.get("GRIP_HEIGHT_MM", "88"))
    APPROACH_HEIGHT_MM = float(os.environ.get("APPROACH_HEIGHT_MM", "160"))

    # Fixed downward orientation for picking (rx, ry)
    # rz is set dynamically from detected block rotation
    PICK_RX = float(os.environ.get("PICK_RX", "179.87"))
    PICK_RY = float(os.environ.get("PICK_RY", "-3.78"))

    def go_to(self, waypoint_name: str) -> None:
        """Move to a named waypoint using joint angles. Blocks until arrived."""
        wp = self.waypoints.get(waypoint_name)
        if wp is None:
            logger.error("Unknown waypoint: %s", waypoint_name)
            return
        if self.mock:
            logger.info("[MOCK] Arm → %s %s", wp.name, wp.angles)
            time.sleep(0.3)
            return
        try:
            logger.info("Arm → %s speed=%d", wp.name, wp.speed)
            self._mc.sync_send_angles(wp.angles, wp.speed, timeout=self.MOTION_TIMEOUT)
        except Exception as exc:
            logger.error("Arm go_to(%s) failed: %s", wp.name, exc)

    def _send_coords_and_wait(self, coords: List[float], speed: int = 40) -> bool:
        """Send Cartesian coords and wait for arrival. Returns True if reached."""
        if self.mock:
            logger.info("[MOCK] Arm → coords [%.1f, %.1f, %.1f]", coords[0], coords[1], coords[2])
            time.sleep(0.3)
            return True
        try:
            logger.info("Arm → coords [%.1f, %.1f, %.1f] speed=%d", coords[0], coords[1], coords[2], speed)
            self._mc.send_coords(coords, speed, 0)
            time.sleep(0.5)  # let the arm start moving before polling

            deadline = time.time() + self.POSITION_TIMEOUT
            while time.time() < deadline:
                try:
                    if self._mc.is_in_position(coords, 1) == 1:
                        return True
                except Exception:
                    pass
                # Also check if arm stopped moving (alternative completion check)
                try:
                    if self._mc.is_moving() == 0:
                        time.sleep(0.3)  # settle
                        return True
                except Exception:
                    pass
                time.sleep(0.1)

            logger.warning("Coords not reached within %ds", self.POSITION_TIMEOUT)
            return False
        except Exception as exc:
            logger.error("send_coords failed: %s", exc)
            return False

    def _lift_via_angles(self) -> None:
        """Lift the arm using joint angles (avoids IK failures on Z moves).

        Reads current J1 and J6 (preserving base rotation + gripper angle),
        uses the SAFE_ABOVE waypoint's J2-J5 for a known-safe raised position.
        Falls back to HOME if SAFE_ABOVE is not calibrated.
        """
        if self.mock:
            logger.info("[MOCK] Arm → lift via angles")
            time.sleep(0.3)
            return

        current = self._mc.get_angles()
        if not current or len(current) != 6:
            logger.warning("Cannot read angles for lift — going HOME")
            self.go_to("HOME")
            return

        safe_wp = self.waypoints.get("SAFE_ABOVE")
        if safe_wp:
            # Preserve J1 (base) and J6 (gripper), use safe J2-J5
            lift_angles = [current[0], safe_wp.angles[1], safe_wp.angles[2],
                           safe_wp.angles[3], safe_wp.angles[4], current[5]]
        else:
            # Fallback: use HOME J2-J5
            home_wp = self.waypoints.get("HOME")
            if home_wp:
                lift_angles = [current[0], home_wp.angles[1], home_wp.angles[2],
                               home_wp.angles[3], home_wp.angles[4], current[5]]
            else:
                logger.warning("No SAFE_ABOVE or HOME waypoint — cannot lift safely")
                return

        try:
            logger.info("Arm → lift via angles (J1=%.1f, J6=%.1f)", current[0], current[5])
            self._mc.sync_send_angles(lift_angles, 25, timeout=self.MOTION_TIMEOUT)
        except Exception as exc:
            logger.error("Lift failed: %s", exc)

    def _go_safe(self, waypoint_name: str) -> None:
        """Move to a waypoint via SAFE_ABOVE to avoid camera collisions."""
        if "SAFE_ABOVE" in self.waypoints:
            self.go_to("SAFE_ABOVE")
        self.go_to(waypoint_name)

    def pick_and_place(
        self,
        decision: str,
        gripper: Gripper,
        pick_target: Optional[PickTarget] = None,
        rotation_angle: float = 0.0,
    ) -> None:
        """Execute pick-and-place following the Elephant Robotics pattern.

        All transit between positions goes via SAFE_ABOVE to clear the
        overhead camera mount.

        Sequence:
          1. HOME → SAFE_ABOVE
          2. Open gripper
          3. Approach above target (coords)
          4. Descend to grip height (coords)
          5. Close gripper
          6. Lift via angles (safe J2-J5)
          7. SAFE_ABOVE → BIN
          8. Release gripper
          9. SAFE_ABOVE → HOME
        """
        bin_name = DECISION_TO_BIN.get(decision, "BIN_REVIEW")
        logger.info("Pick-and-place: %s → %s", decision, bin_name)

        # 1. HOME then raise to safe height
        self.go_to("HOME")
        gripper.open_gripper()

        if pick_target and pick_target.reachable:
            x, y = pick_target.coords[0], pick_target.coords[1]
            rx, ry = pick_target.coords[3], pick_target.coords[4]
            rz = rotation_angle + _get_camera_rotation_offset()

            logger.info("Dynamic pick at (%.1f, %.1f) rx=%.1f ry=%.1f rz=%.1f°", x, y, rx, ry, rz)

            # 2. Lift from current position first (avoid low arcs)
            if not self.mock and self._mc:
                current_coords = self._mc.get_coords()
                if current_coords and len(current_coords) == 6:
                    lift_z = self.APPROACH_HEIGHT_MM + 40
                    self._mc.send_coords(
                        [current_coords[0], current_coords[1], lift_z,
                         current_coords[3], current_coords[4], current_coords[5]],
                        30, 0,
                    )
                    time.sleep(2)

            # 3. Go to SAFE_ABOVE (clears camera)
            if "SAFE_ABOVE" in self.waypoints:
                self.go_to("SAFE_ABOVE")

            # 4. Approach above target (using rx/ry from calibration)
            approach = [x, y, self.APPROACH_HEIGHT_MM, rx, ry, rz]
            reached = self._send_coords_and_wait(approach, speed=40)
            if not reached:
                logger.warning("Could not reach approach — falling back to fixed PICK")
                self._go_safe("PICK")
                gripper.close_gripper()
                self._lift_via_angles()
                self._go_safe(bin_name)
                gripper.release()
                self._go_safe("HOME")
                return

            # 5. Descend to grip height
            grip_pos = [x, y, self.GRIP_HEIGHT_MM, rx, ry, rz]
            self._send_coords_and_wait(grip_pos, speed=25)

            # 5. Close gripper
            gripper.close_gripper()

            # 6. Lift via angles (safe, never fails)
            self._lift_via_angles()
        else:
            if pick_target and not pick_target.reachable:
                logger.warning("Pick target unreachable — using fixed PICK")
            self._go_safe("PICK")
            gripper.close_gripper()
            self._lift_via_angles()

        # 7. Move to bin via SAFE_ABOVE
        self._go_safe(bin_name)
        time.sleep(1.0)  # settle before releasing

        # 8. Release
        logger.info("Releasing at %s", bin_name)
        gripper.release()
        time.sleep(0.5)  # let block fall

        # 9. HOME via SAFE_ABOVE
        self._go_safe("HOME")

    def safe_position(self) -> None:
        """Move to HOME and disable servos (E-STOP safe state)."""
        logger.warning("E-STOP: moving to safe position")
        self.go_to("HOME")
        if not self.mock and self._mc:
            self._mc.release_all_servos()

