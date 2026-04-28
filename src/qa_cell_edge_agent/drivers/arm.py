"""myCobot 280 arm driver — waypoint-based pick-and-place motions.

Waypoints must be calibrated per physical setup using ``scripts/calibrate_arm.py``.
When ``mock=True`` the driver logs motions without moving hardware.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from qa_cell_edge_agent.drivers.connection import get_connection

if TYPE_CHECKING:
    from qa_cell_edge_agent.drivers.gripper import Gripper
    from qa_cell_edge_agent.drivers.transforms import PickTarget

logger = logging.getLogger(__name__)

WAYPOINTS_FILE = os.path.join(os.path.dirname(__file__), "waypoints.json")

# Camera rotation offset: degrees to add to the detected angle to align
# with J6 zero. Measure once with scripts/calibrate_yaw_offset.py.
# CAMERA_ROTATION_OFFSET_DEG is the canonical name; CAMERA_ROTATION_OFFSET
# kept for backwards compatibility.
CAMERA_ROTATION_OFFSET = float(
    os.environ.get(
        "CAMERA_ROTATION_OFFSET_DEG",
        os.environ.get("CAMERA_ROTATION_OFFSET", "0"),
    )
)

# Closed-loop pick parameters (env-tunable; see config/settings.py)
MAX_PICK_RETRIES = int(os.environ.get("MAX_PICK_RETRIES", "2"))
GRIP_LOAD_SUCCESS_THRESHOLD = float(os.environ.get("GRIP_LOAD_SUCCESS_THRESHOLD", "0.15"))
GRIP_VERIFY_DELAY_S = float(os.environ.get("GRIP_VERIFY_DELAY_S", "0.4"))
RETRY_NUDGE_MM = float(os.environ.get("RETRY_NUDGE_MM", "1.5"))


@dataclass
class PickOutcome:
    """Result of a pick_and_place call.

    ``success`` is True iff the gripper reported load above the threshold
    after at least one closed-loop attempt. ``attempts`` counts the total
    number of try_pick invocations (1 for first-try success, ≤ MAX+1 on
    failure). ``last_load`` is the final normalised load reading.
    """

    success: bool
    attempts: int
    last_load: float = 0.0
    final_pose: Optional[Tuple[float, float, float]] = None  # (x_mm, y_mm, yaw_deg)
    total_time_s: float = 0.0
    bin_name: Optional[str] = None
    attempt_loads: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "attempts": self.attempts,
            "last_load": round(self.last_load, 4),
            "final_pose": list(self.final_pose) if self.final_pose else None,
            "total_time_s": round(self.total_time_s, 3),
            "bin_name": self.bin_name,
            "attempt_loads": [round(v, 4) for v in self.attempt_loads],
        }


# Type for the optional re-detection callback. Returns updated
# (coords, rotation_angle) or None if no detection.
RedetectFn = Callable[[], Optional[Tuple["PickTarget", float]]]


@dataclass
class Waypoint:
    """A named joint-angle position for the arm."""

    name: str
    angles: List[float]      # 6 joint angles in degrees
    speed: int = 50          # 0-100


# ── Default waypoints (MUST be calibrated for your setup) ─────────
# These are placeholder values — run calibrate_arm.py to set real ones.
# SCOUT parks the arm fully outside the camera zone so re-detection
# during retry sees an unobstructed workspace (eye-to-hand setup).
DEFAULT_WAYPOINTS: Dict[str, Waypoint] = {
    "HOME":       Waypoint("HOME",       [0, 0, 0, 0, 0, 0]),
    "SCOUT":      Waypoint("SCOUT",      [0, 30, -90, 0, -30, 0]),
    "PICK":       Waypoint("PICK",       [0, -30, -20, 0, 0, 0]),
    "BIN_PASS":   Waypoint("BIN_PASS",   [45, -30, -20, 0, 0, 0]),
    "BIN_FAIL":   Waypoint("BIN_FAIL",   [-45, -30, -20, 0, 0, 0]),
    "BIN_REVIEW": Waypoint("BIN_REVIEW", [90, -30, -20, 0, 0, 0]),
    "BIN_A":      Waypoint("BIN_A",      [60, -30, -20, 0, 0, 0]),
    "BIN_B":      Waypoint("BIN_B",      [30, -30, -20, 0, 0, 0]),
    "BIN_C":      Waypoint("BIN_C",      [-30, -30, -20, 0, 0, 0]),
    "BIN_D":      Waypoint("BIN_D",      [-60, -30, -20, 0, 0, 0]),
}

# Maps fusion decision to bin waypoint name
DECISION_TO_BIN = {
    "PASS": "BIN_PASS",
    "FAIL": "BIN_FAIL",
    "REVIEW": "BIN_REVIEW",
}


def _parse_class_to_bin(env_value: str) -> Dict[str, str]:
    """Parse ``CLASS_TO_BIN`` env var of the form ``"yellow:BIN_A,green:BIN_B"``.

    Empty / malformed values yield an empty mapping (colour sorting disabled,
    falls back to ``DECISION_TO_BIN``).
    """
    mapping: Dict[str, str] = {}
    if not env_value:
        return mapping
    for pair in env_value.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            mapping[key] = value
    return mapping


# Maps detected dominant_color (or class) to a physical bin waypoint.
# When set, takes precedence over DECISION_TO_BIN for the colour-sorting use case.
# Example: CLASS_TO_BIN="yellow:BIN_A,green:BIN_B,red:BIN_C"
CLASS_TO_BIN: Dict[str, str] = _parse_class_to_bin(os.environ.get("CLASS_TO_BIN", ""))


def resolve_bin_name(decision: str, dominant_color: Optional[str] = None) -> str:
    """Resolve target bin given a fusion decision and optional colour class.

    ``dominant_color`` (when supplied) takes precedence via ``CLASS_TO_BIN``,
    so a yellow cube goes to ``BIN_A`` even if the fusion decision is PASS.
    Falls back to ``DECISION_TO_BIN[decision]``, then ``BIN_REVIEW``.
    """
    if dominant_color and dominant_color in CLASS_TO_BIN:
        return CLASS_TO_BIN[dominant_color]
    return DECISION_TO_BIN.get(decision, "BIN_REVIEW")


def _load_waypoints_from_file() -> Optional[Dict[str, Waypoint]]:
    """Load calibrated waypoints from ``waypoints.json`` if it exists.

    The file may also contain a ``via_table`` key for collision routing —
    fetched separately by ``_load_via_table_from_file``.
    """
    if not os.path.isfile(WAYPOINTS_FILE):
        return None
    try:
        with open(WAYPOINTS_FILE) as f:
            data = json.load(f)
        waypoints = {}
        for name, entry in data.items():
            if name == "via_table":
                continue  # routing metadata, not a waypoint
            waypoints[name] = Waypoint(name=name, angles=entry["angles"])
        logger.info("Loaded calibrated waypoints from %s", WAYPOINTS_FILE)
        return waypoints
    except Exception as exc:
        logger.warning("Failed to load waypoints from %s: %s", WAYPOINTS_FILE, exc)
        return None


def _load_via_table_from_file() -> Dict[tuple, List[str]]:
    """Load the waypoint→waypoint via-routing table from ``waypoints.json``.

    Format on disk:

    .. code-block:: json

        "via_table": [
            {"from": "BIN_A", "to": "BIN_C", "via": ["TRANSIT_BACK_LEFT"]},
            {"from": "BIN_C", "to": "BIN_A", "via": ["TRANSIT_BACK_LEFT"]}
        ]

    Returns a dict keyed by ``(from, to)`` for O(1) lookup. Empty when the
    file is missing or the key is absent (default = no routing).
    """
    if not os.path.isfile(WAYPOINTS_FILE):
        return {}
    try:
        with open(WAYPOINTS_FILE) as f:
            data = json.load(f)
        entries = data.get("via_table", [])
        table: Dict[tuple, List[str]] = {}
        for entry in entries:
            src = entry.get("from")
            dst = entry.get("to")
            via = entry.get("via", [])
            if src and dst and isinstance(via, list):
                table[(src, dst)] = list(via)
        if table:
            logger.info("Loaded %d via-routing entries from %s", len(table), WAYPOINTS_FILE)
        return table
    except Exception as exc:
        logger.warning("Failed to load via_table from %s: %s", WAYPOINTS_FILE, exc)
        return {}


class Arm:
    """High-level arm controller for pick-and-place sorting."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        waypoints: Optional[Dict[str, Waypoint]] = None,
        via_table: Optional[Dict[tuple, List[str]]] = None,
        mock: bool = False,
    ) -> None:
        self._mc = get_connection(port, baud, mock=mock)
        self.mock = self._mc is None

        self.waypoints = waypoints or _load_waypoints_from_file() or DEFAULT_WAYPOINTS
        self.via_table = via_table if via_table is not None else _load_via_table_from_file()
        self._last_waypoint_name: Optional[str] = None

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

    MOTION_TIMEOUT = 300  # 5 minutes — effectively no timeout

    def go_to(self, waypoint_name: str) -> None:
        """Move to a named waypoint, routing through any via-points if needed.

        Consults ``self.via_table`` keyed by ``(last_waypoint, target)`` and
        prepends each via-waypoint to avoid known static obstacles (e.g. the
        gooseneck camera mount). If no entry exists, moves directly.
        """
        if waypoint_name not in self.waypoints:
            logger.error("Unknown waypoint: %s", waypoint_name)
            return

        via_chain = self.via_table.get(
            (self._last_waypoint_name, waypoint_name), []
        ) if self._last_waypoint_name else []

        if via_chain:
            logger.info(
                "Routing %s → %s via %s (collision avoidance)",
                self._last_waypoint_name, waypoint_name, via_chain,
            )
            for via_name in via_chain:
                if via_name not in self.waypoints:
                    logger.warning("Via waypoint %s not calibrated — skipping", via_name)
                    continue
                self._move_to_waypoint(via_name)

        self._move_to_waypoint(waypoint_name)
        self._last_waypoint_name = waypoint_name

    def _move_to_waypoint(self, waypoint_name: str) -> None:
        """Direct joint-space move to a calibrated waypoint (no routing)."""
        wp = self.waypoints[waypoint_name]
        if self.mock:
            logger.info("[MOCK] Arm → %s %s", wp.name, wp.angles)
            time.sleep(0.3)
            return
        try:
            logger.info("Arm → %s angles=%s speed=%d", wp.name, wp.angles, wp.speed)
            self._mc.sync_send_angles(wp.angles, wp.speed, timeout=self.MOTION_TIMEOUT)
        except Exception as exc:
            logger.error("Arm _move_to_waypoint(%s) failed: %s", wp.name, exc)

    def go_to_coords(self, coords: List[float], speed: int = 50) -> None:
        """Move to a Cartesian position. Blocks until the arm arrives."""
        if self.mock:
            logger.info("[MOCK] Arm → coords %s", coords)
            time.sleep(0.3)
            return
        try:
            logger.info("Arm → coords [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f] speed=%d",
                        *coords, speed)
            self._mc.sync_send_coords(coords, speed, mode=0, timeout=self.MOTION_TIMEOUT)
        except Exception as exc:
            logger.error("Arm go_to_coords failed: %s", exc)

    # Heights in mm — adjust for your setup
    # Z heights in robot base-frame coordinates (mm)
    # Adjust these to match your setup — check with mc.get_coords() at the surface
    GRIP_HEIGHT_MM = float(os.environ.get("GRIP_HEIGHT_MM", "88"))       # surface level where gripper grabs
    APPROACH_HEIGHT_MM = float(os.environ.get("APPROACH_HEIGHT_MM", "160"))  # above cube, clear for approach
    TRANSIT_HEIGHT_MM = float(os.environ.get("TRANSIT_HEIGHT_MM", "200"))    # safe travel height between positions

    def _try_pick(
        self,
        gripper: Gripper,
        pick_target: PickTarget,
        grip_rz: float,
    ) -> float:
        """Execute one approach → grip → verify cycle. Returns normalised load.

        Caller decides success/failure by comparing the returned load to
        ``GRIP_LOAD_SUCCESS_THRESHOLD``. On any failure the gripper is
        opened and the arm is lifted to TRANSIT height — the caller is
        responsible for parking at SCOUT before re-detection.
        """
        coords = pick_target.coords
        rx, ry = coords[3], coords[4]

        # Move horizontally at transit height to above the target
        above_target = [coords[0], coords[1], self.TRANSIT_HEIGHT_MM, rx, ry, grip_rz]
        self.go_to_coords(above_target)

        # Lower to approach height
        approach = [coords[0], coords[1], self.APPROACH_HEIGHT_MM, rx, ry, grip_rz]
        self.go_to_coords(approach, speed=40)

        # Lower to grip height (slow for precision)
        grip_pos = [coords[0], coords[1], self.GRIP_HEIGHT_MM, rx, ry, grip_rz]
        self.go_to_coords(grip_pos, speed=20)

        # Close gripper
        gripper.close_gripper()

        # Verify load
        time.sleep(GRIP_VERIFY_DELAY_S)
        load = float(gripper.read().normalized_load)

        # Lift to transit height regardless — caller decides next step
        lift_pos = [coords[0], coords[1], self.TRANSIT_HEIGHT_MM, rx, ry, grip_rz]
        self.go_to_coords(lift_pos)

        return load

    def pick_and_place(
        self,
        decision: str,
        gripper: Gripper,
        pick_target: Optional[PickTarget] = None,
        rotation_angle: float = 0.0,
        bin_override: Optional[str] = None,
        redetect_cb: Optional[RedetectFn] = None,
        max_retries: Optional[int] = None,
    ) -> PickOutcome:
        """Execute a full pick → sort cycle with closed-loop retry.

        Sequence (per attempt):
          HOME → open gripper → SCOUT (if retry) → re-detect (if cb)
          → above target → APPROACH → GRIP → close → verify load → lift
          if load ≥ threshold: → bin → release → HOME, return success
          else: → SCOUT → next attempt
        On exhaustion: → SCOUT, return failure (caller forces REVIEW).

        Parameters
        ----------
        decision : str
            Sorting decision: "PASS", "FAIL", or "REVIEW".
        gripper : Gripper
            Gripper driver instance.
        pick_target : PickTarget, optional
            If provided and reachable, uses Cartesian coords. If None, falls
            back to the static PICK waypoint with no closed-loop verify.
        rotation_angle : float
            Detected cube yaw in degrees (CAMERA_ROTATION_OFFSET applied).
        bin_override : str, optional
            Force a specific bin waypoint (overrides DECISION_TO_BIN).
        redetect_cb : callable, optional
            Called between failed attempts to refresh (PickTarget, yaw_deg).
            Signature: ``() -> Optional[(PickTarget, float)]``. If None,
            retries reuse the original pose with a small XY nudge.
        max_retries : int, optional
            Override MAX_PICK_RETRIES for this call (e.g. 0 to disable).
        """
        t0 = time.monotonic()
        bin_name = bin_override or DECISION_TO_BIN.get(decision, "BIN_REVIEW")
        retry_budget = MAX_PICK_RETRIES if max_retries is None else max(0, max_retries)
        logger.info(
            "Pick-and-place: decision=%s → bin=%s (max_retries=%d)",
            decision, bin_name, retry_budget,
        )

        self.go_to("HOME")
        gripper.open_gripper()

        # Static-waypoint fallback — no closed-loop verify possible
        if not (pick_target and pick_target.reachable):
            if pick_target and not pick_target.reachable:
                logger.warning("Pick target unreachable — using fixed PICK waypoint")
            self.go_to("PICK")
            gripper.close_gripper()
            self.go_to(bin_name)
            gripper.release()
            self.go_to("HOME")
            return PickOutcome(
                success=True,
                attempts=1,
                last_load=0.0,
                final_pose=None,
                total_time_s=round(time.monotonic() - t0, 3),
                bin_name=bin_name,
            )

        # Closed-loop retry loop ──────────────────────────────────────
        current_target = pick_target
        current_yaw = rotation_angle
        attempt_loads: List[float] = []
        last_pose: Optional[Tuple[float, float]] = None

        # Lift the arm clear of HOME once before the first approach so
        # the descent path is consistent across attempts.
        home_coords = self._mc.get_coords() if not self.mock else [0, 0, 0, 0, 0, 0]
        if home_coords and len(home_coords) >= 3:
            grip_rz0 = current_yaw + CAMERA_ROTATION_OFFSET
            rx, ry = current_target.coords[3], current_target.coords[4]
            self.go_to_coords(
                [home_coords[0], home_coords[1], self.TRANSIT_HEIGHT_MM, rx, ry, grip_rz0]
            )

        for attempt in range(retry_budget + 1):
            if attempt > 0:
                # Park outside the camera zone for a clean re-detection
                self.go_to("SCOUT")
                gripper.open_gripper()

                if redetect_cb is not None:
                    refreshed = redetect_cb()
                    if refreshed is None:
                        logger.warning(
                            "redetect_cb returned no detection on attempt %d — aborting retries",
                            attempt + 1,
                        )
                        break
                    current_target, current_yaw = refreshed
                    if not current_target.reachable:
                        logger.warning("Re-detected pose unreachable — aborting retries")
                        break

                    # If the new pose hasn't moved, nudge XY a tad to break
                    # any stuck calibration error.
                    new_pose = (current_target.coords[0], current_target.coords[1])
                    if last_pose is not None and (
                        abs(new_pose[0] - last_pose[0]) < 1.0
                        and abs(new_pose[1] - last_pose[1]) < 1.0
                    ):
                        nudged = list(current_target.coords)
                        nudged[0] += RETRY_NUDGE_MM
                        from qa_cell_edge_agent.drivers.transforms import PickTarget as _PT
                        current_target = _PT(
                            coords=nudged,
                            pixel_centre=current_target.pixel_centre,
                            reachable=current_target.reachable,
                            distance_from_base=current_target.distance_from_base,
                        )
                        logger.info("Pose unchanged — applied %.1fmm XY nudge", RETRY_NUDGE_MM)

            grip_rz = current_yaw + CAMERA_ROTATION_OFFSET
            coords = current_target.coords
            logger.info(
                "Pick attempt %d/%d at (%.1f, %.1f) rz=%.1f° — %.1f mm from base",
                attempt + 1, retry_budget + 1,
                coords[0], coords[1], grip_rz, current_target.distance_from_base,
            )
            load = self._try_pick(gripper, current_target, grip_rz)
            attempt_loads.append(load)
            last_pose = (coords[0], coords[1])

            logger.info(
                "Pick attempt %d/%d load=%.3f (threshold=%.3f) angle=%.1f",
                attempt + 1, retry_budget + 1,
                load, GRIP_LOAD_SUCCESS_THRESHOLD, grip_rz,
            )

            if load >= GRIP_LOAD_SUCCESS_THRESHOLD:
                # Success — go to bin, release, home
                self.go_to(bin_name)
                gripper.release()
                self.go_to("HOME")
                return PickOutcome(
                    success=True,
                    attempts=attempt + 1,
                    last_load=load,
                    final_pose=(coords[0], coords[1], current_yaw),
                    total_time_s=round(time.monotonic() - t0, 3),
                    bin_name=bin_name,
                    attempt_loads=attempt_loads,
                )

            gripper.open_gripper()

        # Exhausted retries — leave the part on the workspace, park safely
        logger.error(
            "Pick-and-place failed after %d attempt(s) — last load=%.3f. "
            "Leaving part on workspace; caller should mark REVIEW.",
            len(attempt_loads), attempt_loads[-1] if attempt_loads else 0.0,
        )
        self.go_to("SCOUT")
        gripper.open_gripper()
        return PickOutcome(
            success=False,
            attempts=len(attempt_loads),
            last_load=attempt_loads[-1] if attempt_loads else 0.0,
            final_pose=(current_target.coords[0], current_target.coords[1], current_yaw),
            total_time_s=round(time.monotonic() - t0, 3),
            bin_name=bin_name,
            attempt_loads=attempt_loads,
        )

    def safe_position(self) -> None:
        """Move to HOME and disable servos (E-STOP safe state)."""
        logger.warning("E-STOP: moving to safe position")
        self.go_to("HOME")
        if not self.mock and self._mc:
            self._mc.release_all_servos()

