"""myCobot 280 gripper driver — reads servo load and controls grip state.

When ``mock=True`` the driver returns synthetic load values so the
pipeline can run without the physical robot attached.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from qa_cell_edge_agent.drivers.connection import get_connection

logger = logging.getLogger(__name__)


@dataclass
class GripData:
    """Snapshot of a single gripper measurement."""

    servo_load: float        # raw servo load value
    normalized_load: float   # normalised to [0, 1]
    grip_state: str          # OPEN | CLOSING | CLOSED | RELEASING
    object_detected: bool    # whether the gripper sensed an object


class Gripper:
    """Interface to the myCobot 280 gripper servo."""

    # Typical max servo load — calibrate for your specific unit
    MAX_SERVO_LOAD = 500.0

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        mock: bool = False,
    ) -> None:
        self._mc = get_connection(port, baud, mock=mock)
        self.mock = self._mc is None
        self._grip_state = "OPEN"

        if self.mock:
            logger.info("Gripper running in MOCK mode")
        else:
            logger.info("Gripper ready (shared connection on %s @ %d)", port, baud)

    # ── public API ────────────────────────────────────────────────────

    def read(self) -> GripData:
        """Take a single gripper measurement and return a GripData snapshot."""
        if self.mock:
            return self._mock_read()

        # Use get_gripper_value() (0=fully closed, 100=fully open) as the
        # primary reading.  Fall back to raw servo register if unsupported.
        try:
            raw = self._mc.get_gripper_value()
            if raw is not None:
                # Invert: gripper_value 0=closed (max load) → normalized 1.0
                servo_load = max(0.0, float(100 - raw))
                normalized = min(servo_load / 100.0, 1.0)
            else:
                servo_load = float(self._mc.get_servo_data(7, 14) or 0)
                normalized = min(servo_load / self.MAX_SERVO_LOAD, 1.0)
        except Exception as exc:
            logger.warning("Failed to read gripper: %s", exc)
            servo_load = 0.0
            normalized = 0.0

        detected = normalized > 0.1
        return GripData(
            servo_load=round(servo_load, 2),
            normalized_load=round(normalized, 4),
            grip_state=self._grip_state,
            object_detected=detected,
        )

    def open_gripper(self) -> None:
        """Open the gripper."""
        self._grip_state = "OPEN"
        if not self.mock:
            self._mc.set_gripper_value(100, 50)
            time.sleep(0.5)

    def close_gripper(self) -> None:
        """Close the gripper."""
        self._grip_state = "CLOSING"
        if not self.mock:
            self._mc.set_gripper_value(0, 50)
            time.sleep(1.0)
        self._grip_state = "CLOSED"

    def release(self) -> None:
        """Release (open) the gripper after placing a part."""
        self._grip_state = "RELEASING"
        self.open_gripper()

    def read_joint_temperatures(self) -> list[float]:
        """Read servo temperatures for joints 1-6. Returns 6 floats in degrees C."""
        if self.mock:
            return [round(random.gauss(40.0, 5.0), 1) for _ in range(6)]
        try:
            temps = self._mc.get_joints_temperature()
            if temps and len(temps) >= 6:
                return [float(t) for t in temps[:6]]
            return [0.0] * 6
        except Exception as exc:
            logger.warning("Failed to read joint temps: %s", exc)
            return [0.0] * 6

    # ── internals ─────────────────────────────────────────────────────

    def _mock_read(self) -> GripData:
        """Return a synthetic gripper measurement."""
        servo_load = random.uniform(50.0, 250.0)
        normalized = round(servo_load / self.MAX_SERVO_LOAD, 4)
        return GripData(
            servo_load=round(servo_load, 2),
            normalized_load=normalized,
            grip_state=self._grip_state,
            object_detected=random.random() > 0.1,
        )
