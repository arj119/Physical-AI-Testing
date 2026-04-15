"""myCobot 280 gripper driver — reads servo load and controls grip state.

When ``mock=True`` the driver returns synthetic load values so the
pipeline can run without the physical robot attached.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pymycobot.mycobot import MyCobot
except ImportError:
    MyCobot = None  # type: ignore[assignment,misc]
    logger.warning("pymycobot not available — gripper will run in mock mode")


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
        self.mock = mock or MyCobot is None
        self._mc: Optional[MyCobot] = None
        self._grip_state = "OPEN"

        if not self.mock:
            try:
                self._mc = MyCobot(port, baud)
                time.sleep(0.5)  # allow serial to settle
                logger.info("Gripper connected on %s @ %d", port, baud)
            except Exception as exc:
                logger.error("Gripper init failed: %s — falling back to mock", exc)
                self.mock = True
        if self.mock:
            logger.info("Gripper running in MOCK mode")

    # ── public API ────────────────────────────────────────────────────

    def read(self) -> GripData:
        """Take a single gripper measurement and return a GripData snapshot."""
        if self.mock:
            return self._mock_read()

        try:
            servo_load = float(self._mc.get_servo_data(7, 14) or 0)
        except Exception as exc:
            logger.warning("Failed to read servo load: %s", exc)
            servo_load = 0.0

        normalized = min(servo_load / self.MAX_SERVO_LOAD, 1.0)
        detected = normalized > 0.1
        return GripData(
            servo_load=servo_load,
            normalized_load=round(normalized, 4),
            grip_state=self._grip_state,
            object_detected=detected,
        )

    def open_gripper(self) -> None:
        """Open the gripper."""
        self._grip_state = "OPEN"
        if not self.mock:
            self._mc.set_gripper_state(0, 50)
            time.sleep(0.5)

    def close_gripper(self) -> None:
        """Close the gripper."""
        self._grip_state = "CLOSING"
        if not self.mock:
            self._mc.set_gripper_state(1, 50)
            time.sleep(1.0)
        self._grip_state = "CLOSED"

    def release(self) -> None:
        """Release (open) the gripper after placing a part."""
        self._grip_state = "RELEASING"
        self.open_gripper()

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
