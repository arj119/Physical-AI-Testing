"""Auto-discovery for myCobot 280 serial port and USB camera.

Eliminates the need to hardcode ``/dev/ttyUSB0`` or camera index ``0``.
Discovery results can be overridden via environment variables when the
automatic detection picks the wrong device.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Known USB VID:PID pairs for the myCobot 280 M5Stack serial interface.
_MYCOBOT_USB_IDS: list[tuple[int, int]] = [
    (0x10C4, 0xEA60),  # Silicon Labs CP210x  (M5Stack)
    (0x1A86, 0x7523),  # QinHeng CH340        (some clones / Pi variant)
    (0x0403, 0x6001),  # FTDI FT232R          (rare)
]


def _validate_serial_port(device: str, baud: int = 115200) -> bool:
    """Try to open the port and get a response from the myCobot."""
    try:
        from pymycobot import MyCobot280
        mc = MyCobot280(device, baud)
        time.sleep(0.5)
        angles = mc.get_angles()
        if angles and len(angles) == 6:
            logger.info("Validated myCobot on %s (angles: %s)", device, angles)
            return True
        logger.debug("Port %s open but get_angles() returned %s", device, angles)
        # Port opened and didn't crash — likely correct even if angles are empty
        return True
    except ImportError:
        # pymycobot not installed — can't validate, trust the VID:PID match
        return True
    except Exception as exc:
        logger.debug("Port %s failed validation: %s", device, exc)
        return False


def find_mycobot_port() -> Optional[str]:
    """Scan USB-serial ports and return the first that matches a known myCobot chip.

    Falls back to the ``MYCOBOT_PORT`` environment variable, then ``None``.
    """
    env = os.environ.get("MYCOBOT_PORT")
    if env:
        logger.info("Using MYCOBOT_PORT from env: %s", env)
        return env

    try:
        import serial.tools.list_ports
    except ImportError:
        logger.warning("pyserial not available — cannot auto-discover serial port")
        return None

    vid_pid_set = set(_MYCOBOT_USB_IDS)

    # Collect all candidates by VID:PID
    candidates = []
    for port in serial.tools.list_ports.comports():
        if port.vid is not None and (port.vid, port.pid) in vid_pid_set:
            candidates.append(port.device)

    if len(candidates) > 1:
        logger.warning(
            "Multiple myCobot-compatible serial ports found: %s — using first. "
            "Set MYCOBOT_PORT in .env to choose explicitly.",
            ", ".join(candidates),
        )

    # Validate each candidate
    for device in candidates:
        if _validate_serial_port(device):
            logger.info("Auto-discovered myCobot on %s", device)
            return device

    # Fallback: look for common tty names on Linux (includes Jetson GPIO UART)
    for pattern in ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyTHS*"]:
        matches = sorted(glob.glob(pattern))
        if len(matches) > 1:
            logger.warning(
                "Multiple serial ports found: %s — set MYCOBOT_PORT in .env",
                ", ".join(matches),
            )
        for device in matches:
            if _validate_serial_port(device):
                logger.info("No VID:PID match — validated fallback %s", device)
                return device

    logger.warning("No myCobot serial port found — hardware will be unavailable")
    return None


def find_camera_index() -> int:
    """Scan ``/dev/video*`` devices and return the first that captures a frame.

    Falls back to the ``CAMERA_DEVICE_INDEX`` environment variable, then ``0``.
    """
    env = os.environ.get("CAMERA_DEVICE_INDEX")
    if env is not None:
        idx = int(env)
        logger.info("Using CAMERA_DEVICE_INDEX from env: %d", idx)
        return idx

    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available — defaulting to camera index 0")
        return 0

    video_devices = sorted(glob.glob("/dev/video*"))
    candidate_indices = []
    for dev in video_devices:
        try:
            candidate_indices.append(int(dev.replace("/dev/video", "")))
        except ValueError:
            continue

    if not candidate_indices:
        candidate_indices = [0]

    for idx in candidate_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                logger.info("Auto-discovered camera at index %d (%dx%d)", idx, w, h)
                return idx
        else:
            cap.release()

    logger.warning("No working camera found — defaulting to index 0")
    return 0
