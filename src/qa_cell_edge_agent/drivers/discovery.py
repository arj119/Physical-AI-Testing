"""Auto-discovery for myCobot 280 serial port and USB camera.

Eliminates the need to hardcode ``/dev/ttyUSB0`` or camera index ``0``.
Discovery results can be overridden via environment variables when the
automatic detection picks the wrong device.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Known USB VID:PID pairs for the myCobot 280 M5Stack serial interface.
# The M5Stack base uses a Silicon Labs CP210x USB-UART bridge.
# Add entries here if your unit uses a different chip (e.g. CH340).
_MYCOBOT_USB_IDS: list[tuple[int, int]] = [
    (0x10C4, 0xEA60),  # Silicon Labs CP210x  (M5Stack)
    (0x1A86, 0x7523),  # QinHeng CH340        (some clones / Pi variant)
    (0x0403, 0x6001),  # FTDI FT232R          (rare)
]


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

    for port in serial.tools.list_ports.comports():
        if port.vid is not None and (port.vid, port.pid) in vid_pid_set:
            logger.info(
                "Auto-discovered myCobot on %s (VID:PID %04X:%04X, %s)",
                port.device, port.vid, port.pid, port.description,
            )
            return port.device

    # Fallback: look for common tty names on Linux
    for pattern in ["/dev/ttyUSB*", "/dev/ttyACM*"]:
        matches = sorted(glob.glob(pattern))
        if matches:
            logger.info("No VID:PID match — falling back to %s", matches[0])
            return matches[0]

    logger.warning("No myCobot serial port found — hardware will be unavailable")
    return None


def find_camera_index() -> int:
    """Scan ``/dev/video*`` devices and return the first that opens with OpenCV.

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

    # On Linux, /dev/video{0,2,4,...} are typically capture nodes
    # (odd numbers are metadata nodes for the same device).
    video_devices = sorted(glob.glob("/dev/video*"))
    candidate_indices = []
    for dev in video_devices:
        try:
            idx = int(dev.replace("/dev/video", ""))
            candidate_indices.append(idx)
        except ValueError:
            continue

    if not candidate_indices:
        # macOS or no /dev/video* — just try index 0
        candidate_indices = [0]

    for idx in candidate_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Read one frame to confirm it's a real capture device
            ret, _ = cap.read()
            cap.release()
            if ret:
                logger.info("Auto-discovered camera at index %d (/dev/video%d)", idx, idx)
                return idx
            cap.release()
        else:
            cap.release()

    logger.warning("No working camera found — defaulting to index 0")
    return 0
