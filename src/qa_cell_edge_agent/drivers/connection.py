"""Shared MyCobot280 connection singleton.

Ensures only one serial connection exists per (port, baud) pair within a
process.  ``MyCobot280`` already provides ``thread_lock=True`` by default,
so concurrent access from multiple threads within the same process is safe.

Cross-process access must be avoided by design — all hardware I/O should
live in a single process (defect_detection).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from pymycobot import MyCobot280
except ImportError:
    MyCobot280 = None  # type: ignore[assignment,misc]

_instances: Dict[Tuple[str, int], "MyCobot280"] = {}


def get_connection(
    port: str = "/dev/ttyUSB0",
    baud: int = 115200,
    mock: bool = False,
) -> Optional["MyCobot280"]:
    """Return a shared ``MyCobot280`` instance for the given port.

    Returns ``None`` when *mock* is ``True`` or pymycobot is unavailable.
    """
    if mock or MyCobot280 is None:
        return None

    key = (port, baud)
    if key in _instances:
        return _instances[key]

    try:
        mc = MyCobot280(port, baud)
        time.sleep(0.5)  # allow serial to settle
        logger.info("MyCobot280 connected on %s @ %d", port, baud)
        _instances[key] = mc
        return mc
    except Exception as exc:
        logger.error("MyCobot280 connection failed on %s: %s", port, exc)
        return None
