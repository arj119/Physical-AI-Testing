"""Shared MyCobot280 connection singleton.

Ensures only one serial connection exists per (port, baud) pair within a
process.  ``MyCobot280`` already provides ``thread_lock=True`` by default,
so concurrent access from multiple threads within the same process is safe.

Cross-process access must be avoided by design — all hardware I/O should
live in a single process (defect_detection). A PID ownership check enforces
this at runtime.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from pymycobot import MyCobot280
except ImportError:
    MyCobot280 = None  # type: ignore[assignment,misc]

_instances: Dict[Tuple[str, int], "MyCobot280"] = {}
_owner_pid: Optional[int] = None


def get_connection(
    port: str = "/dev/ttyUSB0",
    baud: int = 115200,
    mock: bool = False,
) -> Optional["MyCobot280"]:
    """Return a shared ``MyCobot280`` instance for the given port.

    Returns ``None`` when *mock* is ``True`` or pymycobot is unavailable.
    Enforces single-process ownership — logs an error if called from a
    different PID than the one that created the first connection.
    """
    global _owner_pid

    if mock or MyCobot280 is None:
        return None

    current_pid = os.getpid()
    if _owner_pid is not None and _owner_pid != current_pid:
        logger.error(
            "Serial connection requested from PID %d but owned by PID %d. "
            "Only defect_detection (Process 2) should access hardware.",
            current_pid, _owner_pid,
        )
        return None

    key = (port, baud)
    if key in _instances:
        return _instances[key]

    try:
        mc = MyCobot280(port, baud)
        time.sleep(0.5)
        _owner_pid = current_pid
        logger.info("MyCobot280 connected on %s @ %d (owner PID %d)", port, baud, current_pid)
        _instances[key] = mc
        return mc
    except Exception as exc:
        logger.error("MyCobot280 connection failed on %s: %s", port, exc)
        return None
