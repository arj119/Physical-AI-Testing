"""Jetson Nano 2GB memory management helpers.

With only 2GB of RAM shared between CPU and GPU, the agent must be careful
about memory usage.  This module provides utilities for:

- Constraining TensorRT workspace size
- Setting OpenCV buffer limits
- Monitoring available memory before operations
- Recommended trtexec flags for the 2GB variant
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# TensorRT workspace limits for Jetson Nano 2GB.
# The default 1024 MB is too large — leaves no room for the OS + Python.
TRTEXEC_WORKSPACE_MB = 256
TRTEXEC_FP16 = True

# Recommended trtexec command for converting ONNX → TensorRT on Nano 2GB:
# trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --workspace=256


def get_trtexec_args(
    onnx_path: str,
    engine_path: str,
    workspace_mb: int = TRTEXEC_WORKSPACE_MB,
) -> list[str]:
    """Build trtexec command-line arguments for Jetson Nano 2GB."""
    trtexec = os.environ.get("TRTEXEC_PATH", "/usr/src/tensorrt/bin/trtexec")
    args = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]
    if TRTEXEC_FP16:
        args.append("--fp16")
    return args


def check_memory(min_free_mb: int = 200) -> bool:
    """Check if enough free memory is available.

    Returns True if at least ``min_free_mb`` MB of RAM is free.
    On non-Linux platforms, always returns True.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    available_mb = available_kb // 1024
                    if available_mb < min_free_mb:
                        logger.warning(
                            "Low memory: %d MB available (minimum %d MB)",
                            available_mb, min_free_mb,
                        )
                        return False
                    return True
    except (OSError, ValueError):
        pass
    return True
