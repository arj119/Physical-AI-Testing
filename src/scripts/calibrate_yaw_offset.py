#!/usr/bin/env python3
"""Calibrate the camera→J6 yaw offset (CAMERA_ROTATION_OFFSET_DEG).

Procedure:
  1. Operator places the cube aligned with the workspace X-axis (so
     the gripper at J6=0 would close along the cube's shorter face).
  2. Script samples N camera frames, runs detection, takes the
     circular median of the reported yaw angles.
  3. Prints the recommended ``CAMERA_ROTATION_OFFSET_DEG`` (= -median)
     and optionally appends/updates ``.env`` with --write-env.

Usage:
    python scripts/calibrate_yaw_offset.py
    python scripts/calibrate_yaw_offset.py --samples 20
    python scripts/calibrate_yaw_offset.py --write-env
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")


def _circular_median_90(angles_deg) -> float:
    """Median of yaws under 90° symmetry (≡ PoseBuffer._circular_median_90)."""
    sin_sum = cos_sum = 0.0
    for a in angles_deg:
        rad = math.radians(a) * 4.0
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)
    if not angles_deg:
        return 0.0
    deg = math.degrees(math.atan2(sin_sum, cos_sum)) / 4.0
    deg = ((deg + 45.0) % 90.0) - 45.0
    return 45.0 if deg == -45.0 else deg


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Calibrate camera→J6 yaw offset")
    parser.add_argument("--samples", type=int, default=10, help="Frames to sample")
    parser.add_argument("--timeout", type=float, default=15.0, help="Total seconds to wait for samples")
    parser.add_argument("--write-env", action="store_true", help="Append/update .env with the result")
    parser.add_argument("--mock", action="store_true", help="Mock camera (return synthetic angles)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print("\n=== Camera → J6 Yaw Offset Calibration ===\n")
    print("Place the cube in the camera zone with one face parallel to the")
    print("workspace X-axis (the direction the gripper closes when J6=0).")
    print("Hold still — the script samples a few frames and reports the median.\n")
    input("Press ENTER when the cube is in position... ")

    if args.mock:
        # Synthetic samples around 12.5° to validate the script offline
        import random
        samples = [12.5 + random.gauss(0, 0.3) for _ in range(args.samples)]
    else:
        samples = _capture_samples(args.samples, args.timeout)

    if not samples:
        print(f"\n  {FAIL} No detections in {args.timeout:.0f}s — check camera and lighting.")
        sys.exit(1)

    median = _circular_median_90(samples)
    spread = max(abs(((a - median + 45.0) % 90.0) - 45.0) for a in samples)
    offset = -median  # negate so applied + measured = 0

    print(f"\n  Samples: {[round(a, 1) for a in samples]}")
    print(f"  Median yaw: {median:+.2f}°  (spread: {spread:.2f}°)")
    print(f"  Recommended CAMERA_ROTATION_OFFSET_DEG = {offset:+.2f}\n")

    if spread > 5.0:
        print(f"  {WARN} Spread > 5° — re-position the cube and run again.\n")

    if args.write_env:
        _write_env_var("CAMERA_ROTATION_OFFSET_DEG", f"{offset:.2f}")
        print(f"  {PASS} Updated {ENV_PATH}")
    else:
        print("  Add to .env:")
        print(f"    CAMERA_ROTATION_OFFSET_DEG={offset:.2f}\n")


def _capture_samples(n_target: int, timeout_s: float):
    """Open the camera + detector, return list of yaws (length ≤ n_target)."""
    try:
        import cv2  # noqa: F401
        from qa_cell_edge_agent.drivers.block_detector import BlockDetector
        from qa_cell_edge_agent.drivers.workspace import WorkspaceMonitor
    except ImportError as exc:
        print(f"\n  {FAIL} Missing dependency: {exc}")
        sys.exit(1)
    import cv2

    cam_idx = int(os.environ.get("CAMERA_DEVICE_INDEX", "0"))
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"  {FAIL} Cannot open camera index {cam_idx}")
        sys.exit(1)

    detector = BlockDetector()
    workspace = WorkspaceMonitor()
    samples = []
    t0 = time.monotonic()

    try:
        while len(samples) < n_target and (time.monotonic() - t0) < timeout_s:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            zone_mask = workspace._zone_mask if workspace.is_configured else None
            det = detector.detect(frame, zone_mask=zone_mask)
            if det is None:
                continue
            samples.append(float(det.rotation_angle))
            print(f"    sample {len(samples):2d}/{n_target}: yaw={det.rotation_angle:+.1f}° "
                  f"({det.dominant_color})")
            time.sleep(0.15)
    finally:
        cap.release()

    return samples


def _write_env_var(key: str, value: str) -> None:
    """Append or update ``key=value`` in .env, preserving other lines."""
    if not os.path.isfile(ENV_PATH):
        with open(ENV_PATH, "w") as f:
            f.write(f"{key}={value}\n")
        return
    with open(ENV_PATH) as f:
        lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(f"{key}={value}\n")
    with open(ENV_PATH, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
