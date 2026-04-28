#!/usr/bin/env python3
"""Hardware test for the closed-loop retry path.

Forces a missed grasp by intentionally offsetting the commanded pick
target by --miss-mm and provides a redetect_cb that returns the *true*
pose on retry. Asserts:
  - first try_pick returns failure (load < threshold),
  - redetect_cb is invoked exactly once,
  - second try_pick succeeds.

Usage:
    python scripts/test_retry_loop.py
    python scripts/test_retry_loop.py --miss-mm 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Closed-loop retry test on real hardware")
    parser.add_argument("--miss-mm", type=float, default=6.0,
                        help="XY offset (mm) applied to first attempt to force a miss")
    parser.add_argument("--bin", type=str, default="BIN_REVIEW",
                        help="Bin to drop the cube in if retry succeeds")
    parser.add_argument("--mock", action="store_true", help="Mock all hardware")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    import cv2
    from qa_cell_edge_agent.drivers.arm import Arm
    from qa_cell_edge_agent.drivers.block_detector import BlockDetector
    from qa_cell_edge_agent.drivers.gripper import Gripper
    from qa_cell_edge_agent.drivers.transforms import CameraTransform, PickTarget
    from qa_cell_edge_agent.drivers.workspace import WorkspaceMonitor

    port = os.environ.get("MYCOBOT_PORT", "/dev/ttyUSB0")
    baud = int(os.environ.get("MYCOBOT_BAUD", "115200"))
    arm = Arm(port=port, baud=baud, mock=args.mock)
    grip = Gripper(port=port, baud=baud, mock=args.mock)
    transform = CameraTransform()
    workspace = WorkspaceMonitor()
    detector = BlockDetector()

    if not transform.is_calibrated:
        print(f"{FAIL} Camera not calibrated.")
        sys.exit(1)
    if args.bin not in arm.waypoints:
        print(f"{FAIL} Bin '{args.bin}' not calibrated.")
        sys.exit(1)

    cam_idx = int(os.environ.get("CAMERA_DEVICE_INDEX", "0"))
    cap = cv2.VideoCapture(cam_idx) if not args.mock else None
    if cap is not None and not cap.isOpened():
        print(f"{FAIL} Cannot open camera {cam_idx}")
        sys.exit(1)

    print(f"\n=== Closed-Loop Retry Test ===")
    print(f"  Miss offset: {args.miss_mm:.1f} mm\n")
    input("  Place a cube in the camera zone, then press ENTER... ")

    # Initial detection
    true_target, true_yaw = _detect(cap, detector, transform, workspace, args.mock)
    if true_target is None:
        print(f"  {FAIL} No detection.")
        sys.exit(1)
    print(f"  True pose: ({true_target.coords[0]:.1f}, {true_target.coords[1]:.1f}) yaw={true_yaw:+.1f}°")

    # Build the deliberately-wrong first target
    wrong_coords = list(true_target.coords)
    wrong_coords[0] += args.miss_mm  # offset X to miss
    wrong_target = PickTarget(
        coords=wrong_coords,
        pixel_centre=true_target.pixel_centre,
        reachable=True,
        distance_from_base=true_target.distance_from_base,
    )

    redetect_calls = {"n": 0}

    def cb():
        redetect_calls["n"] += 1
        # Return the *true* pose so the retry succeeds
        return true_target, true_yaw

    t0 = time.monotonic()
    outcome = arm.pick_and_place(
        decision="REVIEW",
        gripper=grip,
        pick_target=wrong_target,
        rotation_angle=true_yaw,
        bin_override=args.bin,
        redetect_cb=cb,
        max_retries=2,
    )
    elapsed = time.monotonic() - t0

    if cap is not None:
        cap.release()

    print(f"\n  attempts:        {outcome.attempts}")
    print(f"  attempt loads:   {[round(v, 3) for v in outcome.attempt_loads]}")
    print(f"  redetect calls:  {redetect_calls['n']}")
    print(f"  success:         {outcome.success}")
    print(f"  cycle time:      {elapsed:.1f} s\n")

    # Assertions
    fail = False
    if outcome.attempts < 2:
        print(f"  {FAIL} Expected ≥ 2 attempts, got {outcome.attempts}.")
        print("       The first attempt may have succeeded — increase --miss-mm.")
        fail = True
    if redetect_calls["n"] != outcome.attempts - 1:
        print(f"  {FAIL} Expected redetect_cb called exactly {outcome.attempts - 1}× "
              f"(once per retry), got {redetect_calls['n']}.")
        fail = True
    if not outcome.success:
        print(f"  {FAIL} Retry did not succeed — last load {outcome.last_load:.3f}.")
        fail = True

    if fail:
        sys.exit(1)
    print(f"  {PASS} Retry path engaged correctly: miss → redetect → success.\n")


def _detect(cap, detector, transform, workspace, mock):
    if mock:
        from qa_cell_edge_agent.drivers.transforms import PickTarget
        return PickTarget(
            coords=[120.0, 30.0, 50.0, 0.0, 0.0, 0.0],
            pixel_centre=(320, 240),
            reachable=True,
            distance_from_base=124.0,
        ), 12.5
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None
    zone_mask = workspace._zone_mask if workspace.is_configured else None
    det = detector.detect(frame, zone_mask=zone_mask)
    if det is None:
        return None, None
    cx = int(det.bounding_box[0] + det.bounding_box[2] / 2)
    cy = int(det.bounding_box[1] + det.bounding_box[3] / 2)
    return transform.pixel_to_robot(cx, cy), float(det.rotation_angle)


if __name__ == "__main__":
    main()
