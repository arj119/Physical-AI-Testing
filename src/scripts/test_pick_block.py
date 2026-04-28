#!/usr/bin/env python3
"""Single-cube closed-loop pick test.

Operator places a cube in the camera zone; the script runs one
detection cycle, transforms to robot coordinates, and executes a full
pick_and_place with closed-loop retry. Reports PickOutcome.

Usage:
    python scripts/test_pick_block.py
    python scripts/test_pick_block.py --bin BIN_A
    python scripts/test_pick_block.py --no-place           # just pick & lift, don't drop
    python scripts/test_pick_block.py --repeat 5           # loop, restage between
    python scripts/test_pick_block.py --max-retries 3
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
WARN = "\033[93m!\033[0m"


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Single-cube closed-loop pick test")
    parser.add_argument("--bin", type=str, default="BIN_PASS",
                        help="Target bin waypoint (default BIN_PASS)")
    parser.add_argument("--no-place", action="store_true",
                        help="Skip placing in bin — pick and return HOME with cube")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of cycles (operator restages between)")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Override MAX_PICK_RETRIES")
    parser.add_argument("--mock", action="store_true", help="Mock all hardware")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    import cv2
    from qa_cell_edge_agent.drivers.arm import Arm
    from qa_cell_edge_agent.drivers.block_detector import BlockDetector
    from qa_cell_edge_agent.drivers.gripper import Gripper
    from qa_cell_edge_agent.drivers.transforms import CameraTransform
    from qa_cell_edge_agent.drivers.workspace import WorkspaceMonitor

    port = os.environ.get("MYCOBOT_PORT", "/dev/ttyUSB0")
    baud = int(os.environ.get("MYCOBOT_BAUD", "115200"))
    arm = Arm(port=port, baud=baud, mock=args.mock)
    grip = Gripper(port=port, baud=baud, mock=args.mock)
    transform = CameraTransform()
    workspace = WorkspaceMonitor()
    detector = BlockDetector()

    if not transform.is_calibrated:
        print(f"{FAIL} Camera not calibrated. Run scripts/calibrate_camera.py first.")
        sys.exit(1)
    if args.bin not in arm.waypoints:
        print(f"{FAIL} Bin waypoint '{args.bin}' not calibrated.")
        sys.exit(1)

    cam_idx = int(os.environ.get("CAMERA_DEVICE_INDEX", "0"))
    cap = cv2.VideoCapture(cam_idx) if not args.mock else None
    if cap is not None and not cap.isOpened():
        print(f"{FAIL} Cannot open camera index {cam_idx}")
        sys.exit(1)

    print(f"\n=== Closed-Loop Pick Test ===")
    print(f"  Target bin:  {args.bin}")
    print(f"  Repeats:     {args.repeat}")
    print(f"  Max retries: {args.max_retries if args.max_retries is not None else 'env default'}\n")

    successes = 0
    cycle_times = []
    attempts_log = []
    loads_log = []

    try:
        for i in range(args.repeat):
            print(f"\n── Cycle {i + 1}/{args.repeat} ──")
            input("  Place a cube in the camera zone, then press ENTER... ")

            target, yaw = _detect_once(cap, detector, transform, workspace, args.mock)
            if target is None:
                print(f"  {FAIL} No detection — skipping this cycle.")
                continue

            print(f"  Detected at robot ({target.coords[0]:.1f}, {target.coords[1]:.1f}) "
                  f"yaw={yaw:+.1f}°  reachable={target.reachable}")

            redetect_cb = (lambda: _detect_once(cap, detector, transform, workspace, args.mock)) \
                if cap is not None or args.mock else None
            # redetect_cb returns (PickTarget, float) or None — match the type
            def cb():
                pair = _detect_once(cap, detector, transform, workspace, args.mock)
                return None if pair[0] is None else pair

            t0 = time.monotonic()
            outcome = arm.pick_and_place(
                decision="PASS",
                gripper=grip,
                pick_target=target,
                rotation_angle=yaw,
                bin_override=("HOME" if args.no_place else args.bin),
                redetect_cb=cb,
                max_retries=args.max_retries,
            )
            elapsed = time.monotonic() - t0

            cycle_times.append(elapsed)
            attempts_log.append(outcome.attempts)
            loads_log.append(outcome.last_load)
            if outcome.success:
                successes += 1

            marker = PASS if outcome.success else FAIL
            print(f"  {marker} success={outcome.success}  attempts={outcome.attempts}  "
                  f"load={outcome.last_load:.3f}  cycle={elapsed:.1f}s  "
                  f"loads={[round(v, 3) for v in outcome.attempt_loads]}")
    finally:
        if cap is not None:
            cap.release()

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    n = len(cycle_times)
    if n:
        avg_t = sum(cycle_times) / n
        avg_a = sum(attempts_log) / n
        avg_l = sum(loads_log) / n
        sr = successes / n
        print(f"  Cycles:    {n}")
        print(f"  Success:   {successes}/{n}  ({sr:.0%})")
        print(f"  Cycle time mean: {avg_t:.1f} s")
        print(f"  Attempts mean:   {avg_a:.2f}")
        print(f"  Load mean:       {avg_l:.3f}")
    else:
        print(f"  {FAIL} No completed cycles.")
    print("=" * 60 + "\n")
    sys.exit(0 if successes == n and n > 0 else 1)


def _detect_once(cap, detector, transform, workspace, mock):
    """Grab a frame, detect the largest block, return (PickTarget, yaw) or (None, None)."""
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
    target = transform.pixel_to_robot(cx, cy)
    return target, float(det.rotation_angle)


if __name__ == "__main__":
    main()
