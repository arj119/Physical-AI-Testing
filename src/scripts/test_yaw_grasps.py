#!/usr/bin/env python3
"""Yaw-sweep pick test — verify orientation handling end-to-end.

For each angle in --angles, the operator places the cube aligned with
that yaw against a printed protractor sheet (or a 3D-printed jig).
The script detects, picks, places into a discard bin, and records
the residual between the visual yaw and the detected yaw.

Reports per-angle success rate over --reps repetitions. Exits non-zero
if any bucket has < --min-success-rate first-attempt success.

Usage:
    python scripts/test_yaw_grasps.py
    python scripts/test_yaw_grasps.py --angles 0,22.5,45,67.5 --reps 3
    python scripts/test_yaw_grasps.py --bin BIN_REVIEW
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


def _norm_square_yaw(a: float) -> float:
    """Mirror of orientation.normalize_square_yaw to avoid the import."""
    out = ((a + 45.0) % 90.0) - 45.0
    return 45.0 if out == -45.0 else out


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Yaw-sweep pick test")
    parser.add_argument("--angles", type=str, default="0,22.5,45,67.5",
                        help="Comma-separated yaw angles (deg)")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per angle")
    parser.add_argument("--bin", type=str, default="BIN_REVIEW",
                        help="Discard bin (default BIN_REVIEW)")
    parser.add_argument("--max-retries", type=int, default=0,
                        help="Override MAX_PICK_RETRIES (default 0 = no retries → first-attempt SR)")
    parser.add_argument("--min-success-rate", type=float, default=0.80,
                        help="Per-bucket success rate required to exit 0")
    parser.add_argument("--max-yaw-residual-deg", type=float, default=5.0,
                        help="Max |detected - visual| in deg (under 90° symmetry)")
    parser.add_argument("--mock", action="store_true", help="Mock all hardware")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    angles = [float(a) for a in args.angles.split(",") if a.strip()]
    if not angles:
        print(f"{FAIL} No angles parsed.")
        sys.exit(1)

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

    print(f"\n=== Yaw Sweep ===")
    print(f"  Angles: {angles}  (×{args.reps} reps)\n")

    rows = []  # (visual_yaw, detected_yaw, residual, success, load, attempts)
    try:
        for visual in angles:
            for r in range(args.reps):
                print(f"\n── visual_yaw={visual:+.1f}°  rep {r + 1}/{args.reps} ──")
                input("  Align cube at this yaw, then press ENTER... ")

                target, detected = _detect(cap, detector, transform, workspace, args.mock, visual)
                if target is None:
                    print(f"  {FAIL} No detection — recording miss.")
                    rows.append((visual, None, None, False, 0.0, 0))
                    continue

                residual = abs(_norm_square_yaw(detected - visual))
                print(f"  detected={detected:+.1f}°  residual={residual:.1f}°")

                outcome = arm.pick_and_place(
                    decision="REVIEW",
                    gripper=grip,
                    pick_target=target,
                    rotation_angle=detected,
                    bin_override=args.bin,
                    max_retries=args.max_retries,
                )
                rows.append((visual, detected, residual,
                             outcome.success, outcome.last_load, outcome.attempts))
                marker = PASS if outcome.success else FAIL
                print(f"  {marker} success={outcome.success}  load={outcome.last_load:.3f}")
    finally:
        if cap is not None:
            cap.release()

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  {'visual':>8}  {'detected':>9}  {'residual':>9}  {'success':>8}  {'load':>6}")
    for visual, det, res, success, load, _ in rows:
        det_s = f"{det:+.1f}" if det is not None else "  --"
        res_s = f"{res:.1f}" if res is not None else " --"
        print(f"  {visual:+8.1f}  {det_s:>9}  {res_s:>9}  {str(success):>8}  {load:>6.3f}")
    print("=" * 64)

    fail = False
    print(f"\n  Per-yaw first-attempt success rate (target ≥ {args.min_success_rate:.0%}):")
    for visual in angles:
        bucket = [r for r in rows if r[0] == visual]
        n = len(bucket)
        ok = sum(1 for r in bucket if r[3])
        sr = ok / n if n else 0.0
        marker = PASS if sr >= args.min_success_rate else FAIL
        print(f"    {marker} {visual:+6.1f}°: {ok}/{n} ({sr:.0%})")
        if sr < args.min_success_rate:
            fail = True

    # Yaw residual gate
    bad_residuals = [r for r in rows if r[2] is not None and r[2] > args.max_yaw_residual_deg]
    if bad_residuals:
        print(f"\n  {FAIL} {len(bad_residuals)} sample(s) exceeded yaw residual "
              f"{args.max_yaw_residual_deg}°  — recheck CAMERA_ROTATION_OFFSET_DEG.")
        fail = True

    print()
    sys.exit(1 if fail else 0)


def _detect(cap, detector, transform, workspace, mock, visual_yaw):
    if mock:
        from qa_cell_edge_agent.drivers.transforms import PickTarget
        # Mock: report the visual yaw with a small noise injection
        import random
        noisy = visual_yaw + random.gauss(0, 1.0)
        # Map to [-45, 45]
        noisy = ((noisy + 45.0) % 90.0) - 45.0
        target = PickTarget(
            coords=[120.0, 30.0, 50.0, 0.0, 0.0, 0.0],
            pixel_centre=(320, 240),
            reachable=True,
            distance_from_base=124.0,
        )
        return target, noisy

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
