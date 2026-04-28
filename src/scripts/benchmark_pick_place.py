#!/usr/bin/env python3
"""N-cycle pick-and-place benchmark.

Soak test for the closed-loop pipeline. Operator restages the cube
between cycles. Aggregates cycle time, first-attempt success rate,
mean attempts, mean grip load, and writes a CSV per run for A/B
comparison after parameter changes.

Usage:
    python scripts/benchmark_pick_place.py --cycles 20
    python scripts/benchmark_pick_place.py --cycles 50 --bin BIN_A
    python scripts/benchmark_pick_place.py --no-prompt    # auto-cycle (e.g. with feeder)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import statistics
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Pick-and-place benchmark")
    parser.add_argument("--cycles", type=int, default=20, help="Total cycles to run")
    parser.add_argument("--bin", type=str, default="BIN_PASS", help="Drop bin")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Don't wait for ENTER between cycles (use with feeder)")
    parser.add_argument("--inter-cycle-s", type=float, default=2.0,
                        help="Sleep between cycles when --no-prompt (default 2s)")
    parser.add_argument("--mock", action="store_true", help="Mock all hardware")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(message)s",
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

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"pick_place_{stamp}.csv")

    print(f"\n=== Pick-and-Place Benchmark ===")
    print(f"  Cycles: {args.cycles}  Bin: {args.bin}  Output: {csv_path}\n")

    rows = []
    detect_latencies = []
    cycle_times = []
    attempts_log = []
    loads_log = []
    successes = 0

    bench_t0 = time.monotonic()
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cycle", "timestamp", "detect_ms", "cycle_s",
                "success", "attempts", "last_load", "x_mm", "y_mm", "yaw_deg",
            ])

            for i in range(args.cycles):
                if not args.no_prompt:
                    input(f"\n  [{i + 1}/{args.cycles}] Stage cube, press ENTER... ")
                else:
                    print(f"\n  [{i + 1}/{args.cycles}] cycling in {args.inter_cycle_s}s...")
                    time.sleep(args.inter_cycle_s)

                t_det = time.monotonic()
                target, yaw = _detect(cap, detector, transform, workspace, args.mock)
                det_ms = int((time.monotonic() - t_det) * 1000)
                if target is None:
                    print(f"  {FAIL} No detection — skipping.")
                    writer.writerow([i + 1, _ts(), det_ms, 0.0, False, 0, 0.0, 0, 0, 0])
                    continue

                def cb():
                    pair = _detect(cap, detector, transform, workspace, args.mock)
                    return None if pair[0] is None else pair

                t0 = time.monotonic()
                outcome = arm.pick_and_place(
                    decision="PASS",
                    gripper=grip,
                    pick_target=target,
                    rotation_angle=yaw,
                    bin_override=args.bin,
                    redetect_cb=cb,
                )
                cycle_s = time.monotonic() - t0

                detect_latencies.append(det_ms)
                cycle_times.append(cycle_s)
                attempts_log.append(outcome.attempts)
                loads_log.append(outcome.last_load)
                if outcome.success:
                    successes += 1

                writer.writerow([
                    i + 1, _ts(), det_ms, round(cycle_s, 2),
                    outcome.success, outcome.attempts, round(outcome.last_load, 4),
                    round(target.coords[0], 1), round(target.coords[1], 1), round(yaw, 1),
                ])
                f.flush()

                marker = PASS if outcome.success else FAIL
                print(f"  {marker} cycle={cycle_s:.1f}s attempts={outcome.attempts} "
                      f"load={outcome.last_load:.3f}")
    finally:
        if cap is not None:
            cap.release()

    total_s = time.monotonic() - bench_t0
    n = len(cycle_times)

    print("\n" + "=" * 64)
    print(f"  Wrote: {csv_path}")
    print(f"  Total wall time:   {total_s:.0f} s")
    if n:
        first_attempt_ok = sum(1 for a in attempts_log if a == 1)
        print(f"  Cycles completed:  {n}")
        print(f"  Overall success:   {successes}/{n}  ({successes / n:.0%})")
        print(f"  First-attempt SR:  {first_attempt_ok}/{n}  ({first_attempt_ok / n:.0%})")
        print(f"  Cycle time mean:   {statistics.mean(cycle_times):.1f} s")
        print(f"             p50/p95: {_pct(cycle_times, 50):.1f} / {_pct(cycle_times, 95):.1f} s")
        print(f"  Attempts mean:     {statistics.mean(attempts_log):.2f}")
        print(f"  Load mean:         {statistics.mean(loads_log):.3f}")
        print(f"  Detect mean:       {statistics.mean(detect_latencies):.0f} ms")
    print("=" * 64 + "\n")


def _detect(cap, detector, transform, workspace, mock):
    if mock:
        from qa_cell_edge_agent.drivers.transforms import PickTarget
        import random
        return PickTarget(
            coords=[120.0 + random.gauss(0, 1), 30.0 + random.gauss(0, 1), 50.0, 0.0, 0.0, 0.0],
            pixel_centre=(320, 240),
            reachable=True,
            distance_from_base=124.0,
        ), random.uniform(-30, 30)
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


def _ts():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _pct(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


if __name__ == "__main__":
    main()
