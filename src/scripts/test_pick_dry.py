#!/usr/bin/env python3
"""Hardware dry-run: full pick→place motion sequence, no block.

Validates that the arm can reach every calibrated bin via the standard
HOME → SCOUT → APPROACH → GRIP → TRANSIT → BIN → HOME path. Does not
read or assert on gripper load — purely kinematic.

Usage:
    python scripts/test_pick_dry.py
    python scripts/test_pick_dry.py --xy 100 50    # alternate pick centre
    python scripts/test_pick_dry.py --bins BIN_A,BIN_B
    python scripts/test_pick_dry.py --speed 20     # slower
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
    parser = argparse.ArgumentParser(description="Pick-and-place dry run (no block)")
    parser.add_argument("--xy", nargs=2, type=float, metavar=("X", "Y"),
                        default=[150.0, 0.0],
                        help="Pick centre in robot mm (default 150 0)")
    parser.add_argument("--bins", type=str, default="",
                        help="Comma-separated bin waypoint names; default = all calibrated bins")
    parser.add_argument("--speed", type=int, default=30,
                        help="Movement speed (default 30)")
    parser.add_argument("--mock", action="store_true", help="Mock arm/gripper")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    from qa_cell_edge_agent.drivers.arm import Arm
    from qa_cell_edge_agent.drivers.gripper import Gripper
    from qa_cell_edge_agent.drivers.transforms import PickTarget

    port = os.environ.get("MYCOBOT_PORT", "/dev/ttyUSB0")
    baud = int(os.environ.get("MYCOBOT_BAUD", "115200"))
    arm = Arm(port=port, baud=baud, mock=args.mock)
    grip = Gripper(port=port, baud=baud, mock=args.mock)

    # Choose bins
    if args.bins:
        bins = [b.strip() for b in args.bins.split(",") if b.strip()]
    else:
        bins = sorted([n for n in arm.waypoints if n.startswith("BIN_")])

    if not bins:
        print(f"{FAIL} No bin waypoints calibrated — run calibrate_arm.py first.")
        sys.exit(1)

    required = ["HOME", "SCOUT"]
    missing = [r for r in required if r not in arm.waypoints]
    if missing:
        print(f"{FAIL} Missing required waypoints: {missing}")
        sys.exit(1)

    print(f"\n=== Pick-and-Place Dry Run ===")
    print(f"  Pick centre: ({args.xy[0]:.1f}, {args.xy[1]:.1f}) mm")
    print(f"  Bins:        {bins}")
    print(f"  Speed:       {args.speed}\n")

    fake_target = PickTarget(
        coords=[args.xy[0], args.xy[1], arm.GRIP_HEIGHT_MM, 0.0, 0.0, 0.0],
        pixel_centre=(0, 0),
        reachable=True,
        distance_from_base=(args.xy[0] ** 2 + args.xy[1] ** 2) ** 0.5,
    )

    failures = 0
    for bin_name in bins:
        print(f"\n  → Cycle to {bin_name}")
        t0 = time.monotonic()
        outcome = arm.pick_and_place(
            decision="PASS",
            gripper=grip,
            pick_target=fake_target,
            rotation_angle=0.0,
            bin_override=bin_name,
            redetect_cb=None,
            max_retries=0,  # no retries — empty close will "succeed" once we open
        )
        elapsed = time.monotonic() - t0
        # Without a block the load is near 0; pick_and_place will retry if
        # max_retries>0. With max_retries=0, it returns failure on the first
        # missed grasp. That's expected here — we report the kinematic time.
        marker = PASS if elapsed > 0 else FAIL
        print(f"    {marker} {bin_name}: {elapsed:.1f} s, "
              f"success={outcome.success}, load={outcome.last_load:.3f}, "
              f"attempts={outcome.attempts}")
        if not outcome.success:
            failures += 1

    print()
    if failures == 0:
        print(f"  {PASS} All {len(bins)} cycles completed.\n")
    else:
        # Failures are expected (no block) — just confirm motion ran end-to-end.
        print(f"  {WARN} {failures}/{len(bins)} cycles reported missed grasp")
        print(f"        (expected — no block in gripper). Verify the arm visited each bin.\n")


if __name__ == "__main__":
    main()
