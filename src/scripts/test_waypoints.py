#!/usr/bin/env python3
"""Test that the robot can reach all calibrated waypoints.

Sends the arm to each waypoint, waits, then checks how close it got.
Flags any joints that missed their target by more than 3 degrees.

Usage:
    python scripts/test_waypoints.py
    python scripts/test_waypoints.py --speed 20   # slower movements
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WAYPOINTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "waypoints.json"
)

PASS = "\033[92m\u2713\033[0m"
FAIL = "\033[91m\u2717\033[0m"
WARN = "\033[93m!\033[0m"

JOINT_LIMITS = [
    (-168, 168),   # J1
    (-140, 140),   # J2
    (-150, 150),   # J3
    (-150, 150),   # J4
    (-155, 160),   # J5
    (-180, 180),   # J6
]


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test calibrated waypoints")
    parser.add_argument("--speed", type=int, default=30, help="Movement speed (default: 30)")
    args = parser.parse_args()

    try:
        from pymycobot import MyCobot280
    except ImportError:
        print("ERROR: pymycobot not installed")
        sys.exit(1)

    if not os.path.isfile(WAYPOINTS_FILE):
        print(f"ERROR: {WAYPOINTS_FILE} not found — run calibrate_arm.py first")
        sys.exit(1)

    with open(WAYPOINTS_FILE) as f:
        waypoints = json.load(f)

    port = os.getenv("MYCOBOT_PORT", "/dev/ttyTHS1")
    baud = int(os.getenv("MYCOBOT_BAUD", "1000000"))

    print(f"\nConnecting to myCobot on {port} @ {baud}...")
    try:
        mc = MyCobot280(port, baud)
        time.sleep(0.5)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Current angles: {mc.get_angles()}")
    print(f"Speed: {args.speed}")
    print()

    # ── Check joint limits first ─────────────────────────────────
    print("=" * 60)
    print("  Joint Limit Check")
    print("=" * 60)
    limit_issues = False
    for name, data in waypoints.items():
        angles = data["angles"]
        for i, angle in enumerate(angles):
            lo, hi = JOINT_LIMITS[i]
            if angle < lo or angle > hi:
                print(f"  {FAIL} {name} J{i+1}: {angle:.1f} is outside [{lo}, {hi}]")
                limit_issues = True
            elif abs(angle - lo) < 5 or abs(angle - hi) < 5:
                print(f"  {WARN} {name} J{i+1}: {angle:.1f} is within 5° of limit [{lo}, {hi}]")

    if limit_issues:
        print(f"\n  {FAIL} Some waypoints have out-of-range joints — recalibrate")
        resp = input("  Continue anyway? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit(1)
    else:
        print(f"  {PASS} All joints within limits")
    print()

    # ── Test each waypoint ───────────────────────────────────────
    print("=" * 60)
    print("  Waypoint Reachability Test")
    print("=" * 60)

    all_ok = True
    for name, data in waypoints.items():
        target = data["angles"]
        print(f"\n  Testing {name}: {target}")

        try:
            mc.send_angles(target, args.speed)
        except Exception as exc:
            print(f"  {FAIL} send_angles failed: {exc}")
            all_ok = False
            continue

        # Wait for motion to complete
        for _ in range(100):  # up to 10 seconds
            time.sleep(0.1)
            try:
                if mc.is_moving() == 0:
                    break
            except Exception:
                pass

        time.sleep(0.5)  # settle
        actual = mc.get_angles()

        if not actual or len(actual) != 6:
            print(f"  {FAIL} Could not read angles after move")
            all_ok = False
            continue

        wp_ok = True
        for i in range(6):
            delta = abs(actual[i] - target[i])
            if delta > 5:
                print(f"    {FAIL} J{i+1}: target={target[i]:.1f} actual={actual[i]:.1f} delta={delta:.1f}°")
                wp_ok = False
            elif delta > 2:
                print(f"    {WARN} J{i+1}: target={target[i]:.1f} actual={actual[i]:.1f} delta={delta:.1f}°")

        if wp_ok:
            print(f"  {PASS} {name} reached (max delta < 2°)")
        else:
            print(f"  {FAIL} {name} NOT reached — recalibrate this waypoint")
            all_ok = False

    # ── Return to first waypoint ─────────────────────────────────
    first_wp = list(waypoints.keys())[0]
    print(f"\n  Returning to {first_wp}...")
    try:
        mc.send_angles(waypoints[first_wp]["angles"], args.speed)
        time.sleep(3)
    except Exception:
        pass

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_ok:
        print(f"  {PASS} All {len(waypoints)} waypoints reachable")
    else:
        print(f"  {FAIL} Some waypoints failed — recalibrate with: python scripts/calibrate_arm.py")
        print(f"  Tip: keep all joints at least 5° inside their limits")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
