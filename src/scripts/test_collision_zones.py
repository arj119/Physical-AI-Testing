#!/usr/bin/env python3
"""Slow-walk every waypoint pair and capture via-points around obstacles.

The fixed gooseneck camera mount is a static obstacle the arm can swing
into when transiting between certain waypoints (e.g. between bins on
opposite sides of the gooseneck post). This script walks every ordered
pair of calibrated waypoints at a slow speed; for any unsafe move, it
prompts the operator to teach a via-point, then writes a routing
``via_table`` into ``waypoints.json``.

The Arm driver loads the table on init and prepends via-points
transparently in ``go_to``.

Usage:
    python scripts/test_collision_zones.py
    python scripts/test_collision_zones.py --speed 8
    python scripts/test_collision_zones.py --pairs HOME-BIN_A,BIN_A-BIN_C
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

WAYPOINTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "waypoints.json"
)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Calibrate via-points around static obstacles")
    parser.add_argument("--speed", type=int, default=10,
                        help="Movement speed (default 10 — very slow for safety)")
    parser.add_argument("--pairs", type=str, default="",
                        help="Comma-separated pairs (e.g. BIN_A-BIN_C); default = all combinations")
    parser.add_argument("--no-write", action="store_true",
                        help="Run the walk but don't update waypoints.json")
    args = parser.parse_args()

    try:
        from pymycobot import MyCobot280
    except ImportError:
        print(f"{FAIL} pymycobot not installed.")
        sys.exit(1)

    if not os.path.isfile(WAYPOINTS_FILE):
        print(f"{FAIL} {WAYPOINTS_FILE} not found — run calibrate_arm.py first.")
        sys.exit(1)

    with open(WAYPOINTS_FILE) as f:
        data = json.load(f)
    waypoints = {k: v for k, v in data.items() if k != "via_table"}
    via_entries = data.get("via_table", [])
    via_table = {(e["from"], e["to"]): list(e.get("via", [])) for e in via_entries}

    if args.pairs:
        pairs = []
        for p in args.pairs.split(","):
            p = p.strip()
            if "-" not in p:
                continue
            a, b = p.split("-", 1)
            pairs.append((a.strip(), b.strip()))
    else:
        names = list(waypoints.keys())
        pairs = [(a, b) for a in names for b in names if a != b]

    port = os.environ.get("MYCOBOT_PORT", "/dev/ttyUSB0")
    baud = int(os.environ.get("MYCOBOT_BAUD", "115200"))
    print(f"Connecting to myCobot on {port} @ {baud}...")
    mc = MyCobot280(port, baud)
    time.sleep(0.5)

    print(f"\n=== Collision-Zone Walk ===")
    print(f"  Pairs to check: {len(pairs)}  (speed={args.speed})\n")
    print("  For each pair, the arm will move FROM → TO. Watch for any")
    print("  contact or near-miss with the gooseneck or other obstacles.\n")
    print("  After each move:")
    print("    y = clean, no via needed")
    print("    n = unsafe — script will prompt for a via-waypoint")
    print("    s = skip (don't update for this pair)")
    print("    q = quit and save what we have so far\n")

    try:
        for src, dst in pairs:
            print(f"\n── {src} → {dst} ──")
            existing = via_table.get((src, dst), [])
            if existing:
                print(f"  Existing via: {existing}  (re-test)")

            # Move to source first (no routing applied here — direct)
            print(f"  Going to {src}...")
            _send(mc, waypoints[src]["angles"], args.speed)
            time.sleep(0.5)

            # Walk through any existing via-points + dst
            chain = list(existing) + [dst]
            for name in chain:
                if name not in waypoints:
                    print(f"  {WARN} {name} not calibrated — skipping")
                    continue
                print(f"  Going to {name}...")
                _send(mc, waypoints[name]["angles"], args.speed)
                time.sleep(0.3)

            verdict = input("  Verdict [y/n/s/q]: ").strip().lower()
            if verdict == "q":
                break
            if verdict == "s":
                continue
            if verdict == "y":
                # Clean — drop any stale via for this pair
                if (src, dst) in via_table:
                    print(f"  {PASS} Marking clean (was {via_table[(src, dst)]})")
                    via_table.pop((src, dst))
                else:
                    print(f"  {PASS} Marked clean.")
                continue

            # n — capture a via point
            via_chain = []
            while True:
                input("  Manually move arm to a SAFE intermediate pose, then press ENTER... ")
                angles = mc.get_angles()
                if not angles or len(angles) != 6:
                    print(f"  {FAIL} Could not read angles — try again.")
                    continue
                via_name = input(
                    "  Name this via-waypoint (or empty to use existing): "
                ).strip()
                if via_name:
                    if via_name not in waypoints:
                        waypoints[via_name] = {"angles": angles}
                        print(f"  {PASS} Recorded waypoint {via_name}: {angles}")
                    else:
                        print(f"  {WARN} {via_name} already exists — using its calibrated pose.")
                else:
                    # Auto-name VIA_<src>_<dst>_<n>
                    via_name = f"VIA_{src}_{dst}_{len(via_chain) + 1}"
                    waypoints[via_name] = {"angles": angles}
                    print(f"  {PASS} Recorded waypoint {via_name}: {angles}")
                via_chain.append(via_name)

                more = input("  Add another via-point? [y/N] ").strip().lower()
                if more != "y":
                    break

            via_table[(src, dst)] = via_chain
            via_table[(dst, src)] = list(reversed(via_chain))
            print(f"  {PASS} {src}→{dst} via {via_chain}  (and reverse)")
    finally:
        if args.no_write:
            print("\n--no-write set — not updating waypoints.json")
            return
        # Persist
        new_data = dict(waypoints)
        new_data["via_table"] = [
            {"from": a, "to": b, "via": v} for (a, b), v in sorted(via_table.items())
        ]
        with open(WAYPOINTS_FILE, "w") as f:
            json.dump(new_data, f, indent=2)
        print(f"\n{PASS} Wrote {WAYPOINTS_FILE}  ({len(via_table)} via-routes)")


def _send(mc, angles, speed):
    mc.send_angles(angles, speed)
    for _ in range(150):  # up to 15s
        time.sleep(0.1)
        try:
            if mc.is_moving() == 0:
                break
        except Exception:
            pass


if __name__ == "__main__":
    main()
