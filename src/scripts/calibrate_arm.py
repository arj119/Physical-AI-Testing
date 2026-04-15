#!/usr/bin/env python3
"""Interactive waypoint calibration for the myCobot 280.

Usage:
    python scripts/calibrate_arm.py

Guides you through positioning the arm at each waypoint (HOME, PICK,
BIN_PASS, BIN_FAIL, BIN_REVIEW), records the joint angles, and writes
them to a JSON file that the arm driver loads at startup.
"""

import json
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


WAYPOINT_NAMES = ["HOME", "PICK", "BIN_PASS", "BIN_FAIL", "BIN_REVIEW"]
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "waypoints.json")


def main():
    try:
        from pymycobot.mycobot import MyCobot
    except ImportError:
        print("ERROR: pymycobot not installed. Run: pip install pymycobot")
        sys.exit(1)

    port = os.getenv("MYCOBOT_PORT", "/dev/ttyUSB0")
    baud = int(os.getenv("MYCOBOT_BAUD", "115200"))

    print(f"Connecting to myCobot on {port} @ {baud}...")
    mc = MyCobot(port, baud)

    print("\n=== myCobot 280 Waypoint Calibration ===")
    print("For each waypoint, manually move the arm to the desired position,")
    print("then press ENTER to record the joint angles.\n")

    waypoints = {}
    for name in WAYPOINT_NAMES:
        input(f"  Move arm to {name} position, then press ENTER...")
        angles = mc.get_angles()
        coords = mc.get_coords()
        waypoints[name] = {
            "angles": angles,
            "coords": coords,
        }
        print(f"    Recorded {name}: angles={angles}, coords={coords}\n")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(waypoints, f, indent=2)

    print(f"Waypoints saved to {OUTPUT_FILE}")
    print("The arm driver will load these automatically on next startup.")


if __name__ == "__main__":
    main()
