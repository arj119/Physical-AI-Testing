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
import time

from dotenv import load_dotenv

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WAYPOINT_NAMES = ["HOME", "PICK", "BIN_PASS", "BIN_FAIL", "BIN_REVIEW"]
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "waypoints.json")


def main():
    load_dotenv()

    try:
        from pymycobot import MyCobot280
    except ImportError:
        print("ERROR: pymycobot not installed. Run: pip install 'pymycobot>=3.6.0,<5.0.0'")
        sys.exit(1)

    port = os.getenv("MYCOBOT_PORT", "/dev/ttyUSB0")
    baud = int(os.getenv("MYCOBOT_BAUD", "115200"))

    # ── Connect and validate ─────────────────────────────────────
    print(f"Connecting to myCobot on {port} @ {baud}...")
    try:
        mc = MyCobot280(port, baud)
        time.sleep(0.5)
    except Exception as exc:
        print(f"ERROR: Cannot open serial port {port}: {exc}")
        if "Permission denied" in str(exc):
            print("  Fix: sudo usermod -aG dialout $USER && logout/login")
        elif "No such file" in str(exc):
            print("  Fix: Check cable. Run 'python scripts/verify_hardware.py' first.")
        sys.exit(1)

    # Verify communication
    try:
        angles = mc.get_angles()
        if not angles or len(angles) != 6:
            print(f"ERROR: myCobot not responding (get_angles returned: {angles})")
            print("  Fix: Check firmware. Power cycle the robot. Run verify_hardware.py.")
            sys.exit(1)
        print(f"  Connected. Current angles: {angles}")
    except Exception as exc:
        print(f"ERROR: Cannot communicate with myCobot: {exc}")
        sys.exit(1)

    # ── Calibrate waypoints ──────────────────────────────────────
    print("\n=== myCobot 280 Waypoint Calibration ===")
    print("For each waypoint, manually move the arm to the desired position,")
    print("then press ENTER to record the joint angles.")
    print("Press Ctrl+C at any time to abort without saving.\n")

    waypoints = {}
    try:
        for name in WAYPOINT_NAMES:
            input(f"  Move arm to {name} position, then press ENTER...")
            angles = mc.get_angles()
            coords = mc.get_coords()

            if not angles or len(angles) != 6:
                print(f"  WARNING: get_angles() returned {angles} — retrying...")
                time.sleep(0.5)
                angles = mc.get_angles()

            waypoints[name] = {
                "angles": angles if angles else [0.0] * 6,
                "coords": coords if coords else [0.0] * 6,
            }
            print(f"    Recorded {name}: angles={angles}, coords={coords}\n")

    except KeyboardInterrupt:
        print(f"\n\n  Calibration aborted. No file saved.")
        sys.exit(1)

    # ── Save and validate ────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(waypoints, f, indent=2)

    # Validate the file is readable
    try:
        with open(OUTPUT_FILE) as f:
            loaded = json.load(f)
        assert set(WAYPOINT_NAMES) == set(loaded.keys()), "Missing waypoints"
        for name in WAYPOINT_NAMES:
            assert len(loaded[name]["angles"]) == 6, f"{name} has wrong angle count"
    except Exception as exc:
        print(f"ERROR: Saved file is invalid: {exc}")
        sys.exit(1)

    print(f"Waypoints saved to {OUTPUT_FILE}")
    print(f"  {len(waypoints)} waypoints: {', '.join(WAYPOINT_NAMES)}")
    print("The arm driver will load these automatically on next startup.")


if __name__ == "__main__":
    main()
