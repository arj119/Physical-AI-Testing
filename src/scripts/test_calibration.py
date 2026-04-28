#!/usr/bin/env python3
"""Test camera calibration accuracy.

Places the arm at known positions and shows where the camera thinks
they are (and vice versa). Helps diagnose if the homography is correct.

Usage:
    python scripts/test_calibration.py
"""

import os
import sys
import time
import json

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.drivers.transforms import CameraTransform


def main():
    load_dotenv()
    settings = Settings()

    # Load calibration
    transform = CameraTransform()
    if not transform.is_calibrated:
        print("ERROR: No camera calibration found. Run calibrate_camera.py first.")
        sys.exit(1)

    print(f"Calibration mode: {transform._mode}")
    if transform._mode == "homography":
        print(f"Homography matrix:\n{transform._H}")
    print(f"Z pick height: {transform._z_pick} mm")
    print(f"Approach angles: {transform._approach_angles}")
    print()

    # Connect to robot
    try:
        from pymycobot import MyCobot280
    except ImportError:
        print("ERROR: pymycobot not installed")
        sys.exit(1)

    port = os.getenv("MYCOBOT_PORT", "/dev/ttyTHS1")
    baud = int(os.getenv("MYCOBOT_BAUD", "1000000"))
    mc = MyCobot280(port, baud)
    time.sleep(0.5)

    # Open camera
    cap = cv2.VideoCapture(settings.camera_device_index)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    print("=" * 60)
    print("  Camera Calibration Test")
    print("=" * 60)
    print()
    print("This test does two things:")
    print("  1. You click a point in the camera → shows robot coords it maps to")
    print("  2. Move arm to a position → shows where camera thinks it is")
    print()
    print("Instructions:")
    print("  - Click anywhere in the camera view to see predicted robot XY")
    print("  - Press 'r' to read current robot coords and compare")
    print("  - Press 'q' to quit")
    print()

    clicked_points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            target = transform.pixel_to_robot(x, y)
            print(f"  Pixel ({x}, {y}) → Robot ({target.coords[0]:.1f}, {target.coords[1]:.1f}) "
                  f"dist={target.distance_from_base:.1f}mm reachable={target.reachable}")

    cv2.namedWindow("Calibration Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Test", 960, 720)
    cv2.setMouseCallback("Calibration Test", on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Draw clicked points and their robot coords
        for px, py in clicked_points[-10:]:
            target = transform.pixel_to_robot(px, py)
            color = (0, 255, 0) if target.reachable else (0, 0, 255)
            cv2.circle(display, (px, py), 6, color, -1)
            label = f"({target.coords[0]:.0f}, {target.coords[1]:.0f})"
            cv2.putText(display, label, (px + 10, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(display, "Click: pixel->robot | R: read robot coords | Q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Calibration Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            coords = mc.get_coords()
            angles = mc.get_angles()
            if coords and len(coords) >= 3:
                print(f"\n  Robot position: x={coords[0]:.1f}, y={coords[1]:.1f}, z={coords[2]:.1f}")
                print(f"  Robot angles: {[f'{a:.1f}' for a in angles]}")
                print(f"  Distance from base: {(coords[0]**2 + coords[1]**2)**0.5:.1f} mm")
                print()
            else:
                print(f"  Could not read coords: {coords}")

    cap.release()
    cv2.destroyAllWindows()

    # Summary: show calibration points if available
    cal_file = os.path.join(os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "camera_calibration.json")
    if os.path.isfile(cal_file):
        with open(cal_file) as f:
            cal = json.load(f)
        if "pixel_points" in cal and "robot_points" in cal:
            print("\n  Calibration points used:")
            for i, (px, ry) in enumerate(zip(cal["pixel_points"], cal["robot_points"])):
                print(f"    {i+1}. Pixel {px} → Robot {ry}")


if __name__ == "__main__":
    main()
