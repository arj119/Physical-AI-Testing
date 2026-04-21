#!/usr/bin/env python3
"""Define the pick zone by clicking 4 corners in the camera view.

Saves to drivers/workspace_zone.json. The agent only monitors this
region for new objects and ignores everything outside (bins, arm, plate edges).

Usage:
    python scripts/define_zone.py
"""

import json
import os
import sys

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qa_cell_edge_agent.config.settings import Settings

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "workspace_zone.json"
)

points = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"  Point {len(points)}: ({x}, {y})")


def main():
    load_dotenv()
    settings = Settings()

    cap = cv2.VideoCapture(settings.camera_device_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {settings.camera_device_index}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read from camera")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"\n  Define Pick Zone ({w}x{h})")
    print("=" * 50)
    print("  Click 4 corners of the pick zone (clockwise).")
    print("  Press R to reset, Q to quit, ENTER to save.")
    print()

    cv2.namedWindow("Define Zone", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Define Zone", 960, 720)
    cv2.setMouseCallback("Define Zone", on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Draw existing points and lines
        if points:
            pts = np.array(points, dtype=np.int32)
            for i, (px, py) in enumerate(points):
                cv2.circle(display, (px, py), 8, (0, 255, 0), -1)
                cv2.putText(display, str(i + 1), (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(points) > 1:
                cv2.polylines(display, [pts], isClosed=(len(points) == 4),
                              color=(0, 255, 0), thickness=2)
            if len(points) == 4:
                # Fill with transparent overlay
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)

        # Instructions
        remaining = 4 - len(points)
        if remaining > 0:
            text = f"Click {remaining} more corner(s)"
        else:
            text = "Press ENTER to save, R to reset"
        cv2.putText(display, text, (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Define Zone", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("  Aborted.")
            break
        elif key == ord('r'):
            points.clear()
            print("  Reset — click 4 corners again.")
        elif key == 13 and len(points) == 4:  # Enter
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            data = {"points": points}
            with open(OUTPUT_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n  Zone saved to {OUTPUT_FILE}")
            print(f"  Corners: {points}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
