#!/usr/bin/env python3
"""Live camera view with inference overlay.

Shows what the robot sees with bounding boxes, class labels, confidence
scores, and fusion decisions drawn on the frame. No arm movement — just
observation.

Usage:
    python scripts/live_view.py
    python scripts/live_view.py --no-inference   # just camera, no model
"""

from __future__ import annotations

import argparse
import sys
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.models.inference import ModelInference
from qa_cell_edge_agent.fusion.engine import FusionEngine

DECISION_COLORS = {
    "PASS": (0, 200, 0),
    "FAIL": (0, 0, 220),
    "REVIEW": (0, 200, 220),
}


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Live camera view with inference")
    parser.add_argument("--no-inference", action="store_true", help="Just show camera, no model")
    args = parser.parse_args()

    settings = Settings()

    # Open camera
    cap = cv2.VideoCapture(settings.camera_device_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {settings.camera_device_index}")
        sys.exit(1)

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    print(f"Camera: index {settings.camera_device_index} ({w}x{h})")

    # Load workspace + block detector
    from qa_cell_edge_agent.drivers.workspace import WorkspaceMonitor
    from qa_cell_edge_agent.drivers.block_detector import BlockDetector
    workspace = WorkspaceMonitor()
    block_detector = BlockDetector()
    print(f"Workspace: zone configured = {workspace.is_configured}")

    fusion = FusionEngine(
        confidence_threshold=settings.confidence_threshold,
        grip_tolerance=settings.grip_tolerance,
    )
    print("Detection: color-based block detector (green=PASS, red=FAIL, other=REVIEW)")

    cv2.namedWindow("Live View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live View", 960, 720)

    print("Press Q to quit\n")

    fps_start = time.monotonic()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        frame_count += 1

        # Draw workspace zone
        display = workspace.draw_zone(display)

        # Detect colored blocks
        zone_mask = workspace._zone_mask if workspace.is_configured else None
        detection = block_detector.detect(frame, zone_mask=zone_mask)

        if detection is not None:
            bbox = detection.bounding_box
            x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Decision from color
            _CLASS_TO_DECISION = {
                "widget_good": "PASS",
                "widget_defect": "FAIL",
                "widget_unknown": "REVIEW",
            }
            decision = _CLASS_TO_DECISION.get(detection.detected_class, "REVIEW")

            color = DECISION_COLORS.get(decision, (200, 200, 200))

            # Draw bounding box
            cv2.rectangle(display, (x, y), (x + bw, y + bh), color, 2)

            # Label
            label = f"{detection.dominant_color} ({detection.detected_class}) -> {decision}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(display, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Info bar
            cv2.putText(display, f"Block: {detection.dominant_color} conf={detection.confidence:.2f} area={detection.contour_area:.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, "No block detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # FPS
        elapsed = time.monotonic() - fps_start
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Live View", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
