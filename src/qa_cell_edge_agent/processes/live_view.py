"""Optional Process 4 — Live View.

Opens a camera window showing the pick zone, block detection, rotation
angle, and sorting decisions in real-time. Reads detection state from
the shared sensor_state dict. Runs independently — does not interfere
with the main agent processes.

Enabled via ``--live-view`` flag on the main agent.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

DECISION_COLORS = {
    "PASS": (0, 200, 0),
    "FAIL": (0, 0, 220),
    "REVIEW": (0, 200, 220),
}

_CLASS_TO_DECISION = {
    "widget_good": "PASS",
    "widget_defect": "FAIL",
    "widget_unknown": "REVIEW",
}


def run_live_view(
    sensor_state: dict,
    settings=None,
) -> None:
    """Entry point for the live view process."""

    import os
    # Ensure display is available for GUI in subprocess
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    import cv2
    from qa_cell_edge_agent.config.settings import Settings
    from qa_cell_edge_agent.drivers.block_detector import BlockDetector
    from qa_cell_edge_agent.drivers.workspace import WorkspaceMonitor

    settings = settings or Settings()
    workspace = WorkspaceMonitor()
    block_detector = BlockDetector()

    cap = cv2.VideoCapture(settings.camera_device_index)
    if not cap.isOpened():
        logger.error("Live view: cannot open camera at index %d", settings.camera_device_index)
        return

    cv2.namedWindow("QA Cell Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QA Cell Live", 960, 720)

    logger.info("Live view started on camera index %d", settings.camera_device_index)

    fps_start = time.monotonic()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        display = frame.copy()
        frame_count += 1

        # Draw workspace zone
        display = workspace.draw_zone(display)

        # Detect blocks
        zone_mask = workspace._zone_mask if workspace.is_configured else None
        detection = block_detector.detect(frame, zone_mask=zone_mask)

        if detection is not None:
            # Draw rotated bounding box
            box_points = cv2.boxPoints(detection.rotated_bbox)
            box_points = box_points.astype(int)

            decision = _CLASS_TO_DECISION.get(detection.detected_class, "REVIEW")
            color = DECISION_COLORS.get(decision, (200, 200, 200))

            cv2.drawContours(display, [box_points], 0, color, 2)

            # Center dot
            rot_cx = int(detection.rotated_bbox[0][0])
            rot_cy = int(detection.rotated_bbox[0][1])
            cv2.circle(display, (rot_cx, rot_cy), 5, color, -1)

            # Label
            label = f"{detection.dominant_color} -> {decision} ({detection.rotation_angle:.0f} deg)"
            bbox = detection.bounding_box
            x, y = int(bbox[0]), int(bbox[1])
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(display, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Info bar
            cv2.putText(display, f"Block: {detection.dominant_color} angle={detection.rotation_angle:.0f} area={detection.contour_area:.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, "No block detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # Robot state from shared dict
        grip_state = sensor_state.get("grip_state", "?")
        grip_load = sensor_state.get("grip_load", 0.0)
        cv2.putText(display, f"Gripper: {grip_state} ({grip_load:.2f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # FPS
        elapsed = time.monotonic() - fps_start
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("QA Cell Live", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Live view stopped")
