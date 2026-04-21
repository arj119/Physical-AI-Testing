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

    # Load model + workspace monitor
    from qa_cell_edge_agent.drivers.workspace import WorkspaceMonitor
    workspace = WorkspaceMonitor()
    workspace.capture_reference(frame)
    print(f"Workspace: white zone ROI = {workspace.roi_bbox}")

    model = None
    fusion = None
    if not args.no_inference:
        model = ModelInference(model_path=settings.model_path, mock=False)
        fusion = FusionEngine(
            confidence_threshold=settings.confidence_threshold,
            grip_tolerance=settings.grip_tolerance,
        )
        print(f"Model: {settings.model_path} (backend={model._backend}, mock={model.mock})")
    else:
        print("Model: disabled (--no-inference)")

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

        if model and not model.mock:
            masked = workspace.mask_frame(frame) if workspace.is_configured else frame
            t0 = time.monotonic()
            result = model.infer(masked)
            inf_ms = int((time.monotonic() - t0) * 1000)

            bbox = result.bounding_box
            has_detection = (
                result.confidence >= 0.25
                and bbox != [0.0, 0.0, 0.0, 0.0]
            )

            if has_detection:
                x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Fusion decision (using fixed grip load since no hardware)
                fusion_result = fusion.decide(
                    vision_class=result.detected_class,
                    confidence=result.confidence,
                    normalized_load=0.3,
                )

                color = DECISION_COLORS.get(fusion_result.decision, (200, 200, 200))

                # Draw bounding box
                cv2.rectangle(display, (x, y), (x + bw, y + bh), color, 2)

                # Label
                label = f"{result.detected_class} {result.confidence:.2f} -> {fusion_result.decision}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
                cv2.putText(display, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Info bar
            status = f"{result.detected_class} conf={result.confidence:.2f}" if has_detection else "No detection"
            cv2.putText(display, f"Inference: {inf_ms}ms | {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
