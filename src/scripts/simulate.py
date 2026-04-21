#!/usr/bin/env python3
"""Simulate the edge agent loop without robot hardware.

Captures live frames from your laptop/USB camera, runs real ONNX inference
(if model downloaded), applies sensor fusion with synthetic gripper data,
and pushes everything to Foundry — streams, inspection events, telemetry.

This is the closest you can get to a live run without the physical robot.
The arm pick-and-place is logged but not executed.

Usage:
    python scripts/simulate.py                    # run indefinitely at 1Hz
    python scripts/simulate.py --interval 2.0     # run at 0.5Hz
    python scripts/simulate.py --count 20         # run 20 cycles then stop
    python scripts/simulate.py --mock-foundry     # skip Foundry, just log
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))

from foundry_sdk_runtime.types.null_types import Empty

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients
from qa_cell_edge_agent.fusion.engine import FusionEngine
from qa_cell_edge_agent.models.inference import ModelInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulate")

# Simulated servo temperatures — drift upward over time
_temp_base = 35.0
_temp_drift_per_cycle = 0.01  # ~0.6°C per minute at 1Hz


class SimulatedGripper:
    """Generates realistic gripper readings without hardware."""

    def __init__(self):
        self._cycle = 0

    def read(self) -> dict:
        self._cycle += 1
        # Simulate occasional object detection
        has_object = random.random() > 0.3
        if has_object:
            load = round(random.gauss(0.35, 0.10), 4)
        else:
            load = round(random.gauss(0.08, 0.03), 4)
        load = max(0.0, min(1.0, load))
        return {
            "servo_load": round(load * 500, 2),
            "normalized_load": load,
            "grip_state": "CLOSED" if has_object else "OPEN",
            "object_detected": has_object,
        }

    def read_joint_temperatures(self) -> list[float]:
        global _temp_base
        _temp_base += _temp_drift_per_cycle
        return [round(_temp_base + random.gauss(0, 1.5), 1) for _ in range(6)]


def open_camera(settings: Settings):
    """Open the camera. Returns (cv2.VideoCapture, cv2 module) or (None, None)."""
    try:
        import cv2
        cap = cv2.VideoCapture(settings.camera_device_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                logger.info("Camera opened: index %d (%dx%d)", settings.camera_device_index, w, h)
                return cap, cv2
            cap.release()
    except ImportError:
        pass
    logger.warning("No camera available — will generate synthetic frames")
    return None, None


def synthetic_frame() -> np.ndarray:
    """Generate a random 640x480 frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def frame_to_thumbnail_b64(cv2_mod, frame: np.ndarray) -> str:
    """Encode frame as 64x64 JPEG base64."""
    import base64
    if cv2_mod is not None:
        thumb = cv2_mod.resize(frame, (64, 64), interpolation=cv2_mod.INTER_AREA)
        _, buf = cv2_mod.imencode(".jpg", thumb, [cv2_mod.IMWRITE_JPEG_QUALITY, 70])
    else:
        import cv2
        thumb = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    """Encode as full-res JPEG."""
    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def upload_frame(clients: FoundryClients, jpeg_bytes: bytes, inspection_id: str):
    """Upload frame as MediaReference."""
    try:
        from foundry_sdk_runtime import AllowBetaFeatures
        with AllowBetaFeatures():
            return clients.client.ontology.media.upload_media(
                jpeg_bytes, f"{inspection_id}.jpg"
            )
    except Exception as exc:
        logger.warning("Frame upload failed: %s", exc)
        return Empty.value


def run_simulation(
    settings: Settings,
    interval: float = 1.0,
    max_cycles: Optional[int] = None,
    mock_foundry: bool = False,
) -> None:
    """Main simulation loop."""

    if mock_foundry:
        import os
        os.environ["MOCK_FOUNDRY"] = "true"
        settings = Settings()

    clients = FoundryClients(settings=settings) if not mock_foundry else None
    model = ModelInference(model_path=settings.model_path, mock=False)
    fusion = FusionEngine(
        confidence_threshold=settings.confidence_threshold,
        grip_tolerance=settings.grip_tolerance,
    )
    gripper = SimulatedGripper()

    cap, cv2_mod = open_camera(settings)

    logger.info("=" * 60)
    logger.info("QA Cell Simulation")
    logger.info("  Robot ID:    %s", settings.robot_id)
    logger.info("  Model:       %s (mock=%s)", settings.model_path, model.mock)
    logger.info("  Camera:      %s", "live" if cap else "synthetic")
    logger.info("  Foundry:     %s", "live" if not mock_foundry else "mock")
    logger.info("  Interval:    %.1fs", interval)
    logger.info("  Max cycles:  %s", max_cycles or "unlimited")
    logger.info("=" * 60)

    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("Shutting down...")
        shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cycle = 0
    total_inspections = 0

    while not shutdown:
        if max_cycles and cycle >= max_cycles:
            break

        cycle_start = time.monotonic()
        cycle += 1

        try:
            now = datetime.now(timezone.utc)
            ts_str = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            inspection_id = f"insp-{uuid.uuid4()}"
            vr_id = f"vr-{uuid.uuid4()}"
            gr_id = f"gr-{uuid.uuid4()}"

            # ── Capture frame ────────────────────────────────────
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    frame = synthetic_frame()
            else:
                frame = synthetic_frame()

            thumbnail_b64 = frame_to_thumbnail_b64(cv2_mod, frame)

            # ── Simulate gripper ─────────────────────────────────
            grip = gripper.read()
            temps = gripper.read_joint_temperatures()

            # ── Run inference ────────────────────────────────────
            result = model.infer(frame)

            # ── Sensor fusion ────────────────────────────────────
            fusion_result = fusion.decide(
                vision_class=result.detected_class,
                confidence=result.confidence,
                normalized_load=grip["normalized_load"],
            )

            total_inspections += 1
            review_status = "PENDING_REVIEW" if fusion_result.decision == "REVIEW" else "NOT_REQUIRED"

            # ── Log pick-and-place (simulated) ───────────────────
            logger.info(
                "[%d] %s (conf=%.2f, grip=%.2f) -> %s | temps=[%.0f,%.0f,%.0f,%.0f,%.0f,%.0f]",
                cycle,
                result.detected_class,
                result.confidence,
                grip["normalized_load"],
                fusion_result.decision,
                *temps,
            )

            if mock_foundry:
                elapsed = time.monotonic() - cycle_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # ── Push to Foundry ──────────────────────────────────

            # Vision stream
            vision_record = {
                "readingId": vr_id,
                "inspectionId": inspection_id,
                "robotId": settings.robot_id,
                "timestamp": ts_str,
                "detectedClass": result.detected_class,
                "confidence": result.confidence,
                "boundingBox": result.bounding_box,
                "inferenceTimeMs": result.inference_time_ms,
                "modelVersion": model.version,
                "thumbnailBase64": thumbnail_b64,
            }
            clients.push_to_stream(settings.vision_stream_rid, [vision_record])

            # Grip stream
            grip_record = {
                "readingId": gr_id,
                "inspectionId": inspection_id,
                "robotId": settings.robot_id,
                "timestamp": ts_str,
                "servoLoad": grip["servo_load"],
                "normalizedLoad": grip["normalized_load"],
                "gripState": grip["grip_state"],
                "objectDetected": grip["object_detected"],
            }
            clients.push_to_stream(settings.grip_stream_rid, [grip_record])

            # Telemetry stream (8 records)
            telemetry = [
                {"seriesId": f"vision-confidence:{settings.robot_id}", "timestamp": ts_str, "value": round(result.confidence, 4)},
                {"seriesId": f"grip-load:{settings.robot_id}", "timestamp": ts_str, "value": round(grip["normalized_load"], 4)},
            ]
            for i, temp in enumerate(temps):
                telemetry.append({
                    "seriesId": f"j{i+1}-temp:{settings.robot_id}",
                    "timestamp": ts_str,
                    "value": round(temp, 1),
                })
            clients.push_to_stream(settings.telemetry_stream_rid, telemetry)

            # Upload frame + create InspectionEvent
            jpeg_bytes = frame_to_jpeg_bytes(frame)
            media_ref = upload_frame(clients, jpeg_bytes, inspection_id)

            clients.client.ontology.actions.create_inspection_event(
                inspection_id=inspection_id,
                robot_id=settings.robot_id,
                timestamp=now,
                vision_class=result.detected_class,
                vision_confidence=result.confidence,
                grip_load=grip["normalized_load"],
                fusion_decision=fusion_result.decision,
                fusion_reason=fusion_result.reason,
                vision_agrees=fusion_result.vision_agrees,
                model_version=model.version,
                cycle_time_ms=int((time.monotonic() - cycle_start) * 1000),
                review_status=review_status,
                captured_image_ref=media_ref,
            )

            # Heartbeat every 5 cycles
            if cycle % 5 == 0:
                try:
                    clients.client.ontology.actions.update_robot_status(
                        robot=settings.robot_id,
                        status="RUNNING",
                        current_model_version=model.version,
                        total_inspections=total_inspections,
                    )
                except Exception as exc:
                    logger.warning("Heartbeat failed: %s", exc)

        except Exception:
            logger.exception("Simulation cycle %d failed", cycle)

        elapsed = time.monotonic() - cycle_start
        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Cleanup
    if cap is not None:
        cap.release()
    logger.info("Simulation stopped after %d cycles (%d inspections)", cycle, total_inspections)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="QA Cell — Simulate edge agent loop")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between cycles (default: 1.0)")
    parser.add_argument("--count", type=int, default=None, help="Number of cycles (default: unlimited)")
    parser.add_argument("--mock-foundry", action="store_true", help="Skip Foundry, just log locally")
    args = parser.parse_args()

    settings = Settings()
    run_simulation(
        settings,
        interval=args.interval,
        max_cycles=args.count,
        mock_foundry=args.mock_foundry,
    )


if __name__ == "__main__":
    main()
