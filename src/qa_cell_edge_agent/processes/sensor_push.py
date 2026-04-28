"""Process 1 — Sensor Push.

Single source of truth for ALL Foundry stream pushes. Runs at 1Hz and pushes:

- **vision-readings** — camera frame thumbnail + placeholder inference fields
- **grip-readings** — gripper servo load from shared sensor state
- **sensor-telemetry** — scalar time series (joint temps, vision conf, grip load)

Sensor readings come from the shared ``sensor_state`` dict, which Process 2
(defect_detection) writes to on every cycle via the serial connection.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from multiprocessing import Queue
from typing import Any, Dict, List, Optional

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients
from qa_cell_edge_agent.drivers.camera import Camera

logger = logging.getLogger(__name__)


def run_sensor_push(
    queue: Queue,
    sensor_state: dict,
    settings: Optional[Settings] = None,
) -> None:
    """Entry point for the sensor-push process (long-running)."""

    settings = settings or Settings()
    clients = FoundryClients(settings=settings)

    camera = Camera(
        device_index=settings.camera_device_index,
        thumbnail_size=settings.thumbnail_size,
        mock=settings.mock_hardware,
    )

    logger.info(
        "sensor_push started — interval=%.1fs, robot=%s, mock_hw=%s, mock_foundry=%s",
        settings.capture_interval_sec,
        settings.robot_id,
        settings.mock_hardware,
        settings.mock_foundry,
    )

    try:
        while True:
            cycle_start = time.monotonic()
            try:
                _run_one_cycle(settings, clients, camera, queue, sensor_state)
            except Exception:
                logger.exception("sensor_push cycle failed — will retry")

            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, settings.capture_interval_sec - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        camera.release()
        logger.info("Camera released")


def _run_one_cycle(
    settings: Settings,
    clients: FoundryClients,
    camera: Camera,
    queue: Queue,
    sensor_state: dict,
) -> None:
    """Execute one capture → push → enqueue cycle."""

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    inspection_id = f"insp-{uuid.uuid4()}"
    vision_reading_id = f"vr-{uuid.uuid4()}"
    grip_reading_id = f"gr-{uuid.uuid4()}"
    robot_id = settings.robot_id

    # ── 1. Capture frame ──────────────────────────────────────────
    frame = camera.capture()
    if frame is None:
        logger.warning("Skipping cycle — camera returned no frame")
        return
    thumbnail_b64 = camera.make_thumbnail_b64(frame)

    # ── 2. Read shared sensor state (written by Process 2) ────────
    grip_load = sensor_state.get("grip_load", 0.0)
    grip_servo_load = sensor_state.get("grip_servo_load", 0.0)
    grip_state_str = sensor_state.get("grip_state", "OPEN")
    object_detected = sensor_state.get("object_detected", False)
    vision_confidence = sensor_state.get("vision_confidence", 0.0)
    joint_temps = sensor_state.get("joint_temps", [0.0] * 6)

    # ── 3. Build stream payloads ──────────────────────────────────
    vision_record = {
        "readingId": vision_reading_id,
        "inspectionId": inspection_id,
        "robotId": robot_id,
        "timestamp": ts,
        "detectedClass": "pending",
        "confidence": 0.0,
        "boundingBox": [],
        "inferenceTimeMs": 0,
        "modelVersion": "",
        "thumbnailBase64": thumbnail_b64,
    }

    grip_record = {
        "readingId": grip_reading_id,
        "inspectionId": inspection_id,
        "robotId": robot_id,
        "timestamp": ts,
        "servoLoad": grip_servo_load,
        "normalizedLoad": grip_load,
        "gripState": grip_state_str,
        "objectDetected": object_detected,
    }

    telemetry_records: List[Dict[str, Any]] = []
    # Joint temperatures (6 sensors)
    for i, temp in enumerate(joint_temps):
        telemetry_records.append({
            "seriesId": f"j{i + 1}-temp:{robot_id}",
            "timestamp": ts,
            "value": round(float(temp), 1),
        })
    # Vision confidence
    telemetry_records.append({
        "seriesId": f"vision-confidence:{robot_id}",
        "timestamp": ts,
        "value": round(float(vision_confidence), 4),
    })
    # Grip load
    telemetry_records.append({
        "seriesId": f"grip-load:{robot_id}",
        "timestamp": ts,
        "value": round(float(grip_load), 4),
    })

    # ── 4. Push to all three streams ──────────────────────────────
    if not settings.mock_foundry:
        ok_v = clients.push_to_stream(settings.vision_stream_rid, [vision_record])
        if not ok_v:
            logger.error("Failed to push VisionReading %s", vision_reading_id)

        ok_g = clients.push_to_stream(settings.grip_stream_rid, [grip_record])
        if not ok_g:
            logger.error("Failed to push GripReading %s", grip_reading_id)

        ok_t = clients.push_to_stream(settings.telemetry_stream_rid, telemetry_records)
        if not ok_t:
            logger.error("Failed to push telemetry batch (%d records)", len(telemetry_records))
    else:
        logger.debug("[MOCK] Would push VisionReading + GripReading + %d telemetry records", len(telemetry_records))

    # ── 5. Enqueue for Process 2 ──────────────────────────────────
    import queue as _queue_mod
    try:
        queue.put_nowait({
            "inspection_id": inspection_id,
            "vision_reading_id": vision_reading_id,
            "timestamp": ts,
            "frame": frame,
        })
    except _queue_mod.Full:
        logger.debug("Queue full — dropping frame %s", inspection_id)
