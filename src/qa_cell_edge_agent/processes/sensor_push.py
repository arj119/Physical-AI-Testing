"""Process 1 — Sensor Push.

Continuously captures frames from the camera and reads gripper servo load,
pushes raw VisionReading / GripReading JSON to Foundry Streams, and enqueues
the data locally for Process 2 (defect detection).
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from multiprocessing import Queue
from typing import Optional

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients
from qa_cell_edge_agent.drivers.camera import Camera
from qa_cell_edge_agent.drivers.gripper import Gripper

logger = logging.getLogger(__name__)


def run_sensor_push(
    queue: Queue,
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
    gripper = Gripper(
        port=settings.mycobot_port,
        baud=settings.mycobot_baud,
        mock=settings.mock_hardware,
    )

    logger.info(
        "sensor_push started — interval=%.1fs, robot=%s, mock_hw=%s, mock_foundry=%s",
        settings.capture_interval_sec,
        settings.robot_id,
        settings.mock_hardware,
        settings.mock_foundry,
    )

    while True:
        cycle_start = time.monotonic()
        try:
            _run_one_cycle(settings, clients, camera, gripper, queue)
        except Exception:
            logger.exception("sensor_push cycle failed — will retry")

        elapsed = time.monotonic() - cycle_start
        sleep_time = max(0, settings.capture_interval_sec - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


def _run_one_cycle(
    settings: Settings,
    clients: FoundryClients,
    camera: Camera,
    gripper: Gripper,
    queue: Queue,
) -> None:
    """Execute one capture → push → enqueue cycle."""

    now = datetime.now(timezone.utc)
    inspection_id = f"insp-{uuid.uuid4()}"
    vision_reading_id = f"vr-{uuid.uuid4()}"
    grip_reading_id = f"gr-{uuid.uuid4()}"

    # ── 1. Capture frame ──────────────────────────────────────────
    frame = camera.capture()
    if frame is None:
        logger.warning("Skipping cycle — camera returned no frame")
        return
    thumbnail_b64 = camera.make_thumbnail_b64(frame)

    # ── 2. Read gripper ───────────────────────────────────────────
    grip_data = gripper.read()

    # ── 3. Build stream payloads ──────────────────────────────────
    ts = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    vision_record = {
        "readingId": vision_reading_id,
        "inspectionId": inspection_id,
        "robotId": settings.robot_id,
        "timestamp": ts,
        # detectedClass and confidence are NOT known yet — they come from
        # Process 2 after inference.  We push placeholder values here so
        # the stream record is complete; Process 2 will create the
        # authoritative InspectionEvent.
        "detectedClass": "pending",
        "confidence": 0.0,
        "boundingBox": "",
        "inferenceTimeMs": 0,
        "modelVersion": "",
        "thumbnailBase64": thumbnail_b64,
    }

    grip_record = {
        "readingId": grip_reading_id,
        "inspectionId": inspection_id,
        "robotId": settings.robot_id,
        "timestamp": ts,
        "servoLoad": grip_data.servo_load,
        "normalizedLoad": grip_data.normalized_load,
        "gripState": grip_data.grip_state,
        "objectDetected": grip_data.object_detected,
    }

    # ── 4. Push to streams ────────────────────────────────────────
    if not settings.mock_foundry:
        ok_v = clients.push_to_stream(settings.vision_stream_rid, [vision_record])
        ok_g = clients.push_to_stream(settings.grip_stream_rid, [grip_record])
        if not ok_v:
            logger.error("Failed to push VisionReading %s", vision_reading_id)
        if not ok_g:
            logger.error("Failed to push GripReading %s", grip_reading_id)
    else:
        logger.debug("[MOCK] Would push VisionReading %s", vision_reading_id)
        logger.debug("[MOCK] Would push GripReading %s", grip_reading_id)

    # ── 5. Enqueue for Process 2 ──────────────────────────────────
    try:
        queue.put_nowait({
            "inspection_id": inspection_id,
            "vision_reading_id": vision_reading_id,
            "grip_reading_id": grip_reading_id,
            "timestamp": ts,
            "frame": frame,
            "grip_data": grip_data,
        })
    except Exception:
        logger.warning("Local queue full — dropping frame for %s", inspection_id)
