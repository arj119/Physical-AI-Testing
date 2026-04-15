"""Process 2 — Defect Detection.

Dequeues camera frames from Process 1, reads gripper servo load (this process
owns the serial connection to avoid conflicts with the arm driver), runs
YOLOv5 inference, applies sensor fusion, commands the robot arm to sort the
part, and writes results to Foundry via OSDK actions.  Also handles operator
command polling and heartbeat updates.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from multiprocessing import Event, Queue
from typing import Optional

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients
from qa_cell_edge_agent.drivers.arm import Arm
from qa_cell_edge_agent.drivers.gripper import Gripper
from qa_cell_edge_agent.drivers.transforms import CameraTransform
from qa_cell_edge_agent.fusion.engine import FusionEngine
from qa_cell_edge_agent.models.inference import ModelInference

logger = logging.getLogger(__name__)


class RobotState:
    """Mutable state machine for the robot's operational mode."""

    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    E_STOPPED = "E_STOPPED"

    def __init__(self) -> None:
        self.status: str = self.RUNNING
        self.total_inspections: int = 0
        self._e_stop_handled: bool = False


def run_defect_detection(
    queue: Queue,
    reload_event: Event,
    settings: Optional[Settings] = None,
) -> None:
    """Entry point for the defect-detection process (long-running)."""

    settings = settings or Settings()
    clients = FoundryClients(settings=settings)
    fusion = FusionEngine(
        confidence_threshold=settings.confidence_threshold,
        grip_tolerance=settings.grip_tolerance,
    )
    model = ModelInference(model_path=settings.model_path, mock=settings.mock_hardware)
    model.version = "v1.0.0"
    # Arm and Gripper share a single serial connection via drivers.connection
    arm = Arm(
        port=settings.mycobot_port,
        baud=settings.mycobot_baud,
        mock=settings.mock_hardware,
    )
    gripper = Gripper(
        port=settings.mycobot_port,
        baud=settings.mycobot_baud,
        mock=settings.mock_hardware,
    )
    cam_transform = CameraTransform()
    state = RobotState()

    logger.info("defect_detection started — robot=%s", settings.robot_id)

    last_heartbeat = 0.0

    while True:
        try:
            # ── Check for model hot-swap signal from Process 3 ────
            if reload_event.is_set():
                logger.info("Model reload signal received")
                model.reload(settings.model_path)
                reload_event.clear()

            # ── Poll operator commands ────────────────────────────
            _poll_commands(settings, clients, state, fusion, arm)

            # ── Send heartbeat ────────────────────────────────────
            now = time.monotonic()
            if now - last_heartbeat >= settings.heartbeat_interval_sec:
                _send_heartbeat(settings, clients, state, model.version)
                last_heartbeat = now

            # ── Respect PAUSED / E_STOPPED ────────────────────────
            if state.status == RobotState.PAUSED:
                time.sleep(settings.command_poll_interval_sec)
                continue
            if state.status == RobotState.E_STOPPED:
                if not state._e_stop_handled:
                    arm.safe_position()
                    state._e_stop_handled = True
                time.sleep(settings.command_poll_interval_sec)
                continue

            # ── Dequeue sensor data ───────────────────────────────
            try:
                item = queue.get(timeout=2.0)
            except Exception:
                continue  # nothing in queue — loop back to poll commands

            cycle_start = time.monotonic()

            # ── Read gripper (same serial connection as arm) ──────
            grip_data = gripper.read()
            grip_reading_id = f"gr-{uuid.uuid4()}"

            # Push grip reading to Foundry stream
            if not settings.mock_foundry:
                grip_record = {
                    "readingId": grip_reading_id,
                    "inspectionId": item["inspection_id"],
                    "robotId": settings.robot_id,
                    "timestamp": item["timestamp"],
                    "servoLoad": grip_data.servo_load,
                    "normalizedLoad": grip_data.normalized_load,
                    "gripState": grip_data.grip_state,
                    "objectDetected": grip_data.object_detected,
                }
                ok_g = clients.push_to_stream(settings.grip_stream_rid, [grip_record])
                if not ok_g:
                    logger.error("Failed to push GripReading %s", grip_reading_id)

            # ── Run inference ─────────────────────────────────────
            result = model.infer(item["frame"])

            # ── Compute dynamic pick target from bounding box ─────
            pick_target = None
            if cam_transform.is_calibrated and result.bounding_box:
                try:
                    bbox = result.bounding_box  # list[float]
                    cx = int(bbox[0] + bbox[2] / 2)
                    cy = int(bbox[1] + bbox[3] / 2)
                    pick_target = cam_transform.pixel_to_robot(cx, cy)
                except (IndexError, TypeError) as exc:
                    logger.warning("Could not compute pick target: %s", exc)

            # ── Sensor fusion ─────────────────────────────────────
            fusion_result = fusion.decide(
                vision_class=result.detected_class,
                confidence=result.confidence,
                normalized_load=grip_data.normalized_load,
            )

            # ── Sort part ─────────────────────────────────────────
            arm.pick_and_place(fusion_result.decision, gripper, pick_target)

            cycle_time_ms = int((time.monotonic() - cycle_start) * 1000)

            # ── Create InspectionEvent via OSDK ───────────────────
            review_status = (
                "PENDING_REVIEW" if fusion_result.decision == "REVIEW"
                else "NOT_REQUIRED"
            )
            if not settings.mock_foundry:
                ts = datetime.fromisoformat(
                    item["timestamp"].replace("Z", "+00:00")
                )
                # Upload the camera frame and get a MediaReference
                captured_ref = _upload_frame(clients, item)

                clients.client.ontology.actions.create_inspection_event(
                    inspection_id=item["inspection_id"],
                    robot_id=settings.robot_id,
                    timestamp=ts,
                    vision_class=result.detected_class,
                    vision_confidence=result.confidence,
                    grip_load=grip_data.normalized_load,
                    fusion_decision=fusion_result.decision,
                    fusion_reason=fusion_result.reason,
                    vision_agrees=fusion_result.vision_agrees,
                    model_version=model.version,
                    cycle_time_ms=cycle_time_ms,
                    review_status=review_status,
                    captured_image_ref=captured_ref,
                )
            else:
                logger.debug(
                    "[MOCK] InspectionEvent %s → %s (%s)",
                    item["inspection_id"],
                    fusion_result.decision,
                    fusion_result.reason,
                )

            state.total_inspections += 1
            logger.info(
                "Inspection %s: %s (conf=%.2f, grip=%.2f) → %s [%dms]",
                item["inspection_id"],
                result.detected_class,
                result.confidence,
                grip_data.normalized_load,
                fusion_result.decision,
                cycle_time_ms,
            )

        except Exception:
            logger.exception("defect_detection cycle failed — will retry")
            time.sleep(1)


# ── Helper functions ──────────────────────────────────────────────────


def _poll_commands(
    settings: Settings,
    clients: FoundryClients,
    state: RobotState,
    fusion: FusionEngine,
    arm: Arm,
) -> None:
    """Query for PENDING OperatorCommands and process them."""
    if settings.mock_foundry:
        return

    from physical_ai_qa_cell_sdk.ontology.search import OperatorCommandObjectType

    try:
        commands = (
            clients.client.ontology.objects.OperatorCommand
            .where(OperatorCommandObjectType.robot_id.eq(settings.robot_id))
            .where(OperatorCommandObjectType.status.eq("PENDING"))
            .order_by(OperatorCommandObjectType.created_at.desc())
            .take(100)
        )
    except Exception as exc:
        logger.error("Failed to poll commands: %s", exc)
        return

    for cmd in commands:
        cmd_type = cmd.command_type or ""
        cmd_id = cmd.command_id or ""
        payload_str = cmd.payload or "{}"

        logger.info("Processing command %s: %s", cmd_id, cmd_type)

        if cmd_type == "PAUSE":
            state.status = RobotState.PAUSED
        elif cmd_type == "RESUME":
            state.status = RobotState.RUNNING
            state._e_stop_handled = False
        elif cmd_type == "E_STOP":
            state.status = RobotState.E_STOPPED
            state._e_stop_handled = False
            arm.safe_position()
            state._e_stop_handled = True
        elif cmd_type == "UPDATE_TOLERANCE":
            try:
                payload = json.loads(payload_str) if payload_str else {}
                fusion.update_thresholds(
                    grip_tolerance=payload.get("tolerance"),
                    confidence_threshold=payload.get("confidence_threshold"),
                )
            except (json.JSONDecodeError, TypeError) as exc:
                logger.error("Bad UPDATE_TOLERANCE payload: %s", exc)

        # Acknowledge the command
        try:
            clients.client.ontology.actions.acknowledge_command(
                command=cmd_id,
                new_status="EXECUTED",
            )
        except Exception as exc:
            logger.error("Failed to acknowledge command %s: %s", cmd_id, exc)


def _send_heartbeat(
    settings: Settings,
    clients: FoundryClients,
    state: RobotState,
    model_version: str,
) -> None:
    """Push a heartbeat update to the Robot object."""
    if settings.mock_foundry:
        logger.debug("[MOCK] Heartbeat: status=%s, total=%d", state.status, state.total_inspections)
        return

    try:
        clients.client.ontology.actions.update_robot_status(
            robot=settings.robot_id,
            status=state.status,
            current_model_version=model_version,
            total_inspections=state.total_inspections,
        )
    except Exception as exc:
        logger.error("Heartbeat failed: %s", exc)


def _upload_frame(clients: FoundryClients, item: dict):
    """Encode the camera frame as JPEG and upload as a MediaReference.

    Returns a ``MediaReference`` on success, or ``None`` if the upload fails.
    """
    try:
        import cv2
        from foundry_sdk_runtime import AllowBetaFeatures

        frame = item.get("frame")
        if frame is None:
            return None

        _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        filename = f"{item['inspection_id']}.jpg"

        with AllowBetaFeatures():
            media_ref = clients.client.ontology.media.upload_media(
                jpeg_buf.tobytes(), filename
            )
        return media_ref
    except Exception as exc:
        logger.warning("Frame upload failed (will create event without image): %s", exc)
        return None
