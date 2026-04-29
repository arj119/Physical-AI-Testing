"""Process 2 — Defect Detection.

Dequeues camera frames from Process 1, runs YOLOv5 inference, applies sensor
fusion, commands the robot arm to sort the part, and writes results to Foundry
via OSDK actions.

Owns the serial connection (arm + gripper). Writes latest sensor readings to
a shared ``sensor_state`` dict so Process 1 can push them to Foundry streams.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from multiprocessing import Event, Queue
from typing import Optional

from foundry_sdk_runtime.types.null_types import Empty

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients
from qa_cell_edge_agent.drivers.arm import Arm
from qa_cell_edge_agent.drivers.block_detector import BlockDetector
from qa_cell_edge_agent.drivers.gripper import Gripper
from qa_cell_edge_agent.drivers.transforms import CameraTransform
from qa_cell_edge_agent.drivers.workspace import WorkspaceMonitor
from qa_cell_edge_agent.drivers.arm import DECISION_TO_BIN
from qa_cell_edge_agent.fusion.engine import FusionEngine, FusionResult
from qa_cell_edge_agent.models.inference import ModelInference

logger = logging.getLogger(__name__)

# Color class → sorting decision (used in "color" detection mode)
CLASS_TO_DECISION = {
    "widget_good": "PASS",
    "widget_defect": "FAIL",
    "widget_unknown": "REVIEW",
}


class RobotState:
    """Mutable state machine for the robot's operational mode.

    Status transitions are driven by operator commands from Foundry
    (PAUSE/RESUME/E_STOP). The heartbeat carries the current status +
    total_inspections — Foundry derives IDLE/OFFLINE from heartbeat
    timestamps and inspection rate.
    """

    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    E_STOPPED = "E_STOPPED"

    def __init__(self) -> None:
        self.status: str = self.RUNNING
        self.total_inspections: int = 0
        self._e_stop_handled: bool = False


def run_defect_detection(
    sensor_queue: Queue,
    reload_event: Event,
    sensor_state: dict,
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
    workspace = WorkspaceMonitor()
    block_detector = BlockDetector()
    state = RobotState()
    _stable_since: Optional[float] = None
    _stable_bbox: Optional[list] = None
    _arm_busy: bool = False
    SETTLE_SECS = 2.0

    def _on_exit():
        try:
            arm.safe_position()
        except Exception:
            pass

    atexit.register(_on_exit)

    # Move to HOME on startup
    logger.info("Moving to HOME position...")
    arm.go_to("HOME")
    gripper.open_gripper()

    logger.info("defect_detection started — robot=%s", settings.robot_id)

    # ── Heartbeat in background thread (not blocked by arm motion) ─
    def _heartbeat_loop():
        while True:
            _send_heartbeat(settings, clients, state, model.version)
            time.sleep(settings.heartbeat_interval_sec)

    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    logger.info("Heartbeat thread started (every %.0fs)", settings.heartbeat_interval_sec)

    while True:
        try:
            # ── Check for model hot-swap signal from Process 3 ────
            if reload_event.is_set():
                logger.info("Model reload signal received")
                model.reload(settings.model_path)
                reload_event.clear()

            # ── Poll operator commands ────────────────────────────
            _poll_commands(settings, clients, state, fusion, arm)

            # ── Update shared sensor state (every cycle) ──────────
            _update_sensor_state(gripper, sensor_state)

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
                item = sensor_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            cycle_start = time.monotonic()

            frame = item["frame"]

            # ══════════════════════════════════════════════════════
            # DETECTION — either color-based or model-based
            # ══════════════════════════════════════════════════════

            if settings.detection_mode == "color":
                # ── Color detection: HSV block detector ────────────
                zone_mask = workspace._zone_mask if workspace.is_configured else None
                detection = block_detector.detect(frame, zone_mask=zone_mask)

                if detection is None:
                    _stable_since = None
                    _stable_bbox = None
                    continue

                # Verify center is inside workspace zone
                if workspace.is_configured and workspace.roi_bbox:
                    cx = detection.bounding_box[0] + detection.bounding_box[2] / 2
                    cy = detection.bounding_box[1] + detection.bounding_box[3] / 2
                    x_min, y_min, x_max, y_max = workspace.roi_bbox
                    if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                        _stable_since = None
                        _stable_bbox = None
                        continue

                # Wait for block to settle
                if _stable_bbox is None or _bbox_moved(_stable_bbox, detection.bounding_box):
                    _stable_bbox = detection.bounding_box
                    _stable_since = time.monotonic()
                    continue
                if time.monotonic() - _stable_since < SETTLE_SECS:
                    continue

                logger.info(
                    "Block: %s (%s, area=%.0f)",
                    detection.detected_class, detection.dominant_color, detection.contour_area,
                )
                _stable_since = None
                _stable_bbox = None

                detected_class = detection.detected_class
                confidence = detection.confidence
                bounding_box = detection.bounding_box
                rotation_angle = detection.rotation_angle

                # Decision directly from color
                decision = CLASS_TO_DECISION.get(detected_class, "REVIEW")
                reason = f"color_{detection.dominant_color}"

            else:
                # ── Model detection: YOLOv5 inference + fusion ─────
                inference_frame = workspace.mask_frame(frame) if workspace.is_configured else frame
                result = model.infer(inference_frame)

                if result.confidence < 0.25 or result.bounding_box == [0.0, 0.0, 0.0, 0.0]:
                    _stable_since = None
                    _stable_bbox = None
                    continue

                # Wait for detection to settle
                if _stable_bbox is None or _bbox_moved(_stable_bbox, result.bounding_box):
                    _stable_bbox = result.bounding_box
                    _stable_since = time.monotonic()
                    continue
                if time.monotonic() - _stable_since < SETTLE_SECS:
                    continue

                _stable_since = None
                _stable_bbox = None

                detected_class = result.detected_class
                confidence = result.confidence
                bounding_box = result.bounding_box
                rotation_angle = 0.0

                # Decision from sensor fusion
                grip_data_for_fusion = gripper.read()
                fusion_result = fusion.decide(
                    vision_class=detected_class,
                    confidence=confidence,
                    normalized_load=grip_data_for_fusion.normalized_load,
                )
                decision = fusion_result.decision
                reason = fusion_result.reason

            # ══════════════════════════════════════════════════════
            # COMMON: gripper read, pick target, arm, Foundry push
            # ══════════════════════════════════════════════════════

            grip_data = gripper.read()

            # Compute pick target from bounding box
            pick_target = None
            if cam_transform.is_calibrated and bounding_box:
                try:
                    cx = int(bounding_box[0] + bounding_box[2] / 2)
                    cy = int(bounding_box[1] + bounding_box[3] / 2)
                    pick_target = cam_transform.pixel_to_robot(cx, cy)
                except (IndexError, TypeError) as exc:
                    logger.warning("Could not compute pick target: %s", exc)

            fusion_result = FusionResult(decision=decision, reason=reason, vision_agrees=True)

            # ── Sort part ─────────────────────────────────────────
            no_pick = sensor_state.get("no_pick", False)
            if no_pick:
                bin_name = DECISION_TO_BIN.get(decision, "BIN_REVIEW")
                logger.info("No-pick: %s → %s", detected_class, bin_name)
                arm._elevate_to_safe_altitude()
                arm.go_to(bin_name)
                time.sleep(1.0)
                arm._elevate_to_safe_altitude()
                arm.go_to("HOME")
            else:
                arm.pick_and_place(
                    decision, gripper, pick_target,
                    rotation_angle=rotation_angle,
                )

            # Drain ALL stale frames captured during arm motion
            drained = 0
            while True:
                try:
                    sensor_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break

            # Wait for arm to be clear of camera view + flush camera buffer
            # 5 frames at 1Hz = 5 seconds of fresh frames discarded
            logger.info("Waiting for arm to clear camera view...")
            for _ in range(5):
                try:
                    sensor_queue.get(timeout=2.0)
                    drained += 1
                except queue.Empty:
                    break

            if drained:
                logger.debug("Drained %d total frames", drained)

            # Reset settle timer — next block must be stable for full 2s
            _stable_since = None
            _stable_bbox = None

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
                captured_ref = _upload_frame(clients, item)

                try:
                    from foundry_sdk_runtime.types import ActionConfig, ActionMode, ReturnEditsMode
                    response = clients.client.ontology.actions.create_inspection_event(
                        action_config=ActionConfig(
                            mode=ActionMode.VALIDATE_AND_EXECUTE,
                            return_edits=ReturnEditsMode.ALL,
                        ),
                        inspection_id=item["inspection_id"],
                        robot_id=settings.robot_id,
                        timestamp=ts,
                        vision_class=detected_class,
                        vision_confidence=confidence,
                        grip_load=grip_data.normalized_load,
                        fusion_decision=decision,
                        fusion_reason=reason,
                        vision_agrees=True,
                        model_version=model.version if settings.detection_mode == "model" else "color-v1",
                        cycle_time_ms=cycle_time_ms,
                        review_status=review_status,
                        captured_image_ref=captured_ref if captured_ref is not None else Empty.value,
                    )
                    logger.info(
                        "InspectionEvent %s: validation=%s, edits=%s",
                        item["inspection_id"],
                        response.validation,
                        response.edits,
                    )
                except Exception as exc:
                    logger.error("Failed to create InspectionEvent: %s", exc)
            else:
                logger.debug("[MOCK] InspectionEvent %s → %s", item["inspection_id"], decision)

            # ── Update shared state ───────────────────────────────
            sensor_state["vision_confidence"] = confidence
            sensor_state["grip_load"] = grip_data.normalized_load
            sensor_state["grip_servo_load"] = grip_data.servo_load
            sensor_state["grip_state"] = grip_data.grip_state
            sensor_state["object_detected"] = grip_data.object_detected

            state.total_inspections += 1
            logger.info(
                "Inspection #%d: %s (conf=%.2f) → %s [%dms]",
                state.total_inspections,
                detected_class,
                confidence,
                decision,
                cycle_time_ms,
            )

        except Exception:
            logger.exception("defect_detection cycle failed — will retry")
            time.sleep(1)


# ── Helper functions ──────────────────────────────────────────────────


def _update_sensor_state(gripper: Gripper, sensor_state: dict) -> None:
    """Read joint temperatures and write to shared state for Process 1."""
    temps = gripper.read_joint_temperatures()
    sensor_state["joint_temps"] = temps


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
            .where(OperatorCommandObjectType.robot_id == settings.robot_id)
            .where(OperatorCommandObjectType.status == "PENDING")
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


def _bbox_moved(prev: list, curr: list, threshold: float = 20.0) -> bool:
    """Check if bounding box center moved more than threshold pixels."""
    if len(prev) != 4 or len(curr) != 4:
        return True
    prev_cx = prev[0] + prev[2] / 2
    prev_cy = prev[1] + prev[3] / 2
    curr_cx = curr[0] + curr[2] / 2
    curr_cy = curr[1] + curr[3] / 2
    dist = ((prev_cx - curr_cx) ** 2 + (prev_cy - curr_cy) ** 2) ** 0.5
    return dist > threshold


def _upload_frame(clients: FoundryClients, item: dict):
    """Encode the camera frame as JPEG and upload as a MediaReference."""
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
