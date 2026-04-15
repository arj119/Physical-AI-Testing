"""Process 2 — Defect Detection.

Dequeues sensor data from Process 1, runs YOLOv5 inference, applies sensor
fusion, commands the robot arm to sort the part, and writes results to Foundry
via OSDK actions.  Also handles operator command polling and heartbeat updates.
"""

from __future__ import annotations

import json
import logging
import time
from multiprocessing import Event, Queue
from typing import Optional

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients
from qa_cell_edge_agent.drivers.arm import Arm
from qa_cell_edge_agent.drivers.gripper import Gripper
from qa_cell_edge_agent.fusion.engine import FusionEngine
from qa_cell_edge_agent.models.inference import ModelInference

logger = logging.getLogger(__name__)

# Action type API names (used with the v2 apply endpoint)
ACTION_CREATE_INSPECTION = "create-inspection-event"
ACTION_UPDATE_ROBOT = "update-robot-status"
ACTION_ACK_COMMAND = "acknowledge-command"


class RobotState:
    """Mutable state machine for the robot's operational mode."""

    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    E_STOPPED = "E_STOPPED"

    def __init__(self) -> None:
        self.status: str = self.RUNNING
        self.total_inspections: int = 0


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
                arm.safe_position()
                time.sleep(settings.command_poll_interval_sec)
                continue

            # ── Dequeue sensor data ───────────────────────────────
            try:
                item = queue.get(timeout=2.0)
            except Exception:
                continue  # nothing in queue — loop back to poll commands

            cycle_start = time.monotonic()

            # ── Run inference ─────────────────────────────────────
            result = model.infer(item["frame"])

            # ── Sensor fusion ─────────────────────────────────────
            fusion_result = fusion.decide(
                vision_class=result.detected_class,
                confidence=result.confidence,
                normalized_load=item["grip_data"].normalized_load,
            )

            # ── Sort part ─────────────────────────────────────────
            arm.pick_and_place(fusion_result.decision)
            gripper.release()

            cycle_time_ms = int((time.monotonic() - cycle_start) * 1000)

            # ── Create InspectionEvent via OSDK ───────────────────
            review_status = (
                "PENDING_REVIEW" if fusion_result.decision == "REVIEW"
                else "NOT_REQUIRED"
            )
            if not settings.mock_foundry:
                clients.apply_action(ACTION_CREATE_INSPECTION, {
                    "inspectionId": item["inspection_id"],
                    "robotId": settings.robot_id,
                    "timestamp": item["timestamp"],
                    "visionClass": result.detected_class,
                    "visionConfidence": result.confidence,
                    "gripLoad": item["grip_data"].normalized_load,
                    "fusionDecision": fusion_result.decision,
                    "fusionReason": fusion_result.reason,
                    "visionAgrees": fusion_result.vision_agrees,
                    "modelVersion": model.version,
                    "cycleTimeMs": cycle_time_ms,
                    "reviewStatus": review_status,
                })
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
                item["grip_data"].normalized_load,
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

    commands = clients.query_objects(
        "operator-command",
        where={
            "type": "and",
            "value": [
                {"type": "eq", "field": "robotId", "value": settings.robot_id},
                {"type": "eq", "field": "status", "value": "PENDING"},
            ],
        },
        order_by="createdAt",
    )

    for cmd in commands:
        props = cmd.get("properties", cmd)
        cmd_type = props.get("commandType", "")
        cmd_id = props.get("commandId", "")
        payload_str = props.get("payload", "{}")

        logger.info("Processing command %s: %s", cmd_id, cmd_type)

        if cmd_type == "PAUSE":
            state.status = RobotState.PAUSED
        elif cmd_type == "RESUME":
            state.status = RobotState.RUNNING
        elif cmd_type == "E_STOP":
            state.status = RobotState.E_STOPPED
            arm.safe_position()
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
        clients.apply_action(ACTION_ACK_COMMAND, {
            "commandId": cmd_id,
            "status": "EXECUTED",
        })


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

    clients.apply_action(ACTION_UPDATE_ROBOT, {
        "robotId": settings.robot_id,
        "status": state.status,
        "currentModelVersion": model_version,
        "totalInspections": state.total_inspections,
    })
