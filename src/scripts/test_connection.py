#!/usr/bin/env python3
"""Test Connection & Seed Data Script.

Usage:
    python scripts/test_connection.py                  # connectivity checks only
    python scripts/test_connection.py --seed            # checks + seed 50 events
    python scripts/test_connection.py --seed --count 20 # checks + seed 20 events
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv

from foundry_sdk_runtime.types.null_types import Empty

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test-connection")

PASS = "\033[92m\u2713 PASS\033[0m"
FAIL = "\033[91m\u2717 FAIL\033[0m"


# ═══════════════════════════════════════════════════════════════════════
# Connectivity checks
# ═══════════════════════════════════════════════════════════════════════

def check_oauth(clients: FoundryClients) -> bool:
    """Test 1: OAuth2 token acquisition via SDK auth."""
    try:
        token = clients.auth.get_token()
        return bool(token.access_token)
    except Exception as exc:
        logger.error("OAuth2 failed: %s", exc)
        return False


def check_stream_push(clients: FoundryClients, stream_rid: str, name: str) -> bool:
    """Test 2/3: Push a single test record to a stream."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    record: Dict[str, Any] = {
        "readingId": f"test-{uuid.uuid4()}",
        "inspectionId": f"test-insp-{uuid.uuid4()}",
        "robotId": "test-connectivity",
        "timestamp": ts,
    }
    if "vision" in name.lower():
        record.update({
            "detectedClass": "widget_good",
            "confidence": 0.99,
            "boundingBox": [100.0, 100.0, 50.0, 50.0],
            "inferenceTimeMs": 1,
            "modelVersion": "test",
            "thumbnailBase64": "",
        })
    else:
        record.update({
            "servoLoad": 100.0,
            "normalizedLoad": 0.2,
            "gripState": "OPEN",
            "objectDetected": False,
        })
    return clients.push_to_stream(stream_rid, [record])


def check_osdk_read(clients: FoundryClients) -> bool:
    """Test 4: Read Robot objects via SDK."""
    try:
        robots = clients.client.ontology.objects.Robot.take(1)
        return isinstance(robots, list)
    except Exception as exc:
        logger.error("OSDK read failed: %s", exc)
        return False


def check_osdk_write(clients: FoundryClients, settings: Settings) -> bool:
    """Test 5: Create an InspectionEvent via SDK."""
    try:
        clients.client.ontology.actions.create_inspection_event(
            inspection_id=f"test-insp-{uuid.uuid4()}",
            robot_id=settings.robot_id,
            timestamp=datetime.now(timezone.utc),
            vision_class="widget_good",
            vision_confidence=0.99,
            grip_load=0.1,
            fusion_decision="PASS",
            fusion_reason="connectivity_test",
            vision_agrees=True,
            model_version="test",
            cycle_time_ms=1,
            review_status="NOT_REQUIRED",
            captured_image_ref=Empty.value,
        )
        return True
    except Exception as exc:
        logger.error("OSDK write failed: %s", exc)
        return False


def check_model_registry(clients: FoundryClients) -> bool:
    """Test 6: Query ModelRegistry via SDK."""
    try:
        models = clients.client.ontology.objects.ModelRegistry.take(1)
        return isinstance(models, list)
    except Exception as exc:
        logger.error("ModelRegistry query failed: %s", exc)
        return False


def check_commands(clients: FoundryClients, settings: Settings) -> bool:
    """Test 7: Poll OperatorCommands via SDK."""
    try:
        from physical_ai_qa_cell_sdk.ontology.search import OperatorCommandObjectType

        commands = (
            clients.client.ontology.objects.OperatorCommand
            .where(OperatorCommandObjectType.robot_id == settings.robot_id)
            .take(1)
        )
        return isinstance(commands, list)
    except Exception as exc:
        logger.error("Command poll failed: %s", exc)
        return False


def run_checks(settings: Settings, clients: FoundryClients) -> bool:
    """Run all connectivity checks. Returns True if all pass."""
    checks = [
        ("OAuth2 token acquisition", lambda: check_oauth(clients)),
        ("Push to vision-readings stream", lambda: check_stream_push(
            clients, settings.vision_stream_rid, "vision")),
        ("Push to grip-readings stream", lambda: check_stream_push(
            clients, settings.grip_stream_rid, "grip")),
        ("OSDK read (Robot objects)", lambda: check_osdk_read(clients)),
        ("OSDK write (CreateInspectionEvent)", lambda: check_osdk_write(clients, settings)),
        ("Query ModelRegistry", lambda: check_model_registry(clients)),
        ("Poll OperatorCommands", lambda: check_commands(clients, settings)),
    ]

    all_passed = True
    print("\n" + "=" * 60)
    print("  QA Cell Edge Agent — Connectivity Check")
    print("=" * 60)

    for i, (name, fn) in enumerate(checks, 1):
        t0 = time.monotonic()
        try:
            ok = fn()
        except Exception as exc:
            ok = False
            logger.error("Unexpected error: %s", exc)
        elapsed = (time.monotonic() - t0) * 1000
        status = PASS if ok else FAIL
        print(f"  {i}. {status}  {name}  ({elapsed:.0f}ms)")
        if not ok:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print(f"  {PASS}  All {len(checks)} checks passed!")
    else:
        print(f"  {FAIL}  Some checks failed — see errors above.")
    print("=" * 60 + "\n")
    return all_passed


# ═══════════════════════════════════════════════════════════════════════
# Seed data generation
# ═══════════════════════════════════════════════════════════════════════

def seed_data(settings: Settings, clients: FoundryClients, count: int) -> None:
    """Generate and push realistic seed data to Foundry."""

    print(f"\n  Seeding {count} inspection events + supporting data...\n")
    now = datetime.now(timezone.utc)

    # ── 1. Robot status ──────────────────────────────────────────────
    print("  Updating Robot status...")
    clients.client.ontology.actions.update_robot_status(
        robot=settings.robot_id,
        status="RUNNING",
        current_model_version="v1.0.0",
        total_inspections=count,
    )

    # ── 2. Model Registry entry ───────────────────────────────────
    print("  Publishing model v1.0.0...")
    clients.client.ontology.actions.publish_model(
        model_id=f"model-{uuid.uuid4()}",
        version="v1.0.0",
        model_status="PUBLISHED",
        accuracy=0.91,
        model_description="YOLOv5-nano baseline — 3-class widget detector",
        artifact_path="/models/yolov5n_widget_v1.onnx",
    )

    # ── 3. Inspection events + stream readings ────────────────────
    interval = timedelta(hours=2) / max(count, 1)

    vision_batch: List[Dict[str, Any]] = []
    grip_batch: List[Dict[str, Any]] = []

    for i in range(count):
        ts = now - timedelta(hours=2) + (interval * i)
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        inspection_id = f"insp-{uuid.uuid4()}"
        vr_id = f"vr-{uuid.uuid4()}"
        gr_id = f"gr-{uuid.uuid4()}"

        # Generate realistic correlated data
        roll = random.random()
        if roll < 0.70:
            vision_class = "widget_good"
            confidence = round(min(0.99, max(0.5, random.gauss(0.92, 0.04))), 4)
            grip_load = round(min(0.99, max(0.01, random.gauss(0.25, 0.10))), 4)
            decision, reason, agrees = "PASS", "vision_pass + grip_normal", True
        elif roll < 0.90:
            vision_class = "widget_defect"
            confidence = round(min(0.99, max(0.1, random.gauss(0.82, 0.08))), 4)
            grip_load = round(min(0.99, max(0.3, random.gauss(0.78, 0.08))), 4)
            decision, reason, agrees = "FAIL", "vision_fail + grip_anomaly", True
        else:
            if random.random() < 0.5:
                vision_class = "widget_good"
                confidence = round(min(0.99, max(0.5, random.gauss(0.85, 0.06))), 4)
                grip_load = round(min(0.99, max(0.4, random.gauss(0.72, 0.06))), 4)
                reason = "vision_pass + grip_anomaly"
            else:
                vision_class = "widget_defect"
                confidence = round(min(0.99, max(0.2, random.gauss(0.60, 0.10))), 4)
                grip_load = round(min(0.60, max(0.01, random.gauss(0.30, 0.10))), 4)
                reason = "vision_fail + grip_normal"
            decision = "REVIEW"
            agrees = False

        review_status = "PENDING_REVIEW" if decision == "REVIEW" else "NOT_REQUIRED"
        servo_load = round(grip_load * 500, 2)
        cycle_time = random.randint(1800, 3200)
        x, y = random.randint(50, 400), random.randint(50, 300)
        w, h = random.randint(60, 150), random.randint(60, 150)

        # Stream records
        vision_batch.append({
            "readingId": vr_id,
            "inspectionId": inspection_id,
            "robotId": settings.robot_id,
            "timestamp": ts_str,
            "detectedClass": vision_class,
            "confidence": confidence,
            "boundingBox": [float(x), float(y), float(w), float(h)],
            "inferenceTimeMs": random.randint(25, 55),
            "modelVersion": "v1.0.0",
            "thumbnailBase64": "",
        })
        grip_batch.append({
            "readingId": gr_id,
            "inspectionId": inspection_id,
            "robotId": settings.robot_id,
            "timestamp": ts_str,
            "servoLoad": servo_load,
            "normalizedLoad": grip_load,
            "gripState": "CLOSED",
            "objectDetected": True,
        })

        # InspectionEvent via SDK
        clients.client.ontology.actions.create_inspection_event(
            inspection_id=inspection_id,
            robot_id=settings.robot_id,
            timestamp=ts,
            vision_class=vision_class,
            vision_confidence=confidence,
            grip_load=grip_load,
            fusion_decision=decision,
            fusion_reason=reason,
            vision_agrees=agrees,
            model_version="v1.0.0",
            cycle_time_ms=cycle_time,
            review_status=review_status,
            captured_image_ref=Empty.value,
        )

        progress = f"  [{i + 1}/{count}] {inspection_id[:20]}... -> {decision}"
        print(progress, end="\r")

    print(f"\n  Created {count} InspectionEvents via OSDK")

    # Push stream data in batches of 25
    batch_size = 25
    for start in range(0, len(vision_batch), batch_size):
        clients.push_to_stream(
            settings.vision_stream_rid,
            vision_batch[start:start + batch_size],
        )
        clients.push_to_stream(
            settings.grip_stream_rid,
            grip_batch[start:start + batch_size],
        )
    print(f"  Pushed {len(vision_batch)} VisionReadings to stream")
    print(f"  Pushed {len(grip_batch)} GripReadings to stream")

    # ── 4. OperatorCommands ───────────────────────────────────────
    print("  Creating sample OperatorCommands...")
    for cmd_type, ack in [("PAUSE", True), ("RESUME", True), ("UPDATE_TOLERANCE", False)]:
        cmd_id = f"cmd-{uuid.uuid4()}"
        payload = json.dumps({"tolerance": 0.65}) if cmd_type == "UPDATE_TOLERANCE" else ""
        clients.client.ontology.actions.send_command(
            robot=settings.robot_id,
            command_type=cmd_type,
            command_id=cmd_id,
            payload=payload if payload else None,
        )
        if ack:
            clients.client.ontology.actions.acknowledge_command(
                command=cmd_id,
                new_status="EXECUTED",
            )

    # ── Summary ───────────────────────────────────────────────────
    pass_count = int(count * 0.70)
    fail_count = int(count * 0.20)
    review_count = count - pass_count - fail_count

    print("\n" + "=" * 60)
    print("  Seed Data Complete!")
    print(f"    Robot:            {settings.robot_id} (RUNNING)")
    print(f"    Model:            v1.0.0 (PUBLISHED)")
    print(f"    Inspections:      {count} total")
    print(f"      PASS:           ~{pass_count}")
    print(f"      FAIL:           ~{fail_count}")
    print(f"      REVIEW:         ~{review_count} (PENDING_REVIEW)")
    print(f"    VisionReadings:   {count} (streamed)")
    print(f"    GripReadings:     {count} (streamed)")
    print(f"    Commands:         3 (2 executed, 1 pending)")
    print(f"    Time span:        last 2 hours")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="QA Cell — Test Connection & Seed Data")
    parser.add_argument("--seed", action="store_true", help="Seed demo data after checks")
    parser.add_argument("--count", type=int, default=50, help="Number of inspections to seed (default: 50)")
    args = parser.parse_args()

    settings = Settings()
    clients = FoundryClients(settings=settings)

    ok = run_checks(settings, clients)
    if not ok:
        print("  Fix connectivity issues before seeding data.")
        sys.exit(1)

    if args.seed:
        seed_data(settings, clients, args.count)


if __name__ == "__main__":
    main()
