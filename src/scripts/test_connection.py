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

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test-connection")

# Action API names
ACTION_CREATE_INSPECTION = "create-inspection-event"
ACTION_UPDATE_ROBOT = "update-robot-status"
ACTION_SEND_COMMAND = "send-command"
ACTION_ACK_COMMAND = "acknowledge-command"
ACTION_PUBLISH_MODEL = "publish-model"

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"


# ═══════════════════════════════════════════════════════════════════════
# Connectivity checks
# ═══════════════════════════════════════════════════════════════════════

def check_oauth(clients: FoundryClients) -> bool:
    """Test 1: OAuth2 token acquisition."""
    try:
        token = clients._refresh_token()
        return bool(token)
    except Exception as exc:
        logger.error("OAuth2 failed: %s", exc)
        return False


def check_stream_push(clients: FoundryClients, stream_rid: str, name: str) -> bool:
    """Test 2/3: Push a single test record to a stream."""
    record = {
        "readingId": f"test-{uuid.uuid4()}",
        "inspectionId": f"test-insp-{uuid.uuid4()}",
        "robotId": "test-connectivity",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    }
    if "vision" in name.lower():
        record.update({
            "detectedClass": "widget_good",
            "confidence": 0.99,
            "boundingBox": "[100, 100, 50, 50]",
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
    """Test 4: Read Robot objects via OSDK."""
    try:
        result = clients.query_objects("robot", page_size=1)
        return isinstance(result, list)
    except Exception as exc:
        logger.error("OSDK read failed: %s", exc)
        return False


def check_osdk_write(clients: FoundryClients, settings: Settings) -> bool:
    """Test 5: Create an InspectionEvent via OSDK and verify."""
    test_id = f"test-insp-{uuid.uuid4()}"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return clients.apply_action(ACTION_CREATE_INSPECTION, {
        "inspectionId": test_id,
        "robotId": settings.robot_id,
        "timestamp": ts,
        "visionClass": "widget_good",
        "visionConfidence": 0.99,
        "gripLoad": 0.1,
        "fusionDecision": "PASS",
        "fusionReason": "connectivity_test",
        "visionAgrees": True,
        "modelVersion": "test",
        "cycleTimeMs": 1,
        "reviewStatus": "NOT_REQUIRED",
    })


def check_model_registry(clients: FoundryClients) -> bool:
    """Test 6: Query ModelRegistry."""
    try:
        result = clients.query_objects("model-registry", page_size=1)
        return isinstance(result, list)
    except Exception as exc:
        logger.error("ModelRegistry query failed: %s", exc)
        return False


def check_commands(clients: FoundryClients, settings: Settings) -> bool:
    """Test 7: Poll OperatorCommands (expect 0 or more)."""
    try:
        result = clients.query_objects(
            "operator-command",
            where={
                "type": "eq",
                "field": "robotId",
                "value": settings.robot_id,
            },
            page_size=1,
        )
        return isinstance(result, list)
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

    print(f"\n🌱 Seeding {count} inspection events + supporting data...\n")
    now = datetime.now(timezone.utc)

    # ── 1. Robot ──────────────────────────────────────────────────
    print("  Creating Robot...")
    clients.apply_action(ACTION_UPDATE_ROBOT, {
        "robotId": settings.robot_id,
        "status": "RUNNING",
        "currentModelVersion": "v1.0.0",
        "totalInspections": count,
    })

    # ── 2. Model Registry entry ───────────────────────────────────
    print("  Creating ModelRegistry entry (v1.0.0)...")
    clients.apply_action(ACTION_PUBLISH_MODEL, {
        "modelId": f"model-{uuid.uuid4()}",
        "version": "v1.0.0",
        "status": "PUBLISHED",
        "accuracy": 0.91,
        "description": "YOLOv5-nano baseline — 3-class widget detector",
        "artifactPath": "/models/yolov5n_widget_v1.onnx",
    })

    # ── 3. Inspection events + stream readings ────────────────────
    # Distribution: ~70% PASS, ~20% FAIL, ~10% REVIEW
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
            # PASS — high confidence, low grip load
            vision_class = "widget_good"
            confidence = round(min(0.99, max(0.5, random.gauss(0.92, 0.04))), 4)
            grip_load = round(min(0.99, max(0.01, random.gauss(0.25, 0.10))), 4)
            decision, reason, agrees = "PASS", "vision_pass + grip_normal", True
        elif roll < 0.90:
            # FAIL — low confidence, high grip load
            vision_class = "widget_defect"
            confidence = round(min(0.99, max(0.1, random.gauss(0.82, 0.08))), 4)
            grip_load = round(min(0.99, max(0.3, random.gauss(0.78, 0.08))), 4)
            decision, reason, agrees = "FAIL", "vision_fail + grip_anomaly", True
        else:
            # REVIEW — sensors disagree
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
            "boundingBox": f"[{x}, {y}, {w}, {h}]",
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

        # InspectionEvent via OSDK
        clients.apply_action(ACTION_CREATE_INSPECTION, {
            "inspectionId": inspection_id,
            "robotId": settings.robot_id,
            "timestamp": ts_str,
            "visionClass": vision_class,
            "visionConfidence": confidence,
            "gripLoad": grip_load,
            "fusionDecision": decision,
            "fusionReason": reason,
            "visionAgrees": agrees,
            "modelVersion": "v1.0.0",
            "cycleTimeMs": cycle_time,
            "reviewStatus": review_status,
        })

        progress = f"  [{i + 1}/{count}] {inspection_id[:20]}... → {decision}"
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
    for cmd_type, status in [("PAUSE", "EXECUTED"), ("RESUME", "EXECUTED"), ("UPDATE_TOLERANCE", "PENDING")]:
        cmd_id = f"cmd-{uuid.uuid4()}"
        payload = json.dumps({"tolerance": 0.65}) if cmd_type == "UPDATE_TOLERANCE" else ""
        clients.apply_action(ACTION_SEND_COMMAND, {
            "commandId": cmd_id,
            "robotId": settings.robot_id,
            "commandType": cmd_type,
            "payload": payload,
        })
        if status == "EXECUTED":
            clients.apply_action(ACTION_ACK_COMMAND, {
                "commandId": cmd_id,
                "status": "EXECUTED",
            })

    # ── Summary ───────────────────────────────────────────────────
    pass_count = int(count * 0.70)
    fail_count = int(count * 0.20)
    review_count = count - pass_count - fail_count

    print("\n" + "=" * 60)
    print("  🌱 Seed Data Complete!")
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
        print("⚠️  Fix connectivity issues before seeding data.")
        sys.exit(1)

    if args.seed:
        seed_data(settings, clients, args.count)


if __name__ == "__main__":
    main()
