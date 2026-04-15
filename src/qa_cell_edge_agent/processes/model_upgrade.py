"""Process 3 — Model Upgrade.

Polls Foundry's ModelRegistry for newly published model versions, downloads
the artifact, optionally converts to TensorRT, and signals Process 2 to
hot-swap the model.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import time
from multiprocessing import Event
from typing import Optional

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients
from qa_cell_edge_agent.config.jetson import get_trtexec_args

logger = logging.getLogger(__name__)


def run_model_upgrade(
    reload_event: Event,
    settings: Optional[Settings] = None,
) -> None:
    """Entry point for the model-upgrade process (long-running)."""

    settings = settings or Settings()
    clients = FoundryClients(settings=settings)
    current_version: str = "v1.0.0"

    logger.info(
        "model_upgrade started — poll_interval=%ds, auto_upgrade=%s",
        settings.model_poll_interval_sec,
        settings.enable_auto_upgrade,
    )

    os.makedirs(settings.model_staging_dir, exist_ok=True)

    while True:
        try:
            if settings.enable_auto_upgrade and not settings.mock_foundry:
                new_version = _check_and_upgrade(
                    settings, clients, current_version, reload_event
                )
                if new_version:
                    current_version = new_version
        except Exception:
            logger.exception("model_upgrade cycle failed — will retry")

        time.sleep(settings.model_poll_interval_sec)


def _check_and_upgrade(
    settings: Settings,
    clients: FoundryClients,
    current_version: str,
    reload_event: Event,
) -> Optional[str]:
    """Check for a newer published model and upgrade if found.

    Returns the new version string on success, or None.
    """

    # ── 1. Query ModelRegistry for PUBLISHED models ───────────────
    models = clients.query_objects(
        "model-registry",
        where={"type": "eq", "field": "status", "value": "PUBLISHED"},
        order_by="publishedAt",
    )

    if not models:
        logger.debug("No published models found")
        return None

    latest = models[0]
    props = latest.get("properties", latest)
    latest_version = props.get("version", "")
    artifact_path = props.get("artifactPath", "")

    if latest_version <= current_version:
        logger.debug("Already on latest model %s", current_version)
        return None

    logger.info(
        "New model available: %s → %s (artifact: %s)",
        current_version,
        latest_version,
        artifact_path,
    )

    # ── 2. Download artifact ──────────────────────────────────────
    staging_path = os.path.join(settings.model_staging_dir, f"model_{latest_version}")
    if not _download_artifact(settings, clients, artifact_path, staging_path):
        return None

    # ── 3. Convert to TensorRT if ONNX ───────────────────────────
    engine_path = staging_path
    if staging_path.endswith(".onnx"):
        engine_path = staging_path.replace(".onnx", ".engine")
        if not _convert_to_tensorrt(settings, staging_path, engine_path):
            return None

    # ── 4. Atomic swap ────────────────────────────────────────────
    try:
        shutil.move(engine_path, settings.model_path)
        logger.info("Model file swapped to %s", settings.model_path)
    except OSError as exc:
        logger.error("Failed to swap model file: %s", exc)
        return None

    # ── 5. Signal Process 2 to reload ─────────────────────────────
    reload_event.set()
    logger.info("Reload signal sent to defect_detection")

    # ── 6. Update robot status with new version ───────────────────
    clients.apply_action("update-robot-status", {
        "robotId": settings.robot_id,
        "currentModelVersion": latest_version,
    })

    logger.info("Model upgrade complete: %s → %s", current_version, latest_version)
    return latest_version


def _download_artifact(
    settings: Settings,
    clients: FoundryClients,
    artifact_path: str,
    local_path: str,
) -> bool:
    """Download a model artifact from Foundry.

    ``artifact_path`` may be a media-set reference or a direct URL.
    Returns True on success.
    """
    try:
        resp = clients.session.get(
            f"{settings.foundry_url}{artifact_path}",
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info("Downloaded model artifact: %.1f MB → %s", size_mb, local_path)
        return True
    except Exception as exc:
        logger.error("Artifact download failed: %s", exc)
        return False


def _convert_to_tensorrt(
    settings: Settings,
    onnx_path: str,
    engine_path: str,
) -> bool:
    """Convert an ONNX model to a TensorRT engine using trtexec.

    Returns True on success.
    """
    # Use Jetson Nano 2GB-aware workspace size (256 MB default, not 1024)
    cmd = get_trtexec_args(onnx_path, engine_path)
    logger.info("Converting ONNX → TensorRT: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            logger.error("trtexec failed:\n%s", result.stderr[-2000:])
            return False
        logger.info("TensorRT conversion complete → %s", engine_path)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.error("TensorRT conversion error: %s", exc)
        return False
