"""Process 3 — Model Upgrade.

Polls Foundry's ModelRegistry for newly published model versions, downloads
the artifact, optionally converts to TensorRT, and signals Process 2 to
hot-swap the model.
"""

from __future__ import annotations

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
    from physical_ai_qa_cell_sdk.ontology.search import ModelRegistryObjectType

    try:
        models = (
            clients.client.ontology.objects.ModelRegistry
            .where(ModelRegistryObjectType.status.eq("PUBLISHED"))
            .order_by(ModelRegistryObjectType.published_at.desc())
            .take(1)
        )
    except Exception as exc:
        logger.error("Failed to query ModelRegistry: %s", exc)
        return None

    if not models:
        logger.debug("No published models found")
        return None

    latest = models[0]
    latest_version = latest.version or ""

    if latest_version <= current_version:
        logger.debug("Already on latest model %s", current_version)
        return None

    logger.info("New model available: %s → %s", current_version, latest_version)

    # ── 2. Download artifact via SDK media API ───────────────────
    staging_path = _download_model_artifact(clients, latest, settings)
    if staging_path is None:
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
    try:
        clients.client.ontology.actions.update_robot_status(
            robot=settings.robot_id,
            status="RUNNING",
            current_model_version=latest_version,
        )
    except Exception as exc:
        logger.error("Failed to update robot status: %s", exc)

    logger.info("Model upgrade complete: %s → %s", current_version, latest_version)
    return latest_version


def _download_model_artifact(
    clients: FoundryClients,
    model_obj,
    settings: Settings,
) -> Optional[str]:
    """Download the model artifact via the SDK media API.

    Uses ``model_artifact_ref.get_media_content()`` to stream the file bytes
    and ``get_media_metadata()`` to determine the filename extension.

    Returns the local staging path on success, or ``None``.
    """
    from foundry_sdk_runtime import AllowBetaFeatures

    try:
        with AllowBetaFeatures():
            media_ref = model_obj.model_artifact_ref
            if media_ref is None:
                logger.error("Model %s has no model_artifact_ref", model_obj.version)
                return None

            metadata = media_ref.get_media_metadata()
            content = media_ref.get_media_content()

        # Determine file extension from metadata path or default to .onnx
        ext = ".onnx"
        if metadata and metadata.path:
            _, found_ext = os.path.splitext(metadata.path)
            if found_ext:
                ext = found_ext

        version = model_obj.version or "unknown"
        local_path = os.path.join(settings.model_staging_dir, f"model_{version}{ext}")

        with open(local_path, "wb") as f:
            f.write(content.read())

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(
            "Downloaded model artifact: %.1f MB → %s (media: %s)",
            size_mb, local_path, metadata.path if metadata else "unknown",
        )
        return local_path
    except Exception as exc:
        logger.error("Model artifact download failed: %s", exc)
        return None


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
