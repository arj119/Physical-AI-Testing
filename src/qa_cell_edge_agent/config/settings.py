"""Centralised configuration loaded from environment variables with sensible defaults.

Hardware ports and camera index are auto-discovered at startup when not
set via environment variables.  See ``drivers.discovery`` for details.
"""

import os
from dataclasses import dataclass, field
from typing import Dict

from qa_cell_edge_agent.drivers.discovery import find_camera_index, find_mycobot_port


def _parse_class_to_bin(env_value: str) -> Dict[str, str]:
    """Parse ``CLASS_TO_BIN`` env var, e.g. ``yellow:BIN_A,green:BIN_B``."""
    mapping: Dict[str, str] = {}
    if not env_value:
        return mapping
    for pair in env_value.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        key, value = key.strip(), value.strip()
        if key and value:
            mapping[key] = value
    return mapping


@dataclass(frozen=True)
class Settings:
    """All tuneable parameters for the edge agent.

    Values come from environment variables (or .env via python-dotenv).
    Every setting has a safe default so the agent can start in mock mode
    without any env file present.
    """

    # ── Foundry connection ────────────────────────────────────────────
    foundry_url: str = field(
        default_factory=lambda: os.environ.get("FOUNDRY_URL", "https://localhost")
    )
    client_id: str = field(
        default_factory=lambda: os.environ.get("CLIENT_ID", "")
    )
    client_secret: str = field(
        default_factory=lambda: os.environ.get("CLIENT_SECRET", "")
    )

    # ── Stream RIDs ───────────────────────────────────────────────────
    vision_stream_rid: str = field(
        default_factory=lambda: os.environ.get(
            "VISION_STREAM_RID",
            "",
        )
    )
    grip_stream_rid: str = field(
        default_factory=lambda: os.environ.get(
            "GRIP_STREAM_RID",
            "",
        )
    )
    telemetry_stream_rid: str = field(
        default_factory=lambda: os.environ.get(
            "TELEMETRY_STREAM_RID",
            "",
        )
    )

    # ── Robot identity ────────────────────────────────────────────────
    robot_id: str = field(
        default_factory=lambda: os.environ.get("ROBOT_ID", "robot-jetson-01")
    )
    robot_name: str = field(
        default_factory=lambda: os.environ.get("ROBOT_NAME", "QA Cell Robot 01")
    )

    # ── Sensor push (Process 1) ──────────────────────────────────────
    capture_interval_sec: float = field(
        default_factory=lambda: float(os.environ.get("CAPTURE_INTERVAL_SEC", "1.0"))
    )
    camera_device_index: int = field(
        default_factory=find_camera_index,
    )
    thumbnail_size: tuple = (64, 64)
    stream_push_timeout_sec: int = 5
    stream_retry_count: int = 3

    # ── Defect detection (Process 2) ──────────────────────────────────
    confidence_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
    )
    grip_tolerance: float = field(
        default_factory=lambda: float(os.environ.get("GRIP_TOLERANCE", "0.65"))
    )
    command_poll_interval_sec: float = field(
        default_factory=lambda: float(os.environ.get("COMMAND_POLL_INTERVAL_SEC", "1.0"))
    )
    heartbeat_interval_sec: float = field(
        default_factory=lambda: float(os.environ.get("HEARTBEAT_INTERVAL_SEC", "5.0"))
    )

    # ── Model management (Process 3) ──────────────────────────────────
    model_path: str = field(
        default_factory=lambda: os.environ.get("MODEL_PATH", "./models/yolov5n.onnx")
    )
    model_poll_interval_sec: float = field(
        default_factory=lambda: float(os.environ.get("MODEL_POLL_INTERVAL_SEC", "60"))
    )
    model_staging_dir: str = "./models/staging/"

    # ── Detection mode ───────────────────────────────────────────────
    # "color" = HSV block detection (no model needed, sorts by color)
    # "model" = YOLOv5 inference + sensor fusion (needs trained model)
    detection_mode: str = field(
        default_factory=lambda: os.environ.get("DETECTION_MODE", "color")
    )

    # ── Hardware ──────────────────────────────────────────────────────
    mycobot_port: str = field(
        default_factory=lambda: find_mycobot_port() or "/dev/ttyUSB0",
    )
    mycobot_baud: int = field(
        default_factory=lambda: int(os.environ.get("MYCOBOT_BAUD", "115200"))
    )

    # ── Closed-loop pick parameters ──────────────────────────────────
    # See drivers/arm.py for usage. Values mirrored here so callers can
    # introspect / log the active configuration without importing arm.
    max_pick_retries: int = field(
        default_factory=lambda: int(os.environ.get("MAX_PICK_RETRIES", "2"))
    )
    grip_load_success_threshold: float = field(
        default_factory=lambda: float(os.environ.get("GRIP_LOAD_SUCCESS_THRESHOLD", "0.15"))
    )
    grip_verify_delay_s: float = field(
        default_factory=lambda: float(os.environ.get("GRIP_VERIFY_DELAY_S", "0.4"))
    )
    retry_nudge_mm: float = field(
        default_factory=lambda: float(os.environ.get("RETRY_NUDGE_MM", "1.5"))
    )
    camera_rotation_offset_deg: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "CAMERA_ROTATION_OFFSET_DEG",
                os.environ.get("CAMERA_ROTATION_OFFSET", "0"),
            )
        )
    )

    # ── Pose buffer (multi-frame smoothing) ──────────────────────────
    pose_buffer_n: int = field(
        default_factory=lambda: int(os.environ.get("POSE_BUFFER_N", "5"))
    )
    pose_buffer_xy_tol_mm: float = field(
        default_factory=lambda: float(os.environ.get("POSE_BUFFER_XY_TOL_MM", "3.0"))
    )
    pose_buffer_yaw_tol_deg: float = field(
        default_factory=lambda: float(os.environ.get("POSE_BUFFER_YAW_TOL_DEG", "5.0"))
    )

    # ── Bin routing for colour sorting ───────────────────────────────
    # When non-empty, takes precedence over fusion-decision → bin mapping.
    # Format: "yellow:BIN_A,green:BIN_B,red:BIN_C"
    class_to_bin: Dict[str, str] = field(
        default_factory=lambda: _parse_class_to_bin(os.environ.get("CLASS_TO_BIN", ""))
    )

    # ── Workspace geometry (for collision-avoidance documentation) ───
    # Z height (mm in robot base frame) of the gooseneck horizontal beam.
    # TRANSIT_HEIGHT_MM should sit comfortably below this. Measured once
    # with a tape measure during bring-up.
    gooseneck_beam_z_mm: float = field(
        default_factory=lambda: float(os.environ.get("GOOSENECK_BEAM_Z_MM", "350"))
    )

    # ── Operational modes ─────────────────────────────────────────────
    mock_hardware: bool = field(
        default_factory=lambda: os.environ.get("MOCK_HARDWARE", "false").lower() == "true"
    )
    mock_foundry: bool = field(
        default_factory=lambda: os.environ.get("MOCK_FOUNDRY", "false").lower() == "true"
    )
