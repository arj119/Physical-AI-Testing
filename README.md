# qa-cell-edge-agent

Jetson Nano edge agent for the Physical AI QA Cell. Runs on-device vision
inference, fuses it with gripper feedback, sorts parts autonomously, and
reports everything to Palantir Foundry.

**Hardware:** myCobot 280 AI Kit 2023 (6-DOF arm + adaptive gripper + overhead
USB camera + sorting bins), NVIDIA Jetson Nano 2GB  
**SDK:** pymycobot >=3.6.0 (`MyCobot280` class)

## Quick Start

### 1. Install

```bash
git clone <foundry-git-url> qa-cell-edge-agent
cd qa-cell-edge-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Foundry

```bash
cp .env.example .env
# Edit .env — set FOUNDRY_URL, CLIENT_ID, CLIENT_SECRET, stream RIDs
```

Hardware (serial port, camera) is **auto-discovered** — no need to set
`MYCOBOT_PORT` or `CAMERA_DEVICE_INDEX` unless auto-detection picks the
wrong device.

### 3. Run in Mock Mode (no hardware needed)

```bash
# Full mock — synthetic sensor data, no Foundry calls
python -m qa_cell_edge_agent.main --mock
```

You should see all three processes start and the defect detection loop
running with mock inference, fusion decisions, and pick-and-place logging.

### 4. Run with a Real Model (still no hardware)

```bash
# Download YOLOv5-nano ONNX (pretrained on COCO, ~4 MB)
python scripts/download_model.py

# Run with real ONNX inference but mocked hardware
python -m qa_cell_edge_agent.main --mock-hardware
```

COCO detections map to `widget_unknown` — replace with a fine-tuned model
for `widget_good` / `widget_defect` classes in production.

### 5. Set Up the Physical Robot

Plug in the myCobot 280 AI Kit via USB. The agent auto-detects the serial
port (CP210x / CH340 USB-serial chip) and camera.

```bash
# Verify Foundry connectivity + seed demo data
python scripts/test_connection.py --seed

# Calibrate arm waypoints for your AI Kit bin layout
python scripts/calibrate_arm.py

# (Optional) Calibrate camera-to-robot transform for vision-guided picking
python scripts/calibrate_camera.py --method homography --points 6

# Run with real hardware, mock Foundry (for testing)
python -m qa_cell_edge_agent.main --mock-foundry

# Run fully live
python -m qa_cell_edge_agent.main
```

### 6. Deploy as a Service (Jetson)

```bash
sudo cp systemd/qa-cell-edge-agent.service systemd/qa-cell-edge-agent.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qa-cell-edge-agent.target
sudo systemctl start qa-cell-edge-agent.target

# Monitor
journalctl -u qa-cell-edge-agent -f
```

## How It Works

### Physical AI Loop

```
                    ┌─────────────────────────────────────┐
                    │          Palantir Foundry            │
                    │                                     │
                    │  Streams ← raw sensor data          │
                    │  OSDK    ← inspection events        │
                    │  Dashboard → operator commands       │
                    │  ModelRegistry → retrained models    │
                    └──────┬──────────────┬───────────────┘
                           │              │
                    push data up    pull commands +
                                    models down
                           │              │
            ┌──────────────▼──────────────▼───────────────┐
            │          Jetson Nano Edge Agent              │
            │                                             │
            │  sensor_push ──Queue──► defect_detection    │
            │    (camera)              (arm + gripper     │
            │                          + inference)       │
            │                               ▲             │
            │              model_upgrade ───┘ (reload)    │
            └─────────────────────────────────────────────┘
```

1. **Capture** — overhead camera frames + gripper servo load
2. **Infer** — YOLOv5 on-device (ONNX dev / TensorRT prod)
3. **Fuse** — combine vision confidence + grip load → PASS / FAIL / REVIEW
4. **Sort** — arm picks part and places in the correct bin
5. **Report** — push results to Foundry via OSDK
6. **Defer** — uncertain decisions (REVIEW) go to a human operator in Foundry
7. **Learn** — operator labels feed model retraining; Process 3 pulls updated
   models and hot-swaps them on the Jetson

### Architecture

Three long-running processes managed by `main.py`:

| Process | Module | Purpose |
|---------|--------|---------|
| **1. Sensor Push** | `processes/sensor_push.py` | Camera capture → Foundry vision stream |
| **2. Defect Detection** | `processes/defect_detection.py` | Gripper read + YOLOv5 inference + sensor fusion + arm control → OSDK actions |
| **3. Model Upgrade** | `processes/model_upgrade.py` | Polls ModelRegistry → downloads + hot-swaps model |

All serial I/O (arm + gripper) is consolidated in Process 2 via a shared
`MyCobot280` connection singleton (`drivers/connection.py`).

### Sensor Fusion

| Vision | Grip Load | Decision | Reason |
|--------|-----------|----------|--------|
| class=good, conf ≥ 0.75 | load ≤ 0.65 | **PASS** | Both sensors agree |
| class=defect OR conf < 0.75 | load > 0.65 | **FAIL** | Both sensors agree |
| Sensors disagree | — | **REVIEW** | Human review required |
| Grip data unavailable | — | **REVIEW** | Degraded mode |

Thresholds are configurable via `.env` and updatable at runtime via the
`UPDATE_TOLERANCE` operator command from the Foundry dashboard.

### Pick-and-Place Sequence

```
HOME → PICK (fixed or dynamic) → close gripper → BIN → release gripper → HOME
```

When camera calibration is available (`drivers/camera_calibration.json`), the
arm computes a dynamic pick position from the detected bounding box. If the
target is outside the 280mm workspace, it falls back to the fixed PICK waypoint.

### Model Pipeline (ONNX → TensorRT)

```
Train in Foundry → export ONNX → publish to ModelRegistry
    → Process 3 downloads → trtexec converts to TensorRT (FP16, 256MB workspace)
    → atomic swap → Process 2 hot-reloads
```

TensorRT workspace is set to 256 MB for the Jetson Nano 2GB (configured in
`config/jetson.py`).

## Project Structure

```
src/
├── qa_cell_edge_agent/
│   ├── config/
│   │   ├── settings.py      # Centralised config from env vars + auto-discovery
│   │   ├── foundry.py       # OAuth2 + Foundry stream/OSDK clients
│   │   └── jetson.py        # Jetson Nano 2GB memory constraints
│   ├── drivers/
│   │   ├── discovery.py     # Auto-detect serial port + camera
│   │   ├── connection.py    # Shared MyCobot280 serial singleton
│   │   ├── arm.py           # Waypoint + Cartesian arm control
│   │   ├── gripper.py       # Gripper read/close/open
│   │   ├── camera.py        # USB camera capture + thumbnails
│   │   └── transforms.py    # Camera-to-robot coordinate transforms
│   ├── fusion/
│   │   └── engine.py        # Sensor fusion decision engine
│   ├── models/
│   │   └── inference.py     # YOLOv5 inference (ONNX + TensorRT + mock)
│   ├── processes/
│   │   ├── sensor_push.py
│   │   ├── defect_detection.py
│   │   └── model_upgrade.py
│   └── main.py              # Process orchestrator
├── scripts/
│   ├── test_connection.py   # Foundry connectivity check + seed data
│   ├── calibrate_arm.py     # Record arm waypoints for bin layout
│   ├── calibrate_camera.py  # Camera-to-robot calibration (homography / hand-eye)
│   └── download_model.py    # Fetch YOLOv5n ONNX for local testing
└── test/
    └── test_fusion_engine.py
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/test_connection.py` | Verify Foundry OAuth2, streams, OSDK. `--seed` generates demo data |
| `scripts/calibrate_arm.py` | Interactive: move arm to each waypoint, records joint angles to `waypoints.json` |
| `scripts/calibrate_camera.py` | Camera-to-robot calibration. `--method homography` (flat surface) or `--method handeye` (ArUco) |
| `scripts/download_model.py` | Downloads `yolov5n.onnx` (~4 MB) for local inference testing |

## Configuration

All settings load from environment variables (via `.env`). Hardware ports
are auto-discovered when not explicitly set.

| Variable | Default | Notes |
|----------|---------|-------|
| `FOUNDRY_URL` | `https://localhost` | Foundry stack URL |
| `CLIENT_ID` / `CLIENT_SECRET` | — | OAuth2 credentials |
| `MYCOBOT_PORT` | auto-detected | Override with e.g. `/dev/ttyUSB0` |
| `MYCOBOT_BAUD` | `115200` | Fixed for myCobot 280 M5Stack |
| `CAMERA_DEVICE_INDEX` | auto-detected | Override with e.g. `0` |
| `MOCK_HARDWARE` | `false` | Skip all hardware I/O |
| `MOCK_FOUNDRY` | `false` | Skip all Foundry API calls |
| `MODEL_PATH` | `./models/yolov5n.onnx` | Path to ONNX or TensorRT engine |
| `CONFIDENCE_THRESHOLD` | `0.75` | Vision confidence floor |
| `GRIP_TOLERANCE` | `0.65` | Gripper load ceiling for "normal" |
| `CAPTURE_INTERVAL_SEC` | `1.0` | Camera capture rate (seconds) |

## Testing

```bash
pytest src/test/ -v
```

## Foundry Resources

- **Streams:** vision-readings, grip-readings
- **OSDK Actions:** CreateInspectionEvent, UpdateRobotStatus, AcknowledgeCommand
- **Operator Commands:** PAUSE, RESUME, E_STOP, UPDATE_TOLERANCE
- **Model source:** ModelRegistry object type
