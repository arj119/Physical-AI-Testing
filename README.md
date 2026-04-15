# qa-cell-edge-agent

Jetson Nano edge agent for the Physical AI QA Cell. Runs on-device vision inference,
fuses it with gripper feedback, sorts parts autonomously, and reports everything to
Palantir Foundry.

**Hardware:** myCobot 280 (6-DOF) with adaptive gripper, USB camera, NVIDIA Jetson Nano  
**SDK:** pymycobot >=3.6.0 (`MyCobot280` class)

## Architecture

Three long-running processes managed by a single orchestrator (`main.py`):

| Process | Module | Purpose |
|---------|--------|---------|
| **1. Sensor Push** | `processes/sensor_push.py` | Camera capture → Foundry vision stream |
| **2. Defect Detection** | `processes/defect_detection.py` | Gripper read + YOLOv5 inference + sensor fusion + arm control → OSDK actions |
| **3. Model Upgrade** | `processes/model_upgrade.py` | Polls ModelRegistry → downloads + hot-swaps model |
| **Test Connection** | `scripts/test_connection.py` | Connectivity verification + seed data |
| **Calibrate Arm** | `scripts/calibrate_arm.py` | Interactive waypoint calibration → `waypoints.json` |

```
sensor_push ──Queue──► defect_detection ──OSDK──► Foundry
  (camera)               (arm + gripper       ▲
                          + inference)         │
                    model_upgrade ──Event──────┘ (reload signal)
```

All serial I/O (arm + gripper) is consolidated in Process 2 via a shared
`MyCobot280` connection singleton (`drivers/connection.py`) to avoid serial
port conflicts.

**Hardware auto-discovery:** The myCobot serial port (CP210x/CH340 USB VID:PID)
and camera device index are auto-detected at startup. Set `MYCOBOT_PORT` or
`CAMERA_DEVICE_INDEX` in `.env` only if you need to override discovery.

## Quick Start (Development / Mock Mode)

```bash
# 1. Clone from Foundry
git clone <foundry-git-url> qa-cell-edge-agent
cd qa-cell-edge-agent

# 2. Create virtualenv
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your Foundry URL + client credentials

# 5. (Optional) Download a test model for real ONNX inference
python scripts/download_model.py

# 6. Test connectivity
python scripts/test_connection.py

# 6. Seed demo data (optional)
python scripts/test_connection.py --seed --count 50

# 7. Run all processes in mock mode (no hardware needed)
python -m qa_cell_edge_agent.main --mock
```

## Quick Start (Production / Jetson)

```bash
# 1. Clone + install (same as above, but on Jetson)
# 2. Configure .env with real hardware settings (MOCK_HARDWARE=false)

# 3. Calibrate arm waypoints (interactive — move arm to each position)
python scripts/calibrate_arm.py
# Saves to src/qa_cell_edge_agent/drivers/waypoints.json
# Arm driver loads these automatically on next startup

# 4. Install systemd service
sudo cp systemd/qa-cell-edge-agent.service systemd/qa-cell-edge-agent.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qa-cell-edge-agent.target
sudo systemctl start qa-cell-edge-agent.target

# 5. Monitor
journalctl -u qa-cell-edge-agent -f
```

## Project Structure

```
src/
├── qa_cell_edge_agent/
│   ├── config/          # Settings + Foundry client setup
│   ├── drivers/
│   │   ├── connection.py  # Shared MyCobot280 serial singleton
│   │   ├── arm.py         # Waypoint-based arm control + pick-and-place
│   │   ├── gripper.py     # Gripper read/close/open (get_gripper_value API)
│   │   └── camera.py      # USB camera capture + thumbnails
│   ├── fusion/          # Sensor fusion decision engine
│   ├── models/          # YOLOv5/TensorRT inference wrapper
│   ├── processes/       # The 3 long-running processes
│   └── main.py          # Process orchestrator (manages all 3 via multiprocessing)
├── scripts/             # Utility scripts (test_connection, calibrate_arm)
└── test/                # Unit tests
systemd/                 # Single systemd service for Jetson deployment
```

## Sensor Fusion Logic

The fusion engine combines vision confidence with gripper load:

| Vision | Grip Load | Decision | Reason |
|--------|-----------|----------|--------|
| class=good, conf ≥ 0.75 | load ≤ 0.65 | **PASS** | Both sensors agree: good part |
| class=defect OR conf < 0.75 | load > 0.65 | **FAIL** | Both sensors agree: bad part |
| Sensors disagree | — | **REVIEW** | Human review required |
| Grip data unavailable | — | **REVIEW** | Degraded mode: grip sensor missing |

Thresholds are configurable via `.env` and can be updated at runtime via the
`UPDATE_TOLERANCE` operator command from the dashboard.

## Pick-and-Place Sequence

```
HOME → PICK (fixed or dynamic) → close gripper → BIN (PASS/FAIL/REVIEW) → release gripper → HOME
```

The arm always returns to HOME before moving to PICK to ensure a safe,
predictable path. Bin selection is driven by the fusion decision.

### Vision-Guided Picking (non-identical placement)

When camera calibration is available (`drivers/camera_calibration.json`), the
arm computes a dynamic pick position from the bounding box detected by the
vision model, converting pixel coordinates to robot Cartesian coordinates via
either a homography (planar) or full hand-eye transform.

If the computed position is outside the 280mm workspace, the arm falls back to
the fixed PICK waypoint and logs a warning.

**Calibration:**

```bash
# Homography (recommended for flat surfaces — simpler setup)
python scripts/calibrate_camera.py --method homography --points 6

# Full hand-eye (for 3D picking — requires ArUco marker on end-effector)
python scripts/calibrate_camera.py --method handeye --points 12
```

## Testing

```bash
# Run unit tests
pytest src/test/ -v
```

## Foundry Resources

- **Streams:** vision-readings, grip-readings
- **Actions used:** CreateInspectionEvent, UpdateRobotStatus, AcknowledgeCommand
- **Actions polled:** OperatorCommand (PAUSE/RESUME/E_STOP/UPDATE_TOLERANCE)
- **Model source:** ModelRegistry object type
