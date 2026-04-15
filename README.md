# qa-cell-edge-agent

Jetson Nano edge agent for the Physical AI QA Cell. Runs on-device vision inference,
fuses it with gripper feedback, sorts parts autonomously, and reports everything to
Palantir Foundry.

## Architecture

Three long-running processes + one utility script:

| Process | Module | Purpose |
|---------|--------|---------|
| **1. Sensor Push** | `processes/sensor_push.py` | Camera capture + gripper read → Foundry Streams |
| **2. Defect Detection** | `processes/defect_detection.py` | YOLOv5 inference + sensor fusion → OSDK actions |
| **3. Model Upgrade** | `processes/model_upgrade.py` | Polls ModelRegistry → downloads + hot-swaps model |
| **Test Connection** | `scripts/test_connection.py` | Connectivity verification + seed data |

```
sensor_push ──Queue──► defect_detection ──OSDK──► Foundry
                              ▲
model_upgrade ──Event────────┘ (reload signal)
```

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

# 5. Test connectivity
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

# 3. Calibrate arm waypoints
python scripts/calibrate_arm.py

# 4. Install systemd services
sudo cp systemd/*.service systemd/*.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qa-cell-edge-agent.target
sudo systemctl start qa-cell-edge-agent.target

# 5. Monitor
journalctl -u qa-sensor-push -f
journalctl -u qa-defect-detection -f
journalctl -u qa-model-upgrade -f
```

## Project Structure

```
src/
├── qa_cell_edge_agent/
│   ├── config/          # Settings + Foundry client setup
│   ├── drivers/         # Camera, gripper, arm hardware interfaces
│   ├── fusion/          # Sensor fusion decision engine
│   ├── models/          # YOLOv5/TensorRT inference wrapper
│   ├── processes/       # The 3 long-running processes
│   └── main.py          # Process orchestrator
├── scripts/             # Utility scripts (test_connection, calibrate_arm)
└── test/                # Unit tests
systemd/                 # systemd service files for Jetson deployment
```

## Sensor Fusion Logic

The fusion engine combines vision confidence with gripper load:

| Vision | Grip Load | Decision | Reason |
|--------|-----------|----------|--------|
| class=good, conf ≥ 0.75 | load ≤ 0.65 | **PASS** | Both sensors agree: good part |
| class=defect OR conf < 0.75 | load > 0.65 | **FAIL** | Both sensors agree: bad part |
| Sensors disagree | — | **REVIEW** | Human review required |

Thresholds are configurable via `.env` and can be updated at runtime via the
`UPDATE_TOLERANCE` operator command from the dashboard.

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
