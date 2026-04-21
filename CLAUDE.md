# CLAUDE.md

## Project

QA Cell Edge Agent — Jetson Nano 2GB edge agent for a myCobot 280 AI Kit 2023 robotic sorting cell. Pushes sensor data to Palantir Foundry, runs YOLOv5 inference, fuses vision + gripper feedback, and sorts parts into bins autonomously.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e src/
cp .env.example .env  # fill in FOUNDRY_URL, CLIENT_ID, CLIENT_SECRET
```

The `physical-ai-qa-cell-sdk` package comes from a private Foundry artifact repository, not public PyPI.

## Running

```bash
# Mock mode (no hardware, no Foundry)
python -m qa_cell_edge_agent.main --mock

# Mock hardware only (real Foundry)
python -m qa_cell_edge_agent.main --mock-hardware

# Full live
python -m qa_cell_edge_agent.main
```

All commands run from `src/` or after `pip install -e src/`.

## Tests

```bash
pytest src/test/ -v
```

- `test_fusion_engine.py` — sensor fusion decision logic (14 tests, no deps)
- `test_model_hot_swap.py` — model reload during runtime (12 tests, needs `models/yolov5n.onnx` for ONNX tests — run `python scripts/download_model.py` first, otherwise those tests are skipped)

## Architecture

Three multiprocessing processes managed by `main.py`:

1. **sensor_push** — camera frames → Foundry vision stream (via v2 high-scale streams API)
2. **defect_detection** — gripper read + ONNX/TensorRT inference + fusion → arm pick-and-place + OSDK actions
3. **model_upgrade** — polls ModelRegistry → downloads via media API → hot-swaps model

IPC: `multiprocessing.Queue` (sensor→detection), `multiprocessing.Event` (upgrade→detection reload signal).

All serial I/O (myCobot arm + gripper) lives in Process 2 via a shared `MyCobot280` singleton (`drivers/connection.py`).

## Foundry Integration

Uses `physical_ai_qa_cell_sdk` (generated typed OSDK). All actions and object queries go through the SDK — never raw HTTP for ontology operations. Stream push uses raw `requests` with a dedicated `api:use-streams-write` scoped token (separate from the SDK's OSDK token).

**Key pattern:** SDK filter uses Python `==` operator, not `.eq()`:
```python
from physical_ai_qa_cell_sdk.ontology.search import OperatorCommandObjectType
commands = (
    client.ontology.objects.OperatorCommand
    .where(OperatorCommandObjectType.robot_id == settings.robot_id)
    .where(OperatorCommandObjectType.status == "PENDING")
    .take(100)
)
```

**Media properties** (model_artifact_ref, captured_image_ref) require `AllowBetaFeatures` context:
```python
from foundry_sdk_runtime import AllowBetaFeatures
with AllowBetaFeatures():
    media_ref = client.ontology.media.upload_media(image_bytes, "file.jpg")
```

**Null optional params** use `Empty.value`:
```python
from foundry_sdk_runtime.types.null_types import Empty
client.ontology.actions.create_inspection_event(..., captured_image_ref=Empty.value)
```

## Key Files

| File | Purpose |
|------|---------|
| `config/foundry.py` | SDK client + stream push (separate auth tokens) |
| `config/settings.py` | All config from env vars, hardware auto-discovery |
| `drivers/connection.py` | Shared MyCobot280 serial singleton |
| `drivers/discovery.py` | Auto-detect serial port (CP210x VID:PID) and camera |
| `drivers/arm.py` | Waypoint + Cartesian arm control, pick-and-place |
| `drivers/transforms.py` | Camera-to-robot coordinate transforms |
| `fusion/engine.py` | PASS/FAIL/REVIEW decision logic |
| `models/inference.py` | ONNX/TensorRT/mock inference with hot-swap |
| `processes/defect_detection.py` | Main loop: inference → fusion → arm → OSDK |
| `processes/model_upgrade.py` | Poll registry → download → convert → signal reload |

## Conventions

- Robot hardware: `pymycobot` `MyCobot280` class (not deprecated `MyCobot`)
- Action params: snake_case in Python, kebab-case over the wire. Object PK references use the link name (`robot`, `command`), not the property name (`robot_id`, `command_id`)
- `bounding_box` is `list[float]` (`[x, y, w, h]`), not a JSON string
- Fusion thresholds clamped to [0, 1]
- Model versions compared with `packaging.Version`, not string comparison
- TensorRT workspace: 256 MB (Jetson Nano 2GB constraint, see `config/jetson.py`)

## Scripts

| Script | When to use |
|--------|------------|
| `verify_hardware.py` | Before calibration — checks serial port + camera are working |
| `register_robot.py` | Once per robot, before anything else |
| `test_connection.py` | Verify Foundry connectivity. `--seed --count N` for demo data |
| `calibrate_arm.py` | After verify_hardware — records bin waypoints |
| `calibrate_camera.py` | For vision-guided picking (optional) |
| `download_model.py` | Fetch YOLOv5n ONNX for local testing |

## Deployment (Jetson)

```bash
sudo usermod -aG dialout jetson    # serial port access (then logout/login)
sudo cp -r . /opt/qa-cell-edge-agent
sudo chown -R jetson:jetson /opt/qa-cell-edge-agent
```

The `physical-ai-qa-cell-sdk` package needs `--extra-index-url` from your Foundry stack.
