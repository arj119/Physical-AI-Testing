"""Process orchestrator — starts and monitors all three edge agent processes.

Usage:
    python -m qa_cell_edge_agent.main
    python -m qa_cell_edge_agent.main --mock          # mock hardware + Foundry
    python -m qa_cell_edge_agent.main --mock-hardware  # mock hardware only
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from multiprocessing import Event, Process, Queue

from dotenv import load_dotenv

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.processes.sensor_push import run_sensor_push
from qa_cell_edge_agent.processes.defect_detection import run_defect_detection
from qa_cell_edge_agent.processes.model_upgrade import run_model_upgrade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("qa-cell")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="QA Cell Edge Agent")
    parser.add_argument("--mock", action="store_true", help="Mock both hardware and Foundry")
    parser.add_argument("--mock-hardware", action="store_true", help="Mock hardware only")
    parser.add_argument("--mock-foundry", action="store_true", help="Mock Foundry only")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_HARDWARE"] = "true"
        os.environ["MOCK_FOUNDRY"] = "true"
    if args.mock_hardware:
        os.environ["MOCK_HARDWARE"] = "true"
    if args.mock_foundry:
        os.environ["MOCK_FOUNDRY"] = "true"

    settings = Settings()

    logger.info("=" * 60)
    logger.info("QA Cell Edge Agent starting")
    logger.info("  Robot ID:       %s", settings.robot_id)
    logger.info("  Mock hardware:  %s", settings.mock_hardware)
    logger.info("  Mock Foundry:   %s", settings.mock_foundry)
    logger.info("  myCobot port:   %s @ %d", settings.mycobot_port, settings.mycobot_baud)
    logger.info("  Camera index:   %d", settings.camera_device_index)
    logger.info("  Model path:     %s", settings.model_path)
    logger.info("  Capture rate:   %.1f Hz", 1.0 / settings.capture_interval_sec)
    logger.info("=" * 60)

    # Shared IPC primitives
    sensor_queue: Queue = Queue(maxsize=10)
    model_reload_event = Event()

    # Start processes
    processes = {
        "sensor-push": Process(
            target=run_sensor_push,
            args=(sensor_queue, settings),
            name="sensor-push",
            daemon=True,
        ),
        "defect-detection": Process(
            target=run_defect_detection,
            args=(sensor_queue, model_reload_event, settings),
            name="defect-detection",
            daemon=True,
        ),
        "model-upgrade": Process(
            target=run_model_upgrade,
            args=(model_reload_event, settings),
            name="model-upgrade",
            daemon=True,
        ),
    }

    for name, proc in processes.items():
        proc.start()
        logger.info("Started %s (PID %d)", name, proc.pid)

    # Graceful shutdown on SIGINT / SIGTERM
    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("Received signal %d — shutting down", signum)
        shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Monitor loop — restart crashed processes
    while not shutdown:
        for name, proc in processes.items():
            if not proc.is_alive():
                logger.error("%s (PID %d) died — restarting", name, proc.pid)
                # Re-create and start
                if name == "sensor-push":
                    proc = Process(
                        target=run_sensor_push,
                        args=(sensor_queue, settings),
                        name=name,
                        daemon=True,
                    )
                elif name == "defect-detection":
                    proc = Process(
                        target=run_defect_detection,
                        args=(sensor_queue, model_reload_event, settings),
                        name=name,
                        daemon=True,
                    )
                elif name == "model-upgrade":
                    proc = Process(
                        target=run_model_upgrade,
                        args=(model_reload_event, settings),
                        name=name,
                        daemon=True,
                    )
                proc.start()
                processes[name] = proc
                logger.info("Restarted %s (PID %d)", name, proc.pid)
        time.sleep(5)

    # Shutdown
    logger.info("Terminating child processes...")
    for name, proc in processes.items():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            logger.warning("Force-killing %s", name)
            proc.kill()

    logger.info("QA Cell Edge Agent stopped.")


if __name__ == "__main__":
    main()
