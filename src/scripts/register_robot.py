#!/usr/bin/env python3
"""Register a robot in Foundry.

Run once during initial setup before starting the edge agent or seeding data.
Sensors are created automatically by the Pipeline Builder pipeline in Foundry
when telemetry data flows through the sensor-telemetry stream.

Usage:
    python scripts/register_robot.py                      # uses defaults from .env
    python scripts/register_robot.py --robot-id qa-cell-02 --name "QA Cell Robot 02"
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.config.foundry import FoundryClients


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Register a robot in Foundry")
    parser.add_argument("--robot-id", help="Robot ID (default: from .env ROBOT_ID)")
    parser.add_argument("--name", help="Robot display name (default: from .env ROBOT_NAME)")
    args = parser.parse_args()

    settings = Settings()
    robot_id = args.robot_id or settings.robot_id
    robot_name = args.name or settings.robot_name

    clients = FoundryClients(settings=settings)

    print(f"\n  Registering robot: {robot_id} ({robot_name})")
    print("=" * 60)

    try:
        clients.client.ontology.actions.create_robot(
            robot_id=robot_id,
            name=robot_name,
            status="IDLE",
            current_model_version="none",
            grip_tolerance=settings.grip_tolerance,
        )
        print(f"  Robot created: {robot_id}")
    except Exception as exc:
        if "ObjectAlreadyExists" in str(exc) or "CONFLICT" in str(exc):
            print(f"  Robot already exists: {robot_id} (skipping)")
        else:
            print(f"  Failed to create robot: {exc}")
            sys.exit(1)

    print("=" * 60)
    print(f"  Registration complete for {robot_id}")
    print(f"  Sensors will be created automatically when telemetry data flows.")
    print(f"\n  Next steps:")
    print(f"    python scripts/test_connection.py          # verify connectivity")
    print(f"    python scripts/test_connection.py --seed    # seed demo data")
    print(f"    python -m qa_cell_edge_agent.main           # start the agent")
    print()


if __name__ == "__main__":
    main()
