"""Sensor inventory for the QA Cell robot.

Each entry defines a scalar sensor whose readings are pushed to the
sensor-telemetry stream as ``(seriesId, timestamp, value)`` rows.

The ``seriesId`` format is ``{metric}:{robot_id}`` — this must match
the ``seriesId`` property on the Sensor objects in the ontology so the
Time Series Pipeline (TSP) can fan out the stream into per-sensor history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SensorDef:
    """Definition of a single scalar sensor on the robot."""

    series_id_template: str  # e.g. "j1-temp:{robot_id}"
    sensor_id_template: str  # e.g. "j1-temp-{robot_id}"
    name: str                # human-readable
    sensor_type: str         # VISION, GRIP_LOAD, SERVO_TEMP, ...
    location: str            # overhead, gripper, base, shoulder, ...
    units: str               # %, °C, degrees, ...
    source: str              # where the value comes from at runtime


# Phase 1 — Core (derived from inference + gripper readings)
# Phase 2 — Servo temperatures (pymycobot: mc.get_servo_temps())
SENSOR_REGISTRY: List[SensorDef] = [
    # ── Phase 1: Core ────────────────────────────────────────────
    SensorDef("vision-confidence:{robot_id}", "cam-{robot_id}",
              "Camera Confidence", "VISION", "overhead", "%", "inference"),
    SensorDef("grip-load:{robot_id}", "grip-{robot_id}",
              "Gripper Load", "GRIP_LOAD", "gripper", "%", "gripper"),

    # ── Phase 2: Joint temperatures ──────────────────────────────
    SensorDef("j1-temp:{robot_id}", "j1-temp-{robot_id}",
              "J1 Base Temp", "SERVO_TEMP", "base", "°C", "joint_temp:0"),
    SensorDef("j2-temp:{robot_id}", "j2-temp-{robot_id}",
              "J2 Shoulder Temp", "SERVO_TEMP", "shoulder", "°C", "joint_temp:1"),
    SensorDef("j3-temp:{robot_id}", "j3-temp-{robot_id}",
              "J3 Elbow Temp", "SERVO_TEMP", "elbow", "°C", "joint_temp:2"),
    SensorDef("j4-temp:{robot_id}", "j4-temp-{robot_id}",
              "J4 Wrist-Pitch Temp", "SERVO_TEMP", "wrist-pitch", "°C", "joint_temp:3"),
    SensorDef("j5-temp:{robot_id}", "j5-temp-{robot_id}",
              "J5 Wrist-Roll Temp", "SERVO_TEMP", "wrist-roll", "°C", "joint_temp:4"),
    SensorDef("j6-temp:{robot_id}", "j6-temp-{robot_id}",
              "J6 Wrist-Yaw Temp", "SERVO_TEMP", "wrist-yaw", "°C", "joint_temp:5"),
]


def get_sensor_registry(robot_id: str) -> List[SensorDef]:
    """Return the sensor registry with ``{robot_id}`` templates resolved."""
    resolved = []
    for s in SENSOR_REGISTRY:
        resolved.append(SensorDef(
            series_id_template=s.series_id_template.format(robot_id=robot_id),
            sensor_id_template=s.sensor_id_template.format(robot_id=robot_id),
            name=s.name,
            sensor_type=s.sensor_type,
            location=s.location,
            units=s.units,
            source=s.source,
        ))
    return resolved
