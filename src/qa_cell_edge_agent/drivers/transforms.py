"""Camera-to-robot coordinate transforms for vision-guided picking.

Supports two calibration modes:
  1. **Homography** (planar) — 2D pixel→robot mapping for a fixed pick height.
     Simpler to calibrate, sufficient when all items sit on a flat surface.
  2. **Full hand-eye** — 4×4 camera-to-base transform for 3D picking.
     Required when items may be at different heights.

Calibration data is loaded from ``camera_calibration.json`` produced by
``scripts/calibrate_camera.py``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "camera_calibration.json")

# myCobot 280 workspace limits (mm from base centre)
MAX_REACH_MM = 280.0
Z_MIN_MM = -70.0
Z_MAX_MM = 412.0


@dataclass
class PickTarget:
    """A computed pick position in robot coordinates."""

    coords: List[float]          # [x, y, z, rx, ry, rz] in mm/degrees
    pixel_centre: Tuple[int, int]  # (cx, cy) in the source image
    reachable: bool              # whether the target is within the workspace
    distance_from_base: float    # horizontal distance in mm


class CameraTransform:
    """Converts pixel coordinates to robot base-frame coordinates."""

    def __init__(self, calibration_path: Optional[str] = None) -> None:
        self._path = calibration_path or CALIBRATION_FILE
        self._mode: Optional[str] = None  # "homography" or "hand_eye"

        # Homography mode
        self._H: Optional[np.ndarray] = None  # 3×3 homography matrix
        self._z_pick: float = 0.0             # fixed Z height for pick (mm)
        self._approach_angles: List[float] = [0.0, 0.0, 0.0]  # rx, ry, rz

        # Hand-eye mode
        self._T_cam_to_base: Optional[np.ndarray] = None  # 4×4 transform
        self._camera_matrix: Optional[np.ndarray] = None   # 3×3 intrinsics

        self._load()

    @property
    def is_calibrated(self) -> bool:
        return self._mode is not None

    def pixel_to_robot(
        self,
        cx: int,
        cy: int,
        z_override: Optional[float] = None,
    ) -> PickTarget:
        """Convert a pixel centre to a robot pick position.

        Parameters
        ----------
        cx, cy : int
            Bounding box centre in pixel coordinates.
        z_override : float, optional
            Override the calibrated Z pick height (mm).

        Returns
        -------
        PickTarget
            Robot coordinates with reachability status.
        """
        if not self.is_calibrated:
            logger.warning("Camera not calibrated — returning default PICK")
            return PickTarget(
                coords=[0, 0, 0, 0, 0, 0],
                pixel_centre=(cx, cy),
                reachable=False,
                distance_from_base=0.0,
            )

        if self._mode == "homography":
            return self._homography_transform(cx, cy, z_override)
        else:
            return self._hand_eye_transform(cx, cy, z_override)

    # ── Homography (planar) ───────────────────────────────────────────

    def _homography_transform(
        self, cx: int, cy: int, z_override: Optional[float]
    ) -> PickTarget:
        """Map pixel to robot XY via homography, use fixed Z."""
        pixel = np.array([cx, cy, 1.0], dtype=np.float64)
        robot_h = self._H @ pixel
        robot_h /= robot_h[2]  # normalize homogeneous coords

        x, y = float(robot_h[0]), float(robot_h[1])
        z = z_override if z_override is not None else self._z_pick
        rx, ry, rz = self._approach_angles

        dist = math.sqrt(x * x + y * y)
        reachable = dist <= MAX_REACH_MM and Z_MIN_MM <= z <= Z_MAX_MM

        if not reachable:
            logger.warning(
                "Target at pixel (%d, %d) → robot (%.1f, %.1f, %.1f) is "
                "outside workspace (dist=%.1f mm, max=%s mm)",
                cx, cy, x, y, z, dist, MAX_REACH_MM,
            )

        return PickTarget(
            coords=[round(x, 2), round(y, 2), round(z, 2), rx, ry, rz],
            pixel_centre=(cx, cy),
            reachable=reachable,
            distance_from_base=round(dist, 2),
        )

    # ── Hand-eye (full 3D) ────────────────────────────────────────────

    def _hand_eye_transform(
        self, cx: int, cy: int, z_override: Optional[float]
    ) -> PickTarget:
        """Project pixel ray through camera-to-base transform.

        Assumes the object is on a known Z-plane (pick surface).
        """
        z_pick = z_override if z_override is not None else self._z_pick
        K = self._camera_matrix
        T = self._T_cam_to_base

        # Pixel → normalised camera ray
        pixel_h = np.array([cx, cy, 1.0])
        ray_cam = np.linalg.inv(K) @ pixel_h
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        # Camera origin in base frame
        R = T[:3, :3]
        t = T[:3, 3]
        cam_origin = t
        ray_base = R @ ray_cam

        # Intersect ray with Z = z_pick plane
        if abs(ray_base[2]) < 1e-6:
            logger.warning("Ray parallel to pick plane — cannot intersect")
            return PickTarget(
                coords=[0, 0, 0, 0, 0, 0],
                pixel_centre=(cx, cy),
                reachable=False,
                distance_from_base=0.0,
            )

        t_param = (z_pick - cam_origin[2]) / ray_base[2]
        point = cam_origin + t_param * ray_base
        x, y, z = float(point[0]), float(point[1]), float(point[2])

        rx, ry, rz = self._approach_angles
        dist = math.sqrt(x * x + y * y)
        reachable = dist <= MAX_REACH_MM and Z_MIN_MM <= z <= Z_MAX_MM

        if not reachable:
            logger.warning(
                "Target at pixel (%d, %d) → robot (%.1f, %.1f, %.1f) "
                "outside workspace (dist=%.1f mm)",
                cx, cy, x, y, z, dist,
            )

        return PickTarget(
            coords=[round(x, 2), round(y, 2), round(z, 2), rx, ry, rz],
            pixel_centre=(cx, cy),
            reachable=reachable,
            distance_from_base=round(dist, 2),
        )

    # ── Calibration loading ───────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.isfile(self._path):
            logger.info("No camera calibration at %s — vision-guided pick disabled", self._path)
            return
        try:
            with open(self._path) as f:
                data = json.load(f)

            self._mode = data["mode"]  # "homography" or "hand_eye"
            self._z_pick = data.get("z_pick_mm", 0.0)
            self._approach_angles = data.get("approach_angles", [0.0, 0.0, 0.0])

            if self._mode == "homography":
                self._H = np.array(data["homography_matrix"], dtype=np.float64)
                logger.info("Loaded homography calibration from %s", self._path)
            elif self._mode == "hand_eye":
                self._T_cam_to_base = np.array(data["T_camera_to_base"], dtype=np.float64)
                self._camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
                logger.info("Loaded hand-eye calibration from %s", self._path)
            else:
                logger.error("Unknown calibration mode: %s", self._mode)
                self._mode = None
        except Exception as exc:
            logger.error("Failed to load camera calibration: %s", exc)
            self._mode = None
