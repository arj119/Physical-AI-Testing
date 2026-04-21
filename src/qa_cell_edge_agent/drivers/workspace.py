"""Workspace detection and change monitoring for the pick zone.

The pick zone is defined by 4 corners (drawn interactively or loaded
from ``workspace_zone.json``). Only the area inside that polygon is
monitored for new objects via background subtraction.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

ZONE_FILE = os.path.join(os.path.dirname(__file__), "workspace_zone.json")


class WorkspaceMonitor:
    """Monitors a user-defined pick zone for new objects."""

    def __init__(
        self,
        change_threshold: float = 5.0,
        min_change_area: float = 0.005,
    ) -> None:
        self._change_threshold = change_threshold
        self._min_change_area = min_change_area
        self._reference: Optional[np.ndarray] = None
        self._zone_mask: Optional[np.ndarray] = None
        self._zone_points: Optional[List[List[int]]] = None
        self._roi_bbox: Optional[Tuple[int, int, int, int]] = None
        self._load_zone()

    def _load_zone(self) -> None:
        """Load the pick zone polygon from workspace_zone.json."""
        if not os.path.isfile(ZONE_FILE):
            logger.info("No workspace zone defined at %s — run 'python scripts/define_zone.py'", ZONE_FILE)
            return
        try:
            with open(ZONE_FILE) as f:
                data = json.load(f)
            self._zone_points = data["points"]
            logger.info("Loaded workspace zone: %d corners", len(self._zone_points))
        except Exception as exc:
            logger.warning("Failed to load workspace zone: %s", exc)

    @property
    def is_configured(self) -> bool:
        return self._zone_points is not None

    @property
    def is_ready(self) -> bool:
        return self._reference is not None and self._zone_mask is not None

    @property
    def roi_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi_bbox

    @property
    def zone_points(self) -> Optional[List[List[int]]]:
        return self._zone_points

    def capture_reference(self, frame: np.ndarray) -> None:
        """Store a reference frame of the empty workspace."""
        if not self.is_configured:
            return

        # Build polygon mask from zone points
        h, w = frame.shape[:2]
        self._zone_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self._zone_points, dtype=np.int32)
        cv2.fillPoly(self._zone_mask, [pts], 255)

        # Compute bounding rect
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        self._roi_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

        # Store blurred grayscale reference
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._reference = cv2.GaussianBlur(gray, (21, 21), 0)

        zone_pct = (self._zone_mask > 0).sum() / self._zone_mask.size * 100
        logger.info(
            "Reference captured. Zone: (%d,%d)-(%d,%d), %.1f%% of frame",
            x_min, y_min, x_max, y_max, zone_pct,
        )

    def has_new_object(self, frame: np.ndarray) -> bool:
        """Check if something new appeared in the pick zone."""
        if not self.is_ready:
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        diff = cv2.absdiff(self._reference, gray)
        diff_masked = cv2.bitwise_and(diff, diff, mask=self._zone_mask)

        _, thresh = cv2.threshold(diff_masked, 30, 255, cv2.THRESH_BINARY)

        zone_pixels = (self._zone_mask > 0).sum()
        if zone_pixels == 0:
            return False

        changed_pixels = (thresh > 0).sum()
        change_ratio = changed_pixels / zone_pixels
        mean_diff = diff_masked[self._zone_mask > 0].mean()

        has_change = (
            mean_diff > self._change_threshold
            and change_ratio > self._min_change_area
        )

        if has_change:
            logger.debug(
                "Object detected: mean_diff=%.1f, change=%.1f%%",
                mean_diff, change_ratio * 100,
            )

        return has_change

    def mask_frame(self, frame: np.ndarray) -> np.ndarray:
        """Black out everything outside the pick zone."""
        if self._zone_mask is None:
            return frame
        masked = frame.copy()
        masked[self._zone_mask == 0] = 0
        return masked

    def draw_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw the pick zone polygon on a frame (for visualization)."""
        if self._zone_points is None:
            return frame
        display = frame.copy()
        pts = np.array(self._zone_points, dtype=np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return display
