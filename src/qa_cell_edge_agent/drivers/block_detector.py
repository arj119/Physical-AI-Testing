"""Color-based block detection for the pick zone.

Detects colored objects on a white surface using HSV color thresholding
and contour analysis. Classifies blocks by their dominant color:
  - Green → widget_good
  - Red   → widget_defect
  - Other → widget_unknown

No ML model required — works immediately with colored blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class BlockDetection:
    """Result of a single block detection."""

    detected_class: str          # widget_good, widget_defect, widget_unknown
    confidence: float            # 1.0 for color match, 0.5 for unknown
    bounding_box: List[float]    # [x, y, w, h] in pixels
    dominant_color: str          # green, red, blue, yellow, etc.
    contour_area: float          # area in pixels


# HSV ranges for color classification
# (hue_low, hue_high, name, widget_class)
_COLOR_MAP = [
    (35, 85, "green", "widget_good"),
    (0, 10, "red-low", "widget_defect"),
    (170, 180, "red-high", "widget_defect"),
    (100, 130, "blue", "widget_unknown"),
    (20, 35, "yellow", "widget_unknown"),
    (130, 170, "purple", "widget_unknown"),
    (10, 20, "orange", "widget_unknown"),
]


class BlockDetector:
    """Detects colored blocks on a white surface."""

    def __init__(
        self,
        min_area: int = 500,
        sat_min: int = 40,
        val_min: int = 50,
        val_max: int = 255,
    ) -> None:
        """
        Parameters
        ----------
        min_area : int
            Minimum contour area in pixels to count as a block.
        sat_min : int
            Minimum saturation to be "colored" (filters out white/gray).
        val_min : int
            Minimum value to exclude very dark areas.
        val_max : int
            Maximum value.
        """
        self._min_area = min_area
        self._sat_min = sat_min
        self._val_min = val_min
        self._val_max = val_max

    def detect(
        self,
        frame: np.ndarray,
        zone_mask: Optional[np.ndarray] = None,
    ) -> Optional[BlockDetection]:
        """Detect the largest colored block in the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from camera.
        zone_mask : np.ndarray, optional
            Binary mask of the pick zone. If provided, only searches within it.

        Returns None if no block found.
        """
        if cv2 is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find colored pixels (not white, not black)
        color_mask = cv2.inRange(
            hsv,
            (0, self._sat_min, self._val_min),
            (180, 255, self._val_max),
        )

        # Apply zone mask if provided
        if zone_mask is not None:
            color_mask = cv2.bitwise_and(color_mask, zone_mask)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filter by area and pick the largest
        valid = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= self._min_area]
        if not valid:
            return None

        best_contour, best_area = max(valid, key=lambda x: x[1])
        x, y, w, h = cv2.boundingRect(best_contour)

        # Classify by dominant color in the bounding box
        roi_hsv = hsv[y:y+h, x:x+w]
        roi_mask = color_mask[y:y+h, x:x+w]
        color_name, widget_class, confidence = self._classify_color(roi_hsv, roi_mask)

        return BlockDetection(
            detected_class=widget_class,
            confidence=confidence,
            bounding_box=[float(x), float(y), float(w), float(h)],
            dominant_color=color_name,
            contour_area=float(best_area),
        )

    def _classify_color(
        self,
        roi_hsv: np.ndarray,
        roi_mask: np.ndarray,
    ) -> tuple:
        """Classify the dominant color in a region. Returns (color_name, widget_class, confidence)."""
        if roi_mask.sum() == 0:
            return "unknown", "widget_unknown", 0.5

        # Get hue values of the colored pixels only
        hues = roi_hsv[:, :, 0][roi_mask > 0]
        if len(hues) == 0:
            return "unknown", "widget_unknown", 0.5

        median_hue = float(np.median(hues))

        # Match against color ranges
        for hue_low, hue_high, color_name, widget_class in _COLOR_MAP:
            if hue_low <= median_hue <= hue_high:
                # Confidence based on how many pixels match this range
                matching = np.sum((hues >= hue_low) & (hues <= hue_high))
                confidence = min(1.0, matching / len(hues) + 0.3)
                return color_name, widget_class, round(confidence, 2)

        return "unknown", "widget_unknown", 0.5
