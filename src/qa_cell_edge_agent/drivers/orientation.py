"""Yaw / orientation utilities for top-down picking of square parts.

Two helpers cover the two detection paths:

* ``minarearect_yaw`` — given a contour from a colour/HSV detector, returns
  yaw in degrees normalised to ``[-45, 45]`` for 90°-symmetric parts.
  Replaces the inline math in ``block_detector.py``.

* ``estimate_yaw_in_bbox`` — given a frame and an axis-aligned bounding box
  from a YOLO detector (which has no rotation head), runs Otsu thresholding
  inside the bbox, finds the largest contour, and returns yaw the same way.
  Lets the YOLO path align J6 with the cube even though the model itself
  is orientation-blind.

Both return an ``OrientedDetection`` (centre, size, yaw) so callers don't
need to plumb tuples around.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


@dataclass
class OrientedDetection:
    """Yaw + size + centre for a single oriented bounding box."""

    centre: Tuple[float, float]   # (cx, cy) in pixels
    size: Tuple[float, float]     # (w, h) of the rotated rect, pixels
    yaw_deg: float                # normalised to [-45, 45] for square parts


def normalize_square_yaw(angle_deg: float) -> float:
    """Normalise an angle to ``[-45, 45]`` under 90° rotational symmetry.

    A square cube viewed top-down looks identical at 0°, 90°, 180°, 270°,
    so any reported angle can be mapped into the half-quadrant ``(-45, 45]``
    without changing the grasp pose.
    """
    a = ((angle_deg + 45.0) % 90.0) - 45.0
    if a == -45.0:
        return 45.0
    return a


def minarearect_yaw(contour: "np.ndarray") -> Optional[OrientedDetection]:
    """Wrap ``cv2.minAreaRect`` and return a normalised ``OrientedDetection``.

    Returns ``None`` if OpenCV is unavailable or the contour is degenerate.
    """
    if cv2 is None or contour is None or len(contour) < 3:
        return None

    rot_rect = cv2.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), raw_angle = rot_rect

    if w <= 0 or h <= 0:
        return None

    yaw = normalize_square_yaw(float(raw_angle))
    return OrientedDetection(
        centre=(float(cx), float(cy)),
        size=(float(w), float(h)),
        yaw_deg=round(yaw, 1),
    )


def estimate_yaw_in_bbox(
    frame: "np.ndarray",
    bbox_xywh: List[float],
    pad_px: int = 4,
) -> Optional[OrientedDetection]:
    """Estimate cube yaw from a YOLO axis-aligned bbox via Otsu + minAreaRect.

    Crops a small ROI around the bbox (with ``pad_px`` extra pixels per side
    when in-frame), thresholds it with Otsu — the cube against the white
    Camera Zone gives a clean bimodal histogram — and runs ``minAreaRect``
    on the largest contour.

    Returns ``None`` if no usable contour is found, in which case the caller
    should fall back to ``yaw_deg=0`` (the existing behaviour).

    Parameters
    ----------
    frame : np.ndarray
        Full BGR camera frame.
    bbox_xywh : list[float]
        Axis-aligned bounding box ``[x, y, w, h]`` in pixels.
    pad_px : int
        Extra pixels around the bbox ROI (kept inside frame bounds).
    """
    if cv2 is None or frame is None or len(bbox_xywh) != 4:
        return None

    h_frame, w_frame = frame.shape[:2]
    x, y, w, h = (float(v) for v in bbox_xywh)
    if w <= 0 or h <= 0:
        return None

    x0 = max(0, int(x) - pad_px)
    y0 = max(0, int(y) - pad_px)
    x1 = min(w_frame, int(x + w) + pad_px)
    y1 = min(h_frame, int(y + h) + pad_px)

    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu both ways — the cube might be darker than the mat (yellow on
    # black) or lighter (red on black depending on lighting). Pick whichever
    # mask yields the larger central blob.
    _, mask_a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_b = cv2.bitwise_not(mask_a)

    best: Optional[OrientedDetection] = None
    best_area = 0.0
    for mask in (mask_a, mask_b):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < 0.10 * (x1 - x0) * (y1 - y0):
            continue  # contour too small relative to ROI
        det = minarearect_yaw(c)
        if det is None:
            continue
        if area > best_area:
            best_area = area
            # Translate centre back from ROI to full frame coords
            cx, cy = det.centre
            best = OrientedDetection(
                centre=(cx + x0, cy + y0),
                size=det.size,
                yaw_deg=det.yaw_deg,
            )

    return best
