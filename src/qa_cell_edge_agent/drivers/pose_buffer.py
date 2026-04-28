"""Multi-frame pose smoothing for orientation-aware picking.

A single detection's yaw can wobble by a few degrees frame-to-frame from
sensor noise — and around the ±45° wrap, naive averaging is catastrophic
(44° and -44° average to 0°, not ±45°). ``PoseBuffer`` collects a small
window of detections and returns:

* arithmetic median for X and Y (in robot mm),
* circular median for yaw (in the 90°-symmetry domain),
* a ``stable`` flag that is True only once the buffer is full and the
  spread is within configured tolerances.

The defect-detection process uses ``stable`` as a gate before commanding
a pick, and reuses the same buffer (in "quick" mode with a smaller N)
during the closed-loop retry redetect callback.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StablePose:
    """A smoothed pose over the buffer window."""

    x_mm: float
    y_mm: float
    yaw_deg: float       # in [-45, 45], 90° square symmetry
    n_samples: int
    spread_xy_mm: float  # max-min of ||(x,y) - median|| across samples
    spread_yaw_deg: float


def _circular_median_90(angles_deg) -> float:
    """Median of angles under 90° rotational symmetry, in [-45, 45].

    Maps each input angle to a unit vector at ``2θ * (180/90)`` rad (i.e.
    ``4θ`` rad), takes the mean direction, then halves it back. This is
    the standard circular-statistics trick for symmetry of order 2 in the
    90° domain. For square-symmetric parts it's effectively the median in
    the cyclic group of order 4.
    """
    if not angles_deg:
        return 0.0
    sin_sum = 0.0
    cos_sum = 0.0
    for a in angles_deg:
        # Each square has 4-fold symmetry → multiply by 4 to wrap to a full circle
        rad = math.radians(a) * 4.0
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)
    mean_rad = math.atan2(sin_sum, cos_sum) / 4.0
    deg = math.degrees(mean_rad)
    # Renormalise to (-45, 45]
    deg = ((deg + 45.0) % 90.0) - 45.0
    if deg == -45.0:
        return 45.0
    return deg


def _max_circular_spread_90(angles_deg, centre_deg: float) -> float:
    """Max angular distance (under 90° symmetry) of any sample from centre."""
    worst = 0.0
    for a in angles_deg:
        diff = abs(((a - centre_deg + 45.0) % 90.0) - 45.0)
        if diff > worst:
            worst = diff
    return worst


class PoseBuffer:
    """Sliding window of (x_mm, y_mm, yaw_deg, timestamp) samples."""

    def __init__(
        self,
        n: int = 5,
        xy_tol_mm: float = 3.0,
        yaw_tol_deg: float = 5.0,
        max_age_s: float = 1.5,
    ) -> None:
        """
        Parameters
        ----------
        n : int
            Window size. ``stable`` requires this many fresh samples.
        xy_tol_mm : float
            Maximum spread (mm) of XY samples around the median for the
            buffer to be considered stable.
        yaw_tol_deg : float
            Maximum circular spread (deg) of yaw samples around the
            circular median for stability.
        max_age_s : float
            Samples older than this are discarded on each ``add``.
        """
        self._n = n
        self._xy_tol = xy_tol_mm
        self._yaw_tol = yaw_tol_deg
        self._max_age = max_age_s
        self._buf: Deque[Tuple[float, float, float, float]] = deque(maxlen=n)

    def reset(self) -> None:
        self._buf.clear()

    def add(self, x_mm: float, y_mm: float, yaw_deg: float, t: Optional[float] = None) -> None:
        """Append a sample, trimming any that are too old."""
        ts = time.monotonic() if t is None else t
        self._buf.append((float(x_mm), float(y_mm), float(yaw_deg), ts))
        self._evict_stale(ts)

    def _evict_stale(self, now: float) -> None:
        cutoff = now - self._max_age
        while self._buf and self._buf[0][3] < cutoff:
            self._buf.popleft()

    def __len__(self) -> int:  # for tests / debug
        return len(self._buf)

    def median(self) -> Optional[StablePose]:
        """Median pose with stability metrics. Returns None if buffer empty."""
        if not self._buf:
            return None
        xs = sorted(s[0] for s in self._buf)
        ys = sorted(s[1] for s in self._buf)
        ang = [s[2] for s in self._buf]
        n = len(self._buf)

        med_x = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
        med_y = ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])
        med_yaw = _circular_median_90(ang)

        spread_xy = max(
            math.hypot(s[0] - med_x, s[1] - med_y) for s in self._buf
        )
        spread_yaw = _max_circular_spread_90(ang, med_yaw)

        return StablePose(
            x_mm=round(med_x, 2),
            y_mm=round(med_y, 2),
            yaw_deg=round(med_yaw, 1),
            n_samples=n,
            spread_xy_mm=round(spread_xy, 2),
            spread_yaw_deg=round(spread_yaw, 1),
        )

    @property
    def stable(self) -> bool:
        """True when the buffer is full AND within configured spreads."""
        if len(self._buf) < self._n:
            return False
        m = self.median()
        if m is None:
            return False
        return m.spread_xy_mm <= self._xy_tol and m.spread_yaw_deg <= self._yaw_tol
