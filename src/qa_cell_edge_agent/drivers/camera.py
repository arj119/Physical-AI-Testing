"""USB camera driver — captures frames and generates thumbnails.

When ``mock=True`` the driver returns synthetic gradient images so the
full pipeline can run without a physical camera attached.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]
    logger.warning("opencv not available — camera will run in mock mode")

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]
    logger.warning("Pillow not available — thumbnails will be empty")


class Camera:
    """Capture frames from a USB camera (or generate mock frames)."""

    def __init__(
        self,
        device_index: int = 0,
        thumbnail_size: Tuple[int, int] = (64, 64),
        mock: bool = False,
    ) -> None:
        self.device_index = device_index
        self.thumbnail_size = thumbnail_size
        self.mock = mock or cv2 is None
        self._cap = None

        if not self.mock:
            self._cap = cv2.VideoCapture(device_index)
            if not self._cap.isOpened():
                logger.error("Failed to open camera at index %d — falling back to mock", device_index)
                self.mock = True
        if self.mock:
            logger.info("Camera running in MOCK mode")

    # ── public API ────────────────────────────────────────────────────

    def capture(self) -> Optional[np.ndarray]:
        """Return a BGR frame as a numpy array, or None on failure."""
        if self.mock:
            return self._mock_frame()
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Camera capture failed")
            return None
        return frame

    def make_thumbnail_b64(self, frame: np.ndarray) -> str:
        """Resize *frame* and return a base64-encoded JPEG string."""
        if Image is None:
            return ""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 else frame)
        img = img.resize(self.thumbnail_size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    # ── internals ─────────────────────────────────────────────────────

    @staticmethod
    def _mock_frame() -> np.ndarray:
        """Generate a 640×480 synthetic gradient image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
