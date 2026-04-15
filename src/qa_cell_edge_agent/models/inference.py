"""YOLOv5 / TensorRT model inference wrapper.

Supports three backends:
  1. TensorRT engine (.engine) — production on Jetson
  2. ONNX via onnxruntime (.onnx) — development / CI
  3. Mock — returns synthetic detections for pipeline testing

The active backend is auto-detected from the model file extension.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["widget_good", "widget_defect", "widget_unknown"]


@dataclass
class InferenceResult:
    """Output of a single model inference run."""

    detected_class: str    # one of CLASS_NAMES
    confidence: float      # [0, 1]
    bounding_box: str      # JSON "[x, y, w, h]"
    inference_time_ms: int # wall-clock inference duration


class ModelInference:
    """Unified inference interface with hot-swap support."""

    def __init__(self, model_path: str, mock: bool = False) -> None:
        self.model_path = model_path
        self.mock = mock
        self._engine = None
        self._version: str = "unknown"

        if not mock and os.path.exists(model_path):
            self._load_model(model_path)
        elif not mock:
            logger.warning("Model file not found at %s — running in mock mode", model_path)
            self.mock = True

        if self.mock:
            logger.info("ModelInference running in MOCK mode")

    # ── public API ────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray) -> InferenceResult:
        """Run inference on a single BGR frame."""
        if self.mock:
            return self._mock_infer()

        t0 = time.monotonic()
        # --- TensorRT / ONNX inference (implementation depends on backend) ---
        # Placeholder: subclass or replace this block with actual engine call.
        result = self._run_engine(frame)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        result.inference_time_ms = elapsed_ms
        return result

    def reload(self, new_path: str) -> bool:
        """Hot-swap the model file. Returns True on success."""
        logger.info("Hot-swapping model: %s → %s", self.model_path, new_path)
        try:
            self._load_model(new_path)
            self.model_path = new_path
            self.mock = False
            return True
        except Exception as exc:
            logger.error("Model reload failed: %s — keeping previous model", exc)
            return False

    @property
    def version(self) -> str:
        return self._version

    @version.setter
    def version(self, v: str) -> None:
        self._version = v

    # ── internals ─────────────────────────────────────────────────────

    def _load_model(self, path: str) -> None:
        """Load a model file based on its extension."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".engine":
            self._load_tensorrt(path)
        elif ext == ".onnx":
            self._load_onnx(path)
        else:
            raise ValueError(f"Unsupported model format: {ext}")

    def _load_tensorrt(self, path: str) -> None:
        """Load a TensorRT serialised engine."""
        try:
            import tensorrt as trt  # type: ignore[import]
            import pycuda.autoinit  # type: ignore[import]  # noqa: F401

            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(path, "rb") as f, trt.Runtime(trt_logger) as runtime:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            logger.info("TensorRT engine loaded from %s", path)
        except Exception as exc:
            raise RuntimeError(f"TensorRT load failed: {exc}") from exc

    def _load_onnx(self, path: str) -> None:
        """Load an ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort  # type: ignore[import]

            self._engine = ort.InferenceSession(path)
            logger.info("ONNX model loaded from %s", path)
        except Exception as exc:
            raise RuntimeError(f"ONNX load failed: {exc}") from exc

    _engine_warning_logged = False

    def _run_engine(self, frame: np.ndarray) -> InferenceResult:
        """Execute inference on the loaded engine.

        TODO: Replace this stub with actual TensorRT / ONNX pre/post-processing
        once the trained model is available.
        """
        if not ModelInference._engine_warning_logged:
            logger.warning(
                "_run_engine is a STUB — returning mock results despite model "
                "loaded from %s. Implement real pre/post-processing.",
                self.model_path,
            )
            ModelInference._engine_warning_logged = True
        return self._mock_infer()

    @staticmethod
    def _mock_infer() -> InferenceResult:
        """Generate a synthetic inference result with realistic distributions."""
        # 70% good, 20% defect, 10% unknown
        roll = random.random()
        if roll < 0.70:
            cls = "widget_good"
            conf = round(random.gauss(0.92, 0.05), 4)
        elif roll < 0.90:
            cls = "widget_defect"
            conf = round(random.gauss(0.85, 0.08), 4)
        else:
            cls = "widget_unknown"
            conf = round(random.gauss(0.55, 0.12), 4)

        conf = max(0.01, min(conf, 0.99))
        x, y = random.randint(50, 400), random.randint(50, 300)
        w, h = random.randint(60, 150), random.randint(60, 150)

        return InferenceResult(
            detected_class=cls,
            confidence=conf,
            bounding_box=f"[{x}, {y}, {w}, {h}]",
            inference_time_ms=random.randint(25, 60),
        )
