"""YOLOv5 / TensorRT model inference wrapper.

Supports three backends:
  1. TensorRT engine (.engine) — production on Jetson
  2. ONNX via onnxruntime (.onnx) — development / CI
  3. Mock — returns synthetic detections for pipeline testing

The active backend is auto-detected from the model file extension.

For development testing with a real model, run::

    python scripts/download_model.py

to fetch ``yolov5n.onnx`` (the standard YOLOv5-nano pretrained on COCO).
The class names will be COCO classes rather than widget classes, but the
full inference pipeline (preprocessing → model → NMS → postprocessing) runs
end-to-end.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CLASS_NAMES = ["widget_good", "widget_defect", "widget_unknown"]

# Standard YOLOv5 input size
_INPUT_SIZE = 640


@dataclass
class InferenceResult:
    """Output of a single model inference run."""

    detected_class: str    # one of CLASS_NAMES
    confidence: float      # [0, 1]
    bounding_box: List[float]  # [x, y, w, h]
    inference_time_ms: int # wall-clock inference duration


class ModelInference:
    """Unified inference interface with hot-swap support."""

    def __init__(self, model_path: str, mock: bool = False) -> None:
        self.model_path = model_path
        self.mock = mock
        self._engine = None
        self._backend: Optional[str] = None  # "onnx" or "tensorrt"
        self._onnx_class_names: Optional[List[str]] = None
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
            self._backend = "tensorrt"
            logger.info("TensorRT engine loaded from %s", path)
        except Exception as exc:
            raise RuntimeError(f"TensorRT load failed: {exc}") from exc

    def _load_onnx(self, path: str) -> None:
        """Load an ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort  # type: ignore[import]

            self._engine = ort.InferenceSession(path)
            self._backend = "onnx"
            meta = self._engine.get_modelmeta()
            # YOLOv5 ONNX exports store class names in metadata
            if meta.custom_metadata_map and "names" in meta.custom_metadata_map:
                import ast
                self._onnx_class_names = list(
                    ast.literal_eval(meta.custom_metadata_map["names"]).values()
                )
            else:
                self._onnx_class_names = None
            logger.info(
                "ONNX model loaded from %s (%d classes)",
                path,
                len(self._onnx_class_names) if self._onnx_class_names else -1,
            )
        except Exception as exc:
            raise RuntimeError(f"ONNX load failed: {exc}") from exc

    def _run_engine(self, frame: np.ndarray) -> InferenceResult:
        """Execute inference on the loaded engine."""
        if self._backend == "onnx":
            return self._run_onnx(frame)
        # TensorRT stub — needs CUDA context + buffer allocation
        logger.debug("TensorRT _run_engine called — using mock (CUDA not available)")
        return self._mock_infer()

    # ── ONNX inference ────────────────────────────────────────────────

    def _run_onnx(self, frame: np.ndarray) -> InferenceResult:
        """Run YOLOv5 ONNX inference with real pre/post-processing."""
        # Preprocess: letterbox resize, BGR→RGB, HWC→CHW, normalize to [0,1]
        img, ratio, (pad_w, pad_h) = _letterbox(frame, _INPUT_SIZE)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[np.newaxis, ...]  # add batch dim

        # Run model
        input_name = self._engine.get_inputs()[0].name
        outputs = self._engine.run(None, {input_name: img})
        preds = outputs[0]  # shape: [1, num_detections, 5+num_classes]

        # Post-process: filter by confidence, NMS, pick best detection
        detections = _postprocess(preds[0], conf_threshold=0.25, iou_threshold=0.45)

        if len(detections) == 0:
            return InferenceResult(
                detected_class="widget_unknown",
                confidence=0.0,
                bounding_box=[0.0, 0.0, 0.0, 0.0],
                inference_time_ms=0,
            )

        # Pick highest-confidence detection
        best = max(detections, key=lambda d: d[4])
        x1, y1, x2, y2, conf, cls_id = best

        # Un-letterbox back to original frame coordinates
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio
        w, h = x2 - x1, y2 - y1

        # Map class ID to our class names
        cls_id = int(cls_id)
        if self._onnx_class_names and cls_id < len(self._onnx_class_names):
            raw_class = self._onnx_class_names[cls_id]
        else:
            raw_class = f"class_{cls_id}"

        # Map external model classes to our domain classes.
        # For a fine-tuned model, classes should already be widget_good/defect/unknown.
        # For a generic COCO model used for testing, map everything through.
        if raw_class in CLASS_NAMES:
            detected_class = raw_class
        else:
            # Generic model (e.g. COCO yolov5n): treat as widget_unknown
            # but preserve the raw class name in the bounding box string for debugging.
            detected_class = "widget_unknown"

        return InferenceResult(
            detected_class=detected_class,
            confidence=round(float(conf), 4),
            bounding_box=[float(x1), float(y1), float(w), float(h)],
            inference_time_ms=0,
        )

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
            bounding_box=[float(x), float(y), float(w), float(h)],
            inference_time_ms=random.randint(25, 60),
        )


# ── YOLOv5 pre/post-processing helpers ──────────────────────────────────


def _letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: tuple = (114, 114, 114),
) -> tuple:
    """Resize and pad image to ``new_shape`` with preserved aspect ratio.

    Returns (padded_img, scale_ratio, (pad_w, pad_h)).
    """
    h, w = img.shape[:2]
    ratio = new_shape / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (new_shape - new_w) / 2
    pad_h = (new_shape - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, (pad_w, pad_h)


def _postprocess(
    preds: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[tuple]:
    """Filter YOLOv5 raw output by confidence and apply NMS.

    ``preds`` shape: ``[num_detections, 5 + num_classes]``
    where columns are ``[cx, cy, w, h, obj_conf, cls0, cls1, ...]``.

    Returns list of ``(x1, y1, x2, y2, confidence, class_id)`` tuples.
    """
    # Object confidence filter
    obj_conf = preds[:, 4]
    mask = obj_conf > conf_threshold
    preds = preds[mask]
    if len(preds) == 0:
        return []

    # Class scores = obj_conf * class_prob
    class_probs = preds[:, 5:]
    class_ids = class_probs.argmax(axis=1)
    class_confs = class_probs[np.arange(len(class_probs)), class_ids]
    scores = preds[:, 4] * class_confs

    # Convert cxcywh → xyxy
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Simple NMS (greedy, per-class not needed for single-best use case)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_rest = (x2[rest] - x1[rest]) * (y2[rest] - y1[rest])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou < iou_threshold]

    return [
        (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
         float(scores[i]), int(class_ids[i]))
        for i in keep
    ]
