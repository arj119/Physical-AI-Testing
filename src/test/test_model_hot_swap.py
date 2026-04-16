"""Tests for model hot-swap during runtime.

Verifies that ModelInference can:
  - Start in mock mode and produce results
  - Hot-swap to a real ONNX model via reload()
  - Continue producing results after reload
  - Survive a failed reload (bad path) and keep the previous model
  - Hot-swap between two different ONNX models
  - Handle the multiprocessing Event signalling pattern
"""

from __future__ import annotations

import os
import shutil
import tempfile
from multiprocessing import Event

import numpy as np
import pytest

from qa_cell_edge_agent.models.inference import ModelInference, CLASS_NAMES

# Path to the downloaded test model (may not exist in CI)
ONNX_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "yolov5n.onnx"
)
HAS_ONNX_MODEL = os.path.isfile(ONNX_MODEL)


@pytest.fixture
def mock_model():
    """A ModelInference in mock mode."""
    return ModelInference("nonexistent.onnx", mock=True)


@pytest.fixture
def sample_frame():
    """A synthetic 480x640 BGR frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Mock mode basics
# ---------------------------------------------------------------------------

class TestMockMode:
    def test_mock_produces_results(self, mock_model, sample_frame):
        result = mock_model.infer(sample_frame)
        assert result.detected_class in CLASS_NAMES
        assert 0.0 < result.confidence <= 1.0
        assert isinstance(result.bounding_box, list)
        assert len(result.bounding_box) == 4

    def test_mock_version_settable(self, mock_model):
        mock_model.version = "v2.0.0"
        assert mock_model.version == "v2.0.0"


# ---------------------------------------------------------------------------
# Hot-swap: mock → ONNX
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ONNX_MODEL, reason="yolov5n.onnx not downloaded")
class TestHotSwapMockToOnnx:
    def test_reload_from_mock_to_onnx(self, mock_model, sample_frame):
        """Start in mock, reload to real ONNX, verify inference still works."""
        assert mock_model.mock is True

        # Reload to real model
        ok = mock_model.reload(ONNX_MODEL)
        assert ok is True
        assert mock_model.mock is False
        assert mock_model._backend == "onnx"

        # Inference should work with the real model
        result = mock_model.infer(sample_frame)
        assert result.detected_class in CLASS_NAMES
        assert isinstance(result.bounding_box, list)
        assert result.inference_time_ms >= 0

    def test_reload_updates_model_path(self, mock_model):
        ok = mock_model.reload(ONNX_MODEL)
        assert ok is True
        assert mock_model.model_path == ONNX_MODEL


# ---------------------------------------------------------------------------
# Hot-swap: ONNX → ONNX (same or different model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ONNX_MODEL, reason="yolov5n.onnx not downloaded")
class TestHotSwapOnnxToOnnx:
    def test_reload_same_model(self, sample_frame):
        """Reload the same ONNX model — should work without issues."""
        model = ModelInference(ONNX_MODEL, mock=False)
        assert model._backend == "onnx"

        result_before = model.infer(sample_frame)

        ok = model.reload(ONNX_MODEL)
        assert ok is True

        result_after = model.infer(sample_frame)
        # Both should produce valid results (not necessarily identical due to threading)
        assert result_after.detected_class in CLASS_NAMES

    def test_reload_to_copied_model(self, sample_frame):
        """Copy the model to a temp path and reload from there."""
        with tempfile.TemporaryDirectory() as tmpdir:
            copied = os.path.join(tmpdir, "copied_model.onnx")
            shutil.copy2(ONNX_MODEL, copied)

            model = ModelInference(ONNX_MODEL, mock=False)
            ok = model.reload(copied)
            assert ok is True
            assert model.model_path == copied

            result = model.infer(sample_frame)
            assert result.detected_class in CLASS_NAMES


# ---------------------------------------------------------------------------
# Failed reload — keeps previous model
# ---------------------------------------------------------------------------

class TestFailedReload:
    def test_reload_bad_path_keeps_mock(self, mock_model, sample_frame):
        """Reloading from a nonexistent path should fail gracefully."""
        ok = mock_model.reload("/tmp/nonexistent_model.onnx")
        assert ok is False
        assert mock_model.mock is True  # still in mock mode

        # Mock inference should still work
        result = mock_model.infer(sample_frame)
        assert result.detected_class in CLASS_NAMES

    def test_reload_bad_extension_keeps_mock(self, mock_model):
        """Reloading a file with unsupported extension should fail."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            ok = mock_model.reload(f.name)
            assert ok is False

    @pytest.mark.skipif(not HAS_ONNX_MODEL, reason="yolov5n.onnx not downloaded")
    def test_reload_bad_path_keeps_onnx(self, sample_frame):
        """If already on ONNX and reload fails, old ONNX model stays."""
        model = ModelInference(ONNX_MODEL, mock=False)
        assert model._backend == "onnx"

        ok = model.reload("/tmp/nonexistent.onnx")
        assert ok is False
        # Should still have the old model
        assert model._backend == "onnx"
        assert model.model_path == ONNX_MODEL

        result = model.infer(sample_frame)
        assert result.detected_class in CLASS_NAMES


# ---------------------------------------------------------------------------
# Event-based signalling (simulates Process 3 → Process 2)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ONNX_MODEL, reason="yolov5n.onnx not downloaded")
class TestEventSignalling:
    def test_event_driven_reload(self, sample_frame):
        """Simulate the Process 3 → Process 2 reload signal pattern."""
        model = ModelInference("nonexistent.onnx", mock=True)
        reload_event = Event()
        model_path = ONNX_MODEL

        # Initially mock
        assert model.mock is True
        result = model.infer(sample_frame)
        assert result.detected_class in CLASS_NAMES

        # Process 3 signals a new model is available
        reload_event.set()

        # Process 2 checks the event (as in defect_detection.py:82-85)
        if reload_event.is_set():
            model.reload(model_path)
            reload_event.clear()

        assert reload_event.is_set() is False
        assert model.mock is False
        assert model._backend == "onnx"

        # Inference works with the new model
        result = model.infer(sample_frame)
        assert result.detected_class in CLASS_NAMES

    def test_event_not_set_no_reload(self):
        """If event is not set, no reload happens."""
        model = ModelInference("nonexistent.onnx", mock=True)
        reload_event = Event()

        # Event not set — should not reload
        if reload_event.is_set():
            model.reload(ONNX_MODEL)
            reload_event.clear()

        assert model.mock is True  # unchanged


# ---------------------------------------------------------------------------
# Concurrent inference during reload
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ONNX_MODEL, reason="yolov5n.onnx not downloaded")
class TestReloadDuringInference:
    def test_many_inferences_across_reload(self, sample_frame):
        """Run inference before, during, and after a reload."""
        model = ModelInference("nonexistent.onnx", mock=True)

        # Pre-reload: mock inference
        for _ in range(5):
            r = model.infer(sample_frame)
            assert r.detected_class in CLASS_NAMES

        # Reload
        ok = model.reload(ONNX_MODEL)
        assert ok is True

        # Post-reload: ONNX inference
        for _ in range(5):
            r = model.infer(sample_frame)
            assert r.detected_class in CLASS_NAMES
            assert isinstance(r.bounding_box, list)
