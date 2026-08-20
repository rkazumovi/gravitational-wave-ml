"""Tests for src/models/detector.py: architecture shape/compile checks and a
fast synthetic training-step smoke test (no real data dependency, so this
stays fast -- the real-data training run lives in train_detector.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))

from models.detector import build_detector  # noqa: E402


def test_build_detector_output_shape_and_range():
    model = build_detector(input_length=4096)
    x = np.random.default_rng(0).normal(size=(3, 4096, 1)).astype("float32")
    y = model.predict(x, verbose=0)
    assert y.shape == (3, 1)
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_model_is_compiled_with_expected_metrics():
    # In this Keras 3 version, model.metrics/metrics_names report generic
    # placeholders (['loss', 'compile_metrics']) rather than the actual
    # configured metric names, even after fitting -- verified directly.
    # history.history's keys are the reliable way to confirm which metrics
    # are actually being tracked and logged during training.
    model = build_detector(input_length=4096)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4096, 1)).astype("float32")
    y = np.array([0.0, 1.0], dtype="float32")
    history = model.fit(x, y, epochs=1, verbose=0)

    logged = set(history.history.keys())
    assert "auc" in logged
    assert "accuracy" in logged or "binary_accuracy" in logged


def test_model_trains_one_step_without_nan_loss():
    model = build_detector(input_length=4096)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 4096, 1)).astype("float32")
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype="float32")
    history = model.fit(x, y, epochs=1, verbose=0)
    loss = history.history["loss"][0]
    assert np.isfinite(loss)


def test_different_input_lengths_build_successfully():
    for length in (2048, 8192, 16384):
        model = build_detector(input_length=length)
        assert model.input_shape == (None, length, 1)
        assert model.output_shape == (None, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
