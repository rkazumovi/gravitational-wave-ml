"""Tests for src/data/dataset.py: the TF dataset pipeline built on top of
real GW150914 H1 noise (loading -> injection -> whitening -> batching)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

from data.dataset import build_dataset  # noqa: E402
from data.ligo_loader import load_event_strain  # noqa: E402
from data.preprocessor import estimate_psd  # noqa: E402

GW150914_GPS = 1126259462.4
EXCLUDE = ((GW150914_GPS - 60.0, GW150914_GPS + 60.0),)


@pytest.fixture(scope="module")
def h1_full():
    return load_event_strain("GW150914", detector="H1")


@pytest.fixture(scope="module")
def h1_psd(h1_full):
    return estimate_psd(h1_full, fftlength=4.0, overlap=2.0)


def test_batch_shapes_and_dtypes(h1_full, h1_psd):
    ds = build_dataset([h1_full], [h1_psd], window_seconds=4.0, batch_size=4, seed=0,
                        excluded_gps_ranges=EXCLUDE)
    strain_batch, label_batch = next(iter(ds.take(1)))
    assert strain_batch.shape == (4, int(4.0 * h1_full.sample_rate))
    assert label_batch.shape == (4,)
    assert strain_batch.dtype.name == "float32"
    assert set(np.unique(label_batch.numpy())).issubset({0.0, 1.0})


def test_label_balance_roughly_matches_positive_fraction(h1_full, h1_psd):
    ds = build_dataset([h1_full], [h1_psd], window_seconds=4.0, batch_size=8, seed=1,
                        positive_fraction=0.5, excluded_gps_ranges=EXCLUDE)
    labels = np.concatenate([b[1].numpy() for b in ds.take(6)])
    assert 0.2 < labels.mean() < 0.8


def test_fixed_seed_reproduces_identical_sequence_on_restart(h1_full, h1_psd):
    """Documents the intended, tested behavior of an explicit seed: useful
    for reproducible tests, NOT what you want for multi-epoch training (see
    the next test and the module docstring in dataset.py)."""
    ds = build_dataset([h1_full], [h1_psd], window_seconds=4.0, batch_size=4, seed=5,
                        excluded_gps_ranges=EXCLUDE)
    first = [b[1].numpy().tolist() for b in ds.take(2)]
    second = [b[1].numpy().tolist() for b in ds.take(2)]
    assert first == second


def test_default_seed_gives_fresh_data_on_restart(h1_full, h1_psd):
    """Regression test for a real gotcha found during development:
    tf.data.Dataset.from_generator re-invokes the generator from scratch on
    every fresh iteration (e.g. what Keras does each epoch), so a *fixed*
    seed would silently give identical data every epoch. build_dataset
    defaults to seed=None specifically to avoid this."""
    ds = build_dataset([h1_full], [h1_psd], window_seconds=4.0, batch_size=4,
                        excluded_gps_ranges=EXCLUDE)  # seed=None
    first = [b[0].numpy().tolist() for b in ds.take(2)]
    second = [b[0].numpy().tolist() for b in ds.take(2)]
    assert first != second


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
