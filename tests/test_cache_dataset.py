"""Tests for src/data/cache_dataset.py: pre-generated example pool + fast
tf.data pipeline built from it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

from data.cache_dataset import build_cached_dataset, generate_example_pool, load_example_pool  # noqa: E402
from data.ligo_loader import load_event_strain  # noqa: E402
from data.preprocessor import estimate_psd  # noqa: E402

GW150914_GPS = 1126259462.4
EXCLUDE = ((GW150914_GPS - 60.0, GW150914_GPS + 60.0),)


@pytest.fixture(scope="module")
def small_pool_path(tmp_path_factory):
    full = load_event_strain("GW150914", detector="H1")
    psd = estimate_psd(full, fftlength=4.0, overlap=2.0)
    out_path = tmp_path_factory.mktemp("cache") / "pool.npz"
    generate_example_pool([full], [psd], n_examples=20, output_path=out_path,
                           window_seconds=4.0, seed=11, excluded_gps_ranges=EXCLUDE,
                           progress_every=100)
    return out_path


def test_generated_pool_has_expected_shapes_and_labels(small_pool_path):
    pool = load_example_pool(small_pool_path)
    assert pool["strain"].shape == (20, int(4.0 * 4096))
    assert pool["label"].shape == (20,)
    assert set(np.unique(pool["label"])).issubset({0.0, 1.0})
    # SNR/chirp mass recorded for positives, NaN for negatives
    is_positive = pool["label"] == 1.0
    assert np.all(np.isfinite(pool["snr"][is_positive]))
    assert np.all(np.isnan(pool["snr"][~is_positive]))


def test_build_cached_dataset_val_split_has_correct_size(small_pool_path):
    # val_ds does NOT repeat, so one full pass over it is meaningful to count
    # directly. train_ds DOES repeat (see test_train_dataset_repeats_indefinitely
    # below) -- take(n) on it always returns exactly n full batches regardless
    # of the underlying pool size, so counting "one pass" over it isn't a
    # meaningful check and isn't attempted here.
    _, val_ds = build_cached_dataset(small_pool_path, batch_size=4, val_fraction=0.25)
    val_labels = np.concatenate([b[1].numpy() for b in val_ds])
    assert len(val_labels) == 5  # 25% of 20


def test_train_dataset_repeats_indefinitely(small_pool_path):
    train_ds, _ = build_cached_dataset(small_pool_path, batch_size=4, val_fraction=0.25)
    # 15 train examples / batch_size=4 -> 3 full batches + 1 partial per epoch;
    # pulling more batches than one epoch's worth must not raise StopIteration.
    batches = list(train_ds.take(10))
    assert len(batches) == 10


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
