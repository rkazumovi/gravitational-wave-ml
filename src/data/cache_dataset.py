"""
Pre-generate and cache a pool of (whitened_strain, label, metadata) training
examples to disk, then serve them via a fast tf.data pipeline.

Why: profiling found on-the-fly generation (src/data/dataset.py) costs
~1.75s/example -- negligible next to the model's own forward+backward pass
(~0.06s/batch of 8, once warmed up), but that makes full multi-epoch
training on-the-fly impractical on CPU-only hardware (see
docs/step8_detector.md). Standard fix, and the approach most published GW
ML papers actually use: generate a large, fixed, diverse pool of real-noise
+ injection examples ONCE (an unavoidable one-time cost, run in the
background), then train arbitrarily many fast epochs over that in-memory
pool. Trade-off vs. the infinite on-the-fly generator in dataset.py: finite
(though large) example variety instead of a fresh draw every single step.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentor import make_negative_example, make_positive_example  # noqa: E402
from ligo_loader import StrainSegment  # noqa: E402
from preprocessor import PSDEstimate, preprocess_segment  # noqa: E402


class _StrainLike:
    def __init__(self, strain, sample_rate):
        self.strain = strain
        self.sample_rate = sample_rate


def generate_example_pool(full_segments: list[StrainSegment], psds: list[PSDEstimate],
                           n_examples: int, output_path: Path | str, window_seconds: float = 4.0,
                           positive_fraction: float = 0.5, seed: int = 0,
                           mass_range: tuple[float, float] = (10.0, 50.0),
                           snr_range: tuple[float, float] = (8.0, 30.0),
                           whiten_low: float = 35.0, whiten_high: float = 350.0,
                           excluded_gps_ranges: tuple[tuple[float, float], ...] = (),
                           progress_every: int = 50) -> None:
    rng = np.random.default_rng(seed)
    sample_rate = full_segments[0].sample_rate
    n_samples = int(window_seconds * sample_rate)

    strains = np.empty((n_examples, n_samples), dtype=np.float32)
    labels = np.empty(n_examples, dtype=np.float32)
    snrs = np.full(n_examples, np.nan, dtype=np.float32)
    chirp_masses = np.full(n_examples, np.nan, dtype=np.float32)

    t_start = time.time()
    n_segments = len(full_segments)
    for i in range(n_examples):
        seg_idx = int(rng.integers(n_segments))
        full, psd = full_segments[seg_idx], psds[seg_idx]

        if rng.random() < positive_fraction:
            try:
                ex = make_positive_example(full, window_seconds, psd, rng, mass_range=mass_range,
                                            snr_range=snr_range, excluded_gps_ranges=excluded_gps_ranges)
            except (ValueError, RuntimeError):
                ex = make_negative_example(full, window_seconds, rng, excluded_gps_ranges=excluded_gps_ranges)
        else:
            ex = make_negative_example(full, window_seconds, rng, excluded_gps_ranges=excluded_gps_ranges)

        whitened = preprocess_segment(full, _StrainLike(ex.strain, ex.sample_rate),
                                       low=whiten_low, high=whiten_high)
        strains[i] = whitened.astype(np.float32)
        labels[i] = ex.label
        if ex.snr is not None:
            snrs[i] = ex.snr
            chirp_masses[i] = ex.chirp_mass_msun

        if (i + 1) % progress_every == 0 or i == n_examples - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (n_examples - i - 1) / rate if rate > 0 else float("nan")
            print(f"[{i + 1}/{n_examples}] elapsed={elapsed:.0f}s rate={rate:.2f}/s "
                  f"ETA={remaining:.0f}s label_balance={labels[:i + 1].mean():.2f}", flush=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, strain=strains, label=labels, snr=snrs,
                         chirp_mass_msun=chirp_masses, sample_rate=sample_rate,
                         window_seconds=window_seconds)
    print(f"Saved {n_examples} examples to {output_path} "
          f"({output_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t_start:.0f}s total")


def load_example_pool(cache_path: Path | str) -> dict:
    with np.load(cache_path) as data:
        return {k: data[k] for k in data.files}


def build_cached_dataset(cache_path: Path | str, batch_size: int = 32, val_fraction: float = 0.1,
                          shuffle_seed: int = 0):
    """Returns (train_ds, val_ds), both tf.data.Dataset yielding
    (strain[N], label) batches, split by a fixed shuffled index so the same
    cache always gives the same split."""
    import tensorflow as tf

    pool = load_example_pool(cache_path)
    strains, labels = pool["strain"], pool["label"]
    n = len(labels)

    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def _make(idx, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((strains[idx], labels[idx]))
        if training:
            ds = ds.shuffle(len(idx), seed=shuffle_seed, reshuffle_each_iteration=True).repeat()
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return _make(train_idx, training=True), _make(val_idx, training=False)


if __name__ == "__main__":
    from gwosc import datasets

    from ligo_loader import load_event_strain
    from preprocessor import estimate_psd

    merger_gps = datasets.event_gps("GW150914")
    exclude = ((merger_gps - 60.0, merger_gps + 60.0),)

    full_h1 = load_event_strain("GW150914", detector="H1")
    full_l1 = load_event_strain("GW150914", detector="L1")
    psd_h1 = estimate_psd(full_h1, fftlength=4.0, overlap=2.0)
    psd_l1 = estimate_psd(full_l1, fftlength=4.0, overlap=2.0)

    out_path = Path(__file__).resolve().parents[2] / "outputs" / "training_pool.npz"
    generate_example_pool([full_h1, full_l1], [psd_h1, psd_l1], n_examples=2000,
                           output_path=out_path, window_seconds=4.0, seed=42,
                           excluded_gps_ranges=exclude)
