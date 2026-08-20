"""
TensorFlow dataset pipeline: real LIGO noise -> synthetic signal injection
(src/data/augmentor.py) -> whitening/bandpass (src/data/preprocessor.py) ->
batched (whitened_strain, label) pairs for the detection model.

Each pulled example is generated fresh (random noise window, random
injection parameters or none) rather than drawn from a fixed pre-computed
set, so a model trains against effectively unlimited variety rather than
overfitting a small cached dataset -- standard practice for this kind of
synthetic augmentation pipeline.

Gotcha found and fixed during development: `tf.data.Dataset.from_generator`
re-invokes the underlying Python generator function from scratch every time
the Dataset is iterated from the start (a fresh `for batch in ds:` or
`iter(ds)`, e.g. what Keras' `model.fit` does at the start of each epoch
unless the dataset is consumed as one continuous stream). With a *fixed*
seed, that means restarting iteration reproduces the exact same sequence of
examples -- verified directly: two separate `ds.take(2)` passes over the
same fixed-seed dataset returned identical batches. For training that would
silently mean every epoch sees identical data, defeating the entire point
of unlimited synthetic augmentation. `build_dataset` therefore defaults
`seed=None`, which draws fresh OS entropy each time the generator restarts
(via `np.random.default_rng(None)`) -- pass an explicit int only for
reproducible tests, where restart-identical output is actually desired.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentor import make_negative_example, make_positive_example  # noqa: E402
from ligo_loader import StrainSegment  # noqa: E402
from preprocessor import PSDEstimate, preprocess_segment  # noqa: E402


class _StrainLike:
    def __init__(self, strain: np.ndarray, sample_rate: float):
        self.strain = strain
        self.sample_rate = sample_rate


def _example_generator(full_segments: list[StrainSegment], psds: list[PSDEstimate],
                        window_seconds: float, seed: int, positive_fraction: float,
                        mass_range: tuple[float, float], snr_range: tuple[float, float],
                        whiten_low: float, whiten_high: float,
                        excluded_gps_ranges: tuple[tuple[float, float], ...]):
    rng = np.random.default_rng(seed)
    n_segments = len(full_segments)
    while True:
        idx = int(rng.integers(n_segments))
        full, psd = full_segments[idx], psds[idx]

        if rng.random() < positive_fraction:
            try:
                ex = make_positive_example(full, window_seconds, psd, rng, mass_range=mass_range,
                                            snr_range=snr_range, excluded_gps_ranges=excluded_gps_ranges)
            except (ValueError, RuntimeError):
                continue  # e.g. sampled masses too low to fit the window; just retry
        else:
            ex = make_negative_example(full, window_seconds, rng, excluded_gps_ranges=excluded_gps_ranges)

        whitened = preprocess_segment(full, _StrainLike(ex.strain, ex.sample_rate),
                                       low=whiten_low, high=whiten_high)
        yield whitened.astype(np.float32), np.float32(ex.label)


def build_dataset(full_segments: list[StrainSegment], psds: list[PSDEstimate],
                   window_seconds: float = 4.0, batch_size: int = 16, seed: int | None = None,
                   positive_fraction: float = 0.5, mass_range: tuple[float, float] = (10.0, 50.0),
                   snr_range: tuple[float, float] = (8.0, 30.0), whiten_low: float = 35.0,
                   whiten_high: float = 350.0,
                   excluded_gps_ranges: tuple[tuple[float, float], ...] = ()) -> tf.data.Dataset:
    """An infinite tf.data.Dataset of (whitened_strain[N], label) batches.
    `full_segments`/`psds` are parallel lists: one real strain file's full
    segment and its PSD estimate per detector/file used as a noise source."""
    if len({s.sample_rate for s in full_segments}) != 1:
        raise ValueError("all full_segments must share the same sample rate")
    sample_rate = full_segments[0].sample_rate
    n_samples = int(window_seconds * sample_rate)

    output_signature = (
        tf.TensorSpec(shape=(n_samples,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _example_generator(full_segments, psds, window_seconds, seed, positive_fraction,
                                    mass_range, snr_range, whiten_low, whiten_high, excluded_gps_ranges),
        output_signature=output_signature,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    from gwosc import datasets

    from ligo_loader import load_event_strain
    from preprocessor import estimate_psd

    merger_gps = datasets.event_gps("GW150914")
    exclude = ((merger_gps - 60.0, merger_gps + 60.0),)

    full_h1 = load_event_strain("GW150914", detector="H1")
    psd_h1 = estimate_psd(full_h1, fftlength=4.0, overlap=2.0)

    ds = build_dataset([full_h1], [psd_h1], window_seconds=4.0, batch_size=8, seed=42,
                        excluded_gps_ranges=exclude)

    n_batches = 5
    labels_seen = []
    for i, (strain_batch, label_batch) in enumerate(ds.take(n_batches)):
        assert strain_batch.shape == (8, 4 * 4096)
        assert label_batch.shape == (8,)
        assert strain_batch.dtype == tf.float32
        labels_seen.extend(label_batch.numpy().tolist())
        pos_std = strain_batch.numpy()[label_batch.numpy() == 1].std() if (label_batch.numpy() == 1).any() else float("nan")
        print(f"batch {i}: shape={strain_batch.shape}, labels={label_batch.numpy()}, "
              f"positive-example std={pos_std:.3f}")

    labels_seen = np.array(labels_seen)
    frac_positive = labels_seen.mean()
    print(f"\n{len(labels_seen)} examples across {n_batches} batches: "
          f"{frac_positive:.0%} positive (target ~50%)")
    assert 0.2 < frac_positive < 0.8, "label balance should be roughly consistent with positive_fraction"
    assert set(np.unique(labels_seen)) == {0.0, 1.0}
    print("Dataset pipeline validated.")

    print("\n--- Confirming the seed=None re-iteration fix ---")
    ds_fixed = build_dataset([full_h1], [psd_h1], window_seconds=4.0, batch_size=4, seed=1,
                              excluded_gps_ranges=exclude)
    fixed_pass_1 = [b[1].numpy().tolist() for b in ds_fixed.take(2)]
    fixed_pass_2 = [b[1].numpy().tolist() for b in ds_fixed.take(2)]
    print(f"seed=1, two restarts identical: {fixed_pass_1 == fixed_pass_2} (expected True)")
    assert fixed_pass_1 == fixed_pass_2

    ds_random = build_dataset([full_h1], [psd_h1], window_seconds=4.0, batch_size=4,
                               excluded_gps_ranges=exclude)  # seed=None (default)
    # Compare continuous strain values, not just binary labels: with only a
    # few binary labels per pass, two independent draws could coincidentally
    # match by chance and make this check flaky.
    random_pass_1 = [b[0].numpy().tolist() for b in ds_random.take(2)]
    random_pass_2 = [b[0].numpy().tolist() for b in ds_random.take(2)]
    print(f"seed=None, two restarts identical: {random_pass_1 == random_pass_2} (expected False)")
    assert random_pass_1 != random_pass_2, \
        "default seed=None should give fresh data on every restart -- see module docstring"
    print("Re-iteration fix validated.")
