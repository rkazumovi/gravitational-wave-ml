"""Tests for src/data/augmentor.py: synthetic signal injection into real
GW150914 H1 noise, away from the real event."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dsp"))

from data.augmentor import make_negative_example, make_positive_example, sample_noise_window  # noqa: E402
from data.ligo_loader import load_event_strain  # noqa: E402
from data.preprocessor import estimate_psd  # noqa: E402
from dsp.matched_filter import matched_filter_snr_series  # noqa: E402

GW150914_GPS = 1126259462.4
EXCLUDE = ((GW150914_GPS - 60.0, GW150914_GPS + 60.0),)
WINDOW_SECONDS = 4.0


@pytest.fixture(scope="module")
def h1_full():
    return load_event_strain("GW150914", detector="H1")


@pytest.fixture(scope="module")
def h1_psd(h1_full):
    return estimate_psd(h1_full, fftlength=4.0, overlap=2.0)


def test_sample_noise_window_avoids_excluded_range_and_passes_dq_cut(h1_full):
    rng = np.random.default_rng(1)
    for _ in range(20):
        window = sample_noise_window(h1_full, WINDOW_SECONDS, rng, excluded_gps_ranges=EXCLUDE)
        assert not (window.gps_start < EXCLUDE[0][1] and window.gps_end > EXCLUDE[0][0])
        n_seconds = int(window.duration)
        for s in range(n_seconds):
            assert window.is_analysis_ready(window.gps_start + s + 0.5)


def test_make_negative_example_shape_and_label(h1_full):
    rng = np.random.default_rng(2)
    ex = make_negative_example(h1_full, WINDOW_SECONDS, rng, excluded_gps_ranges=EXCLUDE)
    assert ex.label == 0
    assert len(ex.strain) == int(WINDOW_SECONDS * h1_full.sample_rate)
    assert ex.snr is None


def test_make_positive_example_metadata_is_consistent(h1_full, h1_psd):
    rng = np.random.default_rng(3)
    ex = make_positive_example(h1_full, WINDOW_SECONDS, h1_psd, rng, snr_range=(12.0, 12.0),
                                excluded_gps_ranges=EXCLUDE)
    assert ex.label == 1
    assert len(ex.strain) == int(WINDOW_SECONDS * h1_full.sample_rate)
    assert ex.snr == pytest.approx(12.0)
    assert ex.chirp_mass_msun > 0
    assert 0 < ex.injection_time_rel < WINDOW_SECONDS
    assert ex.template is not None
    assert len(ex.template) == len(ex.strain)
    assert np.any(ex.template != 0)


def test_injected_signal_recoverable_at_known_location(h1_full, h1_psd):
    """The injected signal must be recoverable, via matched filtering, at
    its own known location with an SNR statistically consistent with the
    target. Deliberately does NOT check the global SNR-series max: real
    detector noise contains non-Gaussian transients that can produce a
    higher spurious peak elsewhere, unrelated to the injection -- found
    exactly this during development (see docs/step6_augmentor.md). Checking
    the global max would conflate "did injection work" with "does this
    noise window happen to contain a glitch," which is not what this test
    is for.
    """
    rng = np.random.default_rng(4)
    target_snr = 15.0
    recovered = []
    for _ in range(5):
        ex = make_positive_example(h1_full, WINDOW_SECONDS, h1_psd, rng,
                                    snr_range=(target_snr, target_snr), excluded_gps_ranges=EXCLUDE)
        snr_series = matched_filter_snr_series(ex.strain, ex.template, ex.sample_rate, h1_psd,
                                                f_low=30.0, f_high=300.0)
        recovered.append(snr_series[0])  # lag 0: template already matches strain's placement

    recovered = np.array(recovered)
    assert np.all(recovered > 0.5 * target_snr)
    assert np.mean(recovered) < 3.0 * target_snr


def test_make_positive_example_rejects_waveform_too_long_for_window(h1_full, h1_psd):
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        # Very low masses -> very long inspiral duration from f_lower, won't
        # fit in a short window.
        make_positive_example(h1_full, window_seconds=1.0, psd=h1_psd, rng=rng,
                               mass_range=(5.0, 6.0), f_lower=20.0, excluded_gps_ranges=EXCLUDE)


def test_batch_generation_alternating_labels_and_consistent_shape(h1_full, h1_psd):
    rng = np.random.default_rng(6)
    batch = [
        make_positive_example(h1_full, WINDOW_SECONDS, h1_psd, rng, excluded_gps_ranges=EXCLUDE)
        if i % 2 == 0 else make_negative_example(h1_full, WINDOW_SECONDS, rng, excluded_gps_ranges=EXCLUDE)
        for i in range(8)
    ]
    assert sum(ex.label for ex in batch) == 4
    assert {len(ex.strain) for ex in batch} == {int(WINDOW_SECONDS * h1_full.sample_rate)}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
