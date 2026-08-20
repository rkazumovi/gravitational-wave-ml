"""Tests for src/data/preprocessor.py: whitening flattens the noise
spectrum, bandpass is zero-phase and removes out-of-band content, and the
real GW150914 chirp becomes visible after preprocessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, filtfilt, lfilter, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

from data.ligo_loader import load_event_strain  # noqa: E402
from data.preprocessor import bandpass, estimate_psd, preprocess_segment, whiten  # noqa: E402

GW150914_GPS = 1126259462.4
SAMPLE_RATE = 4096.0


def _make_colored_noise(n_seconds: float = 64.0, seed: int = 0) -> np.ndarray:
    """White noise passed through a low-pass filter, giving strongly
    frequency-dependent (colored) power — a stand-in for real detector noise
    with a known, non-flat spectral shape to whiten against."""
    rng = np.random.default_rng(seed)
    n = int(n_seconds * SAMPLE_RATE)
    white = rng.normal(size=n)
    b, a = butter(4, 50.0 / (SAMPLE_RATE / 2), btype="low")
    return lfilter(b, a, white)


class _FakeSegment:
    def __init__(self, strain, sample_rate):
        self.strain = strain
        self.sample_rate = sample_rate


def test_whitening_flattens_a_known_colored_spectrum():
    colored = _make_colored_noise()
    fake_seg = _FakeSegment(colored, SAMPLE_RATE)
    psd = estimate_psd(fake_seg, fftlength=4.0, overlap=2.0)

    chunk = colored[: int(8 * SAMPLE_RATE)]
    whitened = whiten(chunk, SAMPLE_RATE, psd)

    freqs_before, psd_before = welch(chunk, fs=SAMPLE_RATE, nperseg=4096)
    freqs_after, psd_after = welch(whitened, fs=SAMPLE_RATE, nperseg=4096)

    # Restrict to a band clear of the DC bin and the low-pass rolloff edge,
    # where "whitened" vs "colored" should differ most starkly.
    band = (freqs_before > 5) & (freqs_before < SAMPLE_RATE / 2 - 100)
    ratio_before = psd_before[band].max() / psd_before[band].min()
    ratio_after = psd_after[band].max() / psd_after[band].min()

    assert ratio_after < ratio_before / 10, (
        f"expected whitening to flatten the spectrum, got before={ratio_before:.1e} "
        f"after={ratio_after:.1e}"
    )


def test_bandpass_removes_out_of_band_tone_and_keeps_in_band_tone():
    t = np.arange(int(4 * SAMPLE_RATE)) / SAMPLE_RATE
    low_tone = np.sin(2 * np.pi * 10 * t)     # below the 35-350 Hz passband
    in_band_tone = np.sin(2 * np.pi * 100 * t)  # inside the passband
    signal = low_tone + in_band_tone

    filtered = bandpass(signal, SAMPLE_RATE, low=35.0, high=350.0)

    def power_at(x, freq):
        freqs, psd = welch(x, fs=SAMPLE_RATE, nperseg=4096)
        return psd[np.argmin(np.abs(freqs - freq))]

    assert power_at(filtered, 10) < 0.01 * power_at(signal, 10)
    assert power_at(filtered, 100) > 0.5 * power_at(signal, 100)


def test_bandpass_is_zero_phase():
    """filtfilt must not shift a feature in time, unlike a causal lfilter."""
    n = int(2 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    pulse_idx = n // 2
    signal = np.zeros(n)
    signal[pulse_idx - 5:pulse_idx + 5] = np.hanning(10)

    zero_phase = bandpass(signal, SAMPLE_RATE, low=35.0, high=350.0)
    b, a = butter(4, [35 / (SAMPLE_RATE / 2), 350 / (SAMPLE_RATE / 2)], btype="band")
    causal = lfilter(b, a, signal)

    assert abs(np.argmax(np.abs(zero_phase)) - pulse_idx) < abs(np.argmax(np.abs(causal)) - pulse_idx)
    assert abs(np.argmax(np.abs(zero_phase)) - pulse_idx) <= 2


@pytest.fixture(scope="module")
def h1_processed_around_merger():
    full = load_event_strain("GW150914", detector="H1")
    window = full.slice_gps(GW150914_GPS, half_width=2.0)
    return window, preprocess_segment(full, window)


def test_real_gw150914_chirp_peaks_near_merger(h1_processed_around_merger):
    """Quantitative version of 'the chirp is visible by eye': the largest
    whitened+bandpassed amplitude in the window should land within a few
    tens of ms of the real merger GPS time, not at some unrelated time."""
    window, clean = h1_processed_around_merger
    t = window.times() - GW150914_GPS

    crop = int(0.1 * len(t))
    t, clean = t[crop:-crop], clean[crop:-crop]

    peak_time = t[np.argmax(np.abs(clean))]
    assert -0.05 <= peak_time <= 0.05, f"expected merger-adjacent peak, got t={peak_time:.4f}s"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
