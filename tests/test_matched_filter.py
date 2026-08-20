"""Tests for src/dsp/matched_filter.py.

These specifically regression-test two real bugs found during development
(see the module docstring in matched_filter.py for the full story):
  1. Self-match SNR came out as 2x the template's own optimal SNR instead of
     exactly matching it (a normalization bug from building the correlation
     with full two-sided FFT instead of rfft/irfft).
  2. Zero-padding a template into a longer array changed its measured
     optimal SNR by >2x, because integrating the overlap over the full
     0-Nyquist band let a near-zero PSD estimate (routine in a filter
     stopband) blow up the integrand via floating-point spectral leakage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "physics"))

from data.ligo_loader import load_event_strain  # noqa: E402
from data.preprocessor import estimate_psd  # noqa: E402
from dsp.matched_filter import matched_filter_snr_series, optimal_snr, scale_to_target_snr  # noqa: E402
from physics.waveforms import generate_waveform  # noqa: E402

SAMPLE_RATE = 4096.0
F_LOW, F_HIGH = 30.0, 300.0


class _FakeSegment:
    def __init__(self, strain, sample_rate):
        self.strain = strain
        self.sample_rate = sample_rate


@pytest.fixture(scope="module")
def colored_psd():
    rng = np.random.default_rng(0)
    n = int(64 * SAMPLE_RATE)
    white = rng.normal(size=n)
    b, a = butter(4, 300.0 / (SAMPLE_RATE / 2), btype="low")
    colored = lfilter(b, a, white)
    return estimate_psd(_FakeSegment(colored, SAMPLE_RATE), fftlength=4.0, overlap=2.0), (b, a)


@pytest.fixture(scope="module")
def template(colored_psd):
    psd, _ = colored_psd
    wf = generate_waveform(30.0, 30.0, distance_mpc=100.0, sample_rate=SAMPLE_RATE,
                            t_coalescence=0.0, f_lower=30.0)
    scaled, _, _ = scale_to_target_snr(wf.h_plus, SAMPLE_RATE, psd, target_snr=20.0,
                                        f_low=F_LOW, f_high=F_HIGH)
    return scaled


def test_optimal_snr_is_invariant_to_zero_padding(colored_psd, template):
    """Regression test for bug #2: physically, zero-padding cannot change a
    signal's optimal SNR (same energy, same content)."""
    psd, _ = colored_psd
    short_snr = optimal_snr(template, SAMPLE_RATE, psd, f_low=F_LOW, f_high=F_HIGH)

    n_window = int(4 * SAMPLE_RATE)
    padded = np.zeros(n_window)
    padded[:len(template)] = template
    padded_snr = optimal_snr(padded, SAMPLE_RATE, psd, f_low=F_LOW, f_high=F_HIGH)

    assert padded_snr == pytest.approx(short_snr, rel=0.05)


def test_self_match_snr_equals_own_optimal_snr(colored_psd, template):
    """Regression test for bug #1: filtering a template against itself, at
    the same time position (zero shift), is the Cauchy-Schwarz equality case
    and MUST return exactly its own optimal SNR -- not 2x, not 0.5x."""
    psd, _ = colored_psd
    n_window = int(4 * SAMPLE_RATE)
    padded = np.zeros(n_window)
    padded[:len(template)] = template

    sigma = optimal_snr(padded, SAMPLE_RATE, psd, f_low=F_LOW, f_high=F_HIGH)
    snr_series = matched_filter_snr_series(padded, padded, SAMPLE_RATE, psd, f_low=F_LOW, f_high=F_HIGH)

    assert snr_series[0] == pytest.approx(sigma, rel=1e-9)
    assert np.argmax(snr_series) == 0


def test_matched_filter_locates_injection_time_and_snr(colored_psd, template):
    """A template injected into noise at a known offset should be recovered
    by the matched filter at (close to) that exact offset, with a peak SNR
    statistically consistent with the injected target."""
    psd, (b, a) = colored_psd
    n_window = int(4 * SAMPLE_RATE)
    template_padded = np.zeros(n_window)
    template_padded[:len(template)] = template

    inject_at = n_window // 3
    rng = np.random.default_rng(123)

    peaks_snr, peaks_idx = [], []
    for _ in range(5):
        noise = lfilter(b, a, rng.normal(size=n_window))
        injected = np.zeros(n_window)
        injected[inject_at:inject_at + len(template)] = template
        strain = noise + injected

        snr_series = matched_filter_snr_series(strain, template_padded, SAMPLE_RATE, psd,
                                                f_low=F_LOW, f_high=F_HIGH)
        peak_idx = int(np.argmax(snr_series))
        peaks_snr.append(snr_series[peak_idx])
        peaks_idx.append(peak_idx)

    assert all(abs(idx - inject_at) < 5 for idx in peaks_idx)
    mean_snr = np.mean(peaks_snr)
    assert abs(mean_snr - 20.0) < 0.3 * 20.0


def test_scale_to_target_snr_is_self_consistent(colored_psd):
    psd, _ = colored_psd
    wf = generate_waveform(25.0, 25.0, distance_mpc=200.0, sample_rate=SAMPLE_RATE,
                            t_coalescence=0.0, f_lower=30.0)
    for target in (5.0, 15.0, 40.0):
        scaled, _, _ = scale_to_target_snr(wf.h_plus, SAMPLE_RATE, psd, target,
                                            f_low=F_LOW, f_high=F_HIGH)
        measured = optimal_snr(scaled, SAMPLE_RATE, psd, f_low=F_LOW, f_high=F_HIGH)
        assert measured == pytest.approx(target, rel=1e-6)


def test_real_gw150914_search_peak_is_physically_sensible():
    """Regression test for a real bug: searching real GW150914 H1 data
    without band-limiting the strain in the time domain first produced a
    peak SNR of ~700 (vs. GW150914's known network SNR of ~24), because raw
    strain's huge sub-20Hz seismic power leaks into the 30-300Hz analysis
    band via spectral leakage from the un-tapered FFT. After bandpassing
    both strain and template before any FFT, the peak SNR should be in a
    physically sensible range and land at the true merger time."""
    from gwosc import datasets
    from physics.waveforms import generate_waveform as gen_wf

    merger_gps = datasets.event_gps("GW150914")
    full = load_event_strain("GW150914", detector="H1")
    window = full.slice_gps(merger_gps, half_width=2.0)
    psd = estimate_psd(full, fftlength=4.0, overlap=2.0)

    m1_det, m2_det = 36.0 * 1.09, 29.0 * 1.09
    wf = gen_wf(m1_det, m2_det, distance_mpc=410.0, sample_rate=window.sample_rate,
                t_coalescence=0.0, f_lower=30.0)

    n = len(window.strain)
    t_rel = window.times() - merger_gps
    template = np.zeros(n)
    start_idx = int(round((wf.t[0] - t_rel[0]) * window.sample_rate))
    template[start_idx:start_idx + len(wf.h_plus)] = wf.h_plus

    snr_series = matched_filter_snr_series(window.strain, template, window.sample_rate, psd,
                                            f_low=30.0, f_high=300.0)
    peak_idx = int(np.argmax(snr_series))

    # Physically sensible bound: real published GW150914 network SNR ~24;
    # a single-detector, inspiral-only, partial-band template should be
    # comparable, not off by orders of magnitude in either direction.
    assert 5.0 < snr_series[peak_idx] < 100.0
    assert abs(t_rel[peak_idx]) < 0.1, "peak should land close to the true merger time"

    background = np.abs(t_rel) > 0.3
    assert snr_series[peak_idx] > 20 * np.median(snr_series[background])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
