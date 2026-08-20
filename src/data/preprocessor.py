"""
Whitening, PSD estimation, and bandpass filtering for raw LIGO strain.

Math (see also docs/step3_preprocessing.md):

    Raw strain s(t) = h(t) + n(t) is dominated by colored noise n(t) with a
    frequency-dependent power spectral density S_n(f) spanning many orders of
    magnitude. Whitening removes that frequency weighting so noise has a flat
    spectrum and genuine signal power stands out:

        X_white(f) = X(f) / sqrt(S_n(f))   (normalized to unit variance)
        x_white(t) = IFFT[X_white(f)]

    S_n(f) is estimated with Welch's method (scipy.signal.welch): average
    the periodogram of many overlapping, windowed segments of a
    noise-dominated stretch. The whitening normalization used here
    (`norm = 1/sqrt(1/(2*dt))`) matches the convention used in the official
    GWOSC open-data tutorials for this exact dataset, so whitened GW150914
    strain can be sanity-checked against the well-known public result.

    A Tukey window tapers the chunk being whitened before its FFT (avoids
    spectral leakage / edge artifacts), and a zero-phase Butterworth bandpass
    (filtfilt, so no timing smear) restricts the whitened output to the band
    where LIGO is sensitive and a BBH chirp has power.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ligo_loader import StrainSegment  # noqa: E402


@dataclass
class PSDEstimate:
    freqs: np.ndarray
    psd: np.ndarray

    def interpolator(self) -> interp1d:
        return interp1d(self.freqs, self.psd, bounds_error=False, fill_value="extrapolate")


def estimate_psd(segment: StrainSegment, fftlength: float = 4.0, overlap: float = 2.0) -> PSDEstimate:
    """Welch PSD estimate from an entire strain segment. Averaging over many
    windows means a single short transient (a real GW event lasts well under
    a second) contributes negligibly to the estimate, so it's safe to reuse
    the same segment that contains the event."""
    strain = segment.strain
    finite = np.isfinite(strain)
    if not finite.all():
        # NaN gaps (see docs/step2_data_loading.md) would poison the FFT-based
        # Welch estimate; drop them rather than let them propagate as NaN PSD.
        strain = strain[finite]

    nperseg = int(fftlength * segment.sample_rate)
    noverlap = int(overlap * segment.sample_rate)
    freqs, psd = welch(strain, fs=segment.sample_rate, window="hann",
                        nperseg=nperseg, noverlap=noverlap)
    return PSDEstimate(freqs=freqs, psd=psd)


def whiten(strain: np.ndarray, sample_rate: float, psd: PSDEstimate,
           tukey_alpha: float = 0.25) -> np.ndarray:
    """Whiten a chunk of strain against a (typically longer-baseline) PSD estimate."""
    from scipy.signal.windows import tukey

    n = len(strain)
    dt = 1.0 / sample_rate

    tapered = strain * tukey(n, alpha=tukey_alpha)

    freqs = np.fft.rfftfreq(n, dt)
    hf = np.fft.rfft(tapered)

    psd_at_freqs = psd.interpolator()(freqs)
    psd_at_freqs = np.clip(psd_at_freqs, a_min=np.finfo(float).tiny, a_max=None)

    norm = 1.0 / np.sqrt(1.0 / (2.0 * dt))
    white_hf = hf / np.sqrt(psd_at_freqs) * norm

    return np.fft.irfft(white_hf, n=n)


def bandpass(strain: np.ndarray, sample_rate: float, low: float = 35.0, high: float = 350.0,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass (filtfilt avoids the timing smear a
    causal IIR filter would introduce)."""
    nyquist = sample_rate / 2.0
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, strain)


def preprocess_segment(full_segment: StrainSegment, window_segment: StrainSegment,
                        low: float = 35.0, high: float = 350.0) -> np.ndarray:
    """Estimate the PSD from `full_segment` (long baseline, e.g. the whole
    4096s file) and use it to whiten + bandpass `window_segment` (the short
    chunk of interest, e.g. a few seconds around a merger)."""
    psd = estimate_psd(full_segment)
    whitened = whiten(window_segment.strain, window_segment.sample_rate, psd)
    return bandpass(whitened, window_segment.sample_rate, low=low, high=high)


if __name__ == "__main__":
    from gwosc import datasets
    from ligo_loader import load_event_strain

    event = "GW150914"
    merger_gps = datasets.event_gps(event)

    processed = {}
    for detector in ("H1", "L1"):
        full = load_event_strain(event, detector=detector)
        window = full.slice_gps(merger_gps, half_width=2.0)
        processed[detector] = (window, preprocess_segment(full, window))

        print(f"{detector}: whitened+bandpassed std={processed[detector][1].std():.3f} "
              f"(expect ~O(1) for in-band whitened noise)")

    print("\nSaving comparison plot...")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, detector in zip(axes, ("H1", "L1")):
        window, clean = processed[detector]
        t = window.times() - merger_gps
        # crop tapered edges (first/last 10% distorted by the Tukey window)
        crop = int(0.1 * len(t))
        ax.plot(t[crop:-crop], clean[crop:-crop], linewidth=0.8, color="#1f6feb")
        ax.axvline(0.0, color="#d1242f", linestyle="--", linewidth=1, label="merger")
        ax.set_ylabel(f"{detector} whitened strain")
        ax.set_xlim(-0.2, 0.1)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time from merger (s)")
    fig.suptitle(f"{event} — whitened + bandpassed (35-350 Hz)")
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parents[2] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{event}_whitened_chirp.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
