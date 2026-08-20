# Step 3 — Whitening, PSD Estimation, Bandpass Filtering

## Math

See the module docstring in
[`src/data/preprocessor.py`](../src/data/preprocessor.py). Summary:

- **PSD estimation** — Welch's method (`scipy.signal.welch`) on a long
  baseline (the full 4096s file), 4s segments / 2s overlap / Hann window.
  A ~1s transient contributes negligibly to a 4096s average, so it's safe to
  estimate the PSD from the same file that contains the event.
- **Whitening** — `X_white(f) = X(f) / sqrt(S_n(f))`, normalized by
  `1/sqrt(1/(2*dt))`. This normalization matches the convention used in the
  official GWOSC/LOSC open-data tutorials for this exact dataset.
- **Tapering** — a Tukey window (`alpha=0.25`) is applied to the chunk before
  its FFT, to avoid spectral leakage from hard edges. The tapered ~10% at
  each edge is cropped from any downstream plot/analysis.
- **Bandpass** — zero-phase Butterworth (`scipy.signal.filtfilt`), 35-350 Hz,
  order 4. Zero-phase matters: an ordinary causal IIR filter would shift the
  chirp in time, which would corrupt any timing-sensitive downstream use
  (multi-detector time-of-arrival, parameter estimation).

## Validation against GW150914

`src/data/preprocessor.py` run directly: whitens+bandpasses a ±2s window
around the real GW150914 merger for both H1 and L1, independently estimating
each detector's PSD from its own full 4096s file. Result
(`outputs/GW150914_whitened_chirp.png`): a clear burst of increasing
amplitude and frequency peaking within milliseconds of the true merger GPS
time in **both** detectors independently, followed by ringdown — visually
consistent with the published GW150914 waveform (Abbott et al. 2016). This
is the same signal that was **not** visible by eye in the raw strain plot
from Step 2.

## Tests

[`tests/test_preprocessing.py`](../tests/test_preprocessing.py):

- Whitening flattens a synthetic colored-noise spectrum by >10x (max/min PSD
  ratio in-band), independent of any real data.
- Bandpass removes a tone below the passband, keeps one inside it.
- Bandpass is confirmed zero-phase by comparing a filtered impulse's peak
  location against a causal `lfilter` on the same input.
- On real GW150914 H1 data: the whitened+bandpassed peak amplitude lands
  within 50ms of the true merger GPS time — a quantitative version of "the
  chirp is visible."
