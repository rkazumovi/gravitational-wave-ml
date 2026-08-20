# Step 5 — Matched Filtering, Optimal SNR, and Signal Injection Control

## Math

See [`src/dsp/matched_filter.py`](../src/dsp/matched_filter.py) docstring.
Summary: the optimal SNR of a template `h(t)` against noise PSD `S_n(f)` is
`rho^2 = 4 Integral_0^inf |h~(f)|^2 / S_n(f) df` — a property of the template
and noise alone, used to inject training signals at a controlled SNR. The
matched-filter statistic `z(tau) = 4 Re Integral_0^inf [d~(f) h~*(f)/S_n(f)]
e^(2pi i f tau) df`, normalized by `sigma = sqrt(rho^2 of the template)`,
searches for the best-fit time-of-arrival offset between a template and a
data segment. Implemented with `np.fft.rfft`/`irfft` (correct real-signal
Hermitian bookkeeping, chosen after a full-FFT version proved error-prone —
see bugs below).

## Three real bugs found and fixed during development

This module is a good example of why "derive it, then verify it
numerically against a known-correct property" matters more than trusting
the algebra. All three were caught by contradictions between independent
computations, not by inspection.

**Bug 1 — full-FFT normalization was off by a factor of 2.** A first
implementation used `np.fft.fft`/`ifft` (two-sided) and band-masked with
`np.abs(freqs)`, which folds +f and -f bins onto the same one-sided PSD
value without accounting for the implied doubling. Caught by the
fundamental correctness property of a matched filter: filtering a template
against *itself, unshifted* (no noise) must return exactly that template's
own optimal SNR (the Cauchy-Schwarz equality case) — this returned `2x
sigma` instead. Fixed by rebuilding on `rfft`/`irfft`, which handles the
Hermitian symmetry of a real signal's spectrum correctly, and re-deriving
the time-domain normalization until the self-match property held to
machine precision (now
[`tests/test_matched_filter.py::test_self_match_snr_equals_own_optimal_snr`](../tests/test_matched_filter.py)).

**Bug 2 — full-band (0-Nyquist) integration is unsafe.** Zero-padding a
template into a longer array (needed to align it with a strain segment)
changed its measured optimal SNR by >2x, purely from lengthening the array.
Cause: a synthetic test PSD (steep lowpass at 300 Hz) had a Welch estimate
dropping to ~1e-26 near Nyquist; the padded template's finer frequency grid
happened to land bins there, and dividing negligible floating-point spectral
leakage by a near-zero PSD dominated the integral. Fixed by restricting
every overlap integral to an explicit `[f_low, f_high]` band — standard
practice in real pipelines (e.g. LIGO analyses integrate ~20-2048 Hz, never
0-Nyquist), now enforced everywhere in this module. Regression test:
[`test_optimal_snr_is_invariant_to_zero_padding`](../tests/test_matched_filter.py).

**Bug 3 — real strain needs time-domain band-limiting, not just a frequency
mask.** After fixing bugs 1-2, searching real GW150914 H1 data gave a peak
SNR of **~700** — vs. GW150914's published network SNR of ~24, a wildly
nonphysical result. Root cause: raw strain has enormous seismic-noise power
below ~20 Hz (measured: ~500x the 30-300 Hz band's power), and taking a
plain FFT of a hard-edged, un-tapered segment lets that power leak into
in-band bins via sinc sidelobes — even though those bins were themselves
excluded by the frequency-domain band mask, the *leaked* contribution from
outside the mask still landed inside it. Confirmed by directly comparing
in-band FFT power with vs. without a Tukey taper (dropped ~67x once
tapered). Fixed properly by zero-phase bandpassing both the strain and the
template in the *time domain* (reusing the already-validated
`preprocessor.bandpass`) before ever taking an FFT — this removes the
leakage source rather than approximating around it. Regression test:
[`test_real_gw150914_search_peak_is_physically_sensible`](../tests/test_matched_filter.py).

## Validation

- **Synthetic round-trip**: inject a waveform scaled to a known target SNR
  (20) into 5 independent colored-noise realizations; matched filter
  recovers a peak within a few samples of the true injection time and mean
  SNR 18.7 ± 1.2 — statistically consistent with the target.
- **Real GW150914 H1 search**: matched-filtered real strain against a
  restricted-PN template (detector-frame masses from Step 4, real 410 Mpc
  distance) — peak SNR **21.95** landing **39 ms** from the true merger GPS
  time, more than 40x the background median, and comparable to LIGO's own
  published single-detector SNR (~19-24) despite this being a deliberately
  simplified, inspiral-only, single-band template
  (`outputs/GW150914_matched_filter_snr.png`).

## Tests

[`tests/test_matched_filter.py`](../tests/test_matched_filter.py) — 5 tests:
padding invariance, self-match exactness, injection time/SNR recovery on
synthetic noise, `scale_to_target_snr` self-consistency, and the real-data
regression test above. 25/25 tests passing project-wide.
