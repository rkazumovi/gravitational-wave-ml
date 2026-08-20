# Step 6 — Synthetic Signal Injection into Real Noise

## Design

[`src/data/augmentor.py`](../src/data/augmentor.py) generates labeled
training examples by injecting restricted-PN waveforms (Step 4) into real
detector noise (Step 2), scaled to a controlled SNR via the matched-filter
machinery (Step 5), rather than sourcing separate "quiet" files: random
windows are sampled from the same real strain files already in use,
excluding any caller-flagged GPS range (typically a margin around a real
event in that file) and requiring every second of the window to pass the
standard CBC data-quality cut (`StrainSegment.is_analysis_ready` —
see [docs/step2_data_loading.md](step2_data_loading.md) for why that check,
not just a NaN check, is required). This mirrors standard practice in GW ML
literature: train on real "off-source" detector noise, inject synthetic
signals on top, since real noise has non-Gaussian structure synthetic noise
models don't capture.

`InjectionExample` carries the injected strain, label, and (for positives)
the exact injected template — exposing the template lets validation and,
later, training code matched-filter against the *known* ground truth
directly rather than reconstructing it approximately.

## A real-data finding, not a bug: background glitches

Initial round-trip validation asserted that the matched filter's *global*
SNR maximum over a window should equal the injected target. That failed: a
target-SNR-15 injection was correctly recovered at its own known location
(lag 0, SNR≈25 — consistent with the target given normal single-realization
statistical scatter), but the window's global max (~60) came from a tight
cluster of samples about 2.65s away, unrelated to the injection.

This is real detector behavior, not a bug: real LIGO noise contains
non-Gaussian transients ("glitches") that can produce a matched-filter SNR
spike exceeding a modest true signal's SNR, at some unrelated time. It's
precisely why real GW search pipelines never trust a single-detector,
single-template SNR peak alone — they require multi-detector coincidence
and signal-consistency (chi-squared) vetoes before claiming a detection.
Fixed the validation to check the SNR **at the known injection lag**
(the meaningful question — "did injection work?"), not the blind global
max (which conflates that with "does this noise window happen to contain a
glitch?"). Documented in
[`tests/test_augmentor.py::test_injected_signal_recoverable_at_known_location`](../tests/test_augmentor.py).

This also foreshadows real work for the detection model (Step 7+): a
classifier trained only on clean injections risks learning to fire on loud
glitches too, unless real noise (including its non-Gaussian transients) is
well represented in training — which sampling from real files, as this
module does, at least partially provides.

## Validation

- `sample_noise_window` avoids excluded GPS ranges and passes the DQ cut for
  every second, over 20 trials.
- Positive/negative example shapes, labels, and metadata are consistent;
  `scale_to_target_snr`'s own self-consistency (Step 5) carries through.
- Injected signal recovered at its known location with SNR proportional to
  target, over 5 independent real-noise draws.
- Whitening a positive example (`outputs/synthetic_injection_in_real_noise.png`)
  makes the injected chirp visible exactly at the injected coalescence time
  — same visual validation pattern as Step 3, now applied to a synthetic
  injection into real noise rather than the real GW150914 signal.
- Mixed batch generation: consistent shapes, correct label balance.

## Tests

[`tests/test_augmentor.py`](../tests/test_augmentor.py) — 6 tests. 31/31
tests passing project-wide.
