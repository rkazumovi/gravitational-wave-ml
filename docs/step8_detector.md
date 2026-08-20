# Step 8 — CNN + BiLSTM Signal Detector

## Architecture

See [`src/models/detector.py`](../src/models/detector.py) docstring. A 4s
window at 4096 Hz is 16384 samples — too long to feed an LSTM directly
(vanishing gradients, prohibitive per-step cost). Following the standard GW
deep-learning pattern (e.g. George & Huerta 2018), a strided 1D CNN stack —
4 blocks of (Conv1D → BatchNorm → ReLU) ×2 → MaxPool — collapses the
sample-level time axis down to 64 timesteps × 64 channels, acting as a
learned filter bank. A Bidirectional LSTM (bidirectional since the whole
window is available at inference — no causality constraint) then captures
temporal structure across that shorter sequence — the chirp's frequency
sweep and amplitude growth — before a small dense head with dropout ends in
a single sigmoid unit. 69,233 parameters total.

## Profiling finding: data generation, not the model, is the bottleneck

A smoke test (training directly against `src/data/dataset.py`'s on-the-fly
generator) trained without error — loss 0.76→0.70 over 2 epochs — but at
~11s/step for a batch of 8. Profiling data generation and model compute
*separately* found:

- Data generation: ~14s/batch of 8 (~1.75s/example)
- Model inference (forward only): ~0.7s/batch
- Model train_on_batch, warmed up (forward+backward): **~0.06s/batch**

The model itself is essentially free on CPU once TF's graph tracing
overhead (paid once, ~6s) is past — the entire bottleneck is per-example
injection (waveform generation, SNR scaling) and whitening in Python/SciPy.
Full multi-epoch training via on-the-fly generation would be impractical:
even a modest few-hundred-step training run would take hours.

## Fix: pre-generated, cached example pool

Standard approach (and closer to what most published GW ML papers actually
do): generate a large, fixed, diverse pool of real-noise + injection
examples once (one-time cost, run in the background), then train
arbitrarily many fast epochs over that in-memory/on-disk pool.
[`src/data/cache_dataset.py`](../src/data/cache_dataset.py) implements
this: `generate_example_pool` builds and saves a `.npz` of whitened strain
arrays, labels, and injection metadata (SNR, chirp mass — NaN for
negatives); `build_cached_dataset` loads it into a fast, shuffled,
`.repeat()`-ing `tf.data.Dataset` with a fixed train/val split. Trade-off
vs. the infinite on-the-fly generator: finite (though large — 2000
examples here, drawn from both real H1 and L1 GW150914 noise, away from
the true event) example variety instead of a fresh draw every step.

## Training

[`src/models/train_detector.py`](../src/models/train_detector.py) trains
against the cached pool with early stopping on validation AUC (patience 8,
restoring best weights), then runs the ultimate real-data check used
throughout this project: the trained model's predicted signal probability
on the real GW150914 H1 strain itself.

## Results

Trained on 1,800 examples (200 held out for validation) from the 2,000-example
cached pool (both H1 and L1 real noise, away from the true event, 50/50
positive/negative). With the data bottleneck removed, each epoch took
5-8 seconds; training ran 23 epochs before early stopping triggered.

| Metric | Value |
|---|---|
| Best validation AUC-ROC | **0.9906** (target: >0.97 — met) |
| Validation accuracy at best epoch | 0.984 |
| Predicted signal probability on real GW150914 H1 | **0.958** |

The real-event check is the meaningful one: GW150914 itself was excluded
from every noise-sampling window used during training (see the
`excluded_gps_ranges` parameter threaded through `augmentor.py` since Step
6) — the model has never seen this specific signal, synthetic or real, and
still correctly classifies it as a detection with high confidence. This is
a single example, not a substitute for the held-out validation set's
statistics, but it's the same real-data validation standard applied to
every other component in this project.

Trained model saved to `outputs/detector_model.keras` (not committed to
git — regenerable via `cache_dataset.py` then `train_detector.py`, see
main README).
