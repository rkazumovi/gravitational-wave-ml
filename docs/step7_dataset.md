# Step 7 — TensorFlow Dataset Pipeline

## Design

[`src/data/dataset.py`](../src/data/dataset.py) ties the previous steps
together: `build_dataset` wraps a Python generator (drawing on
`augmentor.make_positive_example`/`make_negative_example`, then
`preprocessor.preprocess_segment` for whitening/bandpass) in
`tf.data.Dataset.from_generator`, batched and prefetched. The generator is
`while True` — effectively infinite — so a model trains against endless
fresh variety (new random noise windows, new random injection parameters)
rather than a fixed cached set.

## A real gotcha found and fixed: seeded generators + dataset restarts

`tf.data.Dataset.from_generator` re-invokes the underlying Python generator
function **from scratch** every time the `Dataset` is iterated from the
start — a fresh `for batch in ds:` loop, or what Keras' `model.fit` does at
the start of each epoch unless the dataset is consumed as one continuous
stream. Verified directly: with a fixed seed, two separate `ds.take(2)`
passes over the same dataset returned bit-identical batches.

For training that would be a serious, silent problem: every epoch would see
the *exact same* synthetic examples, defeating the entire point of
unlimited augmentation, and the bug would produce no error — just
suspiciously-fast overfitting a user might not immediately connect to the
data pipeline. Fixed by defaulting `build_dataset(..., seed=None)`, which
draws fresh OS entropy (`np.random.default_rng(None)`) each time the
generator restarts. An explicit int seed is still supported and useful for
reproducible tests, where restart-identical output is exactly what's
wanted.

## Validation

- Batch shapes/dtypes match the requested `window_seconds`/`batch_size`
  (`float32`, labels in `{0, 1}`).
- Label balance across several batches is roughly consistent with
  `positive_fraction`.
- `seed=<int>`: two restarts give identical label sequences (intended,
  reproducible-test behavior).
- `seed=None` (default): two restarts give different strain data (confirms
  the fix above; compared on continuous strain values rather than the
  handful of binary labels per batch, to avoid a flaky coincidental match).

## Tests

[`tests/test_dataset.py`](../tests/test_dataset.py) — 4 tests, including
direct regression tests for both the fixed-seed and default-seed restart
behavior described above.
