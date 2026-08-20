"""
Full training run for the CNN+BiLSTM detector (src/models/detector.py),
consuming the pre-generated example pool (src/data/cache_dataset.py) rather
than generating examples on the fly -- see docs/step8_detector.md for why
on-the-fly generation is impractical for a full multi-epoch training run on
CPU-only hardware (data generation costs ~1.75s/example vs. ~0.06s/batch for
the model's own forward+backward pass once warmed up).

Final validation step trains the model, then evaluates it on the REAL
GW150914 event itself (not held-out synthetic data) -- the same
validate-against-real-data standard used throughout this project.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from detector import build_detector  # noqa: E402
from cache_dataset import build_cached_dataset, load_example_pool  # noqa: E402


def train(cache_path: Path | str, epochs: int = 50, batch_size: int = 32,
          val_fraction: float = 0.1, output_model_path: Path | str | None = None) -> keras.Model:
    pool = load_example_pool(cache_path)
    n_total = len(pool["label"])
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    steps_per_epoch = max(1, n_train // batch_size)
    val_steps = max(1, n_val // batch_size)

    train_ds, val_ds = build_cached_dataset(cache_path, batch_size=batch_size, val_fraction=val_fraction)
    train_ds = train_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))
    val_ds = val_ds.map(lambda x, y: (tf.expand_dims(x, -1), y)).repeat()

    input_length = pool["strain"].shape[1]
    model = build_detector(input_length=input_length)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8,
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=4),
    ]

    print(f"\nTraining on {n_train} examples ({steps_per_epoch} steps/epoch), "
          f"validating on {n_val} examples ({val_steps} steps), up to {epochs} epochs...")
    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds,
                         validation_steps=val_steps, epochs=epochs, callbacks=callbacks, verbose=2)

    best_val_auc = max(history.history["val_auc"])
    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    if output_model_path is not None:
        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_model_path)
        print(f"Saved model to {output_model_path}")

    return model


def validate_on_real_gw150914(model: keras.Model, window_seconds: float = 4.0) -> float:
    """The ultimate real-data check: run the trained model on the real
    whitened GW150914 strain (containing the actual event) and report the
    predicted signal probability. Not a substitute for held-out validation
    metrics -- a single example proves nothing statistically -- but it's the
    same standard applied to every other component in this project."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from ligo_loader import load_event_strain
    from preprocessor import preprocess_segment
    from gwosc import datasets

    merger_gps = datasets.event_gps("GW150914")
    full = load_event_strain("GW150914", detector="H1")
    window = full.slice_gps(merger_gps, half_width=window_seconds / 2.0)
    clean = preprocess_segment(full, window, low=35.0, high=350.0).astype(np.float32)

    x = clean.reshape(1, -1, 1)
    prob = float(model.predict(x, verbose=0)[0, 0])
    print(f"Model's predicted signal probability on real GW150914 H1 data: {prob:.4f}")
    return prob


if __name__ == "__main__":
    cache_path = Path(__file__).resolve().parents[2] / "outputs" / "training_pool.npz"
    if not cache_path.exists():
        raise SystemExit(f"{cache_path} not found -- run src/data/cache_dataset.py first "
                          f"to generate the training pool.")

    model_path = Path(__file__).resolve().parents[2] / "outputs" / "detector_model.keras"
    model = train(cache_path, epochs=50, batch_size=32, output_model_path=model_path)
    validate_on_real_gw150914(model)
