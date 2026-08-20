"""
CNN + Bidirectional LSTM hybrid for gravitational-wave signal detection:
binary classification of a whitened strain window as "contains a GW chirp"
vs. "noise only".

Architecture rationale:

    A raw 4s window at 4096 Hz is 16384 samples -- far too long to feed an
    LSTM directly (vanishing gradients over that many timesteps, and the
    per-step compute is prohibitive). The standard approach in GW deep
    learning (e.g. George & Huerta 2018) is a strided 1D CNN stack that
    acts as a learned, trainable analogue of a spectrogram/filter bank,
    collapsing the sample-level time axis down to a much shorter sequence
    of high-level feature vectors, THEN handing that short sequence to a
    recurrent layer for temporal context across the chirp's evolution
    (frequency sweep, amplitude growth) that pure convolution's limited
    receptive field wouldn't capture as naturally.

    This model: 3 blocks of (Conv1D -> BatchNorm -> ReLU -> Conv1D ->
    BatchNorm -> ReLU -> MaxPool), each halving-or-more the time axis and
    increasing channel depth, followed by a Bidirectional LSTM over the
    resulting short sequence (bidirectional because "does this window
    contain a chirp" doesn't have a causality constraint -- the whole
    window is available at inference time), then a small dense head with
    dropout for regularization, ending in a single sigmoid unit.

    16384 samples -> (CNN stack) -> ~64 timesteps x 64 channels ->
    BiLSTM(32) -> Dense(32) -> Dense(1, sigmoid)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_detector(input_length: int = 16384, learning_rate: float = 1e-3) -> keras.Model:
    inputs = keras.Input(shape=(input_length, 1), name="whitened_strain")

    x = inputs
    for filters, kernel, pool in [(8, 16, 4), (16, 8, 4), (32, 8, 4), (64, 4, 4)]:
        x = layers.Conv1D(filters, kernel, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(filters, kernel, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool1D(pool)(x)

    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="signal_probability")(x)

    model = keras.Model(inputs, outputs, name="gw_cnn_bilstm_detector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from dataset import build_dataset  # noqa: E402
    from ligo_loader import load_event_strain  # noqa: E402
    from preprocessor import estimate_psd  # noqa: E402
    from gwosc import datasets

    model = build_detector(input_length=16384)
    model.summary()

    merger_gps = datasets.event_gps("GW150914")
    exclude = ((merger_gps - 60.0, merger_gps + 60.0),)
    full = load_event_strain("GW150914", detector="H1")
    psd = estimate_psd(full, fftlength=4.0, overlap=2.0)

    ds = build_dataset([full], [psd], window_seconds=4.0, batch_size=8, seed=123,
                        excluded_gps_ranges=exclude)
    ds = ds.map(lambda x, y: (tf.expand_dims(x, -1), y))

    print("\nSmoke-testing training on a handful of real-noise batches "
          "(NOT a full training run -- see docs/step8_detector.md for why)...")
    history = model.fit(ds, steps_per_epoch=10, epochs=2, verbose=2)

    losses = history.history["loss"]
    assert all(np.isfinite(losses)), "loss must stay finite"
    print(f"\nLoss trajectory: {losses}")
    print("Smoke test complete: model trains without error on the real dataset pipeline.")
