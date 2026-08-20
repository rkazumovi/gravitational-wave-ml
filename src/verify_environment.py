"""
Step 1 — Environment verification.

Confirms every core dependency (TensorFlow, SciPy, Matplotlib, gwosc, h5py,
FastAPI/Plotly stack) imports and performs a minimal real operation, and
confirms live connectivity to the real GWOSC API. Run this after any fresh
`pip install -r requirements.txt` to catch a broken environment before
writing physics or model code against it.
"""
from __future__ import annotations

import sys


def check_tensorflow() -> str:
    import tensorflow as tf

    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    result = tf.matmul(a, b).numpy()
    assert result[0, 0] == 1.0 and result[1, 1] == 4.0
    devices = [d.device_type for d in tf.config.list_physical_devices()]
    gpu_note = " (CPU only - TF>=2.11 dropped native Windows GPU support)" if "GPU" not in devices else ""
    return f"TensorFlow {tf.__version__} OK, devices={devices}{gpu_note}"


def check_scipy() -> str:
    import scipy
    from scipy.signal import butter

    b, a = butter(4, 0.2)
    assert len(b) == 5 and len(a) == 5
    return f"SciPy {scipy.__version__} OK (filter design verified)"


def check_matplotlib() -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 1, 10) ** 2)
    plt.close(fig)
    return f"Matplotlib {matplotlib.__version__} OK (figure rendered)"


def check_h5py() -> str:
    import h5py

    return f"h5py {h5py.__version__} OK"


def check_gwosc(live: bool = True) -> str:
    import gwosc

    if not live:
        return f"gwosc {gwosc.__version__} OK (import only, live check skipped)"

    from gwosc import datasets
    from gwosc.locate import get_event_urls

    gps = datasets.event_gps("GW150914")
    urls = get_event_urls("GW150914")
    assert gps > 0 and len(urls) > 0
    return f"gwosc {gwosc.__version__} OK, live GWOSC query succeeded (GW150914 GPS={gps}, {len(urls)} file(s))"


def check_web_stack() -> str:
    import fastapi
    import plotly
    from prometheus_client import Counter

    Counter("verify_env_counter", "sanity check").inc()
    return f"FastAPI {fastapi.__version__}, Plotly {plotly.__version__}, prometheus_client OK"


def main(live: bool = True) -> None:
    checks = [
        ("TensorFlow", check_tensorflow),
        ("SciPy", check_scipy),
        ("Matplotlib", check_matplotlib),
        ("h5py", check_h5py),
        ("gwosc", lambda: check_gwosc(live=live)),
        ("Web/monitoring stack", check_web_stack),
    ]

    print(f"Python {sys.version}\n")
    failures = []
    for name, fn in checks:
        try:
            print(f"[OK] {fn()}")
        except Exception as exc:  # noqa: BLE001 - report every failure, don't stop early
            print(f"[FAIL] {name}: {exc}")
            failures.append(name)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All environment checks passed.")


if __name__ == "__main__":
    live_check = "--no-network" not in sys.argv
    main(live=live_check)
