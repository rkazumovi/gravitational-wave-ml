# Step 1 — Environment Setup (Windows 11, Python 3.13)

## What was done

1. Created a virtual environment with `py -3.13 -m venv venv` (Python 3.13.5, the
   version already on this machine).
2. Installed and individually verified each dependency (see
   [`src/verify_environment.py`](../src/verify_environment.py)):
   - **TensorFlow 2.21.0** — imports, runs a real `tf.matmul`. CPU-only: TensorFlow
     dropped native Windows GPU support after 2.10 (would need WSL2 or the
     DirectML plugin for GPU acceleration — not set up here).
   - **SciPy 1.18.0** — filter design (`scipy.signal.butter`) verified.
   - **Matplotlib 3.11.1** — figure rendering verified with the `Agg` backend.
   - **h5py 3.14.0** — for reading LIGO strain HDF5 files directly.
   - **gwosc 0.8.3** — live-queried the real GWOSC API and resolved GW150914's
     actual strain file URLs.
   - **FastAPI, Plotly, Dash, prometheus_client** — for the later API/dashboard/
     monitoring components.

## Decision: gwosc + manual SciPy pipeline, not gwpy

`gwpy` depends on `igwn-segments`, which compiles a C extension
(`segments.c`, `segment.c`, ...) unconditionally — there is no prebuilt wheel
for Python 3.13 on Windows yet, and no pure-Python fallback. Building it
requires installing Microsoft C++ Build Tools (~6GB, admin rights).

Given the project's own stated philosophy — TensorFlow + SciPy + Matplotlib
as the only ML/scientific libraries, no extra dependencies where avoidable —
we use `gwosc` (pure Python, official GWOSC client, no compiler needed) to
resolve and download real strain files, and implement whitening, bandpass
filtering, PSD estimation, and matched filtering ourselves with
`scipy.signal` / `scipy.fft`. This is consistent with the planned
`src/dsp/whitening.py` and `src/dsp/fourier.py` modules — nothing about the
architecture actually required gwpy's higher-level `TimeSeries` wrapper.

If gwpy is wanted later (e.g. its exact PSD/Q-transform conventions), revisit
once Python 3.13 wheels for `igwn-segments` are published, or install MSVC
Build Tools.

## Bug caught and fixed: `src/signal/` naming collision

The originally planned package `src/signal/` shadows Python's standard
library `signal` module whenever `src/` ends up on `sys.path` (which happens
whenever a script inside `src/` is run directly, per the project's
"every file independently runnable" requirement). This broke `uvicorn` at
import time (`cannot import name 'Signals' from 'signal'`). Renamed to
**`src/dsp/`** (digital signal processing) throughout.

## Reproducing this environment

```bash
py -3.13 -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python src/verify_environment.py
```

Pass `--no-network` to `verify_environment.py` to skip the live GWOSC API
check (e.g. in an offline CI sandbox).

## Verified `pip freeze` snapshot

Direct dependencies are pinned in [`requirements.txt`](../requirements.txt).
All installs and imports above were confirmed working on this machine on
2026-08-20 with Python 3.13.5.
