# Step 2 — Real LIGO Data Loading

## Format

See the docstring in [`src/data/ligo_loader.py`](../src/data/ligo_loader.py) for
the verified GWOSC HDF5 layout. Summary: `strain/Strain` is the raw strain
h(t) time series (float64, 4096 Hz for the O1-era files used here), with GPS
timing in the dataset attributes, and `quality/simple/DQmask` is a 1 Hz
bitmask of data-quality flags (`DATA`, `CBC_CAT1/2/3`, `BURST_CAT1/2/3`).

## Validation against GW150914

Ran [`src/data/ligo_loader.py`](../src/data/ligo_loader.py) directly, which
downloads (and caches under `data/`) the real H1 and L1 strain files GWOSC
published for GW150914 and checks:

- Both files load as 16,777,216 samples at 4096 Hz spanning exactly the
  advertised GPS window.
- Both pass the standard CBC data-quality cut
  (`DATA & CBC_CAT1 & CBC_CAT2`) at the merger GPS time (1126259462.4).
- A ±16 s window can be sliced around the merger from both detectors.

## Finding: NaN gap in the L1 file, correctly flagged by DQmask

The L1 file's raw `strain/Strain` array contains a single contiguous NaN run
of 774,144 samples (189 s) at the very start of the segment (GPS
1126256640.0–1126256829.0) — 2633 seconds away from the merger, so it does
not affect the event window. This is a real data gap (the detector was
presumably not in observing mode), not a bug in the loader.

Checked that `quality/simple/DQmask` correctly reads `0` (no `DATA` bit) for
every second inside that gap, and `127` (full quality) immediately after —
confirming the DQ mask is the authoritative signal for "is this sample
real/usable," not just "is it non-NaN." **Any future code that samples
training windows from the full 4096 s file (`src/data/augmentor.py`,
`src/data/dataset.py`) must check `StrainSegment.is_analysis_ready()` (or the
raw `dq_mask`) before using a window — it cannot assume the strain array is
gap-free just because a given event's own analysis window looks clean.**

## Tests

[`tests/test_data.py`](../tests/test_data.py) covers segment shape/timing,
the data-quality cut at the real merger time, window slicing, and the NaN /
DQmask relationship described above, using the cached real GW150914 files.
