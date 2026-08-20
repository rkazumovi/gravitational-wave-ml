"""
Real LIGO/Virgo strain data loading from GWOSC.

GWOSC strain HDF5 file layout (verified against the real GW150914 H1 file,
see docs/step1_environment_setup.md and docs/step2_data_loading.md):

    strain/Strain               float64[N]   raw dimensionless strain h(t)
        .attrs['Xstart']        GPS time of sample 0 (seconds)
        .attrs['Xspacing']      1 / sample_rate (seconds)
    meta/GPSstart, meta/Duration, meta/Detector, meta/Observatory
    quality/simple/DQmask       uint32[Duration]   1 Hz data-quality bitmask
    quality/simple/DQShortnames bit names, in bit order: DATA, CBC_CAT1,
        CBC_CAT2, CBC_CAT3, BURST_CAT1, BURST_CAT2, BURST_CAT3
    quality/injections/Injmask  uint32[Duration]   1 Hz hardware-injection bitmask

A second is "CBC analysis-ready" when bits 0-2 (DATA, CBC_CAT1, CBC_CAT2) are
all set, i.e. ``mask & 0b0111 == 0b0111``.

No gwpy dependency (see docs/step1_environment_setup.md for why) — files are
resolved via `gwosc`, downloaded with `requests`, and read directly with h5py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data"

# Bit order is fixed by the GWOSC file format (see module docstring).
DQ_BIT_NAMES = (
    "DATA",
    "CBC_CAT1",
    "CBC_CAT2",
    "CBC_CAT3",
    "BURST_CAT1",
    "BURST_CAT2",
    "BURST_CAT3",
)
CBC_ANALYSIS_READY_MASK = 0b0111  # DATA | CBC_CAT1 | CBC_CAT2


@dataclass
class StrainSegment:
    """A contiguous strain time series from one detector, with GPS timing and
    data-quality metadata carried alongside so callers never have to
    re-derive them."""

    detector: str
    gps_start: float
    sample_rate: float
    strain: np.ndarray
    dq_mask: np.ndarray  # 1 Hz, length = duration in seconds
    event: str | None = None

    @property
    def duration(self) -> float:
        return len(self.strain) / self.sample_rate

    @property
    def gps_end(self) -> float:
        return self.gps_start + self.duration

    def times(self) -> np.ndarray:
        """GPS timestamp of every strain sample."""
        return self.gps_start + np.arange(len(self.strain)) / self.sample_rate

    def is_analysis_ready(self, gps_time: float) -> bool:
        """Whether the 1-second interval containing `gps_time` passes the
        standard CBC data-quality cut (DATA & CBC_CAT1 & CBC_CAT2)."""
        idx = int(gps_time - self.gps_start)
        if not (0 <= idx < len(self.dq_mask)):
            raise ValueError(
                f"gps_time={gps_time} outside segment [{self.gps_start}, {self.gps_end})"
            )
        return bool(self.dq_mask[idx] & CBC_ANALYSIS_READY_MASK == CBC_ANALYSIS_READY_MASK)

    def slice_gps(self, gps_center: float, half_width: float) -> "StrainSegment":
        """Return a new StrainSegment trimmed to [gps_center - half_width,
        gps_center + half_width)."""
        start_gps = gps_center - half_width
        end_gps = gps_center + half_width
        if start_gps < self.gps_start or end_gps > self.gps_end:
            raise ValueError(
                f"requested window [{start_gps}, {end_gps}) exceeds segment "
                f"[{self.gps_start}, {self.gps_end})"
            )
        start_idx = int(round((start_gps - self.gps_start) * self.sample_rate))
        n_samples = int(round(2 * half_width * self.sample_rate))
        end_idx = start_idx + n_samples

        dq_start = int(start_gps - self.gps_start)
        dq_end = int(np.ceil(end_gps - self.gps_start))

        return replace(
            self,
            gps_start=self.gps_start + start_idx / self.sample_rate,
            strain=self.strain[start_idx:end_idx],
            dq_mask=self.dq_mask[dq_start:dq_end],
        )


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    os.replace(tmp, dest)
    return dest


def get_event_file_urls(event: str, detector: str, duration: int = 4096) -> list[str]:
    from gwosc.locate import get_event_urls

    urls = get_event_urls(event, detector=detector, duration=duration, format="hdf5")
    if not urls:
        raise ValueError(f"no strain files found for event={event} detector={detector}")
    return urls


def load_strain_file(path: Path, event: str | None = None) -> StrainSegment:
    with h5py.File(path, "r") as f:
        strain_ds = f["strain/Strain"]
        gps_start = float(strain_ds.attrs["Xstart"])
        sample_rate = round(1.0 / float(strain_ds.attrs["Xspacing"]))
        strain = strain_ds[()].astype(np.float64)
        dq_mask = f["quality/simple/DQmask"][()]
        detector = f["meta/Detector"][()].decode()

    return StrainSegment(
        detector=detector,
        gps_start=gps_start,
        sample_rate=sample_rate,
        strain=strain,
        dq_mask=dq_mask,
        event=event,
    )


def load_event_strain(
    event: str,
    detector: str = "H1",
    duration: int = 4096,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> StrainSegment:
    """Download (if not already cached) and load the real strain data GWOSC
    published for a confirmed GW event, e.g. load_event_strain("GW150914")."""
    urls = get_event_file_urls(event, detector=detector, duration=duration)
    url = urls[0]
    dest = Path(cache_dir) / Path(url).name
    _download(url, dest)
    return load_strain_file(dest, event=event)


if __name__ == "__main__":
    from gwosc import datasets

    event = "GW150914"
    merger_gps = datasets.event_gps(event)
    print(f"{event} merger GPS time: {merger_gps}")

    for detector in ("H1", "L1"):
        print(f"\n--- {detector} ---")
        segment = load_event_strain(event, detector=detector)
        print(f"Loaded {len(segment.strain):,} samples at {segment.sample_rate} Hz")
        print(f"Segment spans GPS [{segment.gps_start}, {segment.gps_end})")
        print(f"Strain dtype={segment.strain.dtype}, "
              f"min={segment.strain.min():.3e}, max={segment.strain.max():.3e}, "
              f"std={segment.strain.std():.3e}")

        ready = segment.is_analysis_ready(merger_gps)
        print(f"Analysis-ready (DATA & CBC_CAT1 & CBC_CAT2) at merger time: {ready}")
        assert ready, "expected GW150914 to pass standard CBC data-quality cuts"

        window = segment.slice_gps(merger_gps, half_width=16.0)
        print(f"Sliced {window.duration:.1f}s window around merger: "
              f"{len(window.strain):,} samples")

    print("\nGW150914 strain loading verified for H1 and L1.")
