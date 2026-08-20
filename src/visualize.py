"""Static Matplotlib figures for sanity-checking pipeline stages against real data."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def plot_raw_strain_around_event(event: str, merger_gps: float, half_width: float = 0.5) -> Path:
    """Plot raw (unwhitened) strain from H1 and L1 in a short window around a
    real event's merger time. Raw strain is noise-dominated below ~100 Hz, so
    the chirp is not expected to be visible by eye yet — that requires the
    whitening filter built in the next step. This just confirms the data
    loads with sane values and correct GPS alignment across detectors."""
    from data.ligo_loader import load_event_strain

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, detector in zip(axes, ("H1", "L1")):
        segment = load_event_strain(event, detector=detector)
        window = segment.slice_gps(merger_gps, half_width=half_width)
        t = window.times() - merger_gps
        ax.plot(t, window.strain, linewidth=0.6, color="#1f6feb")
        ax.axvline(0.0, color="#d1242f", linestyle="--", linewidth=1, label="merger (GPS "
                    f"{merger_gps})")
        ax.set_ylabel(f"{detector} strain")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time from merger (s)")
    fig.suptitle(f"{event} — raw strain (unwhitened)")
    fig.tight_layout()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / f"{event}_raw_strain.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    from gwosc import datasets

    event = "GW150914"
    gps = datasets.event_gps(event)
    path = plot_raw_strain_around_event(event, gps)
    print(f"Saved {path}")
