"""
Restricted post-Newtonian (leading order / 0PN) inspiral waveform generator.

No PyCBC/LALSuite (TensorFlow/SciPy-only project constraint) — this is a
from-scratch implementation of the classic "Newtonian quadrupole" chirp,
integrated in closed form from the same frequency-evolution ODE used
elsewhere in this project for the chirp-mass PINN loss:

    df/dt = (96/5) pi^(8/3) (G*Mc/c^3)^(5/3) * f^(11/3)

This is separable; solving with the boundary condition f -> infinity as
t -> t_c (coalescence time) gives closed-form frequency and phase (derived
by hand, see docs/step4_waveforms.md for the full derivation and a check
that dPhi/dt = 2*pi*f holds exactly):

    f(t)   = (1/pi) * (5/256)^(3/8) * (G*Mc/c^3)^(-5/8) * (t_c - t)^(-3/8)
    Phi(t) = phi_c - 2 * [ (t_c - t) / (5*G*Mc/c^3) ]^(5/8)

with restricted (leading-order, quadrupole) amplitude:

    A(t) = (4/d_L) * (G*Mc/c^2)^(5/3) * (pi*f(t)/c)^(2/3)
    h_plus  = A(t) * cos(Phi(t))   [face-on / inclination-averaged factor folded in below]
    h_cross = A(t) * sin(Phi(t))

Valid only for the inspiral, up to the Schwarzschild test-particle ISCO
frequency of the total mass:

    f_isco = c^3 / (6^1.5 * pi * G * M_total)  ~=  4403 Hz * (Msun / M_total)

This underestimates the true numerical-relativity merger frequency for
comparable-mass binaries (e.g. ~67 Hz vs the true ~250 Hz peak for
GW150914-like masses) — a known limitation of the leading-order
approximation, not a bug. It's adequate for generating realistic *inspiral*
chirps to inject into real noise for detector training data; it is not a
merger/ringdown model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 6.674e-11          # m^3 kg^-1 s^-2
C = 2.998e8             # m/s
M_SUN = 1.989e30         # kg
MPC = 3.0857e22          # m


def chirp_mass(m1_msun: float, m2_msun: float) -> float:
    """Chirp mass Mc = (m1*m2)^(3/5) / (m1+m2)^(1/5), in solar masses."""
    return (m1_msun * m2_msun) ** 0.6 / (m1_msun + m2_msun) ** 0.2


def schwarzschild_isco_frequency(total_mass_msun: float) -> float:
    """GW frequency (2x orbital) at the Schwarzschild test-particle ISCO."""
    m_total_kg = total_mass_msun * M_SUN
    return C ** 3 / (6 ** 1.5 * np.pi * G * m_total_kg)


def inspiral_frequency(t: np.ndarray, t_coalescence: float, mc_msun: float) -> np.ndarray:
    """Closed-form GW frequency f(t) for t < t_coalescence. Returns NaN at/after
    t_coalescence, where the leading-order approximation diverges."""
    mc_kg = mc_msun * M_SUN
    tau = t_coalescence - t
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (1.0 / np.pi) * (5.0 / 256.0) ** (3.0 / 8.0) * (G * mc_kg / C ** 3) ** (-5.0 / 8.0) * tau ** (-3.0 / 8.0)
    return np.where(tau > 0, f, np.nan)


def inspiral_phase(t: np.ndarray, t_coalescence: float, mc_msun: float, phase_c: float = 0.0) -> np.ndarray:
    """Closed-form GW phase Phi(t), with dPhi/dt = 2*pi*f(t) by construction."""
    mc_kg = mc_msun * M_SUN
    tau = t_coalescence - t
    tc_scale = 5.0 * G * mc_kg / C ** 3
    with np.errstate(invalid="ignore"):
        phi = phase_c - 2.0 * (tau / tc_scale) ** (5.0 / 8.0)
    return np.where(tau > 0, phi, np.nan)


def restricted_pn_amplitude(t: np.ndarray, t_coalescence: float, mc_msun: float,
                             distance_mpc: float) -> np.ndarray:
    mc_kg = mc_msun * M_SUN
    d_l = distance_mpc * MPC
    f = inspiral_frequency(t, t_coalescence, mc_msun)
    return (4.0 / d_l) * (G * mc_kg / C ** 2) ** (5.0 / 3.0) * (np.pi * f / C) ** (2.0 / 3.0)


@dataclass
class Waveform:
    t: np.ndarray          # seconds, relative to an arbitrary zero (not t_coalescence)
    h_plus: np.ndarray
    h_cross: np.ndarray
    frequency: np.ndarray  # instantaneous GW frequency, Hz
    t_coalescence: float
    chirp_mass_msun: float
    total_mass_msun: float
    f_isco: float


def generate_waveform(m1_msun: float, m2_msun: float, distance_mpc: float,
                       sample_rate: float, t_coalescence: float = 0.0,
                       f_lower: float = 20.0, inclination: float = 0.0,
                       phase_c: float = 0.0) -> Waveform:
    """Generate a restricted-PN inspiral chirp from f_lower up to the
    Schwarzschild ISCO frequency of (m1+m2), sampled at sample_rate."""
    mc = chirp_mass(m1_msun, m2_msun)
    total_mass = m1_msun + m2_msun
    f_isco = schwarzschild_isco_frequency(total_mass)
    if f_lower >= f_isco:
        raise ValueError(f"f_lower={f_lower} Hz is above f_isco={f_isco:.1f} Hz for this total mass")

    # Time at which f(t) == f_lower, from inverting the closed-form f(t):
    # tau = (1/pi * (5/256)^(3/8) * (G*Mc/c^3)^(-5/8) / f_lower)^(8/3)
    mc_kg = mc * M_SUN
    tau_lower = ((1.0 / np.pi) * (5.0 / 256.0) ** (3.0 / 8.0) *
                 (G * mc_kg / C ** 3) ** (-5.0 / 8.0) / f_lower) ** (8.0 / 3.0)
    t_start = t_coalescence - tau_lower

    # Stop a fraction of a sample before t_coalescence to avoid the f->inf singularity;
    # f_isco naturally bounds how close we get since the approximation is only valid until then.
    tau_isco = ((1.0 / np.pi) * (5.0 / 256.0) ** (3.0 / 8.0) *
                (G * mc_kg / C ** 3) ** (-5.0 / 8.0) / f_isco) ** (8.0 / 3.0)
    t_end = t_coalescence - tau_isco

    n_samples = max(2, int(round((t_end - t_start) * sample_rate)))
    t = t_start + np.arange(n_samples) / sample_rate

    freq = inspiral_frequency(t, t_coalescence, mc)
    phase = inspiral_phase(t, t_coalescence, mc, phase_c=phase_c)
    amp = restricted_pn_amplitude(t, t_coalescence, mc, distance_mpc)

    cos_i = np.cos(inclination)
    h_plus = amp * 0.5 * (1 + cos_i ** 2) * np.cos(phase)
    h_cross = amp * cos_i * np.sin(phase)

    return Waveform(t=t, h_plus=h_plus, h_cross=h_cross, frequency=freq,
                     t_coalescence=t_coalescence, chirp_mass_msun=mc,
                     total_mass_msun=total_mass, f_isco=f_isco)


if __name__ == "__main__":
    # GW150914-like source-frame masses (Abbott et al. 2016): ~36 + 29 Msun.
    m1, m2, distance = 36.0, 29.0, 410.0
    mc = chirp_mass(m1, m2)
    f_isco = schwarzschild_isco_frequency(m1 + m2)
    print(f"m1={m1} Msun, m2={m2} Msun -> chirp mass = {mc:.2f} Msun "
          f"(published GW150914 source-frame value: ~28.6 Msun)")
    print(f"f_isco = {f_isco:.1f} Hz (test-particle Schwarzschild ISCO; "
          f"true NR merger frequency is higher, ~250 Hz)")

    wf = generate_waveform(m1, m2, distance, sample_rate=4096.0, t_coalescence=0.0, f_lower=20.0)
    print(f"Generated {len(wf.t)} samples, duration={wf.t[-1]-wf.t[0]:.4f}s, "
          f"frequency sweep {wf.frequency[0]:.1f} -> {wf.frequency[-1]:.1f} Hz")
    assert np.all(np.diff(wf.frequency) > 0), "frequency must increase monotonically"
    print("Frequency increases monotonically: OK")

    # --- Validate the frequency track against real GW150914 data ---
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from ligo_loader import load_event_strain  # noqa: E402
    from preprocessor import preprocess_segment  # noqa: E402
    from gwosc import datasets

    merger_gps = datasets.event_gps("GW150914")
    full = load_event_strain("GW150914", detector="H1")
    window = full.slice_gps(merger_gps, half_width=0.5)
    clean = preprocess_segment(full, window, low=20.0, high=300.0)
    t_real = window.times() - merger_gps

    from scipy.signal import spectrogram

    freqs, times, sxx = spectrogram(clean, fs=window.sample_rate, nperseg=192, noverlap=188)
    times = times - 0.5  # spectrogram times are relative to the start of `clean`

    # Detector-frame chirp mass is redshifted relative to the source-frame
    # value above (GW150914 z~0.09): use it here so the analytic track lines
    # up with real (detector-frame) observed frequencies.
    mc_det = mc * 1.09
    t_model = np.linspace(-0.15, -1.0 / f_isco * 0.0 - 1e-4, 400)
    # model time array up to just before t_c=0 (merger), starting at -0.15s
    t_model = np.linspace(-0.15, -0.002, 400)
    f_model = inspiral_frequency(t_model, t_coalescence=0.0, mc_msun=mc_det)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = 10 * np.log10(sxx + 1e-30)
    # Color scale from the displayed region only, and clipped by percentile:
    # the tapered edges (outside the display window) have huge, irrelevant
    # dynamic range that would otherwise wash out the in-band contrast.
    view = (times >= -0.15) & (times <= 0.05) & True
    freq_view = (freqs >= 20) & (freqs <= 300)
    db_view = db[np.ix_(freq_view, view)]
    vmin, vmax = np.percentile(db_view, [5, 99])

    fig, ax = plt.subplots(figsize=(9, 5))
    pcm = ax.pcolormesh(times, freqs, db, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.plot(t_model, f_model, color="red", linewidth=2, label="analytic restricted-PN f(t)")
    ax.axvline(0.0, color="white", linestyle="--", linewidth=1, label="merger")
    ax.set_ylim(20, 300)
    ax.set_xlim(-0.15, 0.05)
    ax.set_xlabel("Time from merger (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("GW150914 H1: real spectrogram vs analytic inspiral track")
    ax.legend(loc="upper left")
    fig.colorbar(pcm, ax=ax, label="power (dB)")
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parents[2] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "GW150914_waveform_validation.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
