"""Tests for src/physics/waveforms.py — closed-form restricted-PN inspiral."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "physics"))

from physics.waveforms import (  # noqa: E402
    chirp_mass,
    generate_waveform,
    inspiral_frequency,
    inspiral_phase,
    schwarzschild_isco_frequency,
)


def test_chirp_mass_matches_published_gw150914_value():
    # Published source-frame value ~28.6 Msun (Abbott et al. 2016); our
    # generator uses commonly-cited rounded masses (36 + 29 Msun) so an exact
    # match isn't expected, but it should be close.
    mc = chirp_mass(36.0, 29.0)
    assert mc == pytest.approx(28.6, rel=0.03)


def test_chirp_mass_symmetric_in_masses():
    assert chirp_mass(10.0, 20.0) == pytest.approx(chirp_mass(20.0, 10.0))


def test_isco_frequency_matches_known_scaling_relation():
    # Well-known literature shorthand: f_isco ~= 4400 Hz * Msun / M_total.
    f_isco_1msun = schwarzschild_isco_frequency(1.0)
    assert f_isco_1msun == pytest.approx(4400.0, rel=0.01)
    assert schwarzschild_isco_frequency(65.0) == pytest.approx(4400.0 / 65.0, rel=0.01)


def test_inspiral_frequency_diverges_toward_coalescence():
    t_c = 10.0
    mc = 30.0
    f_early = inspiral_frequency(np.array([0.0]), t_c, mc)[0]
    f_late = inspiral_frequency(np.array([9.99]), t_c, mc)[0]
    assert f_late > f_early > 0


def test_inspiral_frequency_nan_after_coalescence():
    result = inspiral_frequency(np.array([5.0, 15.0]), t_coalescence=10.0, mc_msun=30.0)
    assert result[0] > 0
    assert np.isnan(result[1])


def test_phase_derivative_matches_two_pi_f():
    """dPhi/dt = 2*pi*f(t) must hold by construction (derived by hand, see
    module docstring) — check it numerically against a finite difference."""
    t_c, mc = 10.0, 30.0
    t = np.linspace(0.0, 9.9, 500)
    dt = t[1] - t[0]

    phi = inspiral_phase(t, t_c, mc)
    dphi_dt = np.gradient(phi, dt)

    f = inspiral_frequency(t, t_c, mc)
    expected = 2 * np.pi * f

    # skip the very ends where np.gradient is less accurate
    np.testing.assert_allclose(dphi_dt[5:-5], expected[5:-5], rtol=1e-3)


def test_generate_waveform_frequency_increases_monotonically():
    wf = generate_waveform(36.0, 29.0, distance_mpc=410.0, sample_rate=4096.0,
                            t_coalescence=0.0, f_lower=20.0)
    assert np.all(np.diff(wf.frequency) > 0)
    assert wf.frequency[0] == pytest.approx(20.0, rel=0.01)
    assert wf.frequency[-1] <= wf.f_isco


def test_generate_waveform_rejects_f_lower_above_isco():
    with pytest.raises(ValueError):
        generate_waveform(36.0, 29.0, distance_mpc=410.0, sample_rate=4096.0, f_lower=1000.0)


def test_generate_waveform_amplitude_grows_toward_merger():
    wf = generate_waveform(36.0, 29.0, distance_mpc=410.0, sample_rate=4096.0,
                            t_coalescence=0.0, f_lower=20.0)
    envelope = np.abs(wf.h_plus)
    # smoothed trend should be increasing overall (compare first vs last decile)
    n = len(envelope)
    assert envelope[-n // 10:].mean() > envelope[:n // 10].mean()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
