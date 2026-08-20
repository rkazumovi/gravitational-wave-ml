# Step 4 — Restricted Post-Newtonian Inspiral Waveform Generator

## Why not PyCBC / LALSuite

Off the table by this project's stated tech-stack constraint (TensorFlow +
SciPy + Matplotlib only). Instead: derive and implement the leading-order
(0PN, "restricted post-Newtonian") inspiral waveform directly from first
principles, from the same chirp-mass frequency-evolution ODE named in the
project spec for the parameter-estimation PINN loss.

## Derivation

Starting from

```
df/dt = (96/5) pi^(8/3) (G Mc/c^3)^(5/3) f^(11/3)
```

this is separable. Writing `k = (96/5) pi^(8/3) (G Mc/c^3)^(5/3)`:

```
f^(-11/3) df = k dt
-3/8 f^(-8/3) = k t + const
```

Choosing the constant so `f -> infinity` as `t -> t_c` (coalescence time)
gives, after simplification:

```
f(t) = (1/pi) (5/256)^(3/8) (G Mc/c^3)^(-5/8) (t_c - t)^(-3/8)
```

Since `dPhi/dt = 2 pi f(t)`, integrating again gives a second closed form:

```
Phi(t) = phi_c - 2 [(t_c - t) / (5 G Mc/c^3)]^(5/8)
```

**Verified by hand** that differentiating this Phi(t) reproduces `2*pi*f(t)`
exactly (the `5^(3/8)/4` coefficients on both sides match) — and verified
again numerically in
[`tests/test_physics.py::test_phase_derivative_matches_two_pi_f`](../tests/test_physics.py)
via finite differencing.

Leading-order (quadrupole) restricted amplitude:

```
A(t) = (4/d_L) (G Mc/c^2)^(5/3) (pi f(t)/c)^(2/3)
h_+ = A(t) * 0.5*(1+cos^2(iota)) * cos(Phi(t))
h_x = A(t) * cos(iota) * sin(Phi(t))
```

## Validity range and its limits

The whole approximation is *inspiral-only* — it diverges at `t=t_c` and is
not meaningful past the point real physics deviates from a slowly-inspiraling
quadrupole source. That point is approximated here by the Schwarzschild
test-particle ISCO frequency of the total mass:

```
f_isco = c^3 / (6^1.5 * pi * G * M_total) ~= 4403 Hz * (Msun / M_total)
```

(Checked against the widely-cited "~4400 Hz * Msun/M" shorthand — matches to
<0.1%.) For GW150914-like masses (~65 Msun) this gives f_isco ~ 67.6 Hz,
noticeably lower than the true numerical-relativity peak-strain frequency of
GW150914 (~250 Hz). This is a known, expected shortfall of the test-particle
approximation for comparable-mass binaries, not a bug — it means this
generator produces realistic *early-inspiral* chirps, not full
inspiral-merger-ringdown (IMR) waveforms. Documented here rather than glossed
over, since downstream (training-data injection) code must only use this
model up to `f_isco`.

## Validation against real GW150914 data

`src/physics/waveforms.py` run directly:

1. **Chirp mass**: computed 28.10 Msun from (36, 29) Msun vs. the published
   source-frame value ~28.6 Msun (~1.7% off — the small discrepancy is from
   using commonly-cited rounded component masses rather than the exact
   best-fit values).
2. **ISCO scaling**: matches the literature shorthand to <0.1% (tested).
3. **Frequency-track overlay**: computed a real spectrogram of whitened H1
   strain around the merger (`scipy.signal.spectrogram`, `nperseg=192`,
   `noverlap=188` — tuned by comparing a few window sizes for ridge
   clarity) and overlaid the analytic `f(t)` track using the
   *detector-frame* chirp mass (source-frame Mc x (1+z), z~0.09 for
   GW150914). The analytic track visibly follows the real rising-frequency
   ridge in the data up to its ISCO cutoff
   (`outputs/GW150914_waveform_validation.png`).

   Caveat: the ridge isn't razor-sharp — a fixed-window STFT has an inherent
   time-frequency resolution tradeoff (`Delta-t * Delta-f` bounded below)
   that is particularly punishing for a signal this short and fast-chirping.
   LIGO operationally uses a multi-resolution Q-transform for exactly this
   reason; implementing one was judged out of scope here given the
   time-domain whitened-chirp validation in Step 3 already gives a much
   cleaner, unambiguous confirmation of timing.

## Tests

[`tests/test_physics.py`](../tests/test_physics.py) — 9 tests: chirp mass
value and symmetry, ISCO scaling, frequency divergence/NaN behavior near and
past coalescence, the `dPhi/dt = 2 pi f` identity (numerically), monotonic
frequency increase and amplitude growth in a generated waveform, and the
`f_lower > f_isco` error case.
