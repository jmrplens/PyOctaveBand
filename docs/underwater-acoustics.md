← [Documentation index](README.md)

# Underwater acoustics: radiated noise and pile driving (ISO 18405 / 17208 / 18406)

Underwater sound is referenced to **1 µPa** (not the 20 µPa of airborne
acoustics), and its exposure to **1 µPa²·s**. This page covers the ISO 18405
reference levels realised as the shared primitives, the ISO 17208 ship
**radiated noise level** and equivalent **monopole source level**, and the
ISO 18406 percussive **pile-driving** single-strike, peak and cumulative sound
exposure. Every quantity is an exact closed form, verified analytically.

## 1. Reference levels (ISO 18405)

Three primitives, referred to 1 µPa, are computed from a captured pressure
signal (in Pa):

$$
\mathrm{SPL} = 10\lg\frac{\langle p^2\rangle}{p_0^2}, \qquad
\mathrm{SEL} = 10\lg\frac{\int p^2\,dt}{E_0}, \qquad
L_{p,\mathrm{pk}} = 20\lg\frac{\max|p|}{p_0},
$$

with `p₀ = 1 µPa` and `E₀ = 1 µPa²·s`. The sound pressure level is the
mean-square level (ISO 18406 Formula 7); the sound exposure level integrates the
squared pressure over the record (Formulae 3–4); the peak level is the
zero-to-peak value.

```python
from phonometry import underwater

spl = underwater.sound_pressure_level(pressure)        # dB re 1 µPa
sel = underwater.sound_exposure_level(pressure, fs)     # dB re 1 µPa²·s
pk = underwater.peak_sound_pressure_level(pressure)     # dB re 1 µPa
```

To re-reference a level between the underwater (1 µPa) and airborne (20 µPa)
conventions (a `20·lg(20) ≈ 26.02` dB shift, **not** an energy/intensity
equivalence), use `underwater_to_in_air_spl` / `in_air_to_underwater_spl`. For
background-noise subtraction, reuse the ISO 3744 `background_noise_correction`
(`K1`) helper.

## 2. Ship radiated noise and source level (ISO 17208-1/-2)

A surface ship measured in deep water is described first by its **radiated noise
level** and then by an **equivalent monopole source level**:

$$
L_{\mathrm{RN}} = 20\lg\frac{p_{\mathrm{rms}}}{p_0} + 20\lg\frac{r}{r_0}
\ \ \mathrm{dB\ re\ 1\ \mu Pa\!\cdot\! m}, \qquad
L_s = L_{\mathrm{RN}} + \Delta L,
$$

where the Lloyd's-mirror surface correction (ISO 17208-2 Formula 3) for a
nominal source depth `d_s = 0.7·D` (D = mean draught) and `u = k·d_s`,
`k = 2πf/c`, is

$$
\Delta L = -10\lg\frac{2u^4 + 14u^2}{14 + 2u^2 + u^4}\ \mathrm{dB}.
$$

`ΔL` diverges at low `u` (grazing / low frequency) and tends to `−10·lg(2) =
−3.01` dB as `u → ∞`. The reported source level is an *equivalent monopole
broadside* value and must be quoted with its source depth.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/ship_source_level_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/ship_source_level.svg" alt="Ship radiated noise level and equivalent monopole source level versus frequency, with the Lloyd's-mirror surface correction on a twin axis showing its low-frequency divergence and its approach to −3 dB at high frequency" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
from phonometry import underwater

freqs = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                  400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                  4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000.0])
rnl = 175.0 - 12.0 * np.log10(freqs / 20.0)
res = underwater.monopole_source_level(rnl, freqs, draught=6.0)
res.plot()
```

</details>

```python
from phonometry import underwater

lrn = underwater.radiated_noise_level(2e-6, 100.0)   # p_rms = 2 µPa, r = 100 m
res = underwater.monopole_source_level(lrn, 200.0, draught=6.0)
print(res.source_level, res.surface_correction, res.source_depth)
res.plot()   # RNL, Ls and ΔL vs frequency (needs matplotlib)
```

`hydrophone_depths` gives the three ISO 17208-1 measurement depths from the
15°/30°/45° depression angles, and `source_level_uncertainty` the tabulated
expanded uncertainty (5 dB ≤100 Hz, 3 dB 125 Hz–16 kHz, 4 dB >16 kHz).

What turns these closed forms into a *comparable* number is the measurement
discipline ISO 17208-1 wraps around them. The ship transits a straight course
past a vertical string of three hydrophones at a closest point of approach of
100 m or one ship length, whichever is greater, in water at least 150 m or
1.5 ship lengths deep so the bottom stays out of the picture (Clauses 5.2,
5.4). Only the **data window** of ±30° about the CPA is scored: the averaging
runs while the ship crosses a window of length `2·d_CPA·tan 30°` (about
1.15 CPA distances), centred on the beam aspect the radiated noise level is
defined for (Clause 3, Figure 3). Four runs are required, two per side; each
run's three hydrophone levels are power-averaged (Formula 8), the runs are
then arithmetically averaged (Formula 9), and port and starboard are also
reported separately, because a real ship does not radiate symmetrically
(Clause 6.5). Background noise is measured at the start and end of each test
period and the ISO 3744-style correction applied per band; the recommended
wind limit is 20 kn for ships above 100 m (Clause 5.3). Skip any of this and
the number you quote is a level, but not an ISO 17208 radiated noise level.

## 3. Pile-driving sound (ISO 18406)

Percussive pile driving radiates one impulsive pulse per hammer strike. Each
strike has a **single-strike sound exposure level** `SEL_ss`; over a driving
sequence the exposures add to a **cumulative sound exposure level**:

$$
\mathrm{SEL_{cum}} = 10\lg\sum_{n} 10^{\mathrm{SEL}_n/10},
\qquad\text{and for } N \text{ identical strikes } = \mathrm{SEL_{ss}} + 10\lg N.
$$

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/pile_driving_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/pile_driving.svg" alt="A percussive pile-driving strike pressure waveform with its peak marked, and below it the cumulative sound exposure level growing as SEL_ss plus ten times the logarithm of the number of strikes" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
from phonometry import underwater

fs = 48000
t = np.arange(int(0.3 * fs)) / fs
envelope = np.where(t < 0.01, t / 0.01, np.exp(-(t - 0.01) / 0.04))
pressure = 8000.0 * envelope * np.sin(2 * np.pi * 180.0 * t)
res = underwater.pile_strike_metrics(pressure, fs)
res.plot()
```

</details>

```python
from phonometry import underwater

sel_ss = underwater.single_strike_sel(strike_pressure, fs)   # dB re 1 µPa²·s
sel_cum = underwater.cumulative_sel_identical(sel_ss, 2000)   # 2000 strikes
res = underwater.pile_strike_metrics(strike_pressure, fs)
print(res.single_strike_sel, res.peak_spl, res.pulse_duration)
res.plot()   # waveform + cumulative energy (needs matplotlib)
```

`pile_strike_metrics` bundles the single-strike SEL, the peak sound pressure
level, the SPL/Leq and the 90 %-energy pulse duration for one recorded strike;
`cumulative_sel` sums a sequence of differing per-strike SELs.

These are the metrics that regulation is written in. ISO 18406 exists because
offshore wind-farm, oil-and-gas and bridge foundations are consented under
environmental impact frameworks that require the radiated sound to be
monitored, and its scope is drawn accordingly: percussive driving in 4 to
100 m of water, with vibro- and sheet-piling excluded (Clause 1). The minimum
campaign is one measurement position **as close as possible to 750 m** from
the pile, recording the entire driving sequence and reporting the actual
range; the standard is explicit that 750 m is chosen for comparability with
the large body of existing measurements, not because any regulator's limit
lives there, and that a single-range level has no predictive value for other
sites (Clause 6.1.2, Notes 1–2). Impact criteria for marine fauna are phrased
in exactly the quantities of this section, a cap on the single-strike or
cumulative SEL and on the peak level at a stated range, which is why
`pile_strike_metrics` reports them together and `cumulative_sel` follows the
strike-by-strike energy sum of Formulae 8–9.

## References

- Urick, R. J. (1983). *Principles of underwater sound* (3rd ed.).
  McGraw-Hill; reprinted 1996 by Peninsula Publishing.
  ISBN 978-0-932146-62-5.
  [Open Library record](https://openlibrary.org/books/OL9317725M).
  The classic treatment of underwater sound levels and of ship radiated
  noise behind the reference conventions of sections 1-2.
- Ainslie, M. A. (2010). *Principles of sonar performance modelling*.
  Springer.
  [doi:10.1007/978-3-540-87662-5](https://doi.org/10.1007/978-3-540-87662-5).
  The systematic treatment of underwater acoustical quantities in the line
  that ISO 18405 standardised; supports the reference levels and the source
  level of sections 1-2.

## Standards

ISO 18405:2017, *Underwater acoustics — Terminology*: the 1 µPa
sound-pressure and 1 µPa²·s sound-exposure references and the definitions of
sound pressure level, sound exposure level and peak sound pressure level.
ISO 17208-1:2016 and ISO 17208-2:2019, *Underwater acoustics — Quantities and
procedures for description and measurement of underwater sound from ships*: the
radiated noise level, the deep-water three-hydrophone geometry, the tabulated
source-level uncertainty and the equivalent monopole source level via the
Lloyd's-mirror surface correction (Formula 3) for a `0.7·draught` source depth.
ISO 18406:2017, *Underwater acoustics — Measurement of radiated underwater sound
from percussive pile driving*: the single-strike (Formulae 3–4), peak and
cumulative (Formulae 8–9) sound exposure. All quantities are verified against
exact analytic oracles. General underwater propagation modelling (ray, normal
mode, parabolic equation) is out of scope here.
