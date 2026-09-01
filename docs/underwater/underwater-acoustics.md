← [Documentation index](../README.md)

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
\mathrm{SPL} = 10\log_{10}\frac{\langle p^2\rangle}{p_0^2}, \qquad
\mathrm{SEL} = 10\log_{10}\frac{\int p^2\,dt}{E_0}, \qquad
L_{p,\mathrm{pk}} = 20\log_{10}\frac{\max|p|}{p_0},
$$

with $p_0 = 1\ \mu\text{Pa}$ and $E_0 = 1\ \mu\text{Pa}^2\,\text{s}$. The sound pressure level is the
mean-square level (ISO 18406 Formula 7); the sound exposure level integrates the
squared pressure over the record (Formulae 3–4); the peak level is the
zero-to-peak value.

```python
import numpy as np
from phonometry import underwater

# A captured hydrophone record (here a synthetic 250 Hz tone of 1 s).
fs = 48000
pressure = 0.5 * np.sin(2 * np.pi * 250.0 * np.arange(fs) / fs)

spl = underwater.sound_pressure_level(pressure)        # dB re 1 µPa
sel = underwater.sound_exposure_level(pressure, fs)     # dB re 1 µPa²·s
pk = underwater.peak_sound_pressure_level(pressure)     # dB re 1 µPa
```

To re-reference a level between the underwater (1 µPa) and airborne (20 µPa)
conventions (a $20\log_{10}(20) \approx 26.02$ dB shift, **not** an energy/intensity
equivalence), use `underwater_to_in_air_spl` / `in_air_to_underwater_spl`. For
background-noise subtraction, reuse the ISO 3744 `background_noise_correction`
($K_1$) helper.

## 2. Ship radiated noise and source level (ISO 17208-1/-2)

A surface ship measured in deep water is described first by its **radiated noise
level** and then by an **equivalent monopole source level**:

$$
L_{\mathrm{RN}} = 20\log_{10}\frac{p_{\mathrm{rms}}}{p_0} + 20\log_{10}\frac{r}{r_0}
\ \ \mathrm{dB\ re\ 1\ \mu Pa\!\cdot\! m}, \qquad
L_\mathrm{s} = L_{\mathrm{RN}} + \Delta L,
$$

where the Lloyd's-mirror surface correction (ISO 17208-2 Formula 3) for a
nominal source depth $d_\mathrm{s} = 0.7\,D$ ($D$ = mean draught) and $u = k\,d_\mathrm{s}$,
$k = 2\pi f/c$, is

$$
\Delta L = -10\log_{10}\frac{2u^4 + 14u^2}{14 + 2u^2 + u^4}\ \mathrm{dB}.
$$

$\Delta L$ diverges at low $u$ (grazing / low frequency) and tends to
$-10\log_{10}(2) = -3.01$ dB as $u \to \infty$. The reported source level is an *equivalent monopole
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
import numpy as np
from phonometry import underwater

# Band r.m.s. pressures measured at 100 m from a merchant ship, in pascals.
freqs = np.array([31.5, 63.0, 125.0, 250.0, 500.0, 1000.0])
p_rms = np.array([4.3, 2.8, 1.9, 1.2, 0.8, 0.54])
rnl = np.array([underwater.radiated_noise_level(float(x), 100.0) for x in p_rms])
print(np.round(rnl, 1))   # [172.7 168.9 165.6 161.6 158.1 154.6] dB re 1 uPa.m

# The surface correction is frequency-dependent, so the conversion is normally
# run on the whole band vector at once.
res = underwater.monopole_source_level(rnl, freqs, draught=6.0)
print(np.round(res.source_level, 1))       # [177.8 168.4 161.7 157.8 154.8 151.6]
print(np.round(res.surface_correction, 2))  # [ 5.15 -0.51 -3.86 -3.78 -3.27 -3.08]
print(round(res.source_depth, 2))           # 4.2 m = 0.7 x draught
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
runs while the ship crosses a window of length $2\,d_\mathrm{CPA}\tan 30°$ (about
1.15 CPA distances), centred on the beam aspect the radiated noise level is
defined for (Clause 3, Figure 3). Four runs are required, two per side; each
run's three hydrophone levels are power-averaged (Formula 8), the runs are
then arithmetically averaged (Formula 9), and port and starboard are also
reported separately, because a real ship does not radiate symmetrically
(Clause 6.5). Background noise is measured at the start and end of each test
period and the ISO 3744-style correction applied per band; the recommended
wind limit is 20 kn for ships above 100 m (Clause 5.3). Skip any of this and
the number you quote is a level, but not an ISO 17208 radiated noise level.
None of that discipline is implemented in code: the library supplies the
closed-form `radiated_noise_level` and `monopole_source_level`, and the runs,
the Formulae 8-9 averaging, the geometry checks, the ±30° window scoring and
the per-band application of the background correction are the reader's.

All of that discipline fits in one picture: the transit geometry in section
and the data window in plan.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_hydrophone_deployment_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_hydrophone_deployment.svg" alt="ISO 17208-1 deep-water measurement geometry: a ship transiting past a surface buoy that suspends a vertical array of three hydrophones at depths of about 27, 58 and 100 metres set by the 15, 30 and 45 degree depression angles, a lateral distance at the closest point of approach of at least 100 metres or one ship length, water at least 150 metres deep, and a plan view showing the plus or minus 30 degree data window about the CPA" width="92%"></picture>

## 3. Pile-driving sound (ISO 18406)

Percussive pile driving radiates one impulsive pulse per hammer strike. Each
strike has a **single-strike sound exposure level** $\mathrm{SEL_{ss}}$; over a driving
sequence the exposures add to a **cumulative sound exposure level**:

$$
\mathrm{SEL_{cum}} = 10\log_{10}\sum_{n} 10^{\mathrm{SEL}_n/10},
\qquad\text{and for } N \text{ identical strikes } = \mathrm{SEL_{ss}} + 10\log_{10} N.
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
strike_pressure = 8000.0 * envelope * np.sin(2 * np.pi * 180.0 * t)
res = underwater.pile_strike_metrics(strike_pressure, fs)
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

The survey discipline itself — deployment, instrumentation, calibration and
reporting — is described here and enforced nowhere in code: nothing checks
that a strike handed to `pile_strike_metrics` came from a record that meets
it. And because vibro- and sheet-piling sit outside the ISO 18406 scope,
continuous pile-driving noise has no closed form here or anywhere else in
phonometry.

## See also

- [Marine-mammal noise exposure](marine-mammal-exposure.md):
  the assessment the criteria named in section 3 are applied through — auditory
  weighting, accumulation over the strikes and the margin against the published
  onset criteria.
- [Underwater sound propagation](underwater-propagation.md):
  the propagation loss that carries these source levels to a receiver, and the
  sonar equation they feed.
- API reference: [`underwater.acoustics`](https://jmrplens.github.io/phonometry/reference/api/underwater/acoustics/), [`underwater.sources.pile_driving_noise`](https://jmrplens.github.io/phonometry/reference/api/underwater/pile-driving-noise/) and [`underwater.sources.ship_radiated_noise`](https://jmrplens.github.io/phonometry/reference/api/underwater/ship-radiated-noise/).
- General underwater propagation modelling (ray, normal mode, parabolic equation) is covered in [Underwater propagation solvers](underwater-solvers.md).

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
Lloyd's-mirror surface correction (Formula 3) for a 0.7·draught source depth.
ISO 18406:2017, *Underwater acoustics — Measurement of radiated underwater sound
from percussive pile driving*: the single-strike (Formulae 3–4), peak and
cumulative (Formulae 8–9) sound exposure. All quantities are verified against
exact analytic oracles. General underwater propagation modelling (ray, normal
mode, parabolic equation) is out of scope here: see
[Underwater propagation solvers](underwater-solvers.md).
