← [Documentation index](../../README.md)

# Sound power from surface vibration (ISO/TS 7849)

The airborne sound power a machine radiates through the structure-borne
vibration of its outer surface can be estimated from the surface vibratory
velocity and a **radiation factor** $\varepsilon$ (the radiation efficiency),
without an
acoustic measurement. The radiated power is (ISO/TS 7849-1, Formula 6)

$$
P = Z_\mathrm{c} \, \langle v^2 \rangle \, S \, \varepsilon \quad [\mathrm{W}],
$$

with $Z_\mathrm{c}$ the characteristic impedance of air, $\langle v^2 \rangle$ the
mean-square vibratory velocity over the radiating area $S$. Expressed in levels
(velocity level re $v_0 = 5\times10^{-8}\ \text{m/s}$), the A-weighted sound
power level is (Formula 12 / 15)

$$
L_W = L_v + 10\log_{10}\frac{S}{S_0} + 10\log_{10}\varepsilon
      + 10\log_{10}\frac{Z_\mathrm{c,n}}{Z_{\mathrm{c},0}},
$$

where $S_0 = 1\ \text{m}^2$, the normalized impedance
$Z_\mathrm{c,n} = 411\ \text{N·s/m}^3$ and the reference
$Z_{\mathrm{c},0} = 400\ \text{N·s/m}^3$ give the fixed
$10\log_{10}(411/400) = 0.118\ \text{dB}$
term. This module feeds the structure-borne source and building prediction
standards (ISO 9611, EN 15657, EN 12354-5).

Before any levels, the radiator itself. The `radiation_efficiency` plate model
that supplies a predicted radiation factor retains its geometry, and
`sigma.plot_geometry()` draws the plate in its baffle to scale, here 1.5 m by
1.25 m and simply supported.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/radiation_plate_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/radiation_plate_geometry.svg" alt="To-scale front view of the plate of the radiation model: a grey 1.5 m by 1.25 m simply supported plate inside its hatched rigid baffle, both side lengths dimensioned and the boundary condition named in the title" width="82%"></picture>

*The radiator behind the radiation factor, to scale: the 1.5 m by 1.25 m
simply supported plate in its rigid baffle, whose area $S$ enters $L_W$
directly while its size decides how far $\varepsilon$ falls below one under
coincidence.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import vibration

# The simply supported 1.5 x 1.25 m plate of the radiation model.
f = np.geomspace(50.0, 5000.0, 200)
sigma = vibration.radiation_efficiency(f, 1.5, 1.25, 2100.0)
sigma.plot_geometry()
plt.show()
```

</details>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_sound_power_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_sound_power.svg" alt="Radiated sound power level per octave band of a vibrating surface, comparing the ISO/TS 7849-1 upper limit with a fixed radiation factor of one against the ISO/TS 7849-2 engineering value with a measured radiation factor, with the band-summed totals marked" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import emission

# Surface velocity levels and a measured radiation factor per octave band.
bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
lv = np.array([78.0, 82.0, 85.0, 83.0, 79.0, 74.0])
eps = np.array([0.20, 0.45, 0.75, 0.95, 1.00, 1.00])

lw_max = emission.radiated_sound_power_level(lv, 1.6)  # Part 1, eps = 1
lw_eng = emission.radiated_sound_power_level(lv, 1.6, radiation_factor=eps)  # Part 2

# One line — the LW(f) spectrum of one determination as a result object:
res = emission.sound_power_from_vibration(lv, area=1.6, radiation_factor=eps,
                                          frequencies=bands)
res.plot()
plt.show()

# By hand, comparing the two parts:
x = np.arange(bands.size)
fig, ax = plt.subplots()
ax.bar(x - 0.2, lw_max, width=0.4, label="Part 1 upper limit ($\\varepsilon$ = 1)")
ax.bar(x + 0.2, lw_eng, width=0.4, label="Part 2 engineering ($\\varepsilon$ measured)")
ax.set_xticks(x, [f"{b:g}" for b in bands])
ax.set(xlabel="Frequency [Hz]", ylabel="Sound power level $L_W$ [dB re 1 pW]")
ax.legend()
plt.show()
```

</details>

## 1. The two parts

The two parts differ only in the radiation factor. **Part 1 (survey)** assumes
$\varepsilon = 1$ and yields the *upper limit* $L_{W,\max}$, needing only the
velocity level and the area. **Part 2 (engineering)** applies a frequency-band
radiation factor $\varepsilon_j$ determined (per ISO 9614) as
$\varepsilon_j = P_j/(Z_\mathrm{c,n}\,\langle v_j^2 \rangle\,S)$.

```python
import numpy as np
from phonometry import emission

bands = np.array([250.0, 500.0, 1000.0, 2000.0])
lv = np.array([82.0, 85.0, 83.0, 79.0])          # mean velocity level per band [dB]

# Part 1 upper limit (epsilon = 1):
upper = emission.sound_power_from_vibration(lv, area=1.6, frequencies=bands)
print(round(upper.total_level, 1))               # e.g. 89.4  dB re 1 pW

# Part 2 engineering value with a measured radiation factor:
eps = np.array([0.45, 0.75, 0.95, 1.00])
eng = emission.sound_power_from_vibration(lv, area=1.6, radiation_factor=eps, frequencies=bands)
print(np.round(eng.sound_power_level, 1))        # per-band L_W

eng.plot()   # the LW(f) spectrum, as in the figure above (needs matplotlib)
```

## 2. Velocity level, calibration and the radiation factor

The velocity level is $L_v = 20\log_{10}(v/v_0)$ (Formula 3); a sinusoidal
calibration acceleration converts as
$L_v = 20\log_{10}\!\left(\hat{a}/(2\pi f\,v_0\sqrt{2})\right)$ (Formula 8). The
radiation factor comes from an independently measured power:

```python
from phonometry import emission

# The standard's worked calibration EXAMPLE: 9.81 m/s^2 at 100 Hz.
print(round(float(emission.velocity_level_from_acceleration(9.81, 100.0)), 1))   # 106.9 dB

# Radiation factor from a measured power (ISO 9614): eps = P / (Zc <v^2> S).
eps = emission.radiation_factor(3.0e-4, area=2.0, mean_square_velocity=(1e-3)**2)
print(round(float(eps), 3))                                                # 0.365
```

Surface velocity levels from several positions are combined with the energetic
mean `mean_velocity_level` (Formula 10) or its area-weighted form (Formula 11),
and the correction `extraneous_velocity_correction` removes extraneous
vibration per Table 2.

Those positions are not free: the standard divides the radiating surface $S$
into $N$ equal cells and puts one accelerometer at the centre of each. The
area table gives the initial $N$; a strongly non-uniform vibration field can
call for more positions or a redistribution.

What is implemented is the calculation, not the laboratory practice: the
measurement clauses of both parts (instrumentation, source installation,
environmental conditions — clauses 5 to 7), their measurement-uncertainty
clauses and the informative annexes are not code, so the position counts and
the Table 2 ladder are rules the reader keeps, not checks the functions run.
Part 2's clause 8 also defines the radiation factor of a machine batch or
family, averaging $\varepsilon_j$ over several machines with its standard
deviation (Formulae 9 and 10); only the single-machine Formula 8 is
implemented, so pass an already-averaged $\varepsilon$ for a family
determination.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_vibration_sound_power_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_vibration_sound_power.svg" alt="ISO/TS 7849 surface-velocity method: a machine under test with its vibrating measurement surface of 2.5 by 1.6 metres divided into twenty equal cells, an accelerometer at each cell centre, the radiated airborne sound leaving the surface, the table for the initial number of measurement positions and the survey relation LWA = LvA + 10 lg(S/S0) + 10 lg epsilon" width="92%"></picture>

## 3. When the radiation-factor assumption breaks

The whole method stands on one substitution: replacing the acoustic
measurement by $\varepsilon$. The Part 1 value $\varepsilon = 1$ is close to
the true radiation
factor only above the **critical (coincidence) frequency** of plate-like
parts, where bending waves travel faster than sound and the surface radiates
like a piston. Below coincidence, adjacent zones of the
plate move in antiphase and their radiation largely cancels: $\varepsilon$
drops far
below one and falls quickly with decreasing frequency, so the survey method
can overstate the low-frequency bands of a large thin casing by 10 dB and
more. The same cancellation makes small sources radiate poorly (the acoustic
short circuit around an unbaffled panel). Two further assumptions are easy
to violate in the field:

* **The measured vibration must be the machine's own.** Vibration fed in
  from neighbouring machinery inflates $\langle v^2 \rangle$; Table 2
  prescribes the
  source-off check and `extraneous_velocity_correction` applies it.
* **The surface must be the dominant radiator.** Airborne sound from
  openings, intakes or internal sources that bypasses the measured casing is
  invisible to a velocity survey; the method characterises the
  structure-borne part only.

"Adjacent zones move in antiphase and their radiation largely cancels" is a
statement about a phase relationship between a travelling structural wave and
the air on top of it, which no spectrum can draw. The clip below drives a
10 mm steel plate in air along its whole length — a force on the plate, not a
wave arriving at it, and over the whole plate rather than a patch, because a
patch radiates from the patch and below coincidence that would be the only
thing in the air — and runs the same scene twice, at $f_\mathrm{c}/2$ and at
$2f_\mathrm{c}$, with nothing differing between the panels but the drive frequency.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_radiation_efficiency_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_radiation_efficiency.gif" alt="Animation: a bending wave running along a steel plate in air, with alternating pressure lobes that cling to the plate and fade out within centimetres in the lower-frequency panel, and a plane beam departing at forty-five degrees in the higher-frequency panel" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_elastic_radiation_efficiency.webm)

Below coincidence the bending wavelength is **shorter** than the acoustic
one. Adjacent half-waves push and pull the same air in antiphase, and what
the air shows is a skin of alternating lobes that clings to the plate and
dies within a fraction of a wavelength: the pressure at the surface is of the
same order as in the other panel, but a quarter-cycle out of step with the
velocity, so it carries no power away — which is what $\varepsilon \ll 1$
*means*. Above coincidence the bending wave outruns sound, the trace match is
satisfied, and a plane beam leaves at $\arcsin(c_0/c_\mathrm{B}) = 45°$. Each panel
is annotated with the closed forms its regime obeys, from the plate's own
constants: below coincidence $\lambda_\mathrm{B} = 0.40$ m is shorter than the 0.57 m of
sound in air, so $\sin\theta = \lambda/\lambda_\mathrm{B}$ has no solution and the skin
that stands in its place decays over $1/\sqrt{k_\mathrm{B}^2 - k_0^2} = 0.091$ m; above
it $\lambda_\mathrm{B} = 0.20$ m is longer than the 0.14 m in air and the trace match
sends a beam out at 45°. Nothing is measured off the scene: the ends of the
plate reflect the bending wave partially, so the air carries a leftward beam on
top of the rightward one and any reading is the difference of the two. Two cautions: the scene is an effectively infinite plate, so
it shows below-coincidence cancellation and not the short circuit around an
unbaffled panel (the same cancellation at another scale), and the clip prints
no radiation factor, because any finite driving aperture radiates a little on
its own and below coincidence that would be the only thing an intensity line
saw. The factor belongs to the closed-form model below.

Part 2 exists exactly for the radiation-factor problem: it replaces the
fixed $\varepsilon = 1$
with a band-by-band $\varepsilon_j$ determined from one reference measurement
of the
radiated power (ISO 9614 intensity), after which the velocity survey can be
repeated cheaply on nominally identical machines.

## 4. The measurement report (`.report()`)

A determination ends as a *document*. `VibrationSoundPowerResult.report(path)`
writes a one-page PDF fiche laid out like a sound-power test sheet: the
standard-basis line naming the applied method (the ISO/TS 7849-1 survey method
with a fixed radiation factor $\varepsilon = 1$, or the ISO/TS 7849-2
engineering method
with a determined radiation factor), an optional metadata header (client,
machine/source, test environment, instrumentation, climate, date), a per-band
table (nominal octave/one-third-octave frequency, the surface vibratory
velocity level $L_v$ and the band sound-power level $L_W$), the sound-power
spectrum $L_W(f)$ with a nominal band axis, and a boxed A-weighted sound power
level $L_{W\mathrm{A}}$ (dB re 1 pW) with the total $L_W$, the radiating area $S$ and the
applied method alongside.

The relevant `ReportMetadata` fields are `client`, `specimen` (the
machine/source), `test_room` (the test environment), `instrumentation`,
`temperature`, `relative_humidity`, `pressure`, `test_date` and the footer
identity `laboratory`, `operator`, `report_id` and `notes`; the radiating area
$S$ comes from the result itself. Supplying `requirement` adds a PASS/FAIL
verdict against a declared A-weighted sound-power limit (lower is better).
`verbose=True` adds the radiation factor $\varepsilon$ column, and `language="es"`
renders the Spanish fiche (comma decimals). Rendering needs the optional
`phonometry[report]` extra (reportlab), plus matplotlib for the spectrum.

```python
import numpy as np
from phonometry import ReportMetadata, emission

freqs = np.array([125, 250, 500, 1000, 2000, 4000], float)
lv = np.array([78.0, 82, 85, 83, 79, 74])          # surface velocity level [dB]
eps = np.array([0.20, 0.45, 0.75, 0.95, 1.00, 1.00])  # measured radiation factor
res = emission.sound_power_from_vibration(
    lv, area=1.6, radiation_factor=eps, frequencies=freqs,
)
res.report(
    "vibration_sound_power.pdf",
    metadata=ReportMetadata(
        client="Example manufacturing plant",
        specimen="Gearbox casing (steel panel)",
        test_room="Machine hall (source vibration survey)",
        instrumentation="Piezoelectric accelerometer (ISO 16063-21 calibration), s/n 0042",
        laboratory="Phonometry reference example",
        report_id="EXAMPLE-7849",
        requirement=90.0,
    ),
)   # LWA = 88.7 dB(A) re 1 pW -> declared limit 90 dB(A): PASS
```

The rendered example fiche, regenerated with `make reports`, is kept in the
repository. Click the preview to open the PDF:

[![ISO/TS 7849 sound-power-from-vibration example report: a header with the client, the machine/source, the machine-hall test environment and the accelerometer and climate, the octave-band table (125 Hz to 4 kHz) of surface vibratory velocity levels Lv and radiated band sound-power levels LW, the sound-power spectrum LW(f) with a nominal band axis, the boxed A-weighted sound power level LWA = 88.7 dB(A) re 1 pW with the total LW = 90.0 dB, the radiating area S = 1.60 m2 and the engineering method, and a PASS verdict against the declared 90 dB(A) limit](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso7849_vibration_power_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso7849_vibration_power_example.pdf)

*Sound power from vibration fiche (`VibrationSoundPowerResult.report`), an
ISO/TS 7849-2 engineering-method determination with the measured radiation
factor and the boxed LWA.*

## See also

- [Sound Power](sound-power.md): the route chooser
  this method is the sixth entry of, the accuracy grades and the ISO 4871
  declaration a result feeds.
- [Sound Power by Intensity Scanning (ISO 9614)](sound-power-intensity.md):
  the determination that produces the $P_j$ behind a Part 2 radiation factor.
- [Sound Intensity (p-p)](intensity.md): the probe,
  its residual index and the field indicators that qualify that measurement.
- [Panel Sound Insulation](../../buildings/design/panel-sound-insulation.md):
  the same radiation-efficiency model, written $\sigma$ instead of
  $\varepsilon$, and the coincidence physics behind it.
- [Structure-borne source characterisation](../../buildings/design/structure-borne-power.md):
  ISO 9611, EN 15657 and EN 12354-5, the standards that consume this kind of
  result.
- [Mechanical mobility (ISO 7626)](../../vibration/structural/mechanical-mobility.md):
  the transducer practice behind a measured surface velocity.
- API reference: [`emission.vibration_sound_power`](https://jmrplens.github.io/phonometry/reference/api/power/vibration-sound-power/).
- Theory: [Point mobilities and radiation efficiency](../../reference/theory/vibration.md#point-mobilities-and-radiation-efficiency-cremer-5-hopkins-29): the radiation efficiency that turns a surface velocity into a radiated power.

## References

- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-borne
  sound: Structural vibrations and sound radiation at audio frequencies*
  (3rd ed.). Springer. ISBN 978-3-540-22696-3.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728).
  The radiation-efficiency treatment behind section 3: coincidence, the
  cancellation below the critical frequency and the radiation of finite
  plates.
- International Organization for Standardization. (2009). *Acoustics —
  Determination of airborne sound power levels emitted by machinery using
  vibration measurement — Part 1: Survey method using a fixed radiation
  factor* (ISO/TS 7849-1:2009).
  [iso.org catalogue](https://www.iso.org/standard/40537.html).
  The upper-limit method with $\varepsilon = 1$.
- International Organization for Standardization. (2009). *Acoustics —
  Determination of airborne sound power levels emitted by machinery using
  vibration measurement — Part 2: Engineering method including determination
  of the adequate radiation factor* (ISO/TS 7849-2:2009).
  [iso.org catalogue](https://www.iso.org/standard/40538.html).
  The engineering method with a measured band-wise radiation factor.

## Standards

ISO/TS 7849-1:2009 (*survey method using a fixed radiation
factor*) and ISO/TS 7849-2:2009 (*engineering method including determination of
the adequate radiation factor*), *Acoustics — Determination of airborne sound
power levels emitted by machinery using vibration measurement*: the radiated
power $P = Z_\mathrm{c} \, \langle v^2 \rangle \, S \, \varepsilon$ (Formula 6), the
velocity level and its calibration
(Formulae 3, 8), the mean over the surface (Formulae 10/11), the extraneous
correction (Table 2), the radiation factor (Formula 4/8) and the sound power
level (Formulae 12/15). Conformance is anchored on the standard's own worked
calibration example, the exact round-trip between the radiation factor and
$L_W = 10\log_{10}(P/P_0)$, and the fixed impedance term.
