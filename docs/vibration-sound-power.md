← [Documentation index](README.md)

# Sound power from surface vibration (ISO/TS 7849)

The airborne sound power a machine radiates through the structure-borne
vibration of its outer surface can be estimated from the surface vibratory
velocity and a **radiation factor** `ε` (the radiation efficiency), without an
acoustic measurement. The radiated power is (ISO/TS 7849-1, Formula 6)

$$
P = Z_c \, \langle v^2 \rangle \, S \, \varepsilon \quad [\mathrm{W}],
$$

with `Z_c` the characteristic impedance of air, `⟨v²⟩` the mean-square vibratory
velocity over the radiating area `S`. Expressed in levels (velocity level re
`v₀ = 5·10⁻⁸ m/s`), the A-weighted sound power level is (Formula 12 / 15)

$$
L_W = L_v + 10\lg\frac{S}{S_0} + 10\lg\varepsilon
      + 10\lg\frac{Z_{c,n}}{Z_{c,0}},
$$

where `S₀ = 1 m²`, the normalized impedance `Z_{c,n} = 411 N·s/m³` and the
reference `Z_{c,0} = 400 N·s/m³` give the fixed `10 lg(411/400) = 0.118 dB`
term. This module feeds the structure-borne source and building prediction
standards (ISO 9611, EN 15657, EN 12354-5).

Before any levels, the radiator itself. The `radiation_efficiency` plate model
that supplies a predicted radiation factor retains its geometry, and
`sigma.plot_geometry()` draws the plate in its baffle to scale, here 1.5 m by
1.25 m and simply supported.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/radiation_plate_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/radiation_plate_geometry.svg" alt="To-scale front view of the plate of the radiation model: a grey 1.5 m by 1.25 m simply supported plate inside its hatched rigid baffle, both side lengths dimensioned and the boundary condition named in the title" width="82%"></picture>

*The radiator behind the radiation factor, to scale: the 1.5 m by 1.25 m
simply supported plate in its rigid baffle, whose area `S` enters `L_W`
directly while its size decides how far `ε` falls below one under
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
`ε = 1` and yields the *upper limit* `L_W,max`, needing only the velocity level
and the area. **Part 2 (engineering)** applies a frequency-band radiation factor
`εⱼ` determined (per ISO 9614) as `εⱼ = Pⱼ/(Z_{c,n}·⟨vⱼ²⟩·S)`.

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

The velocity level is `L_v = 20·lg(v/v₀)` (Formula 3); a sinusoidal calibration
acceleration converts as `L_v = 20·lg(â/(2πf·v₀·√2))` (Formula 8). The radiation
factor comes from an independently measured power:

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

Those positions are not free: the standard divides the radiating surface `S`
into `N` equal cells and puts one accelerometer at the centre of each, with
`N` set by the area alone.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_vibration_sound_power_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_vibration_sound_power.svg" alt="ISO/TS 7849 surface-velocity method: a machine under test with its vibrating measurement surface of 2.5 by 1.6 metres divided into ten equal cells, an accelerometer at each cell centre, the radiated airborne sound leaving the surface, the table for the number of measurement positions and the survey relation LWA = LvA + 10 lg(S/S0) + 10 lg epsilon" width="92%"></picture>

## 3. When the radiation-factor assumption breaks

The whole method stands on one substitution: replacing the acoustic
measurement by `ε`. The Part 1 value `ε = 1` is close to the true radiation
factor only above the **critical (coincidence) frequency** of plate-like
parts, where bending waves travel faster than sound and the surface radiates
like a piston. Below coincidence, adjacent zones of the
plate move in antiphase and their radiation largely cancels: `ε` drops far
below one and falls quickly with decreasing frequency, so the survey method
can overstate the low-frequency bands of a large thin casing by 10 dB and
more. The same cancellation makes small sources radiate poorly (the acoustic
short circuit around an unbaffled panel). Two further assumptions are easy
to violate in the field:

* **The measured vibration must be the machine's own.** Vibration fed in
  from neighbouring machinery inflates `⟨v²⟩`; Table 2 prescribes the
  source-off check and `extraneous_velocity_correction` applies it.
* **The surface must be the dominant radiator.** Airborne sound from
  openings, intakes or internal sources that bypasses the measured casing is
  invisible to a velocity survey; the method characterises the
  structure-borne part only.

Part 2 exists exactly for the radiation-factor problem: it replaces the
fixed `ε = 1`
with a band-by-band `εⱼ` determined from one reference measurement of the
radiated power (ISO 9614 intensity), after which the velocity survey can be
repeated cheaply on nominally identical machines.

## 4. The measurement report (`.report()`)

A determination ends as a *document*. `VibrationSoundPowerResult.report(path)`
writes a one-page PDF fiche laid out like a sound-power test sheet: the
standard-basis line naming the applied method (the ISO/TS 7849-1 survey method
with a fixed radiation factor `ε = 1`, or the ISO/TS 7849-2 engineering method
with a determined radiation factor), an optional metadata header (client,
machine/source, test environment, instrumentation, climate, date), a per-band
table (nominal octave/one-third-octave frequency, the surface vibratory
velocity level `Lv` and the band sound-power level `LW`), the sound-power
spectrum `LW(f)` with a nominal band axis, and a boxed A-weighted sound power
level `LWA` (dB re 1 pW) with the total `LW`, the radiating area `S` and the
applied method alongside.

The relevant `ReportMetadata` fields are `client`, `specimen` (the
machine/source), `test_room` (the test environment), `instrumentation`,
`temperature`, `relative_humidity`, `pressure`, `test_date` and the footer
identity `laboratory`, `operator`, `report_id` and `notes`; the radiating area
`S` comes from the result itself. Supplying `requirement` adds a PASS/FAIL
verdict against a declared A-weighted sound-power limit (lower is better).
`verbose=True` adds the radiation factor `ε` column, and `language="es"`
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
  The upper-limit method with `ε = 1`.
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
power `P = Z_c⟨v²⟩Sε` (Formula 6), the velocity level and its calibration
(Formulae 3, 8), the mean over the surface (Formulae 10/11), the extraneous
correction (Table 2), the radiation factor (Formula 4/8) and the sound power
level (Formulae 12/15). Conformance is anchored on the standard's own worked
calibration example, the exact round-trip between the radiation factor and
`L_W = 10 lg(P/P₀)`, and the fixed impedance term.
