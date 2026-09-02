← [Documentation index](../../README.md)

# Sound Power by Pressure Methods (ISO 3744 / ISO 3746 / ISO 3745)

Of the standardised routes to the sound power level $L_W$, the pressure
methods are the ones that need nothing more exotic than a sound level meter:
sample the sound pressure on a surface that envelops the source, energy-average
it, correct it, and add the surface term. This guide covers the three of
them. The **enveloping-surface** methods of ISO 3744 (engineering grade) and
ISO 3746 (survey grade) work in situ, over one or more reflecting planes,
and clean the surface level with the background-noise correction $K_1$ and
the environmental correction $K_2$. The **precision method** of ISO 3745
moves the same idea into a qualified anechoic or hemi-anechoic room, where a
fixed microphone array samples the free field directly and the grade-1
corrections are meteorological rather than environmental. Section 3 renders
the determination as the accredited-style test fiche, and section 4 turns the
same surface on a source that does not run steadily at all, the noise burst
whose descriptor is the sound energy level $L_J$ of clause 8.3. Which
route fits which job, and the reverberation-room and intensity alternatives,
are weighed in [Sound Power](sound-power.md).

## 1. Enveloping surface, sound pressure (ISO 3744 / ISO 3746)

Place the source on a reflecting plane and imagine a **measurement surface**
of area $S$ wrapping it: a hemisphere for a compact source, a box (right
parallelepiped) for a large or elongated one. Sample the sound pressure
level at an array of microphone positions on that surface, energy-average
them, and the sound power follows because a diffuse-enough surface captures
all the radiated energy:

$$
\bar{L}_p = 10\log_{10}\left( \frac{1}{N_\mathrm{M}} \sum_i 10^{L_{pi}/10} \right), \qquad
L_W = \bar{L}_p - K_1 - K_2 + 10\log_{10}\frac{S}{S_0},\quad S_0 = 1\ \text{m}^2 .
$$

Two corrections clean up the surface level. The **background-noise
correction** removes the energy that would have been there with the source
switched off, from the margin $\Delta L_p$ between source-on and background
levels,

$$
K_1 = -10\log_{10}\left( 1 - 10^{-\Delta L_p/10} \right),
$$

and the **environmental correction** removes the reverberant build-up of the
test room from its equivalent absorption area $A$,

$$
K_2 = 10\log_{10}\left( 1 + \frac{4 S}{A} \right).
$$

The surface area is a closed form of the geometry: a hemisphere is
$S = 2\pi r^2$ over one reflecting plane (halved and quartered for two and
three planes), and a one-plane box is $S = 4(ab + bc + ca)$ with
$a = 0.5\,l_1 + d$, $b = 0.5\,l_2 + d$, $c = l_3 + d$ for measurement distance
$d$. ISO 3746 (survey) shares every formula but is coarser: fewer
microphone positions, a 3 dB background criterion instead of 6 dB, and
validity up to $K_2 \le 7\ \text{dB}$ instead of 4 dB.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_sound_power_surfaces_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_sound_power_surfaces.svg" alt="Measurement surfaces of ISO 3744: a hemisphere of radius r enveloping a compact source on a reflecting plane, and a right parallelepiped (box) at measurement distance d around a large source, both with microphone positions marked" width="88%"></picture>

```python
import numpy as np
from phonometry import emission

# Octave-band SPL (dB) at the 10 hemisphere positions of ISO 3744 (Annex B),
# with the source running, plus the background spectrum with it switched off.
freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
base = np.array([70.0, 74.0, 78.0, 80.0, 79.0, 76.0, 72.0, 66.0])
rng = np.random.default_rng(0)
levels = base + rng.normal(0.0, 0.5, size=(10, 8))     # (positions, bands)
background = np.full((10, 8), 55.0)

# ISO 3744 Annex B microphone coordinates on a radius-1.5 m hemisphere.
mic_xyz = emission.measurement_positions("hemisphere", radius=1.5, reflecting_planes=1)
print(mic_xyz.shape)                                    # (10, 3)

res = emission.sound_power_pressure(
    levels, "hemisphere", radius=1.5, reflecting_planes=1,
    background_levels=background, frequencies=freqs,
    room=emission.RoomEnvironment(reverberation_time=0.6, volume=300.0),  # -> K2
)
print(round(res.surface_area, 2))                       # 14.14 m^2 (= 2*pi*1.5^2)
print(round(float(res.environmental_correction[0]), 2)) # K2 = 2.32 dB
print(round(res.sound_power_level_a, 1))                # LWA = 92.4 dB
print(round(res.uncertainty, 1))                        # U = 3.0 dB (2*sigma_R0)
print(np.round(res.sound_power_level, 1))               # per-band LW

res.plot()   # sound power level bars per band, LWA in the title (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_power_pressure_result_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_power_pressure_result.svg" alt="The enveloping-surface sound power level spectrum of the ISO 3744 hemisphere example, one bar per octave band from 63 Hz to 8 kHz peaking near 500 Hz, with the A-weighted total of 92.4 dB(A) in the title" width="88%"></picture>

*One bar per band: the energy-averaged surface pressure minus the background
($K_1$) and environmental ($K_2$) corrections plus the surface term
$10\log_{10}(S/S_0)$ gives $L_W(f)$, and the A-weighted energy sum across bands
gives the single-number $L_{W\mathrm{A}}$ in the title.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import emission

# Octave-band SPL (dB) at the 10 hemisphere positions of ISO 3744 (Annex B),
# with the source running, plus the background spectrum with it switched off.
freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
base = np.array([70.0, 74.0, 78.0, 80.0, 79.0, 76.0, 72.0, 66.0])
rng = np.random.default_rng(0)
levels = base + rng.normal(0.0, 0.5, size=(10, 8))     # (positions, bands)
background = np.full((10, 8), 55.0)
res = emission.sound_power_pressure(
    levels, "hemisphere", radius=1.5, reflecting_planes=1,
    background_levels=background, frequencies=freqs,
    room=emission.RoomEnvironment(reverberation_time=0.6, volume=300.0),  # -> K2
)

# res is the SoundPowerResult computed above. One line:
res.plot()
plt.show()

# By hand: a bar spectrum of LW with the A-weighted total in the title.
freqs = res.frequencies
positions = np.arange(freqs.size)
fig, ax = plt.subplots()
ax.bar(positions, res.sound_power_level, width=0.7, color="#1f77b4")
ax.set_xticks(positions)
ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Sound power level LW [dB]")
ax.set_title(
    f"Enveloping-surface sound power (ISO 3744)  "
    f"LWA = {res.sound_power_level_a:.1f} dB(A)")
plt.show()
```

</details>

The A-weighted total $L_{W\mathrm{A}}$ is combined from the band powers with the
ISO 3744
Annex E A-weighting corrections, so it needs `frequencies`. Passing a
`RoomEnvironment` as `room` (`reverberation_time` + `volume`, or
`absorption_area`, or `mean_absorption_coefficient` + `room_surface`) enables
$K_2$; omit it and the field is treated as free ($K_2 = 0$). If the background margin drops below the
grade criterion or $K_2$ exceeds the validity limit, a `SoundPowerWarning`
flags that the levels are upper bounds; the determination still returns.

### K1 and K2 pitfalls

Both corrections subtract energy from the surface level, so overestimating
either one understates the emission. That is why the standards cap them, and
why most disputes over an enveloping-surface result trace back to one of
these habits:

- **$K_1$ has a cliff, not a slope.** At a 15 dB margin the correction is a
  negligible 0.14 dB; at the 6 dB engineering criterion it is already
  1.26 dB, the largest value the grade accepts. Below the criterion the
  standard does not let the formula run on: $K_1$ is capped and the result is
  reported as an upper bound. Never extrapolate the subtraction into a
  smaller margin; raise the margin (quieter site, closer surface) or switch
  to the intensity method.
- **$K_1$ assumes a stationary background.** The source-off reading must be
  taken at the same positions with the room in the same state, and the
  background energy must be the same during both readings. A ventilation
  system that cycles or a vehicle passing during either reading invalidates
  the pair; the energy subtraction also assumes source and background are
  incoherent, which holds for unrelated noise but not for the source's own
  reflections.
- **$K_2$ removes the average room build-up, not discrete reflections.** A
  nearby wall, a trolley or another machine just outside the surface adds a
  specular contribution concentrated at a few microphones. That imbalance
  shows up in the apparent directivity index $DI_i^*$, and no room-average
  correction can remove it: move the surface, remove the reflector or treat
  it with absorption.
- **$K_2$ is only as good as $A$.** With $A$ from Sabine ($0.16\,V/T$),
  errors in the reverberation time or the volume propagate directly. At the
  $K_2 = 4\ \text{dB}$ validity limit about 60 % of the measured energy is
  room, not source, and a 20 % error in $A$ still moves $L_W$ by about
  0.5 dB. Prefer a measured $T_{60}$ over a guessed absorption coefficient,
  and keep the measurement distance small enough that $K_2$ stays well under
  the limit.

### `sound_power_pressure()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `levels_positions` | 2D array | dB | `(NM, NB)` | One row per position, one column per band (or a single A-weighted column) |
| `surface` | str | — | `'hemisphere'` / `'box'` | Measurement-surface shape |
| `radius` | float | m | > 0 (hemisphere) | Hemisphere radius $r$ |
| `dimensions` | (float, float, float) | m | > 0 (box) | Reference-box $(l_1, l_2, l_3)$ |
| `distance` | float | m | > 0 (box) | Measurement distance $d$ |
| `reflecting_planes` | int | — | `1` / `2` / `3`, default `1` | Halves/quarters the hemisphere area |
| `background_levels` | 2D array or spectrum | dB | `(NM, NB)`, or `(NB,)` / `(1, NB)` | Enables $K_1$; a single spectrum broadcasts to every position |
| `frequencies` | 1D array | Hz | nominal band centres | Enables $L_{W\mathrm{A}}$ (Annex E) |
| `room` | `RoomEnvironment` or None | — | default `None` (free field) | The room data behind $K_2$; its fields are the three routes to $A$ below |
| `room.absorption_area` | float or 1D array | m² | > 0 | $A$ for $K_2$ (direct); per-band array → per-band $K_2$ |
| `room.reverberation_time`, `room.volume` | float/array, float | s, m³ | > 0 | $A = 0.16\,V/T$ for $K_2$; per-band $T$ → per-band $K_2$ |
| `room.mean_absorption_coefficient`, `room.room_surface` | float/array, float | —, m² | `(0,1]`, > 0 | $A = \alpha\,S_v$ (Eq. A.7); per-band $\alpha$ → per-band $K_2$ |
| `grade` | str | — | `'engineering'` (default) / `'survey'` | ISO 3744 vs ISO 3746 |
| `omc_uncertainty` | float | dB | default `0.0` | $\sigma_\text{omc}$, operating/mounting instability, folded into $U$ |

Returns a `SoundPowerResult`: `sound_power_level` (per-band $L_W$),
`surface_pressure_level` ($L_p$ after $K_1$/$K_2$), `mean_pressure_level`,
`background_correction`/`environmental_correction` ($K_1$/$K_2$),
`directivity_index` (apparent $DI_i^*$ per microphone position **and** frequency
band, shape `(NM, NB)`; ISO 3744 clause 8.6), `surface_area`,
`sound_power_level_a` ($L_{W\mathrm{A}}$), `uncertainty` (expanded, 95 %) and `grade`.
`measurement_positions('hemisphere', radius=…, reflecting_planes=…, tones=…,
grade=…)` returns the normative `(N, 3)` microphone coordinates (Table B.1 for
tonal sources, B.2 for broadband). Those coordinates plot directly with
`plot_microphone_positions`, which draws the array on its measurement
surface in 3-D, numbered as in the standard.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_positions_hemisphere_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/microphone_positions_hemisphere.svg" alt="Three-dimensional view of the ISO 3744 microphone array: ten numbered microphones on a 2 m wireframe hemisphere standing on a shaded circular reflecting plane, with positions 1 and 2 close to the plane, the others staggered in height up to positions 9 and 10 near the top, and the x, y and z axes graduated in metres" width="88%"></picture>

*Where the ten Annex B microphones actually sit on the 2 m hemisphere: the
heights are staggered so the array samples the whole surface evenly, which
is what lets the plain energy average of the ten levels stand in for the
surface integral.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import emission

# The 10 ISO 3744 Annex B microphones on a 2 m hemisphere.
emission.plot_microphone_positions(emission.measurement_positions("hemisphere", radius=2.0),
                                   radius=2.0)
plt.show()
```

</details>

## 2. Precision grade, anechoic room (ISO 3745)

When the highest accuracy is required, ISO 3745 measures sound power in a
qualified **anechoic** or **hemi-anechoic** room, where the free field lets a
fixed array of microphones sample the radiated sound pressure directly. It is the
grade-1 counterpart to the enveloping-surface method of Section 1, with
standardized microphone coordinates, a per-position background correction and an
explicit meteorological correction.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_precision_anechoic_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_precision_anechoic.svg" alt="ISO 3745 precision sound power in an anechoic room: wedge-lined walls, the device under test at the centre and a hemispherical array of microphones at a fixed radius, with the sound power level formed from the surface-averaged pressure plus the area, background and meteorological corrections" width="92%"></picture>

**Sound power level (Clause 8).** The band sound power level is the
surface-averaged pressure level plus the surface term and the corrections:

$$
L_W = \overline{L_p} + 10\log_{10}\frac{S}{S_0} + C_1 + C_2 + C_3,
$$

with $S = 4\pi r^2$ over the sphere or $S = 2\pi r^2$ over the hemisphere,
$S_0 = 1\ \text{m}^2$. $C_1$ and $C_2$ are the meteorological corrections
(reference and radiation-impedance terms); $C_3$ accounts for air absorption over
the measurement radius. The microphone positions are the standardized
unit-vector arrays of Tables D.1 (sphere), E.1 (hemisphere) and E.2 (hemisphere,
broadband).

```python
import numpy as np
from phonometry import emission

# The 40 standardized hemisphere positions (unit vectors scaled by the radius).
pos = emission.precision_positions("hemisphere", radius=1.0, count=40)
print(pos.shape)                      # (40, 3)

# Octave/third-octave band SPL (dB) at each of the 40 positions; here a uniform
# 74 dB in one band. The result carries S = 2*pi*r^2 and LW with C1+C2+C3.
levels = np.full((40, 1), 74.0)
res = emission.sound_power_anechoic(levels, "hemisphere", radius=1.0)
print(round(res.surface_area, 3))                 # 6.283  (2*pi*1^2)
print(np.round(res.sound_power_level, 2))         # [81.85]
```

**Background and meteorological corrections.** The $K_1$ background correction is
applied **per position** and floored where the signal-to-background difference is
small (Eq. 11); the meteorological correction is evaluated from the measured
temperature and static pressure.

```python
import numpy as np
from phonometry import emission

# K1 for a 6 dB signal-to-background difference in a <=200 Hz edge band: the
# floor is 1.26 dB (Eq. 11). Source and background levels are [positions, bands].
k1 = emission.precision_background_correction(
    np.array([[56.0]]), np.array([[50.0]]), np.array([200.0]))
print(round(float(k1[0, 0]), 4))      # 1.2563

# Meteorological corrections at the 23 C, 101.325 kPa reference (Eq. 16):
mc = emission.meteorological_corrections(23.0, 101.325)
print(round(mc.c1, 4), round(mc.c2, 4))   # -0.1282 0.0

# Expanded uncertainty (Clause 10.5 EXAMPLE): sigma_R0 = 0.5, sigma_omc = 2.0,
# k = 2 -> U = 4.1 dB.
print(round(emission.precision_uncertainty(0.5, 2.0, 2.0), 3))   # 4.123
```

The `MeteorologicalCorrection` is a pair of scalars (plus the per-band $C_3$
when the attenuation coefficient is supplied per band) rather than a
plottable spectrum: the corrections fold into the
`PrecisionSoundPowerResult` as its `c1`/`c2`/`c3` fields, and the `.report()`
fiche prints them on its measurement-basis strip.

Over several bands `sound_power_anechoic` returns a plottable
`PrecisionSoundPowerResult` carrying the per-band $L_W$ and the A-weighted
total:

```python
import numpy as np
from phonometry import emission

# A mid-frequency-peaked machine measured over the 40-position hemisphere array
# (Annex E). levels_positions is the (40, NB) surface pressure spectrum: a base
# spectrum peaked near 1 kHz plus a small per-position spatial spread.
freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], float)
base = 70.0 + 8.0 * np.exp(-(np.log2(freqs / 1000.0) ** 2) / 2.0)
rng = np.random.default_rng(7)
levels = base[None, :] + rng.normal(0.0, 1.0, (40, freqs.size))

result = emission.sound_power_anechoic(levels, "hemisphere", radius=1.0, frequencies=freqs)
print(round(result.sound_power_level_a, 1))   # 89.3
result.plot()   # LW spectrum, LWA in the title (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/precision_anechoic_power_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/precision_anechoic_power.svg" alt="The precision sound power level spectrum of a mid-frequency-peaked machine measured over the ISO 3745 hemisphere array, one bar per band peaking near 1 kHz, with the A-weighted total of 89.3 dB(A) in the title" width="88%"></picture>

*One bar per band: the surface-averaged pressure plus the area, background and
meteorological corrections give $L_W(f)$, and the A-weighted energy sum across
bands gives the single-number $L_{W\mathrm{A}}$ in the title.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import emission

# A mid-frequency-peaked machine measured over the 40-position hemisphere array
# (Annex E). levels_positions is the (40, NB) surface pressure spectrum: a base
# spectrum peaked near 1 kHz plus a small per-position spatial spread.
freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], float)
base = 70.0 + 8.0 * np.exp(-(np.log2(freqs / 1000.0) ** 2) / 2.0)
rng = np.random.default_rng(7)
levels = base[None, :] + rng.normal(0.0, 1.0, (40, freqs.size))
result = emission.sound_power_anechoic(levels, "hemisphere", radius=1.0, frequencies=freqs)

# result is the PrecisionSoundPowerResult computed above. One line:
result.plot()
plt.show()

# By hand: a bar spectrum of LW with the A-weighted total in the title.
freqs = result.frequencies
positions = np.arange(freqs.size)
fig, ax = plt.subplots()
ax.bar(positions, result.sound_power_level, width=0.7, color="#1f77b4")
ax.set_xticks(positions)
ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Sound power level LW [dB]")
ax.set_title(
    f"Precision sound power (ISO 3745)  LWA = {result.sound_power_level_a:.1f} dB(A)")
plt.show()
```

</details>

## 3. The measurement report (`.report()`)

A sound power determination ends as a *document*. Every result of this page
stays plottable while it is being worked on (`res.plot()` draws the same
$L_W$ spectrum interactively that the fiche typesets), and the report step
wraps it into the deliverable. Both the enveloping-surface result
(`SoundPowerResult`, ISO 3744/3746) and the precision result
(`PrecisionSoundPowerResult`, ISO 3745) expose a `.report()` method that writes
a one-page PDF fiche laid out like a sound-power test sheet: the standard-basis
line naming the applied method and accuracy grade, an optional metadata header
(client, noise source, test environment, instrumentation, climate, date), a
per-band table (nominal octave/one-third-octave frequency, the surface
sound-pressure level $L_p$ and the band sound-power level $L_W$), the
sound-power spectrum $L_W(f)$ with a nominal band axis, and a boxed A-weighted
sound power level $L_{W\mathrm{A}}$ (dB re 1 pW) with the total $L_W$, the expanded
uncertainty $U$ and the measurement surface area $S$ alongside.

The metadata is supplied through a `ReportMetadata`, whose applicable fields
here are the **source description** (`specimen`), the **test environment**
(`test_room`), the **client**, the **instrumentation**, the **temperature**,
**relative humidity** and **ambient pressure**, the **date of test**
(`test_date`) and the footer identity (`laboratory`, `operator`, `report_id`,
`notes`); the measurement surface area $S$ comes from the result itself and is
printed in the result box and the basis strip, together with the applied
corrections (the background $K_1$ and environmental $K_2$ for the ISO 3744/3746
surface method, or the meteorological $C_1$/$C_2$/$C_3$ for the ISO 3745
precision method). Supplying `requirement` adds a PASS/FAIL verdict against a
declared A-weighted sound-power limit (a sound-power emission is a quantity where
less is better, so the source passes at or below the limit). `verbose=True`
adds the energy-averaged level $L_p'$ to the table, and for the ISO 3744/3746
surface result it also adds the $K_1$/$K_2$ correction columns (the ISO 3745
precision result carries no $K_1$/$K_2$; its $C_1$/$C_2$/$C_3$ appear in the
basis strip). `language="es"` renders the Spanish fiche with comma decimals.

```python
import numpy as np
from phonometry import ReportMetadata, emission

freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
# Ten identical position spectra over a hemisphere of radius 4 m; background a
# uniform 10 dB below and an equivalent absorption area A = 1500 m^2 (so K1, K2
# are meaningful and within the engineering validity limit).
surface = np.array([72.0, 76, 80, 82, 81, 78, 73, 66])
res = emission.sound_power_pressure(
    np.tile(surface, (10, 1)), "hemisphere", radius=4.0,
    background_levels=np.tile(surface - 10.0, (10, 1)),
    frequencies=freqs, grade="engineering",
    room=emission.RoomEnvironment(absorption_area=1500.0),
)

res.report(
    "sound_power.pdf",
    metadata=ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Hemi-anechoic room over a reflecting floor",
        instrumentation="Class 1 sound level meter (IEC 61672-1), s/n 0042",
        laboratory="Phonometry reference example",
        report_id="EXAMPLE-3744",
        requirement=105.0,
    ),
)   # LWA = 103.7 dB(A) re 1 pW -> declared limit 105 dB(A): PASS
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 3744 sound power determination example report: a header with the client, the noise source, the hemi-anechoic test environment and the instrumentation and climate, the octave-band table (63 Hz to 8 kHz) of surface sound-pressure levels Lp and band sound-power levels LW, the sound-power spectrum LW(f) with a nominal band axis, the boxed A-weighted sound power level LWA = 103.7 dB(A) re 1 pW with the total LW = 105.8 dB, the expanded uncertainty U = 3.0 dB and the measurement surface S = 100.53 m2, and a PASS verdict against the declared 105 dB(A) limit, closed by a basis strip stating the applied K1 = 0.5 dB and K2 = 1.0 dB corrections](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3744_sound_power_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3744_sound_power_example.pdf)

*Sound power determination fiche (`SoundPowerResult.report`), an ISO 3744
engineering-grade hemisphere measurement with the $K_1$/$K_2$ corrections and
the boxed $L_{W\mathrm{A}}$.*

The precision result writes the same sheet from the ISO 3745 side, with the
meteorological corrections on its basis strip in place of the $K_1$/$K_2$ pair.
This is the 40-position hemisphere measurement of section 2, the one whose
spectrum is plotted above:

```python
from phonometry import ReportMetadata

result.report(
    "precision-sound-power.pdf",
    metadata=ReportMetadata(
        client="Example manufacturing plant",
        specimen="Mid-frequency-peaked machine",
        test_room="Qualified anechoic room, 40-position hemisphere array",
        measurement_standard="ISO 3745",
    ),
)   # LWA = 89.3 dB(A) re 1 pW, U = 4.1 dB
```

[![ISO 3745 precision sound power determination example report: a header with the client, the noise source and the qualified anechoic room with its 40-position hemisphere array, the octave-band table from 125 Hz to 8 kHz of surface sound-pressure levels Lp and band sound-power levels LW (78.0, 79.0, 82.6, 85.8, 82.6, 78.7 and 78.0 dB), the LW spectrum peaking at 1 kHz, and the boxed A-weighted sound power level LWA = 89.3 dB(A) re 1 pW with the total LW = 90.1 dB, the expanded uncertainty U = 4.1 dB and the measurement surface S = 6.28 m2, over a basis strip stating the applied meteorological corrections C1 = -0.13 dB, C2 = 0.00 dB and C3 = 0.0 dB](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3745_precision_power_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso3745_precision_power_example.pdf)

*Precision sound power fiche (`PrecisionSoundPowerResult.report`), the ISO 3745
anechoic determination: no $K_2$ at all, a 6.28 m² hemisphere instead of a
100 m² one, and $U$ = 4.1 dB because the operating-and-mounting term dominates
the reproducibility one.*


## 4. Sound energy level of a burst (clause 8.3)

Everything above describes a source that runs steadily for the whole averaging
interval: Eq. 18 reports a *rate* of energy flow. A press stroke, a door slam or
a pneumatic exhaust radiates its energy in a fraction of a second and then
stops, so the standard gives it the **sound energy level**
$L_J = 10\log_{10}(J/J_0)$, $J = \int P(t)\,\mathrm{d}t$, $J_0 = 1$ pJ (clauses
3.22 and 3.23), and determines it (clause 8.3; clause 8.4 of ISO 3746 for the
survey grade) by the chain of section 1 with the **single event time-integrated
sound pressure level** $L_E = 10\log_{10}[\int p^2\,\mathrm{d}t / E_0]$,
$E_0 = (20\ \mu\text{Pa})^2$ s (clause 3.4), in place of $L_p$: measured
through a window that encompasses the whole burst at every position at once
(no traversing microphone, clause 8.3.1), energy-averaged over the positions
(clause 8.3.3), corrected by the same $K_1$ (Eq. 21) and $K_2$ (Eq. 22) and
closed by the surface term (Eq. 23):

$$
L_J = \overline{L'_E} - K_1 - K_2 + 10\log_{10}\frac{S}{S_0}.
$$

The two quantities meet on a steady source: over a window of duration $T$ the
integral of a constant $p^2$ is $T p^2$, so $L_E = L_{p,T} + 10\log_{10}(T/T_0)$,
$T_0 = 1$ s (clause 3.4 NOTE 1), and $L_J = L_W + 10\log_{10}(T/T_0)$. That
identity is what the figure draws and what the library's tests pin the sound
energy chain to, field by field.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_energy_burst_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/sound_energy_burst.svg" alt="Two panels. Left, the running single event level of an impact burst and of a steady 80 dB source through a 10 s window: the burst starts at 2 s and its curve is flat at 90 dB within about a second, the steady source climbs as 10 lg(t/T0) and reaches the same 90 dB at the dashed line T = 10 s. Right, seven octave-band bars from 125 Hz to 8 kHz of the burst's sound energy level over a 2 m hemisphere, rising from 95 dB at 125 Hz to 103.5 dB at 1 kHz and falling to 91 dB at 8 kHz, with the A-weighted total of 107.7 dB(A) in the title" width="96%"></picture>

*Left, the running single event level of an impact burst and of a steady 80 dB
source through a 10 s window: the burst's 90 dB is all there within a second of
the impact, the steady source climbs as $10\log_{10}(t/T_0)$ and reaches the
same 90 dB at $T$. Right, the octave-band $L_J$ of the same press stroke over a
2 m hemisphere, with the A-weighted total in the title.*

The standard asks for at least **five** events (clause 8.3.1), one at a time or
as one reading that spans them all, and reduces both to the level of one event
at each position: the energy average of the $N_\mathrm{e}$ readings (Eq. 19) or
the one reading less $10\log_{10} N_\mathrm{e}$ (Eq. 20). `sound_energy_pressure`
takes `levels_positions` as a `(Ne, NM, NB)` array of the events one at a time,
an `(NM, NB)` reading with `events=Ne`, or the `(NM, NB)` mean already formed;
`mean_single_event_level` is the same reduction on its own. The background is
the time-averaged level measured over the same window (clause 8.3.1), and it is
compared as its exposure over that window, $L_{p(\mathrm{B})} + 10\log_{10}(T/T_0)$,
so that the energies Eq. 21 subtracts share one reference (the two levels are
re $E_0 = p_0^2 \cdot 1$ s and re $p_0^2$ respectively; the
[errata registry](../../ERRATA.md) records the reading); `integration_time` is
therefore required with `background_levels`, and the criteria and clamp are
section 1's.

```python
import numpy as np
from phonometry import emission

# A press stroke measured at the 10 hemisphere positions of ISO 3744 (r = 2 m),
# five strokes one at a time: (events, positions, bands), octaves 125 Hz - 8 kHz.
bands = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
stroke = np.array([84.0, 88.0, 91.0, 92.0, 90.0, 86.0, 80.0])   # single event levels, dB
rng = np.random.default_rng(83)
events = stroke + rng.normal(0.0, 0.8, size=(5, 10, bands.size))

burst = emission.sound_energy_pressure(
    events, "hemisphere", radius=2.0,
    background_levels=np.full((10, bands.size), 62.0),   # time-averaged over the same 10 s
    integration_time=10.0, frequencies=bands,
    room=emission.RoomEnvironment(reverberation_time=1.2, volume=900.0),   # -> K2
)
print(burst.events)                                        # 5 (Eq. 19 over the first axis)
print(round(float(burst.environmental_correction[0]), 2))  # K2 = 2.64 dB
print(np.round(burst.sound_energy_level, 1))               # per-band LJ, 95.1 ... 90.8 dB
print(round(burst.sound_energy_level_a, 1))                # LJA = 107.7 dB(A)

# The section 1 measurement read as a 10 s window: LJ = LW + 10 lg(T/T0) exactly.
ten_seconds = emission.sound_energy_pressure(
    levels + 10.0, "hemisphere", radius=1.5, reflecting_planes=1,
    background_levels=background, integration_time=10.0, frequencies=freqs,
    room=emission.RoomEnvironment(reverberation_time=0.6, volume=300.0),
)
print(np.round(ten_seconds.sound_energy_level - res.sound_power_level, 6))   # 10.0 in every band

# Annex G: a determination at 1 200 m and 8 C carried to 101.325 kPa and 23 C (Eq. G.3).
corr = emission.reference_atmosphere_correction(8.0, altitude=1200.0)
print(round(corr.static_pressure, 1), round(corr.total, 2))   # 87.7 kPa, C1 + C2 = 0.68 dB
lj_ref = burst.sound_energy_level + corr.total

burst.plot()   # LJ per band, LJA in the title (needs matplotlib)
```

Eq. 23 holds for the meteorological conditions of the test; above 500 m of
altitude or below 10 °C clause 8.3.6 requires the Annex G correction
$L_{J\mathrm{ref,atm}} = L_J + C_1 + C_2$ (Eq. G.3), which
`reference_atmosphere_correction` evaluates from the temperature and the static
pressure, or from the altitude through Eq. G.2. `sound_energy_pressure` takes the
surface, position and room arguments of `sound_power_pressure` plus `events`,
`background_levels` with `integration_time`, and returns a `SoundEnergyResult`
(`sound_energy_level`, `surface_event_level`, `mean_event_level`,
`background_correction`, `environmental_correction`, `directivity_index`,
`surface_area`, `sound_energy_level_a`, `uncertainty`, `grade`, `events`,
`integration_time`) with `.plot()`.

## See also

- [Sound Power](sound-power.md): choosing among the seven determination
  routes, what the accuracy grades promise, and the ISO 4871 noise-emission
  declaration a measured $L_{W\mathrm{A}}$ feeds.
- [Sound Power in the Reverberation Room (ISO 3741)](sound-power-reverberation.md):
  the precision-grade diffuse-field alternative when the source can travel
  to a qualified room.
- [Sound Power by Intensity Scanning (ISO 9614)](sound-power-intensity.md):
  the routes that tolerate the steady background noise a pressure method
  cannot subtract.
- [Room Acoustics](../../buildings/rooms/room-acoustics.md): the reverberation time and equivalent
  absorption area that feed $K_2$.
- [Levels](../../signals/levels/levels.md): energy averaging and the A-weighting behind $L_{W\mathrm{A}}$.
- [Theory](../../reference/theory/environment-transport.md): the $K_1$/$K_2$ and $C_1$/$C_2$
  derivations.
- API reference: [`emission.sound_power`](https://jmrplens.github.io/phonometry/reference/api/power/sound-power/).

## References

- Beranek, L. L., & Mellow, T. J. (2012). *Acoustics: Sound fields and
  transducers*. Academic Press. ISBN 978-0-12-391421-7.
  [doi:10.1016/C2011-0-05897-0](https://doi.org/10.1016/C2011-0-05897-0).
  Radiation and sound fields: the free-field relations between pressure and
  power that the enveloping-surface and anechoic methods rest on.
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Engineering methods for an essentially free
  field over a reflecting plane* (ISO 3744:2010).
  [iso.org catalogue](https://www.iso.org/standard/52055.html).
  The enveloping-surface method of section 1.
- International Organization for Standardization. (2012). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for anechoic rooms and
  hemi-anechoic rooms* (ISO 3745:2012).
  [iso.org catalogue](https://www.iso.org/standard/45362.html).
  The precision anechoic-room method of section 2.
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Survey method using an enveloping
  measurement surface over a reflecting plane* (ISO 3746:2010).
  [iso.org catalogue](https://www.iso.org/standard/52056.html).
  The survey grade of section 1, sharing the enveloping-surface formulae
  with coarser criteria.

## Standards

ISO 3744:2010, *Acoustics — Determination of sound power levels
and sound energy levels of noise sources using sound pressure — Engineering
methods for an essentially free field over a reflecting plane*: the
enveloping-surface method: hemisphere and box surface areas, the $K_1$/$K_2$
corrections, the Annex B microphone positions and the Annex E A-weighting.
ISO 3746:2010, *… Survey method using an enveloping measurement surface over a
reflecting plane*: the survey grade sharing the same formulae with coarser
criteria. ISO 3745:2012, *… Precision methods for anechoic rooms and
hemi-anechoic rooms*: the Clause 8 power level, the per-position background
correction (Eq. 11), the meteorological corrections and the standardized
microphone arrays of section 2.

The sound energy level $L_J$ of a noise burst over the same enveloping
surface is section 4: `sound_energy_pressure` (ISO 3744 clause 8.3, ISO 3746
clause 8.4) with `mean_single_event_level` (Eq. 19/20) and the Annex G
correction of `reference_atmosphere_correction` (Eq. G.1/G.3).

**Not covered.** Neither method performs the facility qualification it assumes:
ISO 3745's free-field qualification of the anechoic or hemi-anechoic
environment is taken for granted, and ISO 3744's $K_2$ validity only warns.
ISO 3744 **Annex G**, the correction to reference meteorological conditions
required above 500 m of altitude or below 10 °C (clauses 8.2.5 and 8.3.6), is
evaluated by `reference_atmosphere_correction` but never applied inside a
determination: `sound_power_pressure` and `sound_energy_pressure` take no
temperature or pressure argument, and the sum is added by hand. ISO 3745
defines no sound energy level of its own, so a burst in the precision anechoic
room has no route here. The
$C_3$ meteorological correction of ISO 3745 needs an air-absorption coefficient
the caller supplies through `air_absorption_coefficient=`; this module does not
compute it from **ISO 9613-1** itself.

