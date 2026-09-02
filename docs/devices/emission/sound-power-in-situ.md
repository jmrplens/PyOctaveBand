← [Documentation index](../../README.md)

# Sound Power in Situ by Comparison (ISO 3747)

Some machines never travel to a test room: a compressor bolted to the floor
of the plant, a press line, a packaging machine wired into the rest of the
line. ISO 3747 determines their sound power where they stand, by comparison.
A reference sound source (RSS) of calibrated band sound power is set beside
the machine, the same three or four microphones listen to each source in
turn from the reverberant part of the room, and because both sources see the
same room the room cancels: the sound power of the source under test is the
calibrated power of the reference source carried across by the difference of
the two mean levels. The method reaches engineering grade 2 where the field
is reverberant enough and the source not too directional, and survey grade 3
otherwise; an impulsive source gets a sound energy level instead of a sound
power level by the same route. Which route fits which job is weighed in
[Sound Power](sound-power.md).

## 1. The comparison in situ (ISO 3747)

The microphone positions must lie where the field is **reverberant**, which
the standard measures as the excess of sound pressure level over the free
field, $\Delta L_f \ge 7$ dB (clause 4.1, Annex A). Three or four positions
are used, at least 2 m apart, none closer than 0,5 m to any boundary,
distributed as evenly as possible round the machine (clause 7.4.1); the
reference source stands alongside the machine, never closer than 0,5 m to
its reference box, and where the machine is long it is run at several
locations along the sides (clause 7.3). Both sources and the background are
read at the same positions, the reference source for 30 s (clause 7.5), in
**octave bands from 125 Hz to 8 kHz** (clause 3.11); Table D.1 tabulates a
63 Hz row besides, for use only where the environment and the instrumentation
are satisfactory there (footnote a), which is why the API accepts that band
too.

The background is corrected **position by position** before anything is
averaged (clause 8.1, Eq. 7):

$$
K_{1i} = -10 \log_{10}\left(1 - 10^{-0.1\,\Delta L_{pi}}\right),
\qquad \Delta L_{pi} = L'_{pi(\mathrm{ST})} - L_{pi(\mathrm{B})},
$$

with three rules round it: a margin above 15 dB needs no correction, a
margin between 6 dB and 15 dB takes Eq. 7, and a margin **below 6 dB** caps
the correction at **1,3 dB** and turns the band into an upper bound that the
report must flag as not meeting the background requirement.
`sound_power_in_situ` applies all three, returns the per-position $K_{1i}$
and flags every band where some margin fell below 6 dB in
`background_requirement_met`. The corrected levels are energy-averaged over
the positions (Eq. 8, 9) and the sound power level in each octave band is
the comparison itself (clause 8.3.1, Eq. 11):

$$
L_W = L_{W(\mathrm{RSS})} - \overline{L_{p(\mathrm{RSS})}} + \overline{L_{p(\mathrm{ST})}} .
$$

With the reference source at $m$ locations, its calibrated powers and its
per-location means are each energy-averaged over the locations first
(clause 8.3.2, Eq. 12), and the correction of the reference source is
evaluated per location and per position. Table 1 of the standard, which
sorts the microphone positions by line of sight to each source, is guidance
for placing them and is not evaluated by the library.

```python
import numpy as np
from phonometry import emission

# A floor-standing screw compressor, 2.2 m x 1.4 m x 1.8 m, that cannot
# leave the plant: four microphone positions 1.5 m from its reference box,
# one reference sound source location alongside, octave bands 125 Hz - 8 kHz.
freqs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
lw_rss = np.array([84.5, 88.0, 91.0, 92.5, 92.0, 90.5, 87.0])  # calibrated
st = np.array([  # L'pi(ST): one row per microphone position, dB
    [82.4, 84.9, 86.1, 85.3, 82.6, 78.4, 72.1],
    [80.9, 84.1, 85.4, 84.7, 81.8, 77.5, 71.3],
    [83.1, 85.6, 86.8, 85.9, 83.2, 79.0, 72.8],
    [81.7, 84.5, 85.8, 85.0, 82.3, 78.0, 71.7],
])
rss = np.array([  # L'pi(RSS): the reference source at the same positions
    [78.2, 81.8, 84.6, 86.0, 85.6, 84.1, 80.4],
    [77.5, 81.1, 84.0, 85.4, 85.0, 83.4, 79.8],
    [78.9, 82.4, 85.1, 86.5, 86.1, 84.6, 80.9],
    [78.0, 81.6, 84.4, 85.8, 85.4, 83.9, 80.2],
])
background = np.array([  # Lpi(B): the factory floor, loud at the low end
    [75.0, 74.0, 72.0, 70.0, 67.0, 62.0, 57.0],
    [76.5, 75.0, 73.0, 71.0, 68.0, 63.0, 58.0],
    [74.5, 73.5, 71.5, 69.5, 66.5, 61.5, 56.5],
    [75.5, 74.5, 72.5, 70.5, 67.5, 62.5, 57.5],
])

res = emission.sound_power_in_situ(
    st, rss, lw_rss, freqs, background_levels=background,
    conditions=emission.GradeConditions(
        excess_levels=[8.2, 7.6, 8.9, 8.0],   # dLfA at each position (Annex A)
        directivity_range=4.0,                # +/-4 dB round the machine (7.2)
    ),
    sigma_omc=0.5,                        # a steady source (9.2 NOTE)
)
print(np.round(res.sound_power_level, 1))   # [88.8 91.5 92.6 91.8 88.9 84.7 78.6]
print(round(res.sound_power_level_a, 1))    # 96.1 dB(A)
print(res.background_requirement_met)       # [False  True  True  True  True  True  True]
print(np.round(res.background_correction[1], 2))  # position 2: [1.3 0.57 0.26 0.19 0.18 0.16 0.21]
print(res.grade, res.sigma_r0, round(res.expanded_uncertainty, 1))  # engineering 1.5 3.2
```

The 125 Hz band is the one to read twice: at one of the four positions the
margin over the floor is below 6 dB for the machine, and it is below 6 dB at
every position for the quieter reference source, so $K_1$ takes its 1,3 dB
cap and the 88,8 dB the band reports is an **upper bound**, which is what the
`False` says.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/in_situ_sound_power_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/in_situ_sound_power.svg" alt="Two panels over the seven octave bands from 125 Hz to 8 kHz. The upper panel plots the sound pressure levels at the four microphone positions: the corrected levels of the source under test as filled blue circles with their energy mean as a blue line, the corrected levels of the reference source as green squares with their mean as a green line, the measured levels before the background correction as hollow circles joined to the corrected ones wherever the correction moved them, and the background at each position as grey crosses. A boxed note at 125 Hz says the margin is below 6 dB at that position and at every position for the reference source, K1 is capped at 1.3 dB and the band is an upper bound. The lower panel draws the resulting sound power level of the source under test as blue bars, the 125 Hz bar hatched as an upper bound, with the calibrated sound power level of the reference source as a red dashed line with diamonds, and states in its title an A-weighted level of 96.1 dB(A), grade 2, and an expanded uncertainty of 3.2 dB with k = 2 and a 0.5 dB operating-condition deviation" width="96%"></picture>

*The comparison position by position: both sources at the four microphones
after the Eq. 7 correction above, with the hollow markers showing where each
level was before it, and Eq. 11 band by band below, with the calibrated
$L_{W(\mathrm{RSS})}$ beside it and the 125 Hz upper bound hatched.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt

# res is the InSituSoundPowerResult computed above. One line:
res.plot()   # LW bars, LWA in the title, the upper-bound band hatched
plt.show()

# By hand: both sources per position above, Eq. 11 below.
corrected_st = st - res.background_correction
corrected_rss = rss - res.background_correction_ref[0]
x = np.arange(freqs.size)
fig, (axt, axb) = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True)
for i, offset in enumerate(np.linspace(-0.27, 0.27, st.shape[0])):
    axt.plot(x + offset, corrected_st[i], "o", color="#1f77b4")
    axt.plot(x + offset, corrected_rss[i], "s", color="#2ca02c")
    axt.plot(x + offset, background[i], "x", color="#9e9e9e")
axt.plot(x, res.mean_source_level, "-", color="#1f77b4", label="mean, source under test")
axt.plot(x, res.mean_reference_level, "-", color="#2ca02c", label="mean, reference source")
axt.set_ylabel("Sound pressure level [dB]")
axt.legend()
bars = axb.bar(x, res.sound_power_level, width=0.62, color="#1f77b4")
for bar, met in zip(bars, res.background_requirement_met, strict=True):
    if not met:
        bar.set_hatch("//")
axb.plot(x, res.reference_power_level, "D--", color="#d62728", label="calibrated LW(RSS)")
axb.set_xticks(x, [f"{f:g}" for f in freqs])
axb.set_xlabel("Frequency [Hz]")
axb.set_ylabel("Sound power level LW [dB re 1 pW]")
axb.legend()
plt.show()
```

</details>

A machine that is long relative to the measurement distance gets the
reference source at several locations along its sides (clause 7.3.3), and
the levels arrive as one grid per location:

```python
# The same compressor with the reference source run at two locations, one
# along each long side; the second sits nearer a wall and reads 1.2 dB lower,
# and was calibrated in a similar position 0.4 dB lower.
two = emission.sound_power_in_situ(
    st, np.stack([rss, rss - 1.2]), np.stack([lw_rss, lw_rss - 0.4]), freqs,
    background_levels=background,
)
print(two.reference_levels.shape)               # (2, 7): mean level per location (Eq. 10)
print(np.round(two.sound_power_level, 1))       # [89.1 92.  93.  92.2 89.3 85.1 78.9]
print(round(two.sound_power_level_a, 1))        # 96.5 dB(A)
```

### `sound_power_in_situ()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `levels` | 2D array | dB | `(n, bands)` | Measured $L'_{pi(\mathrm{ST})}$, one row per microphone position, three or four positions (7.4.1) |
| `levels_ref` | 2D or 3D array | dB | `(n, bands)` or `(m, n, bands)` | The reference source at the same positions, one grid per location, corrected per its manufacturer but not for background |
| `lw_ref` | 1D or 2D array | dB | `(bands,)` or `(m, bands)` | Calibrated $L_{W(\mathrm{RSS})}$, per location when each was calibrated in its own similar position (Eq. 12) |
| `frequencies` | 1D array | Hz | 63 Hz to 8 kHz octaves | Nominal octave mid-band frequencies of Table D.1; required. Clause 3.11 puts the general-purpose range at 125 Hz to 8 kHz; the 63 Hz row of Table D.1 carries footnote a, which allows it only where the test environment, the reference sound source and the instrumentation are satisfactory at that frequency |
| `background_levels` | 1D or 2D array | dB | `(bands,)` or `(n, bands)` | $L_{pi(\mathrm{B})}$; one spectrum serves every position. `None` warns and leaves `background_requirement_met` `False` in every band, since 7.5 measures it at each position and 8.1 needs the margin |
| `background_levels_ref` | 1D or 2D array | dB | same shapes | Background for the reference-source reading; `None` reuses `background_levels` (7.5) |
| `temperature` | float | °C | default `23.0` | Air temperature at the test, for $C_2$ |
| `static_pressure` | float | kPa | default `101.325` | Static pressure at the test, for $C_2$; see `static_pressure_from_altitude` |
| `conditions` | `GradeConditions` | | | The two conditions Table 2 reads together: `excess_levels`, the $\Delta L_{f\mathrm{A}}$ at each position `(n,)` in decibels (Annex A), and `directivity_range`, the half-width of the A-weighted directivity survey in decibels (7.2). Either one left out leaves the determination at survey grade |
| `sigma_omc` | float | dB | $\ge 0$ | Standard deviation of the operating and mounting conditions (9.2); `None` leaves the uncertainty `NaN` |
| `coverage_factor` | float | | default `2.0` | $k$ of Eq. 23; 1,6 for a one-sided comparison with a limit |

The function returns an `InSituSoundPowerResult`: `sound_power_level` (Eq. 11
or 12, at the conditions of the test), `mean_source_level` and
`mean_reference_level` (Eq. 8, 9), `reference_levels` per location (Eq. 10),
`reference_power_level`, `background_correction` per position and band,
`background_correction_ref` per location, position and band,
`background_requirement_met` per band, `c2`, `grade`, `sigma_r0`,
`sigma_omc`, `sigma_tot`, `expanded_uncertainty`, `coverage_factor`,
`sound_power_level_a` and `quantity`. `sound_power_level_ref` adds `c2` to
the level (section 3).

## 2. Impulsive sources: the sound energy level (Eq. 13 to 20)

A press stroke or a door slam has no steady power to report. Clause 8.4
measures the **single event level** $L_E$ at each position instead, either
one event at a time, at least five of them (clause 7.6), or once over a run
of $N$ successive events, and clause 8.5 turns the mean into a **sound energy
level** $L_J$ in dB re 1 pJ by the same comparison. Measured one at a time,
each event is corrected for background with Eq. 14 and the corrected levels
are energy-averaged into the mean single event level of the position
(Eq. 13, 15); measured once over $N$ events, the level is corrected and
reduced by $10 \log_{10} N$ to one event (Eq. 16, 17). The per-position
levels are energy-averaged (Eq. 18), and

$$
L_J = L_{W(\mathrm{RSS})} - \overline{L_{p(\mathrm{RSS})}} + \overline{L_{E(\mathrm{ST})}}
$$

(Eq. 19), or its several-location form (Eq. 20). `sound_energy_in_situ`
takes the events as a 3D `(n, N, bands)` array or as a 2D `(n, bands)`
measurement with `events=N`, and returns the same result type with
`quantity='energy'`, the level in `sound_energy_level` and the Annex D total
in `sound_energy_level_a`.

Eq. 14 subtracts the **time-averaged** background level from a
**time-integrated** event level, asking only that both be measured over the
same integration time $T$; as printed the difference is a true margin for
$T$ = 1 s, and for a longer $T$ the background holds $10 \log_{10}(T/T_0)$ dB
more energy over the event's interval (clause 3.4, NOTE 1). ISO 3741 and ISO
3744 print the same line, so it is the family's convention and not a
misprint of this part; the library applies Eq. 14 as printed by default and
offers `integration_time` to carry the background to the event's interval
first.

```python
# The same room, a press beside the compressor: six strokes measured one at a
# time at the four positions, each integrated over 4 s, with the same
# reference source reading and the same floor.
strokes = np.array([
    [96.3, 98.1, 99.4, 98.8, 96.0, 92.1, 86.4],
    [95.1, 97.4, 98.7, 98.0, 95.3, 91.4, 85.6],
    [97.0, 98.9, 100.1, 99.5, 96.6, 92.8, 87.0],
    [95.8, 97.9, 99.0, 98.4, 95.8, 91.9, 86.0],
])
events = np.stack([strokes + d for d in (-0.3, 0.2, 0.0, 0.4, -0.2, 0.1)], axis=1)
energy = emission.sound_energy_in_situ(
    events, rss, lw_rss, freqs,                 # (4 positions, 6 events, 7 bands)
    background_levels=background, integration_time=4.0,
    conditions=emission.GradeConditions(excess_levels=[8.2, 7.6, 8.9, 8.0], directivity_range=4.0), sigma_omc=2.0,
)
print(energy.quantity)                            # energy
print(np.round(energy.sound_energy_level, 1))     # [103.7 105.3 106.1 105.3 102.4  98.6  93. ]
print(round(energy.sound_energy_level_a, 1))      # 109.7 dB(A) re 1 pJ
print(round(energy.expanded_uncertainty, 1))      # 5.0 dB: grade 2 with sigma_omc = 2 dB
```

The 5,0 dB is the standard's own example in clause 9.5, reproduced because
the inputs are the same: grade 2, $\sigma_\mathrm{omc}$ = 2,0 dB, $k$ = 2.

### `sound_energy_in_situ()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `event_levels` | 3D or 2D array | dB | `(n, N, bands)` or `(n, bands)` | $L'_{Ei,q(\mathrm{ST})}$ per event, or $L'_{Ei,N(\mathrm{ST})}$ of one measurement over $N$ events |
| `events` | int | | $\ge 1$ | $N$ for the 2D form (Eq. 17); must be `None` with the 3D form. Clause 7.6 asks for at least five events, so fewer than five warns and the determination is nonconforming, on either form |
| `integration_time` | float | s | $> 0$ | $T$ of the event measurement; `None` applies Eq. 14 as printed |
| `levels_ref`, `lw_ref`, `frequencies`, `background_levels`, `background_levels_ref`, `temperature`, `static_pressure`, `excess_levels`, `directivity_range`, `sigma_omc`, `coverage_factor` | | | | As in `sound_power_in_situ()` |

## 3. Grade, uncertainty and reference conditions

**Grade (Table 2).** Engineering grade 2 needs the A-weighted excess
$\Delta L_{f\mathrm{A}} \ge 7$ dB at every microphone position and a source
directivity range within ±7 dB (clause 7.2); either indicator failing, or
not determined, gives survey grade 3. The table pairs each grade with a
typical upper bound of the reproducibility of the method, **1,5 dB** for
grade 2 and **4,0 dB** for grade 3. The excess itself is Annex A (Eq. A.1),

$$
\Delta L_f(r) = L_{p(\mathrm{RSS}),r} - L_{W(\mathrm{RSS})} + 11\ \mathrm{dB} + 20 \log_{10}\frac{r}{r_0},
\qquad r_0 = 1\ \mathrm{m},
$$

which `excess_sound_pressure_level` evaluates.

**Uncertainty (clause 9).** $\sigma_\mathrm{tot} = \sqrt{\sigma_{R0}^2 +
\sigma_\mathrm{omc}^2}$ (Eq. 22) and $U = k\,\sigma_\mathrm{tot}$ (Eq. 23),
with $k$ = 2 for the two-sided 95 % interval or 1,6 for a one-sided
comparison with a limit value; $\sigma_\mathrm{omc}$ is the user's number
(9.2, E.3), near 0,5 dB for a steady source, 2 dB with material flow, 4 dB
for a press under load. The standard's example (9.5), grade 2 with
$\sigma_\mathrm{omc}$ = 2,0 dB and $k$ = 2, gives
$U = 2\sqrt{1{,}5^2 + 2^2}$ dB = 5 dB. Without a `sigma_omc` the result
leaves `sigma_tot` and `expanded_uncertainty` `NaN`.

**Reference conditions (Annex C).** The radiation-impedance correction
$C_2 = -10 \log_{10}(p_\mathrm{s}/p_{\mathrm{s},0}) + 15 \log_{10}((273{,}15 + \theta)/296)$
carries the level to 101,325 kPa and 23,0 °C, the same $C_2$ as ISO 3741
clause 9.1.4, and Eq. C.2 estimates the static pressure from the altitude,
$p_\mathrm{s} = p_{\mathrm{s},0}\,(1 - a H_\mathrm{a})^b$ with $a$ = 2,2560 × 10⁻⁵ m⁻¹
and $b$ = 5,2553. The result exposes `c2` and the properties
`sound_power_level_ref` and `sound_energy_level_ref` (Eq. C.1, C.3).
$\theta_\mathrm{ref}$ is printed as 296 K beside a reference temperature of
23,0 °C (296,15 K), so at exactly the reference conditions $C_2$ is
+0,003 3 dB and not zero; ISO 3741 and ISO 3744 print the same 296 K, so the
library keeps the family's rounding.

```python
# A plant 640 m above sea level, at 27 degC, with no barometer on site.
ps = emission.static_pressure_from_altitude(640.0)
print(round(ps, 2))                                 # 93.87 kPa (Eq. C.2)
high = emission.sound_power_in_situ(
    st, rss, lw_rss, freqs, background_levels=background,
    temperature=27.0, static_pressure=ps,
)
print(round(high.c2, 3))                            # 0.423 dB
print(round(float(high.sound_power_level[3]), 2), round(float(high.sound_power_level_ref[3]), 2))
# 91.76 at the test conditions, 92.18 under the reference conditions (1 kHz)

# Annex A along a line of sight from the reference source, A-weighted.
print(np.round(emission.excess_sound_pressure_level([81.0, 76.5, 73.8], 92.5, [1.5, 3.0, 6.0]), 2))
# [3.02 4.54 7.86]: reverberant enough for grade 2 only from about 6 m out
```

**A-weighted totals (Annex D).** $L_{W\mathrm{A}}$ and $L_{J\mathrm{A}}$ are the
energy sums of the band levels plus the Table D.1 corrections $C_k$ (−26,2,
−16,1, −8,6, −3,2, 0,0, 1,2, 1,0 and −1,1 dB from 63 Hz to 8 kHz, the ISO
3744 Annex E octave values digit for digit); every `frequencies` entry must
be one of those eight nominal centres, distinct and in ascending order.

## What this guide covers

**Covered.** The ISO 3747 comparison in situ (`sound_power_in_situ`): the
per-position background correction of clause 8.1 with its 6 dB and 15 dB
rules and the 1,3 dB cap, the mean corrected levels of clause 8.2, the sound
power level for one (Eq. 11) or several (Eq. 12) reference-source locations,
the single-event levels of clause 8.4 in both forms and the sound energy
level of clause 8.5 (`sound_energy_in_situ`), the Table 2 grade with its
$\sigma_{R0}$, the clause 9 uncertainty (Eq. 22, 23), the Annex A excess of
sound pressure level (`excess_sound_pressure_level`), the Annex C reference
conditions with Eq. C.2 (`static_pressure_from_altitude`) and the Annex D
A-weighted totals.

**Not covered.** The Annex E uncertainty budget (Table E.2 and its
sensitivity coefficients) is documented but not modelled: $\sigma_{R0}$ is
the typical upper bound of Table 2 by grade, not a machine-specific value
from a round robin (9.3.2) or from the model of 9.3.3. The zoning of Table 1,
the directivity survey of clause 7.2 and the placement rules of clause 7.3
are the operator's decisions, made before the levels exist. The reference
source's own corrections for speed, temperature and static pressure are its
manufacturer's, applied before the levels arrive.

## See also

- [Sound Power](sound-power.md): choosing among the determination routes,
  the accuracy grades and the ISO 4871 noise-emission declaration.
- [Sound Power in the Reverberation Room](sound-power-reverberation.md): the
  precision comparison of ISO 3741 in a qualified room, whose $C_2$ this
  guide shares.
- [Sound Power by Pressure Methods](sound-power-pressure.md): the enveloping
  surface for a source that can be measured in its direct field.
- [Sound Power by Intensity Scanning](sound-power-intensity.md): the routes
  that tolerate steady background noise without a reference source.

## References

- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Engineering/survey methods for use in situ
  in a reverberant environment* (ISO 3747:2010).
  [iso.org catalogue](https://www.iso.org/standard/46426.html).
  The comparison method of this guide: the background correction of clause
  8.1, the mean levels and the sound power and sound energy levels of clauses
  8.2 to 8.5, the Table 2 grades, the clause 9 uncertainty and Annexes A, C
  and D.
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for reverberation test
  rooms* (ISO 3741:2010).
  [iso.org catalogue](https://www.iso.org/standard/52053.html).
  The precision comparison method of a qualified room, whose Eq. 21 is
  Eq. 11 of this guide plus the same radiation-impedance correction $C_2$.

## Standards

ISO 3747:2010, *Acoustics — Determination of sound power levels and sound
energy levels of noise sources using sound pressure — Engineering/survey
methods for use in situ in a reverberant environment*: the comparison with
a reference sound source, the background correction of clause 8.1, the
several-location forms, the single-event path, the Table 2 grades, the
clause 9 uncertainty and Annexes A, C and D.
