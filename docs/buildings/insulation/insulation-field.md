← [Documentation index](../../README.md)

# Field Insulation Measurement (ISO 16283)

This guide continues from the [Room Acoustics guide](../rooms/room-acoustics.md): the
same impulse response, measured either side of a partition, yields its sound
insulation. This page covers the engineering-grade measurement of ISO 16283
in the finished building: the airborne level differences $D$, $D_\mathrm{nT}$ and
$R'$, the impact levels $L'_\mathrm{nT}$ and $L'_\mathrm{n}$, the field test report, and the
ISO 12999-1 uncertainty that qualifies every field value. Three close
relatives have guides of their own: the reference-curve engine behind every
single number in [Insulation Ratings (ISO 717)](insulation-ratings.md), the
building envelope in [Façade Sound Insulation](facade-insulation.md), and the
quick octave-band route in
[Sound Insulation Survey Method (ISO 10052)](insulation-survey.md). The
laboratory characterisation of an element lives in
[Laboratory Insulation Measurement](insulation-lab.md) and the prediction of
in-situ performance in
[Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md).

## Field airborne insulation (ISO 16283-1)

To rate a wall or floor, measure the energy-average level in the **source**
room ($L_1$) and the **receiving** room ($L_2$) per one-third-octave band
and form the level difference $D = L_1 - L_2$. Two normalisations make it
comparable between rooms. The **standardized level difference** references
the receiving-room reverberation time $T$ to $T_0 = 0.5$ s (so with
$T = 0.5$ s, $D_\mathrm{nT} = D$ exactly), and the **apparent sound reduction
index** normalises by the partition area $S$ and the Sabine absorption area
$A$:

$$
D_\mathrm{nT} = D + 10 \log_{10} \frac{T}{T_0}, \qquad
R' = D + 10 \log_{10} \frac{S}{A}, \qquad A = \frac{0.16\ V}{T}.
$$

Positions are energy-averaged with
$L = 10 \log_{10}\left( \frac{1}{n} \sum_j 10^{L_j/10} \right)$.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_insulation_setup_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_insulation_setup.svg" alt="Field airborne insulation setup: a loudspeaker in the source room, microphones energy-averaged in source and receiving rooms across the common partition" width="92%"></picture>

The prime on $R'$ is a convention, not decoration: primed quantities ($R'$,
$L'_\mathrm{n}$, $L'_\mathrm{nT}$) are measured **in the building** and include every
flanking path, while the unprimed $R$ and $L_\mathrm{n}$ are laboratory properties of
the element alone, measured with flanking suppressed. The full lab-to-field
map lives in [Laboratory Insulation Measurement](insulation-lab.md); the
prediction that bridges the two is
[EN 12354](../design/insulation-prediction.md).

```python
import numpy as np
from phonometry import building

# Energy-average several microphone positions in one room (dB)
print(round(float(building.energy_average_level([60.0, 66.0])), 1))   # 64.0

# Field insulation per band; area S and volume V add R'
l1 = np.full(16, 80.0)                                # source-room levels
l2 = np.full(16, 40.0)                                # receiving-room levels
t2 = np.full(16, 0.5)                                 # receiving-room T (s)
ins = building.airborne_insulation(l1, l2, t2, area=10.0, volume=50.0)
print(round(float(ins.dnt[0]), 1))                   # 40.0  (= D since T = T0)
print(round(float(ins.r_prime[0]), 1))               # 38.0

# The single number comes from the ISO 717-1 engine of the ratings guide
print(building.weighted_rating(ins.dnt).rating)      # 40  DnT,w
print(building.weighted_rating(ins.r_prime).rating)  # 38  R'w
```

Compute `l1`, `l2` and `t2` on the same 16 one-third-octave bands from
100 Hz to 3150 Hz (obtain `t2` from
`room_parameters(ir, fs, limits=(100, 3150), fraction=3).t30`, for example) and
pass them to `airborne_insulation`. Feed that function's `dnt` (or `r_prime`)
spectrum to `weighted_rating`, so every band aligns index-by-index with the
ISO 717-1 reference curve.

### `airborne_insulation()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `l1` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Source-room levels (2D is energy-averaged) |
| `l2` | 1D or 2D array | dB | same band count | Receiving-room levels |
| `t2` | 1D array | s | > 0, one per band | Receiving-room reverberation time |
| `area` | float, optional | m² | > 0, with `volume` | Partition area $S$ (enables $R'$) |
| `volume` | float, optional | m³ | > 0, with `area` | Receiving-room volume $V$ |
| `t0` | float | s | default `0.5` | Reference reverberation time $T_0$ |

`airborne_insulation()` returns an `AirborneInsulationResult` (`d`, `dnt`,
`r_prime` or `None`). The reference-curve engine that turns any of those
spectra into $R_\mathrm{w}$, $R'_\mathrm{w}$ or $D_\mathrm{nT,w}$, its spectrum adaptation terms and its
enlarged-range variants are in
[Insulation Ratings (ISO 717)](insulation-ratings.md).

## Field impact insulation (ISO 16283-2)

Footstep noise is rated the other way round. Instead of how much a floor
*blocks*, impact insulation measures how much a standardized **tapping
machine** on the floor above puts into the room below, so a *higher* number
is *worse*. The energy-average impact sound pressure level $L_\mathrm{i}$ in the
receiving room is normalised like the airborne case, but with a sign flip on
the reverberation term:

$$
L'_\mathrm{nT} = L_\mathrm{i} - 10 \log_{10} \frac{T}{T_0}, \qquad
L'_\mathrm{n} = L_\mathrm{i} + 10 \log_{10} \frac{A}{A_0}, \quad
A_0 = 10\ \text{m}^2,\ A = \frac{0.16\ V}{T}.
$$

The **standardized** impact level $L'_\mathrm{nT}$ ($T_0 = 0.5$ s for dwellings)
needs only the receiving-room $T$, so with $T = 0.5$ s it equals $L_\mathrm{i}$; the
**normalized** level $L'_\mathrm{n}$ (referenced to a 10 m² absorption area) also needs
the receiving-room volume. Note the **minus** sign: more reverberation
*lowers* $L'_\mathrm{nT}$, opposite to the airborne $D_\mathrm{nT}$.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_impact_setup_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_impact_setup.svg" alt="Field impact insulation setup: a standardized tapping machine on the floor of the source room above, microphones energy-averaged in the receiving room below, and the receiving-room reverberation time" width="92%"></picture>

```python
import numpy as np
from phonometry import building

# 16 one-third-octave impact levels Li (100 Hz - 3150 Hz), dB, from the
# ISO 717-2 Annex C worked example, and the receiving-room T per band.
li = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
               73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])
t2 = np.full(16, 0.5)

imp = building.impact_insulation(li, t2, volume=50.0)
print(round(float(imp.l_n_t[0]), 1))          # 62.1  (= Li since T = T0)
print(round(float(imp.l_n[0]), 1))            # 64.1  normalized to A0 = 10 m^2

# Weighted impact rating + spectrum adaptation term CI (ISO 717-2)
res_imp = building.weighted_impact_rating(imp.l_n_t)
print(res_imp.rating, res_imp.ci, res_imp.unfavourable_sum)   # 79 -11 28.0  ->  L'nT,w(CI)=79(-11)

# Octave-band data carry the extra -5 dB reduction (Clause 4.3.2)
octave = np.array([65.3, 64.5, 58.0, 55.8, 43.0])
print(building.weighted_impact_rating(octave).rating)  # 54
```

Feed `impact_insulation`'s `l_n_t` (or `l_n`) straight into
`weighted_impact_rating`; the rating and $C_\mathrm{I}$ reproduce the ISO 717-2 Annex C
values (thirds $L'_\mathrm{nT,w} = 79$, $C_\mathrm{I} = -11$; octave 54, $C_\mathrm{I} = 0$).

### `impact_insulation()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `li` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Energy-average impact SPL (2D is averaged over positions) |
| `t2` | 1D array | s | > 0, one per band | Receiving-room reverberation time |
| `volume` | float, optional | m³ | > 0 | Receiving-room $V$ (enables $L'_\mathrm{n}$) |
| `t0` | float | s | default `0.5` | Reference reverberation time $T_0$ |

`impact_insulation()` returns an `ImpactInsulationResult` (`l_n_t`, `l_n` or
`None`); the ISO 717-2 side of the chain is in
[Insulation Ratings (ISO 717)](insulation-ratings.md).

## Small rooms: the low-frequency procedure (ISO 16283 Clause 8)

Below 100 Hz a bedroom-sized room has too few modes for microphones in its
central zone to stand for the whole volume, so ISO 16283 adds a second
measurement in the room corners. It is **not optional**: Part 1 Clause 8.1,
Part 2 Clause 8.1 and Part 3 Clause 7.3.1 all say the procedure *shall* be
used for the 50 Hz, 63 Hz and 80 Hz one-third-octave bands once the room
volume, calculated to the nearest cubic metre, is smaller than 25 m³. Most
bedrooms and every bathroom are under that line.

With the source running, the corner sound pressure level is the highest of the
measured corners, taken band by band (the three bands may come from three
different corners), and energy-averaged over the source positions; the
reported level then weighs it one third against two thirds of the
default-procedure level:

$$
L_\mathrm{Corner} = 10 \log_{10} \frac{p^2_\mathrm{Corner,1} + \cdots +
p^2_\mathrm{Corner,q}}{q\,p_0^2}, \qquad
L_\mathrm{LF} = 10 \log_{10} \left[ \frac{10^{0.1 L_\mathrm{Corner}} +
2 \cdot 10^{0.1 L}}{3} \right].
$$

Under the same trigger, Clause 10.4 (Clause 8.4 in Part 3) stops the 50 Hz,
63 Hz and 80 Hz one-third-octave reverberation times being measured at all and
puts one **63 Hz octave band** value in their place, used for all three.

```python
import numpy as np
from phonometry import building

# The optional low range measured alongside the core bands.
freqs = np.array([50.0, 63.0, 80.0, 100.0, 125.0, 160.0])
l1 = np.array([88.4, 90.1, 87.6, 85.2, 84.7, 84.1])   # source room, dB
l2 = np.array([50.3, 52.8, 49.1, 45.6, 43.2, 41.8])   # receiving room, dB
t2 = np.array([0.62, 0.58, 0.54, 0.49, 0.47, 0.45])   # receiving room, s

# Four corners of the receiving room, 50/63/80 Hz only: that is the whole
# corner sheet, because no other band is measured there.
corners = np.array([[56.4, 58.1, 54.7],
                    [55.2, 60.3, 53.9],
                    [54.8, 57.6, 56.2],
                    [53.1, 56.9, 55.4]])

print(building.low_frequency_procedure_applies(18.0))   # True   (18 m3)
print(building.low_frequency_procedure_applies(25.0))   # False  ("smaller than")

lf = building.LowFrequencyProcedure(
    volume=18.0,
    corner_levels=corners,
    reverberation_63_octave=0.72,      # Clause 10.4, receiving room
)
res = building.airborne_insulation(l1, l2, t2, frequencies=freqs,
                                   receiver_low_frequency=lf)

chain = res.receiver_low_frequency
print(np.round(chain.l_corner, 1))   # [56.4 60.3 56.2]  highest corner per band
print(np.round(chain.l_default, 1))  # [50.3 52.8 49.1]  default procedure
print(np.round(chain.l_lf, 1))       # [53.4 56.9 52.9]  Formula (13)
print(res.t2)                        # [0.72 0.72 0.72 0.49 0.47 0.45]
print(np.round(res.dnt, 1))          # [36.6 34.8 36.3 39.5 41.2 41.8]
```

`impact_insulation()` and `facade_insulation()` take the same object under a
`low_frequency=` keyword and run the same code.

They warn when you forget, everywhere the procedure is allowed: airborne,
impact, and the loudspeaker façade methods. A road-traffic façade stays silent
because Clause 6 of ISO 16283-3 gives it the default procedure and nothing
else, so there is nothing to have forgotten. A `volume` that rounds below
25 m³ beside a `frequencies` vector naming any of the three bands tells
the function
that Clause 8.1 (Clause 7.3.1 for the loudspeaker façade) is in force, and
with no procedure to run it says so with a
`LowFrequencyWarning` rather than returning a number that is not the ISO 16283
one. The corner geometry, the sampling requirements, the differences between
the three parts and a worked measurement with its rating are all in a guide of
its own:
[Small Rooms: the ISO 16283 Low-Frequency
Procedure](low-frequency-procedure.md).

## ISO 16283 field test report (`.report()`)

The per-band field results write the test report of ISO 16283-1:2014 /
ISO 16283-2:2020 Clause 14 directly, laid out like the recommended results
forms (Annex B / Annex C) and the accredited field reports built on them.
`AirborneInsulationResult.report()` renders the standardized level difference
$D_\mathrm{nT}$ fiche (Figure B.1) or, with `quantity="r_prime"`, the apparent sound
reduction index $R'$ fiche (Figure B.2); `ImpactInsulationResult.report()`
renders the standardized $L'_\mathrm{nT}$ fiche (Figure C.1) or, with
`quantity="l_n"`, the normalized $L'_\mathrm{n}$ fiche (Figure C.2). Each fiche names the field standard
in its basis line, evaluates the ISO 717-1 / ISO 717-2 single-number rating
over the 16 core one-third-octave bands (100-3150 Hz), states the quantity to
one decimal place both in tabular form and as a curve against the shifted
reference curve (Clause 12), boxes the field rating (`DnT,w (C; Ctr)`,
`R'w (C; Ctr)`, `L'nT,w (CI)` or `L'n,w (CI)`) and prints the mandatory
statement that the evaluation is based on field measurement results obtained
by an engineering method.

`verbose=True` swaps the two-column table for the per-band measurement chain
(the energy-average $L_1$ and $L_2$, or $L_\mathrm{i}$, and the reverberation time $T$
beside the reported quantity), the content accredited field reports annex; it
needs a result built by `airborne_insulation()` / `impact_insulation()`, which
retain those inputs on the result (`l1`, `l2`/`li`, `t2`, `t0`). Metadata, the
requirement verdict (airborne passes at or above it, impact at or below it),
`language="es"` and the `phonometry[report]` extra behave exactly as in the
ISO 717 fiche above.

```python
import numpy as np
from phonometry import building, ReportMetadata

# Field airborne: source/receiving levels and T per one-third-octave band
l1 = np.array([92.3, 93.1, 94.0, 94.4, 94.8, 95.0, 95.2, 95.4,
               95.3, 95.1, 94.8, 94.4, 93.9, 93.3, 92.5, 91.6])
l2 = l1 - np.array([38.2, 40.1, 42.6, 45.2, 47.8, 50.1, 52.3, 54.0,
                    55.6, 57.1, 58.2, 59.0, 59.6, 60.1, 60.3, 59.8])
t2 = np.array([0.62, 0.58, 0.55, 0.53, 0.52, 0.50, 0.49, 0.48,
               0.47, 0.46, 0.45, 0.45, 0.44, 0.43, 0.43, 0.42])
field = building.airborne_insulation(l1, l2, t2, area=12.5, volume=30.4)
field.plot()   # per-band DnT (and R') of the measured chain (needs matplotlib)
metadata = ReportMetadata(
    specimen="Separating wall, 240 mm brick with independent lining",
    client="Example client",
    area=12.5, source_volume=32.1, receiving_volume=30.4,
    test_room="Dwelling A living room to dwelling B living room",
    test_date="2026-07-20",
    laboratory="Phonometry Reference Laboratory",
    report_id="PHN-2026-0143",
    requirement=50.0,               # DnT,w >= 50 dB -> PASS/FAIL row
)
field.report("DnTw_field.pdf", metadata=metadata)      # DnT,w (C; Ctr)
field.report("Rpw_field.pdf", quantity="r_prime",
             metadata=metadata)                        # R'w (C; Ctr)
field.report("DnTw_chain.pdf", metadata=metadata,
             verbose=True)                             # f | L1 | L2 | T | DnT

# Field impact: tapping-machine levels in the receiving room
li = np.array([58.0, 60.5, 62.0, 63.5, 65.0, 66.0, 66.5, 66.0,
               65.5, 65.0, 64.0, 62.0, 59.0, 56.0, 53.0, 50.0])
imp = building.impact_insulation(li, t2, volume=30.4)
imp.report("LnTw_field.pdf",
           metadata=ReportMetadata(requirement=58.0))  # L'nT,w (CI)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/field_airborne_insulation_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/field_airborne_insulation.svg" alt="Field airborne measurement chain: the raw level difference D and the standardized DnT across the sixteen one-third-octave bands, with the reverberation correction shaded between them and the resulting DnT,w and R'w ratings annotated" width="80%"></picture>

*The receiving-room reverberation time turns the raw level difference $D$
into the standardized $D_\mathrm{nT}$ band by band; with $T$ above $T_0 = 0.5$ s
across the range, the correction lifts the curve slightly. The rating box
carries both single numbers of this measurement, $D_\mathrm{nT,w}$ and $R'_\mathrm{w}$.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# Field airborne: source/receiving levels and T per one-third-octave band
l1 = np.array([92.3, 93.1, 94.0, 94.4, 94.8, 95.0, 95.2, 95.4,
               95.3, 95.1, 94.8, 94.4, 93.9, 93.3, 92.5, 91.6])
l2 = l1 - np.array([38.2, 40.1, 42.6, 45.2, 47.8, 50.1, 52.3, 54.0,
                    55.6, 57.1, 58.2, 59.0, 59.6, 60.1, 60.3, 59.8])
t2 = np.array([0.62, 0.58, 0.55, 0.53, 0.52, 0.50, 0.49, 0.48,
               0.47, 0.46, 0.45, 0.45, 0.44, 0.43, 0.43, 0.42])
field = building.airborne_insulation(l1, l2, t2, area=12.5, volume=30.4)

# One line — the per-band DnT (and R') of the measured chain:
field.plot()
plt.show()

# By hand, from the result's fields:
bands = [100, 125, 160, 200, 250, 315, 400, 500,
         630, 800, 1000, 1250, 1600, 2000, 2500, 3150]
x = np.arange(len(bands))
w = building.weighted_rating(field.dnt)
fig, ax = plt.subplots()
ax.fill_between(x, field.d, field.dnt, alpha=0.2, label="10 log10(T/T0)")
ax.plot(x, field.d, "--o", label="D (level difference)")
ax.plot(x, field.dnt, "-s", label="DnT (standardized)")
ax.set_xticks(x, [str(b) for b in bands], rotation=45)
ax.set(xlabel="Frequency [Hz]", ylabel="Level difference [dB]",
       title=f"DnT,w = {w.rating} dB  (C={w.c:+d}; Ctr={w.ctr:+d})")
ax.legend()
plt.show()
```

</details>

Rendered examples of both field fiches, regenerated with `make reports`, are
kept in the repository. Click either preview to open the PDF:

[![Field airborne ISO 16283-1 example report: metadata header, one-third-octave DnT table beside the measured-versus-shifted-reference curve, boxed DnT,w (C; Ctr), the engineering-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso16283_airborne_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso16283_airborne_example.pdf)

*Field airborne fiche (`AirborneInsulationResult.report`), DnT,w (C; Ctr).*

[![Field impact ISO 16283-2 example report: the same field layout for the standardized impact level L'nT with the 500 Hz read-off, boxed L'nT,w (CI), the engineering-method statement and a FAIL verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso16283_impact_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso16283_impact_example.pdf)

*Field impact fiche (`ImpactInsulationResult.report`), L'nT,w (CI).*

## Measurement uncertainty (ISO 12999-1)

A rating without an uncertainty is only half a result. ISO 12999-1 does not
re-measure anything; it tabulates the **standard uncertainty** $u$ of every
sound-insulation quantity, derived from inter-laboratory tests, and prescribes
how to expand and combine it. Which standard deviation is $u$ depends on the
**measurement situation** (Clause 5.2):

| Situation | Meaning | Standard uncertainty $u$ |
| :--- | :--- | :--- |
| **A** | laboratory characterisation (ISO 10140) | reproducibility $\sigma_\mathrm{R}$ |
| **B** | same location, different teams | in-situ $\sigma_\mathrm{situ}$ |
| **C** | same location, same operator repeated | repeatability $\sigma_\mathrm{r}$ |

The expanded uncertainty is $U = k\ u$ (Formula 2) with the coverage factor $k$
of Table 8. A two-sided interval $Y = y \pm U$ (Formula 3, $k = 1.96$ at 95 %)
*reports* a value; the **one-sided** factor ($k = 1.65$ at 95 %) *declares
conformity* with a requirement (Formulae 4/5).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_iso12999_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_iso12999.svg" alt="ISO 12999-1 uncertainty flow: standard uncertainty from the tables, reduced by repeated measurements and combined in quadrature, then expanded by the Table 8 coverage factor into a two-sided report or a one-sided conformity decision" width="82%"></picture>

```python
from phonometry import building

# Situation B (same building, different teams) -> the in-situ standard deviation.
print(building.single_number_uncertainty("r_w", "B"))       # 0.9  dB  (Table 3)
u = building.band_uncertainty("airborne", "B")              # per-band u (Table 2)
print(len(u.frequencies), u.uncertainties[10])     # 21 1.1  (the 500 Hz band)
u.plot()   # the per-band u(f) spectrum of Table 2 (needs matplotlib)

# Report R'w = 52 dB with a two-sided 95 % interval (k = 1.96, Table 8):
uv = building.uncertain_value(52.0, "rprime_w", "B")        # aliases resolve to r_w
print(uv.coverage_factor, round(uv.expanded_uncertainty, 1))    # 1.96 1.8
print(round(uv.lower, 1), round(uv.upper, 1))      # 50.2 53.8  ->  52 ± 1.8 dB

# Declaring conformity uses the ONE-sided factor (k = 1.65): does R'w provably
# clear a 50 dB requirement?
uc = building.uncertain_value(52.0, "rprime_w", "B", one_sided=True)
print(building.satisfies_lower_requirement(52.0, uc.expanded_uncertainty, 50.0))   # True
```

Impact quantities offer situations B/C only (Table 4, no 500 Hz band in the 2020
edition), and $\Delta L$ only situation A. Descriptors are case-insensitive with
aliases (`rprime_w`/`dnt_w`→`r_w`, `lprime_n_w`→`ln_w`); combine independent
components in quadrature with `combine_uncertainties`, and reduce by $m$
independent measurements with `reduce_by_independent_measurements` ($u/\sqrt{m}$).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/insulation_uncertainty_demo_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/insulation_uncertainty_demo.svg" alt="A weighted rating reported with its two-sided 95 % expanded uncertainty in situations A, B and C, the reproducibility uncertainty widest and the repeatability uncertainty narrowest" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import building

# The same R'w = 52 dB reported in each situation with its two-sided 95 % U.
situations = ["A", "B", "C"]
vals = [building.uncertain_value(52.0, "r_w", s) for s in situations]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(situations, [v.value for v in vals],
            yerr=[v.expanded_uncertainty for v in vals],
            fmt="o", capsize=8, color="tab:blue")
for s, v in zip(situations, vals):
    ax.annotate(f"±{v.expanded_uncertainty:.1f}", (s, v.upper),
                textcoords="offset points", xytext=(8, 4))
ax.set_ylabel("R'w [dB]"); ax.set_xlabel("Measurement situation")
ax.set_title("R'w = 52 dB with 95 % expanded uncertainty (ISO 12999-1)")
fig.tight_layout()
plt.show()
```

</details>

### `band_uncertainty()` / `single_number_uncertainty()` / `uncertain_value()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `measurand` | str | — | `'airborne'` / `'impact'` / `'impact_reduction'` | Selects Table 2 / 4 / 6 |
| `quantity` | str | — | `'r_w'`, `'ln_w'`, `'delta_lw'` (+ aliases, `+c`/`+ctr` variants) | Single-number descriptor |
| `situation` | str | — | `'A'` / `'B'` / `'C'` | Measurement situation (Clause 5.2) |
| `value` | float | dB | — | Best estimate $y$ to attach $U$ to |
| `coverage` | float | — | default `0.95` | Confidence level (Table 8) |
| `one_sided` | bool | — | default `False` | One-sided factor for conformity checks |
| `upper_limit` | bool | — | default `False` | Select the $\sigma_\mathrm{R95}$ upper limit (airborne, situation A) |

`band_uncertainty()` returns a `BandUncertainty` (`frequencies`,
`uncertainties`, `.to_arrays()`); `single_number_uncertainty()` a float;
`uncertain_value()` an `UncertainValue` (`value`, `standard_uncertainty`,
`coverage_factor`, `expanded_uncertainty`, `.lower`, `.upper`). The read-only
`COVERAGE_FACTORS` mapping exposes Table 8 keyed by `(confidence, one_sided)`.

## Beyond the two-room measurement

Three measurements that used to share this page now have guides of their own,
and a fourth sits alongside them. The building envelope, measured against the
level 2 m in front of it and predicted from its elements, is
[Façade Sound Insulation](facade-insulation.md). When a full engineering
measurement is more than the question deserves, the octave-band control method
is [Sound Insulation Survey Method (ISO 10052)](insulation-survey.md). Every
single number quoted above comes from the reference curves of
[Insulation Ratings (ISO 717)](insulation-ratings.md). And the sound-intensity
route to the same quantities, which reads the transmitted power off the
radiating face instead of the receiving-room level, is
[Sound Insulation by Intensity (ISO 15186)](insulation-intensity.md).

## Quick answers

### What does DnT,w mean?

$D_\mathrm{nT,w}$ is the weighted standardized level difference. Per
one-third-octave band, $D_\mathrm{nT} = D + 10 \log_{10}(T/T_0)$ references the
receiving-room reverberation time $T$ to $T_0 = 0.5$ s, with
$D = L_1 - L_2$ the source-to-receiving level difference (ISO 16283-1).
The ISO 717-1 reference-curve method then collapses the 16 bands from
100 Hz to 3150 Hz into the single number read at 500 Hz.

### What is the difference between R and R' in sound insulation?

The prime marks where the measurement was made: primed quantities ($R'$,
$L'_\mathrm{n}$, $L'_\mathrm{nT}$) are measured in the building and include every flanking
path, while the unprimed $R$ and $L_\mathrm{n}$ are laboratory properties of the
element alone, measured with flanking suppressed. In the field
(ISO 16283-1), $R' = D + 10 \log_{10}(S/A)$ with partition area $S$ and
Sabine absorption area $A = 0.16\ V/T$.

## References

- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  The comprehensive treatment of airborne and impact sound insulation: the
  measurement chains and the statistics of rooms behind the field quantities.
- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  A compact textbook companion for the sound-transmission physics behind
  these measurements.
- International Organization for Standardization. (2014). *Acoustics — Field
  measurement of sound insulation in buildings and of building elements —
  Part 1: Airborne sound insulation* (ISO 16283-1:2014).
  [iso.org catalogue](https://www.iso.org/standard/55997.html).
  The field airborne method this page implements.
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 1: Sound insulation* (ISO 12999-1:2020).
  [iso.org catalogue](https://www.iso.org/standard/73930.html).
  The standard uncertainties per measurement situation and the coverage
  factors that expand them.

## Standards

ISO 16283-1:2014 and ISO 16283-2:2020, *Acoustics — Field measurement of
sound insulation in buildings and of building elements*: the airborne and
impact level differences, their normalisations and the Clause 14 test report;
ISO 12999-1:2020, which tabulates the standard uncertainties per measurement
situation and the coverage factors, and whose precision framework builds on
ISO 5725 (context, not implemented directly). The single-number ratings quoted
here are those of ISO 717-1 and ISO 717-2, and the façade part of the same
ISO 16283 family (ISO 16283-3:2016) has its own page.

**Not covered.** Every field function takes levels the caller has already
corrected for background noise (ISO 16283-1 Clause 9.2). Measuring the
background level — source off, same positions, same averaging — is the
operator's job, and nothing here verifies that the 6 dB floor was met.
(`background_correction` is the **ISO 10140-4** *laboratory* variant and does
not match the field thresholds.) The position counts, distances and
placements of ISO 16283-1/-2 are documented above and checked nowhere: energy
averaging happens once positions are supplied, but nothing verifies how many
were taken or where. The low-frequency procedure of Clause 8 *is* implemented
— the 25 m³ trigger, Formula (12), Formula (13) and the 63 Hz octave
reverberation time of Clause 10.4 — but it is fed corner levels the operator
measured, and the only sampling requirement it enforces is a warning below the
four corners per source position Clause 8.3 asks for.
The other members of the family have their own pages: the façade part
(ISO 16283-3), the survey method (ISO 10052), the rating engines (ISO 717-1/-2)
and the sound-intensity route (**ISO 15186-1**/-2).

## See also

- [Insulation Ratings (ISO 717)](insulation-ratings.md): the reference-curve
  engine behind every weighted single number on this page.
- [Façade Sound Insulation](facade-insulation.md): the third part of
  ISO 16283, measured and predicted.
- [Sound Insulation Survey Method (ISO 10052)](insulation-survey.md): the
  octave-band control method these engineering methods are the reference for.
- [Laboratory Insulation Measurement](insulation-lab.md): the
  ISO 10140 element characterisation these field quantities are compared
  against.
- [Sound Insulation by Intensity (ISO 15186)](insulation-intensity.md): the
  direct-power route to the same field and laboratory indices.
- [Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md):
  the in-situ performance predicted from laboratory element data.
- [Room Acoustics](../rooms/room-acoustics.md): the room parameters and reverberation
  times this guide's insulation chain builds on.
- [Levels](../../signals/levels/levels.md): energy averaging and the level metrics behind
  source/receiving-room levels.
- [Filter Banks](../../signals/filters/filter-banks.md): the IEC 61260 fractional-octave filters
  used for the insulation spectra.
- [Theory](../../reference/theory/rooms-buildings.md): the reference-curve derivation behind the
  weighted single-number ratings.
- API reference: [`building.measurement.insulation`](https://jmrplens.github.io/phonometry/reference/api/building/insulation/) and [`building.measurement.uncertainty`](https://jmrplens.github.io/phonometry/reference/api/building/uncertainty/).
