← [Documentation index](../../README.md)

# Human Vibration: Whole-Body and Hand-Arm Exposure

Vibration transmitted to a person is evaluated with the same measurement chain
whatever its origin: the acceleration is **frequency-weighted** to reflect how
the body responds at each frequency, reduced to a **weighted r.m.s.**
acceleration (with dose measures for shocks and long records), condensed across
axes into a single magnitude (the **vibration total value** for hand-arm
exposure and for comfort; the **highest axis value** for the whole-body health
assessment), and finally normalised to an **8-hour daily exposure** $A(8)$ that
is compared against the action and limit values of the European directive.

The weightings themselves are defined once, in **ISO 8041-1:2017**, as a cascade
of analog filters; **ISO 2631-1** applies them to whole-body vibration,
**ISO 2631-2** to vibration in buildings, **ISO 2631-4** to rail ride comfort,
and **ISO 5349-1/-2** to hand-transmitted vibration. This page covers the whole
chain.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_human_vibration_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_human_vibration.svg" alt="Whole-body vibration measurement chain: a triaxial accelerometer at the seat/body interface of a seated person measures the x, y and z acceleration; each axis is band-limited and frequency-weighted (Wk vertical, Wd horizontal) per ISO 8041-1, reduced to a weighted r.m.s. a_w and VDV per ISO 2631-1, and the highest frequency-weighted axis value max(1.4 a_wx, 1.4 a_wy, a_wz) is normalised to the daily exposure A(8) and assessed against the EAV and ELV of Directive 2002/44/EC" width="94%"></picture>

## 1. Frequency weightings (ISO 8041-1)

Every human-vibration weighting is the product of four analog stages evaluated
at $s = j\,2\pi f$ (ISO 8041-1 Formulae (1)–(5)): a second-order Butterworth
**high-pass** and **low-pass** band limiting, an **acceleration–velocity
transition** carrying the overall gain $K$, and an **upward step**:

$$
H(s) = H_h(s)\,H_l(s)\,H_t(s)\,H_s(s).
$$

A single Table 3 parameter set $(f_1, Q_1, f_2, Q_2, f_3, f_4, Q_4, f_5, Q_5,
f_6, Q_6, K)$ realises all nine weightings (`Wb, Wc, Wd, We, Wf, Wh, Wj, Wk,
Wm`), with a corner set to infinity collapsing its stage to unity. The principal
whole-body weighting is `Wk` (vertical, seat surface); `Wd` is the horizontal
weighting, and `Wh` the hand-arm weighting.

```python
from phonometry import vibration

# The overall weighting response at any frequencies (ISO 8041-1 Formula (5)).
resp = vibration.frequency_weighting("Wk", [1.0, 6.3096, 20.0])
print(resp.magnitude.round(3))      # [0.482 1.054 0.636]  (factors)
print(resp.magnitude_db.round(2))   # [-6.33  0.46 -3.93]  (dB)

resp.plot()   # the weighting response in dB, as in the figure below (needs matplotlib)
```

The factors reproduce the ISO 8041-1 Annex B design-goal tables to their four
significant figures: `Wk` plateaus near −6 dB below 2 Hz, peaks at +0.46 dB near
6.3 Hz and rolls off above.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting.svg" alt="The whole-body vertical weighting Wk in decibels over 0.4 to 100 Hz: a plateau near -6 dB below 2 Hz, a small +0.5 dB peak near 6 Hz and a roll-off to about -21 dB at 100 Hz" width="88%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import vibration

result = vibration.frequency_weighting("Wk", np.geomspace(0.4, 100.0, 240))

# One line:
result.plot()
plt.show()

# By hand, from the result's fields, mirroring what WeightingResponse.plot() draws:
fig, ax = plt.subplots()
ax.semilogx(result.frequencies, result.magnitude_db, color="#1f77b4")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Weighting factor [dB]")
ax.set_title("Whole-body vertical weighting Wk (ISO 8041-1)")
plt.show()
```

</details>

To weight a time signal, `apply_weighting` applies the exact complex response in
the frequency domain (so magnitude *and* phase match the standard), which the
time-domain dose metrics below then consume.

### Which weighting on which axis

The weighting is selected by posture, measurement point and axis, not by
application alone. ISO 2631-1 (Tables 1 and 2, clauses 7.2.3 and 8.2.2) maps
them as follows; the multiplying factors $k$ return in the vibration total
value of section 3:

| Posture, measurement point | Axis | Weighting | $k$ health | $k$ comfort |
| :--- | :--- | :--- | :--- | :--- |
| Seated, seat surface | x, y | `Wd` | 1.4 | 1.0 |
| Seated, seat surface | z | `Wk` | 1.0 | 1.0 |
| Seated, seat surface (rotation) | rx / ry / rz | `We` | — | 0.63 / 0.40 / 0.20 m/rad |
| Seated, backrest | x | `Wc` | (0.8)¹ | 0.8 |
| Seated, backrest | y / z | `Wd` | — | 0.5 / 0.4 |
| Seated, feet | x / y / z | `Wk` | — | 0.25 / 0.25 / 0.4 |
| Standing, floor | x, y | `Wd` | — | 1.0 |
| Standing, floor | z | `Wk` | — | 1.0 |
| Recumbent, under the pelvis | horizontal | `Wd` | — | 1.0 |
| Recumbent, under the pelvis | vertical | `Wk` | — | 1.0 |
| Recumbent, under the head | vertical | `Wj` | — | 1.0 |
| Motion sickness (clause 9) | vertical | `Wf` | — | — |

¹ The health assessment of clause 7 is defined on the seat surface; the
backrest x measurement with `Wc`, $k = 0.8$ is encouraged but excluded
from the Annex B severity assessment (7.2.3). The remaining weightings live in the companion parts:
`Wm` for building occupants on all axes (ISO 2631-2), `Wb` for vertical rail
ride comfort (ISO 2631-4), and `Wh` for hand-transmitted vibration on all
three hand axes with every $k = 1$ (ISO 5349-1).

## 2. Weighted acceleration and dose measures (ISO 2631-1)

The basic evaluation is the **weighted r.m.s. acceleration**. From a
one-third-octave spectrum it is (ISO 2631-1 Eq. (9); the identical construction
gives the hand-arm $a_{hw}$ of ISO 5349-1 Eq. (A.1)):

$$
a_w = \sqrt{\sum_i \left(W_i\,a_i\right)^2},
$$

with $W_i$ the weighting factor at band centre $i$ and $a_i$ the measured band
acceleration. The factors are evaluated at exactly the frequencies you pass;
note that the ISO tables (ISO 8041-1 Annex B, ISO 2631-1 Table 3, ISO 5349-1
Table A.2) tabulate $W_i$ at the *true* one-third-octave centres
$10^{n/10}$ Hz (6.31, 7.943, 15.85, ...), not at the nominal band labels
(6.3, 8, 16, ...); pass true centres when comparing against the tabulated
factors.

```python
import numpy as np
from phonometry import vibration

# A measured vertical seat spectrum (r.m.s. per one-third octave, m/s^2).
freqs = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 31.5, 63.0])
accel = np.array([0.20, 0.45, 0.42, 0.25, 0.12, 0.05, 0.02])

result = vibration.weighted_acceleration(accel, freqs, "Wk")
print(round(result.overall, 3))          # 0.555  m/s^2 (a_w)
print(result.weighted.round(3))          # W_i * a_i per band

result.plot()   # unweighted vs weighted bands, as in the figure below (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighted_acceleration_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/weighted_acceleration.svg" alt="A measured vehicle-seat acceleration spectrum (grey) and its Wk-weighted contribution (blue) over the one-third octaves from 1 to 80 Hz: the weighting attenuates the low and high bands but leaves the 4 to 8 Hz range nearly unchanged, giving a weighted r.m.s. a_w of about 1.03 m/s^2" width="90%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import vibration

freqs = np.array([1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0,
                  12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 63.0, 80.0])
accel = np.array([0.18, 0.24, 0.33, 0.46, 0.52, 0.55, 0.48, 0.39, 0.31, 0.26,
                  0.21, 0.17, 0.13, 0.10, 0.078, 0.060, 0.045, 0.028, 0.020])
result = vibration.weighted_acceleration(accel, freqs, "Wk")

# One line:
result.plot()
plt.show()

# By hand, mirroring what WeightedSpectrum.plot() draws:
pos = np.arange(freqs.size)
fig, ax = plt.subplots()
ax.bar(pos - 0.2, result.band_accelerations, 0.4, color="#bbbbbb",
       label="Unweighted $a_i$")
ax.bar(pos + 0.2, result.weighted, 0.4, color="#1f77b4",
       label="Weighted $W_i a_i$ (Wk)")
ax.set_xticks(pos)
ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"r.m.s. acceleration [m/s$^2$]")
ax.set_title(f"Weighted acceleration ($a_w$ = {result.overall:.3f} m/s²)")
ax.legend()
plt.show()
```

</details>

When the r.m.s. value understates an intermittent or shock-laden exposure,
ISO 2631-1 adds dose measures computed on the weighted time signal: the
**running r.m.s.** and its maximum, the **maximum transient vibration value**
`MTVV` (Eq. (4), a 1 s running r.m.s.); the fourth-power **vibration dose
value** $\text{VDV} = \left(\int a_w^4\,dt\right)^{1/4}$ (Eq. (5)); the **motion
sickness dose value** $\text{MSDV} = \left(\int a_w^2\,dt\right)^{1/2}$; and the
**crest factor** (peak / r.m.s.), whose value above 9 signals that the basic
method is inadequate.

```python
import numpy as np
from phonometry import vibration

fs = 1000.0
raw = np.random.default_rng(0).standard_normal(int(60 * fs))   # 60 s record
a_w = vibration.apply_weighting(raw, fs, "Wk")                        # weighted signal

print(round(vibration.vibration_dose_value(a_w, fs), 3))   # 0.744  VDV [m/s^1.75]
print(round(vibration.mtvv(a_w, fs), 3))                   # 0.265  MTVV [m/s^2]
print(round(vibration.crest_factor(a_w), 2))               # 3.74   crest factor
```

## 3. Vibration total value and daily exposure `A(8)`

Across the three axes the **vibration total value** combines the axis-weighted
r.m.s. accelerations with the posture multiplying factors $k_j$ (ISO 2631-1
Eq. (10); for hand-arm, ISO 5349-1 Eq. (1) with every $k = 1$):

$$
a_v = \sqrt{\sum_j k_j^2\,a_{wj}^2}.
$$

```python
from phonometry import vibration

# Health, seated: k = 1.4 / 1.4 / 1.0 (ISO 2631-1, 7.2.3).
a_v = vibration.vibration_total_value([0.35, 0.28, 0.62], k=[1.4, 1.4, 1.0])
print(round(a_v, 3))     # 0.882  m/s^2
```

**Health or comfort? Two different readings of the same measurement.** The
*health* assessment of clause 7 stays per axis: each axis-weighted r.m.s.
(with $k$ = 1.4 / 1.4 / 1.0 seated) is judged by the *highest* single axis
value against the Annex B health guidance caution zone, a band based mainly on
4 h to 8 h exposures below which health effects are not clearly documented,
inside which caution is indicated and above which risks are likely; Directive
2002/44/EC turns that guidance into the enforceable $A(8)$ action and limit
values used below. The *comfort* assessment of clause 8 instead combines all
axes (and, where relevant, backrest, feet and rotation) into the vibration
total value $a_v$ with its own $k$ set and reads it on the Annex C scale for
public transport, whose deliberately overlapping bands run from "not
uncomfortable" below 0.315 m/s² to "extremely uncomfortable" above 2 m/s²; the
standard defines no comfort limit, since acceptable magnitudes depend on trip
duration and what the passengers are trying to do. For orientation, the median
perception threshold of a `Wk`-weighted vibration is about 0.015 m/s² peak
(Annex C).

The **daily exposure** normalises the exposure magnitude $a$ to a reference
8-hour day ($T_0 = 28\,800$ s). For a single operation $A(8) = a\,\sqrt{T/T_0}$;
several operations combine through their partial exposures
$A_i(8) = a_i\,\sqrt{T_i/T_0}$ as $A(8) = \sqrt{\sum_i A_i(8)^2}$ (ISO 5349-1
Eqs. (2)/(3); ISO 5349-2 Eqs. (1)–(3)). Directive 2002/44/EC fixes which
magnitude $a$ each kind is based on (Annex, points 1): for hand-arm vibration
the vector total $a_{hv}$ (Part A), but for whole-body vibration the *highest*
frequency-weighted axis value $\max(1.4\,a_{wx},\,1.4\,a_{wy},\,a_{wz})$
(Part B), not the vector total $a_v$ above. `wbv_exposure_basis` returns that
dominant-axis value:

```python
from phonometry import vibration

# Directive 2002/44/EC whole-body basis (Annex Part B): the dominant axis.
a = vibration.wbv_exposure_basis(0.35, 0.28, 0.62)
print(round(a, 3))    # 0.62  m/s^2 (max of 0.49, 0.392, 0.62; not a_v = 0.882)
```

`daily_vibration_exposure` builds the partial exposures, combines them and
assesses the result against **Directive 2002/44/EC**: hand-arm action value
$A(8) = 2.5$ and limit value 5 m/s²; whole-body action 0.5 and limit 1.15
m/s² (or a VDV of 9.1 / 21 m/s¹·⁷⁵):

```python
from phonometry import vibration

# ISO 5349-2 Annex E.3: a forestry worker's three chain-saw tasks.
result = vibration.daily_vibration_exposure(
    total_values=[4.6, 6.0, 3.6],                 # a_hv per task, m/s^2
    durations_s=[2 * 3600, 1 * 3600, 2 * 3600],   # exposure time per task
    kind="hav",
    labels=["brush-saw", "felling", "stripping"],
)
print(result.partials.round(2))            # [2.3  2.12 1.8 ]  A_i(8)
print(round(result.a8, 2))                 # 3.61  m/s^2
print(result.assessment.zone)             # 'action'  (2.5 <= A(8) < 5.0)

result.plot()   # partial exposures and A(8) against the EAV/ELV, as in the figure below (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/daily_vibration_exposure_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/daily_vibration_exposure.svg" alt="A bar chart of the three partial hand-arm exposures (about 2.3, 2.1 and 1.8 m/s^2) and the combined A(8) of 3.61 m/s^2, with the Directive 2002/44/EC exposure action value at 2.5 and exposure limit value at 5.0 m/s^2 marked as horizontal lines; the daily exposure sits in the action zone between them" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import vibration

result = vibration.daily_vibration_exposure(
    [4.6, 6.0, 3.6], [2 * 3600, 1 * 3600, 2 * 3600], kind="hav",
    labels=["brush-saw", "felling", "stripping"],
)

# One line:
result.plot()
plt.show()

# By hand, mirroring what DailyVibrationExposure.plot() draws:
labels = [*result.labels, "A(8)"]
values = [*result.partials.tolist(), result.a8]
a = result.assessment
fig, ax = plt.subplots()
ax.bar(range(len(values)), values,
       color=["#bbbbbb"] * result.partials.size + ["#1f77b4"])
ax.axhline(a.action_value, color="#2ca02c", ls="--", label=f"EAV = {a.action_value:g}")
ax.axhline(a.limit_value, color="#d62728", ls="--", label=f"ELV = {a.limit_value:g}")
ax.set_xticks(range(len(values)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel(r"Daily exposure A(8) [m/s$^2$]")
ax.legend()
plt.show()
```

</details>

## 4. Exposure–response guidance

For hand-transmitted vibration, ISO 5349-1 Annex C relates the daily exposure to
the group-mean lifetime $D_y$ (in years) that produces vibration-white-finger in
10 % of an exposed group, $D_y = 31.8\,A(8)^{-1.06}$ (Eq. (C.1)):

```python
from phonometry import vibration

print(round(vibration.hav_vwf_lifetime_years(7.0), 1))   # 4.0 years (Table C.1)
```

The standards deliberately define no safe limit; $A(8)$ and the directive's
action and limit values are the basis for any exposure criterion. For whole-body
exposure, `energy_equivalent_acceleration` gives the ISO 2631-1 Eq. (B.3)
energy-equivalent magnitude across periods of different magnitude and duration.

## 5. The exposure assessment report (`.report()`)

A daily exposure exists to be written down and compared with the law.
`DailyVibrationExposure.report(path)` writes a one-page PDF assessment sheet
laid out like the hand-arm and whole-body exposure calculators of
occupational-hygiene practice (the HSE calculators, the EU Good Practice
Guides): the standard-basis line naming the applied ISO method (ISO 5349-1/-2
for hand-transmitted vibration, ISO 2631-1 for whole-body vibration) and the
directive it is assessed against, a header grid (company via `client`,
operator/worker via `specimen`, workplace via `test_room`, and the
`instrumentation` and `calibration` free-text fields of `ReportMetadata`), the
per-operation exposure analysis (each operation's vibration magnitude, the
hand-arm vector total $a_{hv}$ or the whole-body Directive Part B dominant-axis
value $a_{w,max}$, its daily exposure time $T_i$ and the partial exposure
$A_i(8)$, closed by the daily total and the combined $A(8)$) with the
contribution chart, and the boxed $A(8)$ with its exposure zone. The fiche then
assesses $A(8)$ against Directive 2002/44/EC (Article 3): the exposure action
value (EAV) and exposure limit value (ELV) for the vibration kind (hand-arm
$2.5$ / $5$ m/s²; whole-body $0.5$ / $1.15$ m/s²), each marked exceeded / not
exceeded on the value exactly as displayed, with a PASS/FAIL verdict against the
limit value, and a footer note that the ISO standards define no safe exposure
limit. `verbose=True` adds each operation's share of the daily vibration energy;
`language="es"` renders the Spanish fiche (comma decimals). Rendering needs the
optional `phonometry[report]` extra (reportlab), plus matplotlib for the chart.

The relevant `ReportMetadata` fields for a vibration exposure report are
`client` (the company), `specimen` (the operator/worker), `test_room` (the
workplace), `test_date`, `instrumentation`, `calibration`, and the footer
identity `laboratory`, `operator`, `report_id` and `notes`.

```python
from phonometry import vibration, ReportMetadata

# The ISO 5349-2 Annex E.3 forestry worker's hand-arm day.
res = vibration.daily_vibration_exposure(
    [4.6, 6.0, 3.6],
    [2 * 3600, 1 * 3600, 2 * 3600],
    kind="hav",
    labels=["Brush-saw clearance", "Chain-saw felling", "Chain-saw branch stripping"],
)
res.report(
    "a8.pdf",
    metadata=ReportMetadata(
        client="Example forestry contractor",
        specimen="Forestry worker (right hand)",
        test_room="Managed woodland, plot 12",
        instrumentation="Hand-arm vibration meter (ISO 8041-1), s/n 0042",
        report_id="EXAMPLE-5349",
    ),
)   # A(8) = 3.61 m/s² -> action zone (EAV exceeded, ELV not), verdict PASS
```

The rendered example fiche, regenerated with `make reports`, is kept in the
repository. Click the preview to open the PDF:

[![Daily hand-arm vibration exposure example report: a header with the forestry contractor, the worker, the woodland workplace and the instrumentation, the per-operation exposure-analysis table (brush-saw clearance, chain-saw felling and branch stripping with the vibration total values, daily exposure times and partial exposures), the per-operation contribution chart, the boxed daily exposure A(8) = 3.61 m/s2 in the action zone, and the Directive 2002/44/EC assessment where the 2.5 m/s2 exposure action value is exceeded and the 5 m/s2 exposure limit value is not, ending in a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/human_vibration_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/human_vibration_example.pdf)

*Daily vibration exposure fiche (`DailyVibrationExposure.report`), the
ISO 5349-2 Annex E.3 hand-arm day with the Directive 2002/44/EC assessment.*

## See also

- [Multiple Shock Vibration](multiple-shock-vibration.md): where a record goes once its crest factor and its clause 6.3.3 ratios say the r.m.s. cannot describe it — the ISO 2631-5 spinal-response model, and the seat-pad acquisition this page's whole-body half points at.
- [Sound power from surface vibration](../../devices/emission/vibration-sound-power.md): the other thing a surface acceleration is used for, when the question is what the machine radiates rather than what the operator receives.
- [Calibration](../../signals/metrology/calibration.md): the before-and-after check of clause 6.3.1 and the traceability the fiche's `calibration` field records.
- API reference: [`vibration.human.exposure`](https://jmrplens.github.io/phonometry/reference/api/vibration/exposure/).
- Theory: [Human vibration](../../reference/theory/vibration.md#human-vibration-iso-8041-1-iso-2631-12-iso-5349-12-directive-200244ec): the frequency weightings of ISO 2631-1 and ISO 5349-1, and the running r.m.s., VDV and MTVV definitions.

## References

- Griffin, M. J. (1996). *Handbook of human vibration*. Academic Press.
  ISBN 978-0-12-303041-2.
  [Publisher page](https://shop.elsevier.com/books/handbook-of-human-vibration/griffin/978-0-12-303041-2).
  The standard monograph on whole-body and hand-transmitted vibration:
  the biodynamics, discomfort and health-effect evidence behind the
  weightings, dose measures and exposure-response guidance on this page.
- Mansfield, N. J. (2004). *Human response to vibration*. CRC Press.
  ISBN 978-0-415-28239-0.
  [Publisher page](https://www.routledge.com/Human-Response-to-Vibration/Mansfield/p/book/9780415282390).
  A compact modern textbook on the ISO 2631-1 and ISO 5349 evaluation chains,
  from perception and comfort to the occupational exposure limits.

## Standards

ISO 8041-1:2017 (weighting definitions and tolerances);
ISO 2631-1:1997 (whole-body evaluation); ISO 2631-2:2003 (buildings, `Wm`);
ISO 2631-4:2001 (rail ride comfort, `Wb`); ISO 5349-1:2001 and ISO 5349-2:2001
(hand-transmitted vibration); Directive 2002/44/EC (exposure action and limit
values).
