← [Documentation index](../../README.md)

# Multiple-shock whole-body vibration (ISO 2631-5)

Vibration that contains repeated **mechanical shocks** (off-road vehicles,
high-speed marine craft, earth-moving machinery) loads the lumbar spine far
more than its equivalent r.m.s. level suggests. **ISO 2631-5:2018** predicts
the resulting spinal response and the risk of lumbar injury from the measured
vertical seat acceleration. phonometry implements the normative **Clause 5**
spinal-response model and the **Annex C** health-effect assessment. (The Annex A
/ Annex E finite-element model is distributed by ISO as separate software and is
out of scope here.)

The boundary with the basic method is explicit. ISO 2631-1 declares its r.m.s.
evaluation normally sufficient up to a crest factor of 9, and offers the
running r.m.s./MTVV and the VDV beyond it; ISO 2631-5 is the additional method
for the regime past those, where the record contains repeated shocks. Its
clause 4 then splits that regime in two: *severe* conditions with possible
free fall or loss of contact with the seat and a dominant z-axis (military
off-road vehicles, high-speed marine craft) use the clause 5 model implemented
here, while *less severe* conditions in which the occupant stays seated
throughout (tractors, forestry and earth-moving machinery on rough ground)
belong to the Annex A finite-element model. In case of doubt the delineation
is quantitative: when the band-limited vertical peak acceleration exceeds
9.81 m/s² (1 g, the free-fall threshold), clause 5 and Annex C apply.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_iso2631_5_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_iso2631_5.svg" alt="Flow from the vertical seat acceleration az(t), band-limited per ISO 2631-1, through the spinal response Az(t) from the seat-to-spine transfer function (one zero, six poles), the acceleration dose Dz = 1.07 times the sixth root of the sum of the positive response peaks to the sixth power and the daily dose Dzd, the compressive stress Sd = mz times Dzd, the age-cumulated stress variable R, and finally the Weibull probability of lumbar injury" width="86%"></picture>

## 1. Spinal response (clause 5.2)

A seat-to-spine transfer function $H(f)$ (one complex zero and six complex
poles, unity transmissibility at 0 Hz and a resonance near 5 Hz) maps the
measured seat acceleration to the vertical spinal response $A_\mathrm{z}(t)$
(Formula 1/2):

$$
A_\mathrm{z}(t) = \mathcal{F}^{-1}\!\left[H(f)\,\mathcal{F}[a_\mathrm{z}(t)]\right].
$$

The input record must be **conditioned (DC-removed)**: because $H$ is unity at
0 Hz by design, a DC offset (e.g. the 1 g gravity component of a DC-coupled
accelerometer) passes straight into $A_\mathrm{z}(t)$ and corrupts the positive
response peaks of the dose. Subtract the mean (or high-pass) before processing.
The rest of the conditioning chain is the caller's too: `spinal_response`
assumes a record that has already been sign-checked, split at the
loss-of-contact boundaries, offset-corrected, tapered and band-limited.

```python
from phonometry import vibration

# The transmissibility peaks near the ~5 Hz spinal resonance.
print(round(abs(vibration.seat_to_spine_transfer([2.0])[0]), 2))  # 1.06
print(round(abs(vibration.seat_to_spine_transfer([5.0])[0]), 2))  # 1.54
```

## 2. Acceleration dose (clause 5.3)

The **acceleration dose** combines the positive response peaks $A_{\mathrm{z},i}$
(each the maximum between two consecutive zero crossings) with a sixth-power
law, so the largest shocks dominate (Formula 3):

$$
D_\mathrm{z} = 1.07\left(\sum_i A_{\mathrm{z},i}^{\,6}\right)^{1/6}.
$$

A daily dose scales the measured dose to the daily exposure time $t_\mathrm{d}$ over
the measurement time $t_\mathrm{m}$ (Formula 4): $D_\mathrm{zd} = D_\mathrm{z}\,(t_\mathrm{d}/t_\mathrm{m})^{1/6}$.

```python
from phonometry import vibration

# Five 40 m/s2 response peaks in a day (the Annex C worked example).
print(round(vibration.dose_from_peaks([40.0] * 5), 2))  # 55.97  m/s2
```

## 3. Injury risk (Annex C)

The daily dose becomes a daily **compressive stress** $S_\mathrm{d} = m_\mathrm{z}\,D_\mathrm{zd}$
(Formula C.1), where $m_\mathrm{z}$ (0.029 MPa per m/s² for an 82 kg male, 0.025 for a
64 kg female) converts acceleration to vertebral stress. The stress accumulates
over the exposure years against the ageing spine's reducing ultimate strength
$S_\mathrm{u} = 6.75 - S_{\mathrm{age}}\,(b+i)$ (Formulae C.3/C.4):

$$
R = \left[\sum_{i=0}^{n-1}
\left(\frac{S_\mathrm{d}\,N^{1/6}}{S_{\mathrm{u},i} - S_{\mathrm{stat}}}\right)^{6}\right]^{1/6},
$$

and a Weibull model gives the probability of lumbar injury (Formula C.5):

$$
\Pi(R) = 1 - \exp\!\left[-\left(\frac{R}{\alpha}\right)^{\beta}\right].
$$

```python
from phonometry import vibration

# Annex C worked example: 5 x 40 m/s2/day, 82 kg male, age 20 for 20 years,
# 120 days/year.
dz = vibration.dose_from_peaks([40.0] * 5)
sd = vibration.compression_dose(dz)                     # 1.62 MPa
r = vibration.injury_risk(sd, start_age=20, years=20, days_per_year=120)
print(round(r, 2))                               # 1.22
print(round(100 * vibration.injury_probability(r)))     # 37  % risk of injury
```

From a measured time history the whole chain is one call:

```python
import numpy as np
from phonometry import vibration

# A synthetic 10 s seat record at 256 Hz with five 60 m/s2 shocks (stand-in
# for a measured az(t)).
fs = 256.0
az = np.zeros(2560)
az[256::512] = 60.0
result = vibration.multiple_shock_assessment(
    az, fs, start_age=20, years=20, days_per_year=120, sex="male",
)
print(round(result.acceleration_dose, 2))  # 20.94  m/s2
print(round(result.risk, 2))               # 0.46
print(round(result.probability, 2))        # 0.03

result.plot()   # the injury-probability curve with this assessment's R marked (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/multiple_shock_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/multiple_shock.svg" alt="Two panels. Left: the seat-to-spine transmissibility rising to about 1.6 near a 5 Hz resonance then rolling off to near zero by 80 Hz. Right: the probability of lumbar injury as a Weibull function of the stress variable R for male and female, with the 10, 50 and 90 percent risk levels marked and the Annex C male worked example at R = 1.22, about 37 percent" width="96%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import vibration

# Left: the seat-to-spine transfer function magnitude.
f = np.logspace(np.log10(0.5), np.log10(80.0), 400)
plt.plot(f, np.abs(vibration.seat_to_spine_transfer(f)))
plt.xscale("log"); plt.show()

# Right: the injury-probability curve with this assessment's R marked
# (az/fs as in the snippet above).
fs = 256.0
az = np.zeros(2560)
az[256::512] = 60.0
vibration.multiple_shock_assessment(
    az, fs, start_age=20, years=20, days_per_year=120,
).plot()
plt.show()
```

</details>

The `MultipleShockResult` carries the dose $D_\mathrm{z}$, the daily dose $D_\mathrm{zd}$, the
compressive stress $S_\mathrm{d}$, the stress variable $R$, the injury probability and
the response peaks, and its `.plot()` draws the injury-probability curve with
the 10/50/90 % risk thresholds of Table C.2. The model is vertical-axis only:
clause 4 neglects the horizontal contributions to spinal compression by design,
and the horizontal spinal model of the withdrawn 2004 edition is not reinstated,
so horizontal whole-body exposure is assessed instead with the r.m.s.,
running-r.m.s./MTVV and VDV metrics of [Human Vibration](human-vibration.md)
(ISO 2631-1).

## 4. The health-risk report (`.report()`)

The dose chain exists to be written down and read against the Annex C guidance.
`MultipleShockResult.report(path)` writes a one-page PDF health-risk assessment
sheet: the standard-basis line (Clause 5 spinal response and the Annex C risk
model), an optional metadata header (client, subject via `specimen`,
workplace/vehicle via `test_room`, and the `instrumentation` and `calibration`
free-text fields of `ReportMetadata`), the exposure-scenario grid (subject sex,
the age $b$ at which the exposure started, the number of exposure years $n$,
the number of exposure days per year $N$ and the number of counted response
shocks), and the dose-and-stress analysis table with the acceleration dose
$D_\mathrm{z}$ (Formula 3), the daily dose $D_\mathrm{zd}$ (Formula 4), the daily compressive
stress $S_\mathrm{d}$ (Formula C.1), the cumulative stress variable $R$ (Formula C.3)
and the probability of lumbar injury $\Pi$ (Formula C.5).

Because ISO 2631-5:2018 defines no exposure limit, the fiche carries a risk-band
zone row rather than a PASS/FAIL verdict: the boxed $R$ and $\Pi$ name the
Annex C risk classification, and a classification table places $R$ among the
Table C.2 stress variables for $10$ / $50$ / $90$ % risk of injury (low /
moderate / high / very high probability of an adverse health effect), with the
injury-probability chart above it. `language="es"` renders the Spanish fiche (comma decimals). Rendering
needs the optional `phonometry[report]` extra (reportlab), plus matplotlib for
the chart.

The relevant `ReportMetadata` fields for a multiple-shock report are `client`,
`specimen` (the subject), `test_room` (the workplace or vehicle), `test_date`,
`instrumentation`, `calibration`, and the footer identity `laboratory`,
`operator`, `report_id` and `notes`.

```python
import numpy as np
from phonometry import vibration, ReportMetadata

# The Annex C worked example, rebuilt from its five 40 m/s2 response peaks.
peaks = np.array([40.0] * 5)
dz = vibration.dose_from_peaks(peaks)
sd = vibration.compression_dose(dz)
r = vibration.injury_risk(sd, start_age=20, years=20, days_per_year=120)
result = vibration.MultipleShockResult(
    sex="male",
    acceleration_dose=dz,
    daily_dose=dz,
    compression_dose=sd,
    risk=r,
    probability=float(vibration.injury_probability(r)),
    start_age=20.0,
    years=20,
    days_per_year=120.0,
    peaks=peaks,
    risk_thresholds=vibration.RISK_THRESHOLDS_MALE,
)
result.report(
    "multiple_shock.pdf",
    metadata=ReportMetadata(
        client="Example transport operator",
        specimen="82 kg male operator (seated)",
        test_room="Off-road vehicle, driver's seat",
        report_id="EXAMPLE-2631-5",
    ),
)   # R = 1.22, Pi = 37 % -> moderate probability of an adverse health effect
```

The rendered example fiche, regenerated with `make reports`, is kept in the
repository. Click the preview to open the PDF:

[![Whole-body multiple-shock health-risk example report: a header with the transport operator, the subject and the off-road vehicle, the exposure-scenario grid (82 kg male, age 20, 20 years, 120 days per year, five counted shocks), the dose-and-stress analysis table with the acceleration dose Dz = 55.97 m/s2, the daily compressive stress Sd = 1.62 MPa, the cumulative stress variable R = 1.22 and the probability of lumbar injury 37 percent, the injury-probability curve with the Table C.2 risk levels marked, the boxed R = 1.22 and 37 percent classed as a moderate probability of an adverse health effect, and the Annex C classification table with the moderate band as the assessment's band](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso2631_5_multiple_shock_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso2631_5_multiple_shock_example.pdf)

*Whole-body multiple-shock health-risk fiche (`MultipleShockResult.report`), the
ISO 2631-5:2018 Annex C worked example with the Table C.2 risk classification.*

## See also

- [Human Vibration](human-vibration.md): the ISO 2631-1 chain this page hands back to for the horizontal axes, and the crest factor and clause 6.3.3 ratios that decide whether a record belongs here at all.
- API reference: [`vibration.human.multiple_shock`](https://jmrplens.github.io/phonometry/reference/api/vibration/multiple-shock/).
- Theory: [Multiple shocks (ISO 2631-5)](../../reference/theory/vibration.md#multiple-shocks-iso-2631-5): the ISO 2631-5 response model, the acceleration dose and the Weibull risk law.

## References

- Griffin, M. J. (1996). *Handbook of human vibration*. Academic Press.
  ISBN 978-0-12-303041-2.
  [Publisher page](https://shop.elsevier.com/books/handbook-of-human-vibration/griffin/978-0-12-303041-2).
  Background on whole-body shock exposure, spinal biodynamics and the
  lumbar health effects that the ISO 2631-5 dose model quantifies.

## Standards

ISO 2631-5:2018, *Mechanical vibration and shock — Evaluation of
human exposure to whole-body vibration — Part 5: Method for evaluation of
vibration containing multiple shocks*: the seat-to-spine transfer function
(clause 5.2, Formula 1), the acceleration and daily dose (clause 5.3,
Formulae 3-5) and the Annex C assessment of adverse health effects (Formulae
C.1, C.3-C.5, Tables C.1/C.2).
