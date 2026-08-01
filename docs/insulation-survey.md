← [Documentation index](README.md)

# Sound Insulation Survey Method (ISO 10052)

Not every question deserves a full engineering measurement. When a dispute
needs a number this afternoon, ISO 10052 trades accuracy for speed: octave
bands, a hand-held meter and one correction index instead of per-band
reverberation times. This guide covers the survey (control) method: the
reverberation index and its room-class estimate, the airborne, impact,
façade and service-equipment quantities, and the survey fiches. The
engineering-grade methods live in
[Field Insulation Measurement (ISO 16283)](insulation-field.md); the
single-number engines behind the survey ratings in
[Insulation Ratings (ISO 717)](insulation-ratings.md).

## Octave bands and the reverberation index

The [engineering methods](insulation-field.md) buy accuracy with effort: swept microphones,
per-band reverberation times, careful background correction. For a quick check
in a dwelling, ISO 10052 defines a **survey (control) method**: octave bands, a
hand-held meter, and a single quantity, the **reverberation index**
$k = 10\log_{10}(T/T_0)$ ($T_0 = 0.5\ \text{s}$), to carry the receiving-room
correction. Every survey quantity is then just an addition of $k$: the
standardized level difference $D_{nT} = D + k$, the normalized
$D_n = D + k + 10\log_{10}(A_0 T_0/(0.16\,V))$, the apparent
$R' = D + k + 10\log_{10}(S T_0/(0.16\,V))$ (using $V/7.5$ for $S$ where
that is larger), and, for impacts and façades, $L'_{nT} = L_i - k$ and
$D_{2m,nT} = D_{2m} + k$. The clause references follow ISO 10052:2021; the formulas
and the reverberation-index table are identical in the harmonized
EN ISO 10052:2004+A1:2010.

The reverberation index is either **measured** (feed the reverberation time to
`reverberation_index(T)`) or, in a control survey, **estimated** from the room
type and volume with `estimate_reverberation_index(V, room)` (Table 4:
furnished `"kitchen"` / `"bathroom"` / `"furnished"`, or the unfurnished
construction classes `"a"`–`"h"` and the mixed `"a+e"`…`"d+h"`). A fourth
quantity unique to this method is **service-equipment noise** $L_{XY}$: the
energy average of three A- or C-weighted positions.

```python
import numpy as np
from phonometry import building

# Octave-band levels (125-2000 Hz) and the measured receiving-room T.
l1 = np.array([88.0, 90.0, 92.0, 92.0, 90.0])
l2 = np.array([55.0, 51.0, 47.0, 41.0, 35.0])
k = building.reverberation_index([0.70, 0.60, 0.50, 0.45, 0.40])   # k = 10 lg(T/0.5)
res = building.survey_airborne_insulation(l1, l2, k, volume=50.0, area=12.0)
print(np.round(res.d_nt, 1))          # [34.5 39.8 45.  50.5 54. ]  DnT = D + k
print(res.rating.rating, res.rating.c)        # 49 -1  ->  DnT,w (C)
print(res.r_prime_rating.rating)               # 48  ->  R'w

# No reverberation time measured? Estimate k from Table 4 (heavy walls, hard
# floor, 35-60 m3 -> class "g").
k_est = building.estimate_reverberation_index(50.0, "g")
print(k_est)                                   # [4.5 5.  5.5 5.5 5.5]

# Service-equipment noise: energy average of three A-weighted positions.
se = building.survey_service_equipment_level([35.0, 30.0, 32.0], 3.0, volume=50.0)
print(round(float(se.l_xy), 1), round(float(se.l_xy_nt), 1))   # 32.8 29.8

res.plot()   # DnT vs shifted ISO 717-1 reference (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/survey_insulation_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/survey_insulation.svg" alt="Survey-method airborne insulation: the raw level difference D and the standardized DnT across the five octave bands, with the reverberation-index correction k shaded between them" width="80%"></picture>

*The reverberation index $k = 10\log_{10}(T/T_0)$ shifts the raw level difference
$D$ into the standardized $D_{nT}$: up where the room is live ($T > T_0$),
down where
it is dead. The automatic rating is formed only for exactly 5 octave (or 16
one-third-octave) values.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# Octave-band levels (125-2000 Hz) and the measured receiving-room T.
bands = [125, 250, 500, 1000, 2000]
l1 = np.array([88.0, 90.0, 92.0, 92.0, 90.0])
l2 = np.array([55.0, 51.0, 47.0, 41.0, 35.0])
k = building.reverberation_index([0.70, 0.60, 0.50, 0.45, 0.40])   # k = 10 lg(T/0.5)
res = building.survey_airborne_insulation(l1, l2, k, volume=50.0)

x = np.arange(len(bands))
fig, ax = plt.subplots()
ax.fill_between(x, res.d, res.d_nt, alpha=0.2, label="k = 10 log10(T/T0)")
ax.plot(x, res.d, "--o", label="D (level difference)")
ax.plot(x, res.d_nt, "-s", label="DnT (standardized)")
ax.set_xticks(x, [str(b) for b in bands])
ax.set(xlabel="Frequency [Hz]", ylabel="Level difference [dB]",
       title=f"ISO 10052 survey method: DnT,w = {res.rating.rating} dB")
ax.legend()
plt.show()
```

</details>

### `survey_airborne_insulation()` and friends: parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `l1` / `l2` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Source / receiving (or outdoor `l1_2m`) levels |
| `li` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Impact levels (energy-averaged over positions) |
| `reverberation_index` | scalar or 1D array | dB | one per band | $k$ from `reverberation_index` or `estimate_reverberation_index`; `survey_service_equipment_level()` also accepts a scalar $k$ |
| `volume` | float | m³ | > 0 | Receiving-room $V$ (for $D_n$ / $L'_n$ / $R'$ / normalized) |
| `area` | float | m² | > 0 | Common-partition $S$ (airborne $R'$; $V/7.5$ rule applied) |
| `measurements` | array | dB | exactly 3 | Service-equipment positions (`survey_service_equipment_level`) |
| `room` | str | — | `"kitchen"`/`"bathroom"`/`"furnished"`/`"a"`–`"h"`/`"a+e"`… | `estimate_reverberation_index` room class (Table 4) |

`survey_airborne_insulation()` returns a `SurveyAirborneResult` (`d`, `d_nt`,
`d_n`, `r_prime`, `rating`, `r_prime_rating`); `survey_impact_insulation()` a
`SurveyImpactResult` (`l_i`, `l_nt`, `l_n`, `rating`);
`survey_facade_insulation()` a `SurveyFacadeResult`;
`survey_service_equipment_level()` a `SurveyServiceEquipmentResult` (`l_xy`,
`l_xy_nt`, `l_xy_n`).

## ISO 10052 survey reports (`.report()`)

The airborne, impact and façade survey results each carry a `.report(path)` that
writes the one-page ISO 10052 survey (control) method field report: the
standard-basis line naming ISO 10052 (octave bands), an optional metadata
header, the octave-band table beside the measured-versus-shifted-reference
curve, the boxed field rating ($D_{nT,w}$/$R'_w$, $L'_{nT,w}$ or
$D_{2m,nT,w}$), the
survey-method statement, an optional requirement verdict (level differences pass
at or above it, the impact level at or below it) and a footer. The airborne
result reports `quantity="dnt"` (default) or `"r_prime"`; `verbose=True`,
`metadata` and `language="es"` behave as in the fiches above.

```python
import numpy as np

from phonometry import building, ReportMetadata

res.report("DnTw_survey.pdf",
           metadata=ReportMetadata(requirement=40.0))   # DnT,w (C; Ctr)

# Impact and façade surveys reuse the same k; li is the tapping-machine level
# in the room below, l1_2m the level 2 m in front of the façade.
li = np.array([66.0, 64.0, 62.0, 60.0, 55.0])
impact = building.survey_impact_insulation(li, k, volume=50.0)
impact.plot()   # L'nT vs the shifted ISO 717-2 reference (needs matplotlib)
impact.report("LnTw_survey.pdf")                         # L'nT,w (CI)

l1_2m = np.array([76.0, 78.0, 79.0, 79.0, 77.0])
facade = building.survey_facade_insulation(l1_2m, l2, k, volume=40.0)
facade.report("D2mnTw_survey.pdf")                       # D2m,nT,w (C; Ctr)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/survey_impact_insulation_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/survey_impact_insulation.svg" alt="Survey-method impact insulation: the measured octave-band impact level Li and the standardized L'nT with the reverberation-index correction shaded between them, the L'nT,w rating annotated and a note that a live room lowers the standardized impact level" width="80%"></picture>

*The impact survey applies the reverberation index with the opposite sign,
$L'_{nT} = L_i - k$: a live receiving room ($T > T_0$) **lowers** the standardized
impact level. As in the airborne case, the automatic rating appears for
exactly 5 octave (or 16 one-third-octave) values.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# Octave-band tapping-machine levels below the floor, and the measured T.
bands = [125, 250, 500, 1000, 2000]
li = np.array([66.0, 64.0, 62.0, 60.0, 55.0])
k = building.reverberation_index([0.70, 0.60, 0.50, 0.45, 0.40])
impact = building.survey_impact_insulation(li, k, volume=50.0)

# One line — L'nT vs the shifted ISO 717-2 reference:
impact.plot()
plt.show()

# By hand, showing the sign flip of the correction:
x = np.arange(len(bands))
fig, ax = plt.subplots()
ax.fill_between(x, impact.l_i, impact.l_nt, alpha=0.2, label="-k = -10 log10(T/T0)")
ax.plot(x, impact.l_i, "--o", label="Li (impact level)")
ax.plot(x, impact.l_nt, "-s", label="L'nT (standardized)")
ax.set_xticks(x, [str(b) for b in bands])
ax.set(xlabel="Frequency [Hz]", ylabel="Impact sound pressure level [dB]",
       title=f"ISO 10052 survey method: L'nT,w = {impact.rating.rating} dB")
ax.legend()
plt.show()
```

</details>

Rendered examples of the survey fiches, regenerated with `make reports`, are
kept in the repository. Click a preview to open the PDF:

[![Survey airborne ISO 10052 example report: metadata header, octave-band DnT table beside the measured-versus-shifted-reference curve, boxed DnT,w (C; Ctr), the survey-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10052_airborne_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10052_airborne_example.pdf)

*Survey airborne fiche (`SurveyAirborneResult.report`), DnT,w (C; Ctr).*

[![Survey impact ISO 10052 example report: the same survey layout for the standardized impact level L'nT with the 500 Hz read-off, boxed L'nT,w (CI), the survey-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10052_impact_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10052_impact_example.pdf)

*Survey impact fiche (`SurveyImpactResult.report`), L'nT,w (CI).*

[![Survey facade ISO 10052 example report: metadata header, octave-band D2m,nT table beside the measured-versus-shifted-reference curve, boxed D2m,nT,w (C; Ctr), the survey-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10052_facade_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10052_facade_example.pdf)

*Survey façade fiche (`SurveyFacadeResult.report`), D2m,nT,w (C; Ctr).*

## References

- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  The building-acoustics context of the field quantities the survey method
  approximates.
- International Organization for Standardization. (2021). *Acoustics —
  Field measurements of airborne and impact sound insulation and of service
  equipment sound — Survey method* (ISO 10052:2021).
  [iso.org catalogue](https://www.iso.org/standard/76560.html).
  The survey method this page implements: the reverberation-index
  correction, the four survey quantities and their reports.

## Standards

ISO 10052:2021 (harmonized as EN ISO 10052:2004+A1:2010), which defines the
field survey method: the reverberation-index correction, the
standardized/normalized airborne, impact and façade quantities, and
service-equipment noise. The single-number ratings reuse ISO 717-1 and
ISO 717-2.

## See also

- [Field Insulation Measurement (ISO 16283)](insulation-field.md): the
  engineering-grade airborne, impact and façade measurements this method
  approximates, and their ISO 12999-1 uncertainty.
- [Insulation Ratings (ISO 717)](insulation-ratings.md): the reference-curve
  engines behind the survey ratings.
- [Room Acoustics](room-acoustics.md): the measured reverberation times
  behind the reverberation index.
- [Levels](levels.md): the energy averaging behind the octave-band levels.
- API reference: [`building.survey_insulation`](https://jmrplens.github.io/phonometry/reference/api/building/survey-insulation/).
