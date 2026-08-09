← [Documentation index](../../README.md)

# Laboratory Insulation Measurement

To rate a building element on its own (a wall type, a floating floor, a
window) you take it to a qualified laboratory, where suppressed flanking
makes the direct transmission the whole story. This page covers the
ISO 10140 laboratory chain: the sound reduction index $R$, the normalized
impact level $L_n$, the background-noise correction and the accredited test
fiches. Three sibling laboratory methods have guides of their own: the
sound-intensity route in
[Sound Insulation by Intensity (ISO 15186)](insulation-intensity.md), the
floor-covering improvement in
[Floor-Covering Impact Improvement (ISO 16251-1)](../design/impact-improvement.md) and
the junction measurement in
[Laboratory Flanking Transmission (ISO 10848)](flanking-lab.md). Field
measurement lives in
[Field Insulation Measurement (ISO 16283)](insulation-field.md), the
single-number ratings in
[Insulation Ratings (ISO 717)](insulation-ratings.md), and the
prediction that consumes these laboratory ratings in
[Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md).

## Laboratory measurement (ISO 10140)

An [ISO 16283 field measurement](insulation-field.md) yields the primed
quantities ($R'$, $L'_n$): the number a real building achieves, flanking
transmission and all. To rate an
element on its own (a wall type, a floating floor, a window), you take it to a
qualified **laboratory** (ISO 10140), where suppressed flanking makes the
*direct* transmission the whole story. The formulas lose their primes: the
**sound reduction index** $R$ (not $R'$) and the **normalized impact level**
$L_n$ (not $L'_n$), with the receiving room's absorption area $A = 0.16\ V/T$
now a known property of the facility:

$$
R = L_1 - L_2 + 10 \log_{10}\frac{S}{A}, \qquad
L_n = L_i + 10 \log_{10}\frac{A}{A_0}, \quad A_0 = 10\ \text{m}^2.
$$

The facility itself is what suppresses the flanking: two structurally
decoupled reverberation rooms of at least 50 m³ each, with the element under
test mounted in a test opening of about 10 m² between them.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_insulation_lab_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_insulation_lab.svg" alt="Plan view of an ISO 10140 laboratory transmission suite: structurally decoupled source and receiving reverberation rooms of about 59 and 51 cubic metres, the test element mounted in the 10 square metre test opening between them, a corner loudspeaker in the source room and a continuously moving microphone with a sweep radius of at least 1 m in each room" width="92%"></picture>

| | Field (ISO 16283) | Laboratory (ISO 10140) |
| :--- | :--- | :--- |
| Airborne element index | $R'$ apparent (with flanking) | $R$ direct (flanking suppressed) |
| Airborne room pair | $D_{nT}$, $D_n$ (no prime: room quantities) | — |
| Impact | $L'_n$, $L'_{nT}$ apparent | $L_n$ direct |
| Single number | $R'_w$, $D_{nT,w}$, $L'_{n,w}$, $L'_{nT,w}$ | $R_w$, $L_{n,w}$ |
| Absorption area | measured in the room | property of the facility |

The apostrophe is the flanking marker of building acoustics, and it travels
with the quantity into its single number: $R_w$ rates a laboratory spectrum,
$R'_w$ a field one. The standardized and normalized level differences
$D_{nT}$ and $D_n$ carry no prime because they describe the room pair rather
than an element, so there is no flanking-free counterpart to mark. In a
well-built construction $R'_w$ lands a few dB below the laboratory $R_w$ of
the same partition; a much larger gap says flanking dominates, and the
[EN 12354 model](../design/insulation-prediction.md) tells you which path carries it.

The single-number ratings reuse the very same ISO 717-1/2 engines
(`weighted_rating`, `weighted_impact_rating`): an $R$ spectrum rates to $R_w$
exactly as an $R'$ spectrum rated to $R'_w$. Before forming the index the
receiving-room levels must be **corrected for background noise** (Clause 4.3):
the energy subtraction $10 \log_{10}(10^{L_{sb}/10} - 10^{L_b/10})$ applies for a
6–15 dB signal-to-background margin, a fixed 1.3 dB correction (the *limit of
measurement*) at or below 6 dB, and no correction at or above 15 dB.

```python
import numpy as np
from phonometry import building

# Source/receiving levels and receiving-room T over the 16 one-third-octave
# bands; S is the free test-opening area, V the receiving-room volume.
l1 = np.full(16, 80.0)
l2 = np.full(16, 40.0)
t2 = np.full(16, 0.5)
lab = building.lab_airborne_insulation(l1, l2, t2, area=10.0, volume=50.0)
print(round(float(lab.r[0]), 1))              # 38.0  R = L1 - L2 + 10 lg(S/A)
print(round(float(lab.absorption[0]), 1))     # 16.0  A = 0.16 V / T (m^2)
print(lab.rating.rating, lab.rating.c, lab.rating.ctr)   # 38 0 0  ->  Rw(C;Ctr)

# Impact: the tapping-machine level Li normalized to A0 = 10 m^2 gives Ln
li = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
               73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])
imp = building.lab_impact_insulation(li, t2, volume=50.0)
print(round(float(imp.l_n[0]), 1))            # 64.1  Ln = Li + 10 lg(A/A0)
print(imp.rating.rating, imp.rating.ci)       # 81 -11  ->  Ln,w(CI)

# Background correction: margins 6 / 1 / 20 dB -> capped / capped / unchanged
corrected = building.background_correction([30.0, 33.0, 50.0], [24.0, 32.0, 30.0])
print(np.round(corrected, 1))                 # [28.7 31.7 50.0]  (1.3 dB cap twice)

lab.rating.plot()   # measured R vs shifted ISO 717-1 reference (needs matplotlib)
```

A margin at or below 6 dB emits a `LabInsulationWarning` and flags the band as
the limit of measurement; catch it with `warnings.simplefilter("error",
LabInsulationWarning)`. The automatic rating is formed only when exactly 16
one-third-octave or 5 octave values are supplied (`rating` is `None` otherwise).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/lab_insulation_result_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/lab_insulation_result.svg" alt="The two ISO 10140 laboratory quantities side by side: the measured sound reduction index R against the shifted ISO 717-1 reference on the left, and the normalized impact sound pressure level Ln against the shifted ISO 717-2 reference on the right, each panel annotated with its single-number rating" width="92%"></picture>

*The two laboratory quantities of ISO 10140 with their ISO 717 ratings: the
airborne $R$ is rated where the reference sits **above** the measurement,
the impact $L_n$ where the measurement sits **above** the reference (a
higher impact level is worse). `lab.plot()` and `imp.plot()` draw either
panel on its own.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# The ISO 717-1 Annex C wall in an ISO 10140 suite (S = 10 m2, V = 50 m3,
# T = 0.8 s) and the ISO 717-2 Annex C floor under the tapping machine.
r = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
              28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
l1 = np.full(16, 90.0)
t2 = np.full(16, 0.8)
lab = building.lab_airborne_insulation(l1, l1 - r, t2, area=10.0, volume=50.0)
li = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
               73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])
imp = building.lab_impact_insulation(li, t2, volume=50.0)

# One line each — R (or Ln) vs its shifted ISO 717 reference:
lab.plot()
imp.plot()
plt.show()

# By hand, both panels from the results' fields:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.semilogx(lab.rating.band_centers, lab.r, "o-", label="measured R")
ax1.semilogx(lab.rating.band_centers, lab.rating.shifted_reference, "s--",
             label="shifted reference")
ax1.set_title(f"Rw = {lab.rating.rating} dB")
ax2.semilogx(imp.rating.band_centers, imp.l_n, "o-", label="normalized Ln")
ax2.semilogx(imp.rating.band_centers, imp.rating.shifted_reference, "s--",
             label="shifted reference")
ax2.set_title(f"Ln,w = {imp.rating.rating} dB")
for ax in (ax1, ax2):
    ax.set_xlabel("Frequency [Hz]")
    ax.legend()
ax1.set_ylabel("Sound reduction index R [dB]")
ax2.set_ylabel("Impact sound pressure level Ln [dB]")
plt.show()
```

</details>

### `lab_airborne_insulation()` / `lab_impact_insulation()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `l1` / `l2` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Source / receiving levels (airborne) |
| `li` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Impact SPL from the tapping machine (impact) |
| `t2` | 1D array | s | > 0, one per band | Receiving-room reverberation time |
| `area` | float | m² | > 0 | Free test-opening area $S$ (airborne only) |
| `volume` | float | m³ | > 0 | Receiving-room volume $V$ |

`lab_airborne_insulation()` returns a `LabAirborneInsulationResult` (`r`,
`absorption`, `rating`); `lab_impact_insulation()` a
`LabImpactInsulationResult` (`l_n`, `absorption`, `rating`);
`background_correction(signal_and_background, background)` returns the corrected
levels directly.

### ISO 10140 laboratory test report (`.report()`)

Both laboratory results write the one-page ISO 10140 test report directly, laid
out like the accredited laboratory reports rated per ISO 717.
`LabAirborneInsulationResult.report()` renders the sound reduction index $R$
fiche (ISO 10140-2:2010) and `LabImpactInsulationResult.report()` the
normalized impact sound pressure level $L_n$ fiche (ISO 10140-3:2010). Each
fiche names the laboratory standard in its basis line, evaluates the
ISO 717-1 / ISO 717-2 single-number rating (16 one-third-octave bands from
100 Hz to 3150 Hz, or the 5 octave bands), states the quantity to one decimal
place both in tabular form and as a curve against the shifted reference curve,
boxes the laboratory rating (`Rw (C; Ctr)` or `Ln,w (CI)`) and prints the
statement that the evaluation is based on laboratory measurement results
obtained by a precision method. Because a qualified suite suppresses flanking
transmission, the reported quantity is the *direct* $R$ / $L_n$, not the
field $R'$ / $L'_n$.

`verbose=True` annexes the per-band equivalent sound absorption area
$A = 0.16\,V/T$ (ISO 10140-4:2010) beside the reported quantity, the
normalization datum the laboratory report carries. Metadata (client, specimen,
mounting, room volumes, climatic conditions), the requirement verdict (airborne
passes at or above it, impact at or below it), `language="es"` and the
`phonometry[report]` extra behave exactly as in the ISO 717 and ISO 16283
fiches.

```python
import numpy as np
from phonometry import building, ReportMetadata

# Laboratory airborne: source/receiving levels and T per one-third-octave band
l1 = np.full(16, 90.0)
r = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
              28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
lab = building.lab_airborne_insulation(
    l1, l1 - r, np.full(16, 0.8), area=10.0, volume=50.0
)
lab.plot()   # measured R vs shifted ISO 717-1 reference (needs matplotlib)
metadata = ReportMetadata(
    specimen="100 mm autoclaved aerated concrete block wall",
    client="Example client",
    area=10.0, mass_per_area=75.0,
    source_volume=53.0, receiving_volume=50.0,
    test_room="Transmission suite (example)",
    mounting="Type A mounting, mortar-bedded perimeter (ISO 10140-1)",
    measurement_standard="ISO 10140-2",
    laboratory="Phonometry Reference Laboratory",
    report_id="PHN-2026-0143",
    requirement=30.0,               # Rw >= 30 dB -> PASS/FAIL row
)
lab.report("Rw_lab.pdf", metadata=metadata)            # Rw (C; Ctr)
lab.report("Rw_lab_chain.pdf", metadata=metadata,
           verbose=True)                               # f | A | R

# Laboratory impact: tapping-machine levels in the receiving room
li = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
               73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])
imp = building.lab_impact_insulation(li, np.full(16, 0.8), volume=50.0)
imp.report("Lnw_lab.pdf",
           metadata=ReportMetadata(requirement=80.0))  # Ln,w (CI)
```

Rendered examples of both laboratory fiches, regenerated with `make reports`,
are kept in the repository. Click either preview to open the PDF:

[![Laboratory airborne ISO 10140-2 example report: metadata header, one-third-octave R table beside the measured-versus-shifted-reference curve, boxed Rw (C; Ctr), the precision-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10140_airborne_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10140_airborne_example.pdf)

*Laboratory airborne fiche (`LabAirborneInsulationResult.report`), Rw (C; Ctr).*

[![Laboratory impact ISO 10140-3 example report: the same laboratory layout for the normalized impact level Ln with the 500 Hz read-off, boxed Ln,w (CI), the precision-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10140_impact_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10140_impact_example.pdf)

*Laboratory impact fiche (`LabImpactInsulationResult.report`), Ln,w (CI).*

## Beyond the pressure method

Three sibling laboratory measurements used to share this page and now have
guides of their own. When flanking is too high for the pressure method,
[Sound Insulation by Intensity (ISO 15186)](insulation-intensity.md) reads
the transmitted power directly off the radiating face. For a soft floor
covering, [Floor-Covering Impact Improvement (ISO 16251-1)](../design/impact-improvement.md)
replaces the two-room suite with a small heavyweight mock-up. And the
junction data the EN 12354 prediction consumes is measured per
[Laboratory Flanking Transmission (ISO 10848)](flanking-lab.md).

## References

- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  The reference monograph for laboratory sound-insulation measurement and
  the interpretation of the laboratory indices.
- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  The transmission theory of single and double constructions that laboratory
  indices quantify.

## Standards

ISO 10140-2:2010, ISO 10140-3:2010 and ISO 10140-4:2010, which provide the
laboratory $R$ and $L_n$ with the background-noise correction and the measurement
procedures behind them.

**Not covered.** The ISO 10140-4:2010 procedure and the ISO 10140-1:2010 and
**ISO 10140-5:2010** facility and mounting requirements are documented above
and implemented nowhere: nothing checks the position counts, the separations,
the averaging times, the band range, the test-opening geometry or the
loss-factor requirement. The mounting condition survives only as free text in
the report metadata — which is what ISO 10140-1 asks for, since it requires a
mounting to be *described* rather than picked from a list. Nothing estimates
the suite's $R'_\text{max}$ either, so a result approaching the facility's
ceiling is produced without complaint; that qualification comes from the
laboratory's own Annex A tests. The neighbouring laboratory measurements have
their own pages: intensity (ISO 15186), the floor-covering improvement
(ISO 16251-1) and flanking transmission (ISO 10848).

## See also

- [Field Insulation Measurement (ISO 16283)](insulation-field.md):
  the in-building airborne, impact and façade measurements, their single-number
  ratings and their uncertainty.
- [Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md):
  the flanking model that consumes the laboratory $R$, $L_n$ and $K_{ij}$.
- [Sound Insulation by Intensity (ISO 15186)](insulation-intensity.md): the
  direct-power alternative for high-flanking situations.
- [Floor-Covering Impact Improvement (ISO 16251-1)](../design/impact-improvement.md):
  the small-mock-up $\Delta L$ of soft floor coverings.
- [Laboratory Flanking Transmission (ISO 10848)](flanking-lab.md): the
  measured junction vibration reduction index.
- [Insulation Ratings (ISO 717)](insulation-ratings.md): the reference-curve
  engine behind $R_w$ and $L_{n,w}$.
- [Sound Power](../../devices/emission/sound-power.md): the $L_W$ methods that share the
  absorption-area machinery of the receiving room.
- API reference: [`building.measurement.lab_insulation`](https://jmrplens.github.io/phonometry/reference/api/building/lab-insulation/).
