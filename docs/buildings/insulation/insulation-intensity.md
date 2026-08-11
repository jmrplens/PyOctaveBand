← [Documentation index](../../README.md)

# Sound Insulation by Intensity (ISO 15186)

A reverberation room reads transmitted sound power *indirectly*: the
receiving room integrates every watt that arrives, whichever path carried
it, and the absorption area converts the room level back into a power. The
**sound-intensity** method of ISO 15186 replaces that inference with a
direct reading: a p-p intensity probe scans a measurement surface that
encloses the specimen and measures the power radiated by the test element
alone, so flanking transmission stays out of the number. This guide covers
the intensity sound reduction index $R_I$ and its $K_c$-modified form, the
element-normalized level difference of small building elements, the surface
qualification indicator and the accredited test fiches. The pressure-based
laboratory chain lives in
[Laboratory Insulation Measurement](insulation-lab.md); the probe physics
and its field indicators are the subject of the
[Sound Intensity guide](../../devices/emission/intensity.md).

## Measuring the transmitted power directly (ISO 15186-1)


The [ISO 10140 laboratory method](insulation-lab.md) reads the transmitted
power *indirectly*, from the
receiving-room level and its absorption area; this breaks down when flanking
paths leak power the room integrates in anyway. The **sound-intensity** method
(ISO 15186) sidesteps that: an intensity probe scans a measurement surface that
encloses the specimen and measures the radiated power *directly*, so only the
element under test contributes. It is the tool of choice when flanking is high
(ISO 15186-1:2000, Clause 1). From the source-room level $L_{p1}$ and the
average normal intensity level $L_{In}$ over the surface (area $S_m$), for a
specimen of area $S$,

$$
R_I = L_{p1} - 6 - \left[ L_{In} + 10 \log_{10}\frac{S_m}{S} \right],
$$

where the $6$ dB is the diffuse-field offset between the sound pressure level
and the incident intensity level. The same formula gives the apparent index
$R'_I$ in the field (ISO 15186-2). Because the intensity method slightly
*underestimates* the power radiated into a real receiving room, a **modified
index** $R_{I,M} = R_I + K_c$ reproduces the ISO 10140-2 pressure result; the
adaptation term $K_c$ (Annex B) is $10 \log_{10}(1 + S_{b2}\lambda/8V_2)$ for a
well-defined room, or the room-independent $10 \log_{10}(1 + 61.4/f)$. For small
elements the **element normalized level difference** replaces $10\log_{10}(S_m/S)$
with $10\log_{10}(S_m/A_0) + 10\log_{10} N$ ($A_0 = 10\ \text{m}^2$, $N$ element units).

```python
import numpy as np
from phonometry import building

# Source-room level Lp1 and the average normal intensity level LIn over the
# measurement surface (Sm), for a specimen of area S; 16 one-third-octave bands.
lp1 = np.full(16, 85.0)
l_in = np.full(16, 40.0)
freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150]   # nominal 1/3-octave centres
kc = building.adaptation_term_kc(freqs)                  # Annex B (B.2)
res = building.intensity_sound_reduction(lp1, l_in, measurement_area=12.0, area=10.0, kc=kc)
print(round(float(res.r_i[0]), 2))          # 38.21  RI = Lp1 - 6 - [LIn + 10 lg(Sm/S)]
print(round(float(res.r_i_modified[0]), 2)) # 40.29  RI,M = RI + Kc
print(res.rating.rating)                     # 38  ->  RI,w (ISO 717-1 engine)

# Qualify the measurement surface: FpI = Lp - LIn must stay < 10 dB (< 6 dB when
# the receiving side is absorbing); the probe's residual index must exceed FpI+10.
fpi = building.surface_pressure_intensity_indicator(np.full(16, 46.0), l_in)
print(round(float(fpi[0]), 1))               # 6.0

res.plot()   # measured RI vs shifted ISO 717-1 reference (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/intensity_insulation_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/intensity_insulation.svg" alt="Intensity sound reduction index RI and the Kc-modified index RI,M across the one-third-octave bands, with the Annex B adaptation lift shaded between the two curves" width="80%"></picture>

*The modified index $R_{I,M} = R_I + K_c$ lifts $R_I$ (most at the low bands,
where $K_c$ is largest), so an intensity measurement reproduces the ISO 10140-2
pressure result. The automatic rating is formed only for exactly 16
one-third-octave or 5 octave values (`rating`/`rating_modified` are `None`
otherwise). Subareas scanned separately are combined first with
`combine_subareas` (Formulas (11)-(12)); a subarea whose net energy flows back
towards the specimen enters with a negative area, applying the minus-sign rule
of Clause 6.4.6 while $S_m$ keeps the unsigned area sum.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# A light wall: source-room SPL Lp1 = 85 dB and the measured normal intensity
# level LIn over the Sm = 12 m2 surface, 16 one-third-octave bands.
freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150]
l_in = np.array([57.8, 61.9, 60.5, 55.6, 55.8, 55.5, 53.4, 51.6,
                 50.2, 47.7, 46.4, 45.7, 44.8, 45.2, 47.2, 52.7])
kc = building.adaptation_term_kc(freqs)           # Annex B adaptation term
res = building.intensity_sound_reduction(np.full(16, 85.0), l_in,
                                         measurement_area=12.0, area=10.0,
                                         kc=kc)

x = np.arange(len(freqs))
fig, ax = plt.subplots()
ax.fill_between(x, res.r_i, res.r_i_modified, alpha=0.2, label="Kc adaptation")
ax.plot(x, res.r_i, "-o", label="RI (intensity)")
ax.plot(x, res.r_i_modified, "--s", label="RI,M = RI + Kc")
ax.set_xticks(x, [str(f) for f in freqs], rotation=45)
ax.set(xlabel="Frequency [Hz]", ylabel="Sound reduction index [dB]",
       title=f"RI,w = {res.rating.rating} dB, RI,M,w = {res.rating_modified.rating} dB")
ax.legend()
plt.show()
```

</details>

Both levels are already-measured inputs: the scanning probe, the
two-microphone acquisition and the phase-mismatch calibration behind $L_{p1}$
and $L_{In}$ are not implemented, and nothing here enforces the Clause 6.4
acquisition either — the 0.1 m to 0.3 m measurement distance, the 0.1 m/s to
0.3 m/s scan speed, the 90°-rotated second scan with its 1.0 dB validity
test, the 10 dB background margin — so a single scan will produce a number
just as readily as a qualified pair.

### `intensity_sound_reduction()` / `adaptation_term_kc()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `lp1` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Source-room sound pressure level |
| `l_in` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Normal intensity level over the surface |
| `measurement_area` | float | m² | > 0 | Measurement-surface area $S_m$ |
| `area` | float | m² | > 0 | Specimen area $S$ |
| `kc` | 1D array | dB | one per band / `None` | Adaptation term for the modified index |
| `freq` | 1D array | Hz | > 0 | Midband frequencies (`adaptation_term_kc`) |
| `boundary_area` / `volume` | float | m² / m³ | > 0, both or neither | Room $S_{b2}$ / $V_2$ for Formula (B.1) |

`intensity_sound_reduction()` returns an `IntensityReductionResult` (`r_i`,
`r_i_modified`, `rating`, `rating_modified`);
`intensity_element_normalized_difference()` an
`IntensityElementNormalizedResult` (`d_i_n_e`, `rating`);
`surface_pressure_intensity_indicator()` and `combine_subareas()` return arrays.

## ISO 15186-1 intensity test report (`.report()`)

`IntensityReductionResult.report()` writes the one-page ISO 15186-1:2000 test
report of the intensity sound reduction index $R_I$, reusing the same
accredited two-panel layout as the ISO 10140 and ISO 16283 fiches. Because
$R_I$ is an ordinary sound reduction index, its single-number rating $R_{I,w}$
is the ISO 717-1 airborne rating evaluated on the intensity spectrum: the fiche
names ISO 15186-1 in its basis line, tabulates $R_I$ to one decimal place
beside the measured-versus-shifted-reference curve, boxes `RI,w (C; Ctr)` and
prints the statement that the transmitted sound power was measured directly
over the measurement surface. `verbose=True` annexes the $K_c$-modified index
$R_{I,M}$ (Formula (9)) beside $R_I$ when an adaptation term was supplied.

The applicable `ReportMetadata` fields describe the intensity measurement:
`specimen` (the tested element), `area` (specimen area $S$), `client`,
`manufacturer`, `test_room`, `laboratory`, `operator`, `report_id` and
`test_date`, plus the room/climate fields shared with the other insulation
fiches. There is no dedicated field for the measurement-surface geometry or the
scanning-versus-discrete-point acquisition method; record those in `notes` and
name the standard in `measurement_standard` (`"ISO 15186-1"`). The requirement
verdict, `language="es"` and the `phonometry[report]` extra behave exactly as
in the sibling fiches.

```python
import numpy as np
from phonometry import building, ReportMetadata

freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
lp1, sm, s = 85.0, 12.0, 10.0
l_in = np.array([57.8, 61.9, 60.5, 55.6, 55.8, 55.5, 53.4, 51.6,
                 50.2, 47.7, 46.4, 45.7, 44.8, 45.2, 47.2, 52.7])
kc = building.adaptation_term_kc(freqs)                 # Annex B, Formula (B.2)
res = building.intensity_sound_reduction(
    np.full(16, lp1), l_in, measurement_area=sm, area=s, kc=kc
)
metadata = ReportMetadata(
    specimen="100 mm autoclaved aerated concrete block wall",
    area=10.0, measurement_standard="ISO 15186-1",
    test_room="Transmission suite (example)",
    laboratory="Phonometry Reference Laboratory",
    report_id="PHN-2026-0150",
    requirement=30.0,                                   # RI,w >= 30 dB -> PASS
)
res.report("RIw.pdf", metadata=metadata)                # RI,w (C; Ctr)
res.report("RIw_kc.pdf", metadata=metadata, verbose=True)  # f | RI | RI,M
```

The example fiche is regenerated with `make reports` and kept in the
repository. Click the preview to open the PDF:

[![Intensity ISO 15186-1 example report: metadata header, one-third-octave RI table beside the measured-versus-shifted-reference curve, boxed RI,w (C; Ctr), the intensity-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso15186_intensity_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso15186_intensity_example.pdf)

*Intensity fiche (`IntensityReductionResult.report`), $R_{I,w}(C; C_{tr})$.*

## Small building elements: the element-normalized level difference

For a **small building element** (a ventilator, a socket, a small window) the
intensity method reports the element-normalized level difference $D_{I,n,e}$
(Formula (8)) instead, normalized to the reference absorption area
$A_0 = 10\ \text{m}^2$.
`IntensityElementNormalizedResult.report()` writes the same one-page fiche
through the shared renderer, boxing `DI,n,e,w (C; Ctr)` rated per ISO 717-1;
`verbose=True` shows the ISO 717 evaluation per band and a `requirement` adds a
PASS/FAIL verdict (the element insulation passes at or above the target).

```python
import numpy as np
from phonometry import building, ReportMetadata

lp1, sm, n = 85.0, 12.0, 1                              # source SPL, surface, units
l_in = np.array([57.9, 62.0, 60.6, 55.7, 55.9, 55.6, 53.5, 51.7,
                 50.3, 47.8, 46.5, 45.8, 44.9, 45.3, 47.3, 52.8])
res = building.intensity_element_normalized_difference(
    np.full(16, lp1), l_in, measurement_area=sm, n=n
)
metadata = ReportMetadata(
    specimen="Trickle ventilator in a 100 mm masonry wall",
    measurement_standard="ISO 15186-1",
    laboratory="Phonometry Reference Laboratory",
    report_id="PHN-2026-0151",
    requirement=30.0,                                   # DI,n,e,w >= 30 dB -> PASS
)
res.plot()   # DI,n,e vs shifted ISO 717-1 reference (needs matplotlib)
res.report("DIne.pdf", metadata=metadata)               # DI,n,e,w (C; Ctr)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/intensity_element_insulation_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/intensity_element_insulation.svg" alt="Element-normalized level difference DI,n,e of a trickle ventilator per one-third-octave band against the shifted ISO 717-1 reference curve, with the unfavourable deviations shaded and the DI,n,e,w rating annotated" width="80%"></picture>

*The small element is rated exactly like a wall: $D_{I,n,e}$ feeds the
ISO 717-1 engine and the unfavourable deviations (reference above the
measurement) set $D_{I,n,e,w}$. The $10\log_{10}(S_m/A_0)$ normalization replaces
the specimen-area term, so the number describes the element irrespective of
the wall it sits in.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# A trickle ventilator in a masonry wall: source-room SPL 85 dB and the
# normal intensity level over the Sm = 12 m2 measurement surface.
l_in = np.array([57.9, 62.0, 60.6, 55.7, 55.9, 55.6, 53.5, 51.7,
                 50.3, 47.8, 46.5, 45.8, 44.9, 45.3, 47.3, 52.8])
res = building.intensity_element_normalized_difference(
    np.full(16, 85.0), l_in, measurement_area=12.0, n=1
)

# One line — DI,n,e vs the shifted ISO 717-1 reference:
res.plot()
plt.show()

# By hand, from the rating the result carries:
w = res.rating
fig, ax = plt.subplots()
ax.semilogx(w.band_centers, res.d_i_n_e, "o-", label="DI,n,e (element)")
ax.semilogx(w.band_centers, w.shifted_reference, "s--",
            label="shifted reference")
ax.fill_between(w.band_centers, w.measured, w.shifted_reference,
                where=w.measured < w.shifted_reference, interpolate=True,
                alpha=0.3, label="unfavourable deviations")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Element normalized level difference [dB]")
ax.set_title(f"DI,n,e,w = {w.rating} dB  (C={w.c:+d}; Ctr={w.ctr:+d})")
ax.legend()
plt.show()
```

</details>

[![Intensity ISO 15186-1 element example report: metadata header, one-third-octave DI,n,e table beside the measured-versus-shifted-reference curve, boxed DI,n,e,w (C; Ctr), the intensity-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso15186_element_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso15186_element_example.pdf)

*Element intensity fiche (`IntensityElementNormalizedResult.report`), $D_{I,n,e,w}(C; C_{tr})$.*

## Laboratory or field (ISO 15186-2)

Part 1 is a laboratory method: the specimen sits in a test opening between
two rooms, and the suppressed flanking of the facility makes $R_I$ a
property of the element. ISO 15186-2 takes the same probe, the same surface
scan and the same formula into the finished building, where the result is
the apparent index $R'_I$ of the installed element, flanking radiation and
all. The selectivity that motivated the method in the laboratory doubles in
the field: because the probe reads only what its measurement surface
radiates, partial surfaces isolate which element (the wall itself, a window,
a leaky junction) carries the transmitted power, which no pressure-based
field method can resolve. Qualify each scan with the same
pressure-intensity indicator, and expect the probe's own limits (the
[spacer phase error](../../devices/emission/intensity.md)) to set the usable low-frequency range.
Part 2's own field procedure (the loudspeaker positions, the façade cases) and
the low-frequency Part 3 variant are not implemented: the formulas here apply
unchanged to field data, but nothing checks how that data was acquired.

## References

- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  The reference monograph for laboratory sound-insulation measurement,
  including the intensity route and its relation to the pressure methods.
- International Organization for Standardization. (2000). *Acoustics —
  Measurement of sound insulation in buildings and of building elements
  using sound intensity — Part 1: Laboratory measurements*
  (ISO 15186-1:2000).
  [iso.org catalogue](https://www.iso.org/standard/26097.html).
  The intensity sound reduction index, the $K_c$ adaptation and the element
  normalized level difference this page implements.
- International Organization for Standardization. (2003). *Acoustics —
  Measurement of sound insulation in buildings and of building elements
  using sound intensity — Part 2: Field measurements* (ISO 15186-2:2003).
  [iso.org catalogue](https://www.iso.org/standard/30105.html).
  The field counterpart, giving the apparent index $R'_I$ of the installed
  element.

## Standards

ISO 15186-1:2000, which defines the intensity sound reduction index $R_I$,
its $K_c$-modified form $R_{I,M}$ (Annex B), the element normalized level
difference $D_{I,n,e}$ and the surface pressure-intensity indicator;
ISO 15186-2:2003, which carries the same quantities into the field as the
apparent $R'_I$. The single-number ratings reuse ISO 717-1.

## See also

- [Laboratory Insulation Measurement](insulation-lab.md): the ISO 10140
  pressure method that the $K_c$ adaptation reproduces.
- [Field Insulation Measurement (ISO 16283)](insulation-field.md): the
  pressure-based field quantities and their single-number ratings.
- [Insulation Ratings (ISO 717)](insulation-ratings.md): the
  reference-curve engine behind $R_{I,w}$ and $D_{I,n,e,w}$.
- [Sound Intensity](../../devices/emission/intensity.md): the two-microphone probe, its field
  indicators and the residual-index qualification behind every scan.
- API reference: [`building.measurement.intensity_insulation`](https://jmrplens.github.io/phonometry/reference/api/building/intensity-insulation/).
