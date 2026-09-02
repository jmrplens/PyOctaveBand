← [Documentation index](../../README.md)

# Sound Insulation by Intensity (ISO 15186)

A reverberation room reads transmitted sound power *indirectly*: the
receiving room integrates every watt that arrives, whichever path carried
it, and the absorption area converts the room level back into a power. The
**sound-intensity** method of ISO 15186 replaces that inference with a
direct reading: a p-p intensity probe scans a measurement surface that
encloses the specimen and measures the power radiated by the test element
alone, so flanking transmission stays out of the number. This guide covers
the intensity sound reduction index $R_\mathrm{I}$ and its $K_\mathrm{c}$-modified form, the
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
average normal intensity level $L_{I\mathrm{n}}$ over the surface (area $S_\mathrm{m}$), for a
specimen of area $S$,

$$
R_\mathrm{I} = L_{p1} - 6 - \left[ L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{S} \right],
$$

where the $6$ dB is the diffuse-field offset between the sound pressure level
and the incident intensity level. The same formula gives the apparent index
$R'_\mathrm{I}$ in the field (ISO 15186-2). Because the intensity method slightly
*underestimates* the power radiated into a real receiving room, a **modified
index** $R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}$ reproduces the ISO 10140-2 pressure result; the
adaptation term $K_\mathrm{c}$ (Annex B) is $10 \log_{10}(1 + S_{\mathrm{b}2}\lambda/8V_2)$ for a
well-defined room, or the room-independent $10 \log_{10}(1 + 61.4/f)$. For small
elements the **element normalized level difference** replaces $10\log_{10}(S_\mathrm{m}/S)$
with $10\log_{10}(S_\mathrm{m}/A_0) + 10\log_{10} N$ ($A_0 = 10\ \text{m}^2$, $N$ element units).

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

*The modified index $R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}$ lifts $R_\mathrm{I}$ (most at the low bands,
where $K_\mathrm{c}$ is largest), so an intensity measurement reproduces the ISO 10140-2
pressure result. The automatic rating is formed only for exactly 16
one-third-octave or 5 octave values (`rating`/`rating_modified` are `None`
otherwise). Subareas scanned separately are combined first with
`combine_subareas` (Formulas (11)-(12)); a subarea whose net energy flows back
towards the specimen enters with a negative area, applying the minus-sign rule
of Clause 6.4.6 while $S_\mathrm{m}$ keeps the unsigned area sum.*

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
and $L_{I\mathrm{n}}$ are not implemented, and nothing here enforces the Clause 6.4
acquisition either — the 0.1 m to 0.3 m measurement distance, the 0.1 m/s to
0.3 m/s scan speed, the 90°-rotated second scan with its 1.0 dB validity
test, the 10 dB background margin — so a single scan will produce a number
just as readily as a qualified pair.

### `intensity_sound_reduction()` / `adaptation_term_kc()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `lp1` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Source-room sound pressure level |
| `l_in` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Normal intensity level over the surface |
| `measurement_area` | float | m² | > 0 | Measurement-surface area $S_\mathrm{m}$ |
| `area` | float | m² | > 0 | Specimen area $S$ |
| `kc` | 1D array | dB | one per band / `None` | Adaptation term for the modified index |
| `freq` | 1D array | Hz | > 0 | Midband frequencies (`adaptation_term_kc`) |
| `boundary_area` / `volume` | float | m² / m³ | > 0, both or neither | Room $S_{\mathrm{b}2}$ / $V_2$ for Formula (B.1) |

`intensity_sound_reduction()` returns an `IntensityReductionResult` (`r_i`,
`r_i_modified`, `rating`, `rating_modified`);
`intensity_element_normalized_difference()` an
`IntensityElementNormalizedResult` (`d_i_n_e`, `rating`);
`surface_pressure_intensity_indicator()` and `combine_subareas()` return arrays.

## ISO 15186-1 intensity test report (`.report()`)

`IntensityReductionResult.report()` writes the one-page ISO 15186-1:2000 test
report of the intensity sound reduction index $R_\mathrm{I}$, reusing the same
accredited two-panel layout as the ISO 10140 and ISO 16283 fiches. Because
$R_\mathrm{I}$ is an ordinary sound reduction index, its single-number rating $R_\mathrm{I,w}$
is the ISO 717-1 airborne rating evaluated on the intensity spectrum: the fiche
names ISO 15186-1 in its basis line, tabulates $R_\mathrm{I}$ to one decimal place
beside the measured-versus-shifted-reference curve, boxes `RI,w (C; Ctr)` and
prints the statement that the transmitted sound power was measured directly
over the measurement surface. `verbose=True` annexes the $K_\mathrm{c}$-modified index
$R_\mathrm{I,M}$ (Formula (9)) beside $R_\mathrm{I}$ when an adaptation term was supplied.

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

*Intensity fiche (`IntensityReductionResult.report`), $R_\mathrm{I,w}(C; C_\mathrm{tr})$.*

## Small building elements: the element-normalized level difference

For a **small building element** (a ventilator, a socket, a small window) the
intensity method reports the element-normalized level difference $D_\mathrm{I,n,e}$
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

*The small element is rated exactly like a wall: $D_\mathrm{I,n,e}$ feeds the
ISO 717-1 engine and the unfavourable deviations (reference above the
measurement) set $D_\mathrm{I,n,e,w}$. The $10\log_{10}(S_\mathrm{m}/A_0)$ normalization replaces
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

*Element intensity fiche (`IntensityElementNormalizedResult.report`), $D_\mathrm{I,n,e,w}(C; C_\mathrm{tr})$.*

## Laboratory or field (ISO 15186-2)

Part 1 is a laboratory method: the specimen sits in a test opening between
two rooms, and the suppressed flanking of the facility makes $R_\mathrm{I}$ a
property of the element. ISO 15186-2 takes the same probe, the same surface
scan and the same formula into the finished building, where the result is
the apparent index $R'_\mathrm{I}$ of the installed element, flanking radiation and
all. The selectivity that motivated the method in the laboratory doubles in
the field: because the probe reads only what its measurement surface
radiates, partial surfaces isolate which element (the wall itself, a window,
a leaky junction) carries the transmitted power, which no pressure-based
field method can resolve. Qualify each scan with the same
pressure-intensity indicator, and expect the probe's own limits (the
[spacer phase error](../../devices/emission/intensity.md)) to set the usable low-frequency range.
Part 2's own field procedure (the loudspeaker positions, the façade cases) is
not implemented: the formulas here apply unchanged to field data, but nothing
checks how that data was acquired.

## Low frequencies: the pressure on the specimen (ISO 15186-3)

Both parts above put the source microphone in the *room*, and below 100 Hz
that is exactly where a source room stops being able to answer. A room a
laboratory can build has too few modes down there for a space average to
describe the field driving the specimen: move the microphones and the average
moves with them. ISO 15186-3 keeps the intensity probe on the receiving side
and changes the other end of the measurement — the source-room level is read
**on the surface of the test specimen itself**, from fixed microphones no more
than 50 mm from it (Clause 6.3):

$$
R_\mathrm{I} = L_{p\mathrm{S}} - 9 - \left[ L_{I\mathrm{n}} + 10\lg\frac{S_\mathrm{m}}{S} \right]\ \mathrm{dB}
$$

**Nine decibels, not six.** That single constant is the whole difference from
Part 1. Close to a rigid boundary a diffuse field carries twice the
mean-square pressure it carries away from one, so a microphone 50 mm from the
specimen reads 3 dB above the room average of the same field. Part 1's room
average carries no such build-up and subtracts 6; the surface average carries
it and subtracts 9. A NOTE under Formula (7) bounds the assumption: it holds
for a specimen with a reflecting surface in the source room, still works
behind 100 mm of porous absorber, should be read only from 50 Hz to 80 Hz
behind 100 mm to 200 mm of it, and is not valid behind anything thicker.

Clause 6.6 requires filters at 50 Hz, 63 Hz and 80 Hz and allows 100 Hz,
125 Hz and 160 Hz to be added; the ceiling is Clause 1.1's, which applies this
part over 50 Hz to 160 Hz and says it is mainly intended for 50 Hz to 80 Hz.
Six bands is the whole method, which is why no single-number rating comes out
of it, and Clause 1.1 says what to do instead: combine these results with
ISO 140-3 and ISO 15186-1 into one curve over 50 Hz to 5000 Hz. The Clause 6.4.2
indicator $F_{pI} = L_p - L_{I\mathrm{n}}$ is the same as everywhere else in the
series, and both of its levels are read on the measurement surface in the
*receiving* room — not the surface level Formula (7) is built from. Its two
limits distinguish the specimen: $F_{pI} > 10$ dB refuses a sound-reflecting one
and $F_{pI} > 6$ dB one presenting a sound-absorbing surface in the receiving
room. Since the standard asks for that second pressure measurement only "if
possible", `l_p` is optional and its absence leaves the verdict unanswered
rather than guessed.

```python
import numpy as np
from phonometry import building

# A light partition, 10 m2, scanned over a 12.6 m2 box surface.
freqs = [50.0, 63.0, 80.0, 100.0, 125.0, 160.0]
res = building.low_frequency_intensity_reduction(
    np.array([88.4, 89.1, 90.3, 91.0, 91.4, 91.8]),
    np.array([61.6, 60.9, 59.8, 57.9, 56.0, 53.9]),
    measurement_area=12.6, area=10.0,
    l_p=np.array([74.0, 72.0, 69.4, 66.1, 63.1, 60.3]),
    frequencies=freqs,
)
print(np.round(res.r_i, 1))                         # [16.8 18.2 20.5 23.1 25.4 27.9]
print(np.round(res.surface_pressure_intensity, 1))  # [12.4 11.1  9.6  8.2  7.1  6.4]
print(res.qualified.tolist())    # [False, False, True, True, True, True]
```

A refused band is flagged, not dropped: the standard's answer to an indicator
over the limit is to improve the measurement environment, so the index is
still computed and the verdict travels beside it.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/low_frequency_intensity_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/low_frequency_intensity.svg" alt="Left: the low-frequency intensity sound reduction index per one-third-octave band from 50 Hz to 160 Hz as bars, with the 50 Hz and 63 Hz bands hatched because their surface pressure-intensity indicator exceeds 10 dB, and the indicator drawn as a curve on a twin axis against its 10 dB limit line. Right: the calculated limp-panel sound reduction index of Annex A with the 4 dB tolerance the annex allows shaded around it and a measured curve staying inside" width="92%"></picture>

*Left, the measurement: the index rises through the six bands while the
indicator falls past its limit, so the two lowest bands are computed but not
qualified. Right, the facility: Annex A's calculated limp-panel curve and the
4 dB either side of it that a measurement has to stay within.*

### Small elements, and a sign the series disagrees on

Formula (8) normalizes a small element to $A_0 = 10$ m² as Part 1 does, with
the 9 dB of the surface measurement in place of 6:

$$
D_{I\mathrm{n,e}} = L_{p\mathrm{S}} - 9 - \left[ L_{I\mathrm{n}} - 10\lg\frac{A_0}{S_\mathrm{m}} - 10\lg N \right]\ \mathrm{dB}
$$

The $10\lg N$ inside the bracket is subtracted, so it reaches $D_{I\mathrm{n,e}}$
**added** — the sign this library derives, and the opposite of what
ISO 15186-1 prints for the same quantity. Two parts of one series print the
two signs; this is the one that agrees with the physics and with
ISO 10140-2 Formula (6). The Part 1 print is registered in
[ERRATA](../../ERRATA.md).

```python
# Four identical trickle ventilators inside one 2.0 m2 measurement surface.
element = building.low_frequency_element_normalized_difference(
    np.full(6, 90.0), np.array([62.0, 61.0, 60.0, 58.0, 57.0, 55.0]),
    measurement_area=2.0, elements=4, frequencies=freqs,
)
print(np.round(element.d_i_n_e, 1))     # [32. 33. 34. 36. 37. 39.]
```

### Qualifying the facility (Annex A)

Annex A is normative: measure a limp panel of more than 1 m², calculate what
it should read, and require the two to agree within **4,0 dB** from 50 Hz to
160 Hz. The calculated half is forced transmission alone — mass law reduced by
the radiation efficiency of a plate driven by a diffuse field,

$$
R = R_0 - 10\lg 2\sigma_\mathrm{d}, \qquad
R_0 = 20\lg\frac{\pi f m}{\rho c}, \qquad
\sigma_\mathrm{d} = \tfrac{1}{2}\left[0{,}20 + \ln\left(2\pi\frac{f}{c}\sqrt{S}\right)\right]
$$

with the air taken at the climate of the test,
$\rho c = 427\sqrt{273/(273+\theta)}\cdot B/B_0$ and $c = 331 + 0{,}6\,\theta$.

```python
# Annex A, Table A.1: 12,5 mm plaster board, 10 kg/m2, over a 10 m2 opening.
calc = building.limp_panel_reduction_index(
    freqs, surface_mass=10.0, area=10.0,
    temperature=23.0, static_pressure=101300.0,
)
print(np.round(calc, 1))    # [10.7 11.9 13.4 14.8 16.3 17.9]
```

Those six values are the printed table, band for band, and they anchor all
five formulas. The steel-sandwich column beside them is not an oracle: no reading
of the inputs printed next to it reproduces it, and the one that would takes an
area and a surface mass moved together, the mass landing heavier than solid
steel of the stated thickness. It too is in [ERRATA](../../ERRATA.md).

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
  The intensity sound reduction index, the $K_\mathrm{c}$ adaptation and the element
  normalized level difference this page implements.
- International Organization for Standardization. (2003). *Acoustics —
  Measurement of sound insulation in buildings and of building elements
  using sound intensity — Part 2: Field measurements* (ISO 15186-2:2003).
  [iso.org catalogue](https://www.iso.org/standard/30105.html).
  The field counterpart, giving the apparent index $R'_\mathrm{I}$ of the installed
  element.
- International Organization for Standardization (2002). *Acoustics —
  Measurement of sound insulation in buildings and of building elements
  using sound intensity — Part 3: Laboratory measurements at low
  frequencies* (ISO 15186-3:2002).
  [iso.org catalogue](https://www.iso.org/standard/33962.html).
  The low-frequency variant: the source level is read on the surface of the
  specimen and 9 dB is subtracted instead of 6, with the normative limp-panel
  qualification of Annex A.

## Standards

ISO 15186-1:2000, which defines the intensity sound reduction index $R_\mathrm{I}$,
its $K_\mathrm{c}$-modified form $R_\mathrm{I,M}$ (Annex B), the element normalized level
difference $D_\mathrm{I,n,e}$ and the surface pressure-intensity indicator;
ISO 15186-2:2003, which carries the same quantities into the field as the
apparent $R'_\mathrm{I}$; and ISO 15186-3:2002, which reads the source level on the
surface of the specimen so the method reaches 50 Hz, and qualifies the
facility on a limp panel (Annex A). The single-number ratings reuse ISO 717-1,
which Part 3's six bands cannot feed.

## See also

- [Laboratory Insulation Measurement](insulation-lab.md): the ISO 10140
  pressure method that the $K_\mathrm{c}$ adaptation reproduces.
- [Field Insulation Measurement (ISO 16283)](insulation-field.md): the
  pressure-based field quantities and their single-number ratings.
- [Insulation Ratings (ISO 717)](insulation-ratings.md): the
  reference-curve engine behind $R_\mathrm{I,w}$ and $D_\mathrm{I,n,e,w}$.
- [Sound Intensity](../../devices/emission/intensity.md): the two-microphone probe, its field
  indicators and the residual-index qualification behind every scan.
- API reference: [`building.measurement.intensity_insulation`](https://jmrplens.github.io/phonometry/reference/api/building/intensity-insulation/).
