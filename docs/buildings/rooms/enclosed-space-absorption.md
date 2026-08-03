← [Documentation index](../../README.md)

# Sound absorption in enclosed spaces (EN 12354-6)

**EN 12354-6:2003** predicts the **total equivalent sound absorption area** of a
room and its **reverberation time** from the absorption of its surfaces and
objects, the design counterpart of the measured reverberation time. It is the
absorption member of the EN 12354 building-acoustics family (the airborne and
impact insulation members live in
[Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md)).
phonometry implements the normative Clause 4 model. (The informative Annex D
method for irregular spaces is out of scope.)

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_en12354_6_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_en12354_6.svg" alt="Flow from the room surfaces (area and absorption per band) and objects (volume, giving an equivalent area Vobj to the two-thirds power) into the total equivalent absorption area A = sum of alpha times S plus the object areas plus air absorption, then the object fraction psi, and finally the reverberation time T = 55.3/c0 times V times (1 minus psi) over A" width="82%"></picture>

## 1. Equivalent absorption area (clause 4.3)

The total equivalent absorption area sums, over the surfaces $i$, the objects
$j$ and the object arrays $k$, each surface's area times its absorption
coefficient, the equivalent absorption areas of the objects, the object arrays
(groups of identical objects treated as an absorbing surface of area $S_k$),
and the air absorption (Formula 1):

$$
A = \sum_i \alpha_{s,i}\,S_i + \sum_j A_{\mathrm{obj},j}
    + \sum_k \alpha_{s,k}\,S_k + A_{\mathrm{air}}.
$$

For hard, irregular objects whose absorption is not measured, an empirical
estimate from the volume is used (Formula 4):
$A_{\mathrm{obj}} = V_{\mathrm{obj}}^{2/3}$.

```python
from phonometry import room

# EN 12354-6 Annex E, bare room (29.75 m3), 1000 Hz octave band.
surfaces = [(12.39, 0.05), (12.39, 0.02), (10.90, 0.04),
            (10.90, 0.04), (6.55, 0.04), (6.55, 0.04)]
print(round(room.equivalent_absorption_area(surfaces), 2))  # 2.26  m2
print(round(float(room.hard_object_absorption(0.65)), 3))   # 0.75  m2
```

Air absorption uses the power attenuation coefficient $m$ (Formula 2):
$A_{\mathrm{air}} = 4\,m\,V\,(1 - \psi)$. Below 1 kHz and for rooms under
200 m³ it can be neglected.

## 2. Reverberation time (clause 4.4)

The reverberation time follows from the absorption area, the volume and the
object fraction $\psi = \sum V_{\mathrm{obj}}/V$ (Formula 5):

$$
T = \frac{55.3}{c_0}\,\frac{V\,(1 - \psi)}{A},
$$

where the speed of sound $c_0 = 345.6\ \text{m/s}$ makes the factor
$55.3/c_0$ the familiar $0.16$.

```python
from phonometry import room

surfaces = [(12.39, 0.05), (12.39, 0.02), (10.90, 0.04),
            (10.90, 0.04), (6.55, 0.04), (6.55, 0.04)]
a = room.equivalent_absorption_area(surfaces)
print(round(room.reverberation_time(a, 29.75), 1))          # 2.1  s

# Annex E case 2: add furniture (hard objects) to the same room.
volumes = [0.15, 0.60, 0.05, 0.05, 0.65, 0.65]
aobj = room.hard_object_absorption(volumes)
psi = room.object_fraction(volumes, 29.75)                  # 0.072
a2 = room.equivalent_absorption_area(surfaces, objects=aobj)
print(round(a2, 2), round(room.reverberation_time(a2, 29.75, object_fraction=psi), 1))
# 5.03 0.9
```

Per octave band, one call takes the surfaces (with per-band absorption
coefficients) and the air condition and returns the whole spectrum:

```python
from phonometry import room

# Per-band absorption coefficients (125 Hz to 8 kHz) for each surface.
plaster = [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05]
tile = [0.15, 0.35, 0.65, 0.85, 0.90, 0.90, 0.85]
result = room.enclosed_space_reverberation(
    [(54.0, plaster), (20.0, plaster), (20.0, tile)],
    volume=60.0, air_condition="20C_50-70",
)
print(result.reverberation_time.round(2))
# [2.13 1.03 0.62 0.48 0.43 0.42 0.4 ]
result.plot()   # the figure below: A and T per octave band
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/enclosed_space_absorption_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/enclosed_space_absorption.svg" alt="Two panels for a 60 cubic metre office with a bare versus an acoustically-treated ceiling. Left: the equivalent absorption area per octave band, much higher across mid and high frequencies with the acoustic ceiling. Right: the reverberation time per octave band, falling from around five seconds at low frequency for the bare room to under one second with the acoustic ceiling" width="96%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import room

plaster = [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05]
tile = [0.15, 0.35, 0.65, 0.85, 0.90, 0.90, 0.85]
walls_floor = [(54.0, plaster), (20.0, plaster)]
for ceiling in (plaster, tile):
    room.enclosed_space_reverberation(
        [*walls_floor, (20.0, ceiling)], 60.0, air_condition="20C_50-70",
    ).plot()
plt.show()
```

</details>

The `ReverberationResult` carries the per-band absorption area and reverberation
time, the volume and the object fraction, and its `.plot()` draws the
reverberation-time spectrum. This is the prediction counterpart of the measured
reverberation time in [Room Acoustics](room-acoustics.md)
(ISO 3382) and of the reverberation-room absorption of
[Sound Absorption Measurement and Rating](../../materials/absorbers/absorption-measurement.md)
(ISO 354).

## 3. Where the input data comes from

**Surface coefficients.** The standard expects the $\alpha_{s,i}$ to come
from laboratory measurements to EN ISO 354, the reverberation-room method of
[Sound Absorption Measurement and Rating](../../materials/absorbers/absorption-measurement.md);
theoretical, empirical or field
values are admitted as long as the data source is stated. ISO 354 delivers
one-third-octave data, and an octave-band calculation takes the arithmetic
mean of the three thirds as its input. A reverberation-room coefficient can
exceed 1.0 (edge diffraction scatters more energy into the sample than its
flat area intercepts); it enters Formula 1 as measured, without clamping,
because the same diffuse-field convention that produced it is the one the
model assumes.

**Furniture and occupants.** Objects contribute through three routes:
a measured equivalent absorption area $A_{obj}$ when one exists (persons
and seating have tabulated values in the informative Annex C), the
Formula 4 estimate $V_{obj}^{2/3}$ for hard, irregular, unmeasured objects
(furniture, machinery), and object *arrays* rated as an absorbing surface
$\alpha_s S_k$ when many similar objects cover a zone (an audience, a
storage rack). Objects also displace air: their summed volume enters the
object fraction $\psi$ that shortens $T$ in Formula 5 beyond what their
absorption alone would.

**Air.** The air term $A_{air} = 4mV(1-\psi)$ uses the power attenuation
coefficient $m$ from the standard's Table 1, resolved by the
`air_condition` strings (temperature and relative-humidity class, derived
from ISO 9613-1); it only matters above 1 kHz and grows with the volume.
The six built-in profiles, `"10C_30-50"` through `"20C_70-90"` (clause 4.3
recommends `"20C_50-70"` when no conditions are specified), cover the
standard 125 Hz to 8 kHz octave bands only and cannot be combined with a
custom frequency axis; `air_condition=None` (the default) omits the air
term, and for other frequencies or conditions compute $m$ per ISO 9613-1
and chain `air_absorption_area` into `equivalent_absorption_area`.

**Validity limits (clause 4.6).** The model assumes an ordinary,
reasonably diffuse room: no dimension more than 5 times another, opposite
surface pairs whose coefficients differ by less than a factor of 3 (unless
scattering objects are present) and an object fraction below 0.2. Outside
those limits the field is not diffuse and the model errs on the optimistic
side: the standard's own accuracy clause records measured reverberation
times up to twice the prediction in low-diffusivity rooms. The classical
alternatives for those cases live in
[Reverberation-time prediction](reverberation-prediction.md).

## 4. Enclosed-space report (`.report()`)

`ReverberationResult.report(path)` renders a one-page PDF fiche characterising
the enclosed space: a basis line naming EN 12354-6:2003, an optional metadata
header block (client, room, description, room volume, object fraction, climate),
a per-band table of the equivalent sound absorption area $A$ and the
reverberation time $T$ beside the reverberation-time plot (`.plot()`), and the
boxed mid-frequency reverberation time with the mid-frequency absorption area
alongside. EN 12354-6 gives a diffuse-field **estimate**, not a measurement, so
no PASS/FAIL verdict is emitted; a target reverberation time supplied through
the metadata's `requirement` field is printed as a reference line only, since a
room reverberation time is a target range rather than a strictly
higher/lower-is-better quantity. It uses the same `ReportMetadata` container
(documented under
[Insulation ratings](../insulation/insulation-ratings.md#report-metadata-reportmetadata)) and
rendering engine as the other fiches; passing `metadata=None` produces a bare
characterisation fiche. Rendering needs reportlab and, for the figure the fiche
embeds, matplotlib (`pip install "phonometry[report,plot]"`); only
`engine="reportlab"` is supported. The fiche renders in English by default; pass
`language="es"` for a Spanish fiche (translated fixed strings and a comma
decimal separator).

```python
from phonometry import (
    enclosed_space_reverberation, hard_object_absorption, object_fraction,
    ReportMetadata,
)

surfaces = [                                   # per octave band, 125 Hz - 8 kHz
    (20.0, [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55]),  # carpeted floor
    (20.0, [0.20, 0.40, 0.65, 0.75, 0.80, 0.80, 0.75]),  # acoustic ceiling
    (45.0, [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05]),  # painted-plaster walls
]
volumes = [0.5, 0.8, 0.3]                      # furniture, m^3
result = enclosed_space_reverberation(
    surfaces, 50.0,
    objects=hard_object_absorption(volumes),
    object_fraction=object_fraction(volumes, 50.0),
    air_condition="20C_50-70",
)
result.report(
    "enclosed_space_fiche.pdf",
    metadata=ReportMetadata(
        specimen="Meeting room, furnished",
        test_room="Meeting room M2",
        measurement_standard="EN 12354-6",
        temperature=20.0, relative_humidity=55.0,
        laboratory="Phonometry Reference Laboratory",
        requirement=0.6,          # printed as a target reference line, no verdict
    ),
)                                 # the per-band A/T table + the boxed T_mid
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![EN 12354-6 enclosed-space example report: a metadata header with the room volume and object fraction, the octave-band table of the equivalent sound absorption area A and the reverberation time T from 125 Hz to 8 kHz beside the reverberation-time plot, and the boxed mid-frequency reverberation time with the mid-frequency absorption area alongside](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/enclosed_space_absorption_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/enclosed_space_absorption_example.pdf)

*Enclosed-space fiche (`ReverberationResult.report`), the per-band $A$/$T$ table and the boxed $T_\text{mid}$.*

## References

- European Committee for Standardization. (2003). *Building acoustics —
  Estimation of acoustic performance of buildings from the performance of
  elements — Part 6: Sound absorption in enclosed spaces*
  (EN 12354-6:2003).
  [BSI Knowledge record (BS EN 12354-6:2003)](https://knowledge.bsigroup.com/products/building-acoustics-estimation-of-acoustic-performance-of-buildings-from-the-performance-of-elements-sound-absorption-in-enclosed-spaces).
  The Clause 4 model, its input-data rules and its validity limits.
- International Organization for Standardization. (2003). *Acoustics —
  Measurement of sound absorption in a reverberation room* (ISO 354:2003).
  [iso.org catalogue](https://www.iso.org/standard/34545.html).
  The laboratory measurement the surface and array coefficients come from.
- Kuttruff, H. (2016). *Room acoustics* (6th ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  The statistical reverberation theory the standard's formulae specialise.

## Standards

EN 12354-6:2003, *Building acoustics — Estimation of acoustic
performance of buildings from the performance of elements — Part 6: Sound
absorption in enclosed spaces*: the total equivalent absorption area
(clause 4.3, Formulae 1-4, Table 1) and the reverberation time (clause 4.4,
Formula 5), validated against the three worked cases of Annex E.
