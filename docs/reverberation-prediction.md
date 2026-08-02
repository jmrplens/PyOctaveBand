← [Documentation index](README.md)

# Reverberation-time prediction (Sabine · Eyring · Fitzroy · Arau-Puchades)

The **reverberation time** $T$, the time for the sound-energy level to fall by
60 dB after the source stops, is predicted here from a room's **volume**,
**boundary areas** and the **sound-absorption coefficients** of its surfaces,
through the classical statistical-acoustics formulae. This is the design-stage
counterpart of the *measured* reverberation time of
[Room Acoustics](room-acoustics.md) (ISO 3382) and complements the EN 12354-6
model of [Sound absorption in enclosed spaces](enclosed-space-absorption.md),
which specialises the same physics to that standard's Clause 4.

phonometry offers five models, ordered by how much they account for a
**non-uniform** absorption distribution:

| Model | Absorption term in $T = k\,V / (\text{term} + 4mV)$ | Best for |
|:---|:---|:---|
| **Sabine** | $A = \sum_i S_i\alpha_i$ | low, uniform absorption |
| **Eyring** (Norris-Eyring) | $-S\ln(1-\bar\alpha)$ | strong, uniform absorption |
| **Millington-Sette** | $-\sum_i S_i\ln(1-\alpha_i)$ | a few very absorptive surfaces |
| **Fitzroy** | area-weighted **arithmetic** mean of three axial Eyring times | anisotropic rooms |
| **Arau-Puchades** | area-weighted **geometric** mean of the same three | anisotropic rooms (author-preferred) |

with the Sabine constant $k = 24\ln 10 / c_0$ (so $k = 0.161$ for
$c_0 = 343\ \mathrm{m/s}$) and the air-absorption term $4mV$.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/reverberation_models_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/reverberation_models.svg" alt="Reverberation time per octave band for a 10 by 7 by 3.5 metre room with an absorptive floor and ceiling but hard walls, computed by five models. Fitzroy gives the longest times, Sabine and Eyring the mid-range, Millington-Sette the shortest, and Arau-Puchades sits between Eyring and Sabine" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import environment, room

# A 10 x 7 x 3.5 m room: hard end walls, lightly treated side walls and a
# very absorptive floor/ceiling pair (carpet plus an acoustic ceiling).
bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
alpha_x = [0.06, 0.07, 0.08, 0.09, 0.10, 0.10]
alpha_y = [0.12, 0.14, 0.16, 0.18, 0.20, 0.20]
alpha_z = [0.30, 0.50, 0.65, 0.78, 0.82, 0.80]
m = environment.air_attenuation_m(bands, 20.0, 50.0)   # air at 20 C / 50 % RH
res = room.reverberation_time_models((10.0, 7.0, 3.5),
                                     (alpha_x, alpha_y, alpha_z),
                                     air_attenuation=m, frequencies=bands)
res.plot()   # the five model curves per band
plt.show()
```

</details>

## 1. Sabine, Eyring and Millington-Sette

The three statistical models take the room volume and a list of
`(area, absorption_coefficient)` surfaces. **Sabine** is exact only for low,
uniform absorption; **Eyring** replaces the absorption area by
$-S\ln(1-\bar\alpha)$ and is correct where Sabine overestimates $T$ (a live
room with strong absorption); **Millington-Sette** sums the Eyring term surface
by surface, so a single perfectly absorbing surface drives $T$ to zero.

$$
T_{\text{Sab}} = \frac{k V}{\sum_i S_i\alpha_i}, \qquad
T_{\text{Eyr}} = \frac{k V}{-S\ln(1-\bar\alpha)}, \qquad
T_{\text{Mil}} = \frac{k V}{-\sum_i S_i\ln(1-\alpha_i)}.
$$

```python
from phonometry import room

# A shoebox 8 x 5 x 3 m (V = 120 m3, S = 158 m2), uniform alpha = 0.2.
surfaces = [(40.0, 0.2), (40.0, 0.2), (24.0, 0.2),
            (24.0, 0.2), (15.0, 0.2), (15.0, 0.2)]
print(round(room.sabine_reverberation_time(120.0, surfaces), 3))            # 0.612 s
print(round(room.eyring_reverberation_time(120.0, surfaces), 3))            # 0.548 s
print(round(room.millington_sette_reverberation_time(120.0, surfaces), 3))  # 0.548 s
```

For a **uniform** distribution Eyring and Millington-Sette coincide, and both
fall below Sabine; Sabine's over-estimate at high absorption is the reason
Eyring exists. As $\alpha \to 0$, $-S\ln(1-\bar\alpha) \to \sum_i S_i\alpha_i$
and Eyring reduces to Sabine. Air absorption enters every model through the
power-attenuation coefficient $m$ (in neper per metre, from the ISO 9613-1
[atmospheric absorption](outdoor-propagation.md)):

```python
from phonometry import environment, room

m = environment.air_attenuation_m(2000.0, temperature=20.0, relative_humidity=50.0)
surfaces = [(40.0, 0.3), (40.0, 0.3), (24.0, 0.3),
            (24.0, 0.3), (15.0, 0.3), (15.0, 0.3)]
print(round(room.eyring_reverberation_time(120.0, surfaces, air_attenuation=m), 3))
```

Every statistical model also assumes a **diffuse field**, and low
frequencies break that assumption first: below the Schroeder frequency the
room responds as a set of discrete modes, not as a reverberant mixture. The
2D FDTD simulation below drives a rigid 5 m by 3.5 m room exactly on its
(2,1) mode and then between two modes; the standing-wave pattern that
builds up on resonance is what Sabine and Eyring cannot see.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_room_modes_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_room_modes.gif" alt="Animation: a 2D FDTD simulation of a 5 by 3.5 metre room driven at the 84 Hz (2,1) mode and at an off-mode frequency; on resonance a standing-wave pattern with fixed nodal lines grows to dominate the RMS pressure map, off resonance the forced response stays weak and disorganised" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_room_modes.webm)

## 2. Fitzroy and Arau-Puchades (anisotropic rooms)

When the absorption is concentrated on one axis (a carpeted floor and an
acoustic ceiling against otherwise hard walls), a single mean $\bar\alpha$
misrepresents the field. **Fitzroy** and **Arau-Puchades** split a rectangular
(shoebox) room into the three pairs of opposing walls and combine the *axial*
Eyring reverberation times $T_i$ (each using the whole surface $S$ and the mean
absorption $\bar\alpha_i$ of the wall pair perpendicular to axis $i$):

$$
T_{\text{Fitz}} = \sum_i \frac{S_i}{S}\,T_i \quad(\text{arithmetic}), \qquad
T_{\text{Arau}} = \prod_i T_i^{\,S_i/S} \quad(\text{geometric}).
$$

```python
from phonometry import room

# 8 x 5 x 3 m room, absorptive x-wall pair (alpha 0.5), hard elsewhere (0.1).
dims = (8.0, 5.0, 3.0)
absorption = (0.5, 0.1, 0.1)   # mean alpha of the (x, y, z) wall pairs
print(round(room.arau_puchades_reverberation_time(dims, absorption), 3))  # 0.812 s
print(round(room.fitzroy_reverberation_time(dims, absorption), 3))        # 0.974 s
```

By the arithmetic-geometric-mean inequality the Arau-Puchades time never exceeds
the Fitzroy time; Fitzroy is known to over-predict when one wall pair is very
reflective, which is why Arau-Puchades recommends the geometric mean. Both
reduce exactly to Eyring for a uniform absorption distribution.

## 3. Comparing the five models per band

`reverberation_time_models` builds the six boundary surfaces of a rectangular
room from its dimensions and the three wall-pair mean absorptions, then
evaluates all five models on a common footing and returns a
`ReverberationModelResult` whose `.plot()` draws the figure above.

```python
from phonometry import room

# 10 x 7 x 3.5 m room, absorptive floor/ceiling against harder walls.
res = room.reverberation_time_models(
    (10.0, 7.0, 3.5),
    (
        [0.06, 0.07, 0.08, 0.09, 0.10, 0.10],   # x-pair: hard end walls
        [0.12, 0.14, 0.16, 0.18, 0.20, 0.20],   # y-pair: lightly treated walls
        [0.30, 0.50, 0.65, 0.78, 0.82, 0.80],   # z-pair: carpet + acoustic ceiling
    ),
    frequencies=[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0],
)
print(res.sabine.round(2))         # [0.74 0.47 0.37 0.31 0.3  0.3 ]
print(res.arau_puchades.round(2))  # [0.79 0.51 0.38 0.29 0.26 0.27]
print(res.fitzroy.round(2))        # [1.02 0.79 0.66 0.57 0.51 0.51]
res.plot()   # the five model curves per band (the figure above)
```

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import environment, room

m = environment.air_attenuation_m([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0], 20.0, 50.0)
room.reverberation_time_models(
    (10.0, 7.0, 3.5),
    (
        [0.06, 0.07, 0.08, 0.09, 0.10, 0.10],
        [0.12, 0.14, 0.16, 0.18, 0.20, 0.20],
        [0.30, 0.50, 0.65, 0.78, 0.82, 0.80],
    ),
    air_attenuation=m,
    frequencies=[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0],
).plot()
plt.show()
```

</details>

Sabine and Eyring are the two workhorses, and the per-band spread between
them is itself a diagnostic. The diagram runs the room of this section
through both, with the validity boundary every statistical formula shares.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_reverberation_prediction_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_reverberation_prediction.svg" alt="Block diagram of the reverberation-time prediction: a 10 by 7 by 3.5 metre room with 245 cubic metres, 259 square metres and a mean absorption rising from 0.21 at 125 hertz to 0.51 at 4 kilohertz feeds the Sabine and Eyring formulas, whose per-octave-band table runs from 0.74 to 0.30 seconds for Sabine and 0.66 to 0.22 seconds for Eyring, Eyring reading 11 to 29 percent shorter; a closing note bounds the domain of validity to a diffuse field, excluding bands below the Schroeder frequency, coupled volumes and corridor-like rooms" width="92%"></picture>

## 4. Choosing a model, and when every model fails

The five formulae are not rivals on a single axis of accuracy; each has a
domain of validity:

- **Sabine** is the tool for live rooms with low, reasonably even
  absorption (mean $\bar\alpha$ up to roughly 0.2): classrooms, halls,
  reverberation chambers. It is also the convention wired into measurement
  practice, because the ISO 354 absorption coefficient is *defined* through
  Sabine's formula, so feeding reverberation-room data back into Sabine is
  self-consistent even where the formula is strained. Its structural defect
  shows at high absorption: with $\alpha = 1$ on every surface (an opening
  in every direction) it still predicts a finite reverberation time.
- **Eyring** is the choice for evenly treated rooms with substantial
  absorption: studios, treated offices, listening rooms. It reaches
  $T = 0$ for total absorption, and its correction over Sabine grows with
  $\bar\alpha$ (about 10 % shorter at $\bar\alpha = 0.2$, 30 % at 0.5).
- **Millington-Sette** handles a mix of very absorptive and hard surfaces
  better than a single mean, but it is meant for measured, sub-unity
  coefficients: a single surface with $\alpha_i = 1$ drives the whole
  prediction to zero. Reverberation-room coefficients at or above 1.0 (a
  documented ISO 354 outcome, see the absorption section of
  [Room Acoustics](room-acoustics.md)) lie outside the domain of the
  logarithmic term, so phonometry enforces each formula's own domain:
  Sabine accepts such coefficients as supplied (its linear
  $A = \sum_i S_i\alpha_i$ stays finite); Eyring accepts them as long as
  the mean entering $\ln(1-\bar\alpha)$ stays below 1 (Fitzroy and
  Arau-Puchades take the wall-pair means themselves as inputs, so each
  must already be below 1); Millington-Sette rejects any coefficient at
  or above 1. To use Millington anyway, bringing such a
  coefficient into $[0, 1)$ is a modelling decision the formula does not
  prescribe: whatever adjustment you choose (limiting just below 1 is
  common), record it alongside the prediction.
- **Fitzroy** and **Arau-Puchades** target shoebox rooms whose absorption
  is concentrated on one axis, the typical office or dwelling with a soft
  floor and ceiling between hard walls. Arau's geometric mean tempers
  Fitzroy's known over-prediction when one wall pair is very reflective.

**When every formula fails.** All five inherit the same assumption: a
diffuse field, with sound arriving equally from all directions at every
point, that stays diffuse while it decays. The common breakages:

- **Below the Schroeder frequency** the band holds a handful of discrete
  modes (the animation in §1) and a statistical reverberation time is not
  defined at all; each mode decays at its own rate set by the wall
  impedances it actually touches.
- **Coupled volumes** (a hall with an open stage house, two rooms through a
  doorway) produce double-slope decays; no single $T$ exists, and the
  measured T20 and T30 disagree (the curvature diagnostic of
  [Room Acoustics](room-acoustics.md)).
- **Disproportionate rooms** (corridors, low flat halls) with the
  absorption on one surface pair keep a grazing sound field parallel to the
  hard surfaces that the absorber barely touches; the measured time can be
  up to twice any statistical prediction, the practical experience recorded
  in EN 12354-6 (see
  [Sound absorption in enclosed spaces](enclosed-space-absorption.md)).
- **Focusing geometries** (domes, curved rear walls) concentrate late
  energy instead of mixing it, producing position-dependent decays no
  single-number formula can represent.

Scattering objects restore the mixing the models assume: a furnished room
follows the statistical prediction distinctly better than the same room
bare, beyond what the furniture's own absorption area accounts for. In
practice, quote a *band* of predictions (Sabine and Eyring, or Fitzroy and
Arau-Puchades for axial cases) rather than a single value; where the models
spread, the room is telling you its field is not diffuse.

## 5. Prediction report (`.report()`)

`ReverberationModelResult.report(path)` renders a one-page PDF fiche of the
prediction: a basis line marking it a **design-stage prediction** by the five
statistical-acoustics models, an optional metadata header block (client, room,
description, room volume, total surface area, climate), a per-band table with
one reverberation-time column per model beside the model comparison plot
(`.plot()`), and the boxed mid-frequency reverberation time from Arau-Puchades
(the recommended model for a non-uniform absorption distribution) with the
per-model spread alongside. It is a prediction, not a measurement: the five
models bracket the reverberation time likely to occur, so no PASS/FAIL verdict
is emitted. A target reverberation time supplied through the metadata's
`requirement` field is printed as a reference line only, since a room
reverberation time is a target range rather than a strictly
higher/lower-is-better quantity. It uses the same `ReportMetadata` container
(documented under
[Insulation ratings](insulation-ratings.md#report-metadata-reportmetadata)) and
rendering engine as the other fiches; passing `metadata=None` produces a bare
prediction fiche. Rendering needs reportlab and, for the figure the fiche
embeds, matplotlib (`pip install "phonometry[report,plot]"`); only
`engine="reportlab"` is supported. The fiche renders in English by default; pass
`language="es"` for a Spanish fiche (translated fixed strings and a comma
decimal separator).

```python
from phonometry import reverberation_time_models, ReportMetadata

result = reverberation_time_models(
    (8.0, 5.0, 3.0),                       # a shoebox room, one treated wall pair
    ([0.10, 0.15, 0.30, 0.45, 0.55, 0.60], # treated wall pair, per octave band
     [0.08, 0.10, 0.12, 0.15, 0.18, 0.20], # side walls
     [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]),# floor/ceiling
    frequencies=[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0],
)
result.report(
    "reverberation_fiche.pdf",
    metadata=ReportMetadata(
        specimen="Classroom, one wall lined with a broadband absorber",
        test_room="Classroom C1",
        temperature=20.0, relative_humidity=50.0,
        laboratory="Phonometry Reference Laboratory",
        requirement=0.8,          # printed as a target reference line, no verdict
    ),
)                                 # the five-model table + the boxed T_mid
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![Reverberation-time prediction example report: a metadata header with the room volume and total surface area, the octave-band table with one reverberation-time column per model (Sabine, Eyring, Millington-Sette, Fitzroy and Arau-Puchades from 125 Hz to 4 kHz) beside the five-model comparison plot, the boxed mid-frequency reverberation time from Arau-Puchades with the per-model spread alongside, and a target reverberation-time reference line](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/reverberation_prediction_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/reverberation_prediction_example.pdf)

*Reverberation-time prediction fiche (`ReverberationModelResult.report`), the five-model table and the boxed $T_\text{mid}$.*

## References

- Sabine, W. C. (1922). *Collected papers on acoustics*. Harvard University
  Press. [Free scan at the Internet Archive](https://archive.org/details/collectedpaperso00sabi).
  The original reverberation experiments and the $T = 0.161\,V/A$ law of §1.
- Eyring, C. F. (1930). Reverberation time in "dead" rooms. *The Journal of
  the Acoustical Society of America*, 1(2A), 217-241.
  [doi:10.1121/1.1915175](https://doi.org/10.1121/1.1915175).
  The mean-free-path derivation behind the $-S\ln(1-\bar\alpha)$ term of §1.
- Millington, G. (1932). A modified formula for reverberation. *The Journal
  of the Acoustical Society of America*, 4(1), 69-82.
  [doi:10.1121/1.1915588](https://doi.org/10.1121/1.1915588).
  The per-surface logarithmic absorption term of §1.
- Fitzroy, D. (1959). Reverberation formula which seems to be more accurate
  with nonuniform distribution of absorption. *The Journal of the
  Acoustical Society of America*, 31(7), 893-897.
  [doi:10.1121/1.1907814](https://doi.org/10.1121/1.1907814).
  The axial split into three wall-pair decays of §2.
- Arau-Puchades, H. (1988). An improved reverberation formula. *Acustica*,
  65(4), 163-180.
  [Publisher record at Ingenta](https://www.ingentaconnect.com/content/dav/aaua/1988/00000065/00000004/art00003).
  The geometric-mean combination of §2 (its Formula 18).
- Kuttruff, H. (2016). *Room acoustics* (6th ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  The diffuse-field theory, its limits and the modern assessment of the
  classical formulae behind §4.
- Everest, F. A. (2001). *Master handbook of acoustics* (4th ed.).
  McGraw-Hill. ISBN 978-0-07-136097-5.
  [Open Library record](https://openlibrary.org/isbn/9780071360975).
  The Fig. 7-22 worked example the conformance suite reproduces.
- Carrión Isbert, A. (1998). *Diseño acústico de espacios arquitectónicos*.
  Edicions UPC. ISBN 978-84-8301-252-9.
  [Open Library record](https://openlibrary.org/books/OL23159935M).
  A Spanish-language textbook treatment of the reverberation models and
  their use in room design.

## Standards

The classical reverberation formulae predate the normative
world; they enter it through EN 12354-6:2003, whose Clause 4 model is a
Sabine calculation with object and air terms (see
[Sound absorption in enclosed spaces](enclosed-space-absorption.md)), and
through ISO 354:2003, which defines the measured absorption coefficient via
Sabine's formula. Air absorption follows ISO 9613-1:1993, *Acoustics —
Attenuation of sound during propagation outdoors — Part 1: Calculation of
the absorption of sound by the atmosphere* (see
[Outdoor propagation](outdoor-propagation.md)). The conformance suite is
anchored on a real worked example, Everest's Fig. 7-22 Example 1 (an
untreated 23.3 × 16 × 10 ft room), whose six printed Sabine reverberation
times the SI implementation reproduces to ≤ 0.02 s, reinforced by
hand-computed closed-form values and the model identities (every model
collapses to Eyring for uniform absorption; Eyring collapses to Sabine as
$\alpha \to 0$), which transitively carry that real-data anchor to the
whole family.
