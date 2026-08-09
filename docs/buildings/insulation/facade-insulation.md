← [Documentation index](../../README.md)

# Façade Sound Insulation

A façade is the one element of a building that faces the noise everybody
complains about: the road, the railway, the flight path. It is also the
element whose insulation gets asked for twice, once predicted and once
measured, which is why both halves live on this page. ISO 16283-3 measures the
finished envelope, referencing the receiving-room level to the level 2 m in
front of the façade under a 45° loudspeaker or under the road traffic itself.
EN 12354-3 predicts the same standardized level difference from the sound
reduction indices of the wall, the glazing and the air inlet before any of them
is installed, and EN 12354-4 turns the envelope round to radiate an indoor
source outwards. A closing section puts the measured and the predicted
$D_{2m,nT}$ side by side and shows what does, and does not, separate them. The
reference curves behind every single number live in
[Insulation Ratings (ISO 717)](insulation-ratings.md); the internal partitions
of the same building in
[Field Insulation Measurement (ISO 16283)](insulation-field.md) and
[Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md).

## Field measurement (ISO 16283-3)

The same source/receiver logic reaches the building **façade**, but now the
source is *outdoors*: a loudspeaker at 45° or the road traffic itself. Rather
than a level difference across an internal partition, ISO 16283-3 references the
receiving-room level $L_2$ to the level **2 m in front of the façade**
$L_{1,2m}$, giving the level difference $D_{2m}$ and, exactly as in the airborne
case, its standardized and normalized forms:

$$
D_{2m} = L_{1,2m} - L_2, \quad
D_{2m,nT} = D_{2m} + 10 \log_{10}\frac{T}{T_0}, \quad
D_{2m,n} = D_{2m} - 10 \log_{10}\frac{A}{A_0},
$$

with $T_0 = 0.5$ s, $A_0 = 10$ m² and $A = 0.16\ V/T$ (dwellings). When the
microphone sits **on the test element** (surface level $L_{1,s}$) the *element*
method also yields an apparent sound reduction index, carrying a fixed
angle-of-incidence correction: $-1.5$ dB for the 45° loudspeaker method,
$-3$ dB for the all-angle road-traffic method:

$$
R'_{45°} = L_{1,s} - L_2 + 10 \log_{10}\frac{S}{A} - 1.5, \qquad
R'_{tr,s} = L_{1,s} - L_2 + 10 \log_{10}\frac{S}{A} - 3.
$$

The façade quantity is airborne, so its single-number rating uses the
**ISO 717-1** reference curve through `weighted_rating` unchanged (Annex F).

```python
import numpy as np
from phonometry import building

# Outdoor level 2 m in front of the façade, receiving-room level and T per
# one-third-octave band; surface_level is the microphone on the test element.
l1_2m = np.full(16, 75.0)                              # L1,2m outdoors
l2 = np.full(16, 33.0)                                 # receiving-room L2
t2 = np.full(16, 0.5)                                  # receiving-room T (s)

fac = building.facade_insulation(l1_2m, l2, t2, volume=50.0, area=11.5,
                        surface_level=np.full(16, 78.0), method="loudspeaker")
print(round(float(fac.d_2m[0]), 1))                    # 42.0  D2m = L1,2m - L2
print(round(float(fac.d_2m_nt[0]), 1))                 # 42.0  (= D2m since T = T0)
print(round(float(fac.d_2m_n[0]), 1))                  # 40.0  normalized to A0 = 10 m^2
print(round(float(fac.r_prime[0]), 1))                 # 42.1  R'45deg (loudspeaker, -1.5 dB)

# The road-traffic element method carries the -3 dB all-angle correction instead
tr = building.facade_insulation(l1_2m, l2, t2, volume=50.0, area=11.5,
                       surface_level=np.full(16, 78.0), method="road_traffic")
print(round(float(tr.r_prime[0]), 1))                  # 40.6  R'tr,s (traffic, -3 dB)

# The façade quantity is airborne: rate D2m,nT with the ISO 717-1 engine
print(building.weighted_rating(fac.d_2m_nt).rating)             # 42  Dls,2m,nT,w

fac.plot()   # per-band D2m,nT with D2m, D2m,n and R' overlaid (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/facade_field_insulation_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/facade_field_insulation.svg" alt="Field facade insulation of a dwelling under the 45-degree loudspeaker method: the standardized D2m,nT, the raw D2m, the normalized D2m,n and the apparent reduction index R'45 per one-third-octave band, with the Dls,2m,nT,w rating annotated" width="80%"></picture>

*The four façade quantities of one measurement: the raw $D_{2m}$, its
standardized and normalized forms, and the element $R'_{45°}$ carrying the
−1.5 dB angle-of-incidence correction. The rating box reads the single
number $D_{ls,2m,nT,w}$ obtained by feeding $D_{2m,nT}$ to the ISO 717-1
engine.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# A dwelling façade, 45° loudspeaker method: outdoor level 2 m in front,
# receiving-room level and T per one-third-octave band.
bands = np.array([100, 125, 160, 200, 250, 315, 400, 500,
                  630, 800, 1000, 1250, 1600, 2000, 2500, 3150], float)
l1_2m = np.array([76.0, 77.0, 78.0, 78.5, 79.0, 79.0, 79.0, 79.0,
                  78.5, 78.0, 77.5, 77.0, 76.5, 76.0, 75.0, 74.0])
d2m = np.array([24.0, 25.5, 27.0, 28.5, 30.0, 31.5, 33.0, 34.5,
                36.0, 37.0, 38.0, 38.5, 39.0, 39.0, 38.5, 38.0])
t2 = np.array([0.65, 0.62, 0.58, 0.55, 0.52, 0.50, 0.49, 0.48,
               0.47, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42])
fac = building.facade_insulation(l1_2m, l1_2m - d2m, t2, volume=32.0,
                                 area=10.8, surface_level=l1_2m + 3.0,
                                 method="loudspeaker", frequencies=bands)

# One line — D2m,nT with D2m, D2m,n and R' overlaid per band:
fac.plot()
plt.show()

# By hand, from the result's fields:
w = building.weighted_rating(fac.d_2m_nt)
fig, ax = plt.subplots()
ax.semilogx(bands, fac.d_2m_nt, "-s", label="D2m,nT (standardized)")
ax.semilogx(bands, fac.d_2m, "--o", label="D2m")
ax.semilogx(bands, fac.d_2m_n, ":", label="D2m,n (normalized)")
ax.semilogx(bands, fac.r_prime, "-.", label="R'45°")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Level difference / reduction index [dB]")
ax.set_title(f"Dls,2m,nT,w = {w.rating} dB  (C={w.c:+d}; Ctr={w.ctr:+d})")
ax.legend()
plt.show()
```

</details>

`surface_level`, `area` and `volume` are all optional: with only `l1_2m`, `l2`
and `t2` the function returns `d_2m` and `d_2m_nt`; add `volume` for `d_2m_n`;
add `surface_level` **and** `area` **and** `volume` for `r_prime`. Positions are
energy-averaged with the surface-level formula (Clause 9.5.1); band levels are
assumed already corrected for background noise.

### `facade_insulation()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `l1_2m` | 1D or 2D array | dB | one/band, or `(positions, bands)` | Level 2 m in front of the façade $L_{1,2m}$ |
| `l2` | 1D or 2D array | dB | same band count | Receiving-room levels |
| `t2` | 1D array | s | > 0, one per band | Receiving-room reverberation time |
| `area` | float, optional | m² | > 0, with `surface_level`, `volume` | Test-element area $S$ (enables $R'$) |
| `volume` | float, optional | m³ | > 0 | Receiving-room $V$ (enables $D_{2m,n}$; required for $R'$) |
| `surface_level` | 1D/2D array, optional | dB | same band count | Surface level $L_{1,s}$ on the element (enables $R'$) |
| `method` | str | — | `'loudspeaker'` (−1.5 dB) / `'road_traffic'` (−3 dB) | Angle-of-incidence correction of $R'$ |
| `t0` | float | s | default `0.5` | Reference reverberation time $T_0$ |
| `frequencies` | 1D array, optional | Hz | — | Band centres carried on the result for plotting |

`facade_insulation()` returns a `FacadeInsulationResult` (`d_2m`, `d_2m_nt`,
`d_2m_n` or `None`, `r_prime` or `None`, `frequencies`); feed any 16-band façade
quantity to `weighted_rating` for its ISO 717-1 single number.

### ISO 16283-3 field façade report (`.report()`)

`FacadeInsulationResult.report(path)` writes the one-page ISO 16283-3 field
façade test report: the standard-basis line, an optional metadata header, the
one-third-octave table beside the measured-versus-shifted-reference curve, the
boxed field rating $D_{2m,nT,w}\ (C;\ C_{tr})$, the engineering-method
statement, an optional requirement verdict (a level difference passes at or above it) and a
footer. `quantity="d_2m_nt"` (default) reports the standardized level
difference; `"d_2m_n"` the normalized one; `"r_prime"` the apparent sound
reduction index $R'_{45}$. `verbose=True`, `metadata`, `language="es"` and the
`phonometry[report]` extra behave exactly as in the fiches above.

```python
fac.report("D2mnTw_facade.pdf")                        # D2m,nT,w (C; Ctr)
fac.report("Rp45_facade.pdf", quantity="r_prime")      # R'45,w (C; Ctr)
```

[![Field facade ISO 16283-3 example report: metadata header, one-third-octave D2m,nT table beside the measured-versus-shifted-reference curve, boxed D2m,nT,w (C; Ctr), the engineering-method statement and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso16283_facade_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso16283_facade_example.pdf)

*Field façade fiche (`FacadeInsulationResult.report`), $D_{2m,nT,w}\ (C;\ C_{tr})$.*

## Prediction from the elements (EN 12354-3)

Parts 3 and 4 predict the two directions across the building envelope, both from
the same energy summation of the element **transmission factors**
$\tau = 10^{-R/10}$, area-weighted by $S_i/S$ (a small element or air path enters
through its element-normalized level difference $D_{n,e}$ with the reference area
$A_0 = 10\ \text{m}^2$):

$$
R' = -10 \log_{10}\!\Big( \sum_i \tfrac{S_i}{S}\,10^{-R_i/10}
                          + \sum_k \tfrac{A_0}{S}\,10^{-D_{n,e,k}/10} \Big).
$$

**Part 3: outdoor → indoor.** From $R'$ (Formula 10) follow the loudspeaker- and
traffic-referenced indices $R_{45} = R'+1$ and $R_{tr,s} = R'$, and the primary
output, the standardized level difference at 2 m (Formula 13)

$$
D_{2m,nT} = R' + \Delta L_{fs} + 10 \log_{10}\frac{V}{6\,T_0\,S}, \qquad T_0 = 0.5\ \text{s},
$$

with the façade-shape term $\Delta L_{fs}$ (Annex C; 0 dB for a flat reflecting
façade; `facade_shape_level_difference` looks it up from the Figure C.2 table
for galleries, balconies and terraces, interpolating over the underside
absorption $\alpha_w$). Single-number ratings reuse EN ISO 717-1
(`weighted_rating`).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/facade_prediction_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/facade_prediction.svg" alt="Per-element partial sound reduction indices and the resulting façade apparent reduction R' and standardized level difference D2m,nT for the EN 12354-3 Annex F worked example, the air inlet limiting the low bands" width="80%"></picture>

```python
from phonometry import building

# EN 12354-3 Annex F: an 11.3 m² façade (V = 50 m³, flat so ΔLfs = 0) of a double
# wall, a window, a small skylight and an acoustically-treated air inlet (a Dn,e
# element).
elements = [
    building.FacadeElement("wall",     area=6.0, r=[41, 46, 52, 58, 64]),   # octave 125-2000
    building.FacadeElement("window",   area=4.5, r=[23, 22, 30, 36, 37]),
    building.FacadeElement("skylight", area=0.5, r=[24, 27, 30, 33, 30]),
    building.FacadeElement("air inlet", dn_e=[28, 23, 25, 38, 44]),         # small element
]
fac = building.facade_sound_reduction(elements, area=11.3, volume=50.0,
                             frequencies=[125, 250, 500, 1000, 2000], bands="octave")
print(fac.r_tr_s_w, fac.c_tr, fac.d_2m_nt_w)   # 31 -3 33  (R'tr,s,w / Ctr / D2m,nT,w)

fac.plot()   # per-element partial indices with R' and D2m,nT overlaid (needs matplotlib)
```

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from phonometry import building

# EN 12354-3 Annex F: an 11.3 m² façade (V = 50 m³, flat so ΔLfs = 0) of a double
# wall, a window, a small skylight and an acoustically-treated air inlet (a Dn,e
# element).
elements = [
    building.FacadeElement("wall",     area=6.0, r=[41, 46, 52, 58, 64]),   # octave 125-2000
    building.FacadeElement("window",   area=4.5, r=[23, 22, 30, 36, 37]),
    building.FacadeElement("skylight", area=0.5, r=[24, 27, 30, 33, 30]),
    building.FacadeElement("air inlet", dn_e=[28, 23, 25, 38, 44]),         # small element
]
fac = building.facade_sound_reduction(elements, area=11.3, volume=50.0,
                             frequencies=[125, 250, 500, 1000, 2000], bands="octave")

x = np.arange(5)
fig, ax = plt.subplots(figsize=(9, 5.5))
for name, rp in fac.element_r.items():
    ax.plot(x, rp, "--", alpha=0.6, marker=".", label=f"Rp — {name}")
ax.plot(x, fac.r_prime, "k-", lw=2.5, marker="o", label="R′ (façade)")
ax.plot(x, fac.d_2m_nt, lw=2, marker="s", label="D2m,nT")
ax.set_xticks(x); ax.set_xticklabels([125, 250, 500, 1000, 2000])
ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Index / level difference [dB]")
ax.set_title("EN 12354-3 façade sound insulation (Annex F)")
ax.legend(ncol=2); ax.grid(alpha=0.4)
fig.tight_layout(); plt.show()
```

</details>

The composite is easier to reason about drawn as areas. `plot_facade_elements`
tiles the elevation with every element's drawn area equal to its real area
(here a 6 m² masonry wall, a 1.5 m² window and its 0.3 m² roller shutter box),
and a prediction that retained its `elements` redraws its own façade with
`fac.plot_geometry()`.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/facade_elevation_geometry_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/facade_elevation_geometry.svg" alt="To-scale elevation of a composite facade: a hatched 6 m2 masonry wall, a 1.5 m2 window and a narrow 0.3 m2 roller shutter box drawn as tiles of a 3.95 m by 1.97 m facade whose drawn areas equal their real areas, each tile labelled with its area and the overall width and height dimensioned" width="88%"></picture>

*The areas the energy sum weighs, to scale: the window holds a quarter of the
wall's area but each square metre of it transmits a hundred times more, so the
small tiles decide $R'$.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import building

# The prediction's composite drawn as areas: 6 m2 of wall, a 1.5 m2 window
# and its 0.3 m2 roller shutter box.
elements = [
    building.FacadeElement("Masonry wall", area=6.0, r=[50.0] * 5),
    building.FacadeElement("Window", area=1.5, r=[30.0] * 5),
    building.FacadeElement("Roller shutter box", area=0.3, r=[22.0] * 5),
]
building.plot_facade_elements(elements)
plt.show()

# A prediction retains its elements, so it redraws its own elevation:
#   fac = building.facade_sound_reduction(elements, area=7.8, volume=50.0)
#   fac.plot_geometry()
```

</details>

The façade prediction also writes a one-page **prediction** report through a
`report(path)` method, the same layout as the airborne and impact prediction
fiches. `FacadePredictionResult.report()` renders the façade-element table (each
element's weighted partial index $R_{p,w}$) beside the per-element / $R'$ /
$D_{2m,nT}$ plot, the boxed predicted $D_{2m,nT,w}$ (with $R'_{tr,s,w}$ and
$C_{tr}$), the prediction statement and, when a `requirement` is supplied, a
PASS/FAIL verdict
(the level difference passes at or above it). `verbose=True` annexes each
element's share of the transmitted sound energy, which singles out the limiting
element (the air inlet here, not the wall). The report needs the ISO 717-1
single-number ratings, so build the result on the 5 octave or 16 one-third-octave
bands. The applicable `ReportMetadata` fields describe the predicted situation:
`specimen` (the façade element set), `area` (the exposed façade area), the
receiving-room `receiving_volume`, the outdoor/traffic situation in `test_room`,
plus the calculator / laboratory identity fields (`client`, `manufacturer`,
`measurement_standard`, `laboratory`, `operator`, `report_id`, `test_date`), a
free-text façade-shape and model summary in `notes` and the target
$D_{2m,nT,w}$ in `requirement`. Metadata, `language="es"` and the `phonometry[report]` extra
behave as in the measurement fiches.

```python
from phonometry import building, ReportMetadata

# EN 12354-3 Annex F facade -> D2m,nT,w = 33 dB (R'tr,s,w = 31, Ctr = -3).
elements = [
    building.FacadeElement("Masonry wall", area=6.0, r=[41, 46, 52, 58, 64]),
    building.FacadeElement("Glazing",      area=4.5, r=[23, 22, 30, 36, 37]),
    building.FacadeElement("Roof light",   area=0.5, r=[24, 27, 30, 33, 30]),
    building.FacadeElement("Air inlet", dn_e=[28, 23, 25, 38, 44]),
]
fac = building.facade_sound_reduction(elements, area=11.3, volume=50.0,
                             frequencies=[125, 250, 500, 1000, 2000], bands="octave")
fac.report("D2mnT_prediction.pdf", metadata=ReportMetadata(
    specimen="Masonry wall + window + roof light + air inlet", area=11.3,
    receiving_volume=50.0, requirement=30.0,
    notes="Flat facade, ΔLfs = 0 dB (Annex C)."))          # D2m,nT,w = 33 dB
```

[![Predicted facade EN 12354-3 example report: metadata header, the facade-element table beside the per-element partial-index and R' / D2m,nT chart, boxed predicted D2m,nT,w = 33 dB, the prediction statement and a PASS verdict against the 30 dB requirement](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso12354_facade_prediction_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso12354_facade_prediction_example.pdf)

## Indoor sound radiated outdoors (EN 12354-4)

**Part 4: indoor → outdoor.** The sound power level radiated by a segment
(Formula 2) is $L_W = L_{p,in} + C_d - R' + 10 \log_{10}(S/S_0)$ with $S_0 = 1$ m²
and the inside-field diffusivity term $C_d$ (Annex B; −6 dB ideal diffuse, −5 dB
average industrial). Openings are elements whose "R" is the silencer insertion
loss (a bare opening is 0 dB). The exterior level follows from the simplified
Annex E attenuation $A_{tot}$ of a finite radiating side, $L_p = L_W - A_{tot}$.

```python
from phonometry import building

# EN 12354-4 Annex G, side 1: a 10×20 m concrete wall segment with a 6×4 m
# industrial door, inside level Lp,in, Cd = -5 dB. The 40 dB cap on R' is an
# Annex G example footnote (field leaks), not part of Formula (2)/(3): pass it
# explicitly to reproduce Annex G; by default no cap is applied.
bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
seg = building.radiated_sound_power(
    [building.FacadeElement("wall", area=176.0, r=[32, 36, 36, 33, 39, 49, 57, 63]),
     building.FacadeElement("door", area=24.0,  r=[21, 23, 28, 30, 30, 30, 30, 30])],
    lp_in=[70, 74, 76, 72, 70, 67, 62, 57], area=200.0, c_d=-5.0,
    r_prime_cap=40.0, octave_bands=bands)
print(round(seg.l_w[0], 1), round(seg.l_w[1], 1))     # 59.8 61.2  (LW at 63/125 Hz)

# Exterior level 5 m in front of the centre of the 60×10 m side (LWA = 62.9 dB(A)).
a_tot = building.outdoor_attenuation(width=60.0, height=10.0, distance=5.0)
print(round(a_tot, 1), round(building.outdoor_level(62.9, a_tot), 1))   # 26.3 36.6

seg.plot()   # radiated LW per octave with the A-weighted LWA line (needs matplotlib)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/radiated_power_outdoor_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/radiated_power_outdoor.svg" alt="Radiated sound power level per octave band of the EN 12354-4 Annex G wall segment with an industrial door, with the A-weighted single number drawn as a dashed line across the bars" width="80%"></picture>

*The Annex G side-1 segment: the wall dominates the area but the door's
weaker $R$ carries the radiated power, so the octave spectrum stays flat
where the wall alone would fall. The dashed line is the A-weighted single
number $L_{WA}$ formed from the octave bands.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# EN 12354-4 Annex G, side 1: a 10×20 m concrete wall segment with a 6×4 m
# industrial door, inside level Lp,in, Cd = -5 dB.
bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
seg = building.radiated_sound_power(
    [building.FacadeElement("wall", area=176.0, r=[32, 36, 36, 33, 39, 49, 57, 63]),
     building.FacadeElement("door", area=24.0,  r=[21, 23, 28, 30, 30, 30, 30, 30])],
    lp_in=[70, 74, 76, 72, 70, 67, 62, 57], area=200.0, c_d=-5.0,
    r_prime_cap=40.0, octave_bands=bands)

# One line — LW per octave with the A-weighted LWA line:
seg.plot()
plt.show()

# By hand, from the result's fields:
x = np.arange(len(bands))
fig, ax = plt.subplots()
ax.bar(x, seg.l_w, label="radiated LW per octave")
ax.axhline(seg.l_w_dba, ls="--", color="tab:red",
           label=f"LWA = {seg.l_w_dba:.1f} dB(A)")
ax.set_xticks(x, [str(b) for b in bands])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Radiated sound power level [dB re 1 pW]")
ax.set_title("EN 12354-4 radiated sound power (Annex G)")
ax.legend()
plt.show()
```

</details>

> **Worked-example note.** The 2000 worked examples carry small internal rounding
> inconsistencies at the higher octave bands (Part 3's printed $R'$ disagrees with
> its own per-element partial indices at 1 k/2 k; Part 4's $R'$ rows above 500 Hz
> disagree with its Table G.2 inputs). The implementation is faithful to the
> formulas: it reproduces the low bands, every single-number rating and the whole
> Annex E propagation exactly.

### `FacadeElement` / `facade_sound_reduction()` / `radiated_sound_power()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `FacadeElement.area` | float | m² | > 0 for `r` / `insertion_loss` | Element area $S_i$ (ignored for `dn_e`) |
| `FacadeElement.r` / `dn_e` / `insertion_loss` | float or seq | dB | give exactly one | Area element $R_i$ / small-element $D_{n,e}$ / opening insertion loss |
| `facade_sound_reduction(area)` | float | m² | > 0 | Total façade area $S$ |
| `facade_sound_reduction(volume)` | float | m³ | > 0 | Receiving-room volume $V$ (Formula 13) |
| `facade_sound_reduction(delta_l_fs)` | float | dB | default `0` | Façade-shape term $\Delta L_{fs}$ (Annex C; look it up with `facade_shape_level_difference`) |
| `radiated_sound_power(lp_in)` | float or seq | dB | — | Inside level $L_{p,in}$ per band |
| `radiated_sound_power(c_d)` | float | dB | default `-6` | Diffusivity term $C_d$ (Annex B) |
| `radiated_sound_power(r_prime_cap)` | float | dB | default `None` (off) | Optional field cap on $R'$, an Annex G example footnote (it uses 40 dB), not part of Formula (2)/(3) |
| `radiated_sound_power(octave_bands)` | seq of int | Hz | default `None` | Octave centres matching the bands; enables the A-weighted $L_{WA}$ |
| `facade_sound_reduction(frequencies)` | seq | Hz | default `None`; length = band count | Band centres carried on the result for plotting |
| `outdoor_attenuation(width, height, distance)` | float | m | > 0 | Finite radiating side and reception distance (Annex E) |
| `outdoor_level(l_w, attenuation)` | float or seq | dB | broadcast-compatible | Exterior $L_p$ from one or more sides (Formula E.1) |

`facade_sound_reduction()` returns a `FacadePredictionResult` (`r_prime`, `r_45`,
`r_tr_s`, `d_2m_nt`, `element_r`, and the `r_tr_s_w` / `d_2m_nt_w` / `c_tr` single
numbers); `radiated_sound_power()` a `RadiatedPowerResult` (`l_w`, `r_prime`,
`l_w_dba`). Both expose `.plot()`.

## Measurement against prediction

The two halves of this guide describe the same wall from opposite ends of its
life: EN 12354-3 before it exists, ISO 16283-3 once it does. They are worth
comparing because they end on the same quantity, the standardized level
difference at 2 m, and because the geometry term that carries the prediction
from an index to a level difference is the measurement's own normalisation
written another way. Start from the measured definition, replace $D_{2m}$ by
the apparent index it implies and substitute the Sabine absorption area
$A = 0.16\ V/T$:

$$
D_{2m,nT} = D_{2m} + 10 \log_{10}\frac{T}{T_0}
          = R' + 10 \log_{10}\frac{A}{S} + 10 \log_{10}\frac{T}{T_0}
          = R' + 10 \log_{10}\frac{0.16\ V}{S\ T_0}.
$$

The reverberation time cancels: what survives is the room volume against the
façade area. EN 12354-3 Formula (13) writes that same term as
$10 \log_{10} V/(6\,T_0\,S)$, which is the Sabine constant carried as
$1/6 = 0.167$ instead of the $0.16$ of the measurement standard. The two
therefore differ by a fixed $10 \log_{10}(0.167/0.16) = 0.18$ dB in every
band, of any room, and by nothing else. (The 2017 revision of the
prediction standard harmonised its constant to 0.16, so the gap is
specific to the EN 12354-3:2000 edition the module implements.)

That is small enough to check numerically. Take the Annex F façade predicted
above, put it in a receiving room of 50 m³ with $T = T_0$, and reconstruct the
receiving-room level a measurement would have found:

```python
import numpy as np
from phonometry import building

# The Annex F façade predicted above, now "measured" in a receiving room of
# 50 m3 whose reverberation time is exactly T0 = 0.5 s.
elements = [
    building.FacadeElement("wall",     area=6.0, r=[41, 46, 52, 58, 64]),
    building.FacadeElement("window",   area=4.5, r=[23, 22, 30, 36, 37]),
    building.FacadeElement("skylight", area=0.5, r=[24, 27, 30, 33, 30]),
    building.FacadeElement("air inlet", dn_e=[28, 23, 25, 38, 44]),
]
volume, area = 50.0, 11.3
fac = building.facade_sound_reduction(elements, area=area, volume=volume,
                             frequencies=[125, 250, 500, 1000, 2000], bands="octave")

# What the field measurement would read: A = 0.16 V / T is the Sabine area of
# ISO 16283-3, and R' - 10 lg(S/A) is the level difference it implies.
t2 = np.full(5, 0.5)
l1_2m = np.full(5, 75.0)
l2 = l1_2m - (fac.r_prime - 10 * np.log10(area / (0.16 * volume / t2)))
meas = building.facade_insulation(l1_2m, l2, t2, volume=volume)

print(np.round(meas.d_2m_nt - fac.d_2m_nt, 2))   # [-0.18 -0.18 -0.18 -0.18 -0.18]
print(building.weighted_rating(meas.d_2m_nt).rating, fac.d_2m_nt_w)   # 33 33
```

The two curves run parallel to the second decimal and both rate at
$D_{2m,nT,w} = 33$ dB. Everything that separates a real prediction from a real
measurement therefore sits in the inputs, not in the formulas:

* **The element indices.** The prediction consumes laboratory $R$ values
  measured with flanking suppressed and with the element mounted as the
  laboratory mounts it. The built façade adds its perimeter seals, its
  roller-shutter box, the joint between frame and reveal and whatever the site
  did to them.
* **The shape term.** $\Delta L_{fs}$ is 0 dB only for a flat reflecting
  façade. A balcony, a gallery or a terrace changes the field at the 2 m
  position, and a measurement takes that change as it is while a prediction
  has to look it up in Annex C.
* **The source.** ISO 16283-3 labels the loudspeaker result $D_{ls,2m,nT}$
  precisely because a loudspeaker at 45° is not road traffic; its element
  index carries a −1.5 dB angle correction where the all-angle traffic method
  carries −3 dB. EN 12354-3 predicts the traffic-referenced $R_{tr,s} = R'$
  directly and offers $R_{45} = R' + 1$ for the loudspeaker comparison.
* **The receiving room.** The prediction assumes the design volume and a
  diffuse receiving field. The measurement takes the room furnished, and the
  diffuse assumption behind $A = 0.16\ V/T$ weakens exactly in the small rooms
  and low bands where façade requirements bite.

The practical reading: size the glazing and the air inlets with the
prediction, accept the finished façade with the measurement, and do not treat
a couple of decibels between the two as an error in either. A prediction built
on laboratory indices and a field measurement of the finished envelope each
carry an uncertainty of that order on their own; the [ISO 12999-1 uncertainty section](insulation-field.md#measurement-uncertainty-iso-12999-1) puts a number on the
measurement half. And when the two disagree by much more than that, the
per-element share of the transmitted energy in the prediction report is the
place to look first: on this façade it is the air inlet, not the wall, that
decides the result.

## References

- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  The transmission physics of the envelope behind both the measured and the
  predicted façade quantities.
- International Organization for Standardization. (2016). *Acoustics — Field
  measurement of sound insulation in buildings and of building elements —
  Part 3: Façade sound insulation* (ISO 16283-3:2016).
  [iso.org catalogue](https://www.iso.org/standard/59748.html).
  The field façade element and global methods, with the angle-of-incidence
  corrections.
- European Committee for Standardization. (2000). *Building acoustics —
  Estimation of acoustic performance of buildings from the performance of
  elements — Part 3: Airborne sound insulation against outdoor sound*
  (EN 12354-3:2000).
  [BSI Knowledge catalogue](https://knowledge.bsigroup.com/products/building-acoustics-estimation-of-acoustic-performance-in-buildings-from-the-performance-of-elements-airborne-sound-insulation-against-outdoor-sound).
  The façade sound insulation prediction (Annex F worked example).
- European Committee for Standardization. (2000). *Building acoustics —
  Estimation of acoustic performance of buildings from the performance of
  elements — Part 4: Transmission of indoor sound to the outside*
  (EN 12354-4:2000).
  [BSI Knowledge catalogue](https://knowledge.bsigroup.com/products/building-acoustics-estimation-of-acoustic-performance-in-buildings-from-the-performance-of-elements-transmission-of-indoor-sound-to-the-outside).
  The outdoor-radiation prediction (Annex G worked example).

## Standards

ISO 16283-3:2016, which defines the field façade quantities $D_{2m}$,
$D_{2m,nT}$, $D_{2m,n}$, $R'_{45°}$ and $R'_{tr,s}$ of the loudspeaker and
road-traffic methods and their test report; EN 12354-3:2000, which predicts
the same $D_{2m,nT}$ from the element indices with the Annex C façade-shape
term (Annex F worked example); and EN 12354-4:2000, which predicts the sound
power radiated outwards by a building side and its exterior level (Annex G
worked example). The single-number ratings come from ISO 717-1 (Annex F of
ISO 16283-3 and EN ISO 717-1 for the prediction).

**Not covered.** The band levels handed to `facade_insulation` are assumed
already corrected for background noise, and ISO 16283-3's procedural
requirements are documented but never checked: nothing verifies the 45° ± 5°
incidence, the 5 m / 7 m slant distances, the 3 → 10 microphone escalation and
its recess special case, the 50 pass-bys and the simultaneity of a traffic
measurement, or that the method chosen matches the question asked. The
low-frequency corner procedure of Clause 7 — the façade counterpart of the
**ISO 16283-1** procedure, triggered by the same 25 m³ receiving-room threshold
— is not implemented, and neither are the railway and aircraft methods of
Annex E. The EN 12354-3 and EN 12354-4 worked examples of the 2000 editions
carry small internal rounding inconsistencies in the higher octave bands; the
implementation follows the formulae rather than the printed rows.

## See also

- [Field Insulation Measurement (ISO 16283)](insulation-field.md): the
  airborne and impact field measurements of the internal partitions, and the
  ISO 12999-1 uncertainty that qualifies every field rating.
- [Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md): the
  airborne and impact flanking models of EN 12354-1/2 that share the
  transmission-factor summation used here.
- [Insulation Ratings (ISO 717)](insulation-ratings.md): the reference-curve
  engine behind the $D_{2m,nT,w}$, $R'_{45,w}$ and $R'_{tr,s,w}$ single
  numbers.
- [Sound Insulation Survey Method (ISO 10052)](insulation-survey.md): the
  octave-band façade quantity of the quick survey method.
- [Predicting Panel Sound Insulation](../design/panel-sound-insulation.md): the mass
  law, coincidence dip and aperture models behind the element $R$ values the
  prediction consumes.
- [Outdoor Sound Propagation](../../environment/propagation/outdoor-propagation.md): the propagation from
  the source to the 2 m position in front of the façade, and onwards from the
  radiating side of EN 12354-4.
- [Theory](../../reference/theory/rooms-buildings.md): the reference-curve derivation behind
  the weighted single-number ratings.
- API reference: [`building.prediction.facade`](https://jmrplens.github.io/phonometry/reference/api/building/facade/) and [`building.measurement.insulation`](https://jmrplens.github.io/phonometry/reference/api/building/insulation/).
