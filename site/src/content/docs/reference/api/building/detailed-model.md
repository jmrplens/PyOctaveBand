---
title: "building.prediction.detailed_model"
description: "Detailed per-band building prediction (EN/ISO 12354-1/-2:2017)."
sidebar:
  label: "detailed_model"
---

Detailed per-band building prediction (EN/ISO 12354-1/-2:2017).

This is the **detailed model** of the building-prediction chain, the per-band
counterpart of the simplified single-number model implemented in
[`phonometry.building.prediction.global_model`](/phonometry/reference/api/building/global-model/). Where the simplified model
combines the weighted ratings of the elements (`Rw`, `ΔRw`, `Kij`) into a
single `R'w` / `L'n,w`, the detailed model carries every quantity through
the one-third-octave (or octave) bands, converts the laboratory element data to
their **in-situ** values, forms each transmission path per band and only then
rates the result through ISO 717. It is what a consultant runs when the element
spectra are known and the dominant path per band matters.

**Chain, airborne (ISO 12354-1:2017, Clause 4.2).**

1. Element data per band. For homogeneous elements the sound reduction index
   follows from the material properties with the Annex B model: the radiation
   factor for free bending waves `σ` (Formulae B.4 to B.6,
   [`bending_radiation_factor`](/phonometry/reference/api/building/detailed-model/#bending_radiation_factor)), the radiation factor for forced waves `σf`
   (Formula B.3, [`forced_radiation_factor`](/phonometry/reference/api/building/detailed-model/#forced_radiation_factor)) and the three-branch
   transmission factor of Formula (B.2)
   ([`calculated_sound_reduction_index`](/phonometry/reference/api/building/detailed-model/#calculated_sound_reduction_index)).
2. In-situ conversion (Clause 4.2.2). The total loss factor in situ follows
   from Annex C Formula (C.1),

   $$
   \eta_{tot} = \eta_{int} + \frac{2 \rho_o c_o \sigma}{2 \pi f m'} + \frac{c_o}{\pi^2 S \sqrt{f f_c}} \sum_k l_k \alpha_k
   $$

   ([`in_situ_total_loss_factor`](/phonometry/reference/api/building/detailed-model/#in_situ_total_loss_factor)), with the perimeter absorption
   coefficients deduced from the junctions' vibration reduction indices
   (Formula C.4, [`perimeter_absorption_coefficient`](/phonometry/reference/api/building/detailed-model/#perimeter_absorption_coefficient)). From it come the
   structural reverberation time $T_s = 2.2/(f \eta_{tot})$
   ([`structural_reverberation_time`](/phonometry/reference/api/building/detailed-model/#structural_reverberation_time)), the in-situ index
   $R_{situ} = R - 10 \log_{10}(T_{s,situ}/T_{s,lab})$ (Formula 9,
   [`in_situ_reduction_index`](/phonometry/reference/api/building/detailed-model/#in_situ_reduction_index)) and the equivalent absorption length
   $a_{situ} = 2.2\,\pi^2 S \sqrt{f_{ref}/f}/(c_o T_{s,situ})$
   (Formula 11).
3. Junctions (Formula 10).
   $D_{v,ij,situ} = K_{ij} - 10 \log_{10}(l_{ij}/\sqrt{a_{i,situ} a_{j,situ}})$,
   floored at 0 dB ([`in_situ_velocity_level_difference`](/phonometry/reference/api/building/detailed-model/#in_situ_velocity_level_difference)).
4. Paths. The direct path is
   $R_{Dd} = R_{s,situ} + \Delta R_{D,situ} + \Delta R_{d,situ}$
   (Formula 14) and each flanking path (Formula 15) is
   $R_{ij} = R_{i,situ}/2 + \Delta R_{i,situ} + R_{j,situ}/2 + \Delta R_{j,situ} + D_{v,ij,situ} + T$
   with the geometry term $T = 10 \log_{10}(S_s/\sqrt{S_i S_j})$
   ([`flanking_reduction_index`](/phonometry/reference/api/building/detailed-model/#flanking_reduction_index)).
5. Assembly. $R' = -10 \log_{10}(\sum 10^{-R/10})$ over the direct path and
   all flanking paths (Formulae 1 to 4), then `R'w (C; Ctr)` per ISO 717-1
   ([`detailed_airborne_prediction`](/phonometry/reference/api/building/detailed-model/#detailed_airborne_prediction)).

**Chain, impact (ISO 12354-2:2017, Clause 4.2).** The bare floor's normalized
impact sound pressure level per band follows from Annex B Formula (B.2),
$L_n = 155 - 30 \log_{10}(m') + 10 \log_{10}(T_s) + 10 \log_{10}(\sigma) + 10 \log_{10}(f/f_{ref})$
([`bare_floor_impact_level`](/phonometry/reference/api/building/detailed-model/#bare_floor_impact_level)); the direct path is
$L_{n,d} = L_{n,situ} - \Delta L_{situ} - \Delta L_{d,situ}$
(Formula 11) and each flanking path (Formula 12) is
$L_{n,ij} = L_{n,situ} - \Delta L_{situ} + (R_{i,situ} - R_{j,situ})/2 - \Delta R_{j,situ} - D_{v,ij,situ} - 10 \log_{10}(S_i/\sqrt{S_i S_j})$
([`flanking_impact_level`](/phonometry/reference/api/building/detailed-model/#flanking_impact_level)), combined
energetically into `L'n` and rated `L'n,w (CI)` per ISO 717-2
([`detailed_impact_prediction`](/phonometry/reference/api/building/detailed-model/#detailed_impact_prediction)).

The two parts share the same in-situ machinery, so a building is described once
([`HomogeneousElement`](/phonometry/reference/api/building/detailed-model/#homogeneouselement) per element, [`in_situ_element`](/phonometry/reference/api/building/detailed-model/#in_situ_element) per band) and
both the airborne and the impact chain read the same
[`InSituElementResult`](/phonometry/reference/api/building/detailed-model/#insituelementresult).

**Type A and Type B elements.** [`HomogeneousElement`](/phonometry/reference/api/building/detailed-model/#homogeneouselement) and
[`in_situ_element`](/phonometry/reference/api/building/detailed-model/#in_situ_element) describe a **Type A** element, one whose structural
reverberation time is set by the elements connected to it. For a **Type B**
element the standard takes $T_{s,situ} = T_{s,lab}$ (so no in-situ
transfer is
needed) and describes the junction with the normalized direction-averaged
velocity level difference `Dv,ij,n` instead of `Kij`, or with a laboratory
measurement of the flanking level difference `Dn,f`. Those branches are
[`flanking_reduction_index_from_normalized_difference`](/phonometry/reference/api/building/detailed-model/#flanking_reduction_index_from_normalized_difference) (Formula 17),
[`flanking_impact_level_from_normalized_difference`](/phonometry/reference/api/building/detailed-model/#flanking_impact_level_from_normalized_difference) (Part 2, Formula 14)
and [`flanking_reduction_index_from_flanking_level`](/phonometry/reference/api/building/detailed-model/#flanking_reduction_index_from_flanking_level) (Formula 16), with
[`resonant_sound_reduction_index`](/phonometry/reference/api/building/detailed-model/#resonant_sound_reduction_index) for the Annex B.1 correction their
element indices need below `fc`.

Clause and formula citations refer to ISO 12354-1:2017 (airborne) or
ISO 12354-2:2017 (impact). The worked example of ISO 12354-1:2017 Annex L and
ISO 12354-2:2017 Annex G (one heavy homogeneous building driving both parts) is
reproduced band by band in the test suite; the defects found in its printed
tables are recorded in `docs/ERRATA.md`.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## airborne_flanking_path

```python
airborne_flanking_path(
    *,
    label: str,
    kind: FlankingKind,
    element_i: InSituElementResult,
    element_j: InSituElementResult,
    vibration_reduction_index: ArrayLike,
    coupling_length: float,
    separating_area: float,
    delta_r_i: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> BandPath
```

Build one airborne flanking path from two in-situ elements (Formula 15).

The junction velocity level difference is formed from the two elements'
equivalent absorption lengths with Formula (10), then Formula (15) gives
`Rij` per band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `label` | Human-readable path name. |
| `kind` | `"Ff"`, `"Df"` or `"Fd"`. |
| `element_i` | The element excited in the source room. |
| `element_j` | The element radiating in the receiving room. |
| `vibration_reduction_index` | `Kij` of this path (per band or a single value), in dB. |
| `coupling_length` | Common coupling length `lij`, in m. |
| `separating_area` | Separating-element area `Ss`, in m². |
| `delta_r_i` | `ΔRi,situ` on element `i`, per band, in dB. |
| `delta_r_j` | `ΔRj,situ` on element `j`, per band, in dB. |

**Returns:** The [`BandPath`](/phonometry/reference/api/building/detailed-model/#bandpath) carrying `Rij`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `kind` is unknown or a geometry value is not positive. |

## BandPath

```python
BandPath(label: str, kind: str, values: np.ndarray)
```

One transmission path of the detailed model, per band.

**Attributes**

| Name | Description |
| :--- | :--- |
| `label` | Human-readable path name, e.g. `"ext wall 1-Df"`. |
| `kind` | `"Dd"`, `"Ff"`, `"Df"` or `"Fd"`. |
| `values` | `Rij` (airborne) or `Ln,ij` (impact) per band, in dB. |

## bare_floor_impact_level

```python
bare_floor_impact_level(
    frequencies: ArrayLike,
    *,
    mass_per_area: float,
    structural_reverberation_time: ArrayLike,
    radiation_factor: ArrayLike,
) -> np.ndarray
```

Normalized impact level of a bare monolithic floor (Part 2, F. B.2).

$$
L_n = 155 - 30 \log_{10}(m'/1\,\mathrm{kg/m^2}) + 10 \log_{10}(T_s/1\,\mathrm{s}) + 10 \log_{10} \sigma + 10 \log_{10}(f/f_{ref})
$$

with $f_{ref} = 1000$ Hz, the closed form obtained with
the force level of the standard tapping machine on a low-mobility floor.
Supplying the *in-situ* structural reverberation time and radiation factor
returns `Ln,situ` directly.

The reciprocity relation of Part 2 Formulae (B.3)/(B.4),
$R + L_n = 38 + 30 \log_{10} f$ in one-third-octave bands (43 in octave
bands),
holds where forced transmission is negligible and gives an independent
check on the pair.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `mass_per_area` | Mass per unit area `m'` of the floor, in kg/m². |
| `structural_reverberation_time` | Structural reverberation time `Ts` per band, in s. |
| `radiation_factor` | Radiation factor for free bending waves `σ` per band. |

**Returns:** The normalized impact sound pressure level `Ln` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive/finite or the per-band arrays do not share the band count. |

## bending_radiation_factor

```python
bending_radiation_factor(
    frequencies: ArrayLike,
    *,
    critical_frequency: float,
    length1: float,
    length2: float,
    speed_of_sound: float = 340.0,
) -> np.ndarray
```

Radiation factor for free bending waves `σ` (Formulae B.4 to B.6).

The three candidate factors of Formula (B.4) are

- $\sigma_1 = 1/\sqrt{1 - f_c/f}$ (above the critical frequency),
- $\sigma_2 = 4 l_1 l_2 (f/c_o)^2$ (the plate acting as a small
  piston),
- $\sigma_3 = \sqrt{2 \pi f (l_1 + l_2)/(16 c_o)}$ (corner and
  edge modes),

and the first plate mode
$f_{11} = c_o^2/(4 f_c) \cdot (1/l_1^2 + 1/l_2^2)$ selects
between the two regimes. For $f_{11} \le f_c/2$ the element is mode
dense at its critical frequency and Formula (B.5) applies:
$\sigma = \sigma_1$ at and above `fc`, and below it the
edge/corner sum
$\sigma = 2(l_1+l_2)/(l_1 l_2) \cdot (c_o/f_c) \cdot \delta_1 + \delta_2$ with $\lambda = \sqrt{f/f_c}$ and `δ2`
vanishing above `fc/2`. For $f_{11} > f_c/2$ Formula (B.6) picks
`σ3` unless `σ2` (below `fc`) or `σ1` (above `fc`) is smaller.
Every branch is capped at $\sigma \le 2.0$.

These relations hold for a plate in an infinite baffle; the standard notes
that walls and floors surrounded by orthogonal elements radiate 2 (edge
modes) to 4 (corner modes) times more efficiently well below `fc`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `critical_frequency` | Critical frequency $f_c = c_o^2/(1.8\,c_L t)$, Hz. |
| `length1` | One side length of the rectangular element, in m. |
| `length2` | The other side length, in m. |
| `speed_of_sound` | Speed of sound in air `co`, in m/s (Default: 340 m/s, the value ISO 12354-1 Annex A fixes). |

**Returns:** The radiation factor `σ` per band (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is not positive and finite. |

## calculated_sound_reduction_index

```python
calculated_sound_reduction_index(
    frequencies: ArrayLike,
    *,
    mass_per_area: float,
    critical_frequency: float,
    total_loss_factor: ArrayLike,
    radiation_factor: ArrayLike,
    forced_radiation_factor: ArrayLike,
    bands: BandType = 'third',
    resonant_only: bool = False,
    density: float | None = None,
    longitudinal_velocity: float | None = None,
    speed_of_sound: float = 340.0,
    air_density: float = 1.29,
) -> np.ndarray
```

Sound reduction index of a homogeneous element (Formulae B.2, B.10).

$R = -10 \log_{10} \tau$ with the three-branch transmission factor

- $f > f_c$:
  $\tau = (2 \rho_o c_o/(2 \pi f m'))^2 \cdot \pi f_c \sigma^2/(2 f \eta_{tot})$,
- $f \approx f_c$:
  $\tau = (2 \rho_o c_o/(2 \pi f m'))^2 \cdot \pi \sigma^2/(2 \eta_{tot})$,
- $f < f_c$:
  $\tau = (2 \rho_o c_o/(2 \pi f m'))^2 \cdot (F + R)$ with the
  forced term $F = 2 \sigma_f [1 - f^2/f_c^2]^{-2}$ and the
  resonant term $R = 2 (\pi f_c/(4 f)) \sigma^2/\eta_{tot}$.

The $f \approx f_c$ branch is applied to the band whose limits
straddle the
critical frequency, which is how the Annex L worked example selects it.

Below the critical frequency the first term is the *forced* contribution.
Annex B.1 requires flanking paths to use the **resonant** transmission
only; `resonant_only=True` drops that term (Annex B.3: "the contribution
of forced transmission can be neglected for flanking paths"). The Annex L
worked example keeps it on every path, so the default is `False`.

**High-frequency plateau (Formula B.10).** At high frequency the index of
a thick element stops growing; the standard bounds the transmission factor
from below by
$\tau_{plateau} = (4 \rho_o c_o/(1.1\,\rho c_L))^2 \cdot 0.02/\eta_{tot}$. Supplying
both `density` and `longitudinal_velocity` applies that floor,
$\tau = \max(\tau, \tau_{plateau})$, as the Annex L example does
from about 1250 Hz
upwards on its lightweight blockwork.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `mass_per_area` | Mass per unit area `m'`, in kg/m². |
| `critical_frequency` | Critical frequency `fc`, in Hz. |
| `total_loss_factor` | Total loss factor `ηtot` per band (laboratory or in situ, matching the situation being described). |
| `radiation_factor` | Radiation factor for free bending waves `σ` per band (see [`bending_radiation_factor`](/phonometry/reference/api/building/detailed-model/#bending_radiation_factor)). |
| `forced_radiation_factor` | Radiation factor for forced waves `σf` per band (see [`forced_radiation_factor`](/phonometry/reference/api/building/detailed-model/#forced_radiation_factor)); ignored when `resonant_only` is set. |
| `bands` | `"third"` (default) or `"octave"`, setting the band limits used to locate the $f \approx f_c$ branch. |
| `resonant_only` | Drop the forced-transmission term below `fc`. |
| `density` | Density `ρ` of the material, in kg/m³; with `longitudinal_velocity` it enables the Formula (B.10) plateau. |
| `longitudinal_velocity` | Quasi-longitudinal phase velocity `cL` of the material, in m/s. |
| `speed_of_sound` | Speed of sound in air `co`, in m/s. |
| `air_density` | Density of air `ρo`, in kg/m³. |

**Returns:** The sound reduction index `R` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive/finite or the per-band arrays do not share the band count. |

## detailed_airborne_prediction

```python
detailed_airborne_prediction(
    frequencies: ArrayLike,
    *,
    direct_index: ArrayLike,
    flanking_paths: Sequence[BandPath] = (),
    direct_label: str = 'Dd',
    bands: BandType = 'third',
) -> DetailedAirborneResult
```

Combine direct and flanking paths into `R'` per band (F. 1 to 4).

$R' = -10 \log_{10}(\sum 10^{-R/10})$ over the direct path `RDd` and
every
flanking path `Rij`. The result exposes each path's share of the
transmitted energy in every band, which is what identifies the path to
treat first, and the ISO 717-1 rating of the resulting spectrum whenever
the bands cover the rating range (100 Hz to 3150 Hz in one-third octaves,
125 Hz to 2000 Hz in octaves).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz. |
| `direct_index` | `RDd` per band, in dB (see [`direct_reduction_index`](/phonometry/reference/api/building/detailed-model/#direct_reduction_index)). |
| `flanking_paths` | The flanking paths (see [`airborne_flanking_path`](/phonometry/reference/api/building/detailed-model/#airborne_flanking_path)); may be empty. |
| `direct_label` | Label of the direct path (Default: `"Dd"`). |
| `bands` | `"third"` (default) or `"octave"`, selecting the ISO 717-1 rating range. |

**Returns:** The [`DetailedAirborneResult`](/phonometry/reference/api/building/detailed-model/#detailedairborneresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a path does not match the band count. |

## detailed_impact_prediction

```python
detailed_impact_prediction(
    frequencies: ArrayLike,
    *,
    direct_level: ArrayLike | None = None,
    flanking_paths: Sequence[BandPath] = (),
    direct_label: str = 'Dd',
    bands: BandType = 'third',
) -> DetailedImpactResult
```

Combine direct and flanking paths into `L'n` per band (Part 2, (1)).

$L'_n = 10 \log_{10}(\sum 10^{L_n/10})$ over the direct impact path
`Ln,d` and
every flanking path `Ln,ij`, with the ISO 717-2 rating of the resulting
spectrum whenever the bands cover the rating range. For rooms next to each
other there is no direct impact path and the sum runs over the flanking
paths only (Part 2, Formula 2): leave `direct_level` out and the result
carries no direct path at all.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz. |
| `direct_level` | `Ln,d` per band, in dB (see [`direct_impact_level`](/phonometry/reference/api/building/detailed-model/#direct_impact_level)), or `None` for the Formula (2) case of two rooms next to each other, which has no direct path. |
| `flanking_paths` | The flanking paths (see [`impact_flanking_path`](/phonometry/reference/api/building/detailed-model/#impact_flanking_path)); may be empty when `direct_level` is given. |
| `direct_label` | Label of the direct path (Default: `"Dd"`). |
| `bands` | `"third"` (default) or `"octave"`. |

**Returns:** The [`DetailedImpactResult`](/phonometry/reference/api/building/detailed-model/#detailedimpactresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a path does not match the band count, or if neither a direct level nor any flanking path is given. |

## DetailedAirborneResult

```python
DetailedAirborneResult(
    frequencies: np.ndarray,
    paths: tuple[BandPath, ...],
    r_prime: np.ndarray,
    fractions: np.ndarray,
    rating: WeightedRatingResult | None,
)
```

Per-band apparent sound reduction index `R'` (ISO 12354-1, 4.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz. |
| `paths` | Every transmission path (the direct path first), each with its `Rij` per band. |
| `r_prime` | Apparent sound reduction index `R'` per band, in dB. |
| `fractions` | Share of the transmitted energy carried by each path per band (paths x bands), summing to 1 in every band. |
| `rating` | `R'w (C; Ctr)` per ISO 717-1, or `None` when the bands supplied do not cover the rating range. |

### DetailedAirborneResult.dominant

*property*

Label of the path carrying most energy in each band.

### DetailedAirborneResult.plot()

```python
DetailedAirborneResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band path contributions and the resulting `R'`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### DetailedAirborneResult.report()

```python
DetailedAirborneResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a detailed airborne prediction fiche (EN/ISO 12354-1).

Writes a one-page **prediction** report for the per-band detailed
model: a basis line naming ISO 12354-1:2017 Clause 4.2, an optional
metadata header, a two-panel body with the per-path energy-share table
beside the per-band path-contribution figure, the boxed predicted
`R'w`, the prediction statement and, when a requirement is supplied,
a PASS/FAIL verdict, followed by the footer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the path table also gives the band in which each path contributes most. |
| `language` | `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine`/`language` is unknown or the result carries no ISO 717-1 rating. |
| ImportError | If reportlab or matplotlib is missing. |

## DetailedImpactResult

```python
DetailedImpactResult(
    frequencies: np.ndarray,
    paths: tuple[BandPath, ...],
    l_prime_n: np.ndarray,
    fractions: np.ndarray,
    rating: ImpactRatingResult | None,
)
```

Per-band apparent impact level `L'n` (ISO 12354-2, 4.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz. |
| `paths` | Every transmission path (the direct path first), each with its `Ln,ij` per band. |
| `l_prime_n` | Apparent normalized impact level `L'n` per band, in dB. |
| `fractions` | Share of the radiated energy carried by each path per band (paths x bands), summing to 1 in every band. |
| `rating` | `L'n,w (CI)` per ISO 717-2, or `None` when the bands supplied do not cover the rating range. |

### DetailedImpactResult.dominant

*property*

Label of the path carrying most energy in each band.

### DetailedImpactResult.plot()

```python
DetailedImpactResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band path contributions and the resulting `L'n`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### DetailedImpactResult.report()

```python
DetailedImpactResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a detailed impact prediction fiche (EN/ISO 12354-2).

The impact counterpart of [`DetailedAirborneResult.report`](/phonometry/reference/api/building/detailed-model/#detailedairborneresultreport): the
per-band detailed model of ISO 12354-2:2017 Clause 4.2, with the boxed
predicted `L'n,w` and a PASS/FAIL verdict against a requirement (a
lower level passing).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the path table also gives the band in which each path contributes most. |
| `language` | `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine`/`language` is unknown or the result carries no ISO 717-2 rating. |
| ImportError | If reportlab or matplotlib is missing. |

## direct_impact_level

```python
direct_impact_level(
    floor_level: ArrayLike,
    *,
    delta_l: ArrayLike = 0.0,
    delta_l_ceiling: ArrayLike = 0.0,
) -> np.ndarray
```

Normalized impact level of the direct path `Ln,d` (Part 2, F. 11).

$L_{n,d} = L_{n,situ} - \Delta L_{situ} - \Delta L_{d,situ}$: the
in-situ level of the bare
floor reduced by the floor covering and by any additional layer on the
receiving side (a suspended ceiling).

**Parameters**

| Name | Description |
| :--- | :--- |
| `floor_level` | `Ln,situ` of the bare separating floor, per band, dB. |
| `delta_l` | Improvement of the floor covering `ΔLsitu`, per band, dB. |
| `delta_l_ceiling` | Improvement `ΔLd,situ` of a layer on the receiving side, per band, in dB. |

**Returns:** `Ln,d` per band, in dB.

## direct_reduction_index

```python
direct_reduction_index(
    separating_index: ArrayLike,
    *,
    delta_r_source: ArrayLike = 0.0,
    delta_r_receiving: ArrayLike = 0.0,
) -> np.ndarray
```

Sound reduction index of the direct path `RDd` (Formula 14).

$R_{Dd} = R_{s,situ} + \Delta R_{D,situ} + \Delta R_{d,situ}$: the
in-situ index of the
separating element plus the improvement of any lining on its source and
receiving faces (for the in-situ improvement the standard accepts the
laboratory value, Formula 8).

**Parameters**

| Name | Description |
| :--- | :--- |
| `separating_index` | `Rs,situ` per band, in dB. |
| `delta_r_source` | `ΔRD,situ` on the source side, per band, in dB. |
| `delta_r_receiving` | `ΔRd,situ` on the receiving side, in dB. |

**Returns:** `RDd` per band, in dB.

## flanking_impact_level

```python
flanking_impact_level(
    *,
    floor_level: ArrayLike,
    index_i: ArrayLike,
    index_j: ArrayLike,
    velocity_level_difference: ArrayLike,
    area_i: float,
    area_j: float,
    delta_l: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> np.ndarray
```

Flanking normalized impact level `Ln,ij` per band (Part 2, F. 12).

$L_{n,ij} = L_{n,situ} - \Delta L_{situ} + (R_{i,situ} - R_{j,situ})/2 - \Delta R_{j,situ} - D_{v,ij,situ} - T$
with the geometry term $T = 10 \log_{10}(S_i/\sqrt{S_i S_j})$, `i` the
excited floor
and `j` the flanking element radiating in the receiving room.

**Parameters**

| Name | Description |
| :--- | :--- |
| `floor_level` | `Ln,situ` of the excited floor, per band, in dB. |
| `index_i` | `Ri,situ` of the excited floor, per band, in dB. |
| `index_j` | `Rj,situ` of the flanking element, per band, in dB. |
| `velocity_level_difference` | `Dv,ij,situ` per band, in dB. |
| `area_i` | Area `Si` of the excited floor, in m². |
| `area_j` | Area `Sj` of the flanking element, in m². |
| `delta_l` | Improvement of the floor covering `ΔLsitu`, in dB. |
| `delta_r_j` | Improvement `ΔRj,situ` of a lining on the flanking element, per band, in dB. |

**Returns:** `Ln,ij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an area is not positive and finite. |

## flanking_impact_level_from_flanking_level

```python
flanking_impact_level_from_flanking_level(
    normalized_flanking_impact_level: ArrayLike,
    *,
    area: float,
    laboratory_area: float,
    coupling_length: float,
    laboratory_coupling_length: float,
) -> np.ndarray
```

Flanking impact level from a measured `Ln,f` (Part 2, Formula 13).

$L_{n,ij} = L_{n,f,ij,situ} - 10 \log_{10}(S_i l_{lab}/(S_{i,lab} l_{ij}))$, the impact twin of
the airborne [`flanking_reduction_index_from_flanking_level`](/phonometry/reference/api/building/detailed-model/#flanking_reduction_index_from_flanking_level): the
route used when the flanking construction is characterised as a whole by a
laboratory measurement of the normalized flanking impact sound pressure
level (ISO 10848) instead of by the properties of its elements. The
laboratory measurement is transferred to the field situation first, as
ISO 12354-2:2017, Annex D indicates.

**Parameters**

| Name | Description |
| :--- | :--- |
| `normalized_flanking_impact_level` | `Ln,f,ij,situ` per band, in dB. |
| `area` | In-situ area `Si` of the excited floor, in m². |
| `laboratory_area` | Laboratory area `Si,lab` of the excited floor, in m². |
| `coupling_length` | In-situ coupling length `lij`, in m. |
| `laboratory_coupling_length` | Laboratory coupling length `llab`, in m. ISO 12354-1 Clause 4.4.2 gives the usual values: 4,5 m for horizontal flanking elements, 2,5 m for vertical ones. |

**Returns:** `Ln,ij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a geometry value is not positive and finite. |

## flanking_impact_level_from_normalized_difference

```python
flanking_impact_level_from_normalized_difference(
    *,
    floor_level: ArrayLike,
    index_i: ArrayLike,
    index_j: ArrayLike,
    normalized_velocity_level_difference: ArrayLike,
    area_i: float,
    coupling_length: float,
    delta_l: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> np.ndarray
```

Flanking impact level of a Type B junction (Part 2, Formula 14).

$L_{n,ij} = L_{n,ii} - \Delta L_i + (R_i - R_j)/2 - \Delta R_j - D_{v,ij,n} - 10 \log_{10}(S_i/(l_o l_{ij}))$
with the reference length $l_o = 1$ m: Formula (12) with the
junction described by the
normalized direction-averaged velocity level difference instead of
`Kij`, the form used for lightweight constructions.

**Parameters**

| Name | Description |
| :--- | :--- |
| `floor_level` | `Ln,ii` of the excited bare floor, per band, in dB. |
| `index_i` | `Ri` of the excited floor, per band, in dB. |
| `index_j` | `Rj` of the flanking element, per band, in dB. |
| `normalized_velocity_level_difference` | `Dv,ij,n` per band, in dB. |
| `area_i` | Area `Si` of the excited floor, in m². |
| `coupling_length` | Common coupling length `lij`, in m. |
| `delta_l` | Improvement of the floor covering `ΔLi`, per band, dB. |
| `delta_r_j` | Improvement `ΔRj` of a lining on the flanking element, per band, in dB. |

**Returns:** `Ln,ij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a geometry value is not positive and finite. |

## flanking_reduction_index

```python
flanking_reduction_index(
    *,
    index_i: ArrayLike,
    index_j: ArrayLike,
    velocity_level_difference: ArrayLike,
    separating_area: float,
    area_i: float,
    area_j: float,
    delta_r_i: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> np.ndarray
```

Flanking sound reduction index `Rij` per band (Formula 15).

$R_{ij} = R_{i,situ}/2 + \Delta R_{i,situ} + R_{j,situ}/2 + \Delta R_{j,situ} + D_{v,ij,situ} + T$
for `ij = Ff, Fd, Df`, with the geometry term
$T = 10 \log_{10}(S_s/\sqrt{S_i S_j})$. For diagonal transmission the
standard fixes $S_s = 10$ m².

The element indices depend on the path: `Ff` takes the flanking element
on both sides, `Fd` the flanking element as `i` and the separating
element as `j`, and `Df` the separating element as `i` and the
flanking element as `j`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `index_i` | `Ri,situ` of the element excited in the source room, in dB per band. |
| `index_j` | `Rj,situ` of the radiating element in the receiving room, per band, in dB. |
| `velocity_level_difference` | `Dv,ij,situ` per band, in dB (see [`in_situ_velocity_level_difference`](/phonometry/reference/api/building/detailed-model/#in_situ_velocity_level_difference); for a Type B junction pass the Formula (12) value derived from `Dv,ij,n`). |
| `separating_area` | Separating-element area `Ss`, in m². |
| `area_i` | Area `Si` of element `i`, in m². |
| `area_j` | Area `Sj` of element `j`, in m². |
| `delta_r_i` | `ΔRi,situ` on element `i`, per band, in dB. |
| `delta_r_j` | `ΔRj,situ` on element `j`, per band, in dB. |

**Returns:** `Rij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an area is not positive and finite. |

## flanking_reduction_index_from_flanking_level

```python
flanking_reduction_index_from_flanking_level(
    flanking_level_difference: ArrayLike,
    *,
    separating_area: float,
    coupling_length: float,
    laboratory_coupling_length: float,
    reference_absorption_area: float = 10.0,
) -> np.ndarray
```

Flanking index from a measured `Dn,f` (Formula 16).

$R_{ij} = D_{n,f,ij,situ} + 10 \log_{10}(S_s l_{lab}/(A_o l_{ij}))$ with
$A_o = 10$ m², the
route used when the flanking construction is characterised as a whole by a
laboratory measurement of the flanking normalized level difference
(ISO 10848). ISO 12354-1 Clause 4.4.2 gives the usual laboratory coupling
lengths: 4,5 m for horizontal flanking elements such as ceilings, 2,5 m
for vertical ones such as facades.

**Parameters**

| Name | Description |
| :--- | :--- |
| `flanking_level_difference` | `Dn,f,ij,situ` per band, in dB. |
| `separating_area` | Separating-element area `Ss`, in m². |
| `coupling_length` | In-situ coupling length `lij`, in m. |
| `laboratory_coupling_length` | Laboratory coupling length `llab`, m. |
| `reference_absorption_area` | `Ao`, in m² (Default: 10 m²). |

**Returns:** `Rij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a geometry value is not positive and finite. |

## flanking_reduction_index_from_normalized_difference

```python
flanking_reduction_index_from_normalized_difference(
    *,
    index_i: ArrayLike,
    index_j: ArrayLike,
    normalized_velocity_level_difference: ArrayLike,
    separating_area: float,
    coupling_length: float,
    delta_r_i: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> np.ndarray
```

Flanking index of a Type B junction `Rij` (Formula 17).

$R_{ij} = R_{i,situ}/2 + \Delta R_{i,situ} + R_{j,situ}/2 + \Delta R_{j,situ} + D_{v,ij,n} + T$ with
the geometry term $T = 10 \log_{10}(S_s/(l_o l_{ij}))$ and the reference
length $l_o = 1$ m. It is
Formula (15) with Formula (12) substituted, so the junction is described by
the *normalized* direction-averaged velocity level difference `Dv,ij,n`
(ISO 12354-1 Annex F) rather than by `Kij`: the form used for lightweight
double-leaf constructions, where the indices refer either to the double
element as a whole or to its inner leaf and should relate to resonant
transmission only (see [`resonant_sound_reduction_index`](/phonometry/reference/api/building/detailed-model/#resonant_sound_reduction_index)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `index_i` | `Ri,situ` of the element excited in the source room, in dB per band. |
| `index_j` | `Rj,situ` of the radiating element, per band, in dB. |
| `normalized_velocity_level_difference` | `Dv,ij,n` per band, in dB. |
| `separating_area` | Separating-element area `Ss`, in m². |
| `coupling_length` | Common coupling length `lij`, in m. |
| `delta_r_i` | `ΔRi,situ` on element `i`, per band, in dB. |
| `delta_r_j` | `ΔRj,situ` on element `j`, per band, in dB. |

**Returns:** `Rij` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a geometry value is not positive and finite. |

## floating_floor_improvement

```python
floating_floor_improvement(
    frequencies: ArrayLike,
    *,
    resonance_frequency: float,
    slope: float = 30.0,
) -> np.ndarray
```

Improvement of a floating floor `ΔL` per band (Part 2, Formula C.1).

$\Delta L = 30 \log_{10}(f/f_o)$ for sand/cement or calcium-sulfate
screeds and
$\Delta L = 40 \log_{10}(f/f_o)$ (`slope=40`, Formula C.3) for asphalt
or dry
floating floors, with the system resonance
$f_o = 160 \sqrt{s'/m'}$
(Formula C.2) and no improvement at or below it. The Annex L airborne
example reuses the same curve as `ΔR`, noting explicitly that assuming
$\Delta R = \Delta L$ is rough.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `resonance_frequency` | Resonance frequency `fo`, in Hz. |
| `slope` | 30 (screed, Formula C.1) or 40 (asphalt/dry, Formula C.3). |

**Returns:** The improvement `ΔL` per band, in dB (0 at and below `fo`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## forced_radiation_factor

```python
forced_radiation_factor(
    frequencies: ArrayLike,
    *,
    length1: float,
    length2: float,
    speed_of_sound: float = 340.0,
) -> np.ndarray
```

Radiation factor for forced waves `σf` (Formula B.3).

$\sigma_f = 0.5 (\ln(k_o \sqrt{l_1 l_2}) - \Lambda)$ capped at
$\sigma_f \le 2$, with $k_o = 2 \pi f / c_o$ and, for
$l_1 > l_2$,

$$
\Lambda = -0.964 - \left(0.5 + \frac{l_2}{\pi l_1}\right) \ln\frac{l_2}{l_1} + \frac{5 l_2}{2 \pi l_1} - E
$$

with $E = 1/(4 \pi l_1 l_2 k_o^2)$.

ISO 12354-1:2017 Table B.1 tabulates $10 \log_{10} \sigma_f$ for the
two standard laboratory openings (2 m² and 10 m²), which this
implementation reproduces.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `length1` | One side length of the rectangular element, in m. |
| `length2` | The other side length, in m. |
| `speed_of_sound` | Speed of sound in air `co`, in m/s (Default: 340 m/s). |

**Returns:** The forced radiation factor `σf` per band (dimensionless), clipped to $0 \le \sigma_f \le 2$ (the standard prints only the upper bound; the lower one guards the deep low-frequency extrapolation, where the logarithm turns negative).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any input is not positive and finite. |

## HomogeneousElement

```python
HomogeneousElement(
    label: str,
    area: float,
    length1: float,
    length2: float,
    mass_per_area: float,
    critical_frequency: float,
    internal_loss_factor: float = 0.01,
    perimeter_absorption: float = 0.0,
    density: float | None = None,
    longitudinal_velocity: float | None = None,
)
```

A Type A homogeneous element of the detailed model.

**Attributes**

| Name | Description |
| :--- | :--- |
| `label` | Human-readable element name, e.g. `"separating floor"`. |
| `area` | Element area `S`, in m². |
| `length1` | One side length of the rectangular element, in m. |
| `length2` | The other side length, in m. |
| `mass_per_area` | Mass per unit area `m'`, in kg/m². |
| `critical_frequency` | Critical frequency `fc`, in Hz. |
| `internal_loss_factor` | Internal loss factor `ηint` of the material (about 0,01 for common homogeneous building materials; ISO 12354-1 Table B.3 tabulates it per material). |
| `perimeter_absorption` | $\sum l_k \alpha_k$ over the element's perimeter, in m (Formula C.1; build it from [`perimeter_absorption_coefficient`](/phonometry/reference/api/building/detailed-model/#perimeter_absorption_coefficient) times the border lengths). |
| `density` | Density `ρ` of the material, in kg/m³; supplied together with `longitudinal_velocity` it enables the high-frequency plateau of Formula (B.10). `None` (the default) leaves the plateau off. |
| `longitudinal_velocity` | Quasi-longitudinal phase velocity `cL` of the material, in m/s (ISO 12354-1 Table B.3). |

## impact_flanking_path

```python
impact_flanking_path(
    *,
    label: str,
    floor: InSituElementResult,
    element_j: InSituElementResult,
    vibration_reduction_index: ArrayLike,
    coupling_length: float,
    delta_l: ArrayLike = 0.0,
    delta_r_j: ArrayLike = 0.0,
) -> BandPath
```

Build one impact flanking path `Df` (Part 2, Formula 12).

**Parameters**

| Name | Description |
| :--- | :--- |
| `label` | Human-readable path name. |
| `floor` | The excited separating floor, in situ. |
| `element_j` | The flanking element radiating in the receiving room. |
| `vibration_reduction_index` | `Kij` of this path, in dB. |
| `coupling_length` | Common coupling length `lij`, in m. |
| `delta_l` | Improvement of the floor covering `ΔLsitu`, in dB. |
| `delta_r_j` | `ΔRj,situ` of a lining on the flanking element, in dB. |

**Returns:** The [`BandPath`](/phonometry/reference/api/building/detailed-model/#bandpath) carrying `Ln,ij`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a geometry value is not positive and finite. |

## in_situ_element

```python
in_situ_element(
    element: HomogeneousElement,
    frequencies: ArrayLike,
    *,
    bands: BandType = 'third',
    resonant_only: bool = False,
    speed_of_sound: float = 340.0,
    air_density: float = 1.29,
) -> InSituElementResult
```

Evaluate one homogeneous element in situ, per band (Clause 4.2.2).

Runs the whole Annex B / Annex C chain in one call: the two radiation
factors, the in-situ total loss factor and structural reverberation time,
the equivalent absorption length, the calculated in-situ sound reduction
index and, for a floor, the calculated in-situ normalized impact level.

Because the element performance is *calculated from material properties*,
the in-situ loss factor enters Formula (B.2) directly and no
$10 \log_{10}(T_{s,situ}/T_{s,lab})$ transfer is needed
(Annex B.3). Use [`in_situ_reduction_index`](/phonometry/reference/api/building/detailed-model/#in_situ_reduction_index) instead when the element
data come from a laboratory measurement.

**Parameters**

| Name | Description |
| :--- | :--- |
| `element` | The [`HomogeneousElement`](/phonometry/reference/api/building/detailed-model/#homogeneouselement) description. |
| `frequencies` | Band centre frequencies, in Hz. |
| `bands` | `"third"` (default) or `"octave"`. |
| `resonant_only` | Drop the forced-transmission term of Formula (B.2) below `fc` (Annex B.1, flanking paths). |
| `speed_of_sound` | Speed of sound in air `co`, in m/s. |
| `air_density` | Density of air `ρo`, in kg/m³. |

**Returns:** The [`InSituElementResult`](/phonometry/reference/api/building/detailed-model/#insituelementresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any element property is not positive and finite. |

## in_situ_equivalent_absorption_length

```python
in_situ_equivalent_absorption_length(
    frequencies: ArrayLike,
    *,
    area: float,
    situ_reverberation_time: ArrayLike,
    speed_of_sound: float = 340.0,
) -> np.ndarray
```

In-situ equivalent absorption length `asitu` (Formula 11).

$a_{situ} = 2.2\,\pi^2 S \sqrt{f_{ref}/f}/(c_o T_{s,situ})$ with
$f_{ref} = 1000$ Hz. Note
the $\sqrt{f_{ref}/f}$ dependence: the absorption length grows as
the element
rings shorter at high frequency. For a Type B element the standard
replaces it by the element area, $a_{situ} = S/l_o$ (Formula 13).

This is the ISO 10848 Formula (12) quantity
([`phonometry.equivalent_absorption_length`](/phonometry/reference/api/building/flanking-transmission/#equivalent_absorption_length)) evaluated with the
ISO 12354 value $c_o = 340$ m/s.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `area` | Element area `S`, in m². |
| `situ_reverberation_time` | `Ts,situ` per band, in s. |
| `speed_of_sound` | Speed of sound in air `co`, in m/s. |

**Returns:** The equivalent absorption length `asitu` per band, in m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive/finite or the band counts disagree. |

## in_situ_impact_level

```python
in_situ_impact_level(
    impact_level: ArrayLike,
    situ_reverberation_time: ArrayLike,
    laboratory_reverberation_time: ArrayLike,
) -> np.ndarray
```

In-situ normalized impact level `Ln,situ` (Part 2, Formula 5).

$L_{n,situ} = L_n + 10 \log_{10}(T_{s,situ}/T_{s,lab})$, the sign
opposite to
[`in_situ_reduction_index`](/phonometry/reference/api/building/detailed-model/#in_situ_reduction_index): a floor that rings longer in the building
than in the laboratory radiates more impact sound.

**Parameters**

| Name | Description |
| :--- | :--- |
| `impact_level` | Laboratory level `Ln` per band, in dB. |
| `situ_reverberation_time` | `Ts,situ` per band, in s. |
| `laboratory_reverberation_time` | `Ts,lab` per band, in s. |

**Returns:** The in-situ level `Ln,situ` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a reverberation time is not positive/finite or the band counts disagree. |

## in_situ_reduction_index

```python
in_situ_reduction_index(
    sound_reduction_index: ArrayLike,
    situ_reverberation_time: ArrayLike,
    laboratory_reverberation_time: ArrayLike,
) -> np.ndarray
```

In-situ sound reduction index `Rsitu` (Formula 9).

$R_{situ} = R - 10 \log_{10}(T_{s,situ}/T_{s,lab})$: an element that is
better damped in
the building than in the test frame radiates less and gains index. The
standard notes that $R_{situ} = R$ is a usable first approximation,
and the
correction is exactly zero for Type B elements (Clause 4.2.2.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_reduction_index` | Laboratory index `R` per band, in dB. |
| `situ_reverberation_time` | In-situ structural reverberation time `Ts,situ` per band, in s. |
| `laboratory_reverberation_time` | Laboratory structural reverberation time `Ts,lab` per band, in s. |

**Returns:** The in-situ index `Rsitu` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a reverberation time is not positive/finite or the band counts disagree. |

## in_situ_total_loss_factor

```python
in_situ_total_loss_factor(
    frequencies: ArrayLike,
    *,
    internal_loss_factor: float,
    mass_per_area: float,
    area: float,
    critical_frequency: float,
    radiation_factor: ArrayLike,
    perimeter_absorption: float,
    speed_of_sound: float = 340.0,
    air_density: float = 1.29,
) -> np.ndarray
```

Total loss factor in situ `ηtot,situ` (Formula C.1).

$\eta_{tot} = \eta_{int} + 2 \rho_o c_o \sigma/(2 \pi f m') + c_o/(\pi^2 S \sqrt{f f_c}) \cdot \sum_k l_k \alpha_k$: the
internal losses of the material, the losses by radiation into the air
and the losses at the perimeter of the element.
$\sum l_k \alpha_k$ is the junction-length-weighted sum of the
Formula (C.4) absorption coefficients (see
[`perimeter_absorption_coefficient`](/phonometry/reference/api/building/detailed-model/#perimeter_absorption_coefficient)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `internal_loss_factor` | Internal loss factor `ηint` of the material (about 0,01 for common homogeneous building materials). |
| `mass_per_area` | Mass per unit area `m'`, in kg/m². |
| `area` | Element area `S`, in m². |
| `critical_frequency` | Critical frequency `fc`, in Hz. |
| `radiation_factor` | Radiation factor `σ` per band. |
| `perimeter_absorption` | $\sum l_k \alpha_k$ over the element's perimeter, in m (may be zero for a free-edged element). |
| `speed_of_sound` | Speed of sound in air `co`, in m/s. |
| `air_density` | Density of air `ρo`, in kg/m³. |

**Returns:** The total loss factor `ηtot,situ` per band (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive/finite, the perimeter sum is negative, or the band counts disagree. |

## in_situ_velocity_level_difference

```python
in_situ_velocity_level_difference(
    vibration_reduction_index: ArrayLike,
    *,
    coupling_length: float,
    absorption_length_i: ArrayLike,
    absorption_length_j: ArrayLike,
) -> np.ndarray
```

In-situ velocity level difference `Dv,ij,situ` (Formula 10).

$D_{v,ij,situ} = K_{ij} - 10 \log_{10}(l_{ij}/\sqrt{a_{i,situ} a_{j,situ}})$, floored at 0 dB as
the formula prescribes. It converts the situation-invariant junction
descriptor `Kij` (ISO 12354-1 Annex E, or measured per ISO 10848) into
the level drop the junction actually produces between the two elements as
built.

**Parameters**

| Name | Description |
| :--- | :--- |
| `vibration_reduction_index` | `Kij` per band (or a single value broadcast to all bands), in dB. |
| `coupling_length` | Common coupling length `lij`, in m. |
| `absorption_length_i` | `ai,situ` per band, in m. |
| `absorption_length_j` | `aj,situ` per band, in m. |

**Returns:** `Dv,ij,situ` per band, in dB (never negative).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a length is not positive/finite or the band counts disagree. |

## InSituElementResult

```python
InSituElementResult(
    label: str,
    frequencies: np.ndarray,
    area: float,
    radiation_factor: np.ndarray,
    forced_radiation_factor: np.ndarray,
    total_loss_factor: np.ndarray,
    reverberation_time: np.ndarray,
    absorption_length: np.ndarray,
    sound_reduction_index: np.ndarray,
    impact_level: np.ndarray,
)
```

Per-band in-situ description of one element (Clause 4.2.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `label` | The element name. |
| `frequencies` | Band centre frequencies, in Hz. |
| `area` | Element area `S`, in m². |
| `radiation_factor` | Radiation factor for free bending waves `σ`. |
| `forced_radiation_factor` | Radiation factor for forced waves `σf`. |
| `total_loss_factor` | In-situ total loss factor `ηtot,situ`. |
| `reverberation_time` | In-situ structural reverberation time `Ts,situ`, in s. |
| `absorption_length` | In-situ equivalent absorption length `asitu`, in m. |
| `sound_reduction_index` | In-situ sound reduction index `Rsitu`, dB. |
| `impact_level` | In-situ normalized impact level `Ln,situ` of the bare element, in dB (meaningful for the excited floor). |

### InSituElementResult.plot()

```python
InSituElementResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the in-situ `Rsitu` and `Ln,situ` spectra of the element.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## laboratory_total_loss_factor

```python
laboratory_total_loss_factor(
    frequencies: ArrayLike,
    *,
    mass_per_area: float,
    internal_loss_factor: float = 0.01,
) -> np.ndarray
```

Total loss factor in the laboratory `ηtot,lab` (Formula C.3).

$\eta_{tot,lab} \approx \eta_{int} + m'/(485 \sqrt{f})$, the
estimate for the heavy test frame
of an ISO 10140 facility. The relation holds for elements below
$m' = 800$ kg/m² and `ηint` can normally be taken as 0.01.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `mass_per_area` | Mass per unit area `m'`, in kg/m². |
| `internal_loss_factor` | Internal loss factor `ηint` (Default: 0,01). |

**Returns:** The laboratory total loss factor per band (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## perimeter_absorption_coefficient

```python
perimeter_absorption_coefficient(
    critical_frequencies: ArrayLike,
    vibration_reduction_indices: ArrayLike,
) -> float
```

Absorption coefficient for bending waves at one border (Formula C.4).

$\alpha_k = \sum_j \sqrt{f_{c,j}/f_{ref}} \cdot 10^{-K_{ij}/10}$
summed over the elements `j`
connected to the considered element at border `k` (the standard sums
over at most three). Multiplied by the border length and summed over the
perimeter it gives the $\sum l_k \alpha_k$ that
[`in_situ_total_loss_factor`](/phonometry/reference/api/building/detailed-model/#in_situ_total_loss_factor) takes. Annex C.3 places the in-situ
coefficients between 0,05 and 0,5.

**Parameters**

| Name | Description |
| :--- | :--- |
| `critical_frequencies` | Critical frequency `fc,j` of each connected element, in Hz. |
| `vibration_reduction_indices` | Vibration reduction index `Kij` of the path to each connected element, in dB (same order and length). |

**Returns:** The absorption coefficient `αk` at that border (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the two sequences differ in length, a critical frequency is not positive, or an index is not finite. |

## reciprocity_impact_level

```python
reciprocity_impact_level(
    sound_reduction_index: ArrayLike,
    frequencies: ArrayLike,
    *,
    bands: BandType = 'third',
) -> np.ndarray
```

Impact level of a homogeneous floor by reciprocity (Part 2, B.3/B.4).

$R + L_n = 38 + 30 \log_{10}(f/1\,\mathrm{Hz})$ in one-third-octave bands
and
$R + L_n = 43 + 30 \log_{10}(f/1\,\mathrm{Hz})$ in octave bands: for a
homogeneous floor
the sum of the airborne index and the normalized impact level depends only
on frequency, provided forced transmission is negligible (normally up to
about 1 kHz, above which the stiffness of the floor's top layer matters).

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_reduction_index` | `R` of the floor per band, in dB. |
| `frequencies` | Band centre frequencies, in Hz. |
| `bands` | `"third"` (default, constant 38) or `"octave"` (43). |

**Returns:** The normalized impact sound pressure level `Ln` per band, dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive/finite or the band counts disagree. |

## resonant_sound_reduction_index

```python
resonant_sound_reduction_index(
    sound_reduction_index: ArrayLike,
    frequencies: ArrayLike,
    *,
    critical_frequency: float,
    correction: float = 8.0,
) -> np.ndarray
```

Correct a measured `R` to resonant transmission only (Formula B.1).

$R^* = R + 10 \log_{10}(\sigma_a/\sigma_s)$. No standardized method
exists to measure the
two radiation factors, so Annex B.2 gives the estimate this function
applies: no correction for elements separated by one or two cavities, and
a fixed correction (8 dB, the standard's figure for single homogeneous or
layered wood or steel frame elements without a cavity) **below the
critical frequency only**. Above `fc` the laboratory index already
describes resonant transmission and is returned unchanged.

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_reduction_index` | Measured index `R` per band, in dB. |
| `frequencies` | Band centre frequencies, in Hz. |
| `critical_frequency` | Critical frequency `fc`, in Hz. |
| `correction` | Correction applied below `fc`, in dB (Default: 8 dB; Annex B.2 caps the estimate of Formula (B.8) at this value, and the Annex L lightweight example reduces it around the cavity resonance). |

**Returns:** The resonant-only index `R*` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive/finite or the band counts disagree. |

## structural_reverberation_time

```python
structural_reverberation_time(
    frequencies: ArrayLike,
    total_loss_factor: ArrayLike,
) -> np.ndarray
```

Structural reverberation time $T_s = 2.2/(f \eta_{tot})$ (C.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `total_loss_factor` | Total loss factor `ηtot` per band. |

**Returns:** The structural reverberation time `Ts` per band, in s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive/finite or the band counts disagree. |
