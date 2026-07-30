---
title: "building.resilient_layers"
description: "Prediction of resilient-layer performance: tapping force, coverings, floors, linings."
sidebar:
  label: "resilient_layers"
---

Prediction of resilient-layer performance: tapping force, coverings, floors, linings.

The measurement modules of this domain report what a resilient layer *achieved*
([`phonometry.building.floor_covering_improvement`](/phonometry/reference/api/building/floor-covering-improvement/) for the ISO 16251-1
mock-up, [`phonometry.materials.dynamic_stiffness`](/phonometry/reference/api/materials/dynamic-stiffness/) for the EN 29052-1
dynamic stiffness). This module is their **prediction** counterpart: it walks
the physical chain from the material data to the improvement spectrum, so a
covering or a floating floor can be sized before anything is built.

The chain is one story told in four steps.

**1. The excitation (Hopkins 3.6.3).** The ISO tapping machine drops a 0,5 kg
hammer from 40 mm, ten impacts per second, so the impact velocity is
`vo = √(2 g h) = 0,886 m/s` (Eq. 3.85) and, for a short impact, the peak force
per Fourier line is `|Fn| = 2 m vo/Ti` (Eq. 3.90), giving the band mean-square
force `F²rms = 3,9 B` (Eq. 3.92). Real floors are not that simple: the hammer,
the contact stiffness `K` it deforms and the floor's driving-point impedance
`Zdp` form a mass-spring-dashpot (Fig. 3.28) whose force pulse
([`force_pulse`](/phonometry/reference/api/building/resilient-layers/#force_pulse), Eqs. 3.95/3.96) is **over-critical** when `K m ≥ 4 Zdp²`
(a single positive pulse, no rebound) and **under-critical** otherwise (a
rebound; only the first positive lobe is transformed). Its spectrum
([`tapping_force_spectrum`](/phonometry/reference/api/building/resilient-layers/#tapping_force_spectrum)) is flat up to the cut-off `fco`
(Eqs. 3.101/3.102) and falls above it, and it asymptotes at low frequency
between `|Fn|lower = m vo/Ti` and `|Fn|upper = 2 m vo/Ti`, 6 dB apart in
mean square (Eqs. 3.99/3.100).

**2. Soft floor coverings (Hopkins 4.4.3.1).** A soft covering on a heavyweight
floor changes nothing but the force input, so its improvement is the force ratio
`ΔL = 20 lg(|Fn|without/|Fn|with)` (Eq. 4.114). The covering's contact
stiffness `K = E π r²/d` (Eq. 3.98) sets its cut-off, against the bare plate's
`K = 2 r E/(1 − ν²)` (Eq. 3.97), which is why a two-line estimate,
`ΔL ≈ 0` below `fco` and 12 dB/octave above it, captures the whole design
question ([`covering_improvement`](/phonometry/reference/api/building/resilient-layers/#covering_improvement)).

**3. Floating floors (Hopkins 4.4.4, ISO 12354-2 Annex C, Vigran 8.4).** Above
the mass-spring resonance `fo = 160 √(s'/m')` (Formula C.2) the improvement
follows one of three laws ([`floating_floor_improvement_spectrum`](/phonometry/reference/api/building/resilient-layers/#floating_floor_improvement_spectrum)): the
infinite-plate result of Cremer, `ΔL = 40 lg(f/fo)` (Eq. 4.119, Vigran
Eq. 8.40), the empirical `ΔL = 30 lg(f/fo)` that EN 12354-2 adopted for
sand-cement screeds (Formula C.1, Eq. 4.124), and the same 40 lg law with the
hammer-impedance term `10 lg[1 + (f/flimit)²]` that a lightweight walking
surface needs (Eq. 4.123, Vigran Eq. 8.48). A floating floor on discrete
mounts instead of a continuous layer is a two-subsystem SEA problem
([`resilient_mount_improvement`](/phonometry/reference/api/building/resilient-layers/#resilient_mount_improvement), Vér's model as Hopkins Eq. 4.118 and
Vigran Eq. 8.45) and rises at 30 dB/decade, not 40. Two floating floors stacked
give two resonances ([`double_floating_floor_resonances`](/phonometry/reference/api/building/resilient-layers/#double_floating_floor_resonances), Eq. 4.125), and
the weighted single number follows from `m'` and `s'` directly
([`weighted_floating_floor_improvement`](/phonometry/reference/api/building/resilient-layers/#weighted_floating_floor_improvement), Formulae C.4/C.5).

**4. Wall linings (ISO 12354-1 Annex D).** A lining improves or *degrades* the
sound insulation depending on where its resonance falls, so Annex D predicts the
weighted improvement from `fo` alone: Formula (D.1) for a layer bonded
directly to the wall, Formula (D.2) for one on studs over a filled cavity, then
Table D.1 for interior linings ([`weighted_lining_improvement`](/phonometry/reference/api/building/resilient-layers/#weighted_lining_improvement)), Formulae
(D.3) to (D.6) for exterior thermal systems and (D.7) for stud systems
([`lining_improvement`](/phonometry/reference/api/building/resilient-layers/#lining_improvement)), and Formula (D.8) to carry a laboratory rating to
the field ([`lining_improvement_in_situ`](/phonometry/reference/api/building/resilient-layers/#lining_improvement_in_situ)).

Citations are to ISO 12354-1:2017 / ISO 12354-2:2017, to Hopkins, *Sound
Insulation* (2007) and to Vigran, *Building Acoustics* (2008). Where the two
books state the same model in different algebra the test suite pins the identity
rather than either transcription. Two printed defects are relevant here and are
recorded in `docs/ERRATA.md`: the overlap of the last two rows of Table D.1 at
1 600 Hz, and the carpet stiffness in the caption of Vigran's Fig. 8.37.

Several relations used here carry no published worked example, so they are
implemented as printed and checked only for self-consistency: the cavity
stiffness `0,111/d` of Formula (D.2), the asphalt fit of Formula (C.5), and
the exterior-system and stud fits of Formulae (D.3) to (D.8). The guide
"Predicting Resilient-Layer Performance" says which pieces have an oracle and
which do not.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## combined_dynamic_stiffness

```python
combined_dynamic_stiffness(layers: ArrayLike) -> float
```

Total dynamic stiffness of stacked resilient layers (Formula C.6).

`s'tot = (Σ 1/s'i)^(−1)`, springs in series (Hopkins Eq. 4.121 states the
same rule). ISO 12354-2:2017 warns that it holds only if every layer covers
the whole floor without cuts for pipes or electrical devices.

**Parameters**

| Name | Description |
| :--- | :--- |
| `layers` | Dynamic stiffnesses per unit area `s'i`, in N/m³ (any 1-D array-like). |

**Returns:** The total dynamic stiffness `s'tot`, in N/m³.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `layers` is empty or holds a non-positive value. |

## covering_contact_stiffness

```python
covering_contact_stiffness(
    youngs_modulus: float,
    thickness: float,
    *,
    radius: float = 0.015,
) -> float
```

Contact stiffness of a soft floor covering `K = E π r²/d` (Eq. 3.98).

The covering is treated as a linear spring of area `π r²` under the
hammer, so only the ratio `E/d` matters. Vigran's Eq. (8.51) is the same
expression written with the hammer area `Sh`, quoted there as 7 cm²
against the 7,07 cm² of a 15 mm radius.

**Parameters**

| Name | Description |
| :--- | :--- |
| `youngs_modulus` | Young's modulus `E` of the covering, in Pa. |
| `thickness` | Covering thickness `d`, in m. |
| `radius` | Contact radius `r`, in m (Default: 0,015). |

**Returns:** The contact stiffness `K`, in N/m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## covering_improvement

```python
covering_improvement(
    frequencies: ArrayLike,
    covering_stiffness: float,
    plate_stiffness: float,
    impedance: float,
    *,
    mass: float = 0.5,
    impact_rate: float = 10.0,
    band: BandWidth = 'third',
) -> CoveringImprovementResult
```

Improvement of impact sound insulation by a soft covering (Eq. 4.114).

On a heavyweight base floor a soft covering has a negligible effect on the
mass, bending stiffness and total loss factor of the slab, so it alters
only the force the hammer injects. The improvement is then the ratio of the
two force spectra, `ΔL = 20 lg(|Fn|without/|Fn|with)`, computed here from
[`tapping_force_spectrum`](/phonometry/reference/api/building/resilient-layers/#tapping_force_spectrum) with the covering's contact stiffness
(Eq. 3.98) and with the plate's (Eq. 3.97).

The tapping machine excites a **line** spectrum, at multiples of the 10 Hz
impact rate, so Eq. (4.114) is a statement about one Fourier component and
the band value is the ratio of the band mean-square forces (Eq. 3.91),
that is the sum over the lines that fall in the band. `improvement` is
that band value and `line_improvement` is the per-line ratio. The
distinction matters: the undamped model's transform has exact nulls at odd
multiples of `fco`, so a band centre that happens to land on one reads
tens of dB high. With the 100 Hz cut-off of Hopkins's covering No. 2, the
line ratio at 500 Hz is 66,8 dB against a two-line estimate of 27,9 dB,
while the band value is 29,1 dB. Hopkins notes below Fig. 4.64 that the
troughs vanish once the covering's internal damping is included and the
spectrum is averaged into bands.

`two_line` is Hopkins's design estimate: `ΔL ≈ 0` below the covering's
cut-off and a straight 12 dB/octave above it, that is `40 lg(f/fco)`.
Real coverings behave as non-linear springs under the tapping machine's
high force and show two or three slopes between 5 and 22 dB/octave, so the
model identifies the general features rather than replacing a measurement.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `covering_stiffness` | Contact stiffness `K` of the covering, in N/m ([`covering_contact_stiffness`](/phonometry/reference/api/building/resilient-layers/#covering_contact_stiffness)). |
| `plate_stiffness` | Contact stiffness `K` of the bare plate, in N/m ([`plate_contact_stiffness`](/phonometry/reference/api/building/resilient-layers/#plate_contact_stiffness)). |
| `impedance` | Driving-point impedance `Zdp` of the base floor, in N.s/m; unchanged by the covering. |
| `mass` | Hammer mass `m`, in kg (Default: 0,5). |
| `impact_rate` | Impact repetition rate `fi`, in Hz (Default: 10); it sets the spacing of the Fourier lines the bands average over. |
| `band` | `"third"` or `"octave"`. |

**Returns:** A [`CoveringImprovementResult`](/phonometry/reference/api/building/resilient-layers/#coveringimprovementresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, or `band` is unknown. |

## CoveringImprovementResult

```python
CoveringImprovementResult(
    frequencies: np.ndarray,
    improvement: np.ndarray,
    two_line: np.ndarray,
    cut_off_frequency: float,
    bare_cut_off_frequency: float,
    lines: np.ndarray,
    line_improvement: np.ndarray,
    bare: TappingForceResult,
    covered: TappingForceResult,
)
```

Predicted improvement `ΔL` of a soft floor covering (Hopkins 4.4.3.1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `improvement` | Band improvement `ΔL`, in dB: Eq. (4.114) evaluated over the tapping machine's Fourier lines and summed in mean square across each band, `10 lg(Σ\|Fn\|²without/Σ\|Fn\|²with)`. |
| `two_line` | The two-line estimate, in dB: 0 below `fco` and 12 dB/octave (40 dB/decade) above it. |
| `cut_off_frequency` | Cut-off frequency `fco` of the covered floor, in Hz. |
| `bare_cut_off_frequency` | Cut-off frequency of the bare plate, in Hz. |
| `lines` | Fourier line frequencies `n fi` of the tapping machine, in Hz, covering every band in `frequencies`. |
| `line_improvement` | The per-line ratio `ΔL = 20 lg(\|Fn\|without/\|Fn\|with)` of Eq. (4.114) at `lines`, in dB. It carries the deep troughs at odd multiples of `fco` that Hopkins notes below Fig. 4.64, which are an artefact of the undamped model and disappear from `improvement`. |
| `bare` | The bare-plate [`TappingForceResult`](/phonometry/reference/api/building/resilient-layers/#tappingforceresult), at `lines`. |
| `covered` | The [`TappingForceResult`](/phonometry/reference/api/building/resilient-layers/#tappingforceresult) with the covering, at `lines`. |

### CoveringImprovementResult.plot()

```python
CoveringImprovementResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `ΔL(f)` from the force ratio beside the two-line estimate.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## double_floating_floor_resonances

```python
double_floating_floor_resonances(
    lower_stiffness: float,
    lower_mass_per_area: float,
    upper_stiffness: float,
    upper_mass_per_area: float,
) -> tuple[float, float]
```

The two resonances of a double floating floor (Hopkins Eq. 4.125).

One floating floor on top of another over a heavyweight base is a
mass-spring-mass-spring system with

`fmsms = (1/(2^(3/2) π)) √(X ± √(X² − 4 s1 s2/(ρs1 ρs2)))`, where
`X = s1/ρs1 + s2/ρs1 + s2/ρs2`,

subscript 1 being the lower floating floor (on the resilient layer that
rests on the base) and 2 the upper one. The double floor avoids the single
floor's dip at `fms`, but the steep rise in `ΔL` only starts above the
higher of the two resonances. For two identical floors the roots are
`fms √((3 ± √5)/2)`, that is `0,618 fms` and `1,618 fms`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `lower_stiffness` | Dynamic stiffness per unit area `s1` of the lower resilient layer, in N/m³. |
| `lower_mass_per_area` | Mass per unit area `ρs1` of the lower floating floor, in kg/m². |
| `upper_stiffness` | Dynamic stiffness per unit area `s2` of the upper resilient layer, in N/m³. |
| `upper_mass_per_area` | Mass per unit area `ρs2` of the upper floating floor, in kg/m². |

**Returns:** `(lower, upper)` resonance frequencies, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## floating_floor_improvement_spectrum

```python
floating_floor_improvement_spectrum(
    frequencies: ArrayLike,
    *,
    resonance_frequency: float,
    model: FloatingFloorModel = 'en12354',
    limiting_frequency: float | None = None,
    mass_per_area: float | None = None,
    dynamic_stiffness: float | None = None,
) -> FloatingFloorImprovementResult
```

Improvement `ΔL(f)` of a floating floor on a heavyweight base floor.

Three laws share the same anchor, the mass-spring resonance `fo` of the
walking surface on the resilient layer ([`floating_floor_resonance_frequency`](/phonometry/reference/api/building/resilient-layers/#floating_floor_resonance_frequency)),
and all give `ΔL = 0` at and below it (in the band containing `fo`,
`ΔL` is in practice between −5 dB and 0 dB):

* `"cremer"`: `ΔL = 40 lg(f/fo)`, Cremer's 1952 result for two
  infinite, locally reacting plates coupled by a spring layer (Hopkins
  Eq. 4.119, Vigran Eq. 8.40), i.e. 12 dB per octave. It holds for
  constructions with high internal damping, such as asphalt screeds, and is
  the branch ISO 12354-2 Formula (C.3) prescribes for asphalt and dry
  floating floors.
* `"en12354"` (default): `ΔL = 30 lg(f/fo)`, the empirical law of
  ISO 12354-2 Formula (C.1) for sand-cement and calcium-sulfate screeds
  (Hopkins Eq. 4.124). Sand-cement screeds have a low internal loss factor
  and act as finite plates with a reverberant bending field, for which the
  40 lg law overestimates `ΔL`.
* `"cremer_hammer"`: `ΔL = 40 lg(f/fo) + 10 lg[1 + (f/flimit)²]`, the
  40 lg law with the reduction in power input above the limiting frequency
  of the hammer's own impedance (Hopkins Eq. 4.123, Vigran Eq. 8.48). A
  lightweight walking surface such as chipboard needs it, and tends to
  18 dB per octave well above `flimit`.

The laws are stated as valid above `fo`, and Cremer's derivation is
reported to hold in `fo < f < 4 fo`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `resonance_frequency` | Mass-spring resonance `fo`, in Hz. |
| `model` | `"en12354"`, `"cremer"` or `"cremer_hammer"`. |
| `limiting_frequency` | Limiting frequency `flimit`, in Hz; required by `"cremer_hammer"` and ignored otherwise ([`hammer_limiting_frequency`](/phonometry/reference/api/building/resilient-layers/#hammer_limiting_frequency)). |
| `mass_per_area` | Optional `m'` of the floating floor, in kg/m². |
| `dynamic_stiffness` | Optional `s'` of the resilient layer, in N/m³; supplied together with `mass_per_area` it adds `ΔLw` to the result, from Formula (C.4) for `"en12354"` (screeds) and Formula (C.5) for the other two models (asphalt and dry floating floors). |

**Returns:** A [`FloatingFloorImprovementResult`](/phonometry/reference/api/building/resilient-layers/#floatingfloorimprovementresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, `model` is unknown, or `"cremer_hammer"` is used without a limiting frequency. |

## floating_floor_resonance_frequency

```python
floating_floor_resonance_frequency(
    dynamic_stiffness: float,
    mass_per_area: float,
) -> float
```

Resonance `fo = 160 √(s'/m')` of a floating floor (Formula C.2).

ISO 12354-2:2017 Formula (C.2), with `s'` the EN 29052-1 dynamic
stiffness per unit area of the resilient layer measured without pre-load
and `m'` the mass per unit area of the floating floor. The printed
constant 160 rounds the exact mass-spring value `1000/(2 π) = 159,15`
that [`phonometry.materials.natural_frequency`](/phonometry/reference/api/materials/dynamic-stiffness/#natural_frequency) applies, so the two
differ by 0,5 %; this function reproduces the standard, whose own Annex G
example prints `fo = 52,8 Hz` for `s' = 8 MN/m³`, `m' = 73,5 kg/m²`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `dynamic_stiffness` | Dynamic stiffness per unit area `s'`, in N/m³ (i.e. 8e6 for the 8 MN/m³ of the standard's example). |
| `mass_per_area` | Mass per unit area `m'` of the floating floor, in kg/m². |

**Returns:** The resonance frequency `fo`, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## FloatingFloorImprovementResult

```python
FloatingFloorImprovementResult(
    frequencies: np.ndarray,
    improvement: np.ndarray,
    resonance_frequency: float,
    model: str,
    slope: float,
    limiting_frequency: float | None = None,
    delta_lw: float | None = None,
)
```

Predicted improvement `ΔL(f)` of a floating floor.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `improvement` | Improvement `ΔL` per band, in dB (0 at and below `resonance_frequency`). |
| `resonance_frequency` | Mass-spring resonance `fo`, in Hz. |
| `model` | `"en12354"`, `"cremer"` or `"cremer_hammer"`. |
| `slope` | Slope of the law, in dB per decade (30 or 40). |
| `limiting_frequency` | Limiting frequency `flimit` of the hammer term, in Hz, or `None` when the term is not applied. |
| `delta_lw` | Weighted improvement `ΔLw`, in dB, or `None` when the floor data were not supplied: Formula (C.4) for the `"en12354"` model, Formula (C.5) for the other two. |

### FloatingFloorImprovementResult.plot()

```python
FloatingFloorImprovementResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `ΔL(f)` with the resonance and the asymptotic slope marked.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## force_pulse

```python
force_pulse(
    time: ArrayLike,
    contact_stiffness: float,
    impedance: float,
    *,
    mass: float = 0.5,
    impact_velocity: float | None = None,
) -> np.ndarray
```

Force pulse `F1(t)` of a single hammer impact (Eqs. 3.95/3.96).

Lindblad's solution of the mass-spring-dashpot of Hopkins Fig. 3.28, the
hammer mass `m` on the contact stiffness `K` in series with the floor's
driving-point impedance `Zdp`. For an **over-critical** oscillation
(`K m ≥ 4 Zdp²`) the pulse decays to zero without changing sign
(Eq. 3.95); for an **under-critical** one it is a decaying sinusoid
(Eq. 3.96) whose first positive lobe is the impact proper. Hopkins's rule
is stated in terms of the sign of the force rather than of any mechanism:
"only the initial force pulse that has zero or positive force values is
used to determine the force spectrum, with all subsequent values of F1(t)
due to the oscillations set to zero before taking the Fourier transform",
the hammer having rebounded from the plate. That truncation is applied
here, so the under-critical pulse is returned as zero beyond its
first zero crossing at `t = π/β`; it is the same truncation
[`tapping_force_spectrum`](/phonometry/reference/api/building/resilient-layers/#tapping_force_spectrum) transforms, so integrating this pulse over
`0 ≤ t ≤ π/β` reproduces that spectrum.

The over-critical pulse has no such cut and decays for all `t`; it is
evaluated in a form that stays finite over the whole 0,1 s between
impacts rather than one that overflows partway through it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `time` | Time `t` since the impact, in s (scalar or array, `≥ 0`). |
| `contact_stiffness` | Contact stiffness `K`, in N/m. |
| `impedance` | Driving-point impedance `Zdp`, in N.s/m. |
| `mass` | Hammer mass `m`, in kg (Default: 0,5). |
| `impact_velocity` | Impact velocity `vo`, in m/s (Default: [`hammer_impact_velocity`](/phonometry/reference/api/building/resilient-layers/#hammer_impact_velocity)). |

**Returns:** The force `F1(t)`, in N, with the same shape as `time`; never negative.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, or `time` contains a negative value. |

## hammer_impact_velocity

```python
hammer_impact_velocity(
    drop_height: float = 0.04,
    *,
    gravity: float = 9.81,
) -> float
```

Hammer velocity at impact `vo = √(2 g h)` (Hopkins Eq. 3.85).

The ISO tapping machine's nominal 40 mm drop gives `vo = 0,886 m/s`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `drop_height` | Drop height `h`, in m (Default: 0,04). |
| `gravity` | Acceleration of free fall `g`, in m/s² (Default: 9,81). |

**Returns:** The impact velocity `vo`, in m/s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## hammer_limiting_frequency

```python
hammer_limiting_frequency(impedance: float, *, mass: float = 0.5) -> float
```

Limiting frequency `flimit = Zdp/(2 π m)` (Hopkins Eq. 3.106).

The frequency at which the floor's driving-point impedance equals the
magnitude of the hammer's own mass impedance `|Zh| = ω m`; above it the
hammer mass, not the floor, limits the injected power, and the power input
stops rising at 3 dB per doubling of frequency. Vigran's Eq. (8.48) writes
the same frequency as `fz = 4 √(m1 B1)/(π mh)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `impedance` | Driving-point impedance `Zdp`, in N.s/m. |
| `mass` | Hammer mass `m`, in kg (Default: 0,5). |

**Returns:** The limiting frequency `flimit`, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## lining_improvement

```python
lining_improvement(
    resonance_frequency: float,
    *,
    system: LiningSystem = 'mineral_wool',
    anchors: bool = False,
    glued_area: float | None = None,
) -> LiningImprovementResult
```

Single-number ratings of an additional layer (Formulae D.3 to D.7).

For the reference situation of ISO 12354-1:2017 Annex D, a system applied
to a heavy basic wall of about 350 kg/m²:

* `system="mineral_wool"` (Formula D.3), an exterior thermal system on
  mineral wool with 40 % glued area and no anchors:
  `ΔRw = −36 lg(fo) + 82,5`, `ΔRA = −42 lg(fo) + 92,0`,
  `ΔRA,tr = −39 lg(fo) + 87,7`, each floored at −4 dB.
* `system="foam"` (Formula D.4), the same on PS, EPS or EEPS foams:
  `−33 lg(fo) + 76,0`, `−33 lg(fo) + 74,0`, `−36 lg(fo) + 77,0`,
  floored at −3 dB.
* `system="studs"` (Formula D.7), a layer on studs not directly fixed to
  the basic wall: `−20 lg(fo) + 48`, `−22 lg(fo) + 51`,
  `−24 lg(fo) + 54`, floored at −4 dB.

`anchors=True` applies Formula (D.5) for 4 to 10 anchors or battens per
m² (`0,66 ΔRw,ref − 1,2` and its two companions), and `glued_area`
applies Formula (D.6), `ΔR − 0,05 %So + 2,0`, for a glued area other than
the 40 % reference. Both corrections are applied after the floor of the
reference formula, in the order the annex states them.

The annex places the `≥ −4 dB` (or `≥ −3 dB`) floor inside Formulae
(D.3) and (D.4) and says nothing about re-applying it after (D.5) and
(D.6), so this function does not: a fully glued system on anchors can
return about −6,8 dB, below the reference floor. That is the annex read
literally, and the reason the two corrections are exposed as flags rather
than folded into the fit.

**Parameters**

| Name | Description |
| :--- | :--- |
| `resonance_frequency` | Resonance frequency `fo`, in Hz ([`lining_resonance_frequency`](/phonometry/reference/api/building/resilient-layers/#lining_resonance_frequency)). |
| `system` | `"mineral_wool"`, `"foam"` or `"studs"`. |
| `anchors` | Apply the Formula (D.5) anchor/batten correction. |
| `glued_area` | Glued area `%So` as a percentage of the element area (0 to 100), or `None` to keep the 40 % reference. Formula (D.6) corrects the glued exterior systems only, so it is rejected for `system="studs"`. |

**Returns:** A [`LiningImprovementResult`](/phonometry/reference/api/building/resilient-layers/#liningimprovementresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, `system` is unknown, or `glued_area` is out of range or combined with `system="studs"`. |

## lining_improvement_in_situ

```python
lining_improvement_in_situ(
    laboratory_improvement: float,
    resonance_frequency: float,
    base_rating_in_situ: float,
) -> float
```

Transfer a weighted lining improvement to the field (Formula D.8).

Even when the per-band improvement is invariant, its single-number rating
still depends on the basic element it sits on, so ISO 12354-1:2017
Formula (D.8) shifts the laboratory rating by `a X` with

`a = 1,35 lg(fo) − 3,5`, capped at 0, and `X = Rw,situ − 53`, clamped
to `[−10, +7]`.

The same formula applies to `ΔRw`, `ΔRA` and `ΔRA,tr`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `laboratory_improvement` | Laboratory rating `ΔRlab` measured to ISO 10140-1:2016 Annex G for the heavy basic element, in dB. |
| `resonance_frequency` | Resonance frequency `fo` of the system, in Hz. |
| `base_rating_in_situ` | Weighted sound reduction index `Rw,situ` of the basic element in the field situation, in dB. |

**Returns:** The field rating `ΔRsitu`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not finite, or `fo` is not positive. |

## lining_resonance_frequency

```python
lining_resonance_frequency(
    base_mass_per_area: float,
    lining_mass_per_area: float,
    *,
    dynamic_stiffness: float | None = None,
    cavity_depth: float | None = None,
) -> float
```

Resonance `fo` of a lining on a basic element (Formulae D.1/D.2).

Exactly one of the two branches applies:

* `dynamic_stiffness` (Formula D.1), for an insulation layer fixed
  **directly** to the basic construction, without studs or battens:
  `fo = √(s' (1/m'1 + 1/m'2))/(2 π)`.
* `cavity_depth` (Formula D.2), for a layer built on metal or wooden
  studs **not** connected to the basic element, with the cavity filled by a
  porous layer of airflow resistivity `r ≥ 5 kPa·s/m²`:
  `fo = √((0,111/d)(1/m'1 + 1/m'2))/(2 π)`, i.e. the near-isothermal
  stiffness of the filled cavity replaces `s'`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `base_mass_per_area` | Mass per unit area `m'1` of the basic structural element, in kg/m². |
| `lining_mass_per_area` | Mass per unit area `m'2` of the additional layer, in kg/m². |
| `dynamic_stiffness` | Dynamic stiffness per unit area `s'` of the insulation layer (EN 29052-1), in N/m³. |
| `cavity_depth` | Depth `d` of the stud cavity, in m. |

**Returns:** The resonance frequency `fo`, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, or if the two branches are both given or both omitted. |

## LiningImprovementResult

```python
LiningImprovementResult(
    resonance_frequency: float,
    system: LiningSystem,
    delta_rw: float,
    delta_ra: float,
    delta_ratr: float,
    anchors: bool = False,
    glued_area: float | None = None,
)
```

Single-number ratings of an additional layer (ISO 12354-1 Annex D).

**Attributes**

| Name | Description |
| :--- | :--- |
| `resonance_frequency` | Resonance frequency `fo` of the system, in Hz. |
| `system` | `"mineral_wool"`, `"foam"` (exterior systems glued to the wall, Formulae D.3/D.4) or `"studs"` (Formula D.7). |
| `delta_rw` | Improvement of the weighted sound reduction index `ΔRw`, in dB. |
| `delta_ra` | Improvement of the A-weighted rating `ΔRA`, in dB. |
| `delta_ratr` | Improvement of the traffic-weighted rating `ΔRA,tr`, in dB. |
| `anchors` | `True` when the Formula (D.5) anchor correction was applied. |
| `glued_area` | Glued area as a percentage of the element area, or `None` when the 40 % reference was kept. |

### LiningImprovementResult.plot()

```python
LiningImprovementResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the Annex D ratings against the resonance frequency.

Draws the three Annex D curves over the tabulated range with this
system's own resonance marked, the analogue of Figures D.2 and D.3.
Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### LiningImprovementResult.ratings

*property*

`(ΔRw, ΔRA, ΔRA,tr)` as a tuple, in dB.

## plate_contact_stiffness

```python
plate_contact_stiffness(
    youngs_modulus: float,
    *,
    poisson_ratio: float = 0.2,
    radius: float = 0.015,
) -> float
```

Contact stiffness of a plate material `K = 2 r E/(1 − ν²)` (Eq. 3.97).

The stiffness the hammer deforms when it lands on the bare walking surface
(Timoshenko and Goodier's Hertzian contact for a flat circular punch), as
opposed to [`covering_contact_stiffness`](/phonometry/reference/api/building/resilient-layers/#covering_contact_stiffness) for a soft covering laid on
top of it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `youngs_modulus` | Young's modulus `E` of the plate, in Pa. |
| `poisson_ratio` | Poisson's ratio `ν` of the plate (Default: 0,2, the value Hopkins Table A2 estimates for concrete and masonry). |
| `radius` | Contact radius `r`, in m (Default: 0,015). |

**Returns:** The contact stiffness `K`, in N/m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, or `\|ν\| >= 1`. |

## resilient_mount_improvement

```python
resilient_mount_improvement(
    frequencies: ArrayLike,
    *,
    impedance: float,
    mass_per_area: float,
    loss_factor: ArrayLike,
    mount_stiffness: float,
    mount_density: float,
) -> np.ndarray
```

Improvement of a floating floor on discrete resilient mounts (Vér).

Vér's two-subsystem SEA model of a walking surface carrying a reverberant
bending-wave field, connected to a heavyweight base floor by `N` mounts
per unit area of stiffness `k` each, with all transmission through the
mounts and none through the cavity. Hopkins Eq. (4.118) writes it as

`ΔL ≈ 10 lg(2,3 ρs1² cL1 h1 η1 S1 ω³/(N k²))`

where `k` is the dynamic stiffness of each mount, `N` the **number** of
mounts and `S1` the area of the walking surface. Since
`2,3 ρs1² cL1 h1 = Zdp1 ρs1` for `Zdp1 = 2,3 ρ cL h²` (Eq. 2.190), the
same expression reads `10 lg(Zdp1 ρs1 η1 ω³/(N/S1 · k²))`, which is the
form evaluated here: this function takes the mount **density** `N/S1`,
not the count.

Vigran's Eq. (8.45) is a sum of three terms, `Z1/Z2`, `m1 η1/(m2 η2)`
and `Z1 η1 N f³/(2 π m1 fo⁴)` with `fo = √(N k/m1)/(2 π)`. Only the
**third** of them is the model implemented here, and that term is
algebraically identical to Hopkins Eq. (4.118); the first two are the
low-frequency floor, negligible once the third dominates, which is the
regime Vigran states the 9 dB per octave slope for. The dominant-term form
used here therefore rises at **30 dB per decade** (9 dB per octave),
against the 40 dB per decade of a continuous resilient layer: fewer mounts,
a thicker walking surface or more internal damping all raise `ΔL`.

Vigran's simplified Eq. (8.46) inserts `Z1` into that third term and
prints the coefficient as `2/(√3 π) = 0,3676`, which is the same number
as the `2,3094/(2 π) = 0,3676` the substitution gives; the two forms
agree.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `impedance` | Driving-point impedance `Zdp1` of the walking surface, in N.s/m. |
| `mass_per_area` | Mass per unit area `ρs1` of the walking surface, in kg/m². |
| `loss_factor` | Total loss factor `η1` of the walking surface (scalar or per band). |
| `mount_stiffness` | Dynamic stiffness `k` of one mount, in N/m. |
| `mount_density` | Number of mounts per unit area `N/S1`, in 1/m² (Vigran's `N`, which is already a density). |

**Returns:** The improvement `ΔL` per band, in dB, and 0 dB at and below `fo`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## short_pulse_mean_square_force

```python
short_pulse_mean_square_force(
    frequencies: ArrayLike,
    *,
    band: BandWidth = 'third',
) -> np.ndarray
```

Band mean-square force of a short impact `F²rms = 3,9 B` (Eq. 3.92).

The limiting case in which the impact is short enough that the hammer's
momentum alone sets the force: combining `|Fn| = 2 m vo/Ti` (Eq. 3.90)
with `F²rms = |Fn|² B/(2 fi)` (Eq. 3.91) gives 3,925 B, printed as 3,9 B.
Hopkins finds it adequate for bare concrete slabs of at least 100 mm.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `band` | `"third"` (`B = 0,23 f`) or `"octave"` (`B = 0,707 f`). |

**Returns:** The band mean-square force `F²rms`, in N².

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, or `band` is unknown. |

## tapping_cut_off_frequency

```python
tapping_cut_off_frequency(
    contact_stiffness: float,
    impedance: float,
    *,
    mass: float = 0.5,
) -> float
```

Cut-off frequency `fco` of the force spectrum (Eqs. 3.101/3.102).

Above `fco` the tapping machine's force spectrum is no longer flat and
the force falls away. For an under-critical oscillation
(`K m < 4 Zdp²`, the case of a concrete slab with or without a soft
covering) it is the undamped mass-spring value `fco = √(K/m)/(2 π)`
(Eq. 3.102); for an over-critical one (a lightweight walking surface) it is
the lower root `[K/(2 Zdp) − √((K/(2 Zdp))² − K/m)]/(2 π)` (Eq. 3.101).

**Parameters**

| Name | Description |
| :--- | :--- |
| `contact_stiffness` | Contact stiffness `K`, in N/m (see [`plate_contact_stiffness`](/phonometry/reference/api/building/resilient-layers/#plate_contact_stiffness) / [`covering_contact_stiffness`](/phonometry/reference/api/building/resilient-layers/#covering_contact_stiffness)). |
| `impedance` | Driving-point impedance `Zdp` of the floor, in N.s/m (for a homogeneous plate, [`phonometry.vibration.infinite_plate_impedance`](/phonometry/reference/api/vibration/point-mobility/#infinite_plate_impedance)). |
| `mass` | Hammer mass `m`, in kg (Default: 0,5). |

**Returns:** The cut-off frequency `fco`, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite. |

## TAPPING_DROP_HEIGHT

*Constant* (`float`).

```python
TAPPING_DROP_HEIGHT = 0.04
```

## tapping_force_spectrum

```python
tapping_force_spectrum(
    frequencies: ArrayLike,
    contact_stiffness: float,
    impedance: float,
    *,
    mass: float = 0.5,
    impact_rate: float = 10.0,
    impact_velocity: float | None = None,
    band: BandWidth = 'third',
) -> TappingForceResult
```

Force spectrum of the ISO tapping machine on a floor (Hopkins 3.6.3.1).

The Fourier transform of the single-impact force pulse
([`force_pulse`](/phonometry/reference/api/building/resilient-layers/#force_pulse)), scaled by the impact repetition rate. Writing
`a = K/(2 Zdp)` and `ωo² = K/m`, the transform of Eqs. (3.95)/(3.96) is
the same rational function in both critical cases,
`F̂(ω) = vo K/(ωo² − ω² + 2 i a ω)`, multiplied for the under-critical
case by `1 + e^(−a π/β) e^(−i ω π/β)` because only the first positive
lobe (of duration `π/β`) is transformed. That truncation is what produces
the deep troughs at `n fco`, `n = 3, 5, 7` that Hopkins notes below
Fig. 4.64; they vanish once the covering's internal damping is included and
the spectrum is averaged into bands.

The transform is normalised so that the low-frequency asymptote is
`m vo/Ti` for an over-critical impact (no rebound) and `2 m vo/Ti` for
a lightly damped under-critical one (full rebound), the two limits of
Eqs. (3.99)/(3.100).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `contact_stiffness` | Contact stiffness `K`, in N/m. |
| `impedance` | Driving-point impedance `Zdp` of the floor, in N.s/m. |
| `mass` | Hammer mass `m`, in kg (Default: 0,5). |
| `impact_rate` | Impact repetition rate `fi`, in Hz (Default: 10). |
| `impact_velocity` | Impact velocity `vo`, in m/s (Default: [`hammer_impact_velocity`](/phonometry/reference/api/building/resilient-layers/#hammer_impact_velocity)). |
| `band` | `"third"` or `"octave"`, the band width of Eq. (3.91). |

**Returns:** A [`TappingForceResult`](/phonometry/reference/api/building/resilient-layers/#tappingforceresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, or `band` is unknown. |

## TAPPING_HAMMER_MASS

*Constant* (`float`).

```python
TAPPING_HAMMER_MASS = 0.5
```

## TAPPING_HAMMER_RADIUS

*Constant* (`float`).

```python
TAPPING_HAMMER_RADIUS = 0.015
```

## TAPPING_IMPACT_RATE

*Constant* (`float`).

```python
TAPPING_IMPACT_RATE = 10.0
```

## TappingForceResult

```python
TappingForceResult(
    frequencies: np.ndarray,
    peak_force: np.ndarray,
    mean_square_force: np.ndarray,
    power_input: np.ndarray,
    cut_off_frequency: float,
    limiting_frequency: float,
    over_critical: bool,
    contact_stiffness: float,
    impedance: float,
    lower_limit: float,
    upper_limit: float,
    band: str = 'third',
)
```

Force spectrum of the ISO tapping machine on one walking surface.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies `f`, in Hz. |
| `peak_force` | Magnitude of the Fourier force component `\|Fn\|`, in N (Hopkins Fig. 3.32). |
| `mean_square_force` | Band mean-square force `F²rms`, in N² (Eq. 3.91). |
| `power_input` | Power injected into the floor `Win = F²rms/Zdp`, in W (Eq. 3.103). |
| `cut_off_frequency` | Cut-off frequency `fco`, in Hz (Eqs. 3.101/3.102). |
| `limiting_frequency` | Limiting frequency `flimit`, in Hz (Eq. 3.106). |
| `over_critical` | `True` when `K m ≥ 4 Zdp²`, i.e. the hammer does not rebound. |
| `contact_stiffness` | Contact stiffness `K` used, in N/m. |
| `impedance` | Driving-point impedance `Zdp` used, in N.s/m. |
| `lower_limit` | Low-frequency asymptote `\|Fn\|lower = m vo/Ti`, in N (Eq. 3.99). |
| `upper_limit` | Low-frequency asymptote `\|Fn\|upper = 2 m vo/Ti`, in N (Eq. 3.100); 6 dB above `lower_limit` in mean square. |
| `band` | Band width used for `mean_square_force`. |

### TappingForceResult.plot()

```python
TappingForceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the force spectrum `|Fn|` with its asymptotes and `fco`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### TappingForceResult.power_input_level

*property*

Power input level `10 lg(Win/1 pW)`, in dB (Hopkins Fig. 3.33).

## weighted_floating_floor_improvement

```python
weighted_floating_floor_improvement(
    mass_per_area: float,
    dynamic_stiffness: float,
    *,
    floor: FloorType = 'screed',
) -> float
```

Weighted improvement `ΔLw` of a floating floor (Formulae C.4/C.5).

The single number that feeds the simplified prediction
([`phonometry.predicted_impact_insulation`](/phonometry/reference/api/building/building-prediction/#predicted_impact_insulation)), read directly from the
floating floor's mass per unit area and the resilient layer's dynamic
stiffness. ISO 12354-2:2017 gives it as the two nomograms of Figures C.1
and C.2 and prints the fits:

* `floor="screed"` (sand-cement or calcium-sulfate screeds, Formula C.4):
  `ΔLw = 13 lg(m') − 14,2 lg(s') + 20,8`;
* `floor="asphalt"` (asphalt or dry floating floors, Formula C.5):
  `ΔLw = (−0,21 m' − 5,45) lg(s') + 0,46 m' + 23,8`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mass_per_area` | Mass per unit area `m'` of the floating floor, in kg/m². |
| `dynamic_stiffness` | Dynamic stiffness per unit area `s'`, in N/m³. |
| `floor` | `"screed"` (Formula C.4) or `"asphalt"` (Formula C.5). |

**Returns:** The weighted improvement `ΔLw`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, or `floor` is unknown. |

## weighted_lining_improvement

```python
weighted_lining_improvement(
    resonance_frequency: float,
    base_rating: float,
) -> float
```

Weighted improvement `ΔRw` of an interior lining (Table D.1).

ISO 12354-1:2017 Table D.1 reads `ΔRw` off the lining's resonance
frequency, rounded to the centre of the one-third-octave band in which it
falls. Below 200 Hz the improvement also depends on the bare element:
`ΔRw = 74,4 − 20 lg(fo) − Rw/2`, never below 0 dB (NOTE 1). At and above
200 Hz the lining *degrades* the insulation, by 1 dB at 200 Hz down to
10 dB from 630 Hz to 1 600 Hz, recovering to 5 dB from 1 600 Hz to
5 000 Hz.

Table D.1 is stated for basic elements with `20 dB ≤ Rw ≤ 60 dB`.
Its last two rows both cover 1 600 Hz with different values; this function
takes the more conservative −10 dB there (see `docs/ERRATA.md`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `resonance_frequency` | Resonance frequency `fo` of the lining, in Hz ([`lining_resonance_frequency`](/phonometry/reference/api/building/resilient-layers/#lining_resonance_frequency)); must fall in the 30 Hz to 5 000 Hz range Table D.1 covers. |
| `base_rating` | Weighted sound reduction index `Rw` of the bare wall or floor, in dB. |

**Returns:** The weighted improvement `ΔRw`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `fo` is outside the tabulated range or an input is not finite. |
