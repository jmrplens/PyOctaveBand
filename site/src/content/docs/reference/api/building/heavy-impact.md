---
title: "building.measurement.heavy_impact"
description: "Heavy and soft impact sources: rubber ball and bang machine (ISO 16283-2:2020 Annex A, ISO 10140-5:2010 Annex F, JIS A 1418-2:2019, ISO 717-2:2020 Annex D)."
sidebar:
  label: "heavy_impact"
---

Heavy and soft impact sources: rubber ball and bang machine
(ISO 16283-2:2020 Annex A, ISO 10140-5:2010 Annex F, JIS A 1418-2:2019,
ISO 717-2:2020 Annex D).

The ISO tapping machine is a *light* impact source: five 500 g hammers dropped
from 40 mm produce a short, hard, quasi-stationary excitation whose spectrum
peaks well above 100 Hz. Real heavy impacts in a dwelling, a child jumping off
a chair or an adult walking barefoot, are slow, soft and low-frequency, and the
tapping machine says almost nothing about them. The **standard heavy impact
sources** were introduced for exactly that: a hollow silicone **rubber ball**
dropped from 1 m, and the **bang machine**, a car tyre dropped from 85 cm. Both
excite the floor with a single-peak force pulse about 20 ms long that puts most
of its energy below 125 Hz, and both are rated by a **maximum** (Fast
time-weighted) sound pressure level rather than by an energy average.

**Impact force exposure level (ISO 16283-2 Formula (A.1) = JIS A 1418-2
Formula (1)).** A heavy source is specified not by its geometry but by the
octave-band energy of its force pulse:

$$
L_{F\mathrm{E}} = 10 \log_{10}\!\left[ \frac{1}{T_\mathrm{ref}} \int_{t_1}^{t_2} \frac{F(t)^{2}}{F_0^{2}} \,dt \right] \qquad \text{dB re 1 N}
$$

with $F_0 = 1$ N, $T_\mathrm{ref} = 1$ s and
$t_2 - t_1$ the duration of the impact
force. The specification is stated **per octave band**, so the force record is
band-filtered before the integral (JIS A 1418-2:2019 Annex C puts the filter
between the force transducer and the analyser).
[`impact_force_exposure_level`](/phonometry/reference/api/building/heavy-impact/#impact_force_exposure_level) evaluates the integral over whatever record
it is given, i.e. one band at a time;
[`heavy_impact_source_limits`](/phonometry/reference/api/building/heavy-impact/#heavy_impact_source_limits) returns the printed octave-band tolerance
table and [`check_heavy_impact_source`](/phonometry/reference/api/building/heavy-impact/#check_heavy_impact_source) verifies the five band results
against it.

The two source specifications are printed identically in ISO 16283-2:2020
Table A.1 and ISO 10140-5:2010 Table F.1 (rubber ball), and in
JIS A 1418-2:2019 Tables A.2 (rubber ball, *impact force characteristic 2*) and
A.1 (bang machine, *impact force characteristic 1*):

===========  ====================  =====================
Octave (Hz)  Rubber ball LFE (dB)  Bang machine LFE (dB)
===========  ====================  =====================
31,5         39,0 +/- 1,0          47,0 +/- 1,0
63           31,0 +/- 1,5          40,0 +/- 1,5
125          23,0 +/- 1,5          22,0 +/- 1,5
250          17,0 +/- 2,0          11,5 +/- 2,0
500          12,5 +/- 2,0          5,5 +/- 2,0
===========  ====================  =====================

**Standardized maximum impact sound pressure level (ISO 16283-2 Formulae (4),
(5) and (6)).** Because the rated quantity is a *maximum* of a Fast-weighted
level and not an energy average, the receiving room cannot be corrected with
the usual $10 \log_{10}(T/T_0)$: the Fast detector only ever sees the first
`1.7275 s` worth of decay. The standard therefore uses:

$$
L'_{\mathrm{i,Fmax},V,T} = L_\mathrm{i,Fmax} + 10 \log_{10}(V/V_0) - 10 \log_{10}\!\left[ \frac{g(C)}{g(C_0)} \right]
$$

$$
C_0 = \frac{T_0}{1.7275} \tag{5}
$$

$$
C = \frac{T}{1.7275} \tag{6}
$$

with $T_0 = 0.5$ s, $V_0 = 50$ m³ and, writing Formula (4) in
the compact form used here,

$$
g(C) = \frac{C^{1/(1-C)} - C^{-1/(1-1/C)}}{1 - 1/C}
$$

`g` is the peak of the Fast-weighted response to an exponentially decaying
burst; for $T = T_0$ the bracket collapses to 1 and the correction
reduces to the pure volume term $10 \log_{10}(V/V_0)$, as it must. See
[`standardized_maximum_impact_level`](/phonometry/reference/api/building/heavy-impact/#standardized_maximum_impact_level) and
[`fast_reverberation_correction`](/phonometry/reference/api/building/heavy-impact/#fast_reverberation_correction).

**A-weighted rating (ISO 717-2:2020 Annex D, normative).** The single number is
not a shifted reference curve but an A-weighted sum (Formula (D.1)):

$$
X_\mathrm{iA,Fmax} = 10 \log_{10}\!\left( \sum_j 10^{(X_{\mathrm{i,Fmax},j} + A_j)/10} \right)
$$

over the one-third-octave bands 50 Hz to 630 Hz **or** the octave bands 63 Hz to
500 Hz, with the Table D.3 A-weighting corrections, rounded half-up to an
integer. One-third-octave measurements are rated in one-third octaves, never by
first summing them into octaves. See
[`a_weighted_maximum_impact_level`](/phonometry/reference/api/building/heavy-impact/#a_weighted_maximum_impact_level); [`heavy_impact_octave_levels`](/phonometry/reference/api/building/heavy-impact/#heavy_impact_octave_levels)
implements the separate octave-band conversion of ISO 16283-2 Formula (20),
used when the *measurement* is reported in octaves.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## a_weighted_maximum_impact_level

```python
a_weighted_maximum_impact_level(
    level: ArrayLike,
    frequency: ArrayLike | Sequence[float] | None = None,
    *,
    band: str | None = None,
) -> AWeightedMaximumImpactResult
```

A-weighted maximum impact level `XiA,Fmax` (ISO 717-2:2020, (D.1)).

$X_\mathrm{iA,Fmax} = 10 \log_{10}( \sum_j 10^{(X_{\mathrm{i,Fmax},j} + A_j)/10} )$,
rounded half-up to
an integer, over the one-third-octave bands 50 Hz to 630 Hz (12 values) or
the octave bands 63 Hz to 500 Hz (4 values), with the A-weighting
corrections of Table D.3. The same formula rates `LiA,Fmax`,
`LiA,Fmax,V,T`, `L'iA,Fmax` and `L'iA,Fmax,V,T`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | Maximum impact sound pressure levels `Xi,Fmax,j`, in dB, in ascending band order. |
| `frequency` | Optional band centre frequencies, in Hz. When given they must be exactly the rating bands of the chosen band width. |
| `band` | `"third"` or `"octave"`; inferred from the number of values (12 -> third, 4 -> octave) when omitted. |

**Returns:** An [`AWeightedMaximumImpactResult`](/phonometry/reference/api/building/heavy-impact/#aweightedmaximumimpactresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a wrong number of values or mismatched bands. |

## AWeightedMaximumImpactResult

```python
AWeightedMaximumImpactResult(
    frequencies: np.ndarray,
    band: str,
    levels: np.ndarray,
    a_weighting: np.ndarray,
    corrected: np.ndarray,
    unrounded: float,
    rating: int,
)
```

A-weighted maximum impact sound pressure level (ISO 717-2 Annex D).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz. |
| `band` | `"third"` (50-630 Hz) or `"octave"` (63-500 Hz). |
| `levels` | Input maximum levels `Xi,Fmax,j` per band, in dB. |
| `a_weighting` | Table D.3 correction `Aj` per band, in dB. |
| `corrected` | `Xi,Fmax,j + Aj` per band, in dB. |
| `unrounded` | The Formula (D.1) sum before rounding, in dB. |
| `rating` | The single-number quantity `XiA,Fmax`, rounded half-up to an integer, in dB. |

### AWeightedMaximumImpactResult.plot()

```python
AWeightedMaximumImpactResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the band levels, their A-weighted contributions and the rating.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## check_heavy_impact_source

```python
check_heavy_impact_source(
    force_exposure_level: ArrayLike,
    source: str = 'rubber_ball',
) -> HeavyImpactSourceCheck
```

Check a measured heavy impact source against its printed spectrum.

Compares the five measured octave-band impact force exposure levels with
the tolerance band of ISO 16283-2:2020 Table A.1 / ISO 10140-5:2010
Table F.1 / JIS A 1418-2:2019 Table A.2 (rubber ball) or JIS A 1418-2:2019
Table A.1 (bang machine).

**Parameters**

| Name | Description |
| :--- | :--- |
| `force_exposure_level` | Measured `LFE` in the five octave bands 31,5 Hz to 500 Hz, in dB re 1 N. |
| `source` | `"rubber_ball"` or `"bang_machine"`. |

**Returns:** A [`HeavyImpactSourceCheck`](/phonometry/reference/api/building/heavy-impact/#heavyimpactsourcecheck).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown source or a wrong number of bands. |

## fast_reverberation_correction

```python
fast_reverberation_correction(
    reverberation_time: ArrayLike,
    *,
    reference_time: float = 0.5,
) -> np.ndarray
```

Fast time-weighting reverberation correction (Formulae (4), (5), (6)).

The term $10 \log_{10}[g(C)/g(C_0)]$ subtracted in ISO 16283-2:2020
Formula (4),
with $C = T/1.7275$ (Formula (6)) and $C_0 = T_0/1.7275$
(Formula (5)).
It is the *maximum-level* counterpart of the energy-average
$10 \log_{10}(T/T_0)$:
a Fast detector never integrates more than about 1.7 s of decay, so the
correction saturates instead of growing without bound. It is exactly 0 dB
when $T = T_0$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reverberation_time` | Receiving-room reverberation time `T` per band, in seconds (> 0). |
| `reference_time` | Reference reverberation time `T0`, in seconds (Default: 0,5 s). |

**Returns:** The correction per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive reverberation time. |

## HEAVY_IMPACT_A_WEIGHTING

*Constant* (`dict`).

```python
HEAVY_IMPACT_A_WEIGHTING = {'third': {50.0: -30.3, 63.0: -26.2, 80.0: -22.4, 100.0: -19.1, 125.0: -16.2, 160.0: -13.2, 200.0: -10.8, 250.0: -8.7, 315.0: -6.6, 400.0: -4.8, 500.0: -3.2, 630.0: -1.9}, 'octave': {63.0: -26.2, 125.0: -16.2, 250.0: -8.7, 500.0: -3.2}}
```

## HEAVY_IMPACT_OCTAVE_BANDS

*Constant* (`tuple`).

```python
HEAVY_IMPACT_OCTAVE_BANDS = (31.5, 63.0, 125.0, 250.0, 500.0)
```

## heavy_impact_octave_levels

```python
heavy_impact_octave_levels(level: ArrayLike) -> np.ndarray
```

Combine one-third-octave maximum levels into octaves (Formula (20)).

$$
L'_{\mathrm{i,Fmax},V,T,\mathrm{oct}} = 10 \log_{10}\left( \sum_{n=1}^{3} 10^{L'_{\mathrm{i,Fmax},V,T,\mathrm{1/3oct},n} / 10} \right)
$$

(ISO 16283-2:2020, printed p. 19). The input length must be a multiple of
three and the bands must be in ascending order, three per octave.

Note that ISO 717-2:2020 Annex D forbids using this conversion as a route
to the single number: a one-third-octave measurement is rated in one-third
octaves with [`a_weighted_maximum_impact_level`](/phonometry/reference/api/building/heavy-impact/#a_weighted_maximum_impact_level).

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | One-third-octave levels in ascending order, in dB. |

**Returns:** The octave-band levels, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | when the length is not a positive multiple of three. |

## heavy_impact_source_limits

```python
heavy_impact_source_limits(
    source: str = 'rubber_ball',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Octave-band `LFE` tolerance band of a heavy impact source.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source` | `"rubber_ball"` or `"bang_machine"`. |

**Returns:** `(frequencies, lower, upper)`: the five octave-band centre frequencies in Hz and the `LFE` tolerance limits in dB re 1 N.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown source name. |

## heavy_impact_source_specification

```python
heavy_impact_source_specification(
    source: str = 'rubber_ball',
) -> HeavyImpactSourceSpec
```

Printed specification of a standard heavy and soft impact source.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source` | `"rubber_ball"` (ISO 16283-2 Annex A.2, ISO 10140-5 Annex F.2, JIS A 1418-2 impact force characteristic 2) or `"bang_machine"` (JIS A 1418-2 impact force characteristic 1). |

**Returns:** The [`HeavyImpactSourceSpec`](/phonometry/reference/api/building/heavy-impact/#heavyimpactsourcespec).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown source name. |

## HEAVY_IMPACT_SOURCES

*Constant* (`dict`).

```python
HEAVY_IMPACT_SOURCES = {'rubber_ball': ((39.0, 1.0), (31.0, 1.5), (23.0, 1.5), (17.0, 2.0), (12.5, 2.0)), 'bang_machine': ((47.0, 1.0), (40.0, 1.5), (22.0, 1.5), (11.5, 2.0), (5.5, 2.0))}
```

## HeavyImpactSourceCheck

```python
HeavyImpactSourceCheck(
    source: str,
    frequencies: np.ndarray,
    measured: np.ndarray,
    nominal: np.ndarray,
    tolerance: np.ndarray,
    deviation: np.ndarray,
    within_tolerance: np.ndarray,
    passed: bool,
)
```

Conformance of a measured heavy impact source to its printed spectrum.

**Attributes**

| Name | Description |
| :--- | :--- |
| `source` | `"rubber_ball"` or `"bang_machine"`. |
| `frequencies` | Octave-band centre frequencies, in Hz. |
| `measured` | Measured impact force exposure level `LFE`, in dB re 1 N. |
| `nominal` | Printed nominal `LFE` per band, in dB re 1 N. |
| `tolerance` | Printed tolerance per band, in dB. |
| `deviation` | `measured - nominal` per band, in dB. |
| `within_tolerance` | Per-band boolean mask of conforming bands. |
| `passed` | `True` when every band conforms. |

### HeavyImpactSourceCheck.plot()

```python
HeavyImpactSourceCheck.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the measured `LFE` against the printed tolerance band.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## HeavyImpactSourceSpec

```python
HeavyImpactSourceSpec(
    name: str,
    frequencies: tuple[float, ...],
    force_exposure_level: tuple[float, ...],
    tolerance: tuple[float, ...],
    drop_height: float,
    drop_height_tolerance: float,
    effective_mass: float,
    effective_mass_tolerance: float,
    restitution: float,
    restitution_tolerance: float,
    contact_time: float,
    contact_time_tolerance: float,
    description: str,
)
```

Printed specification of a standard heavy and soft impact source.

Annex A is normative in both standards, and the drop height belongs to its
requirements clause (ISO 16283-2 A.2.1: dropped in free fall from
`(100 +/- 1) cm` measured from the bottom of the ball), as do the force
exposure levels and the `20 +/- 2 ms` single-peak impact duration
(JIS A 1418-2 A.2 b)). The remaining mechanical figures, the effective mass,
the coefficient of restitution and the ball geometry, come from the
*example of construction* clauses, which are informative
(ISO 16283-2 A.2.2, JIS A 1418-2 Annex B): a source that meets the printed
spectrum by other means still conforms.

**Attributes**

| Name | Description |
| :--- | :--- |
| `name` | `"rubber_ball"` or `"bang_machine"`. |
| `frequencies` | Octave-band centre frequencies, in Hz. |
| `force_exposure_level` | Nominal `LFE` per band, in dB re 1 N. |
| `tolerance` | Tolerance on `LFE` per band, in dB. |
| `drop_height` | Free-fall drop height, in m. |
| `drop_height_tolerance` | Tolerance on the drop height, in m. |
| `effective_mass` | Effective (equivalent) mass of the source, in kg. |
| `effective_mass_tolerance` | Tolerance on the effective mass, in kg. |
| `restitution` | Coefficient of restitution. |
| `restitution_tolerance` | Tolerance on the coefficient of restitution. |
| `contact_time` | Nominal duration of the single-peak force pulse, in s (JIS A 1418-2:2019 A.2 b): `20 +/- 2 ms` for both characteristics). |
| `contact_time_tolerance` | Tolerance on the contact time, in s. |
| `description` | One-line description of the source. |

## impact_force_exposure_level

```python
impact_force_exposure_level(
    force: ArrayLike,
    sample_rate: float,
    *,
    reference_force: float = 1.0,
    reference_time: float = 1.0,
) -> float
```

Impact force exposure level `LFE` of a force pulse (Formula (A.1)).

$L_{F\mathrm{E}} = 10 \log_{10}[(1/T_\mathrm{ref}) \int F(t)^2 / F_0^2\,dt]$ dB re 1 N
(ISO 16283-2:2020 Formula (A.1) = ISO 10140-5:2010 Formula (F.2) =
JIS A 1418-2:2019 Formula (1)). The integral is taken over the whole
supplied record with the trapezoidal rule, so pass one isolated impact.

.. important::

   The specification tables are **per octave band**: JIS A 1418-2:2019
   Annex C measures the force with a band filter between the force
   transducer and the analyser, and ISO 16283-2 Table A.1 tabulates one
   `LFE` for each of the five octaves. This function integrates whatever
   record it is given, so feeding it an unfiltered pulse returns the
   *broadband* level, which is several decibels above any single band value
   and must not be compared with the table. Band-filter the force record
   first (for example with
   [`OctaveFilterBank`](/phonometry/reference/api/filters/core/#octavefilterbank)) and call this once
   per band, then hand the five results to
   [`check_heavy_impact_source`](/phonometry/reference/api/building/heavy-impact/#check_heavy_impact_source).

**Parameters**

| Name | Description |
| :--- | :--- |
| `force` | Sampled instantaneous force `F(t)`, in newtons (1-D). |
| `sample_rate` | Sampling rate of *force*, in hertz (> 0). |
| `reference_force` | Reference force `F0`, in newtons (Default: 1 N). |
| `reference_time` | Reference time interval `Tref`, in seconds (Default: 1 s). |

**Returns:** The impact force exposure level `LFE`, in dB re 1 N.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a malformed record or a non-positive parameter. |

## standardized_maximum_impact_level

```python
standardized_maximum_impact_level(
    level: ArrayLike,
    volume: float,
    reverberation_time: ArrayLike,
    *,
    frequency: ArrayLike | None = None,
    reference_time: float = 0.5,
    reference_volume: float = 50.0,
) -> StandardizedMaximumImpactResult
```

Standardized maximum impact level `L'i,Fmax,V,T` (Formulae (4)-(6)).

$L'_{\mathrm{i,Fmax},V,T} = L_\mathrm{i,Fmax} + 10 \log_{10}(V/V_0) - 10 \log_{10}[g(C)/g(C_0)]$
(ISO 16283-2:2020, definition 3.16), the field quantity used to rate a
floor excited by the rubber ball. The reverberation term is
[`fast_reverberation_correction`](/phonometry/reference/api/building/heavy-impact/#fast_reverberation_correction).

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | Energy-averaged maximum impact sound pressure level `Li,Fmax` per band, in dB. |
| `volume` | Receiving-room volume `V`, in m3 (> 0). |
| `reverberation_time` | Receiving-room reverberation time `T` per band, in seconds (> 0); a scalar is broadcast over the bands. |
| `frequency` | Optional band centre frequencies, in Hz. |
| `reference_time` | Reference reverberation time `T0`, in seconds (Default: 0,5 s). |
| `reference_volume` | Reference volume `V0`, in m3 (Default: 50 m3 for dwellings). |

**Returns:** A [`StandardizedMaximumImpactResult`](/phonometry/reference/api/building/heavy-impact/#standardizedmaximumimpactresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for mismatched shapes or non-positive inputs. |

## StandardizedMaximumImpactResult

```python
StandardizedMaximumImpactResult(
    frequencies: np.ndarray | None,
    measured: np.ndarray,
    standardized: np.ndarray,
    volume_term: float,
    reverberation_correction: np.ndarray,
    volume: float,
    reverberation_time: np.ndarray,
)
```

Standardized maximum impact sound pressure level (ISO 16283-2 3.16).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz, or `None`. |
| `measured` | Energy-averaged maximum level `Li,Fmax` per band, in dB. |
| `standardized` | `L'i,Fmax,V,T` per band, in dB. |
| `volume_term` | The volume correction $10 \log_{10}(V/V_0)$, in dB (scalar). |
| `reverberation_correction` | The $10 \log_{10}[g(C)/g(C_0)]$ term subtracted per band, in dB. |
| `volume` | Receiving-room volume `V`, in m3. |
| `reverberation_time` | Receiving-room reverberation time `T` per band, in seconds. |

### StandardizedMaximumImpactResult.plot()

```python
StandardizedMaximumImpactResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `Li,Fmax` and `L'i,Fmax,V,T` per band.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.
