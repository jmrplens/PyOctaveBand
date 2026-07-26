---
title: "building.survey_insulation"
description: "Field survey method for sound insulation and service-equipment noise (ISO 10052:2021)."
sidebar:
  label: "survey_insulation"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Field survey method for sound insulation and service-equipment noise
(ISO 10052:2021).

This is the **survey (control) method**: a fast, octave-band field procedure
for dwellings and rooms of comparable size (up to 150 m³). It trades the
resolution of the ISO 16283 engineering method ([`phonometry.insulation`](/phonometry/reference/api/building/insulation/))
for speed: a single hand-held integrating sound level meter swept through the
room. It measures airborne and impact sound insulation between rooms, façade
sound insulation, and the sound pressure level from building service equipment.
Clause references and the reverberation-index table follow the 2021 edition;
the formulas and that table are identical in the harmonized
EN ISO 10052:2004+A1:2010.

**Reverberation index (Clause 3.3).** The correction for the receiving room is
carried by a single quantity, the reverberation index `k = 10 lg(T/T0)` dB
with the reference reverberation time `T0 = 0,5 s`. It may be **measured** (pass
the reverberation time `T` per band to [`reverberation_index`](/phonometry/reference/api/building/survey-insulation/#reverberation_index)) or,
in a control survey, **estimated** from the room type and volume with
[`estimate_reverberation_index`](/phonometry/reference/api/building/survey-insulation/#estimate_reverberation_index) (Clause 6.5, Tables 3 and 4).

**Airborne between rooms (Clauses 3.2-3.6).** From the source- and
receiving-room levels `L1` and `L2` the level difference `D = L1 - L2`
(Clause 3.2) gives the standardized level difference `DnT = D + k`
(Clause 3.4), the normalized level difference `Dn = D + k +
10 lg(A0 T0 / (0,16 V))` (Clause 3.5, `A0 = 10 m²`) and, when a common
partition area `S` is given, the apparent sound reduction index
`R' = D + k + 10 lg(S T0 / (0,16 V))` (Clause 3.6). Where `V/7,5 > S` the
value `V/7,5` is used for `S`, with `V` the smaller room.

**Impact (Clauses 3.7-3.9).** From the impact level `Li` (Clause 3.7,
energy-averaged over the tapping-machine positions; the 2021 edition also
admits the heavy/soft impact source of Clause 3.10 with the maximum level
`Li,Fmax` of Clause 3.11) the standardized impact level `L'nT = Li - k`
(Clause 3.8) and the normalized impact level `L'n = Li - k -
10 lg(A0 T0 / (0,16 V))` (Clause 3.9).

**Façade (Clauses 3.13-3.15).** From the outdoor level 2 m in front of the
façade `L1,2m` and the receiving-room level `L2` the façade level
difference `D2m = L1,2m - L2` (Clause 3.13), the standardized
`D2m,nT = D2m + k` (Clause 3.14) and the normalized
`D2m,n = D2m + k + 10 lg(A0 T0 / (0,16 V))` (Clause 3.15).

**Service equipment (Clauses 3.16-3.18).** From three A- or C-weighted sound
pressure levels (one near a room corner, two in the reverberant field) the
service-equipment level `LXY = 10 lg[(1/3) sum 10^(0,1 LXY,i)]`
(Clause 3.16), its standardized form `LXY,nT = LXY - k` (Clause 3.17) and
normalized form `LXY,n = LXY - k - 10 lg(A0 T0 / (0,16 V))` (Clause 3.18).

**Frequency range (Clause 6.4).** Airborne and tapping-machine impact
quantities are measured in octave bands 125 Hz to 2000 Hz (5 bands); the
heavy/soft impact source uses 63 Hz to 500 Hz. The single-number weighted
ratings reuse the verified [`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating) (ISO 717-1) and
[`phonometry.weighted_impact_rating`](/phonometry/reference/api/building/insulation/#weighted_impact_rating) (ISO 717-2) engines, formed only when
exactly 5 octave (or 16 one-third-octave) values are supplied. No background
correction is applied (Clause 6.2).

## estimate_reverberation_index

```python
estimate_reverberation_index(
    volume: float,
    room: str,
    *,
    weighted: bool = False,
) -> np.ndarray | float
```

Estimate the reverberation index from room type and volume (Clause 6.5).

In a control survey the reverberation time need not be measured: the
reverberation index `k` may be read from ISO 10052:2021 Table 4 (Table 3
of EN ISO 10052:2004+A1:2010) by classifying the room. Furnished rooms use
the `room` categories `"kitchen"` / `"bathroom"` (tabulated only for
`V < 35`) and `"furnished"` (a general furnished living/sleeping room);
unfurnished rooms use the Table 3 (2004: Table 2) construction letters
`"a"`-`"h"` and the area-averaged mixed classes `"a+e"`, `"b+f"`,
`"c+g"` and `"d+h"`. The table is valid for `T0 = 0,5 s` and rooms up
to 150 m³.

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Receiving-room volume `V`, in m³ (0 \< V \<= 150). |
| `room` | Room-type key (see above). |
| `weighted` | When `True` return the single A-/C-weighted index (the Table 4 `A, C` column) instead of the five octave-band values, for globally weighted service-equipment noise (Clause 3.17). |

**Returns:** The reverberation index `k` per octave band (125-2000 Hz), in dB, or the scalar A-/C-weighted index when `weighted` is `True`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `volume` is not in `(0, 150]`, or `room` is not tabulated for that volume range. |

## reverberation_index

```python
reverberation_index(
    t: Sequence[float] | np.ndarray,
    *,
    t0: float = 0.5,
) -> np.ndarray
```

Reverberation index `k = 10 lg(T/T0)` (ISO 10052:2021, Clause 3.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `t` | Receiving-room reverberation time per band, in seconds. |
| `t0` | Reference reverberation time `T0`, in seconds (Default: 0,5 s). |

**Returns:** The reverberation index `k` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `t`/`t0` are not positive or `t` is non-finite. |

## survey_airborne_insulation

```python
survey_airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    reverberation_index: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
    area: float | None = None,
) -> SurveyAirborneResult
```

Airborne sound insulation between rooms, survey method (ISO 10052:2021).

Computes, per octave band, the level difference `D = L1 - L2`
(Clause 3.2), the standardized level difference `DnT = D + k`
(Clause 3.4) and, when `volume` is given, the normalized level
difference `Dn = D + k + 10 lg(A0 T0 / (0,16 V))` (Clause 3.5). When a
common-partition `area` is also given, the apparent sound reduction index
`R' = D + k + 10 lg(S T0 / (0,16 V))` (Clause 3.6) is formed, using
`V/7,5` for `S` where that exceeds the given area (Clause 3.6). The
reverberation index `k` comes from [`reverberation_index`](/phonometry/reference/api/building/survey-insulation/#reverberation_index) (measured
`T`) or a Clause 6.5 estimate.

`l1` and `l2` may be one value per band or a two-dimensional
`(positions, bands)` array (energy-averaged over the positions).

**Parameters**

| Name | Description |
| :--- | :--- |
| `l1` | Source-room sound pressure levels, in dB. |
| `l2` | Receiving-room sound pressure levels, in dB. |
| `reverberation_index` | Reverberation index `k` per band, in dB. |
| `volume` | Receiving-room volume `V`, in m³ (required for `Dn` and `R'`). |
| `area` | Common-partition area `S`, in m² (with `volume`, gives `R'`). |

**Returns:** [`SurveyAirborneResult`](/phonometry/reference/api/building/survey-insulation/#surveyairborneresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If band counts differ, if `area` is given without `volume`, or if `area`/`volume` are not positive. |

## survey_facade_insulation

```python
survey_facade_insulation(
    l1_2m: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    reverberation_index: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
) -> SurveyFacadeResult
```

Façade sound insulation, survey method (ISO 10052:2021).

Computes, per octave band, the façade level difference
`D2m = L1,2m - L2` (Clause 3.13), the standardized façade level
difference `D2m,nT = D2m + k` (Clause 3.14) and, when `volume` is
given, the normalized façade level difference
`D2m,n = D2m + k + 10 lg(A0 T0 / (0,16 V))` (Clause 3.15).

**Parameters**

| Name | Description |
| :--- | :--- |
| `l1_2m` | Outdoor sound pressure levels 2 m in front of the façade, in dB (one value per band or `(positions, bands)`). |
| `l2` | Receiving-room sound pressure levels, in dB. |
| `reverberation_index` | Reverberation index `k` per band, in dB. |
| `volume` | Receiving-room volume `V`, in m³ (required for `D2m,n`). |

**Returns:** [`SurveyFacadeResult`](/phonometry/reference/api/building/survey-insulation/#surveyfacaderesult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If band counts differ or `volume` is not positive. |

## survey_impact_insulation

```python
survey_impact_insulation(
    li: Sequence[float] | np.ndarray,
    reverberation_index: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
) -> SurveyImpactResult
```

Impact sound insulation between rooms, survey method (ISO 10052:2021).

Computes, per octave band, the energy-average impact sound pressure level
`Li` (Clause 3.7), the standardized impact level `L'nT = Li - k`
(Clause 3.8) and, when `volume` is given, the normalized impact level
`L'n = Li - k - 10 lg(A0 T0 / (0,16 V))` (Clause 3.9).

`li` may be one value per band or a two-dimensional
`(positions, bands)` array (energy-averaged over the tapping-machine
positions, Clause 3.7).

**Parameters**

| Name | Description |
| :--- | :--- |
| `li` | Impact sound pressure levels, in dB. |
| `reverberation_index` | Reverberation index `k` per band, in dB. |
| `volume` | Receiving-room volume `V`, in m³ (required for `L'n`). |

**Returns:** [`SurveyImpactResult`](/phonometry/reference/api/building/survey-insulation/#surveyimpactresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If band counts differ or `volume` is not positive. |

## survey_service_equipment_level

```python
survey_service_equipment_level(
    measurements: Sequence[float] | np.ndarray,
    reverberation_index: float | Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
) -> SurveyServiceEquipmentResult
```

Service-equipment sound pressure level, survey method (ISO 10052:2021).

Computes the service-equipment level `LXY = 10 lg[(1/3) sum 10^(0,1
LXY,i)]` (Clause 3.16) as the energy average of the three measurement
positions (one near a corner, two in the reverberant field, Clause 6.3.3),
the standardized level `LXY,nT = LXY - k` (Clause 3.17) and, when
`volume` is given, the normalized level `LXY,n = LXY - k -
10 lg(A0 T0 / (0,16 V))` (Clause 3.18). `X` is the frequency weighting
(A or C) and `Y` the time weighting (F, S or Leq).

**Parameters**

| Name | Description |
| :--- | :--- |
| `measurements` | The three A- or C-weighted levels, in dB; either three scalars, or a `(3, bands)` array for a banded analysis. |
| `reverberation_index` | Reverberation index `k`, in dB; a scalar for a weighted level, or one value per band; for a global weighted level `k` is taken from the mean of the 500/1000/2000 Hz octave reverberation times (Clause 3.17). |
| `volume` | Receiving-room volume `V`, in m³ (required for `LXY,n`). |

**Returns:** [`SurveyServiceEquipmentResult`](/phonometry/reference/api/building/survey-insulation/#surveyserviceequipmentresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If not exactly three measurements are given, if shapes are inconsistent, or if `volume` is not positive. |

## SurveyAirborneResult

```python
SurveyAirborneResult(
    d: np.ndarray,
    d_nt: np.ndarray,
    d_n: np.ndarray | None,
    r_prime: np.ndarray | None,
    rating: WeightedRatingResult | None,
    r_prime_rating: WeightedRatingResult | None,
)
```

Per-band airborne sound insulation, survey method (ISO 10052).

**Attributes**

| Name | Description |
| :--- | :--- |
| `d` | Level difference `D = L1 - L2` per band, in dB (Clause 3.2). |
| `d_nt` | Standardized level difference `DnT = D + k` (Clause 3.4). |
| `d_n` | Normalized level difference `Dn` (Clause 3.5), or `None` when the receiving-room volume was not supplied. |
| `r_prime` | Apparent sound reduction index `R'` (Clause 3.6), or `None` when the partition area and volume were not both supplied. |
| `rating` | Weighted standardized level difference `DnT,w` with `C` / `Ctr` (ISO 717-1), or `None` off the 5/16-band count. |
| `r_prime_rating` | Weighted apparent sound reduction index `R'w`, or `None` when `r_prime` is unavailable or off the band count. |

### SurveyAirborneResult.plot()

```python
SurveyAirborneResult.plot(ax: Axes | None = None, **kwargs: Any) -> Axes
```

Plot `DnT` against the shifted ISO 717-1 reference curve.

Delegates to the weighted-rating plot. Requires the automatic rating
(5 octave or 16 one-third-octave bands) and matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

### SurveyAirborneResult.report()

```python
SurveyAirborneResult.report(
    path: str,
    *,
    quantity: str = 'dnt',
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a survey-method airborne field report to a PDF (ISO 10052).

Writes the one-page survey (control) method field report: the
standard-basis line naming ISO 10052 (octave bands), an optional
metadata header block, the octave-band table beside the
measured-versus-shifted-ISO 717-1-reference curve, the boxed field
rating `DnT,w (C; Ctr)` or `R'w (C; Ctr)`, the survey-method
statement, an optional verdict row (airborne passes at or above the
requirement) and a footer with the identity block and disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `quantity` | The reported quantity: `"dnt"` (default, the standardized level difference `DnT`) or `"r_prime"` (the apparent sound reduction index `R'`; requires the result to carry `r_prime`, built with the `area` and `volume` arguments). |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table shows the ISO 717 evaluation per band instead of the two-column `f \| value` form. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` or `quantity` is unknown, or the selected quantity has no single-number rating (the survey needs 5 octave or 16 one-third-octave bands, and `R'` needs `area` and `volume`). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## SurveyFacadeResult

```python
SurveyFacadeResult(
    d_2m: np.ndarray,
    d_2m_nt: np.ndarray,
    d_2m_n: np.ndarray | None,
    rating: WeightedRatingResult | None,
)
```

Per-band façade sound insulation, survey method (ISO 10052).

**Attributes**

| Name | Description |
| :--- | :--- |
| `d_2m` | Façade level difference `D2m = L1,2m - L2` (Clause 3.13). |
| `d_2m_nt` | Standardized façade level difference `D2m,nT` (Clause 3.14). |
| `d_2m_n` | Normalized façade level difference `D2m,n` (Clause 3.15), or `None` when the receiving-room volume was not supplied. |
| `rating` | Weighted standardized façade level difference `D2m,nT,w` (ISO 717-1), or `None` off the 5/16-band count. |

### SurveyFacadeResult.plot()

```python
SurveyFacadeResult.plot(ax: Axes | None = None, **kwargs: Any) -> Axes
```

Plot `D2m,nT` against the shifted ISO 717-1 reference curve.

### SurveyFacadeResult.report()

```python
SurveyFacadeResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a survey-method facade field report to a PDF (ISO 10052).

Writes the one-page survey (control) method field facade report: the
standard-basis line naming ISO 10052 (octave bands), an optional
metadata header block, the octave-band `D2m,nT` table beside the
measured-versus-shifted-ISO 717-1-reference curve, the boxed field
rating `D2m,nT,w (C; Ctr)`, the survey-method statement, an optional
verdict row (a facade level difference passes at or above the
requirement) and a footer with the identity block and disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table shows the ISO 717 evaluation per band instead of the two-column `f \| value` form. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown or the result has no single-number rating (the survey needs 5 octave or 16 one-third-octave bands). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## SurveyImpactResult

```python
SurveyImpactResult(
    l_i: np.ndarray,
    l_nt: np.ndarray,
    l_n: np.ndarray | None,
    rating: ImpactRatingResult | None,
)
```

Per-band impact sound insulation, survey method (ISO 10052).

**Attributes**

| Name | Description |
| :--- | :--- |
| `l_i` | Energy-average impact sound pressure level `Li` per band, in dB (Clause 3.7). |
| `l_nt` | Standardized impact level `L'nT = Li - k` (Clause 3.8). |
| `l_n` | Normalized impact level `L'n` (Clause 3.9), or `None` when the receiving-room volume was not supplied. |
| `rating` | Weighted standardized impact level `L'nT,w` with `CI` (ISO 717-2), or `None` off the 5/16-band count. |

### SurveyImpactResult.plot()

```python
SurveyImpactResult.plot(ax: Axes | None = None, **kwargs: Any) -> Axes
```

Plot `L'nT` against the shifted ISO 717-2 reference curve.

### SurveyImpactResult.report()

```python
SurveyImpactResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a survey-method impact field report to a PDF (ISO 10052).

Writes the one-page survey (control) method field report: the
standard-basis line naming ISO 10052 (octave bands, tapping machine),
an optional metadata header block, the octave-band `L'nT` table
beside the measured-versus-shifted-ISO 717-2-reference curve, the
boxed field rating `L'nT,w (CI)`, the survey-method statement, an
optional verdict row (impact passes at or below the requirement) and
a footer with the identity block and disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a lightweight fiche (body, rating and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table shows the ISO 717 evaluation per band instead of the two-column `f \| value` form. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown or the result has no single-number rating (the survey needs 5 octave or 16 one-third-octave bands). |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## SurveyServiceEquipmentResult

```python
SurveyServiceEquipmentResult(
    l_xy: np.ndarray,
    l_xy_nt: np.ndarray,
    l_xy_n: np.ndarray | None,
)
```

Service-equipment sound pressure level, survey method (ISO 10052).

**Attributes**

| Name | Description |
| :--- | :--- |
| `l_xy` | Service-equipment level `LXY` (Clause 3.16), the energy average of the three measurement positions, in dB. |
| `l_xy_nt` | Standardized level `LXY,nT = LXY - k` (Clause 3.17). |
| `l_xy_n` | Normalized level `LXY,n` (Clause 3.18), or `None` when the receiving-room volume was not supplied. |
