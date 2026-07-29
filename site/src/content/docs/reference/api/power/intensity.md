---
title: "emission.intensity"
description: "Two-microphone (p-p) sound intensity per IEC 61043:1993 and the ISO 9614-1:1993 field indicators."
sidebar:
  label: "intensity"
---

Two-microphone (p-p) sound intensity per IEC 61043:1993 and the
ISO 9614-1:1993 field indicators.

A p-p probe holds two pressure microphones a fixed distance `spacing`
(dr) apart. The mean of the two pressures is taken as the sound pressure
at the probe reference point, while the pressure differential is used to
derive the particle velocity component along the probe axis
(IEC 61043:1993, definition 3.2):

```text
p(t) = (p1(t) + p2(t)) / 2
u(t) = -(1 / (rho * dr)) * integral of (p2(t) - p1(t)) dt
I    = < p(t) * u(t) >          (time average)
```

For stationary signals the time-averaged intensity reduces to the
imaginary part of the one-sided cross spectrum G12 of the two microphone
pressures (the frequency-domain form of the same finite-difference
estimator):

```text
I(f) = -Im{G12(f)} / (2 * pi * f * rho * dr)
```

The finite-difference gradient underestimates the true plane-wave
intensity by the factor `sin(k*dr) / (k*dr)` with `k = 2*pi*f/c`;
IEC 61043:1993 clause 7.3 specifies the probe intensity response with
exactly this argument (`Ff = dr * f * 2 * pi / c`) and Table 3 lists
the resulting nominal response (e.g. -10,5 dB at 6,3 kHz for a 25 mm
separation). Below `f = 0,1 * c / dr` (k\*dr \< 0,63) the bias stays
under about 0,3 dB (factor >= 0,935).

Field indicators F1 (temporal variability), F2 (surface
pressure-intensity), F3 (negative partial power) and F4 (field
non-uniformity) follow ISO 9614-1:1993 Annex A (normative), equations
(A.1)-(A.9). F1 is measured in the initial test (clause 8.2) at one
typical position on the initial measurement surface and qualifies the
*field*, not the surface: it is the coefficient of variation of M
short-time-averaged samples of the normal intensity at that fixed
point, and Table B.3 calls for action code (e) when it exceeds 0,6.
The dynamic capability index is
`Ld = delta_pI0 - K` (ISO 9614-1 clause 3.12, equation (10)); the
instrument is adequate for a measurement when `Ld > F2` (criterion 1,
Annex B equation (B.1)). The residual index `delta_pI0` that feeds it
is classified against IEC 61043:1993 Table 2 by
[`phonometry.metrology.intensity_compliance.intensity_class_compliance`](/phonometry/reference/api/power/intensity-compliance/#intensity_class_compliance).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## dynamic_capability_index

```python
dynamic_capability_index(
    pressure_residual_intensity_index: float,
    bias_error_factor: float = 10.0,
) -> float
```

Dynamic capability index Ld (ISO 9614-1:1993, clause 3.12).

`Ld = delta_pI0 - K` (equation (10)), where `delta_pI0` is the
instrument pressure-residual intensity index (clause 3.11, equation
(9); determined per IEC 61043:1993, which requires the Table 2
minima per class) and `K` the bias error factor of Table 1: 10 dB
for precision (grade 1) and engineering (grade 2) measurements, 7 dB
for survey (grade 3). The measurement arrangement is adequate when
`Ld > F2` (criterion 1, Annex B equation (B.1)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure_residual_intensity_index` | delta_pI0 in decibels. |
| `bias_error_factor` | K in decibels (default 10,0). |

**Returns:** Ld in decibels.

## field_indicators

```python
field_indicators(
    pressure_levels: list[float] | np.ndarray,
    normal_intensity: list[float] | np.ndarray,
    frequencies: list[float] | np.ndarray | None = None,
    *,
    temporal_intensity: list[float] | np.ndarray | None = None,
) -> FieldIndicators
```

ISO 9614-1:1993 Annex A field indicators F1 to F4.

Given the sound pressure level `Lpi` (dB) and the signed normal
sound intensity `Ini` (W/m^2) measured at each of the N discrete
positions on the measurement surface:

- F2 = Lp - L|In| (equation (A.3)), with the surface pressure level
  from equation (A.4) and the level of the mean magnitude of the
  normal intensity from equation (A.5);
- F3 = Lp - LIn (equation (A.6)), with the algebraic surface
  intensity level from equation (A.7);
- F4 = (1/|mean In|) * sqrt(sum((Ini - mean In)^2) / (N - 1))
  (equations (A.8)-(A.9)).

F1, the temporal variability indicator (equation (A.1)), does not come
from the surface scan: it is evaluated in the initial test at one
typical position from M short-time-averaged intensity samples
(clause 8.2), and again immediately before and after the measurement
on any one measurement surface (Annex B, B.1.4). Passing those samples
as `temporal_intensity` fills `f1` on the result; see
[`temporal_variability_indicator`](/phonometry/reference/api/power/intensity/#temporal_variability_indicator).

The inputs are either 1D per-position arrays (one frequency band,
scalar indicators) or 2D `(positions, bands)` arrays (the
indicators are evaluated band by band and returned as per-band
arrays; the plottable form). If the algebraic mean intensity of any
band is not positive the test conditions do not satisfy ISO 9614-1
in that band (clause A.2.3) and a `ValueError` is raised.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure_levels` | Lpi at each position, in decibels; 1D `(positions,)` or 2D `(positions, bands)`. |
| `normal_intensity` | Signed normal intensity Ini at each position, in W/m^2; same shape as `pressure_levels`. |
| `frequencies` | Band centre frequencies in Hz, one per column of the 2D input (optional for 1D input, where it is ignored beyond a length check). |
| `temporal_intensity` | Optional M short-time-averaged normal intensity samples `Ink` at one fixed position, in W/m^2; 1D `(samples,)` for a single band or 2D `(samples, bands)` with one column per band of the surface input. Supplying it fills `f1` on the returned result. |

**Returns:** [`FieldIndicators`](/phonometry/reference/api/power/intensity/#fieldindicators).

## FieldIndicators

```python
FieldIndicators(
    f2: float | np.ndarray,
    f3: float | np.ndarray,
    f4: float | np.ndarray,
    frequency: np.ndarray | None = None,
    f1: float | np.ndarray | None = None,
)
```

ISO 9614-1:1993 Annex A field indicators over a measurement surface.

`f2` is the surface pressure-intensity indicator (equation (A.3)),
`f3` the negative partial power indicator (equation (A.6)) and
`f4` the field non-uniformity indicator (equation (A.8)).
`f3 - f2 > 0` reveals negative partial power flowing through parts
of the surface. The instrument's dynamic capability index must
satisfy `Ld > f2` (criterion 1, equation (B.1)); the number of
positions N must satisfy `N > C * f4**2` (criterion 2, equation
(B.2)).

`f1` is the temporal variability indicator (equation (A.1)), present
only when [`field_indicators`](/phonometry/reference/api/power/intensity/#field_indicators) was given the `temporal_intensity`
samples it is computed from; it is `None` otherwise. Unlike the other
three it describes the *field* rather than the surface: it is evaluated
at one typical position from M short-time-averaged samples of the
normal intensity (clause 8.2, the initial test), and Table B.3
requires action when it exceeds [`TEMPORAL_VARIABILITY_LIMIT`](/phonometry/reference/api/power/intensity/#temporal_variability_limit).

With per-position *and* per-band input (2D arrays passed to
[`field_indicators`](/phonometry/reference/api/power/intensity/#field_indicators)) the indicators are per-band arrays and
`frequency` carries the band centres; with 1D per-position input they
are scalars and `frequency` is `None`.

### FieldIndicators.field_is_stationary()

```python
FieldIndicators.field_is_stationary(limit: float = 0.6) -> bool | np.ndarray
```

Whether the temporal variability stays within the Table B.3 limit.

ISO 9614-1:1993 Table B.3 lists `F1 > 0,6` as the condition that
calls for action code (e), i.e. reducing the temporal variability of
the extraneous intensity, measuring during quieter periods or
lengthening the averaging time at each position. A field at exactly
the limit is not flagged.

**Parameters**

| Name | Description |
| :--- | :--- |
| `limit` | The Table B.3 threshold (default [`TEMPORAL_VARIABILITY_LIMIT`](/phonometry/reference/api/power/intensity/#temporal_variability_limit), 0,6). |

**Returns:** `True`/`False` for a scalar `f1`, or a boolean array per band for a per-band `f1`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result carries no `f1` (call [`field_indicators`](/phonometry/reference/api/power/intensity/#field_indicators) with `temporal_intensity`, or use [`temporal_variability_indicator`](/phonometry/reference/api/power/intensity/#temporal_variability_indicator) directly). |

### FieldIndicators.plot()

```python
FieldIndicators.plot(
    ax: Axes | None = None,
    *,
    dynamic_capability: float | np.ndarray | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band indicators F2/F3, the Ld line and F4.

F1 and its Table B.3 limit of 0,6 are drawn beside F4 when the
result carries them, that is when `temporal_intensity` was
supplied to [`field_indicators`](/phonometry/reference/api/power/intensity/#field_indicators).

Requires per-band data (call [`field_indicators`](/phonometry/reference/api/power/intensity/#field_indicators) with 2D
`(positions, bands)` arrays and `frequencies`) and matplotlib
(`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `dynamic_capability` | Optional instrument dynamic capability index `Ld` in dB (scalar or per band), drawn as the criterion-1 reference line (`Ld > F2`, equation (B.1)); see [`dynamic_capability_index`](/phonometry/reference/api/power/intensity/#dynamic_capability_index). |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the F2 curve `plot` call. |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result carries no per-band data. |

## IntensityResult

```python
IntensityResult(
    frequency: np.ndarray | None,
    intensity: np.ndarray | None,
    intensity_level: np.ndarray | None,
    pressure_level: np.ndarray | None,
    pressure_intensity_index: np.ndarray | None,
    direction: np.ndarray | None,
    bias_correction: np.ndarray | None,
    total_intensity: float,
    total_intensity_level: float,
    total_pressure_level: float,
    total_pressure_intensity_index: float,
    total_direction: int,
    max_valid_frequency: float,
    spacing: float | None = None,
)
```

Result of a p-p sound intensity measurement.

Per-band arrays are `None` unless a band `fraction` was requested.
`intensity` is signed (positive along the probe axis, from
microphone 1 towards microphone 2); `intensity_level` is computed
from the magnitude, `10*lg(|I|/1e-12)` dB, with the sign reported
separately in `direction` (+1/-1). `pressure_intensity_index` is
`Lp - LI` (the single-position form of the ISO 9614-1:1993 F2
indicator, equation (A.3)). `bias_correction` is the multiplicative
factor `(k*dr)/sin(k*dr)` compensating the finite-difference
underestimation at each band centre (IEC 61043:1993, 7.3); it is NaN
at and beyond the first null `k*dr >= pi`. `max_valid_frequency`
is the usable-bandwidth bound `0,1*c/spacing` (bias \< ~0,3 dB).
`spacing` retains the microphone separation the measurement was
reduced with so `plot_geometry` can draw the probe; it is
appended after the original fields and `None` for hand-built
results.

### IntensityResult.plot()

```python
IntensityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot Lp vs LI per band with the pressure-intensity index.

Requires per-band data (call `sound_intensity(..., fraction=...)`)
and matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### IntensityResult.plot_geometry()

```python
IntensityResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the p-p probe with its spacer to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

## plot_pp_probe_geometry

```python
plot_pp_probe_geometry(
    spacing: float = 0.012,
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the face-to-face p-p intensity probe to scale.

Two phase-matched microphones separated by the solid spacer, with the
intensity axis through both; default is the classic 12 mm spacer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `spacing` | Microphone separation `dr`, in metres. |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the spacer rectangle. |

**Returns:** The axes.

## sound_intensity

```python
sound_intensity(
    p1: list[float] | np.ndarray,
    p2: list[float] | np.ndarray,
    fs: int,
    spacing: float,
    rho: float = 1.204,
    c: float = 343.0,
    fraction: int | None = None,
    limits: list[float] | None = None,
    bias_correct: bool = False,
) -> IntensityResult
```

Sound intensity from a two-microphone (p-p) probe (IEC 61043:1993).

The one-sided cross spectrum `G12` of the two microphone pressures
is estimated with Welch-averaged, Hann-windowed segments
(`scipy.signal.csd`) and converted to the intensity spectral
density along the probe axis (definition 3.2 of IEC 61043:1993 gives
the underlying p-p formulation):

```text
I(f) = -Im{G12(f)} / (2 * pi * f * rho * spacing)
```

Positive intensity flows from microphone 1 towards microphone 2.
The mean-square pressure is taken from the mean signal
`(p1 + p2)/2` at the probe reference point. When `fraction` is
given, both quantities are integrated into octave (1) or one-third
octave (3) bands using the ANSI S1.11/IEC 61260-1 band edges of
[`phonometry.nominal_frequencies`](/phonometry/reference/api/filters/frequencies/#nominal_frequencies); bands without any spectral
bin are dropped. Broadband totals are always computed (over
`limits` when provided, otherwise over all positive frequencies).

The pressure-intensity index `Lp - LI` is reported per band and
broadband; in a free plane progressive wave it equals
`10*lg(rho*c/400)` = 0,14 dB (IEC 61043:1993 clause 5 note), while
large values flag reactive or noisy fields (compare with the
instrument dynamic capability, ISO 9614-1:1993 criterion 1).

Usable bandwidth: the finite-difference gradient biases the result
by the factor `sin(k*spacing)/(k*spacing)` (IEC 61043:1993, 7.3);
results are essentially unbiased (\< ~0,3 dB) below
`max_valid_frequency = 0,1 * c / spacing`, and `bias_correction`
provides the per-band compensation factor.

**Parameters**

| Name | Description |
| :--- | :--- |
| `p1` | Pressure signal of microphone 1, in pascals (1D). |
| `p2` | Pressure signal of microphone 2, in pascals (1D). |
| `fs` | Sample rate in Hz. |
| `spacing` | Microphone separation dr, in metres. |
| `rho` | Air density, in kg/m^3. Default 1,204 (20 degC). |
| `c` | Speed of sound, in m/s. Default 343,0. |
| `fraction` | `None` (broadband only), 1 (octave bands) or 3 (one-third octave bands). |
| `limits` | [f_min, f_max] band limits in Hz (default [12, 20000], as in [`phonometry.nominal_frequencies`](/phonometry/reference/api/filters/frequencies/#nominal_frequencies)). |
| `bias_correct` | If True, apply the per-bin finite-difference correction `(k*spacing)/sin(k*spacing)` (IEC 61043:1993, 7.3) to the intensity spectral density before summing the band and broadband totals, so the totals no longer under-read as the frequency approaches `max_valid_frequency`. The reciprocal diverges as `k*spacing -> pi` (the first spatial-aliasing null at `c/(2*spacing)`, inside the default band range for close spacings), so it is applied only over the probe's usable range (up to `k*spacing = pi/2`) and held constant beyond, keeping the totals bounded instead of letting a few near-null bins dominate them. Default False keeps the exact legacy totals; the per-band `bias_correction` factor (same clamped definition) is reported either way. |

**Returns:** [`IntensityResult`](/phonometry/reference/api/power/intensity/#intensityresult).

## temporal_variability_indicator

```python
temporal_variability_indicator(
    short_time_intensity: list[float] | np.ndarray,
) -> float | np.ndarray
```

ISO 9614-1:1993 temporal variability indicator F1 (equation (A.1)).

In the initial test a "typical" measurement position is chosen on an
initial measurement surface and the normal sound intensity is sampled
there M times with a short averaging time (clause 8.2). The indicator
is the coefficient of variation of those samples:

```text
F1 = (1 / In) * sqrt( sum_k (Ink - In)**2 / (M - 1) )
```

with `In` the mean of the M samples (equation (A.2)). It is
dimensionless, and it is zero for a perfectly steady field, so it
qualifies the *stationarity of the field*, not the uniformity of the
surface (that is F4). Note 9 of Annex A recommends M = 10 samples, with
a short averaging time of 8 s to 12 s (or a whole number of cycles) for
periodic signals. Table B.3 requires corrective action when
`F1 > 0,6` ([`TEMPORAL_VARIABILITY_LIMIT`](/phonometry/reference/api/power/intensity/#temporal_variability_limit)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `short_time_intensity` | The M short-time-averaged signed normal intensity samples `Ink` at the fixed position, in W/m^2; 1D `(samples,)` for one frequency band, or 2D `(samples, bands)` to evaluate every band at once (Annex A.1 requires F1 in each band used for the sound-power determination). |

**Returns:** F1 as a float for 1D input, or a per-band `numpy.ndarray` for 2D input.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If fewer than two samples are supplied, the input is not 1D or 2D, or the mean intensity of a band is not positive. A.2.1 states no positivity condition on the M samples, but F1 normalizes by that mean, so a non-positive one leaves the indicator meaningless and the Table B.3 criterion vacuous; this library rejects it rather than return a negative F1. |

## TEMPORAL_VARIABILITY_LIMIT

*Constant* (`float`).

```python
TEMPORAL_VARIABILITY_LIMIT = 0.6
```
