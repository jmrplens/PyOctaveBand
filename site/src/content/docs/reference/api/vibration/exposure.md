---
title: "vibration.human.exposure"
description: "Human exposure to whole-body and hand-transmitted vibration."
sidebar:
  label: "exposure"
---

Human exposure to whole-body and hand-transmitted vibration.

The measurement chain of the ISO human-vibration family is implemented from the
standards' own analog definitions, clean-room:

* **ISO 8041-1:2017** - the authoritative *master* definition of every
  frequency weighting.  A single cascade
  $H(s) = H_\mathrm{h}(s) H_\mathrm{l}(s) H_\mathrm{t}(s) H_\mathrm{s}(s)$
  (Formula (5)) of second-order band-limiting Butterworth sections
  (Formulae (1)/(2)), an acceleration-velocity transition (Formula (3)) and an
  upward step (Formula (4)) realises all nine weightings from the one Table 3
  parameter set: `Wb, Wc, Wd, We, Wf, Wh, Wj, Wk, Wm`.  The tabulated
  design-goal factors of Annex B (Tables B.1-B.9) are reproduced to their
  four-significant-figure precision.

* **ISO 2631-1:1997** - whole-body vibration: the weighted r.m.s. acceleration
  `a_w` (Eq. (1)/(9)), the vibration total value `a_v` with axis
  multiplying factors `k` (Eq. (10)), the running r.m.s. and maximum
  transient vibration value `MTVV` (Eqs. (2)-(4)), the vibration dose value
  `VDV` (Eq. (5)), the crest factor (6.2.1) and the energy-equivalent
  magnitude relations of Annex B (Eqs. (B.1)-(B.3)).

* **ISO 2631-2:2003** - whole-body vibration in buildings: the direction-
  independent weighting `Wm` (Annex A).

* **ISO 2631-4:2001** - ride comfort in fixed-guideway (rail) transport: the
  vertical weighting `Wb` (Annex A) with the `Wk` axis multiplying
  factors.  `Wb` is realised here from its exact ISO 8041-1 Table 3
  parameters (of which ISO 2631-4 Table A.1 is the two-decimal rounding).

* **ISO 5349-1:2001 / ISO 5349-2:2001** - hand-transmitted vibration: the
  vibration total value `a_hv` (5349-1 Eq. (1)), the daily exposure `A(8)`
  for single and multiple operations (5349-1 Eqs. (2)/(3); 5349-2 Eqs. (1)-(3))
  and the vibration-white-finger dose relation of 5349-1 Annex C (Eq. (C.1)).

* **Directive 2002/44/EC** - the daily exposure action and limit values
  (Article 3) that ISO does not fix: hand-arm `A(8)` EAV `2.5` /
  ELV `5` m/s2; whole-body `A(8)` EAV `0.5` / ELV `1.15` m/s2 (or VDV
  EAV `9.1` / ELV `21` m/s^1.75).  The hand-arm `A(8)` is based on the
  ISO 5349-1 vector total `a_hv` (Annex, Part A, point 1); the whole-body
  `A(8)` is based on the *highest* of the frequency-weighted axis values
  $1.4 a_{\mathrm{w}x}$, $1.4 a_{\mathrm{w}y}$, $a_{\mathrm{w}z}$ (Annex, Part B,
  point 1; see [`wbv_exposure_basis`](/phonometry/reference/api/vibration/exposure/#wbv_exposure_basis)), not on the ISO 2631-1 Eq. (10)
  vector total.

The band (spectrum) method and the exposure arithmetic carry the standards'
worked-example oracles; the time-domain metrics operate on a weighted
acceleration signal, which [`apply_weighting`](/phonometry/reference/api/vibration/exposure/#apply_weighting) produces from a raw record
by applying the exact analog response of ISO 8041-1 in the frequency domain.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## apply_weighting

```python
apply_weighting(
    signal: SignalInput,
    fs: float | None = None,
    *,
    name: str,
) -> Real
```

Apply frequency weighting `name` to a time signal (ISO 8041-1).

The exact analog response [`frequency_weighting`](/phonometry/reference/api/vibration/exposure/#frequency_weighting) is applied in the
frequency domain (real FFT), so the weighted signal reproduces both the
magnitude and phase of the standard's cascade without bilinear warping.
The multiplication is circular, so the record wraps at its ends; apply it
to a record long enough (or pre-tapered) that the boundary transient is
negligible, as for any block frequency-domain filtering.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Unweighted acceleration time history (1-D), in m/s2. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) for its rate; a calibration factor it carries is deliberately not applied, because this quantity is an acceleration in m/s2 and not a pressure. |
| `fs` | Sampling frequency, in hertz (> 0). Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `name` | Weighting name (one of [`WEIGHTING_NAMES`](/phonometry/reference/api/vibration/exposure/#weighting_names)). |

**Returns:** The frequency-weighted acceleration signal, same length as input.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `signal` is not 1-D, `fs` is not positive, or `name` is unknown. |

## combine_partial_exposures

```python
combine_partial_exposures(partials: ArrayLike) -> float
```

Combine partial exposures into `A(8)` (ISO 5349-1/-2 Eq. (3)).

$A(8) = \sqrt{\sum_i A_i(8)^2}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `partials` | Partial exposures $A_i(8)$, in m/s2. |

**Returns:** The combined daily exposure `A(8)`, in m/s2.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `partials` is empty. |

## crest_factor

```python
crest_factor(signal: SignalInput) -> float
```

Crest factor of a weighted signal (ISO 2631-1 clause 6.2.1).

The modulus of the ratio of the peak weighted acceleration to its r.m.s.
value.  ISO 2631-1 6.2.2 deems the basic (r.m.s.) method adequate for a
crest factor up to 9.

Emits a [`HumanVibrationWarning`](/phonometry/reference/api/vibration/exposure/#humanvibrationwarning) when the crest factor exceeds 9,
the threshold above which ISO 2631-1 6.2.2 deems the basic method
inadequate.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Frequency-weighted acceleration signal (1-D), in m/s2. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), of which only the samples are read: a peak-to-r.m.s. ratio needs no sample rate, and a calibration factor the object carries is deliberately not applied, because this quantity is an acceleration in m/s2 and not a pressure. |

**Returns:** The crest factor (dimensionless); `0` for an all-zero signal.

## daily_exposure

```python
daily_exposure(total_value: float, duration_s: float) -> float
```

Daily exposure `A(8)` for one operation (ISO 5349-1 Eq. (2)).

$A(8) = a_\mathrm{hv} \sqrt{T / T_0}$ with $T_0 = 8$ h.  The
identical form gives the whole-body `A(8)` of Directive 2002/44/EC,
whose magnitude is the Annex Part B dominant-axis value (see
[`wbv_exposure_basis`](/phonometry/reference/api/vibration/exposure/#wbv_exposure_basis)) rather than a vector total.

**Parameters**

| Name | Description |
| :--- | :--- |
| `total_value` | Vibration total value `a_hv` (or `a_v`), in m/s2. |
| `duration_s` | Daily exposure duration `T`, in seconds. |

**Returns:** The daily exposure `A(8)`, in m/s2.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-finite or negative magnitude or duration. |

## daily_vibration_exposure

```python
daily_vibration_exposure(
    total_values: ArrayLike,
    durations_s: ArrayLike,
    *,
    kind: str,
    labels: list[str] | tuple[str, ...] | None = None,
) -> DailyVibrationExposure
```

Daily exposure from several operations, assessed (ISO 5349 + Directive).

Combines the operations via the partial exposures of ISO 5349-1/-2
Eqs. (2)/(3) and assesses the resulting `A(8)` against the Directive
2002/44/EC action and limit values for `kind`.

The Directive fixes the per-operation magnitude each kind must be fed
with (Annex, points 1): for `kind="hav"` the ISO 5349-1 Eq. (1) vector
total `a_hv` (Part A); for `kind="wbv"` the *highest* frequency-
weighted axis value `max(1,4*a_wx, 1,4*a_wy, a_wz)` that
[`wbv_exposure_basis`](/phonometry/reference/api/vibration/exposure/#wbv_exposure_basis) returns (Part B), **not** the ISO 2631-1
Eq. (10) vector total `a_v`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `total_values` | Per-operation vibration magnitude, in m/s2: the hand-arm vibration total value `a_hv,i` (`kind="hav"`) or the Directive Part B dominant-axis value of [`wbv_exposure_basis`](/phonometry/reference/api/vibration/exposure/#wbv_exposure_basis) (`kind="wbv"`). |
| `durations_s` | Duration `T_i` per operation, in seconds. |
| `kind` | `"hav"` or `"wbv"` (selects the EAV/ELV). |
| `labels` | Optional operation labels; defaults to `op 1`, `op 2`, ... |

**Returns:** A [`DailyVibrationExposure`](/phonometry/reference/api/vibration/exposure/#dailyvibrationexposure) with `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the inputs differ in length, carry non-finite values, or `labels` mismatches. |

## DailyVibrationExposure

```python
DailyVibrationExposure(
    a8: float,
    labels: tuple[str, ...],
    total_values: Real,
    durations_s: Real,
    partials: Real,
    assessment: ExposureAssessment,
)
```

A daily exposure built from several operations, with its assessment.

**Attributes**

| Name | Description |
| :--- | :--- |
| `a8` | The daily exposure `A(8)`, in m/s2. |
| `labels` | A label per operation. |
| `total_values` | Vibration total value `a_hvi` per operation, in m/s2. |
| `durations_s` | Duration `T_i` per operation, in seconds. |
| `partials` | Partial exposure `A_i(8)` per operation, in m/s2. |
| `assessment` | The [`ExposureAssessment`](/phonometry/reference/api/vibration/exposure/#exposureassessment) of `a8`. |

### DailyVibrationExposure.plot()

```python
DailyVibrationExposure.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the partial exposures against the EAV / ELV thresholds.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

### DailyVibrationExposure.report()

```python
DailyVibrationExposure.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a daily vibration exposure assessment fiche to a PDF.

Writes a one-page assessment sheet laid out like the hand-arm /
whole-body vibration exposure calculators of the occupational-hygiene
practice: the standard-basis line naming the applied ISO method
(ISO 5349-1/-2:2001 for hand-transmitted vibration, ISO 2631-1:1997 for
whole-body vibration) and the Directive 2002/44/EC it is assessed
against, an optional metadata header (company, operator/worker,
workplace, instrumentation, calibration), the per-operation exposure
analysis (the vibration magnitude - the hand-arm vector total `a_hv`
or the whole-body Directive Part B dominant-axis value `a_w,max` -
the daily exposure time `T_i` and
the partial exposure `A_i(8)` of ISO 5349-1/-2 Eq. (2), closed by the
daily total and the combined `A(8)` of Eq. (3)), the per-operation
contribution chart, the boxed `A(8)` with its exposure zone, an
assessment table against the exposure action value (EAV) and exposure
limit value (ELV) of Directive 2002/44/EC (Article 3), a PASS/FAIL
verdict against the limit value, and a footer identity/disclaimer block.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`client` is the company, `specimen` the operator/worker, `test_room` the workplace) plus the `instrumentation` and `calibration` free-text fields and the footer identity. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When True, the operations table adds each operation's share of the daily vibration energy $A_i(8)^2 / A(8)^2$. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab or matplotlib is not installed. The fiche always embeds the contribution chart, so both are required (`pip install "phonometry[report,plot]"`). |

## energy_equivalent_acceleration

```python
energy_equivalent_acceleration(
    magnitudes: ArrayLike,
    durations_s: ArrayLike,
) -> float
```

Energy-equivalent weighted acceleration (ISO 2631-1 Eq. (B.3)).

$a_\mathrm{w,e} = \sqrt{\sum a_{\mathrm{w}i}^2 T_i / \sum T_i}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `magnitudes` | Weighted r.m.s. magnitudes $a_{\mathrm{w}i}$, in m/s2. |
| `durations_s` | Duration $T_i$ per period, in seconds. |

**Returns:** The energy-equivalent magnitude $a_\mathrm{w,e}$, in m/s2.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the inputs differ in length, are empty, or the total duration is zero. |

## exposure_assessment

```python
exposure_assessment(
    value: float,
    *,
    kind: str,
    metric: str = 'a8',
) -> ExposureAssessment
```

Assess a daily exposure against Directive 2002/44/EC (Article 3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `value` | The daily exposure `A(8)` in m/s2 (`metric="a8"`) or the vibration dose value in m/s^1,75 (`metric="vdv"`, whole-body only). |
| `kind` | `"hav"` (hand-arm) or `"wbv"` (whole-body). |
| `metric` | `"a8"` (default) or `"vdv"`. |

**Returns:** An [`ExposureAssessment`](/phonometry/reference/api/vibration/exposure/#exposureassessment).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown `kind`/`metric` combination or a non-finite or negative `value`. |

## ExposureAssessment

```python
ExposureAssessment(
    value: float,
    kind: str,
    metric: str,
    action_value: float,
    limit_value: float,
    exceeds_action: bool,
    exceeds_limit: bool,
    zone: str,
)
```

A daily exposure assessed against the Directive 2002/44/EC values.

**Attributes**

| Name | Description |
| :--- | :--- |
| `value` | The assessed daily exposure `A(8)` (or VDV), in its unit. |
| `kind` | `"hav"` or `"wbv"`. |
| `metric` | `"a8"` (m/s2) or `"vdv"` (m/s^1,75). |
| `action_value` | The exposure action value (EAV). |
| `limit_value` | The exposure limit value (ELV). |
| `exceeds_action` | Whether `value` reaches or exceeds the EAV. |
| `exceeds_limit` | Whether `value` reaches or exceeds the ELV. |
| `zone` | `"below action"`, `"action"` (EAV\<=value\<ELV) or `"limit"` (value>=ELV). |

## frequency_weighting

```python
frequency_weighting(name: str, frequencies: ArrayLike) -> WeightingResponse
```

Frequency-weighting response `H(f)` (ISO 8041-1:2017, Formula (5)).

Evaluates the overall weighting - band limiting, acceleration-velocity
transition and upward step - of weighting `name` at `frequencies`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `name` | Weighting name (one of [`WEIGHTING_NAMES`](/phonometry/reference/api/vibration/exposure/#weighting_names)). |
| `frequencies` | Frequencies at which to evaluate, in hertz (> 0). |

**Returns:** A [`WeightingResponse`](/phonometry/reference/api/vibration/exposure/#weightingresponse) with `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `name` is unknown or `frequencies` is empty. |

## hav_daily_exposure

```python
hav_daily_exposure(total_values: ArrayLike, durations_s: ArrayLike) -> float
```

Daily exposure `A(8)` for several operations (ISO 5349-1 Eq. (3)).

$A(8) = \sqrt{(1/T_0) \sum_i a_{\mathrm{hv}i}^2 T_i}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `total_values` | Vibration total value $a_{\mathrm{hv}i}$ per operation, in m/s2. |
| `durations_s` | Duration $T_i$ per operation, in seconds. |

**Returns:** The daily exposure `A(8)`, in m/s2.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the inputs differ in length, are empty or negative. |

## HAV_EAV_A8

*Constant* (`float`).

```python
HAV_EAV_A8 = 2.5
```

## HAV_ELV_A8

*Constant* (`float`).

```python
HAV_ELV_A8 = 5.0
```

## hav_vwf_lifetime_years

```python
hav_vwf_lifetime_years(a8: float) -> float
```

Years to 10 % vibration-white-finger prevalence (ISO 5349-1
Eq. (C.1)).

$D_\mathrm{y} = 31.8 \, A(8)^{-1.06}$ - the group-mean lifetime exposure
that produces finger blanching in 10 % of an exposed group
(informative Annex C).

**Parameters**

| Name | Description |
| :--- | :--- |
| `a8` | Daily vibration exposure `A(8)`, in m/s2 (> 0). |

**Returns:** The lifetime exposure duration $D_\mathrm{y}$, in years.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `a8` is not positive. |

## HumanVibrationWarning

Advisory for out-of-range human-vibration measurement conditions.

## motion_sickness_dose_value

```python
motion_sickness_dose_value(
    signal: SignalInput,
    fs: float | None = None,
) -> float
```

Motion sickness dose value `MSDV` (ISO 2631-1 clause 9; 8041-1
3.1.2.5).

$\mathrm{MSDV} = \left( \int a_\mathrm{w}(t)^2 \, dt \right)^{1/2}$, in
m/s^1.5; the `Wf`-weighted signal is the intended input.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Frequency-weighted acceleration signal (1-D), in m/s2. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) for its rate; a calibration factor it carries is deliberately not applied, because this quantity is an acceleration in m/s2 and not a pressure. |
| `fs` | Sampling frequency, in hertz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |

**Returns:** The MSDV, in m/s^1.5.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a bad signal or non-positive `fs`. |

## mtvv

```python
mtvv(
    signal: SignalInput,
    fs: float | None = None,
    *,
    integration_time: float = 1.0,
) -> float
```

Maximum transient vibration value (ISO 2631-1 Eq. (4)).

`MTVV = max a_w(t0)`, the peak of the 1 s running r.m.s. value.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Frequency-weighted acceleration signal (1-D), in m/s2. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) for its rate; a calibration factor it carries is deliberately not applied, because this quantity is an acceleration in m/s2 and not a pressure. |
| `fs` | Sampling frequency, in hertz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `integration_time` | Running-r.m.s. averaging time, in seconds (1 s). |

**Returns:** The MTVV, in m/s2.

## partial_exposure

```python
partial_exposure(total_value: float, duration_s: float) -> float
```

Daily exposure `A(8)` for one operation (ISO 5349-1 Eq. (2)).

$A(8) = a_\mathrm{hv} \sqrt{T / T_0}$ with $T_0 = 8$ h.  The
identical form gives the whole-body `A(8)` of Directive 2002/44/EC,
whose magnitude is the Annex Part B dominant-axis value (see
[`wbv_exposure_basis`](/phonometry/reference/api/vibration/exposure/#wbv_exposure_basis)) rather than a vector total.

**Parameters**

| Name | Description |
| :--- | :--- |
| `total_value` | Vibration total value `a_hv` (or `a_v`), in m/s2. |
| `duration_s` | Daily exposure duration `T`, in seconds. |

**Returns:** The daily exposure `A(8)`, in m/s2.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-finite or negative magnitude or duration. |

## REFERENCE_ACCELERATION

*Constant* (`float`).

```python
REFERENCE_ACCELERATION = 1e-06
```

## REFERENCE_DURATION_S

*Constant* (`float`).

```python
REFERENCE_DURATION_S = 28800.0
```

## running_rms

```python
running_rms(
    signal: SignalInput,
    fs: float | None = None,
    *,
    integration_time: float = 1.0,
    method: str = 'linear',
) -> Real
```

Running r.m.s. of a weighted signal (ISO 2631-1 Eqs. (2)/(3)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Frequency-weighted acceleration signal (1-D), in m/s2. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) for its rate; a calibration factor it carries is deliberately not applied, because this quantity is an acceleration in m/s2 and not a pressure. |
| `fs` | Sampling frequency, in hertz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `integration_time` | Averaging time `tau`, in seconds (default 1 s, the "slow" constant of ISO 2631-1 6.3.1). |
| `method` | `"linear"` (Eq. (2), a sliding rectangular average) or `"exponential"` (Eq. (3), a single-pole average). |

**Returns:** The running r.m.s. value `a_w(t0)` per sample, in m/s2.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a bad signal, non-positive `fs`/`tau` or an unknown `method`. |

## vibration_dose_value

```python
vibration_dose_value(signal: SignalInput, fs: float | None = None) -> float
```

Vibration dose value `VDV` (ISO 2631-1 Eq. (5)).

$\mathrm{VDV} = \left( \int a_\mathrm{w}(t)^4 \, dt \right)^{1/4}$, in
m/s^1.75; more sensitive to peaks than the r.m.s. value.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Frequency-weighted acceleration signal (1-D), in m/s2. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) for its rate; a calibration factor it carries is deliberately not applied, because this quantity is an acceleration in m/s2 and not a pressure. |
| `fs` | Sampling frequency, in hertz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |

**Returns:** The VDV, in m/s^1.75.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a bad signal or non-positive `fs`. |

## vibration_total_value

```python
vibration_total_value(
    components: ArrayLike,
    *,
    k: ArrayLike | None = None,
) -> float
```

Vibration total value `a_v` / `a_hv` (ISO 2631-1 Eq. (10)).

$a_\mathrm{v} = \sqrt{\sum_j k_j^2 a_{\mathrm{w}j}^2}$ over the (up to three)
axis-weighted r.m.s. accelerations.  With `k = None` the unweighted
vector sum of ISO 5349-1 Eq. (1) (`a_hv`, all $k = 1$) is
returned.

**Parameters**

| Name | Description |
| :--- | :--- |
| `components` | Axis-weighted r.m.s. accelerations $a_{\mathrm{w}j}$, in m/s2. |
| `k` | Optional per-axis multiplying factors $k_j$ (ISO 2631-1 7.2.3); `None` uses unity for every axis. |

**Returns:** The vibration total value, in m/s2.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `components` is empty or `k` differs in length. |

## WBV_EAV_A8

*Constant* (`float`).

```python
WBV_EAV_A8 = 0.5
```

## WBV_EAV_VDV

*Constant* (`float`).

```python
WBV_EAV_VDV = 9.1
```

## WBV_ELV_A8

*Constant* (`float`).

```python
WBV_ELV_A8 = 1.15
```

## WBV_ELV_VDV

*Constant* (`float`).

```python
WBV_ELV_VDV = 21.0
```

## wbv_exposure_basis

```python
wbv_exposure_basis(a_wx: float, a_wy: float, a_wz: float) -> float
```

Whole-body exposure basis of Directive 2002/44/EC (Annex, Part B).

$\max(1.4 a_{\mathrm{w}x}, 1.4 a_{\mathrm{w}y}, a_{\mathrm{w}z})$ - the Directive bases
the whole-body daily exposure `A(8)` on the *highest* of the
frequency-weighted axis values $1.4 a_{\mathrm{w}x}$,
$1.4 a_{\mathrm{w}y}$, $a_{\mathrm{w}z}$ for a seated or standing worker
(Annex, Part B, point 1, with the ISO 2631-1 clause 7.2.3
multiplying factors), **not** on the ISO 2631-1 Eq. (10) vector total
`a_v`.  Feed the returned dominant-axis value to
[`daily_vibration_exposure`](/phonometry/reference/api/vibration/exposure/#daily_vibration_exposure) (`kind="wbv"`) for a
Directive-conforming whole-body assessment; the hand-arm basis is the
vector total `a_hv` instead (Annex, Part A, point 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `a_wx` | `Wd`-weighted r.m.s. acceleration on the x axis, in m/s2. |
| `a_wy` | `Wd`-weighted r.m.s. acceleration on the y axis, in m/s2. |
| `a_wz` | `Wk`-weighted r.m.s. acceleration on the z axis, in m/s2. |

**Returns:** The dominant-axis value $\max(1.4 a_{\mathrm{w}x}, 1.4 a_{\mathrm{w}y}, a_{\mathrm{w}z})$, in m/s2.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a negative or non-finite axis value. |

## weighted_acceleration

```python
weighted_acceleration(
    band_accelerations: ArrayLike,
    frequencies: ArrayLike,
    weighting: str,
) -> WeightedSpectrum
```

Weighted r.m.s. acceleration from a band spectrum (ISO 2631-1
Eq. (9)).

$a_\mathrm{w} = \sqrt{\sum_i (W_i a_i)^2}$ with the per-band weighting
factors $W_i$ of ISO 8041-1 evaluated at the band centres
(ISO 5349-1 Eq. (A.1) is the identical construction for the hand-arm
weighting `Wh`).

:::note
The weightings are evaluated at exactly the frequencies you pass. The
ISO tables (ISO 8041-1 Annex B, ISO 2631-1 Table 3, ISO 5349-1
Table A.2) tabulate $W_i$ at the *true* one-third-octave
centres $10^{n/10}$ Hz (6.31, 7.943, 15.85, ...), not at the
nominal band labels (6.3, 8, 16, ...). Pass true centres when
comparing against the tabulated factors; a nominal label (e.g.
16.0 instead of 15.85) gives a slightly different $W_i$.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `band_accelerations` | r.m.s. acceleration $a_i$ per band, in m/s2. |
| `frequencies` | Band centre frequencies, in hertz (true one-third-octave centres $10^{n/10}$ for table-conformant band values). |
| `weighting` | Weighting name (one of [`WEIGHTING_NAMES`](/phonometry/reference/api/vibration/exposure/#weighting_names)). |

**Returns:** A [`WeightedSpectrum`](/phonometry/reference/api/vibration/exposure/#weightedspectrum) with `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the inputs differ in length, are empty or carry non-finite values. |

## WeightedSpectrum

```python
WeightedSpectrum(
    frequencies: Real,
    band_accelerations: Real,
    weighting_name: str,
    weighting_factors: Real,
    weighted: Real,
    overall: float,
)
```

A weighted one-third-octave acceleration spectrum and its `a_w`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in hertz. |
| `band_accelerations` | Unweighted r.m.s. acceleration per band, in m/s2. |
| `weighting_name` | Weighting applied (one of [`WEIGHTING_NAMES`](/phonometry/reference/api/vibration/exposure/#weighting_names)). |
| `weighting_factors` | Weighting factor `W_i` per band. |
| `weighted` | Weighted band contribution `W_i*a_i`, in m/s2. |
| `overall` | Overall weighted r.m.s. acceleration `a_w`, in m/s2. |

### WeightedSpectrum.plot()

```python
WeightedSpectrum.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the unweighted and weighted band spectra with `a_w`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

## weighting_factors

```python
weighting_factors(name: str, frequencies: ArrayLike) -> Real
```

Weighting factors `|H(f)|` of weighting `name` (ISO 8041-1).

Convenience wrapper over [`frequency_weighting`](/phonometry/reference/api/vibration/exposure/#frequency_weighting) returning only the
magnitude array (the `W_i` of ISO 2631-1 Eq. (9) / ISO 5349-1 Eq. (A.1)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `name` | Weighting name (one of [`WEIGHTING_NAMES`](/phonometry/reference/api/vibration/exposure/#weighting_names)). |
| `frequencies` | Band centre frequencies, in hertz. |

**Returns:** Weighting factor per frequency.

## WEIGHTING_NAMES

*Constant* (`tuple`).

```python
WEIGHTING_NAMES = ('Wb', 'Wc', 'Wd', 'We', 'Wf', 'Wh', 'Wj', 'Wk', 'Wm')
```

## WeightingResponse

```python
WeightingResponse(
    name: str,
    frequencies: Real,
    response: Complex,
    magnitude: Real,
    magnitude_db: Real,
)
```

A frequency-weighting magnitude response (ISO 8041-1, Formula (5)).

**Attributes**

| Name | Description |
| :--- | :--- |
| `name` | Weighting name (one of [`WEIGHTING_NAMES`](/phonometry/reference/api/vibration/exposure/#weighting_names)). |
| `frequencies` | Frequencies at which the response was evaluated, in Hz. |
| `response` | Complex weighting $H(j2\pi f)$ per frequency. |
| `magnitude` | Weighting factor $\lvert H \rvert$ per frequency. |
| `magnitude_db` | $20 \log_{10}\lvert H \rvert$ per frequency, in decibels. |

### WeightingResponse.plot()

```python
WeightingResponse.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the weighting factor (dB) versus frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.
