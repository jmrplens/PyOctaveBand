---
title: "vibration.multiple_shock_vibration"
description: "Whole-body vibration containing multiple shocks (ISO 2631-5:2018)."
sidebar:
  label: "multiple_shock_vibration"
---

Whole-body vibration containing multiple shocks (ISO 2631-5:2018).

Implements the normative Clause 5 spinal-response model and the Annex C
assessment of adverse health effects for the vertical (`z`) axis.

The 2018 edition is vertical-axis only by design: clause 4 (delineation,
item a) neglects the `x` and `y` contributions to spinal compression, the
seat-to-spine transfer function of clause 5.2 is the vertical seat-to-lumbar
response, and the Annex C stress conversion $m_z$ is the vertical one.
The horizontal spinal model of the withdrawn 2004 edition is not reproduced.
Assess horizontal whole-body exposure with the ISO 2631-1 metrics in this
domain instead: the weighted r.m.s. acceleration
([`weighted_acceleration`](/phonometry/reference/api/vibration/human-vibration/#weighted_acceleration)) and the vibration dose
value ([`vibration_dose_value`](/phonometry/reference/api/vibration/human-vibration/#vibration_dose_value)).

A seat-to-spine transfer function $H(\omega)$ (clause 5.2, Formula 1)
maps the measured seat acceleration $a_z(t)$ to the spinal response
acceleration

$$
A_z(t) = F^{-1}[H(\omega) \, F[a_z(t)]] \tag{Formula 2}
$$

The standard assumes a *conditioned* input: $H$ has unity
transmissibility at 0 Hz, so any DC offset in the record (e.g. the gravity
component of a non-AC-coupled accelerometer) passes straight into
$A_z(t)$ and corrupts the response peaks; remove the mean (high-pass)
before processing. The acceleration dose is

$$
D_z = 1.07 \left( \sum_i A_{z,i}^6 \right)^{1/6} \tag{Formula 3}
$$

over the positive response peaks, scaled to a daily dose

$$
D_{zd} = D_z \, (t_d/t_m)^{1/6} \tag{Formula 4/5}
$$

Annex C turns the daily dose into an injury risk: the daily compressive
stress $S_d$ (Formula C.1), the age-cumulated stress variable
$R$ (Formulae C.3/C.4) and the Weibull probability of lumbar injury
$P$ (Formula C.5, Table C.1):

$$
S_d = m_z D_{zd} \tag{Formula C.1}
$$

$$
R = \left[ \sum_i \left( S_d N^{1/6} / (S_{u,i} - S_\text{stat}) \right)^6 \right]^{1/6} \tag{Formulae C.3/C.4}
$$

$$
P = 1 - \exp\left(-(R/\alpha)^\beta\right) \tag{Formula C.5}
$$

The Annex A / Annex E model (intervertebral compressive forces via a
finite-element model distributed by ISO) is not reproducible from the standard
text and is out of scope.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## acceleration_dose

```python
acceleration_dose(acceleration: ArrayLike, fs: float) -> float
```

Acceleration dose $D_z$ from a seat acceleration time history.

Filters the acceleration through the seat-to-spine transfer function
(Formula 2), takes the positive response peaks and combines them by
Formula 3. The input must be conditioned (DC-removed); see
[`spinal_response`](/phonometry/reference/api/vibration/multiple-shock-vibration/#spinal_response).

**Parameters**

| Name | Description |
| :--- | :--- |
| `acceleration` | Measured, conditioned (zero-mean) vertical seat acceleration $a_z(t)$, m/s2. |
| `fs` | Sampling frequency, in hertz. |

**Returns:** The acceleration dose $D_z$, m/s2.

## compression_dose

```python
compression_dose(daily_dose_value: float, *, mz: float = 0.029) -> float
```

Daily compressive stress $S_d$ (Annex C, Formula C.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `daily_dose_value` | The daily acceleration dose $D_{zd}$, m/s2. |
| `mz` | Stress conversion $m_z$ (MPa per m/s2); default the 82 kg male value `MZ_MALE`. See `MZ_FEMALE`. |

**Returns:** The daily compressive stress $S_d = m_z D_{zd}$, MPa.

## daily_dose

```python
daily_dose(
    dose: float,
    exposure_time: float,
    measurement_time: float,
) -> float
```

Daily acceleration dose $D_{zd}$ (clause 5.3, Formula 4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `dose` | The measured acceleration dose $D_z$, m/s2. |
| `exposure_time` | Daily exposure period $t_d$ (any time unit). |
| `measurement_time` | Period $t_m$ over which $D_z$ was measured (same unit as `exposure_time`). |

**Returns:** The daily dose $D_{zd} = D_z (t_d/t_m)^{1/6}$, m/s2.

## daily_dose_multi

```python
daily_dose_multi(
    doses: ArrayLike,
    exposure_times: ArrayLike,
    measurement_times: ArrayLike,
) -> float
```

Daily dose from several exposure conditions (clause 5.3, Formula 5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `doses` | Acceleration dose $D_{z,j}$ of each condition, m/s2. |
| `exposure_times` | Daily exposure duration $t_{d,j}$ of each condition. |
| `measurement_times` | Measurement duration $t_{m,j}$ of each condition. |

**Returns:** The combined daily dose $D_{zd} = \left[ \sum_j D_{z,j}^6 \, (t_{d,j}/t_{m,j}) \right]^{1/6}$, m/s2.

## dose_from_peaks

```python
dose_from_peaks(peaks: ArrayLike) -> float
```

Acceleration dose $D_z$ from response peaks (clause 5.3,
Formula 3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `peaks` | The positive response peaks $A_{z,i}$, m/s2. |

**Returns:** The acceleration dose $D_z = 1.07 \left( \sum A_{z,i}^6 \right)^{1/6}$, m/s2.

## injury_probability

```python
injury_probability(
    risk: ArrayLike,
    *,
    sex: Literal['male', 'female'] = 'male',
) -> np.ndarray | float
```

Probability of lumbar injury $P(R)$ (Annex C, Formula C.5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `risk` | The stress variable $R$ (see [`injury_risk`](/phonometry/reference/api/vibration/multiple-shock-vibration/#injury_risk)); scalar or array-like. |
| `sex` | `"male"` or `"female"` (sets the Weibull coefficients). |

**Returns:** The injury probability $P = 1 - \exp(-(R/\alpha)^\beta)$ in 0-1; a float for a scalar input, otherwise an array. Negative $R$ gives 0.

## injury_risk

```python
injury_risk(
    daily_compression: float,
    *,
    start_age: float,
    years: int,
    days_per_year: float,
    sex: Literal['male', 'female'] = 'male',
    mz: float | None = None,
) -> float
```

Cumulative injury stress variable $R$ (Annex C, Formula C.3).

Accumulates the daily compressive stress over the exposure years, each year
weighted by the reducing ultimate strength of the ageing spine.

**Parameters**

| Name | Description |
| :--- | :--- |
| `daily_compression` | The daily compressive stress $S_d$, MPa. |
| `start_age` | Age $b$ at which the exposure started, in years. |
| `years` | Number of exposure years `n`. |
| `days_per_year` | Number of exposure days per year `N`. |
| `sex` | `"male"` or `"female"`. |
| `mz` | Stress conversion for the static stress $S_\text{stat} = m_z \cdot 9.81$; defaults to the sex-specific value. |

**Returns:** The stress variable $R$.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `years` is not positive or the spine strength is exhausted ($S_u - S_\text{stat} \le 0$) within the exposure period. |

## multiple_shock_assessment

```python
multiple_shock_assessment(
    acceleration: ArrayLike,
    fs: float,
    *,
    start_age: float,
    years: int,
    days_per_year: float,
    exposure_time: float | None = None,
    measurement_time: float | None = None,
    sex: Literal['male', 'female'] = 'male',
    mz: float | None = None,
) -> MultipleShockResult
```

Full multiple-shock assessment from a seat acceleration time
history.

Chains the Clause 5 dose and the Annex C risk: spinal response
(Formula 2), acceleration dose (Formula 3), daily dose (Formula 4),
compressive stress (C.1), stress variable $R$ (C.3) and injury
probability (C.5). The input must be conditioned (DC-removed); see
[`spinal_response`](/phonometry/reference/api/vibration/multiple-shock-vibration/#spinal_response).

The model is vertical-axis only (clause 4a of the 2018 edition); for
horizontal whole-body exposure use the ISO 2631-1 metrics in this domain
([`weighted_acceleration`](/phonometry/reference/api/vibration/human-vibration/#weighted_acceleration),
[`vibration_dose_value`](/phonometry/reference/api/vibration/human-vibration/#vibration_dose_value)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `acceleration` | Measured, conditioned (zero-mean) vertical seat acceleration $a_z(t)$, m/s2. |
| `fs` | Sampling frequency, in hertz. |
| `start_age` | Age `b` at which the exposure started, in years. |
| `years` | Number of exposure years `n`. |
| `days_per_year` | Number of exposure days per year `N`. |
| `exposure_time` | Daily exposure period $t_d$; when given with `measurement_time` the dose is scaled to a daily dose (Formula 4), otherwise the measured dose is taken as the daily dose. |
| `measurement_time` | Period $t_m$ over which the record was measured. |
| `sex` | `"male"` or `"female"`. |
| `mz` | Stress conversion $m_z$ (MPa per m/s2); defaults to the sex-specific value. |

**Returns:** The [`MultipleShockResult`](/phonometry/reference/api/vibration/multiple-shock-vibration/#multipleshockresult).

## MultipleShockResult

```python
MultipleShockResult(
    sex: Literal['male', 'female'],
    acceleration_dose: float,
    daily_dose: float,
    compression_dose: float,
    risk: float,
    probability: float,
    start_age: float,
    years: int,
    days_per_year: float,
    peaks: np.ndarray,
    risk_thresholds: tuple[float, float, float],
)
```

Multiple-shock health assessment (ISO 2631-5:2018, Clause 5 +
Annex C).

**Attributes**

| Name | Description |
| :--- | :--- |
| `sex` | `"male"` or `"female"`. |
| `acceleration_dose` | The acceleration dose $D_z$, m/s2. |
| `daily_dose` | The daily acceleration dose $D_{zd}$, m/s2. |
| `compression_dose` | The daily compressive stress $S_d$, MPa. |
| `risk` | The cumulative stress variable $R$. |
| `probability` | The probability of lumbar injury $P(R)$ in 0-1. |
| `start_age` | Age at which the exposure started, in years. |
| `years` | Number of exposure years. |
| `days_per_year` | Number of exposure days per year. |
| `peaks` | The positive response peaks $A_{z,i}$ used for the dose, m/s2. |
| `risk_thresholds` | The $R$ values for 10 %, 50 % and 90 % risk of injury for this sex (Table C.2). |

### MultipleShockResult.plot()

```python
MultipleShockResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the injury-probability curve with this assessment's
$R$.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### MultipleShockResult.report()

```python
MultipleShockResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a whole-body multiple-shock health-risk fiche to a PDF.

Writes a one-page health-risk assessment sheet for whole-body vibration
containing multiple shocks (ISO 2631-5:2018): the standard-basis line
(Clause 5 spinal response and Annex C risk model), an optional metadata
header (client, subject, workplace/vehicle, instrumentation,
calibration), the exposure-scenario grid (subject sex, the age
`b` at which the exposure started, the number of exposure years
`n`, the number of exposure days per year `N` and the number
of counted response shocks), the dose-and-stress analysis table
(the acceleration dose $D_z$ of Formula 3, the daily dose
$D_{zd}$ of Formula 4, the daily compressive stress
$S_d$ of Formula C.1, the cumulative stress variable
$R$ of Formula C.3 and the probability of lumbar injury
$P$ of Formula C.5), the injury-probability chart, the boxed
$R$ and $P$ with the Annex C risk classification, a
classification table against the Table C.2 risk levels with a zone
row, and a footer identity/disclaimer block.

The Annex C classification is informative (ISO 2631-5:2018 defines no
exposure limit), so the fiche carries a risk-band zone row rather than a
PASS/FAIL verdict: $R$ is placed among the Table C.2 stress
variables for 10 / 50 / 90 % risk of injury (low / moderate /
high / very high probability of an adverse health effect), the
moderate band matching the Annex C worked example.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`client`, `specimen` the subject, `test_room` the workplace or vehicle) plus the `instrumentation` and `calibration` free-text fields and the footer identity. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the fiche has one stacked body layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab or matplotlib is not installed. The fiche always embeds the injury-probability chart, so both are required (`pip install "phonometry[report,plot]"`). |

## response_peaks

```python
response_peaks(response: ArrayLike) -> np.ndarray
```

Positive response peaks $A_{z,i}$ (clause 5.3).

A peak is the maximum value of the response between two consecutive zero
crossings; only positive peaks are counted.

**Parameters**

| Name | Description |
| :--- | :--- |
| `response` | The spinal response acceleration $A_z(t)$. |

**Returns:** The positive peak values, in the order they occur.

## seat_to_spine_transfer

```python
seat_to_spine_transfer(frequencies: ArrayLike) -> np.ndarray
```

Seat-to-spine transfer function $H(\omega)$ (clause 5.2,
Formula 1).

A single complex zero and six complex poles map the seat acceleration to
the vertical spinal response; the transmissibility is unity at 0 Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies at which to evaluate `H`, in hertz. |

**Returns:** The complex frequency response, aligned with `frequencies`.

## spinal_response

```python
spinal_response(acceleration: ArrayLike, fs: float) -> np.ndarray
```

Vertical spinal response $A_z(t)$ (clause 5.2, Formula 2).

Applies the seat-to-spine transfer function to the measured conditioned
seat acceleration in the frequency domain and returns the time-domain
response by the inverse transform.

The input must be **conditioned (DC-removed)**: the transfer function is
unity at 0 Hz by design (clause 5.2), so a DC offset (e.g. the 1 g
gravity component of a DC-coupled accelerometer) is passed unattenuated
and produces a spurious constant shift in $A_z(t)$ that corrupts
the positive response peaks of the dose. Subtract the mean (or
high-pass) of $a_z(t)$ before calling.

**Parameters**

| Name | Description |
| :--- | :--- |
| `acceleration` | Measured, conditioned (zero-mean) vertical seat acceleration $a_z(t)$, m/s2. |
| `fs` | Sampling frequency, in hertz. |

**Returns:** The spinal response acceleration $A_z(t)$, m/s2, same length.

## static_stress

```python
static_stress(mz: float = 0.029) -> float
```

Static compressive stress $S_\text{stat} = m_z \cdot 9.81$
(Annex C), MPa.

## ultimate_strength

```python
ultimate_strength(
    age: ArrayLike,
    *,
    sex: Literal['male', 'female'] = 'male',
) -> np.ndarray
```

Ultimate lumbar strength $S_u$ at an age (Annex C,
Formula C.4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `age` | Age $b + i$, in years. |
| `sex` | `"male"` or `"female"` (sets the age slope $S_\text{age}$). |

**Returns:** The ultimate strength $S_u = 6.75 - S_\text{age} (b+i)$, MPa.
