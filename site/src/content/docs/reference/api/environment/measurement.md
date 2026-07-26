---
title: "environmental.measurement"
description: "Determination of environmental-noise sound pressure levels (ISO 1996-2:2017)."
sidebar:
  label: "measurement"
---

Determination of environmental-noise sound pressure levels (ISO 1996-2:2017).

The measurement companion of the ISO 1996-1 descriptors in
`phonometry.environmental`. ISO 1996-2 covers *how* the levels that feed
those descriptors are obtained: the tonal adjustment for prominent tones, the
residual-noise correction, and the measurement-uncertainty budget.

**Tonal audibility (engineering method, ISO 1996-2:2007 Annex C).** From the
energy-summed tone level `Lpt` (Formula (C.1)) and the masking-noise level
`Lpn` in the critical band around the tone, the tonal audibility above the
masking threshold is
`ΔLta = Lpt − Lpn + 2 + lg[1 + (fc/502)²·⁵]` dB (Formula (C.3)), and the
tonal adjustment is the piecewise function `Kt = 0` for `ΔLta < 4`,
`Kt = ΔLta − 4` for `4 ≤ ΔLta ≤ 10` and `Kt = 6` for `ΔLta > 10`
(Formulae (C.4)–(C.6)). The critical bandwidth is 100 Hz for centre
frequencies up to 500 Hz and 20 % of the centre frequency above (Table C.1).
[`assess_tonal_audibility`](/phonometry/reference/api/environment/measurement/#assess_tonal_audibility) returns both in a plottable result. (The 2017
edition defers the full engineering method to ISO/PAS 20065; the detailed,
self-contained algorithm implemented here is the 2007/2009 Annex C one.)

**Survey method (ISO 1996-2:2017 Annex K).**
[`tonal_seeking_survey`](/phonometry/reference/api/environment/measurement/#tonal_seeking_survey) flags a one-third-octave band that exceeds *both*
neighbours by 15 dB (25–125 Hz), 8 dB (160–400 Hz) or 5 dB (500–10 000 Hz).

**Mean-audibility route (ISO 1996-2:2017 Table J.1).**
[`tonal_adjustment_from_mean_audibility`](/phonometry/reference/api/environment/measurement/#tonal_adjustment_from_mean_audibility) maps the ISO/PAS 20065 mean
audibility `ΔL` to `Kt` (0–6 dB).

**Residual-noise correction (Clause 10.4).**
`L = 10 lg(10^(L'/10) − 10^(Lres/10))` (Formula (16)); with a residual
within 3 dB of the measured level no correction is allowed; the
*uncorrected* measured level `L'` is then the reportable value, as an upper
bound of the specific sound. [`gaussian_residual_level`](/phonometry/reference/api/environment/measurement/#gaussian_residual_level) estimates the
residual from percentile levels (Annex I, Formulae (I.1)/(I.2)).

**Measurement uncertainty (Clause 4, Annex F).** `u = √(Σ (cⱼ·uⱼ)²)`
(Formula (2)) expanded by `k = 2` (95 %) or `k = 1.3` (80 %). The
residual-correction sensitivity coefficients (Formulae (F.7)/(F.8)) and the
repeated-measurement standard uncertainty (Formulae (17)–(20)) are provided.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## assess_tonal_audibility

```python
assess_tonal_audibility(
    tone_level: float,
    masking_noise_level: float,
    centre_frequency: float,
) -> TonalAssessmentResult
```

Assess a tone's audibility and adjustment (ISO 1996-2 Annex C).

Combines [`tonal_audibility`](/phonometry/reference/api/environment/measurement/#tonal_audibility) and [`tonal_adjustment`](/phonometry/reference/api/environment/measurement/#tonal_adjustment) with the
[`critical_bandwidth`](/phonometry/reference/api/environment/measurement/#critical_bandwidth) into a plottable result.

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_level` | Energy-summed tone level `Lpt`, in dB. |
| `masking_noise_level` | Masking-noise level `Lpn`, in dB. |
| `centre_frequency` | Critical-band centre frequency `fc`, in Hz. |

**Returns:** A [`TonalAssessmentResult`](/phonometry/reference/api/environment/measurement/#tonalassessmentresult).

## combined_standard_uncertainty

```python
combined_standard_uncertainty(
    contributions: Sequence[float] | Sequence[tuple[float, float]] | np.ndarray,
) -> float
```

Combined standard uncertainty `u = √(Σ (cⱼ·uⱼ)²)` (Formula (2)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `contributions` | Either the per-component products `cⱼ·uⱼ` (dB), or `(uⱼ, cⱼ)` pairs whose product is formed. Independent inputs are assumed (no covariance term). |

**Returns:** The combined standard uncertainty, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `contributions` is empty or non-finite. |

## critical_bandwidth

```python
critical_bandwidth(centre_frequency: float) -> float
```

Critical bandwidth around a tone (ISO 1996-2 Annex C, Table C.1).

100 Hz for a centre frequency up to 500 Hz, 20 % of the centre frequency
above 500 Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `centre_frequency` | Critical-band centre frequency `fc`, in Hz. |

**Returns:** Critical bandwidth, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `centre_frequency` is not positive/finite. |

## environmental_expanded_uncertainty

```python
environmental_expanded_uncertainty(
    standard_uncertainty: float,
    *,
    confidence: float = 0.95,
) -> float
```

Expanded uncertainty `U = k·u` (Clause 4).

Coverage factor `k = 2` for 95 % or `k = 1.3` for 80 %.

**Parameters**

| Name | Description |
| :--- | :--- |
| `standard_uncertainty` | Combined standard uncertainty `u`, in dB. |
| `confidence` | Coverage probability (0.95 or 0.80). |

**Returns:** The expanded uncertainty `U`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `u` is negative/non-finite or `confidence` is not one of the tabulated values. |

## gaussian_residual_level

```python
gaussian_residual_level(
    l50: float,
    *,
    l90: float | None = None,
    l95: float | None = None,
) -> float
```

Estimate the residual equivalent level from percentiles (Annex I).

`Leq = L50 + 0.115·((L50 − L90)/1.28)²` (Formula (I.1)) or, with `L95`,
`Leq = L50 + 0.115·((L50 − L95)/1.65)²` (Formula (I.2)). Supply exactly
one of `l90` / `l95`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `l50` | Median level `L50`, in dB. |
| `l90` | Level exceeded 90 % of the time `L90`, in dB. |
| `l95` | Level exceeded 95 % of the time `L95`, in dB. |

**Returns:** The estimated Gaussian residual equivalent level, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If not exactly one of `l90` / `l95` is given, the inputs are not finite, or the percentile ordering is inverted (`L90`/`L95` cannot exceed `L50`, almost certainly swapped arguments, which the squared spread would otherwise hide). |

## RepeatedMeasurementResult

```python
RepeatedMeasurementResult(
    mean_level: float,
    standard_uncertainty: float,
    approximate_uncertainty: float,
    n: int,
)
```

Energy-mean level and its uncertainty from repeats (Formulae (17)–(20)).

**Attributes**

| Name | Description |
| :--- | :--- |
| `mean_level` | Energy-mean level `Lk = 10 lg((1/N)·Σ 10^(0.1·Li))`, dB (Formula (18)). |
| `standard_uncertainty` | Standard uncertainty `uk` by the primary route, Formulae (17)+(19): the sample standard deviation `sk` of the energy values `10^(0.1·Li)` mapped back to level, `uk = 10 lg(10^(0.1·Lk) + sk) − Lk`, in dB. |
| `approximate_uncertainty` | The Note 2 substitute (Formula (20)), `√(Σ(Li − Lk)²/(N − 1))`, in dB; valid only when the spread of the `Li` is small; it grossly inflates for spread levels. |
| `n` | Number of measurements. |

## residual_correction_uncertainty

```python
residual_correction_uncertainty(
    measured_level: float,
    residual_level: float,
    measured_uncertainty: float,
    residual_uncertainty: float,
) -> float
```

Uncertainty of the residual-corrected level (Formulae (F.7)–(F.9)).

With `m = 10^(−0.1(L'−Lres))`, the sensitivity coefficients are
`cL' = 1/(1 − m)` and `cres = −m/(1 − m)`, and
`uL = √(cL'²·uL'² + cres²·ures²)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `measured_level` | Measured level `L'`, in dB. |
| `residual_level` | Residual level `Lres`, in dB. |
| `measured_uncertainty` | Standard uncertainty of `L'`, in dB. |
| `residual_uncertainty` | Standard uncertainty of `Lres`, in dB. |

**Returns:** The combined standard uncertainty of the corrected level, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the residual is not below the measured level or an uncertainty is negative/non-finite. |

## residual_sound_correction

```python
residual_sound_correction(
    measured_level: float,
    residual_level: float,
) -> ResidualCorrectionResult
```

Correct a measured level for residual sound (Formula (16)).

`L = 10 lg(10^(L'/10) − 10^(Lres/10))`. When the residual is within 3 dB
of the measured level, §10.4 allows **no** correction: the *uncorrected*
measured level `L'` is the reportable value, as an upper bound of the
specific sound (the corrected value would understate reliability, being
the lower-side estimate). The result is then flagged `reliable = False`
and an `EnvironmentalMeasurementWarning` is issued; report
`reportable_upper_bound` (= `L'`), not `corrected_level`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `measured_level` | Measured level `L'` including residual, in dB. |
| `residual_level` | Residual (background) level `Lres`, in dB. |

**Returns:** A [`ResidualCorrectionResult`](/phonometry/reference/api/environment/measurement/#residualcorrectionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the levels are not finite or the residual is not below the measured level. |

## ResidualCorrectionResult

```python
ResidualCorrectionResult(
    corrected_level: float,
    reportable_upper_bound: float,
    margin: float,
    reliable: bool,
)
```

Residual-noise-corrected level (ISO 1996-2:2017 Clause 10.4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `corrected_level` | The corrected level `L` (Formula (16)), in dB. When `reliable` is `False` the standard allows *no* correction; this value is then informative only (it estimates the source from below) and must not be reported as the result. |
| `reportable_upper_bound` | The *measured* level `L'`, in dB. When the margin is 3 dB or less, §10.4 permits reporting the measured level as an upper bound of the specific sound level; this field carries that reportable value. |
| `margin` | `L' − Lres`, in dB (measured minus residual). |
| `reliable` | `True` when the residual is more than 3 dB below the measured level; `False` when no correction is allowed and only the uncorrected `L'` may be reported, as an upper bound. |

## tonal_adjustment

```python
tonal_adjustment(audibility: float) -> float
```

Tonal adjustment `Kt` from the audibility (Formulae (C.4)–(C.6)).

`Kt = 0` for `ΔLta < 4`, `Kt = ΔLta − 4` for `4 ≤ ΔLta ≤ 10` and
`Kt = 6` for `ΔLta > 10`. `Kt` is not restricted to integers.

**Parameters**

| Name | Description |
| :--- | :--- |
| `audibility` | Tonal audibility `ΔLta`, in dB. |

**Returns:** Tonal adjustment `Kt`, in dB (0 to 6).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `audibility` is not finite. |

## tonal_adjustment_from_mean_audibility

```python
tonal_adjustment_from_mean_audibility(
    mean_audibility: float,
    *,
    coarse: bool = False,
) -> int
```

Tonal adjustment `Kt` from the mean audibility `ΔL` (Table J.1).

The ISO 1996-2:2017 route that maps the ISO/PAS 20065 mean audibility
`ΔL` to an integer adjustment. With `coarse=True` the 3-dB-step
alternative applies (`ΔL ≤ 2` → 0, `2 < ΔL ≤ 9` → 3, `ΔL > 9` → 6).

**Parameters**

| Name | Description |
| :--- | :--- |
| `mean_audibility` | Mean audibility `ΔL`, in dB. |
| `coarse` | Use the coarse 3-dB-step mapping instead of Table J.1. |

**Returns:** Tonal adjustment `Kt`, in dB (integer, 0 to 6).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `mean_audibility` is not finite. |

## tonal_audibility

```python
tonal_audibility(
    tone_level: float,
    masking_noise_level: float,
    centre_frequency: float,
) -> float
```

Tonal audibility above the masking threshold (Formula (C.3)).

`ΔLta = Lpt − Lpn + 2 + lg[1 + (fc/502)²·⁵]` dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_level` | Energy-summed tone level `Lpt` in the critical band, in dB (see Formula (C.1)). |
| `masking_noise_level` | Masking-noise level `Lpn` in the critical band, in dB (see Formula (C.2)/(C.11)). |
| `centre_frequency` | Critical-band centre frequency `fc`, in Hz. |

**Returns:** Tonal audibility `ΔLta`, in dB (dB above the masking threshold).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `centre_frequency` is not positive/finite or the levels are not finite. |

## tonal_seeking_survey

```python
tonal_seeking_survey(
    levels: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray,
) -> np.ndarray
```

Flag prominent tones by the one-third-octave survey method (Annex K).

A band is flagged when it exceeds *both* adjacent one-third-octave bands by
the level difference for its range: 15 dB (25–125 Hz), 8 dB (160–400 Hz),
5 dB (500–10 000 Hz). The two end bands (no pair of neighbours) are never
flagged.

:::note
Annex K defines the thresholds for 25 Hz to 10 kHz only. Bands
supplied outside that span are extrapolated with the nearest rule
(15 dB below 25 Hz, 5 dB above 10 kHz), outside the standard.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | One-third-octave-band time-average levels, in dB. |
| `frequencies` | The band centre frequencies, in Hz (same length). |

**Returns:** Boolean array, `True` where a prominent tone is present.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are empty, non-finite, or differ in length. |

## TonalAssessmentResult

```python
TonalAssessmentResult(
    tone_level: float,
    masking_noise_level: float,
    centre_frequency: float,
    critical_bandwidth: float,
    audibility: float,
    adjustment: float,
)
```

Tonal-audibility assessment of a tone in noise (ISO 1996-2 Annex C).

**Attributes**

| Name | Description |
| :--- | :--- |
| `tone_level` | Energy-summed tone level `Lpt`, in dB. |
| `masking_noise_level` | Masking-noise level `Lpn`, in dB. |
| `centre_frequency` | Critical-band centre frequency `fc`, in Hz. |
| `critical_bandwidth` | Critical bandwidth, in Hz (Table C.1). |
| `audibility` | Tonal audibility `ΔLta`, in dB (Formula (C.3)). |
| `adjustment` | Tonal adjustment `Kt`, in dB (Formulae (C.4)–(C.6)). |

### TonalAssessmentResult.plot()

```python
TonalAssessmentResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the `Kt(ΔLta)` adjustment curve with this tone marked.

## uncertainty_from_repeated_measurements

```python
uncertainty_from_repeated_measurements(
    levels: Sequence[float] | np.ndarray,
) -> RepeatedMeasurementResult
```

Energy mean and its uncertainty from repeated levels (Formulae (17)–(20)).

`Lk = 10 lg((1/N)·Σ 10^(0.1·Li))` (Formula (18)). The standard
uncertainty follows the primary §10.5 route: the sample standard
deviation `sk` of the energy values `10^(0.1·Li)` (Formula (17))
propagated back to level, `uk = 10 lg(10^(0.1·Lk) + sk) − Lk`
(Formula (19)). The Note 2 level-domain approximation
`√(Σ(Li − Lk)²/(N − 1))` (Formula (20)) is also reported as
`approximate_uncertainty`; it is valid only "if the difference between
different Li is small", so a spread above 3 dB triggers an
`EnvironmentalMeasurementWarning` (e.g. [50, 60, 70] dB gives
3.94 dB by Formulae (17)+(19) but 12.18 dB by Formula (20)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | The repeated measured levels `Li`, in dB (at least two). |

**Returns:** A [`RepeatedMeasurementResult`](/phonometry/reference/api/environment/measurement/#repeatedmeasurementresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If fewer than two finite levels are given. |
