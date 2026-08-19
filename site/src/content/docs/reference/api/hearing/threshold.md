---
title: "hearing.threshold"
description: "Age-related hearing threshold (ISO 7029:2017) and audiometric reference zero (ISO 389-7:2005)."
sidebar:
  label: "threshold"
---

Age-related hearing threshold (ISO 7029:2017) and audiometric reference zero
(ISO 389-7:2005).

Implements the statistical distribution of the hearing threshold of an
otologically normal population as a function of age and sex (ISO 7029:2017),
and the reference threshold of hearing under free-field and diffuse-field
listening (ISO 389-7:2005, Table 1), over the audiometric frequencies from
125 Hz to 8000 Hz.

ISO 7029 gives the median threshold deviation from the value at age 18 as
$dH_\mathrm{md} = a \, (\mathrm{age} - 18)^b$ (clause 4.2, Table 1) and the
spread around the
median as two half-Gaussian standard deviations `su` (worse than median) and
`sl` (better than median), each a fifth-degree polynomial in `age - 18`
(clause 4.3, Tables 2-5). A population fractile is obtained by shifting the
median by the standard-normal quantile times the appropriate spread
(clause 4.4 / Annex A).

The noise-induced permanent threshold shift of ISO 1999 (which combines a noise
component with this age component) is not part of this module.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## age_threshold

```python
age_threshold(
    age: float,
    sex: Literal['male', 'female'] = 'male',
    fractile: float = 0.5,
    frequencies: ArrayLike | None = None,
) -> AgeThresholdResult
```

Age-related hearing threshold distribution (ISO 7029:2017).

Returns, per audiometric frequency, the median threshold deviation from the
value at age 18 (clause 4.2), the upper/lower half-Gaussian spreads
(clause 4.3) and the threshold at the requested population `fractile`
(clause 4.4): `median + z * spread` where `z` is the standard-normal
quantile of `fractile` and the spread is the upper one for `z >= 0`
(worse than the median) or the lower one otherwise.

**Parameters**

| Name | Description |
| :--- | :--- |
| `age` | Listener age, in years (must be at least 18). The standard's formulae are established up to 80 years for frequencies at or below 2000 Hz and up to 70 years above; ages beyond that extrapolate. |
| `sex` | `"male"` or `"female"`. |
| `fractile` | Population fractile in the open interval (0, 1); `0.5` gives the median. |
| `frequencies` | Optional subset of the audiometric frequencies, in hertz; `None` uses all eleven (125 Hz - 8000 Hz). |

**Returns:** An [`AgeThresholdResult`](/phonometry/reference/api/hearing/threshold/#agethresholdresult) with the distribution and `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an age below 18, an unknown sex, a fractile outside (0, 1), or an unknown frequency. |

## AgeThresholdResult

```python
AgeThresholdResult(
    age: float,
    sex: str,
    fractile: float,
    frequencies: np.ndarray,
    median: np.ndarray,
    spread_upper: np.ndarray,
    spread_lower: np.ndarray,
    threshold: np.ndarray,
)
```

Age-related hearing threshold distribution (ISO 7029:2017).

All arrays are in dB and aligned with [`AUDIOMETRIC_FREQUENCIES`](/phonometry/reference/api/hearing/threshold/#audiometric_frequencies).

**Attributes**

| Name | Description |
| :--- | :--- |
| `age` | Listener age, in years. |
| `sex` | `"male"` or `"female"`. |
| `fractile` | Population fractile of `threshold` (0-1). |
| `frequencies` | Audiometric frequencies, in hertz. |
| `median` | Median threshold deviation from age 18 (clause 4.2). |
| `spread_upper` | Upper half-Gaussian standard deviation `su`. |
| `spread_lower` | Lower half-Gaussian standard deviation `sl`. |
| `threshold` | Threshold deviation at `fractile` (clause 4.4). |

### AgeThresholdResult.plot()

```python
AgeThresholdResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the median threshold with the fractile band over frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## AUDIOMETRIC_FREQUENCIES

*Constant* (`numpy.ndarray, shape (11,)`).

## FIELDS

*Constant* (`tuple`).

```python
FIELDS = ('free-field', 'diffuse-field')
```

## reference_threshold

```python
reference_threshold(
    field: str = 'free-field',
    frequencies: ArrayLike | None = None,
) -> np.ndarray
```

Reference threshold of hearing (ISO 389-7:2005, Table 1).

The sound pressure level, in dB, that corresponds to the audiometric zero
(0 dB HL) under the given listening condition, at the audiometric
frequencies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `field` | `"free-field"` (frontal incidence) or `"diffuse-field"`. |
| `frequencies` | Optional subset of the audiometric frequencies, in hertz; `None` uses all eleven (125 Hz - 8000 Hz). |

**Returns:** The reference threshold, in dB, aligned with the frequencies.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown field or frequency. |

## SEXES

*Constant* (`tuple`).

```python
SEXES = ('male', 'female')
```
