---
title: "hearing.noise_induced_hearing_loss"
description: "Estimation of noise-induced hearing loss (ISO 1999:2013)."
sidebar:
  label: "noise_induced_hearing_loss"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Estimation of noise-induced hearing loss (ISO 1999:2013).

Implements the noise-induced permanent threshold shift (NIPTS) of a
noise-exposed population and its combination with the age-related threshold
into the hearing threshold level associated with age and noise (HTLAN), over
the six audiometric frequencies 500 Hz to 6000 Hz.

The median NIPTS for exposure durations of 10 to 40 years is
`N50 = [u + v*lg(t/t0)] * (L_EX,8h - L0)**2` (clause 6.3.1, Formula 2, with
the values `u, v, L0` of Table 1), extrapolated below 10 years by Formula 3.
The statistical distribution about the median is two half-Gaussians whose
spreads `du` (worse than the median) and `dl` (better) follow Formulae 6/7
with the coefficients of Table 3; a population fractile is
`N50 + z * spread` with `z` the standard-normal quantile (clause 6.3.2,
Formulae 4/5, Table 2), clamped at zero. The HTLAN combines the age component
`H` (HTLA, database A = ISO 7029) with the noise component `N` by
`H' = H + N - H*N/120` (clause 6.1, Formula 1).

## combine_age_and_noise

```python
combine_age_and_noise(htla: ArrayLike, nipts_value: ArrayLike) -> np.ndarray
```

Combine the age and noise components by Formula (1) (clause 6.1).

`H' = H + N - H*N/120`, the hearing threshold level associated with age
and noise. The formula "is applicable only to corresponding percentage
values of H', H and N", so both components must be taken at the same
population percentage.

Exposed separately from [`htlan`](/phonometry/reference/api/hearing/noise-induced-hearing-loss/#htlan) because clause 6.2 lets the user
supply their own HTLA database: 6.2.3 recommends a database B collected on
a control population of the country under consideration, and 6.2.4 makes
the choice between databases A and B depend on the question being answered.
Feeding such a database here combines it with the library's NIPTS.

**Parameters**

| Name | Description |
| :--- | :--- |
| `htla` | Age-associated hearing threshold level `H`, in dB. |
| `nipts_value` | Noise-induced permanent threshold shift `N`, in dB. |

**Returns:** The combined threshold `H'`, in dB.

## htlan

```python
htlan(
    age: float,
    sex: Literal['male', 'female'],
    l_ex: float,
    years: float,
    fractile: float = 0.5,
    frequencies: ArrayLike | None = None,
) -> HtlanResult
```

Hearing threshold level associated with age and noise (clause 6.1).

Combines the age component `H` (HTLA from database A, evaluated from
ISO 7029:2017 - the edition ISO 1999:2013 references undated in 6.2.2 - at
the same population fractile) with the noise component `N` (the NIPTS at
that fractile) by Formula (1): `H' = H + N - H*N/120`. The formula applies
to corresponding percentage values, so the same `fractile` drives both
components.

**Parameters**

| Name | Description |
| :--- | :--- |
| `age` | Listener age, in years (at least 18, the ISO 7029 lower limit). |
| `sex` | `"male"` or `"female"`. |
| `l_ex` | Noise exposure level normalized to 8 h, `L_EX,8h`, in dB. |
| `years` | Exposure duration, in years. |
| `fractile` | Population fractile in (0, 1) applied to both components. |
| `frequencies` | Optional subset of the audiometric frequencies, in hertz; `None` uses all six (500 Hz - 6000 Hz). |

**Returns:** An [`HtlanResult`](/phonometry/reference/api/hearing/noise-induced-hearing-loss/#htlanresult) with `htla`, `nipts`, `threshold` and `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an age below 18, an unknown sex, a non-positive duration, a fractile outside (0, 1), or an unknown frequency. |

The noise component inherits the validated-domain checks of [`nipts`](/phonometry/reference/api/hearing/noise-induced-hearing-loss/#nipts),
so exposure conditions outside the standard's stated ranges raise a
[`NoiseInducedHearingLossWarning`](/phonometry/reference/api/hearing/noise-induced-hearing-loss/#noiseinducedhearinglosswarning) and the `.report()` fiche prints
the extrapolation caveat.

## HtlanResult

```python
HtlanResult(
    age: float,
    sex: str,
    l_ex: float,
    years: float,
    fractile: float,
    frequencies: np.ndarray,
    htla: np.ndarray,
    nipts: np.ndarray,
    threshold: np.ndarray,
)
```

Hearing threshold level associated with age and noise (clause 6.1).

All arrays are in dB and aligned with `NIPTS_FREQUENCIES`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `age` | Listener age, in years. |
| `sex` | `"male"` or `"female"`. |
| `l_ex` | Noise exposure level normalized to 8 h, in dB. |
| `years` | Exposure duration, in years. |
| `fractile` | Population fractile (0-1) applied to both components. |
| `frequencies` | Audiometric frequencies, in hertz. |
| `htla` | Age component `H` (HTLA, database A, evaluated from ISO 7029:2017, the edition ISO 1999:2013 references undated in 6.2.2). |
| `nipts` | Noise component `N` (NIPTS at `fractile`). |
| `threshold` | Combined HTLAN `H' = H + N - H*N/120`. |

### HtlanResult.plot()

```python
HtlanResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the age, noise and combined threshold components over frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### HtlanResult.report()

```python
HtlanResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an HTLAN hearing-threshold prediction fiche to a PDF.

Writes a one-page statistical-prediction report of the hearing
threshold level associated with age and noise (ISO 1999:2013 clause
6.1): a prediction-basis line, an optional metadata header, a
per-audiometric-frequency table of the age component `H`, the noise
component `N` and the combined threshold `H' = H + N - H*N/120`
beside the result's own plot, the boxed representative threshold
averaged over the 2/3/4 kHz hearing-handicap set with the
listener/exposure conditions, and a statistical-prediction note. The
fiche is a population estimate, not a clinical diagnosis of any
individual.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`client` is the company, `specimen` the worker(s)/group, `test_room` the workplace) and, via `requirement`, a maximum acceptable representative HTLAN in dB that adds a PASS/FAIL verdict (a lower threshold is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When True, the table adds the compression term `H*N/120`. |
| `language` | `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab, matplotlib or svglib is not installed (`pip install "phonometry[report,plot]"`). |

## nipts

```python
nipts(
    l_ex: float,
    years: float,
    fractile: float = 0.5,
    frequencies: ArrayLike | None = None,
) -> NiptsResult
```

Noise-induced permanent threshold shift (ISO 1999:2013, clause 6.3).

Returns, per audiometric frequency, the median NIPTS `N50` (Formula 2,
extrapolated below 10 years by Formula 3), the upper/lower spreads
`du`/`dl` (Formulae 6/7) and the NIPTS at the requested population
`fractile` (Formulae 4/5): `N50 + z * spread` where `z` is the
standard-normal quantile of `fractile` and the spread is the upper one for
`z >= 0` (worse than the median) or the lower one otherwise, clamped at
zero.

**Parameters**

| Name | Description |
| :--- | :--- |
| `l_ex` | Noise exposure level normalized to a nominal 8 h working day, `L_EX,8h`, in dB. |
| `years` | Exposure duration, in years (> 0; the standard establishes 10-40 years and extrapolates 1-10 years by Formula 3, below 1 year the result is a further extrapolation). |
| `fractile` | Population fractile in the open interval (0, 1); `0.5` gives the median. The reliable range of the standard is 0.05-0.95. |
| `frequencies` | Optional subset of the audiometric frequencies, in hertz; `None` uses all six (500 Hz - 6000 Hz). |

**Returns:** A [`NiptsResult`](/phonometry/reference/api/hearing/noise-induced-hearing-loss/#niptsresult) with the distribution and `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive duration, a fractile outside (0, 1), or an unknown frequency. |

Outside the standard's validated domain (durations of
`VALIDATED_YEARS`, fractiles of `VALIDATED_FRACTILES`,
exposure levels up to `VALIDATED_L_EX_MAX`) the computation still
runs but a [`NoiseInducedHearingLossWarning`](/phonometry/reference/api/hearing/noise-induced-hearing-loss/#noiseinducedhearinglosswarning) marks the result as an
extrapolation, and the `.report()` fiche prints the matching caveat.

## NiptsResult

```python
NiptsResult(
    l_ex: float,
    years: float,
    fractile: float,
    frequencies: np.ndarray,
    median: np.ndarray,
    value: np.ndarray,
    spread_upper: np.ndarray,
    spread_lower: np.ndarray,
)
```

Noise-induced permanent threshold shift (ISO 1999:2013, clause 6.3).

All arrays are in dB and aligned with `NIPTS_FREQUENCIES`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `l_ex` | Noise exposure level normalized to 8 h, `L_EX,8h`, in dB. |
| `years` | Exposure duration, in years. |
| `fractile` | Population fractile of `value` (0-1); the fraction of the population with a smaller shift, so `0.9` is the most-susceptible 10 %. |
| `frequencies` | Audiometric frequencies, in hertz. |
| `median` | Median NIPTS `N50` (Formula 2/3). |
| `value` | NIPTS at `fractile` (Formula 4/5), clamped at zero. |
| `spread_upper` | Upper half-Gaussian spread `du` (Formula 6). |
| `spread_lower` | Lower half-Gaussian spread `dl` (Formula 7). |

### NiptsResult.plot()

```python
NiptsResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the NIPTS spectrum with the fractile band over frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### NiptsResult.report()

```python
NiptsResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a NIPTS hearing-loss prediction fiche to a PDF.

Writes a one-page statistical-prediction report of the noise-induced
permanent threshold shift (ISO 1999:2013 clause 6.3): a
prediction-basis line, an optional metadata header (company,
worker(s)/group, workplace, date), a per-audiometric-frequency table of
the median `N50` and the NIPTS at the chosen population fractile
beside the result's own spectrum plot, the boxed representative shift
averaged over the 2/3/4 kHz hearing-handicap set with the exposure
conditions (`L_EX,8h`, exposure years, fractile), and a
statistical-prediction note. The fiche is a population estimate, not a
clinical diagnosis of any individual.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`client` is the company, `specimen` the worker(s)/group, `test_room` the workplace) and, via `requirement`, a maximum acceptable representative NIPTS in dB that adds a PASS/FAIL verdict (a lower shift is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When True, the table adds the upper/lower spread columns (`du`/`dl`). |
| `language` | `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab, matplotlib or svglib is not installed (`pip install "phonometry[report,plot]"`). |

## NoiseInducedHearingLossWarning

Warns when an ISO 1999 prediction leaves the standard's validated domain.
