---
title: "psychoacoustics.tonality"
description: "Prominent discrete tone assessment per ECMA-418-1:2024 (3rd edition)."
sidebar:
  label: "tonality"
---

Prominent discrete tone assessment per ECMA-418-1:2024 (3rd edition).

Implements the tone-to-noise ratio (TNR, clause 11) and prominence ratio
(PR, clause 12) methods on Hann-windowed, RMS-averaged FFT spectra
(clauses 11.1 / 12.1), with the clause 10 critical-band model.

The `prominent` verdict is the **numeric criterion only** (Formulae
(12)/(13) resp. (25)/(26)). The standard additionally requires prominent
tones to be confirmed by aural examination (clauses 11.8/12.8) and to pass
the clause 8/9 lower-threshold-of-hearing screen (its Formula (1), which
needs calibrated absolute levels); both audibility requirements are the
caller's responsibility.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## prominence_ratio

```python
prominence_ratio(
    x: list[float] | np.ndarray,
    fs: int,
    tone_freq: float | None = None,
    resolution_hz: float = 1.0,
) -> ToneAssessment
```

Prominence ratio of a discrete tone (ECMA-418-1:2024, clause 12).

Compares the level of the critical band centred on the tone (`LM`)
with the mean of the two contiguous critical bands (`LL`, `LU`,
band edges from the fitted Formulae 21-22 with Tables 2-3). For tones
at or below 171.4 Hz the lower band is truncated at 20 Hz and rescaled
to a 100 Hz bandwidth (Formula 24; Formula 23 otherwise, per the
clause 12.5 prose - the inline condition printed next to Formula 23 in
the PDF is a typo).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D). |
| `fs` | Sample rate in Hz. |
| `tone_freq` | Approximate tone frequency in Hz (default: highest peak in the range of interest). |
| `resolution_hz` | FFT bin spacing (default 1.0 Hz). |

**Returns:** [`ToneAssessment`](/phonometry/reference/api/psychoacoustics/tonality/#toneassessment) with `ratio_db` = PR in dB.

## TonalityWarning

Warns about biased tonality estimates (e.g. coarse FFT resolution).

## tone_to_noise_ratio

```python
tone_to_noise_ratio(
    x: list[float] | np.ndarray,
    fs: int,
    tone_freq: float | None = None,
    resolution_hz: float = 1.0,
) -> ToneAssessment
```

Tone-to-noise ratio of a discrete tone (ECMA-418-1:2024, clause 11).

The spectrum is Hann-windowed and RMS-averaged (clause 11.1). The tone
level `Lt` is the tone-band level above the line connecting the band
edges (Formula 9); the masking-noise level `Ln` is the remaining
critical-band level rescaled to the full critical bandwidth
(Formula 10); TNR = Lt - Ln (Formula 11). Proximate secondary tones in
the same critical band are combined per clause 11.6 (Formulae 14-16).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D). |
| `fs` | Sample rate in Hz. |
| `tone_freq` | Approximate tone frequency in Hz. Default (None) assesses the highest spectral peak in the 89.1 Hz - 11.2 kHz range of interest. For harmonic complexes, call once per component (clause 11.7). |
| `resolution_hz` | FFT bin spacing (default 1.0 Hz; the tone band must stay within 15 % of the critical bandwidth, clause 11.2). |

**Returns:** [`ToneAssessment`](/phonometry/reference/api/psychoacoustics/tonality/#toneassessment) with `ratio_db` = TNR in dB.

## ToneAssessment

```python
ToneAssessment(
    frequency: float,
    ratio_db: float,
    criterion_db: float,
    prominent: bool,
)
```

Result of a discrete-tone prominence assessment.

`ratio_db` is the tone-to-noise ratio or the prominence ratio in
decibels depending on the producing function; `criterion_db` is the
prominence limit at `frequency` and `prominent` the verdict.

`prominent` applies the numeric criterion only; the standard's
audibility requirements (aural examination per clauses 11.8/12.8 and
the clause 8/9 lower-threshold-of-hearing screen, which needs
calibrated absolute levels) are the caller's responsibility.

### ToneAssessment.plot()

```python
ToneAssessment.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the assessed tone against the ECMA-418-1 prominence criterion.

Draws the frequency-dependent criterion curve over the 89.1 Hz -
11.2 kHz range of interest with this tone at its own frequency and
the margin to the criterion marked; a tone on or above the curve is
prominent. The criterion family (tone-to-noise ratio, clause 11, or
prominence ratio, clause 12) is recovered from `criterion_db`.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes to draw on, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the assessed-tone marker `plot` call. |

**Returns:** The axes.
