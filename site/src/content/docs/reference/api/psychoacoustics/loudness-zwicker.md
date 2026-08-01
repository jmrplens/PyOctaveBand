---
title: "psychoacoustics.loudness_zwicker"
description: "Zwicker loudness for stationary and time-varying sounds per ISO 532-1:2017."
sidebar:
  label: "loudness_zwicker"
---

Zwicker loudness for stationary and time-varying sounds per ISO 532-1:2017.

Clean-room Python port of the normative reference implementation given in
Annex A of ISO 532-1:2017 (program "ISO_532-1", Annex A.4).  Two entry
points are provided:

* [`loudness_zwicker_from_spectrum`](/phonometry/reference/api/psychoacoustics/loudness-zwicker/#loudness_zwicker_from_spectrum) - stationary loudness from 28
  one-third-octave band levels, 25 Hz to 12.5 kHz (clause 5.3 / A.2).
* [`loudness_zwicker`](/phonometry/reference/api/psychoacoustics/loudness-zwicker/#loudness_zwicker) - loudness from a calibrated time signal,
  either with the stationary method (clause 5) or the time-varying
  method (clause 6), including the one-third-octave filterbank of
  Annex A (Tables A.1/A.2), the nonlinear temporal decay and the
  temporal weighting of the total loudness.

All numeric tables live in `phonometry._zwicker_data` and reproduce
Tables A.1 to A.9 of the standard digit for digit.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## loudness_zwicker

```python
loudness_zwicker(
    x: list[float] | np.ndarray,
    fs: int,
    field: Literal['free', 'diffuse'] = 'free',
    stationary: bool = False,
    calibration_factor: float = 1.0,
    time_skip: float = 0.0,
) -> ZwickerLoudness
```

Zwicker loudness of a calibrated time signal per ISO 532-1:2017.

The signal is resampled to the internal 48 kHz rate if needed, split
into 28 one-third-octave bands with the Annex A filterbank (Tables
A.1/A.2), squared and smoothed.  With `stationary=True` the method
for stationary sounds (clause 5) is applied to the per-band mean
square of the whole signal.  Otherwise the method for time-varying
sounds (clause 6) is used: core loudness every 0.5 ms, nonlinear
temporal decay, specific-loudness slopes, temporal weighting of the
total loudness and the percentile values N5/N10 from the full-rate
(2000 Hz) weighted loudness series, while the public 500 Hz
loudness-vs-time trace is left unchanged.

Input scaling follows the reference implementation's WAV convention:
`x * calibration_factor` must be the instantaneous sound pressure in
pascals, so that band levels are
$10 \lg(p^2 / (20~\mu\text{Pa})^2)$ dB SPL.
The reference program reads 32-bit float WAV files as pressure in Pa
directly (`calibration_factor = 1`), while 16-bit PCM samples are
divided by 32768 (full scale = +-1) and multiplied by a calibration
factor derived from a reference recording of known level Lref:

$$
\text{calibration factor} = \sqrt{ 10^{L_{\text{ref}}/10} \cdot 4\times10^{-10} / \overline{\text{ref}^2} }
$$

with `ref` scaled to +-1 full scale as well.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Single-channel time signal (see scaling convention above). |
| `fs` | Sampling rate in Hz (positive integer; resampled to 48 kHz with `scipy.signal.resample_poly` when not 48000). |
| `field` | Sound field of the recording: 'free' or 'diffuse'. |
| `stationary` | Use the stationary method (clause 5) instead of the time-varying method (clause 6). |
| `calibration_factor` | Multiplier converting `x` to pascals. |
| `time_skip` | Leading time, in seconds, excluded from the stationary mean square (the reference implementation's TimeSkip). Annex B.1 states the stationary calculation "shall start from 0,2 s" when validating against the official Annex B WAV files, excluding the filterbank transient; the default 0.0 preserves the whole-signal behaviour for synthetic steady signals. Validated always (negative, non-finite or whole-signal skips raise `ValueError`) but applied only by the stationary method -- clause 6 has no TimeSkip. |

**Returns:** [`ZwickerLoudness`](/phonometry/reference/api/psychoacoustics/loudness-zwicker/#zwickerloudness).  Stationary: as in [`loudness_zwicker_from_spectrum`](/phonometry/reference/api/psychoacoustics/loudness-zwicker/#loudness_zwicker_from_spectrum).  Time-varying: `loudness` is the maximum loudness Nmax, `loudness_level` its phon mapping, `specific` the pattern at the loudness maximum, `n5`/`n10` the percentile values and `time` / `loudness_vs_time` the loudness trace at 500 Hz.

## loudness_zwicker_from_spectrum

```python
loudness_zwicker_from_spectrum(
    levels: list[float] | np.ndarray,
    field: Literal['free', 'diffuse'] = 'free',
) -> ZwickerLoudness
```

Stationary Zwicker loudness from one-third-octave band levels.

Implements the method for stationary sounds of ISO 532-1:2017
(clause 5) starting from the 28 one-third-octave band levels with
center frequencies 25 Hz to 12.5 kHz (base-ten bands, IEC 61260-1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | 28 band levels in dB SPL (re 20 uPa), 25 Hz..12.5 kHz. |
| `field` | Sound field the levels were measured in: 'free' or 'diffuse'. |

**Returns:** [`ZwickerLoudness`](/phonometry/reference/api/psychoacoustics/loudness-zwicker/#zwickerloudness) with `loudness` (N in sone), `loudness_level` (LN in phon) and `specific` (N' in sone/Bark, 240 values at 0.1-Bark steps); the time-varying fields are None.

## ZwickerLoudness

```python
ZwickerLoudness(
    loudness: float,
    loudness_level: float,
    specific: np.ndarray,
    n5: float | None = None,
    n10: float | None = None,
    time: np.ndarray | None = None,
    loudness_vs_time: np.ndarray | None = None,
    field: str | None = None,
)
```

Result of an ISO 532-1:2017 Zwicker loudness calculation.

`loudness` is the total loudness N in sone (the stationary value, or
the maximum of the time-varying loudness); `loudness_level` is the
loudness level LN in phon obtained from `loudness` with the
sone-to-phon mapping of the reference implementation.  `specific` holds the
specific loudness N' in sone/Bark at 0.1-Bark steps (240 values; for
the time-varying method it is the pattern at the instant of maximum
loudness).  `n5`/`n10` are the percentile loudness values N5/N10
and `time`/`loudness_vs_time` the 500 Hz loudness-vs-time trace
(clause 6.5); these four are `None` for stationary results.
`field` records the sound field the calculation assumed (`"free"`
or `"diffuse"`), one of the items clause 7 requires a loudness report
to state; it defaults to `None` only for backward-compatible manual
construction (the library constructors always set it).

### ZwickerLoudness.plot()

```python
ZwickerLoudness.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the specific loudness N'(z) over Bark (see `._plotting`).

Adds a loudness-vs-time panel when the time-varying trace is
present.  Requires matplotlib (`pip install phonometry[plot]`);
returns the `Axes` (or array thereof).

### ZwickerLoudness.report()

```python
ZwickerLoudness.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 532-1 Zwicker loudness fiche to a PDF.

Writes a one-page accredited loudness report with the items clause 7
makes mandatory: the standard-basis line stating the method used
(stationary, clause 5, or time-varying, clause 6) and the sound field
(free or diffuse, when the result carries it), an optional metadata
header block, a compact metrics table (total loudness N, or maximum
loudness Nmax with the N5/N10 percentiles for a time-varying result,
and the loudness level LN) beside the specific-loudness pattern (the
result's own `plot`), for a time-varying result the loudness-
versus-time function N(t) in sones (clause 7 f)), the boxed
`N = X sone (LN = Y phon)` result, an optional verdict row and a
footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a prediction fiche (body, result and disclaimer only). A supplied `requirement` is read as the maximum permitted loudness in sone. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform signature; it has no effect on the single-layout loudness fiche. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |
