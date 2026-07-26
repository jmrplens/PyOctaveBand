---
title: "hearing.sii"
description: "Speech Intelligibility Index (SII) per ANSI S3.5-1997 (R2017)."
sidebar:
  label: "sii"
---

Speech Intelligibility Index (SII) per ANSI S3.5-1997 (R2017).

Implements the one-third-octave-band method (18 bands, 160 Hz - 8000 Hz) of
ANSI S3.5-1997, *American National Standard Methods for the Calculation of the
Speech Intelligibility Index*. From an equivalent speech spectrum level, an
equivalent noise spectrum level and an equivalent hearing threshold, the
procedure forms the self-speech masking, the upward spread of masking, the
equivalent internal noise and disturbance, the level-distortion factor and the
band-audibility function, and weights the latter by the band-importance function
of Table 3 to give the index `SII` in [0, 1] (clause 6).

The band-importance function (Table 3, average speech material), the standard
speech spectrum levels by vocal effort (Table 3) and the reference internal
noise spectrum level (Table 3) are the standard's own tabulated constants.
Spectrum levels are as defined in clauses 3.11 and 3.55.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## SIIResult

```python
SIIResult(
    sii: float,
    band_audibility: np.ndarray,
    band_importance: np.ndarray,
    frequencies: np.ndarray,
    speech_spectrum: np.ndarray,
    disturbance: np.ndarray,
    masking: np.ndarray,
)
```

Result of a Speech Intelligibility Index computation (ANSI S3.5-1997).

**Attributes**

| Name | Description |
| :--- | :--- |
| `sii` | The overall Speech Intelligibility Index in [0, 1] (clause 6). |
| `band_audibility` | Per-band audibility function `Ai` (clause 5.8). |
| `band_importance` | Per-band importance function `Ii` used (Table 3). |
| `frequencies` | One-third-octave band centre frequencies, in hertz. |
| `speech_spectrum` | Equivalent speech spectrum level `Ei'` per band. |
| `disturbance` | Equivalent disturbance spectrum level `Di` (clause 5.6). |
| `masking` | Equivalent masking spectrum level `Zi` (clause 5.4). |

### SIIResult.plot()

```python
SIIResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band audibility weighted by importance, with the SII.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### SIIResult.report()

```python
SIIResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ANSI S3.5-1997 speech-intelligibility-index fiche to a PDF.

Writes a one-page speech-audibility report: a standard-basis line, an
optional metadata header block, a per-one-third-octave-band table (the
equivalent speech spectrum `Ei'`, the band-importance function `Ii`
of Table 3 and the band-audibility function `Ai`) beside the
audibility and importance-weighted contribution bars (the result's own
`plot`), the boxed `SII = X` single number, an optional verdict
row and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare fiche (body, result and disclaimer only). A supplied `requirement` is read as the minimum required SII (a higher SII passes). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the left table adds the equivalent disturbance spectrum level `Di` column (clause 5.6). |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is not a supported language. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## speech_intelligibility_index

```python
speech_intelligibility_index(
    speech_spectrum: ArrayLike,
    noise_spectrum: ArrayLike | None = None,
    *,
    threshold: ArrayLike | None = None,
) -> SIIResult
```

Speech Intelligibility Index (ANSI S3.5-1997, one-third-octave method).

All spectra are equivalent spectrum levels (clauses 3.11/3.55) sampled at
the 18 one-third-octave band centres from 160 Hz to 8000 Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speech_spectrum` | Equivalent speech spectrum level `Ei'`, in dB SPL. A vocal-effort name (`"normal"`, `"raised"`, `"loud"` or `"shout"`) selects the corresponding standard speech spectrum (Table 3). |
| `noise_spectrum` | Equivalent noise spectrum level `Ni'`, in dB SPL; `None` uses a quiet field (`-80` dB in every band). |
| `threshold` | Equivalent hearing threshold `Ti'`, in dB HL; `None` uses normal hearing (`0` in every band). |

**Returns:** An [`SIIResult`](/phonometry/reference/api/speech/sii/#siiresult) with the overall index and its `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if a spectrum has the wrong length or effort name. |

## standard_speech_spectra

```python
standard_speech_spectra(
    vocal_efforts: str | Sequence[str] = ('normal', 'raised', 'loud', 'shout'),
) -> StandardSpeechSpectrum
```

Build the plottable ANSI S3.5-1997 standard speech spectra (Table 3).

Collects the standard speech spectrum level `Ui` of the requested vocal
efforts (via [`standard_speech_spectrum`](/phonometry/reference/api/speech/sii/#standard_speech_spectrum)) into a
[`StandardSpeechSpectrum`](/phonometry/reference/api/speech/sii/#standardspeechspectrum) that exposes `.plot()`. The band levels
are unchanged; this is a thin, plottable wrapper around the existing
function, which still returns the bare per-band array.

**Parameters**

| Name | Description |
| :--- | :--- |
| `vocal_efforts` | A single vocal-effort name or a sequence of names, each one of `"normal"`, `"raised"`, `"loud"` or `"shout"`. Defaults to the full family in the Table 3 order. |

**Returns:** A frozen [`StandardSpeechSpectrum`](/phonometry/reference/api/speech/sii/#standardspeechspectrum).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown vocal effort, or an empty selection. |

## standard_speech_spectrum

```python
standard_speech_spectrum(vocal_effort: str = 'normal') -> np.ndarray
```

Standard speech spectrum level by vocal effort (ANSI S3.5-1997 Table 3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `vocal_effort` | One of `"normal"`, `"raised"`, `"loud"`, `"shout"`. |

**Returns:** The 18-band equivalent speech spectrum level `Ui`, in dB SPL.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown vocal effort. |

## StandardSpeechSpectrum

```python
StandardSpeechSpectrum(
    frequencies: np.ndarray,
    vocal_efforts: tuple[str, ...],
    levels: np.ndarray,
)
```

The ANSI S3.5-1997 standard speech spectra by vocal effort (Table 3).

Bundles the standard speech spectrum level `Ui` of one or more vocal
efforts (ANSI S3.5-1997 Table 3) over the 18 one-third-octave bands, so the
spectra can be drawn with `plot`. Build it with
[`standard_speech_spectra`](/phonometry/reference/api/speech/sii/#standard_speech_spectra); the frozen instance is a thin, plottable
wrapper and re-runs none of the maths (the band levels are the tabulated
constants that [`standard_speech_spectrum`](/phonometry/reference/api/speech/sii/#standard_speech_spectrum) returns).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | The 18 one-third-octave band centre frequencies, in hertz (160 Hz to 8000 Hz). |
| `vocal_efforts` | The vocal efforts carried, in order; each one of `"normal"`, `"raised"`, `"loud"` or `"shout"`. |
| `levels` | The standard speech spectrum level `Ui`, in dB SPL, as a `(len(vocal_efforts), 18)` array; row `i` is the spectrum for `vocal_efforts[i]`. |

### StandardSpeechSpectrum.plot()

```python
StandardSpeechSpectrum.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the standard speech spectrum level versus frequency band.

Draws the standard speech spectrum level (dB SPL) over the 18
one-third-octave bands (160 Hz to 8000 Hz) on a categorical band axis;
each vocal effort in `vocal_efforts` is one labelled line, so the
whole spectrum lifting with vocal effort reads at a glance.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the per-effort `plot` calls. |

**Returns:** The axes.
