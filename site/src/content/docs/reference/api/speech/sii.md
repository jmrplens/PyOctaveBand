---
title: "hearing.sii"
description: "Speech Intelligibility Index (SII) per ANSI S3.5-1997 (R2017)."
sidebar:
  label: "sii"
---

Speech Intelligibility Index (SII) per ANSI S3.5-1997 (R2017).

Implements all four band procedures of ANSI S3.5-1997, *American National
Standard Methods for the Calculation of the Speech Intelligibility Index*:

- `method="critical-band"`: 21 critical bands, 100 Hz to 9500 Hz (Table 1).
- `method="equally-contributing"`: 17 equally-contributing critical bands,
  300 Hz to 6400 Hz (Table 2).
- `method="one-third-octave"`: 18 one-third-octave bands, 160 Hz to 8000 Hz
  (Table 3). The library default.
- `method="octave"`: 6 octave bands, 177 Hz to 11314 Hz (Table 4).

From an equivalent speech spectrum level, an equivalent noise spectrum level
and an equivalent hearing threshold, every procedure runs the same chain: the
self-speech masking, the upward spread of masking, the equivalent internal
noise and disturbance, the level-distortion factor and the band-audibility
function, whose importance-weighted sum is the index `SII` in [0, 1]
(clause 6). Only the band table and the geometry of the spread of masking
change from procedure to procedure: the critical-band and
equally-contributing procedures spread the masking between tabulated band
limits, the one-third-octave procedure between band centre frequencies, and
the octave-band procedure omits the spread entirely (its bands are already
wider than the spread being modelled).

The band-importance functions, the standard speech spectrum levels by vocal
effort and the reference internal noise spectrum levels are the standard's own
tabulated constants (Tables 1 to 4). Spectrum levels are as defined in clauses
3.11 and 3.55.

The implementation reproduces the reference implementation `SII.C` of ASA
Working Group S3-79 (the committee that maintains ANSI S3.5) to double
precision on all eight of its official test cases (`CB.TST`, `CB_1.TST`,
`ECB.TST`, `ECB_1.TST`, `TO.TST`, `TO_1.TST`, `OCTAVE.TST` and
`OCTAVE_1.TST`, two per procedure, the `_1` variants exercising an
alternative band-importance function), and computes the Annex C worked
examples with the working group's official errata applied (see
`docs/ERRATA.md`).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## SII_METHODS

*Constant* (`tuple`).

```python
SII_METHODS = ('critical-band', 'equally-contributing', 'one-third-octave', 'octave')
```

## sii_procedure

```python
sii_procedure(method: str = 'one-third-octave') -> SIIProcedure
```

Build the plottable band table of an ANSI S3.5-1997 band procedure.

**Parameters**

| Name | Description |
| :--- | :--- |
| `method` | The band procedure, one of [`SII_METHODS`](/phonometry/reference/api/speech/sii/#sii_methods). |

**Returns:** A frozen [`SIIProcedure`](/phonometry/reference/api/speech/sii/#siiprocedure) carrying the band centre frequencies, band limits, band-importance function, reference internal noise spectrum level and normal-effort standard speech spectrum level of that procedure.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown procedure name. |

## SIIProcedure

```python
SIIProcedure(
    method: str,
    frequencies: np.ndarray,
    band_edges: np.ndarray,
    band_importance: np.ndarray,
    internal_noise: np.ndarray,
    speech_spectrum: np.ndarray,
)
```

The tabulated band table of one ANSI S3.5-1997 band procedure.

Bundles the normative constants of one of the standard's four band
procedures (Tables 1 to 4) so they can be inspected and drawn with
`plot`. Build it with [`sii_procedure`](/phonometry/reference/api/speech/sii/#sii_procedure); the frozen instance is a
plottable view of the tabulated constants and runs none of the SII maths.

**Attributes**

| Name | Description |
| :--- | :--- |
| `method` | The procedure name, one of [`SII_METHODS`](/phonometry/reference/api/speech/sii/#sii_methods). |
| `frequencies` | Nominal band centre frequencies, in hertz. |
| `band_edges` | Band limits, in hertz; one value more than the number of bands, so band `i` runs from `band_edges[i]` to `band_edges[i+1]`. |
| `band_importance` | Band-importance function `Ii` for average speech material. It sums to one, except for the equally-contributing procedure, whose printed 0.0588 per band sums to 0.9996. |
| `internal_noise` | Reference internal noise spectrum level `Xi`, in dB SPL. |
| `speech_spectrum` | Standard speech spectrum level `Ui` for normal vocal effort, in dB SPL. |

### SIIProcedure.plot()

```python
SIIProcedure.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the band-importance function of the procedure versus frequency.

Draws the band-importance function `Ii` as a step over the band
limits, so procedures with different band counts and widths can be
overlaid on the same logarithmic frequency axis and compared directly.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the step `plot` call. |

**Returns:** The axes.

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
    level_distortion: np.ndarray,
    method: str = 'one-third-octave',
)
```

Result of a Speech Intelligibility Index computation (ANSI S3.5-1997).

**Attributes**

| Name | Description |
| :--- | :--- |
| `sii` | The overall Speech Intelligibility Index in [0, 1] (clause 6). |
| `band_audibility` | Per-band audibility function `Ai` (clause 5.8). |
| `band_importance` | Per-band importance function `Ii` used (the procedure's own table, or the alternative function supplied). |
| `frequencies` | Band centre frequencies of the procedure, in hertz. |
| `speech_spectrum` | Equivalent speech spectrum level `Ei'` per band. |
| `disturbance` | Equivalent disturbance spectrum level `Di` (clause 5.6). |
| `masking` | Equivalent masking spectrum level `Zi` (clause 5.4). |
| `level_distortion` | Per-band level-distortion factor `Li` in [0, 1] (clause 5.7), unity until the speech spectrum level rises above the standard normal-effort spectrum by more than 10 dB. |
| `method` | The band procedure used, one of [`SII_METHODS`](/phonometry/reference/api/speech/sii/#sii_methods). |

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

Writes a one-page speech-audibility report: a standard-basis line
naming the band procedure, an optional metadata header block, a per-band
table over that procedure's bands (the equivalent speech spectrum
`Ei'`, the band-importance function `Ii` and the band-audibility
function `Ai`) beside the audibility and importance-weighted
contribution bars (the result's own `plot`), the boxed
`SII = X` single number, an optional verdict row and a footer with the
fixed disclaimer.

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
    method: str = 'one-third-octave',
    band_importance: ArrayLike | None = None,
) -> SIIResult
```

Speech Intelligibility Index (ANSI S3.5-1997, any of the four methods).

All spectra are equivalent spectrum levels (clauses 3.11/3.55) sampled at
the band centres of the chosen procedure: 18 one-third-octave bands from
160 Hz to 8000 Hz by default, or the 21 critical bands, the 17
equally-contributing critical bands or the 6 octave bands.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speech_spectrum` | Equivalent speech spectrum level `Ei'`, in dB SPL. A vocal-effort name (`"normal"`, `"raised"`, `"loud"` or `"shout"`) selects the corresponding standard speech spectrum; only `"normal"` is tabulated outside the one-third-octave procedure. |
| `noise_spectrum` | Equivalent noise spectrum level `Ni'`, in dB SPL; `None` uses a quiet field (`-80` dB in every band). |
| `threshold` | Equivalent hearing threshold `Ti'`, in dB HL; `None` uses normal hearing (`0` in every band). |
| `method` | The band procedure, one of [`SII_METHODS`](/phonometry/reference/api/speech/sii/#sii_methods): `"critical-band"` (21 bands, Table 1), `"equally-contributing"` (17 bands, Table 2), `"one-third-octave"` (18 bands, Table 3, the default) or `"octave"` (6 bands, Table 4). |
| `band_importance` | Alternative band-importance function `Ii`, one value per band, replacing the procedure's tabulated function (the standard's Annex B tabulates functions for specific speech test materials); `None` uses the tabulated average-speech function. |

**Returns:** An [`SIIResult`](/phonometry/reference/api/speech/sii/#siiresult) with the overall index and its `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if a spectrum has the wrong length, or the method or effort name is unknown. |

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
