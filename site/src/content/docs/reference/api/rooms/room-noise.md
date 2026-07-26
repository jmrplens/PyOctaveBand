---
title: "room.room_noise"
description: "Room-noise rating curves per ANSI/ASA S12.2-2019."
sidebar:
  label: "room_noise"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Room-noise rating curves per ANSI/ASA S12.2-2019.

Implements the two spectrum-in rating methods of ANSI/ASA S12.2-2019, *Criteria
for Evaluating Room Noise*:

* **Noise Criteria (NC)** by the two-step procedure of clause 5.2.2. The
  speech interference level SIL (clause 3.2, the average of the 500, 1000,
  2000 and 4000 Hz octave-band levels) selects the NC-(SIL) curve; when no
  octave band exceeds that curve, the spectrum is designated NC-(SIL), and
  otherwise the rating is determined by the tangency method (clause 5.2.3:
  the value of the highest NC curve of Table 1 touched by the spectrum,
  reported together with the governing band). The tangency rating is always
  evaluated and kept available on the result.
* **Room Criteria Mark II (RC)** (Annex D, Table D.1). The numerical rating is
  the mid-frequency average `LMF` (500/1000/2000 Hz) rounded to the nearest
  decibel (clause D.4); the spectral tag `N`/`R`/`H` follows the
  deviation rules of clause D.3.

Both methods evaluate octave-band sound pressure levels over the 16 Hz to
8000 Hz bands tabulated by the standard. The RC Mark II curves are generated
from the -5 dB/octave rule of Annex D (16 Hz equal to 31.5 Hz, with the low
frequencies not dropping below 55 dB), which reproduces Table D.1 exactly.

The balanced noise criteria (NCB), the room noise criterion for fluctuating
low-frequency noise (RNC, Annex A), the acoustically induced vibration and
rattle classification (`RV`, clause D.3.4, which needs the Table 6 test)
and the numeric quality-assessment index (QAI, clause D.5 - deferred by the
standard to external references) are not implemented here.

## nc_curve

```python
nc_curve(index: float) -> np.ndarray
```

Octave-band levels of a Noise Criteria curve (ANSI/ASA S12.2-2019 Table 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `index` | The NC designation. Integer designations from 15 to 70 in steps of five return the tabulated curve; intermediate values are linearly interpolated band by band. |

**Returns:** The 10-band curve levels, in dB, aligned with [`OCTAVE_BANDS`](/phonometry/reference/api/materials/absorption-rating/#octave_bands).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `index` is outside the tabulated range. |

## NCResult

```python
NCResult(
    rating: float,
    governing_frequency: float,
    frequencies: np.ndarray,
    levels: np.ndarray,
    sil: float = nan,
    tangency_rating: float = nan,
    method: str = 'tangency',
    out_of_range: str | None = None,
)
```

Result of a Noise Criteria (NC) rating (ANSI/ASA S12.2-2019, 5.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `rating` | The reported NC designation, following the two-step procedure of clause 5.2.2: when no octave band exceeds the NC-(SIL) curve chosen from the speech interference level, the designation is NC-(SIL); otherwise it is the tangency rating (clause 5.2.3). NaN when the spectrum lies outside the NC-15 to NC-70 family of Table 1 (see `out_of_range`). |
| `governing_frequency` | Band, in hertz, where the tangency touch occurs (for a spectrum above the family, the band of maximum exceedance over the NC-70 curve); NaN for a SIL-designated spectrum, which has no governing band. |
| `frequencies` | Octave-band centre frequencies evaluated, in hertz. |
| `levels` | Measured octave-band sound pressure levels, in dB. |
| `sil` | Speech interference level, in dB: the arithmetic average of the 500, 1000, 2000 and 4000 Hz octave-band levels (clause 3.2); NaN when any of the four bands is missing. |
| `tangency_rating` | The tangency-method rating (clause 5.2.3), always evaluated; NaN when the spectrum lies outside the Table 1 family. |
| `method` | `"SIL"` when the designation is the clause 5.2.2 NC-(SIL) value, `"tangency"` otherwise. |
| `out_of_range` | `None` inside the NC-15 to NC-70 family; `"above"` when a band exceeds the NC-70 curve (no NC rating is defined, `label` reads `">NC-70"`); `"below"` when every measured band lies below the NC-15 curve (`label` reads `"<NC-15"`). |

### NCResult.label

*property*

The textual NC designation.

`"NC-44"` for a SIL-designated spectrum, `"NC-51 (125 Hz)"` for a
tangency rating with its governing band (the form of clause 5.2.3),
and `">NC-70 (63 Hz)"` / `"<NC-15"` for a spectrum outside the
Table 1 family, for which the standard defines no NC rating.

### NCResult.plot()

```python
NCResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the measured spectrum against the NC curves.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### NCResult.report()

```python
NCResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a Noise Criteria (NC) assessment fiche to a PDF.

Writes a one-page room-noise assessment report (ANSI/ASA S12.2-2019):
the standard-basis line, an optional metadata header block, the
measured octave-band levels beside the measured spectrum plotted
against the NC curve family (the result's own `plot`), the boxed
NC designation (the clause 5.2.2 NC-(SIL) value, or the tangency
rating with its governing band; `">NC-70"` / `"<NC-15"` outside
the Table 1 family), an optional verdict row and a footer with the
fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare assessment fiche (body, result and disclaimer only). A supplied `requirement` is read as the maximum acceptable NC rating (a lower rating is quieter, so the room passes at or below it). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table adds the per-band NC contour value read by the tangency method. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## noise_criterion

```python
noise_criterion(
    levels: ArrayLike,
    frequencies: ArrayLike | None = None,
) -> NCResult
```

Noise Criteria (NC) rating of a spectrum (ANSI/ASA S12.2-2019, 5.2.2).

Follows the standard's two-step rating procedure. First the speech
interference level SIL (clause 3.2: the average of the 500, 1000, 2000
and 4000 Hz levels) selects the NC-(SIL) curve; when no octave band
exceeds that curve, the spectrum is designated NC-(SIL). Otherwise the
rating is determined by the tangency method (clause 5.2.3): for each band
the NC index whose Table 1 curve passes through the measured level is
found by interpolation, the rating is the maximum over bands and the
governing band is where that maximum occurs. The tangency rating is
always evaluated and kept on `tangency_rating`; when the four SIL bands
are not all supplied, the tangency rating is the designation.

A spectrum outside the Table 1 family has no NC rating: when a band
exceeds the NC-70 curve the result is flagged `out_of_range="above"`
(with the governing band at the maximum exceedance over NC-70), and when
every measured band lies below the NC-15 curve it is flagged
`out_of_range="below"`; in both cases `rating` is NaN and `label`
reads `">NC-70"` / `"<NC-15"`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Octave-band sound pressure levels, in dB. Without `frequencies` this must be the 10 bands from 16 Hz to 8000 Hz. |
| `frequencies` | Optional band centre frequencies, in hertz, matching `levels`; a subset of the ANSI S12.2 octave bands may be supplied. |

**Returns:** An [`NCResult`](/phonometry/reference/api/rooms/room-noise/#ncresult) with the designation and its `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for malformed inputs or unknown band frequencies. |

## rc_curve

```python
rc_curve(index: float) -> np.ndarray
```

Octave-band levels of a Room Criteria Mark II curve (Annex D, Table D.1).

The curve has a constant slope of -5 dB/octave keyed to its value at
1000 Hz; the 31.5 Hz level does not drop below 55 dB and the 16 Hz level
equals the 31.5 Hz level.

**Parameters**

| Name | Description |
| :--- | :--- |
| `index` | The RC designation (value at 1000 Hz). |

**Returns:** The 10-band curve levels, in dB, aligned with [`OCTAVE_BANDS`](/phonometry/reference/api/materials/absorption-rating/#octave_bands).

## RCResult

```python
RCResult(
    rating: int,
    lmf: float,
    classification: str,
    reference_curve: np.ndarray,
    frequencies: np.ndarray,
    levels: np.ndarray,
)
```

Result of a Room Criteria Mark II rating (ANSI/ASA S12.2-2019, Annex D).

**Attributes**

| Name | Description |
| :--- | :--- |
| `rating` | Numerical RC designation `LMF` rounded to the nearest dB. |
| `lmf` | Mid-frequency average (500/1000/2000 Hz), in dB (clause D.4). |
| `classification` | Spectral tag, `"N"` (neutral), `"R"` (rumble) or `"H"` (hiss) per clause D.3, or `"RH"` when both the rumble and the hiss deviation tests fire. Clause D.3.5 admits only the letters N, R, H or the combination RV, so the combined `"RH"` tag is a diagnostic extension of this library, not a clause D.3.5 designation. The vibration/rattle tag `RV` (clause D.3.4) needs the Table 6 criterion test and is not implemented. |
| `reference_curve` | The RC Mark II curve used for classification, in dB. |
| `frequencies` | Octave-band centre frequencies evaluated, in hertz. |
| `levels` | Measured octave-band sound pressure levels, in dB. |

### RCResult.label

*property*

The room-criterion label in the `RC-NN(A)` form of clause D.3.5.

Clause D.3.5 admits N, R, H or RV as the tag; when both the rumble
and hiss deviations fire, this library labels the spectrum with the
combined `RH` extension (see `classification`).

### RCResult.out_of_family

*property*

True when the rating falls outside the tabulated RC-25 to RC-50 family.

Table D.1 tabulates the RC Mark II curves for ratings 25 through 50
and clause D.3.5 defines the `RC-NN(A)` label for integers in that
range. Outside it the reference curve is generated by the same
-5 dB/octave rule of Annex D, but the designation extrapolates beyond
the standard's tabulated family.

### RCResult.plot()

```python
RCResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the measured spectrum against the reference RC Mark II curve.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### RCResult.report()

```python
RCResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a Room Criteria Mark II (RC) assessment fiche to a PDF.

Writes a one-page room-noise assessment report (ANSI/ASA S12.2-2019,
Annex D): the standard-basis line, an optional metadata header block,
the measured octave-band levels beside the measured spectrum plotted
against the reference RC Mark II curve (the result's own `plot`),
the boxed `RC-nn(tag)` rating with its mid-frequency average and
spectral-quality tag, an optional verdict row and a footer with the
fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare assessment fiche (body, result and disclaimer only). A supplied `requirement` is read as the maximum acceptable RC rating (a lower rating is quieter, so the room passes at or below it). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table adds the reference RC Mark II curve and the measured deviation from it. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## room_criterion

```python
room_criterion(
    levels: ArrayLike,
    frequencies: ArrayLike | None = None,
) -> RCResult
```

Room Criteria Mark II rating (ANSI/ASA S12.2-2019, Annex D).

The numerical rating is the mid-frequency average `LMF` of the 500,
1000 and 2000 Hz levels rounded to the nearest decibel (clause D.4). The
spectral tag is neutral (`"N"`) when the levels at and below 500 Hz do
not exceed the reference RC curve by more than 5 dB and the levels at and
above 1000 Hz do not exceed it by more than 3 dB; rumble (`"R"`) when a
low band exceeds by more than 5 dB; hiss (`"H"`) when a high band
exceeds by more than 3 dB (clause D.3). When both deviations fire, the
library tags the spectrum `"RH"`, a diagnostic extension beyond the
clause D.3.5 letters (see [`RCResult`](/phonometry/reference/api/rooms/room-noise/#rcresult)).

Clause D.4 evaluates a spectrum that includes at least the octave bands
from 31.5 Hz through 4000 Hz; when any of those bands is missing a
`UserWarning` is emitted, since the absent bands are silently
skipped by the spectral-tag deviation tests. A rating outside the
tabulated RC-25 to RC-50 family is flagged by
[`RCResult.out_of_family`](/phonometry/reference/api/rooms/room-noise/#rcresultout_of_family).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Octave-band sound pressure levels, in dB. Without `frequencies` this must be the 10 bands from 16 Hz to 8000 Hz. |
| `frequencies` | Optional band centre frequencies, in hertz, matching `levels`. |

**Returns:** An [`RCResult`](/phonometry/reference/api/rooms/room-noise/#rcresult) with the rating, tag and its `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the 500/1000/2000 Hz mid-frequency bands are absent. |
