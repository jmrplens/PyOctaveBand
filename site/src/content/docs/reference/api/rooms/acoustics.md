---
title: "room.acoustics"
description: "Room acoustic parameters from impulse responses per ISO 3382-1:2009 (performance spaces) and ISO 3382-2:2008 (ordinary rooms)."
sidebar:
  label: "acoustics"
---

Room acoustic parameters from impulse responses per ISO 3382-1:2009
(performance spaces) and ISO 3382-2:2008 (ordinary rooms).

The measured impulse response (acquired e.g. with the swept-sine or MLS
front end of [`phonometry.room.impulse_response`](/phonometry/reference/api/rooms/impulse-response/), ISO 18233) is filtered into
fractional-octave bands (IEC 61260) and converted to a decay curve by
Schroeder backward integration of the squared impulse response
(ISO 3382-1:2009, 5.3.3, Equation (1)). To limit the influence of
background noise, the integration is truncated at the crossing point
between the background-noise level and a sloping line fitted to the
squared impulse response, and the missing tail is compensated assuming
an exponential decay with the fitted rate (5.3.3, Equation (3)).

From the decay curve the reverberation times are evaluated by
least-squares line fits (ISO 3382-2:2008, Clause 6 and Annex C):
EDT over 0 dB to -10 dB (ISO 3382-1:2009, A.2.2), T20 over -5 dB to
-25 dB and T30 over -5 dB to -35 dB, each extrapolated to a 60 dB decay
(T = -60/slope). The energy parameters follow ISO 3382-1:2009 Annex A:
clarity C50/C80 (Equation (A.10)), definition D50 (Equation (A.11)) and
centre time Ts (Equation (A.13)), with t = 0 at the start of the direct
sound (A.2.1).

Validity flags implement the dynamic-range criterion of ISO 3382-1:2009,
5.3.3: the background noise must lie at least the evaluation range plus
15 dB below the maximum of the (squared) impulse response - 25 dB for EDT
(equivalently, the noise floor sits at least 10 dB below the lowest
evaluation point). The +15 dB rule is derived for finite forward
integration without tail compensation (C = 0), which under-estimates T;
because this module compensates the truncated tail (5.3.3, Equation (3),
C != 0) with a residual positive bias, the T20 and T30 flags add extra
headroom (46 dB for T20, 54 dB for T30) so that a flagged-valid decay
time stays within the 5 % just-noticeable difference of ISO 3382-1:2009,
Table A.1. The curvature indicator C = 100*(T30/T20 - 1) follows
ISO 3382-2:2008, B.3; values above 10 % flag a decay curve that is far
from a straight line.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## decay_curve

```python
decay_curve(
    ir: list[float] | np.ndarray,
    fs: int,
    band: float | None = None,
    fraction: int = 1,
    zero_phase: bool = False,
) -> DecayCurve
```

Schroeder decay curve of an impulse response.

Backward integration of the squared impulse response
(ISO 3382-1:2009, 5.3.3, Equation (1)), with noise truncation at the
crossing of the background-noise level with the fitted decay slope
and exponential compensation of the missing tail (Equation (3)).
Time zero is the start of the direct sound (A.2.1) and the level is
referenced to the steady-state level (the total energy of the
integrated impulse response, Clause 6).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Measured impulse response (1D), e.g. from [`phonometry.impulse_response`](/phonometry/reference/api/rooms/impulse-response/#impulse_response) (ISO 18233). |
| `fs` | Sample rate in Hz. |
| `band` | Optional band centre frequency in Hz. When given, the impulse response is first filtered with the matching IEC 61260 fractional-octave filter; when None the broadband response is integrated directly. |
| `fraction` | Bandwidth fraction of the band filter (1 = octave, 3 = one-third octave). Only used when `band` is not None. |
| `zero_phase` | If True, filter the band with forward-backward (zero-phase) filtering, removing the octave filter's group delay before the backward integration. ISO 3382-2:2008 Clause 7.3 NOTE permits time-reversed filtering (it relaxes the B\*T > 16 rule to B\*T > 4); it roughly halves the low-frequency short-decay bias at 125 Hz. Only used when `band` is not None. Default False (causal). |

**Returns:** A [`DecayCurve`](/phonometry/reference/api/rooms/acoustics/#decaycurve) with `time` in seconds from the direct sound and `level` in dB (0 dB at time zero), up to the noise truncation point. It unpacks as `time, level = decay_curve(...)` for backward compatibility and exposes [`DecayCurve.plot`](/phonometry/reference/api/rooms/acoustics/#decaycurveplot).

## DecayCurve

```python
DecayCurve(time: np.ndarray, level: np.ndarray, band: float | None = None)
```

Schroeder backward-integrated decay curve of an impulse response.

`time` holds the sample times in seconds from the direct sound and
`level` the decay levels in dB (0 dB at time zero), up to the noise
truncation point (ISO 3382-1:2009, 5.3.3). `band` is the
octave/third-octave band centre in Hz, or `None` for a broadband decay.

For backward compatibility with the previous `(time, level)` tuple
return of [`decay_curve`](/phonometry/reference/api/rooms/acoustics/#decay_curve), the dataclass is iterable and unpacks as
`time, level = decay_curve(...)`.

### DecayCurve.plot()

```python
DecayCurve.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the decay curve with optional straight T-fit overlays.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`. Pass `fits=False` to omit the
EDT/T20/T30 fit lines.

## room_parameters

```python
room_parameters(
    ir: list[float] | np.ndarray,
    fs: int,
    limits: tuple[float, float] | None = (125.0, 4000.0),
    fraction: int = 1,
    zero_phase: bool = False,
) -> RoomAcousticsResult
```

Room acoustic parameters per ISO 3382-1:2009 / ISO 3382-2:2008.

The impulse response (e.g. acquired with the ISO 18233 swept-sine or
MLS methods of [`phonometry.room.impulse_response`](/phonometry/reference/api/rooms/impulse-response/)) is filtered into
fractional-octave bands (IEC 61260) and each band decay curve is
obtained by Schroeder backward integration with noise truncation and
tail compensation (ISO 3382-1:2009, 5.3.3). Least-squares line fits
(ISO 3382-2:2008, Annex C) yield EDT (0 dB to -10 dB, ISO 3382-1,
A.2.2), T20 (-5 dB to -25 dB) and T30 (-5 dB to -35 dB), each
extrapolated to 60 dB. Clarity C50/C80, definition D50 and centre
time Ts follow ISO 3382-1:2009, Equations (A.10), (A.11) and (A.13),
with t = 0 at the start of the direct sound.

Values that cannot be evaluated (evaluation range unreachable, or
reaching below the noise floor + 10 dB) are NaN. The validity flags
apply the dynamic-range criterion of ISO 3382-1:2009, 5.3.3 (noise
at least evaluation range + 15 dB below the maximum of the impulse
response: 25 dB for EDT), with T20 and T30 raised to 46 dB and 54 dB
to absorb the positive bias of the tail compensation and keep a
flagged-valid decay time within the 5 % JND (ISO 3382-1:2009,
Table A.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Measured impulse response (1D). |
| `fs` | Sample rate in Hz. |
| `limits` | `(f_min, f_max)` band-centre limits in Hz; default octave bands 125 Hz to 4 kHz (ISO 3382-1:2009, 5.1). Use `(100.0, 5000.0)` with `fraction=3` for the one-third-octave engineering/precision range. `None` analyses the broadband response as a single band (`frequency` is then `None`). |
| `fraction` | Bandwidth fraction (1 = octave, 3 = one-third octave). Default 1. |
| `zero_phase` | If True, use forward-backward (zero-phase) octave filtering, removing the filter group delay before the backward integration. ISO 3382-2:2008 Clause 7.3 NOTE permits time-reversed filtering (relaxing B\*T > 16 to B\*T > 4); it roughly halves the 125 Hz short-decay T30 bias (about +4.9 % -> +2.4 % at T = 0.2 s). The benefit is small next to the ~10 % measurement variance but is free and standards-sanctioned. Default False (causal filtering). |

**Returns:** [`RoomAcousticsResult`](/phonometry/reference/api/rooms/acoustics/#roomacousticsresult) with one entry per band.

## RoomAcousticsResult

```python
RoomAcousticsResult(
    frequency: np.ndarray | None,
    edt: np.ndarray,
    t20: np.ndarray,
    t30: np.ndarray,
    c50: np.ndarray,
    c80: np.ndarray,
    d50: np.ndarray,
    ts: np.ndarray,
    dynamic_range: np.ndarray,
    edt_valid: np.ndarray,
    t20_valid: np.ndarray,
    t30_valid: np.ndarray,
    curvature: np.ndarray,
)
```

Per-band room acoustic parameters from one impulse response.

All arrays have one entry per analysis band (`frequency` holds the
exact band centre frequencies; it is `None` for a broadband
analysis, in which case the arrays have length 1). `edt`, `t20`
and `t30` are decay times in seconds extrapolated to 60 dB
(ISO 3382-1:2009, A.2.2; ISO 3382-2:2008, Clause 6); `c50`/`c80`
are early-to-late indices in dB (Equation (A.10)), `d50` the
definition ratio (Equation (A.11)) and `ts` the centre time in
seconds (Equation (A.13); the Table A.1 JND is 10 ms).

`dynamic_range` is the peak-to-noise-floor distance of the squared
band impulse response in dB. `edt_valid`, `t20_valid` and
`t30_valid` apply the ISO 3382-1:2009, 5.3.3 criterion (noise at
least evaluation range + 15 dB below the maximum: 25 dB for EDT), with
T20 and T30 tightened to 46 dB and 54 dB to absorb the positive bias of
the tail compensation (5.3.3, Eq. (3)) and keep a flagged-valid value
within the 5 % JND (ISO 3382-1:2009, Table A.1); they are False when the
value could not be evaluated. `curvature` is
C = 100*(T30/T20 - 1) in percent (ISO 3382-2:2008, B.3); values
above 10 % indicate an unreliable, non-straight decay.

### RoomAcousticsResult.plot()

```python
RoomAcousticsResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot per-band decay times (EDT/T20/T30) and clarity (C50/C80).

Invalid bands are hatched and greyed. With `ax` given, only the
decay-times panel is drawn on it. Requires matplotlib
(`pip install phonometry[plot]`); returns the
`Axes` (or array thereof).

### RoomAcousticsResult.report()

```python
RoomAcousticsResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a room acoustic parameters fiche to a PDF (ISO 3382-1/-2).

Writes a one-page report laid out like a room-acoustics measurement
report: the standard-basis line, an optional metadata header block
(room, volume, source/receiver positions, climate ...), the full-width
per-band parameter table (T20/T30/EDT and C50/C80/D50/Ts) above the
result's own per-band decay-time plot (`plot`), the boxed
mid-frequency reverberation time T_mid (the mean of the 500 Hz and
1000 Hz band T30; a one-third-octave analysis averages the 500 Hz and
1 kHz one-third-octave bands and labels them as such), an optional
verdict row and a footer with the fixed disclaimer. ISO 3382-1/-2 are
characterisation standards with no intrinsic pass/fail, so the verdict
row appears only when a target mid-frequency T is supplied through
`metadata.requirement` (read as the maximum acceptable T_mid). A
broadband result (`frequency` is `None`) has no 500 Hz and 1000 Hz
bands to average, so the box and the verdict fall back to the plain
broadband T30 instead of a mid-frequency average, with no
"500-1000 Hz" label; a band range without both mid bands boxes its
first finite T30 band, named explicitly.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare characterisation fiche (body, result and disclaimer only). The room-specific fields `room_volume`, `source_positions` and `receiver_positions` populate the header; `requirement` is read as the maximum mid-frequency reverberation time. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for parity with the other fiches; the room table already shows every computed parameter, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |
