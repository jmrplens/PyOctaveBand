---
title: "environment.assessment.impulsive_sound"
description: "Prominence of impulsive sounds and the LAeq adjustment (NT ACOU 112:2002, ISO/PAS 1996-3:2022)."
sidebar:
  label: "impulsive_sound"
---

Prominence of impulsive sounds and the `LAeq` adjustment (NT ACOU 112:2002,
ISO/PAS 1996-3:2022).

Noise with prominent impulses is more annoying than a steady sound of the same
equivalent level, so both methods add an adjustment `KI` to the measured
`LAeq`. They share the prominence and adjustment formulae, which both take
from Pedersen's method, and differ in what the caller supplies.

**NT ACOU 112:2002** is the closed form: the caller brings the onset rate and
the level difference of the impulse, and the **predicted prominence**

$$
P = 3 \log_{10}(\text{onset rate}) + 2 \log_{10}(\text{level difference}) \tag{Formula 1}
$$

follows (clause 7). From the impulse with the highest prominence over a
30-minute period, a graduated adjustment follows (clause 8):

$$
K_\mathrm{I} = 1.8 \, (P - 5)~\text{dB} \quad \text{for } P > 5, \text{ else } 0 \tag{Formula 2}
$$

and the rating level over a reference time interval combines the adjusted
sub-interval levels (clause 8, Note 1). An impulse qualifies when its onset
rate exceeds 10 dB/s (clauses 4.5-4.7); non-qualifying level rises receive
no adjustment (clause 8 applies only "for sounds with onset rates larger
than 10 dB/s").

**ISO/PAS 1996-3:2022** is the measurement chain that reads those same two
quantities from a calibrated time signal, and categorises the source by the
adjustment it earns (typically 0.0 dB to 9.0 dB):

* the A frequency-weighted, F time-weighted sound pressure level `LpAF` is
  computed from the signal and sampled at 10-25 ms intervals (Clause 4);
* an *onset* is a contiguous part of the positive slope of `LpAF` where the
  gradient exceeds 10 dB/s; its **starting** and **end** points are found from
  procedures a) to d) of Clause 4, merging events separated by less than 50 ms
  (Clause 3.3, Figure 2);
* for each onset the **level difference** $\mathrm{LD} = L_\mathrm{e} - L_\mathrm{s}$
  and the **onset rate** `OR` (the least-squares slope over the onset) are
  measured (Clauses 3.4, 3.5, Figures 1 and 2);
* the prominence (Clause 5, Formula 2) and the adjustment
  $K_\mathrm{I} = 1.8 \cdot (P - 5)$ dB for $P > 5$ (Clause 6, Formula 3)
  are the NT ACOU 112 formulae above;
* the source is categorised (Clause 7) as *not impulsive* ($K_\mathrm{I} = 0$),
  *regular impulsive* ($0 < K_\mathrm{I} \le 5$) or *highly impulsive*
  ($K_\mathrm{I} > 5$).

The ISO/PAS method for determining `KI` is not sensitive to the absolute
calibration of the equipment (Clause 8): onset rate and level difference are
level *differences*, so the adjustment is unchanged by a constant offset. Only
the reported `LAeq` and the adjusted `LAeq` depend on calibration.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## detect_onsets

```python
detect_onsets(
    levels: ArrayLike,
    dt: float,
    *,
    onset_rate_method: _OnsetRateMethod = 'least_squares',
) -> tuple[ImpulseOnset, ...]
```

Detect the onsets in an `LpAF` level history (Clause 4).

Applies procedures a) to d) of the standard: the starting point is the
first sample where the gradient exceeds 10 dB/s, the end point the first
later sample where it drops below 10 dB/s, and onsets separated by less
than 50 ms are merged. Each onset carries its level difference
$\mathrm{LD} = L_\mathrm{e} - L_\mathrm{s}$ (3.4), its onset rate (3.5) and its
prominence `P`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | A-weighted, F time-weighted level history `LpAF`, in dB, uniformly sampled with interval `dt`. |
| `dt` | Sampling interval of `levels`, in seconds (must be positive). |
| `onset_rate_method` | `"least_squares"` (default) fits the whole onset; `"upper_half"` fits the upper half of the slope, the variant for pass-bys of road vehicles, trains or aircraft (3.5, Note 1). |

**Returns:** The detected onsets, ordered in time (empty when none is found).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive `dt` or fewer than two samples. |

## impulse_adjustment

```python
impulse_adjustment(prominence: ArrayLike) -> np.ndarray
```

Adjustment `KI` to `LAeq` from the prominence (clause 8, Formula 2).

$K_\mathrm{I} = 1.8 \, (P - 5)$ dB for $P > 5$, else 0 dB. The
adjustment is made
to `LAeq,30min` on the basis of the single impulse with the highest `P`.
This helper applies the bare Formula 2; the clause 8 onset-rate
qualification (> 10 dB/s, clause 4.5) is enforced by
[`impulse_prominence`](/phonometry/reference/api/environment/impulsive-sound/#impulse_prominence).

**Parameters**

| Name | Description |
| :--- | :--- |
| `prominence` | Predicted prominence `P`. |

**Returns:** The adjustment `KI`, in dB, clamped at zero.

## impulse_prominence

```python
impulse_prominence(
    onset_rates: ArrayLike,
    level_differences: ArrayLike,
    *,
    assessment_period_min: float = 30.0,
) -> ImpulseProminenceResult
```

Governing prominence and adjustment of a set of impulses (clauses 7-8).

Evaluates the predicted prominence of each candidate impulse (Formula 1),
takes the highest among the *qualifying* impulses as the governing
prominence (clause 7) and derives its `LAeq` adjustment (Formula 2).
An event qualifies as an impulse only when its onset rate exceeds
10 dB/s (clause 4.5); clause 8 applies the adjustment "for sounds with
onset rates larger than 10 dB/s" only, so non-qualifying events cannot
produce a `KI` (an [`ImpulseProminenceWarning`](/phonometry/reference/api/environment/impulsive-sound/#impulseprominencewarning) reports them and
the adjustment is 0 dB when no event qualifies).

**Parameters**

| Name | Description |
| :--- | :--- |
| `onset_rates` | Onset rate of each impulse, in dB/s (> 0). |
| `level_differences` | Level difference of each impulse, in dB (> 0). |
| `assessment_period_min` | The assessment time interval the impulses were selected over, in minutes; the standard's default is 30 min (Clause 5), and the value is carried through to the fiche. |

**Returns:** An [`ImpulseProminenceResult`](/phonometry/reference/api/environment/impulsive-sound/#impulseprominenceresult) with the per-impulse and governing values and `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for empty input, mismatched lengths, a non-positive onset rate or level difference, or an assessment period that is not positive and finite. |

## ImpulseOnset

```python
ImpulseOnset(
    index_start: int,
    index_end: int,
    time_start: float,
    time_end: float,
    level_start: float,
    level_end: float,
    level_difference: float,
    onset_rate: float,
    prominence: float,
    qualifies: bool,
)
```

A single detected onset of `LpAF` (ISO/PAS 1996-3, Clause 3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `index_start` | Sample index of the starting point `s`. |
| `index_end` | Sample index of the end point `e`. |
| `time_start` | Time of the starting point, in seconds. |
| `time_end` | Time of the end point, in seconds. |
| `level_start` | Level `Ls` at the starting point, in dB. |
| `level_end` | Level `Le` at the end point, in dB. |
| `level_difference` | Level difference $\mathrm{LD} = L_\mathrm{e} - L_\mathrm{s}$, in dB (3.4). |
| `onset_rate` | Onset rate `OR`, in dB/s, the least-squares slope over the onset (3.5). |
| `prominence` | Predicted prominence `P` of this onset (Formula 2). |
| `qualifies` | Whether the onset rate exceeds 10 dB/s, so the onset can contribute an adjustment (Clause 6). |

## ImpulseProminenceResult

```python
ImpulseProminenceResult(
    onset_rates: np.ndarray,
    level_differences: np.ndarray,
    per_impulse: np.ndarray,
    qualifies: np.ndarray,
    prominence: float,
    adjustment: float,
    assessment_period_min: float = 30.0,
)
```

Prominence of a set of candidate impulses (NT ACOU 112:2002).

**Attributes**

| Name | Description |
| :--- | :--- |
| `onset_rates` | Onset rate of each impulse, in dB/s. |
| `level_differences` | Level difference of each impulse, in dB. |
| `per_impulse` | Predicted prominence `P` of each impulse (Formula 1). |
| `qualifies` | Whether each event qualifies as an impulse: onset rate above 10 dB/s (clause 4.5; clause 8 applies the adjustment "for sounds with onset rates larger than 10 dB/s" only). |
| `prominence` | The governing prominence: the highest `P` among the qualifying impulses (clause 7), or the highest overall (informational) when none qualifies. |
| `adjustment` | The LAeq adjustment `KI`, in dB, of the governing qualifying impulse (Formula 2); 0 dB when no event qualifies. |
| `assessment_period_min` | The assessment time interval the impulses were selected over, in minutes (Clause 5; 30 min by default). |

### ImpulseProminenceResult.plot()

```python
ImpulseProminenceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the adjustment curve `KI(P)` with the impulses marked.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ImpulseProminenceResult.report()

```python
ImpulseProminenceResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an impulsive-sound prominence assessment fiche to a PDF.

Writes a one-page assessment report following NT ACOU 112:2002 (carried
into ISO/PAS 1996-3:2022): the standard-basis line, an optional metadata
header (source/situation, client, measurement position, instrumentation
and date, always followed by this result's `assessment_period_min`), a
full-width per-impulse table (onset rate, level difference, predicted
prominence `P` and whether the onset qualifies as an impulse) above the
adjustment-curve plot `KI(P)` with the candidate impulses marked, the
boxed governing prominence `P` and the derived `LAeq` adjustment
`KI` (Formula 2), an optional verdict row and a prominence-category
note, and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare assessment fiche (body, result and disclaimer only). A supplied `requirement` is read as the maximum acceptable governing prominence `P` (a lower prominence passes). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for signature parity with the other fiches; the per-impulse table already shows every candidate, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `language` is not one of the supported languages, or if `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## ImpulseProminenceWarning

A supplied level rise does not qualify as an impulse (clause 4.5).

## impulsive_sound_adjustment

```python
impulsive_sound_adjustment(
    signal: SignalInput,
    fs: float | None = None,
    *,
    dt: float = 0.02,
    reference_pressure: float = 2e-05,
    calibration_offset: float = 0.0,
    onset_rate_method: _OnsetRateMethod = 'least_squares',
    laeq: float | None = None,
) -> ImpulsiveSoundResult
```

Objective prominence adjustment of an impulsive interval (ISO/PAS 1996-3).

Computes the `LpAF` history from the calibrated signal (Clause 4), detects
the onsets, evaluates the prominence of each and returns the governing
adjustment `KI` (Formula 3) of the most prominent qualifying impulse,
together with the source category (Clause 7) and the adjusted `LAeq`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Calibrated sound pressure signal of the candidate event, in pascal. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples: that factor is what puts them in pascal, and it is a different knob from `calibration_offset`, which shifts the finished level in decibels for a record that never was. |
| `fs` | Sampling rate of `signal`, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `dt` | Target `LpAF` sampling interval, in seconds (10-25 ms). |
| `reference_pressure` | Reference pressure, in pascal (default 20 uPa). |
| `calibration_offset` | Level offset, in dB, for signals not scaled to pascal. The adjustment `KI` is unaffected by it (Clause 8); only the reported levels shift. |
| `onset_rate_method` | `"least_squares"` (default) or `"upper_half"` for pass-bys (3.5, Note 1). |
| `laeq` | Equivalent level of the interval, in dB; when omitted it is computed from the A-weighted signal energy. |

**Returns:** An [`ImpulsiveSoundResult`](/phonometry/reference/api/environment/impulsive-sound/#impulsivesoundresult) with the level history, onsets, prominence, adjustment, category and adjusted `LAeq`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for invalid `fs`, `dt` or an empty signal. |

## ImpulsiveSoundResult

```python
ImpulsiveSoundResult(
    times: np.ndarray,
    levels: np.ndarray,
    dt: float,
    onsets: tuple[ImpulseOnset, ...],
    prominence: float,
    adjustment: float,
    category: str,
    laeq: float,
    adjusted_laeq: float,
)
```

Objective prominence of an impulsive interval (ISO/PAS 1996-3:2022).

**Attributes**

| Name | Description |
| :--- | :--- |
| `times` | Time of each `LpAF` sample, in seconds. |
| `levels` | A-weighted, F time-weighted level `LpAF`, in dB. |
| `dt` | Sampling interval of `levels`, in seconds. |
| `onsets` | The detected onsets, ordered in time (Clause 4). |
| `prominence` | Governing prominence `P`: the highest `P` among the qualifying onsets (Clause 5); `nan` when none qualifies. |
| `adjustment` | The `LAeq` adjustment `KI`, in dB (Formula 3); 0 dB when no onset qualifies. |
| `category` | Source category (Clause 7): `"not impulsive"`, `"regular impulsive"` or `"highly impulsive"`. |
| `laeq` | A-weighted equivalent level of the interval, in dB. |
| `adjusted_laeq` | `laeq + adjustment`, in dB. |

### ImpulsiveSoundResult.governing_onset

*property*

The qualifying onset with the highest prominence, or `None`.

### ImpulsiveSoundResult.plot()

```python
ImpulsiveSoundResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `LpAF` versus time with the detected onsets marked.

Draws the level history, the starting/end points of each onset, the
least-squares onset line and the level difference of the governing
impulse, annotated with the prominence, adjustment and category.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## ImpulsiveSoundWarning

No qualifying onset (gradient > 10 dB/s) was found in the interval.

## LevelHistory

```python
LevelHistory(times: np.ndarray, levels: np.ndarray, dt: float)
```

A frequency- and time-weighted level trace, and the axis it lives on.

What [`sound_pressure_level_history`](/phonometry/reference/api/environment/impulsive-sound/#sound_pressure_level_history) computes is `LpAF`: the
A-weighted, F time-weighted sound pressure level sampled every `dt`
seconds. It came back as a bare `(times, levels)` pair, which meant
the one thing every other result in this library offers, a plot that
knows what it is drawing, had to be written by hand each time.

For backward compatibility with that pair, the dataclass is iterable
and unpacks as `times, levels = sound_pressure_level_history(...)`,
the same way [`DecayCurve`](/phonometry/reference/api/rooms/acoustics/#decaycurve) replaced its own
tuple return.

**Attributes**

| Name | Description |
| :--- | :--- |
| `times` | Time of each sample, in seconds from the start. |
| `levels` | The level at each of those times, in dB. |
| `dt` | Sampling interval of `levels`, in seconds. |

### LevelHistory.plot()

```python
LevelHistory.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the level trace against time.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## predicted_prominence

```python
predicted_prominence(
    onset_rate: ArrayLike,
    level_difference: ArrayLike,
) -> np.ndarray
```

Predicted prominence `P` of an impulse (NT ACOU 112, clause 7).

$P = 3 \log_{10}(\text{onset rate}) + 2 \log_{10}(\text{level difference})$
(Formula 1), with $\log_{10}$
the base-10 logarithm. Both quantities are read from the A-weighted,
time-weighting-F level history: the onset rate is the slope of the onset in
dB/s and the level difference is the level rise over the onset in dB
(clauses 4.6-4.7). An impulse qualifies when its onset rate exceeds
`ONSET_RATE_LIMIT` (10 dB/s).

**Parameters**

| Name | Description |
| :--- | :--- |
| `onset_rate` | Onset rate(s), in dB/s (> 0). |
| `level_difference` | Level difference(s), in dB (> 0). |

**Returns:** The predicted prominence `P` (scalar inputs give a 0-d array).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive onset rate or level difference. |

## rating_level

```python
rating_level(
    laeq: ArrayLike,
    adjustment: ArrayLike,
    durations: ArrayLike,
    reference_time: float,
) -> float
```

Rating level over a reference time interval (clause 8, Note 1).

Combines the impulse-adjusted equivalent levels of the measurement
sub-intervals into a single rating level:

$$
L_{\mathrm{Ar},T} = 10 \log_{10}\!\left[ \frac{1}{T} \sum_N \Delta t_N \, 10^{(L_{\mathrm{Aeq},N} + K_{\mathrm{I},N})/10} \right]
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `laeq` | Equivalent level `LAeq,N` of each sub-interval, in dB. |
| `adjustment` | Adjustment `KI,N` of each sub-interval, in dB. |
| `durations` | Duration `dt_N` of each sub-interval (any time unit, consistent with `reference_time`). |
| `reference_time` | Reference time interval `T` (same unit as `durations`); commonly the sum of the durations. |

**Returns:** The rating level `LAr,T`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for mismatched lengths or a non-positive time. |

## sound_pressure_level_history

```python
sound_pressure_level_history(
    signal: SignalInput,
    fs: float | None = None,
    *,
    dt: float = 0.02,
    reference_pressure: float = 2e-05,
    calibration_offset: float = 0.0,
) -> LevelHistory
```

A frequency-weighted, F time-weighted level history `LpAF` (Clause 4).

The signal is A-weighted (IEC 61672-1), F time-weighted
($\tau = 125$ ms)
and sampled at intervals `dt` in the 10-25 ms range required by the
standard.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Calibrated sound pressure signal, in pascal. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples: that factor is what puts them in pascal, and it is a different knob from `calibration_offset`, which shifts the finished level in decibels for a record that never was. |
| `fs` | Sampling rate of `signal`, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `dt` | Target sampling interval of `LpAF`, in seconds (10-25 ms). |
| `reference_pressure` | Reference pressure, in pascal (default 20 uPa). |
| `calibration_offset` | Level offset added to `LpAF`, in dB, for signals recorded on a scale other than pascal. |

**Returns:** A [`LevelHistory`](/phonometry/reference/api/environment/impulsive-sound/#levelhistory), which unpacks as `(times, levels)` for the callers that always did: the sample times in seconds and `LpAF` in dB. The realised interval is `times[1] - times[0]` and may differ slightly from `dt` because it is an integer number of samples.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive `fs` or `dt` outside 10-25 ms. |
