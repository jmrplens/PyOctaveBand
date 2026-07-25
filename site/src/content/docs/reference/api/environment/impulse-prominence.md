---
title: "environmental.impulse_prominence"
description: "Public API of phonometry.environmental.impulse_prominence (auto-generated)."
sidebar:
  label: "impulse_prominence"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Prominence of impulsive sounds and the LAeq adjustment (NT ACOU 112:2002).

Noise with prominent impulses is more annoying than a steady sound of the same
equivalent level, so the Nordtest method adds an adjustment `KI` to the
measured `LAeq`. The audibility of an impulse is captured by the **predicted
prominence** `P`, a logarithmic measure of the onset rate and the level
difference of the impulse (clause 7):

`P = 3*lg(onset_rate) + 2*lg(level_difference)`   (Formula 1)

From the impulse with the highest prominence over a 30-minute period, a
graduated adjustment follows (clause 8):

`KI = 1.8*(P - 5)` dB for `P > 5`, else `0`   (Formula 2)

and the rating level over a reference time interval combines the adjusted
sub-interval levels (clause 8, Note 1). An impulse qualifies when its onset
rate exceeds 10 dB/s (clauses 4.5-4.7); non-qualifying level rises receive
no adjustment (clause 8 applies only "for sounds with onset rates larger
than 10 dB/s").

## impulse_adjustment

```python
impulse_adjustment(prominence: ArrayLike) -> np.ndarray
```

Adjustment `KI` to `LAeq` from the prominence (clause 8, Formula 2).

`KI = 1.8*(P - 5)` dB for `P > 5`, else `0` dB. The adjustment is made
to `LAeq,30min` on the basis of the single impulse with the highest `P`.
This helper applies the bare Formula 2; the clause 8 onset-rate
qualification (> 10 dB/s, clause 4.5) is enforced by
[`impulse_prominence`](/phonometry/reference/api/environment/impulse-prominence/#impulse_prominence).

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
produce a `KI` (an [`ImpulseProminenceWarning`](/phonometry/reference/api/environment/impulse-prominence/#impulseprominencewarning) reports them and
the adjustment is 0 dB when no event qualifies).

**Parameters**

| Name | Description |
| :--- | :--- |
| `onset_rates` | Onset rate of each impulse, in dB/s (> 0). |
| `level_differences` | Level difference of each impulse, in dB (> 0). |
| `assessment_period_min` | The assessment time interval the impulses were selected over, in minutes; the standard's default is 30 min (Clause 5), and the value is carried through to the fiche. |

**Returns:** An [`ImpulseProminenceResult`](/phonometry/reference/api/environment/impulse-prominence/#impulseprominenceresult) with the per-impulse and governing values and `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for empty input, mismatched lengths, a non-positive onset rate or level difference, or an assessment period that is not positive and finite. |

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
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## ImpulseProminenceWarning

A supplied level rise does not qualify as an impulse (clause 4.5).

## predicted_prominence

```python
predicted_prominence(
    onset_rate: ArrayLike,
    level_difference: ArrayLike,
) -> np.ndarray
```

Predicted prominence `P` of an impulse (NT ACOU 112, clause 7).

`P = 3*lg(onset_rate) + 2*lg(level_difference)` (Formula 1), with `lg`
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

```text
LAr,T = 10*lg( (1/T) * sum_N dt_N * 10**((LAeq,N + KI,N) / 10) )
```

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
