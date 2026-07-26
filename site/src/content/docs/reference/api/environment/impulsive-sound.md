---
title: "environmental.impulsive_sound"
description: "Objective prominence of impulsive sounds and the LAeq adjustment (ISO/PAS 1996-3:2022)."
sidebar:
  label: "impulsive_sound"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Objective prominence of impulsive sounds and the `LAeq` adjustment
(ISO/PAS 1996-3:2022).

ISO/PAS 1996-3 objectively categorises a source by how prominently its
impulsive sound is perceived and derives an adjustment `KI` (typically in the
range 0,0 dB to 9,0 dB) that is added to `LAeq`. Unlike the closed-form
[`impulse_prominence`](/phonometry/reference/api/environment/impulse-prominence/) helpers (NT ACOU 112, which
take the onset rate and level difference as inputs), this module implements the
*objective measurement chain* that reads those quantities directly from a
calibrated time signal.

The chain follows the standard:

* the A frequency-weighted, F time-weighted sound pressure level `LpAF` is
  computed from the signal and sampled at 10-25 ms intervals (Clause 4);
* an *onset* is a contiguous part of the positive slope of `LpAF` where the
  gradient exceeds 10 dB/s; its **starting** and **end** points are found from
  procedures a) to d) of Clause 4, merging events separated by less than 50 ms
  (Clause 3.3, Figure 2);
* for each onset the **level difference** `LD = Le - Ls` and the **onset
  rate** `OR` (the least-squares slope over the onset) are measured
  (Clauses 3.4, 3.5, Figures 1 and 2);
* the **prominence** `P = 3*lg(OR) + 2*lg(LD)` follows (Clause 5, Formula 2)
  and the impulse with the highest `P` gives the **adjustment**
  `KI = 1.8*(P - 5)` dB for `P > 5`, else 0 dB (Clause 6, Formula 3);
* the source is categorised (Clause 7) as *not impulsive* (`KI = 0`),
  *regular impulsive* (`0 < KI <= 5`) or *highly impulsive* (`KI > 5`).

The prominence and adjustment formulae are shared with NT ACOU 112 (both derive
from Pedersen's method) and are reused from
[`impulse_prominence`](/phonometry/reference/api/environment/impulse-prominence/). The method for determining
`KI` is not sensitive to the absolute calibration of the equipment
(Clause 8): onset rate and level difference are level *differences*, so the
adjustment is unchanged by a constant offset. Only the reported `LAeq` and the
adjusted `LAeq` depend on calibration.

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
`LD = Le - Ls` (3.4), its onset rate (3.5) and its prominence `P`.

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
| `level_difference` | Level difference `LD = Le - Ls`, in dB (3.4). |
| `onset_rate` | Onset rate `OR`, in dB/s, the least-squares slope over the onset (3.5). |
| `prominence` | Predicted prominence `P` of this onset (Formula 2). |
| `qualifies` | Whether the onset rate exceeds 10 dB/s, so the onset can contribute an adjustment (Clause 6). |

## impulsive_sound_adjustment

```python
impulsive_sound_adjustment(
    signal: ArrayLike,
    fs: float,
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
| `signal` | Calibrated sound pressure signal of the candidate event, in pascal. |
| `fs` | Sampling rate of `signal`, in Hz. |
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

## sound_pressure_level_history

```python
sound_pressure_level_history(
    signal: ArrayLike,
    fs: float,
    *,
    dt: float = 0.02,
    reference_pressure: float = 2e-05,
    calibration_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]
```

A frequency-weighted, F time-weighted level history `LpAF` (Clause 4).

The signal is A-weighted (IEC 61672-1), F time-weighted (`tau = 125 ms`)
and sampled at intervals `dt` in the 10-25 ms range required by the
standard.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Calibrated sound pressure signal, in pascal. |
| `fs` | Sampling rate of `signal`, in Hz. |
| `dt` | Target sampling interval of `LpAF`, in seconds (10-25 ms). |
| `reference_pressure` | Reference pressure, in pascal (default 20 uPa). |
| `calibration_offset` | Level offset added to `LpAF`, in dB, for signals recorded on a scale other than pascal. |

**Returns:** `(times, levels)`, the sample times in seconds and `LpAF` in dB. The realised interval is `times[1] - times[0]` and may differ slightly from `dt` because it is an integer number of samples.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive `fs` or `dt` outside 10-25 ms. |
