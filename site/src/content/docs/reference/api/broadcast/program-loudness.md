---
title: "broadcast.program_loudness"
description: "Programme loudness and true-peak level (ITU-R BS.1770-5, EBU R 128)."
sidebar:
  label: "program_loudness"
---

Programme loudness and true-peak level (ITU-R BS.1770-5, EBU R 128).

Implements the objective multichannel loudness measurement algorithm of
ITU-R BS.1770-5 Annex 1 and its companions:

* **K-weighting** (Annex 1): the two-stage pre-filter (a shelving filter
  modelling the acoustic effect of a spherical head, then the RLB high-pass),
  specified as second-order sections with tabulated coefficients at 48 kHz
  (Tables 1 and 2) and re-derived here for other rates so the frequency
  response is preserved, as the Recommendation requires.
* **Programme (integrated) loudness** (Annex 1): mean-square power in gating
  blocks of 400 ms overlapping 75 %, channel-weighted summation (surround
  +1.5 dB, LFE excluded, Table 3), and the two-stage gate: an absolute
  threshold at -70 LKFS and a relative threshold 10 LU below the
  absolute-gated loudness (Formulae 3-7).
* **Channel weights for advanced sound systems** (Annex 3): the
  position-dependent weighting $G_i$ from the azimuth and elevation of
  each loudspeaker (Table 4), covering the BS.2051 configurations (Table 5).
* **True-peak level** (Annex 2): the inter-sample peak in dBTP estimated by
  oversampling to at least 192 kHz before taking the absolute maximum.
* **EBU Mode momentary and short-term loudness** (EBU Tech 3341): the
  ungated sliding-window loudness over 400 ms (M) and 3 s (S), with their
  maxima; EBU R 128 normalises Programme Loudness to -23.0 LUFS and limits
  the true peak to -1 dBTP.
* **Loudness range** (EBU Tech 3342): the 10th-to-95th percentile spread of
  the short-term loudness distribution after a cascaded gate (absolute
  -70 LUFS, then relative -20 LU, deliberately different from the -10 LU of
  the integrated measure).

The whole module is validated against the EBU Tech 3341 and Tech 3342
"minimum requirements" test signals with their official tolerances, and
against the Recommendation's own 997 Hz anchor (a 0 dB FS sine on one front
channel reads -3.01 LKFS).

Loudness values are returned in LUFS (the EBU name; identical to the LKFS of
ITU-R BS.1770). 1 LU is equivalent to 1 dB.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## channel_weight

```python
channel_weight(
    azimuth: ArrayLike,
    elevation: ArrayLike = 0.0,
) -> float | np.ndarray
```

Position-dependent channel weight `Gi` (BS.1770-5 Annex 3, Table 4).

Extends the Table 3 weights to arbitrarily placed loudspeakers of the
advanced sound systems (Recommendation ITU-R BS.2051): a mid-layer
loudspeaker to the side of the listener weighs 1.41 (+1.5 dB), every
other position 1.0. The LFE channels are excluded from the measurement
altogether (use a weight of 0 for them).

**Parameters**

| Name | Description |
| :--- | :--- |
| `azimuth` | Azimuth angle(s) of the loudspeaker position, in degrees (0 in front, positive to either side; only the magnitude matters). |
| `elevation` | Elevation angle(s), in degrees. |

**Returns:** The weight `Gi`: 1.41 where `|elevation| < 30` and `60 <= |azimuth| <= 120`, else 1.0. Float for scalar inputs.

## DEFAULT_CHANNEL_WEIGHTS

*Constant* (`dict`).

```python
DEFAULT_CHANNEL_WEIGHTS = {1: (1.0,), 2: (1.0, 1.0), 5: (1.0, 1.0, 1.0, 1.41, 1.41), 6: (1.0, 1.0, 1.0, 0.0, 1.41, 1.41)}
```

## integrated_loudness

```python
integrated_loudness(
    x: list[float] | np.ndarray,
    fs: float,
    weights: ArrayLike | None = None,
) -> float
```

Programme (integrated) loudness in LUFS (BS.1770-5 Annex 1).

K-weights each channel, averages the power over gating blocks of 400 ms
overlapping 75 %, sums the channels with their weights `Gi` and
applies the two-stage gate: blocks below -70 LKFS are dropped, a
relative threshold 10 LU below the loudness of the survivors is
computed, and the loudness of the blocks above both thresholds is the
programme loudness (Formulae 3-7). This is the quantity EBU R 128
normalises to -23.0 LUFS.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D mono or 2D `[channels, samples]`), full-scale units (1.0 = 0 dBFS). |
| `fs` | Sample rate, Hz. |
| `weights` | Per-channel weights `Gi`, or `None` for the Table 3 defaults by channel count (see [`DEFAULT_CHANNEL_WEIGHTS`](/phonometry/reference/api/broadcast/program-loudness/#default_channel_weights) and [`channel_weight`](/phonometry/reference/api/broadcast/program-loudness/#channel_weight)). |

**Returns:** The programme loudness, LUFS (`-inf` when the signal is shorter than one gating block or entirely below the absolute gate).

## k_weighting

```python
k_weighting(x: list[float] | np.ndarray, fs: float) -> np.ndarray
```

Apply the two-stage K-weighting pre-filter (BS.1770-5 Annex 1).

Stage 1 models the acoustic effect of the head as a rigid sphere (a
high-frequency shelf of about +4 dB); stage 2 is the RLB revised
low-frequency B-curve high-pass. Their concatenation is the K-weighting.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D `[channels, samples]`), linear units. |
| `fs` | Sample rate, Hz. |

**Returns:** The K-weighted signal, same shape as the input.

## k_weighting_coefficients

```python
k_weighting_coefficients(
    fs: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
```

K-weighting biquad coefficients (BS.1770-5 Annex 1, Tables 1-2).

At 48 kHz the tabulated values are returned verbatim; at any other rate
the biquads are re-derived through their analog prototypes so the
frequency response matches the 48 kHz specification, as the
Recommendation requires. At 32 kHz and above the redesigned response
stays within 0.02 dB of the specification across the audio band; at
16 kHz the bilinear warping near Nyquist grows to about 0.13 dB while
the 997 Hz anchor still holds within 0.03 LU. Below 16 kHz the warping
would break the +/-0.1 LU metering tolerance, so such rates are
rejected.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate, Hz (16 kHz or higher). |

**Returns:** `(stage1, stage2)`, each a `(b, a)` coefficient pair: the spherical-head shelving filter and the RLB high-pass filter.

## k_weighting_response

```python
k_weighting_response(
    fs: float = 48000.0,
    *,
    frequencies: ArrayLike | None = None,
    n: int = 512,
) -> KWeightingResponse
```

K-weighting magnitude frequency response (BS.1770-5 Annex 1, Tables 1-2).

Evaluates the two-stage K-weighting pre-filter as a transfer function and
returns its magnitude in dB. The biquad coefficients come from
[`k_weighting_coefficients`](/phonometry/reference/api/broadcast/program-loudness/#k_weighting_coefficients) (the verbatim 48 kHz tables, or the
analog-prototype redesign at other rates), so the response is the exact
frequency-domain counterpart of the filter [`k_weighting`](/phonometry/reference/api/broadcast/program-loudness/#k_weighting) applies; no
coefficients are re-derived here. Each stage is evaluated with
`scipy.signal.freqz` and their magnitudes summed in dB.

The combined response tends to the +4 dB shelf plateau of the spherical-head
stage at high frequency and rolls off below a few hundred hertz through the
RLB high-pass; near 1 kHz it passes through the ~0.69 dB gain that the
`-0.691` constant of Formula 2 cancels.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate the response is evaluated at, Hz (16 kHz or higher, as [`k_weighting_coefficients`](/phonometry/reference/api/broadcast/program-loudness/#k_weighting_coefficients) requires; default 48 kHz). |
| `frequencies` | Explicit frequency grid, in Hz (each in the half-open interval `(0, fs/2]`); `None` (the default) uses `n` log-spaced points from 10 Hz to the Nyquist rate. |
| `n` | Number of log-spaced points when `frequencies` is `None` (default 512); ignored otherwise. |

**Returns:** A frozen [`KWeightingResponse`](/phonometry/reference/api/broadcast/program-loudness/#kweightingresponse).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `fs` is below 16 kHz, `n` is not positive, or a supplied frequency is outside `(0, fs/2]`. |

## KWeightingResponse

```python
KWeightingResponse(
    frequencies: np.ndarray,
    magnitude_db: np.ndarray,
    shelf_db: np.ndarray,
    highpass_db: np.ndarray,
    fs: float,
)
```

Magnitude frequency response of the K-weighting pre-filter (BS.1770-5).

The two-stage K-weighting of Annex 1 evaluated as a transfer function: the
spherical-head high-frequency shelf (stage 1, about +4 dB above 2 kHz), the
RLB high-pass (stage 2) and their combination, each as a magnitude in dB
over a shared logarithmic frequency grid. It is the frequency-domain view of
the same biquads that [`k_weighting`](/phonometry/reference/api/broadcast/program-loudness/#k_weighting) applies in the time domain, built
from the tabulated (or redesigned) coefficients of [`k_weighting_coefficients`](/phonometry/reference/api/broadcast/program-loudness/#k_weighting_coefficients).

Build it with [`k_weighting_response`](/phonometry/reference/api/broadcast/program-loudness/#k_weighting_response); the frozen instance then exposes
`plot` (the magnitude response versus frequency).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency grid, in Hz (log-spaced up to the Nyquist rate by default). |
| `magnitude_db` | Combined K-weighting magnitude response, in dB (the product of the two stages). It tends to the +4 dB shelf plateau at high frequency and rolls off below a few hundred hertz. |
| `shelf_db` | Stage 1 (spherical-head shelf) magnitude response, in dB. |
| `highpass_db` | Stage 2 (RLB high-pass) magnitude response, in dB. |
| `fs` | Sample rate the response was evaluated at, in Hz. |

### KWeightingResponse.plot()

```python
KWeightingResponse.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the K-weighting magnitude response versus frequency.

Draws the combined K-weighting magnitude (dB) on a logarithmic
frequency axis, with the two stages (the +4 dB shelf and the RLB
high-pass) as light companion curves.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the combined-curve `plot` call. |

**Returns:** The axes.

## loudness_range

```python
loudness_range(short_term_loudness: ArrayLike) -> float
```

Loudness range LRA of a short-term loudness series (EBU Tech 3342).

The input is the ungated short-term loudness (3 s sliding window,
computed at 10 Hz or faster, as specified in ITU-R BS.1770). A cascaded
gate first drops readings below the absolute threshold (-70 LUFS), then
readings more than 20 LU below the power mean of what survived; the LRA
is the spread between the 10th and 95th percentiles of the remaining
distribution, following the Tech 3342 reference implementation.

**Parameters**

| Name | Description |
| :--- | :--- |
| `short_term_loudness` | Short-term loudness readings, LUFS. |

**Returns:** The loudness range, in LU (0.0 when no reading survives the gate).

## program_loudness

```python
program_loudness(
    x: list[float] | np.ndarray,
    fs: float,
    weights: ArrayLike | None = None,
    *,
    momentary_step: float = 0.01,
    short_term_step: float = 0.1,
    oversample: int | None = None,
) -> ProgramLoudnessResult
```

Measure a programme per EBU R 128: I, M, S, LRA and true peak.

One call computes the full EBU Mode measurement set (EBU Tech 3341) of a
finished programme:

* the **integrated loudness** `I` with the BS.1770-5 two-stage gate
  (the quantity normalised to -23.0 LUFS by EBU R 128);
* the ungated **momentary** (400 ms) and **short-term** (3 s) loudness
  series with their maxima;
* the **loudness range** `LRA` from the short-term series with the
  Tech 3342 cascaded gate and 10-95 percentile spread; and
* the **true-peak level** per channel and overall (Annex 2; EBU R 128
  permits at most -1 dBTP during production).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D mono or 2D `[channels, samples]`), full-scale units (1.0 = 0 dBFS). |
| `fs` | Sample rate, Hz. |
| `weights` | Per-channel weights `Gi`, or `None` for the Table 3 defaults by channel count (see [`DEFAULT_CHANNEL_WEIGHTS`](/phonometry/reference/api/broadcast/program-loudness/#default_channel_weights); for other layouts derive the weights with [`channel_weight`](/phonometry/reference/api/broadcast/program-loudness/#channel_weight)). |
| `momentary_step` | Hop of the momentary series, s (default 10 ms, a 100 Hz update rate; EBU Tech 3341 requires at least 10 Hz). |
| `short_term_step` | Hop of the short-term series, s (default 100 ms, the 10 Hz minimum rate that Tech 3342 requires for the LRA input). |
| `oversample` | True-peak oversampling factor; `None` picks the smallest factor reaching 192 kHz (4 at 48 kHz). |

**Returns:** The frozen [`ProgramLoudnessResult`](/phonometry/reference/api/broadcast/program-loudness/#programloudnessresult).

## ProgramLoudnessResult

```python
ProgramLoudnessResult(
    integrated: float,
    loudness_range: float,
    true_peak: float,
    momentary: np.ndarray,
    momentary_time: np.ndarray,
    short_term: np.ndarray,
    short_term_time: np.ndarray,
    max_momentary: float,
    max_short_term: float,
    relative_threshold: float,
    lra_low: float,
    lra_high: float,
    true_peak_per_channel: np.ndarray,
    channel_weights: np.ndarray,
    fs: float,
)
```

EBU Mode loudness measurement of a programme (BS.1770-5 / EBU R 128).

**Attributes**

| Name | Description |
| :--- | :--- |
| `integrated` | Programme loudness `I` (gated, Annex 1), LUFS. |
| `loudness_range` | Loudness range `LRA` (EBU Tech 3342), LU. |
| `true_peak` | Maximum true-peak level over the channels, dBTP. |
| `momentary` | Momentary loudness series `M` (400 ms, ungated), LUFS. |
| `momentary_time` | Time of each `M` reading (window end), s. |
| `short_term` | Short-term loudness series `S` (3 s, ungated), LUFS. |
| `short_term_time` | Time of each `S` reading (window end), s. |
| `max_momentary` | Maximum momentary loudness, LUFS. |
| `max_short_term` | Maximum short-term loudness, LUFS. |
| `relative_threshold` | Relative gating threshold of the integrated measurement (10 LU below the absolute-gated loudness), LUFS. |
| `lra_low` | Lower (10th percentile) edge of the loudness range, LUFS. |
| `lra_high` | Upper (95th percentile) edge of the loudness range, LUFS. |
| `true_peak_per_channel` | True-peak level of each channel, dBTP. |
| `channel_weights` | The channel weights `Gi` used. |
| `fs` | Sample rate of the analysed signal, Hz. |

### ProgramLoudnessResult.plot()

```python
ProgramLoudnessResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot momentary and short-term loudness over time, with the
integrated loudness and the loudness range annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

### ProgramLoudnessResult.report()

```python
ProgramLoudnessResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    tolerance: str = 'qc',
) -> str
```

Render an EBU R 128 programme-loudness compliance fiche to a PDF.

Writes a one-page broadcast loudness-compliance sheet: the
standard-basis line, an optional metadata header block, a full-width
compliance table (integrated loudness and maximum true peak carry the
verdict; the loudness range and the momentary/short-term maxima are
informational), the result's own loudness-vs-time `plot`, the
boxed `I = X LUFS (LRA = Y LU, max TP = Z dBTP)` result, a combined
PASS/FAIL verdict row and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a measurement fiche (compliance table, plot and verdict only). A supplied `requirement` is read as the target programme loudness in LUFS (defaulting to the EBU R 128 -23.0 LUFS). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform signature; it has no effect on the single-layout programme-loudness fiche. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |
| `tolerance` | Programme-loudness tolerance rule of EBU R 128: `"qc"` (default) applies the +-0.2 LU allowance of item i) for measurement errors in loudness workflows such as Quality Control; `"live"` applies the +-1.0 LU tolerance of item h), permitted only where attaining the Target Level is not achievable practically (for example, live programmes). The fiche prints the applied rule and its R 128 item. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `tolerance` is not `"qc"`/`"live"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## true_peak_level

```python
true_peak_level(
    x: list[float] | np.ndarray,
    fs: float,
    oversample: int | None = None,
) -> float | np.ndarray
```

True-peak level in dBTP (BS.1770-5 Annex 2).

Estimates the inter-sample peak by oversampling the signal to at least
192 kHz with a polyphase FIR interpolator before taking the absolute
maximum (the same machinery behind
[`phonometry.signals.levels.lc_peak`](/phonometry/reference/api/signals/levels/#lc_peak)). At 48 kHz this is the
4-times oversampling of the Annex 2 block diagram; higher input rates
need proportionately less. The initial 12.04 dB attenuation of the
Annex 2 integer pipeline is unnecessary in floating point and omitted.

dBTP is referred to 100 % full scale: a sample value of 1.0 is 0 dBTP,
and inter-sample peaks above full scale give positive values.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D `[channels, samples]`), full-scale units (1.0 = 0 dBFS). |
| `fs` | Sample rate, Hz. |
| `oversample` | Integer oversampling factor >= 1, or `None` (the default) for the smallest factor whose oversampled rate reaches 192 kHz (4 at 48 kHz, 2 at 96 kHz, 1 at 192 kHz and above, matching the Annex 2 guidance that higher input rates need proportionately less oversampling). |

**Returns:** The true-peak level in dBTP: a float for 1D input, an array of shape `(channels,)` for 2D input.
