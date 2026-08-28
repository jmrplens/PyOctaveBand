---
title: "signals.levels"
description: "Integrated and statistical sound levels (Leq, LAeq, LN percentiles)."
sidebar:
  label: "levels"
---

Integrated and statistical sound levels (Leq, LAeq, LN percentiles).

Every function here accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) in place of
the bare `(x, fs)` pair: the object read from a measurement file
already knows its sample rate and, when calibrated, its
digital-to-pascal factor, so asking the caller to repeat either is
asking for a transcription error. All of them or none: a function that
took the object but silently dropped its calibration -- one row away
from functions that honour it -- would compute a wrong level that looks
right, the exact failure the object exists to prevent. The bare-array
signatures are unchanged -- a plain array with an explicit `fs` and
`calibration_factor` computes exactly what it always did.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## laeq

```python
laeq(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    calibration_factor: float | None = None,
    dbfs: bool = False,
) -> float | np.ndarray
```

A-weighted equivalent continuous sound level (LAeq).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D [channels, samples]) in raw pressure units, or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. |
| `fs` | Sample rate in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `calibration_factor` | Multiplier converting digital units to Pascals. Precedence as in [`leq`](/phonometry/reference/api/signals/levels/#leq): explicit value, then a calibrated Signal's own factor, then 1.0. |
| `dbfs` | If True, return dBFS instead of dB SPL. |

**Returns:** Scalar for 1D input, array of shape (channels,) for 2D input.

## lc_peak

```python
lc_peak(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    calibration_factor: float | None = None,
    dbfs: bool = False,
    oversample: int = 8,
) -> float | np.ndarray
```

C-weighted peak sound level, LCpeak (IEC 61672-1:2013, subclause 5.13).

The absolute maximum of the C-weighted signal, expressed in dB. This is
the quantity used by occupational-noise regulations (e.g. 135/137/140
dB(C) action limits). Verified against the reference one-cycle and
half-cycle responses of BS EN 61672-1:2013 Table 5 in the test suite.

The true peak of a continuous waveform generally falls *between* samples.
A raw on-grid maximum therefore under-reads sustained high-frequency
tones (worst near integer samples-per-cycle rates, e.g. an 8 kHz tone at
fs = 48 kHz is 6.0 samples/cycle and under-reads by up to ~1.15 dB). The
C-weighted signal is polyphase-oversampled by `oversample` before the
maximum is taken, recovering the inter-sample peak to within about
+/-0.5 dB of the analytic value.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D [channels, samples]) in raw pressure units, or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. |
| `fs` | Sample rate in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `calibration_factor` | Multiplier converting digital units to Pascals. Precedence as in [`leq`](/phonometry/reference/api/signals/levels/#leq): explicit value, then a calibrated Signal's own factor, then 1.0. |
| `dbfs` | If True, return dBFS (0 dB = peak 1.0) instead of dB SPL. |
| `oversample` | Integer oversampling factor applied before peak detection (default 8, the audit-validated value). Use 1 to disable oversampling and detect the peak on the original sample grid. |

**Returns:** Scalar for 1D input, array of shape (channels,) for 2D input.

## leq

```python
leq(
    x: Signal | list[float] | np.ndarray,
    calibration_factor: float | None = None,
    dbfs: bool = False,
) -> float | np.ndarray
```

Equivalent continuous sound level (Leq) over the whole signal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D [channels, samples]) in raw pressure units, or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. |
| `calibration_factor` | Multiplier converting digital units to Pascals. Precedence: an explicit value always wins; `None` (the default) takes the factor a calibrated [`Signal`](/phonometry/reference/api/io/io/#signal) carries, and falls back to 1.0 (levels in digital units) for everything else. |
| `dbfs` | If True, return dBFS (0 dB = RMS 1.0) instead of dB SPL; calibration does not apply. |

**Returns:** Scalar for 1D input, array of shape (channels,) for 2D input.

## lex_8h

```python
lex_8h(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    duration_hours: float | None = None,
    calibration_factor: float | None = None,
) -> float | np.ndarray
```

Normalized 8-h average sound level, LEX,8h (IEC 61252, 3.3).

The daily personal noise exposure level: the steady level that, sustained
over a nominal 8 h working day, carries the same A-weighted sound
exposure as the measured event. Identical to LEP,d (Directive 86/188/EEC)
and LEX,8h of ISO 1999 (BS EN 61252:1995, 3.3 NOTES 5-6).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal in raw pressure units (1D or 2D), or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. |
| `fs` | Sample rate in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `duration_hours` | Exposure period the input represents, in hours. Default: the recording duration itself. |
| `calibration_factor` | Multiplier converting digital units to Pascals. Precedence as in [`leq`](/phonometry/reference/api/signals/levels/#leq): explicit value, then a calibrated Signal's own factor, then 1.0. |

**Returns:** LEX,8h in dB (scalar or per-channel array).

## ln_levels

```python
ln_levels(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    n: Sequence[int] = (10, 50, 90),
    mode: str = 'fast',
    weighting: str | None = None,
    calibration_factor: float | None = None,
    dbfs: bool = False,
) -> dict[int, float | np.ndarray]
```

Statistical percentile levels (LN) from the time-weighted level envelope.

L10 is the level exceeded 10% of the time (90th percentile of the level
distribution), L90 the level exceeded 90% of the time, etc.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D [channels, samples]) in raw pressure units, or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. |
| `fs` | Sample rate in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `n` | Percentile exceedance values, e.g. (10, 50, 90). |
| `mode` | Time weighting for the envelope: 'fast', 'slow' or 'impulse'. |
| `weighting` | Optional frequency weighting, any curve accepted by [`weighting_filter`](/phonometry/reference/api/filters/weighting/#weighting_filter): 'A', 'B', 'C', 'D', 'G', 'AU', '468' or 'Z'. None (the default) and 'Z' both leave the signal unweighted. |
| `calibration_factor` | Multiplier converting digital units to Pascals. Precedence as in [`leq`](/phonometry/reference/api/signals/levels/#leq): explicit value, then a calibrated Signal's own factor, then 1.0. |
| `dbfs` | If True, return dBFS instead of dB SPL. |

**Returns:** Dict mapping each N to its level (scalar for 1D input, array (channels,) for 2D input).

## sel

```python
sel(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    weighting: str | None = None,
    calibration_factor: float | None = None,
    dbfs: bool = False,
) -> float | np.ndarray
```

Sound exposure level (SEL / LAE): the event level normalized to 1 second.

$\text{SEL} = L_{\mathrm{eq},T} + 10 \log_{10}(T / 1\,\text{s})$, the
standard single-event metric
(aircraft flyovers, train passes). With `weighting="A"` this is LAE as
defined by IEC 61672-1:2013 (verified against the Table 4 toneburst
reference responses, Equation 8, in the test suite).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal covering the whole event (1D or 2D), or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. |
| `fs` | Sample rate in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `weighting` | Optional frequency weighting, any curve accepted by [`weighting_filter`](/phonometry/reference/api/filters/weighting/#weighting_filter): 'A', 'B', 'C', 'D', 'G', 'AU', '468' or 'Z'. None (the default) and 'Z' both leave the signal unweighted. |
| `calibration_factor` | Multiplier converting digital units to Pascals. Precedence as in [`leq`](/phonometry/reference/api/signals/levels/#leq): explicit value, then a calibrated Signal's own factor, then 1.0. |
| `dbfs` | If True, reference digital full scale instead of 20 uPa. |

**Returns:** Scalar for 1D input, array of shape (channels,) for 2D input.

## sound_exposure

```python
sound_exposure(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    duration_hours: float | None = None,
    calibration_factor: float | None = None,
) -> float | np.ndarray
```

A-weighted sound exposure E in pascal-squared hours (IEC 61252, 3.1).

The time integral of the squared A-weighted sound pressure. By default
the input is the whole event (E integrates over `len(x)/fs`); pass
`duration_hours` to treat the input as a representative sample of a
longer exposure period (E = mean-square * duration). Anchors from
BS EN 61252:1995 (3.3 NOTE 4): 3.2 Pa²h \<-> LEX,8h of exactly 90 dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal in raw pressure units (1D or 2D), or a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) read from a measurement file. |
| `fs` | Sample rate in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `duration_hours` | Exposure period the input represents, in hours. Default: the recording duration itself. |
| `calibration_factor` | Multiplier converting digital units to Pascals. Precedence as in [`leq`](/phonometry/reference/api/signals/levels/#leq): explicit value, then a calibrated Signal's own factor, then 1.0. |

**Returns:** Exposure in Pa²·h (scalar or per-channel array).
