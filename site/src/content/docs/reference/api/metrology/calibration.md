---
title: "metrology.calibration"
description: "Calibration utilities for mapping digital signals to physical SPL levels."
sidebar:
  label: "calibration"
---

Calibration utilities for mapping digital signals to physical SPL levels.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## CalibrationWarning

The calibration reference recording looks unreliable.

## sensitivity

```python
sensitivity(
    ref_signal: SignalInput,
    target_spl: float = ...,
    ref_pressure: float = ...,
    *,
    fs: int,
    validate: bool = ...,
    max_fluctuation_db: float | None = ...,
    frequency: float = ...,
    narrowband: Literal[True],
) -> float

sensitivity(
    ref_signal: SignalInput,
    target_spl: float = ...,
    ref_pressure: float = ...,
    fs: int | None = ...,
    validate: bool = ...,
    max_fluctuation_db: float | None = ...,
    frequency: float = ...,
    narrowband: Literal[False] = ...,
) -> float
```

Calculate the calibration factor (multiplier) to convert digital units
to Pascals based on a reference recording (e.g., 1kHz @ 94dB).

When `fs` is provided (and `validate` is True), the recording's
stability is checked the way IEC 60942:2017 specifies for the calibrator
itself (5.3.3): levels are measured with time-weighting F and the
*short-term level fluctuation* (the absolute difference between each of
the maximum and minimum levels and the mean level) must not exceed the
Table 2 acceptance limit for the calibrator class (class 1: 0.07 dB at
and above 160 Hz, relaxed to 0.10 dB above 63 Hz and below 160 Hz, and to
0.20 dB for the 31.5-63 Hz rows where the F time-weighting itself ripples;
below Table 2's 31.5 Hz span the strict 0.07 dB applies). A larger
fluctuation usually means a badly coupled microphone or handling noise in
the recording, which would silently corrupt every calibrated level; a
[`CalibrationWarning`](/phonometry/reference/api/metrology/calibration/#calibrationwarning) is emitted.

:::note
IEC 60942 certifies calibrators over 60 s of operation sampled
at least 30 times; this check applies the same criterion to whatever
settled portion of the recording is available (>= 1 s after the 1 s
integrator attack).
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `ref_signal` | Recording of the calibration tone. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) for its rate; a calibration factor it carries is deliberately **not** applied, because this function is what produces such a factor and folding an existing one in would calibrate the calibration. |
| `target_spl` | The known SPL level of the calibrator (default 94 dB). |
| `ref_pressure` | Reference pressure (default 20 microPascals). |
| `fs` | Sample rate of the recording in Hz. Required for the stability validation; without it the check is skipped. A [`Signal`](/phonometry/reference/api/io/io/#signal) supplies it, so a read take gets the validation for free, and an explicit value that disagrees raises. |
| `validate` | If True (default) and `fs` is given, warn when the recording's short-term level fluctuation exceeds the limit. |
| `max_fluctuation_db` | Explicit fluctuation limit in dB. Default (None) resolves the IEC 60942:2017 Table 2 class 1 limit for `frequency`. |
| `frequency` | Nominal frequency of the calibration tone in Hz (default 1000.0), used to select the Table 2 row. |
| `narrowband` | If True (requires `fs`), estimate the tone level with a coherent single-frequency (Goertzel) detector locked to the tone near `frequency` instead of the full-band RMS. This rejects broadband hum/noise in the reference take, which otherwise inflates the RMS and shrinks the factor by $-10 \log_{10}(1 + 1/\mathrm{SNR})$ (about -0.44 dB at 20 dB SNR), silently biasing every subsequent level. The default (False) keeps the exact legacy broadband-RMS behaviour; enable it for noisy coupler recordings. |

**Returns:** Calibration factor (sensitivity multiplier).
