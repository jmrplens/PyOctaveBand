---
title: "signals.windows"
description: "Figures of merit of a spectral-analysis taper (Harris 1978)."
sidebar:
  label: "windows"
---

Figures of merit of a spectral-analysis taper (Harris 1978).

The window is the one choice every Fourier estimator forces and the one
the estimators of [`phonometry.signals.spectra`](/phonometry/reference/api/signals/spectra/) leave open in their
`window` parameter. Harris (1978, *On the use of windows for harmonic
analysis with the discrete Fourier transform*) turned that choice into a
table of numbers: equivalent noise bandwidth, coherent gain, scalloping
loss, worst-case processing loss, highest sidelobe level and the -3 dB
main-lobe width - the trade-off between resolution, leakage and
amplitude accuracy, read rather than guessed.

[`window_metrics`](/phonometry/reference/api/signals/windows/#window_metrics) computes those figures for any taper
`scipy.signal.get_window` accepts, sampling it DFT-even (periodic)
exactly as the Welch estimators of
[`phonometry.signals.spectra`](/phonometry/reference/api/signals/spectra/) and the multitaper estimator of
[`phonometry.signals.multitaper`](/phonometry/reference/api/signals/multitaper/) apply it, so the numbers describe
the window as this library actually uses it.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## window_metrics

```python
window_metrics(
    window: str | tuple[Any, ...],
    n: int = 1024,
) -> WindowMetricsResult
```

Figures of merit of a spectral-analysis taper (Harris 1978).

Computes the numbers behind the window trade-off for any taper the
`window` parameter of the [`phonometry.signals.spectra`](/phonometry/reference/api/signals/spectra/)
estimators accepts: the
equivalent noise bandwidth and coherent gain (closed forms of the
samples, machine-exact), and the scalloping loss, worst-case
processing loss, highest sidelobe level and -3 dB main-lobe width
(measured on the spectrum of the sampled window, oversampled by
zero-padding). The window is sampled DFT-even (periodic), exactly as
the Welch estimators apply it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `window` | Window name or `(name, param)` tuple, anything `scipy.signal.get_window` accepts (e.g. `'hann'`, `('kaiser', 8.6)`, `('tukey', 0.5)`). |
| `n` | Window length, in samples (at least 16). |

**Returns:** A [`WindowMetricsResult`](/phonometry/reference/api/signals/windows/#windowmetricsresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## WindowMetricsResult

```python
WindowMetricsResult(
    window: str | tuple[Any, ...],
    n: int,
    taps: NDArray[np.float64],
    coherent_gain: float,
    enbw_bins: float,
    scalloping_loss_db: float,
    worst_case_processing_loss_db: float,
    highest_sidelobe_db: float,
    mainlobe_width_3db_bins: float,
)
```

Figures of merit of a taper (Harris 1978), DFT-even sampling.

Losses are positive dB (how much is lost), sidelobe levels negative dB
(relative to the main lobe), bandwidths in DFT bins (multiply by
`fs/n` for Hz), matching the conventions of Harris' Table 1. The
window is sampled DFT-even (periodic), exactly as
`scipy.signal.welch` and the estimators of
[`phonometry.signals.spectra`](/phonometry/reference/api/signals/spectra/) use it.

**Attributes**

| Name | Description |
| :--- | :--- |
| `window` | The window specification as given (any name or `(name, param)` tuple `scipy.signal.get_window` accepts). |
| `n` | Window length, in samples. |
| `taps` | The window samples `w[m]` (DFT-even). |
| `coherent_gain` | Normalized DC gain $\sum w / n$ (1 for rectangular); the amplitude a bin-centered tone is scaled by before correction. |
| `enbw_bins` | Equivalent noise bandwidth $n \sum w^2 / \left( \sum w \right)^2$, in bins: the width of the ideal rectangular filter that would pass the same white-noise power (1 rectangular, 1.5 Hann, 1987/1458 Hamming). |
| `scalloping_loss_db` | Attenuation of a tone midway between two bins, $-20 \log_{10} \lvert W(1/2)/W(0) \rvert$, in dB (positive). |
| `worst_case_processing_loss_db` | Scalloping loss plus the ENBW processing loss $10 \log_{10}(\mathrm{ENBW})$, in dB: the worst-case reduction in output signal-to-noise ratio for a tone in white noise. |
| `highest_sidelobe_db` | Level of the highest sidelobe relative to the main lobe, in dB (negative; -13.3 rectangular, -31.5 Hann). |
| `mainlobe_width_3db_bins` | Two-sided -3 dB width of the main lobe, in bins. |

### WindowMetricsResult.enbw_hz()

```python
WindowMetricsResult.enbw_hz(fs: float) -> float
```

Equivalent noise bandwidth in Hz for a sample rate `fs`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate, in Hz. |

**Returns:** `enbw_bins·fs/n`, in Hz.

### WindowMetricsResult.plot()

```python
WindowMetricsResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the window shape and its spectrum with the metrics marked.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
