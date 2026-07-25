---
title: "metrology.synchronous_average"
description: "Public API of phonometry.metrology.synchronous_average (auto-generated)."
sidebar:
  label: "synchronous_average"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Time synchronous averaging (TSA) of a periodic waveform in noise.

Time domain averaging extracts a repetitive signal of known period `T`
from additive noise by ensemble-averaging successive length-`T` blocks,
following P. D. McFadden, "A revised model for the extraction of periodic
waveforms by time domain averaging", *Mechanical Systems and Signal
Processing* 1(1) 1987, 83-95. Given a signal `y(t) = x(t) + e(t)` with
`x` periodic in `T` and `e` asynchronous, the average

    `a(t) = (1/N) Σ_{n=0}^{N-1} y(t + n·T)`   (McFadden Eq. 5)

reinforces every component synchronous with `T` and suppresses the rest.

**Two models, one implementation.** McFadden distinguishes the *existing*
comb-filter model from the *revised* model. In the frequency domain the
average is the multiplication of `Y(f)` by the comb filter (Eq. 8)

    `C(f) = (1/N)·sin(N·π·f·T) / sin(π·f·T)`,

whose magnitude `|C(f)| = |sin(N·π·f·T) / (N·sin(π·f·T))|` is a Dirichlet
kernel: unity at every harmonic `k/T` (the teeth, Eq. 9, of unit height
regardless of `N`) and zero at the nodes `j/(N·T)` with `j` not a
multiple of `N`. That model assumes knowledge of `y` over infinite time
and produces a result that is not exactly periodic. McFadden's *revised*
model applies a rectangular window of width `T` in the time domain and
samples the transform in the frequency domain, so it needs only a finite
block of the signal and yields a result that is exactly periodic and can be
stored as a single period. The digital block average computed here, `N`
consecutive periods of an integer number of samples reduced to one period,
*is* that revised model: the returned `period_waveform`, repeated,
is exactly periodic.

**Noise reduction.** Asynchronous noise of variance `σ²` averaged over
`N` periods has residual variance `σ²/N`: the residual standard
deviation falls as `1/√N` and the amplitude signal-to-noise ratio
improves by `√N` (a power reduction of `10·log₁₀ N` dB, reported as
`noise_reduction_db`).

**Choosing N (McFadden's revised-model correction).** Because a discrete
interfering tone at a *non-harmonic* order `q = f·T` is only attenuated,
not removed, its rejection is optimised by choosing `N` so that a comb
node lands exactly on it, i.e. the smallest `N` with `N·q` an integer.
McFadden's own example, a tone at 32.05 orders, is suppressed by more than
100 dB with `N = 20` (since `20·32.05 = 641`) yet -- evaluating Eq. 8
at that order -- by only about 14 dB with the common power-of-two choice
`N = 32` (`32·32.05 = 1025.6`; the paper makes the comparison but does
not print this figure). Thus the habit of taking a power-of-two number of
averages is not, in general, optimal.

**Non-integer samples per period.** When `fs·T` is not an integer the
period boundaries fall between samples. Each block is then aligned to a
common integer grid by the band-limited fractional delay of
[`phonometry.metrology.signals.fractional_delay`](/phonometry/reference/api/spectra/signals/#fractional_delay) before averaging, so
the periodic waveform is recovered within the interpolation error of that
band-limited shift. An integer `fs·T` needs no interpolation and the
waveform is recovered to machine precision. The averaged samples stay on
the `1/fs` sampling grid throughout: output sample `m` is the average
of the input at the times `n·T + m/fs`, so the returned time axis is
`m/fs` and the `M = round(fs·T)` samples cover one period exactly when
`fs·T` is an integer and to within half a sample otherwise. No
resampling onto an `M`-point angular grid is performed.

## comb_filter_response

```python
comb_filter_response(
    frequencies: NDArray[np.float64] | list[float],
    period: float,
    n_averages: int,
) -> NDArray[np.float64]
```

Magnitude of the N-period synchronous-averaging comb filter.

The closed form of McFadden Eq. 8, `|C(f)| = |sin(N·π·f·T) /
(N·sin(π·f·T))|`, a Dirichlet kernel with unit-height teeth at the
harmonics `k/T` (Eq. 9) and nodes at `j/(N·T)` for `j` not a
multiple of `N`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies at which to evaluate, in Hz. |
| `period` | Repetition period `T`, in seconds. |
| `n_averages` | Number of averaged periods `N` (at least 1). |

**Returns:** The filter magnitude at each frequency (unitless; bounded by 1, though floating-point cancellation immediately beside a tooth can return values above 1 by a few parts in 10⁹).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the parameters are invalid. |

## SynchronousAverageResult

```python
SynchronousAverageResult(
    period_waveform: NDArray[np.float64],
    times: NDArray[np.float64],
    residual: NDArray[np.float64],
    n_averages: int,
    samples_per_period: int,
    period: float,
    fs: float,
    interpolated: bool,
    noise_reduction_db: float,
    residual_rms: float,
    comb_frequencies: NDArray[np.float64],
    comb_response: NDArray[np.float64],
)
```

Time synchronous average of a periodic waveform in noise.

**Attributes**

| Name | Description |
| :--- | :--- |
| `period_waveform` | The averaged periodic waveform, one period of `samples_per_period` samples. |
| `times` | Time axis of `period_waveform`, in seconds: the sampling grid `m/fs`, `m = 0 .. M-1` (the averaged samples stay on the `1/fs` grid; see the module note). The axis spans one period exactly when `fs·T` is an integer, and to within half a sample otherwise. |
| `residual` | Input minus the periodic reconstruction, over the analysed span (`n_averages·samples_per_period` samples, aligned to the integer period grid): what is left after the synchronous component is removed. |
| `n_averages` | Number of periods averaged, `N`. |
| `samples_per_period` | Integer samples per period `M` after any alignment. |
| `period` | Repetition period `T`, in seconds. |
| `fs` | Sample rate, in Hz. |
| `interpolated` | Whether band-limited fractional-delay alignment was applied (`True` when `fs·T` is not an integer). |
| `noise_reduction_db` | Power reduction of asynchronous noise, `10·log₁₀ N` dB (amplitude SNR gain `√N`). |
| `residual_rms` | Root-mean-square of `residual`. |
| `comb_frequencies` | Frequency axis of the comb-filter response, in Hz (from DC over a whole number of harmonics of `1/T`). |
| `comb_response` | Magnitude of the comb filter (McFadden Eq. 8) on `comb_frequencies`. |

### SynchronousAverageResult.amplitude_snr_gain

*property*

Amplitude signal-to-noise improvement `√N` from averaging.

### SynchronousAverageResult.plot()

```python
SynchronousAverageResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the averaged waveform and the comb-filter magnitude.

With `ax` given, only the averaged-waveform panel is drawn on it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## time_synchronous_average

```python
time_synchronous_average(
    x: NDArray[np.float64] | list[float],
    fs: float,
    period: float,
    *,
    n_averages: int | None = None,
    n_harmonics: int = 8,
) -> SynchronousAverageResult
```

Extract a periodic waveform of known period by time domain averaging.

Ensemble-averages `N` successive periods of the record (McFadden
Eq. 5) to reinforce the component synchronous with `period` and
suppress asynchronous noise, whose residual standard deviation falls as
`1/√N`. When `fs·period` is an integer the periods are sliced
directly and a noiseless periodic signal is recovered exactly; otherwise
each period is aligned to a common integer grid by the band-limited
fractional delay of [`fractional_delay`](/phonometry/reference/api/spectra/signals/#fractional_delay)
and recovered within that interpolation error.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D, containing the periodic component plus noise. |
| `fs` | Sample rate, in Hz. |
| `period` | Known repetition period `T`, in seconds (e.g. one revolution of a rotating machine). |
| `n_averages` | Number of whole periods to average (default: as many as the record holds). Choosing `N` so that `N·q` is an integer places a comb node on an interfering tone at order `q` and maximises its rejection (McFadden's revised-model result). |
| `n_harmonics` | Number of harmonics of `1/T` spanned by the returned comb-filter response (default 8). |

**Returns:** A [`SynchronousAverageResult`](/phonometry/reference/api/spectra/synchronous-average/#synchronousaverageresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |
