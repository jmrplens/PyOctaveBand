---
title: "signals.inversion"
description: "Regularized spectral inversion with frequency-dependent regularization."
sidebar:
  label: "inversion"
---

Regularized spectral inversion with frequency-dependent regularization.

Inverting a measured transfer function -- to equalize a measurement
loudspeaker, flatten a microphone response or post-process an acquired
impulse response -- cannot be a plain reciprocal $1/H(f)$: wherever
the
system radiates little energy (outside its passband, in deep notches) the
reciprocal explodes and the inverse filter amplifies nothing but noise.
Mueller & Massarani ("Transfer-Function Measurement with Sweeps", JAES
49(6), 2001, Secs. 3.1 and 4.5) therefore confine the inversion to the
transmission band: unity equalization in-band and a controlled, bounded
gain outside.

This module implements that behaviour with the frequency-dependent
Tikhonov regularization of Kirkeby & Nelson ("Digital Filter Design for
Inversion Problems in Sound Reproduction", JAES 47(7/8), 1999, Eq. (17)):

$$
H_{\mathrm{inv}}(f) = \frac{\overline{H(f)}}{\lvert H(f) \rvert^2 + \epsilon(f)}
$$

where the regularization profile $\epsilon(f)$ is small inside the
band $[f_1, f_2]$ (the filter equalizes to unity within an analytic
bound) and large outside (the out-of-band gain is capped at
$1/(2\sqrt{\epsilon})$, the maximum of
$x/(x^2 + \epsilon)$), with a smooth geometric cross-fade
over a transition zone bordering the band edges. A modeling delay makes
the mixed-phase inverse causal (Kirkeby & Nelson Sec. 2.4: the inverse of
a non-minimum-phase response is anticausal and must be delayed to be
realisable).

The closed forms above are the module's oracles: in-band the equalized
magnitude $\lvert H \, H_{\mathrm{inv}} \rvert$ deviates from unity
by exactly $\epsilon/(\lvert H \rvert^2 + \epsilon)$, and
out-of-band the filter gain never exceeds the
$1/(2\sqrt{\epsilon})$ bound.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## InverseFilterResult

```python
InverseFilterResult(
    inverse: np.ndarray,
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    response_spectrum: np.ndarray,
    regularization: np.ndarray,
    f_range: tuple[float, float],
    delay: int,
    fs: float,
    flatness_db: float,
    max_gain_db: float,
)
```

A regularized inverse filter with its achieved equalization.

Returned by [`regularized_inverse_filter`](/phonometry/reference/api/signals/inversion/#regularized_inverse_filter). The causal filter
samples live in `inverse` (the equalized response arrives `delay`
samples late; `apply` compensates it). `spectrum` is the
complex inverse spectrum *including* the modeling delay;
`regularization` is the $\epsilon(f)$ profile actually used,
and `flatness_db` reports how flat the equalized magnitude
$\lvert H(f) \, H_{\mathrm{inv}}(f) \rvert$ actually is across
the requested band.

**Attributes**

| Name | Description |
| :--- | :--- |
| `inverse` | Inverse-filter samples (time domain, length `n_fft`). |
| `frequencies` | Frequency grid of the design, in Hz. |
| `spectrum` | Complex inverse spectrum on `frequencies`, including the $\exp(-j 2 \pi f \, \text{delay} / f_\mathrm{s})$ modeling delay. |
| `response_spectrum` | Complex spectrum of the measured response the filter was designed from, on the same grid. |
| `regularization` | The frequency-dependent profile $\epsilon(f)$ (absolute units of $\lvert H \rvert^2$). |
| `f_range` | `(f1, f2)` band equalized to unity, in Hz. |
| `delay` | Modeling delay of the filter, in samples. |
| `fs` | Sample rate, in Hz. |
| `flatness_db` | Largest deviation of the equalized magnitude $20 \log_{10} \lvert H \, H_{\mathrm{inv}} \rvert$ from 0 dB inside $[f_1, f_2]$. |
| `max_gain_db` | Largest filter gain *outside* the transition-padded band, peak-normalized as $20 \log_{10}(\max \lvert H_{\mathrm{inv}} \rvert \cdot \text{peak}_h)$ where `peak_h` is the peak of $\lvert H \rvert$ -- the achieved out-of-band boost the regularization allowed where the measurement carries no signal. |

### InverseFilterResult.apply()

```python
InverseFilterResult.apply(x: list[float] | np.ndarray) -> np.ndarray
```

Equalize a signal with the inverse filter.

Convolves `x` with the filter and removes the modeling delay, so
the output is time-aligned with the input and has the same length.
Feeding the response the filter was designed from returns (in-band)
a band-limited unit impulse at sample 0.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | The signal to equalize (1-D). |

**Returns:** The equalized, delay-compensated signal, `len(x)` samples.

### InverseFilterResult.plot()

```python
InverseFilterResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the measured, inverse and equalized magnitudes.

One panel: $\lvert H \rvert$,
$\lvert H_{\mathrm{inv}} \rvert$ and the equalized product
$\lvert H \, H_{\mathrm{inv}} \rvert$ in dB over
log-frequency, with the equalized band
shaded. Requires matplotlib (`pip install phonometry[plot]`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

### InverseFilterResult.size

*property*

Number of samples in the inverse filter.

## regularized_inverse_filter

```python
regularized_inverse_filter(
    response: Signal | list[float] | np.ndarray | ImpulseResponseResult,
    fs: float | None = None,
    *,
    f_range: tuple[float, float],
    regularization_inside: float = 1e-06,
    regularization_outside: float = 1.0,
    transition_octaves: float = 0.3333333333333333,
    n_fft: int | None = None,
    delay: int | None = None,
) -> InverseFilterResult
```

Design a regularized inverse filter for a measured impulse response.

Computes the frequency-dependent Tikhonov inverse of Kirkeby & Nelson
(JAES 47(7/8), 1999, Eq. (17)),

$$
H_{\mathrm{inv}}(f) = \frac{\overline{H(f)}}{\lvert H(f) \rvert^2 + \epsilon(f)}
$$

with $\epsilon(f)$ small across $[f_1, f_2]$ and large
outside, so the filter equalizes the response to unity in-band while
the out-of-band gain stays bounded by
$1/(2\sqrt{\epsilon_{\mathrm{outside}}})$ -- the behaviour
Mueller & Massarani (JAES 49(6), 2001, Secs. 3.1/4.5) obtain by
band-passing the plain inverse. A modeling delay of `delay` samples
(default half the FFT block) shifts the generally anticausal inverse of
a mixed-phase response into a causal filter (Kirkeby & Nelson,
Sec. 2.4).

Both regularization levels are *relative* to the peak of
$\lvert H \rvert^2$ (like the scalar `regularization` of
[`phonometry.room.impulse_response`](/phonometry/reference/api/rooms/impulse-response/), which this generalises): in-band
the equalized magnitude deviates from unity by at most
`regularization_inside * max|H|**2 / min|H|**2` -- the analytic residue
$\epsilon/(\lvert H \rvert^2 + \epsilon)$ -- and the achieved figure
is reported as [`InverseFilterResult.flatness_db`](/phonometry/reference/api/signals/inversion/#inversefilterresult).

Use the result's [`InverseFilterResult.apply`](/phonometry/reference/api/signals/inversion/#inversefilterresultapply) to equalize
recordings (or the excitation, for pre-emphasis) and read
[`InverseFilterResult.spectrum`](/phonometry/reference/api/signals/inversion/#inversefilterresult) to apply it spectrally.

**Parameters**

| Name | Description |
| :--- | :--- |
| `response` | Measured impulse response (1-D array), or an [`phonometry.room.ImpulseResponseResult`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresult) from the sweep/MLS/Golay front ends (its sample rate is used when `fs` is omitted). Also accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples, though an inverse filter undoes the response it is built from, so it comes out in 1/Pa, with the regularization floor in Pa². |
| `fs` | Sample rate in Hz. Optional when `response` carries one. |
| `f_range` | `(f1, f2)` band, in Hz, over which the response is equalized to unity. Choose it inside the band actually excited and radiated; inverting unexcited regions only amplifies noise. |
| `regularization_inside` | In-band regularization, as a fraction of the peak spectral power $\max \lvert H \rvert^2$. Default 1e-6. |
| `regularization_outside` | Out-of-band regularization, same units. Default 1.0, which caps the out-of-band gain at 6 dB below the peak-normalised unity ($1/(2\sqrt{1}) = 0.5$). |
| `transition_octaves` | Width of the geometric cross-fade between the two regularization levels, in octaves outside each band edge. Default 1/3. |
| `n_fft` | FFT block length of the design (also the filter length). Default: the next power of two of `2*len(response)`, so the circular design has room for the anticausal (delayed) part. |
| `delay` | Modeling delay in samples. Default `n_fft // 2`. |

**Returns:** An [`InverseFilterResult`](/phonometry/reference/api/signals/inversion/#inversefilterresult).
