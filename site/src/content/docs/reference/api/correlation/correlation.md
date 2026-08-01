---
title: "metrology.correlation"
description: "Correlation analysis and time-delay estimation."
sidebar:
  label: "correlation"
---

Correlation analysis and time-delay estimation.

Auto- and cross-correlation estimators with the three standard
normalizations, generalized cross-correlation (GCC) time-delay estimation
with the Knapp & Carter weightings, and sub-sample delay location for
impulse responses, following Bendat & Piersol, *Random Data: Analysis and
Measurement Procedures* (4th ed., 2010) and Knapp & Carter (1976):

* **correlation estimates** computed via FFT with zero padding so the
  circular product never wraps (B&P Section 11.4.2, Eq. 11.95), with the
  $1/N$ *biased*, $1/(N-r)$ *unbiased* (Eq. 11.96) and
  *coefficient*
  $\rho_{xy}(\tau) = C_{xy}(\tau)/(\sigma_x \sigma_y)$ (Eq. 5.16)
  normalizations, plus the
  large-`T` normalized random error of the estimate for bandwidth-limited
  Gaussian data,
  $\varepsilon[\hat{R}_{xy}(\tau)] = [1 + \rho_{xy}^{-2}(\tau)]^{1/2} / \sqrt{2BT}$
  (Eqs. 8.109/8.112), exposed as [`correlation_random_error`](/phonometry/reference/api/correlation/correlation/#correlation_random_error);
* **time-delay estimation**: the peak of the cross-correlation locates the
  delay of a common signal between two sensors (B&P Section 5.1.4,
  Eq. 5.21). [`time_delay`](/phonometry/reference/api/correlation/correlation/#time_delay) implements the direct correlator, the
  weighted-phase-slope estimator of the cross-spectrum (Eq. 5.101b) and the
  **generalized cross-correlation** of Knapp & Carter (1976): the averaged
  cross-spectrum is weighted by $\psi(f)$ before the inverse
  transform, with the Table I processors `'roth'`
  ($1/G_{xx}$), `'scot'`
  ($1/\sqrt{G_{xx} G_{yy}}$), `'phat'`
  ($1/\lvert G_{xy} \rvert$) and the maximum-likelihood
  `'ml'` (Hannan-Thomson) weighting
  $\lvert \gamma \rvert^2 / (\lvert G_{xy} \rvert (1 - \lvert \gamma \rvert^2))$
  that attains the Cramér-Rao bound;
* **sub-sample peak location** by three-point parabolic interpolation,
  optionally after band-limited local upsampling of the correlation around
  its peak, and the **peak-location uncertainty** of the delay estimate,
  $\sigma(\hat{\tau}_0) \approx (3/4)^{1/4} \sqrt{\varepsilon} / (\pi B)$ (Eq. 8.129), with the 95 % interval
  $\hat{\tau}_0 \pm 2\sigma$ (Eq. 8.130);
* **impulse-response utilities**: the sub-sample arrival time of a single
  IR (its peak is the cross-correlation with an ideal impulse) and the
  alignment of an IR pair by the estimated delay, applied as an exact
  fractional shift in the frequency domain.

The GCC estimators run on the same Welch core (segmentation, tapering,
overlap policy) as [`phonometry.metrology.spectra`](/phonometry/reference/api/spectra/spectra/), so a GCC and a
cross-spectral density computed with the same segment length are mutually
consistent bin by bin.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## align_impulse_responses

```python
align_impulse_responses(
    ir: NDArray[np.float64] | list[float],
    reference: NDArray[np.float64] | list[float],
    fs: float,
    *,
    interpolation: _Interpolation = 'parabolic',
    upsample: int = 8,
) -> AlignedImpulseResponseResult
```

Align an impulse response onto a reference by its estimated delay.

Estimates the sub-sample delay of `ir` relative to `reference`
([`impulse_response_delay`](/phonometry/reference/api/correlation/correlation/#impulse_response_delay)) and removes it with an exact
band-limited fractional shift (frequency-domain phase ramp over a
zero-padded record). Use it to average IR ensembles or to compare
measurements taken at slightly different distances.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Impulse response to align, 1-D. |
| `reference` | Reference impulse response, same length. |
| `fs` | Sample rate, in Hz. |
| `interpolation` | `'parabolic'` (default) or `'none'`. |
| `upsample` | Integer local-upsampling factor (default 8). |

**Returns:** An [`AlignedImpulseResponseResult`](/phonometry/reference/api/correlation/correlation/#alignedimpulseresponseresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## AlignedImpulseResponseResult

```python
AlignedImpulseResponseResult(
    aligned: NDArray[np.float64],
    reference: NDArray[np.float64],
    delay: float,
    delay_samples: float,
    fs: float,
)
```

An impulse response aligned onto a reference.

**Attributes**

| Name | Description |
| :--- | :--- |
| `aligned` | The input IR advanced by the estimated delay (exact fractional shift applied as a frequency-domain phase ramp over a zero-padded record, so nothing wraps around). |
| `reference` | The reference IR. |
| `delay` | Estimated delay removed from the IR, in seconds. |
| `delay_samples` | The same delay in (fractional) samples. |
| `fs` | Sample rate, in Hz. |

### AlignedImpulseResponseResult.plot()

```python
AlignedImpulseResponseResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the reference and the aligned impulse response.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## correlation

```python
correlation(
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float] | None = None,
    fs: float = 1.0,
    *,
    normalization: _Normalization = 'unbiased',
    max_lag: float | None = None,
) -> CorrelationResult
```

Auto- or cross-correlation estimate with a chosen normalization.

Computed via zero-padded FFT so the circular product never wraps (B&P
Section 11.4.2). `y=None` gives the autocorrelation of `x`. The
sign convention follows B&P Eq. 5.19-5.20: with
$y(t) = \alpha x(t - \tau_0) + n(t)$ the estimate peaks at
$\tau = +\tau_0$.

Normalizations:

* `'biased'` - the raw lag sums divided by `N`; tapers toward the
  record ends and stays bounded by
  $[\hat{R}_{xx}(0) \hat{R}_{yy}(0)]^{1/2}$;
* `'unbiased'` - divided by $N - \lvert r \rvert$
  (Eq. 11.96), an unbiased estimate of $R_{xy}(\tau)$ whose
  variance grows toward the ends;
* `'coefficient'` - the correlation coefficient function
  $\hat{\rho}_{xy}(\tau) = \hat{C}_{xy}(\tau)/(\sigma_x \sigma_y) \in [-1, 1]$ over the
  mean-removed records (Eq. 5.16).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | First signal, 1-D. |
| `y` | Second signal, same length, or `None` for autocorrelation. |
| `fs` | Sample rate, in Hz. |
| `normalization` | See above (default `'unbiased'`). |
| `max_lag` | Largest lag magnitude to keep, in seconds (default: the full `N-1` samples). |

**Returns:** A [`CorrelationResult`](/phonometry/reference/api/correlation/correlation/#correlationresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## correlation_random_error

```python
correlation_random_error(
    coefficient: float,
    signal_bandwidth: float,
    duration: float,
) -> float
```

Normalized random error of a correlation estimate.

B&P Eqs. 8.109/8.112:
$\varepsilon[\hat{R}_{xy}(\tau)] = [1 + \rho_{xy}^{-2}(\tau)]^{1/2} / \sqrt{2BT}$ for
bandwidth-limited
Gaussian data of bandwidth `B` observed for `T` seconds, with
$\rho_{xy}(\tau)$ the correlation coefficient at the lag of
interest. At the zero lag of an autocorrelation ($\rho = 1$)
this is $1/\sqrt{BT}$
(Eq. 8.111). Valid for $T \ge 10 \lvert \tau \rvert$ and
$BT \ge 5$ (Section 8.4.1).

For the two-detector time-delay problem
$x = s + m$, $y = s' + n$
(Section 8.4.2) the peak coefficient is
$\rho = S/\sqrt{(S+M)(S+N)}$, which reproduces the book's
Example 8.5:
$B = 100$ Hz, $T = 5$ s, $M/S = N/S = 10$ give
$\varepsilon \approx 0.35$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `coefficient` | Correlation coefficient $\rho_{xy}(\tau)$ at the lag. |
| `signal_bandwidth` | Signal bandwidth `B`, in Hz. |
| `duration` | Record length `T`, in seconds. |

**Returns:** Normalized random error (dimensionless; `inf` at $\rho = 0$).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the bandwidth or duration is not positive, or the coefficient is outside [-1, 1]. |

## CorrelationResult

```python
CorrelationResult(
    lags: NDArray[np.float64],
    values: NDArray[np.float64],
    coefficient: NDArray[np.float64],
    normalization: str,
    kind: str,
    fs: float,
    n_samples: int,
    duration: float,
)
```

Auto- or cross-correlation estimate (B&P Sections 5.1, 8.4, 11.4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `lags` | Lag axis $\tau$, in seconds, symmetric about zero. Positive lag means the second signal is delayed relative to the first ($\hat{R}_{xy}(\tau) \sim E[x(t) y(t+\tau)]$, B&P Eq. 5.19-5.20 convention). |
| `values` | Correlation estimate on `lags` with the requested `normalization`. |
| `coefficient` | Correlation coefficient function $\hat{\rho}_{xy}(\tau) \in [-1, 1]$ on the same lags (Eq. 5.16; equals `values` when `normalization='coefficient'`). |
| `normalization` | `'biased'` ($1/N$), `'unbiased'` ($1/(N - \lvert r \rvert)$, Eq. 11.96) or `'coefficient'`. |
| `kind` | `'autocorrelation'` or `'cross-correlation'`. |
| `fs` | Sample rate, in Hz. |
| `n_samples` | Record length `N`, in samples. |
| `duration` | Record length $T = N/f_s$, in seconds. |

### CorrelationResult.plot()

```python
CorrelationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the correlation estimate against the lag in seconds.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

### CorrelationResult.random_error()

```python
CorrelationResult.random_error(
    signal_bandwidth: float,
) -> NDArray[np.float64]
```

Normalized random error of the estimate at each lag.

For bandwidth-limited Gaussian data of bandwidth `B` and record
length `T` (B&P Eqs. 8.109 and 8.112, identical in form for the
cross- and autocorrelation, with the measured coefficient in place
of the true value):

$$
\varepsilon[\hat{R}_{xy}(\tau)] = \frac{[1 + \rho_{xy}^{-2}(\tau)]^{1/2}}{\sqrt{2BT}}
$$

so $\varepsilon[\hat{R}_{xx}(0)] = 1/\sqrt{BT}$
(Eq. 8.111). The large-`T` approximation behind it assumes
$T \ge 10 \lvert \tau \rvert$ and $BT \ge 5$
(Section 8.4.1). Lags where the measured coefficient is zero
return `inf`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal_bandwidth` | Signal bandwidth `B`, in Hz. |

**Returns:** Normalized random error per lag (dimensionless).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the bandwidth is not positive. |

## impulse_response_delay

```python
impulse_response_delay(
    ir: NDArray[np.float64] | list[float],
    fs: float,
    *,
    reference: NDArray[np.float64] | list[float] | None = None,
    interpolation: _Interpolation = 'parabolic',
    upsample: int = 8,
) -> float
```

Sub-sample delay of an impulse response, in seconds.

Without a reference, the arrival time of the IR itself: the
cross-correlation of an IR with an ideal unit impulse *is* the IR, so
its peak magnitude location - refined to sub-sample resolution by
band-limited local upsampling plus parabolic interpolation - is the
delay relative to $t = 0$. With a `reference` IR, the delay of
`ir` relative to it, from the peak of their full-record
cross-correlation with the same refinement (one-shot transients are
not stationary records, so the direct correlator is used rather than
the Welch-averaged GCC).

Sub-sample accuracy presumes the IR is band-limited below Nyquist;
the synthetic fractional-delay tests pin the achievable accuracy
(about 1e-3 samples for a $0.4 f_s$ band-limited pulse at the
default `upsample=8`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Impulse response, 1-D. |
| `fs` | Sample rate, in Hz. |
| `reference` | Optional reference IR (same length) the delay is measured against. |
| `interpolation` | `'parabolic'` (default) or `'none'`. |
| `upsample` | Integer local-upsampling factor (default 8). |

**Returns:** Delay in seconds (relative to $t = 0$ or to `reference`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## time_delay

```python
time_delay(
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    fs: float,
    *,
    method: _Method = 'gcc',
    weighting: _Weighting = 'phat',
    window: str = 'hann',
    nperseg: int | None = None,
    overlap: float = 0.5,
    max_delay: float | None = None,
    interpolation: _Interpolation = 'parabolic',
    upsample: int = 1,
    signal_bandwidth: float | None = None,
) -> TimeDelayResult
```

Time delay of `y` relative to `x` (TDE).

Three estimators of the delay $\tau_0$ in the two-sensor model
$y(t) = \alpha x(t - \tau_0) + n(t)$ (B&P Section 5.1.4):

* `'direct'` - the peak of the full-record correlation coefficient
  function (Eq. 5.21);
* `'gcc'` - the peak of the generalized cross-correlation of
  Knapp & Carter (1976): the Welch-averaged cross-spectrum (shared
  core with [`cross_spectral_density`](/phonometry/reference/api/spectra/spectra/#cross_spectral_density))
  is weighted by $\psi(f)$ before the inverse transform.
  Weightings (Table I): `'none'` (plain correlator), `'roth'`
  ($1/G_{xx}$,
  suppresses bands where the first sensor is noisy), `'scot'`
  ($1/\sqrt{G_{xx} G_{yy}}$, prewhitens both channels),
  `'phat'`
  ($1/\lvert G_{xy} \rvert$: for uncorrelated noises the ideal
  GCC is a delta at
  the delay, but errors are accentuated wherever signal power is
  small), and `'ml'` (Hannan-Thomson,
  $\lvert \gamma \rvert^2 / (\lvert G_{xy} \rvert (1 - \lvert \gamma \rvert^2))$):
  the maximum-likelihood processor that
  weights the phase by its coherence-derived reliability and attains
  the Cramér-Rao bound - provided the coherence estimate is averaged
  over at least two segments, the discrete form of Knapp & Carter's
  $\lvert \gamma \rvert^2 \ne 1$ existence condition (enforced
  here). The delay must
  fit within half a segment; raise `nperseg` for long delays.
* `'phase'` - the $\lvert \hat{G}_{xy} \rvert$-weighted
  least-squares slope of the
  cross-spectrum phase (Eq. 5.101b); accurate for clean, moderate
  delays where the unwrapped phase is unambiguous, and independent of
  any peak interpolation (`interpolation`/`upsample` do not
  apply).

The correlation-peak methods refine the sample peak to sub-sample
resolution by three-point parabolic interpolation, optionally after
band-limited local upsampling (`upsample > 1`); this presumes the
signals are band-limited below Nyquist so the peak is oversampled.
With `signal_bandwidth` given, the result carries the B&P
peak-location uncertainty (Eq. 8.129) and its $\pm 2\sigma$
interval (Eq. 8.130).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Reference record, 1-D. |
| `y` | Delayed record, 1-D, same length. |
| `fs` | Sample rate, in Hz. |
| `method` | `'gcc'` (default), `'direct'` or `'phase'`. |
| `weighting` | GCC weighting (default `'phat'`; ignored otherwise). |
| `window` | Welch taper for `'gcc'`/`'phase'` (default Hann). |
| `nperseg` | Welch segment length for `'gcc'`/`'phase'` (`None` picks the shared default). |
| `overlap` | Welch overlap fraction for `'gcc'`/`'phase'`. |
| `max_delay` | Largest delay magnitude searched, in seconds. |
| `interpolation` | `'parabolic'` (default) or `'none'`. |
| `upsample` | Integer local-upsampling factor (default 1: off). |
| `signal_bandwidth` | Signal bandwidth `B` in Hz for the Eq. 8.129 delay uncertainty (`None`: no error reported). |

**Returns:** A [`TimeDelayResult`](/phonometry/reference/api/correlation/correlation/#timedelayresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## TimeDelayResult

```python
TimeDelayResult(
    delay: float,
    delay_samples: float,
    method: str,
    weighting: str | None,
    lags: NDArray[np.float64],
    correlation: NDArray[np.float64],
    peak_correlation: float,
    delay_std: float | None,
    delay_interval: tuple[float, float] | None,
    signal_bandwidth: float | None,
    fs: float,
)
```

Time-delay estimate between two records.

**Attributes**

| Name | Description |
| :--- | :--- |
| `delay` | Estimated delay $\hat{\tau}_0$ of the second record relative to the first, in seconds (positive: `y` lags `x`). |
| `delay_samples` | The same delay in (fractional) samples. |
| `method` | `'direct'`, `'gcc'` or `'phase'`. |
| `weighting` | GCC weighting name (`None` unless `method='gcc'`). |
| `lags` | Lag axis of [`correlation`](/phonometry/reference/api/correlation/correlation/#correlation), in seconds. |
| `correlation` | The correlation function whose peak was located: the correlation coefficient $\hat{\rho}_{xy}(\tau)$ for `'direct'`, the weighted GCC $\hat{R}_\psi(\tau)$ (normalized to unit peak magnitude) for `'gcc'`, and the unweighted equivalent for `'phase'` (whose estimate comes from Eq. 5.101b, not from this curve). |
| `peak_correlation` | Plain correlation coefficient $\hat{\rho}_{xy}$ at the estimated delay (rounded to the nearest sample) - the quantity entering the B&P error formulas, whatever the method. |
| `delay_std` | Standard deviation of the peak-location estimate, $\sigma(\hat{\tau}_0) \approx (3/4)^{1/4} \sqrt{\varepsilon} / (\pi B)$ (Eq. 8.129), in seconds; `None` unless `signal_bandwidth` was given. |
| `delay_interval` | Approximate 95 % confidence interval $\hat{\tau}_0 \pm 2\sigma$ (Eq. 8.130), in seconds; `None` without a bandwidth. |
| `signal_bandwidth` | The bandwidth `B` used for the error, Hz. |
| `fs` | Sample rate, in Hz. |

### TimeDelayResult.plot()

```python
TimeDelayResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the correlation function with the estimated delay marked.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
