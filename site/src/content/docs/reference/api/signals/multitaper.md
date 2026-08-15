---
title: "signals.multitaper"
description: "Thomson multitaper spectral estimation (Percival & Walden 1993, Ch. 7)."
sidebar:
  label: "multitaper"
---

Thomson multitaper spectral estimation (Percival & Walden 1993, Ch. 7).

Thomson's multitaper estimator (Thomson 1982), as developed in Percival &
Walden, *Spectral Analysis for Physical Applications* (1993, Chapter 7),
is the whole-record alternative to the Welch segment averaging of
[`phonometry.signals.spectra`](/phonometry/reference/api/signals/spectra/): `K` orthogonal discrete prolate
spheroidal (Slepian) tapers of time-half-bandwidth `NW` produce `K`
nearly uncorrelated eigenspectra whose (adaptively weighted) average
carries about $2K$ chi-square degrees of freedom *without*
splitting the record - the estimator of choice for short records where
Welch would leave too few segments.

The statistical apparatus is the same as that of the Welch estimators,
one step further: the chi-square confidence interval keeps the form of
Bendat & Piersol Eq. 8.163, but its degrees of freedom are
per-frequency, because Thomson's adaptive weights (P&W Eq. 368a)
downweight leakage-prone tapers wherever the spectrum is locally weak
and that costs degrees of freedom there (P&W Eq. 370b). Calibration -
no detrending, one-sided `'density'` scaling integrating to the signal
power, `'spectrum'` scaling reading $A^2/2$ at a tone's peak -
matches [`phonometry.signals.spectra`](/phonometry/reference/api/signals/spectra/) exactly, so the two estimators
of the same record are directly comparable.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## multitaper_psd

```python
multitaper_psd(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    time_half_bandwidth: float = 4.0,
    n_tapers: int | None = None,
    adaptive: bool = True,
    scaling: Literal['density', 'spectrum'] = 'density',
    confidence: float = 0.95,
) -> MultitaperSpectralDensityResult
```

Thomson multitaper spectral density with chi-square interval.

Implements the multitaper estimator of Thomson (1982) as developed in
Percival & Walden (1993, Chapter 7): the record is multiplied by
`K` orthogonal discrete prolate spheroidal (Slepian) data tapers -
the sequences that maximize spectral concentration in the design band
$[-W, W]$, computed by `scipy.signal.windows.dpss` - and
the
`K` resulting eigenspectra (P&W Eq. 333) are averaged. Because the
tapers are orthogonal the eigenspectra are nearly uncorrelated, so the
average has about $2K$ chi-square degrees of freedom and
$1/K$ of
the periodogram's variance *without* segmenting the record: the
estimator of choice for short records, where Welch's method
([`power_spectral_density`](/phonometry/reference/api/signals/spectra/#power_spectral_density)) would
leave too few segments.

With `adaptive=True` (default) the eigenspectra are combined with
Thomson's frequency-dependent weights (P&W Eqs. 368a/370a, iterated to
convergence): wherever the local spectrum is weak relative to the
broad-band leakage each taper could carry, the leakier high-order
tapers are downweighted, trading degrees of freedom (Eq. 370b) for
leakage protection in high-dynamic-range spectra. The broadband
$\sigma^2$ driving the weights is
$\operatorname{mean}(x^2)$ with no mean removal,
consistent with the no-detrending calibration below. For a locally
white spectrum the weights converge to uniform and nothing is lost.
With `adaptive=False` the eigenvalue-weighted average of P&W
Eq. 369a is returned.

Calibration matches the Welch estimators of
[`phonometry.signals.spectra`](/phonometry/reference/api/signals/spectra/) exactly: no
detrending, `'density'` scaling integrates to the signal power
(units²/Hz, one-sided) and `'spectrum'` scaling reads
$A^2/2$ at
the peak of a sinusoid of amplitude `A` (the tone calibration is
exact for the taper set in use, computed from the taper DC gains
$\left( \sum_t h_{tk} \right)^2$; a tone's power in
`'density'` scaling is spread over
the resolution bandwidth $2W$).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D (used whole; no segmentation). |
| `fs` | Sample rate, in Hz. |
| `time_half_bandwidth` | Duration x half-bandwidth product `NW` (dimensionless; default 4, P&W's worked choice). The design half-bandwidth is $W = NW f_\mathrm{s} / N$ Hz; larger `NW` admits more tapers (lower variance) at the cost of resolution $2W$. |
| `n_tapers` | Number of tapers `K`; `None` picks $2 NW - 1$ (all tapers with near-unity concentration, P&W Section 7.1). At most the Shannon number $2 NW$. |
| `adaptive` | Use Thomson's adaptive weights (default) or the eigenvalue-weighted average. |
| `scaling` | `'density'` (units²/Hz) or `'spectrum'` (units², sinusoid-peak reading). |
| `confidence` | Confidence level for the chi-square interval. |

**Returns:** A [`MultitaperSpectralDensityResult`](/phonometry/reference/api/signals/multitaper/#multitaperspectraldensityresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## MultitaperSpectralDensityResult

```python
MultitaperSpectralDensityResult(
    frequencies: NDArray[np.float64],
    psd: NDArray[np.float64],
    ci_lower: NDArray[np.float64],
    ci_upper: NDArray[np.float64],
    confidence: float,
    degrees_of_freedom: NDArray[np.float64],
    random_error: NDArray[np.float64],
    weights: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    time_half_bandwidth: float,
    n_tapers: int,
    resolution_bandwidth: float,
    adaptive: bool,
    scaling: str,
)
```

Thomson multitaper spectral density (Percival & Walden Ch. 7).

One whole-record estimate from `K` orthogonal Slepian (dpss) tapers:
the `K` eigenspectra are nearly uncorrelated, so their weighted
average trades the two chi-square degrees of freedom of a periodogram
for about $2K$ - without segmenting the record as Welch's
method does. The chi-square machinery mirrors
[`SpectralDensityResult`](/phonometry/reference/api/signals/spectra/#spectraldensityresult), but here
the degrees of freedom are
per-frequency: Thomson's adaptive weights (P&W Eq. 368a) downweight
leakage-prone tapers wherever the spectrum is locally weak, which
costs degrees of freedom there (P&W Eq. 370b).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-sided frequency axis, in Hz. |
| `psd` | Multitaper spectral density $\hat{S}^{(mt)}(f)$ (units²/Hz for `'density'` scaling, units² for `'spectrum'`). |
| `ci_lower` | Lower chi-square confidence bound, $\nu \hat{S}/\chi^2_{\nu;\alpha/2}$ with the per-frequency $\nu$ (the same interval form as B&P Eq. 8.163, with $\nu$ from P&W Eq. 370b). |
| `ci_upper` | Upper chi-square confidence bound. |
| `confidence` | Confidence level of the interval (e.g. `0.95`). |
| `degrees_of_freedom` | Per-frequency equivalent chi-square degrees of freedom $\nu(f) = 2 \left( \sum_k d_k \right)^2 / \sum_k d_k^2$ with $d_k = b_k^2(f) \lambda_k$ (P&W Eq. 370b); $2K \left( \sum \lambda_k / K \right)^2 K / \sum \lambda_k^2 \approx 2K$ for unity weights. The DC bin - and the Nyquist bin for an even record length - carries half (a single real Fourier component per eigenspectrum). |
| `random_error` | Per-frequency normalized random error $\varepsilon[\hat{S}^{(mt)}] = \sqrt{2/\nu}$ ($\approx 1/\sqrt{K}$), the multitaper counterpart of B&P Eq. 8.158. |
| `weights` | Normalized combination weights $d_k(f) / \sum_j d_j(f)$, shape `(n_tapers, n_frequencies)`. Adaptive weighting makes them frequency dependent; they converge to $\approx 1/K$ where the spectrum is locally white (exactly uniform weights would be $\lambda_k / \sum \lambda_j$). |
| `eigenvalues` | Concentration ratios $\lambda_k(N, W)$ of the tapers - the fraction of each taper's spectral-window energy inside the design band $[-W, W]$ (P&W Section 7.1; near unity for $k < 2NW$). |
| `time_half_bandwidth` | The duration x half-bandwidth product `NW` (dimensionless; $W = NW/(N \Delta t)$). |
| `n_tapers` | Number of tapers `K` averaged. |
| `resolution_bandwidth` | The resolution bandwidth $2W$ of the estimator, in Hz - the multitaper analog of the Welch $B_\mathrm{e}$ (P&W call $2W$ *the* natural resolution measure of the method). |
| `adaptive` | Whether Thomson's adaptive weights were used. |
| `scaling` | `'density'` or `'spectrum'`. |

### MultitaperSpectralDensityResult.plot()

```python
MultitaperSpectralDensityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the multitaper density in dB with its confidence band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
