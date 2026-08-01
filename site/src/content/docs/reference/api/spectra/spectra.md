---
title: "metrology.spectra"
description: "Calibrated spectral-density estimation with statistical error analysis."
sidebar:
  label: "spectra"
---

Calibrated spectral-density estimation with statistical error analysis.

Welch-averaged auto- and cross-spectral density estimators that report,
alongside the spectrum itself, the statistical quality of the estimate,
following Bendat & Piersol, *Random Data: Analysis and Measurement
Procedures* (4th ed., 2010):

* the **number of averages**: the raw segment count and the effective number
  of independent averages $n_d$ once the correlation between
  overlapped,
  tapered segments is accounted for (Section 11.5.2.2 and its Ref. 11,
  Welch 1967);
* the **normalized random error** of the autospectrum estimate,
  $\varepsilon[\hat{G}_{xx}] = 1/\sqrt{n_d}$ (Eq. 8.158), and of
  the cross-spectrum magnitude and phase,
  $\varepsilon[\lvert \hat{G}_{xy} \rvert] = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_d})$ (Eq. 9.33) and
  $\mathrm{s.d.}[\hat{\theta}_{xy}] = (1 - \gamma^2_{xy})^{1/2} / (\lvert \gamma_{xy} \rvert \sqrt{2 n_d})$ (Eq. 9.52);
* **chi-square confidence intervals** for the autospectrum: the sampling
  distribution is $n \hat{G}_{xx}/G_{xx} \sim \chi^2_n$ with
  $n = 2 n_d$ degrees of freedom
  (Eq. 8.162), giving the interval
  $n \hat{G}_{xx}/\chi^2_{n;\alpha/2} \le G_{xx} \le n \hat{G}_{xx}/\chi^2_{n;1-\alpha/2}$ (Eq. 8.163);
* the **first-order resolution-bias error**:
  $b[\hat{G}_{xx}] \approx (B_e^2/24) \, G''_{xx}$
  (Eq. 8.139), which for a resonance peak of half-power bandwidth
  $B_r$
  becomes $\varepsilon_b \approx -(B_e/B_r)^2/3$ (Eq. 8.141) -
  exposed here as
  [`resolution_bias_error`](/phonometry/reference/api/spectra/spectra/#resolution_bias_error);
* the **coherent output spectrum**
  $G_{vv} = \gamma^2_{xy} G_{yy}$ and the noise output
  spectrum $G_{nn} = (1 - \gamma^2_{xy}) G_{yy}$ of the
  single-input/single-output model
  (Eqs. 9.55-9.56), with the spectral signal-to-noise ratio
  $\gamma^2/(1 - \gamma^2)$ and the random error
  $\varepsilon[\hat{G}_{vv}] = (2 - \gamma^2_{xy})^{1/2} / (\lvert \gamma_{xy} \rvert \sqrt{n_d})$ (Eq. 9.73).

The same Welch core (Hann taper and 50% overlap by default, `detrend`
off so absolute calibration is preserved) also backs the H1/H2 frequency
response and coherence estimators of
[`phonometry.electroacoustics.frequency_response`](/phonometry/reference/api/electroacoustics/frequency-response/) and the p-p intensity
probe of [`phonometry.emission.intensity`](/phonometry/reference/api/power/intensity/).

A **fractional-octave smoothing** utility completes the module: a
constant-power rectangular kernel of 1/n-octave width in log-frequency
(the constant-percentage resolution bandwidth that Bendat & Piersol,
Section 8.5.3, recommend for resonant-response spectra), applicable to
power spectra, magnitude responses and dB curves. A flat spectrum is left
exactly unchanged.

[`window_metrics`](/phonometry/reference/api/spectra/spectra/#window_metrics) characterizes any taper the `window` parameter
accepts with the figures of merit of Harris (1978, *On the use of windows
for harmonic analysis with the discrete Fourier transform*): equivalent
noise bandwidth, coherent gain, scalloping loss, worst-case processing
loss, highest sidelobe level and the -3 dB main-lobe width - the numbers
that turn "which window should I use?" into a trade-off one can read.

[`multitaper_psd`](/phonometry/reference/api/spectra/spectra/#multitaper_psd) adds Thomson's multitaper estimator (Thomson 1982;
Percival & Walden, *Spectral Analysis for Physical Applications*, 1993,
Chapter 7) as the whole-record alternative to Welch segment averaging:
`K` orthogonal discrete prolate spheroidal (Slepian) tapers of
time-half-bandwidth `NW` produce `K` nearly uncorrelated eigenspectra
whose (adaptively weighted) average carries about $2K$ chi-square
degrees of freedom without splitting the record - the estimator of choice
for short records where Welch would leave too few segments.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## coherent_output_spectrum

```python
coherent_output_spectrum(
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    fs: float,
    *,
    window: str = 'hann',
    nperseg: int | None = None,
    overlap: float = 0.5,
    scaling: Literal['density', 'spectrum'] = 'density',
) -> CoherentOutputSpectrumResult
```

Coherent output spectrum and spectral SNR (Bendat & Piersol 9.2.2).

Splits the measured output autospectrum $G_{yy}$ into the
coherent part
$G_{vv} = \gamma^2_{xy} G_{yy}$ linearly explained by the
input `x` and the noise
remainder $G_{nn} = (1 - \gamma^2_{xy}) G_{yy}$, and reports
the spectral
signal-to-noise ratio $\gamma^2/(1 - \gamma^2)$ together with
the Bendat & Piersol
random errors (Eqs. 9.73 and 9.82). For additive uncorrelated output
noise of known level the coherence satisfies
$\gamma^2 = \mathrm{SNR}/(1 + \mathrm{SNR})$, which is the
closed-form oracle used to verify
the implementation.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input (reference) signal, 1-D. |
| `y` | Output (response) signal, 1-D, same length as `x`. |
| `fs` | Sample rate, in Hz. |
| `window` | Segment taper (default Hann). |
| `nperseg` | Welch segment length; `None` picks a default. |
| `overlap` | Segment overlap fraction in [0, 1) (default 0.5). |
| `scaling` | `'density'` or `'spectrum'`. |

**Returns:** A [`CoherentOutputSpectrumResult`](/phonometry/reference/api/spectra/spectra/#coherentoutputspectrumresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## CoherentOutputSpectrumResult

```python
CoherentOutputSpectrumResult(
    frequencies: NDArray[np.float64],
    output_psd: NDArray[np.float64],
    coherent_psd: NDArray[np.float64],
    noise_psd: NDArray[np.float64],
    coherence: NDArray[np.float64],
    snr: NDArray[np.float64],
    snr_db: NDArray[np.float64],
    random_error: NDArray[np.float64],
    snr_random_error: NDArray[np.float64],
    coherence_bias: NDArray[np.float64],
    n_segments: int,
    n_averages: float,
    resolution_bandwidth: float,
    window: str,
    nperseg: int,
    overlap: float,
    scaling: str,
)
```

Coherent output spectrum of a single-input/single-output model.

The measured output autospectrum splits into the part linearly
explained by the input,
$G_{vv} = \gamma^2_{xy} G_{yy}$ (Eq. 9.55), and the
uncorrelated noise remainder
$G_{nn} = (1 - \gamma^2_{xy}) G_{yy}$ (Eq. 9.56), with
$G_{yy} = G_{vv} + G_{nn}$
(Eq. 9.57). Their ratio is the spectral signal-to-noise ratio.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-sided frequency axis, in Hz. |
| `output_psd` | Measured output autospectrum $\hat{G}_{yy}(f)$. |
| `coherent_psd` | Coherent output spectrum $\hat{G}_{vv} = \hat{\gamma}^2_{xy} \hat{G}_{yy}$. |
| `noise_psd` | Noise output spectrum $\hat{G}_{nn} = (1 - \hat{\gamma}^2_{xy}) \hat{G}_{yy}$. |
| `coherence` | Ordinary coherence $\hat{\gamma}^2_{xy}(f) \in [0, 1]$. |
| `snr` | Spectral signal-to-noise ratio $\hat{\gamma}^2/(1 - \hat{\gamma}^2)$ ($\infty$ at $\hat{\gamma}^2 = 1$). |
| `snr_db` | $10 \lg$ of `snr`, in dB. |
| `random_error` | Normalized random error of $\hat{G}_{vv}$, $\varepsilon = (2 - \gamma^2_{xy})^{1/2} / (\lvert \gamma_{xy} \rvert \sqrt{n_d})$ (Eq. 9.73), with the measured coherence in place of the true value. |
| `snr_random_error` | Normalized random error of the SNR, $\varepsilon = \sqrt{2} / (\lvert \gamma_{xy} \rvert \sqrt{n_d})$, first-order propagation of the coherence random error of Eq. 9.82 through $\gamma^2/(1 - \gamma^2)$. |
| `coherence_bias` | First-order bias of the coherence estimate, $b[\hat{\gamma}^2] \approx (1 - \gamma^2)^2 / n_d$ (Eq. 9.75). |
| `n_segments` | Raw number of segments averaged. |
| `n_averages` | Effective number of independent averages $n_d$. |
| `resolution_bandwidth` | Effective noise bandwidth $B_e$, in Hz. |
| `window` | Taper name. |
| `nperseg` | Segment length, in samples. |
| `overlap` | Segment overlap fraction. |
| `scaling` | `'density'` or `'spectrum'`. |

### CoherentOutputSpectrumResult.plot()

```python
CoherentOutputSpectrumResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the output/coherent/noise spectra and the spectral SNR.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## cross_spectral_density

```python
cross_spectral_density(
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    fs: float,
    *,
    window: str = 'hann',
    nperseg: int | None = None,
    overlap: float = 0.5,
    scaling: Literal['density', 'spectrum'] = 'density',
) -> CrossSpectralDensityResult
```

Calibrated cross-spectral density with statistical error analysis.

Welch's method on both channels; alongside
$\hat{G}_{xy}(f)$ the result
reports the ordinary coherence and the Bendat & Piersol random errors:
$\varepsilon[\lvert \hat{G}_{xy} \rvert] = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_d})$ (Eq. 9.33) for the
magnitude and
$\mathrm{s.d.}[\hat{\theta}_{xy}] = (1 - \gamma^2_{xy})^{1/2} / (\lvert \gamma_{xy} \rvert \sqrt{2 n_d})$ (Eq. 9.52) for the phase,
with the measured coherence in place of the unknown true value.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | First signal, 1-D. |
| `y` | Second signal, 1-D, same length as `x`. |
| `fs` | Sample rate, in Hz. |
| `window` | Segment taper (default Hann). |
| `nperseg` | Welch segment length; `None` picks a default. |
| `overlap` | Segment overlap fraction in [0, 1) (default 0.5). |
| `scaling` | `'density'` or `'spectrum'`. |

**Returns:** A [`CrossSpectralDensityResult`](/phonometry/reference/api/spectra/spectra/#crossspectraldensityresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## CrossSpectralDensityResult

```python
CrossSpectralDensityResult(
    frequencies: NDArray[np.float64],
    csd: NDArray[np.complex128],
    magnitude: NDArray[np.float64],
    phase: NDArray[np.float64],
    coherence: NDArray[np.float64],
    magnitude_random_error: NDArray[np.float64],
    phase_std: NDArray[np.float64],
    n_segments: int,
    n_averages: float,
    resolution_bandwidth: float,
    window: str,
    nperseg: int,
    overlap: float,
    scaling: str,
)
```

Welch cross-spectral density with its statistical error (B&P Ch. 9).

The error formulas replace the unknown true coherence with the computed
estimate, as Bendat & Piersol recommend for measured data (Section 9.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-sided frequency axis, in Hz. |
| `csd` | Complex cross-spectral density $\hat{G}_{xy}(f)$. |
| `magnitude` | $\lvert \hat{G}_{xy}(f) \rvert$. |
| `phase` | Cross-spectrum phase $\hat{\theta}_{xy}(f)$, in radians (unwrapped). |
| `coherence` | Ordinary coherence $\hat{\gamma}^2_{xy}(f) \in [0, 1]$. |
| `magnitude_random_error` | Normalized random error of $\lvert \hat{G}_{xy} \rvert$, $\varepsilon = 1/(\lvert \gamma_{xy} \rvert \sqrt{n_d})$ (Eq. 9.33). |
| `phase_std` | Standard deviation of the phase estimate, in radians, $\mathrm{s.d.} = (1 - \gamma^2_{xy})^{1/2} / (\lvert \gamma_{xy} \rvert \sqrt{2 n_d})$ (Eq. 9.52). |
| `n_segments` | Raw number of segments averaged. |
| `n_averages` | Effective number of independent averages $n_d$. |
| `resolution_bandwidth` | Effective noise bandwidth $B_e$, in Hz. |
| `window` | Taper name. |
| `nperseg` | Segment length, in samples. |
| `overlap` | Segment overlap fraction. |
| `scaling` | `'density'` or `'spectrum'`. |

### CrossSpectralDensityResult.plot()

```python
CrossSpectralDensityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the magnitude, phase (with ±σ band) and coherence.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## fractional_octave_smoothing

```python
fractional_octave_smoothing(
    frequencies: NDArray[np.float64] | list[float],
    values: NDArray[np.float64] | list[float],
    fraction: float = 3.0,
    *,
    domain: Literal['power', 'amplitude', 'db'] = 'power',
) -> NDArray[np.float64]
```

Smooth a spectrum with a constant-power 1/n-octave kernel.

Each output point is the power average of the input over a rectangular
window of 1/`fraction` octave centred (geometrically) on its
frequency: $[f \cdot 2^{-1/2n}, f \cdot 2^{+1/2n}]$. This is
the
constant-percentage resolution bandwidth that Bendat & Piersol
(Section 8.5.3) recommend for spectra of resonant systems, and the de
facto standard presentation of loudspeaker and room responses. The
average is computed on power regardless of `domain` (amplitudes are
squared first, dB levels converted), so smoothing conserves band power
rather than amplitude; a flat spectrum is left exactly unchanged.

The window is clipped at the ends of the frequency axis, and points at
non-positive frequencies (where a log-frequency window is undefined)
are copied unchanged.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency axis, 1-D, strictly increasing. |
| `values` | Spectrum sampled on `frequencies`: power-like values (`'power'`), magnitudes (`'amplitude'`) or levels in dB (`'db'`). |
| `fraction` | The `n` of the 1/n-octave width (default 3, one-third octave). |
| `domain` | How `values` map to power (see above). The output is returned in the same domain. |

**Returns:** Smoothed spectrum, same shape and domain as `values`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

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
([`power_spectral_density`](/phonometry/reference/api/spectra/spectra/#power_spectral_density)) would leave too few segments.

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

Calibration matches the Welch estimators of this module exactly: no
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
| `time_half_bandwidth` | Duration x half-bandwidth product `NW` (dimensionless; default 4, P&W's worked choice). The design half-bandwidth is $W = NW f_s / N$ Hz; larger `NW` admits more tapers (lower variance) at the cost of resolution $2W$. |
| `n_tapers` | Number of tapers `K`; `None` picks $2 NW - 1$ (all tapers with near-unity concentration, P&W Section 7.1). At most the Shannon number $2 NW$. |
| `adaptive` | Use Thomson's adaptive weights (default) or the eigenvalue-weighted average. |
| `scaling` | `'density'` (units²/Hz) or `'spectrum'` (units², sinusoid-peak reading). |
| `confidence` | Confidence level for the chi-square interval. |

**Returns:** A [`MultitaperSpectralDensityResult`](/phonometry/reference/api/spectra/spectra/#multitaperspectraldensityresult).

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
[`SpectralDensityResult`](/phonometry/reference/api/spectra/spectra/#spectraldensityresult), but here the degrees of freedom are
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
| `resolution_bandwidth` | The resolution bandwidth $2W$ of the estimator, in Hz - the multitaper analog of the Welch $B_e$ (P&W call $2W$ *the* natural resolution measure of the method). |
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

## power_spectral_density

```python
power_spectral_density(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    window: str = 'hann',
    nperseg: int | None = None,
    overlap: float = 0.5,
    scaling: Literal['density', 'spectrum'] = 'density',
    confidence: float = 0.95,
) -> SpectralDensityResult
```

Calibrated autospectral density with chi-square confidence interval.

Welch's method (Bendat & Piersol Section 11.5.2: tapered, overlapped
segment averaging, no detrending so absolute calibration is preserved).
Alongside $\hat{G}_{xx}(f)$ the result reports the effective
number of
independent averages $n_d$, the normalized random error
$\varepsilon = 1/\sqrt{n_d}$ (Eq. 8.158) and the chi-square
confidence interval with
$2 n_d$ degrees of freedom (Eq. 8.163). For the first-order
resolution-bias error at a resonance peak see
[`resolution_bias_error`](/phonometry/reference/api/spectra/spectra/#resolution_bias_error).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `window` | Segment taper (any scipy window name; default Hann, the B&P Section 11.5.2 recommendation for side-lobe suppression). |
| `nperseg` | Welch segment length; `None` picks a length giving a bin spacing of at most 4 Hz (the resolution bandwidth $B_e$ further depends on the taper; see [`SpectralDensityResult.resolution_bandwidth`](/phonometry/reference/api/spectra/spectra/#spectraldensityresult)). |
| `overlap` | Segment overlap fraction in [0, 1) (default 0.5, which with a Hann taper retrieves most of the stability lost to tapering, B&P Section 11.5.2.2). |
| `scaling` | `'density'` (units²/Hz) or `'spectrum'` (units² per segment bandwidth). |
| `confidence` | Confidence level for the chi-square interval. |

**Returns:** A [`SpectralDensityResult`](/phonometry/reference/api/spectra/spectra/#spectraldensityresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## resolution_bias_error

```python
resolution_bias_error(
    resolution_bandwidth: float,
    half_power_bandwidth: float,
) -> float
```

First-order resolution-bias error at a resonance peak (Eq. 8.141).

$\varepsilon_b[\hat{G}_{xx}(f_r)] \approx -(B_e/B_r)^2/3$ for
a resonance of half-power bandwidth
$B_r$ analysed with resolution bandwidth $B_e$: peaks are
underestimated (and valleys overestimated) by frequency smoothing, in
the direction of reduced dynamic range (B&P Section 8.5.1). The
approximation assumes $B_e < B_r$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `resolution_bandwidth` | Analysis resolution bandwidth $B_e$, Hz ([`SpectralDensityResult.resolution_bandwidth`](/phonometry/reference/api/spectra/spectra/#spectraldensityresult)). |
| `half_power_bandwidth` | Half-power (-3 dB) bandwidth $B_r$ of the spectral peak, in Hz. |

**Returns:** Normalized bias error (dimensionless, negative at a peak).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If either bandwidth is not positive. |

## SpectralDensityResult

```python
SpectralDensityResult(
    frequencies: NDArray[np.float64],
    psd: NDArray[np.float64],
    ci_lower: NDArray[np.float64],
    ci_upper: NDArray[np.float64],
    confidence: float,
    random_error: float,
    n_segments: int,
    n_averages: float,
    degrees_of_freedom: float,
    resolution_bandwidth: float,
    window: str,
    nperseg: int,
    overlap: float,
    scaling: str,
)
```

Welch autospectral density with its statistical error (B&P Ch. 8).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-sided frequency axis, in Hz. |
| `psd` | Autospectral density $\hat{G}_{xx}(f)$ (units²/Hz for `'density'` scaling, units² for `'spectrum'`). |
| `ci_lower` | Lower chi-square confidence bound on $G_{xx}$ (Eq. 8.163; the DC bin, and the Nyquist bin for an even segment length, use $n = n_d$ degrees of freedom - a wider interval - because those bins carry a single real Fourier component). |
| `ci_upper` | Upper chi-square confidence bound on $G_{xx}$. |
| `confidence` | Confidence level of the interval (e.g. `0.95`). |
| `random_error` | Normalized random error $\varepsilon[\hat{G}_{xx}] = 1/\sqrt{n_d}$ (Eq. 8.158) of the interior bins ($\sqrt{2/n_d}$ at DC/Nyquist). |
| `n_segments` | Raw number of (possibly overlapped) segments averaged. |
| `n_averages` | Effective number of independent averages $n_d$ (equals `n_segments` without overlap; smaller with overlap). |
| `degrees_of_freedom` | Chi-square degrees of freedom $n = 2 n_d$ of the interior bins (Eq. 8.162; $n_d$ at DC/Nyquist). |
| `resolution_bandwidth` | Effective noise bandwidth $B_e$ of the tapered segment, in Hz (drives the bias error of Eq. 8.139). |
| `window` | Taper name. |
| `nperseg` | Segment length, in samples. |
| `overlap` | Segment overlap fraction. |
| `scaling` | `'density'` or `'spectrum'`. |

### SpectralDensityResult.plot()

```python
SpectralDensityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the spectral density in dB with its confidence band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## window_metrics

```python
window_metrics(
    window: str | tuple[Any, ...],
    n: int = 1024,
) -> WindowMetricsResult
```

Figures of merit of a spectral-analysis taper (Harris 1978).

Computes the numbers behind the window trade-off for any taper the
`window` parameter of this module's estimators accepts: the
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

**Returns:** A [`WindowMetricsResult`](/phonometry/reference/api/spectra/spectra/#windowmetricsresult).

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
`scipy.signal.welch` and the estimators of this module use it.

**Attributes**

| Name | Description |
| :--- | :--- |
| `window` | The window specification as given (any name or `(name, param)` tuple `scipy.signal.get_window` accepts). |
| `n` | Window length, in samples. |
| `taps` | The window samples `w[m]` (DFT-even). |
| `coherent_gain` | Normalized DC gain $\sum w / n$ (1 for rectangular); the amplitude a bin-centered tone is scaled by before correction. |
| `enbw_bins` | Equivalent noise bandwidth $n \sum w^2 / \left( \sum w \right)^2$, in bins: the width of the ideal rectangular filter that would pass the same white-noise power (1 rectangular, 1.5 Hann, 1987/1458 Hamming). |
| `scalloping_loss_db` | Attenuation of a tone midway between two bins, $-20 \lg \lvert W(1/2)/W(0) \rvert$, in dB (positive). |
| `worst_case_processing_loss_db` | Scalloping loss plus the ENBW processing loss $10 \lg(\mathrm{ENBW})$, in dB: the worst-case reduction in output signal-to-noise ratio for a tone in white noise. |
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
