---
title: "signals.spectra"
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
  $b[\hat{G}_{xx}] \approx (B_\mathrm{e}^2/24) \, G''_{xx}$
  (Eq. 8.139), which for a resonance peak of half-power bandwidth
  $B_\mathrm{r}$
  becomes $\varepsilon_\mathrm{b} \approx -(B_\mathrm{e}/B_\mathrm{r})^2/3$ (Eq. 8.141) -
  exposed here as
  [`resolution_bias_error`](/phonometry/reference/api/signals/spectra/#resolution_bias_error);
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

The taper the `window` parameter selects is characterized, with the
figures of merit of Harris (1978), by
[`phonometry.signals.windows.window_metrics`](/phonometry/reference/api/signals/windows/#window_metrics); the whole-record
alternative to Welch segment averaging, Thomson's multitaper estimator,
is [`phonometry.signals.multitaper.multitaper_psd`](/phonometry/reference/api/signals/multitaper/#multitaper_psd).

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

**Returns:** A [`CoherentOutputSpectrumResult`](/phonometry/reference/api/signals/spectra/#coherentoutputspectrumresult).

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
| `snr_db` | $10 \log_{10}$ of `snr`, in dB. |
| `random_error` | Normalized random error of $\hat{G}_{vv}$, $\varepsilon = (2 - \gamma^2_{xy})^{1/2} / (\lvert \gamma_{xy} \rvert \sqrt{n_d})$ (Eq. 9.73), with the measured coherence in place of the true value. |
| `snr_random_error` | Normalized random error of the SNR, $\varepsilon = \sqrt{2} / (\lvert \gamma_{xy} \rvert \sqrt{n_d})$, first-order propagation of the coherence random error of Eq. 9.82 through $\gamma^2/(1 - \gamma^2)$. |
| `coherence_bias` | First-order bias of the coherence estimate, $b[\hat{\gamma}^2] \approx (1 - \gamma^2)^2 / n_d$ (Eq. 9.75). |
| `n_segments` | Raw number of segments averaged. |
| `n_averages` | Effective number of independent averages $n_d$. |
| `resolution_bandwidth` | Effective noise bandwidth $B_\mathrm{e}$, in Hz. |
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

**Returns:** A [`CrossSpectralDensityResult`](/phonometry/reference/api/signals/spectra/#crossspectraldensityresult).

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
| `resolution_bandwidth` | Effective noise bandwidth $B_\mathrm{e}$, in Hz. |
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
[`resolution_bias_error`](/phonometry/reference/api/signals/spectra/#resolution_bias_error).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `window` | Segment taper (any scipy window name; default Hann, the B&P Section 11.5.2 recommendation for side-lobe suppression). |
| `nperseg` | Welch segment length; `None` picks a length giving a bin spacing of at most 4 Hz (the resolution bandwidth $B_\mathrm{e}$ further depends on the taper; see [`SpectralDensityResult.resolution_bandwidth`](/phonometry/reference/api/signals/spectra/#spectraldensityresult)). |
| `overlap` | Segment overlap fraction in [0, 1) (default 0.5, which with a Hann taper retrieves most of the stability lost to tapering, B&P Section 11.5.2.2). |
| `scaling` | `'density'` (units²/Hz) or `'spectrum'` (units² per segment bandwidth). |
| `confidence` | Confidence level for the chi-square interval. |

**Returns:** A [`SpectralDensityResult`](/phonometry/reference/api/signals/spectra/#spectraldensityresult).

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

$\varepsilon_\mathrm{b}[\hat{G}_{xx}(f_\mathrm{r})] \approx -(B_\mathrm{e}/B_\mathrm{r})^2/3$ for
a resonance of half-power bandwidth
$B_\mathrm{r}$ analysed with resolution bandwidth $B_\mathrm{e}$: peaks are
underestimated (and valleys overestimated) by frequency smoothing, in
the direction of reduced dynamic range (B&P Section 8.5.1). The
approximation assumes $B_\mathrm{e} < B_\mathrm{r}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `resolution_bandwidth` | Analysis resolution bandwidth $B_\mathrm{e}$, Hz ([`SpectralDensityResult.resolution_bandwidth`](/phonometry/reference/api/signals/spectra/#spectraldensityresult)). |
| `half_power_bandwidth` | Half-power (-3 dB) bandwidth $B_\mathrm{r}$ of the spectral peak, in Hz. |

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
| `resolution_bandwidth` | Effective noise bandwidth $B_\mathrm{e}$ of the tapered segment, in Hz (drives the bias error of Eq. 8.139). |
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
