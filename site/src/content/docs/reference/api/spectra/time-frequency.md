---
title: "metrology.time_frequency"
description: "Calibrated time-frequency analysis: STFT spectrogram and zoom FFT."
sidebar:
  label: "time_frequency"
---

Calibrated time-frequency analysis: STFT spectrogram and zoom FFT.

Fine-band time-frequency views of a record, following Bendat & Piersol,
*Random Data: Analysis and Measurement Procedures* (4th ed., 2010):

* the **spectrogram** (Section 12.6.4.2): short-time Fourier transforms of
  contiguous, tapered, possibly overlapped segments, displayed over the
  time-frequency plane (Eq. 12.173 defines the unweighted magnitude
  version; this module computes the power version with the exact
  `'density'`/`'spectrum'` calibration of
  [`power_spectral_density`](/phonometry/reference/api/spectra/spectra/#power_spectral_density), so a
  signal in pascals reads directly in Pa²/Hz or Pa² and averaging the
  columns reproduces the Welch estimate bin by bin). Each cell trades the
  time resolution $T_B = \text{nperseg}/f_s$ against the frequency
  resolution
  $1/T_B$ at a resolution-bandwidth-time product of one, so a single
  cell of random data is an unaveraged estimate: the power carries a
  normalized random error of $1/\sqrt{n_d} = 1$ with
  $n_d = 1$ (Eq. 8.158),
  and the magnitude display an error of
  $\sqrt{2}/1.25 \approx 1.13$ - the
  Rayleigh-ratio result Bendat & Piersol quote in Section 12.6.4.2.
  Deterministic structure (tones, sweeps, transients) is unaffected by
  that caveat and is what the spectrogram is for.

* the **zoom FFT** (Section 11.5.4): the spectrum of a narrow band
  $[f_{\min}, f_{\max}]$ computed on an arbitrarily fine frequency
  grid without the giant FFT block a full-band analysis would need
  (Eq. 11.122). The book's procedure - bandpass, complex demodulation by
  $\exp(-j 2 \pi f_1 t)$, decimation by
  $d = k_2/(k_2 - k_1)$ and an FFT of the
  decimated record (Eqs. 11.123-11.130) - is realized here in its exact
  single-pass digital equivalent, the chirp-Z evaluation of the DFT on
  the zoom grid ([`scipy.signal.zoom_fft`](/phonometry/reference/api/spectra/time-frequency/#zoom_fft)): both compute the same
  DFT samples of the record, which the test suite verifies to machine
  precision against the demodulate-decimate-DFT chain. The bin spacing
  can be made arbitrarily fine, but the true resolution stays set by the
  record length and taper (the reported effective noise bandwidth
  $B_e = f_s \sum w^2 / \left( \sum w \right)^2$): zooming refines
  the grid, only a longer record
  refines the resolution (Eq. 11.127).

Amplitudes are calibrated so that a sine of peak amplitude `A` on an
analysis frequency reads $\lvert \text{spectrum} \rvert = A$,
$\text{power} = A^2/2$ (its mean
square) - consistent with the `'spectrum'` scaling of the Welch module.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## spectrogram

```python
spectrogram(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    window: str = 'hann',
    nperseg: int | None = None,
    overlap: float = 0.5,
    scaling: Literal['density', 'spectrum'] = 'density',
) -> SpectrogramResult
```

Calibrated STFT power spectrogram (Bendat & Piersol 12.6.4.2).

The record is split into tapered (Hann by default), overlapped
segments - exactly the segmentation of
[`power_spectral_density`](/phonometry/reference/api/spectra/spectra/#power_spectral_density) - and
each segment's one-sided periodogram becomes one column of the
time-frequency display, without the averaging that the Welch
estimate applies (averaging the columns reproduces it bin by bin).
No detrending is applied, so absolute calibration is preserved: a
signal in pascals yields Pa²/Hz (`'density'`) or Pa²
(`'spectrum'`), and a sine of amplitude `A` on an analysis
frequency reads $A^2/2$ - its mean square - in every
`'spectrum'` column it spans.

The display trades time against frequency resolution through the
segment length: $T_B = \text{nperseg}/f_s$ of time resolution
against
$B_e \approx 1/T_B$ of frequency resolution
(Section 12.6.4.2). Because
each cell is a single unaveraged estimate
($B_e T_B \approx 1$), random
data carries a per-cell normalized random error of 1 (Eq. 8.158 with
$n_d = 1$): the spectrogram is a tool for deterministic
structure -
tones, sweeps, transients - not a low-variance spectral estimator.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `window` | Segment taper (any scipy window name; default Hann). |
| `nperseg` | Segment length; `None` picks a length giving a bin spacing of at most 4 Hz (the Welch-module default). |
| `overlap` | Segment overlap fraction in [0, 1) (default 0.5). |
| `scaling` | `'density'` (units²/Hz) or `'spectrum'` (units²). |

**Returns:** A [`SpectrogramResult`](/phonometry/reference/api/spectra/time-frequency/#spectrogramresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## SpectrogramResult

```python
SpectrogramResult(
    times: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    power: NDArray[np.float64],
    time_resolution: float,
    resolution_bandwidth: float,
    random_error: float,
    n_segments: int,
    hop: int,
    window: str,
    nperseg: int,
    overlap: float,
    scaling: str,
)
```

Calibrated STFT power spectrogram (B&P Section 12.6.4.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `times` | Segment-centre times, in seconds (one per column). |
| `frequencies` | One-sided frequency axis, in Hz (one per row). |
| `power` | Power spectrogram, shape `(frequencies, times)` (units²/Hz for `'density'` scaling, units² for `'spectrum'`). Each column is the tapered periodogram of one segment, with the exact calibration of [`power_spectral_density`](/phonometry/reference/api/spectra/spectra/#power_spectral_density): the column mean over time reproduces the Welch spectrum bin by bin. Integrating a `'density'` column over frequency gives that segment's taper-weighted mean square $\sum (x w)^2 / \sum w^2$; summing those over time *and multiplying by the hop duration* `hop/fs` recovers the record energy $\sum x^2 / f_s$ when the squared taper overlap-adds to a constant (e.g. Hann at 75 % overlap), up to the taper roll-off at the record edges (the first and last segments are under-weighted: about 1-2 % low for typical records). |
| `time_resolution` | Segment duration $T_B = \text{nperseg}/f_s$, in seconds - the time resolution of the display. |
| `resolution_bandwidth` | Effective noise bandwidth $B_e$ of the tapered segment, in Hz - the frequency resolution ($\approx 1/T_B$ for a light taper; the $B_e T_B$ product per cell is close to 1). |
| `random_error` | Normalized random error of each (unaveraged) power cell for random data, $1/\sqrt{n_d} = 1$ with $n_d = 1$ (Eq. 8.158); Bendat & Piersol quote $\sqrt{2}/1.25 \approx 1.13$ for the magnitude display (Section 12.6.4.2). Deterministic components are unaffected. |
| `n_segments` | Number of segments (columns). |
| `hop` | Hop between segment starts, in samples. |
| `window` | Taper name. |
| `nperseg` | Segment length, in samples. |
| `overlap` | Segment overlap fraction. |
| `scaling` | `'density'` or `'spectrum'`. |

### SpectrogramResult.plot()

```python
SpectrogramResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the spectrogram in dB over the time-frequency plane.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## zoom_fft

```python
zoom_fft(
    x: NDArray[np.float64] | list[float],
    fs: float,
    f_min: float,
    f_max: float,
    *,
    n_points: int | None = None,
    window: str = 'hann',
) -> ZoomFFTResult
```

Zoom FFT: the spectrum of a narrow band on a fine grid (B&P 11.5.4).

Resolves closely spaced tones - gear sidebands, twin machines, power
hum - separated by less than a practical full-band FFT bin, without
the giant block size Eq. 11.122 would demand. Bendat & Piersol's
procedure (bandpass, complex demodulation to shift `f_min` to zero,
decimation by the bandwidth ratio, FFT of the decimated record;
Eqs. 11.123-11.130) is computed here in its exact single-pass digital
equivalent: the chirp-Z evaluation of the tapered record's DFT on the
zoom grid ([`scipy.signal.zoom_fft`](/phonometry/reference/api/spectra/time-frequency/#zoom_fft)), which yields the same DFT
samples to machine precision.

Amplitudes are calibrated per taper coherent gain
($2 X / \sum w$), so a
sine of peak amplitude `A` on an analysis frequency reads
$\text{amplitude} = A$ and $\text{power} = A^2/2$
exactly. The grid can be made
arbitrarily fine, but the true resolution remains the reported
effective noise bandwidth of the tapered record ($1/T$ for no
taper): the zoom refines the *grid*; only a longer record separates
tones closer than $B_e$ (Eq. 11.127).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `f_min` | Lower edge of the zoom band, in Hz ($\ge 0$). |
| `f_max` | Upper edge of the zoom band, in Hz ($\le f_s/2$). |
| `n_points` | Grid points across `[f_min, f_max]` (endpoints included); `None` places one point per record-length resolution $f_s/N$. |
| `window` | Record taper (any scipy window name; default Hann; `'boxcar'` for none). |

**Returns:** A [`ZoomFFTResult`](/phonometry/reference/api/spectra/time-frequency/#zoomfftresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## ZoomFFTResult

```python
ZoomFFTResult(
    frequencies: NDArray[np.float64],
    spectrum: NDArray[np.complex128],
    amplitude: NDArray[np.float64],
    power: NDArray[np.float64],
    bin_spacing: float,
    resolution_bandwidth: float,
    window: str,
    n_points: int,
)
```

Narrow-band zoom spectrum on a fine frequency grid (B&P 11.5.4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Zoom frequency grid from `f_min` to `f_max` inclusive, in Hz. |
| `spectrum` | Complex amplitude-calibrated coefficients: a sine of peak amplitude `A` on an analysis frequency reads $\lvert \text{spectrum} \rvert = A$ (calibration $2 X(f) / \sum w$; no one-sided doubling at exactly 0 Hz or the Nyquist frequency). |
| `amplitude` | $\lvert \text{spectrum} \rvert$ - peak-amplitude spectrum. |
| `power` | Mean-square spectrum $\text{amplitude}^2/2$ ($\text{amplitude}^2$ at DC/Nyquist), consistent with the `'spectrum'` scaling of the Welch module: a tone reads its mean square $A^2/2$. |
| `bin_spacing` | Grid spacing, in Hz - freely chosen, finer than $f_s/N$ if requested (the zoom gain of Eq. 11.127). |
| `resolution_bandwidth` | Effective noise bandwidth $B_e = f_s \sum w^2 / \left( \sum w \right)^2$ of the tapered record, in Hz - the true resolution, set by the record length and taper, that no grid refinement improves. |
| `window` | Taper name. |
| `n_points` | Number of grid points. |

### ZoomFFTResult.plot()

```python
ZoomFFTResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the zoom power spectrum in dB over the zoom band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
