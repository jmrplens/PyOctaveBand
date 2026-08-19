---
title: "electroacoustics.distortion"
description: "Harmonic distortion of electroacoustic equipment (IEC 60268-3 / AES17)."
sidebar:
  label: "distortion"
---

Harmonic distortion of electroacoustic equipment (IEC 60268-3 / AES17).

Single-tone distortion metrics of amplifiers and audio equipment, from a
captured signal:

* **Total harmonic distortion** `THD` (IEC 60268-3 14.12.2-3), relative to
  the fundamental (`kind='F'`, the widespread convention) or to the total
  RMS (`kind='R'`, the 14.12.3.2 quantity), and the **nth-order harmonic
  distortion** (14.12.5).
* **THD+N** and the derived **SINAD** (AES17-2015 6.3.1): the fundamental is
  removed with the standard notch filter and the residual is compared with
  the total signal, both through the AES17 measurement bandwidth (20 Hz to
  20 kHz by default).
* **Weighted THD** (14.12.11), the harmonic residual weighted by the
  IEC 60268-1 / ITU-R BS.468-4 network (A/C optional), whose nominal
  response [`itu_r_468_weighting`](/phonometry/reference/api/electroacoustics/distortion/#itu_r_468_weighting) exposes on its own.

All metrics have an exact analytic oracle: a signal synthesised with known
harmonic amplitudes reproduces the closed-form ratio. The functions assume
the tones fall on (or very near) FFT bins -- use coherent sampling (an
integer number of periods) or supply a low-leakage window, as audio
analysers do.

The two-tone metrics of IEC 60268-3 14.12.7-10 live in
[`phonometry.electroacoustics.intermodulation`](/phonometry/reference/api/electroacoustics/intermodulation/), and the AES17-2015 6.4
noise levels in [`phonometry.electroacoustics.noise_measurements`](/phonometry/reference/api/electroacoustics/noise-measurements/); both
are built on the spectrum, notch and weighting helpers of this module.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## harmonic_analysis

```python
harmonic_analysis(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    n_harmonics: int = 10,
    notch_q: float = 2.0,
    bandwidth: float | None = 20000.0,
    window: str = 'hann',
) -> HarmonicDistortionResult
```

Full harmonic analysis of a signal (THD, THD+N, SINAD).

Bundles the fundamental, the harmonic amplitudes and the THD (both
conventions), THD+N and SINAD into a plottable result.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples: the three distortion ratios come out unchanged, and so does `sinad_db`, which is the decibel form of one such ratio rather than a level. Only `harmonic_amplitudes` carries the unit, and so lands in pascals when the record is calibrated. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `fundamental` | Fundamental frequency, or `None` to auto-detect. |
| `n_harmonics` | Highest harmonic order (default 10). |
| `notch_q` | Effective notch quality factor for THD+N (default 2.0). |
| `bandwidth` | AES17 measurement bandwidth for THD+N/SINAD, in Hz (default 20 kHz; `None` measures the full Nyquist band). |
| `window` | FFT window (default `'hann'`). |

**Returns:** A [`HarmonicDistortionResult`](/phonometry/reference/api/electroacoustics/distortion/#harmonicdistortionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## harmonic_distortion

```python
harmonic_distortion(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    fundamental: float,
    order: int,
    n_harmonics: int = 10,
    window: str = 'hann',
) -> float
```

nth-order harmonic distortion $d_n$ (IEC 60268-3 14.12.5).

$d_n = a_n / \sqrt{\sum_{k \ge 1} a_k^2}$ -- the nth harmonic
amplitude relative to the total RMS.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: this is a ratio of amplitudes drawn from the same record, so the factor divides out and the answer is the same calibrated or not. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `fundamental` | Fundamental frequency $f_1$, in Hz. |
| `order` | Harmonic order `n` (>= 2). |
| `n_harmonics` | Highest harmonic order used for the total RMS. |
| `window` | FFT window (default `'hann'`). |

**Returns:** nth-order harmonic distortion, as a ratio.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `order` \< 2 or the inputs are invalid. |

## HarmonicDistortionResult

```python
HarmonicDistortionResult(
    fundamental: float,
    harmonic_frequencies: NDArray[np.float64],
    harmonic_amplitudes: NDArray[np.float64],
    thd_f: float,
    thd_r: float,
    thd_plus_noise: float,
    sinad_db: float,
)
```

Harmonic analysis of a signal (IEC 60268-3 / AES17).

**Attributes**

| Name | Description |
| :--- | :--- |
| `fundamental` | Fundamental frequency $f_1$, in Hz. |
| `harmonic_frequencies` | Harmonic frequencies $n f_1$ present, in Hz. |
| `harmonic_amplitudes` | Peak amplitudes $a_n$ of the harmonics. |
| `thd_f` | Total harmonic distortion relative to the fundamental. |
| `thd_r` | Total harmonic distortion relative to the total RMS. |
| `thd_plus_noise` | THD+N ratio (AES17). |
| `sinad_db` | SINAD, in dB. |

### HarmonicDistortionResult.plot()

```python
HarmonicDistortionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the magnitude spectrum with the harmonics marked.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## itu_r_468_weighting

```python
itu_r_468_weighting(frequencies: ArrayLike) -> NDArray[np.float64]
```

ITU-R BS.468-4 weighting response, in dB re 1 kHz.

The nominal response of the Recommendation's Table 1 (identical to the
IEC 60268-1 Appendix A network required by IEC 60268-3 14.12.11),
interpolated linearly in dB over log-frequency -- the Recommendation's
own rule for values between the mask frequencies -- and extrapolated
beyond the table with the end-segment slopes. Zero frequency (DC) maps
to `-inf` dB. AES17-2015 5.2.7 tabulates the same curve with an
additional gain of -5,63 dB (unity at 2 kHz, the "CCIR-RMS" filter).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in Hz (scalar or array-like, >= 0). |

**Returns:** Response in dB re the 1 kHz value, same shape as the input.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for negative or non-finite frequencies. |

## sinad

```python
sinad(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    notch_q: float = 2.0,
    bandwidth: float | None = 20000.0,
    window: str = 'hann',
) -> float
```

Signal-to-noise-and-distortion ratio SINAD, in dB.

$$
\mathrm{SINAD} = -(\text{THD+N in dB}) = 20 \log_{10}(V_\text{total} / V_\text{residual}),
$$

the reciprocal, in dB, of the THD+N ratio. AES17-2015 does not itself
define SINAD; this value is derived from the AES17 6.3.1 THD+N
measurement (same notch, same measurement bandwidth).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: SINAD is the decibel form of a ratio of amplitudes drawn from the same record rather than a level, so the factor divides out and the answer is the same calibrated or not. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `fundamental` | Fundamental frequency, or `None` to auto-detect. |
| `notch_q` | Effective notch quality factor (AES17: 1.2..3; default 2.0). |
| `bandwidth` | Upper band-edge frequency of the AES17 chain, in Hz (default 20 kHz); `None` measures the full Nyquist band. |
| `window` | FFT window used only for fundamental auto-detection. |

**Returns:** SINAD, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## thd

```python
thd(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    kind: Literal['F', 'R'] = 'F',
    n_harmonics: int = 10,
    window: str = 'hann',
) -> float
```

Total harmonic distortion (IEC 60268-3 14.12.2-3).

From the harmonic amplitudes $a_n$:

$$
\mathrm{THD}_F = \sqrt{\sum_{n \ge 2} a_n^2} / a_1
$$

$$
\mathrm{THD}_R = \sqrt{\sum_{n \ge 2} a_n^2} / \sqrt{\sum_{n \ge 1} a_n^2}
$$

relative to the fundamental (`kind='F'`) or to the total RMS
(`kind='R'`), respectively.

Convention note: the quantity the IEC 60268-3 14.12.3.2 formula defines
is the R form (harmonic RMS over total RMS). The default `kind='F'` is
the fundamental-referenced convention widespread in audio practice and
datasheets; the two agree to first order for small distortion.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: this is a ratio of amplitudes drawn from the same record, so the factor divides out and the answer is the same calibrated or not. Coherent sampling (integer periods) or a low-leakage window gives the exact value. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `fundamental` | Fundamental frequency $f_1$ in Hz, or `None` to take the largest spectral peak. |
| `kind` | `'F'` (relative to the fundamental, the default) or `'R'` (relative to the total RMS, the 14.12.3.2 quantity). |
| `n_harmonics` | Highest harmonic order summed (default 10). |
| `window` | FFT window (default `'hann'`). |

**Returns:** Total harmonic distortion, as a ratio (0..).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the signal/parameters are invalid, `kind` is unknown, or no harmonic of the fundamental lies below Nyquist. |

## thd_plus_noise

```python
thd_plus_noise(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    notch_q: float = 2.0,
    bandwidth: float | None = 20000.0,
    window: str = 'hann',
    as_db: bool = False,
) -> float
```

THD+N ratio (AES17-2015 6.3.1).

The fundamental is removed with the standard notch filter
($1.2 \le Q \le 3$, validated on the applied zero-phase response
per 5.2.8) and the residual RMS is compared with the total RMS:
$\mathrm{THD{+}N} = V_\text{residual} / V_\text{total}$ (a
ratio, or $20 \log_{10}$ of it in dB). Both voltages are measured
through the AES17 measurement bandwidth -- a 20 Hz high-pass plus the
standard low-pass at `bandwidth` (5.2.5 / 6.3.1) -- so DC offsets
and out-of-band noise do not inflate the result.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: this is a ratio of amplitudes drawn from the same record, so the factor divides out and the answer is the same calibrated or not. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `fundamental` | Fundamental frequency, or `None` to auto-detect. |
| `notch_q` | Effective notch quality factor (AES17: 1.2..3; default 2.0). |
| `bandwidth` | Upper band-edge frequency of the AES17 chain, in Hz (default 20 kHz, the 5.2.5 standard value; capped at Nyquist). `None` disables the chain and measures the full Nyquist band (20 Hz high-pass included only when the chain is active). |
| `window` | FFT window used only for fundamental auto-detection. |
| `as_db` | Return $20 \log_{10}(\text{ratio})$ in dB instead of the ratio. |

**Returns:** THD+N as a ratio (default) or in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or `notch_q` out of range. |

## weighted_thd

```python
weighted_thd(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    fundamental: float | None = None,
    *,
    notch_q: float = 2.0,
    weighting: Literal['468', 'A', 'C'] = '468',
    window: str = 'hann',
) -> float
```

Weighted total harmonic distortion (IEC 60268-3 14.12.11).

The fundamental is notched out and the residual is frequency-weighted
before its RMS is compared with the total signal RMS, so the perceptual
emphasis of the distortion products is accounted for. The default
weighting is the network required by the clause -- IEC 60268-1:1985
Appendix A, the ITU-R BS.468-4 curve (peaking +12,2 dB near 6,3 kHz) with
its standard 0 dB at 1 kHz normalization; `'A'` and `'C'` (IEC
61672-1) are kept as explicitly labelled alternatives, not 14.12.11
quantities.

Validity note (14.12.11): because of the shape of the weighting response,
the weighted measurement is valid only for fundamental frequencies
between 31,5 Hz and 400 Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: this is a ratio of amplitudes drawn from the same record, so the factor divides out and the answer is the same calibrated or not. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `fundamental` | Fundamental frequency, or `None` to auto-detect. |
| `notch_q` | Effective notch quality factor (default 2.0). |
| `weighting` | Frequency weighting applied to the residual: `'468'` (ITU-R BS.468-4 / IEC 60268-1, the 14.12.11 default), `'A'` or `'C'`. |
| `window` | FFT window used only for fundamental auto-detection. |

**Returns:** Weighted THD, as a ratio.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |
