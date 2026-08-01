---
title: "electroacoustics.distortion"
description: "Distortion metrics for electroacoustic equipment (IEC 60268-3 / AES17)."
sidebar:
  label: "distortion"
---

Distortion metrics for electroacoustic equipment (IEC 60268-3 / AES17).

Harmonic and intermodulation distortion of amplifiers and audio equipment, from
a captured signal:

* **Total harmonic distortion** `THD` (IEC 60268-3 14.12.2-3), relative to
  the fundamental (`kind='F'`, the widespread convention) or to the total
  RMS (`kind='R'`, the 14.12.3.2 quantity), and the **nth-order harmonic
  distortion** (14.12.5).
* **THD+N** and the derived **SINAD** (AES17-2015 6.3.1): the fundamental is
  removed with the standard notch filter and the residual is compared with
  the total signal, both through the AES17 measurement bandwidth (20 Hz to
  20 kHz by default).
* **Modulation distortion** `d_m,2`/`d_m,3` (IEC 60268-3 14.12.7) and
  **difference-frequency distortion** `d_d,2`/`d_d,3` (14.12.8), plus the
  **total difference-frequency distortion** (14.12.10) -- the IEC per-order
  definitions, with the SMPTE combined-RMS convention alongside.
* **Dynamic intermodulation distortion** `DIM` (14.12.9) from the 15 kHz sine /
  3.15 kHz square-wave test signal.
* **Weighted THD** (14.12.11), the harmonic residual weighted by the
  IEC 60268-1 / ITU-R BS.468-4 network (A/C optional).

All metrics have an exact analytic oracle: a signal synthesised with known
harmonic or intermodulation amplitudes reproduces the closed-form ratio. The
functions assume the tones fall on (or very near) FFT bins -- use coherent
sampling (an integer number of periods) or supply a low-leakage window, as audio
analysers do.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## difference_frequency_distortion

```python
difference_frequency_distortion(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    f1: float,
    f2: float,
    *,
    order: int = 2,
    window: str = 'hann',
) -> float
```

Difference-frequency distortion of the nth order (IEC 60268-3
14.12.8).

Two equal-amplitude tones $f_1 < f_2$ are applied. Per 14.12.8.1
the reference voltage is $U_{2,\mathrm{ref}} = 2 U_{2,f_2}$ --
realised here as the sum of both measured tone amplitudes, identical
for the standard equal-amplitude tones -- and

$$
d_{d,2} = a_{f_2-f_1} / (a_{f_1} + a_{f_2})
$$

$$
d_{d,3} = (a_{2f_2-f_1} + a_{2f_1-f_2}) / (a_{f_1} + a_{f_2})
$$

with the third order an *arithmetic* sum of the two products. Products
that fall outside (0, Nyquist) or that cannot be separated from a primary
tone or DC read zero.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
| `f1` | Lower tone, in Hz. |
| `f2` | Upper tone, in Hz. |
| `order` | Product order (2 or 3). |
| `window` | FFT window (default `'hann'`). |

**Returns:** nth-order difference-frequency distortion, as a ratio.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `order` is not 2 or 3 or the inputs are invalid. |

## dynamic_intermodulation_distortion

```python
dynamic_intermodulation_distortion(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    *,
    f_sine: float = 15000.0,
    f_square: float = 3150.0,
    window: str = 'hann',
) -> float
```

Dynamic intermodulation distortion DIM (IEC 60268-3 14.12.9).

From the standard test signal -- a `f_sine` = 15 kHz sine plus a
low-pass-filtered `f_square` = 3.15 kHz square wave in a 1:4 peak
ratio -- the DIM is the RMS of the intermodulation products
$\lvert k \, f_\mathrm{square} \pm f_\mathrm{sine} \rvert$ that
fall below `f_sine` (IEC 60268-3 Table 2), relative to the 15 kHz
sine amplitude.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
| `f_sine` | High sine frequency, in Hz (default 15 kHz). |
| `f_square` | Square-wave fundamental, in Hz (default 3.15 kHz). |
| `window` | FFT window (default `'hann'`). |

**Returns:** Dynamic intermodulation distortion, as a ratio.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## dynamic_range

```python
dynamic_range(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    fundamental: float | None = None,
    *,
    notch_q: float = 2.0,
    bandwidth: float | None = 20000.0,
    full_scale: float = 1.0,
    window: str = 'hann',
) -> float
```

Dynamic range of an audio device (AES17-2015 6.4.1), in dB CCIR-RMS.

The device is driven with a 997 Hz sine 60 dB below full scale; the
captured output has its fundamental removed by the standard notch filter
(5.2.8), and the residual noise-plus-distortion is weighted by the
CCIR-RMS filter (5.2.7) over the AES17 measurement band. The dynamic range
is the ratio of the maximum output level (a full-scale sine, 6.2.6) to
that weighted residual level:

$$
\mathrm{DR} = 20 \lg\left( (\text{full\_scale} / \sqrt{2}) / V_\text{residual,CCIR-RMS} \right)
$$

It includes all harmonic, inharmonic and noise components and is also
known as the signal-to-noise ratio (6.4.1 note).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured output of the device under test (1-D), scaled so that `full_scale` is the digital full-scale peak amplitude. |
| `fs` | Sample rate, in Hz. |
| `fundamental` | Test frequency, in Hz; `None` uses the 997 Hz of the standard. |
| `notch_q` | Effective notch quality factor (AES17 5.2.8: 1.2..3; default 2.0). |
| `bandwidth` | AES17 measurement bandwidth, in Hz (default 20 kHz; `None` measures the full Nyquist band). |
| `full_scale` | Digital full-scale peak amplitude (default 1.0). |
| `window` | FFT window used only for fundamental auto-detection. |

**Returns:** The dynamic range, in dB CCIR-RMS (a positive number).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or `notch_q` is out of range. |

## harmonic_analysis

```python
harmonic_analysis(
    signal: NDArray[np.float64] | list[float],
    fs: float,
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
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
    fundamental: float,
    order: int,
    *,
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
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
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

## idle_channel_noise

```python
idle_channel_noise(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    *,
    bandwidth: float | None = 20000.0,
    full_scale: float = 1.0,
) -> float
```

Idle channel noise level (AES17-2015 6.4.2), in dBFS CCIR-RMS.

The weighted output of the device when driven with no signal (a
short-circuited analogue input or digital zero at the input). The captured
idle output is weighted by the CCIR-RMS filter (5.2.7) over the AES17
measurement band and reported relative to full scale:

$$
L_\text{idle} = 20 \lg\left( V_\text{idle,CCIR-RMS} / (\text{full\_scale} / \sqrt{2}) \right)
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured idle output of the device under test (1-D), scaled so that `full_scale` is the digital full-scale peak amplitude. |
| `fs` | Sample rate, in Hz. |
| `bandwidth` | AES17 measurement bandwidth, in Hz (default 20 kHz; `None` measures the full Nyquist band). |
| `full_scale` | Digital full-scale peak amplitude (default 1.0). |

**Returns:** The idle channel noise level, in dBFS CCIR-RMS (a negative number for any real device).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

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

## modulation_distortion

```python
modulation_distortion(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    f_low: float,
    f_high: float,
    *,
    window: str = 'hann',
) -> ModulationDistortionResult
```

Modulation distortion of the nth order (IEC 60268-3 14.12.7).

A low-frequency tone $f_1$ (`f_low`, large) and a
high-frequency tone $f_2$ (`f_high`; small, amplitude ratio
preferably 4:1) are applied; the nth-order distortion shows up as
modulation sidebands at $f_2 \pm (n-1) f_1$. Per 14.12.7.2
g)-h) the per-order values use the *arithmetic* sum of the two
sideband amplitudes, referenced to the output voltage at `f2`:

$$
d_{m,2} = (a_{f_2+f_1} + a_{f_2-f_1}) / a_{f_2}
$$

$$
d_{m,3} = (a_{f_2+2f_1} + a_{f_2-2f_1}) / a_{f_2}
$$

(The alternative presentation $d'_{m,n} = 5 d_{m,n}$ references
the 4:1 reference output voltage
$U_{2,\mathrm{ref}} = 5 U_{2,f_2}$ instead.) The combined
root-sum-square that SMPTE-type analyzers report is returned alongside
as `smpte`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
| `f_low` | Low modulating tone `f1`, in Hz (e.g. 60 Hz). |
| `f_high` | High carrier tone `f2`, in Hz (e.g. 7 kHz). |
| `window` | FFT window (default `'hann'`). |

**Returns:** A [`ModulationDistortionResult`](/phonometry/reference/api/electroacoustics/distortion/#modulationdistortionresult) with `d2`, `d3` and the `smpte` combined RMS.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## ModulationDistortionResult

```python
ModulationDistortionResult(
    d2: float,
    d3: float,
    smpte: float,
    f_low: float | None = None,
    f_high: float | None = None,
    carrier_amplitude: float | None = None,
    sideband_frequencies: NDArray[np.float64] | None = None,
    sideband_amplitudes: NDArray[np.float64] | None = None,
)
```

Modulation (intermodulation) distortion (IEC 60268-3 14.12.7).

**Attributes**

| Name | Description |
| :--- | :--- |
| `d2` | Second-order modulation distortion $d_{m,2}$ (14.12.7.2 g): the *arithmetic* sum of the sideband amplitudes at $f_2 \pm f_1$ relative to the output amplitude at `f2`. |
| `d3` | Third-order modulation distortion $d_{m,3}$ (14.12.7.2 h): the arithmetic sum of the sidebands at $f_2 \pm 2 f_1$ relative to the output amplitude at `f2`. |
| `smpte` | Combined-RMS convention of SMPTE-type analyzers (not an IEC 60268-3 quantity): $\sqrt{\sum a_s^2} / a_{f_2}$ over all four sidebands. |
| `f_low` | Low modulating tone `f1`, in Hz. |
| `f_high` | High carrier tone `f2`, in Hz. |
| `carrier_amplitude` | Measured output amplitude at `f2` (the reference of the per-order ratios). |
| `sideband_frequencies` | The four intermodulation product frequencies in ascending order: $f_2 - 2f_1$, $f_2 - f_1$, $f_2 + f_1$ and $f_2 + 2f_1$, in Hz. |
| `sideband_amplitudes` | Measured peak amplitudes at `sideband_frequencies` (zero for a product that falls outside the analysis band or cannot be separated from a primary tone). |

### ModulationDistortionResult.plot()

```python
ModulationDistortionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the carrier and its modulation sidebands, with d2/d3
annotated.

Draws the output amplitude at `f2` (the 0 dB reference) and the
four intermodulation sidebands at $f_2 \pm f_1$ and
$f_2 \pm 2f_1$ as a stem-style spectrum in dB relative to
the carrier, the modulation counterpart of
[`HarmonicDistortionResult.plot`](/phonometry/reference/api/electroacoustics/distortion/#harmonicdistortionresultplot).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the marker `plot` call. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result carries no sideband spectrum data (a result constructed by hand without the spectral fields). |

## sinad

```python
sinad(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    fundamental: float | None = None,
    *,
    notch_q: float = 2.0,
    bandwidth: float | None = 20000.0,
    window: str = 'hann',
) -> float
```

Signal-to-noise-and-distortion ratio SINAD, in dB.

$$
\mathrm{SINAD} = -(\text{THD+N in dB}) = 20 \lg(V_\text{total} / V_\text{residual}),
$$

the reciprocal, in dB, of the THD+N ratio. AES17-2015 does not itself
define SINAD; this value is derived from the AES17 6.3.1 THD+N
measurement (same notch, same measurement bandwidth).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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
| `signal` | Captured signal (1-D). Coherent sampling (integer periods) or a low-leakage window gives the exact value. |
| `fs` | Sample rate, in Hz. |
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
    signal: NDArray[np.float64] | list[float],
    fs: float,
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
ratio, or $20 \lg$ of it in dB). Both voltages are measured
through the AES17 measurement bandwidth -- a 20 Hz high-pass plus the
standard low-pass at `bandwidth` (5.2.5 / 6.3.1) -- so DC offsets
and out-of-band noise do not inflate the result.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
| `fundamental` | Fundamental frequency, or `None` to auto-detect. |
| `notch_q` | Effective notch quality factor (AES17: 1.2..3; default 2.0). |
| `bandwidth` | Upper band-edge frequency of the AES17 chain, in Hz (default 20 kHz, the 5.2.5 standard value; capped at Nyquist). `None` disables the chain and measures the full Nyquist band (20 Hz high-pass included only when the chain is active). |
| `window` | FFT window used only for fundamental auto-detection. |
| `as_db` | Return $20 \lg(\text{ratio})$ in dB instead of the ratio. |

**Returns:** THD+N as a ratio (default) or in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or `notch_q` out of range. |

## total_difference_frequency_distortion

```python
total_difference_frequency_distortion(
    signal: NDArray[np.float64] | list[float],
    fs: float,
    f1: float = 8000.0,
    f2: float = 11950.0,
    *,
    window: str = 'hann',
) -> float
```

Total difference-frequency distortion (IEC 60268-3 14.12.10).

A specific two-tone test with $f_1 = 2 f_0$ and
$f_2 = 3 f_0 - \delta$ (the standard values, kept as defaults,
are $f_1 = 8$ kHz, $f_2 = 11.95$ kHz, so
$f_0 = 4$ kHz and $\delta = 50$ Hz). Only the two in-band
products at $f_0 \mp \delta$ enter -- the second-order product
at $f_2 - f_1$ and the third-order product at
$2 f_1 - f_2$ -- combined in RMS over the arithmetic sum of the
two tone output amplitudes (14.12.10.2 g):

$$
d_{\mathrm{TDFD}} = \sqrt{a_{f_2-f_1}^2 + a_{2f_1-f_2}^2} / (a_{f_1} + a_{f_2})
$$

(The out-of-band product at $2 f_2 - f_1$ is explicitly not
part of it.)

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
| `f1` | Lower tone, in Hz (default 8 kHz, per 14.12.10.2 b). |
| `f2` | Upper tone, in Hz (default 11.95 kHz, per 14.12.10.2 b). |
| `window` | FFT window (default `'hann'`). |

**Returns:** Total difference-frequency distortion, as a ratio.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## weighted_thd

```python
weighted_thd(
    signal: NDArray[np.float64] | list[float],
    fs: float,
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
| `signal` | Captured signal (1-D). |
| `fs` | Sample rate, in Hz. |
| `fundamental` | Fundamental frequency, or `None` to auto-detect. |
| `notch_q` | Effective notch quality factor (default 2.0). |
| `weighting` | Frequency weighting applied to the residual: `'468'` (ITU-R BS.468-4 / IEC 60268-1, the 14.12.11 default), `'A'` or `'C'`. |
| `window` | FFT window used only for fundamental auto-detection. |

**Returns:** Weighted THD, as a ratio.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |
