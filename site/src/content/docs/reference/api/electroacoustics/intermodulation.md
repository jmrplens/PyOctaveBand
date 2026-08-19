---
title: "electroacoustics.intermodulation"
description: "Intermodulation distortion of audio equipment (IEC 60268-3 14.12.7-10)."
sidebar:
  label: "intermodulation"
---

Intermodulation distortion of audio equipment (IEC 60268-3 14.12.7-10).

Where the harmonic metrics of [`phonometry.electroacoustics.distortion`](/phonometry/reference/api/electroacoustics/distortion/)
drive the equipment with a single tone, these drive it with two (or with a
sine against a square wave) and read the products that a non-linearity puts
*between* the excitation frequencies -- the components a harmonic
measurement cannot see and the ear finds least forgiving, because they are
inharmonic with the programme:

* **Modulation distortion** `d_m,2`/`d_m,3` (14.12.7): a large low tone
  $f_1$ modulating a small high tone $f_2$, read on the
  sidebands at $f_2 \pm (n-1) f_1$.
* **Difference-frequency distortion** `d_d,2`/`d_d,3` (14.12.8) from two
  equal-amplitude tones, and the **total difference-frequency distortion**
  (14.12.10) of the standard 8 kHz / 11,95 kHz pair.
* **Dynamic intermodulation distortion** `DIM` (14.12.9) from the 15 kHz
  sine / 3,15 kHz square-wave test signal.

The per-order definitions are the IEC ones (arithmetic sums of the product
amplitudes, referenced as each clause prescribes), with the SMPTE
combined-RMS convention reported alongside where analyzers use it. As with
the harmonic metrics, every quantity has an exact analytic oracle: a signal
synthesised with known product amplitudes reproduces the closed-form ratio,
and the tones are assumed to fall on (or very near) FFT bins.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## difference_frequency_distortion

```python
difference_frequency_distortion(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    f1: float,
    f2: float,
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
d_{\mathrm{d},2} = a_{f_2-f_1} / (a_{f_1} + a_{f_2})
$$

$$
d_{\mathrm{d},3} = (a_{2f_2-f_1} + a_{2f_1-f_2}) / (a_{f_1} + a_{f_2})
$$

with the third order an *arithmetic* sum of the two products. Products
that fall outside (0, Nyquist) or that cannot be separated from a primary
tone or DC read zero.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: this is a ratio of product amplitudes drawn from the same record, so the factor divides out and the answer is the same calibrated or not. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
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
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
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
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: this is a ratio of product amplitudes drawn from the same record, so the factor divides out and the answer is the same calibrated or not. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `f_sine` | High sine frequency, in Hz (default 15 kHz). |
| `f_square` | Square-wave fundamental, in Hz (default 3.15 kHz). |
| `window` | FFT window (default `'hann'`). |

**Returns:** Dynamic intermodulation distortion, as a ratio.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## modulation_distortion

```python
modulation_distortion(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    f_low: float,
    f_high: float,
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
d_{\mathrm{m},2} = (a_{f_2+f_1} + a_{f_2-f_1}) / a_{f_2}
$$

$$
d_{\mathrm{m},3} = (a_{f_2+2f_1} + a_{f_2-2f_1}) / a_{f_2}
$$

(The alternative presentation $d'_{\mathrm{m},n} = 5 d_{\mathrm{m},n}$ references
the 4:1 reference output voltage
$U_{2,\mathrm{ref}} = 5 U_{2,f_2}$ instead.) The combined
root-sum-square that SMPTE-type analyzers report is returned alongside
as `smpte`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples: the distortion ratios are ratios and come out unchanged, while `carrier_amplitude` and `sideband_amplitudes` carry the unit and so land in pascals when the record is calibrated. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `f_low` | Low modulating tone `f1`, in Hz (e.g. 60 Hz). |
| `f_high` | High carrier tone `f2`, in Hz (e.g. 7 kHz). |
| `window` | FFT window (default `'hann'`). |

**Returns:** A [`ModulationDistortionResult`](/phonometry/reference/api/electroacoustics/intermodulation/#modulationdistortionresult) with `d2`, `d3` and the `smpte` combined RMS.

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
| `d2` | Second-order modulation distortion $d_{\mathrm{m},2}$ (14.12.7.2 g): the *arithmetic* sum of the sideband amplitudes at $f_2 \pm f_1$ relative to the output amplitude at `f2`. |
| `d3` | Third-order modulation distortion $d_{\mathrm{m},3}$ (14.12.7.2 h): the arithmetic sum of the sidebands at $f_2 \pm 2 f_1$ relative to the output amplitude at `f2`. |
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
[`plot`](/phonometry/reference/api/electroacoustics/distortion/#harmonicdistortionresultplot).

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

## total_difference_frequency_distortion

```python
total_difference_frequency_distortion(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
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
| `signal` | Captured signal (1-D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: this is a ratio of product amplitudes drawn from the same record, so the factor divides out and the answer is the same calibrated or not. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `f1` | Lower tone, in Hz (default 8 kHz, per 14.12.10.2 b). |
| `f2` | Upper tone, in Hz (default 11.95 kHz, per 14.12.10.2 b). |
| `window` | FFT window (default `'hann'`). |

**Returns:** Total difference-frequency distortion, as a ratio.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |
