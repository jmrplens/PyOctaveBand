---
title: "filters.core"
description: "Core processing logic and FilterBank class for phonometry."
sidebar:
  label: "core"
---

Core processing logic and FilterBank class for phonometry.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## BlockProcessing

```python
BlockProcessing(stateful: bool = False, steady_ic: bool = False)
```

How the bank carries its filter state from one block to the next.

**Attributes**

| Name | Description |
| :--- | :--- |
| `stateful` | If True, carry filter state between calls. Useful for block processing (default False). |
| `steady_ic` | If True, calculate steady state initial conditions for filter (default False). |

## FilterBankWarning

Warns about fractional-octave filter-bank processing pitfalls.

## FilterDesign

```python
FilterDesign(
    filter_type: str = 'butter',
    ripple: float = 0.1,
    attenuation: float = 72.0,
    resample: bool = True,
)
```

How the band-pass filters of a bank are designed.

The defaults are the design used everywhere in the library: Butterworth
sections, evaluated band by band on a decimated sample rate.

**Attributes**

| Name | Description |
| :--- | :--- |
| `filter_type` | Type of filter ('butter', 'cheby1', 'cheby2', 'ellip', 'bessel'). Only `butter` meets IEC 61260-1:2014 class 1 with the default parameters (`cheby2` also does once `attenuation` >= 70 dB, see below); `cheby1`/`ellip`/`bessel` fail on passband ripple or roll-off regardless of parameters (default 'butter'). |
| `ripple` | Passband ripple in dB, for cheby1 and ellip (default 0.1). |
| `attenuation` | Stopband attenuation in dB (default 72.0). For the `cheby2` filter scipy pins the equiripple deep-stopband floor at exactly this value, so it must be >= 70 dB for the bank to meet the IEC 61260-1:2014 class 1 deep-stopband limit (Omega >= G^4). The 72 dB default clears class 1 with the same +0.400 dB passband margin as `butter`. |
| `resample` | If True, resampling is performed: each band is filtered on a decimated sample rate (default True). |

## LevelCalibration

```python
LevelCalibration(factor: float = 1.0, dbfs: bool = False)
```

How the energy in a band becomes a level reading.

**Attributes**

| Name | Description |
| :--- | :--- |
| `factor` | Calibration factor for SPL calculation: multiplies the digital amplitude to obtain pascals (default 1.0). |
| `dbfs` | If True, calculate SPL in dBFS, where 0 dB is a full-scale RMS of 1.0, instead of dB SPL re 20 uPa; `factor` is then not applied (default False). |

## octave_filter

```python
octave_filter(
    x: list[float] | np.ndarray,
    fs: int,
    fraction: float = 1,
    order: int = 6,
    limits: list[float] | None = None,
    *,
    sigbands: bool = False,
    detrend: bool = True,
    mode: str = 'rms',
    nominal: bool = False,
    design: FilterDesign = ...,
    calibration: LevelCalibration = ...,
    response_plot: ResponsePlot = ...,
) -> tuple[np.ndarray, list[float]] | tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[float], list[np.ndarray]] | tuple[np.ndarray, list[str], list[np.ndarray]]
```

Filter a signal with octave or fractional octave filter bank.

This method uses a filter bank with Second-Order Sections (SOS) coefficients.
To obtain the correct coefficients, automatic subsampling is applied to the
signal in each filtered band.

Multichannel support: If x is 2D (channels, samples), each channel is filtered.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | (*Union[List[float], np.ndarray]*) Input signal (1D array or 2D array [channels, samples]). |
| `fs` | (*int*) Sample rate in Hz. |
| `fraction` | (*float*) Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1, 2/3-octave b=1.5. Default: 1. |
| `order` | (*int*) Order of the filter. Default: 6. |
| `limits` | (*Optional[List[float]]*) Minimum and maximum limit frequencies [f_min, f_max]. Default [12, 20000]. |
| `sigbands` | (*bool*) If True, also return the signal in the time domain divided into bands. |
| `detrend` | (*bool*) If True, remove DC offset before filtering. Default: True. |
| `mode` | 'rms' or 'peak'. Default: 'rms'. |
| `nominal` | If True, return IEC 61260-1 nominal frequency labels (List[str]) instead of exact floats. |
| `design` | How the band filters are designed: family, ripple, stopband attenuation and multirate decimation ([`FilterDesign`](/phonometry/reference/api/filters/core/#filterdesign)). The default 'butter' family is the only one that meets IEC 61260-1 class 1 with the default parameters; for `cheby2` scipy pins the deep-stopband floor at exactly `attenuation`, so it must be >= 70 dB to clear the class 1 limit (matches [`OctaveFilterBank`](/phonometry/reference/api/filters/core/#octavefilterbank)). |
| `calibration` | How band energy becomes a level: calibration factor and dBFS switch ([`LevelCalibration`](/phonometry/reference/api/filters/core/#levelcalibration)). |
| `response_plot` | Whether to show or save the filter response plot ([`ResponsePlot`](/phonometry/reference/api/filters/core/#responseplot)). Plotting bypasses the design cache. |

**Returns:** A tuple containing (SPL_array, Frequencies_list) or (SPL_array, Frequencies_list, signals). When *nominal=True*, the frequency list contains `List[str]` labels instead of floats. (*Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[float], List[np.ndarray]], Tuple[np.ndarray, List[str], List[np.ndarray]]]*)

## OctaveFilterBank

```python
OctaveFilterBank(
    fs: int,
    fraction: float = 1,
    order: int = 6,
    limits: list[float] | None = None,
    *,
    design: FilterDesign = ...,
    calibration: LevelCalibration = ...,
    block_processing: BlockProcessing = ...,
    response_plot: ResponsePlot = ...,
)
```

A class-based representation of an Octave Filter Bank.
Allows for pre-calculating and reusing filter coefficients.

Initialize the Octave Filter Bank.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate in Hz. |
| `fraction` | Bandwidth fraction (e.g., 1 for octave, 3 for 1/3 octave). |
| `order` | Filter order. |
| `limits` | Frequency limits [f_min, f_max]. |
| `design` | How the band filters are designed: family, ripple, stopband attenuation and multirate decimation ([`FilterDesign`](/phonometry/reference/api/filters/core/#filterdesign)). Only `butter` meets IEC 61260-1:2014 class 1 with the default parameters (`cheby2` also does once `attenuation` >= 70 dB); `cheby1`/`ellip`/`bessel` fail on passband ripple or roll-off regardless of parameters. |
| `calibration` | How band energy becomes a level: calibration factor and dBFS switch ([`LevelCalibration`](/phonometry/reference/api/filters/core/#levelcalibration)). |
| `block_processing` | Whether the bank carries its filter state between calls, and how that state starts ([`BlockProcessing`](/phonometry/reference/api/filters/core/#blockprocessing)). |
| `response_plot` | Whether to show or save the filter response plot drawn while designing the bank ([`ResponsePlot`](/phonometry/reference/api/filters/core/#responseplot)). |

### OctaveFilterBank.filter()

```python
OctaveFilterBank.filter(
    x: list[float] | np.ndarray,
    sigbands: bool = False,
    mode: str = 'rms',
    detrend: bool = True,
    calculate_level: bool = True,
    nominal: bool = False,
    zero_phase: bool = False,
) -> tuple[np.ndarray | None, list[float] | list[str]] | tuple[np.ndarray | None, list[float] | list[str], list[np.ndarray]]
```

Apply the pre-designed filter bank to a signal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D array or 2D array [channels, samples]). |
| `sigbands` | If True, also return the signal in the time domain divided into bands. |
| `mode` | 'rms' for energy-based level, 'peak' for peak-holding level. Note: 'peak' includes the filter's onset transient; a tone that starts abruptly can overshoot by ~1 dB. For steady signals, discard the first ~5/f_low seconds or use longer signals. |
| `detrend` | If True, remove DC offset from signal before filtering (Default: True). |
| `calculate_level` | If True, calculate SPL. |
| `nominal` | If True, return IEC 61260-1 nominal frequency labels (List[str]) instead of exact floats. |
| `zero_phase` | If True, filter with `sosfiltfilt` (forward-backward): no group delay, but the effective stopband attenuation doubles and the effective passband narrows. The narrowing lowers the measured broadband band level by about 0.2 to 0.3 dB per band relative to forward filtering (a pure in-band tone is unaffected, since it sits where both passes are ~0 dB). Prefer forward filtering when the absolute band SPL must match single-pass conventions; use zero-phase when preserving the temporal envelope matters (e.g. reverberation decay, ISO 3382-2 Clause 7.3). Offline analysis only; incompatible with stateful mode. |

**Returns:** A tuple containing (SPL_array, Frequencies_list) or (SPL_array, Frequencies_list, signals).

### OctaveFilterBank.spectrogram()

```python
OctaveFilterBank.spectrogram(
    x: list[float] | np.ndarray,
    window_time: float = 0.125,
    overlap: float = 0.5,
    mode: str = 'rms',
    detrend: bool = True,
    zero_phase: bool = False,
) -> tuple[np.ndarray, list[float], np.ndarray]
```

Short-time fractional-octave analysis: level per band over time.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D array or 2D array [channels, samples]). |
| `window_time` | Analysis window length in seconds. |
| `overlap` | Window overlap fraction in [0, 1). |
| `mode` | 'rms' or 'peak' (per window). |
| `detrend` | If True, remove DC offset before filtering. |
| `zero_phase` | If True, filter bands forward-backward so their group delays don't skew the frames (offline analysis only). |

**Returns:** Tuple (levels, freq, times). `levels` has shape (num_bands, num_frames) for 1D input and (channels, num_bands, num_frames) for 2D input; `times` holds each window's center in seconds.

## ResponsePlot

```python
ResponsePlot(show: bool = False, file: str | None = None)
```

The filter-response plot drawn while the bank is being designed.

**Attributes**

| Name | Description |
| :--- | :--- |
| `show` | If True, show the filter response plot (default False). |
| `file` | Path to save the filter response plot (default None). |
