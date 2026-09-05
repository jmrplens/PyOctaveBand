---
title: "electroacoustics.noise_measurements"
description: "Noise measurements of audio equipment (AES17-2015 6.4)."
sidebar:
  label: "noise_measurements"
---

Noise measurements of audio equipment (AES17-2015 6.4).

The two levels that say how quiet a device is, both measured through the
AES17 measurement chain -- the standard notch (5.2.8) where a test tone has
to be removed, the CCIR-RMS weighting (5.2.7, the ITU-R BS.468-4 curve with
the flat -5,63 dB gain that puts unity at 2 kHz) and the 20 Hz to 20 kHz
band (5.2.5):

* **Dynamic range** (6.4.1), the ratio of the maximum output level to the
  weighted noise-plus-distortion residual left when the device is driven
  with a 997 Hz sine 60 dB below full scale -- what the standard's own note
  also calls the signal-to-noise ratio, since it includes every harmonic,
  inharmonic and noise component.
* **Idle channel noise** (6.4.2), the weighted output level, relative to
  full scale, of a device driven with no signal at all.

They share the notch, the weighting and the band with the distortion
metrics of [`phonometry.electroacoustics.distortion`](/phonometry/reference/api/electroacoustics/distortion/), and differ from
them in what is being asked: not how much of the output is not the tone,
but how much output there is when there should be none.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## dynamic_range

```python
dynamic_range(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
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
\mathrm{DR} = 20 \log_{10}\left( (\text{full\_scale} / \sqrt{2}) / V_\text{residual,CCIR-RMS} \right)
$$

It includes all harmonic, inharmonic and noise components and is also
known as the signal-to-noise ratio (6.4.1 note).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured output of the device under test (1-D), scaled so that `full_scale` is the digital full-scale peak amplitude. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) for the rate, but a calibration factor it carries is deliberately **not** applied: this quantity is referenced to digital full scale, not to 20 uPa, so scaling the samples to pascals would move the reading by `20 lg(factor)` under a full-scale name. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `fundamental` | Test frequency, in Hz. `None` reads it off the captured signal's own spectrum rather than assuming the 997 Hz the standard prescribes, so a capture that drifted is still notched where its fundamental actually is. |
| `notch_q` | Effective notch quality factor (AES17 5.2.8: 1.2..3; default 2.0). |
| `bandwidth` | AES17 measurement bandwidth, in Hz (default 20 kHz; `None` measures the full Nyquist band). |
| `full_scale` | Digital full-scale peak amplitude (default 1.0). |
| `window` | FFT window used only for fundamental auto-detection. |

**Returns:** The dynamic range, in dB CCIR-RMS (a positive number).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or `notch_q` is out of range. |

## idle_channel_noise

```python
idle_channel_noise(
    signal: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
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
L_\text{idle} = 20 \log_{10}\left( V_\text{idle,CCIR-RMS} / (\text{full\_scale} / \sqrt{2}) \right)
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Captured idle output of the device under test (1-D), scaled so that `full_scale` is the digital full-scale peak amplitude. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) for the rate, but a calibration factor it carries is deliberately **not** applied: this quantity is referenced to digital full scale, not to 20 uPa, so scaling the samples to pascals would move the reading by `20 lg(factor)` under a full-scale name. |
| `fs` | Sample rate, in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `bandwidth` | AES17 measurement bandwidth, in Hz (default 20 kHz; `None` measures the full Nyquist band). |
| `full_scale` | Digital full-scale peak amplitude (default 1.0). |

**Returns:** The idle channel noise level, in dBFS CCIR-RMS (a negative number for any real device).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |
