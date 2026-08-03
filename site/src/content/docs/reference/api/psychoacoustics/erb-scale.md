---
title: "psychoacoustics.erb_scale"
description: "The ERB_N scale: auditory-filter bandwidth and the Cam frequency scale."
sidebar:
  label: "erb_scale"
---

The ERB_N scale: auditory-filter bandwidth and the Cam frequency scale.

The cochlea behaves as a bank of overlapping band-pass **auditory filters**.
The width of the filter centred on a given frequency is summarised by its
**equivalent rectangular bandwidth** `ERB_N`: the bandwidth of the
rectangular filter that passes the same power and has the same peak response.
Glasberg and Moore (1990), fitting notched-noise data for young listeners at
moderate levels, give it as a straight line in frequency (Moore, *An
Introduction to the Psychology of Hearing* 6th ed., p. 76):

$$
\text{ERB}_N = 24.7 \, (4.37 F + 1)~\text{Hz}, \qquad F~\text{in kHz}
$$

Integrating $df / \text{ERB}_N(f)$ turns that into a frequency scale
whose unit is
one auditory-filter width, the **ERB_N number**, whose unit is called the
**Cam** (after Cambridge, following a suggestion by Hartmann):

$$
\text{ERB}_N\ \text{number} = 21.4 \log_{10}(4.37 F + 1)~\text{Cam}
$$

The Cam scale plays the same role as the Bark scale of Zwicker and Terhardt or
the mel scale of pitch: it is the axis on which masking patterns, excitation
patterns and specific loudness are naturally expressed. It differs from Bark
numerically, mostly below 500 Hz where the "old" critical-band function
flattens out but direct measurements show `ERB_N` continuing to shrink.

This module states the constants to the precision used by the ISO 532-2
implementation of the Moore-Glasberg loudness model
([`phonometry.psychoacoustics.loudness.moore_glasberg`](/phonometry/reference/api/psychoacoustics/moore-glasberg/)), which shares
them:

$$
\text{ERB}_N = 24.673 \, (0.004368 f + 1)~\text{Hz}, \qquad f~\text{in Hz}
$$

$$
\text{ERB}_N\ \text{number} = 21.366 \log_{10}(0.004368 f + 1)~\text{Cam}
$$

They are the same fit written with one more digit: the two forms agree to
better than 0.2 % over the whole audible range. The extra digits matter for
the round trip through [`frequency_from_cam`](/phonometry/reference/api/psychoacoustics/erb-scale/#frequency_from_cam), and they are what reproduce
the check value Moore prints on p. 77, that 1000 Hz corresponds to 15.59 Cam
(the two-significant-digit constants give 15.62).

Reference: Glasberg, B. R. and Moore, B. C. J. (1990), "Derivation of auditory
filter shapes from notched-noise data", *Hearing Research* 47, 103-138.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## CAM_C

*Constant* (`float`).

```python
CAM_C = 21.366
```

## cam_from_frequency

```python
cam_from_frequency(frequency: ArrayLike) -> np.ndarray | float
```

ERB_N number (Cam) of a frequency.

$i = 21.366 \log_{10}(0.004368 f + 1)$ (Glasberg and Moore 1990;
Moore 6th ed., p. 76, printed there as
$21.4 \log_{10}(4.37 F + 1)$ for `F` in kHz).
One Cam is one auditory-filter width.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, Hz (scalar or array, non-negative). |

**Returns:** The ERB_N number, Cam; a float for a scalar input.

## erb_bandwidth

```python
erb_bandwidth(frequency: ArrayLike) -> np.ndarray | float
```

Equivalent rectangular bandwidth of the auditory filter, Hz.

$\text{ERB}_N = 24.673 \, (0.004368 f + 1)$ (Glasberg and Moore
1990; Moore 6th ed., p. 76, printed there with the rounded constants
$24.7 \, (4.37 F + 1)$ for `F` in kHz).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Filter centre frequency `f`, Hz (scalar or array, non-negative). |

**Returns:** The bandwidth `ERB_N`, Hz; a float for a scalar input.

## ERB_C1

*Constant* (`float`).

```python
ERB_C1 = 24.673
```

## ERB_C2

*Constant* (`float`).

```python
ERB_C2 = 0.004368
```

## frequency_from_cam

```python
frequency_from_cam(cam: ArrayLike) -> np.ndarray | float
```

Frequency of an ERB_N number, the inverse of [`cam_from_frequency`](/phonometry/reference/api/psychoacoustics/erb-scale/#cam_from_frequency).

$f = (10^{i / 21.366} - 1) / 0.004368$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `cam` | ERB_N number `i`, Cam (scalar or array, non-negative). |

**Returns:** The frequency `f`, Hz; a float for a scalar input.
