---
title: "materials.absorbers.standing_wave"
description: "Standing-wave-ratio method for normal-incidence absorption and impedance."
sidebar:
  label: "standing_wave"
---

Standing-wave-ratio method for normal-incidence absorption and impedance.

**BS EN ISO 10534-1:2001**, the probe-traverse method of the impedance tube: a
loudspeaker drives one pure tone at a time, a probe microphone traverses the
tube and reads the maximum and minimum sound pressure levels of the standing
wave the specimen sets up, and the position of the first pressure minimum
fixes the phase. The reflection magnitude, phase, absorption coefficient and
normalised impedance follow from those two readings alone (Clause 5,
Eqs. (12)-(26)).

That is the whole method, and it is why it is one module: it consumes a
standing-wave ratio and a distance, and needs neither a measured transfer
function nor the complex wavenumber and air properties every broadband
reduction in the tube goes through. The quantities are the same
normal-incidence quantities the two-microphone method reports -
$\alpha = 1 - \lvert r\rvert^2$ is Eq. (9) here and Eq. (18) there -
measured one frequency at a time; see
[`impedance_tube`](/phonometry/reference/api/materials/impedance-tube/).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## standing_wave_absorption

```python
standing_wave_absorption(swr: ArrayLike) -> Real
```

Absorption coefficient from the standing-wave ratio (ISO 10534-1).

Combining $\alpha = 1 - |r|^2$ (Eq. (9)) with
$|r| = (s - 1)/(s + 1)$ (Eq. (14)) gives
$\alpha = 4s/(s + 1)^2$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `swr` | Standing-wave ratio `s` (>= 1). |

**Returns:** Absorption coefficient `alpha` in `[0, 1]`.

## standing_wave_normalized_impedance

```python
standing_wave_normalized_impedance(
    swr: ArrayLike,
    first_min_distance: ArrayLike,
    wavelength: ArrayLike,
) -> Complex
```

Normalised impedance from the standing wave (ISO 10534-1, Eqs. (24)-(26)).

$z = Z/Z_0 = (1 + r)/(1 - r)$; the real/imaginary split is
Eqs. (25)/(26).

**Parameters**

| Name | Description |
| :--- | :--- |
| `swr` | Standing-wave ratio `s` (>= 1). |
| `first_min_distance` | Distance `x_min1` to the first minimum, in metres. |
| `wavelength` | Wavelength `lambda0`, in metres. |

**Returns:** Normalised surface impedance `z` (complex).

## standing_wave_ratio_from_level

```python
standing_wave_ratio_from_level(level_difference: ArrayLike) -> Real
```

Standing-wave ratio from a level difference (ISO 10534-1, Eq. (15)).

$s = 10^{\Delta L / 20}$ with
$\Delta L = L_{\max} - L_{\min}$ in decibels.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level_difference` | Level difference $\Delta L = L_{\max} - L_{\min}$, in dB. |

**Returns:** Standing-wave ratio `s` (>= 1).

## standing_wave_reflection

```python
standing_wave_reflection(
    swr: ArrayLike,
    first_min_distance: ArrayLike,
    wavelength: ArrayLike,
) -> Complex
```

Complex reflection factor from the standing wave (ISO 10534-1, Eqs. (17)-(23)).

$r = |r| e^{j\phi}$ with $|r| = (s - 1)/(s + 1)$ (Eq. (14))
and the phase at the first pressure minimum
$\phi = \pi (4 x_{\text{min},1} / \lambda_0 - 1)$ (Eq. (20)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `swr` | Standing-wave ratio `s` (>= 1). |
| `first_min_distance` | Distance `x_min1` from the reference plane to the first pressure minimum (toward the source), in metres. |
| `wavelength` | Wavelength `lambda0`, in metres (Eq. (27)). |

**Returns:** Complex reflection factor `r`.

## standing_wave_reflection_magnitude

```python
standing_wave_reflection_magnitude(swr: ArrayLike) -> Real
```

Reflection magnitude from the standing-wave ratio (ISO 10534-1, Eq. (14)).

$|r| = (s - 1) / (s + 1)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `swr` | Standing-wave ratio `s` (>= 1). |

**Returns:** Reflection magnitude `|r|` in `[0, 1]`.
