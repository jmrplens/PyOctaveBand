---
title: "underwater.acoustics"
description: "Underwater-acoustics reference levels (ISO 18405:2017)."
sidebar:
  label: "acoustics"
---

Underwater-acoustics reference levels (ISO 18405:2017).

Underwater sound levels use a reference pressure of **1 µPa** (not the 20 µPa of
airborne acoustics) and a reference sound exposure of **1 µPa²·s**. This module
realises the ISO 18405 terminology as the shared level primitives used by the
ship-radiated-noise and pile-driving modules:

* [`sound_pressure_level`](/phonometry/reference/api/underwater/acoustics/#sound_pressure_level) -- the mean-square sound pressure level,
  `SPL = 10·lg(<p²>/p₀²)` dB re 1 µPa.
* [`sound_exposure_level`](/phonometry/reference/api/underwater/acoustics/#sound_exposure_level) -- the time-integrated exposure level,
  `SEL = 10·lg(∫p² dt / E₀)` dB re 1 µPa²·s.
* [`peak_sound_pressure_level`](/phonometry/reference/api/underwater/acoustics/#peak_sound_pressure_level) -- the zero-to-peak level
  `20·lg(max|p|/p₀)` dB re 1 µPa.

[`underwater_to_in_air_spl`](/phonometry/reference/api/underwater/acoustics/#underwater_to_in_air_spl) / [`in_air_to_underwater_spl`](/phonometry/reference/api/underwater/acoustics/#in_air_to_underwater_spl) convert a
level between the two reference pressures (a `20·lg(20) ≈ 26.02` dB reference
change, **not** an energy/intensity equivalence, which would additionally
involve the media impedances). For background-noise subtraction of a measured
level, reuse the ISO 3744 `background_noise_correction` (`K1`) helper.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## in_air_to_underwater_spl

```python
in_air_to_underwater_spl(level: float) -> float
```

Re-reference an in-air SPL (re 20 µPa) to the underwater 1 µPa reference.

Adds `20·lg(20) ≈ 26.02` dB (a reference-pressure change only; see
[`underwater_to_in_air_spl`](/phonometry/reference/api/underwater/acoustics/#underwater_to_in_air_spl)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | Level in dB re 20 µPa. |

**Returns:** The same pressure expressed in dB re 1 µPa.

## peak_sound_pressure_level

```python
peak_sound_pressure_level(
    pressure: NDArray[np.float64] | list[float],
    *,
    reference: float = 1e-06,
) -> float
```

Zero-to-peak sound pressure level (ISO 18406 6.4.2.1.3).

`L_p,pk = 20·lg(max|p|/p₀)` dB re 1 µPa.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure` | Sound-pressure time series (1-D), in Pa. |
| `reference` | Reference pressure `p₀`, in Pa (default 1 µPa). |

**Returns:** Peak sound pressure level, in dB re the reference.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the signal is invalid or is all zero. |

## sound_exposure_level

```python
sound_exposure_level(
    pressure: NDArray[np.float64] | list[float],
    fs: float,
    *,
    reference: float = 1e-12,
) -> float
```

Sound exposure level (ISO 18405 / ISO 18406 Formulae 3-4).

`SEL = 10·lg(E/E₀)` dB re 1 µPa²·s, with the sound exposure
`E = ∫p² dt ≈ (1/f_s)·Σ p²` over the record.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure` | Sound-pressure time series (1-D), in Pa. |
| `fs` | Sample rate, in Hz. |
| `reference` | Reference exposure `E₀`, in Pa²·s (default 1 µPa²·s). |

**Returns:** Sound exposure level, in dB re the reference.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or the signal has no energy. |

## sound_pressure_level

```python
sound_pressure_level(
    pressure: NDArray[np.float64] | list[float],
    *,
    reference: float = 1e-06,
) -> float
```

Mean-square sound pressure level (ISO 18405 / ISO 18406 Formula 7).

`SPL = 10·lg(<p²>/p₀²)` dB, with `p` in pascals and the underwater
reference `p₀ = 1 µPa` by default.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure` | Sound-pressure time series (1-D), in Pa. |
| `reference` | Reference pressure `p₀`, in Pa (default 1 µPa). |

**Returns:** Sound pressure level, in dB re the reference.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the signal is invalid or has no energy. |

## UNDERWATER_REFERENCE_EXPOSURE

*Constant* (`float`).

```python
UNDERWATER_REFERENCE_EXPOSURE = 1e-12
```

## UNDERWATER_REFERENCE_PRESSURE

*Constant* (`float`).

```python
UNDERWATER_REFERENCE_PRESSURE = 1e-06
```

## underwater_to_in_air_spl

```python
underwater_to_in_air_spl(level: float) -> float
```

Re-reference an underwater SPL (re 1 µPa) to the in-air 20 µPa reference.

Subtracts `20·lg(20) ≈ 26.02` dB. This is a **reference-pressure change
only** -- it is not the in-water/in-air intensity equivalence, which also
involves the media characteristic impedances.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | Level in dB re 1 µPa. |

**Returns:** The same pressure expressed in dB re 20 µPa.
