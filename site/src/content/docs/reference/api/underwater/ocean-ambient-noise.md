---
title: "underwater.ocean_ambient_noise"
description: "Ocean ambient-noise spectrum levels (Wenz framework)."
sidebar:
  label: "ocean_ambient_noise"
---

Ocean ambient-noise spectrum levels (Wenz framework).

Deep-water ambient-noise **spectrum levels** (dB re 1 µPa²/Hz) from the two
physically grounded components of the Wenz curves:

* [`wind_noise_spectrum`](/phonometry/reference/api/underwater/ocean-ambient-noise/#wind_noise_spectrum) -- wind / sea-surface (Knudsen) noise via Wenz's
  "rule of fives",
  $\mathrm{NL} = 51.02 - (5/3) \cdot 10 (\lg f - \lg(U/5))$
  (`f` in kHz, `U` in knots; the historical 25 dB anchor is re 20 µPa and
  becomes $25 + 20 \lg(20)$ re 1 µPa), valid over roughly 500 Hz-5 kHz.
* [`thermal_noise_spectrum`](/phonometry/reference/api/underwater/ocean-ambient-noise/#thermal_noise_spectrum) -- the molecular thermal-noise limit (Mellen
  1952), $\langle p^2(f) \rangle = 4 \pi k T \rho f^2 / c$ (Pa²/Hz),
  dominant above ~50 kHz.

[`ocean_ambient_noise`](/phonometry/reference/api/underwater/ocean-ambient-noise/#ocean_ambient_noise) energy-sums the enabled components (and an optional
caller-supplied shipping spectrum) into a composite [`AmbientNoiseResult`](/phonometry/reference/api/underwater/ocean-ambient-noise/#ambientnoiseresult)
with a `.plot()` of the Wenz-style curves.

The low-frequency turbulence band and a built-in distant-shipping model are out
of scope: Wenz (1962) and Carey & Evans note these bands are strongly variable
and shipping-dependent, with no single fixed analytic parametrisation; a shipping
spectrum may be supplied by the caller. Source: Carey & Evans, *Ocean Ambient
Noise* (2011) -- the rule of fives (p. 2) and the thermal-noise derivation
(Appendix F).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AmbientNoiseResult

```python
AmbientNoiseResult(
    frequency: NDArray[np.float64],
    spectrum_level: NDArray[np.float64],
    wind: NDArray[np.float64],
    thermal: NDArray[np.float64],
    shipping: NDArray[np.float64] | None,
    wind_speed_knots: float,
)
```

Composite ambient-noise spectrum (Wenz framework).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequencies, in Hz. |
| `spectrum_level` | Composite spectrum level (energy sum of the enabled components), in dB re 1 µPa²/Hz. |
| `wind` | Wind-noise component per frequency, in dB re 1 µPa²/Hz. |
| `thermal` | Thermal-noise component per frequency, in dB re 1 µPa²/Hz. |
| `shipping` | Caller-supplied shipping component, or `None`. |
| `wind_speed_knots` | The wind speed used, in knots. |

### AmbientNoiseResult.plot()

```python
AmbientNoiseResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the composite spectrum and its components versus frequency.

## ocean_ambient_noise

```python
ocean_ambient_noise(
    frequency_hz: NDArray[np.float64] | list[float],
    *,
    wind_speed_knots: float,
    shipping: NDArray[np.float64] | list[float] | None = None,
    temperature: float = 16.85,
    density: float = 1025.0,
    sound_speed: float = 1500.0,
) -> AmbientNoiseResult
```

Composite deep-water ambient-noise spectrum (wind + thermal [+ shipping]).

Energy-sums the wind-noise (rule of fives) and thermal-noise (Mellen)
components, plus an optional caller-supplied shipping spectrum.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Frequencies, in Hz (1-D, strictly positive). |
| `wind_speed_knots` | Wind speed `U`, in knots. |
| `shipping` | Optional shipping-noise spectrum level per frequency, in dB re 1 µPa²/Hz (same length as `frequency_hz`), or `None`. |
| `temperature` | Water temperature, in degrees Celsius. |
| `density` | Water density, in kg/m³. |
| `sound_speed` | Sound speed, in m/s. |

**Returns:** An [`AmbientNoiseResult`](/phonometry/reference/api/underwater/ocean-ambient-noise/#ambientnoiseresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## thermal_noise_spectrum

```python
thermal_noise_spectrum(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    *,
    temperature: float = 16.85,
    density: float = 1025.0,
    sound_speed: float = 1500.0,
) -> NDArray[np.float64]
```

Molecular thermal-noise spectrum level (Mellen 1952), dB re 1 µPa²/Hz.

$\langle p^2(f) \rangle = 4 \pi k T \rho f^2 / c$ (Pa²/Hz); the
level is $10 \lg(\langle p^2 \rangle / p_0^2)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Frequency, in Hz (scalar or array). |
| `temperature` | Water temperature, in degrees Celsius (default 16.85 °C = 290 K). |
| `density` | Water density $\rho$, in kg/m³ (default 1025). |
| `sound_speed` | Sound speed `c`, in m/s (default 1500). |

**Returns:** Thermal-noise spectrum level per frequency, in dB re 1 µPa²/Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## wind_noise_spectrum

```python
wind_noise_spectrum(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    wind_speed_knots: float,
) -> NDArray[np.float64]
```

Wind / sea-surface noise spectrum level (Wenz rule of fives), dB re
1 µPa²/Hz.

$\mathrm{NL}(f, U) = 51.02 - (5/3) \cdot 10 (\lg f - \lg(U/5))$
with `f` in kHz and `U`
in knots: −5 dB per octave and +5 dB per doubling of wind speed about the
canonical anchor, which Wenz/Knudsen state as "25 dB (5 × 5)" at 1 kHz for
5 knots **re 0.0002 dyn/cm² (20 µPa)**, i.e.
$25 + 20 \lg(20) \approx 51.02$ dB
once referenced to the ISO 18405 1 µPa. Valid over roughly 500 Hz-5 kHz
and winds of 2.5-40 knots (the stated range of the wind-doubling law);
outside both the formula extrapolates.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Frequency, in Hz (scalar or array). |
| `wind_speed_knots` | Wind speed `U`, in knots. A calm sea (`0`) has no wind-driven noise, returning `-inf` (zero contribution in the energy sum). |

**Returns:** Wind-noise spectrum level per frequency, in dB re 1 µPa²/Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid (a negative wind speed). |
