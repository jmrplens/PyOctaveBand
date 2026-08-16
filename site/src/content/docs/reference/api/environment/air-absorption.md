---
title: "environment.propagation.air_absorption"
description: "Atmospheric absorption of sound: ISO 9613-1:1993."
sidebar:
  label: "air_absorption"
---

Atmospheric absorption of sound: ISO 9613-1:1993.

The attenuation of a pure tone propagating through the atmosphere is governed by
a pure-tone attenuation coefficient `alpha` (in dB/m) that depends on frequency,
temperature, humidity and pressure through the vibrational relaxation of the
oxygen and nitrogen molecules plus classical and rotational losses
(ISO 9613-1:1993, clause 6).

The attenuation coefficient, in decibels per metre (ISO 9613-1:1993):

$$
\alpha = 8.686 f^2 \left\{ 1.84 \times 10^{-11} \left( \frac{p_\mathrm{a}}{p_\mathrm{r}} \right)^{-1} \left( \frac{T}{T_0} \right)^{1/2} + \left( \frac{T}{T_0} \right)^{-5/2} \left[ 0.01275 \, e^{-2239.1/T} \left( f_\mathrm{rO} + \frac{f^2}{f_\mathrm{rO}} \right)^{-1} + 0.1068 \, e^{-3352.0/T} \left( f_\mathrm{rN} + \frac{f^2}{f_\mathrm{rN}} \right)^{-1} \right] \right\} \tag{Eq. 5}
$$

with the oxygen and nitrogen relaxation frequencies (ISO 9613-1:1993):

$$
f_\mathrm{rO} = \frac{p_\mathrm{a}}{p_\mathrm{r}} \left[ 24 + 4.04 \times 10^{4} \, h \, \frac{0.02 + h}{0.391 + h} \right] \tag{Eq. 3}
$$

$$
f_\mathrm{rN} = \frac{p_\mathrm{a}}{p_\mathrm{r}} \left( \frac{T}{T_0} \right)^{-1/2} \left\{ 9 + 280 \, h \exp\!\left[ -4.170 \left( \left( \frac{T}{T_0} \right)^{-1/3} - 1 \right) \right] \right\} \tag{Eq. 4}
$$

Here `T` is the ambient temperature (K), $T_0 = 293.15$ K and
$p_\mathrm{r} = 101.325$ kPa are the reference conditions (ISO 9613-1:1993,
clause 4.2), `pa` is the ambient pressure (kPa) and `h` is the molar
concentration of water vapour as a percentage, obtained from the relative
humidity by the psychrometric conversion (ISO 9613-1:1993, clause 6.4 /
Annex B):

$$
h = h_\mathrm{r} \, \frac{p_\mathrm{sat}/p_\mathrm{r}}{p_\mathrm{a}/p_\mathrm{r}}
$$

$$
\frac{p_\mathrm{sat}}{p_\mathrm{r}} = 10^{-6.8346 \, (T_{01}/T)^{1.261} + 4.6151}, \qquad T_{01} = 273.16~\text{K}
$$

with `hr` the relative humidity (%) and `T01` the triple-point temperature of
water.

Table 1 of ISO 9613-1:1993 tabulates `alpha` (in dB/km) at the reference
pressure for a grid of temperature, relative humidity and one-third-octave
frequency; its rows are labelled with the ISO 266 preferred frequencies but
the coefficients are computed at the exact midband frequencies (Note 5)
$f_\mathrm{m} = 1000 \cdot 10^{k/10}$, `k` integer. Pass
`exact_midband=True` to snap the requested frequencies onto that grid and
reproduce Table 1 exactly.

This module closes the loop with [`sound_absorption`](/phonometry/reference/api/materials/sound-absorption/) (ISO 354),
whose air power-attenuation coefficient `m` (1/m) is defined only through
the ISO 9613-1 `alpha` via $m = \alpha / (10 \log_{10} e)$.
[`air_attenuation_m`](/phonometry/reference/api/environment/air-absorption/#air_attenuation_m) returns that `m` directly.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## air_attenuation

```python
air_attenuation(
    frequencies: ArrayLike,
    temperature: float = 20.0,
    relative_humidity: float = 50.0,
    pressure: float = 101.325,
    *,
    exact_midband: bool = False,
) -> NDArray[np.float64]
```

Pure-tone atmospheric attenuation coefficient (ISO 9613-1, Eq. (5)).

Evaluates `alpha` in decibels per metre from the oxygen and nitrogen
relaxation frequencies (Eq. (3)/(4)) and the classical, rotational and
vibrational absorption terms (Eq. (5)). Fully vectorized over
`frequencies`; `temperature`, `relative_humidity` and `pressure` are
scalars.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency or frequencies `f`, in hertz (array-like). |
| `temperature` | Ambient air temperature, in degrees Celsius (default 20 degC, i.e. the reference `T0`). A value outside the -20..+50 degC tabulated range emits an [`AtmosphericAbsorptionWarning`](/phonometry/reference/api/environment/air-absorption/#atmosphericabsorptionwarning); a value at or below absolute zero raises `ValueError`. |
| `relative_humidity` | Relative humidity, in percent, with respect to saturation over liquid water (default 50 %). Outside 10..100 % emits an [`AtmosphericAbsorptionWarning`](/phonometry/reference/api/environment/air-absorption/#atmosphericabsorptionwarning); outside [0, 100] % raises `ValueError`. |
| `pressure` | Ambient atmospheric pressure `pa`, in kilopascals (default 101.325 kPa = one standard atmosphere = `pr`). Above 200 kPa emits an [`AtmosphericAbsorptionWarning`](/phonometry/reference/api/environment/air-absorption/#atmosphericabsorptionwarning); non-positive raises `ValueError`. |
| `exact_midband` | When `True`, each requested frequency is snapped to the nearest exact one-third-octave midband $f_\mathrm{m} = 1000 \cdot 10^{k/10}$ (Eq. (6)) before evaluation, reproducing the frequencies used for Table 1 (Note 5). Default `False` (use `frequencies` verbatim). |

**Returns:** Attenuation coefficient `alpha`, in dB/m, with the shape of `frequencies`.

:::note
ISO 354:2003 defers its air power-attenuation coefficient `m` (1/m)
entirely to this `alpha` via $m = \alpha / (10 \log_{10} e)$. Use
[`air_attenuation_m`](/phonometry/reference/api/environment/air-absorption/#air_attenuation_m) to obtain that `m` for
[`absorption_area`](/phonometry/reference/api/materials/sound-absorption/#absorption_area) /
[`absorption_coefficient`](/phonometry/reference/api/materials/sound-absorption/#absorption_coefficient).
:::

## air_attenuation_m

```python
air_attenuation_m(
    frequencies: ArrayLike,
    temperature: float = 20.0,
    relative_humidity: float = 50.0,
    pressure: float = 101.325,
    *,
    exact_midband: bool = False,
) -> NDArray[np.float64]
```

ISO 354 air power-attenuation coefficient `m` (1/m) from conditions.

Convenience composition of [`air_attenuation`](/phonometry/reference/api/environment/air-absorption/#air_attenuation) (ISO 9613-1 `alpha` in
dB/m) with the ISO 354:2003 (8.1.2.1) conversion
$m = \alpha / (10 \log_{10} e)$
(via [`attenuation_from_alpha`](/phonometry/reference/api/materials/sound-absorption/#attenuation_from_alpha)). It lets an
ISO 354 caller feed real atmospheric conditions into
[`absorption_area`](/phonometry/reference/api/materials/sound-absorption/#absorption_area) /
[`absorption_coefficient`](/phonometry/reference/api/materials/sound-absorption/#absorption_coefficient) instead of
hand-entering `m`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency or frequencies `f`, in hertz (array-like). |
| `temperature` | Ambient air temperature, in degrees Celsius (default 20). |
| `relative_humidity` | Relative humidity, in percent (default 50). |
| `pressure` | Ambient atmospheric pressure, in kilopascals (default 101.325). |
| `exact_midband` | Snap frequencies to exact midbands; see [`air_attenuation`](/phonometry/reference/api/environment/air-absorption/#air_attenuation). |

**Returns:** Power attenuation coefficient `m`, in 1/m, with the shape of `frequencies`.

## atmospheric_attenuation

```python
atmospheric_attenuation(
    frequencies: ArrayLike,
    temperature: float = 20.0,
    relative_humidity: float = 50.0,
    pressure: float = 101.325,
    *,
    exact_midband: bool = False,
    distance: float | None = None,
) -> AtmosphericAttenuation
```

Build a plottable ISO 9613-1 atmospheric-attenuation curve.

Evaluates [`air_attenuation`](/phonometry/reference/api/environment/air-absorption/#air_attenuation) at `frequencies` for the given
atmospheric conditions and bundles the result into an
[`AtmosphericAttenuation`](/phonometry/reference/api/environment/air-absorption/#atmosphericattenuation) that exposes `.plot()`. The maths is
unchanged; this is a thin, plottable wrapper around the existing function
(the same warnings and the same `ValueError` cases apply).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency or frequencies `f`, in hertz (array-like). |
| `temperature` | Ambient air temperature, in degrees Celsius (default 20). |
| `relative_humidity` | Relative humidity, in percent (default 50). |
| `pressure` | Ambient atmospheric pressure, in kilopascals (default 101.325 kPa, one standard atmosphere). |
| `exact_midband` | Snap the frequencies to the exact one-third-octave midbands $f_\mathrm{m} = 1000 \cdot 10^{k/10}$ (Eq. (6)) before evaluation; see [`air_attenuation`](/phonometry/reference/api/environment/air-absorption/#air_attenuation). When `True` the stored `frequencies` are the snapped midbands the coefficient was computed at. |
| `distance` | Optional propagation distance `d`, in metres. When given, the result's [`total_attenuation`](/phonometry/reference/api/environment/air-absorption/#atmosphericattenuationtotal_attenuation) returns the total attenuation $A = \alpha d$ over that distance (ISO 9613-2 Eq. (8)). Must be finite and non-negative. |

**Returns:** A frozen [`AtmosphericAttenuation`](/phonometry/reference/api/environment/air-absorption/#atmosphericattenuation).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `distance` is negative or non-finite (NaN/inf); the check lives on [`AtmosphericAttenuation`](/phonometry/reference/api/environment/air-absorption/#atmosphericattenuation) so it also guards direct construction. |

## AtmosphericAbsorptionWarning

Advisory for ISO 9613-1 inputs outside the tabulated/validity ranges.

## AtmosphericAttenuation

```python
AtmosphericAttenuation(
    frequencies: NDArray[np.float64],
    attenuation_coefficient: NDArray[np.float64],
    temperature: float,
    relative_humidity: float,
    pressure: float,
    distance: float | None = None,
)
```

A pure-tone atmospheric attenuation curve (ISO 9613-1:1993).

Bundles the ISO 9613-1 attenuation coefficient `alpha` (Eq. (5)) over a
frequency grid with the atmospheric conditions it was evaluated for, so the
classic `alpha` versus frequency curve can be drawn with `plot`.
Build it with [`atmospheric_attenuation`](/phonometry/reference/api/environment/air-absorption/#atmospheric_attenuation); the frozen instance is a thin,
plottable wrapper and re-runs none of the maths.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f` the coefficient is evaluated at, in Hz (the exact one-third-octave midbands when `exact_midband` was used). |
| `attenuation_coefficient` | Pure-tone attenuation coefficient `alpha`, per frequency, in decibels per metre (Table 1 prints dB/km, i.e. $\times 1000$). |
| `temperature` | Ambient air temperature, in degrees Celsius. |
| `relative_humidity` | Relative humidity, in percent. |
| `pressure` | Ambient atmospheric pressure `pa`, in kilopascals. |
| `distance` | Propagation distance `d`, in metres, or `None` when the result carries only the coefficient. When given, `total_attenuation` returns the total attenuation $A = \alpha d$ over that distance. |

### AtmosphericAttenuation.plot()

```python
AtmosphericAttenuation.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the attenuation coefficient `alpha` versus frequency.

Draws `alpha` (in dB/km, as Table 1 tabulates it) on a logarithmic
frequency axis, the classic ISO 9613-1 curve for the stored atmospheric
conditions. Requires matplotlib (`pip install phonometry[plot]`);
returns the `Axes` and never calls `plt.show`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the `alpha` curve `plot` call. |

**Returns:** The axes.

### AtmosphericAttenuation.total_attenuation

*property*

Total attenuation $A = \alpha d$ over `distance`.

The pure-tone attenuation `alpha` (dB/m) accumulated over the
propagation distance `d` (m), per frequency, in decibels; this is the
ISO 9613-2:1996 `Aatm` (Eq. (8)) form. `None` when no
`distance` was supplied.
