---
title: "underwater.weston_regimes"
description: "Weston's shallow-water propagation regimes (flux theory)."
sidebar:
  label: "weston_regimes"
---

Weston's shallow-water propagation regimes (flux theory).

A source in a shallow-water waveguide loses energy in four successive range
regimes, each with its own power law. The boundaries between them follow from
the seabed reflectivity alone, which makes the set an inexpensive analytic
reference for any numerical propagation model:

* **spherical spreading** -- `F = 1/r²` (`20·lg r`), while the sound has not
  yet felt the boundaries;
* **cylindrical spreading** -- `F = 2·ψc/(r·H)` (`10·lg r`), once the energy
  is confined to a cylinder of height `H` and only rays within the critical
  angle `ψc` survive;
* **mode stripping** -- `F = (π/(η·H))^½ · r^−3/2` (`15·lg r`), once the
  accumulated reflection loss has eroded the steep paths;
* **single mode** -- an exponential decay dominated by the lowest-order mode.

Everything here is implemented clean-room from Ainslie, *Principles of Sonar
Performance Modelling* (Springer 2010), §9.1.1.2 (printed pp. 452-458):
Equations (9.42) to (9.61) and the seabed properties of Table 9.1
([`WESTON_SEABEDS`](/phonometry/reference/api/underwater/weston-regimes/#weston_seabeds)). The quantity computed is Ainslie's **propagation
factor** `F` (units m⁻²), reported as the propagation loss
`PL = −10·lg F` dB re 1 m², which equals the usual transmission loss for a
point source in free water.

The regime formulae are energy-flux (incoherent) results: they describe the
range-averaged field, not its modal interference. That is exactly what makes
them a usable cross-check for [`phonometry.underwater.numerical_propagation`](/phonometry/reference/api/underwater/numerical-propagation/)
-- the range average of a normal-mode or parabolic-equation field over many
interference cycles converges on the cylindrical-spreading law, with
`ψc = π/2` for a totally reflecting (pressure-release) bottom.

:::note
Ainslie's Equation (9.57) for the mode-stripping/single-mode transition is
printed as `r_MS ≈ k²·He³/(9·η)`. Carrying out the derivation the
accompanying text prescribes -- "equating `θ_n` and `θ_eff` with
`n = 3/2`" -- with the two equations exactly as they are printed, namely
`θ_eff = (π·H/(4·η·r))^½` (Equation 9.47, with the **true** water depth
`H`) and `θ_n = n·π/(k·He)` (Equation 9.56, with the **effective**
depth `He`), gives `r_MS = k²·He²·H/(9·π·η)` instead. The printed form
is larger by `π·He/H`. This module implements the derivation-consistent
value, which also keeps `θ_eff` defined with `H` everywhere it is used
(the composite loss below evaluates Equation 9.47 the same way), and
records the discrepancy in `docs/ERRATA.md`.
:::

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## critical_grazing_angle

```python
critical_grazing_angle(sound_speed_ratio: float) -> float
```

Critical grazing angle `ψc = arccos(c_w/c_sed)`, in radians.

A seabed slower than the water (`c_sed ≤ c_w`, e.g. mud) has **no**
critical angle; the function then returns `0`, which correctly switches
the reflection-loss gradient to the refracting-sediment branch of
[`reflection_loss_gradient`](/phonometry/reference/api/underwater/weston-regimes/#reflection_loss_gradient).

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_speed_ratio` | `c_sed/c_w`, dimensionless and positive. |

**Returns:** The critical grazing angle, in radians (`0` if none exists).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the ratio is not positive and finite. |

## effective_depth

```python
effective_depth(
    water_depth: float,
    frequency_hz: float,
    *,
    seabed: str | WestonSeabed = 'sand',
    sound_speed: float = 1500.0,
) -> float
```

Weston effective water depth `He` (Ainslie Eq. 9.55), in metres.

`He = H + (ρ_sed/ρ_w)/((ω/c_w)·sin ψc)`: the depth at which a
pressure-release boundary appears to lie, a short distance below the true
seabed. Only meaningful for a seabed with a critical angle.

**Parameters**

| Name | Description |
| :--- | :--- |
| `water_depth` | Water-column depth `H`, in metres. |
| `frequency_hz` | Acoustic frequency, in Hz. |
| `seabed` | `"sand"`, `"mud"` or a [`WestonSeabed`](/phonometry/reference/api/underwater/weston-regimes/#westonseabed). |
| `sound_speed` | Water sound speed `c_w`, in m/s. |

**Returns:** The effective depth `He`, in metres.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the seabed has no critical angle or an input is invalid. |

## loss_parameter

```python
loss_parameter(attenuation_db_per_wavelength: float) -> float
```

Sediment loss parameter `ε = β_sed/(40·π·lg e)` (Ainslie Eq. 9.23).

**Parameters**

| Name | Description |
| :--- | :--- |
| `attenuation_db_per_wavelength` | `β_sed`, in dB per wavelength. |

**Returns:** The dimensionless loss parameter `ε`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the attenuation is negative or non-finite. |

## reflection_loss_gradient

```python
reflection_loss_gradient(
    seabed: str | WestonSeabed = 'sand',
    *,
    frequency_hz: float | None = None,
) -> float
```

Reflection loss gradient `η`, in nepers per radian.

The rate at which the seabed reflection loss grows with grazing angle,
`|R(θ)| ≈ exp(−η·θ)` (Ainslie Eq. 9.45). Two branches:

* a **reflecting** seabed with a critical angle (sand, coarse silt),
  `η = 2·ε·(ρ_sed/ρ_w)·cos²ψc/sin³ψc` (Eq. 9.51), frequency-independent;
* a **refracting** seabed with none (mud, clay, fine silt),
  `η = 2·ω·ε/c'` (Eq. 9.53), proportional to frequency.

**Parameters**

| Name | Description |
| :--- | :--- |
| `seabed` | `"sand"`, `"mud"` or an explicit [`WestonSeabed`](/phonometry/reference/api/underwater/weston-regimes/#westonseabed). |
| `frequency_hz` | Acoustic frequency, in Hz; required only for the refracting branch (`c' > 0`). |

**Returns:** The reflection loss gradient `η`, in Np/rad.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the frequency is missing or invalid for a refracting seabed. |

## waveguide_cutoff_frequency

```python
waveguide_cutoff_frequency(
    water_depth: float,
    *,
    seabed: str | WestonSeabed = 'sand',
    sound_speed: float = 1500.0,
) -> float
```

Shallow-water waveguide cut-off frequency `fc` (Ainslie Eq. 9.60), in Hz.

`fc = (π − ρ_sed/ρ_w)/(2·π·sin ψc) · c_w/H` -- below it no mode is cut on
and ducted propagation does not occur.

**Parameters**

| Name | Description |
| :--- | :--- |
| `water_depth` | Water-column depth `H`, in metres. |
| `seabed` | `"sand"`, `"mud"` or a [`WestonSeabed`](/phonometry/reference/api/underwater/weston-regimes/#westonseabed). |
| `sound_speed` | Water sound speed `c_w`, in m/s. |

**Returns:** The cut-off frequency, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the seabed has no critical angle or an input is invalid. |

## weston_propagation_loss

```python
weston_propagation_loss(
    range_m: NDArray[np.float64] | list[float] | float,
    frequency_hz: float,
    water_depth: float,
    *,
    seabed: str | WestonSeabed = 'sand',
    sound_speed: float = 1500.0,
    source_depth: float | None = None,
    receiver_depth: float | None = None,
    critical_angle: float | None = None,
    reflection_loss_gradient_value: float | None = None,
) -> WestonPropagationResult
```

Propagation loss across Weston's four shallow-water regimes.

Assembles the piecewise loss from Ainslie's Equations (9.42), (9.49) and
(9.54), switching regime at the boundaries of
[`weston_regime_boundaries`](/phonometry/reference/api/underwater/weston-regimes/#weston_regime_boundaries), and returns each regime's own law over the
whole range grid so the transitions can be drawn.

**Parameters**

| Name | Description |
| :--- | :--- |
| `range_m` | Range(s) from the source, in metres (scalar or array, strictly positive). |
| `frequency_hz` | Acoustic frequency, in Hz. |
| `water_depth` | Water-column depth `H`, in metres. |
| `seabed` | `"sand"`, `"mud"` or a [`WestonSeabed`](/phonometry/reference/api/underwater/weston-regimes/#westonseabed). |
| `sound_speed` | Water sound speed `c_w`, in m/s. |
| `source_depth` | Source depth `z0`, in metres; defaults to `H/2` (used only by the single-mode formula). |
| `receiver_depth` | Receiver depth `z`, in metres; defaults to `H/2`. |
| `critical_angle` | Override `ψc`, in degrees (`90` for an ideal totally reflecting waveguide). |
| `reflection_loss_gradient_value` | Override `η`, in Np/rad (`0` for a lossless bottom: no mode stripping, no single-mode regime). |

**Returns:** A [`WestonPropagationResult`](/phonometry/reference/api/underwater/weston-regimes/#westonpropagationresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

## weston_regime_boundaries

```python
weston_regime_boundaries(
    frequency_hz: float,
    water_depth: float,
    *,
    seabed: str | WestonSeabed = 'sand',
    sound_speed: float = 1500.0,
    critical_angle: float | None = None,
    reflection_loss_gradient_value: float | None = None,
) -> WestonRegimeBoundaries
```

Regime boundaries of a shallow-water waveguide (Ainslie §9.1.1.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Acoustic frequency, in Hz. |
| `water_depth` | Water-column depth `H`, in metres. |
| `seabed` | `"sand"`, `"mud"` or a [`WestonSeabed`](/phonometry/reference/api/underwater/weston-regimes/#westonseabed). |
| `sound_speed` | Water sound speed `c_w`, in m/s. |
| `critical_angle` | Override the seabed critical angle `ψc`, in degrees. Use `90` for the ideal totally reflecting waveguide. |
| `reflection_loss_gradient_value` | Override `η`, in Np/rad. Use `0` for a lossless bottom (no mode stripping, no single-mode regime). |

**Returns:** A [`WestonRegimeBoundaries`](/phonometry/reference/api/underwater/weston-regimes/#westonregimeboundaries).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

:::note
The two overrides are independent: overriding `critical_angle`
alone leaves `η` computed from the seabed's *own* critical angle
through Equation (9.51), which mixes two different bottoms. Pass both
together (as the ideal-waveguide case `critical_angle=90` with
`reflection_loss_gradient_value=0` does) whenever the intent is a
hypothetical seabed rather than a tweak of the tabulated one.
:::

## WESTON_REGIMES

*Constant* (`tuple`).

```python
WESTON_REGIMES = ('spherical', 'cylindrical', 'mode-stripping', 'single-mode')
```

## WESTON_SEABEDS

*Constant* (`dict`).

```python
WESTON_SEABEDS = {'sand': WestonSeabed(name='sand', grain_size=1.5, sound_speed_ratio=1.2, density_ratio=2.1, attenuation_db_per_wavelength=0.88, loss_parameter=0.0161, sound_speed_gradient=0.0), 'mud': WestonSeabed(name='mud', grain_size=8.0, sound_speed_ratio=1.0, density_ratio=1.4, attenuation_db_per_wavelength=0.09, loss_parameter=0.00165, sound_speed_gradient=1.0)}
```

## WestonPropagationResult

```python
WestonPropagationResult(
    range_m: NDArray[np.float64],
    propagation_loss: NDArray[np.float64],
    propagation_factor: NDArray[np.float64],
    regime: NDArray[np.str_],
    spherical: NDArray[np.float64],
    cylindrical: NDArray[np.float64],
    mode_stripping: NDArray[np.float64],
    single_mode: NDArray[np.float64],
    multipath: NDArray[np.float64],
    boundaries: WestonRegimeBoundaries,
    frequency: float,
    water_depth: float,
    source_depth: float,
    receiver_depth: float,
    seabed: str,
)
```

Weston regime propagation loss versus range.

**Attributes**

| Name | Description |
| :--- | :--- |
| `range_m` | Ranges from the source, in metres. |
| `propagation_loss` | Composite propagation loss `PL = −10·lg F` per range, in dB re 1 m². |
| `propagation_factor` | The composite propagation factor `F`, in m⁻². |
| `regime` | The active regime label at each range (one of [`WESTON_REGIMES`](/phonometry/reference/api/underwater/weston-regimes/#weston_regimes)). |
| `spherical` | Spherical-spreading loss `20·lg r` at every range, in dB. |
| `cylindrical` | Cylindrical-spreading loss (Eq. 9.42) at every range, dB. |
| `mode_stripping` | Mode-stripping loss (Eq. 9.49) at every range, dB (`nan` when the bottom is lossless: without reflection loss there is nothing to strip). |
| `single_mode` | Single-mode loss (Eq. 9.54) at every range, in dB. |
| `multipath` | Loss from the continuous multipath integral (Eq. 9.46), which joins the cylindrical and mode-stripping regimes smoothly, in dB. |
| `boundaries` | The [`WestonRegimeBoundaries`](/phonometry/reference/api/underwater/weston-regimes/#westonregimeboundaries) in force. |
| `frequency` | Acoustic frequency, in Hz. |
| `water_depth` | Water-column depth `H`, in metres. |
| `source_depth` | Source depth `z0`, in metres. |
| `receiver_depth` | Receiver depth `z`, in metres. |
| `seabed` | Name of the seabed used. |

### WestonPropagationResult.plot()

```python
WestonPropagationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the composite loss with each regime's law and the boundaries.

## WestonRegimeBoundaries

```python
WestonRegimeBoundaries(
    spherical_to_cylindrical: float,
    cylindrical_to_mode_stripping: float,
    mode_stripping_to_single_mode: float,
    critical_angle: float,
    reflection_loss_gradient: float,
    effective_depth: float,
    cutoff_frequency: float,
    mode_count: float,
)
```

Range boundaries between Weston's four propagation regimes.

**Attributes**

| Name | Description |
| :--- | :--- |
| `spherical_to_cylindrical` | Range at which `1/r²` and `2ψc/(rH)` are equal, `H/(2·ψc)`, in metres. |
| `cylindrical_to_mode_stripping` | Ainslie Eq. (9.50) `r_CS = π·H/(4·η·ψc²)`, in metres (`inf` for a lossless bottom). |
| `mode_stripping_to_single_mode` | `r_MS = k²·He²·H/(9·π·η)`, in metres (`inf` for a lossless bottom). See the module note on Eq. (9.57). |
| `critical_angle` | Critical grazing angle `ψc`, in radians. |
| `reflection_loss_gradient` | `η`, in Np/rad. |
| `effective_depth` | Weston effective depth `He`, in metres. |
| `cutoff_frequency` | Waveguide cut-off frequency, in Hz (`nan` when the seabed has no critical angle). |
| `mode_count` | Number of cut-on modes, `(ω/c_w)·He·sin ψc/π` (Eq. 9.58), as a real number. |

## WestonSeabed

```python
WestonSeabed(
    name: str,
    grain_size: float,
    sound_speed_ratio: float,
    density_ratio: float,
    attenuation_db_per_wavelength: float,
    loss_parameter: float,
    sound_speed_gradient: float,
)
```

Characteristic seabed properties (Ainslie Table 9.1, printed p. 454).

**Attributes**

| Name | Description |
| :--- | :--- |
| `name` | Sediment name. |
| `grain_size` | Grain size `Mz` (phi units). |
| `sound_speed_ratio` | `c_sed/c_w`. |
| `density_ratio` | `ρ_sed/ρ_w`. |
| `attenuation_db_per_wavelength` | `β_sed`, in dB per wavelength. |
| `loss_parameter` | `ε = β_sed/(40·π·lg e)` (Equation 9.23). |
| `sound_speed_gradient` | `c'`, the sediment sound-speed gradient, in s⁻¹ (0 for sand, 1 for mud). |
