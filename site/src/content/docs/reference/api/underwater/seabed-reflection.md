---
title: "underwater.propagation.seabed_reflection"
description: "Plane-wave reflection at the seabed (fluid-fluid Rayleigh model)."
sidebar:
  label: "seabed_reflection"
---

Plane-wave reflection at the seabed (fluid-fluid Rayleigh model).

A plane wave in the water column (density $\rho_1$, sound speed
`c1`) striking a fluid sediment half-space ($\rho_2$, `c2`)
reflects with the Rayleigh pressure reflection coefficient (Medwin & Clay,
Eq. 2.6.11a). Using the grazing angle $\varphi$ (measured from the
interface), the angle of incidence from the normal is
$\theta_1 = 90^\circ - \varphi$, Snell's law gives
$\sin\theta_2 = (c_2/c_1) \cos\varphi$ and

$$
R = \frac{\rho_2 c_2 \sin\varphi - \rho_1 c_1 \cos\theta_2} {\rho_2 c_2 \sin\varphi + \rho_1 c_1 \cos\theta_2}
$$

* [`critical_angle`](/phonometry/reference/api/underwater/seabed-reflection/#critical_angle) -- the critical grazing angle
  $\varphi_c = \arccos(c_1/c_2)$ (only when `c2 > c1`); below it the
  wave is totally reflected ($\lvert R \rvert = 1$).
* [`reflection_coefficient`](/phonometry/reference/api/underwater/seabed-reflection/#reflection_coefficient) -- the complex `R` per grazing angle.
* [`seabed_reflection`](/phonometry/reference/api/underwater/seabed-reflection/#seabed_reflection) -- the complex `R`, its magnitude `|R|` and the
  bottom loss bundled with the interface parameters into a
  [`SeabedReflection`](/phonometry/reference/api/underwater/seabed-reflection/#seabedreflection) whose `.plot()` draws `|R|` versus grazing angle.
* [`bottom_reflection_loss`](/phonometry/reference/api/underwater/seabed-reflection/#bottom_reflection_loss) -- the bottom loss
  $\mathrm{BL} = -20 \log_{10} \lvert R \rvert$ (dB), returned as a
  [`BottomLossResult`](/phonometry/reference/api/underwater/seabed-reflection/#bottomlossresult) with a `.plot()`.

Lossless fluid-fluid model (real $\rho$/`c`); sediment attenuation is
out of scope. Densities enter only through the impedance ratio, so any
consistent unit works (kg/m³ by convention).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## bottom_reflection_loss

```python
bottom_reflection_loss(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float = 1000.0,
    c1: float = 1500.0,
    rho2: float,
    c2: float,
) -> BottomLossResult
```

Bottom reflection loss $\mathrm{BL} = -20 \log_{10} \lvert R \rvert$
versus grazing angle (dB).

At an angle of intromission $R = 0$ and the loss is `inf` (total
transmission into the sediment); that is a legitimate bottom-loss value,
returned without warning.

**Parameters**

| Name | Description |
| :--- | :--- |
| `grazing_angle` | Grazing angle(s) $\varphi$ from the interface, in degrees. |
| `rho1` | Water density $\rho_1$ (default 1000 kg/m³). |
| `c1` | Sound speed in the water `c1`, in m/s (default 1500). |
| `rho2` | Sediment density $\rho_2$. |
| `c2` | Sound speed in the sediment `c2`, in m/s. |

**Returns:** A [`BottomLossResult`](/phonometry/reference/api/underwater/seabed-reflection/#bottomlossresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## BottomLossResult

```python
BottomLossResult(
    grazing_angle: NDArray[np.float64],
    reflection_loss: NDArray[np.float64],
    reflection_coefficient: NDArray[np.complex128],
    critical_angle: float | None,
)
```

Bottom reflection loss versus grazing angle (fluid-fluid Rayleigh
model).

**Attributes**

| Name | Description |
| :--- | :--- |
| `grazing_angle` | Grazing angles, in degrees. |
| `reflection_loss` | Bottom loss $\mathrm{BL} = -20 \log_{10} \lvert R \rvert$ per angle, in dB. |
| `reflection_coefficient` | Complex reflection coefficient per angle. |
| `critical_angle` | The critical grazing angle, in degrees, or `None` if the sediment is not faster than the water. |

### BottomLossResult.plot()

```python
BottomLossResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the bottom loss versus grazing angle with the critical angle.

## critical_angle

```python
critical_angle(c1: float, c2: float) -> float
```

Critical grazing angle $\varphi_c = \arccos(c_1/c_2)$, in
degrees.

Defined only when the sediment is faster than the water (`c2 > c1`); at and
below this grazing angle the wave is totally reflected.

**Parameters**

| Name | Description |
| :--- | :--- |
| `c1` | Sound speed in the water, in m/s. |
| `c2` | Sound speed in the sediment, in m/s. |

**Returns:** The critical grazing angle, in degrees.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `c2 <= c1` (no critical angle exists). |

## reflection_coefficient

```python
reflection_coefficient(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float,
    c1: float,
    rho2: float,
    c2: float,
) -> NDArray[np.complex128]
```

Complex plane-wave pressure reflection coefficient at the seabed.

**Parameters**

| Name | Description |
| :--- | :--- |
| `grazing_angle` | Grazing angle(s) $\varphi$ from the interface, in degrees (`0` grazing to `90` normal incidence). |
| `rho1` | Water density $\rho_1$ (any consistent unit; kg/m³ by convention). |
| `c1` | Sound speed in the water `c1`, in m/s. |
| `rho2` | Sediment density $\rho_2$. |
| `c2` | Sound speed in the sediment `c2`, in m/s. |

**Returns:** The complex reflection coefficient per grazing angle.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## seabed_reflection

```python
seabed_reflection(
    grazing_angle: NDArray[np.float64] | list[float] | float,
    *,
    rho1: float = 1000.0,
    c1: float = 1500.0,
    rho2: float,
    c2: float,
) -> SeabedReflection
```

Build a plottable seabed reflection-coefficient result (Rayleigh
model).

Evaluates [`reflection_coefficient`](/phonometry/reference/api/underwater/seabed-reflection/#reflection_coefficient) at `grazing_angle` for the given
fluid-fluid interface and bundles the complex `R`, its magnitude `|R|`,
the bottom loss $\mathrm{BL} = -20 \log_{10} \lvert R \rvert$ and the
interface parameters into a [`SeabedReflection`](/phonometry/reference/api/underwater/seabed-reflection/#seabedreflection) that exposes
`.plot()`. The maths is unchanged; this is a thin, plottable wrapper
around the existing function (the same `ValueError` cases apply). At an
angle of intromission $R = 0$ and the bottom loss is `inf` (total
transmission), returned without warning.

**Parameters**

| Name | Description |
| :--- | :--- |
| `grazing_angle` | Grazing angle(s) $\varphi$ from the interface, in degrees (`0` grazing to `90` normal incidence). |
| `rho1` | Water density $\rho_1$ (default 1000 kg/m³). |
| `c1` | Sound speed in the water `c1`, in m/s (default 1500). |
| `rho2` | Sediment density $\rho_2$. |
| `c2` | Sound speed in the sediment `c2`, in m/s. |

**Returns:** A frozen [`SeabedReflection`](/phonometry/reference/api/underwater/seabed-reflection/#seabedreflection).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## SeabedReflection

```python
SeabedReflection(
    grazing_angle: NDArray[np.float64],
    reflection_coefficient: NDArray[np.complex128],
    magnitude: NDArray[np.float64],
    bottom_loss: NDArray[np.float64],
    critical_angle: float | None,
    rho1: float,
    c1: float,
    rho2: float,
    c2: float,
)
```

Plane-wave seabed reflection coefficient versus grazing angle.

Bundles the complex Rayleigh reflection coefficient `R` over a
grazing-angle grid with its magnitude `|R|`, the bottom loss
$\mathrm{BL} = -20 \log_{10} \lvert R \rvert$ and the fluid-fluid
interface parameters, so the classic `|R|` versus grazing-angle curve
can be drawn with `plot`. Build it with [`seabed_reflection`](/phonometry/reference/api/underwater/seabed-reflection/#seabed_reflection);
the frozen instance is a thin, plottable wrapper and re-runs none of the
maths.

**Attributes**

| Name | Description |
| :--- | :--- |
| `grazing_angle` | Grazing angles $\varphi$ from the interface, in degrees. |
| `reflection_coefficient` | Complex pressure reflection coefficient `R` per grazing angle. |
| `magnitude` | Reflection-coefficient magnitude `\|R\|` per grazing angle (`1` below the critical angle for a faster sediment). |
| `bottom_loss` | Bottom loss $\mathrm{BL} = -20 \log_{10} \lvert R \rvert$ per grazing angle, in dB. |
| `critical_angle` | The critical grazing angle, in degrees, or `None` if the sediment is not faster than the water. |
| `rho1` | Water density $\rho_1$. |
| `c1` | Sound speed in the water `c1`, in m/s. |
| `rho2` | Sediment density $\rho_2$. |
| `c2` | Sound speed in the sediment `c2`, in m/s. |

### SeabedReflection.plot()

```python
SeabedReflection.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the reflection-coefficient magnitude `|R|` versus grazing angle.

Draws `|R|` on a linear grazing-angle axis (0..90°), marking the
critical angle when the sediment is faster than the water. Requires
matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the `\|R\|` curve `plot` call. |

**Returns:** The axes.
