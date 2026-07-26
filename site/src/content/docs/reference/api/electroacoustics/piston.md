---
title: "electroacoustics.piston"
description: "Radiation of a rigid circular piston set in an infinite baffle."
sidebar:
  label: "piston"
---

Radiation of a rigid circular piston set in an infinite baffle.

The baffled circular piston is the canonical acoustic radiator: a flat rigid
disc of radius `a` vibrating with a uniform normal velocity in an otherwise
rigid infinite plane. It is the model behind a loudspeaker cone in a large
cabinet, the open end of a duct, and the reference source for the radiation
efficiency of any finite vibrating surface, so its two results -- the
**radiation impedance** the air presents to the piston and the **directivity**
of the far field -- are the base of the electroacoustics domain (Beranek &
Mellow, *Acoustics: Sound Fields, Transducers and Vibration* 2nd ed., §4.4;
Bies, Hansen & Howard, *Engineering Noise Control* 5th ed.).

**Radiation impedance.** The reaction force of the air on the piston is
`F = Z_r u` with the mechanical radiation impedance

    Z_r = rho c S ( R1(2ka) + j X1(2ka) ),      S = pi a^2,

where `k = omega / c` is the wavenumber, `rho c` the characteristic
impedance of air and `S` the piston area. The dimensionless **piston
resistance** and **reactance** functions are (Beranek & Mellow Eq. (4.30))

    R1(x) = 1 - 2 J1(x) / x,        X1(x) = 2 H1(x) / x,

with `J1` the Bessel function of the first kind and `H1` the Struve
function, both of order one, evaluated at `x = 2ka`.

* **Low frequency** (`ka << 1`): `R1 -> (ka)^2 / 2` so the radiated power
  rises as `f^2`, and `X1 -> (8 / 3 pi) ka`. The reactance is mass-like,
  `X_r = rho c S X1 = omega M_r` with the **radiation (accreted) mass**

      M_r = 8 rho a^3 / 3

  (Beranek & Mellow Eq. (4.32)): the piston drags an extra `8 rho a^3 / 3` of
  air, equivalent to a layer `8a / 3 pi` thick over its face.
* **High frequency** (`ka >> 1`): `R1 -> 1` and `X1 -> 0`, so
  `Z_r -> rho c S` -- the piston radiates as if into an infinite tube and the
  air loads it purely resistively.

**Directivity.** The far-field pressure of the baffled piston varies with the
polar angle `theta` from the axis as (Beranek & Mellow Eq. (4.42))

    D(theta) = 2 J1(ka sin theta) / (ka sin theta),      D(0) = 1.

The main lobe narrows as `ka` grows; its first null is at
`ka sin theta = 3.8317` (the first zero of `J1`), which exists only once
`ka > 3.8317`. The **directivity factor** `Q` (on-axis intensity over the
intensity of a point source of equal power radiating into the full sphere) and
the **directivity index** `DI = 10 log10 Q` follow from integrating
`|D|^2` over the radiating hemisphere,

    Q = 2 / integral_0^(pi/2) |D(theta)|^2 sin theta d theta,

which tends to `Q = 2` (`DI = 3.01 dB`, the half-space baffle gain) at low
`ka` and to `Q ~ (ka)^2` (`DI ~ 20 log10 ka`) at high `ka`.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## piston_directivity

```python
piston_directivity(ka: ArrayLike, theta: ArrayLike) -> np.ndarray | float
```

Far-field directivity `D = 2 J1(ka sin theta) / (ka sin theta)`.

The pressure amplitude of a baffled circular piston relative to its on-axis
value (Beranek & Mellow Eq. (4.42)), normalized so `D(0) = 1`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ka` | Wavenumber-radius product `ka` (scalar or array). |
| `theta` | Polar angle from the axis, rad (scalar or array). Broadcast against `ka`. |

**Returns:** `D` (float for scalar inputs, else an array).

## piston_directivity_pattern

```python
piston_directivity_pattern(
    ka: ArrayLike,
    angles: ArrayLike | None = None,
) -> PistonDirectivity
```

Far-field directivity pattern of one or more baffled circular pistons.

Samples the directivity `D(theta) = 2 J1(ka sin theta) / (ka sin theta)`
(Beranek & Mellow Eq. (4.42)) at every `ka` over a polar-angle grid and
bundles it into a [`PistonDirectivity`](/phonometry/reference/api/electroacoustics/piston/#pistondirectivity) that exposes `.plot()`. The
main lobe narrows as `ka` grows; its first null appears once `ka` passes
the first zero of `J1` (`ka sin theta = 3.8317`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ka` | Wavenumber-radius product(s) `ka` (scalar or 1-D array), each non-negative. |
| `angles` | Polar angles `theta` from the axis, rad (1-D). `None` (default) uses 361 points spanning the front hemisphere `-90 deg` to `+90 deg`, 0.5 deg apart. |

**Returns:** A [`PistonDirectivity`](/phonometry/reference/api/electroacoustics/piston/#pistondirectivity).

## piston_reactance

```python
piston_reactance(x: ArrayLike) -> np.ndarray | float
```

Piston reactance function `X1(x) = 2 H1(x) / x` (`H1` Struve order 1).

The imaginary part of the normalized radiation impedance of a baffled
circular piston (Beranek & Mellow Eq. (4.30)). It rises as
`(8 / 3 pi) ka` (mass-like) at low `x = 2ka` and decays to 0 at high
`x`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Argument `x = 2ka` (scalar or array), dimensionless. |

**Returns:** `X1(x)` (float for scalar input, else an array).

## piston_resistance

```python
piston_resistance(x: ArrayLike) -> np.ndarray | float
```

Piston resistance function `R1(x) = 1 - 2 J1(x) / x`.

The real part of the normalized radiation impedance of a baffled circular
piston, as a function of `x = 2ka` (Beranek & Mellow Eq. (4.30)). It
rises as `x^2 / 8 = (ka)^2 / 2` at low `x` and tends to 1 at high `x`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Argument `x = 2ka` (scalar or array), dimensionless. |

**Returns:** `R1(x)` (float for scalar input, else an array).

## PistonDirectivity

```python
PistonDirectivity(
    angles: np.ndarray,
    ka: np.ndarray,
    directivity: np.ndarray,
    directivity_db: np.ndarray,
)
```

Far-field directivity pattern of a baffled circular piston.

Bundles the far-field directivity
`D(theta) = 2 J1(ka sin theta) / (ka sin theta)` (Beranek & Mellow
Eq. (4.42)) of one or more baffled circular pistons over a shared
polar-angle grid, so the classic beam pattern can be drawn with
`plot`. The maths is [`piston_directivity`](/phonometry/reference/api/electroacoustics/piston/#piston_directivity); this is a thin,
plottable bundle around it.

**Attributes**

| Name | Description |
| :--- | :--- |
| `angles` | Polar angles `theta` from the axis, rad. |
| `ka` | Wavenumber-radius products `ka`, one per pattern (a 1-D array). |
| `directivity` | Linear directivity `D(theta)`, normalized so `D(0) = 1`, as a `(len(ka), len(angles))` array; row `i` is the pattern for `ka[i]`. |
| `directivity_db` | Directivity in dB, `20 log10 \|D\|`, same shape as `directivity` (the side-lobe nulls floor at a large negative value rather than `-inf`). |

### PistonDirectivity.plot()

```python
PistonDirectivity.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the far-field directivity (beam) pattern on a polar axes.

Draws the directivity in dB against the polar angle: one curve per
`ka` value as a single family (still one concept, the directivity
pattern). A polar axes is created when `ax` is `None`. Requires
matplotlib (`pip install phonometry[plot]`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing (polar) axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the `Axes.plot` call when the result holds a single `ka`; ignored for a multi-`ka` family so one user color or label cannot collapse the per-curve styling. |

**Returns:** The axes.

## radiating_piston

```python
radiating_piston(
    radius: float,
    frequencies: ArrayLike,
    *,
    speed_of_sound: float = 343.0,
    density: float = 1.206,
    angles: ArrayLike | None = None,
) -> RadiatingPistonResult
```

Radiation impedance and directivity of a rigid baffled circular piston.

Evaluates the piston resistance `R1(2ka)` and reactance `X1(2ka)`, the
mechanical radiation impedance `rho c S (R1 + j X1)`, the low-frequency
radiation mass `8 rho a^3 / 3` and the directivity index over the given
frequencies (Beranek & Mellow §4.4). Pass `angles` to also sample the
far-field directivity pattern `D(theta)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `radius` | Piston radius `a`, m. |
| `frequencies` | Frequencies `f`, Hz (scalar or 1-D array), all > 0. |
| `speed_of_sound` | Speed of sound `c`, m/s (default 343). |
| `density` | Air density `rho`, kg/m3 (default 1.206). |
| `angles` | Optional polar angles `theta` from the axis, rad, at which to sample the directivity pattern. |

**Returns:** A [`RadiatingPistonResult`](/phonometry/reference/api/electroacoustics/piston/#radiatingpistonresult).

## RadiatingPistonResult

```python
RadiatingPistonResult(
    frequencies: np.ndarray,
    ka: np.ndarray,
    resistance: np.ndarray,
    reactance: np.ndarray,
    radiation_resistance: np.ndarray,
    radiation_reactance: np.ndarray,
    radiation_mass: float,
    directivity_index: np.ndarray,
    angles: np.ndarray | None,
    directivity: np.ndarray | None,
    radius: float,
    speed_of_sound: float,
    density: float,
)
```

Radiation impedance and directivity of a baffled circular piston.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz. |
| `ka` | Wavenumber-radius product `ka` at each frequency. |
| `resistance` | Normalized piston resistance `R1(2ka)` (real part of `Z_r / (rho c S)`). |
| `reactance` | Normalized piston reactance `X1(2ka)` (imaginary part of `Z_r / (rho c S)`). |
| `radiation_resistance` | Mechanical radiation resistance `rho c S R1`, N s/m. |
| `radiation_reactance` | Mechanical radiation reactance `rho c S X1`, N s/m. |
| `radiation_mass` | Low-frequency accreted air mass `M_r = 8 rho a^3/3`, kg (a single value; the mass limit of `radiation_reactance / omega`). |
| `directivity_index` | Directivity index `DI = 10 log10 Q`, dB. |
| `angles` | Polar angles of `directivity`, rad, or `None` if not requested. |
| `directivity` | Far-field directivity `D(theta)` as a `(n_freq, n_angle)` array, or `None` if `angles` was not given. |
| `radius` | Piston radius `a`, m. |
| `speed_of_sound` | Speed of sound `c`, m/s. |
| `density` | Air density `rho`, kg/m3. |

### RadiatingPistonResult.plot()

```python
RadiatingPistonResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the normalized piston resistance and reactance against `ka`.

Reproduces the classic Beranek & Mellow figure: `R1` rising to 1 and
`X1` peaking then decaying, over the `ka` range of the result.
Requires matplotlib (`pip install phonometry[plot]`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
