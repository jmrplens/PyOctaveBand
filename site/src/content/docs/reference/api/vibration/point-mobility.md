---
title: "vibration.point_mobility"
description: "Point mobilities and impedances of infinite structures (Cremer, Heckl & Petersson 2005, Chapter 5, Table 5.1)."
sidebar:
  label: "point_mobility"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Point mobilities and impedances of infinite structures (Cremer, Heckl &
Petersson 2005, Chapter 5, Table 5.1).

The **point mobility** `Y` of a structure is the complex ratio of the
velocity response at a driving point to the point force that produces it, and
its reciprocal is the **point impedance** `Z = 1/Y` (the same
motion-per-force / force-per-motion pair as ISO 7626-1 mechanical mobility, so
these theoretical values slot straight into [`MobilityResult`](/phonometry/reference/api/vibration/mechanical-mobility/#mobilityresult)
and [`convert_frf`](/phonometry/reference/api/vibration/mechanical-mobility/#convert_frf)). For an *infinite* structure the driving
point never sees a reflected wave, so the mobility is the free-field value that
sets the vibrational power a source injects (Cremer 5.5): with a point force of
amplitude `F` the time-averaged injected power is (Cremer Eq. 5.23):

```text
W = 0.5 * |F|**2 * Re{Y}                                    [W]
```

These are the theoretical companions of the *measured* driving-point mobilities
of ISO 7626 and the isolator transfer stiffnesses of ISO 10846, and they supply
the receiver mobility that the installed structure-borne prediction of
EN 12354-5 needs when no measurement is available.

**Compilation (Cremer Table 5.1).** With `m'` the mass per unit length
(kg/m), `m''` the mass per unit area (kg/m^2), `B` the bending stiffness of
a beam (N.m^2) and `B'` the bending stiffness of a plate *per unit width*
(N.m):

===============================  =========================  =============
Structure (point force)          Impedance `Z`            Mobility `Y`
===============================  =========================  =============
Longitudinal rod                 `rho cL S`               `1/(rho cL S)`
Slender beam, bending, centre    `2 m' cB (1 + j)`        `(1 - j)/(4 m' cB)`
Slender beam, bending, end       `(m' cB / 2)(1 + j)`     `(1 - j)/(m' cB)`
Thin plate, bending, centre      `8 sqrt(B' m'')`         `1/(8 sqrt(B' m''))`
Thin plate, bending, edge        `3.5 sqrt(B' m'')`       `1/(3.5 sqrt(B' m''))`
===============================  =========================  =============

The thin-plate driving-point impedance `Z = 8 sqrt(B' m'')` is real and
frequency independent (the plate behaves as a pure resistance to a point
force), so a plate absorbs power like a matched resistance. The beam impedance
grows as `cB = (B omega**2 / m')**(1/4)` (the bending wave speed), so its
mobility falls as `omega**(-1/2)`; the `(1 - j)` factor means half the
input goes into a reactive near field. A moment excitation of the beam has the
mobility (Cremer Eq. 5.75) `Y_M = omega (1 + j) / (4 B kB)` with
`kB = omega / cB` the bending wavenumber.

## infinite_beam_mobility

```python
infinite_beam_mobility(
    frequency: ArrayLike,
    bending_stiffness: float,
    mass_per_length: float,
    *,
    location: str = 'centre',
) -> NDArray[np.complex128]
```

Point mobility of an infinite beam in bending (Cremer Table 5.1).

`Y = (1 - j) / (4 m' cB)` for a force at the centre and
`Y = (1 - j) / (m' cB)` for a force at a free end, the reciprocal of
`infinite_beam_impedance`. The mobility falls as `omega**(-1/2)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array, > 0). |
| `bending_stiffness` | Beam bending stiffness `B = E I`, in N.m^2. |
| `mass_per_length` | Mass per unit length `m'`, in kg/m. |
| `location` | `"centre"` or `"end"`. |

**Returns:** The complex point mobility `Y`, in m/(N.s).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or unknown location. |

## infinite_beam_moment_mobility

```python
infinite_beam_moment_mobility(
    frequency: ArrayLike,
    bending_stiffness: float,
    mass_per_length: float,
) -> NDArray[np.complex128]
```

Moment (rotational) mobility of an infinite beam (Cremer Eq. 5.75).

`Y_M = omega (1 + j) / (4 B kB)` with the bending wavenumber
`kB = omega / cB`, the angular velocity per unit applied moment at the
driving point.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array, > 0). |
| `bending_stiffness` | Beam bending stiffness `B = E I`, in N.m^2. |
| `mass_per_length` | Mass per unit length `m'`, in kg/m. |

**Returns:** The complex moment mobility `Y_M`, in rad/(N.m.s).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## infinite_beam_point_mobility

```python
infinite_beam_point_mobility(
    frequency: ArrayLike,
    bending_stiffness: float,
    mass_per_length: float,
    *,
    location: str = 'centre',
) -> MobilityResult
```

Infinite-beam point mobility bundled as a [`MobilityResult`](/phonometry/reference/api/vibration/mechanical-mobility/#mobilityresult).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequencies `f`, in hertz (array, > 0). |
| `bending_stiffness` | Beam bending stiffness `B = E I`, in N.m^2. |
| `mass_per_length` | Mass per unit length `m'`, in kg/m. |
| `location` | `"centre"` or `"end"`. |

**Returns:** The [`MobilityResult`](/phonometry/reference/api/vibration/mechanical-mobility/#mobilityresult) (driving point).

## infinite_plate_impedance

```python
infinite_plate_impedance(
    bending_stiffness: float,
    mass_per_area: float,
    *,
    location: str = 'centre',
) -> float
```

Point impedance of an infinite thin plate (Cremer Table 5.1).

`Z = C sqrt(B' m'')` with `C = 8` for a force at the plate centre and
`C = 3.5` for a force at a free edge. The impedance is purely real and
frequency independent: an infinite plate presents a matched resistance to a
point force.

**Parameters**

| Name | Description |
| :--- | :--- |
| `bending_stiffness` | Plate bending stiffness per unit width `B'`, in N.m (see [`plate_bending_stiffness`](/phonometry/reference/api/vibration/point-mobility/#plate_bending_stiffness)). |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2. |
| `location` | `"centre"` (`C = 8`) or `"edge"` (`C = 3.5`). |

**Returns:** The point impedance `Z`, in N.s/m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive stiffness/mass or unknown location. |

## infinite_plate_mobility

```python
infinite_plate_mobility(
    bending_stiffness: float,
    mass_per_area: float,
    *,
    location: str = 'centre',
) -> float
```

Point mobility of an infinite thin plate `Y = 1 / (C sqrt(B' m''))`.

The reciprocal of [`infinite_plate_impedance`](/phonometry/reference/api/vibration/point-mobility/#infinite_plate_impedance) (real, frequency
independent).

**Parameters**

| Name | Description |
| :--- | :--- |
| `bending_stiffness` | Plate bending stiffness per unit width `B'`, in N.m. |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2. |
| `location` | `"centre"` (`C = 8`) or `"edge"` (`C = 3.5`). |

**Returns:** The point mobility `Y`, in m/(N.s).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive stiffness/mass or unknown location. |

## infinite_plate_point_mobility

```python
infinite_plate_point_mobility(
    frequency: ArrayLike,
    bending_stiffness: float,
    mass_per_area: float,
    *,
    location: str = 'centre',
) -> MobilityResult
```

Infinite-plate point mobility bundled as a [`MobilityResult`](/phonometry/reference/api/vibration/mechanical-mobility/#mobilityresult).

The plate mobility is frequency independent, so the returned spectrum is
constant across *frequency*; bundling it lets it be plotted and converted
with the ISO 7626 mobility machinery.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequencies `f`, in hertz (array, > 0). |
| `bending_stiffness` | Plate bending stiffness per unit width `B'`, in N.m. |
| `mass_per_area` | Mass per unit area `m''`, in kg/m^2. |
| `location` | `"centre"` or `"edge"`. |

**Returns:** The [`MobilityResult`](/phonometry/reference/api/vibration/mechanical-mobility/#mobilityresult) (driving point).

## injected_power

```python
injected_power(force: ArrayLike, mobility: ArrayLike) -> NDArray[np.float64]
```

Time-averaged vibrational power injected by a point force (Cremer 5.23).

`W = 0.5 |F|**2 Re{Y}`: only the real part (conductance) of the mobility
carries power; the reactive part stores near-field energy.

**Parameters**

| Name | Description |
| :--- | :--- |
| `force` | Point-force amplitude `F` (peak, scalar or array), in N. |
| `mobility` | Complex point mobility `Y` (broadcast with *force*), in m/(N.s). |

**Returns:** The injected power `W`, in W.

## longitudinal_rod_impedance

```python
longitudinal_rod_impedance(
    density: float,
    longitudinal_wave_speed: float,
    cross_section_area: float,
) -> float
```

Point impedance of an infinite rod in longitudinal motion (Table 5.1).

`Z = rho cL S`, real and frequency independent.

**Parameters**

| Name | Description |
| :--- | :--- |
| `density` | Material density `rho`, in kg/m^3. |
| `longitudinal_wave_speed` | Longitudinal wave speed `cL`, in m/s. |
| `cross_section_area` | Cross-section area `S`, in m^2. |

**Returns:** The point impedance `Z`, in N.s/m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input. |

## plate_bending_stiffness

```python
plate_bending_stiffness(
    youngs_modulus: float,
    thickness: float,
    poisson_ratio: float = 0.0,
) -> float
```

Bending stiffness of a thin plate per unit width (Cremer Eq. 4.22).

`B' = E h**3 / (12 (1 - nu**2))`, the plate bending stiffness `B'` in
N.m used throughout this module and by the coincidence frequency of
[`phonometry.vibration.radiation_efficiency.coincidence_frequency`](/phonometry/reference/api/vibration/radiation-efficiency/#coincidence_frequency).

**Parameters**

| Name | Description |
| :--- | :--- |
| `youngs_modulus` | Young's modulus `E` of the plate material, in Pa. |
| `thickness` | Plate thickness `h`, in m. |
| `poisson_ratio` | Poisson's ratio `nu` (Default: 0.0). |

**Returns:** The bending stiffness per unit width `B'`, in N.m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive modulus/thickness or `\|nu\| >= 1`. |
