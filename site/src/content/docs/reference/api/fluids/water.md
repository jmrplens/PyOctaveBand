---
title: "fluids.water"
description: "Sea water: its speed of sound, and its density."
sidebar:
  label: "water"
---

Sea water: its speed of sound, and its density.

Four coexisting equations for the sound speed `c` as a function of
temperature, salinity and depth or pressure, selectable through
`sound_speed_model`:

* `"unesco"` -- the UNESCO / Chen & Millero (1977) algorithm, the
  international standard, in the Wong & Zhu (1995) ITS-90 recalculation.
  Default.
* `"del_grosso"` -- the Del Grosso (1974) equation (Wong & Zhu 1995 form),
  a high-accuracy alternative over a narrower domain.
* `"mackenzie"` -- the Mackenzie (1981) nine-term depth-based equation.
* `"medwin"` -- the Medwin (1975) six-term short formula.

Unlike air, sea water has four competing fits to one quantity rather than one
model of a substance, which is why this constructor takes a model and
[`air`](/phonometry/reference/api/fluids/air/) does not: those four are answers to the same
question, while the air formulas scattered through the library are clauses of
different measurement standards and must never be substitutable.

The density comes from Ainslie, *Principles of Sonar Performance Modelling*
(Springer 2010), Equation (4.6) on printed folio 127, which the book attributes
to Pierce (1989, p. 34). Its pressure argument is **absolute**, defined by
Equation (4.4) as the atmosphere plus the water column above.

Sea water has no ratio of specific heats or thermal diffusivity here: no source
in this library prints them, so a [`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid) built from
this module raises [`FluidPropertyUnavailable`](/phonometry/reference/api/fluids/fluids/#fluidpropertyunavailable) for
them rather than inventing a number.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## depth_to_absolute_pressure_pa

```python
depth_to_absolute_pressure_pa(
    *,
    depth_m: float,
    latitude_deg: float = 45.0,
) -> float
```

Absolute static pressure at an ocean depth, in pascals.

Ainslie Equation (4.11), printed folio 128. **Absolute**, not gauge:
Equation (4.4) defines the static pressure as the atmosphere plus the
weight of the water above, so this returns one atmosphere at the surface
rather than zero.

That is the pressure [`sea_water`](/phonometry/reference/api/fluids/water/#sea_water)'s density wants. The sound speeds
want the other one, from [`depth_to_gauge_pressure_mpa`](/phonometry/reference/api/fluids/water/#depth_to_gauge_pressure_mpa), and the two
names say which is which because the difference is a factor of a million
and an offset of an atmosphere.

**Parameters**

| Name | Description |
| :--- | :--- |
| `depth_m` | Depth below the surface `z`, in metres (`>= 0`). |
| `latitude_deg` | Latitude $\varphi$, in degrees (default 45). |

**Returns:** Absolute static pressure, in pascals.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the depth is negative or non-finite. |

## depth_to_gauge_pressure_mpa

```python
depth_to_gauge_pressure_mpa(
    *,
    depth_m: float,
    latitude_deg: float = 45.0,
) -> float
```

Gauge pressure at an ocean depth (Leroy & Parthiot 1998), in MPa.

The standard-ocean formula, for an ideal medium of 0 degC and 35 ppt; no
local corrections are applied. This is the pressure the UNESCO and Del
Grosso sound speeds want, and it is **gauge**: zero at the surface.

Its companion [`depth_to_absolute_pressure_pa`](/phonometry/reference/api/fluids/water/#depth_to_absolute_pressure_pa) answers the other
question, in the other unit and from the other datum, which is why both
names carry theirs. Handing one to what wants the other is a factor of a
million and an offset of an atmosphere.

**Parameters**

| Name | Description |
| :--- | :--- |
| `depth_m` | Depth below the surface `Z`, in metres (`>= 0`). |
| `latitude_deg` | Latitude $\varphi$, in degrees (default 45). |

**Returns:** Gauge pressure, in megapascals.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the depth is negative or non-finite. |

## sea_water

```python
sea_water(
    *,
    temperature_c: float,
    salinity_psu: float = 35.0,
    depth_m: float = 0.0,
    latitude_deg: float = 45.0,
    sound_speed_model: str = 'unesco',
) -> Fluid
```

Sea water at one point of the ocean.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature_c` | Temperature `T`, in degrees Celsius. |
| `salinity_psu` | Practical salinity `S`, dimensionless (default 35, the salinity of the standard ocean). |
| `depth_m` | Depth below the surface, in metres (default 0). |
| `latitude_deg` | Latitude, in degrees, for the depth-to-pressure conversions (default 45). |
| `sound_speed_model` | Which of the four fits to use; see the module docstring. |

**Returns:** The [`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid) at that point.

The density and the speed of sound come from different sources, which is
why both are named in the result's `model`. Neither source in this
library prints a ratio of specific heats, a viscosity or a thermal
diffusivity for sea water, so reading one raises rather than guessing.

## sea_water_density

```python
sea_water_density(
    *,
    temperature_c: float,
    salinity_psu: float,
    absolute_pressure_pa: float,
) -> float
```

Density of sea water, in kilograms per cubic metre.

Ainslie Equation (4.6), printed folio 127, attributed there to Pierce
(1989, p. 34):

$$
\rho = 1027 + 4{,}3\times10^{-7} P_\mathrm{w} + 0{,}75\,(S - 35) - 0{,}16\,(T - 10) - 0{,}004\,(T - 10)^2
$$

The pressure is **absolute**, in pascals, as Equations (4.4) and (4.7) to
(4.10) define it. Use [`depth_to_absolute_pressure_pa`](/phonometry/reference/api/fluids/water/#depth_to_absolute_pressure_pa) to get one from
a depth.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature_c` | Temperature `T`, in degrees Celsius. |
| `salinity_psu` | Practical salinity `S`, dimensionless. |
| `absolute_pressure_pa` | Absolute static pressure `P_w`, in pascals. |

**Returns:** Density `rho`, in kg/m3.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For a temperature at or below absolute zero, a negative salinity, or a non-positive pressure. |

## sea_water_sound_speed

```python
sea_water_sound_speed(
    temperature: float,
    salinity: float,
    depth: float,
    *,
    model: str = ...,
    latitude: float = ...,
) -> float

sea_water_sound_speed(
    temperature: ArrayLike,
    salinity: ArrayLike,
    depth: ArrayLike,
    *,
    model: str = ...,
    latitude: float = ...,
) -> float | NDArray[np.float64]
```

Speed of sound in sea water, in metres per second.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Temperature `T`, in degrees Celsius. |
| `salinity` | Salinity `S`, in parts per thousand (PSU). |
| `depth` | Depth below the surface, in metres (`>= 0`). |
| `model` | `"unesco"` (default), `"del_grosso"`, `"mackenzie"` or `"medwin"`. |
| `latitude` | Latitude for the depth→pressure conversion, in degrees (used by `"unesco"` and `"del_grosso"`; default 45°). |

**Returns:** The sound speed `c`, in m/s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `model` is unknown or an input is non-finite. |

:::note
Each equation is a fit over a bounded oceanographic domain and
**extrapolates silently outside it** (e.g. Del Grosso abused at
T = 40 °C, S = 0, z = 11 km returns an unphysical ~1995 m/s).
Published validity domains: UNESCO/Chen-Millero T 0-40 °C, S 0-40,
P 0-1000 bar; Del Grosso T 0-30 °C, S 30-40, P 0-1000 kg/cm²;
Mackenzie T 2-30 °C, S 25-40, depth 0-8000 m. Medwin is a
deliberately simplified fit ("not accurate by modern standards", in
Ainslie's words) and drifts by a few m/s against the UNESCO standard
away from mid-range temperatures and shallow depths.
:::
