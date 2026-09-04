---
title: "fluids.air"
description: "Humid air (IEC 61094-2:2009, Annex F)."
sidebar:
  label: "air"
---

Humid air (IEC 61094-2:2009, Annex F).

Annex F carries the CIPM-2007 formulation for the density of moist air together
with fits for the speed of sound, the ratio of specific heats, the viscosity and
the thermal diffusivity. It is the only model of air in this library that is
computed rather than quoted: every other air in the tree is a value some
standard printed for its own procedure, and those stay where their clause prints
them.

Table F.1 (printed folio 40) prints five quantities at two condition sets, and
every one of them reproduces here from the Table F.2 coefficients (printed folio
41) to better than 1,5e-7 relative, which is inside the rounding of the last
printed figure. Thermal conductivity and specific heat capacity come from the
two expressions of Clause F.6 (printed folio 39); the annex prints their formulae
and units but tabulates only the diffusivity they form, so those two are anchored
in closed form rather than against a printed number.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## air

```python
air(
    *,
    temperature_c: float,
    static_pressure_pa: float | None = None,
    relative_humidity_percent: float | None = None,
    co2_mole_fraction: float | None = None,
) -> Fluid
```

Humid air at one state (IEC 61094-2:2009, Annex F).

Returns the density, speed of sound, ratio of specific heats, viscosity and
thermal diffusivity Table F.1 tabulates, and the thermal conductivity and
specific heat capacity Clause F.6 gives expressions for.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature_c` | Air temperature `t`, in **degrees Celsius**. Required: there is no defensible default for the one condition the caller actually measured. |
| `static_pressure_pa` | Absolute static pressure `p_s`, in **pascals**. `None` assumes [`DEFAULT_STATIC_PRESSURE_PA`](/phonometry/reference/api/fluids/air/#default_static_pressure_pa) and warns. |
| `relative_humidity_percent` | Relative humidity `H`, in **per cent**. `None` assumes [`DEFAULT_RELATIVE_HUMIDITY_PERCENT`](/phonometry/reference/api/fluids/air/#default_relative_humidity_percent) and warns. |
| `co2_mole_fraction` | Carbon dioxide mole fraction `x_c`. `None` takes [`DEFAULT_CO2_MOLE_FRACTION`](/phonometry/reference/api/fluids/air/#default_co2_mole_fraction), the value Clause F.2 recommends for laboratory conditions, and does **not** warn: unlike the other two it is a value the annex names, and it reaches the fifth figure of the density at most. |

**Returns:** The [`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid) at that state.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the temperature is at or below -273,15 degC, the pressure is not positive, the humidity is outside 0 % to 100 %, the carbon dioxide mole fraction is outside 0 to 1, any of the four is not finite, or the pressure and humidity cannot hold together at that temperature. That last one is a combination rather than an argument: at 20 degC and 1 kPa, 50 % relative humidity asks for a water vapour mole fraction of 1,17, and a mole fraction cannot reach 1. |

Nothing else is refused. Annex F states a domain for its equations and this
warns outside it, because a fit past its range is still arithmetic; what it
refuses is a state that cannot exist.

## DEFAULT_CO2_MOLE_FRACTION

*Constant* (`float`).

```python
DEFAULT_CO2_MOLE_FRACTION = 0.0004
```

## DEFAULT_RELATIVE_HUMIDITY_PERCENT

*Constant* (`float`).

```python
DEFAULT_RELATIVE_HUMIDITY_PERCENT = 50.0
```

## DEFAULT_STATIC_PRESSURE_PA

*Constant* (`float`).

```python
DEFAULT_STATIC_PRESSURE_PA = 101325.0
```
