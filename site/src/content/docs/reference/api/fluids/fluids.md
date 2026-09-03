---
title: "fluids"
description: "fluids domain of phonometry (see module docstrings)."
sidebar:
  label: "fluids"
---

fluids domain of phonometry (see module docstrings).

The state of the medium a sound travels through, computed from the conditions
that were measured rather than assumed. Every other package may import this one
without an architecture edge, the way they import `phonometry.filters`,
`phonometry.signals` and `phonometry.metrology`: a medium is not a
domain of application but something every domain needs, and eleven identical
edges would record nothing.

What lives here is the *physics* of a fluid. A simplified formula a measurement
standard prints inside its own procedure stays in that standard's module, where
its clause can be cited beside it, and a constant frozen by a conformance row
never moves at all. Those three are different things, and keeping them apart is
what lets better physics reach a caller without any measurement silently
ceasing to reproduce the standard it claims.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## characteristic_impedance

```python
characteristic_impedance(density: float, speed_of_sound: float) -> float
```

Characteristic impedance `rho c`, in pascal seconds per metre.

The product of a medium's density and its speed of sound. It belongs to the
medium rather than to any procedure, which is why it lives here and not with
the impedance tube that used to publish it; ISO 10534-2 Clause 7.2 and
ASTM E2611-19 Clauses 8.2/8.3 both reach for the same product, and so does
every reflection coefficient in the library.

[`Fluid`](/phonometry/reference/api/fluids/fluids/#fluid) exposes the same quantity as a property, closed from the two
it was built with. This function is for a caller who has a density and a
speed of sound and no fluid to go with them.

The arguments carry no unit in their names, unlike the temperatures and
pressures elsewhere in the library: those name a unit because two are in
play and a caller can supply the wrong one. A density in this tree is
kilograms per cubic metre everywhere, and a speed of sound is metres per
second everywhere, so there is no second unit to be confused with.

**Parameters**

| Name | Description |
| :--- | :--- |
| `density` | Density `rho`, in kg/m3. |
| `speed_of_sound` | Speed of sound `c`, in m/s. |

**Returns:** Characteristic impedance `rho c`, in Pa\*s/m (rayl).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if either argument is not positive. |

## Fluid

```python
Fluid(
    temperature_c: float,
    static_pressure_pa: float,
    composition: Mapping[str, float],
    model: str,
    validity: str,
    properties: Mapping[str, float],
)
```

One fluid at one state, and the properties its model fixed there.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature_c` | Temperature, in degrees Celsius. |
| `static_pressure_pa` | Absolute static pressure, in pascals. |
| `composition` | What distinguishes this fluid from another at the same temperature and pressure: the relative humidity of air, the salinity and depth of sea water. Read it to know what was assumed. |
| `model` | The model that produced the properties, named so a result can cite it. |
| `validity` | The domain the model states for itself, in words, or the empty string where it states none. Prose, because sources state these in prose and reducing them to a box loses the conditions attached to them. |
| `properties` | The quantities the model determined, in SI. Reached through the named accessors, which raise [`FluidPropertyUnavailable`](/phonometry/reference/api/fluids/fluids/#fluidpropertyunavailable) for a quantity that is absent. |

### Fluid.characteristic_impedance

*property*

`rho c`, in pascal seconds per metre.

Closed by identity from two quantities the model fixed, so it is
available whenever both of those are.

### Fluid.density

*property*

Density `rho`, in kilograms per cubic metre.

### Fluid.heat_capacity_ratio

*property*

Ratio of specific heats `kappa` (`gamma`), dimensionless.

### Fluid.kinematic_viscosity

*property*

`nu = eta / rho`, in square metres per second.

### Fluid.prandtl_number

*property*

`Pr = eta / (rho alpha_t)`, dimensionless.

A model that prints its own Prandtl number keeps it: a published fit
carries the value it was fitted with, and closing the identity from a
better air would silently change the model rather than correct it. A
model that does not print one has it closed from the three that it did.

### Fluid.specific_heat_capacity

*property*

Specific heat capacity at constant pressure `C_P`, in J/(kg K).

### Fluid.speed_of_sound

*property*

Speed of sound `c`, in metres per second.

### Fluid.thermal_conductivity

*property*

Thermal conductivity `k_a`, in watts per metre kelvin.

### Fluid.thermal_diffusivity

*property*

Thermal diffusivity `alpha_t`, in square metres per second.

### Fluid.viscosity

*property*

Dynamic viscosity `eta`, in pascal seconds.

## FluidAssumptionWarning

A fluid was built from a default the caller did not supply.

The default is a documented value, not a measurement of the caller's air or
water, so it is announced. Passing every argument silences it. Python shows
a warning once per call site by default, and
`warnings.filterwarnings("error", category=FluidAssumptionWarning)` turns
it into a hard failure for anyone who wants one.

## FluidPropertyUnavailable

A quantity the model that built this fluid does not determine.

## FluidWarning

A fluid state outside the domain its model states for itself.
