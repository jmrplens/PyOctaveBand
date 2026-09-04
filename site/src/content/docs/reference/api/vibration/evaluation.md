---
title: "vibration.machinery.evaluation"
description: "Evaluation of machine vibration by measurement (ISO 20816-1:2016)."
sidebar:
  label: "evaluation"
---

Evaluation of machine vibration by measurement (ISO 20816-1:2016).

Condition monitoring answers two questions, and they are not the same one.
[`diagnostics`](/phonometry/reference/api/vibration/diagnostics/) answers *where a fault
would show*, by turning the geometry of a bearing or a gear pair into the
frequencies it excites. This module answers the prior question a plant
actually asks: **is this machine acceptable at all**, from one broad-band
magnitude measured at a bearing.

ISO 20816-1 is the basis document of the series, merging what used to be
ISO 10816-1 and ISO 7919-1. It fixes the shape of the answer and leaves the
numbers to the machine-specific parts.

**Four evaluation zones** (6.3.2.3) grade a machine rather than pass or fail
it. Zone **A** is where newly commissioned machines normally fall; zone **B**
is acceptable for unrestricted long-term operation; zone **C** is
unsatisfactory for long-term continuous running, though the machine may run a
limited period until remedial action can be arranged; zone **D** is severe
enough to cause damage. Three boundaries separate them, and
[`evaluation_zone`](/phonometry/reference/api/vibration/evaluation/#evaluation_zone) is the comparison itself, blind to the quantity: the
specific parts set boundaries on shaft displacement, housing velocity or
housing acceleration, and the grading is the same in all three.

**Criterion I** (6.3.2) is that comparison applied to the vibration severity,
the largest broad-band magnitude measured at any bearing at rated speed.
Velocity carries it over a wide speed range, but a single velocity limit
regardless of frequency allows unacceptable displacement at low frequency and
unacceptable acceleration at high frequency. So the criterion is a curve, flat
between two corner frequencies and sloped outside them
(Figure 9, Formula (C.1)):

$$
v_\mathrm{rms} = v_A \, Z_\mathrm{bound} \left(\frac{f_z}{f_x}\right)^{k} \left(\frac{f_y}{f_w}\right)^{m} \tag{C.1}
$$

with $f_z = f$ below the lower corner $f_x$ and $f_x$ above
it, and $f_w = f$ above the upper corner $f_y$ and $f_y$
below it, so both bracketed factors are unity between the corners.
$Z_\mathrm{bound}$ moves the one curve onto the three boundaries, and
Annex C.2 prints the factors it takes: 1 for the limit of zone A, **2,56** for
zone B and **6,4** for zone C.

**Criterion II** (6.3.3) judges a *change* from an established baseline, and a
change is a vector. Annex D makes the point with a machine whose magnitude
fell from 3 mm/s to 2,5 mm/s while its phase swung from 40° to 180°: the
magnitude moved by half a millimetre per second, and the vibration itself
moved by **5,2 mm/s**, ten times as much. [`vibration_vector_change`](/phonometry/reference/api/vibration/evaluation/#vibration_vector_change) is
that subtraction.

Where no part of the series covers a machine and no experience is available,
Annex C.1 offers Table C.1: a ladder of preferred magnitudes and the range
each boundary is typically drawn from, with small machines at the low end and
large flexibly supported ones at the high end. They are a starting point for
agreement between supplier and customer, not an acceptance specification.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## allowable_velocity

```python
allowable_velocity(
    frequency: ArrayLike,
    *,
    constant_velocity_mm_s: float,
    zone_factor: float = 1.0,
    corner_low_hz: float,
    corner_high_hz: float,
    exponent_low: float = 1.0,
    exponent_high: float = 1.0,
) -> np.ndarray | float
```

Frequency-shaped velocity criterion of Figure 9 (Formula (C.1)).

Flat between the two corner frequencies and sloped outside them. The
default exponents of 1 are the physical reading of that shape: below the
lower corner the criterion holds displacement constant, so the allowable
velocity falls with frequency, and above the upper corner it holds
acceleration constant, so it falls with the reciprocal. A machine-specific
part that states its own `k` and `m` overrides them.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array). |
| `constant_velocity_mm_s` | The constant r.m.s. velocity `vA` that applies between the corners for zone A, in millimetres per second. |
| `zone_factor` | `Zbound`, the factor that moves the curve onto a zone limit; see [`ZONE_LIMIT_FACTORS`](/phonometry/reference/api/vibration/evaluation/#zone_limit_factors) for the 1 / 2,56 / 6,4 of Annex C.2. |
| `corner_low_hz` | The lower corner `fx`, in hertz. |
| `corner_high_hz` | The upper corner `fy`, in hertz; must exceed `corner_low_hz`. |
| `exponent_low` | `k`, the slope below the lower corner. |
| `exponent_high` | `m`, the slope above the upper corner. |

**Returns:** The allowable r.m.s. velocity, in millimetres per second; a float for a scalar frequency, otherwise an array.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a frequency is not positive, a velocity or corner is not positive and finite, or the corners are not in order. |

## evaluation_zone

```python
evaluation_zone(
    magnitude: ArrayLike,
    boundaries: ZoneBoundaries,
) -> EvaluationZone | NDArray[np.str_]
```

Grade a vibration magnitude into zone A, B, C or D (6.3.2.3).

The boundaries belong to a zone each: a magnitude exactly on the A/B
boundary is the limit of zone A and is graded `"A"`, which is how a
limit reads in the tables of the machine-specific parts.

**Parameters**

| Name | Description |
| :--- | :--- |
| `magnitude` | The vibration severity, in the same quantity and unit as `boundaries` (scalar or array). |
| `boundaries` | The three zone boundaries of the applicable part of ISO 20816. |

**Returns:** `"A"`, `"B"`, `"C"` or `"D"`; a string for a scalar input, otherwise an array of them.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a magnitude is negative or not finite. |

## TYPICAL_BOUNDARY_LADDER_MM_S

*Constant* (`tuple`).

```python
TYPICAL_BOUNDARY_LADDER_MM_S = (0.28, 0.45, 0.71, 1.12, 1.8, 2.8, 4.5, 7.1, 9.3, 11.2, 14.7, 18.0, 28.0, 45.0)
```

## TYPICAL_ZONE_BOUNDARY_RANGES_MM_S

*Constant* (`dict`).

```python
TYPICAL_ZONE_BOUNDARY_RANGES_MM_S = {'A/B': (0.71, 4.5), 'B/C': (1.8, 9.3), 'C/D': (4.5, 14.7)}
```

## VectorChangeResult

```python
VectorChangeResult(
    magnitude: float,
    phase_deg: float,
    initial: tuple[float, float],
    final: tuple[float, float],
)
```

A change in vibration between two states, as a vector (Annex D).

**Attributes**

| Name | Description |
| :--- | :--- |
| `magnitude` | The magnitude of the change, in the unit of the two states it was built from. |
| `phase_deg` | The direction of the change, in degrees within [0, 360). |
| `initial` | The magnitude and phase of the initial state. |
| `final` | The magnitude and phase of the final state. |

### VectorChangeResult.magnitude_change

*property*

What a magnitude-only comparison would have reported.

The difference of the two magnitudes, signed. Annex D exists because
this number and `magnitude` can disagree by an order of
magnitude, and only the second is the change in the vibration.

### VectorChangeResult.plot()

```python
VectorChangeResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the polar diagram of Figure D.1.

The two states as vectors from the origin and the change as the vector
joining their tips, which is the picture that makes the point.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing polar axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | `unit`, the name of the unit the two states were given in, and anything forwarded to the change chord; see `phonometry._plot.vibration.plot_vector_change`. |

## vibration_vector_change

```python
vibration_vector_change(
    initial_magnitude: float,
    initial_phase_deg: float,
    final_magnitude: float,
    final_phase_deg: float,
) -> VectorChangeResult
```

The vector change in vibration between two steady states (Annex D).

Criterion II is written on a change from an established baseline, and a
broad-band magnitude cannot express one: a component that swings in phase
changes the vibration even as the magnitude it contributes falls. Annex D
prints the case, 3 mm/s at 40 degrees becoming 2,5 mm/s at 180 degrees,
where the magnitude drops by half a millimetre per second and the
vibration moves by 5,2.

**Parameters**

| Name | Description |
| :--- | :--- |
| `initial_magnitude` | Magnitude of the reference state, in any unit; the result carries the same one. |
| `initial_phase_deg` | Phase of the reference state, in degrees. |
| `final_magnitude` | Magnitude of the later state, in the same unit. |
| `final_phase_deg` | Phase of the later state, in degrees. |

**Returns:** The [`VectorChangeResult`](/phonometry/reference/api/vibration/evaluation/#vectorchangeresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a magnitude is negative or a value is not finite. |

## ZONE_LIMIT_FACTORS

*Constant* (`dict`).

```python
ZONE_LIMIT_FACTORS = {'A': 1.0, 'B': 2.56, 'C': 6.4}
```

## ZoneBoundaries

```python
ZoneBoundaries(a_b: float, b_c: float, c_d: float)
```

The three magnitudes that separate the four evaluation zones.

The unit is whichever the machine-specific part states: micrometres of
shaft displacement, millimetres per second of housing velocity or metres
per second squared of housing acceleration. Nothing here converts between
them, and the boundaries and the magnitude judged against them have to be
the same quantity.

**Attributes**

| Name | Description |
| :--- | :--- |
| `a_b` | The zone A/B boundary, below which a newly commissioned machine normally sits. |
| `b_c` | The zone B/C boundary, the limit of unrestricted long-term operation. |
| `c_d` | The zone C/D boundary, above which the vibration is severe enough to damage the machine. |

### ZoneBoundaries.as_tuple

*property*

The three boundaries in order, for a plot or a table.
