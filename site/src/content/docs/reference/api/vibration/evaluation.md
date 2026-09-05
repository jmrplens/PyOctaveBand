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

## alarm_limit

```python
alarm_limit(baseline: float, zone_b_upper: float) -> float
```

The ALARM setting a baseline and a zone B/C boundary imply (5.4.1).

The recommendation is two rules at once: set the ALARM a quarter of the
upper limit of zone B above the established baseline, and do not normally
let it exceed 1,25 times that limit. A machine with a low baseline
therefore alarms below zone C, which the clause says in as many words, and
one with a high baseline is capped rather than allowed to drift up with
it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `baseline` | The steady-state baseline for that measurement position and direction, in the unit the boundaries are in. |
| `zone_b_upper` | The upper limit of zone B, which is the B/C boundary. |

**Returns:** The ALARM setting.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the baseline is negative or the limit is not positive. |

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
velocity *rises* with frequency at 6 dB per octave, and above the upper
corner it holds acceleration constant, so it falls at the same rate. The
curve is therefore lowest where the machine is slowest and where it is
fastest, and flat in between. A machine-specific part that states its own
`k` and `m` overrides them.

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

## GEAR_ACCEPTANCE_HEADROOM

*Constant* (`float`).

```python
GEAR_ACCEPTANCE_HEADROOM = 1.25
```

## GEAR_DISPLACEMENT_CORNER_HZ

*Constant* (`float`).

```python
GEAR_DISPLACEMENT_CORNER_HZ = 50.0
```

## GEAR_DISPLACEMENT_SLOPE_DB_PER_DECADE

*Constant* (`float`).

```python
GEAR_DISPLACEMENT_SLOPE_DB_PER_DECADE = 10.0
```

## gear_housing_velocity_limit

```python
gear_housing_velocity_limit(
    frequency: ArrayLike,
    *,
    rating: float,
) -> float | NDArray[np.float64]
```

The housing velocity rating curve of Figure A.2.

Flat between 45 Hz and 1590 Hz and falling outside both corners at 14 dB
per decade, which is the shape [`allowable_velocity`](/phonometry/reference/api/vibration/evaluation/#allowable_velocity) already draws
for ISO 20816-1 Formula (C.1); this is that formula with the corners and
the exponents Part 9 states, so the two parts of the series share one
curve rather than two implementations of it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array). |
| `rating` | The velocity rating `VR`, in millimetres per second. |

**Returns:** The allowable r.m.s. velocity, in millimetres per second; a float for a scalar frequency, otherwise an array.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a frequency or the rating is not positive. |

## gear_shaft_displacement_limit

```python
gear_shaft_displacement_limit(
    frequency: ArrayLike,
    *,
    rating: float,
) -> float | NDArray[np.float64]
```

The shaft displacement rating curve of Figure A.1.

$$
d(f) = \mathrm{DR} \qquad f \leq 50\ \mathrm{Hz}
$$

$$
d(f) = \mathrm{DR}\,(f/50)^{-1/2} \qquad f > 50\ \mathrm{Hz}
$$

The note under the figure states both halves: the rating number is the
displacement of the curve up to 50 Hz, and above 50 Hz the curves decrease
by 10 dB per decade, which on an amplitude is an exponent of one half.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in hertz (scalar or array). |
| `rating` | The displacement rating `DR`, in micrometres. |

**Returns:** The allowable peak-to-peak displacement, in micrometres; a float for a scalar frequency, otherwise an array.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a frequency or the rating is not positive. |

## GEAR_UNIT_CLASSES

*Constant* (`dict`).

```python
GEAR_UNIT_CLASSES = {('I', 'a'): GearUnitRatings(displacement=31.5, velocity=3.15, acceleration=50.0), ('I', 'b_low'): GearUnitRatings(displacement=31.5, velocity=3.15, acceleration=None), ('I', 'b_high'): GearUnitRatings(displacement=50.0, velocity=5.0, acceleration=None), ('II', 'a'): GearUnitRatings(displacement=50.0, velocity=5.0, acceleration=80.0), ('II', 'b_low'): GearUnitRatings(displacement=50.0, velocity=5.0, acceleration=None), ('II', 'b_high'): GearUnitRatings(displacement=80.0, velocity=8.0, acceleration=None), ('III', 'a'): GearUnitRatings(displacement=80.0, velocity=8.0, acceleration=125.0), ('III', 'b_low'): GearUnitRatings(displacement=80.0, velocity=8.0, acceleration=None), ('III', 'b_high'): GearUnitRatings(displacement=125.0, velocity=12.5, acceleration=None), ('IV', 'a'): GearUnitRatings(displacement=125.0, velocity=20.0, acceleration=125.0), ('IV', 'b_low'): GearUnitRatings(displacement=125.0, velocity=12.5, acceleration=None), ('IV', 'b_high'): GearUnitRatings(displacement=200.0, velocity=20.0, acceleration=None)}
```

## gear_unit_zone_boundaries

```python
gear_unit_zone_boundaries(quantity: str, rating: float) -> ZoneBoundaries
```

The three boundaries Table 2, 3 or 4 prints for one rating.

The tables are printed for the rating numbers they list and for no others,
so a rating between two rows is refused rather than interpolated: the
ladder is a choice made with the manufacturer, not a continuum. The
comparison is a relative one to a part in a thousand million, so a rating
that arrived through a computation and carries floating-point noise still
finds its row; nothing a reader would call a different rating does.

**Parameters**

| Name | Description |
| :--- | :--- |
| `quantity` | `"displacement"` (Table 2, shaft relative peak-to-peak, µm), `"velocity"` (Table 3, housing r.m.s., mm/s) or `"acceleration"` (Table 4, housing true peak, m/s²). |
| `rating` | The rating number, `DR`, `VR` or `AR`. |

**Returns:** The A/B, B/C and C/D boundaries of that row.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the quantity is unknown or the table prints no row for that rating. |

## GEAR_UNIT_ZONES

*Constant* (`dict`).

```python
GEAR_UNIT_ZONES = {'displacement': {31.5: ZoneBoundaries(a_b=20.0, b_c=31.5, c_d=50.0), 50.0: ZoneBoundaries(a_b=31.5, b_c=50.0, c_d=80.0), 80.0: ZoneBoundaries(a_b=50.0, b_c=80.0, c_d=125.0), 125.0: ZoneBoundaries(a_b=80.0, b_c=125.0, c_d=200.0), 200.0: ZoneBoundaries(a_b=125.0, b_c=200.0, c_d=315.0)}, 'velocity': {3.15: ZoneBoundaries(a_b=2.0, b_c=3.15, c_d=5.0), 5.0: ZoneBoundaries(a_b=3.15, b_c=5.0, c_d=8.0), 8.0: ZoneBoundaries(a_b=5.0, b_c=8.0, c_d=12.5), 12.5: ZoneBoundaries(a_b=8.0, b_c=12.5, c_d=20.0), 20.0: ZoneBoundaries(a_b=12.5, b_c=20.0, c_d=31.5)}, 'acceleration': {5.0: ZoneBoundaries(a_b=3.15, b_c=5.0, c_d=8.0), 8.0: ZoneBoundaries(a_b=5.0, b_c=8.0, c_d=12.5), 12.5: ZoneBoundaries(a_b=8.0, b_c=12.5, c_d=20.0), 20.0: ZoneBoundaries(a_b=12.5, b_c=20.0, c_d=31.5), 31.5: ZoneBoundaries(a_b=20.0, b_c=31.5, c_d=50.0), 50.0: ZoneBoundaries(a_b=31.5, b_c=50.0, c_d=80.0), 80.0: ZoneBoundaries(a_b=50.0, b_c=80.0, c_d=125.0), 125.0: ZoneBoundaries(a_b=80.0, b_c=125.0, c_d=200.0), 200.0: ZoneBoundaries(a_b=125.0, b_c=200.0, c_d=315.0)}}
```

## GEAR_VELOCITY_CORNERS_HZ

*Constant* (`tuple`).

```python
GEAR_VELOCITY_CORNERS_HZ = (45.0, 1590.0)
```

## GEAR_VELOCITY_SLOPE_DB_PER_DECADE

*Constant* (`float`).

```python
GEAR_VELOCITY_SLOPE_DB_PER_DECADE = 14.0
```

## GearUnitRatings

```python
GearUnitRatings(
    displacement: float,
    velocity: float,
    acceleration: float | None,
)
```

The three rating numbers Table 5 gives one class of gear unit.

**Attributes**

| Name | Description |
| :--- | :--- |
| `displacement` | The displacement rating `DR`, which indexes Table 2. |
| `velocity` | The velocity rating `VR`, which indexes Table 3. |
| `acceleration` | The acceleration rating `AR`, which indexes Table 4, or `None` where the table prints "no information available at this time", which it does for every subclass b) row. |

## industrial_machine_zone

```python
industrial_machine_zone(
    group: str,
    support: str,
    *,
    displacement_um: float | None = None,
    velocity_mm_s: float | None = None,
) -> EvaluationZone
```

Grade an industrial machine against Table A.1 or A.2.

Give whichever quantities were measured. With both, 5.2.3 applies and the
result is the more restrictive of the two gradings, which is the whole
reason the tables state each class twice.

**Parameters**

| Name | Description |
| :--- | :--- |
| `group` | `"group_1"` (above 300 kW and not more than 50 MW, or an electrical machine of shaft height 315 mm or more) or `"group_2"` (above 15 kW up to and including 300 kW, or shaft height from 160 mm up to but not including 315 mm), from 4.2. |
| `support` | `"rigid"` or `"flexible"`; rigid means the lowest natural frequency of the support in the measuring direction is at least 25 % above the main excitation frequency (4.3). |
| `displacement_um` | Measured broad-band r.m.s. displacement of the bearing, pedestal or housing, in micrometres. |
| `velocity_mm_s` | Measured broad-band r.m.s. velocity, in millimetres per second. |

**Returns:** `"A"`, `"B"`, `"C"` or `"D"`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the group or support class is unknown, or neither measured quantity is given. |

## INDUSTRIAL_MACHINE_ZONES

*Constant* (`dict`).

```python
INDUSTRIAL_MACHINE_ZONES = {('group_1', 'rigid'): MachineZoneLimits(displacement_um=ZoneBoundaries(a_b=29.0, b_c=57.0, c_d=90.0), velocity_mm_s=ZoneBoundaries(a_b=2.3, b_c=4.5, c_d=7.1)), ('group_1', 'flexible'): MachineZoneLimits(displacement_um=ZoneBoundaries(a_b=45.0, b_c=90.0, c_d=140.0), velocity_mm_s=ZoneBoundaries(a_b=3.5, b_c=7.1, c_d=11.0)), ('group_2', 'rigid'): MachineZoneLimits(displacement_um=ZoneBoundaries(a_b=22.0, b_c=45.0, c_d=71.0), velocity_mm_s=ZoneBoundaries(a_b=1.4, b_c=2.8, c_d=4.5)), ('group_2', 'flexible'): MachineZoneLimits(displacement_um=ZoneBoundaries(a_b=37.0, b_c=71.0, c_d=113.0), velocity_mm_s=ZoneBoundaries(a_b=2.3, b_c=4.5, c_d=7.1))}
```

## is_significant_change

```python
is_significant_change(change: float, zone_b_upper: float) -> bool
```

Whether a change from the baseline is significant (5.3).

A change exceeding a quarter of the upper limit of zone B, in either
direction, is one that 5.3 asks to be investigated even when zone C of
Criterion I has not been reached: the machine can go wrong well inside
zone B, and the criterion that catches it is the change rather than the
magnitude.

**Parameters**

| Name | Description |
| :--- | :--- |
| `change` | The change from the established baseline, in the unit the boundaries are in; the sign is ignored. |
| `zone_b_upper` | The upper limit of zone B, which is the B/C boundary. |

**Returns:** Whether the change is significant.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the change is not finite or the limit is not positive. |

## MachineZoneLimits

```python
MachineZoneLimits(
    displacement_um: ZoneBoundaries,
    velocity_mm_s: ZoneBoundaries,
)
```

The two boundary sets one machine class is judged on at once.

ISO 10816-3:2009 states each class twice, once in displacement and once
in velocity, and 5.2.3 says which wins when they disagree: the more
restrictive zone applies. Both are read on the same non-rotating part,
and A.1 says velocity alone is enough in most cases and that a spectrum
expected to carry low-frequency components should be judged on both.

**Attributes**

| Name | Description |
| :--- | :--- |
| `displacement_um` | The three boundaries in r.m.s. displacement, micrometres. |
| `velocity_mm_s` | The three boundaries in r.m.s. velocity, millimetres per second. |

## OPERATIONAL_LIMIT_HEADROOM

*Constant* (`float`).

```python
OPERATIONAL_LIMIT_HEADROOM = 1.25
```

## SIGNIFICANT_CHANGE_FRACTION

*Constant* (`float`).

```python
SIGNIFICANT_CHANGE_FRACTION = 0.25
```

## trip_limit

```python
trip_limit(zone_c_upper: float) -> float
```

The largest TRIP setting 5.4.2 recommends.

Unlike an ALARM, a TRIP is not set from a baseline: it relates to the
mechanical integrity of the machine and is the same for every machine of
a design. The clause declines to give absolute values and gives a ceiling
instead, 1,25 times the upper limit of zone C.

**Parameters**

| Name | Description |
| :--- | :--- |
| `zone_c_upper` | The upper limit of zone C, which is the C/D boundary. |

**Returns:** The recommended maximum TRIP setting.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the limit is not positive. |

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
