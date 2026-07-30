---
title: "environmental.cnossos_road"
description: "CNOSSOS-EU road traffic source emission (Directive 2002/49/EC Annex II, 2.2)."
sidebar:
  label: "cnossos_road"
---

CNOSSOS-EU road traffic source emission (Directive 2002/49/EC Annex II, 2.2).

The common noise assessment methods of the European Union describe a road as an
incoherent **source line** of point sources 0,05 m above the pavement. Each
vehicle category `m` radiates a directional sound power per metre of line

`L'_W,eq,line,i,m = L_W,i,m + 10 lg( Q_m / (1000 v_m) )`   (2.2.1)

built from a **rolling** term (2.2.4) and a **propulsion** term (2.2.11), each
with its own corrections for road surface, air temperature, studded tyres, road
gradient and the acceleration or deceleration near a junction. This module
implements the whole of 2.2 together with the coefficient database of
Appendix F, in the eight octave bands from 63 Hz to 8 kHz.

Which text is implemented
-------------------------
Annex II was replaced by Commission Directive (EU) 2015/996, corrected by the
corrigendum of OJ L 5, 10.1.2018 and amended by Commission Delegated Directive
(EU) 2021/1226. The consolidated text (02002L0049) is what is implemented here:

* the equations of 2.2.2 to 2.2.6 come unchanged from 2015/996;
* the octave-band range of 2.2.1 is **63 Hz to 8 kHz** as corrected in 2018
  (the 2015 text said "125 Hz to 4 kHz", which never matched Appendix F);
* Tables F-1 and F-4 are the versions **replaced by 2021/1226**; Tables F-2 and
  F-3 are unchanged since 2015/996.

Each shipped table records in its comment which instrument it comes from. The
superseded 2015/996 coefficients are not shipped, but any coefficient set can be
supplied through [`RoadEmissionCoefficients`](/phonometry/reference/api/environment/cnossos-road/#roademissioncoefficients) and
[`RoadSurfaceCoefficients`](/phonometry/reference/api/environment/cnossos-road/#roadsurfacecoefficients), which is also how a Member State substitutes
its own national database (the reason Appendix F is called a *database* and not
a table of constants).

Scope
-----
This is the **emission** stage only. It produces the source power that the
propagation stage consumes; splitting the source line into point sources is
explicitly outside the scope of the method (2.5.3), so
[`line_source_segment_power`](/phonometry/reference/api/environment/cnossos-road/#line_source_segment_power) is offered as plain arithmetic, clearly
labelled as such. The CNOSSOS propagation model itself is *not* ISO 9613-2; the
hand-off to [`predicted_receiver_level`](/phonometry/reference/api/environment/outdoor-propagation/#predicted_receiver_level)
mixes two methods and is a convenience, not a normative chain.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## CNOSSOS_A_WEIGHTING

*Constant* (`tuple`).

```python
CNOSSOS_A_WEIGHTING = (-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1)
```

## JunctionType

```python
JunctionType(*values)
```

Junction types `k` of Table F-3 for the speed-variation correction.

## line_source_segment_power

```python
line_source_segment_power(
    line_power: NDArray[np.float64] | list[float] | float,
    length: float,
) -> NDArray[np.float64]
```

Sound power of the point source representing a segment of source line.

`L_W,segment,i = L'_W,eq,line,i + 10 lg(dL)`. This is arithmetic, not a
normative rule: section 2.5.3 of Annex II states that how a line source is
split into equivalent point sources "is outside the scope of the current
methodology". Only the per-metre line power is defined by the method.

**Parameters**

| Name | Description |
| :--- | :--- |
| `line_power` | Per-metre line power `L'_W,eq,line,i`, in dB re 1 pW per metre. |
| `length` | Length `dL` of the segment, in m. |

**Returns:** The segment sound power, in dB re 1 pW.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `length` is not positive. |

## ROAD_COEFFICIENTS

*Constant* (`phonometry.environmental.cnossos_road.RoadEmissionCoefficients`).

## ROAD_OCTAVE_BANDS

*Constant* (`tuple`).

```python
ROAD_OCTAVE_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
```

## road_propulsion_noise

```python
road_propulsion_noise(
    category: RoadVehicleCategory | str,
    speed: float,
    *,
    surface: RoadSurface | str | RoadSurfaceCoefficients = ...,
    gradient: float = 0.0,
    junction_distance: float | None = None,
    junction_type: JunctionType = ...,
    coefficients: RoadEmissionCoefficients = ...,
) -> NDArray[np.float64]
```

Propulsion-noise sound power `L_WP,i,m` of one vehicle (2.2.11)/(2.2.12).

`L_WP,i,m = A_P,i,m + B_P,i,m (v_m - v_ref)/v_ref + dL_WP,i,m` with
`dL_WP` collecting the road surface (2.2.20), the road gradient
(2.2.13)-(2.2.16) and the junction (2.2.18). Unlike rolling noise, the
surface term is `min{alpha_i,m ; 0}`: an absorbing surface reduces
propulsion noise, a noisy one does not increase it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `category` | Vehicle category (Table [2.2.a]). |
| `speed` | Average speed `v_m`, in km/h; values below 20 km/h are raised to 20 km/h (2.2.1). |
| `surface` | Road surface, as a [`RoadSurface`](/phonometry/reference/api/environment/cnossos-road/#roadsurface), its description or an explicit [`RoadSurfaceCoefficients`](/phonometry/reference/api/environment/cnossos-road/#roadsurfacecoefficients). |
| `gradient` | Road slope `s`, in per cent, positive uphill in the direction of travel. For a bidirectional flow, split the flow in two and correct one half uphill and the other downhill. |
| `junction_distance` | Distance `x` to the junction, in m. |
| `junction_type` | Junction type `k` (Table F-3). |
| `coefficients` | The Appendix F database to use. |

**Returns:** `L_WP,i,m` over the eight octave bands, in dB re 1 pW.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

## ROAD_REFERENCE_SPEED

*Constant* (`float`).

```python
ROAD_REFERENCE_SPEED = 70.0
```

## ROAD_REFERENCE_TEMPERATURE

*Constant* (`float`).

```python
ROAD_REFERENCE_TEMPERATURE = 20.0
```

## road_rolling_noise

```python
road_rolling_noise(
    category: RoadVehicleCategory | str,
    speed: float,
    *,
    surface: RoadSurface | str | RoadSurfaceCoefficients = ...,
    temperature: float = 20.0,
    studded_fraction: float = 0.0,
    studded_months: float = 0.0,
    junction_distance: float | None = None,
    junction_type: JunctionType = ...,
    coefficients: RoadEmissionCoefficients = ...,
) -> NDArray[np.float64]
```

Rolling-noise sound power `L_WR,i,m` of one vehicle (2.2.4)/(2.2.5).

`L_WR,i,m = A_R,i,m + B_R,i,m lg(v_m/v_ref) + dL_WR,i,m` with the
correction term `dL_WR` collecting the road surface (2.2.19), the studded
tyres (2.2.8), the junction (2.2.17) and the air temperature (2.2.10).
Categories 4a and 4b have no rolling noise: their Table F-1 rows are zero,
and their total power is the propulsion term alone (2.2.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `category` | Vehicle category (Table [2.2.a]). |
| `speed` | Average speed `v_m`, in km/h; values below 20 km/h are raised to 20 km/h (2.2.1). |
| `surface` | Road surface, as a [`RoadSurface`](/phonometry/reference/api/environment/cnossos-road/#roadsurface), its description or an explicit [`RoadSurfaceCoefficients`](/phonometry/reference/api/environment/cnossos-road/#roadsurfacecoefficients). |
| `temperature` | Air temperature `tau`, in degrees Celsius. |
| `studded_fraction` | `Q_stud,ratio` of (2.2.7). |
| `studded_months` | `T_s` of (2.2.7), the months per year over which studded tyres are in use. |
| `junction_distance` | Distance `x` to the junction, in m. |
| `junction_type` | Junction type `k` (Table F-3). |
| `coefficients` | The Appendix F database to use. |

**Returns:** `L_WR,i,m` over the eight octave bands, in dB re 1 pW.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

## ROAD_SOURCE_HEIGHT

*Constant* (`float`).

```python
ROAD_SOURCE_HEIGHT = 0.05
```

## road_source_power

```python
road_source_power(
    traffic: RoadTraffic | list[RoadTraffic] | tuple[RoadTraffic, ...],
    *,
    surface: RoadSurface | str | RoadSurfaceCoefficients = ...,
    temperature: float = 20.0,
    gradient: float = 0.0,
    studded_months: float = 0.0,
    junction_distance: float | None = None,
    junction_type: JunctionType = ...,
    coefficients: RoadEmissionCoefficients = ...,
) -> RoadEmissionResult
```

Directional sound power per metre of a road source line (2.2.1).

Evaluates `L'_W,eq,line,i,m = L_W,i,m + 10 lg(Q_m/(1000 v_m))` for every
category of the traffic mix and sums the categories energetically. The
flow term uses the true average speed even where the sound power itself is
frozen at 20 km/h (2.2.1).

All corrections other than the per-category traffic are properties of the
road segment, so they are passed once here: the surface, the air
temperature, the gradient, the studded-tyre season and the junction.

**Parameters**

| Name | Description |
| :--- | :--- |
| `traffic` | One [`RoadTraffic`](/phonometry/reference/api/environment/cnossos-road/#roadtraffic) or a sequence of them, at most one per vehicle category. |
| `surface` | Road surface (Table F-4). |
| `temperature` | Yearly average air temperature `tau`, in degrees Celsius (the reference condition is 20 degC). |
| `gradient` | Road slope `s`, in per cent, positive uphill. |
| `studded_months` | `T_s` of (2.2.7), the months per year over which studded tyres are in use. |
| `junction_distance` | Distance `x` from the source to the nearest junction, in m; `None` (the default) means no junction in range. |
| `junction_type` | Junction type `k` (Table F-3). |
| `coefficients` | The Appendix F database to use. |

**Returns:** A [`RoadEmissionResult`](/phonometry/reference/api/environment/cnossos-road/#roademissionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the traffic is empty, repeats a category, or carries an invalid flow rate or speed. |

## road_surface_coefficients

```python
road_surface_coefficients(
    surface: RoadSurface | str,
) -> RoadSurfaceCoefficients
```

Look up a Table F-4 row.

**Parameters**

| Name | Description |
| :--- | :--- |
| `surface` | A [`RoadSurface`](/phonometry/reference/api/environment/cnossos-road/#roadsurface) member or its description string. |

**Returns:** The [`RoadSurfaceCoefficients`](/phonometry/reference/api/environment/cnossos-road/#roadsurfacecoefficients) of that surface.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the surface is not tabulated in Table F-4. |

## road_vehicle_sound_power

```python
road_vehicle_sound_power(
    category: RoadVehicleCategory | str,
    speed: float,
    *,
    surface: RoadSurface | str | RoadSurfaceCoefficients = ...,
    temperature: float = 20.0,
    gradient: float = 0.0,
    studded_fraction: float = 0.0,
    studded_months: float = 0.0,
    junction_distance: float | None = None,
    junction_type: JunctionType = ...,
    coefficients: RoadEmissionCoefficients = ...,
) -> NDArray[np.float64]
```

Total sound power `L_W,i,m` of a single vehicle (2.2.2)/(2.2.3).

For categories 1, 2 and 3 the rolling and propulsion terms are summed
energetically (2.2.2); for the powered two-wheelers of category 4 the
propulsion term is the whole of it (2.2.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `category` | Vehicle category (Table [2.2.a]). |
| `speed` | Average speed `v_m`, in km/h. |
| `surface` | Road surface (Table F-4). |
| `temperature` | Air temperature `tau`, in degrees Celsius. |
| `gradient` | Road slope `s`, in per cent. |
| `studded_fraction` | `Q_stud,ratio` of (2.2.7). |
| `studded_months` | `T_s` of (2.2.7), in months. |
| `junction_distance` | Distance `x` to the junction, in m. |
| `junction_type` | Junction type `k` (Table F-3). |
| `coefficients` | The Appendix F database to use. |

**Returns:** `L_W,i,m` over the eight octave bands, in dB re 1 pW.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

## RoadEmissionCoefficients

```python
RoadEmissionCoefficients(
    rolling_a: dict[str, tuple[float, ...]],
    rolling_b: dict[str, tuple[float, ...]],
    propulsion_a: dict[str, tuple[float, ...]],
    propulsion_b: dict[str, tuple[float, ...]],
    studded_a: tuple[float, ...],
    studded_b: tuple[float, ...],
    junction_c: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    temperature_k: dict[str, float],
)
```

Appendix F coefficient database for the road source.

Holds Tables F-1, F-2 and F-3 together with the temperature coefficients
`K_m` of (2.2.10). The default instance,
[`ROAD_COEFFICIENTS`](/phonometry/reference/api/environment/cnossos-road/#road_coefficients), carries the tables of the consolidated
Directive; supplying another instance is how a Member State substitutes a
national database, and how the superseded 2015/996 coefficients can be
reproduced for comparison with pre-2021 studies.

**Attributes**

| Name | Description |
| :--- | :--- |
| `rolling_a` | `A_R,i,m` per category, eight octave bands, in dB. |
| `rolling_b` | `B_R,i,m` per category, eight octave bands, in dB. |
| `propulsion_a` | `A_P,i,m` per category, eight octave bands, in dB. |
| `propulsion_b` | `B_P,i,m` per category, eight octave bands, in dB. |
| `studded_a` | `a_i` of Table F-2 (category 1 only), in dB. |
| `studded_b` | `b_i` of Table F-2 (category 1 only), in dB. |
| `junction_c` | `(C_R, C_P)` of Table F-3 per category and junction type, in dB. |
| `temperature_k` | `K_m` of (2.2.10), in dB per degree Celsius. |

## RoadEmissionResult

```python
RoadEmissionResult(
    frequencies: NDArray[np.float64],
    categories: tuple[RoadVehicleCategory, ...],
    rolling: NDArray[np.float64],
    propulsion: NDArray[np.float64],
    vehicle_power: NDArray[np.float64],
    line_power: NDArray[np.float64],
    total_line_power: NDArray[np.float64],
    source_height: float = 0.05,
)
```

Directional sound power per metre of a CNOSSOS-EU road source line.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band midband frequencies, in Hz (63 Hz to 8 kHz). |
| `categories` | The vehicle categories of the modelled flows, in the order of the rows of the per-category arrays. |
| `rolling` | `L_WR,i,m` per category and band, in dB re 1 pW. The row of a powered two-wheeler is the zero row Table F-1 prints for it, and does **not** enter its sound power, which is the propulsion term alone (2.2.3). |
| `propulsion` | `L_WP,i,m` per category and band, in dB re 1 pW. |
| `vehicle_power` | `L_W,i,m` per category and band (2.2.2)/(2.2.3), in dB re 1 pW. |
| `line_power` | `L'_W,eq,line,i,m` per category and band (2.2.1), in dB re 1 pW per metre. |
| `total_line_power` | The energy sum of `line_power` over the categories, in dB re 1 pW per metre: the source strength of the line. |
| `source_height` | Height of the equivalent point source above the road surface, in m (0,05 m, fixed by 2.2.1). |

### RoadEmissionResult.a_weighted_line_power

*property*

A-weighted total line power, in dB(A) re 1 pW per metre.

Uses the octave-band weighting `AWC_f,i` printed in 2.5.5 as amended
by (EU) 2021/1226, i.e. [`CNOSSOS_A_WEIGHTING`](/phonometry/reference/api/environment/cnossos-road/#cnossos_a_weighting).

### RoadEmissionResult.plot()

```python
RoadEmissionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-metre line power spectrum and its category breakdown.

## RoadSurface

```python
RoadSurface(*values)
```

Road surfaces tabulated in Table F-4 (as replaced by 2021/1226).

## RoadSurfaceCoefficients

```python
RoadSurfaceCoefficients(
    name: str,
    alpha: dict[str, tuple[float, ...]],
    beta: dict[str, float],
    speed_range: tuple[float, float] | None,
)
```

One row of Table F-4: the acoustic signature of a road surface.

**Attributes**

| Name | Description |
| :--- | :--- |
| `name` | The surface description as printed in Table F-4. |
| `alpha` | `alpha_i,m` per category, eight octave bands, in dB. |
| `beta` | `beta_m` per category (dimensionless speed coefficient). |
| `speed_range` | `(v_min, v_max)` over which the row is declared valid, in km/h, or `None` for the reference surface, which carries no range. |

## RoadTraffic

```python
RoadTraffic(
    category: RoadVehicleCategory,
    flow_rate: float,
    speed: float,
    studded_fraction: float = 0.0,
)
```

The traffic of one vehicle category on a source line.

**Attributes**

| Name | Description |
| :--- | :--- |
| `category` | The [`RoadVehicleCategory`](/phonometry/reference/api/environment/cnossos-road/#roadvehiclecategory) of the flow. |
| `flow_rate` | Hourly flow `Q_m`, in vehicles per hour. |
| `speed` | Average speed `v_m`, in km/h. Sound powers are frozen at 20 km/h below that speed (2.2.1); the flow term still uses the true speed. |
| `studded_fraction` | `Q_stud,ratio` of (2.2.7), the fraction of the light-vehicle flow fitted with studded tyres over the studded period. Ignored for every category other than category 1 (2.2.9). |

## RoadVehicleCategory

```python
RoadVehicleCategory(*values)
```

Vehicle categories of Table [2.2.a].

Categories 1 to 4 are mandatory; the "open" category 5 has no coefficients
in Appendix F and is therefore not modelled.
