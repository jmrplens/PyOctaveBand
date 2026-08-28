---
title: "aircraft.flight_performance"
description: "ECAC Doc 29 flight performance: procedural steps into a flight profile."
sidebar:
  label: "flight_performance"
---

ECAC Doc 29 flight performance: procedural steps into a flight profile.

A published departure or arrival procedure is not a trajectory. It is a list of
*procedural steps* -- "climb at take-off thrust to 1500 ft", "accelerate to
210.6 kt at 984.3 ft/min", "descend on a 3 degree slope from 3000 ft at 180 kt"
-- and the aeroplane's own aerodynamic and engine coefficients. **ECAC Doc 29
5th ed., Volume 2, Appendix B** is the flight-mechanics model that turns the one
into the other: a *flight profile*, an ordered list of profile points carrying
distance along the ground track, height above the aerodrome, true airspeed and
corrected net thrust per engine. Corrected net thrust is what the NPD tables of
[`phonometry.aircraft.airport_noise`](/phonometry/reference/api/aeroacoustics/airport-noise/) are indexed on, so this model is what
stands between a published procedure and a noise contour.

* [`Aerodrome`](/phonometry/reference/api/aeroacoustics/flight-performance/#aerodrome) -- the aerodrome and its weather, and the five atmosphere
  ratios of B3 that every equation below reads.
* [`PerformanceAircraft`](/phonometry/reference/api/aeroacoustics/flight-performance/#performanceaircraft) with [`JetEngineCoefficients`](/phonometry/reference/api/aeroacoustics/flight-performance/#jetenginecoefficients),
  [`PropellerEngineCoefficients`](/phonometry/reference/api/aeroacoustics/flight-performance/#propellerenginecoefficients) and [`AerodynamicCoefficients`](/phonometry/reference/api/aeroacoustics/flight-performance/#aerodynamiccoefficients) --
  the ANP coefficient tables the equations take their constants from.
* [`DepartureStep`](/phonometry/reference/api/aeroacoustics/flight-performance/#departurestep) and [`ApproachStep`](/phonometry/reference/api/aeroacoustics/flight-performance/#approachstep) -- one row each of a
  published procedure.
* [`departure_profile`](/phonometry/reference/api/aeroacoustics/flight-performance/#departure_profile) and [`approach_profile`](/phonometry/reference/api/aeroacoustics/flight-performance/#approach_profile) -- the model, returning
  a [`FlightProfile`](/phonometry/reference/api/aeroacoustics/flight-performance/#flightprofile) of [`ProfilePoint`](/phonometry/reference/api/aeroacoustics/flight-performance/#profilepoint).

Units are the standard's and they are English throughout (B2): feet, knots,
pounds, pounds of thrust per engine, degrees Celsius in the thrust equations and
inches of mercury for pressure. Doc 29 keeps them "due to the history of the
overarching method [...] and the strong association that aviation has with
English units", and pins two conversion constants at deliberately imprecise
legacy values that must not be improved (footnotes 30 and 31, folios B-7/B-8).

Departures run forward from brake release and arrivals run **backwards** from
touchdown, which is why an arrival profile carries negative distances until the
aeroplane is on the runway (folio B-5).

Source (clean-room, implemented from the published standard): ECAC.CEAC Doc 29,
5th edition, Volume 2 "Technical Guide", Appendix B, folios B-1 to B-49.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## Aerodrome

```python
Aerodrome(
    elevation_ft: float,
    temperature_c: float = 15.0,
    sea_level_pressure_inhg: float = 29.92,
    headwind_kt: float = 8.0,
    runway_gradient: float = 0.0,
)
```

The aerodrome, its weather and the runway a procedure is flown from.

Every atmosphere ratio of B3 is a function of altitude above mean sea level
given these five numbers, and every equation of Appendix B reads at least
one of them.

**Attributes**

| Name | Description |
| :--- | :--- |
| `elevation_ft` | Aerodrome elevation above mean sea level `Eapt`, ft. |
| `temperature_c` | Air temperature at the aerodrome `Tapt`, in degC. Doc 29 writes it in degF; it is taken in degC here because that is what the reference cases, the thrust equations and the rest of this package use, and converted on the way in. |
| `sea_level_pressure_inhg` | Aerodrome pressure reduced to sea level `Papt` -- the QNH, not the pressure at the field -- in inHg. |
| `headwind_kt` | Headwind component `w`, kt; negative for a tailwind. Defaults to Doc 29's own modelling default of 8 kt (B4.4). |
| `runway_gradient` | Runway gradient `GR`, positive uphill, dimensionless: the rise over the run between the two runway ends. |

The validity envelope Doc 29 claims for the coefficients is "air
temperatures up to 43 degrees C, aerodrome altitudes up to 6,000 ft and
across the range of weights specified in the ANP database" (B1). Nothing
here enforces it: outside it the equations still evaluate, and the
coefficients, not the arithmetic, are what stop being adequate.

### Aerodrome.calibrated_airspeed_kt()

```python
Aerodrome.calibrated_airspeed_kt(
    true_airspeed_kt: float,
    altitude_ft: float,
) -> float
```

Calibrated airspeed from a true one at *altitude_ft* (Eq. B-8), kt.

### Aerodrome.density_ratio()

```python
Aerodrome.density_ratio(altitude_ft: float) -> float
```

Density ratio `sigma` at *altitude_ft* above MSL (Eq. B-5).

The ratio calibrated and true airspeed differ by: Eq. B-7 divides a
calibrated airspeed by its square root to get the true one.

### Aerodrome.pressure_altitude_ft()

```python
Aerodrome.pressure_altitude_ft(altitude_ft: float) -> float
```

Pressure altitude `h` for *altitude_ft* above MSL, ft (Eq. B-6).

The altitude the standard atmosphere would put this pressure at, which
is what Eq. B-9's `Ga h` and `Gb h^2` terms read -- not the
geometric altitude. The two coincide only at a QNH of exactly 29.92
inHg; at 30.71 inHg over a sea-level aerodrome the aeroplane sits at
0 ft and flies at a pressure altitude of -723 ft.

### Aerodrome.pressure_ratio()

```python
Aerodrome.pressure_ratio(altitude_ft: float) -> float
```

Pressure ratio `delta` at *altitude_ft* above MSL (Eq. B-4).

Ambient pressure over 29.92 inHg. Every force balance in Appendix B
divides the weight by it, because `W/delta` is the weight the thrust
equations' *corrected* thrust has to lift.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | above the tropopause the lapsed temperature Eq. B-4 divides by reaches absolute zero and the bracket turns negative, where raising it to a fractional power leaves the reals; Python answers that with a complex number rather than an error, and a complex pressure ratio propagates silently into every thrust of the profile. Doc 29 claims the coefficients only up to 6,000 ft anyway. |

### Aerodrome.temperature_c_at()

```python
Aerodrome.temperature_c_at(altitude_ft: float) -> float
```

Air temperature at *altitude_ft* above mean sea level, degC.

Eq. B-2 lapses from the temperature *at the aerodrome*, not from a
sea-level value: at field elevation the temperature is `Tapt` however
high the field is. Eq. B-9 reads this as its `T`.

### Aerodrome.temperature_ratio()

```python
Aerodrome.temperature_ratio(altitude_ft: float) -> float
```

Temperature ratio `theta` at *altitude_ft* above MSL (Eq. B-3).

Air temperature at the aeroplane over standard sea-level temperature,
both absolute. Eq. B-16 reads it directly and Eq. B-5 divides by it.

### Aerodrome.true_airspeed_kt()

```python
Aerodrome.true_airspeed_kt(
    calibrated_airspeed_kt: float,
    altitude_ft: float,
) -> float
```

True airspeed from a calibrated one at *altitude_ft* (Eq. B-7), kt.

## AerodynamicCoefficients

```python
AerodynamicCoefficients(
    drag_ratio: float,
    ground_roll_coefficient: float | None = None,
    speed_coefficient: float | None = None,
)
```

One ANP `Aerodynamic_Coefficients` row: a flap configuration.

**Attributes**

| Name | Description |
| :--- | :--- |
| `drag_ratio` | `R`, the drag-over-lift ratio of the configuration, dimensionless. Every force balance in Appendix B carries it. |
| `ground_roll_coefficient` | `B`, ft/lb, of Eq. B-16, or `None` for a configuration no take-off is flown in. |
| `speed_coefficient` | The take-off speed coefficient `C` of Eq. B-15 on a departure and the landing speed coefficient `D` of Eq. B-75 on an arrival, kt/sqrt(lb), or `None` for a configuration that is neither taken off nor landed in. One field for the two because no flap configuration is ever both: the ANP table keys them by operation and fills the matching column, and Doc 29 Volume 3 merges the pair into a single `C/D` column for the same reason. |

A missing coefficient is `None`, the dash the printed table prints, not a
zero: a zero `B` is a take-off with no ground roll at all.

## approach_profile

```python
approach_profile(
    aircraft: PerformanceAircraft,
    steps: Sequence[ApproachStep],
    *,
    aerodrome: Aerodrome,
    weight_lb: float | None = None,
    procedure_id: str = '',
) -> FlightProfile
```

Fly an approach procedure's steps into a flight profile (Doc 29 B7).

Approaches are solved **backwards**. Every airborne step computes its own
Point1 from the following step's Point1 (Eq. B-42, Eq. B-64), and the
recursion is anchored by the Land step, whose Point1 sits at distance zero:
hence the negative distances before touchdown. The rollout is then solved
forwards from the same anchor, so touchdown is where the two sweeps meet.

**Parameters**

| Name | Description |
| :--- | :--- |
| `aircraft` | The aeroplane's coefficient set. |
| `steps` | The procedure's steps, in order, containing one Land step followed by its Decelerate steps. |
| `aerodrome` | Aerodrome and weather. |
| `weight_lb` | Approach weight, lb. `None` (default) takes Doc 29's own rule, 90 % of the aeroplane's maximum landing weight (folio B-31). |
| `procedure_id` | Identifier of the procedure, carried into the result. |

**Returns:** A [`FlightProfile`](/phonometry/reference/api/aeroacoustics/flight-performance/#flightprofile) with `operation="A"`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the procedure carries no Land step to anchor it, or a step cannot be flown as specified. |

## ApproachStep

```python
ApproachStep(
    step_type: str,
    flap_id: str,
    start_altitude_ft: float | None = None,
    start_calibrated_airspeed_kt: float | None = None,
    descent_angle_deg: float | None = None,
    touchdown_roll_ft: float | None = None,
    distance_ft: float | None = None,
    start_thrust_percent: float | None = None,
    bank_angle_deg: float = 0.0,
)
```

One row of an ANP approach procedural-step table (B7.1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `step_type` | `"Descend"`, `"Descend-Decel"`, `"Descend-Idle"`, `"Level"`, `"Level-Decel"`, `"Level-Idle"`, `"Land"` or `"Decelerate"`, in whatever case the table spells it. |
| `flap_id` | Flap identifier, as the table spells it. |
| `start_altitude_ft` | Height above the aerodrome at the *start* of the step, ft. An approach step is anchored at its top, not its bottom. |
| `start_calibrated_airspeed_kt` | Calibrated airspeed at the start of the step, kt. |
| `descent_angle_deg` | Descent angle, degrees, **positive by convention** as the 5th edition declares it and as the ANP tables store it. The 4th edition took it negative and wrote its equations to suit. |
| `touchdown_roll_ft` | Distance from touchdown to the Land step's Point2, ft; defined only for a Land step. |
| `distance_ft` | Track length of a Level, Level-Decel, Level-Idle or Decelerate step, ft. |
| `start_thrust_percent` | Start thrust of a Decelerate step, as a percentage of maximum sea-level static thrust (Eq. B-79, Eq. B-81). |
| `bank_angle_deg` | Bank angle `eps` over the step, degrees. |

Doc 29 Volume 3's own workbook keeps a Level-Idle step's length in the
*descent angle* column, with the distance column empty. That is a defect of
that workbook, not of the format: the length belongs in
`distance_ft` here, where the ANP release also puts it.

### ApproachStep.kind

*property*

The step type folded onto Doc 29's own vocabulary, lowercase.

## departure_profile

```python
departure_profile(
    aircraft: PerformanceAircraft,
    steps: Sequence[DepartureStep],
    *,
    weight_lb: float,
    aerodrome: Aerodrome,
    procedure_id: str = '',
) -> FlightProfile
```

Fly a departure procedure's steps into a flight profile (Doc 29 B6).

The profile is built forward from brake release, "the starting parameters
for each segment being equal to those at the end of the preceding segment"
(B1). Every step contributes one point, except the Take-off step, which
contributes two, and any step that changes thrust rating, which is preceded
by an inserted transition point (B6.1.6).

**Parameters**

| Name | Description |
| :--- | :--- |
| `aircraft` | The aeroplane's coefficient set. |
| `steps` | The procedure's steps, in order, starting with a Take-off. |
| `weight_lb` | Take-off weight, lb -- the ANP `Default_weights` entry for the stage length being flown. |
| `aerodrome` | Aerodrome, weather and runway gradient. |
| `procedure_id` | Identifier of the procedure, carried into the result. |

**Returns:** A [`FlightProfile`](/phonometry/reference/api/aeroacoustics/flight-performance/#flightprofile) with `operation="D"`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the procedure does not start with a Take-off step, or a step cannot be flown as specified. |

## DepartureStep

```python
DepartureStep(
    step_type: str,
    thrust_rating: str,
    flap_id: str,
    end_altitude_ft: float | None = None,
    rate_of_climb_ft_per_min: float | None = None,
    end_calibrated_airspeed_kt: float | None = None,
    energy_share_percent: float | None = None,
    distance_ft: float | None = None,
    bank_angle_deg: float = 0.0,
)
```

One row of an ANP departure procedural-step table (B6.1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `step_type` | `"Takeoff"`, `"Climb"`, `"Accelerate"`, `"Level"` or `"Level-Accelerate"`, in whatever case and hyphenation the table spells it; `kind` is the folded form the model works in. |
| `thrust_rating` | Thrust rating the step is flown at, as the table spells it. `"AdaptedThrust"` marks a Level step, whose thrust is solved rather than looked up, and `"MinimumThrust"` the engine-out floor of Eq. B-13. |
| `flap_id` | Flap identifier, as the table spells it. |
| `end_altitude_ft` | End-point height above the aerodrome of a Climb step, ft; an em dash for every other step type, which is why it is optional. |
| `rate_of_climb_ft_per_min` | Rate of climb of an Accelerate step, ft/min. |
| `end_calibrated_airspeed_kt` | End-point calibrated airspeed of an Accelerate or Level-Accelerate step, kt. |
| `energy_share_percent` | Acceleration percentage (energy share factor) of an Accelerate or Level-Accelerate step, per cent. |
| `distance_ft` | Track distance of a Level step, ft. |
| `bank_angle_deg` | Bank angle `eps` over the step, degrees. |

Four of these are quantities the step type simply does not define, and the
ANP table leaves each blank; they are `None` here and rendered as an em
dash, never as a zero, since a zero rate of climb is a level acceleration
and a zero distance is a step that goes nowhere.

A step given both a rate of climb and an energy share factor keeps both, and
the model prefers the energy share factor: "The ROC-values are altitude and
atmosphere conditions dependent whereas ESF values adapt to changing airport
elevations and atmosphere conditions" (B6.1.3, folio B-21), so of the two
only the energy share factor still means what the manufacturer intended at
another aerodrome. B6.1.3 leaves the choice to the model, putting it as
advice: "it is preferable to use ESF values".

The bank angle is an input rather than something derived, because Eq. B-14
needs a turn radius and a turn radius needs the ground track, which this
model does not build and Appendix B assumes it is given. Zero, the default,
is straight flight,
where every `R/cos(eps)` in Appendix B reduces to `R`.

### DepartureStep.kind

*property*

The step type folded onto Doc 29's own vocabulary, lowercase.

The raw `step_type` is kept as the table spells it so a
transcription stays diffable against the sheet it came from; this is
what the model branches on.

## FlightProfile

```python
FlightProfile(
    aircraft_id: str,
    operation: str,
    procedure_id: str,
    points: tuple[ProfilePoint, ...],
)
```

A flight profile: the fixed-point trajectory a procedure flies (B1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier. |
| `operation` | `"D"` (departure) or `"A"` (arrival). |
| `procedure_id` | Identifier of the procedure the steps came from. |
| `points` | The profile points, ordered along the ground track. |

This is the vertical-plane half of a Doc 29 flight path. Section 3.6 is what
turns it into three dimensions -- splitting segments at ground-track nodes,
sub-segmenting the rolls and the climb, merging in the ground track -- and
none of that happens here.

### FlightProfile.altitude_ft

*property*

Height above the aerodrome per point, ft.

### FlightProfile.corrected_net_thrust_lb

*property*

Corrected net thrust per engine per point, lb.

### FlightProfile.distance_ft

*property*

Distance along the ground track per point, ft.

### FlightProfile.plot()

```python
FlightProfile.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the profile's height and thrust against distance along the track.

### FlightProfile.true_airspeed_kt

*property*

True airspeed per point, kt.

## JetEngineCoefficients

```python
JetEngineCoefficients(e: float, f: float, ga: float, gb: float, h: float)
```

One ANP `Jet_Engine_Coefficients` row: the Eq. B-9 thrust polynomial.

`CNT = E + F Vc + Ga h + Gb h^2 + H T` gives the corrected net thrust per
engine in lb, for one aeroplane and one thrust rating.

**Attributes**

| Name | Description |
| :--- | :--- |
| `e` | Constant term `E`, lb. |
| `f` | Calibrated-airspeed coefficient `F`, lb/kt. |
| `ga` | Pressure-altitude coefficient `Ga`, lb/ft. |
| `gb` | Squared pressure-altitude coefficient `Gb`, lb/ft2. |
| `h` | Temperature coefficient `H`, lb/degC. |

The units are the 5th edition's symbol list (folio B-2) and the Volume 3
column headers; the 4th edition printed four units for these five symbols.

### JetEngineCoefficients.corrected_net_thrust_lb()

```python
JetEngineCoefficients.corrected_net_thrust_lb(
    *,
    calibrated_airspeed_kt: float,
    pressure_altitude_ft: float,
    temperature_c: float,
) -> float
```

Corrected net thrust per engine, lb (Eq. B-9, and Eq. B-10 in kind).

**Parameters**

| Name | Description |
| :--- | :--- |
| `calibrated_airspeed_kt` | Calibrated airspeed `Vc`, kt. |
| `pressure_altitude_ft` | Pressure altitude `h` of Eq. B-6, ft. |
| `temperature_c` | Air temperature at the aeroplane `T`, degC. |

## PerformanceAircraft

```python
PerformanceAircraft(
    aircraft_id: str,
    engines: int,
    max_static_thrust_lb: float,
    max_landing_weight_lb: float,
    jet_coefficients: Mapping[str, JetEngineCoefficients] = ...,
    propeller_coefficients: Mapping[str, PropellerEngineCoefficients] = ...,
    aerodynamic_coefficients: Mapping[tuple[str, str], AerodynamicCoefficients] = ...,
)
```

One aeroplane's Appendix B coefficient set.

**Attributes**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier. |
| `engines` | Number of engines supplying thrust `N`. |
| `max_static_thrust_lb` | Maximum sea-level static thrust per engine, lb. Read only by Eq. B-79 and Eq. B-81, where a Decelerate step's start thrust is a percentage of it. |
| `max_landing_weight_lb` | Maximum gross landing weight, lb. The approach weight is 90 % of it (Eq. B-75, Eq. B-76). |
| `jet_coefficients` | Eq. B-9 coefficients per thrust rating. |
| `propeller_coefficients` | Eq. B-12 coefficients per thrust rating. |
| `aerodynamic_coefficients` | Flap configurations per `(operation, flap identifier)`, with the operation `"A"` or `"D"`. |

Which of the two thrust forms applies is decided by which table carries a
row, not by the engine-type label: B4.1 is headed "jet and (certain)
turboprop" for Eq. B-9 and B4.2 "piston and (some) turboprop" for Eq. B-12,
and neither says which turboprop is which, so a turboprop appears under
either heading and only its coefficient rows say under which.

### PerformanceAircraft.approach_weight_lb

*property*

Approach weight, lb: 90 % of the maximum landing weight (folio B-31).

Not the ANP `Default_weights` arrival entry, which is a different
number for most of the fleet; Doc 29 names the aircraft table's landing
weight and the fraction explicitly, three times.

### PerformanceAircraft.flap()

```python
PerformanceAircraft.flap(
    operation: str,
    flap_id: str,
) -> AerodynamicCoefficients
```

Aerodynamic coefficients for one flap configuration.

**Parameters**

| Name | Description |
| :--- | :--- |
| `operation` | `"A"` (arrival) or `"D"` (departure). |
| `flap_id` | Flap identifier as the procedure spells it. |

**Raises**

| Exception | When |
| :--- | :--- |
| KeyError | if the aeroplane has no such configuration. Eq. B-21 takes `R` from the ANP `Aerodynamic_Coefficients` table for the step's own `Flap_ID` and names no fallback for an identifier that is not in it, so this raises rather than substituting a default: a silently substituted drag ratio changes every climb angle of the profile and nothing downstream can tell. |

## ProfilePoint

```python
ProfilePoint(
    distance_ft: float,
    altitude_ft: float,
    true_airspeed_kt: float,
    corrected_net_thrust_lb: float,
)
```

One point of a Doc 29 flight profile.

**Attributes**

| Name | Description |
| :--- | :--- |
| `distance_ft` | Distance along the ground track, ft. Measured from brake release on a departure, and from touchdown on an arrival, where it is negative while the aeroplane is still airborne (folio B-5). |
| `altitude_ft` | Height above the aerodrome elevation, ft. |
| `true_airspeed_kt` | True airspeed, kt. |
| `corrected_net_thrust_lb` | Corrected net thrust `Fn/delta` per engine, lb. This is the power setting the NPD tables are indexed on. |

## PropellerEngineCoefficients

```python
PropellerEngineCoefficients(efficiency: float, power_hp: float)
```

One ANP `Propeller_Engine_Coefficients` row: the Eq. B-12 thrust.

`CNT = (326 eta Pp / Vt) / delta` for a piston or turboprop aeroplane.

**Attributes**

| Name | Description |
| :--- | :--- |
| `efficiency` | Propeller efficiency `eta`, dimensionless. |
| `power_hp` | Installed net propulsive power `Pp` per engine, hp. |

### PropellerEngineCoefficients.corrected_net_thrust_lb()

```python
PropellerEngineCoefficients.corrected_net_thrust_lb(
    *,
    true_airspeed_kt: float,
    pressure_ratio: float,
) -> float
```

Corrected net thrust per engine, lb (Eq. B-12).

**Parameters**

| Name | Description |
| :--- | :--- |
| `true_airspeed_kt` | True airspeed `Vt`, kt. Eq. B-12 is singular at rest, so the caller supplies the floor B4.2 pins for the ground roll: "the minimum value of V_T is assumed to be the initial climb speed", which at the take-off Point1 is the Point2 true airspeed. |
| `pressure_ratio` | `delta` of Eq. B-4 at the point's altitude. |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the true airspeed is not positive. |
