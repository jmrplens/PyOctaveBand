---
title: "aircraft.airport_noise"
description: "Noise-Power-Distance (NPD) event-level interpolation (ECAC Doc 29)."
sidebar:
  label: "airport_noise"
---

Noise-Power-Distance (NPD) event-level interpolation (ECAC Doc 29).

The ECAC Doc 29 airport-noise method describes an aircraft's noise emission with
**Noise-Power-Distance (NPD)** tables: the event noise level (`LAmax` or the
sound exposure level `SEL`) of an aircraft in steady straight flight on an
infinite path, tabulated over a grid of engine power settings and slant
distances at reference conditions. Placing an aircraft's noise at a receiver
starts by reading a level from this table for an arbitrary power and distance.

* [`npd_level`](/phonometry/reference/api/aeroacoustics/airport-noise/#npd_level) -- the interpolated event level `L(P, d)`, linear in power
  and log-linear in distance (Eqs. 4-3/4-4), with extrapolation beyond the
  tabulated envelope.
* [`npd_curve`](/phonometry/reference/api/aeroacoustics/airport-noise/#npd_curve) -- the level over a distance sweep at one power, returned as
  an [`NpdLevelResult`](/phonometry/reference/api/aeroacoustics/airport-noise/#npdlevelresult) with a `.plot()`.

The single-event stage segments a flight path and adjusts the NPD baseline per
segment (§4.3-4.5): [`impedance_adjustment`](/phonometry/reference/api/aeroacoustics/airport-noise/#impedance_adjustment) (Eq. 4-6/4-7),
[`lateral_attenuation`](/phonometry/reference/api/aeroacoustics/airport-noise/#lateral_attenuation) (β, ℓ), [`engine_installation_correction`](/phonometry/reference/api/aeroacoustics/airport-noise/#engine_installation_correction)
(φ, mounting), [`duration_correction`](/phonometry/reference/api/aeroacoustics/airport-noise/#duration_correction), the finite-segment
[`noise_fraction`](/phonometry/reference/api/aeroacoustics/airport-noise/#noise_fraction) and, behind takeoff ground-roll segments, the
[`start_of_roll_directivity`](/phonometry/reference/api/aeroacoustics/airport-noise/#start_of_roll_directivity) `ΔSOR` (§4.5.7). [`event_level`](/phonometry/reference/api/aeroacoustics/airport-noise/#event_level)
assembles and sums them into the `SEL`/`LAmax` of a movement (mark takeoff
ground-roll segments with the `ground_roll` mask of its
[`FlightSegmentState`](/phonometry/reference/api/aeroacoustics/airport-noise/#flightsegmentstate)), and [`noise_contour`](/phonometry/reference/api/aeroacoustics/airport-noise/#noise_contour)
evaluates it over a ground grid.

Source (clean-room, implemented from the standard text): ECAC Doc 29, 4th ed.,
Vol 2 (2016), §4.2-4.5. Validated per-term and end-to-end against the ECAC
Doc 29 5th ed. Vol 3 Part 1 reference workbook.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AerodromeAtmosphere

```python
AerodromeAtmosphere(temperature: float = 15.0, pressure: float = 101.325)
```

The aerodrome air the NPD levels are corrected to (Eq. 4-6/4-7).

The two quantities [`impedance_adjustment`](/phonometry/reference/api/aeroacoustics/airport-noise/#impedance_adjustment) needs, at the standard
atmosphere by default, so an event at reference conditions needs no
atmosphere at all.

The Doc 32 rotorcraft chain keeps its own
[`RotorcraftAtmosphere`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraftatmosphere)
instead of sharing this one: it propagates band by band, so it also needs
the humidity and the absorption method, and its reference conditions are
the ICAO 25 °C/70 % of the hemisphere database, not the 15 °C of Eq. 4-7.

**Attributes**

| Name | Description |
| :--- | :--- |
| `temperature` | Aerodrome air temperature `T`, in °C (default 15). |
| `pressure` | Aerodrome air pressure `p`, in kPa (default 101.325). |

## duration_correction

```python
duration_correction(reference_speed: float, segment_speed: float) -> float
```

Duration correction (Eq. 4-14, exposure only).

$\Delta V = 10 \cdot \log_{10}(V_{\mathrm{ref}}/V_{\mathrm{seg}})$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reference_speed` | NPD reference speed `Vref` (any consistent unit). |
| `segment_speed` | Segment speed `Vseg` (same unit). |

**Returns:** The duration correction `ΔV`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a speed is not strictly positive. |

## engine_installation_correction

```python
engine_installation_correction(
    depression_deg: float,
    mounting: str = 'wing',
) -> float
```

Engine-installation lateral-directivity correction `ΔI(φ)` (Eq. 4-15/4-16).

**Parameters**

| Name | Description |
| :--- | :--- |
| `depression_deg` | Depression angle `φ` (from the wing plane), in degrees. |
| `mounting` | `"wing"`, `"fuselage"` or `"propeller"`. |

**Returns:** The correction `ΔI`, in dB (added to the level).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `mounting` is unknown or the angle is non-finite. |

## event_level

```python
event_level(
    path: NDArray[np.float64] | list[list[float]],
    observer: NDArray[np.float64] | list[float],
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    exposure_levels: NDArray[np.float64] | list[list[float]],
    maximum_levels: NDArray[np.float64] | list[list[float]],
    *,
    reference_speed: float = 82.31104,
    mounting: str = 'wing',
    metric: str = 'exposure',
    atmosphere: AerodromeAtmosphere = ...,
    segments: FlightSegmentState = ...,
) -> FlyoverResult
```

Single-event noise level of a flight path at a receiver (ECAC Doc 29).

Assembles the segment event levels (Eq. 4-8/4-9), the NPD baseline plus the
duration, engine-installation, lateral-attenuation and finite-segment
(noise-fraction) corrections, and the start-of-roll directivity behind takeoff
ground-roll segments, and combines them into the exposure level `SEL`
(energy sum, Eq. 4-11) or the maximum level (Eq. 4-10).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Flight-path points, shape `(N, 5)`: columns `x, y, z` (m), engine power setting and speed (m/s). `N-1` segments are formed. |
| `observer` | Receiver position `(x, y, z)`, in metres. |
| `powers` | NPD tabulated power settings. |
| `distances` | NPD tabulated slant distances, in metres. |
| `exposure_levels` | NPD exposure (SEL) levels, shape `(P, D)`. |
| `maximum_levels` | NPD maximum levels, shape `(P, D)`. |
| `reference_speed` | NPD reference speed, in m/s (default 160 kn). |
| `mounting` | Engine mounting (`"wing"`/`"fuselage"`/`"propeller"`). |
| `metric` | `"exposure"` (SEL) or `"maximum"` (LAmax). |
| `atmosphere` | Aerodrome air temperature and pressure, an [`AerodromeAtmosphere`](/phonometry/reference/api/aeroacoustics/airport-noise/#aerodromeatmosphere) (default: the standard atmosphere); they set the impedance adjustment. |
| `segments` | What each segment is doing, a [`FlightSegmentState`](/phonometry/reference/api/aeroacoustics/airport-noise/#flightsegmentstate) (default: airborne and unbanked): the takeoff ground-roll and landing-rollout masks and the per-segment bank angle. |

**Returns:** A [`FlyoverResult`](/phonometry/reference/api/aeroacoustics/airport-noise/#flyoverresult). If every segment is degenerate (zero length) the level is `-inf`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## FlightSegmentState

```python
FlightSegmentState(
    ground_roll: NDArray[np.bool_] | list[bool] | None = None,
    landing_roll: NDArray[np.bool_] | list[bool] | None = None,
    bank: NDArray[np.float64] | list[float] | None = None,
)
```

What each segment of a flight path is doing (§4.5.2/4.5.5-4.5.7).

One entry per segment, so every field an `N`-point path fills has length
`N-1`. Everything left unset describes an airborne, unbanked movement.

**Attributes**

| Name | Description |
| :--- | :--- |
| `ground_roll` | Optional boolean mask marking takeoff ground-roll segments; these receive the start-of-roll directivity `ΔSOR` and reduced noise fraction behind the aircraft (§4.5.6-4.5.7). |
| `landing_roll` | Optional boolean mask marking landing rollout segments; ahead of them the reduced fraction (Eq. 4-21b), the nearest-end lateral geometry and no directivity term apply (§4.5.5-4.5.6). |
| `bank` | Optional per-segment bank angle `ε` in degrees, measured counter-clockwise about the roll axis (positive in a left turn, starboard wing up); the depression angle becomes $\phi = \beta + \varepsilon$ for observers to starboard (right) of the track and $\phi = \beta - \varepsilon$ for observers to port (§4.5.2). |

## FlyoverResult

```python
FlyoverResult(
    level: float,
    metric: str,
    segment_levels: NDArray[np.float64],
    observer: NDArray[np.float64],
)
```

Single-event noise level of an aircraft movement at a receiver.

**Attributes**

| Name | Description |
| :--- | :--- |
| `level` | The event level, in dB (SEL for `metric="exposure"`, else the maximum level). |
| `metric` | `"exposure"` (SEL) or `"maximum"` (LAmax). |
| `segment_levels` | Per-segment contribution, in dB. |
| `observer` | Receiver position `(x, y, z)`, in metres. |

### FlyoverResult.plot()

```python
FlyoverResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-segment contributions to the event level.

## impedance_adjustment

```python
impedance_adjustment(
    temperature: float = 15.0,
    pressure: float = 101.325,
) -> float
```

Acoustic-impedance adjustment of the standard NPD data (Eq. 4-6/4-7).

The ANP NPD levels are normalised to a reference specific acoustic impedance
of 409.81 N·s/m³. At the aerodrome's temperature and pressure the air
impedance is $\rho c = 416.86 \cdot (\delta/\sqrt{\theta})$ with
$\delta = p/p_0$ and
$\theta = (T+273.15)/(T_0+273.15)$, and the adjustment
$10 \cdot \log_{10}(\rho c/409.81)$ is
added to the NPD levels. Under the standard atmosphere it is +0.074 dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Aerodrome air temperature `T`, in °C (default 15 °C). |
| `pressure` | Aerodrome air pressure `p`, in kPa (default 101.325 kPa). |

**Returns:** The impedance adjustment, in dB (added to the NPD level).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the pressure is not positive or inputs are non-finite. |

## lateral_attenuation

```python
lateral_attenuation(elevation_deg: float, lateral_m: float) -> float
```

Excess lateral attenuation `Λ(β, ℓ)` over soft ground (Eq. 4-18/4-19).

**Parameters**

| Name | Description |
| :--- | :--- |
| `elevation_deg` | Elevation angle `β` of the (equivalent level) path, in degrees. |
| `lateral_m` | Lateral displacement `ℓ` from the ground track, in metres. |

**Returns:** The lateral attenuation `Λ`, in dB (subtracted from the level).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `lateral_m` is negative or non-finite. |

## noise_contour

```python
noise_contour(
    path: NDArray[np.float64] | list[list[float]],
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    exposure_levels: NDArray[np.float64] | list[list[float]],
    maximum_levels: NDArray[np.float64] | list[list[float]],
    *,
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    reference_speed: float = 82.31104,
    mounting: str = 'wing',
    metric: str = 'exposure',
    atmosphere: AerodromeAtmosphere = ...,
    segments: FlightSegmentState = ...,
) -> NoiseContourResult
```

Single-event noise level over a ground grid (ECAC Doc 29 contour).

Evaluates [`event_level`](/phonometry/reference/api/aeroacoustics/airport-noise/#event_level) at every grid point `(xi, yj, 0)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Flight-path points, shape `(N, 5)` (see [`event_level`](/phonometry/reference/api/aeroacoustics/airport-noise/#event_level)). |
| `powers` | NPD tabulated power settings. |
| `distances` | NPD tabulated slant distances, in metres. |
| `exposure_levels` | NPD exposure (SEL) levels, shape `(P, D)`. |
| `maximum_levels` | NPD maximum levels, shape `(P, D)`. |
| `x` | Grid x coordinates, in metres. |
| `y` | Grid y coordinates, in metres. |
| `reference_speed` | NPD reference speed, in m/s. |
| `mounting` | Engine mounting. |
| `metric` | `"exposure"` (SEL) or `"maximum"` (LAmax). |
| `atmosphere` | Aerodrome air temperature and pressure, an [`AerodromeAtmosphere`](/phonometry/reference/api/aeroacoustics/airport-noise/#aerodromeatmosphere) (impedance adjustment). |
| `segments` | What each segment is doing, a [`FlightSegmentState`](/phonometry/reference/api/aeroacoustics/airport-noise/#flightsegmentstate) (see [`event_level`](/phonometry/reference/api/aeroacoustics/airport-noise/#event_level)). |

**Returns:** A [`NoiseContourResult`](/phonometry/reference/api/aeroacoustics/airport-noise/#noisecontourresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## noise_fraction

```python
noise_fraction(
    q: float,
    segment_length: float,
    scaled_distance: float,
) -> float
```

Finite-segment correction (noise fraction) `ΔF` (Eq. 4-20, exposure only).

**Parameters**

| Name | Description |
| :--- | :--- |
| `q` | Signed distance from the segment start `S1` to the perpendicular foot `Sp`, in metres (negative behind the segment). |
| `segment_length` | Segment length `λ`, in metres (`> 0`). |
| `scaled_distance` | The scaled distance `dλ` (Appendix E), in metres (`> 0`). |

**Returns:** The finite-segment correction `ΔF`, in dB (`<= 0`, floored at −150 dB).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `segment_length` or `scaled_distance` is not positive. |

## NoiseContourResult

```python
NoiseContourResult(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    level: NDArray[np.float64],
    metric: str,
)
```

Single-event noise level over a ground grid (ECAC Doc 29).

**Attributes**

| Name | Description |
| :--- | :--- |
| `x` | Grid x coordinates, in metres. |
| `y` | Grid y coordinates, in metres. |
| `level` | Event level over the grid `(len(y), len(x))`, in dB. |
| `metric` | `"exposure"` (SEL) or `"maximum"` (LAmax). |

### NoiseContourResult.plot()

```python
NoiseContourResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot filled noise contours over the ground plane.

## npd_curve

```python
npd_curve(
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    levels: NDArray[np.float64] | list[list[float]],
    power: float,
    query_distances: NDArray[np.float64] | list[float] | None = None,
) -> NpdLevelResult
```

NPD event level over a distance sweep at one power setting.

**Parameters**

| Name | Description |
| :--- | :--- |
| `powers` | Tabulated engine power settings. |
| `distances` | Tabulated slant distances, in metres. |
| `levels` | Tabulated event levels, shape `(len(powers), len(distances))`. |
| `power` | Query engine power setting. |
| `query_distances` | Distances to evaluate, in metres; defaults to a log sweep across the tabulated envelope. |

**Returns:** An [`NpdLevelResult`](/phonometry/reference/api/aeroacoustics/airport-noise/#npdlevelresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## npd_level

```python
npd_level(
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    levels: NDArray[np.float64] | list[list[float]],
    power: float,
    distance: NDArray[np.float64] | list[float] | float,
) -> NDArray[np.float64]
```

Interpolated NPD event level `L(P, d)` (ECAC Doc 29 §4.2, Eqs. 4-3/4-4).

Interpolates log-linearly in slant distance (Eq. 4-4) at the two bracketing
tabulated powers, then linearly in power (Eq. 4-3). Queries outside the
tabulated envelope are extrapolated from the terminal segments.

**Parameters**

| Name | Description |
| :--- | :--- |
| `powers` | Tabulated engine power settings (1-D, strictly increasing). |
| `distances` | Tabulated slant distances, in metres (1-D, strictly increasing, positive). |
| `levels` | Tabulated event levels, shape `(len(powers), len(distances))`, in dB. |
| `power` | Query engine power setting `P`. |
| `distance` | Query slant distance(s) `d`, in metres. |

**Returns:** The interpolated event level per query distance, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the table or inputs are invalid. |

## NpdLevelResult

```python
NpdLevelResult(
    distance: NDArray[np.float64],
    level: NDArray[np.float64],
    power: float,
    table_distances: NDArray[np.float64],
    table_levels: NDArray[np.float64],
)
```

NPD event level over a distance sweep at one power (ECAC Doc 29).

**Attributes**

| Name | Description |
| :--- | :--- |
| `distance` | Slant distances, in metres. |
| `level` | Interpolated event level per distance, in dB. |
| `power` | The engine power setting queried. |
| `table_distances` | The tabulated slant distances, in metres. |
| `table_levels` | The tabulated levels at the queried power, in dB. |

### NpdLevelResult.plot()

```python
NpdLevelResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the interpolated level versus slant distance (log axis).

## start_of_roll_directivity

```python
start_of_roll_directivity(
    azimuth_deg: float,
    distance_m: float,
    engine: str = 'jet',
) -> float
```

Start-of-roll (ground-roll) directivity correction `ΔSOR` (Eq. 4-22/4-25).

Behind a takeoff ground-roll segment, jet-exhaust noise radiates a lobed
rearward pattern. `ΔSOR` adjusts the segment level relative to the level to
the side of the start of roll, as a function of the azimuth `ψ` between the
aircraft forward axis and the observer (Eq. 4-24a for turbofan jets, 4-24b for
turboprops), scaled beyond 762 m by $d_{SOR,0}/d_{SOR}$ (Eq. 4-25).
It is only
applied behind takeoff ground-roll segments
($90^\circ \le \psi \le 180^\circ$); ahead of the
aircraft ($\psi < 90^\circ$) it is zero.

**Parameters**

| Name | Description |
| :--- | :--- |
| `azimuth_deg` | Azimuth `ψ` from the forward axis to the observer, in degrees ($\psi = \arccos(q/d_{SOR})$, in `[90, 180]` behind the aircraft). Values below 90° return 0; values above 180° are clamped to 180°. |
| `distance_m` | Distance `dSOR` from the observer to the segment start, in metres. |
| `engine` | `"jet"` (turbofan, Eq. 4-24a) or `"turboprop"` (Eq. 4-24b). |

**Returns:** The directivity correction `ΔSOR`, in dB (added to the level).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is unknown or the inputs are invalid. |
