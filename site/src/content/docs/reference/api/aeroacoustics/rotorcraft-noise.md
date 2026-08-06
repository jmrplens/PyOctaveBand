---
title: "aircraft.rotorcraft_noise"
description: "Rotorcraft noise by the hemisphere method (ECAC Doc 32 / NORAH2)."
sidebar:
  label: "rotorcraft_noise"
---

Rotorcraft noise by the hemisphere method (ECAC Doc 32 / NORAH2).

The ECAC Doc 32 rotorcraft-noise method describes a helicopter's highly directive
source with a **noise hemisphere**: one-third-octave-band sound pressure levels on
a spherical grid of azimuth `φ` and polar angle `θ` at a fixed 60 m reference
distance (at ICAO reference atmospheric conditions). Placing that source at a
receiver adds the propagation adjustment
$\Delta L_p = \Delta L_s + \Delta L_a + \Delta L_g$ (plus
$\Delta L_d$ with shielding): spherical spreading, atmospheric
absorption, ground effect and, later, shielding. Those adjustments depend on the
path and not on the rotorcraft, and live in
[`rotorcraft_propagation`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/); this module is the source
that emits and the event that receives.

This module provides the source primitives and the single-event method built on
them (clean-room, from the NORAH2 guidance SC01.D1.5d, the basis of ECAC
Doc 32):

* [`hemisphere_source_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#hemisphere_source_level) -- the interpolated source level `L(fc, φ, θ)`
  from a [`RotorcraftHemisphere`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcrafthemisphere), bilinear over the 10° grid (Eq. 13) with
  nearest-bin fill outside the measured coverage (Eq. 14/15).
* [`flight_condition_weights`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_condition_weights) / [`interpolated_source_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#interpolated_source_level) -- the
  flight-condition interpolation across a hemisphere set: distance-scaled
  triangulation inside the convex hull of the normalised `(V̄, γ̄)` database
  conditions, nearest neighbour outside (Eq. 3-10).
* [`flight_path_kinematics`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_path_kinematics) -- track kinematics by central finite
  differences: ground speed, airspeed, heading, curvature, bank and path angle
  (Eq. 16-21 / Doc 32 Eq. 8-10).
* [`rotorcraft_event_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraft_event_level) -- the received one-third-octave time history of
  a single event at recorded time (Eq. 1/22/23) and its integrated metrics:
  `LASmax`, `SEL` (Doc 32 Eq. 27) and `EPNL` (Doc 32 Eq. 28, ICAO Annex 16).
* [`rotorcraft_noise_contour`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraft_noise_contour) -- the single-event `SEL`/`LASmax` ground
  grid.

Source (clean-room): ECAC Doc 32, 1st ed.; NORAH2 rotorcraft-noise modelling
guidance (EASA.2020.FC.06 SC01.D1.5d), §A.3 and §A.5. The event chain is
validated end to end against the NORAH2 reference implementation outputs for the
ARP verification cases (angles, retarded times, hemisphere selection, per-step
levels and event metrics).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## flight_condition_weights

```python
flight_condition_weights(
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    airspeed: float,
    path_angle: float,
    *,
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
) -> list[tuple[int, float]]
```

Hemisphere blending weights for a flight condition (Eq. 3-10).

The database flight conditions and the query are scaled by the database
spans, $\bar{V} = V/(V_{\mathrm{max}} - V_{\mathrm{min}})$ and
$\bar{\gamma} = F_{fc} \cdot \gamma / (\gamma_{\mathrm{max}} - \gamma_{\mathrm{min}})$ with
the empirical flight-condition scaling factor $F_{fc} = 2$: the
guidance's
normalisation (Eq. 3-6), which subtracts no minima -- a shared offset
cancels in the distances `δ_j` (Eq. 7) either way. Inside the
convex hull of the database conditions the enveloping Delaunay triangle
contributes with inverse-distance weights
$(1/\delta_j)/\sum (1/\delta_j)$,
$\delta_j = \sqrt{(\bar{\gamma}-\bar{\gamma}_j)^2 + (\bar{V}-\bar{V}_j)^2}$ (Eq. 7/8); outside it (and whenever no
triangulation exists, e.g. collinear conditions) the nearest database
condition is adopted unblended (Eq. 9/10). A query on a database condition
returns that hemisphere alone. ECAC Doc 32, 1st ed., §4.1 defines no
interpolation ("select the most appropriate hemisphere"); this is the
interpolation of the NORAH2 guidance §A.3.1 on which the NORAH database and
reference implementation operate, and it degrades to the Doc 32 behaviour
outside the measured envelope.

The scaling is span-based, so the weights do not depend on the units of
`airspeeds` or `path_angles` as long as the query uses the same units
as the database conditions.

**Parameters**

| Name | Description |
| :--- | :--- |
| `airspeeds` | Database hemisphere airspeeds `V_j`, shape `(J,)`. |
| `path_angles` | Database hemisphere path angles `γ_j`, in degrees, shape `(J,)` (negative for descent). |
| `airspeed` | Query airspeed `V_A` (the airspeed, not the ground speed, selects the hemisphere; guidance §A.3.3). |
| `path_angle` | Query path angle `γ`, in degrees. |
| `scaling_factor` | Flight-condition scaling factor `F_fc` applied to the normalised path angle (default 2, the guidance's empirical value). |
| `triangles` | Optional precomputed triangulation, shape `(T, 3)` 0-based indices into the database conditions (guidance §A.3.1 step 4 admits a lookup table; the NORAH database ships one per type). Default `None` computes the Delaunay triangulation of the normalised conditions. The shipped NORAH lookup tables triangulate the raw `(V, γ)` plane instead of the normalised one, so passing them reproduces the reference implementation bin for bin. |

**Returns:** The `(index, weight)` pairs, weights summing to 1.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## flight_path_kinematics

```python
flight_path_kinematics(
    times: NDArray[np.float64] | list[float],
    positions: NDArray[np.float64] | list[list[float]],
    *,
    gravity: float = 9.80665,
) -> FlightPathKinematics
```

Track kinematics by central finite differences (Eq. 16-21 / Doc 32 Eq. 8-10).

Computes, at every point of a time-stamped track, the ground speed `V_g`
(Eq. 16), the zero-wind airspeed `V_A` (Eq. 17), the heading
$\Theta = \operatorname{atan2}(\Delta X, \Delta Y)$ (Eq. 19), the
curvature $K = \Delta\Theta/\Delta S$ (Eq. 18), the
bank angle $\Phi = \arctan(K \cdot V_g^2/g)$ (Eq. 20) and the path
angle
$\gamma = \arctan(\Delta Z/\Delta S)$ (Doc 32 Eq. 10). The
airspeed, not the ground speed,
selects the hemisphere (guidance §A.3.3); the guidance recommends smoothing
radar tracks (e.g. spline resampling) before differentiating.

**Parameters**

| Name | Description |
| :--- | :--- |
| `times` | Track times, in s, strictly increasing, shape `(N,)`, $N \ge 2$. |
| `positions` | Track positions `(x, y, z)`, in metres, shape `(N, 3)` (x east, y north, z up; any consistent right-handed ground frame works, headings are then relative to its y axis). |
| `gravity` | Acceleration of gravity `g` in m/s² (default 9.80665). |

**Returns:** A [`FlightPathKinematics`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flightpathkinematics).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## FlightConditionInterpolation

```python
FlightConditionInterpolation(
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
)
```

How a flight condition blends the database hemispheres (Eq. 3-10).

The two settings of [`flight_condition_weights`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_condition_weights), which the event and
contour entry points hand it per track point.

**Attributes**

| Name | Description |
| :--- | :--- |
| `scaling_factor` | Flight-condition scaling factor `F_fc` applied to the normalised path angle (default 2, the guidance's empirical value). |
| `triangles` | Optional precomputed triangulation, shape `(T, 3)` 0-based indices into the database conditions (default `None`: the Delaunay triangulation of the normalised conditions). See [`flight_condition_weights`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_condition_weights). |

## FlightPathKinematics

```python
FlightPathKinematics(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    ground_speed: NDArray[np.float64],
    airspeed: NDArray[np.float64],
    heading: NDArray[np.float64],
    curvature: NDArray[np.float64],
    bank_angle: NDArray[np.float64],
    path_angle: NDArray[np.float64],
)
```

Kinematics of a rotorcraft track (guidance Eq. 16-21 / Doc 32 Eq. 8-10).

All rates come from central finite differences around each track point.

**Attributes**

| Name | Description |
| :--- | :--- |
| `times` | Track times, in s, shape `(N,)`. |
| `positions` | Track positions `(x, y, z)`, in metres, shape `(N, 3)`. |
| `ground_speed` | Ground speed `V_g` (Eq. 16), in m/s, shape `(N,)`. |
| `airspeed` | Airspeed `V_A` (Eq. 17, zero-wind), in m/s, shape `(N,)`. |
| `heading` | Heading $\Theta = \operatorname{atan2}(\Delta X, \Delta Y)$ (Eq. 19), in degrees, shape `(N,)`. |
| `curvature` | Track curvature $K = \Delta\Theta/\Delta S$ (Eq. 18), in rad/m, shape `(N,)` (zero where the ground speed vanishes). |
| `bank_angle` | Bank angle $\Phi = \arctan(K \cdot V_g^2/g)$ (Eq. 20), in degrees, positive starboard down, shape `(N,)`. |
| `path_angle` | Path angle $\gamma = \arctan(\Delta Z/\Delta S)$ (Doc 32 Eq. 10), in degrees, positive climbing, shape `(N,)`. |

:::note
The guidance prints Eq. 21 as
$\gamma = \arccos(\Delta Z/\Delta S)$, which returns the
complement of the path angle (90° in level flight) and is dimensionally
inconsistent with its use; ECAC Doc 32 Eq. 10 states the correct
`atan` form, which this implementation follows.
:::

### FlightPathKinematics.plot()

```python
FlightPathKinematics.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the speed and angle profiles along the track.

## hemisphere_source_level

```python
hemisphere_source_level(
    hemisphere: RotorcraftHemisphere,
    azimuth_deg: float,
    polar_deg: float,
) -> NDArray[np.float64]
```

Interpolated source level `L(fc, φ, θ)` from a hemisphere (Eq. 13-15).

The grid is first gap-filled by nearest-bin constant-value extrapolation
(Eq. 14/15, computed once per hemisphere and cached), then the query is a
bilinear interpolation in the energy domain over the four neighbouring
azimuth/polar bins (Eq. 13). Filling the grid before interpolating keeps
partially-measured cells continuous with their fully-measured neighbours
(the valid corners still contribute) instead of snapping to a single bin.

Queries outside the grid clamp to the boundary node and edge-interpolate;
Eq. 14/15 taken literally would return the single nearest node, which
coincides on the boundary nodes but is discontinuous alongside them, so the
smoother clamp is intentional. Bands with no filled bin anywhere in the
grid return `NaN`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `hemisphere` | The [`RotorcraftHemisphere`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcrafthemisphere) source description. |
| `azimuth_deg` | Emission azimuth `φ`, in degrees. |
| `polar_deg` | Emission polar angle `θ`, in degrees. |

**Returns:** Band levels at `(φ, θ)`, in dB, shape `(F,)`.

## interpolated_source_level

```python
interpolated_source_level(
    hemispheres: Sequence[RotorcraftHemisphere],
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    airspeed: float,
    path_angle: float,
    azimuth_deg: float,
    polar_deg: float,
    *,
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
) -> NDArray[np.float64]
```

Source level at a flight condition between hemispheres (Eq. 8/10 over Eq. 13).

Blends [`hemisphere_source_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#hemisphere_source_level) lookups of the hemispheres selected
by [`flight_condition_weights`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_condition_weights) in the energy domain (Eq. 8).

**Parameters**

| Name | Description |
| :--- | :--- |
| `hemispheres` | The database hemispheres, one per flight condition. |
| `airspeeds` | Database airspeeds `V_j`, shape `(J,)`. |
| `path_angles` | Database path angles `γ_j`, in degrees, shape `(J,)`. |
| `airspeed` | Query airspeed `V_A` (same units as `airspeeds`). |
| `path_angle` | Query path angle `γ`, in degrees. |
| `azimuth_deg` | Emission azimuth `φ`, in degrees. |
| `polar_deg` | Emission polar angle `θ`, in degrees. |
| `scaling_factor` | Flight-condition scaling factor `F_fc` (default 2). |
| `triangles` | Optional precomputed triangulation (see [`flight_condition_weights`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_condition_weights)). |

**Returns:** Band levels at the reference distance, in dB, shape `(F,)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## rotorcraft_event_level

```python
rotorcraft_event_level(
    hemispheres: Sequence[RotorcraftHemisphere],
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    times: NDArray[np.float64] | list[float],
    positions: NDArray[np.float64] | list[list[float]],
    receiver: tuple[float, float] | NDArray[np.float64] | list[float],
    *,
    level_offset: float | NDArray[np.float64] | list[float] = 0.0,
    atmosphere: RotorcraftAtmosphere = ...,
    ground: RotorcraftGround = ...,
    track_state: RotorcraftTrackState = ...,
    interpolation: FlightConditionInterpolation = ...,
) -> RotorcraftEventResult
```

Rotorcraft single-event level at a receiver (Doc 32 §6.1 / guidance §A.5.1).

For every track point the flight condition selects (or blends, Eq. 3-10)
the hemispheres, the emission angles address the source level (Eq. 13-15)
and the propagation adjustment
$\Delta L_p = \Delta L_s + \Delta L_a + \Delta L_g$ (Eq. 23-35)
places
it at the receiver. The received one-third-octave history is expressed at
recorded time $t_r = t_e + r/c$ (Eq. 22) and integrated into
`LASmax`,
`SEL` (Doc 32 Eq. 27) and `EPNL` (Doc 32 Eq. 28, ICAO Annex 16 App. 2,
reusing
[`epnl_from_pnlt`](/phonometry/reference/api/aeroacoustics/certification/#epnl_from_pnlt)).

The flight condition per point comes from the `track_state` overrides
when given (e.g. the smoothed values of a radar-track workflow),
otherwise from [`flight_path_kinematics`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_path_kinematics) on the track itself, in which
case the database `airspeeds` must be in m/s. The hemisphere frame is
oriented by the heading and tilted by the bank angle in turns (guidance
§A.3.4); pitch attitude is implicit in the hemispheres.

**Parameters**

| Name | Description |
| :--- | :--- |
| `hemispheres` | The database hemispheres, one per flight condition. |
| `airspeeds` | Database airspeeds `V_j`, shape `(J,)` (same units as the `airspeed` values used for selection). |
| `path_angles` | Database path angles `γ_j`, in degrees, shape `(J,)`. |
| `times` | Track times, in s, strictly increasing, shape `(N,)`. |
| `positions` | Track positions `(x, y, z)`, in metres, shape `(N, 3)` (z up, above the ground elevation datum). |
| `receiver` | Receiver ground position `(x, y)`, in metres. |
| `level_offset` | Source-level offset `ΔEPNL` added to the hemisphere levels (Eq. 2 class substitution), in dB (default 0). Scalar or per track point, shape `(N,)`: Chapter-8 substitutions correct climb, level and descent conditions with different certification levels. |
| `atmosphere` | The air the event propagates through, a [`RotorcraftAtmosphere`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraftatmosphere) (default: the ICAO reference conditions of the database). |
| `ground` | The ground under the event, a [`RotorcraftGround`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraftground) (default: flat ground at the track datum, CNOSSOS class `"G"`, a 1.2 m microphone). A single receiver takes its scalar fields only; the per-grid-point arrays are for the contour. |
| `track_state` | Per-point airspeed, path angle, heading and bank angle, a [`RotorcraftTrackState`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcrafttrackstate) (default: all derived from the track by [`flight_path_kinematics`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_path_kinematics)). |
| `interpolation` | How the flight condition blends the database hemispheres, a [`FlightConditionInterpolation`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flightconditioninterpolation) (default: `F_fc = 2` over the Delaunay triangulation). |

**Returns:** A [`RotorcraftEventResult`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcrafteventresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## rotorcraft_noise_contour

```python
rotorcraft_noise_contour(
    hemispheres: Sequence[RotorcraftHemisphere],
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    times: NDArray[np.float64] | list[float],
    positions: NDArray[np.float64] | list[list[float]],
    *,
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    metric: str = 'exposure',
    level_offset: float | NDArray[np.float64] | list[float] = 0.0,
    atmosphere: RotorcraftAtmosphere = ...,
    ground: RotorcraftGround = ...,
    track_state: RotorcraftTrackState = ...,
    interpolation: FlightConditionInterpolation = ...,
) -> RotorcraftNoiseContourResult
```

Rotorcraft single-event level over a ground grid (Doc 32 §6.3).

Evaluates the event of [`rotorcraft_event_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraft_event_level) at every grid point
`(xi, yj)` in one vectorised pass per emission step, and reduces the
received histories to the exposure (`SEL`, Doc 32 Eq. 27) or maximum
(`LASmax`) level.

**Parameters**

| Name | Description |
| :--- | :--- |
| `hemispheres` | The database hemispheres, one per flight condition. |
| `airspeeds` | Database airspeeds `V_j`, shape `(J,)`. |
| `path_angles` | Database path angles `γ_j`, in degrees, shape `(J,)`. |
| `times` | Track times, in s, strictly increasing, shape `(N,)`. |
| `positions` | Track positions `(x, y, z)`, in metres, shape `(N, 3)`. |
| `x` | Grid x coordinates, in metres (at least 2). |
| `y` | Grid y coordinates, in metres (at least 2). |
| `metric` | `"exposure"` (SEL) or `"maximum"` (LASmax). |
| `level_offset` | Source-level offset `ΔEPNL` (Eq. 2), in dB, scalar or per track point. |
| `atmosphere` | The air the event propagates through, a [`RotorcraftAtmosphere`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraftatmosphere). |
| `ground` | The ground under the grid, a [`RotorcraftGround`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraftground). Its `ground_elevation` and `flow_resistivity` also accept one value per grid point (shape `(len(y), len(x))`), and its `terrain` model must cover the whole track and grid: every emission-receiver pair then samples its own vertical section, so the cost grows with track points times grid points; keep contour grids modest with terrain. |
| `track_state` | Per-point airspeed, path angle, heading and bank angle (see [`rotorcraft_event_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraft_event_level)). |
| `interpolation` | How the flight condition blends the database hemispheres, a [`FlightConditionInterpolation`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flightconditioninterpolation). |

**Returns:** A [`RotorcraftNoiseContourResult`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraftnoisecontourresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## RotorcraftAtmosphere

```python
RotorcraftAtmosphere(
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
    atmospheric_method: str = 'iso9613',
)
```

The air a rotorcraft event propagates through (Eq. 26/27).

The ICAO reference conditions of the hemisphere database are the defaults,
so an event flown at those conditions needs no atmosphere at all. The Doc 29
airport chain keeps its own
[`AerodromeAtmosphere`](/phonometry/reference/api/aeroacoustics/airport-noise/#aerodromeatmosphere) instead of
sharing this one: it corrects a broadband NPD level with the impedance of
Eq. 4-7 alone, with no band-by-band absorption to ask the humidity or the
method about, and at a different reference temperature.

**Attributes**

| Name | Description |
| :--- | :--- |
| `temperature` | Air temperature, in °C (default 25, ICAO reference). |
| `relative_humidity` | Relative humidity, in % (default 70). |
| `pressure` | Ambient pressure, in kPa (default 101.325). |
| `atmospheric_method` | `"iso9613"` for the pure-tone Eq. 26/27 term (the guidance text), or `"sae"` for the SAE ARP 5534 band-integrated mapping used by the NORAH2 reference implementation (they agree to ~0.05 dB below 3.15 kHz). |

## RotorcraftEventResult

```python
RotorcraftEventResult(
    frequencies: NDArray[np.float64],
    emission_times: NDArray[np.float64],
    times: NDArray[np.float64],
    distance: NDArray[np.float64],
    azimuth: NDArray[np.float64],
    polar: NDArray[np.float64],
    band_levels: NDArray[np.float64],
    a_levels: NDArray[np.float64],
    la_max: float,
    sel: float,
    sel_10db: float,
    pnlt: NDArray[np.float64],
    pnltm: float,
    epnl: float,
)
```

A rotorcraft single-event time history at a receiver (Doc 32 §6.1).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz, shape `(F,)`. |
| `emission_times` | Emission times `t_e`, in s, shape `(K,)`. |
| `times` | Recorded times $t_r = t_e + r/c$ (Eq. 22), in s, shape `(K,)`. |
| `distance` | Slant distance `r` per step, in metres, shape `(K,)`. |
| `azimuth` | Emission azimuth `φ` per step, in degrees, shape `(K,)`. |
| `polar` | Emission polar angle `θ` per step, in degrees, shape `(K,)`. |
| `band_levels` | Received (unweighted) band levels, in dB, shape `(K, F)`. |
| `a_levels` | A-weighted overall level `L_A(t)` per step, in dB(A), shape `(K,)`. |
| `la_max` | Maximum A-weighted level `LASmax`, in dB(A). |
| `sel` | Sound exposure level over the full history (Doc 32 Eq. 27, $t_0 = 1$ s), in dB(A). The full-history integration is the land-use planning convention of the NORAH2 reference implementation. |
| `sel_10db` | Sound exposure level restricted to the 10 dB-down window about `LASmax` (the certification convention), in dB(A). |
| `pnlt` | Tone-corrected perceived noise level per step, in TPNdB, shape `(K,)`; `NaN` where undefined (zero total noisiness, or the band grid does not cover the 24 noy bands 50 Hz-10 kHz). |
| `pnltm` | Maximum `PNLT` (with the Annex 16 bandsharing adjustment), in TPNdB; `NaN` if no step has a defined `PNLT`. |
| `epnl` | Effective perceived noise level (Doc 32 Eq. 28 / ICAO Annex 16), in EPNdB; `NaN` if no step has a defined `PNLT`. |

### RotorcraftEventResult.plot()

```python
RotorcraftEventResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the A-weighted level time history with its event metrics.

## RotorcraftGround

```python
RotorcraftGround(
    receiver_height: float = 1.2,
    ground_elevation: float | NDArray[np.float64] | list[float] | list[list[float]] = 0.0,
    flow_resistivity: float | str | np.floating[Any] | np.integer[Any] | NDArray[np.float64] | list[float] | list[list[float]] = 'G',
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | Sequence[NDArray[np.float64]] | None = None,
    terrain_resolution: float | None = None,
)
```

The ground a rotorcraft event stands on (guidance §A.4.3-A.4.5).

Flat ground at the track datum by default: the microphone height, the
elevation of the site and the ground type feed the two-ray ground effect,
and an optional elevation model replaces the flat plane with real terrain.

**Attributes**

| Name | Description |
| :--- | :--- |
| `receiver_height` | Microphone height above local ground, in metres (default 1.2). |
| `ground_elevation` | Ground elevation `z` at the receivers, in metres on the track datum (default 0); source and receiver heights above ground follow from it. A contour grid also accepts one value per grid point (shape `(len(y), len(x))`) for receivers on uneven sites without a full elevation model. |
| `flow_resistivity` | Ground flow resistivity `σ` in Pa·s/m², or a CNOSSOS class letter (see [`ground_effect_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#ground_effect_adjustment)). A contour grid also accepts one value per grid point (shape `(len(y), len(x))`) for heterogeneous ground across the receivers (each receiver's two-ray model uses its local value). |
| `terrain` | Optional digital elevation model `(x, y, z)` on the track frame (`x` and `y` strictly increasing, `z` of shape `(len(y), len(x))`, all in metres on the track datum). When given, every emission-receiver pair is evaluated over its sampled vertical section (guidance §A.4.4/A.4.5): mean-ground-plane ground effect with equivalent heights, and rubber-band diffraction where terrain blocks the line of sight; `ground_elevation` is then taken from the model. The model must cover the whole track and every receiver (fabricating terrain beyond its edges is refused). |
| `terrain_resolution` | Section sampling step along the path, in metres (default: the elevation model's cell size; sections are capped at 20000 sampling intervals). |

## RotorcraftHemisphere

```python
RotorcraftHemisphere(
    frequencies: NDArray[np.float64],
    azimuth: NDArray[np.float64],
    polar: NDArray[np.float64],
    levels: NDArray[np.float64],
    distance: float = 60.0,
)
```

A rotorcraft noise hemisphere (ECAC Doc 32 §A.3.2).

One-third-octave-band sound pressure levels on a regular azimuth/polar grid at
the 60 m reference distance (ICAO reference atmosphere). Missing bins (outside
the measured coverage) are `NaN` and filled by nearest-bin extrapolation on
lookup.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz, shape `(F,)`. |
| `azimuth` | Azimuth angles `φ`, in degrees, shape `(A,)` (`-90` port … `+90` starboard). |
| `polar` | Polar angles `θ`, in degrees, shape `(P,)` (`0` forward … `180` rearward). |
| `levels` | Band levels, in dB, shape `(A, P, F)`. |
| `distance` | Reference distance, in metres (default 60). The standard NORAH database uses 60 m; when the data uses another polar distance (e.g. 70 m hover rings), pass this value as `reference_distance` to [`spherical_spreading_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#spherical_spreading_adjustment) and [`atmospheric_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#atmospheric_adjustment) so the propagation chain honours it. |

### RotorcraftHemisphere.mirrored()

```python
RotorcraftHemisphere.mirrored() -> RotorcraftHemisphere
```

The hemisphere with the azimuth axis reversed (`φ → −φ`).

Doc 32 Eq. 2 substitutes a class member whose main/tail-rotor
configuration is mirrored with respect to the class reference (the
bracketed types of its Table 2, e.g. `[A600]` in the `R22` class)
by reversing the hemisphere azimuth angle.

**Returns:** A new [`RotorcraftHemisphere`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcrafthemisphere) with mirrored azimuth.

### RotorcraftHemisphere.plot()

```python
RotorcraftHemisphere.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the hemisphere directivity for one band (polar section).

## RotorcraftNoiseContourResult

```python
RotorcraftNoiseContourResult(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    level: NDArray[np.float64],
    metric: str,
)
```

Rotorcraft single-event noise level over a ground grid (Doc 32 §6.3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `x` | Grid x coordinates, in metres, shape `(nx,)`. |
| `y` | Grid y coordinates, in metres, shape `(ny,)`. |
| `level` | Event level over the grid, in dB(A), shape `(ny, nx)`. |
| `metric` | `"exposure"` (SEL) or `"maximum"` (LASmax). |

### RotorcraftNoiseContourResult.plot()

```python
RotorcraftNoiseContourResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot filled noise contours over the ground plane.

## RotorcraftTrackState

```python
RotorcraftTrackState(
    airspeed: float | NDArray[np.float64] | list[float] | None = None,
    path_angle: float | NDArray[np.float64] | list[float] | None = None,
    heading: float | NDArray[np.float64] | list[float] | None = None,
    bank_angle: float | NDArray[np.float64] | list[float] | None = None,
)
```

Per-point flight state of a rotorcraft track (Eq. 16-21).

Every field left unset is derived from the track itself by
[`flight_path_kinematics`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_path_kinematics); a radar-track workflow that has already
smoothed these quantities hands them over instead. Each is a scalar
(broadcast over the track) or an array of shape `(N,)`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `airspeed` | Airspeed `V_A`, in the units of the database `airspeeds` (the derived values are in m/s). |
| `path_angle` | Path angle `γ`, in degrees (negative descending). |
| `heading` | Heading `Θ`, in degrees. |
| `bank_angle` | Bank angle `Φ`, in degrees (positive starboard down). |
