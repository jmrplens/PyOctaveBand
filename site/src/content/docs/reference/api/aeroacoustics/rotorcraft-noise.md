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
absorption, ground effect and, later, shielding.

This module provides the source and propagation primitives and the single-event
method built on them (clean-room, from the NORAH2 guidance SC01.D1.5d, the basis
of ECAC Doc 32):

* [`hemisphere_source_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#hemisphere_source_level) -- the interpolated source level `L(fc, φ, θ)`
  from a [`RotorcraftHemisphere`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcrafthemisphere), bilinear over the 10° grid (Eq. 13) with
  nearest-bin fill outside the measured coverage (Eq. 14/15).
* [`spherical_spreading_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#spherical_spreading_adjustment) --
  $\Delta L_s = -20 \cdot \log_{10}(r/60)$ (Eq. 24).
* [`atmospheric_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#atmospheric_adjustment) --
  $\Delta L_a = -\alpha(f) \cdot (r - 60)$ with the ISO 9613-1
  pure-tone coefficient (Eq. 26/27), reusing
  [`air_attenuation`](/phonometry/reference/api/environment/air-absorption/#air_attenuation).
* [`ground_effect_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#ground_effect_adjustment) -- `ΔLg` for a point source over an impedance
  plane (Chien-Soroka, Eq. 28-35) with the Delany-Bazley one-parameter impedance
  and the CNOSSOS flow-resistivity classes.
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
guidance (EASA.2020.FC.06 SC01.D1.5d), §A.3-A.5. The atmospheric term is validated
against the guidance Table 4 (one-third-octave attenuation per km at ICAO
reference conditions); the event chain is validated end to end against the NORAH2
reference implementation outputs for the ARP verification cases (angles, retarded
times, hemisphere selection, per-step levels and event metrics).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## atmospheric_adjustment

```python
atmospheric_adjustment(
    frequencies: NDArray[np.float64] | list[float],
    distance: float,
    *,
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
    reference_distance: float = 60.0,
) -> NDArray[np.float64]
```

Atmospheric-absorption adjustment `ΔLa` of the hemisphere level (Eq. 26/27).

The hemisphere already includes absorption out to the reference distance
`rh`, so only the excess path $r - r_h$ is corrected:
$\Delta L_a = -\alpha(f) \cdot (r - r_h)$ with the ISO 9613-1
pure-tone coefficient `α`
evaluated at the exact band centre (Eq. 26/27, ICAO reference atmosphere by
default). This matches the guidance Eq. 27 to 0.02 dB/km and the NORAH2
reference implementation. The guidance's alternative per-band mapping (SAE
method by Rickley et al., its Table 4) coincides below 3.15 kHz and deviates
by up to 2.2 dB/km at 8-10 kHz; for a path-dependent band mapping use
[`sae_band_attenuation`](/phonometry/reference/api/aeroacoustics/atmospheric-absorption/#sae_band_attenuation).

:::note
The printed guidance Eq. 27 pairs the coefficient `6.6928e-6` with
$f_{r,O} = 630.7$ Hz, which evaluates to nonsense (14.3 dB/km
at 500 Hz against Table 4's 3.1). The physically correct pairing
(`6.6928e-6`
with the oxygen relaxation frequency, `1.3415e-6` with 630.7 Hz)
reproduces Table 4 and this implementation to 0.02 dB/km; do not
"fix" the code by transcribing the typo.
:::

Bands below the 50 Hz floor of the ISO 9613-1 tabulation (the NORAH grid
starts at 10 Hz) use the same analytic formulas; the advisory out-of-range
warning is suppressed because `α` is negligible there (Table 4 lists
0.0 dB/km for every band up to 50 Hz). The suppression only applies while
every band stays within the 10 kHz top of the NORAH grid; above that the
advisory warning propagates, since `α` is large and extrapolated.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave-band centre frequencies, in Hz. |
| `distance` | Slant distance `r`, in metres (`> 0`; below `rh` the adjustment is a small positive value, i.e. less absorption than the reference path). |
| `temperature` | Air temperature, in °C (default 25 °C, ICAO reference). |
| `relative_humidity` | Relative humidity, in % (default 70 %). |
| `pressure` | Ambient pressure, in kPa (default 101.325). |
| `reference_distance` | Hemisphere reference distance `rh`, in metres (default 60). Pass [`RotorcraftHemisphere.distance`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcrafthemisphere) when the data uses a non-standard polar distance. |

**Returns:** The adjustment `ΔLa` per band, in dB (added to the level, $\le 0$ for $r \ge r_h$).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a distance is not strictly positive. |

## diffraction_attenuation

```python
diffraction_attenuation(
    frequencies: NDArray[np.float64] | list[float],
    path_difference: float,
    *,
    edge_height: float,
    edge_span: float = 0.0,
    capped: bool = True,
) -> NDArray[np.float64]
```

Pure diffraction attenuation `ΔLd` per band (guidance Eq. 42-44).

$\Delta L_d = 10 \cdot C_h \cdot \log_{10}(3 + (40/\lambda) \cdot C'' \cdot \delta)$ where the argument is at least 1
(below it the attenuation is 0),
$C_h = \min(f_m \cdot h_0/250, 1)$ (Eq. 43) and
$C''$ accounts for multiple diffraction (Eq. 44: 1 for a single
edge or an edge span $e \le 0.3$ m,
$(1 + (5\lambda/e)^2)/(1/3 + (5\lambda/e)^2)$ otherwise).
A negative path difference (edge below the line of sight) still yields a
small attenuation down to
$(40/\lambda) \cdot C'' \cdot \delta = -2$; for bands with
$\delta < -\lambda/20$ the screening chain evaluates the
clear-path ground effect
instead of the diffraction (§A.4.5). At grazing incidence
($\delta = 0$) the attenuation is the classical
$10 \cdot \log_{10}(3) \approx 4.8$ dB.

The attenuation is returned positive (a loss); in the Doc 32 Eq. 23
chain, whose adjustments are added to the level, it enters with a minus
sign. The wavelength uses the Doc 32 reference speed of sound
$c = 346.1$ m/s.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave-band centre frequencies, in Hz. |
| `path_difference` | Path difference `δ` between the diffracted and the direct path, in metres (negative when the edge lies below the line of sight). |
| `edge_height` | Edge height `h0` above the mean ground plane(s), in metres (the greatest of the two side values for a terrain edge; `≥ 0`). |
| `edge_span` | Distance `e` between the first and last diffraction edges, in metres (default 0: single diffraction). |
| `capped` | Apply the 25 dB upper bound of §A.4.5 (default). The image-path terms inside the ground-diffraction weighting (Eq. 46/47) are evaluated unbounded. |

**Returns:** The attenuation `ΔLd` per band, in dB (`≥ 0`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

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

## ground_effect_adjustment

```python
ground_effect_adjustment(
    frequencies: NDArray[np.float64] | list[float],
    source_height: float,
    receiver_height: float,
    horizontal_distance: float,
    *,
    flow_resistivity: float | str = 'G',
) -> NDArray[np.float64]
```

Ground-effect adjustment `ΔLg` over an impedance plane (Eq. 28-35).

A point source over a locally-reacting impedance ground produces interference
between the direct and reflected rays. With the spherical reflection
coefficient `Q` (Chien-Soroka) and the Delany-Bazley impedance,
$\Delta L_g = 10 \cdot \log_{10}\{1 + (r_1/r_2)^2 \lvert Q \rvert^2 + 2 (r_1/r_2) \lvert Q \rvert \cdot I\}$ (Eq. 29), where `I`
(Eq. 30) is the in-band interference factor.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave-band centre frequencies, in Hz. |
| `source_height` | Source height above the ground `hs`, in metres (clamped to `>= 0.1`). |
| `receiver_height` | Receiver height above the ground `hr`, in metres (clamped to `>= 0.1`). |
| `horizontal_distance` | Horizontal source-receiver distance `dp`, in metres (`> 0`). |
| `flow_resistivity` | Ground flow resistivity `σ` in Pa·s/m², or a CNOSSOS class letter `"A"`-`"H"`. The default `"G"` (20e6, hard surfaces) is the CNOSSOS class covering the paved surroundings typical of heliports; the guidance's own suggestions, concrete $\sigma = 65 \times 10^6$ for city areas and grass $\sigma = 200 \times 10^3$ for rural areas (§A.4.3), can be passed as numeric values. |

**Returns:** The adjustment `ΔLg` per band, in dB (added to the level).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

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

## mean_flow_resistivity

```python
mean_flow_resistivity(
    lengths: NDArray[np.float64] | list[float],
    resistivities: NDArray[np.float64] | list[float],
) -> float
```

Logarithmic mean flow resistivity along a path (guidance Eq. 41).

When the ground type changes along a terrain profile, the guidance
averages the flow resistivity by the logarithm, weighted by the length of
each ground segment:
$\bar{\sigma} = 10^{\sum d_i \cdot \log_{10}(\sigma_i) / \sum d_i}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `lengths` | Segment lengths `dᵢ`, in metres (`> 0`), shape `(n,)`. |
| `resistivities` | Segment flow resistivities `σᵢ`, in Pa·s/m² (`> 0`), shape `(n,)`. |

**Returns:** The mean flow resistivity `σ̄`, in Pa·s/m².

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## mean_ground_plane

```python
mean_ground_plane(
    distances: NDArray[np.float64] | list[float],
    heights: NDArray[np.float64] | list[float],
) -> MeanGroundPlaneResult
```

The mean ground plane of a terrain section (guidance Eq. 36-40).

Fits $z = a \cdot d + b$ to the polyline of straight segments that
form the
terrain profile by continuous least squares (the residual is integrated
along `d`, not summed over the vertices), using the closed forms of
Eq. 37/38 with the segment integrals `A` and `B` of Eq. 39/40.

**Parameters**

| Name | Description |
| :--- | :--- |
| `distances` | Section distances `d`, in metres, strictly increasing, shape `(M,)` with $M \ge 2$ (arbitrary spacing). |
| `heights` | Terrain heights `z(d)`, in metres, shape `(M,)`. |

**Returns:** A [`MeanGroundPlaneResult`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#meangroundplaneresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## MeanGroundPlaneResult

```python
MeanGroundPlaneResult(
    slope: float,
    intercept: float,
    distances: NDArray[np.float64],
    heights: NDArray[np.float64],
)
```

A mean ground plane fitted to a terrain section (guidance Eq. 36-40).

ECAC Doc 32, 1st ed., assumes flat terrain; its guidance (§A.4.4)
represents a varying vertical section by the least-squares line
$z = a \cdot d + b$ through the terrain polyline, evaluated in
closed form from the per-segment integrals (Eq. 37-40). Equivalent source
and
receiver heights are then measured orthogonally to this plane and
substituted into the flat-ground equations.

**Attributes**

| Name | Description |
| :--- | :--- |
| `slope` | The fitted slope `a` (Eq. 37). |
| `intercept` | The fitted intercept `b`, in metres (Eq. 38). |
| `distances` | The section distances `d`, in metres, shape `(M,)`. |
| `heights` | The terrain heights `z(d)`, in metres, shape `(M,)`. |

### MeanGroundPlaneResult.equivalent_height()

```python
MeanGroundPlaneResult.equivalent_height(
    distance: float,
    height: float,
) -> float
```

The orthogonal (equivalent) height of a point above the plane.

Positive above the plane; the guidance substitutes these equivalent
heights, floored at 0.1 m for source and receiver, into the
flat-ground equations (§A.4.4).

### MeanGroundPlaneResult.height()

```python
MeanGroundPlaneResult.height(
    distance: float | NDArray[np.float64],
) -> NDArray[np.float64]
```

The plane height `a·d + b` at `distance`, in metres.

### MeanGroundPlaneResult.plot()

```python
MeanGroundPlaneResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the terrain section and the fitted mean ground plane.

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
    receiver_height: float = 1.2,
    ground_elevation: float = 0.0,
    airspeed: float | NDArray[np.float64] | list[float] | None = None,
    path_angle: float | NDArray[np.float64] | list[float] | None = None,
    heading: float | NDArray[np.float64] | list[float] | None = None,
    bank_angle: float | NDArray[np.float64] | list[float] | None = None,
    flow_resistivity: float | str | np.floating[Any] | np.integer[Any] = 'G',
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
    level_offset: float | NDArray[np.float64] | list[float] = 0.0,
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
    atmospheric_method: str = 'iso9613',
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | Sequence[NDArray[np.float64]] | None = None,
    terrain_resolution: float | None = None,
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
[`epnl_from_pnlt`](/phonometry/reference/api/aeroacoustics/aircraft-noise/#epnl_from_pnlt)).

The flight condition per point comes from the `airspeed`/`path_angle`
overrides when given (e.g. the smoothed values of a radar-track workflow),
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
| `receiver_height` | Microphone height above local ground, in metres (default 1.2). |
| `ground_elevation` | Ground elevation `z` at the site, in metres on the track datum (default 0); source and receiver heights above ground follow from it. |
| `airspeed` | Per-point airspeed override, scalar or shape `(N,)`. |
| `path_angle` | Per-point path-angle override, in degrees. |
| `heading` | Per-point heading override, in degrees. |
| `bank_angle` | Per-point bank-angle override, in degrees (positive starboard down). |
| `flow_resistivity` | Ground flow resistivity `σ` in Pa·s/m², or a CNOSSOS class letter (see [`ground_effect_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#ground_effect_adjustment)). |
| `temperature` | Air temperature, in °C (default 25, ICAO reference). |
| `relative_humidity` | Relative humidity, in % (default 70). |
| `pressure` | Ambient pressure, in kPa (default 101.325). |
| `level_offset` | Source-level offset `ΔEPNL` added to the hemisphere levels (Eq. 2 class substitution), in dB (default 0). Scalar or per track point, shape `(N,)`: Chapter-8 substitutions correct climb, level and descent conditions with different certification levels. |
| `scaling_factor` | Flight-condition scaling factor `F_fc` (default 2). |
| `triangles` | Optional precomputed flight-condition triangulation (see [`flight_condition_weights`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#flight_condition_weights)). |
| `atmospheric_method` | `"iso9613"` for the pure-tone Eq. 26/27 term (the guidance text), or `"sae"` for the SAE ARP 5534 band-integrated mapping used by the NORAH2 reference implementation (they agree to ~0.05 dB below 3.15 kHz). |
| `terrain` | Optional digital elevation model `(x, y, z)` on the track frame (`x` and `y` strictly increasing, `z` of shape `(len(y), len(x))`, all in metres on the track datum). When given, every emission-receiver pair is evaluated over its sampled vertical section (guidance §A.4.4/A.4.5): mean-ground-plane ground effect with equivalent heights, and rubber-band diffraction where terrain blocks the line of sight; `ground_elevation` is then taken from the model. The model must cover the whole track and the receiver (fabricating terrain beyond its edges is refused). |
| `terrain_resolution` | Section sampling step along the path, in metres (default: the elevation model's cell size; sections are capped at 20000 sampling intervals). |

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
    receiver_height: float = 1.2,
    ground_elevation: float | NDArray[np.float64] | list[list[float]] = 0.0,
    airspeed: float | NDArray[np.float64] | list[float] | None = None,
    path_angle: float | NDArray[np.float64] | list[float] | None = None,
    heading: float | NDArray[np.float64] | list[float] | None = None,
    bank_angle: float | NDArray[np.float64] | list[float] | None = None,
    flow_resistivity: float | str | NDArray[np.float64] | list[list[float]] = 'G',
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
    level_offset: float | NDArray[np.float64] | list[float] = 0.0,
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
    atmospheric_method: str = 'iso9613',
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | Sequence[NDArray[np.float64]] | None = None,
    terrain_resolution: float | None = None,
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
| `receiver_height` | Microphone height above local ground, in metres. |
| `ground_elevation` | Ground elevation, in metres on the track datum: a scalar, or one value per grid point (shape `(len(y), len(x))`) for receivers on uneven sites without a full elevation model. |
| `airspeed` | Per-point airspeed override (see [`rotorcraft_event_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraft_event_level)). |
| `path_angle` | Per-point path-angle override, in degrees. |
| `heading` | Per-point heading override, in degrees. |
| `bank_angle` | Per-point bank-angle override, in degrees. |
| `flow_resistivity` | Ground flow resistivity `σ` in Pa·s/m², a CNOSSOS class letter, or one value per grid point (shape `(len(y), len(x))`) for heterogeneous ground across the receivers (each receiver's two-ray model uses its local value). |
| `temperature` | Air temperature, in °C. |
| `relative_humidity` | Relative humidity, in %. |
| `pressure` | Ambient pressure, in kPa. |
| `level_offset` | Source-level offset `ΔEPNL` (Eq. 2), in dB, scalar or per track point. |
| `scaling_factor` | Flight-condition scaling factor `F_fc` (default 2). |
| `triangles` | Optional precomputed flight-condition triangulation. |
| `atmospheric_method` | `"iso9613"` or `"sae"` (see [`rotorcraft_event_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraft_event_level)). |
| `terrain` | Optional digital elevation model `(x, y, z)` (see [`rotorcraft_event_level`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraft_event_level)); it must cover the whole track and grid. Every emission-receiver pair then samples its own vertical section, so the cost grows with track points times grid points; keep contour grids modest with terrain. |
| `terrain_resolution` | Section sampling step, in metres (default: the elevation model's cell size; sections are capped at 20000 sampling intervals). |

**Returns:** A [`RotorcraftNoiseContourResult`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcraftnoisecontourresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

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
| `distance` | Reference distance, in metres (default 60). The standard NORAH database uses 60 m; when the data uses another polar distance (e.g. 70 m hover rings), pass this value as `reference_distance` to [`spherical_spreading_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#spherical_spreading_adjustment) and [`atmospheric_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#atmospheric_adjustment) so the propagation chain honours it. |

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

## spherical_spreading_adjustment

```python
spherical_spreading_adjustment(
    distance: float,
    *,
    reference_distance: float = 60.0,
) -> float
```

Spherical-spreading adjustment `ΔLs` of the hemisphere level (Eq. 24).

The hemisphere levels are defined at the reference distance `rh` (60 m in
the standard database), so at slant distance `r` the geometric spreading
adjustment is $\Delta L_s = -20 \cdot \log_{10}(r/r_h)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `distance` | Slant distance `r` from the rotorcraft to the observer, in metres (`> 0`). |
| `reference_distance` | Hemisphere reference distance `rh`, in metres (default 60). Pass [`RotorcraftHemisphere.distance`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#rotorcrafthemisphere) when the data uses a non-standard polar distance (e.g. 70 m hover rings). |

**Returns:** The spreading adjustment `ΔLs`, in dB (added to the level).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a distance is not strictly positive. |

## terrain_screening_adjustment

```python
terrain_screening_adjustment(
    frequencies: NDArray[np.float64] | list[float],
    source: tuple[float, float],
    receiver: tuple[float, float],
    distances: NDArray[np.float64] | list[float],
    heights: NDArray[np.float64] | list[float],
    *,
    flow_resistivity: float | str | NDArray[np.float64] | list[float] = 'G',
) -> TerrainScreeningResult
```

Ground effect and terrain screening over a vertical section (§A.4.4-A.4.5).

The terrain profile between the source and the receiver decides the
propagation regime:

* **Line of sight clear** (no profile point strictly above it): the
  section's mean ground plane (Eq. 36-40) supplies equivalent orthogonal
  heights (floored at 0.1 m) and the flat-ground two-ray model of
  §A.4.3 evaluates on the plane, with the log-mean flow resistivity
  (Eq. 41) when it varies along the path. Terrain points below the line
  of sight are never treated as diffracting obstacles (the guidance's
  topography rule, which avoids accidental screening in flat terrain).
* **Blocked**: the sound follows the shortest convex path over the
  terrain (the guidance's rubber band); its vertices are the diffraction
  edges. The attenuation combines the pure diffraction of the path
  difference `δ` (Eq. 42-44, capped at 25 dB) with the source-side and
  receiver-side ground effects weighted by their image-path diffractions
  (Eq. 45-47), each side using its own mean ground plane, equivalent
  heights and log-mean flow resistivity. The ground effect is not
  evaluated separately in this regime; bands with
  $\delta < -\lambda/20$ fall
  back to the clear-path evaluation (with terrain-only obstacles
  $\delta > 0$, so the rule engages for constructed screens below
  the line of sight rather than for terrain).

ECAC Doc 32, 1st ed., defines no screening or topography (its Eq. 12
propagation chain ends at the flat-ground `ΔLg`); this implements the
NORAH2 guidance sections A.4.4/A.4.5 and its noise-path appendices,
whose diffraction equations follow CNOSSOS-EU.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave-band centre frequencies, in Hz. |
| `source` | Source `(d, z)` in the section, in metres. |
| `receiver` | Receiver `(d, z)` in the section, in metres (the microphone point, i.e. ground plus microphone height). |
| `distances` | Terrain section distances `d`, in metres, strictly increasing, covering `[source d, receiver d]`. |
| `heights` | Terrain heights `z(d)`, in metres. |
| `flow_resistivity` | Ground flow resistivity: a value in Pa·s/m², a CNOSSOS class letter, or one value per profile segment (shape `(M−1,)`) averaged per sub-path by Eq. 41. |

**Returns:** A [`TerrainScreeningResult`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/#terrainscreeningresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## TerrainScreeningResult

```python
TerrainScreeningResult(
    frequencies: NDArray[np.float64],
    adjustment: NDArray[np.float64],
    screened: bool,
    path_difference: float,
    diffraction_points: NDArray[np.float64],
    source: tuple[float, float],
    receiver: tuple[float, float],
    distances: NDArray[np.float64],
    heights: NDArray[np.float64],
)
```

Ground and screening over a terrain section (guidance §A.4.4-A.4.5).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz, shape `(F,)`. |
| `adjustment` | The combined ground-and-screening adjustment per band, in dB, added to the received level in the Doc 32 Eq. 23 chain (it replaces the flat-ground `ΔLg`): the mean-ground-plane ground effect when the line of sight is clear, $-(\Delta L_d + \Delta L_g)$ of Eq. 45 when terrain blocks it. |
| `screened` | Whether terrain blocks the line of sight (any profile point strictly above it). |
| `path_difference` | The rubber-band path difference `δ`, in metres (`NaN` when unscreened). |
| `diffraction_points` | The diffracting edges `(d, z)` on the convex propagation path, shape `(n, 2)` (empty when unscreened). |
| `source` | The source `(d, z)`, in metres. |
| `receiver` | The receiver `(d, z)`, in metres. |
| `distances` | The section distances, in metres, shape `(M,)`. |
| `heights` | The section terrain heights, in metres, shape `(M,)`. |

### TerrainScreeningResult.plot()

```python
TerrainScreeningResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the section geometry: terrain, line of sight and sound path.
