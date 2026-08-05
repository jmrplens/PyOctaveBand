---
title: "aircraft.rotorcraft_propagation"
description: "Rotorcraft propagation, ground effect and screening (ECAC Doc 32 / NORAH2)."
sidebar:
  label: "rotorcraft_propagation"
---

Rotorcraft propagation, ground effect and screening (ECAC Doc 32 / NORAH2).

Between the noise hemisphere that describes a helicopter and the receiver on the
ground lies the path, and the path knows nothing about rotorcraft. The ECAC
Doc 32 propagation chain
$\Delta L_p = \Delta L_s + \Delta L_a + \Delta L_g$ adds spherical
spreading, atmospheric absorption and the ground effect of a point source over
an impedance plane; the NORAH2 guidance extends that last term to real terrain,
where a fitted mean ground plane replaces the flat ground and a blocked line of
sight becomes diffraction. Every function here takes a geometry and a spectrum;
[`rotorcraft_noise`](/phonometry/reference/api/aeroacoustics/rotorcraft-noise/) holds the hemisphere source and the
event chain that call them.

This module provides the propagation primitives of the method (clean-room, from
the NORAH2 guidance SC01.D1.5d, the basis of ECAC Doc 32):

* [`spherical_spreading_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#spherical_spreading_adjustment) --
  $\Delta L_s = -20 \cdot \log_{10}(r/60)$ (Eq. 24).
* [`atmospheric_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#atmospheric_adjustment) --
  $\Delta L_a = -\alpha(f) \cdot (r - 60)$ with the ISO 9613-1
  pure-tone coefficient (Eq. 26/27), reusing
  [`air_attenuation`](/phonometry/reference/api/environment/air-absorption/#air_attenuation).
* [`ground_effect_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#ground_effect_adjustment) -- `ΔLg` for a point source over an impedance
  plane (Chien-Soroka, Eq. 28-35) with the Delany-Bazley one-parameter impedance
  and the CNOSSOS flow-resistivity classes.
* [`mean_ground_plane`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#mean_ground_plane) -- the least-squares plane through a terrain section
  (Eq. 36-40), whose equivalent orthogonal heights carry a varying profile into
  the flat-ground equations.
* [`mean_flow_resistivity`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#mean_flow_resistivity) -- the log-mean flow resistivity of a path that
  crosses several ground types (Eq. 41).
* [`diffraction_attenuation`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#diffraction_attenuation) -- the pure diffraction attenuation `ΔLd` of
  a path difference (Eq. 42-44).
* [`terrain_screening_adjustment`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#terrain_screening_adjustment) -- the combined ground-and-screening
  adjustment over a vertical section (§A.4.4-A.4.5, Eq. 45-47): the
  mean-ground-plane ground effect while the line of sight is clear, the
  rubber-band diffraction over the terrain once it is blocked.

ECAC Doc 32, 1st ed., defines no topography or screening at all: its Eq. 12
chain ends at the flat-ground `ΔLg`. The mean ground plane, the log-mean flow
resistivity and the diffraction of §A.4.4-A.4.5 come from the guidance, whose
diffraction equations follow CNOSSOS-EU.

Source (clean-room): ECAC Doc 32, 1st ed.; NORAH2 rotorcraft-noise modelling
guidance (EASA.2020.FC.06 SC01.D1.5d), §A.4. The atmospheric term is validated
against the guidance Table 4 (one-third-octave attenuation per km at ICAO
reference conditions); the ground and screening chain is validated end to end,
inside the event chain, against the NORAH2 reference implementation outputs for
the ARP verification cases.

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

**Returns:** A [`MeanGroundPlaneResult`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#meangroundplaneresult).

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

The plane height $a\,d + b$ at `distance`, in metres.

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

**Returns:** A [`TerrainScreeningResult`](/phonometry/reference/api/aeroacoustics/rotorcraft-propagation/#terrainscreeningresult).

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
