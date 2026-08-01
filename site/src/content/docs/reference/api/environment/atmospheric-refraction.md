---
title: "environmental.atmospheric_refraction"
description: "Atmospheric refraction: ray tracing and the parabolic equation (PE)."
sidebar:
  label: "atmospheric_refraction"
---

Atmospheric refraction: ray tracing and the parabolic equation (PE).

Sound propagation outdoors is bent by vertical gradients of the effective sound
speed $c_{eff}(z) = c(z) + u(z)$ (the adiabatic sound speed plus the
component of the wind in the propagation direction, Salomons Eq. (4.4)). This
module
predicts that refraction with two complementary models, clean-room from
Salomons, *Computational Atmospheric Acoustics* (Springer, 2001) and
Attenborough & Van Renterghem, *Predicting Outdoor Sound* (2e, CRC, 2021,
Ch. 11), and it is the refracting-atmosphere counterpart of the range-independent
ocean solvers in [`phonometry.underwater.numerical_propagation`](/phonometry/reference/api/underwater/numerical-propagation/):

* [`atmospheric_ray_paths`](/phonometry/reference/api/environment/atmospheric-refraction/#atmospheric_ray_paths) -- geometrical acoustics. Integrates Snell's law
  for sound rays (Salomons Eq. (4.3)) with a fixed-step Runge-Kutta scheme,
  returning the curved ray paths, their turning points, travel times and ground
  reflections. It shares its ray core with the ocean `ray_trace`: the
  atmospheric version reflects at the ground ($z = 0$) instead of the sea
  surface and marches an upward-open half space.
* [`atmospheric_parabolic_equation`](/phonometry/reference/api/environment/atmospheric-refraction/#atmospheric_parabolic_equation) -- the Green's Function Parabolic
  Equation (GFPE, Salomons Appendix H). Marches the one-way wave equation in
  range with the split-step Fourier algorithm (the same range-marching family
  as the ocean `parabolic_equation`), a Gaussian starter (Salomons
  Eq. (G.64)), an absorbing layer at the top of the grid (Salomons Sec. G.9)
  and a finite-impedance ground condition through the plane-wave reflection
  coefficient $R(k_z) = (k_z Z - k_0)/(k_z Z + k_0)$ plus the
  surface-wave residue of its pole (Salomons Eqs. (H.28), (H.49)). It returns
  the relative sound level (dB re free field) over the range-height plane.

For a linear effective sound-speed profile the ray paths are exact circular
arcs of radius [`ray_curvature_radius`](/phonometry/reference/api/environment/atmospheric-refraction/#ray_curvature_radius), and an upward-refracting linear
profile has a closed-form [`shadow_zone_distance`](/phonometry/reference/api/environment/atmospheric-refraction/#shadow_zone_distance); both anchor the ray
model. The PE is anchored against the exact spherical-wave ground effect
([`phonometry.environmental.ground_effect`](/phonometry/reference/api/environment/ground-barriers/#ground_effect)) in the homogeneous limit
(gradient zero), which it reproduces to a few tenths of a dB on the default
grid (finer `height_step` converges it further).

The ground impedance is taken in the $e^{-i \omega t}$ convention of
Salomons (a passive ground has $\operatorname{Im}(Z) > 0$), shared with
[`phonometry.environmental.ground_barriers`](/phonometry/reference/api/environment/ground-barriers/). The porous models of
`phonometry.materials` work in the opposite $e^{+j \omega t}$
convention ($\operatorname{Im}(Z) < 0$), so an impedance derived from
them (`flow_resistivity=` or a
`PorousMediumResult`) is conjugated internally before entering the PE ground
condition. Heights and ranges are in metres, sound speeds in m/s and
frequencies in Hz.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## atmospheric_parabolic_equation

```python
atmospheric_parabolic_equation(
    frequency_hz: float,
    profile: EffectiveSoundSpeedProfile,
    *,
    source_height: float,
    impedance: ArrayLike | PorousMediumResult | None = None,
    flow_resistivity: float | None = None,
    model: Literal['delany_bazley', 'miki'] = 'delany_bazley',
    max_range: float = 1000.0,
    max_height: float = 100.0,
    range_step: float | None = None,
    height_step: float | None = None,
    air_density: float = 1.205,
) -> AtmosphericPEResult
```

Relative-level field from the Green's Function Parabolic Equation (GFPE).

Marches the split-step Fourier solution of the one-way wave equation
(Salomons Appendix H) in range. Each step transforms the field to the
vertical-wavenumber domain, applies the free-space propagator
$\exp\!\left[ i \, dr \left( \sqrt{k_a^2 - k_z^2} - k_a \right) \right]$ together with the ground reflection
$R(k_z) = (k_z Z - k_0)/(k_z Z + k_0)$ (Eq. (H.28)), transforms
back, adds the surface-wave residue of the reflection pole at
$k_z = -k_0/Z$ (the third term of Eq. (H.49), present for a passive
ground, $\operatorname{Im}(Z) > 0$) and applies the refraction phase
screen $\exp\!\left[ i \, dr \left( k(z) - k_a \right) \right]$
(Eq. (H.58)). The source is a Gaussian starter with its ground image
(Eqs. (G.64), (G.76)) and an absorbing layer at the top of the grid
(Sec. G.9) suppresses top-boundary reflections. The reference wavenumber
$k_a = k_0$ is taken at the ground.

The relative sound level re the free field is
$\Delta L(z, r) = 20 \lg(|p(z, r)| \, R_1)$ with $R_1$ the
direct source-receiver distance (Salomons Eq. (3.6)); in a homogeneous
atmosphere it reproduces the spherical-wave ground effect of
[`ground_effect`](/phonometry/reference/api/environment/ground-barriers/#ground_effect).

The ground surface impedance is either supplied through `impedance` (a
normalized complex value/array, or a
[`PorousMediumResult`](/phonometry/reference/api/materials/porous-absorber/#porousmediumresult)) or derived from an
effective `flow_resistivity` (Pa s/m2) via the `model` porous model.
Exactly one of the two must be given. A plain `impedance` value is taken
in the $e^{-i \omega t}$ convention ($\operatorname{Im}(Z) > 0$
for a passive ground); a `PorousMediumResult` or `flow_resistivity` is
conjugated internally from the materials' $e^{+j \omega t}$
convention.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Source frequency, in Hz. |
| `profile` | The effective sound-speed profile. |
| `source_height` | Source height `zs`, in metres (> 0). |
| `impedance` | Normalized ground impedance, or a `PorousMediumResult`. |
| `flow_resistivity` | Effective flow resistivity `sigma` (Pa s/m2), as an alternative to `impedance`. |
| `model` | Porous model for `flow_resistivity`. |
| `max_range` | Maximum range, in metres. |
| `max_height` | Top of the output height grid, in metres (the receiver region of interest; an absorbing layer is added above it). |
| `range_step` | Range marching step `dr`, in metres. Default (`None`): one wavelength. |
| `height_step` | Vertical grid spacing `dz`, in metres. Default (`None`): a tenth of a wavelength (Salomons Sec. G.2). |
| `air_density` | Air density `rho`, in kg/m3 (for the porous model). |

**Returns:** An [`AtmosphericPEResult`](/phonometry/reference/api/environment/atmospheric-refraction/#atmosphericperesult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or the impedance is unspecified. |

## atmospheric_ray_paths

```python
atmospheric_ray_paths(
    profile: EffectiveSoundSpeedProfile,
    *,
    source_height: float,
    launch_angles_deg: ArrayLike,
    max_range: float = 1000.0,
    n_steps: int = 2000,
) -> AtmosphericRayResult
```

Trace sound rays through a refracting atmosphere over a ground surface.

Integrates Snell's law for sound rays (Salomons Eq. (4.3),
$\cos(\gamma)/c(z) = \text{const}$) with a fixed-step fourth-order
Runge-Kutta scheme, marching in range and reflecting specularly at the
ground ($z = 0$). The state is the height `z` and the vertical
slowness $\zeta = \sin(\gamma)/c$; with the range-invariant
$\xi = \cos(\gamma_0)/c(z_s)$ the equations are
$dz/dr = \zeta/\xi$ and
$d\zeta/dr = -(dc/dz)/(c^3 \xi)$, the same ray core as the ocean
[`ray_trace`](/phonometry/reference/api/underwater/numerical-propagation/#ray_trace) (with a ground
reflection in place of the sea surface). The travel time accumulates
$dt/dr = 1/(\xi c^2)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `profile` | The effective sound-speed profile (see [`linear_sound_speed_profile`](/phonometry/reference/api/environment/atmospheric-refraction/#linear_sound_speed_profile) / [`log_linear_sound_speed_profile`](/phonometry/reference/api/environment/atmospheric-refraction/#log_linear_sound_speed_profile)). |
| `source_height` | Source height `zs`, in metres (>= 0). |
| `launch_angles_deg` | Launch angles from the horizontal, in degrees (positive upward), within `(-90, 90)`. |
| `max_range` | Maximum horizontal range to trace, in metres. |
| `n_steps` | Number of range steps per ray (>= 2). |

**Returns:** An [`AtmosphericRayResult`](/phonometry/reference/api/environment/atmospheric-refraction/#atmosphericrayresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## AtmosphericPEResult

```python
AtmosphericPEResult(
    frequency: float,
    ranges: NDArray[np.float64],
    heights: NDArray[np.float64],
    relative_level: NDArray[np.float64],
    source_height: float,
    normalized_impedance: complex,
)
```

Parabolic-equation relative-level field in a refracting atmosphere.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Source frequency, in Hz. |
| `ranges` | Range grid, in metres. |
| `heights` | Height grid of the output field, in metres. |
| `relative_level` | Relative sound level $\Delta L(z, r)$ (dB re free field), shape `(n_heights, n_ranges)`. |
| `source_height` | Source height, in metres. |
| `normalized_impedance` | Normalized ground impedance used (complex). |

### AtmosphericPEResult.level_at_height()

```python
AtmosphericPEResult.level_at_height(height: float) -> NDArray[np.float64]
```

Relative sound level versus range at the grid height nearest `height`.

### AtmosphericPEResult.plot()

```python
AtmosphericPEResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the relative-level field over the range-height plane.

## AtmosphericRayResult

```python
AtmosphericRayResult(
    launch_angles: NDArray[np.float64],
    ranges: NDArray[np.float64],
    heights: NDArray[np.float64],
    travel_times: NDArray[np.float64],
    turning_points: NDArray[np.int_],
    ground_reflections: NDArray[np.int_],
    source_height: float,
)
```

Ray-tracing solution through an effective sound-speed profile.

**Attributes**

| Name | Description |
| :--- | :--- |
| `launch_angles` | Launch angles from the horizontal, in degrees. |
| `ranges` | Per-ray horizontal ranges, in metres, shape `(n_rays, n_steps)`. |
| `heights` | Per-ray heights, in metres, shape `(n_rays, n_steps)`. |
| `travel_times` | Per-ray cumulative travel times, in seconds, shape `(n_rays, n_steps)`. |
| `turning_points` | Number of turning points (height extrema) per ray. |
| `ground_reflections` | Number of ground reflections per ray. |
| `source_height` | Source height, in metres. |

### AtmosphericRayResult.plot()

```python
AtmosphericRayResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the curved ray paths (height on the vertical axis).

## EffectiveSoundSpeedProfile

```python
EffectiveSoundSpeedProfile(
    heights: NDArray[np.float64],
    sound_speeds: NDArray[np.float64],
    description: str = '',
)
```

Vertical profile of the effective sound speed `c_eff(z)`.

The profile is sampled on a strictly increasing height grid starting at the
ground ($z = 0$); intermediate heights are taken as piecewise linear,
so a two-point profile represents an exact linear gradient.

**Attributes**

| Name | Description |
| :--- | :--- |
| `heights` | Heights `z` above the ground, in metres (from $z = 0$). |
| `sound_speeds` | Effective sound speed at each height, in m/s. |
| `description` | Short human-readable label (e.g. `"linear, +0.1 s^-1"`). |

### EffectiveSoundSpeedProfile.plot()

```python
EffectiveSoundSpeedProfile.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the effective sound-speed profile (height on the vertical axis).

### EffectiveSoundSpeedProfile.speed_at()

```python
EffectiveSoundSpeedProfile.speed_at(
    height: ArrayLike,
) -> NDArray[np.float64]
```

Piecewise-linear effective sound speed at one or more heights.

## linear_sound_speed_profile

```python
linear_sound_speed_profile(
    gradient: float,
    *,
    ground_speed: float = 343.0,
    max_height: float = 100.0,
) -> EffectiveSoundSpeedProfile
```

Linear effective sound-speed profile (constant vertical gradient).

The profile is $c_{eff}(z) = c_0 + \text{gradient} \cdot z$. A
positive `gradient` (sound speed increasing with height) refracts sound
downward (favourable propagation); a negative gradient refracts it upward
and creates an acoustic shadow near the ground (Salomons Sec. 4.2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `gradient` | Vertical gradient `dc/dz`, in s^-1 (m/s per m). |
| `ground_speed` | Sound speed `c0` at the ground, in m/s. |
| `max_height` | Top of the sampled profile, in metres. |

**Returns:** A two-point [`EffectiveSoundSpeedProfile`](/phonometry/reference/api/environment/atmospheric-refraction/#effectivesoundspeedprofile).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `max_height` or `ground_speed` is not positive, or the profile turns non-positive within `[0, max_height]`. |

## log_linear_sound_speed_profile

```python
log_linear_sound_speed_profile(
    b: float,
    *,
    ground_speed: float = 340.0,
    roughness_length: float = 0.1,
    max_height: float = 100.0,
    n_points: int = 128,
) -> EffectiveSoundSpeedProfile
```

Logarithmic effective sound-speed profile (surface layer).

The profile is $c_{eff}(z) = c_0 + b \ln(1 + z/z_0)$, the realistic
surface-layer profile of Salomons Eq. (4.5): `b` is
the strength of the gradient (`+1 m/s` for a typical downward-refracting
atmosphere, `-1 m/s` for an upward-refracting one) and `z0` is the
aerodynamic roughness length (about 0.1 m for grassland). The steep gradient
near the ground is resolved by sampling the height grid logarithmically.

**Parameters**

| Name | Description |
| :--- | :--- |
| `b` | Profile strength `b`, in m/s (positive: downward refraction). |
| `ground_speed` | Sound speed `c0` at the ground, in m/s. |
| `roughness_length` | Aerodynamic roughness length `z0`, in metres. |
| `max_height` | Top of the sampled profile, in metres. |
| `n_points` | Number of samples of the height grid (>= 2). |

**Returns:** An [`EffectiveSoundSpeedProfile`](/phonometry/reference/api/environment/atmospheric-refraction/#effectivesoundspeedprofile).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On non-positive `c0`, `z0` or `max_height`, fewer than two points, or a profile that turns non-positive. |

## ray_curvature_radius

```python
ray_curvature_radius(
    gradient: float,
    *,
    ground_speed: float = 343.0,
    launch_angle_deg: float = 0.0,
) -> float
```

Radius of curvature of a sound ray in a linear sound-speed gradient.

In a linear effective sound-speed profile a sound ray is an exact circular
arc of radius $R_c = 1/(|\text{gradient}| \, \xi)$ with the Snell
invariant $\xi = \cos(\theta_0)/c$, evaluated at the launch height
(Salomons Sec. 4.4; Attenborough Ch. 11). For a ray launched from the
height where the speed is `ground_speed` ($c_0$),
$R_c = c_0 / (|\text{gradient}| \cos\theta_0)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `gradient` | Vertical gradient `dc/dz`, in s^-1 (must be non-zero). |
| `ground_speed` | Sound speed at the launch height, in m/s. |
| `launch_angle_deg` | Launch angle from the horizontal, in degrees. |

**Returns:** The radius of curvature `Rc`, in metres (always positive).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the gradient is zero (a straight ray) or the launch angle is not within `(-90, 90)` degrees. |

## shadow_zone_distance

```python
shadow_zone_distance(
    gradient: float,
    source_height: float,
    receiver_height: float,
    *,
    ground_speed: float = 343.0,
) -> float
```

Distance to the acoustic shadow boundary, upward-refracting profile.

For a linear upward-refracting profile (`gradient < 0`) the ground-level
ray that just grazes the surface bounds a region beyond which no direct or
once-reflected ray arrives (Salomons Sec. 4.4; Attenborough Ch. 11). With a
ray radius $R_c = c_0/|\text{gradient}|$ the limiting horizontal
distance is the closed form:

$$
x_{shadow} = \sqrt{2 R_c} \left( \sqrt{h_s} + \sqrt{h_r} \right)
$$

valid for source and receiver heights $h_s$, $h_r$ small
compared with $R_c$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `gradient` | Vertical gradient `dc/dz`, in s^-1 (must be negative for a shadow zone to exist). |
| `source_height` | Source height `hs`, in metres (>= 0). |
| `receiver_height` | Receiver height `hr`, in metres (>= 0). |
| `ground_speed` | Sound speed `c0` at the ground, in m/s. |

**Returns:** The horizontal shadow-boundary distance, in metres.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the gradient is not negative, or a height is negative. |
