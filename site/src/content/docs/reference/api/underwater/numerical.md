---
title: "underwater.propagation.numerical"
description: "Numerical models of underwater sound propagation (range-independent ocean)."
sidebar:
  label: "numerical"
---

Numerical models of underwater sound propagation (range-independent ocean).

Three complementary numerical solvers for the acoustic field in a
horizontally-stratified ocean waveguide, complementing the closed-form
transmission loss of [`phonometry.underwater.propagation.closed_form`](/phonometry/reference/api/underwater/closed-form/):

* [`normal_modes`](/phonometry/reference/api/underwater/numerical/#normal_modes) -- the normal-mode expansion. Solves the depth-separated
  Sturm-Liouville eigenvalue problem by finite differences and assembles the
  transmission loss from the propagating modes.
* [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace) -- ray tracing. Integrates the ray-trajectory equations
  through a sound-speed profile (Runge-Kutta), returning the ray paths and the
  travel time accumulated along each of them.
* [`parabolic_equation`](/phonometry/reference/api/underwater/numerical/#parabolic_equation) -- the standard (Tappert) parabolic equation, solved
  with the split-step Fourier algorithm, returning the transmission-loss field.

All three are implemented clean-room from Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics* (2nd ed., Springer 2011): the modal derivation
(Ch. 5, Eqs. 5.3-5.17), the ray equations (Ch. 3, Eqs. 3.23-3.24) and the
split-step Fourier PE (Ch. 6). They are validated against analytic oracles: the
ideal (pressure-release) waveguide's exact modes, the circular-arc ray paths of
a linear sound-speed gradient together with the closed-form travel time along
them (Medwin & Clay, *Fundamentals of Acoustical Oceanography*, Academic Press
1998, Eq. (3.3.20)), and mutual agreement of the PE and normal-mode
transmission loss for a range-independent waveguide.

Densities are in kg/m3, sound speeds in m/s, depths and ranges in metres,
frequencies in Hz. The water column has a pressure-release surface at z = 0.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## normal_modes

```python
normal_modes(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    receiver_depth: float,
    ranges_m: NDArray[np.float64] | list[float] | None = None,
    density: float = 1000.0,
    bottom: str = 'pressure-release',
    n_depth_points: int | None = None,
) -> NormalModeResult
```

Normal-mode transmission loss for a range-independent waveguide.

Solves the depth-separated Sturm-Liouville problem (Jensen Eq. 5.3) on a
uniform finite-difference grid, then assembles the coherent transmission
loss from the propagating modes (Eq. 5.17).

The finite-difference eigenvalues carry an $O(dz^2)$ error that
grows with the mode's vertical wavenumber, so near-cutoff modes need a
fine grid. Two guards apply: eigenvalues inside the scheme's error band
($k_r^2 \le \max(k^2)^2 \, dz^2 / 12$) are discarded as numerically
indistinguishable from cutoff, and a
[`PhonometryWarning`](/phonometry/reference/api/filters/phonometry/#phonometrywarning) is emitted when a
retained mode sits within ten times that band (increase `n_depth_points`
to resolve it).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Source frequency, in Hz. |
| `depths` | Depth samples of the sound-speed profile, in metres, starting at the surface `z = 0` and strictly increasing to the bottom. |
| `sound_speeds` | Sound speed at each depth, in m/s. |
| `source_depth` | Source depth `zs`, in metres. |
| `receiver_depth` | Receiver depth for the transmission-loss slice, in m. |
| `ranges_m` | Ranges at which to evaluate the loss, in metres; defaults to 100 m to 10 km. |
| `density` | Water density (constant), in kg/m3. |
| `bottom` | `"pressure-release"` (default) or `"rigid"`. |
| `n_depth_points` | Number of finite-difference depth points. Default (`None`): derived from the physics as $\max(400, \operatorname{ceil}(60 D f / c_{\mathrm{min}}))$, which keeps the near-cutoff eigenvalue error small at any frequency/depth combination, capped at 20 000 points (very high $f D$ products exceed the cap; the near-cutoff warning then indicates whether the capped grid suffices, and an explicit `n_depth_points` overrides the cap). |

**Returns:** A [`NormalModeResult`](/phonometry/reference/api/underwater/numerical/#normalmoderesult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## NormalModeResult

```python
NormalModeResult(
    frequency: float,
    wavenumbers: NDArray[np.float64],
    mode_depths: NDArray[np.float64],
    mode_functions: NDArray[np.float64],
    ranges: NDArray[np.float64],
    transmission_loss: NDArray[np.float64],
    receiver_depth: float,
    source_depth: float,
)
```

Normal-mode solution of a range-independent waveguide.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Source frequency, in Hz. |
| `wavenumbers` | Horizontal wavenumbers `krm` of the propagating modes, in rad/m (descending order). |
| `mode_depths` | Depth grid of the mode functions, in metres. |
| `mode_functions` | Orthonormalised mode shapes `Ψm(z)`, shape `(n_modes, n_depths)`. |
| `ranges` | Ranges at which the transmission loss is evaluated, in metres. |
| `transmission_loss` | Coherent transmission loss at `receiver_depth` per range, in dB. |
| `receiver_depth` | Receiver depth of the transmission-loss slice, in m. |
| `source_depth` | Source depth, in metres. |

### NormalModeResult.plot()

```python
NormalModeResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the transmission loss versus range (loss increasing downward).

## parabolic_equation

```python
parabolic_equation(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    max_range: float = 10000.0,
    range_step: float = 10.0,
    n_depth_points: int = 1024,
) -> ParabolicEquationResult
```

Transmission-loss field from the standard (Tappert) parabolic
equation.

Marches the split-step Fourier solution (Jensen Ch. 6) in range with a
discrete sine transform in depth, enforcing a pressure-release surface at
`z = 0` and bottom at `z = water_depth`. The envelope is related to
pressure by $p = \psi \, e^{i(k_0 r - \pi/4)} / \sqrt{r}$ and
$\mathrm{TL} = -20 \log_{10}(\lvert \psi \rvert / \sqrt{r})$
(Eqs. 6.70-6.71), using a Gaussian starter.

The standard PE is **paraxial**: it is accurate for propagation within
roughly ±15-20° of the horizontal (Jensen §6.2). Steep modes therefore
carry a phase error that shows at short and intermediate range in
shallow-waveguide problems (a few dB against the exact field below a few
water depths of range), converging at long range; the free-field
calibration itself is exact to ~1e-4 dB at the default `range_step`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Source frequency, in Hz. |
| `depths` | Depth samples of the profile, in metres, from `z = 0`. |
| `sound_speeds` | Sound speed at each depth, in m/s. |
| `source_depth` | Source depth, in metres. |
| `max_range` | Maximum range, in metres. |
| `range_step` | Range marching step $\Delta r$, in metres. |
| `n_depth_points` | Number of depth points (interior sine-transform grid). |

**Returns:** A [`ParabolicEquationResult`](/phonometry/reference/api/underwater/numerical/#parabolicequationresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## ParabolicEquationResult

```python
ParabolicEquationResult(
    frequency: float,
    ranges: NDArray[np.float64],
    depths: NDArray[np.float64],
    transmission_loss: NDArray[np.float64],
    source_depth: float,
)
```

Parabolic-equation transmission-loss field.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Source frequency, in Hz. |
| `ranges` | Range grid, in metres. |
| `depths` | Depth grid, in metres. |
| `transmission_loss` | Transmission-loss field `TL(z, r)`, in dB, shape `(n_depths, n_ranges)`. |
| `source_depth` | Source depth, in metres. |

### ParabolicEquationResult.plot()

```python
ParabolicEquationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the transmission-loss field (depth increasing downward).

## ray_trace

```python
ray_trace(
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    launch_angles_deg: NDArray[np.float64] | list[float],
    max_range: float = 10000.0,
    n_steps: int = 2000,
) -> RayTraceResult
```

Trace acoustic rays through a range-independent sound-speed profile.

Integrates the ray-trajectory equations (Jensen Eqs. 3.23-3.24) with a
fixed-step fourth-order Runge-Kutta scheme, reflecting at the pressure-release
surface (`z = 0`) and the bottom (`z = water_depth`).

The travel time is a third state of that same Runge-Kutta step rather than a
quadrature run over the finished path: with the range-invariant Snell
parameter $\xi = \cos\theta_0 / c(z_s)$ it obeys
$dt/dr = 1/(\xi c^2)$, so it is integrated with the very stages that
place the ray and cannot drift from the geometry actually returned. This is
the same ray core, and the same travel-time equation, as the atmospheric
[`atmospheric_ray_paths`](/phonometry/reference/api/environment/refraction/#atmospheric_ray_paths)
(which reflects at the ground instead of at the sea surface). Reflections
cost no time, so the accumulated time stays continuous across them.

**Parameters**

| Name | Description |
| :--- | :--- |
| `depths` | Depth samples of the profile, in metres, from `z = 0`. |
| `sound_speeds` | Sound speed at each depth, in m/s. |
| `source_depth` | Source depth, in metres. |
| `launch_angles_deg` | Launch angles from the horizontal, in degrees (positive downward). |
| `max_range` | Maximum horizontal range to trace, in metres. |
| `n_steps` | Number of integration steps per ray. |

**Returns:** A [`RayTraceResult`](/phonometry/reference/api/underwater/numerical/#raytraceresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## RayTraceResult

```python
RayTraceResult(
    launch_angles: NDArray[np.float64],
    ranges: NDArray[np.float64],
    depths: NDArray[np.float64],
    travel_times: NDArray[np.float64],
    source_depth: float,
    water_depth: float,
)
```

Ray-tracing solution through a sound-speed profile.

**Attributes**

| Name | Description |
| :--- | :--- |
| `launch_angles` | Launch angles from the horizontal, in degrees. |
| `ranges` | Per-ray horizontal ranges, in metres, shape `(n_rays, n_steps)`. |
| `depths` | Per-ray depths, in metres, shape `(n_rays, n_steps)`. |
| `travel_times` | Per-ray cumulative travel times, in seconds, shape `(n_rays, n_steps)` (zero at the source, increasing along the ray). |
| `source_depth` | Source depth, in metres. |
| `water_depth` | Water-column depth, in metres. |

### RayTraceResult.plot()

```python
RayTraceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the ray paths (depth increasing downward).
