---
title: "underwater.propagation.numerical"
description: "Numerical models of underwater sound propagation (range-independent ocean)."
sidebar:
  label: "numerical"
---

Numerical models of underwater sound propagation (range-independent ocean).

Four complementary numerical solvers for the acoustic field in a
horizontally-stratified ocean waveguide, complementing the closed-form
propagation loss of [`phonometry.underwater.propagation.closed_form`](/phonometry/reference/api/underwater/closed-form/):

* [`normal_modes`](/phonometry/reference/api/underwater/numerical/#normal_modes) -- the normal-mode expansion. Solves the depth-separated
  Sturm-Liouville eigenvalue problem by finite differences and assembles the
  propagation loss from the propagating modes.
* [`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace) -- ray tracing. Integrates the ray-trajectory equations
  through a sound-speed profile (Runge-Kutta), returning the ray paths and the
  travel time accumulated along each of them, and no amplitude.
* [`gaussian_beams`](/phonometry/reference/api/underwater/numerical/#gaussian_beams) -- Gaussian beam tracing. Hangs a beam on each of those
  rays and sums them into a propagation-loss field, which is finite at a caustic
  and decays smoothly into a shadow zone where ray theory has nothing to say.
* [`parabolic_equation`](/phonometry/reference/api/underwater/numerical/#parabolic_equation) -- the standard (Tappert) parabolic equation, solved
  with the split-step Fourier algorithm, returning the propagation-loss field.

All four are implemented clean-room from Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics* (2nd ed., Springer 2011): the modal derivation
(Ch. 5, Eqs. 5.3-5.17), the ray equations (Ch. 3, Eqs. 3.23-3.24), the Gaussian
beams of Sect. 3.5 (Eqs. 3.88-3.92) and the split-step Fourier PE (Ch. 6). They
are validated against analytic oracles: the ideal (pressure-release) waveguide's
exact modes and its image-source sum, the circular-arc ray paths of a linear
sound-speed gradient together with the closed-form travel time along them
(Medwin & Clay, *Fundamentals of Acoustical Oceanography*, Academic Press 1998,
Eq. (3.3.20)), free-field spherical spreading, and mutual agreement of the PE
and normal-mode propagation loss for a range-independent waveguide.

The three field solvers report the same quantity on the same terms, so their
propagation losses can be laid side by side: `normal_modes` on a range slice
at one receiver depth, `gaussian_beams` and `parabolic_equation` on a
(depth, range) grid. Which of them to reach for is a question of frequency and
of what is being asked; the guide's solver table sets it out.

Densities are in kg/m3, sound speeds in m/s, depths and ranges in metres,
frequencies in Hz. The water column has a pressure-release surface at z = 0.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## gaussian_beams

```python
gaussian_beams(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    max_range: float = 10000.0,
    ranges_m: NDArray[np.float64] | list[float] | None = None,
    receiver_depths_m: NDArray[np.float64] | list[float] | None = None,
    n_depth_points: int = 200,
    max_angle_deg: float = 80.0,
    n_beams: int | None = None,
    beam_width: float | None = None,
    range_step: float = 25.0,
    bottom: str = 'pressure-release',
) -> GaussianBeamResult
```

Propagation-loss field from Gaussian beam tracing.

Hangs a Gaussian beam on each ray of a launch fan (Jensen Eq. 3.88) and sums
them over the fan with the weight of Eq. (3.92). The rays are the ones
[`ray_trace`](/phonometry/reference/api/underwater/numerical/#ray_trace) draws, integrated by the same marcher through the same
profile; what is added is the dynamic pair $(q, p)$ of Eq. (3.58),
started from the complex conditions of Eq. (3.91) that make each ray the
axis of a beam of initial half-width `beam_width` and flat wavefront.

The point of the beams is that the answer stays finite. Ray theory's
amplitude, Eq. (3.65), divides by the ray-tube spreading, which vanishes on
a caustic and gives an infinity there (Sect. 3.4.1) and nothing at all in a
shadow zone. Complex $q$ cannot vanish, so this field needs no KMAH
index and no minimum-width floor, is finite wherever a beam reaches, and
falls into a shadow zone gradually rather than off a cliff, which is what
the exact solution does (Figs. 3.11, 3.17). See
[`GaussianBeamResult`](/phonometry/reference/api/underwater/numerical/#gaussianbeamresult) for the one place it still reports an infinity,
which is the wedge no beam of the fan illuminates at all.

The limits are worth knowing before the numbers are believed.

* **Ray theory's own regime** (Sect. 3.4.2): "the wavelength should be
  substantially smaller than any physical scale in the problem". This is the
  limit that bites hardest and the one a plausible-looking answer hides
  best. At 20 Hz in 100 m of water the depth is 1.3 wavelengths, two modes
  propagate, and the quarter-depth cap on the beam width leaves a beam a
  third of a wavelength across: against the image-source sum from 200 m to
  5 km the loss then comes out 2 to 8 dB high, and it moves by decibels when
  the fan is opened or the beam count multiplied by 150, so there is nothing
  it is converging to. Use [`normal_modes`](/phonometry/reference/api/underwater/numerical/#normal_modes) there, which is exact in
  that regime for the cost of two modes.
* **The fan is truncated** at `max_angle_deg`, and a waveguide with two
  perfectly reflecting boundaries is the worst case for that, because
  nothing but $1/R$ attenuates the steep multiple bounces. Measured on
  the ideal 1000 m guide at 300 Hz, source at 300 m and receiver at 600 m,
  against the image-source sum at 2, 5 and 10 km: a fan to 80 degrees is
  0.27, 4.06 and 2.52 dB out, a fan to 85 degrees 0.21, 1.32 and 1.91 dB,
  and a fan to 88 degrees 0.0002, 0.0003 and 0.0004 dB. Cutting the *oracle*
  to the same half-angle moves it by 0.25, 3.95 and 2.31 dB, so what is left
  at 80 degrees is the fan and not the method. A real seabed absorbs those
  bounces and the default is then ample; a perfect reflector needs the
  fan opened and `range_step` cut with it, since a step has to resolve
  $\tan\theta_\mathrm{max}$ depth units of climb per unit range. The
  warning below says when that pairing is wrong.
* **The beam must be small compared to the channel**, which the default
  `beam_width` enforces and an explicit one is checked against.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Source frequency, in Hz. |
| `depths` | Depth samples of the profile, in metres, from `z = 0`. |
| `sound_speeds` | Sound speed at each depth, in m/s. |
| `source_depth` | Source depth, in metres, inside the water column. |
| `max_range` | Maximum range to march to, in metres. |
| `ranges_m` | Ranges at which to evaluate the field, in metres. Default (`None`): the marching grid itself, which puts every receiver on a column the rays were actually sampled at. |
| `receiver_depths_m` | Depths at which to evaluate the field, in metres. Default (`None`): `n_depth_points` points spread over the water column, on the interior grid [`parabolic_equation`](/phonometry/reference/api/underwater/numerical/#parabolic_equation) uses, so the two fields land on the same depths. |
| `n_depth_points` | Size of that default depth grid. |
| `max_angle_deg` | Half-angle of the launch fan, in degrees from the horizontal. Beams are spread symmetrically over `[-max_angle_deg, +max_angle_deg]`. |
| `n_beams` | Number of beams in the fan. Default (`None`): from the overlap condition. Adjacent beams are $s\,\delta\theta_0$ apart at arc length $s$ while each has spread to $W \to s\lambda/(\pi W_0)$, so the condition that they still overlap, $\delta\theta_0 \lesssim \lambda/(\pi W_0)$, is range-independent; the default takes four times that margin. Too coarse a fan shows as a periodic ripple in range at the beam spacing, which is easy to mistake for physical interference. |
| `beam_width` | The $W_0$ of Eq. (3.91), in metres: the beam's initial half-width, at the $e^{-2}$ folding distance in intensity. Default (`None`): see `_default_beam_width`. |
| `range_step` | Marching step in range, in metres, and the spacing of the default `ranges_m`. |
| `bottom` | `"pressure-release"` (default) or `"rigid"`. The sea surface is always pressure-release. |

**Returns:** A [`GaussianBeamResult`](/phonometry/reference/api/underwater/numerical/#gaussianbeamresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

.. warning::

   A [`PhonometryWarning`](/phonometry/reference/api/filters/phonometry/#phonometrywarning) is emitted when the source sits
   on a kink of the profile (Sect. 3.7.4's spurious horizontal jet), when an
   explicit `beam_width` exceeds a quarter of the water depth, and when
   one marching step carries the steepest beam of the fan across more than a
   quarter of the water column, which is the pairing between
   `max_angle_deg` and `range_step` that is easiest to get wrong.

## GaussianBeamResult

```python
GaussianBeamResult(
    frequency: float,
    ranges: NDArray[np.float64],
    depths: NDArray[np.float64],
    propagation_loss: NDArray[np.float64],
    pressure: NDArray[np.complex128],
    launch_angles: NDArray[np.float64],
    ray_ranges: NDArray[np.float64],
    ray_depths: NDArray[np.float64],
    beam_widths: NDArray[np.float64],
    wavefront_curvatures: NDArray[np.float64],
    initial_beam_width: float,
    source_depth: float,
    water_depth: float,
)
```

Gaussian beam solution of a range-independent waveguide.

The propagation-loss field is on the same footing as
[`ParabolicEquationResult`](/phonometry/reference/api/underwater/numerical/#parabolicequationresult)'s: same shape, same reference, so the two
can be subtracted.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Source frequency, in Hz. |
| `ranges` | Range grid of the field, in metres. |
| `depths` | Depth grid of the field, in metres. |
| `propagation_loss` | Propagation-loss field `PL(z, r)`, in dB, shape `(n_depths, n_ranges)`. Infinite where the field is exactly zero, which happens at the source range itself and in the wedge no beam of the fan reaches: each beam is summed out to four half-widths, 140 dB below its own axis, so a point that far from every one of them is outside the traced aperture rather than merely in shadow. The graded penumbra just past a limiting ray, which is the part of a shadow zone worth having, is finite and carries the beams' tails. |
| `pressure` | The complex field the loss was taken from, same shape, in the module's own $e^{-i\omega t}$ convention (the conjugate of the one Jensen Eq. (3.88) is printed in) and normalised to unit pressure at 1 m, so `propagation_loss = -20 lg\|pressure\|`. |
| `launch_angles` | Launch angle of each beam's central ray, from the horizontal, in degrees. |
| `ray_ranges` | Range of each central ray at each marching step, in metres, shape `(n_beams, n_steps)`. This is the marching grid, which is finer than (and independent of) `ranges`. |
| `ray_depths` | Depth of each central ray on that grid, in metres. |
| `beam_widths` | Beam half-width $W(s)$ on that grid, in metres: Jensen Eq. (3.89), the distance at which the beam's own pressure has fallen by $e^{-1}$ and its intensity by $e^{-2}$. |
| `wavefront_curvatures` | Beam wavefront curvature $K(s)$ on that grid, in 1/m: Jensen Eq. (3.90) with the sign that belongs to the conjugated field this result exposes, so that a beam spreading in free space reproduces Eq. (3.85), $K = x/(x^2 + a^2)$, as a positive number. |
| `initial_beam_width` | The $W_0$ of Eq. (3.91) actually used, in metres, whether it was passed or defaulted. |
| `source_depth` | Source depth, in metres. |
| `water_depth` | Water-column depth, in metres. |

### GaussianBeamResult.plot()

```python
GaussianBeamResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the propagation-loss field (depth increasing downward).

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

Normal-mode propagation loss for a range-independent waveguide.

Solves the depth-separated Sturm-Liouville problem (Jensen Eq. 5.3) on a
uniform finite-difference grid, then assembles the coherent propagation
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
| `receiver_depth` | Receiver depth for the propagation-loss slice, in m. |
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
    propagation_loss: NDArray[np.float64],
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
| `ranges` | Ranges at which the propagation loss is evaluated, in metres. |
| `propagation_loss` | Coherent propagation loss at `receiver_depth` per range, in dB. |
| `receiver_depth` | Receiver depth of the propagation-loss slice, in m. |
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

Plot the propagation loss versus range (loss increasing downward).

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

Propagation-loss field from the standard (Tappert) parabolic
equation.

Marches the split-step Fourier solution (Jensen Ch. 6) in range with a
discrete sine transform in depth, enforcing a pressure-release surface at
`z = 0` and bottom at `z = water_depth`. The envelope is related to
pressure by $p = \psi \, e^{i(k_0 r - \pi/4)} / \sqrt{r}$ and
$\mathrm{PL} = -20 \log_{10}(\lvert \psi \rvert / \sqrt{r})$
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
    propagation_loss: NDArray[np.float64],
    source_depth: float,
)
```

Parabolic-equation propagation-loss field.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Source frequency, in Hz. |
| `ranges` | Range grid, in metres. |
| `depths` | Depth grid, in metres. |
| `propagation_loss` | Propagation-loss field `PL(z, r)`, in dB, shape `(n_depths, n_ranges)`. |
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

Plot the propagation-loss field (depth increasing downward).

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
parameter $\xi = \cos\theta_0 / c(z_\mathrm{s})$ it obeys
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
