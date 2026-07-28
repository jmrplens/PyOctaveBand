---
title: "simulation.ntff"
description: "2D near-to-far-field (NTFF) transformation over a closed contour."
sidebar:
  label: "ntff"
---

2D near-to-far-field (NTFF) transformation over a closed contour.

Given the steady-state pressure and outward normal velocity phasors on a
closed contour that encloses a scatterer (or any source region), the
exterior field is fully determined by the Kirchhoff-Helmholtz boundary
integral. In two dimensions, with the `exp(+j omega t)` time convention
used throughout the library, the free-space Green function is

    `G(R) = -(j / 4) H0(2)(k R)`,

with `H0(2)` the Hankel function of the second kind (outgoing waves) and
`R` the source-observer distance, and the exterior representation reads

    `p(r) = oint_S [ p(r') dG/dn' + j omega rho v_n(r') G ] dl'`,

where `n'` is the outward normal of the contour `S` and the momentum
equation `dp/dn = -j omega rho v_n` eliminated the pressure gradient. The
normal derivative of the Green function brings in `H1(2)`:
`dG/dR = (j k / 4) H1(2)(k R)`. This is the same construction full-wave
solvers use to report far-field scattering patterns from near-field data
(e.g. the finite-element polar responses of Jimenez, Cox, Romero-Garcia
and Groby, *Metadiffusers: Deep-subwavelength sound diffusers*, Sci. Rep.
7, 5389, 2017); the classical background is Williams, *Fourier Acoustics*
(Academic Press, 1999), chapter 8.

[`far_field_from_contour`](/phonometry/reference/api/simulation/ntff/#far_field_from_contour) evaluates that integral for phasors captured
by `add_contour_probe` (or assembled by
hand into a [`ContourPhasors`](/phonometry/reference/api/simulation/ntff/#contourphasors)), either in the true far-field limit,
where `H0(2)(kR) -> sqrt(2 / (pi k R)) exp(-j (k R - pi / 4))` turns the
integral into an angular pattern `F(theta)` with
`p(r, theta) -> F(theta) exp(-j k r) / sqrt(r)`, or at a finite
observation radius with the exact Hankel kernels.

Because the integral representation is source-free inside `S` for any
field whose sources lie outside the contour, the incident wave of a
scattering run contributes (analytically) nothing to the exterior integral:
transforming total-field phasors already yields the scattered far field.
Subtracting a reference run without the scatterer
([`ContourPhasors.subtract`](/phonometry/reference/api/simulation/ntff/#contourphasorssubtract)) removes the residual of that cancellation
caused by grid dispersion; on the validation scenes the residual stays
below 0.01 dB, so the subtraction is optional in practice.

Validated against analytic oracles: the omnidirectional pattern and the
absolute level of a monopole line source reconstructed from an enclosing
contour, the null pattern of an antiphase source pair, the extinction of a
contour that does not enclose the source, and the far-field polar response
of meshed Schroeder-type diffusers against the library's Fraunhofer model
(`tests/simulation/test_fdtd_ntff.py`).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ContourPhasors

```python
ContourPhasors(
    frequency: float,
    positions: NDArray[np.float64],
    normals: NDArray[np.float64],
    pressure: NDArray[np.complex128],
    normal_velocity: NDArray[np.complex128],
    segment: float,
)
```

Steady-state `p` and `v_n` phasors sampled on a closed contour.

The phasors follow the library's `exp(+j omega t)` convention:
`p(t) = Re{ pressure * exp(+j omega t) }`. `normals` point outward
(away from the enclosed region) and `normal_velocity` is the particle
velocity component along them. Instances are produced by
`add_contour_probe` probes; they can
also be assembled by hand from any solver's near field.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency of the phasors [Hz]. |
| `positions` | Sample positions `(x, y)` [m], shape `(n, 2)`. |
| `normals` | Outward unit normals, shape `(n, 2)`. |
| `pressure` | Complex pressure phasor at each sample [Pa]. |
| `normal_velocity` | Complex outward normal velocity phasor [m/s]. |
| `segment` | Contour length carried by each sample `dl` [m]. |

### ContourPhasors.subtract()

```python
ContourPhasors.subtract(reference: ContourPhasors) -> ContourPhasors
```

Phasor difference on the same contour: `self - reference`.

The standard way to isolate the scattered field of a plane-wave
scattering run: capture the same contour in a second, otherwise
identical simulation without the scatterer and subtract its
(incident-only) phasors. Geometry and frequency must match.

## far_field_from_contour

```python
far_field_from_contour(
    contour: ContourPhasors,
    angles: ArrayLike,
    *,
    distance: float | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    speed_of_sound: float = 343.0,
    air_density: float = 1.2,
) -> NDArray[np.complex128]
```

Exterior field of a closed contour: the 2D Kirchhoff-Helmholtz integral.

Evaluates, for each observation angle `a` (degrees, measured from the
`+x` axis towards `+y` of the grid coordinates, so the unit
direction is `u = (cos a, sin a)`),

    `p(r) = oint_S [ p dG/dn' + j omega rho v_n G ] dl'`

with the outgoing 2D free-space Green function
`G = -(j/4) H0(2)(k R)` and its normal derivative through
`dG/dR = (j k / 4) H1(2)(k R)` (`exp(+j omega t)` convention,
`scipy.special.hankel2`).

With `distance=None` (the default) the far-field limit
`H0(2)(kR) -> sqrt(2 / (pi k R)) exp(-j (k R - pi/4))` is taken
analytically and the returned complex pattern `F(a)` is the relative
far-field amplitude defined by

    `p(r, a) -> F(a) exp(-j k r) / sqrt(r)`  as `r -> oo`,

with `r` measured from `origin` (the phase reference; magnitudes at
infinity do not depend on it). With a finite `distance` the exact
Hankel kernels are used instead and the return value is the complex
pressure [Pa] on the circle of that radius around `origin` (which must
lie outside the contour samples).

The contour must enclose every source of the field being transformed;
for scattering runs, subtract a no-scatterer reference first
([`ContourPhasors.subtract`](/phonometry/reference/api/simulation/ntff/#contourphasorssubtract)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `contour` | The contour phasors (from `add_contour_probe`, or hand built). |
| `angles` | Observation angles [degrees], 1D. `0` is `+x`, `90` is `+y` of the coordinates `contour.positions` live in. |
| `distance` | Observation radius [m] for the exact evaluation, or `None` (default) for the far-field pattern. |
| `origin` | Phase-reference point `(x, y)` [m]; the centre of the observation circle when `distance` is given. Default the grid origin. |
| `speed_of_sound` | Speed of sound `c` [m/s] (default 343). |
| `air_density` | Density `rho` [kg/m3] (default 1.2). |

**Returns:** Complex array, one value per angle: the far-field pattern `F(a)` (units Pa sqrt(m)), or the pressure [Pa] at `distance`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | For invalid angles, a non-positive `distance`, or an observation circle that does not clear the contour. |
