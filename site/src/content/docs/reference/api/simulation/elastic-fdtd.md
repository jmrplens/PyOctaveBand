---
title: "simulation.elastic_fdtd"
description: "2D elastic finite-difference time-domain (P-SV) simulation."
sidebar:
  label: "elastic_fdtd"
---

2D elastic finite-difference time-domain (P-SV) simulation.

A staggered-grid velocity-stress leapfrog solver for the 2D elastodynamic
equations in an isotropic linear medium, following the reference formulation
of Virieux, "P-SV wave propagation in heterogeneous media: velocity-stress
finite-difference method", *Geophysics* 51(4), 889-901 (1986):

* the governing first-order velocity-stress system in `(vx, vy)` and
  `(txx, tyy, txy)` (Eq. 2), which handles heterogeneous media without any
  explicit interface treatment;
* the fully staggered discretisation of Fig. 1 / Eq. 5, second order in
  space and time, arranged here so that the normal stresses share the cell
  centres of the acoustic solver ([`FDTD2D`](/phonometry/reference/api/simulation/fdtd/#fdtd2d))
  and the velocities live on the cell faces;
* the Courant stability condition `c_P dt sqrt(1/dx^2 + 1/dy^2) < 1`
  (Eqs. 6-7), which depends only on the P-wave speed, and the numerical
  dispersion relations of Eqs. 13-14 with the 10-cells-per-wavelength rule;
* liquids as the `c_s = 0` limit: shear-free cells propagate the acoustic
  wave equation and a fluid-solid contact needs no special treatment.

Heterogeneous media use the effective grid parameters of Moczo, Kristek,
Galis, Pazak & Balazovjech, "The finite-difference and finite-element
modeling of seismic wave propagation and earthquake motion", *Acta Physica
Slovaca* 57(2), 177-406 (2007): density arithmetically averaged onto the
faces and the shear modulus harmonically averaged onto the corners
(Eqs. 7.37-7.39), which is what makes internal interfaces (including
fluid-solid ones) converge to the physical traction continuity. Free
surfaces use the stress-imaging condition (Levander 1988; Moczo et al.
Eq. 9.9): zero normal stress on the surface plane and an antisymmetric
image of the shear stress above it.

Two API levels are exposed, mirroring the acoustic module.
[`elastic_fdtd_simulation`](/phonometry/reference/api/simulation/elastic-fdtd/#elastic_fdtd_simulation) builds the grid, runs a deterministic
simulation and returns a frozen [`ElasticFDTDResult`](/phonometry/reference/api/simulation/elastic-fdtd/#elasticfdtdresult) with per-probe
signal histories, optional field snapshots and a `.plot()` method.
[`ElasticFDTD2D`](/phonometry/reference/api/simulation/elastic-fdtd/#elasticfdtd2d) is the underlying stepping engine for callers that
need frame-by-frame access to the five field arrays.

The solver is deliberately deterministic: float64 arithmetic throughout, no
random numbers and single-threaded numpy execution, so identical inputs give
bit-identical outputs on the same platform.

Validated against analytic oracles: P- and S-wave times of flight
`c_P = sqrt((lambda + 2 mu) / rho)` and `c_S = sqrt(mu / rho)`, the
Rayleigh-wave speed from the exact characteristic equation (Cremer, Heckl &
Petersson 2005, Eq. 3.149), the Kirchhoff thin-plate flexural dispersion
`c_B = (B'/m'')^(1/4) sqrt(omega)` in its `lambda_B >> h` domain
(Eqs. 3.83-3.89), the normal-incidence fluid-solid reflection coefficient
`(Z2 - Z1)/(Z2 + Z1)`, the normal-incidence mass law of a thin immersed
panel, and the exact reduction to the acoustic solver when `c_s = 0`
everywhere. The fluid-solid coupling is further pinned to the oblique
plane-wave reflection coefficient of Brekhovskikh & Godin, *Acoustics of
Layered Media I* (Springer 1990), Eqs. 4.2.22-4.2.26 (with the shear-wave
mode conversion active), to the Scholte interface-wave speed from the exact
characteristic equation of Eq. 4.4.20 (see [`scholte_speed`](/phonometry/reference/api/simulation/elastic-fdtd/#scholte_speed)), and to
the exact three-media transmission of an immersed plate (B&G Eqs. 2.4.10,
2.4.14) including its first thickness resonance `f_1 = c_P / (2 h)`
(Eq. 2.4.19), following the fluid-solid finite-difference benchmark of
van Vossen, Robertsson & Chapman, *Geophysics* 67(2), 618-624 (2002).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AIR

*Constant* (`phonometry.simulation.elastic_fdtd.Material`).

## ALUMINIUM

*Constant* (`phonometry.simulation.elastic_fdtd.Material`).

## CONCRETE

*Constant* (`phonometry.simulation.elastic_fdtd.Material`).

## elastic_fdtd_simulation

```python
elastic_fdtd_simulation(
    c_p: float | Field2D,
    c_s: float | Field2D,
    dx: float,
    duration: float,
    *,
    sources: Sequence[ElasticSource],
    rho: float | Field2D,
    shape: tuple[int, int] | None = None,
    cfl: float = 0.6,
    probes: Sequence[tuple[int, int]] = (),
    probe_fields: Sequence[str] = ('vy',),
    boundaries: str | Mapping[str, str] = 'rigid',
    absorbing_layer_cells: int = 20,
    obstacle_mask: NDArray[np.bool_] | None = None,
    damping: float = 0.0,
    snapshot_every: int | None = None,
    snapshot_field: str = 'p',
) -> ElasticFDTDResult
```

Run a deterministic 2D elastic (P-SV) FDTD simulation.

Builds the staggered velocity-stress grid (Virieux 1986, Eq. 5), applies
the requested boundary conditions, injects the sources and integrates
for `duration` seconds, recording the selected fields at every probe
each time step and, optionally, full-field snapshots.

The grid covers `(nx * dx, ny * dx)` metres; a cell index `(ix, iy)`
maps to the physical cell centre `((ix + 0.5) * dx, (iy + 0.5) * dx)`.
Resolve at least 10 cells per shortest wavelength
(`dx <= c_s_min / (10 f)` with `c_s_min` the smallest non-zero
`c_s` over the solid cells, since the S wave is always the shortest;
a wholly fluid map falls back to the acoustic rule on the smallest
`c_p`; Virieux's rule from the dispersion relations
Eqs. 13-14), and 15-20 cells per wavelength when a Rayleigh wave along a
free surface matters, the second-order stress-imaging surface being the
most dispersive part of the scheme. The simulation is 2D (plane strain):
a point source is physically a line source with cylindrical
`1/sqrt(r)` amplitude spreading.

**Parameters**

| Name | Description |
| :--- | :--- |
| `c_p` | P-wave speed map [m/s], shape `(ny, nx)`, or a scalar with an explicit `shape`. |
| `c_s` | S-wave speed map [m/s]; `c_s = 0` marks fluid cells. |
| `dx` | Grid spacing [m] (square cells). |
| `duration` | Physical time to simulate [s]. |
| `sources` | One or more of [`ExplosionSource`](/phonometry/reference/api/simulation/elastic-fdtd/#explosionsource) or [`ForceSource`](/phonometry/reference/api/simulation/elastic-fdtd/#forcesource). |
| `rho` | Density map [kg/m3]; scalar or `(ny, nx)` array. |
| `shape` | Grid shape `(ny, nx)`, required when `c_p` is scalar. |
| `cfl` | Courant number in `(0, 1)` (Virieux Eqs. 6-7); the time step is `dt = cfl * dx / (c_p_max * sqrt(2))`. Default 0.6. |
| `probes` | Probe cells as `(ix, iy)` index pairs. |
| `probe_fields` | Which fields each probe records, drawn from `("p", "vx", "vy")` (default `("vy",)`, the component a surface accelerometer would see). |
| `boundaries` | `"rigid"` (default), `"absorbing"`, `"free"`, or a mapping from side name (`left`/`right`/`top`/`bottom`) to one of those. |
| `absorbing_layer_cells` | Sponge-layer thickness for absorbing sides, in cells. |
| `obstacle_mask` | Boolean map, shape `(ny, nx)`, of rigid cells (rasterised interior geometry). |
| `damping` | Uniform bulk amplitude decay rate [1/s]. |
| `snapshot_every` | Record a full field snapshot every this many steps (and at `t = 0`); `None` records none. |
| `snapshot_field` | Field recorded in the snapshots (`"p"`/`"vx"`/`"vy"`, interpolated to cell centres). |

**Returns:** An [`ElasticFDTDResult`](/phonometry/reference/api/simulation/elastic-fdtd/#elasticfdtdresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## ElasticFDTD2D

```python
ElasticFDTD2D(
    c_p: float | Field2D,
    c_s: float | Field2D,
    dx: float,
    *,
    rho: float | Field2D,
    cfl: float = 0.6,
    sponge_width: int = 0,
    sponge_sides: str | Iterable[str] | None = None,
    sponge_reflection: float = 0.0001,
    damping: float | NDArray[np.float64] = 0.0,
    shape: tuple[int, int] | None = None,
    free_sides: str | Iterable[str] | None = None,
    obstacle_mask: NDArray[np.bool_] | None = None,
)
```

2D elastic (P-SV) FDTD stepping engine on a staggered grid.

The Virieux (1986) cell is shifted half a cell so it lands on the layout
of the acoustic [`FDTD2D`](/phonometry/reference/api/simulation/fdtd/#fdtd2d): the normal
stresses `txx` and `tyy` share the cell centres, shape `(ny, nx)`
(row = y, column = x, the `imshow` convention; Virieux's downward z is
the growing row here); `vx` lives at interior x-faces, shape
`(ny, nx - 1)`; `vy` at interior y-faces, shape `(ny - 1, nx)`;
the shear stress `txy` at interior corners, shape `(ny - 1, nx - 1)`.
Because only interior faces are stored, the domain boundary has zero
normal velocity by construction and, with the corner shear stress taken
as zero on the edge, is a shear-free rigid wall; `free_sides` turns
selected sides into traction-free surfaces via stress imaging and sponge
layers into absorbing ones. Material maps are given as the measurable
wave speeds and converted internally to the Lame parameters
`mu = rho c_s**2` and `lambda = rho (c_p**2 - 2 c_s**2)`; density is
arithmetically averaged onto the faces and `mu` harmonically averaged
onto the corners (zero whenever any neighbour is a fluid), the Moczo
et al. (2007) effective parameters (Eqs. 7.37-7.39) that make internal
interfaces converge to the physical traction continuity.

**Parameters**

| Name | Description |
| :--- | :--- |
| `c_p` | P-wave speed map [m/s], shape `(ny, nx)`. A scalar with an explicit `shape` is also accepted. |
| `c_s` | S-wave speed map [m/s]; scalar or `(ny, nx)` array. `c_s = 0` marks a fluid cell (the acoustic limit); every cell must satisfy `c_p**2 >= 2 c_s**2` (non-negative `lambda`). |
| `dx` | Grid spacing [m] (square cells). |
| `rho` | Density map [kg/m3]; scalar or `(ny, nx)` array. |
| `cfl` | Courant number `CN = c_p_max dt sqrt(2) / dx`; the scheme is stable for `CN < 1` (Virieux Eqs. 6-7, a bound on `c_P` alone, independent of `c_S` and of the Poisson ratio) and values in `(0, 1)` are accepted. The default 0.6 keeps a wide stability margin with moderate numerical dispersion. |
| `sponge_width` | Thickness of the absorbing layer in cells (0 = no absorbing sides). |
| `sponge_sides` | Which sides absorb: a single side name or an iterable drawn from `{"left", "right", "top", "bottom"}` (default: all four when `sponge_width > 0`). |
| `sponge_reflection` | Target round-trip amplitude reflection of the sponge layer; sets the peak absorption rate. |
| `damping` | Bulk amplitude decay rate [1/s]: a scalar for the whole field or an `(ny, nx)` map for locally lossy regions. |
| `shape` | Grid shape `(ny, nx)`, required only when `c_p` is a scalar. |
| `free_sides` | Sides carrying a traction-free surface (stress imaging, Moczo et al. Eq. 9.9): the surface plane runs through the boundary row/column of cell centres, the normal stress is pinned to zero there and the shear stress is imaged antisymmetrically above it. A side cannot be both free and absorbing. |
| `obstacle_mask` | Boolean map, shape `(ny, nx)`, of rigid cells: every face adjacent to a masked cell is closed (zero normal velocity) and every touching corner shear stress is zeroed, i.e. a shear-free rigid inclusion, rasterising arbitrary interior geometry. |

### ElasticFDTD2D.add_source()

```python
ElasticFDTD2D.add_source(source: ElasticSource) -> None
```

Register an [`ExplosionSource`](/phonometry/reference/api/simulation/elastic-fdtd/#explosionsource) or a [`ForceSource`](/phonometry/reference/api/simulation/elastic-fdtd/#forcesource).

Explosions inject additively at one cell centre; forces at one
staggered velocity node. Positions are validated against the grid
and the obstacle mask.

### ElasticFDTD2D.collocated()

```python
ElasticFDTD2D.collocated(field: str) -> Field2D
```

One field interpolated to the cell centres, shape `(ny, nx)`.

`"p"` is the synthetic pressure; `"vx"`/`"vy"` average the two
adjacent faces (domain-edge faces count as zero, their built-in
value), so probes and snapshots of every field share the cell-centre
positions.

### ElasticFDTD2D.energy()

```python
ElasticFDTD2D.energy() -> float
```

Total elastic field energy [J per metre of depth].

Kinetic energy plus the plane-strain elastic energy expressed in
stresses by inverting the 2D stiffness (determinant
`4 mu (lambda + mu)`); fluid cells (`mu = 0`) degenerate to the
acoustic `p**2 / (2 lambda)` and shear-free corners contribute
nothing.

### ElasticFDTD2D.from_regions()

*classmethod*

```python
ElasticFDTD2D.from_regions(
    shape: tuple[int, int],
    dx: float,
    *,
    background: Material | tuple[float, float, float],
    regions: Iterable[tuple[Any, Material | tuple[float, float, float]]] = (),
    **kwargs: Any,
) -> ElasticFDTD2D
```

Build the engine from named materials painted over a background.

Thin sugar over the map constructor for the common layered set-ups
(fluid over a solid half-space, an immersed plate, an inclusion):
the `(ny, nx)` maps of `c_p`, `c_s` and `rho` start uniform
at `background` and each `(where, material)` entry of
`regions` is painted over them in order (later entries overwrite
earlier ones), then everything is delegated to the normal
constructor. No new physics: a fluid-solid contact still needs no
explicit interface treatment, because the constructor's effective
parameters (Moczo et al. 2007, Eqs. 7.37-7.39) handle it from the
maps alone.

`where` selects the painted cells: a boolean `(ny, nx)` mask,
or any basic numpy index expression over the `(row, column)`
maps, e.g. `(slice(120, None), slice(None))` for the lower half
or `numpy.s_[120:, :]` for the same thing spelled as a slice.

**Parameters**

| Name | Description |
| :--- | :--- |
| `shape` | Grid shape `(ny, nx)`. |
| `dx` | Grid spacing [m] (square cells). |
| `background` | The material filling the whole grid first: a [`Material`](/phonometry/reference/api/simulation/elastic-fdtd/#material) or a `(c_p, c_s, rho)` triple, e.g. [`WATER`](/phonometry/reference/api/simulation/elastic-fdtd/#water). |
| `regions` | `(where, material)` pairs painted in order. |
| `kwargs` | Forwarded to [`ElasticFDTD2D`](/phonometry/reference/api/simulation/elastic-fdtd/#elasticfdtd2d) (`cfl`, `sponge_width`, `sponge_sides`, `sponge_reflection`, `damping`, `free_sides`, `obstacle_mask`). |

**Returns:** The configured stepping engine.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a mask does not match `shape` or a material spec is invalid. |

### ElasticFDTD2D.p

*property*

Synthetic pressure `-(txx + tyy) / 2` at cell centres [Pa].

The mean compressive normal stress: in fluid (`c_s = 0`) regions
it is exactly the acoustic pressure, and in solids it is the
convenient scalar for probes and snapshots.

### ElasticFDTD2D.run()

```python
ElasticFDTD2D.run(
    steps: int,
    record_every: int | None = None,
    decimate: int = 1,
) -> NDArray[np.float64]
```

Advance `steps` steps, optionally recording pressure frames.

With `record_every = k` a snapshot of the synthetic pressure `p`
is stored after every `k`-th step (and one of the initial state),
spatially subsampled by `decimate`; the stacked
`(n_frames, ny', nx')` array plugs straight into a
`FuncAnimation` `imshow` update. Without `record_every` an
empty array is returned and only the final state is kept.

### ElasticFDTD2D.step()

```python
ElasticFDTD2D.step() -> None
```

Advance the leapfrog scheme by one time step (Virieux Eq. 5).

### ElasticFDTD2D.time

*property*

Elapsed simulated time [s].

## ElasticFDTDResult

```python
ElasticFDTDResult(
    times: NDArray[np.float64],
    signals: NDArray[np.float64],
    probe_fields: tuple[str, ...],
    probes: NDArray[np.int_],
    probe_positions: NDArray[np.float64],
    dx: float,
    dt: float,
    shape: tuple[int, int],
    sources: tuple[ElasticSource, ...],
    snapshots: NDArray[np.float64] | None,
    snapshot_times: NDArray[np.float64] | None,
    snapshot_field: str,
    obstacle_mask: NDArray[np.bool_] | None,
    free_sides: tuple[str, ...],
)
```

Frozen result of an [`elastic_fdtd_simulation`](/phonometry/reference/api/simulation/elastic-fdtd/#elastic_fdtd_simulation) run.

**Attributes**

| Name | Description |
| :--- | :--- |
| `times` | Time axis [s], length `n_steps + 1` (includes `t = 0`). |
| `signals` | Recorded histories, shape `(n_probes, n_fields, n_steps + 1)`, one row per probe and one layer per entry of `probe_fields` (velocities are interpolated to the probe cell centre). |
| `probe_fields` | The recorded fields, drawn from `("p", "vx", "vy")`. |
| `probes` | Probe cell indices `(ix, iy)`, shape `(n_probes, 2)`. |
| `probe_positions` | Probe cell-centre positions `(x, y)` [m], shape `(n_probes, 2)`. |
| `dx` | Grid spacing [m]. |
| `dt` | Time step [s]. |
| `shape` | Grid shape `(ny, nx)`. |
| `sources` | The source definitions of the run. |
| `snapshots` | Recorded cell-centre fields of `snapshot_field`, shape `(n_frames, ny, nx)`, or `None` when no snapshots were requested. |
| `snapshot_times` | Time of each snapshot [s], or `None`. |
| `snapshot_field` | The field recorded in `snapshots`. |
| `obstacle_mask` | Boolean map of rigid cells, or `None`. |
| `free_sides` | Sides carrying a traction-free surface. |

### ElasticFDTDResult.plot()

```python
ElasticFDTDResult.plot(
    ax: Axes | None = None,
    *,
    kind: str = 'probes',
    frame: int = -1,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the probe histories or one recorded field snapshot.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `kind` | `"probes"` (default) draws the per-probe signal time histories; `"snapshot"` renders one recorded field with the geometry overlaid (`imshow` raster). |
| `frame` | Snapshot index for `kind="snapshot"` (default: the last recorded frame). |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the underlying `plot`/`imshow`. |

**Returns:** The axes.

### ElasticFDTDResult.size

*property*

Domain size `(lx, ly)` [m].

## ExplosionSource

```python
ExplosionSource(
    ix: int,
    iy: int,
    waveform: Callable[[float], float],
    amplitude: float = 1.0,
)
```

An isotropic (explosive) stress injection at one cell centre.

Virieux (1986) drives an explosion with equal increments on both normal
stresses at their shared node, which avoids the infinite amplitudes of a
velocity-node source. The sign convention here is pressure-like: the
waveform value is the injected compression, added as `-s(t)` to both
`txx` and `tyy` so the synthetic pressure `p = -(txx + tyy) / 2`
increases with a positive waveform (and a `c_s = 0` run reproduces the
acoustic solver driven by the same waveform, sample by sample).

`waveform` maps time in seconds to the injected pressure in pascals:
any callable, typically the `value` method of a
[`GaussianPulse`](/phonometry/reference/api/simulation/fdtd/#gaussianpulse),
[`CWSource`](/phonometry/reference/api/simulation/fdtd/#cwsource) or
[`SignalSource`](/phonometry/reference/api/simulation/fdtd/#signalsource) (their own `ix`/`iy`
are not used here).

**Attributes**

| Name | Description |
| :--- | :--- |
| `ix` | Source column (x) index; the cell centre is at `x = (ix + 0.5) * dx`. |
| `iy` | Source row (y) index. |
| `waveform` | Callable `t -> s(t)` in pascals. |
| `amplitude` | Extra gain applied to `waveform`. |

## ForceSource

```python
ForceSource(
    ix: int,
    iy: int,
    direction: str,
    waveform: Callable[[float], float],
    amplitude: float = 1.0,
)
```

A directional body-force injection at one velocity node.

The standard body-force term of the equation of motion (the `f` of the
Moczo et al. 2007 formulation): each step the target velocity component
receives `dt * f(t) / (rho * dx**2)`, i.e. `waveform` is a line
force per unit depth [N/m] spread over one cell. A vertical force just
below a free surface reproduces Lamb's problem and launches the Rayleigh
wave (Virieux 1986, Fig. 5).

`ix`/`iy` index the staggered velocity array directly: for
`direction = "x"` the x-face between cells `ix` and `ix + 1`
(position `((ix + 1) * dx, (iy + 0.5) * dx)`), for
`direction = "y"` the y-face between rows `iy` and `iy + 1`
(position `((ix + 0.5) * dx, (iy + 1) * dx)`).

**Attributes**

| Name | Description |
| :--- | :--- |
| `ix` | Source column index into the velocity array. |
| `iy` | Source row index into the velocity array. |
| `direction` | `"x"` (horizontal) or `"y"` (vertical, positive downward in the `imshow` axes). |
| `waveform` | Callable `t -> f(t)` in newtons per metre of depth. |
| `amplitude` | Extra gain applied to `waveform`. |

## Material

```python
Material(c_p: float, c_s: float, rho: float)
```

An isotropic elastic medium as measurable wave speeds and density.

The three numbers the solver's material maps are built from: the
compressional speed `c_p = sqrt((lambda + 2 mu) / rho)`, the shear
speed `c_s = sqrt(mu / rho)` and the density `rho`. `c_s = 0`
marks a fluid (the acoustic `mu = 0` limit of the elastic scheme,
Virieux 1986), so the same dataclass names both fluids and solids.
Every material must satisfy `c_p**2 >= 2 c_s**2` (non-negative first
Lame parameter), the constructor bound of [`ElasticFDTD2D`](/phonometry/reference/api/simulation/elastic-fdtd/#elasticfdtd2d).

The module constants [`AIR`](/phonometry/reference/api/simulation/elastic-fdtd/#air), [`WATER`](/phonometry/reference/api/simulation/elastic-fdtd/#water), [`STEEL`](/phonometry/reference/api/simulation/elastic-fdtd/#steel),
[`ALUMINIUM`](/phonometry/reference/api/simulation/elastic-fdtd/#aluminium) and [`CONCRETE`](/phonometry/reference/api/simulation/elastic-fdtd/#concrete) carry the nominal round-number
properties used throughout the documentation and the validation suite,
mirroring the documented default media of the acoustic solver.

**Attributes**

| Name | Description |
| :--- | :--- |
| `c_p` | Compressional (P) wave speed [m/s], strictly positive. |
| `c_s` | Shear (S) wave speed [m/s], non-negative; 0 marks a fluid. |
| `rho` | Density [kg/m3], strictly positive. |

### Material.is_fluid

*property*

`True` when the material carries no shear (`c_s = 0`).

## scholte_speed

```python
scholte_speed(
    fluid: Material | tuple[float, float, float],
    solid: Material | tuple[float, float, float],
) -> float
```

Exact Scholte-wave speed of a fluid over an elastic half-space [m/s].

The Scholte wave is the true interface wave of a fluid-solid contact:
evanescent on both sides, elliptical particle motion, no low-frequency
cut-off, and non-dispersive over homogeneous half-spaces (Jensen,
Kuperman, Porter & Schmidt, *Computational Ocean Acoustics* 2e,
Sections 4.5.2 and 8.5.4). Its speed `v` lies below both the fluid
sound speed and the solid shear speed, and solves the exact
characteristic equation of Brekhovskikh & Godin, *Acoustics of Layered
Media I* (1990), Eq. 4.4.20, written in the notation of Eq. 4.4.18
(`q = c_S**2/c_P**2`, `r = c_S**2/c**2`, `s = v**2/c_S**2`,
`m = rho_solid/rho_fluid`):

```text
4 sqrt(1-s) sqrt(1-qs) - (2-s)**2 = (s**2/m) sqrt((1-sq)/(1-sr))
```

A root always exists (B&G Section 4.4.3); the `m -> inf` limit of the
left side is the Rayleigh equation. For stiff beds the root hugs the
fluid speed (water over steel: 1479.6 m/s, 0.027 % below `c`) while
for soft sediments it drops well below it, which is why measured
seabed interface waves probe the sediment shear speed
(`v approx 0.85 c_S` rule, Jensen et al. Section 5.10.5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `fluid` | Fluid half-space: a [`Material`](/phonometry/reference/api/simulation/elastic-fdtd/#material) with `c_s = 0` (or a `(c_p, 0.0, rho)` triple). |
| `solid` | Elastic half-space: a [`Material`](/phonometry/reference/api/simulation/elastic-fdtd/#material) with `c_s > 0`. |

**Returns:** The Scholte-wave phase speed [m/s].

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `fluid` carries shear or `solid` does not. |

## STEEL

*Constant* (`phonometry.simulation.elastic_fdtd.Material`).

## WATER

*Constant* (`phonometry.simulation.elastic_fdtd.Material`).
