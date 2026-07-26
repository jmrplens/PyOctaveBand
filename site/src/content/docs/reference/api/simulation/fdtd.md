---
title: "simulation.fdtd"
description: "2D acoustic finite-difference time-domain (FDTD) simulation."
sidebar:
  label: "fdtd"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

2D acoustic finite-difference time-domain (FDTD) simulation.

A staggered-grid (Yee-style) pressure-velocity leapfrog solver for the
linear acoustic equations in a non-moving medium, following the reference
formulation of Attenborough & Van Renterghem, *Predicting Outdoor Sound*
(2nd ed., CRC Press 2021), chapter 4:

* the governing first-order system in `p` and `v` (Eqs. 4.3-4.4);
* the staggered-in-place, staggered-in-time discretisation (Eqs. 4.11-4.12),
  with pressure at cell centres and velocity components on cell faces;
* the Courant stability condition `CN <= 1` with
  `CN = c dt sqrt(1/dx^2 + 1/dy^2)` (Eqs. 4.13-4.14);
* rigid boundaries as zero normal face velocity (Eq. 4.32) and the
  frequency-independent real-impedance boundary update (Eqs. 4.33-4.35);
* absorbing edges as a graded sponge layer, the simple precursor of the
  perfectly-matched-layer treatment discussed in section 4.2.3.

Two API levels are exposed. [`fdtd_simulation`](/phonometry/reference/api/simulation/fdtd/#fdtd_simulation) is the result-object
entry point: it builds the grid, runs a deterministic simulation and returns
a frozen [`FDTDResult`](/phonometry/reference/api/simulation/fdtd/#fdtdresult) with per-probe pressure histories, optional
field snapshots and a `.plot()` method. [`FDTD2D`](/phonometry/reference/api/simulation/fdtd/#fdtd2d) is the underlying
stepping engine (also used by the documentation animations) for callers that
need frame-by-frame access to the field.

The solver is deliberately deterministic: float64 arithmetic throughout, no
random numbers and single-threaded numpy execution, so identical inputs give
bit-identical outputs on the same platform.

Validated against analytic oracles: the eigenfrequencies of a rigid
rectangular box and of an effectively 1D tube, the numerical dispersion
relation of the leapfrog scheme (the discrete counterpart of Eq. 4.15),
free-field pulse arrival times and cylindrical `1/sqrt(r)` amplitude
decay, the image-source echo of a rigid wall, the normal-incidence
reflection coefficient `(Z - rho c)/(Z + rho c)` of an impedance edge,
and second-order convergence under grid refinement.

## CWSource

```python
CWSource(
    ix: int,
    iy: int,
    frequency: float,
    amplitude: float = 1.0,
    ramp_cycles: float = 3.0,
)
```

A continuous sine drive with a smooth cosine-ramped onset.

The first `ramp_cycles` periods fade the amplitude in with a raised
cosine so the start does not splash a broadband transient over the field.

**Attributes**

| Name | Description |
| :--- | :--- |
| `ix` | Source column (x) index. |
| `iy` | Source row (y) index. |
| `frequency` | Drive frequency [Hz]. |
| `amplitude` | Steady-state source amplitude [Pa]. |
| `ramp_cycles` | Onset ramp length in periods of `frequency`. |

### CWSource.value()

```python
CWSource.value(t: float) -> float
```

Source waveform at time `t` (seconds).

## FDTD2D

```python
FDTD2D(
    c: float | Field2D,
    dx: float,
    *,
    rho: float | Field2D = 1.2,
    cfl: float = 0.6,
    sponge_width: int = 0,
    sponge_sides: str | Iterable[str] | None = None,
    sponge_reflection: float = 0.0001,
    damping: float = 0.0,
    shape: tuple[int, int] | None = None,
    edge_impedance: Mapping[str, float | NDArray[np.float64]] | None = None,
    obstacle_mask: NDArray[np.bool_] | None = None,
)
```

2D acoustic FDTD stepping engine on a staggered grid.

Pressure `p` lives at cell centres, shape `(ny, nx)` (row = y,
column = x, the `imshow` convention); `vx` at interior x-faces,
shape `(ny, nx - 1)`; `vy` at interior y-faces, shape
`(ny - 1, nx)`. Because only interior faces are stored, the domain
boundary is perfectly rigid (zero normal velocity, Eq. 4.32) by
construction; sponge layers and per-cell real impedances turn selected
sides into absorbing or locally reacting boundaries. Sources are soft
(additive) pressure injections.

**Parameters**

| Name | Description |
| :--- | :--- |
| `c` | Sound-speed map [m/s], shape `(ny, nx)`. A scalar with an explicit `shape` is also accepted. |
| `dx` | Grid spacing [m] (square cells). |
| `rho` | Density map [kg/m3]; scalar or `(ny, nx)` array (default 1.2). |
| `cfl` | Courant number `CN = c_max dt sqrt(2) / dx` (Eq. 4.13); the explicit scheme is stable for `CN <= 1` (Eq. 4.14) and values in `(0, 1)` are accepted. The default 0.6 keeps a wide stability margin with moderate numerical dispersion. |
| `sponge_width` | Thickness of the absorbing layer in cells (0 = no absorbing sides). |
| `sponge_sides` | Which sides absorb: a single side name or an iterable drawn from `{"left", "right", "top", "bottom"}` (default: all four when `sponge_width > 0`). `left`/`right` are the low/high column edges and `top`/`bottom` the low/high row edges (the default `imshow` origin). |
| `sponge_reflection` | Target round-trip amplitude reflection of the sponge layer; sets the peak absorption rate. |
| `damping` | Uniform bulk amplitude decay rate [1/s] applied to the whole field (a simple stand-in for air/wall absorption; `6.91 / T60` gives a `T60` seconds reverberant decay). |
| `shape` | Grid shape `(ny, nx)`, required only when `c` is a scalar. |
| `edge_impedance` | Locally reacting boundary sides: a mapping from side name to a real specific acoustic impedance [Pa s/m], either a scalar or a per-edge-cell 1D array (length `ny` for `left`/ `right`, `nx` for `top`/`bottom`). Implements Eqs. (4.33)-(4.35); `Z = rho c` is a normal-incidence matched (anechoic) edge. A side cannot be both a sponge and an impedance boundary. |
| `obstacle_mask` | Boolean map, shape `(ny, nx)`, of rigid cells: every face adjacent to a masked cell is closed (zero normal velocity, Eq. 4.32), rasterising arbitrary interior geometry. |

### FDTD2D.add_source()

```python
FDTD2D.add_source(source: Source) -> None
```

Register a soft pressure source (additive injection at one cell).

### FDTD2D.energy()

```python
FDTD2D.energy() -> float
```

Total acoustic field energy [J per metre of depth].

### FDTD2D.run()

```python
FDTD2D.run(
    steps: int,
    record_every: int | None = None,
    decimate: int = 1,
) -> NDArray[np.float64]
```

Advance `steps` steps, optionally recording pressure frames.

With `record_every = k` a snapshot of `p` is stored after every
`k`-th step (and one of the initial state), spatially subsampled by
`decimate`; the stacked `(n_frames, ny', nx')` array plugs
straight into a `FuncAnimation` `imshow` update. Without
`record_every` an empty array is returned and only the final state
is kept (read it from `self.p`).

### FDTD2D.step()

```python
FDTD2D.step() -> None
```

Advance the leapfrog scheme by one time step.

### FDTD2D.time

*property*

Elapsed simulated time [s].

## fdtd_simulation

```python
fdtd_simulation(
    c: float | Field2D,
    dx: float,
    duration: float,
    *,
    sources: Sequence[Source],
    shape: tuple[int, int] | None = None,
    rho: float | Field2D = 1.2,
    cfl: float = 0.6,
    probes: Sequence[tuple[int, int]] = (),
    boundaries: str | Mapping[str, str | float | NDArray[np.float64]] = 'rigid',
    absorbing_layer_cells: int = 20,
    obstacle_mask: NDArray[np.bool_] | None = None,
    damping: float = 0.0,
    snapshot_every: int | None = None,
) -> FDTDResult
```

Run a deterministic 2D acoustic FDTD simulation.

Builds the staggered-grid domain (Attenborough & Van Renterghem 2021,
Eqs. 4.11-4.12), applies the requested boundary conditions, injects the
sources and integrates for `duration` seconds, recording the pressure
at every probe each time step and, optionally, full-field snapshots.

The grid covers `(nx * dx, ny * dx)` metres; a cell index `(ix, iy)`
maps to the physical cell centre `((ix + 0.5) * dx, (iy + 0.5) * dx)`.
Resolve at least 10 cells per shortest wavelength using the smallest
sound speed of the domain (`dx <= c_min / (10 f)`), the usual rule for
this lowest-order scheme: the worst-case (on-axis) numerical dispersion
error magnitude, `(k dx)^2 / 24` from the discrete counterpart of
Eq. 4.15 (the modelled frequency under-reads, so the signed error is
negative), is then about 1.6 % (about 1.4 % at the default `cfl`;
in a heterogeneous domain the slower cells run at a lower local Courant
number and sit nearer the 1.6 % bound) and finer grids reduce it
quadratically. The
simulation is 2D, so a point source is
physically a line source with cylindrical `1/sqrt(r)` amplitude
spreading rather than the 3D spherical `1/r`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `c` | Sound-speed map [m/s], shape `(ny, nx)`, or a scalar with an explicit `shape`. |
| `dx` | Grid spacing [m] (square cells). |
| `duration` | Physical time to simulate [s]. |
| `sources` | One or more of [`GaussianPulse`](/phonometry/reference/api/simulation/fdtd/#gaussianpulse), [`CWSource`](/phonometry/reference/api/simulation/fdtd/#cwsource) or [`SignalSource`](/phonometry/reference/api/simulation/fdtd/#signalsource). |
| `shape` | Grid shape `(ny, nx)`, required when `c` is a scalar. |
| `rho` | Density map [kg/m3]; scalar or `(ny, nx)` array. |
| `cfl` | Courant number in `(0, 1)` (Eqs. 4.13-4.14); the time step is `dt = cfl * dx / (c_max * sqrt(2))`. Default 0.6. |
| `probes` | Pressure-probe cells as `(ix, iy)` index pairs. |
| `boundaries` | `"rigid"` (default), `"absorbing"`, or a mapping from side name (`left`/`right`/`top`/`bottom`) to `"rigid"`, `"absorbing"`, or a real specific impedance [Pa s/m] (scalar or per-edge-cell 1D array, Eqs. 4.33-4.35). |
| `absorbing_layer_cells` | Sponge-layer thickness for absorbing sides, in cells. |
| `obstacle_mask` | Boolean map, shape `(ny, nx)`, of rigid cells (rasterised interior geometry). |
| `damping` | Uniform bulk amplitude decay rate [1/s]. |
| `snapshot_every` | Record a full pressure-field snapshot every this many steps (and at `t = 0`); `None` records none. |

**Returns:** A [`FDTDResult`](/phonometry/reference/api/simulation/fdtd/#fdtdresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## FDTDResult

```python
FDTDResult(
    times: NDArray[np.float64],
    pressures: NDArray[np.float64],
    probes: NDArray[np.int_],
    probe_positions: NDArray[np.float64],
    dx: float,
    dt: float,
    shape: tuple[int, int],
    sources: tuple[Source, ...],
    snapshots: NDArray[np.float64] | None,
    snapshot_times: NDArray[np.float64] | None,
    obstacle_mask: NDArray[np.bool_] | None,
)
```

Frozen result of a [`fdtd_simulation`](/phonometry/reference/api/simulation/fdtd/#fdtd_simulation) run.

**Attributes**

| Name | Description |
| :--- | :--- |
| `times` | Time axis [s], length `n_steps + 1` (includes `t = 0`). |
| `pressures` | Pressure history at each probe [Pa], shape `(n_probes, n_steps + 1)`. |
| `probes` | Probe cell indices `(ix, iy)`, shape `(n_probes, 2)`. |
| `probe_positions` | Probe cell-centre positions `(x, y)` [m], shape `(n_probes, 2)`. |
| `dx` | Grid spacing [m]. |
| `dt` | Time step [s]. |
| `shape` | Grid shape `(ny, nx)`. |
| `sources` | The source definitions of the run. |
| `snapshots` | Recorded pressure fields, shape `(n_frames, ny, nx)`, or `None` when no snapshots were requested. |
| `snapshot_times` | Time of each snapshot [s], or `None`. |
| `obstacle_mask` | Boolean map of rigid cells, or `None`. |

### FDTDResult.plot()

```python
FDTDResult.plot(
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
| `kind` | `"probes"` (default) draws the per-probe pressure time histories; `"snapshot"` renders one recorded pressure field with the geometry overlaid (`imshow` raster). |
| `frame` | Snapshot index for `kind="snapshot"` (default: the last recorded frame). |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the underlying `plot`/`imshow`. |

**Returns:** The axes.

### FDTDResult.size

*property*

Domain size `(lx, ly)` [m].

## GaussianPulse

```python
GaussianPulse(
    ix: int,
    iy: int,
    width: float,
    t0: float | None = None,
    amplitude: float = 1.0,
)
```

A soft Gaussian pressure pulse injected at one cell.

`s(t) = amplitude * exp(-((t - t0) / width)**2)` with `t0` defaulting
to `4 * width` so the pulse starts from (numerically) zero.

**Attributes**

| Name | Description |
| :--- | :--- |
| `ix` | Source column (x) index; the cell centre is at `x = (ix + 0.5) * dx`. |
| `iy` | Source row (y) index. |
| `width` | Gaussian half-width [s]; sets the pulse bandwidth. |
| `t0` | Pulse centre time [s] (default `4 * width`). |
| `amplitude` | Peak source amplitude [Pa]. |

### GaussianPulse.value()

```python
GaussianPulse.value(t: float) -> float
```

Source waveform at time `t` (seconds).

## SignalSource

```python
SignalSource(
    ix: int,
    iy: int,
    samples: NDArray[np.float64],
    sample_rate: float,
    amplitude: float = 1.0,
)
```

An arbitrary sampled waveform injected at one cell.

The samples are interpreted as the source signal at `sample_rate` and
linearly interpolated onto the simulation time steps; outside the sampled
span the source is zero. `sample_rate` therefore does not need to match
the simulation rate `1/dt`, although a rate well above the highest
frequency of interest avoids interpolation roll-off.

**Attributes**

| Name | Description |
| :--- | :--- |
| `ix` | Source column (x) index. |
| `iy` | Source row (y) index. |
| `samples` | Source signal samples [Pa] (stored as a read-only 1D float64 array). |
| `sample_rate` | Sampling rate of `samples` [Hz]. |
| `amplitude` | Scale factor applied to the samples. |

### SignalSource.value()

```python
SignalSource.value(t: float) -> float
```

Source waveform at time `t` (seconds), zero outside the span.
