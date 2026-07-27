#  Copyright (c) 2026. Jose M. Requena-Plens
"""
2D acoustic finite-difference time-domain (FDTD) simulation.

A staggered-grid (Yee-style) pressure-velocity leapfrog solver for the
linear acoustic equations in a non-moving medium, following the reference
formulation of Attenborough & Van Renterghem, *Predicting Outdoor Sound*
(2nd ed., CRC Press 2021), chapter 4:

* the governing first-order system in ``p`` and ``v`` (Eqs. 4.3-4.4);
* the staggered-in-place, staggered-in-time discretisation (Eqs. 4.11-4.12),
  with pressure at cell centres and velocity components on cell faces;
* the Courant stability condition ``CN <= 1`` with
  ``CN = c dt sqrt(1/dx^2 + 1/dy^2)`` (Eqs. 4.13-4.14);
* rigid boundaries as zero normal face velocity (Eq. 4.32) and the
  frequency-independent real-impedance boundary update (Eqs. 4.33-4.35);
* absorbing edges as a graded sponge layer, the simple precursor of the
  perfectly-matched-layer treatment discussed in section 4.2.3.

Two API levels are exposed. :func:`fdtd_simulation` is the result-object
entry point: it builds the grid, runs a deterministic simulation and returns
a frozen :class:`FDTDResult` with per-probe pressure histories, optional
field snapshots and a ``.plot()`` method. :class:`FDTD2D` is the underlying
stepping engine (also used by the documentation animations) for callers that
need frame-by-frame access to the field.

The solver is deliberately deterministic: float64 arithmetic throughout, no
random numbers and single-threaded numpy execution, so identical inputs give
bit-identical outputs on the same platform.

Validated against analytic oracles: the eigenfrequencies of a rigid
rectangular box and of an effectively 1D tube, the numerical dispersion
relation of the leapfrog scheme (the discrete counterpart of Eq. 4.15),
free-field pulse arrival times and cylindrical ``1/sqrt(r)`` amplitude
decay, the image-source echo of a rigid wall, the normal-incidence
reflection coefficient ``(Z - rho c)/(Z + rho c)`` of an impedance edge,
and second-order convergence under grid refinement.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Field2D = NDArray[np.float64]

_SIDES = ("left", "right", "top", "bottom")

#: Boundary-condition names accepted by :func:`fdtd_simulation`.
_BOUNDARY_NAMES = ("rigid", "absorbing")


def _positive_finite(name: str, value: float) -> float:
    """Validate that *value* is a strictly positive finite scalar."""
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _finite(name: str, value: float) -> float:
    """Validate that *value* is a finite scalar."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _integer(name: str, value: int) -> int:
    """Validate that *value* is an integral scalar (bool is rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004 - ValueError keeps the module validation errors uniform
    return int(value)


def _positive_map(name: str, field: Field2D) -> None:
    """Validate that every cell of *field* is strictly positive and finite."""
    if not np.all(np.isfinite(field)) or bool(np.any(field <= 0.0)):
        raise ValueError(f"{name} must be strictly positive and finite "
                         "everywhere")


@dataclass(frozen=True)
class GaussianPulse:
    """A soft Gaussian pressure pulse injected at one cell.

    ``s(t) = amplitude * exp(-((t - t0) / width)**2)`` with ``t0`` defaulting
    to ``4 * width`` so the pulse starts from (numerically) zero.

    :ivar ix: Source column (x) index; the cell centre is at
        ``x = (ix + 0.5) * dx``.
    :ivar iy: Source row (y) index.
    :ivar width: Gaussian half-width [s]; sets the pulse bandwidth.
    :ivar t0: Pulse centre time [s] (default ``4 * width``).
    :ivar amplitude: Peak source amplitude [Pa].
    """

    ix: int
    iy: int
    width: float
    t0: float | None = None
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        _positive_finite("width", self.width)
        _finite("amplitude", self.amplitude)
        if self.t0 is not None:
            _finite("t0", self.t0)

    def value(self, t: float) -> float:
        """Source waveform at time ``t`` (seconds)."""
        t0 = 4.0 * self.width if self.t0 is None else self.t0
        return self.amplitude * float(np.exp(-(((t - t0) / self.width) ** 2)))


@dataclass(frozen=True)
class CWSource:
    """A continuous sine drive with a smooth cosine-ramped onset.

    The first ``ramp_cycles`` periods fade the amplitude in with a raised
    cosine so the start does not splash a broadband transient over the field.

    :ivar ix: Source column (x) index.
    :ivar iy: Source row (y) index.
    :ivar frequency: Drive frequency [Hz].
    :ivar amplitude: Steady-state source amplitude [Pa].
    :ivar ramp_cycles: Onset ramp length in periods of ``frequency``.
    """

    ix: int
    iy: int
    frequency: float
    amplitude: float = 1.0
    ramp_cycles: float = 3.0

    def __post_init__(self) -> None:
        _positive_finite("frequency", self.frequency)
        _finite("amplitude", self.amplitude)
        if not np.isfinite(self.ramp_cycles) or self.ramp_cycles < 0.0:
            raise ValueError("ramp_cycles must be non-negative and finite")

    def value(self, t: float) -> float:
        """Source waveform at time ``t`` (seconds)."""
        ramp_time = self.ramp_cycles / self.frequency
        if t < ramp_time and ramp_time > 0.0:
            envelope = 0.5 * (1.0 - float(np.cos(np.pi * t / ramp_time)))
        else:
            envelope = 1.0
        return (self.amplitude * envelope
                * float(np.sin(2.0 * np.pi * self.frequency * t)))


@dataclass(frozen=True)
class SignalSource:
    """An arbitrary sampled waveform injected at one cell.

    The samples are interpreted as the source signal at ``sample_rate`` and
    linearly interpolated onto the simulation time steps; outside the sampled
    span the source is zero. ``sample_rate`` therefore does not need to match
    the simulation rate ``1/dt``, although a rate well above the highest
    frequency of interest avoids interpolation roll-off.

    :ivar ix: Source column (x) index.
    :ivar iy: Source row (y) index.
    :ivar samples: Source signal samples [Pa] (stored as a read-only 1D
        float64 array).
    :ivar sample_rate: Sampling rate of ``samples`` [Hz].
    :ivar amplitude: Scale factor applied to the samples.
    """

    ix: int
    iy: int
    samples: NDArray[np.float64]
    sample_rate: float
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        _positive_finite("sample_rate", self.sample_rate)
        _finite("amplitude", self.amplitude)
        arr = np.array(self.samples, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("samples must be a 1D array")
        if arr.size == 0:
            raise ValueError("samples must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError("samples must be finite")
        arr.flags.writeable = False
        object.__setattr__(self, "samples", arr)

    def value(self, t: float) -> float:
        """Source waveform at time ``t`` (seconds), zero outside the span."""
        pos = t * self.sample_rate
        last = self.samples.size - 1
        if pos < 0.0 or pos > last:
            return 0.0
        i = int(pos)
        if i >= last:
            return self.amplitude * float(self.samples[last])
        w = pos - i
        return self.amplitude * float((1.0 - w) * self.samples[i]
                                      + w * self.samples[i + 1])


Source = GaussianPulse | CWSource | SignalSource


def _sponge_profile(n: int, width: int, sides: tuple[bool, bool],
                    sigma_max: float) -> Field2D:
    """1D absorption rate sigma(i) [1/s]: quadratic ramp into each sponge side.

    ``sides`` selects (low-index side, high-index side).
    """
    sigma = np.zeros(n, dtype=np.float64)
    if width <= 0:
        return sigma
    depth = (width - np.arange(width, dtype=np.float64)) / width
    ramp = sigma_max * depth**2
    if sides[0]:
        sigma[:width] = np.maximum(sigma[:width], ramp)
    if sides[1]:
        sigma[n - width:] = np.maximum(sigma[n - width:], ramp[::-1])
    return sigma


def _resolve_c_map(c: float | Field2D,
                   shape: tuple[int, int] | None) -> Field2D:
    """Broadcast/validate the sound-speed spec into a positive 2D map."""
    if np.isscalar(c):
        if shape is None:
            raise ValueError("shape is required when c is a scalar")
        c_map = np.full(shape, float(np.real(c)), dtype=np.float64)
    else:
        c_map = np.asarray(c, dtype=np.float64)
    if c_map.ndim != 2:
        raise ValueError("c must be a 2D (ny, nx) map")
    _positive_map("c", c_map)
    return c_map


def _resolve_rho_map(rho: float | Field2D, ny: int, nx: int) -> Field2D:
    """Broadcast/validate the density spec into a positive ``(ny, nx)`` map."""
    rho_map = (np.full((ny, nx), float(np.real(rho)), dtype=np.float64)
               if np.isscalar(rho) else np.asarray(rho, dtype=np.float64))
    if rho_map.shape != (ny, nx):
        raise ValueError("rho map must match the shape of c")
    _positive_map("rho", rho_map)
    return rho_map


def _resolve_sponge_sides(
    sponge_sides: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalise the sponge-side spec into a validated tuple of side names."""
    if sponge_sides is None:
        sides: tuple[str, ...] = _SIDES
    elif isinstance(sponge_sides, str):
        # A bare string would iterate per character; treat it as one side.
        sides = (sponge_sides,)
    else:
        sides = tuple(sponge_sides)
    unknown = set(sides) - set(_SIDES)
    if unknown:
        raise ValueError(f"unknown sponge sides: {sorted(unknown)}")
    return sides


def _edge_impedance_profile(side: str, value: float | NDArray[np.float64],
                            n_edge: int) -> Field2D:
    """Broadcast/validate one side's impedance into a positive 1D profile."""
    z = np.asarray(value, dtype=np.float64)
    if z.ndim == 0:
        z = np.full(n_edge, float(z), dtype=np.float64)
    if z.shape != (n_edge,):
        raise ValueError(
            f"impedance for side {side!r} must be a scalar or a 1D "
            f"array of length {n_edge}")
    if not np.all(np.isfinite(z)) or bool(np.any(z <= 0.0)):
        raise ValueError(f"impedance for side {side!r} must be "
                         "strictly positive and finite")
    return z


class _ImpedanceEdge:
    """One locally reacting boundary side with a real specific impedance.

    Implements the frequency-independent real-impedance velocity update of
    Attenborough & Van Renterghem Eqs. (4.33)-(4.35): the boundary-face
    normal velocity ``vb`` is stored on the wall, updated implicitly from the
    half-cell pressure gradient with the surface pressure eliminated through
    ``p_surf = Z * v_out`` (time-averaged over the two half steps), and its
    flux enters the divergence of the adjacent pressure cells.
    """

    __slots__ = ("c1", "c2", "side", "vb")

    def __init__(self, side: str, impedance: Field2D, rho_edge: Field2D,
                 dt: float, dx: float) -> None:
        self.side = side
        # a = dt Z / (rho dx): the dimensionless boundary Courant product.
        a = dt * impedance / (rho_edge * dx)
        self.c1 = (1.0 - a) / (1.0 + a)
        self.c2 = (2.0 * dt / (rho_edge * dx)) / (1.0 + a)
        self.vb: Field2D = np.zeros(impedance.shape, dtype=np.float64)

    def update(self, p: Field2D) -> None:
        """Advance the boundary-face velocity by one leapfrog step."""
        if self.side == "left":
            self.vb = self.c1 * self.vb - self.c2 * p[:, 0]
        elif self.side == "right":
            self.vb = self.c1 * self.vb + self.c2 * p[:, -1]
        elif self.side == "top":
            self.vb = self.c1 * self.vb - self.c2 * p[0, :]
        else:                                          # bottom
            self.vb = self.c1 * self.vb + self.c2 * p[-1, :]

    def add_flux(self, div: Field2D) -> None:
        """Add the boundary-face flux to the velocity divergence."""
        if self.side == "left":
            div[:, 0] -= self.vb
        elif self.side == "right":
            div[:, -1] += self.vb
        elif self.side == "top":
            div[0, :] -= self.vb
        else:                                          # bottom
            div[-1, :] += self.vb


class FDTD2D:
    """2D acoustic FDTD stepping engine on a staggered grid.

    Pressure ``p`` lives at cell centres, shape ``(ny, nx)`` (row = y,
    column = x, the ``imshow`` convention); ``vx`` at interior x-faces,
    shape ``(ny, nx - 1)``; ``vy`` at interior y-faces, shape
    ``(ny - 1, nx)``. Because only interior faces are stored, the domain
    boundary is perfectly rigid (zero normal velocity, Eq. 4.32) by
    construction; sponge layers and per-cell real impedances turn selected
    sides into absorbing or locally reacting boundaries. Sources are soft
    (additive) pressure injections.

    :param c: Sound-speed map [m/s], shape ``(ny, nx)``. A scalar with an
        explicit ``shape`` is also accepted.
    :param dx: Grid spacing [m] (square cells).
    :param rho: Density map [kg/m3]; scalar or ``(ny, nx)`` array
        (default 1.2).
    :param cfl: Courant number ``CN = c_max dt sqrt(2) / dx`` (Eq. 4.13);
        the explicit scheme is stable for ``CN <= 1`` (Eq. 4.14) and values
        in ``(0, 1)`` are accepted. The default 0.6 keeps a wide stability
        margin with moderate numerical dispersion.
    :param sponge_width: Thickness of the absorbing layer in cells
        (0 = no absorbing sides).
    :param sponge_sides: Which sides absorb: a single side name or an
        iterable drawn from ``{"left", "right", "top", "bottom"}``
        (default: all four when ``sponge_width > 0``). ``left``/``right``
        are the low/high column edges and ``top``/``bottom`` the low/high
        row edges (the default ``imshow`` origin).
    :param sponge_reflection: Target round-trip amplitude reflection of the
        sponge layer; sets the peak absorption rate.
    :param damping: Uniform bulk amplitude decay rate [1/s] applied to the
        whole field (a simple stand-in for air/wall absorption;
        ``6.91 / T60`` gives a ``T60`` seconds reverberant decay).
    :param shape: Grid shape ``(ny, nx)``, required only when ``c`` is a
        scalar.
    :param edge_impedance: Locally reacting boundary sides: a mapping from
        side name to a real specific acoustic impedance [Pa s/m], either a
        scalar or a per-edge-cell 1D array (length ``ny`` for ``left``/
        ``right``, ``nx`` for ``top``/``bottom``). Implements Eqs.
        (4.33)-(4.35); ``Z = rho c`` is a normal-incidence matched
        (anechoic) edge. A side cannot be both a sponge and an impedance
        boundary.
    :param obstacle_mask: Boolean map, shape ``(ny, nx)``, of rigid cells:
        every face adjacent to a masked cell is closed (zero normal
        velocity, Eq. 4.32), rasterising arbitrary interior geometry.
    """

    def __init__(
        self,
        c: float | Field2D,
        dx: float,
        *,
        rho: float | Field2D = 1.2,
        cfl: float = 0.6,
        sponge_width: int = 0,
        sponge_sides: str | Iterable[str] | None = None,
        sponge_reflection: float = 1e-4,
        damping: float = 0.0,
        shape: tuple[int, int] | None = None,
        edge_impedance: Mapping[str, float | NDArray[np.float64]]
        | None = None,
        obstacle_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        c_map = _resolve_c_map(c, shape)
        if not np.isfinite(cfl) or not 0.0 < cfl < 1.0:
            raise ValueError("cfl must lie in (0, 1): the leapfrog scheme "
                             "is unstable beyond the Courant bound CN = 1")
        ny, nx = c_map.shape
        rho_map = _resolve_rho_map(rho, ny, nx)
        sponge_width = _integer("sponge_width", sponge_width)
        if sponge_width < 0:
            raise ValueError("sponge_width must be non-negative")
        if sponge_width >= min(nx, ny):
            raise ValueError("sponge_width must be narrower than the "
                             "smallest grid side")
        if not 0.0 < sponge_reflection < 1.0:
            raise ValueError("sponge_reflection must lie strictly between "
                             "0 and 1")
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("damping must be non-negative and finite")

        self.dx = _positive_finite("dx", dx)
        self.c = c_map
        self.rho = rho_map
        c_max = float(c_map.max())
        self.dt = cfl * self.dx / (c_max * float(np.sqrt(2.0)))
        self.kappa = rho_map * c_map**2          # bulk modulus at centres
        # Density averaged onto the faces where each velocity lives.
        self._rho_x = 0.5 * (rho_map[:, 1:] + rho_map[:, :-1])
        self._rho_y = 0.5 * (rho_map[1:, :] + rho_map[:-1, :])

        self.p: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self.vx: Field2D = np.zeros((ny, nx - 1), dtype=np.float64)
        self.vy: Field2D = np.zeros((ny - 1, nx), dtype=np.float64)
        self._div: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self._sources: list[Source] = []
        self.n = 0                                # completed steps

        sides = _resolve_sponge_sides(sponge_sides)
        #: Sponge-layer width in cells (read-only configuration record).
        self.sponge_width = int(sponge_width)
        #: Sides carrying a sponge layer when ``sponge_width > 0``.
        self.sponge_sides: tuple[str, ...] = (
            sides if sponge_width > 0 else ()
        )
        #: Per-side impedance boundaries as supplied (immutable record).
        self.edge_impedance: Mapping[str, float | NDArray[np.float64]] = (
            MappingProxyType(dict(edge_impedance) if edge_impedance else {})
        )
        self._init_decay(sides, sponge_width, sponge_reflection, damping,
                         c_max)
        self._edges = self._build_edges(edge_impedance, sponge_width, sides,
                                        ny, nx)
        self._init_obstacle(obstacle_mask, ny, nx)

    def _init_decay(self, sides: tuple[str, ...], sponge_width: int,
                    sponge_reflection: float, damping: float,
                    c_max: float) -> None:
        """Precompute the sponge/damping decay factors of every field."""
        ny, nx = self.p.shape
        sigma_max = 0.0
        if sponge_width > 0:
            # Quadratic-profile PML-style rate for a target reflection R:
            # exp(-2 * sigma_max * (w dx) / (3 c)) = R  (two-way transit).
            sigma_max = (-3.0 * c_max * float(np.log(sponge_reflection))
                         / (2.0 * sponge_width * self.dx))
        sig_x = _sponge_profile(nx, sponge_width,
                                ("left" in sides, "right" in sides), sigma_max)
        sig_y = _sponge_profile(ny, sponge_width,
                                ("top" in sides, "bottom" in sides), sigma_max)
        sigma = sig_x[np.newaxis, :] + sig_y[:, np.newaxis] + damping
        self._decay_p: Field2D = np.exp(-sigma * self.dt)
        self._decay_vx: Field2D = np.exp(
            -(0.5 * (sigma[:, 1:] + sigma[:, :-1])) * self.dt)
        self._decay_vy: Field2D = np.exp(
            -(0.5 * (sigma[1:, :] + sigma[:-1, :])) * self.dt)

    def _init_obstacle(self, obstacle_mask: NDArray[np.bool_] | None,
                       ny: int, nx: int) -> None:
        """Validate the obstacle mask into closed-face velocity factors."""
        self._obstacle: NDArray[np.bool_] | None = None
        self._vx_open: Field2D | None = None
        self._vy_open: Field2D | None = None
        if obstacle_mask is None:
            return
        mask = np.asarray(obstacle_mask)
        if mask.shape != (ny, nx):
            raise ValueError("obstacle_mask must match the grid shape")
        if mask.dtype != np.bool_:
            raise ValueError("obstacle_mask must be a boolean array")
        if bool(mask.all()):
            raise ValueError("obstacle_mask must leave open cells")
        if bool(mask.any()):
            self._obstacle = mask.copy()
            # A face between two cells is open only when both are open
            # (rigid obstacle boundary: zero normal velocity, Eq. 4.32).
            self._vx_open = (
                ~(mask[:, 1:] | mask[:, :-1])).astype(np.float64)
            self._vy_open = (
                ~(mask[1:, :] | mask[:-1, :])).astype(np.float64)

    def _build_edges(
        self,
        edge_impedance: Mapping[str, float | NDArray[np.float64]] | None,
        sponge_width: int,
        sponge_sides: tuple[str, ...],
        ny: int,
        nx: int,
    ) -> list[_ImpedanceEdge]:
        """Validate the per-side impedance spec into edge updaters."""
        edges: list[_ImpedanceEdge] = []
        if not edge_impedance:
            return edges
        unknown = set(edge_impedance) - set(_SIDES)
        if unknown:
            raise ValueError(f"unknown impedance sides: {sorted(unknown)}")
        absorbing = set(sponge_sides) if sponge_width > 0 else set()
        rho_edges = {"left": self.rho[:, 0], "right": self.rho[:, -1],
                     "top": self.rho[0, :], "bottom": self.rho[-1, :]}
        for side in _SIDES:                      # deterministic order
            if side not in edge_impedance:
                continue
            if side in absorbing:
                raise ValueError(f"side {side!r} cannot be both absorbing "
                                 "and an impedance boundary")
            n_edge = ny if side in ("left", "right") else nx
            z = _edge_impedance_profile(side, edge_impedance[side], n_edge)
            edges.append(_ImpedanceEdge(side, z, rho_edges[side],
                                        self.dt, self.dx))
        return edges

    @property
    def time(self) -> float:
        """Elapsed simulated time [s]."""
        return self.n * self.dt

    def plot_geometry(
        self,
        ax: Axes | None = None,
        *,
        probes: ArrayLike | None = None,
        language: str = "en",
        **kwargs: Any,
    ) -> Axes:
        """Draw the configured domain before running it.

        Domain extent, obstacles, sponge layers, impedance and rigid edges
        and the added sources, with optional probe positions previewed; no
        time stepping happens. Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param probes: Optional probe positions ``(x, y)`` in metres, shape
            ``(N, 2)`` (the :func:`fdtd_simulation` convention).
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the obstacle ``imshow``.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_fdtd_domain

        check_language(language)
        return plot_fdtd_domain(
            self, ax=ax, probes=probes, language=language, **kwargs
        )

    def add_source(self, source: Source) -> None:
        """Register a soft pressure source (additive injection at one cell)."""
        ny, nx = self.p.shape
        ix = _integer("source ix", source.ix)
        iy = _integer("source iy", source.iy)
        if not (0 <= ix < nx and 0 <= iy < ny):
            raise ValueError("source position lies outside the grid")
        if self._obstacle is not None and self._obstacle[iy, ix]:
            raise ValueError("source position lies inside an obstacle")
        self._sources.append(source)

    def step(self) -> None:
        """Advance the leapfrog scheme by one time step."""
        dt_dx = self.dt / self.dx
        # Velocity half-step from the pressure gradient (rigid walls are the
        # implicit zero normal velocity at the domain edge).
        self.vx -= dt_dx / self._rho_x * (self.p[:, 1:] - self.p[:, :-1])
        self.vy -= dt_dx / self._rho_y * (self.p[1:, :] - self.p[:-1, :])
        if self._vx_open is not None and self._vy_open is not None:
            self.vx *= self._vx_open
            self.vy *= self._vy_open
        self.vx *= self._decay_vx
        self.vy *= self._decay_vy
        for edge in self._edges:
            edge.update(self.p)
        # Pressure step from the velocity divergence (reused buffer).
        div = self._div
        div.fill(0.0)
        div[:, :-1] += self.vx
        div[:, 1:] -= self.vx
        div[:-1, :] += self.vy
        div[1:, :] -= self.vy
        for edge in self._edges:
            edge.add_flux(div)
        self.p -= self.kappa * dt_dx * div
        t_next = (self.n + 1) * self.dt
        for src in self._sources:
            self.p[src.iy, src.ix] += src.value(t_next)
        self.p *= self._decay_p
        self.n += 1

    def energy(self) -> float:
        """Total acoustic field energy [J per metre of depth]."""
        e_p = float(np.sum(self.p**2 / (2.0 * self.kappa)))
        e_v = 0.5 * (float(np.sum(self._rho_x * self.vx**2))
                     + float(np.sum(self._rho_y * self.vy**2)))
        return (e_p + e_v) * self.dx**2

    def run(
        self,
        steps: int,
        record_every: int | None = None,
        decimate: int = 1,
    ) -> NDArray[np.float64]:
        """Advance ``steps`` steps, optionally recording pressure frames.

        With ``record_every = k`` a snapshot of ``p`` is stored after every
        ``k``-th step (and one of the initial state), spatially subsampled by
        ``decimate``; the stacked ``(n_frames, ny', nx')`` array plugs
        straight into a ``FuncAnimation`` ``imshow`` update. Without
        ``record_every`` an empty array is returned and only the final state
        is kept (read it from ``self.p``).
        """
        steps = _integer("steps", steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if record_every is not None:
            record_every = _integer("record_every", record_every)
            if record_every < 1:
                raise ValueError("record_every must be >= 1")
        decimate = _integer("decimate", decimate)
        if decimate < 1:
            raise ValueError("decimate must be >= 1")
        frames: list[Field2D] = []
        if record_every is not None:
            frames.append(self.p[::decimate, ::decimate].copy())
        for i in range(steps):
            self.step()
            if record_every is not None and (i + 1) % record_every == 0:
                frames.append(self.p[::decimate, ::decimate].copy())
        if not frames:
            return np.zeros((0, 0, 0), dtype=np.float64)
        return np.stack(frames)


@dataclass(frozen=True)
class FDTDResult:
    """Frozen result of a :func:`fdtd_simulation` run.

    :ivar times: Time axis [s], length ``n_steps + 1`` (includes ``t = 0``).
    :ivar pressures: Pressure history at each probe [Pa], shape
        ``(n_probes, n_steps + 1)``.
    :ivar probes: Probe cell indices ``(ix, iy)``, shape ``(n_probes, 2)``.
    :ivar probe_positions: Probe cell-centre positions ``(x, y)`` [m], shape
        ``(n_probes, 2)``.
    :ivar dx: Grid spacing [m].
    :ivar dt: Time step [s].
    :ivar shape: Grid shape ``(ny, nx)``.
    :ivar sources: The source definitions of the run.
    :ivar snapshots: Recorded pressure fields, shape ``(n_frames, ny, nx)``,
        or ``None`` when no snapshots were requested.
    :ivar snapshot_times: Time of each snapshot [s], or ``None``.
    :ivar obstacle_mask: Boolean map of rigid cells, or ``None``.
    """

    times: NDArray[np.float64]
    pressures: NDArray[np.float64]
    probes: NDArray[np.int_]
    probe_positions: NDArray[np.float64]
    dx: float
    dt: float
    shape: tuple[int, int]
    sources: tuple[Source, ...]
    snapshots: NDArray[np.float64] | None
    snapshot_times: NDArray[np.float64] | None
    obstacle_mask: NDArray[np.bool_] | None

    @property
    def size(self) -> tuple[float, float]:
        """Domain size ``(lx, ly)`` [m]."""
        ny, nx = self.shape
        return nx * self.dx, ny * self.dx

    def plot(self, ax: Axes | None = None, *, kind: str = "probes",
             frame: int = -1, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the probe histories or one recorded field snapshot.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param kind: ``"probes"`` (default) draws the per-probe pressure
            time histories; ``"snapshot"`` renders one recorded pressure
            field with the geometry overlaid (``imshow`` raster).
        :param frame: Snapshot index for ``kind="snapshot"`` (default: the
            last recorded frame).
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the underlying ``plot``/``imshow``.
        :return: The axes.
        """
        from .._i18n import check_language

        check_language(language)
        if kind == "probes":
            from .._plot.simulation import plot_fdtd_probes

            return plot_fdtd_probes(self, ax=ax, language=language, **kwargs)
        if kind == "snapshot":
            from .._plot.simulation import plot_fdtd_snapshot

            return plot_fdtd_snapshot(self, ax=ax, frame=frame,
                                      language=language, **kwargs)
        raise ValueError("kind must be 'probes' or 'snapshot'")


def _parse_boundaries(
    boundaries: str | Mapping[str, str | float | NDArray[np.float64]],
) -> tuple[tuple[str, ...],
           dict[str, float | NDArray[np.float64]]]:
    """Split the boundary spec into sponge sides and impedance sides."""
    spec: dict[str, str | float | NDArray[np.float64]]
    if isinstance(boundaries, str):
        spec = dict.fromkeys(_SIDES, boundaries)
    else:
        unknown = set(boundaries) - set(_SIDES)
        if unknown:
            raise ValueError(f"unknown boundary sides: {sorted(unknown)}")
        spec = {side: boundaries.get(side, "rigid") for side in _SIDES}
    absorbing: list[str] = []
    impedance: dict[str, float | NDArray[np.float64]] = {}
    for side in _SIDES:
        value = spec[side]
        if isinstance(value, str):
            if value not in _BOUNDARY_NAMES:
                raise ValueError(
                    f"boundary for side {side!r} must be one of "
                    f"{_BOUNDARY_NAMES} or a real impedance, got {value!r}")
            if value == "absorbing":
                absorbing.append(side)
        else:
            impedance[side] = value
    return tuple(absorbing), impedance


def _probe_indices(probes: Sequence[tuple[int, int]],
                   shape: tuple[int, int],
                   obstacle: NDArray[np.bool_] | None) -> NDArray[np.int_]:
    """Validate the probe cells into an ``(n_probes, 2)`` index array."""
    ny, nx = shape
    probe_ix = np.zeros((len(probes), 2), dtype=np.int_)
    for k, (ix, iy) in enumerate(probes):
        ix = _integer("probe ix", ix)
        iy = _integer("probe iy", iy)
        if not (0 <= ix < nx and 0 <= iy < ny):
            raise ValueError(f"probe ({ix}, {iy}) lies outside the grid")
        if obstacle is not None and obstacle[iy, ix]:
            raise ValueError(f"probe ({ix}, {iy}) lies inside an obstacle")
        probe_ix[k] = (ix, iy)
    return probe_ix


def _record_run(
    sim: FDTD2D,
    steps: int,
    probe_ix: NDArray[np.int_],
    snapshot_every: int | None,
) -> tuple[NDArray[np.float64], list[Field2D], list[int]]:
    """Step the engine, recording probe histories and field snapshots."""
    pressures = np.zeros((probe_ix.shape[0], steps + 1), dtype=np.float64)
    frames: list[Field2D] = []
    frame_steps: list[int] = []
    if snapshot_every is not None:
        frames.append(sim.p.copy())
        frame_steps.append(0)
    rows = probe_ix[:, 1]
    cols = probe_ix[:, 0]
    for i in range(steps):
        sim.step()
        if probe_ix.shape[0]:
            pressures[:, i + 1] = sim.p[rows, cols]
        if snapshot_every is not None and (i + 1) % snapshot_every == 0:
            frames.append(sim.p.copy())
            frame_steps.append(i + 1)
    return pressures, frames, frame_steps


def fdtd_simulation(
    c: float | Field2D,
    dx: float,
    duration: float,
    *,
    sources: Sequence[Source],
    shape: tuple[int, int] | None = None,
    rho: float | Field2D = 1.2,
    cfl: float = 0.6,
    probes: Sequence[tuple[int, int]] = (),
    boundaries: str | Mapping[str, str | float | NDArray[np.float64]]
    = "rigid",
    absorbing_layer_cells: int = 20,
    obstacle_mask: NDArray[np.bool_] | None = None,
    damping: float = 0.0,
    snapshot_every: int | None = None,
) -> FDTDResult:
    """Run a deterministic 2D acoustic FDTD simulation.

    Builds the staggered-grid domain (Attenborough & Van Renterghem 2021,
    Eqs. 4.11-4.12), applies the requested boundary conditions, injects the
    sources and integrates for ``duration`` seconds, recording the pressure
    at every probe each time step and, optionally, full-field snapshots.

    The grid covers ``(nx * dx, ny * dx)`` metres; a cell index ``(ix, iy)``
    maps to the physical cell centre ``((ix + 0.5) * dx, (iy + 0.5) * dx)``.
    Resolve at least 10 cells per shortest wavelength using the smallest
    sound speed of the domain (``dx <= c_min / (10 f)``), the usual rule for
    this lowest-order scheme: the worst-case (on-axis) numerical dispersion
    error magnitude, ``(k dx)^2 / 24`` from the discrete counterpart of
    Eq. 4.15 (the modelled frequency under-reads, so the signed error is
    negative), is then about 1.6 % (about 1.4 % at the default ``cfl``;
    in a heterogeneous domain the slower cells run at a lower local Courant
    number and sit nearer the 1.6 % bound) and finer grids reduce it
    quadratically. The
    simulation is 2D, so a point source is
    physically a line source with cylindrical ``1/sqrt(r)`` amplitude
    spreading rather than the 3D spherical ``1/r``.

    :param c: Sound-speed map [m/s], shape ``(ny, nx)``, or a scalar with an
        explicit ``shape``.
    :param dx: Grid spacing [m] (square cells).
    :param duration: Physical time to simulate [s].
    :param sources: One or more of :class:`GaussianPulse`,
        :class:`CWSource` or :class:`SignalSource`.
    :param shape: Grid shape ``(ny, nx)``, required when ``c`` is a scalar.
    :param rho: Density map [kg/m3]; scalar or ``(ny, nx)`` array.
    :param cfl: Courant number in ``(0, 1)`` (Eqs. 4.13-4.14); the time step
        is ``dt = cfl * dx / (c_max * sqrt(2))``. Default 0.6.
    :param probes: Pressure-probe cells as ``(ix, iy)`` index pairs.
    :param boundaries: ``"rigid"`` (default), ``"absorbing"``, or a mapping
        from side name (``left``/``right``/``top``/``bottom``) to
        ``"rigid"``, ``"absorbing"``, or a real specific impedance
        [Pa s/m] (scalar or per-edge-cell 1D array, Eqs. 4.33-4.35).
    :param absorbing_layer_cells: Sponge-layer thickness for absorbing
        sides, in cells.
    :param obstacle_mask: Boolean map, shape ``(ny, nx)``, of rigid cells
        (rasterised interior geometry).
    :param damping: Uniform bulk amplitude decay rate [1/s].
    :param snapshot_every: Record a full pressure-field snapshot every this
        many steps (and at ``t = 0``); ``None`` records none.
    :return: A :class:`FDTDResult`.
    :raises ValueError: If the inputs are invalid.
    """
    if len(sources) == 0:
        raise ValueError("at least one source is required")
    duration = _positive_finite("duration", duration)
    if snapshot_every is not None:
        snapshot_every = _integer("snapshot_every", snapshot_every)
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be >= 1")
    absorbing_sides, edge_impedance = _parse_boundaries(boundaries)
    if absorbing_sides:
        absorbing_layer_cells = _integer("absorbing_layer_cells",
                                         absorbing_layer_cells)
        if absorbing_layer_cells < 1:
            raise ValueError("absorbing_layer_cells must be >= 1")

    sim = FDTD2D(
        c, dx,
        rho=rho,
        cfl=cfl,
        sponge_width=absorbing_layer_cells if absorbing_sides else 0,
        sponge_sides=absorbing_sides if absorbing_sides else None,
        damping=damping,
        shape=shape,
        edge_impedance=edge_impedance if edge_impedance else None,
        obstacle_mask=obstacle_mask,
    )
    for source in sources:
        sim.add_source(source)

    ny, nx = sim.p.shape
    probe_ix = _probe_indices(probes, (ny, nx), sim._obstacle)

    steps = round(duration / sim.dt)
    if steps < 1:
        raise ValueError("duration must cover at least one time step "
                         f"(dt = {sim.dt:.3e} s)")
    times = np.arange(steps + 1, dtype=np.float64) * sim.dt
    pressures, frames, frame_steps = _record_run(sim, steps, probe_ix,
                                                 snapshot_every)

    positions = (probe_ix.astype(np.float64) + 0.5) * sim.dx
    return FDTDResult(
        times=times,
        pressures=pressures,
        probes=probe_ix,
        probe_positions=positions,
        dx=sim.dx,
        dt=sim.dt,
        shape=(ny, nx),
        sources=tuple(sources),
        snapshots=np.stack(frames) if frames else None,
        snapshot_times=(np.asarray(frame_steps, dtype=np.float64) * sim.dt
                        if frame_steps else None),
        obstacle_mask=(sim._obstacle.copy()
                       if sim._obstacle is not None else None),
    )
