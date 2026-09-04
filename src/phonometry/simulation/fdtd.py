#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""2D acoustic finite-difference time-domain (FDTD) simulation.

A staggered-grid (Yee-style) pressure-velocity leapfrog solver for the
linear acoustic equations in a non-moving medium, following the reference
formulation of Attenborough & Van Renterghem, *Predicting Outdoor Sound*
(2nd ed., CRC Press 2021), chapter 4:

* the governing first-order system in ``p`` and ``v`` (Eqs. 4.3-4.4);
* the staggered-in-place, staggered-in-time discretisation (Eqs. 4.11-4.12),
  with pressure at cell centres and velocity components on cell faces;
* the Courant stability condition :math:`C_\mathrm{N} \le 1` with
  :math:`C_\mathrm{N} = c\,\Delta t \sqrt{1/\Delta x^2 + 1/\Delta y^2}`
  (Eqs. 4.13-4.14);
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
free-field pulse arrival times and cylindrical :math:`1/\sqrt{r}` amplitude
decay, the image-source echo of a rigid wall, the normal-incidence
reflection coefficient :math:`(Z - \rho c)/(Z + \rho c)` of an impedance
edge, and second-order convergence under grid refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.validation import (
    require_equal_counts,
    require_positive,
    require_ranks,
    require_same_length,
)
from .ntff import ContourPhasors

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from matplotlib.axes import Axes

Field2D = NDArray[np.float64]

_SIDES = ("left", "right", "top", "bottom")
#: Plane-wave travel directions accepted by ``FDTD2D.add_plane_wave``.
_SIDE_TRAVEL = ("down", "up", "left", "right")

#: Boundary-condition names accepted by :func:`fdtd_simulation`.
_BOUNDARY_NAMES = ("rigid", "absorbing")


def _positive_finite(name: str, value: float) -> float:
    """Validate that *value* is a strictly positive finite scalar."""
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        msg = f"{name} must be positive and finite"
        raise ValueError(msg)
    return out


def _finite(name: str, value: float) -> float:
    """Validate that *value* is a finite scalar."""
    out = float(value)
    if not np.isfinite(out):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return out


def _integer(name: str, value: int) -> int:
    """Validate that *value* is an integral scalar (bool is rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer"
        # ValueError keeps the module validation errors uniform.
        raise ValueError(msg)  # noqa: TRY004
    return int(value)


def _positive_map(name: str, field: Field2D) -> None:
    """Validate that every cell of *field* is strictly positive and finite."""
    if not np.all(np.isfinite(field)) or bool(np.any(field <= 0.0)):
        msg = f"{name} must be strictly positive and finite everywhere"
        raise ValueError(msg)


@dataclass(frozen=True)
class GaussianPulse:
    r"""A soft Gaussian pressure pulse injected at one cell.

    :math:`s(t) = \text{amplitude} \cdot e^{-((t - t_0)/\text{width})^2}`
    with ``t0`` defaulting to ``4 * width`` so the pulse starts from
    (numerically) zero.

    :ivar ix: Source column (x) index; the cell centre is at
        :math:`x = (i_x + 0.5)\,\Delta x`.
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
        """Require a positive ``width`` and finite ``amplitude`` and ``t0``."""
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
        """Require a positive ``frequency`` and a non-negative ``ramp_cycles``."""
        _positive_finite("frequency", self.frequency)
        _finite("amplitude", self.amplitude)
        if not np.isfinite(self.ramp_cycles) or self.ramp_cycles < 0.0:
            msg = "ramp_cycles must be non-negative and finite"
            raise ValueError(msg)

    def value(self, t: float) -> float:
        """Source waveform at time ``t`` (seconds)."""
        ramp_time = self.ramp_cycles / self.frequency
        if t < ramp_time and ramp_time > 0.0:
            envelope = 0.5 * (1.0 - float(np.cos(np.pi * t / ramp_time)))
        else:
            envelope = 1.0
        return (
            self.amplitude * envelope * float(np.sin(2.0 * np.pi * self.frequency * t))
        )


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
        """Require a positive ``sample_rate`` and freeze finite 1D ``samples``."""
        _positive_finite("sample_rate", self.sample_rate)
        _finite("amplitude", self.amplitude)
        arr = np.array(self.samples, dtype=np.float64)
        if arr.ndim != 1:
            msg = "samples must be a 1D array"
            raise ValueError(msg)
        if arr.size == 0:
            msg = "samples must not be empty"
            raise ValueError(msg)
        if not np.all(np.isfinite(arr)):
            msg = "samples must be finite"
            raise ValueError(msg)
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
        return self.amplitude * float(
            (1.0 - w) * self.samples[i] + w * self.samples[i + 1]
        )


@dataclass(frozen=True)
class PlaneWaveSource:
    """A sustained one-way plane wave injected on a line near one edge.

    A total-field/scattered-field style injection: each step the incident
    wave is added simultaneously to the pressure on the injection line and
    to the particle velocity on the adjacent face, so the wave launches
    only toward ``direction`` and anything scattered back crosses the line
    untouched (and can be absorbed by a sponge behind it).

    ``direction`` is the travel direction (``"down"``, ``"up"``,
    ``"left"``, ``"right"`` in the :meth:`FDTD2D.plot_geometry` axes) and
    the line sits ``offset`` cells in from the opposite edge (place it just
    inside the sponge layer when one is configured on that side).
    ``waveform`` maps time in seconds to the incident pressure in pascals
    (any callable, or reuse the ``value`` method of a point source).

    :ivar direction: Travel direction of the launched wave.
    :ivar waveform: Callable ``t -> p_inc(t)`` in pascals.
    :ivar offset: Line position, in cells from the launch edge.
    :ivar amplitude: Extra gain applied to ``waveform``.
    """

    direction: str
    waveform: Callable[[float], float]
    offset: int = 0
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        """Require a callable ``waveform`` and a finite ``amplitude``.

        ``direction`` and ``offset`` are checked against the grid in
        :meth:`FDTD2D.add_source`, which is the first place a grid exists.
        """
        if not callable(self.waveform):
            msg = "waveform must be callable"
            # ValueError keeps the module validation errors uniform.
            raise ValueError(msg)  # noqa: TRY004
        _finite("amplitude", self.amplitude)


Source = GaussianPulse | CWSource | SignalSource

#: Everything :meth:`FDTD2D.add_source` accepts.
AnySource = Source | PlaneWaveSource


def _sponge_profile(
    n: int, width: int, sides: tuple[bool, bool], sigma_max: float
) -> Field2D:
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
        sigma[n - width :] = np.maximum(sigma[n - width :], ramp[::-1])
    return sigma


def _sigma_map(
    shape: tuple[int, int],
    sides: tuple[str, ...],
    sponge_width: int,
    sponge_reflection: float,
    damping: NDArray[np.float64],
    c_max: float,
    dx: float,
) -> Field2D:
    """Absorption-rate map sigma(x, y) [1/s]: sponge ramps plus damping."""
    ny, nx = shape
    sigma_max = 0.0
    if sponge_width > 0:
        # Quadratic-profile PML-style rate for a target reflection R:
        # exp(-2 * sigma_max * (w dx) / (3 c)) = R  (two-way transit).
        sigma_max = (
            -3.0 * c_max * float(np.log(sponge_reflection)) / (2.0 * sponge_width * dx)
        )
    sig_x = _sponge_profile(
        nx, sponge_width, ("left" in sides, "right" in sides), sigma_max
    )
    sig_y = _sponge_profile(
        ny, sponge_width, ("top" in sides, "bottom" in sides), sigma_max
    )
    return sig_x[np.newaxis, :] + sig_y[:, np.newaxis] + damping


def _validate_sponge_damping(
    sponge_width: int,
    sponge_reflection: float,
    damping: float | NDArray[np.float64],
    ny: int,
    nx: int,
) -> tuple[int, NDArray[np.float64]]:
    """Validate the sponge/damping spec into ``(width, damping map)``."""
    sponge_width = _integer("sponge_width", sponge_width)
    if sponge_width < 0:
        msg = "sponge_width must be non-negative"
        raise ValueError(msg)
    if sponge_width >= min(nx, ny):
        msg = "sponge_width must be narrower than the smallest grid side"
        raise ValueError(msg)
    if not 0.0 < sponge_reflection < 1.0:
        msg = "sponge_reflection must lie strictly between 0 and 1"
        raise ValueError(msg)
    damping_map = np.asarray(damping, dtype=np.float64)
    if damping_map.ndim not in (0, 2):
        msg = "damping must be a scalar or an (ny, nx) map"
        raise ValueError(msg)
    if not np.all(np.isfinite(damping_map)) or np.any(damping_map < 0.0):
        msg = "damping must be non-negative and finite"
        raise ValueError(msg)
    if damping_map.ndim == 2 and damping_map.shape != (ny, nx):  # noqa: PLR2004
        msg = (
            f"damping map shape {damping_map.shape} does not match the grid {(ny, nx)}"
        )
        raise ValueError(msg)
    return sponge_width, damping_map


class _PressureEngine(Protocol):
    """The stepping interface the shared ``run()`` driver relies on."""

    def step(self) -> None:
        """Advance the leapfrog scheme by one time step."""

    @property
    def p(self) -> Field2D:
        """Pressure at the cell centres, shape ``(ny, nx)``."""


def _run_recording(
    engine: _PressureEngine,
    steps: int,
    record_every: int | None,
    decimate: int,
) -> NDArray[np.float64]:
    """Validate the ``run()`` arguments and march, stacking pressure frames."""
    steps = _integer("steps", steps)
    if steps < 0:
        msg = "steps must be non-negative"
        raise ValueError(msg)
    if record_every is not None:
        record_every = _integer("record_every", record_every)
        if record_every < 1:
            msg = "record_every must be >= 1"
            raise ValueError(msg)
    decimate = _integer("decimate", decimate)
    if decimate < 1:
        msg = "decimate must be >= 1"
        raise ValueError(msg)
    frames: list[Field2D] = []
    if record_every is not None:
        frames.append(engine.p[::decimate, ::decimate].copy())
    for i in range(steps):
        engine.step()
        if record_every is not None and (i + 1) % record_every == 0:
            frames.append(engine.p[::decimate, ::decimate].copy())
    if not frames:
        return np.zeros((0, 0, 0), dtype=np.float64)
    return np.stack(frames)


def _validated_obstacle(
    obstacle_mask: NDArray[np.bool_] | None, ny: int, nx: int
) -> NDArray[np.bool_] | None:
    """Validate the obstacle mask; ``None`` when absent or all open."""
    if obstacle_mask is None:
        return None
    mask = np.asarray(obstacle_mask)
    if mask.shape != (ny, nx):
        msg = "obstacle_mask must match the grid shape"
        raise ValueError(msg)
    if mask.dtype != np.bool_:
        msg = "obstacle_mask must be a boolean array"
        raise ValueError(msg)
    if bool(mask.all()):
        msg = "obstacle_mask must leave open cells"
        raise ValueError(msg)
    return mask.copy() if bool(mask.any()) else None


def _resolve_c_map(c: float | Field2D, shape: tuple[int, int] | None) -> Field2D:
    """Broadcast/validate the sound-speed spec into a positive 2D map."""
    if np.iscomplexobj(c):
        msg = "c must be real; a complex sound speed is not supported"
        raise ValueError(msg)
    if np.isscalar(c):
        if shape is None:
            msg = "shape is required when c is a scalar"
            raise ValueError(msg)
        valid = isinstance(shape, (tuple, list)) and len(shape) == 2  # noqa: PLR2004
        if valid:
            valid = all(
                not isinstance(n, bool) and isinstance(n, (int, np.integer)) and n >= 1
                for n in shape
            )
        if not valid:
            msg = f"shape must be a pair of positive integers (ny, nx); got {shape!r}"
            raise ValueError(msg)
        c_map = np.full(shape, float(np.real(c)), dtype=np.float64)
    else:
        c_map = np.asarray(c, dtype=np.float64)
    if c_map.ndim != 2:  # noqa: PLR2004
        msg = "c must be a 2D (ny, nx) map"
        raise ValueError(msg)
    _positive_map("c", c_map)
    return c_map


def _resolve_rho_map(rho: float | Field2D, ny: int, nx: int) -> Field2D:
    """Broadcast/validate the density spec into a positive ``(ny, nx)`` map."""
    if np.iscomplexobj(rho):
        msg = "rho must be real; a complex density is not supported"
        raise ValueError(msg)
    rho_map = (
        np.full((ny, nx), float(np.real(rho)), dtype=np.float64)
        if np.isscalar(rho)
        else np.asarray(rho, dtype=np.float64)
    )
    if rho_map.shape != (ny, nx):
        msg = "rho map must match the shape of c"
        raise ValueError(msg)
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
        msg = f"unknown sponge sides: {sorted(unknown)}"
        raise ValueError(msg)
    return sides


def _edge_impedance_profile(
    side: str, value: float | NDArray[np.float64], n_edge: int
) -> Field2D:
    """Broadcast/validate one side's impedance into a positive 1D profile."""
    z = np.asarray(value, dtype=np.float64)
    if z.ndim == 0:
        z = np.full(n_edge, float(z), dtype=np.float64)
    if z.shape != (n_edge,):
        msg = (
            f"impedance for side {side!r} must be a scalar or a 1D "
            f"array of length {n_edge}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(z)) or bool(np.any(z <= 0.0)):
        msg = f"impedance for side {side!r} must be strictly positive and finite"
        raise ValueError(msg)
    return z


class _ImpedanceEdge:
    r"""One locally reacting boundary side with a real specific impedance.

    Implements the frequency-independent real-impedance velocity update of
    Attenborough & Van Renterghem Eqs. (4.33)-(4.35): the boundary-face
    normal velocity ``vb`` is stored on the wall, updated implicitly from the
    half-cell pressure gradient with the surface pressure eliminated through
    :math:`p_{\text{surf}} = Z v_{\text{out}}` (time-averaged over the two
    half steps), and its flux enters the divergence of the adjacent pressure
    cells.
    """

    __slots__ = ("c1", "c2", "side", "vb")

    def __init__(
        self, side: str, impedance: Field2D, rho_edge: Field2D, dt: float, dx: float
    ) -> None:
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
        else:  # bottom
            self.vb = self.c1 * self.vb + self.c2 * p[-1, :]

    def add_flux(self, div: Field2D) -> None:
        """Add the boundary-face flux to the velocity divergence."""
        if self.side == "left":
            div[:, 0] -= self.vb
        elif self.side == "right":
            div[:, -1] += self.vb
        elif self.side == "top":
            div[0, :] -= self.vb
        else:  # bottom
            div[-1, :] += self.vb


class ContourProbe:
    r"""On-the-fly DFT of ``p`` and ``v_n`` on a closed rectangular contour.

    Created by :meth:`FDTD2D.add_contour_probe`. The contour is the closed
    rectangle of cell faces around the cell block ``ix0..ix1`` x
    ``iy0..iy1`` (inclusive); on each face the engine samples the outward
    normal velocity (which lives exactly there on the staggered grid) and
    the pressure averaged from the two adjacent cell centres, and folds
    them into complex accumulators at each requested frequency, so a
    continuous-wave run never stores full time histories.

    After every step the accumulators gain :math:`p\, e^{-j \omega t}` with
    the fields' own leapfrog time stamps (:math:`t = n\,\Delta t` for
    pressure, :math:`t = (n - 1/2)\,\Delta t` for velocity, so the half-step
    stagger is handled
    exactly). :meth:`phasors` scales the sums by ``2 / n_samples`` into the
    steady-state complex amplitudes of the library's :math:`e^{+j \omega t}`
    convention. Accumulate only over the steady state: run the transient
    out, call :meth:`reset`, then integrate a window as close as possible
    to a whole number of periods (the residual leakage falls as one over
    the number of periods).

    :ivar positions: Face-sample positions ``(x, y)`` [m], shape
        ``(n_points, 2)``, ordered left, right, top, bottom face.
    :ivar normals: Outward unit normals of each sample, same shape.
    :ivar frequencies: The tracked frequencies [Hz].
    :ivar samples: Number of steps accumulated since the last reset.
    """

    def __init__(
        self,
        sim: FDTD2D,
        ix0: int,
        ix1: int,
        iy0: int,
        iy1: int,
        frequencies: ArrayLike,
    ) -> None:
        ny, nx = sim.p.shape
        ix0 = _integer("ix0", ix0)
        ix1 = _integer("ix1", ix1)
        iy0 = _integer("iy0", iy0)
        iy1 = _integer("iy1", iy1)
        if not (1 <= ix0 <= ix1 <= nx - 2 and 1 <= iy0 <= iy1 <= ny - 2):
            msg = (
                "the contour block needs 1 <= ix0 <= ix1 <= nx - 2 and "
                "1 <= iy0 <= iy1 <= ny - 2, so every sampled face has an "
                "open cell on both sides"
            )
            raise ValueError(msg)
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        if (
            freqs.ndim != 1
            or freqs.size == 0
            or not np.all(np.isfinite(freqs))
            or bool(np.any(freqs <= 0.0))
        ):
            msg = (
                "frequencies must be a non-empty 1D sequence of positive finite values"
            )
            raise ValueError(msg)
        if sim._obstacle is not None:
            # Every sampled face needs open cells on both of its sides:
            # the two cell columns flanking the vertical faces and the two
            # cell rows flanking the horizontal ones.
            blocked = bool(
                sim._obstacle[iy0 : iy1 + 1, [ix0 - 1, ix0, ix1, ix1 + 1]].any()
                or sim._obstacle[[iy0 - 1, iy0, iy1, iy1 + 1], ix0 : ix1 + 1].any()
            )
            if blocked:
                msg = (
                    "the contour faces touch obstacle cells; the closed "
                    "rectangle must run through open air with the whole "
                    "scatterer strictly inside"
                )
                raise ValueError(msg)
        self._ix0, self._ix1 = ix0, ix1
        self._iy0, self._iy1 = iy0, iy1
        self._ys = slice(iy0, iy1 + 1)
        self._xs = slice(ix0, ix1 + 1)
        n_side = iy1 - iy0 + 1
        n_band = ix1 - ix0 + 1
        dx = sim.dx
        y_faces = (np.arange(iy0, iy1 + 1, dtype=np.float64) + 0.5) * dx
        x_faces = (np.arange(ix0, ix1 + 1, dtype=np.float64) + 0.5) * dx
        pos = np.empty((2 * (n_side + n_band), 2), dtype=np.float64)
        nrm = np.zeros_like(pos)
        s_left = slice(0, n_side)
        s_right = slice(n_side, 2 * n_side)
        s_top = slice(2 * n_side, 2 * n_side + n_band)
        s_bottom = slice(2 * n_side + n_band, 2 * (n_side + n_band))
        pos[s_left, 0], pos[s_left, 1] = ix0 * dx, y_faces
        pos[s_right, 0], pos[s_right, 1] = (ix1 + 1) * dx, y_faces
        pos[s_top, 0], pos[s_top, 1] = x_faces, iy0 * dx
        pos[s_bottom, 0], pos[s_bottom, 1] = x_faces, (iy1 + 1) * dx
        nrm[s_left, 0], nrm[s_right, 0] = -1.0, 1.0
        nrm[s_top, 1], nrm[s_bottom, 1] = -1.0, 1.0
        pos.flags.writeable = False
        nrm.flags.writeable = False
        self.positions: NDArray[np.float64] = pos
        self.normals: NDArray[np.float64] = nrm
        self.frequencies: tuple[float, ...] = tuple(float(f) for f in freqs)
        self._omega = 2.0 * np.pi * freqs
        self._dt = sim.dt
        self._dx = dx
        self._acc_p = np.zeros((freqs.size, pos.shape[0]), dtype=np.complex128)
        self._acc_v = np.zeros_like(self._acc_p)
        self.samples = 0

    def reset(self) -> None:
        """Clear the accumulators (call once the field is steady)."""
        self._acc_p.fill(0.0)
        self._acc_v.fill(0.0)
        self.samples = 0

    def _accumulate(self, p: Field2D, vx: Field2D, vy: Field2D, n: int) -> None:
        """Fold the current fields into the running DFT sums."""
        ys, xs = self._ys, self._xs
        ix0, ix1 = self._ix0, self._ix1
        iy0, iy1 = self._iy0, self._iy1
        p_now = np.concatenate(
            (
                0.5 * (p[ys, ix0 - 1] + p[ys, ix0]),
                0.5 * (p[ys, ix1] + p[ys, ix1 + 1]),
                0.5 * (p[iy0 - 1, xs] + p[iy0, xs]),
                0.5 * (p[iy1, xs] + p[iy1 + 1, xs]),
            )
        )
        v_now = np.concatenate(
            (
                -vx[ys, ix0 - 1],
                vx[ys, ix1],
                -vy[iy0 - 1, xs],
                vy[iy1, xs],
            )
        )
        # Leapfrog time stamps: p sits at n dt, v half a step earlier.
        phase_p = np.exp(self._omega * (-1j * (n * self._dt)))
        phase_v = np.exp(self._omega * (-1j * ((n - 0.5) * self._dt)))
        self._acc_p += phase_p[:, np.newaxis] * p_now[np.newaxis, :]
        self._acc_v += phase_v[:, np.newaxis] * v_now[np.newaxis, :]
        self.samples += 1

    def phasors(self, frequency: float) -> ContourPhasors:
        """The accumulated contour phasors at one tracked frequency.

        :param frequency: One of the frequencies the probe was created
            with, in hertz.
        :return: A :class:`~phonometry.simulation.ntff.ContourPhasors`
            ready for
            :func:`~phonometry.simulation.ntff.far_field_from_contour`.
        :raises ValueError: If the frequency is not tracked or nothing has
            been accumulated yet.
        """
        matches = [
            i for i, f in enumerate(self.frequencies) if np.isclose(f, float(frequency))
        ]
        if not matches:
            msg = (
                f"frequency {frequency!r} Hz is not tracked by this probe "
                f"(tracked: {self.frequencies})"
            )
            raise ValueError(msg)
        if self.samples == 0:
            msg = "no samples accumulated yet; step the simulation first"
            raise ValueError(msg)
        scale = 2.0 / self.samples
        i = matches[0]
        return ContourPhasors(
            frequency=self.frequencies[i],
            positions=self.positions,
            normals=self.normals,
            pressure=np.asarray(scale * self._acc_p[i]),
            normal_velocity=np.asarray(scale * self._acc_v[i]),
            segment=self._dx,
        )


class FDTD2D:
    r"""2D acoustic FDTD stepping engine on a staggered grid.

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
    :param cfl: Courant number
        :math:`C_\mathrm{N} = c_{\max}\, \Delta t \sqrt{2} / \Delta x` (Eq. 4.13);
        the explicit scheme is stable for :math:`C_\mathrm{N} \le 1` (Eq. 4.14) and
        values
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
    :param damping: Bulk amplitude decay rate [1/s]: a scalar applied to
        the whole field (a simple stand-in for air/wall absorption;
        ``6.91 / T60`` gives a ``T60`` seconds reverberant decay) or an
        ``(ny, nx)`` map for locally lossy regions, e.g. an equivalent
        fluid modelling a porous sample (plane waves inside a uniform
        lossy region follow :math:`k = (\omega - j \sigma) / c` with the
        real characteristic impedance :math:`\rho c`).
    :param shape: Grid shape ``(ny, nx)``, required only when ``c`` is a
        scalar.
    :param edge_impedance: Locally reacting boundary sides: a mapping from
        side name to a real specific acoustic impedance [Pa s/m], either a
        scalar or a per-edge-cell 1D array (length ``ny`` for ``left``/
        ``right``, ``nx`` for ``top``/``bottom``). Implements Eqs.
        (4.33)-(4.35); :math:`Z = \rho c` is a normal-incidence matched
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
        damping: float | NDArray[np.float64] = 0.0,
        shape: tuple[int, int] | None = None,
        edge_impedance: Mapping[str, float | NDArray[np.float64]] | None = None,
        obstacle_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        c_map = _resolve_c_map(c, shape)
        if not np.isfinite(cfl) or not 0.0 < cfl < 1.0:
            msg = (
                "cfl must lie in (0, 1): the leapfrog scheme "
                "is unstable beyond the Courant bound CN = 1"
            )
            raise ValueError(msg)
        ny, nx = c_map.shape
        rho_map = _resolve_rho_map(rho, ny, nx)
        sponge_width, damping_map = _validate_sponge_damping(
            sponge_width, sponge_reflection, damping, ny, nx
        )

        self.dx = _positive_finite("dx", dx)
        self.c = c_map
        self.rho = rho_map
        c_max = float(c_map.max())
        self.dt = cfl * self.dx / (c_max * float(np.sqrt(2.0)))
        #: Mean sound speed, cached for the plane-wave machinery.
        self._c_ref = float(c_map.mean())
        self.kappa = rho_map * c_map**2  # bulk modulus at centres
        # Density averaged onto the faces where each velocity lives.
        self._rho_x = 0.5 * (rho_map[:, 1:] + rho_map[:, :-1])
        self._rho_y = 0.5 * (rho_map[1:, :] + rho_map[:-1, :])

        self.p: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self.vx: Field2D = np.zeros((ny, nx - 1), dtype=np.float64)
        self.vy: Field2D = np.zeros((ny - 1, nx), dtype=np.float64)
        self._div: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self._sources: list[Source] = []
        self._plane_sources: list[PlaneWaveSource] = []
        self._contour_probes: list[ContourProbe] = []
        self.n = 0  # completed steps

        sides = _resolve_sponge_sides(sponge_sides)
        #: Sponge-layer width in cells (read-only configuration record).
        self.sponge_width = int(sponge_width)
        #: Sides carrying a sponge layer when ``sponge_width > 0``.
        self.sponge_sides: tuple[str, ...] = sides if sponge_width > 0 else ()
        #: Per-side impedance boundaries as supplied (immutable record).
        self.edge_impedance: Mapping[str, float | NDArray[np.float64]] = (
            MappingProxyType(dict(edge_impedance) if edge_impedance else {})
        )
        self._init_decay(sides, sponge_width, sponge_reflection, damping_map, c_max)
        self._edges = self._build_edges(edge_impedance, sponge_width, sides, ny, nx)
        self._init_obstacle(obstacle_mask, ny, nx)

    def _init_decay(
        self,
        sides: tuple[str, ...],
        sponge_width: int,
        sponge_reflection: float,
        damping: NDArray[np.float64],
        c_max: float,
    ) -> None:
        """Precompute the sponge/damping decay factors of every field."""
        sigma = _sigma_map(
            self.p.shape,
            sides,
            sponge_width,
            sponge_reflection,
            damping,
            c_max,
            self.dx,
        )
        self._decay_p: Field2D = np.exp(-sigma * self.dt)
        self._decay_vx: Field2D = np.exp(
            -(0.5 * (sigma[:, 1:] + sigma[:, :-1])) * self.dt
        )
        self._decay_vy: Field2D = np.exp(
            -(0.5 * (sigma[1:, :] + sigma[:-1, :])) * self.dt
        )

    def _init_obstacle(
        self, obstacle_mask: NDArray[np.bool_] | None, ny: int, nx: int
    ) -> None:
        """Validate the obstacle mask into closed-face velocity factors."""
        self._obstacle: NDArray[np.bool_] | None = _validated_obstacle(
            obstacle_mask, ny, nx
        )
        self._vx_open: Field2D | None = None
        self._vy_open: Field2D | None = None
        if self._obstacle is not None:
            mask = self._obstacle
            # A face between two cells is open only when both are open
            # (rigid obstacle boundary: zero normal velocity, Eq. 4.32).
            self._vx_open = (~(mask[:, 1:] | mask[:, :-1])).astype(np.float64)
            self._vy_open = (~(mask[1:, :] | mask[:-1, :])).astype(np.float64)

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
            msg = f"unknown impedance sides: {sorted(unknown)}"
            raise ValueError(msg)
        absorbing = set(sponge_sides) if sponge_width > 0 else set()
        rho_edges = {
            "left": self.rho[:, 0],
            "right": self.rho[:, -1],
            "top": self.rho[0, :],
            "bottom": self.rho[-1, :],
        }
        for side in _SIDES:  # deterministic order
            if side not in edge_impedance:
                continue
            if side in absorbing:
                msg = (
                    f"side {side!r} cannot be both absorbing and an impedance boundary"
                )
                raise ValueError(msg)
            n_edge = ny if side in ("left", "right") else nx
            z = _edge_impedance_profile(side, edge_impedance[side], n_edge)
            edges.append(_ImpedanceEdge(side, z, rho_edges[side], self.dt, self.dx))
        return edges

    @property
    def time(self) -> float:
        """Elapsed simulated time [s]."""
        return self.n * self.dt

    def add_plane_wave(
        self,
        direction: str,
        *,
        center: float,
        width: float,
        amplitude: float = 1.0,
        wavelength: float | None = None,
    ) -> None:
        """Superimpose a one-way plane wave packet as an initial condition.

        A Gaussian envelope (optionally carrying a sine at ``wavelength``)
        is written onto the pressure field, and the leapfrog-consistent
        particle velocity is written a half time step back, so the packet
        propagates only toward ``direction`` (the axes of
        :meth:`plot_geometry`: ``"down"``/``"up"`` along y,
        ``"right"``/``"left"`` along x). The packet is uniform across the
        transverse direction and adds to whatever fields are present.

        Obstacles are not carved out of the initial condition: place the
        packet in free field (its envelope clear of ``obstacle_mask``
        cells), as a physical incident wave would be.

        :param direction: Travel direction, one of ``"down"``, ``"up"``,
            ``"left"``, ``"right"``.
        :param center: Envelope centre along the travel axis, in metres.
        :param width: Gaussian envelope width (the ``1/e`` half-width), in
            metres.
        :param amplitude: Peak pressure of the envelope, in pascals.
        :param wavelength: Optional carrier wavelength, in metres; ``None``
            gives the pure Gaussian pulse.
        :raises ValueError: For an unknown direction, a non-positive or
            non-finite ``width``/``wavelength``, or a non-finite
            ``center``/``amplitude``.
        """
        if direction not in _SIDE_TRAVEL:
            msg = "'direction' must be 'down', 'up', 'left' or 'right'."
            raise ValueError(msg)
        width = require_positive(width, "width")
        if wavelength is not None:
            wavelength = require_positive(wavelength, "wavelength")
        center = _finite("center", center)
        amplitude = _finite("amplitude", amplitude)

        def profile(coord: NDArray[np.floating]) -> NDArray[np.float64]:
            envelope = amplitude * np.exp(-(((coord - center) / width) ** 2))
            if wavelength is not None:
                envelope = envelope * np.sin(
                    2.0 * np.pi * (coord - center) / wavelength
                )
            return np.asarray(envelope, dtype=np.float64)

        ny, nx = self.p.shape
        axis_y = direction in ("down", "up")
        sign = 1.0 if direction in ("down", "right") else -1.0
        c_ref = self._c_ref
        if axis_y:
            centres = (np.arange(ny) + 0.5) * self.dx
            faces = np.arange(1, ny) * self.dx
            self.p += profile(centres)[:, np.newaxis]
            # v = sign * p / (rho c) for travel along +/- y, sampled on the
            # faces a half time step earlier: g(y -/+ c t) at t = -dt/2.
            v_prof = profile(faces + sign * 0.5 * c_ref * self.dt)
            rho_face = self._rho_y
            self.vy += sign * v_prof[:, np.newaxis] / (rho_face * c_ref)
        else:
            centres = (np.arange(nx) + 0.5) * self.dx
            faces = np.arange(1, nx) * self.dx
            self.p += profile(centres)[np.newaxis, :]
            v_prof = profile(faces + sign * 0.5 * c_ref * self.dt)
            rho_face = self._rho_x
            self.vx += sign * v_prof[np.newaxis, :] / (rho_face * c_ref)

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
        return plot_fdtd_domain(self, ax=ax, probes=probes, language=language, **kwargs)

    def add_source(self, source: AnySource) -> None:
        """Register a source: a point injection or a plane-wave line.

        Point sources (:class:`GaussianPulse`, :class:`CWSource`,
        :class:`SignalSource`) inject additively at one cell. A
        :class:`PlaneWaveSource` injects a sustained one-way plane wave on
        a full line of cells near its launch edge.
        """
        if isinstance(source, PlaneWaveSource):
            self._add_plane_source(source)
            return
        ny, nx = self.p.shape
        ix = _integer("source ix", source.ix)
        iy = _integer("source iy", source.iy)
        if not (0 <= ix < nx and 0 <= iy < ny):
            msg = "source position lies outside the grid"
            raise ValueError(msg)
        if self._obstacle is not None and self._obstacle[iy, ix]:
            msg = "source position lies inside an obstacle"
            raise ValueError(msg)
        self._sources.append(source)

    def add_contour_probe(
        self, ix0: int, ix1: int, iy0: int, iy1: int, *, frequencies: ArrayLike
    ) -> ContourProbe:
        r"""Record ``p`` and ``v_n`` phasors on a closed rectangular contour.

        The contour is the rectangle of cell faces enclosing the cell
        block ``ix0..ix1`` x ``iy0..iy1`` (both ends inclusive): its sides
        lie on :math:`x = i_{x0}\,\Delta x`, :math:`x = (i_{x1}+1)\,\Delta x`,
        :math:`y = i_{y0}\,\Delta x` and :math:`y = (i_{y1}+1)\,\Delta x`. From
        the step after registration the engine
        folds the face pressures (averaged from the two adjacent cell
        centres) and the outward face normal velocities into running DFT
        accumulators at each requested frequency; see :class:`ContourProbe`
        for the steady-state protocol and
        :func:`~phonometry.simulation.ntff.far_field_from_contour` for the
        far-field transformation of the captured phasors.

        Place the contour in open air: strictly around the scatterer,
        clear of sponge layers and of any source (point sources and
        plane-wave injection lines must stay outside so the enclosed
        region is source-free in a scattering run, or inside when the
        radiated field itself is the quantity of interest).

        :param ix0: First cell column inside the contour.
        :param ix1: Last cell column inside the contour.
        :param iy0: First cell row inside the contour.
        :param iy1: Last cell row inside the contour.
        :param frequencies: Frequencies to track [Hz].
        :return: The registered :class:`ContourProbe`.
        :raises ValueError: For a block without open faces on all sides or
            invalid frequencies.
        """
        probe = ContourProbe(self, ix0, ix1, iy0, iy1, frequencies)
        self._contour_probes.append(probe)
        return probe

    def _add_plane_source(self, source: PlaneWaveSource) -> None:
        """Validate and register a plane-wave injection line."""
        if source.direction not in _SIDE_TRAVEL:
            msg = "'direction' must be 'down', 'up', 'left' or 'right'."
            raise ValueError(msg)
        offset = _integer("plane-wave offset", source.offset)
        ny, nx = self.p.shape
        limit = ny if source.direction in ("down", "up") else nx
        if not 0 <= offset < limit - 1:
            msg = "plane-wave offset lies outside the grid"
            raise ValueError(msg)
        if self._obstacle is not None:
            if source.direction == "down":
                line = self._obstacle[offset, :]
            elif source.direction == "up":
                line = self._obstacle[ny - 1 - offset, :]
            elif source.direction == "right":
                line = self._obstacle[:, offset]
            else:
                line = self._obstacle[:, nx - 1 - offset]
            if bool(np.any(line)):
                msg = "plane-wave injection line crosses an obstacle"
                raise ValueError(msg)
        self._plane_sources.append(source)

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
        for plane in self._plane_sources:
            self._inject_plane(plane, t_next)
        self.p *= self._decay_p
        self.n += 1
        for probe in self._contour_probes:
            probe._accumulate(self.p, self.vx, self.vy, self.n)

    def _inject_plane(self, plane: PlaneWaveSource, t_next: float) -> None:
        r"""Add the incident plane wave on the injection line (one-way).

        Simultaneous pressure and adjacent-face velocity increments carry
        the incident wave into the domain only along the travel direction;
        the increments are the discrete equivalent of superimposing
        :math:`p = s(t - d/c)` entering through the line each step.
        """
        c_ref = self._c_ref
        gain = plane.amplitude * self.dt / (self.dx / c_ref)
        value_p = plane.waveform(t_next)
        value_v = plane.waveform(t_next - 0.5 * self.dt + 0.5 * self.dx / c_ref)
        direction = plane.direction
        k = plane.offset
        if direction == "down":
            self.p[k, :] += gain * value_p
            self.vy[k, :] += gain * value_v / (self._rho_y[k, :] * c_ref)
        elif direction == "up":
            row = self.p.shape[0] - 1 - k
            self.p[row, :] += gain * value_p
            self.vy[row - 1, :] -= gain * value_v / (self._rho_y[row - 1, :] * c_ref)
        elif direction == "right":
            self.p[:, k] += gain * value_p
            self.vx[:, k] += gain * value_v / (self._rho_x[:, k] * c_ref)
        else:
            col = self.p.shape[1] - 1 - k
            self.p[:, col] += gain * value_p
            self.vx[:, col - 1] -= gain * value_v / (self._rho_x[:, col - 1] * c_ref)

    def energy(self) -> float:
        """Total acoustic field energy [J per metre of depth]."""
        e_p = float(np.sum(self.p**2 / (2.0 * self.kappa)))
        e_v = 0.5 * (
            float(np.sum(self._rho_x * self.vx**2))
            + float(np.sum(self._rho_y * self.vy**2))
        )
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
        return _run_recording(self, steps, record_every, decimate)


def _require_grid_axes(
    owner: str,
    shape: tuple[int, int],
    snapshots: NDArray[np.float64] | None,
    obstacle_mask: NDArray[np.bool_] | None,
) -> None:
    """Require every whole-domain raster to cover the grid ``shape`` names.

    Shared by the acoustic and the elastic result, which record different
    fields over the same domain and hand them to the same snapshot renderer.
    That renderer takes the physical extent of the picture from ``shape``
    alone and draws the obstacle map on top of it as a second image over the
    same extent, so a raster recorded on a different grid is stretched to fit
    a domain it never covered, with the geometry landing on cells that were
    never simulated. Neither imshow nor the eye objects: the frame is a
    perfectly ordinary field plot of the wrong thing.

    :param owner: Name of the result type being checked, for the message.
    :param shape: The grid the run covered, ``(ny, nx)``.
    :param snapshots: Recorded frames, ``(n_frames, ny, nx)``, or ``None``.
    :param obstacle_mask: Rigid-cell map, ``(ny, nx)``, or ``None``.
    :raises ValueError: if a raster disagrees with ``shape``.
    """
    ny, nx = shape
    rows = {"shape (rows)": ny}
    columns = {"shape (columns)": nx}
    if snapshots is not None:
        rows["snapshots (axis 1)"] = snapshots.shape[1]
        columns["snapshots (axis 2)"] = snapshots.shape[2]
    if obstacle_mask is not None:
        rows["obstacle_mask"] = obstacle_mask.shape[0]
        columns["obstacle_mask (axis 1)"] = obstacle_mask.shape[1]
    require_equal_counts(owner, rows, "grid row")
    require_equal_counts(owner, columns, "grid column")


@dataclass(frozen=True)
class FDTDResult:
    r"""Frozen result of a :func:`fdtd_simulation` run.

    :ivar times: Time axis [s], length ``n_steps + 1`` (includes
        :math:`t = 0`).
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

    def __post_init__(self) -> None:
        """Reject a run whose probes, time axis and grid do not line up.

        Every reader of a run is a picture, and neither picture can defend
        itself. The probe plot pairs each row of ``pressures`` with the label
        built from the matching row of ``probe_positions``, so a positions
        array of another length surfaces only at drawing time, as a zip that
        reports two lengths and names neither array, an entire run after the
        mistake was made. The snapshot plot does not surface it at all: it
        stamps the frame across an extent taken from ``shape``, which turns a
        frame recorded on a different grid into a plausible field over a
        domain it never covered.

        :raises ValueError: if the probe, time-step, snapshot or grid axes
            disagree.
        """
        require_ranks(
            self,
            times=1,
            pressures=2,
            probes=2,
            probe_positions=2,
            snapshots=3,
            snapshot_times=1,
            obstacle_mask=2,
        )
        require_same_length(
            self, "pressures", "probes", "probe_positions", axis="probe"
        )
        require_same_length(
            self, ("probes", 1), ("probe_positions", 1), axis="coordinate"
        )
        require_same_length(self, "times", ("pressures", 1), axis="time step")
        require_same_length(self, "snapshots", "snapshot_times", axis="snapshot")
        _require_grid_axes(
            type(self).__name__, self.shape, self.snapshots, self.obstacle_mask
        )

    @property
    def size(self) -> tuple[float, float]:
        """Domain size ``(lx, ly)`` [m]."""
        ny, nx = self.shape
        return nx * self.dx, ny * self.dx

    def plot(
        self,
        ax: Axes | None = None,
        *,
        kind: str = "probes",
        frame: int = -1,
        language: str = "en",
        **kwargs: Any,
    ) -> Axes:
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

            return plot_fdtd_snapshot(
                self, ax=ax, frame=frame, language=language, **kwargs
            )
        msg = "kind must be 'probes' or 'snapshot'"
        raise ValueError(msg)


def _parse_boundaries(
    boundaries: str | Mapping[str, str | float | NDArray[np.float64]],
) -> tuple[tuple[str, ...], dict[str, float | NDArray[np.float64]]]:
    """Split the boundary spec into sponge sides and impedance sides."""
    spec: dict[str, str | float | NDArray[np.float64]]
    if isinstance(boundaries, str):
        spec = dict.fromkeys(_SIDES, boundaries)
    else:
        unknown = set(boundaries) - set(_SIDES)
        if unknown:
            msg = f"unknown boundary sides: {sorted(unknown)}"
            raise ValueError(msg)
        spec = {side: boundaries.get(side, "rigid") for side in _SIDES}
    absorbing: list[str] = []
    impedance: dict[str, float | NDArray[np.float64]] = {}
    for side in _SIDES:
        value = spec[side]
        if isinstance(value, str):
            if value not in _BOUNDARY_NAMES:
                msg = (
                    f"boundary for side {side!r} must be one of "
                    f"{_BOUNDARY_NAMES} or a real impedance, got {value!r}"
                )
                raise ValueError(msg)
            if value == "absorbing":
                absorbing.append(side)
        else:
            impedance[side] = value
    return tuple(absorbing), impedance


def _probe_indices(
    probes: Sequence[tuple[int, int]],
    shape: tuple[int, int],
    obstacle: NDArray[np.bool_] | None,
) -> NDArray[np.int_]:
    """Validate the probe cells into an ``(n_probes, 2)`` index array."""
    ny, nx = shape
    probe_ix = np.zeros((len(probes), 2), dtype=np.int_)
    for k, entry in enumerate(probes):
        try:
            ix, iy = entry
        except (TypeError, ValueError):
            msg = f"probes must hold (ix, iy) index pairs; entry {k} is {entry!r}"
            raise ValueError(msg) from None
        ix = _integer("probe ix", ix)
        iy = _integer("probe iy", iy)
        if not (0 <= ix < nx and 0 <= iy < ny):
            msg = f"probe ({ix}, {iy}) lies outside the grid"
            raise ValueError(msg)
        if obstacle is not None and obstacle[iy, ix]:
            msg = f"probe ({ix}, {iy}) lies inside an obstacle"
            raise ValueError(msg)
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
    boundaries: str | Mapping[str, str | float | NDArray[np.float64]] = "rigid",
    absorbing_layer_cells: int = 20,
    obstacle_mask: NDArray[np.bool_] | None = None,
    damping: float = 0.0,
    snapshot_every: int | None = None,
) -> FDTDResult:
    r"""Run a deterministic 2D acoustic FDTD simulation.

    Builds the staggered-grid domain (Attenborough & Van Renterghem 2021,
    Eqs. 4.11-4.12), applies the requested boundary conditions, injects the
    sources and integrates for ``duration`` seconds, recording the pressure
    at every probe each time step and, optionally, full-field snapshots.

    The grid covers ``(nx * dx, ny * dx)`` metres; a cell index ``(ix, iy)``
    maps to the physical cell centre ``((ix + 0.5) * dx, (iy + 0.5) * dx)``.
    Resolve at least 10 cells per shortest wavelength using the smallest
    sound speed of the domain (:math:`\Delta x \le c_{\min} / (10 f)`), the
    usual rule for
    this lowest-order scheme: the worst-case (on-axis) numerical dispersion
    error magnitude, :math:`(k \Delta x)^2 / 24` from the discrete counterpart
    of
    Eq. 4.15 (the modelled frequency under-reads, so the signed error is
    negative), is then about 1.6 % (about 1.4 % at the default ``cfl``;
    in a heterogeneous domain the slower cells run at a lower local Courant
    number and sit nearer the 1.6 % bound) and finer grids reduce it
    quadratically. The
    simulation is 2D, so a point source is
    physically a line source with cylindrical :math:`1/\sqrt{r}` amplitude
    spreading rather than the 3D spherical :math:`1/r`.

    :param c: Sound-speed map [m/s], shape ``(ny, nx)``, or a scalar with an
        explicit ``shape``.
    :param dx: Grid spacing [m] (square cells).
    :param duration: Physical time to simulate [s].
    :param sources: One or more of :class:`GaussianPulse`,
        :class:`CWSource` or :class:`SignalSource`.
    :param shape: Grid shape ``(ny, nx)``, required when ``c`` is a scalar.
    :param rho: Density map [kg/m3]; scalar or ``(ny, nx)`` array.
    :param cfl: Courant number in ``(0, 1)`` (Eqs. 4.13-4.14); the time step
        is :math:`\Delta t = C_\mathrm{N}\, \Delta x / (c_{\max} \sqrt{2})`.
        Default 0.6.
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
        many steps (and at :math:`t = 0`); ``None`` records none.
    :return: A :class:`FDTDResult`.
    :raises ValueError: If the inputs are invalid.
    """
    if len(sources) == 0:
        msg = "at least one source is required"
        raise ValueError(msg)
    duration = _positive_finite("duration", duration)
    if snapshot_every is not None:
        snapshot_every = _integer("snapshot_every", snapshot_every)
        if snapshot_every < 1:
            msg = "snapshot_every must be >= 1"
            raise ValueError(msg)
    absorbing_sides, edge_impedance = _parse_boundaries(boundaries)
    if absorbing_sides:
        absorbing_layer_cells = _integer("absorbing_layer_cells", absorbing_layer_cells)
        if absorbing_layer_cells < 1:
            msg = "absorbing_layer_cells must be >= 1"
            raise ValueError(msg)

    sim = FDTD2D(
        c,
        dx,
        rho=rho,
        cfl=cfl,
        sponge_width=absorbing_layer_cells if absorbing_sides else 0,
        sponge_sides=absorbing_sides or None,
        damping=damping,
        shape=shape,
        edge_impedance=edge_impedance or None,
        obstacle_mask=obstacle_mask,
    )
    for source in sources:
        sim.add_source(source)

    ny, nx = sim.p.shape
    probe_ix = _probe_indices(probes, (ny, nx), sim._obstacle)

    steps = round(duration / sim.dt)
    if steps < 1:
        msg = f"duration must cover at least one time step (dt = {sim.dt:.3e} s)"
        raise ValueError(msg)
    times = np.arange(steps + 1, dtype=np.float64) * sim.dt
    pressures, frames, frame_steps = _record_run(sim, steps, probe_ix, snapshot_every)

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
        snapshot_times=(
            np.asarray(frame_steps, dtype=np.float64) * sim.dt if frame_steps else None
        ),
        obstacle_mask=(sim._obstacle.copy() if sim._obstacle is not None else None),
    )
