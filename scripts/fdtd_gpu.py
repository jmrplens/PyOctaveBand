#  Copyright (c) 2026. Jose Manuel Requena Plens
"""GPU-portable 2D acoustic FDTD stepping engine for the animation renders.

A backend-agnostic replica of the leapfrog scheme of
``phonometry.simulation.fdtd.FDTD2D`` restricted to the feature subset the
documentation animations use: per-cell sound-speed and density maps, rigid
walls by construction, graded sponge layers, per-side real-impedance edges,
scalar or per-cell bulk damping, rasterised rigid obstacles and the one-way
plane-wave initial condition. The array module is injected (the ``xp``
parameter, NumPy or CuPy), so the same stepping code runs on the CPU or on
an NVIDIA GPU without any change.

The module is deliberately self-contained (NumPy is its only import) so it
can be copied as a single file into a bare CuPy container by the remote
runner (``fdtd_gpu_remote.py``); it must not import ``phonometry``.

Numerical contract: with ``xp=numpy`` every update expression is written in
the exact operation order of the library engine, so the pressure history is
bit-identical to ``FDTD2D`` for the supported subset (the parity test in
``tests/test_fdtd_gpu_parity.py`` locks this). All grid setup (validation,
time step, face densities, decay maps, sponge profiles) is computed in
float64 NumPy and only then transferred to the backend, so a GPU run departs
from the CPU run only through the floating-point reassociation of the
backend's elementwise kernels, at the ULP level.

Scheme reference: Attenborough & Van Renterghem, *Predicting Outdoor Sound*
(2nd ed., CRC Press 2021), chapter 4; see the library module docstring for
the equation-by-equation mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Field2D = NDArray[np.float64]
#: Backend array: ``numpy.ndarray`` or ``cupy.ndarray`` depending on ``xp``.
XPArray = Any

_SIDES = ("left", "right", "top", "bottom")
_SIDE_TRAVEL = ("down", "up", "left", "right")


def _positive_finite(name: str, value: float) -> float:
    """Validate that *value* is a strictly positive finite scalar."""
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _integer(name: str, value: int) -> int:
    """Validate that *value* is an integral scalar (bool is rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004 - ValueError keeps the module validation errors uniform
    return int(value)


def _resolve_cfl(cfl: float) -> float:
    """Validate the Courant number into a float strictly inside (0, 1)."""
    out = float(cfl)
    if not np.isfinite(out) or not 0.0 < out < 1.0:
        raise ValueError("cfl must lie in (0, 1)")
    return out


def _positive_map(name: str, field: Field2D) -> None:
    """Validate that every cell of *field* is strictly positive and finite."""
    if not np.all(np.isfinite(field)) or bool(np.any(field <= 0.0)):
        raise ValueError(f"{name} must be strictly positive and finite "
                         "everywhere")


def _sponge_profile(n: int, width: int, sides: tuple[bool, bool],
                    sigma_max: float) -> Field2D:
    """1D absorption rate sigma(i) [1/s]: quadratic ramp into each sponge side.

    ``sides`` selects (low-index side, high-index side). Identical to the
    library helper of the same name.
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


def _resolve_sponge(sponge_width: Any, sponge_reflection: float,
                    ny: int, nx: int) -> tuple[int, float]:
    """Validate the sponge configuration exactly as the library does."""
    width = _integer("sponge_width", sponge_width)
    if width < 0:
        raise ValueError("sponge_width must be non-negative")
    if width >= min(nx, ny):
        raise ValueError("sponge_width must be narrower than the "
                         "smallest grid side")
    if not 0.0 < sponge_reflection < 1.0:
        raise ValueError("sponge_reflection must lie strictly between "
                         "0 and 1")
    return width, float(sponge_reflection)


def _resolve_damping(damping: float | NDArray[np.float64],
                     ny: int, nx: int) -> NDArray[np.float64]:
    """Validate the damping spec into a non-negative 0D or ``(ny, nx)`` array."""
    damping_map = np.asarray(damping, dtype=np.float64)
    if damping_map.ndim not in (0, 2):
        raise ValueError("damping must be a scalar or an (ny, nx) map")
    if not np.all(np.isfinite(damping_map)) or np.any(damping_map < 0.0):
        raise ValueError("damping must be non-negative and finite")
    if damping_map.ndim == 2 and damping_map.shape != (ny, nx):
        raise ValueError(
            f"damping map shape {damping_map.shape} does not match the "
            f"grid {(ny, nx)}")
    return damping_map


def _resolve_sponge_sides(sponge_sides: Any) -> tuple[str, ...]:
    """Normalise the sponge-side spec into a validated tuple of side names."""
    if sponge_sides is None:
        sides: tuple[str, ...] = _SIDES
    elif isinstance(sponge_sides, str):
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


def _resolve_edge_impedance(
    edge_impedance: dict[str, float | NDArray[np.float64]] | None,
    sponge_width: int,
    sponge_sides: tuple[str, ...],
    ny: int,
    nx: int,
) -> dict[str, Field2D]:
    """Validate the per-side impedance spec into 1D per-edge-cell profiles.

    Rejects unknown side names and sides that are absorbing (sponge) at the
    same time; scalars are broadcast to the edge length. The returned dict
    iterates in the deterministic ``_SIDES`` order.
    """
    if not edge_impedance:
        return {}
    unknown = set(edge_impedance) - set(_SIDES)
    if unknown:
        raise ValueError(f"unknown impedance sides: {sorted(unknown)}")
    absorbing = set(sponge_sides) if sponge_width > 0 else set()
    profiles: dict[str, Field2D] = {}
    for side in _SIDES:                          # deterministic order
        if side not in edge_impedance:
            continue
        if side in absorbing:
            raise ValueError(f"side {side!r} cannot be both absorbing "
                             "and an impedance boundary")
        n_edge = ny if side in ("left", "right") else nx
        profiles[side] = _edge_impedance_profile(side, edge_impedance[side],
                                                 n_edge)
    return profiles


def _resolve_obstacle_mask(obstacle_mask: NDArray[np.bool_] | None,
                           ny: int, nx: int) -> NDArray[np.bool_] | None:
    """Validate the obstacle spec into a boolean ``(ny, nx)`` map (or None)."""
    if obstacle_mask is None:
        return None
    mask = np.asarray(obstacle_mask)
    if mask.shape != (ny, nx):
        raise ValueError("obstacle_mask must match the grid shape")
    if mask.dtype != np.bool_:
        raise ValueError("obstacle_mask must be a boolean array")
    if bool(mask.all()):
        raise ValueError("obstacle_mask must leave open cells")
    return mask


class _ImpedanceEdge:
    """One locally reacting boundary side with a real specific impedance.

    Backend-portable replica of the library ``_ImpedanceEdge``: coefficients
    ``c1 = (1 - a) / (1 + a)`` and ``c2 = (2 dt / (rho dx)) / (1 + a)`` with
    ``a = dt Z / (rho dx)``, computed in NumPy and transferred to ``xp``.
    """

    __slots__ = ("c1", "c2", "side", "vb")

    def __init__(self, xp: Any, side: str, impedance: Field2D,
                 rho_edge: Field2D, dt: float, dx: float) -> None:
        self.side = side
        a = dt * impedance / (rho_edge * dx)
        self.c1: XPArray = xp.asarray((1.0 - a) / (1.0 + a))
        self.c2: XPArray = xp.asarray((2.0 * dt / (rho_edge * dx)) / (1.0 + a))
        self.vb: XPArray = xp.zeros(impedance.shape, dtype=np.float64)

    def update(self, p: XPArray) -> None:
        """Advance the boundary-face velocity by one leapfrog step."""
        if self.side == "left":
            self.vb = self.c1 * self.vb - self.c2 * p[:, 0]
        elif self.side == "right":
            self.vb = self.c1 * self.vb + self.c2 * p[:, -1]
        elif self.side == "top":
            self.vb = self.c1 * self.vb - self.c2 * p[0, :]
        else:                                          # bottom
            self.vb = self.c1 * self.vb + self.c2 * p[-1, :]

    def add_flux(self, div: XPArray) -> None:
        """Add the boundary-face flux to the velocity divergence."""
        if self.side == "left":
            div[:, 0] -= self.vb
        elif self.side == "right":
            div[:, -1] += self.vb
        elif self.side == "top":
            div[0, :] -= self.vb
        else:                                          # bottom
            div[-1, :] += self.vb


class GpuFDTD2D:
    """2D acoustic FDTD stepping engine with an injectable array backend.

    Same staggered grid as the library engine: pressure ``p`` at cell
    centres, shape ``(ny, nx)``; ``vx`` at interior x-faces ``(ny, nx - 1)``;
    ``vy`` at interior y-faces ``(ny - 1, nx)``. The domain boundary is
    rigid by construction; sponges and impedance edges soften selected
    sides. Time step: ``dt = cfl * dx / (c_max * sqrt(2))``.

    :param c: Sound-speed map [m/s], shape ``(ny, nx)``, or a scalar with
        an explicit ``shape``.
    :param dx: Grid spacing [m] (square cells).
    :param xp: Array module: ``numpy`` (default) or ``cupy``. All state
        arrays (``p``, ``vx``, ``vy``, decay maps) live on this backend.
    :param rho: Density map [kg/m3]; scalar or ``(ny, nx)`` array.
    :param cfl: Courant number in ``(0, 1)``; default 0.6 as the library.
    :param sponge_width: Absorbing-layer thickness in cells (0 = none).
    :param sponge_sides: Side name or iterable of side names
        (default: all four when ``sponge_width > 0``).
    :param sponge_reflection: Target round-trip amplitude reflection.
    :param damping: Bulk amplitude decay rate [1/s], scalar or map.
    :param shape: Grid shape ``(ny, nx)``, required when ``c`` is scalar.
    :param edge_impedance: Mapping side name -> real specific impedance
        [Pa s/m], scalar or per-edge-cell 1D array.
    :param obstacle_mask: Boolean ``(ny, nx)`` map of rigid cells.
    """

    def __init__(
        self,
        c: float | Field2D,
        dx: float,
        *,
        xp: Any = np,
        rho: float | Field2D = 1.2,
        cfl: float = 0.6,
        sponge_width: int = 0,
        sponge_sides: Any = None,
        sponge_reflection: float = 1e-4,
        damping: float | NDArray[np.float64] = 0.0,
        shape: tuple[int, int] | None = None,
        edge_impedance: dict[str, float | NDArray[np.float64]] | None = None,
        obstacle_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        c_map = _resolve_c_map(c, shape)
        cfl = _resolve_cfl(cfl)
        ny, nx = c_map.shape
        rho_map = _resolve_rho_map(rho, ny, nx)
        sponge_width, sponge_reflection = _resolve_sponge(
            sponge_width, sponge_reflection, ny, nx)
        damping_map = _resolve_damping(damping, ny, nx)

        self._xp = xp
        self.dx = _positive_finite("dx", dx)
        c_max = float(c_map.max())
        #: Time step [s]: ``cfl * dx / (c_max * sqrt(2))``, as the library.
        self.dt = cfl * self.dx / (c_max * float(np.sqrt(2.0)))
        self._c_ref = float(c_map.mean())
        kappa = rho_map * c_map**2               # bulk modulus at centres
        rho_x = 0.5 * (rho_map[:, 1:] + rho_map[:, :-1])
        rho_y = 0.5 * (rho_map[1:, :] + rho_map[:-1, :])

        self.p: XPArray = xp.zeros((ny, nx), dtype=np.float64)
        self.vx: XPArray = xp.zeros((ny, nx - 1), dtype=np.float64)
        self.vy: XPArray = xp.zeros((ny - 1, nx), dtype=np.float64)
        self._div: XPArray = xp.zeros((ny, nx), dtype=np.float64)
        self.kappa: XPArray = xp.asarray(kappa)
        self._rho_x: XPArray = xp.asarray(rho_x)
        self._rho_x_np = rho_x                   # host copy for plane waves
        self._rho_y: XPArray = xp.asarray(rho_y)
        self._rho_y_np = rho_y
        self.n = 0                               # completed steps

        sides = _resolve_sponge_sides(sponge_sides)
        sigma_max = 0.0
        if sponge_width > 0:
            sigma_max = (-3.0 * c_max * float(np.log(sponge_reflection))
                         / (2.0 * sponge_width * self.dx))
        sig_x = _sponge_profile(nx, sponge_width,
                                ("left" in sides, "right" in sides), sigma_max)
        sig_y = _sponge_profile(ny, sponge_width,
                                ("top" in sides, "bottom" in sides), sigma_max)
        sigma = sig_x[np.newaxis, :] + sig_y[:, np.newaxis] + damping_map
        self._decay_p: XPArray = xp.asarray(np.exp(-sigma * self.dt))
        self._decay_vx: XPArray = xp.asarray(np.exp(
            -(0.5 * (sigma[:, 1:] + sigma[:, :-1])) * self.dt))
        self._decay_vy: XPArray = xp.asarray(np.exp(
            -(0.5 * (sigma[1:, :] + sigma[:-1, :])) * self.dt))

        self._edges = self._build_edges(edge_impedance, sponge_width, sides,
                                        rho_map, ny, nx)
        self._init_obstacle(obstacle_mask, ny, nx)

    def _build_edges(
        self,
        edge_impedance: dict[str, float | NDArray[np.float64]] | None,
        sponge_width: int,
        sponge_sides: tuple[str, ...],
        rho_map: Field2D,
        ny: int,
        nx: int,
    ) -> list[_ImpedanceEdge]:
        """Validate the per-side impedance spec into edge updaters."""
        profiles = _resolve_edge_impedance(edge_impedance, sponge_width,
                                           sponge_sides, ny, nx)
        rho_edges = {"left": rho_map[:, 0], "right": rho_map[:, -1],
                     "top": rho_map[0, :], "bottom": rho_map[-1, :]}
        return [_ImpedanceEdge(self._xp, side, z, rho_edges[side],
                               self.dt, self.dx)
                for side, z in profiles.items()]

    def _init_obstacle(self, obstacle_mask: NDArray[np.bool_] | None,
                       ny: int, nx: int) -> None:
        """Validate the obstacle mask into closed-face velocity factors."""
        self._vx_open: XPArray | None = None
        self._vy_open: XPArray | None = None
        mask = _resolve_obstacle_mask(obstacle_mask, ny, nx)
        if mask is not None and bool(mask.any()):
            xp = self._xp
            self._vx_open = xp.asarray(
                (~(mask[:, 1:] | mask[:, :-1])).astype(np.float64))
            self._vy_open = xp.asarray(
                (~(mask[1:, :] | mask[:-1, :])).astype(np.float64))

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

        Replica of ``FDTD2D.add_plane_wave``: a Gaussian envelope
        (optionally carrying a sine at ``wavelength``) written onto the
        pressure, with the leapfrog-consistent particle velocity written a
        half time step back (``profile(faces + sign * 0.5 * c_ref * dt)``,
        divided by ``rho_face * c_ref`` with ``c_ref = mean(c)``), so the
        packet propagates only toward ``direction``. Profiles are computed
        in NumPy and added on the backend.

        :param direction: ``"down"``, ``"up"``, ``"left"`` or ``"right"``.
        :param center: Envelope centre along the travel axis [m].
        :param width: Gaussian ``1/e`` half-width [m].
        :param amplitude: Peak pressure of the envelope [Pa].
        :param wavelength: Optional carrier wavelength [m].
        """
        if direction not in _SIDE_TRAVEL:
            raise ValueError(
                "'direction' must be 'down', 'up', 'left' or 'right'."
            )
        if width <= 0.0:
            raise ValueError("'width' must be positive.")
        if wavelength is not None and wavelength <= 0.0:
            raise ValueError("'wavelength' must be positive.")

        def profile(coord: NDArray[np.floating]) -> Field2D:
            envelope = amplitude * np.exp(-(((coord - center) / width) ** 2))
            if wavelength is not None:
                envelope = envelope * np.sin(
                    2.0 * np.pi * (coord - center) / wavelength
                )
            return np.asarray(envelope, dtype=np.float64)

        xp = self._xp
        ny, nx = self.p.shape
        axis_y = direction in ("down", "up")
        sign = 1.0 if direction in ("down", "right") else -1.0
        c_ref = self._c_ref
        if axis_y:
            centres = (np.arange(ny) + 0.5) * self.dx
            faces = np.arange(1, ny) * self.dx
            self.p += xp.asarray(profile(centres)[:, np.newaxis])
            v_prof = profile(faces + sign * 0.5 * c_ref * self.dt)
            v_add = sign * v_prof[:, np.newaxis] / (self._rho_y_np * c_ref)
            self.vy += xp.asarray(v_add)
        else:
            centres = (np.arange(nx) + 0.5) * self.dx
            faces = np.arange(1, nx) * self.dx
            self.p += xp.asarray(profile(centres)[np.newaxis, :])
            v_prof = profile(faces + sign * 0.5 * c_ref * self.dt)
            v_add = sign * v_prof[np.newaxis, :] / (self._rho_x_np * c_ref)
            self.vx += xp.asarray(v_add)

    def step(self) -> None:
        """Advance the leapfrog scheme by one time step.

        Update order is identical to the library engine: velocity from the
        pressure gradient, obstacle face closure, velocity decay, impedance
        edge update from the pre-step pressure, divergence scatter plus
        edge flux, pressure update, pressure decay.
        """
        dt_dx = self.dt / self.dx
        self.vx -= dt_dx / self._rho_x * (self.p[:, 1:] - self.p[:, :-1])
        self.vy -= dt_dx / self._rho_y * (self.p[1:, :] - self.p[:-1, :])
        if self._vx_open is not None and self._vy_open is not None:
            self.vx *= self._vx_open
            self.vy *= self._vy_open
        self.vx *= self._decay_vx
        self.vy *= self._decay_vy
        for edge in self._edges:
            edge.update(self.p)
        div = self._div
        div.fill(0.0)
        div[:, :-1] += self.vx
        div[:, 1:] -= self.vx
        div[:-1, :] += self.vy
        div[1:, :] -= self.vy
        for edge in self._edges:
            edge.add_flux(div)
        self.p -= self.kappa * dt_dx * div
        self.p *= self._decay_p
        self.n += 1

    def pressure(self) -> Field2D:
        """The current pressure field as a host (NumPy) array copy."""
        if hasattr(self.p, "get"):               # CuPy device array
            return np.asarray(self.p.get(), dtype=np.float64)
        return np.array(self.p, dtype=np.float64)
