#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
2D elastic finite-difference time-domain (P-SV) simulation.

A staggered-grid velocity-stress leapfrog solver for the 2D elastodynamic
equations in an isotropic linear medium, following the reference formulation
of Virieux, "P-SV wave propagation in heterogeneous media: velocity-stress
finite-difference method", *Geophysics* 51(4), 889-901 (1986):

* the governing first-order velocity-stress system in ``(vx, vy)`` and
  ``(txx, tyy, txy)`` (Eq. 2), which handles heterogeneous media without any
  explicit interface treatment;
* the fully staggered discretisation of Fig. 1 / Eq. 5, second order in
  space and time, arranged here so that the normal stresses share the cell
  centres of the acoustic solver (:class:`~phonometry.simulation.FDTD2D`)
  and the velocities live on the cell faces;
* the Courant stability condition
  :math:`c_\mathrm{P} \Delta t \sqrt{1/\Delta x^2 + 1/\Delta y^2} < 1`
  (Eqs. 6-7), which depends only on the P-wave speed, and the numerical
  dispersion relations of Eqs. 13-14 with the 10-cells-per-wavelength rule;
* liquids as the ``c_s = 0`` limit: shear-free cells propagate the acoustic
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
:func:`elastic_fdtd_simulation` builds the grid, runs a deterministic
simulation and returns a frozen :class:`ElasticFDTDResult` with per-probe
signal histories, optional field snapshots and a ``.plot()`` method.
:class:`ElasticFDTD2D` is the underlying stepping engine for callers that
need frame-by-frame access to the five field arrays.

The solver is deliberately deterministic: float64 arithmetic throughout, no
random numbers and single-threaded numpy execution, so identical inputs give
bit-identical outputs on the same platform.

Validated against analytic oracles: P- and S-wave times of flight
:math:`c_\mathrm{P} = \sqrt{(\lambda + 2\mu) / \rho}` and
:math:`c_\mathrm{S} = \sqrt{\mu / \rho}`, the
Rayleigh-wave speed from the exact characteristic equation (Cremer, Heckl &
Petersson 2005, Eq. 3.149), the Kirchhoff thin-plate flexural dispersion
:math:`c_\mathrm{B} = (B'/m'')^{1/4} \sqrt{\omega}` in its :math:`\lambda_\mathrm{B} \gg h`
domain
(Eqs. 3.83-3.89), the normal-incidence fluid-solid reflection coefficient
:math:`(Z_2 - Z_1)/(Z_2 + Z_1)`, the normal-incidence mass law of a thin
immersed
panel, and the exact reduction to the acoustic solver when ``c_s = 0``
everywhere. The fluid-solid coupling is further pinned to the oblique
plane-wave reflection coefficient of Brekhovskikh & Godin, *Acoustics of
Layered Media I* (Springer 1990), Eqs. 4.2.22-4.2.26 (with the shear-wave
mode conversion active), to the Scholte interface-wave speed from the exact
characteristic equation of Eq. 4.4.20 (see :func:`scholte_speed`), and to
the exact three-media transmission of an immersed plate (B&G Eqs. 2.4.10,
2.4.14) including its first thickness resonance :math:`f_1 = c_\mathrm{P} / (2 h)`
(Eq. 2.4.19), following the fluid-solid finite-difference benchmark of
van Vossen, Robertsson & Chapman, *Geophysics* 67(2), 618-624 (2002).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .fdtd import (
    _SIDES,
    Field2D,
    _finite,
    _integer,
    _positive_finite,
    _positive_map,
    _probe_indices,
    _resolve_sponge_sides,
    _run_recording,
    _sigma_map,
    _validate_sponge_damping,
    _validated_obstacle,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Boundary-condition names accepted by :func:`elastic_fdtd_simulation`.
_BOUNDARY_NAMES = ("rigid", "absorbing", "free")

#: Force directions accepted by :class:`ForceSource`.
_DIRECTIONS = ("x", "y")

#: Fields a probe or a snapshot can record.
_FIELD_NAMES = ("p", "vx", "vy")


@dataclass(frozen=True)
class ExplosionSource:
    r"""An isotropic (explosive) stress injection at one cell centre.

    Virieux (1986) drives an explosion with equal increments on both normal
    stresses at their shared node, which avoids the infinite amplitudes of a
    velocity-node source. The sign convention here is pressure-like: the
    waveform value is the injected compression, added as ``-s(t)`` to both
    ``txx`` and ``tyy`` so the synthetic pressure
    :math:`p = -(t_{xx} + t_{yy})/2` increases with a positive waveform (and
    a ``c_s = 0`` run reproduces the acoustic solver driven by the same
    waveform, sample by sample).

    ``waveform`` maps time in seconds to the injected pressure in pascals:
    any callable, typically the ``value`` method of a
    :class:`~phonometry.simulation.GaussianPulse`,
    :class:`~phonometry.simulation.CWSource` or
    :class:`~phonometry.simulation.SignalSource` (their own ``ix``/``iy``
    are not used here).

    :ivar ix: Source column (x) index; the cell centre is at
        :math:`x = (i_x + 0.5)\,\Delta x`.
    :ivar iy: Source row (y) index.
    :ivar waveform: Callable ``t -> s(t)`` in pascals.
    :ivar amplitude: Extra gain applied to ``waveform``.
    """

    ix: int
    iy: int
    waveform: Callable[[float], float]
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        _finite("amplitude", self.amplitude)


@dataclass(frozen=True)
class ForceSource:
    """A directional body-force injection at one velocity node.

    The standard body-force term of the equation of motion (the ``f`` of the
    Moczo et al. 2007 formulation): each step the target velocity component
    receives ``dt * f(t) / (rho * dx**2)``, i.e. ``waveform`` is a line
    force per unit depth [N/m] spread over one cell. A vertical force just
    below a free surface reproduces Lamb's problem and launches the Rayleigh
    wave (Virieux 1986, Fig. 5).

    ``ix``/``iy`` index the staggered velocity array directly: for
    ``direction = "x"`` the x-face between cells ``ix`` and ``ix + 1``
    (position ``((ix + 1) * dx, (iy + 0.5) * dx)``), for
    ``direction = "y"`` the y-face between rows ``iy`` and ``iy + 1``
    (position ``((ix + 0.5) * dx, (iy + 1) * dx)``).

    :ivar ix: Source column index into the velocity array.
    :ivar iy: Source row index into the velocity array.
    :ivar direction: ``"x"`` (horizontal) or ``"y"`` (vertical, positive
        downward in the ``imshow`` axes).
    :ivar waveform: Callable ``t -> f(t)`` in newtons per metre of depth.
    :ivar amplitude: Extra gain applied to ``waveform``.
    """

    ix: int
    iy: int
    direction: str
    waveform: Callable[[float], float]
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        _finite("amplitude", self.amplitude)
        if self.direction not in _DIRECTIONS:
            raise ValueError("direction must be 'x' or 'y'")


ElasticSource = ExplosionSource | ForceSource


@dataclass(frozen=True)
class Material:
    r"""An isotropic elastic medium as measurable wave speeds and density.

    The three numbers the solver's material maps are built from: the
    compressional speed :math:`c_\mathrm{P} = \sqrt{(\lambda + 2\mu) / \rho}`,
    the shear
    speed :math:`c_\mathrm{S} = \sqrt{\mu / \rho}` and the density ``rho``.
    ``c_s = 0``
    marks a fluid (the acoustic ``mu = 0`` limit of the elastic scheme,
    Virieux 1986), so the same dataclass names both fluids and solids.
    Every material must satisfy :math:`c_\mathrm{P}^2 \ge 2 c_\mathrm{S}^2`
    (non-negative first
    Lame parameter), the constructor bound of :class:`ElasticFDTD2D`.

    The module constants :data:`AIR`, :data:`WATER`, :data:`STEEL`,
    :data:`ALUMINIUM` and :data:`CONCRETE` carry the nominal round-number
    properties used throughout the documentation and the validation suite,
    mirroring the documented default media of the acoustic solver.

    :ivar c_p: Compressional (P) wave speed [m/s], strictly positive.
    :ivar c_s: Shear (S) wave speed [m/s], non-negative; 0 marks a fluid.
    :ivar rho: Density [kg/m3], strictly positive.
    """

    c_p: float
    c_s: float
    rho: float

    def __post_init__(self) -> None:
        c_p = _positive_finite("c_p", self.c_p)
        c_s = _finite("c_s", self.c_s)
        if c_s < 0.0:
            raise ValueError("c_s must be non-negative (0 marks a fluid)")
        rho = _positive_finite("rho", self.rho)
        if c_p**2 < 2.0 * c_s**2 * (1.0 - 1e-9):
            raise ValueError("c_p**2 must be at least 2 * c_s**2 "
                             "(non-negative lambda)")
        object.__setattr__(self, "c_p", c_p)
        object.__setattr__(self, "c_s", c_s)
        object.__setattr__(self, "rho", rho)

    @property
    def is_fluid(self) -> bool:
        """``True`` when the material carries no shear (``c_s = 0``)."""
        return self.c_s <= 0.0


#: Air at room conditions (fluid), the acoustic solver's default medium.
AIR = Material(c_p=343.0, c_s=0.0, rho=1.2)

#: Fresh water at room temperature (fluid).
WATER = Material(c_p=1480.0, c_s=0.0, rho=1000.0)

#: Structural steel (nominal bulk wave speeds).
STEEL = Material(c_p=5900.0, c_s=3200.0, rho=7850.0)

#: Aluminium (nominal bulk wave speeds).
ALUMINIUM = Material(c_p=6320.0, c_s=3130.0, rho=2700.0)

#: Dense concrete (nominal bulk wave speeds).
CONCRETE = Material(c_p=3800.0, c_s=2250.0, rho=2400.0)


def _as_material(name: str, value: object) -> Material:
    """Coerce a :class:`Material` or a ``(c_p, c_s, rho)`` triple."""
    if isinstance(value, Material):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str) \
            and len(value) == 3:
        c_p, c_s, rho = (float(np.real(v)) for v in value)
        return Material(c_p=c_p, c_s=c_s, rho=rho)
    raise ValueError(f"{name} must be a Material or a (c_p, c_s, rho) "
                     "triple")


def scholte_speed(
    fluid: Material | tuple[float, float, float],
    solid: Material | tuple[float, float, float],
) -> float:
    r"""Exact Scholte-wave speed of a fluid over an elastic half-space [m/s].

    The Scholte wave is the true interface wave of a fluid-solid contact:
    evanescent on both sides, elliptical particle motion, no low-frequency
    cut-off, and non-dispersive over homogeneous half-spaces (Jensen,
    Kuperman, Porter & Schmidt, *Computational Ocean Acoustics* 2e,
    Sections 4.5.2 and 8.5.4). Its speed ``v`` lies below both the fluid
    sound speed and the solid shear speed, and solves the exact
    characteristic equation of Brekhovskikh & Godin, *Acoustics of Layered
    Media I* (1990), Eq. 4.4.20, written in the notation of Eq. 4.4.18
    (:math:`q = c_\mathrm{S}^2/c_\mathrm{P}^2`, :math:`r = c_\mathrm{S}^2/c^2`,
    :math:`s = v^2/c_\mathrm{S}^2`,
    :math:`m = \rho_{\mathrm{solid}}/\rho_{\mathrm{fluid}}`):

    .. math::

       4 \sqrt{1-s} \sqrt{1-qs} - (2-s)^2
       = \frac{s^2}{m} \sqrt{\frac{1-sq}{1-sr}}

    A root always exists (B&G Section 4.4.3); the :math:`m \to \infty`
    limit of the left side is the Rayleigh equation. For stiff beds the
    root hugs the fluid speed (water over steel: 1479.6 m/s, 0.027 % below
    ``c``) while for soft sediments it drops well below it, which is why
    measured seabed interface waves probe the sediment shear speed
    (:math:`v \approx 0.85 c_\mathrm{S}` rule, Jensen et al. Section 5.10.5).

    :param fluid: Fluid half-space: a :class:`Material` with ``c_s = 0``
        (or a ``(c_p, 0.0, rho)`` triple).
    :param solid: Elastic half-space: a :class:`Material` with ``c_s > 0``.
    :return: The Scholte-wave phase speed [m/s].
    :raises ValueError: If ``fluid`` carries shear or ``solid`` does not.
    """
    flu = _as_material("fluid", fluid)
    sol = _as_material("solid", solid)
    if not flu.is_fluid:
        raise ValueError("fluid must have c_s = 0")
    if sol.is_fluid:
        raise ValueError("solid must have c_s > 0")
    q = sol.c_s**2 / sol.c_p**2
    r = sol.c_s**2 / flu.c_p**2
    m = sol.rho / flu.rho

    def characteristic(s: float) -> float:
        # B&G Eq. 4.4.20 rearranged to LHS - RHS; real and single-signed
        # branches over 0 < s < min(1, 1/r), where v < min(c, c_S).
        return float(4.0 * np.sqrt(1.0 - s) * np.sqrt(1.0 - q * s)
                     - (2.0 - s) ** 2
                     - (s**2 / m) * np.sqrt((1.0 - s * q) / (1.0 - s * r)))

    # The function is positive as s -> 0+ (expansion 2 s (1 - q) + O(s^2))
    # and diverges to -inf at the upper end, so a sign sweep brackets every
    # root; the largest one is the Scholte root.
    s_hi = min(1.0, 1.0 / r) * (1.0 - 1e-12)
    grid = np.linspace(1e-9, s_hi, 4001)
    values = np.array([characteristic(s) for s in grid])
    signs = np.sign(values)
    crossings = np.nonzero(np.diff(signs) != 0)[0]
    if crossings.size == 0:     # pragma: no cover - a root always exists
        raise ValueError("no Scholte root found; check the material pair")
    k = int(crossings[-1])
    root = float(brentq(characteristic, grid[k], grid[k + 1], xtol=1e-15,
                        rtol=8.9e-16))
    return sol.c_s * float(np.sqrt(root))


def _resolve_speed_map(name: str, value: float | Field2D,
                       shape: tuple[int, int] | None) -> Field2D:
    """Broadcast/validate a wave-speed spec into a finite 2D map."""
    if np.isscalar(value):
        if shape is None:
            raise ValueError(f"shape is required when {name} is a scalar")
        out = np.full(shape, float(np.real(value)), dtype=np.float64)
    else:
        out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D (ny, nx) map")
    return out


def _resolve_free_sides(
    free_sides: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalise the free-surface side spec into a validated tuple."""
    if free_sides is None:
        return ()
    if isinstance(free_sides, str):
        # A bare string would iterate per character; treat it as one side.
        sides: tuple[str, ...] = (free_sides,)
    else:
        sides = tuple(free_sides)
    unknown = set(sides) - set(_SIDES)
    if unknown:
        raise ValueError(f"unknown free sides: {sorted(unknown)}")
    return sides


class ElasticFDTD2D:
    r"""2D elastic (P-SV) FDTD stepping engine on a staggered grid.

    The Virieux (1986) cell is shifted half a cell so it lands on the layout
    of the acoustic :class:`~phonometry.simulation.FDTD2D`: the normal
    stresses ``txx`` and ``tyy`` share the cell centres, shape ``(ny, nx)``
    (row = y, column = x, the ``imshow`` convention; Virieux's downward z is
    the growing row here); ``vx`` lives at interior x-faces, shape
    ``(ny, nx - 1)``; ``vy`` at interior y-faces, shape ``(ny - 1, nx)``;
    the shear stress ``txy`` at interior corners, shape ``(ny - 1, nx - 1)``.
    Because only interior faces are stored, the domain boundary has zero
    normal velocity by construction and, with the corner shear stress taken
    as zero on the edge, is a shear-free rigid wall; ``free_sides`` turns
    selected sides into traction-free surfaces via stress imaging and sponge
    layers into absorbing ones. Material maps are given as the measurable
    wave speeds and converted internally to the Lame parameters
    :math:`\mu = \rho c_\mathrm{S}^2` and
    :math:`\lambda = \rho (c_\mathrm{P}^2 - 2 c_\mathrm{S}^2)`;
    density is
    arithmetically averaged onto the faces and ``mu`` harmonically averaged
    onto the corners (zero whenever any neighbour is a fluid), the Moczo
    et al. (2007) effective parameters (Eqs. 7.37-7.39) that make internal
    interfaces converge to the physical traction continuity.

    :param c_p: P-wave speed map [m/s], shape ``(ny, nx)``. A scalar with an
        explicit ``shape`` is also accepted.
    :param c_s: S-wave speed map [m/s]; scalar or ``(ny, nx)`` array.
        ``c_s = 0`` marks a fluid cell (the acoustic limit); every cell must
        satisfy :math:`c_\mathrm{P}^2 \ge 2 c_\mathrm{S}^2` (non-negative
        ``lambda``).
    :param dx: Grid spacing [m] (square cells).
    :param rho: Density map [kg/m3]; scalar or ``(ny, nx)`` array.
    :param cfl: Courant number
        :math:`C_\mathrm{N} = c_{\mathrm{p},\max}\, \Delta t \sqrt{2} / \Delta x`; the scheme
        is stable for :math:`C_\mathrm{N} < 1` (Virieux Eqs. 6-7, a bound on ``c_P``
        alone, independent of ``c_S`` and of the Poisson ratio) and values
        in ``(0, 1)`` are accepted. The default 0.6 keeps a wide stability
        margin with moderate numerical dispersion.
    :param sponge_width: Thickness of the absorbing layer in cells
        (0 = no absorbing sides).
    :param sponge_sides: Which sides absorb: a single side name or an
        iterable drawn from ``{"left", "right", "top", "bottom"}``
        (default: all four when ``sponge_width > 0``).
    :param sponge_reflection: Target round-trip amplitude reflection of the
        sponge layer; sets the peak absorption rate.
    :param damping: Bulk amplitude decay rate [1/s]: a scalar for the whole
        field or an ``(ny, nx)`` map for locally lossy regions.
    :param shape: Grid shape ``(ny, nx)``, required only when ``c_p`` is a
        scalar.
    :param free_sides: Sides carrying a traction-free surface (stress
        imaging, Moczo et al. Eq. 9.9): the surface plane runs through the
        boundary row/column of cell centres, the normal stress is pinned to
        zero there and the shear stress is imaged antisymmetrically above
        it. A side cannot be both free and absorbing.
    :param obstacle_mask: Boolean map, shape ``(ny, nx)``, of rigid cells:
        every face adjacent to a masked cell is closed (zero normal
        velocity) and every touching corner shear stress is zeroed, i.e. a
        shear-free rigid inclusion, rasterising arbitrary interior geometry.
    """

    def __init__(
        self,
        c_p: float | Field2D,
        c_s: float | Field2D,
        dx: float,
        *,
        rho: float | Field2D,
        cfl: float = 0.6,
        sponge_width: int = 0,
        sponge_sides: str | Iterable[str] | None = None,
        sponge_reflection: float = 1e-4,
        damping: float | NDArray[np.float64] = 0.0,
        shape: tuple[int, int] | None = None,
        free_sides: str | Iterable[str] | None = None,
        obstacle_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        cp_map = _resolve_speed_map("c_p", c_p, shape)
        _positive_map("c_p", cp_map)
        ny, nx = cp_map.shape
        if ny < 2 or nx < 2:
            raise ValueError("the grid must be at least 2 x 2 cells")
        cs_map = _resolve_speed_map("c_s", c_s, (ny, nx))
        if cs_map.shape != (ny, nx):
            raise ValueError("c_s map must match the shape of c_p")
        if not np.all(np.isfinite(cs_map)) or bool(np.any(cs_map < 0.0)):
            raise ValueError("c_s must be non-negative and finite "
                             "everywhere")
        # lambda >= 0 <=> c_p**2 >= 2 c_s**2 (a small relative tolerance
        # admits c_s = c_p / sqrt(2) given in rounded floats).
        if bool(np.any(cp_map**2 < 2.0 * cs_map**2 * (1.0 - 1e-9))):
            raise ValueError("c_p**2 must be at least 2 * c_s**2 in every "
                             "cell (non-negative lambda); c_s = 0 marks a "
                             "fluid cell")
        if not np.isfinite(cfl) or not 0.0 < cfl < 1.0:
            raise ValueError("cfl must lie in (0, 1): the leapfrog scheme "
                             "is unstable beyond the Courant bound CN = 1")
        rho_map = (np.full((ny, nx), float(np.real(rho)), dtype=np.float64)
                   if np.isscalar(rho) else np.asarray(rho, dtype=np.float64))
        if rho_map.shape != (ny, nx):
            raise ValueError("rho map must match the shape of c_p")
        _positive_map("rho", rho_map)
        sponge_width, damping_map = _validate_sponge_damping(
            sponge_width, sponge_reflection, damping, ny, nx)

        self.dx = _positive_finite("dx", dx)
        self.c_p = cp_map
        self.c_s = cs_map
        self.rho = rho_map
        c_max = float(cp_map.max())
        self.dt = cfl * self.dx / (c_max * float(np.sqrt(2.0)))
        #: Shear modulus ``mu = rho c_s**2`` at cell centres.
        self.mu: Field2D = rho_map * cs_map**2
        #: First Lame parameter ``lambda = rho (c_p**2 - 2 c_s**2)``.
        self.lam: Field2D = np.maximum(
            rho_map * (cp_map**2 - 2.0 * cs_map**2), 0.0)
        self._mu2 = 2.0 * self.mu
        # Plane-strain surface modulus E' = 4 mu (lambda + mu) /
        # (lambda + 2 mu): on a free surface the zero normal stress ties
        # the normal strain rate to the tangential one, and the in-plane
        # stress evolves with E' times the tangential strain rate alone.
        self._e_surf: Field2D = np.divide(
            4.0 * self.mu * (self.lam + self.mu),
            self.lam + self._mu2,
            out=np.zeros_like(self.mu),
            where=(self.lam + self._mu2) > 0.0)
        # Density averaged onto the faces where each velocity lives
        # (arithmetic mean, Moczo et al. Eq. 7.39).
        self._rho_x = 0.5 * (rho_map[:, 1:] + rho_map[:, :-1])
        self._rho_y = 0.5 * (rho_map[1:, :] + rho_map[:-1, :])
        self._mu_c = self._corner_mu(self.mu)

        self.txx: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self.tyy: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self.txy: Field2D = np.zeros((ny - 1, nx - 1), dtype=np.float64)
        self.vx: Field2D = np.zeros((ny, nx - 1), dtype=np.float64)
        self.vy: Field2D = np.zeros((ny - 1, nx), dtype=np.float64)
        self._div: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self._dvx: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self._dvy: Field2D = np.zeros((ny, nx), dtype=np.float64)
        self._dty: Field2D = np.zeros((ny, nx - 1), dtype=np.float64)
        self._dtx: Field2D = np.zeros((ny - 1, nx), dtype=np.float64)
        self._sources: list[ElasticSource] = []
        self.n = 0                                # completed steps

        sides = _resolve_sponge_sides(sponge_sides)
        #: Sponge-layer width in cells (read-only configuration record).
        self.sponge_width = int(sponge_width)
        #: Sides carrying a sponge layer when ``sponge_width > 0``.
        self.sponge_sides: tuple[str, ...] = (
            sides if sponge_width > 0 else ()
        )
        #: Sides carrying a traction-free surface (immutable record).
        self.free_sides: tuple[str, ...] = _resolve_free_sides(free_sides)
        overlap = set(self.free_sides) & set(self.sponge_sides)
        if overlap:
            raise ValueError(f"sides {sorted(overlap)} cannot be both free "
                             "and absorbing")
        self._init_decay(sides, sponge_width, sponge_reflection, damping_map,
                         c_max)
        self._init_obstacle(obstacle_mask, ny, nx)

    @classmethod
    def from_regions(
        cls,
        shape: tuple[int, int],
        dx: float,
        *,
        background: Material | tuple[float, float, float],
        regions: Iterable[
            tuple[Any, Material | tuple[float, float, float]]
        ] = (),
        **kwargs: Any,
    ) -> ElasticFDTD2D:
        """Build the engine from named materials painted over a background.

        Thin sugar over the map constructor for the common layered set-ups
        (fluid over a solid half-space, an immersed plate, an inclusion):
        the ``(ny, nx)`` maps of ``c_p``, ``c_s`` and ``rho`` start uniform
        at ``background`` and each ``(where, material)`` entry of
        ``regions`` is painted over them in order (later entries overwrite
        earlier ones), then everything is delegated to the normal
        constructor. No new physics: a fluid-solid contact still needs no
        explicit interface treatment, because the constructor's effective
        parameters (Moczo et al. 2007, Eqs. 7.37-7.39) handle it from the
        maps alone.

        ``where`` selects the painted cells: a boolean ``(ny, nx)`` mask,
        or any basic numpy index expression over the ``(row, column)``
        maps, e.g. ``(slice(120, None), slice(None))`` for the lower half
        or ``numpy.s_[120:, :]`` for the same thing spelled as a slice.

        :param shape: Grid shape ``(ny, nx)``.
        :param dx: Grid spacing [m] (square cells).
        :param background: The material filling the whole grid first: a
            :class:`Material` or a ``(c_p, c_s, rho)`` triple, e.g.
            :data:`WATER`.
        :param regions: ``(where, material)`` pairs painted in order.
        :param kwargs: Forwarded to :class:`ElasticFDTD2D` (``cfl``,
            ``sponge_width``, ``sponge_sides``, ``sponge_reflection``,
            ``damping``, ``free_sides``, ``obstacle_mask``).
        :return: The configured stepping engine.
        :raises ValueError: If a mask does not match ``shape`` or a
            material spec is invalid.
        """
        ny = _integer("shape[0]", shape[0])
        nx = _integer("shape[1]", shape[1])
        if ny < 2 or nx < 2:
            raise ValueError("the grid must be at least 2 x 2 cells")
        bg = _as_material("background", background)
        c_p = np.full((ny, nx), bg.c_p, dtype=np.float64)
        c_s = np.full((ny, nx), bg.c_s, dtype=np.float64)
        rho = np.full((ny, nx), bg.rho, dtype=np.float64)
        for k, (where, material) in enumerate(regions):
            mat = _as_material(f"regions[{k}]", material)
            index = where
            if isinstance(index, np.ndarray) and (
                    index.dtype != np.bool_ or index.shape != (ny, nx)):
                raise ValueError(
                    f"regions[{k}] mask must be a boolean array of "
                    f"shape {(ny, nx)}")
            c_p[index] = mat.c_p
            c_s[index] = mat.c_s
            rho[index] = mat.rho
        return cls(c_p, c_s, dx, rho=rho, **kwargs)

    @staticmethod
    def _corner_mu(mu: Field2D) -> Field2D:
        """Harmonic mean of ``mu`` over the 4 cells meeting at each corner.

        The Moczo et al. (2007) effective shear modulus (Eqs. 7.37-7.38):
        the harmonic mean enforces traction continuity across internal
        interfaces, and any fluid neighbour (``mu = 0``) zeroes the corner,
        which is exactly the shear-free wetted-wall condition of a
        fluid-solid contact (Virieux 1986).
        """
        quads = (mu[:-1, :-1], mu[:-1, 1:], mu[1:, :-1], mu[1:, 1:])
        all_solid = np.logical_and.reduce([q > 0.0 for q in quads])
        inv_sum = np.zeros_like(quads[0])
        for q in quads:
            inv_sum += np.divide(1.0, q, out=np.zeros_like(q),
                                 where=q > 0.0)
        return np.where(all_solid, 4.0 / np.where(inv_sum > 0.0, inv_sum,
                                                  1.0), 0.0)

    def _init_decay(self, sides: tuple[str, ...], sponge_width: int,
                    sponge_reflection: float,
                    damping: NDArray[np.float64],
                    c_max: float) -> None:
        """Precompute the sponge/damping decay factors of every field."""
        sigma = _sigma_map(self.txx.shape, sides, sponge_width,
                           sponge_reflection, damping, c_max, self.dx)
        self._decay_c: Field2D = np.exp(-sigma * self.dt)
        self._decay_vx: Field2D = np.exp(
            -(0.5 * (sigma[:, 1:] + sigma[:, :-1])) * self.dt)
        self._decay_vy: Field2D = np.exp(
            -(0.5 * (sigma[1:, :] + sigma[:-1, :])) * self.dt)
        self._decay_xy: Field2D = np.exp(
            -(0.25 * (sigma[1:, 1:] + sigma[1:, :-1]
                      + sigma[:-1, 1:] + sigma[:-1, :-1])) * self.dt)

    def _init_obstacle(self, obstacle_mask: NDArray[np.bool_] | None,
                       ny: int, nx: int) -> None:
        """Validate the obstacle mask into closed-face velocity factors."""
        self._obstacle: NDArray[np.bool_] | None = _validated_obstacle(
            obstacle_mask, ny, nx)
        self._vx_open: Field2D | None = None
        self._vy_open: Field2D | None = None
        self._txy_open: Field2D | None = None
        if self._obstacle is not None:
            mask = self._obstacle
            # A face between two cells is open only when both are open
            # (rigid obstacle boundary: zero normal velocity), and a corner
            # shear stress survives only when all four cells are open
            # (shear-free rigid inclusion, matching the domain edges).
            self._vx_open = (
                ~(mask[:, 1:] | mask[:, :-1])).astype(np.float64)
            self._vy_open = (
                ~(mask[1:, :] | mask[:-1, :])).astype(np.float64)
            self._txy_open = (
                ~(mask[1:, 1:] | mask[1:, :-1]
                  | mask[:-1, 1:] | mask[:-1, :-1])).astype(np.float64)

    @property
    def time(self) -> float:
        """Elapsed simulated time [s]."""
        return self.n * self.dt

    @property
    def p(self) -> Field2D:
        """Synthetic pressure ``-(txx + tyy) / 2`` at cell centres [Pa].

        The mean compressive normal stress: in fluid (``c_s = 0``) regions
        it is exactly the acoustic pressure, and in solids it is the
        convenient scalar for probes and snapshots.
        """
        return -0.5 * (self.txx + self.tyy)

    def add_source(self, source: ElasticSource) -> None:
        """Register an :class:`ExplosionSource` or a :class:`ForceSource`.

        Explosions inject additively at one cell centre; forces at one
        staggered velocity node. Positions are validated against the grid
        and the obstacle mask.
        """
        ix = _integer("source ix", source.ix)
        iy = _integer("source iy", source.iy)
        if isinstance(source, ExplosionSource):
            self._validate_explosion_position(ix, iy)
        else:
            self._validate_force_position(source, ix, iy)
        self._sources.append(source)

    def _validate_explosion_position(self, ix: int, iy: int) -> None:
        """Check an explosion cell against the grid and the obstacle mask."""
        ny, nx = self.txx.shape
        if not (0 <= ix < nx and 0 <= iy < ny):
            raise ValueError("source position lies outside the grid")
        if self._obstacle is not None and self._obstacle[iy, ix]:
            raise ValueError("source position lies inside an obstacle")

    def _validate_force_position(self, source: ForceSource, ix: int,
                                 iy: int) -> None:
        """Check a force velocity node against the grid and the obstacle."""
        ny, nx = self.txx.shape
        open_map = (self._vx_open if source.direction == "x"
                    else self._vy_open)
        limit = (ny, nx - 1) if source.direction == "x" else (ny - 1, nx)
        if not (0 <= ix < limit[1] and 0 <= iy < limit[0]):
            raise ValueError("source position lies outside the grid")
        # The open map holds exact 0.0/1.0 factors cast from booleans:
        # exact zero marks a face closed by the obstacle mask (<= avoids
        # the float-equality lint without changing the semantics).
        if open_map is not None and open_map[iy, ix] <= 0.0:
            raise ValueError("source position lies inside an obstacle")

    def step(self) -> None:
        """Advance the leapfrog scheme by one time step (Virieux Eq. 5)."""
        dt_dx = self.dt / self.dx
        txx, tyy = self.txx, self.tyy
        # Velocity half-step from the stress gradients; the one-sided txy
        # differences fold in the shear-free rigid edges and the imaged
        # free surfaces.
        self.vx += dt_dx / self._rho_x * ((txx[:, 1:] - txx[:, :-1])
                                          + self._shear_dty())
        self.vy += dt_dx / self._rho_y * ((tyy[1:, :] - tyy[:-1, :])
                                          + self._shear_dtx())
        if self._vx_open is not None and self._vy_open is not None:
            self.vx *= self._vx_open
            self.vy *= self._vy_open
        t_half = (self.n + 0.5) * self.dt
        for src in self._sources:
            if isinstance(src, ForceSource):
                self._inject_force(src, t_half)
        self.vx *= self._decay_vx
        self.vy *= self._decay_vy
        # Stress step from the velocity gradients (Hooke's law in rate
        # form): txx and tyy share the lambda * div(v) part, plus 2 mu
        # times their own normal strain rate; txy is driven by the shear
        # strain rate through the corner-averaged mu. Free surfaces stash
        # their E' in-plane stress first and rewrite it after the bulk
        # update.
        div, dvx, dvy = self._velocity_gradients()
        txx_surf, tyy_surf = self._stash_surface_stress(dt_dx, dvx, dvy)
        self.txx += self.lam * dt_dx * div + self._mu2 * dt_dx * dvx
        self.tyy += self.lam * dt_dx * div + self._mu2 * dt_dx * dvy
        self.txy += self._mu_c * dt_dx * (
            (self.vx[1:, :] - self.vx[:-1, :])
            + (self.vy[:, 1:] - self.vy[:, :-1]))
        if self._txy_open is not None:
            self.txy *= self._txy_open
        # Pin the free surfaces before the sources, so a stress source on
        # the surface row acts as a boundary traction drive.
        self._apply_free_surfaces(txx_surf, tyy_surf)
        t_next = (self.n + 1) * self.dt
        for src in self._sources:
            if isinstance(src, ExplosionSource):
                inj = src.amplitude * src.waveform(t_next)
                self.txx[src.iy, src.ix] -= inj
                self.tyy[src.iy, src.ix] -= inj
        self.txx *= self._decay_c
        self.tyy *= self._decay_c
        self.txy *= self._decay_xy
        self.n += 1

    def _shear_dty(self) -> Field2D:
        r"""One-sided y-difference of txy, free surfaces imaged (Moczo 9.9).

        The shear stress is zero on the domain edge (shear-free rigid
        wall) except on free sides, where the antisymmetric image
        :math:`t_{xy}(-d) = -t_{xy}(+d)` about the surface plane doubles the
        one-sided difference.
        """
        txy, dty = self.txy, self._dty
        dty[0, :] = txy[0, :]
        dty[1:-1, :] = txy[1:, :] - txy[:-1, :]
        dty[-1, :] = -txy[-1, :]
        if "top" in self.free_sides:
            dty[0, :] *= 2.0
        if "bottom" in self.free_sides:
            dty[-1, :] *= 2.0
        return dty

    def _shear_dtx(self) -> Field2D:
        """One-sided x-difference of txy, free surfaces imaged (Moczo 9.9)."""
        txy, dtx = self.txy, self._dtx
        dtx[:, 0] = txy[:, 0]
        dtx[:, 1:-1] = txy[:, 1:] - txy[:, :-1]
        dtx[:, -1] = -txy[:, -1]
        if "left" in self.free_sides:
            dtx[:, 0] *= 2.0
        if "right" in self.free_sides:
            dtx[:, -1] *= 2.0
        return dtx

    def _velocity_gradients(self) -> tuple[Field2D, Field2D, Field2D]:
        """Velocity divergence and one-sided vx/vy differences at centres."""
        div = self._div
        div.fill(0.0)
        div[:, :-1] += self.vx
        div[:, 1:] -= self.vx
        div[:-1, :] += self.vy
        div[1:, :] -= self.vy
        dvx = self._dvx
        dvx[:, 0] = self.vx[:, 0]
        dvx[:, 1:-1] = self.vx[:, 1:] - self.vx[:, :-1]
        dvx[:, -1] = -self.vx[:, -1]
        dvy = self._dvy
        dvy[0, :] = self.vy[0, :]
        dvy[1:-1, :] = self.vy[1:, :] - self.vy[:-1, :]
        dvy[-1, :] = -self.vy[-1, :]
        return div, dvx, dvy

    def _surface_lines(self) -> tuple[tuple[tuple[bool, int], ...],
                                      tuple[tuple[bool, int], ...]]:
        """Free-surface (active, index) pairs for the rows and columns."""
        free = self.free_sides
        rows = (("top" in free, 0), ("bottom" in free, -1))
        cols = (("left" in free, 0), ("right" in free, -1))
        return rows, cols

    def _stash_surface_stress(
        self, dt_dx: float, dvx: Field2D, dvy: Field2D,
    ) -> tuple[list[Field2D], list[Field2D]]:
        """E'-updated in-plane stress of each free surface, pre-bulk-update.

        On a free-surface row/column the zero normal stress ties the
        normal strain rate to the tangential one, so the in-plane normal
        stress must evolve with E' times the tangential strain rate alone
        (the vy/vx value beyond the surface is vacuum, not the rigid
        edge).
        """
        surf_rows, surf_cols = self._surface_lines()
        txx_surf = [self.txx[k, :] + self._e_surf[k, :] * dt_dx * dvx[k, :]
                    for on, k in surf_rows if on]
        tyy_surf = [self.tyy[:, k] + self._e_surf[:, k] * dt_dx * dvy[:, k]
                    for on, k in surf_cols if on]
        return txx_surf, tyy_surf

    def _apply_free_surfaces(self, txx_surf: list[Field2D],
                             tyy_surf: list[Field2D]) -> None:
        """Pin the surface normal stress to zero and restore the E' update.

        Where two orthogonal free sides meet, the corner cell lies on both
        surface planes: pin both normal stresses to zero there (the
        sequential row/column passes would otherwise leave the later
        pass's E' value on the earlier pass's pinned stress).
        """
        surf_rows, surf_cols = self._surface_lines()
        it_rows = iter(txx_surf)
        for on, k in surf_rows:
            if on:
                self.tyy[k, :] = 0.0
                self.txx[k, :] = next(it_rows)
        it_cols = iter(tyy_surf)
        for on, k in surf_cols:
            if on:
                self.txx[:, k] = 0.0
                self.tyy[:, k] = next(it_cols)
        for on_r, r in surf_rows:
            for on_c, c in surf_cols:
                if on_r and on_c:
                    self.txx[r, c] = 0.0
                    self.tyy[r, c] = 0.0

    def _inject_force(self, src: ForceSource, t: float) -> None:
        """Add one body-force increment to the target velocity node."""
        f = src.amplitude * src.waveform(t)
        if src.direction == "x":
            self.vx[src.iy, src.ix] += (
                self.dt * f / (self._rho_x[src.iy, src.ix] * self.dx**2))
        else:
            self.vy[src.iy, src.ix] += (
                self.dt * f / (self._rho_y[src.iy, src.ix] * self.dx**2))

    def energy(self) -> float:
        r"""Total elastic field energy [J per metre of depth].

        Kinetic energy plus the plane-strain elastic energy expressed in
        stresses by inverting the 2D stiffness (determinant
        :math:`4 \mu (\lambda + \mu)`); fluid cells (``mu = 0``) degenerate
        to the acoustic :math:`p^2 / (2 \lambda)` and shear-free corners
        contribute nothing.
        """
        e_v = 0.5 * (float(np.sum(self._rho_x * self.vx**2))
                     + float(np.sum(self._rho_y * self.vy**2)))
        lam, mu = self.lam, self.mu
        solid = mu > 0.0
        den = np.where(solid, 8.0 * mu * (lam + mu), 1.0)
        e_solid = ((lam + 2.0 * mu) * (self.txx**2 + self.tyy**2)
                   - 2.0 * lam * self.txx * self.tyy) / den
        # Fluid cells always have lam = rho c_p**2 > 0; guard the division
        # anyway so a lambda = 0 solid does not raise a spurious warning.
        e_fluid = np.divide(self.p**2, 2.0 * lam,
                            out=np.zeros_like(lam), where=~solid)
        e_n = float(np.sum(np.where(solid, e_solid, e_fluid)))
        mu_c = self._mu_c
        e_s = float(np.sum(np.where(
            mu_c > 0.0, self.txy**2 / np.where(mu_c > 0.0, 2.0 * mu_c, 1.0),
            0.0)))
        return (e_v + e_n + e_s) * self.dx**2

    def collocated(self, field: str) -> Field2D:
        """One field interpolated to the cell centres, shape ``(ny, nx)``.

        ``"p"`` is the synthetic pressure; ``"vx"``/``"vy"`` average the two
        adjacent faces (domain-edge faces count as zero, their built-in
        value), so probes and snapshots of every field share the cell-centre
        positions.
        """
        if field == "p":
            return self.p
        if field not in _FIELD_NAMES:
            raise ValueError(f"field must be one of {_FIELD_NAMES}")
        ny, nx = self.txx.shape
        out = np.zeros((ny, nx), dtype=np.float64)
        if field == "vx":
            out[:, :-1] += self.vx
            out[:, 1:] += self.vx
        else:
            out[:-1, :] += self.vy
            out[1:, :] += self.vy
        out *= 0.5
        return out

    def run(
        self,
        steps: int,
        record_every: int | None = None,
        decimate: int = 1,
    ) -> NDArray[np.float64]:
        """Advance ``steps`` steps, optionally recording pressure frames.

        With ``record_every = k`` a snapshot of the synthetic pressure ``p``
        is stored after every ``k``-th step (and one of the initial state),
        spatially subsampled by ``decimate``; the stacked
        ``(n_frames, ny', nx')`` array plugs straight into a
        ``FuncAnimation`` ``imshow`` update. Without ``record_every`` an
        empty array is returned and only the final state is kept.
        """
        return _run_recording(self, steps, record_every, decimate)


@dataclass(frozen=True)
class ElasticFDTDResult:
    r"""Frozen result of an :func:`elastic_fdtd_simulation` run.

    :ivar times: Time axis [s], length ``n_steps + 1`` (includes
        :math:`t = 0`).
    :ivar signals: Recorded histories, shape
        ``(n_probes, n_fields, n_steps + 1)``, one row per probe and one
        layer per entry of ``probe_fields`` (velocities are interpolated to
        the probe cell centre).
    :ivar probe_fields: The recorded fields, drawn from
        ``("p", "vx", "vy")``.
    :ivar probes: Probe cell indices ``(ix, iy)``, shape ``(n_probes, 2)``.
    :ivar probe_positions: Probe cell-centre positions ``(x, y)`` [m], shape
        ``(n_probes, 2)``.
    :ivar dx: Grid spacing [m].
    :ivar dt: Time step [s].
    :ivar shape: Grid shape ``(ny, nx)``.
    :ivar sources: The source definitions of the run.
    :ivar snapshots: Recorded cell-centre fields of ``snapshot_field``,
        shape ``(n_frames, ny, nx)``, or ``None`` when no snapshots were
        requested.
    :ivar snapshot_times: Time of each snapshot [s], or ``None``.
    :ivar snapshot_field: The field recorded in ``snapshots``.
    :ivar obstacle_mask: Boolean map of rigid cells, or ``None``.
    :ivar free_sides: Sides carrying a traction-free surface.
    """

    times: NDArray[np.float64]
    signals: NDArray[np.float64]
    probe_fields: tuple[str, ...]
    probes: NDArray[np.int_]
    probe_positions: NDArray[np.float64]
    dx: float
    dt: float
    shape: tuple[int, int]
    sources: tuple[ElasticSource, ...]
    snapshots: NDArray[np.float64] | None
    snapshot_times: NDArray[np.float64] | None
    snapshot_field: str
    obstacle_mask: NDArray[np.bool_] | None
    free_sides: tuple[str, ...]

    @property
    def size(self) -> tuple[float, float]:
        """Domain size ``(lx, ly)`` [m]."""
        ny, nx = self.shape
        return nx * self.dx, ny * self.dx

    def plot(self, ax: Axes | None = None, *, kind: str = "probes",
             frame: int = -1, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the probe histories or one recorded field snapshot.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param kind: ``"probes"`` (default) draws the per-probe signal time
            histories; ``"snapshot"`` renders one recorded field with the
            geometry overlaid (``imshow`` raster).
        :param frame: Snapshot index for ``kind="snapshot"`` (default: the
            last recorded frame).
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the underlying ``plot``/``imshow``.
        :return: The axes.
        """
        from .._i18n import check_language

        check_language(language)
        if kind == "probes":
            from .._plot.simulation import plot_elastic_probes

            return plot_elastic_probes(self, ax=ax, language=language,
                                       **kwargs)
        if kind == "snapshot":
            from .._plot.simulation import plot_elastic_snapshot

            return plot_elastic_snapshot(self, ax=ax, frame=frame,
                                         language=language, **kwargs)
        raise ValueError("kind must be 'probes' or 'snapshot'")


@dataclass(frozen=True)
class ElasticRecording:
    """What an elastic run writes down: probe traces and field snapshots.

    The two ways out of a running simulation. Probes record a time history at
    named cells, one trace per field of ``probe_fields``; snapshots record the
    whole domain every ``snapshot_every`` steps for one field. Recording
    nothing (the default) runs the physics and keeps only the time axis.

    :param probes: Probe cells as ``(ix, iy)`` index pairs.
    :param probe_fields: Which fields each probe records, drawn from
        ``("p", "vx", "vy")`` (default ``("vy",)``, the component a surface
        accelerometer would see).
    :param snapshot_every: Record a full field snapshot every this many steps
        (and at :math:`t = 0`); ``None`` records none.
    :param snapshot_field: Field recorded in the snapshots
        (``"p"``/``"vx"``/``"vy"``, interpolated to cell centres).
    """

    probes: Sequence[tuple[int, int]] = ()
    probe_fields: Sequence[str] = ("vy",)
    snapshot_every: int | None = None
    snapshot_field: str = "p"


@dataclass(frozen=True)
class ElasticBoundaries:
    """How the four edges of the elastic domain are terminated.

    The sponge thickness only means anything where a side is absorbing, which
    is why it travels with the sides rather than beside them.

    :param sides: ``"rigid"`` (default), ``"absorbing"``, ``"free"``, or a
        mapping from side name (``left``/``right``/``top``/``bottom``) to one
        of those.
    :param absorbing_layer_cells: Sponge-layer thickness for absorbing sides,
        in cells.
    """

    sides: str | Mapping[str, str] = "rigid"
    absorbing_layer_cells: int = 20


def _parse_elastic_boundaries(
    boundaries: str | Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the boundary spec into absorbing sides and free sides."""
    spec: dict[str, str]
    if isinstance(boundaries, str):
        spec = dict.fromkeys(_SIDES, boundaries)
    else:
        unknown = set(boundaries) - set(_SIDES)
        if unknown:
            raise ValueError(f"unknown boundary sides: {sorted(unknown)}")
        spec = {side: boundaries.get(side, "rigid") for side in _SIDES}
    absorbing: list[str] = []
    free: list[str] = []
    for side in _SIDES:
        value = spec[side]
        if value not in _BOUNDARY_NAMES:
            raise ValueError(
                f"boundary for side {side!r} must be one of "
                f"{_BOUNDARY_NAMES}, got {value!r}")
        if value == "absorbing":
            absorbing.append(side)
        elif value == "free":
            free.append(side)
    return tuple(absorbing), tuple(free)


def _resolve_probe_fields(probe_fields: Sequence[str]) -> tuple[str, ...]:
    """Validate the probe-field spec into a non-empty unique tuple."""
    fields = tuple(probe_fields)
    if not fields:
        raise ValueError("probe_fields must not be empty")
    unknown = set(fields) - set(_FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown probe fields: {sorted(unknown)}")
    if len(set(fields)) != len(fields):
        raise ValueError("probe_fields must not repeat a field")
    return fields


def _sample_probes(sim: ElasticFDTD2D, field: str,
                   rows: NDArray[np.int_],
                   cols: NDArray[np.int_]) -> NDArray[np.float64]:
    """Sample one field at the probe cell centres (velocities averaged)."""
    if field == "p":
        return -0.5 * (sim.txx[rows, cols] + sim.tyy[rows, cols])
    if field == "vx":
        nx = sim.txx.shape[1]
        left = np.where(cols > 0, sim.vx[rows, np.maximum(cols - 1, 0)], 0.0)
        right = np.where(cols < nx - 1,
                         sim.vx[rows, np.minimum(cols, nx - 2)], 0.0)
        return np.asarray(0.5 * (left + right), dtype=np.float64)
    ny = sim.txx.shape[0]
    above = np.where(rows > 0, sim.vy[np.maximum(rows - 1, 0), cols], 0.0)
    below = np.where(rows < ny - 1,
                     sim.vy[np.minimum(rows, ny - 2), cols], 0.0)
    return np.asarray(0.5 * (above + below), dtype=np.float64)


def _record_elastic_run(
    sim: ElasticFDTD2D,
    steps: int,
    probe_ix: NDArray[np.int_],
    probe_fields: tuple[str, ...],
    snapshot_every: int | None,
    snapshot_field: str,
) -> tuple[NDArray[np.float64], list[Field2D], list[int]]:
    """Step the engine, recording probe histories and field snapshots."""
    signals = np.zeros((probe_ix.shape[0], len(probe_fields), steps + 1),
                       dtype=np.float64)
    frames: list[Field2D] = []
    frame_steps: list[int] = []
    if snapshot_every is not None:
        frames.append(sim.collocated(snapshot_field))
        frame_steps.append(0)
    rows = probe_ix[:, 1]
    cols = probe_ix[:, 0]
    for i in range(steps):
        sim.step()
        if probe_ix.shape[0]:
            for k, field in enumerate(probe_fields):
                signals[:, k, i + 1] = _sample_probes(sim, field, rows, cols)
        if snapshot_every is not None and (i + 1) % snapshot_every == 0:
            frames.append(sim.collocated(snapshot_field))
            frame_steps.append(i + 1)
    return signals, frames, frame_steps


def _validated_snapshot_options(
    recording: ElasticRecording,
) -> tuple[int | None, str]:
    """Validate and normalise the snapshot cadence and field."""
    snapshot_every = recording.snapshot_every
    snapshot_field = recording.snapshot_field
    if snapshot_field not in _FIELD_NAMES:
        raise ValueError(f"snapshot_field must be one of {_FIELD_NAMES}")
    if snapshot_every is not None:
        snapshot_every = _integer("snapshot_every", snapshot_every)
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be >= 1")
    return snapshot_every, snapshot_field


def _validated_absorbing_layer(
    boundaries: ElasticBoundaries, absorbing_sides: tuple[str, ...],
) -> int:
    """Validate the absorbing-layer thickness; 0 when no side absorbs."""
    if not absorbing_sides:
        return 0
    cells = _integer("absorbing_layer_cells", boundaries.absorbing_layer_cells)
    if cells < 1:
        raise ValueError("absorbing_layer_cells must be >= 1")
    return cells


def elastic_fdtd_simulation(
    c_p: float | Field2D,
    c_s: float | Field2D,
    dx: float,
    duration: float,
    *,
    sources: Sequence[ElasticSource],
    rho: float | Field2D,
    shape: tuple[int, int] | None = None,
    cfl: float = 0.6,
    recording: ElasticRecording | None = None,
    boundaries: ElasticBoundaries | None = None,
    obstacle_mask: NDArray[np.bool_] | None = None,
    damping: float = 0.0,
) -> ElasticFDTDResult:
    r"""Run a deterministic 2D elastic (P-SV) FDTD simulation.

    Builds the staggered velocity-stress grid (Virieux 1986, Eq. 5), applies
    the requested boundary conditions, injects the sources and integrates
    for ``duration`` seconds, recording the selected fields at every probe
    each time step and, optionally, full-field snapshots.

    The grid covers ``(nx * dx, ny * dx)`` metres; a cell index ``(ix, iy)``
    maps to the physical cell centre ``((ix + 0.5) * dx, (iy + 0.5) * dx)``.
    Resolve at least 10 cells per shortest wavelength
    (:math:`\Delta x \le c_{\mathrm{s},\min} / (10 f)` with ``c_s_min`` the smallest
    non-zero
    ``c_s`` over the solid cells, since the S wave is always the shortest;
    a wholly fluid map falls back to the acoustic rule on the smallest
    ``c_p``; Virieux's rule from the dispersion relations
    Eqs. 13-14), and 15-20 cells per wavelength when a Rayleigh wave along a
    free surface matters, the second-order stress-imaging surface being the
    most dispersive part of the scheme. The simulation is 2D (plane strain):
    a point source is physically a line source with cylindrical
    :math:`1/\sqrt{r}` amplitude spreading.

    :param c_p: P-wave speed map [m/s], shape ``(ny, nx)``, or a scalar with
        an explicit ``shape``.
    :param c_s: S-wave speed map [m/s]; ``c_s = 0`` marks fluid cells.
    :param dx: Grid spacing [m] (square cells).
    :param duration: Physical time to simulate [s].
    :param sources: One or more of :class:`ExplosionSource` or
        :class:`ForceSource`.
    :param rho: Density map [kg/m3]; scalar or ``(ny, nx)`` array.
    :param shape: Grid shape ``(ny, nx)``, required when ``c_p`` is scalar.
    :param cfl: Courant number in ``(0, 1)`` (Virieux Eqs. 6-7); the time
        step is :math:`\Delta t = C_\mathrm{N}\, \Delta x / (c_{\mathrm{p},\max} \sqrt{2})`.
        Default 0.6.
    :param recording: What the run records (:class:`ElasticRecording`): the
        probe cells and their fields, and the snapshot cadence and field.
        ``None`` records nothing but the time axis.
    :param boundaries: How the domain edges are terminated
        (:class:`ElasticBoundaries`); ``None`` is rigid on all four sides.
    :param obstacle_mask: Boolean map, shape ``(ny, nx)``, of rigid cells
        (rasterised interior geometry).
    :param damping: Uniform bulk amplitude decay rate [1/s].
    :return: An :class:`ElasticFDTDResult`.
    :raises ValueError: If the inputs are invalid.
    """
    recording = ElasticRecording() if recording is None else recording
    boundaries = ElasticBoundaries() if boundaries is None else boundaries
    if len(sources) == 0:
        raise ValueError("at least one source is required")
    duration = _positive_finite("duration", duration)
    fields = _resolve_probe_fields(recording.probe_fields)
    snapshot_every, snapshot_field = _validated_snapshot_options(recording)
    absorbing_sides, free_sides = _parse_elastic_boundaries(boundaries.sides)
    absorbing_layer_cells = _validated_absorbing_layer(
        boundaries, absorbing_sides)

    sim = ElasticFDTD2D(
        c_p, c_s, dx,
        rho=rho,
        cfl=cfl,
        sponge_width=absorbing_layer_cells,
        sponge_sides=absorbing_sides if absorbing_sides else None,
        damping=damping,
        shape=shape,
        free_sides=free_sides,
        obstacle_mask=obstacle_mask,
    )
    for source in sources:
        sim.add_source(source)

    ny, nx = sim.txx.shape
    probe_ix = _probe_indices(recording.probes, (ny, nx), sim._obstacle)

    steps = round(duration / sim.dt)
    if steps < 1:
        raise ValueError("duration must cover at least one time step "
                         f"(dt = {sim.dt:.3e} s)")
    times = np.arange(steps + 1, dtype=np.float64) * sim.dt
    signals, frames, frame_steps = _record_elastic_run(
        sim, steps, probe_ix, fields, snapshot_every, snapshot_field)

    positions = (probe_ix.astype(np.float64) + 0.5) * sim.dx
    return ElasticFDTDResult(
        times=times,
        signals=signals,
        probe_fields=fields,
        probes=probe_ix,
        probe_positions=positions,
        dx=sim.dx,
        dt=sim.dt,
        shape=(ny, nx),
        sources=tuple(sources),
        snapshots=np.stack(frames) if frames else None,
        snapshot_times=(np.asarray(frame_steps, dtype=np.float64) * sim.dt
                        if frame_steps else None),
        snapshot_field=snapshot_field,
        obstacle_mask=(sim._obstacle.copy()
                       if sim._obstacle is not None else None),
        free_sides=sim.free_sides,
    )
