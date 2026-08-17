#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Numerical models of underwater sound propagation (range-independent ocean).

Four complementary numerical solvers for the acoustic field in a
horizontally-stratified ocean waveguide, complementing the closed-form
propagation loss of :mod:`phonometry.underwater.propagation.closed_form`:

* :func:`normal_modes` -- the normal-mode expansion. Solves the depth-separated
  Sturm-Liouville eigenvalue problem by finite differences and assembles the
  propagation loss from the propagating modes.
* :func:`ray_trace` -- ray tracing. Integrates the ray-trajectory equations
  through a sound-speed profile (Runge-Kutta), returning the ray paths and the
  travel time accumulated along each of them, and no amplitude.
* :func:`gaussian_beams` -- Gaussian beam tracing. Hangs a beam on each of those
  rays and sums them into a propagation-loss field, which is finite at a caustic
  and decays smoothly into a shadow zone where ray theory has nothing to say.
* :func:`parabolic_equation` -- the standard (Tappert) parabolic equation, solved
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
propagation losses can be laid side by side: ``normal_modes`` on a range slice
at one receiver depth, ``gaussian_beams`` and ``parabolic_equation`` on a
(depth, range) grid. Which of them to reach for is a question of frequency and
of what is being asked; the guide's solver table sets it out.

Densities are in kg/m3, sound speeds in m/s, depths and ranges in metres,
frequencies in Hz. The water column has a pressure-release surface at z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ..._internal.rays import DynamicRays, march_rays
from ..._internal.validation import require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..._internal.rays import RayDerivative, RayMarch

_BOTTOM_TYPES = ("pressure-release", "rigid")
#: Pressure reflection coefficient of each bottom kind. The sea surface is
#: always pressure-release, so its own coefficient is the -1 below.
_BOTTOM_REFLECTION = {"pressure-release": -1.0, "rigid": 1.0}
_SURFACE_REFLECTION = -1.0


def _clean_profile(
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    z = np.asarray(depths, dtype=np.float64)
    c = np.asarray(sound_speeds, dtype=np.float64)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("'depths' must be a 1-D array of at least two points.")
    if c.shape != z.shape:
        raise ValueError("'sound_speeds' must match 'depths' in length.")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(c))):
        raise ValueError("'depths' and 'sound_speeds' must be finite.")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("'depths' must be strictly increasing.")
    if abs(float(z[0])) > 1e-9:
        raise ValueError("'depths' must start at the surface z = 0.")
    if np.any(c <= 0.0):
        raise ValueError("'sound_speeds' must be strictly positive.")
    return z, c


def _ocean_ray_derivative(
    z_prof: NDArray[np.float64], c_prof: NDArray[np.float64],
    xi: NDArray[np.float64],
) -> RayDerivative:
    r"""The ocean's ray equations in range, vectorised over rays.

    March in range :math:`r` (not arc length): every valid ray then spans
    ``[0, max_range]`` in the same number of steps regardless of its launch
    angle. The state is :math:`(z, \zeta, t)` and
    :math:`\xi = \cos\theta_0/c(z_\mathrm{s})` is invariant for a
    range-independent :math:`c(z)`, so from
    :math:`dz/ds`, :math:`d\zeta/ds`, :math:`dt/ds = 1/c` and
    :math:`dr/ds = c\,\xi`,

    .. math::

        \frac{dz}{dr} = \frac{\zeta}{\xi}, \qquad
        \frac{d\zeta}{dr} = -\frac{dc/dz}{c^3 \xi}, \qquad
        \frac{dt}{dr} = \frac{1}{\xi c^2} .

    The time shares the sound speed the other two derivatives already need, so
    carrying it costs one multiply per stage and inherits the Runge-Kutta order:
    at the default step it reproduces the linear-gradient closed form to
    ~1e-14 s, where accumulating :math:`dr/(\xi c^2)` over the finished path
    would be first order.

    The profile is piecewise linear, so :math:`c(z)` interpolates exactly and
    :math:`dc/dz` is piecewise *constant* with jumps at the nodes; evaluating
    the gradient per segment keeps thermocline kinks sharp, where a smoothed
    gradient on an interpolated fine grid biases turning depths by metres. Which
    segment a node itself belongs to is settled by the direction of travel
    rather than by rounding, which is what a march that lands its sub-steps
    exactly on the nodes needs (see :mod:`phonometry._internal.rays`).
    """
    seg_grad = np.diff(c_prof) / np.diff(z_prof)

    def deriv(
        z_arr: NDArray[np.float64], zeta_arr: NDArray[np.float64], /
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        cc = np.interp(z_arr, z_prof, c_prof)
        seg = np.where(zeta_arr >= 0.0,
                       np.searchsorted(z_prof, z_arr, side="right") - 1,
                       np.searchsorted(z_prof, z_arr, side="left") - 1)
        grad = seg_grad[np.clip(seg, 0, seg_grad.size - 1)]
        return (zeta_arr / xi, -grad / (cc**3 * xi), 1.0 / (xi * cc**2))

    return deriv


# ===========================================================================
# 1. Normal modes (Jensen Ch. 5)
# ===========================================================================


@dataclass(frozen=True)
class NormalModeResult:
    """Normal-mode solution of a range-independent waveguide.

    :ivar frequency: Source frequency, in Hz.
    :ivar wavenumbers: Horizontal wavenumbers ``krm`` of the propagating modes,
        in rad/m (descending order).
    :ivar mode_depths: Depth grid of the mode functions, in metres.
    :ivar mode_functions: Orthonormalised mode shapes ``Ψm(z)``, shape
        ``(n_modes, n_depths)``.
    :ivar ranges: Ranges at which the propagation loss is evaluated, in metres.
    :ivar propagation_loss: Coherent propagation loss at ``receiver_depth``
        per range, in dB.
    :ivar receiver_depth: Receiver depth of the propagation-loss slice, in m.
    :ivar source_depth: Source depth, in metres.
    """

    frequency: float
    wavenumbers: NDArray[np.float64]
    mode_depths: NDArray[np.float64]
    mode_functions: NDArray[np.float64]
    ranges: NDArray[np.float64]
    propagation_loss: NDArray[np.float64]
    receiver_depth: float
    source_depth: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the propagation loss versus range (loss increasing downward)."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_normal_modes

        return plot_normal_modes(self, ax=ax, language=check_language(language), **kwargs)


def _propagating_band(
    eigvals: NDArray[np.float64], eigvecs: NDArray[np.float64],
    k2_max: float, dz: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagating wavenumbers (descending) and mode columns, cutoff-guarded.

    Discards eigenvalues inside the O(dz²) discretisation-error band about
    cutoff: the FD scheme can push a truly evanescent mode marginally above
    zero (a spurious propagating mode) and biases genuine near-cutoff modes;
    warns when a retained mode sits within ten times that band.
    """
    fd_floor = k2_max**2 * dz**2 / 12.0
    prop = eigvals > fd_floor
    if np.any(prop & (eigvals <= 10.0 * fd_floor)):
        import warnings

        from ..._internal.warnings import PhonometryWarning

        warnings.warn(
            "normal_modes: retained near-cutoff mode(s) lie within 10x the"
            " finite-difference error band; increase 'n_depth_points' to"
            " resolve them accurately.", PhonometryWarning, stacklevel=3)
    kr = np.sqrt(eigvals[prop])
    order = np.argsort(kr)[::-1]  # descending kr (mode 1 first)
    return kr[order], eigvecs[:, prop][:, order]


def normal_modes(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    receiver_depth: float,
    ranges_m: NDArray[np.float64] | list[float] | None = None,
    density: float = 1000.0,
    bottom: str = "pressure-release",
    n_depth_points: int | None = None,
) -> NormalModeResult:
    r"""Normal-mode propagation loss for a range-independent waveguide.

    Solves the depth-separated Sturm-Liouville problem (Jensen Eq. 5.3) on a
    uniform finite-difference grid, then assembles the coherent propagation
    loss from the propagating modes (Eq. 5.17).

    The finite-difference eigenvalues carry an :math:`O(dz^2)` error that
    grows with the mode's vertical wavenumber, so near-cutoff modes need a
    fine grid. Two guards apply: eigenvalues inside the scheme's error band
    (:math:`k_r^2 \le \max(k^2)^2 \, dz^2 / 12`) are discarded as numerically
    indistinguishable from cutoff, and a
    :class:`~phonometry.PhonometryWarning` is emitted when a
    retained mode sits within ten times that band (increase ``n_depth_points``
    to resolve it).

    :param frequency_hz: Source frequency, in Hz.
    :param depths: Depth samples of the sound-speed profile, in metres, starting
        at the surface ``z = 0`` and strictly increasing to the bottom.
    :param sound_speeds: Sound speed at each depth, in m/s.
    :param source_depth: Source depth ``zs``, in metres.
    :param receiver_depth: Receiver depth for the propagation-loss slice, in m.
    :param ranges_m: Ranges at which to evaluate the loss, in metres; defaults to
        100 m to 10 km.
    :param density: Water density (constant), in kg/m3.
    :param bottom: ``"pressure-release"`` (default) or ``"rigid"``.
    :param n_depth_points: Number of finite-difference depth points. Default
        (``None``): derived from the physics as
        :math:`\max(400, \operatorname{ceil}(60 D f / c_{\mathrm{min}}))`,
        which keeps the near-cutoff
        eigenvalue error small at any frequency/depth combination, capped at
        20 000 points (very high :math:`f D` products exceed the cap; the
        near-cutoff warning then indicates whether the capped grid suffices,
        and an explicit ``n_depth_points`` overrides the cap).
    :return: A :class:`NormalModeResult`.
    :raises ValueError: If the inputs are invalid.
    """
    f = require_positive(frequency_hz, "frequency_hz")
    rho = require_positive(density, "density")
    z_prof, c_prof = _clean_profile(depths, sound_speeds)
    water_depth = float(z_prof[-1])
    zs = float(source_depth)
    zr = float(receiver_depth)
    if not (0.0 < zs < water_depth) or not (0.0 < zr < water_depth):
        raise ValueError("'source_depth'/'receiver_depth' must lie within the water column.")
    key = bottom.strip().lower()
    if key not in _BOTTOM_TYPES:
        raise ValueError(f"'bottom' must be one of {_BOTTOM_TYPES}, got {bottom!r}.")
    if n_depth_points is None:
        n_depth_points = min(20_000, max(400, int(np.ceil(
            60.0 * water_depth * f / float(np.min(c_prof))))))
    if int(n_depth_points) < 8:
        raise ValueError("'n_depth_points' must be at least 8.")

    ranges = (
        np.linspace(100.0, 10_000.0, 400)
        if ranges_m is None
        else np.asarray(ranges_m, dtype=np.float64).ravel()
    )
    if np.any(ranges <= 0.0) or not np.all(np.isfinite(ranges)):
        raise ValueError("'ranges_m' must be finite and positive.")

    omega = 2.0 * np.pi * f
    n = int(n_depth_points)
    z = np.linspace(0.0, water_depth, n)
    dz = z[1] - z[0]
    c = np.interp(z, z_prof, c_prof)
    k2 = (omega / c) ** 2  # ω²/c²(z)

    # Discretise Ψ'' + (k² − kr²)Ψ = 0 (constant ρ) on the interior grid with a
    # pressure-release surface Ψ_0 = 0. The eigenvalue is kr².
    inv_dz2 = 1.0 / dz**2
    if key == "rigid":
        # Unknowns Ψ_1..Ψ_{n-1}; the bottom node n-1 has a Neumann condition
        # dΨ/dz|_D = 0 (ghost Ψ_n = Ψ_{n-2}), giving an asymmetric row with a
        # doubled coupling 2/dz². Symmetrise by the similarity D = diag(1,…,1/√2)
        # so the coupling becomes √2/dz² (eigenvalues preserved); the true mode's
        # last node is then recovered by multiplying that component by √2.
        idx = np.arange(1, n)
        main = k2[idx] - 2.0 * inv_dz2
        off = np.full(idx.size - 1, inv_dz2)
        off[-1] = np.sqrt(2.0) * inv_dz2
    else:
        # Unknowns Ψ_1..Ψ_{n-2}; pressure-release bottom Ψ_{n-1} = 0.
        idx = np.arange(1, n - 1)
        main = k2[idx] - 2.0 * inv_dz2
        off = np.full(idx.size - 1, inv_dz2)
    # The operator is symmetric tridiagonal; solve directly from the diagonals
    # and restrict the solve to the propagating band kr² > 0 (`select="v"`)
    # rather than computing the full spectrum.
    from scipy.linalg import eigh_tridiagonal

    k2_max = float(k2.max())
    eigvals, eigvecs = eigh_tridiagonal(
        main, off, select="v", select_range=(0.0, k2_max * (1.0 + 1e-12)))
    kr, shapes_int = _propagating_band(eigvals, eigvecs, k2_max, dz)

    # Rebuild full-depth mode functions with the boundary nodes.
    n_modes = kr.size
    psi = np.zeros((n_modes, n), dtype=np.float64)
    if key == "rigid":
        shapes = shapes_int.T.copy()
        shapes[:, -1] *= np.sqrt(2.0)  # un-scale the Neumann boundary node
        psi[:, 1:] = shapes
    else:
        psi[:, 1:-1] = shapes_int.T
    # Normalise: ∫ Ψ²/ρ dz = 1 (trapezoid), constant ρ.
    for m in range(n_modes):
        norm = np.trapezoid(psi[m] ** 2 / rho, z)
        if norm > 0.0:
            psi[m] /= np.sqrt(norm)

    # Linear interpolation of every mode at zs/zr in one shot (uniform grid).
    def _modes_at(zq: float) -> NDArray[np.float64]:
        i = int(np.clip(np.searchsorted(z, zq) - 1, 0, z.size - 2))
        w = (zq - z[i]) / (z[i + 1] - z[i])
        return np.asarray(psi[:, i] * (1.0 - w) + psi[:, i + 1] * w)

    psi_s = _modes_at(zs)
    psi_r = _modes_at(zr)

    # Coherent PL (Eq. 5.14/5.17): p = i/(ρ√(8πr)) e^{-iπ/4} Σ Ψm(zs)Ψm(zr) e^{i kr r}/√kr
    r = ranges
    modal = (psi_s * psi_r / np.sqrt(kr))[:, None] * np.exp(1j * kr[:, None] * r[None, :])
    field = (1j / (rho * np.sqrt(8.0 * np.pi * r))) * np.exp(-1j * np.pi / 4.0) * modal.sum(axis=0)
    p0 = 1.0 / (4.0 * np.pi)  # free-field pressure magnitude at r = 1 m
    with np.errstate(divide="ignore"):
        pl = -20.0 * np.log10(np.abs(field) / p0)

    return NormalModeResult(
        frequency=f,
        wavenumbers=kr,
        mode_depths=z,
        mode_functions=psi,
        ranges=r,
        propagation_loss=np.asarray(pl, dtype=np.float64),
        receiver_depth=zr,
        source_depth=zs,
    )


# ===========================================================================
# 2. Ray tracing (Jensen Ch. 3, Eqs. 3.23-3.24)
# ===========================================================================


@dataclass(frozen=True)
class RayTraceResult:
    """Ray-tracing solution through a sound-speed profile.

    :ivar launch_angles: Launch angles from the horizontal, in degrees.
    :ivar ranges: Per-ray horizontal ranges, in metres, shape
        ``(n_rays, n_steps)``.
    :ivar depths: Per-ray depths, in metres, shape ``(n_rays, n_steps)``.
    :ivar travel_times: Per-ray cumulative travel times, in seconds, shape
        ``(n_rays, n_steps)`` (zero at the source, increasing along the ray).
    :ivar source_depth: Source depth, in metres.
    :ivar water_depth: Water-column depth, in metres.
    """

    launch_angles: NDArray[np.float64]
    ranges: NDArray[np.float64]
    depths: NDArray[np.float64]
    travel_times: NDArray[np.float64]
    source_depth: float
    water_depth: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the ray paths (depth increasing downward)."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_ray_trace

        return plot_ray_trace(self, ax=ax, language=check_language(language), **kwargs)


def ray_trace(
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    launch_angles_deg: NDArray[np.float64] | list[float],
    max_range: float = 10_000.0,
    n_steps: int = 2000,
) -> RayTraceResult:
    r"""Trace acoustic rays through a range-independent sound-speed profile.

    Integrates the ray-trajectory equations (Jensen Eqs. 3.23-3.24) with a
    fixed-step fourth-order Runge-Kutta scheme, reflecting at the pressure-release
    surface (``z = 0``) and the bottom (``z = water_depth``).

    The travel time is a third state of that same Runge-Kutta step rather than a
    quadrature run over the finished path: with the range-invariant Snell
    parameter :math:`\xi = \cos\theta_0 / c(z_\mathrm{s})` it obeys
    :math:`dt/dr = 1/(\xi c^2)`, so it is integrated with the very stages that
    place the ray and cannot drift from the geometry actually returned. This is
    the same ray core, and the same travel-time equation, as the atmospheric
    :func:`~phonometry.environment.propagation.refraction.atmospheric_ray_paths`
    (which reflects at the ground instead of at the sea surface). Reflections
    cost no time, so the accumulated time stays continuous across them.

    :param depths: Depth samples of the profile, in metres, from ``z = 0``.
    :param sound_speeds: Sound speed at each depth, in m/s.
    :param source_depth: Source depth, in metres.
    :param launch_angles_deg: Launch angles from the horizontal, in degrees
        (positive downward).
    :param max_range: Maximum horizontal range to trace, in metres.
    :param n_steps: Number of integration steps per ray.
    :return: A :class:`RayTraceResult`.
    :raises ValueError: If the inputs are invalid.
    """
    z_prof, c_prof = _clean_profile(depths, sound_speeds)
    water_depth = float(z_prof[-1])
    zs = float(source_depth)
    if not (0.0 <= zs <= water_depth):
        raise ValueError("'source_depth' must lie within the water column.")
    rmax = require_positive(max_range, "max_range")
    if int(n_steps) < 2:
        raise ValueError("'n_steps' must be at least 2.")
    angles = np.asarray(launch_angles_deg, dtype=np.float64).ravel()
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("'launch_angles_deg' must be finite and non-empty.")
    if np.any(np.abs(angles) >= 90.0):
        raise ValueError("'launch_angles_deg' must be within (-90, 90) degrees (forward rays).")

    ns = int(n_steps)
    ranges = np.linspace(0.0, rmax, ns)
    c0 = float(np.interp(zs, z_prof, c_prof))
    th = np.radians(angles)
    xi = np.cos(th) / c0  # Snell invariant per ray (> 0 since |θ0| < 90°)
    deriv = _ocean_ray_derivative(z_prof, c_prof, xi)

    # The marcher splits every range step at the surface or bottom it crosses,
    # so a reflected ray keeps the order the rest of the path is integrated
    # with; see :mod:`phonometry._internal.rays`.
    march = march_rays(deriv, xi=xi, z0=np.full(angles.size, zs),
                       zeta0=np.sin(th) / c0, range_step=rmax / (ns - 1),
                       n_steps=ns, lower=0.0, upper=water_depth)
    ray_r = np.broadcast_to(ranges, march.positions.shape).copy()

    return RayTraceResult(
        launch_angles=angles,
        ranges=ray_r,
        depths=march.positions,
        travel_times=march.times,
        source_depth=zs,
        water_depth=water_depth,
    )


# ===========================================================================
# 3. Gaussian beam tracing (Jensen Ch. 3, Sect. 3.5, Eqs. 3.88-3.92)
# ===========================================================================
#
# A ray carries a travel time and nothing else; the amplitude lives in the
# dynamic pair (q, p) of Eq. (3.58) that :func:`ray_trace`'s marcher can carry
# alongside it. Give that pair the *complex* initial conditions of Eq. (3.91)
# and each ray becomes the central ray of a Gaussian beam, Eq. (3.88):
#
#     p^beam(s, n) = A sqrt( c(s) / (r q(s)) )
#                    exp{ -i omega [ tau(s) + (p(s)/(2 q(s))) n^2 ] }   (3.88)
#
# with ``n`` the normal distance from the central ray and ``r`` the cylindrical
# range of the point on it, which is there because r q = J (Eq. 3.64) carries
# the extra factor r that a point source in a cylindrically symmetric ocean
# brings with it (Eqs. 3.39, 3.46). The field is the sum of Eq. (3.88) over the
# launch fan, weighted by Eq. (3.92). Three things are worth stating up front,
# because each of them is a way to be quietly wrong.
#
# WHY THE FIELD IS FINITE. Write the pair as q = q_R + i q_I, p = p_R + i p_I.
# Eq. (3.58) is linear with real coefficients, so (q_R, p_R) and (q_I, p_I) are
# two real solutions of the same equation, and their Wronskian
# q_R p_I - q_I p_R is conserved: its derivative is
# c p_R p_I - (c_nn/c^2) q_R q_I - c p_I p_R + (c_nn/c^2) q_I q_R = 0. The
# impulse at a profile node and the one at a reflection are both q -> q,
# p -> p + (something) q, of unit determinant, so they conserve it too. With
# Eq. (3.91) it starts at -omega W_0^2 / 2 and stays there, which says three
# things at once:
#
#   * q never vanishes, because vanishing needs q_R = q_I = 0. There is no
#     caustic singularity to patch, and the KMAH index of Eq. (3.79) is not
#     needed: the -pi/2 per caustic is carried by the complex square root.
#   * Im[p/q] = Im[p conj(q)]/|q|^2 = -omega W_0^2/(2|q|^2) is strictly
#     negative, so Eq. (3.88) always decays away from its central ray and the
#     transverse exponent can never overflow.
#   * the beam half-width of Eq. (3.89) has the closed form
#     W(s) = sqrt(-2/(omega Im[p/q])) = 2|q(s)|/(omega W_0), which is what this
#     module computes and what makes the reach of a beam cheap to bound.
#
# THE SIGN CONVENTIONS, WHICH DO NOT AGREE AS PRINTED. Eq. (3.89) needs
# Im[p/q] < 0, and Eq. (3.91) delivers it, so Eq. (3.88) is written in the
# exp(+i omega t) convention: its propagation factor is exp(-i omega tau), the
# conjugate of the exp(i k_r r) of Eq. (5.14) that :func:`normal_modes` uses and
# of the exp(i k_0 r) of the PE. Eq. (3.92) as printed carries e^(+i pi/4),
# which belongs with the *other* convention; against Eqs. (3.88) and (3.91) it
# puts the free field exactly pi/2 out of phase (measured: arg of the ratio to
# exp(-i omega s/c_0)/s is +1.5712 rad with e^(+i pi/4) and +4.4e-4 rad with
# e^(-i pi/4), at r = 2 km, 100 Hz, W_0 = 20 lambda). So Eqs. (3.88), (3.91) and
# (3.92)-with-e^(-i pi/4) are implemented as one consistent triple and the
# summed field is conjugated once at the end, which puts the exposed complex
# pressure in the same exp(-i omega t) convention as the other two solvers.
# This is a textbook inconsistency, not a defect of a published standard, so it
# is recorded here rather than in docs/ERRATA.md.
#
# Substituting Eq. (3.91) into that corrected Eq. (3.92) makes sqrt(i) e^(-i
# pi/4) = 1, and the weight comes out real and positive in closed form,
#
#     A(theta_0) = dtheta_0 (omega W_0 / (2 c_0)) sqrt(cos(theta_0) / pi),
#
# which agrees with the square-root form to 2.3e-16 relative and is cheaper and
# branch-free. Its normalisation is the one of Sect. 3.5, Eq. (3.80),
# p = e^(iks)/s, and NOT the 1/(4 pi s) of Eq. (3.51): the beam sum converges to
# unit pressure at 1 m, so the propagation loss is -20 lg|sum| with no division
# by p_0. (:func:`normal_modes` divides by p_0 = 1/(4 pi) because its own field
# carries that factor.) Getting this wrong is a flat 20 lg(4 pi) = 21.98 dB.
#
# THE BRANCH OF THE SQUARE ROOT. sqrt(c/(r q)) with complex q has to be taken on
# a continuously tracked branch: :func:`numpy.sqrt` jumps by pi whenever its
# argument crosses the negative real axis, and it would do so exactly where the
# geometric ray had its caustic, re-creating the phase error of Fig. 3.14 that
# this whole method exists to remove. In free space q stays in one half plane
# and never crosses, so a free-field test cannot catch it. The argument of q is
# therefore unwrapped along each ray, and continued to the influence point by
# the increment arg(q_infl conj(q_sample)): that increment can never reach
# +-pi, because q_infl = q + c p ds runs along a straight line in the complex
# plane which, by the Wronskian above, misses the origin, and every point of
# such a line lies within an open half plane about it.


@dataclass(frozen=True)
class GaussianBeamResult:
    """Gaussian beam solution of a range-independent waveguide.

    The propagation-loss field is on the same footing as
    :class:`ParabolicEquationResult`'s: same shape, same reference, so the two
    can be subtracted.

    :ivar frequency: Source frequency, in Hz.
    :ivar ranges: Range grid of the field, in metres.
    :ivar depths: Depth grid of the field, in metres.
    :ivar propagation_loss: Propagation-loss field ``PL(z, r)``, in dB, shape
        ``(n_depths, n_ranges)``. Infinite where the field is exactly zero,
        which happens at the source range itself and in the wedge no beam of
        the fan reaches: each beam is summed out to four half-widths, 140 dB
        below its own axis, so a point that far from every one of them is
        outside the traced aperture rather than merely in shadow. The graded
        penumbra just past a limiting ray, which is the part of a shadow zone
        worth having, is finite and carries the beams' tails.
    :ivar pressure: The complex field the loss was taken from, same shape, in
        the module's own :math:`e^{-i\\omega t}` convention (the conjugate of
        the one Jensen Eq. (3.88) is printed in) and normalised to unit
        pressure at 1 m, so ``propagation_loss = -20 lg|pressure|``.
    :ivar launch_angles: Launch angle of each beam's central ray, from the
        horizontal, in degrees.
    :ivar ray_ranges: Range of each central ray at each marching step, in
        metres, shape ``(n_beams, n_steps)``. This is the marching grid, which
        is finer than (and independent of) ``ranges``.
    :ivar ray_depths: Depth of each central ray on that grid, in metres.
    :ivar beam_widths: Beam half-width :math:`W(s)` on that grid, in metres:
        Jensen Eq. (3.89), the distance at which the beam's own pressure has
        fallen by :math:`e^{-1}` and its intensity by :math:`e^{-2}`.
    :ivar wavefront_curvatures: Beam wavefront curvature :math:`K(s)` on that
        grid, in 1/m: Jensen Eq. (3.90) with the sign that belongs to the
        conjugated field this result exposes, so that a beam spreading in free
        space reproduces Eq. (3.85), :math:`K = x/(x^2 + a^2)`, as a positive
        number.
    :ivar initial_beam_width: The :math:`W_0` of Eq. (3.91) actually used, in
        metres, whether it was passed or defaulted.
    :ivar source_depth: Source depth, in metres.
    :ivar water_depth: Water-column depth, in metres.
    """

    frequency: float
    ranges: NDArray[np.float64]
    depths: NDArray[np.float64]
    propagation_loss: NDArray[np.float64]
    pressure: NDArray[np.complex128]
    launch_angles: NDArray[np.float64]
    ray_ranges: NDArray[np.float64]
    ray_depths: NDArray[np.float64]
    beam_widths: NDArray[np.float64]
    wavefront_curvatures: NDArray[np.float64]
    initial_beam_width: float
    source_depth: float
    water_depth: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the propagation-loss field (depth increasing downward)."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_gaussian_beams

        return plot_gaussian_beams(self, ax=ax, language=check_language(language), **kwargs)


#: How many beam half-widths out the transverse Gaussian is still summed.
#: exp(-16) is 1.1e-7 in pressure, 140 dB below the beam's own axis, so even a
#: fan of a thousand beams adding in phase at the cut sits 80 dB under the
#: field. It exists to bound the work, and the work is linear in it: the reach
#: in depth of a beam is this many half-widths over the cosine of its angle,
#: which is what decides both how many cells survive the admission test and how
#: many times the transverse profile wraps the water column.
_BEAM_CUTOFF = 4.0
#: Ceiling on how many times the transverse profile of one beam is allowed to
#: wrap the water column (see :func:`_image_ladder`). Only a fan within a
#: degree of the vertical ever reaches it.
_MAX_BEAM_WRAPS = 512
#: Complex elements one block of the influence sum may hold, which sets how
#: many receiver depths are evaluated at once. 2^20 keeps each temporary at
#: 16 MB, so the peak is tens of megabytes whatever grid is asked for.
_INFLUENCE_BLOCK = 1 << 20
#: Fraction of the water column the steepest beam of the fan may climb in one
#: marching step before the step is reported as too coarse to resolve it.
_MAX_STEEP_CLIMB = 0.25


def _default_beam_width(
    wavelength: float, max_range: float, water_depth: float,
) -> float:
    r"""The :math:`W_0` of Eq. (3.91), from the book's own optimality argument.

    Sect. 3.5.1 does the calculation explicitly for a beam in free space: with
    the waist at the source the half-width evolves as Eq. (3.86),
    :math:`W(x; a) = \sqrt{(2/k)(a + x^2/a)}` with :math:`a = k W_0^2/2`, and
    "differentiating :math:`W(x; a)` with respect to :math:`a` and setting the
    result to 0, we find that the optimal :math:`a` to minimize the beamwidth is
    :math:`a = x`". Evaluated at the far end of the run that is

    .. math::

        W_0 = \sqrt{2 r_\mathrm{max} / k} = \sqrt{\lambda r_\mathrm{max}/\pi},

    the width that resolves the field best where it is resolved worst. It is
    also the width at which the launch-angle integral behind Eq. (3.92) is a
    genuine Gaussian rather than a Fresnel integral: the quadratic coefficient
    of that integral is proportional to :math:`q(0)/q(r)`, whose real part
    vanishes as :math:`q(0)` grows, and the sum then stops converging on a
    truncated fan. Measured against the free field at 100 Hz, the relative error
    in :math:`|p|` is 7.5e-5 at this width at every range tried (500 m, 2 km,
    8 km); it grows to 5e-4 at a fifth of it, where each beam accepts too wide a
    cone of launch angles for the paraxial expansion, and to 2.4e-2 and 3.6e-1
    at six and fifteen times it, where the Fresnel behaviour sets in.

    Two clamps stand around it. The floor of ten wavelengths and the ceiling of
    fifty are the band the book recommends ("typically, this will lead to an
    initial beamwidth of 10-50 wavelengths"), and the formula lands inside them
    on its own across most of the useful parameter space: over a 10 km run it
    gives 14.6 wavelengths at 100 Hz and 46 at 1 kHz. The last clamp is the
    channel, and it has the final word over the other two, because a beam
    comparable to the water depth breaks the bookkeeping that folds a reflected
    ray back into the column rather than merely costing accuracy.
    """
    return float(min(max(np.sqrt(wavelength * max_range / np.pi),
                         10.0 * wavelength),
                     50.0 * wavelength, water_depth / 4.0))


def _image_ladder(
    water_depth: float, bottom_reflection: float, n_wrap: int,
) -> list[tuple[float, float, float]]:
    """``(shift, side, strength)`` of the receiver's images in the folded column.

    The marcher folds a reflected ray back into the water column, which is the
    right thing for the geometry and only half the story for a beam: what folds
    is the *central ray*, while the beam's transverse profile keeps its full
    unfolded extent. A beam of half-width :math:`W` crossing the column at
    :math:`\\theta` to the horizontal spans :math:`W/\\cos\\theta` in depth at
    fixed range, so any beam steep enough, or wide enough, straddles a boundary
    and its folded copies overlap. Summing only the copy nearest the receiver
    throws the rest away, and it does so silently and in one direction: measured
    on the ideal 1000 m guide at 300 Hz against the image-source sum, with the
    fan at 88 degrees, the loss at 2, 5 and 10 km comes out -0.49, +4.50 and
    +4.97 dB off with the nearest copy alone, and +0.0002, +0.0003 and
    -0.0004 dB off with the ladder restored.

    Restoring them is the method of images applied to the receiver rather than
    to the source, which reciprocity allows. In the folded frame the images of a
    receiver at ``z_r`` sit at ``2 l D + z_r`` and ``2 l D - z_r`` for every
    integer ``l``, and the strength of each is the product of the reflection
    coefficients of the boundaries between it and the receiver: the surface
    planes stand at even multiples of ``D`` and the bottom planes at odd ones,
    so counting them gives the exponents below. Both boundary conditions come
    out of that sum identically rather than approximately, for either bottom:
    at ``z_r = 0`` the two families coincide with opposite signs and cancel, and
    at ``z_r = D`` with a rigid bottom they coincide with equal signs, so the
    field doubles and its depth derivative cancels.

    :param n_wrap: How many wraps each way to carry. What it costs is bounded
        per beam rather than globally: :func:`gaussian_beams` admits each beam
        only to the wraps its own reach can populate.
    :return: One entry per image, with ``side`` the sign multiplying ``z_r``.
    """
    surface = _SURFACE_REFLECTION
    ladder = []
    for wrap in range(-n_wrap, n_wrap + 1):
        shift = 2.0 * wrap * water_depth
        ladder.append((shift, 1.0, (surface * bottom_reflection) ** abs(wrap)))
        mirrored = (bottom_reflection**wrap * surface ** (wrap - 1) if wrap >= 1
                    else surface ** (abs(wrap) + 1) * bottom_reflection ** abs(wrap))
        ladder.append((shift, -1.0, mirrored))
    return ladder


class _BeamSamples(NamedTuple):
    """Each beam read at the marching column that brackets a receiver range.

    All the ``(n_beams, n_ranges)`` fields are the march's own history indexed
    at the column nearest each requested range, so the influence sum is
    arithmetic on aligned arrays rather than a search. ``xi`` is
    ``(n_beams, 1)`` and the two range fields are ``(1, n_ranges)``, so they
    broadcast against the rest.

    :ivar weight: :math:`A(\\theta_0)` of Eq. (3.92) times the reflection
        coefficients the central ray has accumulated by that column.
    :ivar phase: The argument of ``spreading``, unwrapped along the ray, which
        is the branch the square root of Eq. (3.88) is taken on.
    :ivar reach: How far in depth, at fixed range, the beam still counts:
        ``_BEAM_CUTOFF`` half-widths divided by the cosine of the local ray
        angle, at the widest point of the ray. Used to admit each beam to as
        many wraps of the column as it can populate and no more.
    """

    xi: NDArray[np.float64]
    column_range: NDArray[np.float64]
    range_offset: NDArray[np.float64]
    depth: NDArray[np.float64]
    vertical: NDArray[np.float64]
    speed: NDArray[np.float64]
    spreading: NDArray[np.complex128]
    slope: NDArray[np.complex128]
    time: NDArray[np.float64]
    phase: NDArray[np.float64]
    weight: NDArray[np.complex128]
    reach: NDArray[np.float64]


def _beam_influence(
    s: _BeamSamples, receiver_depths: NDArray[np.float64], *,
    water_depth: float, bottom_reflection: float, omega: float,
    beam_width: float,
) -> NDArray[np.complex128]:
    r"""Sum Eq. (3.88) over every beam at every point of the receiver grid.

    The ray-centred coordinates of Eqs. (3.182)-(3.183) collapse to arithmetic
    here. ``march_rays`` samples every ray on one uniform range grid and
    :math:`\xi > 0` for every valid ray, so no ray ever reverses in range and
    the containment test of Eq. (3.181) is an index. With the receiver at
    :math:`(r_\mathrm{R}, z_\mathrm{R})`, the ray's sample at
    :math:`(r_j, z_j)`, the offsets :math:`\Delta r`, :math:`\Delta z` between
    them, and the unit tangent and normal of Eqs. (3.25)-(3.26),
    :math:`\mathbf{t} = c\,(\xi, \zeta)` and :math:`\mathbf{n} = c\,(-\zeta,
    \xi)`,

    .. math::

        \frac{s}{c} = \xi\,\Delta r + \zeta\,\Delta z, \qquad
        n = c\,(\xi\,\Delta z - \zeta\,\Delta r),

    and the beam's state follows the central ray to the foot of that
    perpendicular in closed form, because ``p`` is constant between events and
    ``q`` is then a straight line in arc length:

    .. math::

        q_\mathrm{infl} = q_j + c^2 p_j (s/c), \qquad
        \tau_\mathrm{infl} = \tau_j + s/c, \qquad
        r_\mathrm{infl} = r_j + c^2 \xi\,(s/c) .

    That is Eq. (3.184) done exactly rather than by the linear fit the book
    settles for, and the bracketing column is the *nearest* one rather than the
    one below, which halves :math:`|\Delta r|` and with it the only part of the
    step the extrapolation cannot follow, a reflection falling between the
    sample and the receiver. The correction to the travel time is vertical
    slowness times depth offset, the paraxial phase term; dropping it leaves the
    near-horizontal interference pattern wrong.

    THE ONE FACTOR NOTHING ELSE PROTECTS. Eq. (3.88) divides by :math:`r q`,
    which is the Jacobian :math:`J` of Eq. (3.46), and Eq. (3.46) factors it as
    range times ray-tube width. Complex initial data keeps the *second* factor
    off zero for good (see the Wronskian argument above), and that is the whole
    point of the method; the *first* has no such protection, and it vanishes on
    the axis, where every ray of the fan begins. That is the ordinary point
    source singularity the book flags on p. 167, "the amplitude goes to infinity
    as :math:`s \to 0`", and it is not rare here but guaranteed: the foot of the
    perpendicular from a receiver lands exactly on the source for whichever beam
    is launched perpendicular to the source-receiver line, and a fan dense enough
    to sum always has one. Refining the fan aims at it more accurately rather
    than avoiding it. Measured, before this floor was put in, on a deep
    isovelocity column at 100 Hz with everything at its default: a receiver 50 m
    down range and 50 m below the source drew the beam launched at exactly
    -45.0 degrees, whose foot landed at :math:`r = 1.4\times10^{-14}` m, and the
    single term :math:`\sqrt{c/r} = 3.2\times10^{8}` carried the answer to
    :math:`|p| = 8.9\times10^{3}` against an exact :math:`1.4\times10^{-2}`, a
    propagation loss of -79 dB. Over a +-500 m cut the error reached 116 dB and
    it was one-directional, always too loud.

    The floor is one wavelength of the local sound speed, which is Sect. 3.4.2's
    own criterion ("the wavelength should be substantially smaller than any
    physical scale in the problem") applied to the one length that appears here:
    a foot within a wavelength of the axis is a physical scale the ray
    description does not resolve, so the spreading is held at its value there
    rather than allowed to run away. It bounds the term without touching
    anything the method can actually say. That -79 dB cell now reads 37.51 dB
    against an exact 36.99 dB. Measured on the same column, worst error against
    :math:`20\lg R` over a +-500 m cut: 116 dB before the floor and 4.93 dB
    after at 100 m, 108 dB and 0.62 dB at 200 m, 67.7 dB and 0.058 dB at 400 m,
    0.0007 dB before and 0.0006 dB after from 700 m out. On the ideal 1000 m
    guide against the image-source sum at 2, 5 and 10 km the floor is invisible:
    0.00044 dB worst with it and without. Dropping the extrapolation of ``r``
    altogether removes the singularity too, and is *not* the fix, because that
    same guide then comes out 0.0077 dB off instead, seventeen times worse. What
    is left inside a few hundred metres is the method declining to have a near
    field, not this clamp: see :func:`gaussian_beams`.

    Nothing here is an eigenray hunt and nothing interpolates between rays, so
    the interpolation hazard of Sect. 3.7.5.1 (Fig. 3.34) does not arise: the
    beams are summed independently, which is the structural advantage of beam
    tracing over ray interpolation.

    :return: The complex field, shape ``(n_receiver_depths, n_ranges)``, in the
        convention Eq. (3.88) is printed in; the caller conjugates it.
    """
    n_ranges = s.depth.shape[1]
    n_wrap = min(_MAX_BEAM_WRAPS,
                 int(np.ceil(float(s.reach.max()) / (2.0 * water_depth))))
    plan = []
    for shift, side, strength in _image_ladder(water_depth, bottom_reflection, n_wrap):
        # This image sits at ``shift + side*z_r - z_j`` in depth, so with both
        # depths inside the column its offset is at least ``|shift| - D`` away
        # for the upright family and ``|shift| - 2D`` for the mirrored one,
        # whose two depths subtract rather than cancel. A beam that cannot reach
        # that far cannot contribute to the image at any receiver depth and is
        # dropped before a single array is built for it.
        span = water_depth if side > 0.0 else 2.0 * water_depth
        rows = np.flatnonzero(s.reach >= abs(shift) - span)
        if rows.size:
            plan.append((shift, side, strength, rows))

    half_omega_width = 0.5 * omega * beam_width
    cutoff_sq = _BEAM_CUTOFF**2
    field = np.zeros((receiver_depths.size, n_ranges), dtype=np.complex128)
    for shift, side, strength, rows in plan:
        xi = s.xi[rows][:, :, None]
        offset = s.range_offset[:, :, None]
        column = s.column_range[:, :, None]
        depth = s.depth[rows][:, :, None]
        vertical = s.vertical[rows][:, :, None]
        speed2d, slope2d = s.speed[rows], s.slope[rows]
        spread2d, weight2d = s.spreading[rows], s.weight[rows]
        phase2d, time2d = s.phase[rows], s.time[rows]
        speed = speed2d[:, :, None]
        speed_sq = speed**2
        wavelength = 2.0 * np.pi * speed / omega
        spreading = spread2d[:, :, None]
        slope = slope2d[:, :, None]
        # A block of receiver depths at a time, sized so the temporaries stay
        # bounded however large the requested grid is.
        step = max(1, _INFLUENCE_BLOCK // (rows.size * n_ranges))
        for lo in range(0, receiver_depths.size, step):
            zr = receiver_depths[lo:lo + step]
            dz = (shift + side * zr[None, None, :]) - depth
            along = xi * offset + vertical * dz  # s / c
            normal = speed * (xi * dz - vertical * offset)
            q_infl = spreading + speed_sq * slope * along
            # Held off the axis by a wavelength; see the note above on why the
            # cylindrical half of the Jacobian is the one factor nothing else
            # protects.
            r_infl = np.maximum(column + speed_sq * xi * along, wavelength)
            # Admit a cell when n^2/W^2 < cutoff^2 with W = 2|q|/(omega W_0),
            # written as a comparison of squares so that neither side needs a
            # square root: this test runs over the whole block while everything
            # after it runs over the survivors, which in a waveguide are around
            # a third of the cells, so it must stay arithmetic.
            hits = np.flatnonzero(
                ((normal * half_omega_width) ** 2
                 < cutoff_sq * (q_infl.real**2 + q_infl.imag**2)).ravel())
            if hits.size == 0:
                continue
            beam_at, within = np.divmod(hits, n_ranges * zr.size)
            range_at, depth_at = np.divmod(within, zr.size)
            q_hit = q_infl.ravel()[hits]
            spread_hit = spread2d[beam_at, range_at]
            # The travel-time phase, the tracked branch of 1/sqrt(q) and the
            # transverse Gaussian all ride in one exponent rather than three,
            # because a complex exponential over tens of millions of survivors
            # is where the run time goes.
            exponent = (
                -0.5j * (phase2d[beam_at, range_at]
                         + np.angle(q_hit * np.conj(spread_hit)))
                - 1j * omega * (time2d[beam_at, range_at] + along.ravel()[hits]
                                + slope2d[beam_at, range_at] / (2.0 * q_hit)
                                * normal.ravel()[hits] ** 2)
            )
            value = (
                weight2d[beam_at, range_at] * strength
                * np.sqrt(speed2d[beam_at, range_at] / r_infl.ravel()[hits])
                / np.sqrt(np.abs(q_hit)) * np.exp(exponent)
            )
            cells = zr.size * n_ranges
            target = depth_at * n_ranges + range_at
            field[lo:lo + zr.size] += (
                np.bincount(target, value.real, minlength=cells)
                + 1j * np.bincount(target, value.imag, minlength=cells)
            ).reshape(zr.size, n_ranges)
    return field


def _warn_beams(
    message: str,
) -> None:
    """Raise a :class:`~phonometry.PhonometryWarning` from the caller's frame."""
    import warnings

    from ..._internal.warnings import PhonometryWarning

    warnings.warn(f"gaussian_beams: {message}", PhonometryWarning, stacklevel=3)


def _check_source_on_kink(
    z_prof: NDArray[np.float64], c_prof: NDArray[np.float64], source_depth: float,
) -> None:
    """Warn when the source sits on a gradient discontinuity of the profile.

    Sect. 3.7.4 records the artefact: "a further feature of interest is the
    formation of an acoustic jet emanating horizontally from the source when
    piecewise-linear interpolation is used, and when the source is located at
    the discontinuity in the gradient of the sound speed". A measured profile
    handed in with the source at one of its listed depths is the ordinary case
    rather than a corner one, so it is worth saying out loud; moving the source
    by a metre, or handing in a profile without the kink, removes it.
    """
    if z_prof.size < 3:
        return
    grad = np.diff(c_prof) / np.diff(z_prof)
    kinked = z_prof[1:-1][np.diff(grad) != 0.0]
    if kinked.size and np.min(np.abs(kinked - source_depth)) <= 1e-6:
        _warn_beams(
            "'source_depth' sits on a gradient discontinuity of the profile,"
            " which concentrates the near-horizontal beams into a spurious jet;"
            " offset the source or smooth the profile there.")


def gaussian_beams(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    max_range: float = 10_000.0,
    ranges_m: NDArray[np.float64] | list[float] | None = None,
    receiver_depths_m: NDArray[np.float64] | list[float] | None = None,
    n_depth_points: int = 200,
    max_angle_deg: float = 80.0,
    n_beams: int | None = None,
    beam_width: float | None = None,
    range_step: float = 25.0,
    bottom: str = "pressure-release",
) -> GaussianBeamResult:
    r"""Propagation-loss field from Gaussian beam tracing.

    Hangs a Gaussian beam on each ray of a launch fan (Jensen Eq. 3.88) and sums
    them over the fan with the weight of Eq. (3.92). The rays are the ones
    :func:`ray_trace` draws, integrated by the same marcher through the same
    profile; what is added is the dynamic pair :math:`(q, p)` of Eq. (3.58),
    started from the complex conditions of Eq. (3.91) that make each ray the
    axis of a beam of initial half-width ``beam_width`` and flat wavefront.

    The point of the beams is that the answer stays finite. Ray theory's
    amplitude, Eq. (3.65), divides by the ray-tube spreading, which vanishes on
    a caustic and gives an infinity there (Sect. 3.4.1) and nothing at all in a
    shadow zone. Complex :math:`q` cannot vanish, so this field needs no KMAH
    index and no minimum-width floor, is finite wherever a beam reaches, and
    falls into a shadow zone gradually rather than off a cliff, which is what
    the exact solution does (Figs. 3.11, 3.17). See
    :class:`GaussianBeamResult` for the one place it still reports an infinity,
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
      it is converging to. Use :func:`normal_modes` there, which is exact in
      that regime for the cost of two modes.
    * **The fan is truncated** at ``max_angle_deg``, and a waveguide with two
      perfectly reflecting boundaries is the worst case for that, because
      nothing but :math:`1/R` attenuates the steep multiple bounces. Measured on
      the ideal 1000 m guide at 300 Hz, source at 300 m and receiver at 600 m,
      against the image-source sum at 2, 5 and 10 km: a fan to 80 degrees is
      0.27, 4.06 and 2.52 dB out, a fan to 85 degrees 0.21, 1.32 and 1.91 dB,
      and a fan to 88 degrees 0.0002, 0.0003 and 0.0004 dB. Cutting the *oracle*
      to the same half-angle moves it by 0.25, 3.95 and 2.31 dB, so what is left
      at 80 degrees is the fan and not the method. A real seabed absorbs those
      bounces and the default is then ample; a perfect reflector needs the
      fan opened and ``range_step`` cut with it, since a step has to resolve
      :math:`\tan\theta_\mathrm{max}` depth units of climb per unit range. The
      warning below says when that pairing is wrong.
    * **The beam must be small compared to the channel**, which the default
      ``beam_width`` enforces and an explicit one is checked against.

    What it costs is ``n_beams`` times the size of the receiver grid, and none
    of the three factors depends on the frequency: the ray core does not have to
    resolve a wavelength on a grid, and the fan only widens as
    :math:`\lambda/W_0`, which the default width holds nearly fixed. On a
    5000 m Munk column at 100 Hz over 10 km, everything left at its default
    (512 beams, a 200 by 401 field), this takes 14 s against 0.1 s for
    :func:`parabolic_equation` and 177 s for :func:`normal_modes`; raise the
    frequency and the first number stays where it is while the other two climb.
    Shrinking ``n_depth_points`` or handing in a coarser ``ranges_m`` is the
    direct way to trade resolution for time.

    :param frequency_hz: Source frequency, in Hz.
    :param depths: Depth samples of the profile, in metres, from ``z = 0``.
    :param sound_speeds: Sound speed at each depth, in m/s.
    :param source_depth: Source depth, in metres, inside the water column.
    :param max_range: Maximum range to march to, in metres.
    :param ranges_m: Ranges at which to evaluate the field, in metres. Default
        (``None``): the marching grid itself, which puts every receiver on a
        column the rays were actually sampled at.
    :param receiver_depths_m: Depths at which to evaluate the field, in metres.
        Default (``None``): ``n_depth_points`` points spread over the water
        column, on the interior grid :func:`parabolic_equation` uses, so the two
        fields land on the same depths.
    :param n_depth_points: Size of that default depth grid.
    :param max_angle_deg: Half-angle of the launch fan, in degrees from the
        horizontal. Beams are spread symmetrically over
        ``[-max_angle_deg, +max_angle_deg]``.
    :param n_beams: Number of beams in the fan. Default (``None``): from the
        overlap condition. Adjacent beams are :math:`s\,\delta\theta_0` apart at
        arc length :math:`s` while each has spread to
        :math:`W \to s\lambda/(\pi W_0)`, so the condition that they still
        overlap, :math:`\delta\theta_0 \lesssim \lambda/(\pi W_0)`, is
        range-independent; the default takes four times that margin. Too coarse
        a fan shows as a periodic ripple in range at the beam spacing, which is
        easy to mistake for physical interference.
    :param beam_width: The :math:`W_0` of Eq. (3.91), in metres: the beam's
        initial half-width, at the :math:`e^{-2}` folding distance in intensity.
        Default (``None``): see :func:`_default_beam_width`.
    :param range_step: Marching step in range, in metres, and the spacing of the
        default ``ranges_m``.
    :param bottom: ``"pressure-release"`` (default) or ``"rigid"``. The sea
        surface is always pressure-release.
    :return: A :class:`GaussianBeamResult`.
    :raises ValueError: If the inputs are invalid.

    .. warning::

       A :class:`~phonometry.PhonometryWarning` is emitted when the source sits
       on a kink of the profile (Sect. 3.7.4's spurious horizontal jet), when an
       explicit ``beam_width`` exceeds a quarter of the water depth, and when
       one marching step carries the steepest beam of the fan across more than a
       quarter of the water column, which is the pairing between
       ``max_angle_deg`` and ``range_step`` that is easiest to get wrong.
    """
    f = require_positive(frequency_hz, "frequency_hz")
    z_prof, c_prof = _clean_profile(depths, sound_speeds)
    water_depth = float(z_prof[-1])
    zs = float(source_depth)
    if not (0.0 < zs < water_depth):
        raise ValueError("'source_depth' must lie within the water column.")
    rmax = require_positive(max_range, "max_range")
    dr_step = require_positive(range_step, "range_step")
    if dr_step > rmax:
        raise ValueError("'range_step' must not exceed 'max_range'.")
    key = bottom.strip().lower()
    if key not in _BOTTOM_TYPES:
        raise ValueError(f"'bottom' must be one of {_BOTTOM_TYPES}, got {bottom!r}.")
    theta_max = float(max_angle_deg)
    if not (0.0 < theta_max < 90.0):
        raise ValueError("'max_angle_deg' must lie in (0, 90) degrees.")

    omega = 2.0 * np.pi * f
    c0 = float(np.interp(zs, z_prof, c_prof))
    wavelength = c0 / f
    w0 = (_default_beam_width(wavelength, rmax, water_depth) if beam_width is None
          else require_positive(beam_width, "beam_width"))
    if beam_width is not None and w0 > water_depth / 4.0:
        _warn_beams(
            "'beam_width' exceeds a quarter of the water depth, so the beams"
            " straddle boundaries their central rays reflected off and the"
            " folded field drifts from the true one.")
    _check_source_on_kink(z_prof, c_prof, zs)

    span = 2.0 * np.radians(theta_max)
    n_fan = (int(np.ceil(span * 4.0 * np.pi * w0 / wavelength)) + 1
             if n_beams is None else int(n_beams))
    if n_fan < 2:
        raise ValueError("'n_beams' must be at least 2.")
    launch = np.linspace(-np.radians(theta_max), np.radians(theta_max), n_fan)
    dtheta = float(launch[1] - launch[0])

    n_steps = int(np.ceil(rmax / dr_step)) + 1
    dr = rmax / (n_steps - 1)
    xi = np.cos(launch) / c0
    march = march_rays(
        _ocean_ray_derivative(z_prof, c_prof, xi), xi=xi,
        z0=np.full(n_fan, zs), zeta0=np.sin(launch) / c0, range_step=dr,
        n_steps=n_steps, lower=0.0, upper=water_depth,
        dynamic=DynamicRays(np.full(n_fan, 0.5j * omega * w0**2),
                            np.full(n_fan, 1.0 + 0.0j), z_prof, c_prof))

    ranges = np.asarray(
        np.arange(n_steps) * dr if ranges_m is None else ranges_m,
        dtype=np.float64).ravel()
    if ranges.size == 0 or not np.all(np.isfinite(ranges)) or np.any(ranges < 0.0):
        raise ValueError("'ranges_m' must be finite, non-negative and non-empty.")
    # Past the end of the march there is nothing to read a beam off, and the
    # nearest-column arithmetic would answer with a silent extrapolation of the
    # last one rather than with an error.
    if np.any(ranges > rmax + 0.5 * dr):
        raise ValueError("'ranges_m' must not run past 'max_range'.")
    if receiver_depths_m is None:
        n_z = int(n_depth_points)
        if n_z < 2:
            raise ValueError("'n_depth_points' must be at least 2.")
        dz = water_depth / (n_z + 1)
        receivers = np.asarray(dz * np.arange(1, n_z + 1), dtype=np.float64)
    else:
        receivers = np.asarray(receiver_depths_m, dtype=np.float64).ravel()
        if receivers.size == 0 or not np.all(np.isfinite(receivers)):
            raise ValueError("'receiver_depths_m' must be finite and non-empty.")
        # The image ladder folds the receiver about the two boundaries, which
        # only means anything for a receiver between them.
        if np.any(receivers < 0.0) or np.any(receivers > water_depth):
            raise ValueError("'receiver_depths_m' must lie within the water column.")

    climb = dr * np.tan(np.radians(theta_max))
    if climb > _MAX_STEEP_CLIMB * water_depth:
        _warn_beams(
            f"one marching step carries the steepest beam of the fan {climb:.0f} m"
            f" across a {water_depth:.0f} m column, so its trajectory is not"
            " resolved; cut 'range_step' or narrow 'max_angle_deg'.")

    result = _assemble_beam_field(
        march, ranges=ranges, receivers=receivers, launch=launch, xi=xi,
        dtheta=dtheta, dr=dr, z_prof=z_prof, c_prof=c_prof, omega=omega,
        c0=c0, w0=w0, water_depth=water_depth,
        bottom_reflection=_BOTTOM_REFLECTION[key])
    field, widths, curvatures = result

    # Eq. (3.88) is written in the exp(+i omega t) convention; conjugating once
    # here hands back a field in the exp(-i omega t) one the rest of the module
    # speaks. The loss is untouched by that, and the weights of Eq. (3.92)
    # normalise the sum to Eq. (3.80)'s unit pressure at 1 m, so there is no
    # p_0 to divide by.
    pressure = np.conj(field)
    with np.errstate(divide="ignore"):
        pl = -20.0 * np.log10(np.abs(pressure))

    return GaussianBeamResult(
        frequency=f,
        ranges=ranges,
        depths=receivers,
        propagation_loss=np.asarray(pl, dtype=np.float64),
        pressure=np.asarray(pressure, dtype=np.complex128),
        launch_angles=np.degrees(launch),
        ray_ranges=np.broadcast_to(np.arange(n_steps) * dr,
                                   march.positions.shape).copy(),
        ray_depths=march.positions,
        beam_widths=widths,
        wavefront_curvatures=curvatures,
        initial_beam_width=float(w0),
        source_depth=zs,
        water_depth=water_depth,
    )


def _assemble_beam_field(
    march: RayMarch, *, ranges: NDArray[np.float64],
    receivers: NDArray[np.float64], launch: NDArray[np.float64],
    xi: NDArray[np.float64], dtheta: float, dr: float,
    z_prof: NDArray[np.float64], c_prof: NDArray[np.float64],
    omega: float, c0: float, w0: float, water_depth: float,
    bottom_reflection: float,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    r"""Read the march into :class:`_BeamSamples` and sum the beams over it.

    Two per-ray quantities are formed here and nowhere else. The reflection
    factor is the running product of the coefficients the central ray has met,
    which is why the marcher has to say *which* boundary each bounce was at: a
    pressure-release sea surface inverts the pressure and a seabed need not.
    And the branch of :math:`\sqrt{q}` is fixed by unwrapping the argument of
    ``q`` along the ray before anything is sampled off it, since the increment
    per range step is small while a principal-value square root taken sample by
    sample would jump by :math:`\pi` at the first caustic.
    """
    if march.spreadings is None or march.spreading_slopes is None:  # pragma: no cover
        raise ValueError("the march must carry the dynamic ray states.")
    q = np.asarray(march.spreadings, dtype=np.complex128)
    p = np.asarray(march.spreading_slopes, dtype=np.complex128)
    speed = np.interp(march.positions, z_prof, c_prof)
    # W = 2|q|/(omega W_0) is Eq. (3.89) with Im[p/q] replaced by the conserved
    # Wronskian; K is Eq. (3.90) with the sign of the conjugated field.
    widths = 2.0 * np.abs(q) / (omega * w0)
    curvatures = speed * np.real(p / q)

    at_bottom = np.cumsum(march.upper_reflections, axis=1)
    at_surface = np.cumsum(march.reflections - march.upper_reflections, axis=1)
    reflected = (_SURFACE_REFLECTION**at_surface) * (bottom_reflection**at_bottom)
    # A(theta_0) of Eq. (3.92) with Eq. (3.91) substituted in, real and positive.
    weight = dtheta * (omega * w0 / (2.0 * c0)) * np.sqrt(np.cos(launch) / np.pi)

    column = np.clip(np.rint(ranges / dr).astype(np.intp), 0,
                     march.positions.shape[1] - 1)
    cosine = speed[:, column] * xi[:, None]
    samples = _BeamSamples(
        xi=xi[:, None],
        column_range=np.asarray(column * dr, dtype=np.float64)[None, :],
        range_offset=(ranges - column * dr)[None, :],
        depth=march.positions[:, column],
        vertical=march.verticals[:, column],
        speed=speed[:, column],
        spreading=q[:, column],
        slope=p[:, column],
        time=march.times[:, column],
        phase=np.unwrap(np.angle(q), axis=1)[:, column],
        weight=(weight[:, None] * reflected[:, column]).astype(np.complex128),
        reach=_BEAM_CUTOFF * widths[:, column].max(axis=1) / cosine.min(axis=1),
    )
    field = _beam_influence(
        samples, receivers, water_depth=water_depth,
        bottom_reflection=bottom_reflection, omega=omega, beam_width=w0)
    return field, widths, curvatures


# ===========================================================================
# 4. Parabolic equation (Jensen Ch. 6, split-step Fourier)
# ===========================================================================


# ===========================================================================
# 4. Parabolic equation (Jensen Ch. 6, split-step Fourier)
# ===========================================================================


@dataclass(frozen=True)
class ParabolicEquationResult:
    """Parabolic-equation propagation-loss field.

    :ivar frequency: Source frequency, in Hz.
    :ivar ranges: Range grid, in metres.
    :ivar depths: Depth grid, in metres.
    :ivar propagation_loss: Propagation-loss field ``PL(z, r)``, in dB, shape
        ``(n_depths, n_ranges)``.
    :ivar source_depth: Source depth, in metres.
    """

    frequency: float
    ranges: NDArray[np.float64]
    depths: NDArray[np.float64]
    propagation_loss: NDArray[np.float64]
    source_depth: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the propagation-loss field (depth increasing downward)."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_parabolic_equation

        return plot_parabolic_equation(self, ax=ax, language=check_language(language), **kwargs)


def parabolic_equation(
    frequency_hz: float,
    depths: NDArray[np.float64] | list[float],
    sound_speeds: NDArray[np.float64] | list[float],
    *,
    source_depth: float,
    max_range: float = 10_000.0,
    range_step: float = 10.0,
    n_depth_points: int = 1024,
) -> ParabolicEquationResult:
    r"""Propagation-loss field from the standard (Tappert) parabolic
    equation.

    Marches the split-step Fourier solution (Jensen Ch. 6) in range with a
    discrete sine transform in depth, enforcing a pressure-release surface at
    ``z = 0`` and bottom at ``z = water_depth``. The envelope is related to
    pressure by :math:`p = \psi \, e^{i(k_0 r - \pi/4)} / \sqrt{r}` and
    :math:`\mathrm{PL} = -20 \log_{10}(\lvert \psi \rvert / \sqrt{r})`
    (Eqs. 6.70-6.71), using a Gaussian starter.

    The standard PE is **paraxial**: it is accurate for propagation within
    roughly ±15-20° of the horizontal (Jensen §6.2). Steep modes therefore
    carry a phase error that shows at short and intermediate range in
    shallow-waveguide problems (a few dB against the exact field below a few
    water depths of range), converging at long range; the free-field
    calibration itself is exact to ~1e-4 dB at the default ``range_step``.

    :param frequency_hz: Source frequency, in Hz.
    :param depths: Depth samples of the profile, in metres, from ``z = 0``.
    :param sound_speeds: Sound speed at each depth, in m/s.
    :param source_depth: Source depth, in metres.
    :param max_range: Maximum range, in metres.
    :param range_step: Range marching step :math:`\Delta r`, in metres.
    :param n_depth_points: Number of depth points (interior sine-transform grid).
    :return: A :class:`ParabolicEquationResult`.
    :raises ValueError: If the inputs are invalid.
    """
    from scipy.fft import dst, idst

    f = require_positive(frequency_hz, "frequency_hz")
    z_prof, c_prof = _clean_profile(depths, sound_speeds)
    water_depth = float(z_prof[-1])
    zs = float(source_depth)
    if not (0.0 < zs < water_depth):
        raise ValueError("'source_depth' must lie within the water column.")
    rmax = require_positive(max_range, "max_range")
    dr = require_positive(range_step, "range_step")
    if dr > rmax:
        raise ValueError("'range_step' must not exceed 'max_range'.")
    n = int(n_depth_points)
    if n < 16:
        raise ValueError("'n_depth_points' must be at least 16.")

    # Interior depth grid z_j = j·Δz, j = 1..n (pressure-release at 0 and D).
    dz = water_depth / (n + 1)
    z = np.asarray(dz * np.arange(1, n + 1), dtype=np.float64)
    c = np.interp(z, z_prof, c_prof)
    c0 = float(np.interp(zs, z_prof, c_prof))  # reference speed at the source
    k0 = 2.0 * np.pi * f / c0
    nsq = (c0 / c) ** 2  # n²(z) = (c0/c)²

    # Sine-transform vertical wavenumbers: kz_m = mπ/D, m = 1..n.
    kz = np.pi * np.arange(1, n + 1) / water_depth

    # Gaussian starter with a pressure-release surface image (Jensen §6.4.1).
    psi = np.sqrt(k0) * (
        np.exp(-0.5 * k0**2 * (z - zs) ** 2) - np.exp(-0.5 * k0**2 * (z + zs) ** 2)
    )

    # ceil so the range grid always covers max_range even when range_step does
    # not divide it evenly (the last sample may sit just beyond max_range).
    n_r = int(np.ceil(rmax / dr)) + 1
    ranges = np.asarray(np.arange(n_r) * dr, dtype=np.float64)
    pl = np.zeros((n, n_r), dtype=np.float64)
    half_phase = np.exp(0.5j * k0 * (nsq - 1.0) * dr)  # phase screen exp(i k0/2 (n²−1) Δr)
    free_phase = np.exp(-0.5j * kz**2 / k0 * dr)  # free propagation exp(−i kz²/(2k0) Δr)

    # PL = −20·log10(|ψ|/√r) (Eq. 6.71); the √k0 Gaussian starter reproduces
    # free-field spherical spreading (PL = 20·log10 r) exactly.
    for j in range(n_r):
        r = ranges[j]
        if r <= 0.0:
            pl[:, j] = np.inf
        else:
            with np.errstate(divide="ignore"):
                pl[:, j] = -20.0 * np.log10(np.abs(psi) / np.sqrt(r))
        # March one step: phase screen, then free propagation via sine transform.
        psi = half_phase * psi
        spectrum = dst(psi, type=1)
        spectrum = free_phase * spectrum
        psi = idst(spectrum, type=1)

    return ParabolicEquationResult(
        frequency=f,
        ranges=ranges,
        depths=z,
        propagation_loss=np.asarray(pl, dtype=np.float64),
        source_depth=zs,
    )
