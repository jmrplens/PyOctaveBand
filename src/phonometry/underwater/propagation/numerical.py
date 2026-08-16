#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Numerical models of underwater sound propagation (range-independent ocean).

Three complementary numerical solvers for the acoustic field in a
horizontally-stratified ocean waveguide, complementing the closed-form
propagation loss of :mod:`phonometry.underwater.propagation.closed_form`:

* :func:`normal_modes` -- the normal-mode expansion. Solves the depth-separated
  Sturm-Liouville eigenvalue problem by finite differences and assembles the
  propagation loss from the propagating modes.
* :func:`ray_trace` -- ray tracing. Integrates the ray-trajectory equations
  through a sound-speed profile (Runge-Kutta), returning the ray paths and the
  travel time accumulated along each of them.
* :func:`parabolic_equation` -- the standard (Tappert) parabolic equation, solved
  with the split-step Fourier algorithm, returning the propagation-loss field.

All three are implemented clean-room from Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics* (2nd ed., Springer 2011): the modal derivation
(Ch. 5, Eqs. 5.3-5.17), the ray equations (Ch. 3, Eqs. 3.23-3.24) and the
split-step Fourier PE (Ch. 6). They are validated against analytic oracles: the
ideal (pressure-release) waveguide's exact modes, the circular-arc ray paths of
a linear sound-speed gradient together with the closed-form travel time along
them (Medwin & Clay, *Fundamentals of Acoustical Oceanography*, Academic Press
1998, Eq. (3.3.20)), and mutual agreement of the PE and normal-mode
propagation loss for a range-independent waveguide.

Densities are in kg/m3, sound speeds in m/s, depths and ranges in metres,
frequencies in Hz. The water column has a pressure-release surface at z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.rays import march_rays
from ..._internal.validation import require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

_BOTTOM_TYPES = ("pressure-release", "rigid")


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

    # The profile is piecewise linear, so c(z) interpolates exactly and dc/dz
    # is piecewise CONSTANT with jumps at the profile nodes; evaluating the
    # gradient per segment keeps thermocline kinks sharp (a smoothed gradient
    # on an interpolated fine grid biases turning depths by metres).
    seg_grad = np.diff(c_prof) / np.diff(z_prof)

    # March in range r (not arc length): every valid ray then spans [0, rmax] in
    # n_steps regardless of its launch angle. State is (z, ζ, t); ξ = cosθ0/c(zs)
    # is invariant for c(z). From dz/ds, dζ/ds, dt/ds = 1/c and dr/ds = c·ξ:
    #   dz/dr = ζ/ξ,   dζ/dr = −(dc/dz)/(c³·ξ),   dt/dr = 1/(ξ·c²).
    # The time shares the sound speed the other two derivatives already need, so
    # carrying it costs one multiply per stage and inherits the RK4 order: at the
    # default step it reproduces the linear-gradient closed form to ~1e-14 s,
    # where accumulating dr/(ξc²) over the finished path would be first-order.
    ns = int(n_steps)
    ranges = np.linspace(0.0, rmax, ns)
    c0 = float(np.interp(zs, z_prof, c_prof))
    th = np.radians(angles)
    xi = np.cos(th) / c0  # Snell invariant per ray (> 0 since |θ0| < 90°)

    def deriv(
        z_arr: NDArray[np.float64], zeta_arr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        # Vectorised over all rays at once (data-parallel).
        cc = np.interp(z_arr, z_prof, c_prof)
        seg = np.clip(np.searchsorted(z_prof, z_arr, side="right") - 1,
                      0, seg_grad.size - 1)
        return (zeta_arr / xi, -seg_grad[seg] / (cc**3 * xi), 1.0 / (xi * cc**2))

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
# 3. Parabolic equation (Jensen Ch. 6, split-step Fourier)
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
