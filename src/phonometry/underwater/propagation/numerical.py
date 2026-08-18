#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Numerical models of underwater sound propagation (range-independent water,
with an optionally sloping bottom for the ray-based solvers).

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
* :func:`eigenrays` -- the arrival structure. Takes a traced fan and a receiver
  and refines, by bisection on fresh traces, the rays that actually connect the
  source to that receiver, each with its travel time, launch and arrival
  angles, boundary-touch counts and classical complex amplitude: the list the
  sonar equation, a channel impulse response and communications work consume.
* :func:`parabolic_equation` -- the standard (Tappert) parabolic equation, solved
  with the split-step Fourier algorithm, returning the propagation-loss field.

All four are implemented clean-room from Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics* (2nd ed., Springer 2011): the modal derivation
(Ch. 5, Eqs. 5.3-5.17), the ray equations (Ch. 3, Eqs. 3.23-3.24), the Gaussian
beams of Sect. 3.5 (Eqs. 3.88-3.92) and the split-step Fourier PE (Ch. 6). They
are validated against analytic oracles: the ideal (pressure-release) waveguide's
exact modes and its image-source sum, that same image sum over a lossy fluid
seabed with the Rayleigh coefficient of each image's own grazing angle raised
to its count of bottom touches (Jensen Eq. 2.138 with Eq. 3.126 at every
touch), the circular-arc ray paths of a linear
sound-speed gradient together with the closed-form travel time along them
(Medwin & Clay, *Fundamentals of Acoustical Oceanography*, Academic Press 1998,
Eq. (3.3.20)), free-field spherical spreading, mutual agreement of the PE
and normal-mode propagation loss for a range-independent waveguide, and, for
the sloping bottom the two ray-based solvers accept, the ideal wedge's exact
image fan (the folded geometry to eleven digits, the beam field in dB).

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

from ..._internal.rays import DynamicRays, SlopingBoundary, march_rays
from ..._internal.validation import require_positive
from .closed_form import _ABSORPTION_MODELS, _M_PER_KM, seawater_absorption
from .seabed_reflection import reflection_coefficient

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..._internal.rays import RayDerivative, RayMarch

_BOTTOM_TYPES = ("pressure-release", "rigid")
#: What every solver here says when the source is handed to it outside the
#: water column. One string rather than four copies: the three field solvers
#: and the ray tracer all reject the same thing for the same reason, and a
#: caller who has read the message once should not have to notice which of
#: them phrased it.
_SOURCE_OUTSIDE = "'source_depth' must lie within the water column."
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


def _clean_bathymetry(
    bathymetry_ranges_m: NDArray[np.float64] | list[float] | None,
    bathymetry_depths_m: NDArray[np.float64] | list[float] | None,
    z_prof: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """One validated depth(r) polyline out of the pair of arrays, or ``None``.

    The polyline is clamped level beyond its last node (the same
    :func:`numpy.interp` convention the sound-speed profile lives by), so it
    need not reach ``max_range``; but it must start at ``r = 0`` so the water
    column at the source is stated rather than extrapolated, and it may not
    dive below the profile's last node, because the profile *is* the medium
    and a bottom below it would put water where no sound speed was given.
    """
    if bathymetry_ranges_m is None and bathymetry_depths_m is None:
        return None
    if bathymetry_ranges_m is None or bathymetry_depths_m is None:
        raise ValueError(
            "'bathymetry_ranges_m' and 'bathymetry_depths_m' describe one"
            " bottom profile and must be passed together.")
    br = np.asarray(bathymetry_ranges_m, dtype=np.float64).ravel()
    bd = np.asarray(bathymetry_depths_m, dtype=np.float64).ravel()
    if br.size < 2 or bd.shape != br.shape:
        raise ValueError(
            "'bathymetry_ranges_m' and 'bathymetry_depths_m' must be 1-D"
            " arrays of equal length, at least two points.")
    if not (np.all(np.isfinite(br)) and np.all(np.isfinite(bd))):
        raise ValueError("the bathymetry must be finite.")
    if np.any(np.diff(br) <= 0.0):
        raise ValueError("'bathymetry_ranges_m' must be strictly increasing.")
    if abs(float(br[0])) > 1e-9:
        raise ValueError("'bathymetry_ranges_m' must start at the source, r = 0.")
    if np.any(bd <= 0.0):
        raise ValueError(
            "'bathymetry_depths_m' must be strictly positive: the wedge apex"
            " itself, where the water ends, cannot carry a water column.")
    if float(bd.max()) > float(z_prof[-1]) + 1e-9:
        raise ValueError(
            "'bathymetry_depths_m' must not run below the sound-speed"
            " profile: the profile is the medium, so it must reach the"
            " deepest point of the bottom.")
    return br, bd


def _resolve_boundary(
    bottom: str, seabed_density: float | None, seabed_sound_speed: float | None,
    density: float,
) -> tuple[str, tuple[float, float, float] | None]:
    """One bottom description out of the two ways a caller may give one.

    ``bottom`` names a perfect reflector; the ``seabed_density`` /
    ``seabed_sound_speed`` pair names the lossy fluid half-space of
    :func:`~phonometry.underwater.propagation.seabed_reflection.reflection_coefficient`.
    They describe the same boundary, so the pair must arrive whole and must not
    arrive alongside ``bottom="rigid"``. Returns the validated bottom key and
    the ``(water_density, sediment_density, sediment_speed)`` triple, or
    ``None`` for a perfect reflector. Shared by :func:`gaussian_beams` and
    :func:`eigenrays`, which charge the same coefficient to different carriers
    (a beam's running product there, an arrival's amplitude here).
    """
    key = bottom.strip().lower()
    if key not in _BOTTOM_TYPES:
        raise ValueError(f"'bottom' must be one of {_BOTTOM_TYPES}, got {bottom!r}.")
    seabed: tuple[float, float, float] | None = None
    if seabed_density is not None or seabed_sound_speed is not None:
        if seabed_density is None or seabed_sound_speed is None:
            raise ValueError(
                "'seabed_density' and 'seabed_sound_speed' describe one fluid"
                " seabed and must be passed together.")
        if key != "pressure-release":
            raise ValueError(
                "'bottom' and the seabed pair are two descriptions of the same"
                " boundary; leave 'bottom' at its default when passing a fluid"
                " seabed.")
        seabed = (require_positive(density, "density"),
                  require_positive(seabed_density, "seabed_density"),
                  require_positive(seabed_sound_speed, "seabed_sound_speed"))
    return key, seabed


def _seabed_grazing_deg(
    xi: NDArray[np.float64], c_bottom: float,
) -> NDArray[np.float64]:
    """The grazing angle Snell's invariant fixes at the seabed, in degrees.

    :math:`\\cos\\varphi = \\xi\\,c(D)`: the same at every touch of one ray,
    because the direction a ray crosses a depth with is set by the invariant
    and not by how many times it has bounced. A ray that turns above the
    seabed has :math:`\\xi\\,c(D) > 1` (clipped to a grazing angle of zero);
    its bottom count stays zero, so the coefficient it never earned is never
    applied.
    """
    return np.asarray(np.degrees(np.arccos(np.clip(xi * c_bottom, 0.0, 1.0))))


def _ocean_ray_derivative(
    z_prof: NDArray[np.float64], c_prof: NDArray[np.float64],
) -> RayDerivative:
    r"""The ocean's ray equations in range, vectorised over rays.

    March in range :math:`r` (not arc length): every valid ray then spans
    ``[0, max_range]`` in the same number of steps regardless of its launch
    angle. The state is :math:`(z, \zeta, t, s)` and
    :math:`\xi = \cos\theta_0/c(z_\mathrm{s})` is invariant for a
    range-independent :math:`c(z)` between boundary reflections, which is why
    the marcher passes it in per call rather than letting this closure freeze
    it: a level boundary never touches it, and a sloping one rotates it at
    each bounce (see :mod:`phonometry._internal.rays`). From
    :math:`dz/ds`, :math:`d\zeta/ds`, :math:`dt/ds = 1/c` and
    :math:`dr/ds = c\,\xi`,

    .. math::

        \frac{dz}{dr} = \frac{\zeta}{\xi}, \qquad
        \frac{d\zeta}{dr} = -\frac{dc/dz}{c^3 \xi}, \qquad
        \frac{dt}{dr} = \frac{1}{\xi c^2}, \qquad
        \frac{ds}{dr} = \frac{1}{\xi c} .

    The time shares the sound speed the other two derivatives already need, so
    carrying it costs one multiply per stage and inherits the Runge-Kutta order:
    at the default step it reproduces the linear-gradient closed form to
    ~1e-14 s, where accumulating :math:`dr/(\xi c^2)` over the finished path
    would be first order. The arc length rides along on the same argument, and
    is kept as its own expression rather than derived from the time's, so that
    the three states already there come out bit for bit what they were before
    it existed.

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
        z_arr: NDArray[np.float64], zeta_arr: NDArray[np.float64],
        xi_arr: NDArray[np.float64], /
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64]]:
        cc = np.interp(z_arr, z_prof, c_prof)
        seg = np.where(zeta_arr >= 0.0,
                       np.searchsorted(z_prof, z_arr, side="right") - 1,
                       np.searchsorted(z_prof, z_arr, side="left") - 1)
        grad = seg_grad[np.clip(seg, 0, seg_grad.size - 1)]
        return (zeta_arr / xi_arr, -grad / (cc**3 * xi_arr),
                1.0 / (xi_arr * cc**2), 1.0 / (xi_arr * cc))

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
    :ivar arc_lengths: Per-ray cumulative arc length along the ray, in metres,
        same shape (zero at the source). It is never less than the range
        column it stands in, exceeds it by the obliquity of the path, and a
        reflection leaves it continuous. This, and not the range, is the
        measure seawater absorption acts along: Jensen Sect. 3.6.2 carries a
        volume loss :math:`\\alpha` into the ray solution by perturbing the
        eikonal and lands on :math:`e^{-\\int_0^s \\alpha(s')\\,ds'}`
        (Eq. 3.116), an integral over the path actually flown, so a caller
        hanging amplitudes on these rays multiplies by
        :math:`e^{-\\alpha s}` with the :math:`s` read off here.
    :ivar surface_reflections: Per-ray cumulative count of sea-surface
        reflections by each range sample, same shape (zero at the source).
    :ivar bottom_reflections: The same count for the seabed. The two counts,
        and not the reflection coefficients themselves, are the whole of the
        per-bounce record an amplitude carrier needs from the geometry.
        Jensen Sect. 3.6.3 treats a boundary interaction as multiplying the
        ray amplitude by :math:`|\\mathcal{R}(\\theta)|` and adding
        :math:`\\arg \\mathcal{R}(\\theta)` to its phase (Eqs. 3.125-3.126),
        with :math:`\\theta` the local angle of incidence; and in a
        range-independent medium that angle is the *same* at every touch of
        the same flat boundary, because the direction a ray crosses a depth
        with is fixed by Snell's invariant, :math:`\\cos\\theta = \\xi\\,c`,
        not by how many times it has bounced. Any boundary coefficient
        therefore enters a path's amplitude only as :math:`\\mathcal{R}^n`
        with the :math:`n` read off here, which is how
        :func:`gaussian_beams` charges its lossy seabed and how
        :func:`eigenrays` charges each arrival; ``ray_trace`` itself carries
        no amplitude, so the counts are what it can meaningfully expose.
    :ivar source_depth: Source depth, in metres.
    :ivar water_depth: Water-column depth, in metres.
    :ivar profile_depths: Depth samples of the sound-speed profile the rays
        were traced through, in metres, exactly as cleaned on the way in. The
        pair below *is* the medium (both solvers interpolate it piecewise
        linearly and nothing else about the water enters the geometry), so
        recording it makes the result self-contained: :func:`eigenrays` needs
        it to put fresh rays through the same water the fan flew.
    :ivar profile_speeds: Sound speed at each of those depths, in m/s.
    :ivar bathymetry_ranges: Node ranges of the bottom profile the rays were
        traced over, in metres, or ``None`` for the level bottom at
        ``water_depth`` (the default). With a sloping bottom two of the flat
        record's invariants fall, and the arrays here say so honestly rather
        than quietly keep their old meaning. A ray reflected past the vertical
        by the accumulating slope cannot be carried by a range march (see
        :func:`ray_trace`); its ``depths``, ``travel_times`` and
        ``arc_lengths`` are ``NaN`` from the sample of the terminating bounce
        on, so a plot simply ends where the ray turned and nothing downstream
        can mistake a frozen sample for a traced one (the reflection counts,
        being integers, instead hold their last value). And the crossing angle
        at the bottom is no longer one per ray: each slope bounce rotates
        Snell's invariant, so the per-bounce record that sufficed for a flat
        guide (counts alone) does not price a sloping one, which is why
        :func:`eigenrays` declines such a trace.
    :ivar bathymetry_depths: Bottom depth at each of those nodes, in metres
        (``None`` likewise). Between nodes the bottom is the straight facet,
        beyond the last node it continues level: exactly the boundary the
        marcher reflected off.
    """

    launch_angles: NDArray[np.float64]
    ranges: NDArray[np.float64]
    depths: NDArray[np.float64]
    travel_times: NDArray[np.float64]
    arc_lengths: NDArray[np.float64]
    surface_reflections: NDArray[np.int_]
    bottom_reflections: NDArray[np.int_]
    source_depth: float
    water_depth: float
    profile_depths: NDArray[np.float64]
    profile_speeds: NDArray[np.float64]
    bathymetry_ranges: NDArray[np.float64] | None = None
    bathymetry_depths: NDArray[np.float64] | None = None

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
    bathymetry_ranges_m: NDArray[np.float64] | list[float] | None = None,
    bathymetry_depths_m: NDArray[np.float64] | list[float] | None = None,
) -> RayTraceResult:
    r"""Trace acoustic rays through a range-independent sound-speed profile.

    Integrates the ray-trajectory equations (Jensen Eqs. 3.23-3.24) with a
    fixed-step fourth-order Runge-Kutta scheme, reflecting at the pressure-release
    surface (``z = 0``) and the bottom (``z = water_depth``).

    **The bottom may slope.** Passing the ``bathymetry_ranges_m`` /
    ``bathymetry_depths_m`` pair replaces the level bottom with a
    piecewise-linear depth profile ``depth(r)``, the faceted boundary model of
    Jensen Fig. 3.20, and the first range dependence in this module; the
    sound-speed profile stays range independent (see the scope note below).
    The marcher then finds each boundary crossing against the interpolated
    polyline and reflects the ray specularly about the local facet
    (Eq. 3.121), so a bounce off a slope of angle :math:`\beta` changes the
    ray's inclination by :math:`2\beta`: upslope bounces steepen a ray, which
    is the whole one-line physics of wedge propagation, and downslope bounces
    flatten it. Snell's invariant :math:`\xi` is therefore no longer a
    constant of each ray but a constant *between* its bottom bounces, and one
    consequence is drawn honestly rather than papered over: a ray steepened
    past the vertical runs backward in range, which a marcher whose
    independent variable is range cannot carry (the same one-way surgery the
    parabolic equation performs), so such a ray is terminated at that bounce
    and its samples are ``NaN`` from there on -- see
    :attr:`RayTraceResult.bathymetry_ranges`. The polyline continues level
    past its last node exactly as :func:`numpy.interp` clamps the sound-speed
    profile; a bathymetric feature narrower than one range step can hide
    between two samples of the crossing search, so ``n_steps`` must resolve
    the bathymetry as well as the rays.

    **Scope, stated plainly.** Range dependence enters here through the
    boundary alone: full :math:`c(r, z)` is *not* implemented, deliberately.
    Every solver in this module ships with an exact published oracle, and the
    sloping-bottom geometry has one (the ideal wedge unfolds into a closed
    fan of images, which is what the tests hold it to), while a
    range-dependent water column has none: there is no closed form to hold a
    :math:`c(r, z)` marcher to, so it would ship on trust, and this module
    does not ship on trust.

    The travel time is a third state of that same Runge-Kutta step rather than a
    quadrature run over the finished path: with the range-invariant Snell
    parameter :math:`\xi = \cos\theta_0 / c(z_\mathrm{s})` it obeys
    :math:`dt/dr = 1/(\xi c^2)`, so it is integrated with the very stages that
    place the ray and cannot drift from the geometry actually returned. The arc
    length is a fourth state on the same footing, :math:`ds/dr = 1/(\xi c)`,
    because it is the measure volume absorption needs (see
    :class:`RayTraceResult`) and reading it off the finished path would demote
    it to first order. This is
    the same ray core, and the same travel-time equation, as the atmospheric
    :func:`~phonometry.environment.propagation.refraction.atmospheric_ray_paths`
    (which reflects at the ground instead of at the sea surface). Reflections
    cost no time and no path, so both odometers stay continuous across them.
    They are counted, though, per boundary: see :class:`RayTraceResult` on why
    the two cumulative counts, with the crossing angle Snell's invariant fixes
    per ray, are the entire per-bounce record a downstream amplitude needs.

    :param depths: Depth samples of the profile, in metres, from ``z = 0``.
    :param sound_speeds: Sound speed at each depth, in m/s.
    :param source_depth: Source depth, in metres.
    :param launch_angles_deg: Launch angles from the horizontal, in degrees
        (positive downward).
    :param max_range: Maximum horizontal range to trace, in metres.
    :param n_steps: Number of integration steps per ray.
    :param bathymetry_ranges_m: Node ranges of a piecewise-linear bottom
        profile, in metres, strictly increasing from ``r = 0``; level past the
        last node. Default (``None``): the level bottom at the profile's last
        depth. Passed together with ``bathymetry_depths_m``.
    :param bathymetry_depths_m: Bottom depth at each node, in metres, strictly
        positive and never below the sound-speed profile's last depth
        (``None`` likewise).
    :return: A :class:`RayTraceResult`.
    :raises ValueError: If the inputs are invalid.
    """
    z_prof, c_prof = _clean_profile(depths, sound_speeds)
    bathymetry = _clean_bathymetry(bathymetry_ranges_m, bathymetry_depths_m,
                                   z_prof)
    water_depth = float(z_prof[-1])
    depth_at_source = (water_depth if bathymetry is None
                       else float(bathymetry[1][0]))
    zs = float(source_depth)
    if not (0.0 <= zs <= depth_at_source):
        raise ValueError(_SOURCE_OUTSIDE)
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
    deriv = _ocean_ray_derivative(z_prof, c_prof)

    # The marcher splits every range step at the surface or bottom it crosses,
    # so a reflected ray keeps the order the rest of the path is integrated
    # with; see :mod:`phonometry._internal.rays`.
    upper: float | SlopingBoundary = (
        water_depth if bathymetry is None else SlopingBoundary(*bathymetry))
    march = march_rays(deriv, xi=xi, z0=np.full(angles.size, zs),
                       zeta0=np.sin(th) / c0, range_step=rmax / (ns - 1),
                       n_steps=ns, lower=0.0, upper=upper)
    ray_r = np.broadcast_to(ranges, march.positions.shape).copy()

    ray_z, ray_t, ray_s = march.positions, march.times, march.arc_lengths
    if bathymetry is not None and march.stopped_columns is not None and np.any(
            march.stopped_columns < ns):
        # A terminated ray's samples from the stopping bounce on are frozen at
        # a point that is not on those columns' own ranges; NaN says "the ray
        # ended here" in every consumer at once, plots included.
        gone = np.arange(ns)[None, :] >= march.stopped_columns[:, None]
        ray_z = np.where(gone, np.nan, ray_z)
        ray_t = np.where(gone, np.nan, ray_t)
        ray_s = np.where(gone, np.nan, ray_s)

    return RayTraceResult(
        launch_angles=angles,
        ranges=ray_r,
        depths=ray_z,
        travel_times=ray_t,
        arc_lengths=ray_s,
        # The marcher reports per-step bounce counts and says which boundary
        # each was at; accumulated along the ray they become the exponent any
        # per-boundary reflection coefficient enters the amplitude with.
        surface_reflections=np.cumsum(
            march.reflections - march.upper_reflections, axis=1),
        bottom_reflections=np.cumsum(march.upper_reflections, axis=1),
        source_depth=zs,
        water_depth=water_depth,
        profile_depths=z_prof,
        profile_speeds=c_prof,
        bathymetry_ranges=None if bathymetry is None else bathymetry[0],
        bathymetry_depths=None if bathymetry is None else bathymetry[1],
    )


# ===========================================================================
# 2b. Eigenrays and the arrival structure (Jensen Sect. 3.3.5, Eqs. 3.65-3.68)
# ===========================================================================
#
# A traced fan draws every path the profile supports and says nothing about
# which of them pass through a given point. The pressure at a receiver is the
# sum of Eq. (3.66) over precisely the rays that do, "the eigenrays, that is,
# the rays which pass through that point" (Sect. 3.3.5.2), and the list of
# them, each with its delay, its angles and its complex amplitude, is a
# quantity in its own right: it is what the sonar equation consumes as
# multipath structure, what a channel impulse response is made of, and what
# communications work equalises against. The field solvers above collapse that
# structure into one number per grid cell; this section keeps it apart.

#: Bisection iterations refining an eigenray's launch angle inside its fan
#: bracket. Each halving is one fresh march of the bracket set, so the cost is
#: linear in this while the interval shrinks geometrically; the loop leaves as
#: soon as every bracket is converged, which from a fan spaced in hundredths
#: of a radian takes ~35 of the 60. The ceiling only stops a fan spaced in
#: whole radians from spinning.
_EIGENRAY_BISECTIONS = 60
#: Interval width, in radians, at which a bracket is converged: a picoradian.
#: The quantities an arrival reports move linearly with the launch angle at
#: ordinary sensitivities (seconds per radian, and so on), so halving past
#: this buys digits far below the marcher's own discretisation; it is not the
#: float64 limit, which would cost twenty more marches to reach and change
#: nothing an assertion can see.
_EIGENRAY_CONVERGED = 1e-12
#: Refined launch angles closer than this, in radians, are one eigenray found
#: twice: a root landing within roundoff of a fan node gets bracketed from
#: both sides, and bisection then walks both brackets to the same ray.
_EIGENRAY_DISTINCT = 1e-9


@dataclass(frozen=True)
class EigenrayResult:
    r"""The eigenrays connecting one source to one receiver, earliest first.

    Every per-arrival array has one entry per eigenray, sorted by travel time.
    The frequency-independent pieces of each arrival are recorded separately
    (delay, complex amplitude, angles, boundary counts) so that one search
    serves every frequency: in the module's :math:`e^{-i\omega t}` convention
    the pressure a tone of angular frequency :math:`\omega` produces at the
    receiver is

    .. math::

        p(\omega) = \sum_j a_j\, e^{i \omega \tau_j},

    with :math:`a_j` the ``amplitudes`` and :math:`\tau_j` the
    ``travel_times``; the band-limited channel impulse response is the inverse
    transform of that sum, a spike of complex weight :math:`a_j` at each
    :math:`\tau_j`.

    :ivar launch_angles: Launch angle of each eigenray at the source, from the
        horizontal, in degrees, positive downward: the same convention the
        fan was launched with.
    :ivar arrival_angles: The angle each eigenray crosses the receiver with,
        same convention. In a range-independent medium its magnitude is fixed
        by Snell's invariant at the receiver depth; its sign says whether the
        arrival is descending or climbing, which is what a vertical array
        steers on.
    :ivar travel_times: Travel time of each eigenray, in seconds: the
        marcher's third Runge-Kutta state read at the receiver, not a
        quadrature over the finished path.
    :ivar amplitudes: Complex amplitude of each arrival, dimensionless,
        normalised to unit pressure at 1 m from the source (the reference of
        Jensen Eqs. (3.67)-(3.68), the same one every field solver of this
        module reports its loss against). The magnitude is the classical ray
        amplitude of Eq. (3.65),
        :math:`|c(z_\mathrm{R})\cos\theta_0 /
        (c(z_\mathrm{S})\, r\, q(r))|^{1/2}` with :math:`q` integrated from
        the point-source initial conditions of Eq. (3.63); the phase is the
        caustic factor :math:`(-i)^m` of Eq. (3.79) times the boundary
        factors :math:`(-1)^{n_\mathrm{s}}` and :math:`\mathcal{R}^{n_b}`
        (Eqs. 3.125-3.126). See :func:`eigenrays` for why that convention and
        not another.
    :ivar surface_reflections: Sea-surface touches of each eigenray.
    :ivar bottom_reflections: Seabed touches of each eigenray. The pair
        classifies the arrivals the way Fig. 3.7 colours them: refracted
        paths carry zeros, and every multipath family is named by its counts.
    :ivar caustic_crossings: The KMAH index :math:`m` of Eq. (3.79): how many
        times each eigenray's ray-tube spreading vanished on the way, each
        crossing turning the amplitude by :math:`-\pi/2`. Zero for every path
        in an isovelocity channel, where straight rays cannot form caustics.
    :ivar receiver_range: Range of the receiver the list connects to, in m.
    :ivar receiver_depth: Its depth, in metres.
    :ivar source_depth: Source depth, in metres.
    :ivar water_depth: Water-column depth, in metres.
    """

    launch_angles: NDArray[np.float64]
    arrival_angles: NDArray[np.float64]
    travel_times: NDArray[np.float64]
    amplitudes: NDArray[np.complex128]
    surface_reflections: NDArray[np.int_]
    bottom_reflections: NDArray[np.int_]
    caustic_crossings: NDArray[np.int_]
    receiver_range: float
    receiver_depth: float
    source_depth: float
    water_depth: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the arrival structure (per-path loss stems against delay)."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_eigenrays

        return plot_eigenrays(self, ax=ax, language=check_language(language), **kwargs)


def _march_arrival_rays(
    z_prof: NDArray[np.float64], c_prof: NDArray[np.float64], *,
    source_depth: float, thetas: NDArray[np.float64], receiver_range: float,
    n_steps: int,
) -> RayMarch:
    """March candidate eigenrays so the last sample lands on the receiver.

    Geometry and amplitude in one call, which the marcher's own doctrine
    requires (see :mod:`phonometry._internal.rays`): asking for the dynamic
    pair makes the march split its steps at the profile nodes, so a ray traced
    with it is not bit-for-bit the ray traced without, and an amplitude read
    off one while the root was found on the other would answer for a slightly
    different path. Every march of the search therefore carries the pair, with
    the *real* point-source initial conditions of Jensen Eq. (3.63),
    :math:`q(0) = 0`, :math:`p(0) = 1/c(0)`, under which :math:`r\\,q = J`
    (Eq. 3.64) and the classical amplitude of Eq. (3.65) is a read-off. The
    receiver range is the last sample of the march by construction, so nothing
    is interpolated at the point everything is asserted at.
    """
    c0 = float(np.interp(source_depth, z_prof, c_prof))
    xi = np.cos(thetas) / c0
    deriv = _ocean_ray_derivative(z_prof, c_prof)
    return march_rays(
        deriv, xi=xi, z0=np.full(thetas.size, source_depth),
        zeta0=np.sin(thetas) / c0,
        range_step=receiver_range / (n_steps - 1), n_steps=n_steps,
        lower=0.0, upper=float(z_prof[-1]),
        dynamic=DynamicRays(np.zeros(thetas.size),
                            np.full(thetas.size, 1.0 / c0), z_prof, c_prof))


def _caustic_crossings(spreadings: NDArray[np.float64]) -> NDArray[np.int_]:
    """The KMAH index per ray: sign changes of the real spreading history.

    With the initial conditions of Eq. (3.63) the pair is real, so
    :math:`q \\propto J` (Eq. 3.64) and every caustic is a sign change of
    ``q`` along the row: "the number of times J(s) vanishes in [0, s]"
    (Eq. 3.79). The launch sample is q = 0 by those initial conditions and is
    not a caustic, which is why zeros are dropped before the signs are
    compared rather than counted as crossings of their own.
    """
    counts = np.zeros(spreadings.shape[0], dtype=np.int_)
    for i, row in enumerate(spreadings):
        # Exact nonzero mask, deliberately: only the launch sample is q = 0 by
        # construction, and a tolerance would count the small real spreadings
        # either side of a caustic as zeros and miss their sign change.
        signs = np.sign(row[row.astype(bool)])
        counts[i] = int(np.count_nonzero(np.diff(signs)))
    return counts


def _bracket_launches(
    trace: RayTraceResult, r_grid: NDArray[np.float64], r_rec: float,
    z_rec: float,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Read the fan at the receiver range and raise the sign-change brackets.

    Linear between the two bracketing columns: this is only the bracket hunt,
    and every sign found here is re-established on a fresh march before the
    bisection of :func:`_refine_brackets` trusts it.
    """
    launch = np.radians(np.sort(np.asarray(trace.launch_angles,
                                           dtype=np.float64).ravel()))
    col = int(np.clip(np.searchsorted(r_grid, r_rec), 1, r_grid.size - 1))
    w = (r_rec - r_grid[col - 1]) / (r_grid[col] - r_grid[col - 1])
    fan_order = np.argsort(np.asarray(trace.launch_angles,
                                      dtype=np.float64).ravel())
    fan_depth = (trace.depths[fan_order, col - 1] * (1.0 - w)
                 + trace.depths[fan_order, col] * w)
    # A fan ray landing exactly on the receiver depth counts to one side, so
    # the root right under it still raises exactly one bracket.
    sign = np.where(fan_depth == z_rec, 1.0, np.sign(fan_depth - z_rec))
    crossing = np.flatnonzero(sign[:-1] * sign[1:] < 0.0)
    return launch, crossing


def _refine_brackets(
    z_prof: NDArray[np.float64], c_prof: NDArray[np.float64], zs: float,
    launch: NDArray[np.float64], crossing: NDArray[np.intp], *,
    r_rec: float, z_rec: float, ns: int,
) -> NDArray[np.float64]:
    """Close each bracket by bisection on fresh marches through the profile.

    Returns the distinct refined launch angles, possibly none: a bracket the
    re-established endpoints disown is dropped rather than believed.
    """
    lo, hi = launch[crossing], launch[crossing + 1]
    if lo.size == 0:
        return np.zeros(0)
    # The brackets are only as good as the fan's reading of them; make
    # both endpoints real marches before bisecting between them.
    ends = _march_arrival_rays(
        z_prof, c_prof, source_depth=zs,
        thetas=np.concatenate([lo, hi]), receiver_range=r_rec, n_steps=ns)
    f_lo = ends.positions[:lo.size, -1] - z_rec
    f_hi = ends.positions[lo.size:, -1] - z_rec
    # A root standing within the two integrations' disagreement of a fan
    # rung (the fan was traced at its own step, the search marches at ns)
    # can slip just past the endpoint and un-bracket itself; one rung of
    # slack on each side recovers it before the bracket is disbelieved.
    bad = np.flatnonzero(np.sign(f_lo) * np.sign(f_hi) > 0.0)
    if bad.size:
        wide_lo = launch[np.maximum(crossing[bad] - 1, 0)]
        wide_hi = launch[np.minimum(crossing[bad] + 2, launch.size - 1)]
        wide = _march_arrival_rays(
            z_prof, c_prof, source_depth=zs,
            thetas=np.concatenate([wide_lo, wide_hi]),
            receiver_range=r_rec, n_steps=ns)
        lo[bad], hi[bad] = wide_lo, wide_hi
        f_lo[bad] = wide.positions[:bad.size, -1] - z_rec
        f_hi[bad] = wide.positions[bad.size:, -1] - z_rec
    # An endpoint whose marched depth lands on the receiver to the last
    # bit *is* the eigenray; the zero test is exact (a nonzero mask,
    # inverted) because any tolerance would promote near-misses to roots
    # and close their brackets before the bisection has refined them.
    exact_lo, exact_hi = ~f_lo.astype(bool), ~f_hi.astype(bool)
    keep = (np.sign(f_lo) * np.sign(f_hi) < 0.0) | exact_lo | exact_hi
    lo, hi, f_lo = lo[keep], hi[keep], f_lo[keep]
    exact_lo, exact_hi = exact_lo[keep], exact_hi[keep]
    hi = np.where(exact_lo, lo, hi)
    lo = np.where(exact_hi & ~exact_lo, hi, lo)
    s_lo = np.sign(f_lo)
    for _ in range(_EIGENRAY_BISECTIONS):
        open_ = (hi - lo) > _EIGENRAY_CONVERGED
        if not open_.any():
            break
        mid = 0.5 * (lo + hi)
        f_mid = _march_arrival_rays(
            z_prof, c_prof, source_depth=zs, thetas=mid,
            receiver_range=r_rec, n_steps=ns).positions[:, -1] - z_rec
        s_mid = np.sign(f_mid)
        # A midpoint landing exactly on the receiver depth is a root the
        # march itself certified; the exact test (sign code zero, taken as
        # an inverted nonzero mask) keeps it, where a tolerance would
        # declare roots the marcher never confirmed.
        hit = ~s_mid.astype(bool) & open_
        same = (s_mid == s_lo) & open_ & ~hit
        lo = np.where(hit | same, mid, lo)
        hi = np.where(hit | (open_ & ~same), mid, hi)
    refined = 0.5 * (lo + hi)
    if refined.size:
        refined = refined[np.concatenate(
            ([True], np.diff(refined) > _EIGENRAY_DISTINCT))]
    return np.asarray(refined, dtype=np.float64)


def _arrival_bottom_coefficient(
    seabed: tuple[float, float, float] | None, key: str,
    xi: NDArray[np.float64], c_bottom: float,
) -> NDArray[np.complex128]:
    """One bottom coefficient per arrival, at the angle its invariant fixes."""
    if seabed is None:
        return np.full(xi.size, _BOTTOM_REFLECTION[key], dtype=np.complex128)
    rho1, rho2, c2 = seabed
    # As printed, unconjugated: see the docstring's convention note.
    return np.asarray(reflection_coefficient(
        _seabed_grazing_deg(xi, c_bottom), rho1=rho1, c1=c_bottom,
        rho2=rho2, c2=c2), dtype=np.complex128)


def _earliest_arrivals(
    times: NDArray[np.float64], max_arrivals: int,
) -> NDArray[np.intp]:
    """The stable order of the arrival times, truncated to the cap and said."""
    order = np.argsort(times, kind="stable")
    if order.size > int(max_arrivals):
        import warnings

        from ..._internal.warnings import PhonometryWarning

        warnings.warn(
            f"eigenrays: {order.size} eigenrays connect the receiver within"
            f" the traced fan; keeping the {int(max_arrivals)} earliest."
            " Raise 'max_arrivals' to keep them all.",
            PhonometryWarning, stacklevel=3)
        order = order[:int(max_arrivals)]
    return order


def eigenrays(
    trace: RayTraceResult,
    *,
    receiver_range: float,
    receiver_depth: float,
    bottom: str = "pressure-release",
    seabed_density: float | None = None,
    seabed_sound_speed: float | None = None,
    density: float = 1000.0,
    max_arrivals: int = 64,
    n_steps: int | None = None,
) -> EigenrayResult:
    r"""The eigenrays a traced fan brackets between a source and a receiver.

    Finding an eigenray is finding a launch angle whose ray's depth at the
    receiver range equals the receiver depth: a root of
    :math:`f(\theta_0) = z(r_\mathrm{R}; \theta_0) - z_\mathrm{R}`, which is
    continuous in :math:`\theta_0` because a specular reflection folds the
    trajectory continuously. The fan of ``trace`` supplies the brackets, one
    per adjacent pair of rays whose :math:`f` changes sign, and each bracket
    is then closed by bisection on *fresh marches through the same profile*,
    never by interpolating between the traced rays: interpolation across a
    fan is exactly the hazard Jensen Sect. 3.7.5.1 illustrates (two rays of
    one bracket taking different bounce histories have no path between them),
    and a root polished on real traces is a real ray, whose travel time,
    bounce counts and amplitude are its own rather than a blend's. The
    marcher that traces the fan is the marcher that closes the brackets, with
    the dynamic pair of Eq. (3.58) riding along under the real point-source
    initial conditions of Eq. (3.63), so every arrival's amplitude is the
    Jacobian of the very trajectory that hit the receiver.

    **The amplitude convention, stated once.** Each arrival's complex
    amplitude is

    .. math::

        a_j = \left| \frac{c(z_\mathrm{R})\,\cos\theta_0}
        {c(z_\mathrm{S})\, r_\mathrm{R}\, q_j} \right|^{1/2}
        (-i)^{m_j}\, (-1)^{n_{\mathrm{s},j}}\, \mathcal{R}^{n_{b,j}},

    in the module's :math:`e^{-i\omega t}` convention, normalised to unit
    pressure at 1 m. Why each factor:

    * The magnitude is Eq. (3.65) with the :math:`1/(4\pi)` cancelled against
      the free-field reference of Eqs. (3.67)-(3.68), which is how the book
      itself defines transmission loss from these amplitudes and how every
      solver of this module already normalises: the coherent sum
      :math:`\sum_j a_j e^{i\omega\tau_j}` over a complete arrival set is
      directly comparable to :attr:`GaussianBeamResult.pressure`, and
      :math:`-20\lg|{\sum}|` to every propagation loss here.
    * :math:`(-i)^m` is Eq. (3.79) exactly as printed. Sect. 3.3 writes the
      ray field as :math:`A\,e^{i\omega\tau}` (Eq. 3.57), which *is* the
      :math:`e^{-i\omega t}` convention, so the printed factor transfers
      unchanged; it is the same :math:`-\pi/2` per caustic the beam solver's
      tracked square-root branch spends continuously, taken here in the
      discrete form the classical amplitude needs, since with real initial
      conditions :math:`q` passes through zero instead of around it.
    * :math:`(-1)^{n_\mathrm{s}}` and :math:`\mathcal{R}^{n_b}` are
      Eqs. (3.125)-(3.126) applied at every boundary touch, collapsed to
      powers because Snell's invariant fixes one crossing angle per ray at a
      flat boundary. With the ``seabed_density`` / ``seabed_sound_speed``
      pair, :math:`\mathcal{R}` is the Rayleigh coefficient of
      :func:`~phonometry.underwater.propagation.seabed_reflection.reflection_coefficient`
      at that angle, **not conjugated**: that function returns the
      coefficient in the :math:`e^{-i\omega t}` convention these amplitudes
      are declared in, so it enters as printed. (:func:`gaussian_beams`
      conjugates the very same coefficient because its internal sum is
      assembled in the conjugate convention and conjugated once at the end;
      neither solver's choice transfers to the other, which is why both
      spell it out.)

    A receiver standing exactly on a caustic is the one place the list is
    honest rather than useful: :math:`q_j \to 0` there and the classical
    amplitude of Eq. (3.65) diverges, as Sect. 3.4.1 says it must. The
    infinity is ray theory's own, not an artefact to clamp;
    :func:`gaussian_beams` is the solver whose field stays finite there.

    **What the fan resolves is what the search can find.** A bracket exists
    where :math:`f` changes sign between adjacent fan rays, so a pair of
    eigenrays standing between the same two rays (the two sides of a fold
    near a caustic do this first) merges into no bracket at all, and an
    eigenray steeper than the fan's aperture does not exist to it. The fan's
    density and half-angle are the completeness levers, and they belong to
    the caller's :func:`ray_trace`, where they are visible, rather than to a
    hidden retrace here. Tangential contact (an :math:`f` that touches zero
    without crossing) is likewise invisible, which is also why a receiver on
    a boundary is rejected: a folded trajectory only ever grazes the
    boundaries.

    **The cap.** The multipath count grows without bound as paths steepen: in
    an ideal waveguide nothing but spreading attenuates the high-order
    bounces, and every extra :math:`2D` of unfolded depth is two more
    arrivals. ``max_arrivals`` bounds what is returned: the earliest
    ``max_arrivals`` arrivals are kept, because the early paths are the flat,
    least-bounced, least-attenuated ones that carry the energy, and the tail
    being discarded is the part every physical seabed drains fastest. A
    :class:`~phonometry.PhonometryWarning` says when the cap truncated; raise
    it to keep everything the fan bracketed.

    :param trace: A :func:`ray_trace` result: the fan to bracket on, the
        profile to retrace through, and the source the rays leave from.
    :param receiver_range: Receiver range, in metres, within the traced range.
    :param receiver_depth: Receiver depth, in metres, strictly inside the
        water column.
    :param bottom: ``"pressure-release"`` (default) or ``"rigid"``: the
        perfect reflector whose coefficient (:math:`-1` or :math:`+1`) each
        bottom touch multiplies into the amplitude. The sea surface is always
        pressure-release. Superseded by the fluid seabed when the pair below
        is passed. The choice touches amplitudes only; the geometry, times
        and angles of the eigenrays are specular either way.
    :param seabed_density: Sediment density of a lossy fluid seabed (kg/m3 by
        convention; only the ratio to ``density`` enters), passed together
        with ``seabed_sound_speed`` and not alongside ``bottom="rigid"``.
        Default (``None``): the perfect reflector named by ``bottom``.
    :param seabed_sound_speed: Sediment sound speed of that seabed, in m/s
        (``None`` likewise).
    :param density: Water density above the seabed, in kg/m3. Ignored unless
        the seabed pair is passed.
    :param max_arrivals: Most arrivals to return (earliest kept, see above).
    :param n_steps: Range samples per refinement march, receiver column
        included. Default (``None``): the step of ``trace`` itself carried
        over, so the search resolves what the fan resolved.
    :return: An :class:`EigenrayResult`, possibly with zero arrivals: a
        receiver the traced fan never crosses (a shadow zone, or simply
        outside the aperture) has no eigenrays to list, which is an answer
        and not an error.
    :raises ValueError: If the inputs are invalid.
    """
    key, seabed = _resolve_boundary(bottom, seabed_density, seabed_sound_speed,
                                    density)
    if trace.bathymetry_ranges is not None:
        raise ValueError(
            "'trace' was made over a sloping bottom, which this search does"
            " not price: each slope bounce rotates Snell's invariant, so an"
            " arrival's bottom touches no longer share one grazing angle and"
            " the amplitude convention below (one coefficient per ray, raised"
            " to the touch count) stops holding. Trace over a level bottom to"
            " list eigenrays.")
    z_prof = np.asarray(trace.profile_depths, dtype=np.float64)
    c_prof = np.asarray(trace.profile_speeds, dtype=np.float64)
    water_depth = float(trace.water_depth)
    zs = float(trace.source_depth)
    r_rec = require_positive(receiver_range, "receiver_range")
    r_grid = np.asarray(trace.ranges[0], dtype=np.float64)
    if r_rec > float(r_grid[-1]):
        raise ValueError("'receiver_range' must not run past the traced fan.")
    z_rec = float(receiver_depth)
    if not (0.0 < z_rec < water_depth):
        raise ValueError(
            "'receiver_depth' must lie strictly inside the water column: a"
            " folded ray only ever grazes the boundaries, so a receiver on"
            " one is touched tangentially and never crossed.")
    if int(max_arrivals) < 1:
        raise ValueError("'max_arrivals' must be at least 1.")
    if trace.launch_angles.size < 2:
        raise ValueError("'trace' must carry at least two rays to bracket between.")
    if n_steps is None:
        ns = max(2, int(np.ceil(r_rec / float(r_grid[1] - r_grid[0]))) + 1)
    else:
        ns = int(n_steps)
        if ns < 2:
            raise ValueError("'n_steps' must be at least 2.")

    launch, crossing = _bracket_launches(trace, r_grid, r_rec, z_rec)
    refined = _refine_brackets(z_prof, c_prof, zs, launch, crossing,
                               r_rec=r_rec, z_rec=z_rec, ns=ns)

    if refined.size == 0:
        empty = np.zeros(0)
        return EigenrayResult(
            launch_angles=empty, arrival_angles=np.zeros(0),
            travel_times=np.zeros(0),
            amplitudes=np.zeros(0, dtype=np.complex128),
            surface_reflections=np.zeros(0, dtype=np.int_),
            bottom_reflections=np.zeros(0, dtype=np.int_),
            caustic_crossings=np.zeros(0, dtype=np.int_),
            receiver_range=r_rec, receiver_depth=z_rec, source_depth=zs,
            water_depth=water_depth)

    final = _march_arrival_rays(z_prof, c_prof, source_depth=zs,
                                thetas=refined, receiver_range=r_rec,
                                n_steps=ns)
    if final.spreadings is None:  # pragma: no cover
        raise ValueError("the march must carry the dynamic ray states.")
    c0 = float(np.interp(zs, z_prof, c_prof))
    z_end = final.positions[:, -1]
    c_end = np.asarray(np.interp(z_end, z_prof, c_prof))
    q_end = np.asarray(final.spreadings[:, -1], dtype=np.float64)
    # Eq. (3.65) over the 1 m reference of Eqs. (3.67)-(3.68). A receiver on a
    # caustic has q = 0 and honestly infinite classical amplitude; the
    # errstate only keeps numpy from narrating what the docstring already has.
    with np.errstate(divide="ignore"):
        magnitude = np.sqrt(np.abs(
            c_end * np.cos(refined) / (c0 * r_rec * q_end)))
    n_bottom = np.asarray(final.upper_reflections.sum(axis=1))
    n_surface = np.asarray((final.reflections - final.upper_reflections).sum(axis=1))
    kmah = _caustic_crossings(np.asarray(final.spreadings, dtype=np.float64))
    xi = np.cos(refined) / c0
    bottom_coeff = _arrival_bottom_coefficient(seabed, key, xi,
                                               float(c_prof[-1]))
    amplitudes = (magnitude * (-1j) ** kmah
                  * _SURFACE_REFLECTION ** n_surface * bottom_coeff ** n_bottom)
    times = final.times[:, -1]
    arrival = np.degrees(np.arcsin(np.clip(
        final.verticals[:, -1] * c_end, -1.0, 1.0)))

    order = _earliest_arrivals(times, max_arrivals)

    return EigenrayResult(
        launch_angles=np.degrees(refined[order]),
        arrival_angles=np.asarray(arrival)[order],
        travel_times=np.asarray(times)[order],
        amplitudes=np.asarray(amplitudes, dtype=np.complex128)[order],
        surface_reflections=n_surface[order],
        bottom_reflections=n_bottom[order],
        caustic_crossings=kmah[order],
        receiver_range=r_rec,
        receiver_depth=z_rec,
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
# is recorded here rather than in docs/ERRATA.md. The final conjugation has one
# further consequence that stays invisible until a complex coefficient enters
# the sum: everything multiplied into the field before it must be the conjugate
# of its exp(-i omega t) self. The reflection coefficients of the perfect
# boundaries are -1 and +1 and hid this; the complex R of a lossy seabed is
# conjugated on the way in (see :func:`gaussian_beams`).
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
        which happens in the wedge no beam of the fan reaches: each beam is
        summed out to four half-widths, 140 dB below its own axis, so a point
        that far from every one of them is outside the traced aperture rather
        than merely in shadow. The graded penumbra just past a limiting ray,
        which is the part of a shadow zone worth having, is finite and carries
        the beams' tails. Many ordinary cases have no infinity at all: an
        isovelocity 1000 m guide at 300 Hz over 10 km, everything default, has
        none in 80200 cells.

        The source column is **not** one of the infinities, and is not to be
        read. :func:`parabolic_equation` divides by :math:`\\sqrt{r}` and so
        genuinely diverges at ``r = 0``; the beam sum does not, and hands back a
        finite number there instead, 13.6 dB in the case above. It means
        nothing, and neither does anything else within about three initial beam
        widths of the source: see :func:`gaussian_beams` on why this method has
        no near field. The plausible size of these numbers is the point worth
        knowing about them.
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
    :ivar initial_beam_widths: The :math:`W_0` of Eq. (3.91) actually used by
        each beam of the fan, in metres, shape ``(n_beams,)``. An explicit
        ``beam_width`` fills it with one value; the default is per launch
        angle (see :func:`_default_beam_widths`), widest on the axis of the
        fan whenever a shallow channel's modal-resolution term is in play and
        flat across it otherwise.
    :ivar absorption_model: The seawater absorption model applied along the
        beams, or ``None`` when the run propagated without volume absorption
        (the default).
    :ivar absorption_coefficient: The absorption coefficient :math:`\\alpha`
        actually applied, in dB/km (0.0 when ``absorption_model`` is ``None``),
        as :func:`~phonometry.underwater.propagation.closed_form.seawater_absorption`
        evaluated it at the source frequency and depth. Recorded so a run's
        loss can be decomposed without re-deriving what was subtracted.
    :ivar seabed_density: Sediment density of the fluid seabed the bottom
        bounces were charged with, or ``None`` when the bottom was one of the
        perfect reflectors (the default).
    :ivar seabed_sound_speed: Sediment sound speed of that seabed, in m/s, or
        ``None`` likewise. Together the pair names the Rayleigh interface of
        :func:`~phonometry.underwater.propagation.seabed_reflection.reflection_coefficient`
        each beam's bottom reflections multiplied it by.
    :ivar source_depth: Source depth, in metres.
    :ivar water_depth: Water-column depth, in metres: the sound-speed
        profile's last depth, which over a sloping bottom is the deepest
        water the medium description reaches while the column itself is the
        bathymetry below.
    :ivar bathymetry_ranges: Node ranges of the bottom profile the run was
        marched over, in metres, or ``None`` for the level bottom (the
        default). When present, the per-beam histories (``ray_depths``,
        ``beam_widths``, ``wavefront_curvatures``) are ``NaN`` from the
        column at which a beam was terminated by a reflection past the
        vertical, the same convention :class:`RayTraceResult` uses and for
        the same reason: from there on the beam no longer exists in the
        forward field, and its weight in the sum is zero.
    :ivar bathymetry_depths: Bottom depth at each of those nodes, in metres
        (``None`` likewise).
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
    initial_beam_widths: NDArray[np.float64]
    absorption_model: str | None
    absorption_coefficient: float
    seabed_density: float | None
    seabed_sound_speed: float | None
    source_depth: float
    water_depth: float
    bathymetry_ranges: NDArray[np.float64] | None = None
    bathymetry_depths: NDArray[np.float64] | None = None

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
#: How far from its sample a beam's analytic tail may still be evaluated at a
#: *sloped* receiver column, in multiples of the marched range extent. The fan
#: ladder has no admission floor on a slope (see :func:`_fold_margins`), so
#: this budget is what stands between the influence sum and pricing wedge
#: geometry the caller never described: a wrapped rung's image is the local
#: facet's wedge continued clear around its apex, and where that apex stands
#: tens of extents beyond the march (a locally tilted bottom in deep water),
#: the arrivals those tails would reconstruct bounce on pure fiction --
#: measured at up to 11 dB of pollution against a two-path oracle whose bottom
#: no beam can even reach, against 0.001 dB with the budget in place. Two
#: extents is the up-and-back allowance: the farthest genuine rung of the
#: ideal-wedge oracle, a return leg that turns just past the far edge of the
#: march, stands 1.9 extents out and is untouched (the oracle's numbers are
#: identical with the budget at 2 or at 4), while one extent alone clips it
#: and costs that field over 2 dB. Level columns are never capped, which is
#: part of why a level polyline keeps the flat ladder bit for bit.
_TAIL_TRUST = 2.0


def _default_beam_widths(
    wavelength: float, max_range: float, water_depth: float,
    launch: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""The :math:`W_0` of Eq. (3.91) per launch angle: the free-space optimum,
    raised in a shallow channel to the width that resolves the guide's modes.

    **The free-space term.** Sect. 3.5.1 does the calculation explicitly for a
    beam in free space: with the waist at the source the half-width evolves as
    Eq. (3.86), :math:`W(x; a) = \sqrt{(2/k)(a + x^2/a)}` with
    :math:`a = k W_0^2/2`, and "differentiating :math:`W(x; a)` with respect
    to :math:`a` and setting the result to 0, we find that the optimal
    :math:`a` to minimize the beamwidth is :math:`a = x`". Evaluated at the
    far end of the run that is

    .. math::

        W_0 = \sqrt{2 r_\mathrm{max} / k} = \sqrt{\lambda r_\mathrm{max}/\pi},

    the width that resolves the field best where it is resolved worst. It is
    also the width at which the launch-angle integral behind Eq. (3.92) is a
    genuine Gaussian rather than a Fresnel integral: the quadratic coefficient
    of that integral is proportional to :math:`q(0)/q(r)`, whose real part
    vanishes as :math:`q(0)` grows, and the sum then stops converging on a
    truncated fan. Measured against the free field at 100 Hz at 2, 5 and 8 km,
    the relative error in :math:`|p|` is 7.5e-5 at this width; it is 2.7e-2 at
    a fifth of it, where each beam accepts too wide a cone of launch angles for
    the paraxial expansion, and 3.7e-3 and 4.1e-2 at six and fifteen times it,
    where the Fresnel behaviour sets in. The shallowest part of the curve is a
    little above the formula rather than on it, 1.9e-5 at twice the width,
    which is as close to the optimum as this measurement can place it. (The
    range set matters: taken on a set reaching in to 500 m the last two
    numbers come out 5.4e-2 and 6.3e-2, because a wide beam is exactly what
    pushes the perpendicular feet of the fan in towards the source; that is
    the axis floor of :func:`_beam_influence` being read as a property of the
    width, and it is not one. The same measurement is why widening a beam
    beyond what a criterion asks for is never free.)

    **The guide term, and why it is per launch angle.** A waveguide's trapped
    field is a discrete sum of modes, and mode :math:`m` of a channel of depth
    :math:`D` stands at the launch angle :math:`\sin\theta_m = m\lambda/(2D)`
    (its vertical wavenumber is :math:`m\pi/D`; Jensen Eq. 5.13's
    :math:`\sin(m\pi z/D)` says the same thing for the ideal guide, and the
    spacing is asymptotically the same for a refracting one). Adjacent modes
    are therefore :math:`\delta(\sin\theta) = \lambda/(2D)` apart, which at
    the launch angle :math:`\theta_0` is a gap of
    :math:`\lambda/(2D\cos\theta_0)` in angle; and a beam of initial width
    :math:`W_0` cannot tell launch angles apart more finely than its own
    far-field divergence, :math:`\lambda/(\pi W_0)` by Eq. (3.86). Asking the
    beam to resolve its neighbours' gap to half, so that the modal
    interference a receiver kilometres out actually shows, gives

    .. math::

        \frac{\lambda}{\pi W_0} \le \frac{1}{2}\,
        \frac{\lambda}{2 D \cos\theta_0}
        \quad\Longleftrightarrow\quad
        W_0(\theta_0) \ge \frac{4 D \cos\theta_0}{\pi},

    one width per launch angle, widest for the flat beams (whose modes crowd
    together in angle) and relaxing by the cosine for the steep ones (whose
    modes stand apart). Its vertical footprint :math:`W_0/\cos\theta_0` is the
    constant :math:`4D/\pi`: every beam of the fan spans the one column. The
    half-a-gap margin is a choice, and the oracle brackets it: on an
    :math:`n^2`-linear 200 m guide at 200 Hz, against the exact Airy modes
    (energy-averaged over 0.5 to 4 km), this default measures +0.19 dB in the
    mean where the free-space optimum alone (100 m) is +1.15 dB, a full-gap
    margin leaves +1.12 dB, a third of a gap overshoots to -0.62 dB, and flat
    300 and 400 m widths to -0.91 and -1.18 dB; a 100 m guide at 250 Hz walks
    the same ladder (+0.74, +0.39, +0.14 dB at full, half and a third of a
    gap) without the overshoot, so half the gap is where the two cases agree.

    The guide term stands only where the book's own band can hold it: for
    :math:`4D/\pi > 50\lambda` (deep water, in wavelengths) no admissible
    width resolves the modes, and rather than pinning every beam to the
    ceiling, which the free-space measurement above prices at a percent-level
    error for the refracted paths that dominate a deep field, the default
    falls back to the free-space optimum for the whole fan. The changeover is
    deliberately all-or-nothing: a criterion that cannot be met inside the
    band is not met halfway.

    **The band.** The floor of ten wavelengths and the ceiling of fifty are
    the band the book recommends ("typically, this will lead to an initial
    beamwidth of 10-50 wavelengths"), and the free-space term lands inside
    them on its own across most of the useful parameter space: over a 10 km
    run it gives 14.6 wavelengths at 100 Hz and 46 at 1 kHz.

    **What the text supports, and what it cost here.** A quarter-depth cap
    used to stand over all of this, :math:`W_0 \le D/4` whatever the angle,
    and its retirement is worth the paragraph. The text following Eq. (3.91)
    does ask for beams "not large compared to the water depth" (while calling
    the choice "a matter of current research"), and p. 184 warns that a beam
    large compared to the channel "causes a variety of problems": intuition
    that presumes a beam summed only in the folded column, where a wide beam's
    tail straddles the boundaries and is lost. The folded-image ladder of
    :func:`_image_ladder` restores exactly what that folding drops, and with
    it in place the measurements come down against the cap: on the 200 m guide
    above the cap's 50 m width is +3.08 dB in the mean, one-sided and silent,
    against this default's +0.19 dB; and on the isovelocity 200 m guide at
    50 Hz (source 30.5 m, receiver 120.5 m, the same range window), where the
    cap even undercut the book's own ten-wavelength floor (50 m against 300),
    it costs +0.17 dB against the exact modal sum where the floor's 300 m
    measures -0.001 dB. What survives of the water-depth intuition is its
    geometry, made per-angle: the guide term holds every beam's *vertical
    footprint* at :math:`4D/\pi`, about the water depth -- the cap misread
    that footprint as the width itself, and charged the steep beams, which
    fit the column many times over, the same toll as the flat ones.
    """
    widths = np.full(launch.shape, np.sqrt(wavelength * max_range / np.pi))
    guide = 4.0 * water_depth / np.pi
    if guide <= 50.0 * wavelength:
        widths = np.maximum(widths, guide * np.cos(launch))
    return np.asarray(np.clip(widths, 10.0 * wavelength, 50.0 * wavelength))


def _image_ladder(
    n_wrap: int,
) -> list[tuple[int, float, int, int]]:
    """The receiver's images in the folded column, as boundary-touch counts.

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
    and *counting* them is all that happens here. The counts are returned
    rather than the product, because the bottom's coefficient need not be a
    number: a lossy seabed's :math:`\\mathcal{R}` depends on the grazing angle
    and so differs beam by beam, while the count of planes between an image and
    the receiver is geometry and differs only rung by rung.
    :func:`_beam_influence` raises each beam's own coefficient to these
    exponents, which for the unfolded beam is exact in an isovelocity column:
    every bottom plane the straight unfolded beam crosses, it crosses at the
    one angle its central ray makes with the horizontal. Both boundary
    conditions come out of the resulting sum identically rather than
    approximately, for either perfect bottom: at ``z_r = 0`` the two families
    coincide with opposite signs and cancel, and at ``z_r = D`` with a rigid
    bottom they coincide with equal signs, so the field doubles and its depth
    derivative cancels.

    With a sloping bottom there is no single ``D`` to fold at, and the rungs
    returned here are the dimensionless part of the ladder only, the wrap
    count and the two exponents: :func:`_beam_influence` prices the level
    columns as ``shift = 2 l D`` and the sloping ones as the dihedral fan of
    :func:`_fold_images`, whose rung ``(l, side)`` is the rotation of the
    receiver by ``2 l`` facet angles about the local apex. The counts carry
    over unchanged because the fan's plane-crossing structure is the stack's:
    the image at angle :math:`2l\\beta \\pm \\gamma` stands behind exactly
    ``l`` bottom planes and ``l`` (or ``l - 1``) surface planes, the same
    words the level ladder counts, which is also where the sign consistency
    condition (an even number of facets in the half turn) comes from.

    :param n_wrap: How many wraps each way to carry. What it costs is bounded
        per beam rather than globally: :func:`gaussian_beams` admits each beam
        only to the wraps its own reach can populate.
    :return: One ``(wrap, side, n_surface, n_bottom)`` entry per image, with
        ``wrap`` the integer ``l`` of the fold ``2 l D``, ``side`` the sign
        multiplying ``z_r`` and the two counts the exponents of the surface
        and bottom reflection coefficients.
    """
    ladder = []
    for wrap in range(-n_wrap, n_wrap + 1):
        ladder.append((wrap, 1.0, abs(wrap), abs(wrap)))
        mirrored = ((wrap - 1, wrap) if wrap >= 1
                    else (abs(wrap) + 1, abs(wrap)))
        ladder.append((wrap, -1.0, *mirrored))
    return ladder


class _BeamSamples(NamedTuple):
    """Each beam read at the marching column that brackets a receiver range.

    All the ``(n_beams, n_ranges)`` fields are the march's own history indexed
    at the column nearest each requested range, so the influence sum is
    arithmetic on aligned arrays rather than a search. The two range fields
    are ``(1, n_ranges)``, so they broadcast against the rest. ``xi`` is the
    marched horizontal slowness read per column like the other state, because
    a sloping bottom rotates it at each bounce; over a level bottom every row
    is its launch value repeated.

    :ivar weight: :math:`A(\\theta_0)` of Eq. (3.92) times the reflection
        coefficients the central ray has accumulated by that column.
    :ivar phase: The argument of ``spreading``, unwrapped along the ray, which
        is the branch the square root of Eq. (3.88) is taken on.
    :ivar path: Cumulative arc length of the central ray at the column, in
        metres: the odometer the marcher integrated with the very stages that
        placed the ray, which is what a volume absorption multiplies on.
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
    path: NDArray[np.float64]
    phase: NDArray[np.float64]
    weight: NDArray[np.complex128]
    reach: NDArray[np.float64]


class _FoldColumns(NamedTuple):
    """The local bottom each receiver column folds its images at.

    One entry per receiver range: the receiver's own range, the bottom depth
    there and the facet slope there (the facet ahead at a vertex, the level
    clamp beyond the polyline's ends: the marcher's own conventions). A
    column with slope zero folds like the level guide, bit for bit.
    """

    ranges: NDArray[np.float64]
    depths: NDArray[np.float64]
    slopes: NDArray[np.float64]


def _fold_images(
    fold: _FoldColumns, cols: NDArray[np.intp], wrap: int, side: float,
    zr: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Receiver images of one ladder rung, per (column, receiver depth).

    Where the column's facet is level this is the flat ladder's image,
    ``z = 2 l D + side z_r`` at the receiver's own range, to the last bit.
    Where it slopes, the surface plane and the extended facet plane meet at a
    local apex, and the compositions of their two mirrors are the dihedral
    fan about it: the rung ``(l, side)`` image sits at polar angle
    :math:`2 l \\beta + \\mathrm{side}\\,\\gamma` on the receiver's own circle
    of radius :math:`\\rho` about that apex, with :math:`\\beta` the facet
    angle and :math:`(\\rho, \\gamma)` the receiver's polar coordinates. For
    a single facet that is the exact unfolding (it is how the ideal wedge's
    closed-form image fan is built), which is what lets the tail sum keep the
    flat ladder's accuracy on a slope; a polyline of several facets makes it
    local to each column's facet, the honest first order in facet changes.
    Both slope signs map onto one canonical wedge by mirroring range about
    the column, which preserves every distance the influence sum consumes.

    :return: ``(z_i, r_i)`` arrays of shape ``(cols.size, zr.size)``: the
        image positions in the water-column plane.
    """
    d_col = fold.depths[cols][:, None]
    m_col = fold.slopes[cols][:, None]
    r_col = fold.ranges[cols][:, None]
    # Exact nonzero mask: a level facet has slope exactly 0.0 by construction
    # (the polyline's own differences, or the level clamp past its ends), and
    # a tolerance would fold gently sloping facets as if they were flat --
    # small slopes are what a finely sampled bathymetry is made of.
    sloped = m_col.astype(bool)
    z_flat = 2.0 * wrap * d_col + side * zr[None, :]
    # Distance from the local apex to the column along the surface, guarded
    # where the facet is level (the apex is then at infinity and the flat
    # branch above is the one selected).
    x_col = d_col / np.abs(np.where(sloped, m_col, 1.0))
    beta = np.arctan(np.abs(m_col))
    rho = np.hypot(x_col, zr[None, :])
    gamma = np.arctan2(zr[None, :], x_col)
    angle = 2.0 * wrap * beta + side * gamma
    z_i = np.where(sloped, rho * np.sin(angle), z_flat)
    r_i = np.where(sloped,
                   r_col + np.sign(m_col) * (rho * np.cos(angle) - x_col),
                   r_col)
    return z_i, r_i


def _fold_margins(
    fold: _FoldColumns, wrap: int, side: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Per-column admission floor and validity of one ladder rung.

    The floor is a lower bound on the *transverse* distance from the rung's
    images to the beams, compared against the beams' reach so a rung no beam
    can populate is never built. Level columns use the flat ladder's exact
    ``|2 l D| - span``, which is transverse because a level fold displaces
    the image in depth alone. Sloping columns get **no** floor at all, and
    the reason is worth a sentence: a fold of the fan displaces the image in
    range as much as in depth, and an image kilometres away *along* a beam's
    axis is at zero transverse distance -- it is exactly how a wrapped tail
    represents an arrival that went up the slope, turned past the vertical
    and came back, which the marched axis cannot do but the analytic tail
    does. An earlier chord-distance floor here silently pruned those
    return-leg images and measured whole decibels against the ideal wedge;
    only the per-cell admission test can prune a fan rung. What keeps that
    freedom honest is the tail budget of :data:`_TAIL_TRUST`, applied inside
    the per-cell test itself: a wrapped tail may carry an up-and-back
    arrival, but no farther out than the march could have carried the beam
    there and back, so a rung whose images stand on wedge geometry beyond
    anything the caller described is refused by every beam at once.

    Validity is the fan closing on itself: a wedge of angle :math:`\\beta`
    has :math:`2\\pi/\\beta` distinct images in the whole circle, so a rung
    rotated past :math:`\\pi` re-enters from the other side and would count
    an image twice; it is dropped, and at exactly :math:`\\pi` (the seam,
    where the two directions land on the same image) only the positive wrap
    keeps it. Level columns have no fan to close and every rung is valid.
    """
    d_col = fold.depths
    m_col = fold.slopes
    # The same exact nonzero mask as :func:`_fold_images`, for the same
    # physics: level means slope exactly 0.0, and a tolerance would hand a
    # gently sloping column the flat ladder's floor it is not entitled to.
    sloped = m_col.astype(bool)
    span = d_col if side > 0.0 else 2.0 * d_col
    flat_margin = np.abs(2.0 * wrap * d_col) - span
    beta = np.arctan(np.abs(m_col))
    margin = np.where(sloped, 0.0, flat_margin)
    turned = 2.0 * abs(wrap) * beta
    valid = ~sloped | (turned < np.pi - 1e-9) | (
        (np.abs(turned - np.pi) <= 1e-9) & (wrap > 0))
    return margin, valid


class _Influence(NamedTuple):
    """What every rung of one influence sum shares.

    The receiver grid, the constants of Eq. (3.88) and the sloped-column tail
    budget travel together so the rung workers take a context rather than an
    argument list. ``capped`` and ``tail_limit`` are the fold's tail budget
    (see :func:`_beam_influence` on :data:`_TAIL_TRUST`): ``None`` and an
    unused limit over a level bottom, where no tail needs a leash.
    """

    receiver_depths: NDArray[np.float64]
    water_depth: float
    omega: float
    attenuation: float
    fold: _FoldColumns | None
    half_omega_width: NDArray[np.float64]
    cutoff_sq: float
    capped: NDArray[np.bool_] | None
    tail_limit: float


def _wrap_count(
    s: _BeamSamples, water_depth: float, fold: _FoldColumns | None,
) -> int:
    """How many rungs each way of the ladder the widest reach demands."""
    if fold is None:
        return min(_MAX_BEAM_WRAPS,
                   int(np.ceil(float(s.reach.max()) / (2.0 * water_depth))))
    # Level columns ask for the depth-stack count the reach implies; the
    # sloped ones ask for the whole half fan, seam included, because the
    # wrapped tails carry the up-and-back arrivals whatever the reach.
    n_wrap = int(np.ceil(float(s.reach.max()) / (2.0 * fold.depths.min())))
    sloped = np.abs(fold.slopes) > 0.0
    if np.any(sloped):
        beta_min = float(np.arctan(np.abs(fold.slopes[sloped])).min())
        n_wrap = max(n_wrap, int(np.ceil(0.5 * np.pi / beta_min)) + 1)
    return min(_MAX_BEAM_WRAPS, n_wrap)


def _rung_plan(
    s: _BeamSamples, fold: _FoldColumns | None, water_depth: float,
    r_bottom: NDArray[Any], n_wrap: int,
) -> list[tuple[int, float, NDArray[Any], NDArray[np.intp],
                NDArray[np.intp] | None]]:
    """The rungs some beam can populate: strength, rows and columns of each.

    Each image sits at ``shift + side*z_r - z_j`` in depth, so with both
    depths inside the column its offset is at least ``|shift| - D`` away
    for the upright family and ``|shift| - 2D`` for the mirrored one,
    whose two depths subtract rather than cancel. A beam that cannot reach
    that far cannot contribute to the image at any receiver depth and is
    dropped before a single array is built for it. On a slope the floor
    and the rung's validity come from the local fan instead.
    """
    reach_max = float(s.reach.max())
    plan: list[tuple[int, float, NDArray[Any], NDArray[np.intp],
                     NDArray[np.intp] | None]] = []
    for wrap, side, n_surface, n_bottom in _image_ladder(n_wrap):
        cols = None
        if fold is None:
            shift = 2.0 * wrap * water_depth
            span = water_depth if side > 0.0 else 2.0 * water_depth
            rows = np.flatnonzero(s.reach >= abs(shift) - span)
        else:
            margin, valid = _fold_margins(fold, wrap, side)
            usable = valid & (margin <= reach_max)
            if not usable.all():
                cols = np.flatnonzero(usable)
                if cols.size == 0:
                    continue
            rows = np.flatnonzero(s.reach >= float(margin[usable].min()))
        if rows.size == 0:
            continue
        strength = _SURFACE_REFLECTION**n_surface * r_bottom[rows]**n_bottom
        plan.append((wrap, side, strength, rows, cols))
    return plan


def _rung_samples(
    s: _BeamSamples, rows: NDArray[np.intp], cols: NDArray[np.intp] | None,
) -> _BeamSamples:
    """The march history of ``rows``, at every column or at ``cols`` alone.

    A rung only some columns can populate is priced on those columns alone;
    ``np.ix_`` gathers the (row, column) cross product once. The subset is a
    :class:`_BeamSamples` again, so the block worker reads whichever subset
    it is handed with the same words.
    """
    if cols is None:
        return _BeamSamples(
            xi=s.xi[rows], column_range=s.column_range,
            range_offset=s.range_offset, depth=s.depth[rows],
            vertical=s.vertical[rows], speed=s.speed[rows],
            spreading=s.spreading[rows], slope=s.slope[rows],
            time=s.time[rows], path=s.path[rows], phase=s.phase[rows],
            weight=s.weight[rows], reach=s.reach[rows])
    sub = np.ix_(rows, cols)
    return _BeamSamples(
        xi=s.xi[sub], column_range=s.column_range[:, cols],
        range_offset=s.range_offset[:, cols], depth=s.depth[sub],
        vertical=s.vertical[sub], speed=s.speed[sub],
        spreading=s.spreading[sub], slope=s.slope[sub],
        time=s.time[sub], path=s.path[sub], phase=s.phase[sub],
        weight=s.weight[sub], reach=s.reach[rows])


def _image_offsets(
    grid: _Influence, col_index: NDArray[np.intp], wrap: int, side: float,
    zr: NDArray[np.float64], depth: NDArray[np.float64],
    offset: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Depth and range offsets from each beam sample to one rung's images."""
    if grid.fold is None:
        shift = 2.0 * wrap * grid.water_depth
        return (shift + side * zr[None, None, :]) - depth, offset
    z_i, r_i = _fold_images(grid.fold, col_index, wrap, side, zr)
    dz = z_i[None, :, :] - depth
    # The image's own range enters through the offset to the sample column; a
    # level column reduces to the receiver's.
    d_r = offset + (r_i - grid.fold.ranges[col_index][:, None])[None, :, :]
    return dz, d_r


def _survivor_block(
    r: _BeamSamples, grid: _Influence, strength: NDArray[Any], *,
    admitted: NDArray[np.bool_], q_infl: NDArray[np.complex128],
    r_infl: NDArray[np.float64], along: NDArray[np.float64],
    normal: NDArray[np.float64], n_z: int,
) -> NDArray[np.complex128] | None:
    """Eq. (3.88) summed over one block's admitted cells, or ``None``."""
    nc = r.depth.shape[1]
    hits = np.flatnonzero(admitted.ravel())
    if hits.size == 0:
        return None
    beam_at, within = np.divmod(hits, nc * n_z)
    range_at, depth_at = np.divmod(within, n_z)
    q_hit = q_infl.ravel()[hits]
    spread_hit = r.spreading[beam_at, range_at]
    along_hit = along.ravel()[hits]
    # The travel-time phase, the tracked branch of 1/sqrt(q) and the
    # transverse Gaussian all ride in one exponent rather than three,
    # because a complex exponential over tens of millions of survivors
    # is where the run time goes.
    exponent = (
        -0.5j * (r.phase[beam_at, range_at]
                 + np.angle(q_hit * np.conj(spread_hit)))
        - 1j * grid.omega * (r.time[beam_at, range_at] + along_hit
                             + r.slope[beam_at, range_at] / (2.0 * q_hit)
                             * normal.ravel()[hits] ** 2)
    )
    if grid.attenuation > 0.0:
        # e^{-alpha s} of Eq. (3.116): the marched arc length continued
        # to the foot of the perpendicular (along = s/c), floored at
        # zero; see the docstring. Real, so it joins the exponent as
        # pure decay whichever time convention the caller settles on.
        exponent -= grid.attenuation * np.maximum(
            r.path[beam_at, range_at]
            + r.speed[beam_at, range_at] * along_hit, 0.0)
    value = (
        r.weight[beam_at, range_at] * strength[beam_at]
        * np.sqrt(r.speed[beam_at, range_at] / r_infl.ravel()[hits])
        / np.sqrt(np.abs(q_hit)) * np.exp(exponent)
    )
    cells = n_z * nc
    target = depth_at * nc + range_at
    return (np.bincount(target, value.real, minlength=cells)
            + 1j * np.bincount(target, value.imag, minlength=cells)
            ).reshape(n_z, nc)


def _sum_rung(
    field: NDArray[np.complex128], s: _BeamSamples, grid: _Influence, *,
    wrap: int, side: float, strength: NDArray[Any],
    rows: NDArray[np.intp], cols: NDArray[np.intp] | None,
) -> None:
    """Add one rung's beams into ``field``, a block of depths at a time.

    The block is sized so the temporaries stay bounded however large the
    requested receiver grid is.
    """
    n_ranges = s.depth.shape[1]
    nc = n_ranges if cols is None else cols.size
    r = _rung_samples(s, rows, cols)
    col_index = np.arange(n_ranges, dtype=np.intp) if cols is None else cols
    xi = r.xi[:, :, None]
    offset = r.range_offset[:, :, None]
    column = r.column_range[:, :, None]
    depth = r.depth[:, :, None]
    vertical = r.vertical[:, :, None]
    speed = r.speed[:, :, None]
    speed_sq = speed**2
    wavelength = 2.0 * np.pi * speed / grid.omega
    spreading = r.spreading[:, :, None]
    slope = r.slope[:, :, None]
    half_width = grid.half_omega_width[rows][:, None, None]
    sub_capped = (None if grid.capped is None
                  else grid.capped if cols is None else grid.capped[cols])
    step = max(1, _INFLUENCE_BLOCK // (rows.size * nc))
    for lo in range(0, grid.receiver_depths.size, step):
        zr = grid.receiver_depths[lo:lo + step]
        dz, d_r = _image_offsets(grid, col_index, wrap, side, zr, depth,
                                 offset)
        along = xi * d_r + vertical * dz  # s / c
        normal = speed * (xi * dz - vertical * d_r)
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
        admitted = ((normal * half_width) ** 2
                    < grid.cutoff_sq * (q_infl.real**2 + q_infl.imag**2))
        if sub_capped is not None and sub_capped.any():
            admitted &= (~sub_capped[None, :, None]
                         | (np.abs(speed * along) <= grid.tail_limit))
        block = _survivor_block(r, grid, strength, admitted=admitted,
                                q_infl=q_infl, r_infl=r_infl, along=along,
                                normal=normal, n_z=zr.size)
        if block is None:
            continue
        if cols is None:
            field[lo:lo + zr.size] += block
        else:
            field[lo:lo + zr.size, cols] += block


def _beam_influence(
    s: _BeamSamples, receiver_depths: NDArray[np.float64], *,
    water_depth: float,
    bottom_reflection: float | NDArray[np.complex128],
    omega: float, beam_width: float | NDArray[np.float64],
    attenuation: float, fold: _FoldColumns | None = None,
    march_extent: float = np.inf,
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

    VOLUME ABSORPTION rides in the same exponent when ``attenuation`` is
    nonzero. Sect. 3.6.2 derives it by perturbing the eikonal with a complex
    sound speed: the real rays stand, and each acquires the factor
    :math:`e^{-\int_0^s \alpha(s')\,ds'}` of Eq. (3.116), an integral along the
    ray's own arc length, "a loss proportional to the path length times the
    loss per meter" for constant :math:`\alpha`. The path length here is the
    marcher's cumulative arc length at the bracketing column continued to the
    foot of the perpendicular by :math:`c \cdot (s/c)`, the same closed-form
    continuation the travel time takes three lines up, so the loss is charged
    over exactly the path whose phase is summed. It is *not*
    :math:`\alpha \times` range: the section closes by noting that adding
    :math:`\alpha r` "is used in many ray models", and that approximation is
    precisely what a steep or many-times-reflected path breaks, its arc length
    exceeding its range by the obliquity the marcher already integrated. The
    continued length is floored at zero because the foot of a perpendicular can
    land marginally behind the source, where a negative path would read as
    gain; the clamp is dormant everywhere the method has anything to say (the
    near field within a few beam widths is already not to be read, see
    :func:`gaussian_beams`).

    THE LADDER'S STRENGTHS ARE PER BEAM, because the bottom's coefficient may
    be. ``bottom_reflection`` is either the scalar of a perfect reflector or
    one complex Rayleigh coefficient per beam, evaluated by the caller at the
    grazing angle Snell's invariant fixes for that beam at the seabed; each
    rung of :func:`_image_ladder` says how many surface and bottom planes
    stand between the image and the receiver, and the strength is the
    coefficients raised to those counts. For the unfolded beam in an
    isovelocity column that is exact, not paraxial: the unfolded beam is
    straight, so it crosses every bottom plane at its central ray's own angle.

    THE FOLD MAY BE A LOCAL WEDGE FAN. With ``fold`` given (a sloping bottom),
    each receiver column folds its images at its own facet: the flat vertical
    stack of mirrors becomes the dihedral fan about the local apex that
    :func:`_fold_images` builds, which for a single facet is the exact
    unfolding. The straight continuation the influence sum evaluates *is* the
    unfolded beam, so evaluating it at those rotated image points keeps on a
    slope the very property that makes the flat ladder exact: each stationary
    image contributes :math:`e^{ikR}/R` of its own unfolded distance. What
    breaks it is a fold plane that is wrong by the facet's tilt: an earlier
    version of this branch folded at the local *depth* but not the local
    *slope*, and the tilt displaces a first-fold image by
    :math:`2\beta\,(D - z_\mathrm{r})`, metres against a wavelength, which
    measured 5 to 21 dB of fringe displacement on the ideal wedge where the
    fan ladder measures a small fraction of a decibel. The fan is local in
    reach as well as in tilt: at a sloped column a tail is evaluated no
    farther from its sample than :data:`_TAIL_TRUST` marched extents,
    because a rung whose image circles an apex standing tens of extents
    beyond the march represents bounces on boundary the polyline never
    described, and evaluating those tails anyway measured up to 11 dB of
    pollution on a configuration whose bottom no beam can even reach; two
    extents is the out-and-back allowance that keeps every genuine return
    leg of the ideal wedge (its farthest stands 1.9 extents out) while
    refusing the fiction. A rung whose images
    stand beyond every beam's reach at some columns but not others is
    evaluated on the columns that need it alone, which is what keeps the
    thinning column of a wedge from pricing every rung everywhere: near an
    apex the local depth gets small, the wrap count the reach demands grows
    as its inverse, and without the column subset the ladder would build
    full-width arrays for rungs one column asked for.

    :param attenuation: Volume absorption :math:`\alpha` in nepers per metre
        (0.0 propagates without absorption and skips the work).
    :param fold: Per-column fold geometry of a sloping bottom, or ``None``
        for the level guide at ``water_depth`` (the flat ladder, bit for
        bit; a ``fold`` whose slopes are all zero is that same ladder by
        construction).
    :param march_extent: Range span the beams were marched over, in metres;
        with ``fold`` given it sets the tail budget
        :data:`_TAIL_TRUST` ``* march_extent`` at the sloped columns.
    :return: The complex field, shape ``(n_receiver_depths, n_ranges)``, in the
        convention Eq. (3.88) is printed in; the caller conjugates it.
    """
    r_bottom = np.broadcast_to(np.asarray(bottom_reflection),
                               (s.depth.shape[0],))
    plan = _rung_plan(s, fold, water_depth, r_bottom,
                      _wrap_count(s, water_depth, fold))
    # One W_0 per beam (a scalar is every beam's): the admission test of
    # :func:`_sum_rung` reads W = 2|q|/(omega W_0) with each row's own width.
    half_omega_width = 0.5 * omega * np.broadcast_to(
        np.asarray(beam_width, dtype=np.float64), (s.depth.shape[0],))
    grid = _Influence(
        receiver_depths=receiver_depths, water_depth=water_depth, omega=omega,
        attenuation=attenuation, fold=fold,
        half_omega_width=half_omega_width, cutoff_sq=_BEAM_CUTOFF**2,
        capped=None if fold is None else np.abs(fold.slopes) > 0.0,
        tail_limit=_TAIL_TRUST * march_extent)
    field = np.zeros((receiver_depths.size, s.depth.shape[1]),
                     dtype=np.complex128)
    for wrap, side, strength, rows, cols in plan:
        _sum_rung(field, s, grid, wrap=wrap, side=side, strength=strength,
                  rows=rows, cols=cols)
    return field


def _warn_beams(
    message: str, *, nested: int = 0,
) -> None:
    """Raise a :class:`~phonometry.PhonometryWarning` at the caller's call site.

    The frame the warning is reported against is the one that called
    :func:`gaussian_beams`, which is where the input that provoked it was
    written. ``nested`` counts the frames between this one and
    :func:`gaussian_beams` for a check that lives in a helper of its own.
    """
    import warnings

    from ..._internal.warnings import PhonometryWarning

    warnings.warn(f"gaussian_beams: {message}", PhonometryWarning,
                  stacklevel=3 + nested)


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
    # A jump of exactly zero is a node the profile runs straight through; any
    # jump at all is a kink, so the test is deliberately exact and carries no
    # tolerance (see :func:`phonometry._internal.rays._prepare_impulses`).
    kinked = z_prof[1:-1][np.diff(grad).astype(bool)]
    if kinked.size and np.min(np.abs(kinked - source_depth)) <= 1e-6:
        _warn_beams(
            "'source_depth' sits on a gradient discontinuity of the profile,"
            " which concentrates the near-horizontal beams into a spurious jet;"
            " offset the source or smooth the profile there.", nested=1)


def _beam_range_grid(
    ranges_m: NDArray[np.float64] | list[float] | None, *,
    n_steps: int, dr: float, rmax: float,
) -> NDArray[np.float64]:
    """The ranges the field is evaluated at, defaulting to the marching grid.

    Past the end of the march there is nothing to read a beam off, and the
    nearest-column arithmetic of :func:`_beam_influence` would answer with a
    silent extrapolation of the last column rather than with an error, so a
    range beyond it is refused. Half a step of slack is allowed because the
    last column is the one nearest ``max_range`` and a caller who asks for
    exactly that is asking for a column that exists.
    """
    ranges = np.asarray(
        np.arange(n_steps) * dr if ranges_m is None else ranges_m,
        dtype=np.float64).ravel()
    if ranges.size == 0 or not np.all(np.isfinite(ranges)) or np.any(ranges < 0.0):
        raise ValueError("'ranges_m' must be finite, non-negative and non-empty.")
    if np.any(ranges > rmax + 0.5 * dr):
        raise ValueError("'ranges_m' must not run past 'max_range'.")
    return ranges


def _beam_receiver_grid(
    receiver_depths_m: NDArray[np.float64] | list[float] | None, *,
    n_depth_points: int, water_depth: float,
) -> NDArray[np.float64]:
    """The depths the field is evaluated at.

    The default is the interior grid :func:`parabolic_equation` uses, so the
    two fields land on the same depths and subtract. An explicit grid has to
    stay inside the column: the image ladder of :func:`_image_ladder` folds the
    receiver about the two boundaries, which only means anything between them.
    """
    if receiver_depths_m is None:
        n_z = int(n_depth_points)
        if n_z < 2:
            raise ValueError("'n_depth_points' must be at least 2.")
        dz = water_depth / (n_z + 1)
        return np.asarray(dz * np.arange(1, n_z + 1), dtype=np.float64)
    receivers = np.asarray(receiver_depths_m, dtype=np.float64).ravel()
    if receivers.size == 0 or not np.all(np.isfinite(receivers)):
        raise ValueError("'receiver_depths_m' must be finite and non-empty.")
    if np.any(receivers < 0.0) or np.any(receivers > water_depth):
        raise ValueError("'receiver_depths_m' must lie within the water column.")
    return receivers


class _Fan(NamedTuple):
    """The launch fan, in the three forms the beam sum reads it in.

    They are one quantity written three ways and have to agree, which is why
    they travel together: ``xi`` is the Snell invariant the marcher is handed,
    and ``dtheta`` is the spacing the weight of Eq. (3.92) integrates over.

    :ivar launch: Launch angle of each beam from the horizontal, in radians.
    :ivar xi: ``cos(launch) / c(z_s)``, per beam, in s/m.
    :ivar dtheta: Spacing of the fan, in radians.
    """

    launch: NDArray[np.float64]
    xi: NDArray[np.float64]
    dtheta: float


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
    seabed_density: float | None = None,
    seabed_sound_speed: float | None = None,
    density: float = 1000.0,
    absorption_model: str | None = None,
    temperature: float = 10.0,
    salinity: float = 35.0,
    ph: float = 8.0,
    bathymetry_ranges_m: NDArray[np.float64] | list[float] | None = None,
    bathymetry_depths_m: NDArray[np.float64] | list[float] | None = None,
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
      substantially smaller than any physical scale in the problem". This is
      the limit that bites hardest and the one a plausible-looking answer
      hides best -- and one earlier version of this paragraph blamed it for an
      error that was really the beam width's. At 20 Hz in 100 m of water the
      depth is 1.3 wavelengths and two modes propagate; the quarter-depth cap
      this module used to put on :math:`W_0` left a beam a third of a
      wavelength across, and the loss came out decibels high against the
      image-source sum. With the cap retired the same guide (source 36 m,
      receiver 64 m, energy-averaged 0.2 to 5 km) measures -0.001 dB in the
      mean and 0.03 dB at worst, at the ten-wavelength floor's 750 m width.
      That clean bill is narrower than it looks: an isovelocity column over
      perfect reflectors is pure geometry, which the folded receiver images
      reproduce exactly at any frequency, so it says nothing about a channel
      the low-frequency field actually refracts through, and there
      :func:`normal_modes` remains the solver to trust, exact in that regime
      for the cost of two modes.
    * **There is no near field**, and this is the largest error the function
      makes. Eq. (3.92) weights the fan by matching it to a point source in the
      far field, and Eq. (3.88) divides by a cylindrical range that goes to zero
      on the axis every ray leaves from, so close in the sum has nothing to
      converge to. The cylindrical range is floored at one wavelength,
      which is what keeps the answer bounded rather than what makes it right.
      The scale it recovers on is the initial beam width, not a fixed distance.
      Worst error against :math:`20\lg R` over a +-500 m depth cut in an
      unbounded medium at 100 Hz, at three settings whose :math:`W_0` spans
      150 to 437 m: 17, 13 and 4.1 dB at a quarter of :math:`W_0`, 1.2, 0.64 and
      0.36 dB at :math:`W_0`, 0.012, 0.005 and 0.002 dB at 2.5 :math:`W_0`, and
      a thousandth of a decibel or better from 3 :math:`W_0` out. Read nothing
      inside about three beam widths of the source; since the default's
      free-space term grows as :math:`\sqrt{r_\mathrm{max}}`, a longer run
      pushes that boundary out rather than in. :func:`parabolic_equation` is
      the solver to reach for close to the source.
    * **The fan is truncated** at ``max_angle_deg``, and a waveguide with two
      perfectly reflecting boundaries is the worst case for that, because
      nothing but :math:`1/R` attenuates the steep multiple bounces. Measured on
      the ideal 1000 m guide at 300 Hz, source at 300 m and receiver at 600 m,
      against the image-source sum at 2, 5 and 10 km: a fan to 80 degrees is
      0.27, 4.06 and 2.52 dB out, a fan to 85 degrees 0.21, 1.32 and 1.91 dB,
      and a fan to 88 degrees 0.0002, 0.0003 and 0.0004 dB. Cutting the *oracle*
      to the same half-angle moves it by 0.25, 3.95 and 2.31 dB, so what is left
      at 80 degrees is the fan and not the method. A real seabed (the
      ``seabed_density``/``seabed_sound_speed`` pair below) absorbs those
      bounces and the default is then ample; a perfect reflector needs the
      fan opened and ``range_step`` cut with it, since a step has to resolve
      :math:`\tan\theta_\mathrm{max}` depth units of climb per unit range. The
      warning below says when that pairing is wrong.
    * **A shallow channel sets its own width**, and the default now pays it
      per launch angle rather than clamping against it. An earlier version of
      this module capped :math:`W_0` at a quarter of the water depth, reading
      Sect. 3.5's caution that a beam "large compared to the channel ...
      causes a variety of problems" as a ceiling; measured against the
      closed-form Airy modes of an :math:`n^2`-linear 200 m guide at 200 Hz
      (source 30.5 m, receiver 120.5 m, energy-averaged over 0.5 to 4 km),
      that cap's 50 m width came out +3.08 dB in the mean and +5.86 dB at
      worst, systematically too quiet, while the per-angle default measures
      +0.19 dB on the same cut. What a shallow guide actually demands is the
      opposite bound, a beam wide enough to resolve the channel's modes in
      launch angle, and :func:`_default_beam_widths` says why that is
      :math:`W_0 \ge 4D\cos\theta_0/\pi` and what the folded receiver images
      do to make the width affordable. The same profile in 1000 m of water,
      where the modal criterion is out of the band's reach and the free-space
      optimum stands, comes out at +0.72 dB with a 1.37 dB worst bin, closer
      to the exact field than :func:`normal_modes` on the same cut. An
      explicit ``beam_width`` is taken as given, whatever its size: the old
      quarter-depth warning went with the cap, since the measurements put the
      fault on the cap's side.

    **Seawater absorption is off by default** and the field is then optimistic
    beyond a few kilometres at sonar frequencies, exactly as ray theory without
    a volume loss must be. Passing ``absorption_model`` multiplies each beam by
    :math:`e^{-\alpha s}` with :math:`s` the **arc length along its central
    ray**, which is Sect. 3.6.2 done as printed: perturbing the eikonal with
    the complex sound speed a volume loss implies leaves the real rays standing
    and attaches :math:`e^{-\int_0^s \alpha(s')\,ds'}` to each (Eq. 3.116),
    an integral along the path flown, not along the range axis. The distinction
    is not pedantry. The same section notes that adding :math:`\alpha r` to the
    loss "is used in many ray models", and that shortcut under-charges every
    steep or multiply-reflected path by the obliquity of its climb: a path at
    60 degrees is twice as long as the range it covers, and it is precisely the
    steep multiples of a waveguide that absorption is supposed to be killing.
    The marcher integrates :math:`s` with the very Runge-Kutta stages that
    place the ray (:math:`ds/dr = 1/(\xi c)`), so the length the loss is
    charged over is the length of the geometry actually summed. The
    coefficient itself comes from
    :func:`~phonometry.underwater.propagation.closed_form.seawater_absorption`,
    one :math:`\alpha` per run, evaluated at the source frequency and at the
    source depth (the same point the reference sound speed :math:`c_0` is read
    at); over a water column the coefficient's own depth terms move it by
    around a percent per hundred metres, which is far inside the method's
    error budget. The default stays off so the validation figures quoted
    throughout, all measured without absorption, remain reproducible as
    printed.

    **The seabed is a perfect reflector by default**, for the same reason, and
    real shallow-water propagation loss is dominated by what that default
    leaves out: the seabed absorbs part of every bottom bounce. Passing
    ``seabed_density`` and ``seabed_sound_speed`` replaces the perfect
    reflector with the lossy fluid half-space of
    :func:`~phonometry.underwater.propagation.seabed_reflection.reflection_coefficient`
    (the Rayleigh interface, ``density`` standing for the water above it and
    the profile's own bottom sound speed for its ``c1``). This is Sect. 3.6.3
    done as printed: "most ray codes treat the bottom simply as a reflector",
    and each boundary touch multiplies the ray amplitude by
    :math:`|\mathcal{R}(\theta)|` and adds :math:`\arg \mathcal{R}(\theta)`
    to its phase (Eqs. 3.125-3.126), while the dynamic pair :math:`(q, p)`
    crosses the reflection exactly as before, the curvature term of
    Eq. (3.122) vanishing at a flat bottom, so the beam keeps its width and
    only its complex amplitude is docked. The phase is not a refinement to
    skip: below the critical angle :math:`|\mathcal{R}| = 1`, *only* the
    phase distinguishes the lossy seabed from a perfect one, and it moves the
    interference fringes of every bottom-interacting path. The grazing angle
    each beam is charged at is the one Snell's invariant fixes,
    :math:`\cos\theta = \xi\,c(z)` along the whole ray, so at a flat seabed a
    given beam arrives at one and the same angle at every touch whatever the
    profile above did in between: its coefficient is evaluated once, exactly,
    and raised to the marcher's count of bottom touches, in the running
    product and in the receiver-image ladder alike. The book is candid that a
    plane-wave coefficient applied to a field that is not a plane wave is an
    approximation (p. 189), and it is the approximation the whole method
    already breathes; sediment attenuation and elasticity are outside the
    fluid-fluid model here as they are outside ``seabed_reflection`` itself.

    **The bottom may slope.** The ``bathymetry_ranges_m`` /
    ``bathymetry_depths_m`` pair replaces the level bottom with the same
    piecewise-linear ``depth(r)`` polyline :func:`ray_trace` takes, and it is
    the first range dependence in this module; the sound-speed profile stays
    range independent, deliberately -- see :func:`ray_trace`'s scope note for
    why full :math:`c(r, z)` is excluded (no exact oracle exists to hold it
    to). Four things follow from the slope, each stated with its cost:

    * Every beam's central ray reflects specularly off the local facet
      (Eq. 3.121), so an upslope bounce steepens it by twice the slope, and
      the dynamic pair takes the reflection impulse of Eqs. (3.122)-(3.123)
      evaluated on that facet (curvature zero, since the facets are straight;
      :mod:`phonometry._internal.rays` records the closed form and its
      flat-bottom limit).
    * A beam steepened past the vertical would run backward in range, which a
      range-marching solver cannot carry: it is terminated at that bounce and
      its weight is zero from there on, so the field keeps only what still
      travels forward, exactly as the one-way parabolic equation keeps no
      backscatter. Upslope propagation toward an apex therefore *loses* the
      energy the wedge sends back down the slope; the ideal-wedge oracle in
      the tests prices that truncation next to everything else.
    * The receiver-image ladder folds each receiver column about its own
      local facet: the vertical stack of mirrors becomes the dihedral fan
      about the local apex, exact for a single facet and local to each
      column's facet on a general polyline; at slope zero it is the level
      ladder bit for bit, which a test pins. The fan is what lets the
      *tails* keep representing paths the marched axes cannot: an arrival
      that went up the slope, turned past the vertical and came back is a
      wrapped rung of the fan, analytic rather than marched, and the solver
      recovers most of it.
    * The lossy fluid seabed cannot be combined with a slope, and the
      rejection is of the wiring, not the physics: the one-coefficient-per-
      beam collapse rests on Snell's invariant fixing a single grazing angle
      per ray at the bottom, and a slope rotates that invariant at every
      touch. A sloping run takes the perfect reflectors of ``bottom``.

    Validated against the ideal wedge, which has an exact solution by images:
    an isovelocity wedge under a pressure-release surface with a rigid sloping
    bottom of angle :math:`\beta = \pi/n` (:math:`n` even) unfolds into a
    closed fan of :math:`2\pi/\beta` image sources on a circle about the apex.
    The tests build that fan from pure geometry and quantify the agreement in
    dB, cross-sections in range and depth, upslope: a tenth of a decibel
    where a single facet bounce carries the field, and within about two
    decibels of the *complete* wedge field (mean two thirds of one) across a
    thin 2.8-degree wedge whose every cell is dense multipath, most of it
    arrivals near their own turning point, which is where a one-way marcher
    pays its way; the test module's docstring records the measurements and
    their stability under step, width and aperture.

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
        initial half-width, at the :math:`e^{-2}` folding distance in
        intensity, applied to every beam of the fan when passed. Default
        (``None``): one width per launch angle, the free-space optimum of
        each beam's own flight; see :func:`_default_beam_widths`.
    :param range_step: Marching step in range, in metres, and the spacing of the
        default ``ranges_m``.
    :param bottom: ``"pressure-release"`` (default) or ``"rigid"``. The sea
        surface is always pressure-release. Superseded by the fluid seabed
        when the pair below is passed.
    :param seabed_density: Sediment density of a lossy fluid seabed, in the
        same unit as ``density`` (kg/m3 by convention; only the ratio
        enters). Default (``None``): the perfect reflector named by
        ``bottom``, so every published validation number of this module is
        what the solver returns. Passed together with
        ``seabed_sound_speed``, and not alongside ``bottom="rigid"``.
    :param seabed_sound_speed: Sediment sound speed of that seabed, in m/s
        (``None`` likewise). A sediment faster than the water at the bottom
        has a critical grazing angle, below which the reflection is total in
        magnitude and lossy in phase alone.
    :param density: Water density above the seabed, in kg/m3. Ignored unless
        the seabed pair is passed; it enters only through the seabed's
        impedance ratio, the field itself being density-normalised already.
    :param absorption_model: Seawater volume absorption applied along each
        beam's central ray: ``"francois-garrison"``, ``"ainslie-mccolm"`` or
        ``"thorp"``, the same models, spelled the same way, as
        :func:`~phonometry.underwater.propagation.closed_form.seawater_absorption`.
        Default (``None``): no volume absorption, so the published validation
        numbers of this module are what the solver returns.
    :param temperature: Temperature ``T`` for the absorption model, in degrees
        Celsius (ignored when ``absorption_model`` is ``None``).
    :param salinity: Salinity ``S`` for the absorption model, in parts per
        thousand (ignored when ``absorption_model`` is ``None``).
    :param ph: Acidity for the absorption model (ignored when
        ``absorption_model`` is ``None``; Thorp ignores it always).
    :param bathymetry_ranges_m: Node ranges of a piecewise-linear bottom
        profile, in metres, strictly increasing from ``r = 0``; level past
        the last node. Default (``None``): the level bottom at the profile's
        last depth, which is every validation number of this module. Passed
        together with ``bathymetry_depths_m``, and not alongside the seabed
        pair.
    :param bathymetry_depths_m: Bottom depth at each node, in metres,
        strictly positive and never below the sound-speed profile's last
        depth (``None`` likewise).
    :return: A :class:`GaussianBeamResult`.
    :raises ValueError: If the inputs are invalid.
    :warns PhonometryWarning: when the source sits on a kink of the profile
        (Sect. 3.7.4's spurious horizontal jet), and when one marching step
        carries the steepest beam of the fan across more than a quarter of
        the water column, which is the pairing between ``max_angle_deg`` and
        ``range_step`` that is easiest to get wrong.
    """
    f = require_positive(frequency_hz, "frequency_hz")
    z_prof, c_prof = _clean_profile(depths, sound_speeds)
    bathymetry = _clean_bathymetry(bathymetry_ranges_m, bathymetry_depths_m,
                                   z_prof)
    water_depth = float(z_prof[-1])
    depth_at_source = (water_depth if bathymetry is None
                       else float(bathymetry[1][0]))
    zs = float(source_depth)
    if not (0.0 < zs < depth_at_source):
        raise ValueError(_SOURCE_OUTSIDE)
    rmax = require_positive(max_range, "max_range")
    dr_step = require_positive(range_step, "range_step")
    if dr_step > rmax:
        raise ValueError("'range_step' must not exceed 'max_range'.")
    key, seabed = _resolve_boundary(bottom, seabed_density, seabed_sound_speed,
                                    density)
    if seabed is not None and bathymetry is not None:
        raise ValueError(
            "a lossy fluid seabed and a sloping bottom cannot be combined:"
            " the one-coefficient-per-beam collapse (and the image ladder's"
            " powers of R) rests on Snell's invariant fixing a single grazing"
            " angle per ray at a level bottom, and a slope rotates that"
            " invariant at every touch. Sloping runs take the perfect"
            " reflectors of 'bottom'.")
    theta_max = float(max_angle_deg)
    if not (0.0 < theta_max < 90.0):
        raise ValueError("'max_angle_deg' must lie in (0, 90) degrees.")

    absorption_key: str | None = None
    alpha = 0.0
    if absorption_model is not None:
        absorption_key = absorption_model.strip().lower()
        if absorption_key not in _ABSORPTION_MODELS:
            raise ValueError(
                f"'absorption_model' must be one of {_ABSORPTION_MODELS} or"
                f" None, got {absorption_model!r}.")
        alpha = float(seawater_absorption(
            f, temperature=temperature, salinity=salinity, depth=zs, ph=ph,
            model=absorption_key)[0])
    # dB/km to nepers/m: N dB is e^(N ln 10 / 20) in amplitude.
    attenuation = alpha * np.log(10.0) / (20.0 * _M_PER_KM)

    omega = 2.0 * np.pi * f
    c0 = float(np.interp(zs, z_prof, c_prof))
    wavelength = c0 / f
    _check_source_on_kink(z_prof, c_prof, zs)

    # The width the fan's density is sized for has to exist before the fan
    # does, and the overlap condition below must hold for every beam, so it is
    # the *widest* width of the run: an explicit one applies everywhere, and
    # both terms of the per-angle default are widest on the axis, so its
    # maximum is its value at theta_0 = 0.
    if beam_width is not None:
        w_fan = require_positive(beam_width, "beam_width")
    else:
        w_fan = float(_default_beam_widths(
            wavelength, rmax, water_depth, np.zeros(1))[0])

    span = 2.0 * np.radians(theta_max)
    n_fan = (int(np.ceil(span * 4.0 * np.pi * w_fan / wavelength)) + 1
             if n_beams is None else int(n_beams))
    if n_fan < 2:
        raise ValueError("'n_beams' must be at least 2.")
    launch = np.linspace(-np.radians(theta_max), np.radians(theta_max), n_fan)
    fan = _Fan(launch, np.cos(launch) / c0, float(launch[1] - launch[0]))
    w0 = (np.full(n_fan, float(w_fan)) if beam_width is not None
          else _default_beam_widths(wavelength, rmax, water_depth, launch))

    n_steps = int(np.ceil(rmax / dr_step)) + 1
    dr = rmax / (n_steps - 1)

    bottom_factor: float | NDArray[np.complex128] = _BOTTOM_REFLECTION[key]
    if seabed is not None:
        rho1, rho2, c2 = seabed
        c_bottom = float(c_prof[-1])
        # One coefficient per beam is exact rather than sampled: see
        # :func:`_seabed_grazing_deg` for why Snell's invariant fixes a single
        # grazing angle per ray at a flat seabed.
        grazing = _seabed_grazing_deg(fan.xi, c_bottom)
        # Conjugated, deliberately. The beam sum is assembled in the
        # exp(+i omega t) convention Eq. (3.88) is printed in and conjugated
        # once at the end (see the sign-convention note above the result
        # class), so every complex factor fed into it must be the conjugate of
        # its exp(-i omega t) self or the final conjugation turns its phase
        # backwards. The perfect reflectors are real and never showed this;
        # a below-critical R is where it bites, |R| = 1 and only the phase
        # carrying the seabed, and applying it un-conjugated measured 15 dB
        # wrong against the lossy image sum at 4 km where the conjugate
        # measures 0.02 dB.
        bottom_factor = np.conj(reflection_coefficient(
            grazing, rho1=rho1, c1=c_bottom, rho2=rho2, c2=c2))

    upper: float | SlopingBoundary = (
        water_depth if bathymetry is None else SlopingBoundary(*bathymetry))
    march = march_rays(
        _ocean_ray_derivative(z_prof, c_prof), xi=fan.xi,
        z0=np.full(n_fan, zs), zeta0=np.sin(launch) / c0, range_step=dr,
        n_steps=n_steps, lower=0.0, upper=upper,
        dynamic=DynamicRays(np.asarray(0.5j * omega * w0**2, dtype=np.complex128),
                            np.full(n_fan, 1.0 + 0.0j), z_prof, c_prof))

    ranges = _beam_range_grid(ranges_m, n_steps=n_steps, dr=dr, rmax=rmax)
    receivers = _beam_receiver_grid(receiver_depths_m, n_depth_points=n_depth_points,
                                    water_depth=water_depth)

    climb = dr * np.tan(np.radians(theta_max))
    if climb > _MAX_STEEP_CLIMB * water_depth:
        _warn_beams(
            f"one marching step carries the steepest beam of the fan {climb:.0f} m"
            f" across a {water_depth:.0f} m column, so its trajectory is not"
            " resolved; cut 'range_step' or narrow 'max_angle_deg'.")

    # The image ladder folds each receiver column at its own local facet when
    # the bottom slopes: depth and slope both, since a fold plane wrong by the
    # facet's tilt displaces the images by wavelengths (see _beam_influence).
    fold: _FoldColumns | None = None
    if bathymetry is not None:
        br, bd = bathymetry
        facet = np.concatenate(([0.0], np.diff(bd) / np.diff(br), [0.0]))
        fold = _FoldColumns(
            ranges, np.asarray(np.interp(ranges, br, bd)),
            facet[np.searchsorted(br, ranges, side="right")])
    field, widths, curvatures = _assemble_beam_field(
        march, ranges=ranges, receivers=receivers, fan=fan, dr=dr,
        z_prof=z_prof, c_prof=c_prof, omega=omega, c0=c0, w0=w0,
        water_depth=water_depth, bottom_reflection=bottom_factor,
        attenuation=attenuation, fold=fold)

    # Eq. (3.88) is written in the exp(+i omega t) convention; conjugating once
    # here hands back a field in the exp(-i omega t) one the rest of the module
    # speaks. The loss is untouched by that, and the weights of Eq. (3.92)
    # normalise the sum to Eq. (3.80)'s unit pressure at 1 m, so there is no
    # p_0 to divide by.
    pressure = np.conj(field)
    with np.errstate(divide="ignore"):
        pl = -20.0 * np.log10(np.abs(pressure))

    ray_depths = march.positions
    ray_widths, ray_curvatures = widths, curvatures
    if (bathymetry is not None and march.stopped_columns is not None
            and np.any(march.stopped_columns < n_steps)):
        # A terminated beam's history is frozen at its last bounce; the field
        # above already retired it there, and the per-ray records say so the
        # same way ray_trace does: NaN from the stop column on.
        gone = np.arange(n_steps)[None, :] >= march.stopped_columns[:, None]
        ray_depths = np.where(gone, np.nan, ray_depths)
        ray_widths = np.where(gone, np.nan, ray_widths)
        ray_curvatures = np.where(gone, np.nan, ray_curvatures)

    return GaussianBeamResult(
        frequency=f,
        ranges=ranges,
        depths=receivers,
        propagation_loss=np.asarray(pl, dtype=np.float64),
        pressure=np.asarray(pressure, dtype=np.complex128),
        launch_angles=np.degrees(launch),
        ray_ranges=np.broadcast_to(np.arange(n_steps) * dr,
                                   march.positions.shape).copy(),
        ray_depths=ray_depths,
        beam_widths=ray_widths,
        wavefront_curvatures=ray_curvatures,
        initial_beam_widths=np.asarray(w0, dtype=np.float64),
        absorption_model=absorption_key,
        absorption_coefficient=alpha,
        seabed_density=None if seabed is None else seabed[1],
        seabed_sound_speed=None if seabed is None else seabed[2],
        source_depth=zs,
        water_depth=water_depth,
        bathymetry_ranges=None if bathymetry is None else bathymetry[0],
        bathymetry_depths=None if bathymetry is None else bathymetry[1],
    )


def _assemble_beam_field(
    march: RayMarch, *, ranges: NDArray[np.float64],
    receivers: NDArray[np.float64], fan: _Fan, dr: float,
    z_prof: NDArray[np.float64], c_prof: NDArray[np.float64],
    omega: float, c0: float, w0: NDArray[np.float64],
    water_depth: float,
    bottom_reflection: float | NDArray[np.complex128], attenuation: float,
    fold: _FoldColumns | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    r"""Read the march into :class:`_BeamSamples` and sum the beams over it.

    Two per-ray quantities are formed here and nowhere else. The reflection
    factor is the running product of the coefficients the central ray has met,
    which is why the marcher has to say *which* boundary each bounce was at: a
    pressure-release sea surface inverts the pressure and a seabed need not.
    With a lossy seabed the bottom's coefficient is one complex
    :math:`\mathcal{R}` per beam rather than a scalar, and the product is
    Jensen Eq. (3.126) applied at every touch, magnitude and phase together;
    it collapses to a power because the touches of one ray all share one
    grazing angle (see :func:`gaussian_beams`).
    And the branch of :math:`\sqrt{q}` is fixed by unwrapping the argument of
    ``q`` along the ray before anything is sampled off it, since the increment
    per range step is small while a principal-value square root taken sample by
    sample would jump by :math:`\pi` at the first caustic.

    A beam the sloping bottom terminated (see :func:`ray_trace` on rays
    reflected past the vertical) has its weight zeroed from the stop column
    on: its history there is a state frozen at the terminating bounce, and a
    frozen beam summed as if it kept flying would shine from a point the ray
    never passed. What the one-way march drops with it is the energy the
    wedge sends back down the slope, exactly as a one-way parabolic equation
    drops backscatter.
    """
    if (march.spreadings is None or march.spreading_slopes is None
            or march.horizontals is None
            or march.stopped_columns is None):  # pragma: no cover
        raise ValueError("the march must carry the dynamic ray states.")
    q = np.asarray(march.spreadings, dtype=np.complex128)
    p = np.asarray(march.spreading_slopes, dtype=np.complex128)
    speed = np.interp(march.positions, z_prof, c_prof)
    # W = 2|q|/(omega W_0) is Eq. (3.89) with Im[p/q] replaced by the conserved
    # Wronskian; K is Eq. (3.90) with the sign of the conjugated field. W_0 is
    # per beam, so each row of the history divides by its own.
    widths = 2.0 * np.abs(q) / (omega * w0[:, None])
    curvatures = speed * np.real(p / q)

    at_bottom = np.cumsum(march.upper_reflections, axis=1)
    at_surface = np.cumsum(march.reflections - march.upper_reflections, axis=1)
    r_bottom = (bottom_reflection if isinstance(bottom_reflection, float)
                else np.asarray(bottom_reflection)[:, None])
    reflected = (_SURFACE_REFLECTION**at_surface) * (r_bottom**at_bottom)
    # A(theta_0) of Eq. (3.92) with Eq. (3.91) substituted in, real and
    # positive. The weight carries each beam's own q(0) through W_0, which is
    # what lets the fan mix initial widths without renormalising anything.
    weight = (fan.dtheta * (omega * w0 / (2.0 * c0))
              * np.sqrt(np.cos(fan.launch) / np.pi))

    column = np.clip(np.rint(ranges / dr).astype(np.intp), 0,
                     march.positions.shape[1] - 1)
    cosine = speed[:, column] * march.horizontals[:, column]
    alive = column[None, :] < march.stopped_columns[:, None]
    samples = _BeamSamples(
        xi=march.horizontals[:, column],
        column_range=np.asarray(column * dr, dtype=np.float64)[None, :],
        range_offset=(ranges - column * dr)[None, :],
        depth=march.positions[:, column],
        vertical=march.verticals[:, column],
        speed=speed[:, column],
        spreading=q[:, column],
        slope=p[:, column],
        time=march.times[:, column],
        path=march.arc_lengths[:, column],
        phase=np.unwrap(np.angle(q), axis=1)[:, column],
        weight=np.where(alive, weight[:, None] * reflected[:, column],
                        0.0).astype(np.complex128),
        reach=_BEAM_CUTOFF * widths[:, column].max(axis=1) / cosine.min(axis=1),
    )
    field = _beam_influence(
        samples, receivers, water_depth=water_depth,
        bottom_reflection=bottom_reflection, omega=omega, beam_width=w0,
        attenuation=attenuation, fold=fold,
        march_extent=(march.positions.shape[1] - 1) * dr)
    return field, widths, curvatures


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
        raise ValueError(_SOURCE_OUTSIDE)
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
