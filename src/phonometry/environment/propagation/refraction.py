#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Atmospheric refraction: ray tracing and the parabolic equation (PE).

Sound propagation outdoors is bent by vertical gradients of the effective sound
speed :math:`c_{eff}(z) = c(z) + u(z)` (the adiabatic sound speed plus the
component of the wind in the propagation direction, Salomons Eq. (4.4)). This
module
predicts that refraction with two complementary models, clean-room from
Salomons, *Computational Atmospheric Acoustics* (Springer, 2001) and
Attenborough & Van Renterghem, *Predicting Outdoor Sound* (2e, CRC, 2021,
Ch. 11), and it is the refracting-atmosphere counterpart of the range-independent
ocean solvers in :mod:`phonometry.underwater.propagation.numerical`:

* :func:`atmospheric_ray_paths` -- geometrical acoustics. Integrates Snell's law
  for sound rays (Salomons Eq. (4.3)) with a fixed-step Runge-Kutta scheme,
  returning the curved ray paths, their turning points, travel times and ground
  reflections. It shares its ray core with the ocean ``ray_trace``: the
  atmospheric version reflects at the ground (:math:`z = 0`) instead of the sea
  surface and marches an upward-open half space.
* :func:`atmospheric_parabolic_equation` -- the Green's Function Parabolic
  Equation (GFPE, Salomons Appendix H). Marches the one-way wave equation in
  range with the split-step Fourier algorithm (the same range-marching family
  as the ocean ``parabolic_equation``), a Gaussian starter (Salomons
  Eq. (G.64)), an absorbing layer at the top of the grid (Salomons Sec. G.9)
  and a finite-impedance ground condition through the plane-wave reflection
  coefficient :math:`R(k_z) = (k_z Z - k_0)/(k_z Z + k_0)` plus the
  surface-wave residue of its pole (Salomons Eqs. (H.28), (H.49)). It returns
  the relative sound level (dB re free field) over the range-height plane.

For a linear effective sound-speed profile the ray paths are exact circular
arcs of radius :func:`ray_curvature_radius`, and an upward-refracting linear
profile has a closed-form :func:`shadow_zone_distance`; both anchor the ray
model. The PE is anchored against the exact spherical-wave ground effect
(:func:`phonometry.environment.ground_effect`) in the homogeneous limit
(gradient zero), which it reproduces to a few tenths of a dB on the default
grid (finer ``height_step`` converges it further).

The ground impedance is taken in the :math:`e^{-i \omega t}` convention of
Salomons (a passive ground has :math:`\operatorname{Im}(Z) > 0`), shared with
:mod:`phonometry.environment.propagation.ground_barriers`. The porous models of
:mod:`phonometry.materials` work in the opposite :math:`e^{+j \omega t}`
convention (:math:`\operatorname{Im}(Z) < 0`), so an impedance derived from
them (``flow_resistivity=`` or a
``PorousMediumResult``) is conjugated internally before entering the PE ground
condition. Heights and ranges are in metres, sound speeds in m/s and
frequencies in Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ..._internal.validation import require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ...materials.absorbers.porous import PorousMediumResult

#: Default reference speed of sound ``c`` in air at the ground, in m/s (matches
#: the environmental domain).
_C_SOUND = 343.0
#: Default air density ``rho``, in kg/m3 (matches the environmental domain).
_AIR_DENSITY = 1.205


# ===========================================================================
# Effective sound-speed profiles
# ===========================================================================
@dataclass(frozen=True)
class EffectiveSoundSpeedProfile:
    r"""Vertical profile of the effective sound speed ``c_eff(z)``.

    The profile is sampled on a strictly increasing height grid starting at the
    ground (:math:`z = 0`); intermediate heights are taken as piecewise linear,
    so a two-point profile represents an exact linear gradient.

    :ivar heights: Heights ``z`` above the ground, in metres (from
        :math:`z = 0`).
    :ivar sound_speeds: Effective sound speed at each height, in m/s.
    :ivar description: Short human-readable label (e.g. ``"linear, +0.1 s^-1"``).
    """

    heights: NDArray[np.float64]
    sound_speeds: NDArray[np.float64]
    description: str = ""

    def speed_at(self, height: ArrayLike) -> NDArray[np.float64]:
        """Piecewise-linear effective sound speed at one or more heights."""
        z = np.asarray(height, dtype=np.float64)
        return np.asarray(np.interp(z, self.heights, self.sound_speeds),
                          dtype=np.float64)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the effective sound-speed profile (height on the vertical axis)."""
        from ..._i18n import check_language
        from ..._plot.environment import plot_sound_speed_profile

        return plot_sound_speed_profile(self, ax=ax, language=check_language(language), **kwargs)


def linear_sound_speed_profile(
    gradient: float,
    *,
    ground_speed: float = _C_SOUND,
    max_height: float = 100.0,
) -> EffectiveSoundSpeedProfile:
    r"""Linear effective sound-speed profile (constant vertical gradient).

    The profile is :math:`c_{eff}(z) = c_0 + \text{gradient} \cdot z`. A
    positive ``gradient`` (sound speed increasing with height) refracts sound
    downward (favourable propagation); a negative gradient refracts it upward
    and creates an acoustic shadow near the ground (Salomons Sec. 4.2).

    :param gradient: Vertical gradient ``dc/dz``, in s^-1 (m/s per m).
    :param ground_speed: Sound speed ``c0`` at the ground, in m/s.
    :param max_height: Top of the sampled profile, in metres.
    :return: A two-point :class:`EffectiveSoundSpeedProfile`.
    :raises ValueError: If ``max_height`` or ``ground_speed`` is not positive, or
        the profile turns non-positive within ``[0, max_height]``.
    """
    c0 = require_positive(ground_speed, "ground_speed")
    h = require_positive(max_height, "max_height")
    if not np.isfinite(gradient):
        raise ValueError("'gradient' must be finite.")
    top = c0 + gradient * h
    if top <= 0.0:
        raise ValueError(
            "the linear profile reaches a non-positive sound speed within "
            "'max_height'; reduce 'max_height' or the gradient magnitude."
        )
    sign = "+" if gradient >= 0 else "-"
    return EffectiveSoundSpeedProfile(
        heights=np.array([0.0, h], dtype=np.float64),
        sound_speeds=np.array([c0, top], dtype=np.float64),
        description=f"linear, {sign}{abs(gradient):g} s^-1",
    )


def log_linear_sound_speed_profile(
    b: float,
    *,
    ground_speed: float = 340.0,
    roughness_length: float = 0.1,
    max_height: float = 100.0,
    n_points: int = 128,
) -> EffectiveSoundSpeedProfile:
    r"""Logarithmic effective sound-speed profile (surface layer).

    The profile is :math:`c_{eff}(z) = c_0 + b \ln(1 + z/z_0)`, the realistic
    surface-layer profile of Salomons Eq. (4.5): ``b`` is
    the strength of the gradient (``+1 m/s`` for a typical downward-refracting
    atmosphere, ``-1 m/s`` for an upward-refracting one) and ``z0`` is the
    aerodynamic roughness length (about 0.1 m for grassland). The steep gradient
    near the ground is resolved by sampling the height grid logarithmically.

    :param b: Profile strength ``b``, in m/s (positive: downward refraction).
    :param ground_speed: Sound speed ``c0`` at the ground, in m/s.
    :param roughness_length: Aerodynamic roughness length ``z0``, in metres.
    :param max_height: Top of the sampled profile, in metres.
    :param n_points: Number of samples of the height grid (>= 2).
    :return: An :class:`EffectiveSoundSpeedProfile`.
    :raises ValueError: On non-positive ``c0``, ``z0`` or ``max_height``, fewer
        than two points, or a profile that turns non-positive.
    """
    c0 = require_positive(ground_speed, "ground_speed")
    z0 = require_positive(roughness_length, "roughness_length")
    h = require_positive(max_height, "max_height")
    if not np.isfinite(b):
        raise ValueError("'b' must be finite.")
    if int(n_points) < 2:
        raise ValueError("'n_points' must be at least 2.")
    # Logarithmic height grid from the ground to max_height (z = 0 included).
    z = np.concatenate((
        [0.0],
        np.geomspace(min(z0, h) * 0.1, h, int(n_points) - 1),
    ))
    z = np.unique(np.clip(z, 0.0, h))
    c = c0 + b * np.log1p(z / z0)
    if np.any(c <= 0.0):
        raise ValueError(
            "the logarithmic profile reaches a non-positive sound speed within "
            "'max_height'; reduce 'max_height' or |b|."
        )
    sign = "+" if b >= 0 else "-"
    return EffectiveSoundSpeedProfile(
        heights=np.asarray(z, dtype=np.float64),
        sound_speeds=np.asarray(c, dtype=np.float64),
        description=f"log-linear, b={sign}{abs(b):g} m/s, z0={z0:g} m",
    )


def _clean_profile(
    profile: EffectiveSoundSpeedProfile,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return the height/speed arrays of a profile."""
    z = np.asarray(profile.heights, dtype=np.float64)
    c = np.asarray(profile.sound_speeds, dtype=np.float64)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("the profile must have at least two height samples.")
    if c.shape != z.shape:
        raise ValueError("'sound_speeds' must match 'heights' in length.")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(c))):
        raise ValueError("the profile heights and speeds must be finite.")
    if abs(float(z[0])) > 1e-9:
        raise ValueError("the profile must start at the ground z = 0.")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("the profile heights must be strictly increasing.")
    if np.any(c <= 0.0):
        raise ValueError("the profile sound speeds must be strictly positive.")
    return z, c


# ===========================================================================
# Closed-form ray geometry (linear profile)
# ===========================================================================
def ray_curvature_radius(
    gradient: float,
    *,
    ground_speed: float = _C_SOUND,
    launch_angle_deg: float = 0.0,
) -> float:
    r"""Radius of curvature of a sound ray in a linear sound-speed gradient.

    In a linear effective sound-speed profile a sound ray is an exact circular
    arc of radius :math:`R_c = 1/(|\text{gradient}| \, \xi)` with the Snell
    invariant :math:`\xi = \cos(\theta_0)/c`, evaluated at the launch height
    (Salomons Sec. 4.4; Attenborough Ch. 11). For a ray launched from the
    height where the speed is ``ground_speed`` (:math:`c_0`),
    :math:`R_c = c_0 / (|\text{gradient}| \cos\theta_0)`.

    :param gradient: Vertical gradient ``dc/dz``, in s^-1 (must be non-zero).
    :param ground_speed: Sound speed at the launch height, in m/s.
    :param launch_angle_deg: Launch angle from the horizontal, in degrees.
    :return: The radius of curvature ``Rc``, in metres (always positive).
    :raises ValueError: If the gradient is zero (a straight ray) or the launch
        angle is not within ``(-90, 90)`` degrees.
    """
    c0 = require_positive(ground_speed, "ground_speed")
    if not (np.isfinite(gradient) and abs(gradient) > 0.0):
        raise ValueError("'gradient' must be finite and non-zero (a curved ray).")
    if abs(launch_angle_deg) >= 90.0:
        raise ValueError("'launch_angle_deg' must be within (-90, 90) degrees.")
    return float(c0 / (abs(gradient) * np.cos(np.radians(launch_angle_deg))))


def shadow_zone_distance(
    gradient: float,
    source_height: float,
    receiver_height: float,
    *,
    ground_speed: float = _C_SOUND,
) -> float:
    r"""Distance to the acoustic shadow boundary, upward-refracting profile.

    For a linear upward-refracting profile (``gradient < 0``) the ground-level
    ray that just grazes the surface bounds a region beyond which no direct or
    once-reflected ray arrives (Salomons Sec. 4.4; Attenborough Ch. 11). With a
    ray radius :math:`R_c = c_0/|\text{gradient}|` the limiting horizontal
    distance is the closed form:

    .. math::

       x_{shadow} = \sqrt{2 R_c} \left( \sqrt{h_s} + \sqrt{h_r} \right)

    valid for source and receiver heights :math:`h_s`, :math:`h_r` small
    compared with :math:`R_c`.

    :param gradient: Vertical gradient ``dc/dz``, in s^-1 (must be negative for a
        shadow zone to exist).
    :param source_height: Source height ``hs``, in metres (>= 0).
    :param receiver_height: Receiver height ``hr``, in metres (>= 0).
    :param ground_speed: Sound speed ``c0`` at the ground, in m/s.
    :return: The horizontal shadow-boundary distance, in metres.
    :raises ValueError: If the gradient is not negative, or a height is negative.
    """
    c0 = require_positive(ground_speed, "ground_speed")
    if not np.isfinite(gradient) or gradient >= 0.0:
        raise ValueError(
            "'gradient' must be negative (an upward-refracting profile) for an "
            "acoustic shadow zone to exist."
        )
    if source_height < 0.0 or receiver_height < 0.0:
        raise ValueError("Source and receiver heights must be non-negative.")
    rc = c0 / abs(gradient)
    return float(np.sqrt(2.0 * rc) * (np.sqrt(source_height) + np.sqrt(receiver_height)))


# ===========================================================================
# Ray tracing (Snell's law, RK4)
# ===========================================================================
@dataclass(frozen=True)
class AtmosphericRayResult:
    """Ray-tracing solution through an effective sound-speed profile.

    :ivar launch_angles: Launch angles from the horizontal, in degrees.
    :ivar ranges: Per-ray horizontal ranges, in metres, shape
        ``(n_rays, n_steps)``.
    :ivar heights: Per-ray heights, in metres, shape ``(n_rays, n_steps)``.
    :ivar travel_times: Per-ray cumulative travel times, in seconds, shape
        ``(n_rays, n_steps)``.
    :ivar turning_points: Number of turning points (height extrema) per ray.
    :ivar ground_reflections: Number of ground reflections per ray.
    :ivar source_height: Source height, in metres.
    """

    launch_angles: NDArray[np.float64]
    ranges: NDArray[np.float64]
    heights: NDArray[np.float64]
    travel_times: NDArray[np.float64]
    turning_points: NDArray[np.int_]
    ground_reflections: NDArray[np.int_]
    source_height: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the curved ray paths (height on the vertical axis)."""
        from ..._i18n import check_language
        from ..._plot.environment import plot_atmospheric_rays

        return plot_atmospheric_rays(self, ax=ax, language=check_language(language), **kwargs)


def atmospheric_ray_paths(
    profile: EffectiveSoundSpeedProfile,
    *,
    source_height: float,
    launch_angles_deg: ArrayLike,
    max_range: float = 1000.0,
    n_steps: int = 2000,
) -> AtmosphericRayResult:
    r"""Trace sound rays through a refracting atmosphere over a ground surface.

    Integrates Snell's law for sound rays (Salomons Eq. (4.3),
    :math:`\cos(\gamma)/c(z) = \text{const}`) with a fixed-step fourth-order
    Runge-Kutta scheme, marching in range and reflecting specularly at the
    ground (:math:`z = 0`). The state is the height ``z`` and the vertical
    slowness :math:`\zeta = \sin(\gamma)/c`; with the range-invariant
    :math:`\xi = \cos(\gamma_0)/c(z_s)` the equations are
    :math:`dz/dr = \zeta/\xi` and
    :math:`d\zeta/dr = -(dc/dz)/(c^3 \xi)`, the same ray core as the ocean
    :func:`~phonometry.underwater.propagation.numerical.ray_trace` (with a ground
    reflection in place of the sea surface). The travel time accumulates
    :math:`dt/dr = 1/(\xi c^2)`.

    :param profile: The effective sound-speed profile (see
        :func:`linear_sound_speed_profile` /
        :func:`log_linear_sound_speed_profile`).
    :param source_height: Source height ``zs``, in metres (>= 0).
    :param launch_angles_deg: Launch angles from the horizontal, in degrees
        (positive upward), within ``(-90, 90)``.
    :param max_range: Maximum horizontal range to trace, in metres.
    :param n_steps: Number of range steps per ray (>= 2).
    :return: An :class:`AtmosphericRayResult`.
    :raises ValueError: If the inputs are invalid.
    """
    z_prof, c_prof = _clean_profile(profile)
    top = float(z_prof[-1])
    zs = float(source_height)
    if zs < 0.0:
        raise ValueError("'source_height' must be non-negative.")
    rmax = require_positive(max_range, "max_range")
    if int(n_steps) < 2:
        raise ValueError("'n_steps' must be at least 2.")
    angles = np.asarray(launch_angles_deg, dtype=np.float64).ravel()
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("'launch_angles_deg' must be finite and non-empty.")
    if np.any(np.abs(angles) >= 90.0):
        raise ValueError("'launch_angles_deg' must be within (-90, 90) degrees.")

    # Piecewise-constant gradient per profile segment (sharp kinks preserved);
    # above the profile top the atmosphere is taken as homogeneous (gradient 0).
    seg_grad = np.diff(c_prof) / np.diff(z_prof)

    def _grad_at(zq: NDArray[np.float64]) -> NDArray[np.float64]:
        seg = np.clip(np.searchsorted(z_prof, zq, side="right") - 1,
                      0, seg_grad.size - 1)
        g = seg_grad[seg]
        return np.asarray(np.where(zq > top, 0.0, g), dtype=np.float64)

    def _speed_at(zq: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(np.interp(zq, z_prof, c_prof), dtype=np.float64)

    def deriv(
        z_arr: NDArray[np.float64], zeta_arr: NDArray[np.float64],
        xi_arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        cc = _speed_at(z_arr)
        return zeta_arr / xi_arr, -_grad_at(z_arr) / (cc**3 * xi_arr)

    ns = int(n_steps)
    dr = rmax / (ns - 1)
    ranges = np.linspace(0.0, rmax, ns)
    c0 = float(np.interp(zs, z_prof, c_prof))
    th = np.radians(angles)
    xi = np.cos(th) / c0
    z = np.full(angles.size, zs)
    zeta = np.sin(th) / c0
    ray_z = np.zeros((angles.size, ns))
    ray_t = np.zeros((angles.size, ns))
    ray_z[:, 0] = zs
    turns = np.zeros(angles.size, dtype=np.int_)
    bounces = np.zeros(angles.size, dtype=np.int_)
    prev_dz = np.sign(zeta)
    for s in range(1, ns):
        k1z, k1zeta = deriv(z, zeta, xi)
        k2z, k2zeta = deriv(z + 0.5 * dr * k1z, zeta + 0.5 * dr * k1zeta, xi)
        k3z, k3zeta = deriv(z + 0.5 * dr * k2z, zeta + 0.5 * dr * k2zeta, xi)
        k4z, k4zeta = deriv(z + dr * k3z, zeta + dr * k3zeta, xi)
        z = z + dr / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)
        zeta = zeta + dr / 6.0 * (k1zeta + 2 * k2zeta + 2 * k3zeta + k4zeta)
        # Specular ground reflection: fold z < 0 back up and flip zeta.
        below = z < 0.0
        z = np.where(below, -z, z)
        zeta = np.where(below, -zeta, zeta)
        bounces += below.astype(np.int_)
        # Travel time increment (dt/dr = 1/(xi c^2)) at the true (folded) height.
        cc = _speed_at(z)
        ray_t[:, s] = ray_t[:, s - 1] + dr / (xi * cc**2)
        # Turning points: sign change of the vertical velocity (away from a bounce).
        cur = np.sign(zeta)
        turns += ((cur != prev_dz) & (~below) & (prev_dz != 0)).astype(np.int_)
        prev_dz = cur
        ray_z[:, s] = z
    ray_r = np.broadcast_to(ranges, ray_z.shape).copy()

    return AtmosphericRayResult(
        launch_angles=angles,
        ranges=ray_r,
        heights=ray_z,
        travel_times=ray_t,
        turning_points=turns,
        ground_reflections=bounces,
        source_height=zs,
    )


# ===========================================================================
# Parabolic equation (Green's Function PE, split-step Fourier)
# ===========================================================================
@dataclass(frozen=True)
class AtmosphericPEResult:
    r"""Parabolic-equation relative-level field in a refracting atmosphere.

    :ivar frequency: Source frequency, in Hz.
    :ivar ranges: Range grid, in metres.
    :ivar heights: Height grid of the output field, in metres.
    :ivar relative_level: Relative sound level :math:`\Delta L(z, r)` (dB re
        free field), shape ``(n_heights, n_ranges)``.
    :ivar source_height: Source height, in metres.
    :ivar normalized_impedance: Normalized ground impedance used (complex).
    """

    frequency: float
    ranges: NDArray[np.float64]
    heights: NDArray[np.float64]
    relative_level: NDArray[np.float64]
    source_height: float
    normalized_impedance: complex

    def level_at_height(self, height: float) -> NDArray[np.float64]:
        """Relative sound level versus range at the grid height nearest ``height``."""
        i = int(np.argmin(np.abs(self.heights - float(height))))
        return np.asarray(self.relative_level[i], dtype=np.float64)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the relative-level field over the range-height plane."""
        from ..._i18n import check_language
        from ..._plot.environment import plot_atmospheric_pe

        return plot_atmospheric_pe(self, ax=ax, language=check_language(language), **kwargs)


def _resolve_pe_impedance(
    frequency: float,
    impedance: ArrayLike | PorousMediumResult | None,
    flow_resistivity: float | None,
    model: str,
    speed_of_sound: float,
    air_density: float,
) -> complex:
    """Resolve a single complex normalized ground impedance at ``frequency``."""
    from .ground_barriers import _normalized_ground_impedance

    f = np.array([float(frequency)], dtype=np.float64)
    z = _normalized_ground_impedance(f, impedance, flow_resistivity, model,
                                     speed_of_sound, air_density)
    return complex(z[0])


def atmospheric_parabolic_equation(
    frequency_hz: float,
    profile: EffectiveSoundSpeedProfile,
    *,
    source_height: float,
    impedance: ArrayLike | PorousMediumResult | None = None,
    flow_resistivity: float | None = None,
    model: Literal["delany_bazley", "miki"] = "delany_bazley",
    max_range: float = 1000.0,
    max_height: float = 100.0,
    range_step: float | None = None,
    height_step: float | None = None,
    air_density: float = _AIR_DENSITY,
) -> AtmosphericPEResult:
    r"""Relative-level field from the Green's Function Parabolic Equation (GFPE).

    Marches the split-step Fourier solution of the one-way wave equation
    (Salomons Appendix H) in range. Each step transforms the field to the
    vertical-wavenumber domain, applies the free-space propagator
    :math:`\exp\!\left[ i \, dr \left( \sqrt{k_a^2 - k_z^2} - k_a \right)
    \right]` together with the ground reflection
    :math:`R(k_z) = (k_z Z - k_0)/(k_z Z + k_0)` (Eq. (H.28)), transforms
    back, adds the surface-wave residue of the reflection pole at
    :math:`k_z = -k_0/Z` (the third term of Eq. (H.49), present for a passive
    ground, :math:`\operatorname{Im}(Z) > 0`) and applies the refraction phase
    screen :math:`\exp\!\left[ i \, dr \left( k(z) - k_a \right) \right]`
    (Eq. (H.58)). The source is a Gaussian starter with its ground image
    (Eqs. (G.64), (G.76)) and an absorbing layer at the top of the grid
    (Sec. G.9) suppresses top-boundary reflections. The reference wavenumber
    :math:`k_a = k_0` is taken at the ground.

    The relative sound level re the free field is
    :math:`\Delta L(z, r) = 20 \log_{10}(|p(z, r)| \, R_1)` with :math:`R_1` the
    direct source-receiver distance (Salomons Eq. (3.6)); in a homogeneous
    atmosphere it reproduces the spherical-wave ground effect of
    :func:`ground_effect`.

    The ground surface impedance is either supplied through ``impedance`` (a
    normalized complex value/array, or a
    :class:`~phonometry.materials.PorousMediumResult`) or derived from an
    effective ``flow_resistivity`` (Pa s/m2) via the ``model`` porous model.
    Exactly one of the two must be given. A plain ``impedance`` value is taken
    in the :math:`e^{-i \omega t}` convention (:math:`\operatorname{Im}(Z) > 0`
    for a passive ground); a ``PorousMediumResult`` or ``flow_resistivity`` is
    conjugated internally from the materials' :math:`e^{+j \omega t}`
    convention.

    :param frequency_hz: Source frequency, in Hz.
    :param profile: The effective sound-speed profile.
    :param source_height: Source height ``zs``, in metres (> 0).
    :param impedance: Normalized ground impedance, or a ``PorousMediumResult``.
    :param flow_resistivity: Effective flow resistivity ``sigma`` (Pa s/m2), as
        an alternative to ``impedance``.
    :param model: Porous model for ``flow_resistivity``.
    :param max_range: Maximum range, in metres.
    :param max_height: Top of the output height grid, in metres (the receiver
        region of interest; an absorbing layer is added above it).
    :param range_step: Range marching step ``dr``, in metres. Default
        (``None``): one wavelength.
    :param height_step: Vertical grid spacing ``dz``, in metres. Default
        (``None``): a tenth of a wavelength (Salomons Sec. G.2).
    :param air_density: Air density ``rho``, in kg/m3 (for the porous model).
    :return: An :class:`AtmosphericPEResult`.
    :raises ValueError: If the inputs are invalid or the impedance is unspecified.
    """
    f = require_positive(frequency_hz, "frequency_hz")
    z_prof, c_prof = _clean_profile(profile)
    zs = float(source_height)
    if zs <= 0.0:
        raise ValueError("'source_height' must be positive.")
    rmax = require_positive(max_range, "max_range")
    zmax = require_positive(max_height, "max_height")
    c_ref = float(c_prof[0])  # reference speed at the ground (ka = k0)
    lam = c_ref / f
    dz = require_positive(height_step, "height_step") if height_step is not None else lam / 10.0
    dr = require_positive(range_step, "range_step") if range_step is not None else lam
    if dr > rmax:
        raise ValueError("'range_step' must not exceed 'max_range'.")
    z_imp = _resolve_pe_impedance(f, impedance, flow_resistivity, model, c_ref,
                                  air_density)

    # Grid: physical region [0, zmax] plus a 50-wavelength absorbing layer, then
    # padded to N = 2M so only the lower half carries the (positive-height) field
    # and the Fourier wrap-around is discarded each step (Salomons Sec. H.11).
    absorber = 50.0 * lam
    z_top = zmax + absorber
    m = int(np.ceil(z_top / dz))
    n = 2 * m
    j = np.arange(n)
    z = (j + 0.5) * dz  # midpoint grid (Salomons Eq. (H.76))
    k0 = 2.0 * np.pi * f / c_ref
    ka = k0

    # Effective wavenumber profile with the top absorbing layer (Sec. G.9): an
    # imaginary term i*At*((z - zt)/(zM - zt))^2 grows quadratically to the top.
    c_eff = np.interp(np.clip(z, 0.0, z_prof[-1]), z_prof, c_prof)
    kprof = (2.0 * np.pi * f / c_eff).astype(np.complex128)
    zt = z[m - 1] - absorber
    at = _absorbing_strength(f)
    in_layer = (z >= zt) & (j < m)
    denom = max(z[m - 1] - zt, dz)
    kprof[in_layer] += 1j * at * ((z[in_layer] - zt) / denom) ** 2

    dk = 2.0 * np.pi / (n * dz)
    kz = np.where(j < n // 2, j * dk, (j - n) * dk)
    sq = np.sqrt(ka**2 - kz**2 + 0j)
    sq = np.where(sq.imag < 0.0, -sq, sq)  # decaying (evanescent) branch
    propagator = np.exp(1j * dr * (sq - ka))
    reflection = (kz * z_imp - k0) / (kz * z_imp + k0)
    refraction = np.exp(1j * dr * (kprof - ka))

    # Surface-wave term of the GFPE step (third term of Salomons Eq. (H.40)/
    # (H.49)): the residue of the reflection pole at kz = -beta, beta = k0/Z.
    # The contour of Eq. (H.31) encloses that pole only for Im(Z) > 0 (a
    # passive ground in the e^{-i omega t} convention), where exp(-i beta z)
    # decays with height; otherwise the term is absent.
    beta = k0 / z_imp
    has_surface_wave = beta.imag < 0.0
    if has_surface_wave:
        sw_sq = np.sqrt(ka**2 - beta**2 + 0j)
        if sw_sq.imag < 0.0:
            sw_sq = -sw_sq  # decaying branch, as for the spectral propagator
        sw_propagator = np.exp(1j * dr * (sw_sq - ka))
        sw_profile = np.exp(-1j * beta * z)
        sw_profile[m:] = 0.0

    # Gaussian starter with the ground image (Eqs. (G.64), (G.76)); the normal-
    # incidence plane-wave reflection coefficient weights the image (Eq. (G.77)).
    cp = (z_imp - 1.0) / (z_imp + 1.0)
    starter = np.sqrt(1j * ka) * (
        np.exp(-0.5 * ka**2 * (z - zs) ** 2)
        + cp * np.exp(-0.5 * ka**2 * (z + zs) ** 2)
    )
    psi = starter.astype(np.complex128)
    psi[m:] = 0.0

    n_r = int(np.ceil(rmax / dr)) + 1
    ranges = np.arange(n_r) * dr
    # Output only the physical height band [0, zmax], decimated to keep the
    # stored field light while resolving the interference structure.
    phys = int(np.searchsorted(z, zmax))
    stride = max(1, phys // 400)
    out_idx = np.arange(0, phys, stride)
    out_heights = z[out_idx]
    field = np.zeros((out_idx.size, n_r), dtype=np.float64)

    mid_fwd = np.exp(-1j * kz * 0.5 * dz)
    mid_inv = np.exp(1j * kz * 0.5 * dz)
    for step in range(n_r):
        r = ranges[step]
        if r <= 0.0:
            # The source range is physically singular; NaN is masked by
            # matplotlib and by reductions, unlike +inf which breaks autoscale.
            field[:, step] = np.nan
        else:
            r1 = np.hypot(r, out_heights - zs)
            with np.errstate(divide="ignore"):
                field[:, step] = 20.0 * np.log10(np.abs(psi[out_idx]) * r1 / np.sqrt(r))
        # One GFPE step: forward transform, direct + ground-reflected spectrum,
        # inverse transform, plus the surface-wave residue (Eq. (H.49)), then
        # the refraction phase screen and discard of the wrap-around.
        spec = dz * mid_fwd * np.fft.fft(psi)
        spec_neg = np.empty_like(spec)
        spec_neg[0] = spec[0]
        spec_neg[1:] = spec[1:][::-1]
        if has_surface_wave:
            # Psi(r, beta) = int_0^inf exp(-i beta z') psi(r, z') dz'
            # (Eq. (H.50) evaluated at the pole), on the same midpoint grid.
            phi = dz * np.sum(psi * sw_profile)
            surface = 2j * beta * phi * sw_propagator * sw_profile
        else:
            surface = 0.0
        psi = (np.fft.ifft((spec + reflection * spec_neg) * propagator * mid_inv)
               / dz + surface)
        psi = refraction * psi
        psi[m:] = 0.0

    return AtmosphericPEResult(
        frequency=f,
        ranges=np.asarray(ranges, dtype=np.float64),
        heights=np.asarray(out_heights, dtype=np.float64),
        relative_level=field,
        source_height=zs,
        normalized_impedance=z_imp,
    )


def _absorbing_strength(frequency: float) -> float:
    r"""Absorbing-layer strength ``At`` for the top boundary (Salomons Sec. G.9).

    Salomons tabulates :math:`A_t = 1, 0.5, 0.4, 0.2` at 1000, 500, 125, 30 Hz;
    values in between are interpolated (in log-frequency) and clamped to the
    tabulated ends.
    """
    freqs = np.array([30.0, 125.0, 500.0, 1000.0])
    strengths = np.array([0.2, 0.4, 0.5, 1.0])
    lf = np.log10(np.clip(frequency, freqs[0], freqs[-1]))
    return float(np.interp(lf, np.log10(freqs), strengths))
