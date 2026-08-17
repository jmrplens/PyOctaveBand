#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Tests for Gaussian beam tracing.

Every oracle here is built in the test from a published closed form and shares
nothing with the solver.

* **Free space.** A point source in a homogeneous medium radiates
  :math:`e^{ikR}/R` at unit pressure at 1 m. That single comparison pins the
  three ways the calibration of Jensen Eq. (3.92) can go wrong at once: its
  normalisation is the :math:`1/s` of Eq. (3.80) rather than the
  :math:`1/(4\pi s)` of Eq. (3.51), so a spurious :math:`4\pi` would show as a
  flat 21.98 dB; the printed :math:`e^{+i\pi/4}` is a quarter turn out against
  the :math:`q(0) = i\omega W^2/2` of Eq. (3.91), so it would show as
  :math:`\pi/2` of phase; and Eq. (3.88) is written in the conjugate of the time
  convention the rest of the module uses, so a missing conjugation would show as
  the phase running backwards. None of the last two touches the loss, which is
  exactly why the phase is asserted here and not only the magnitude.

* **One reflection.** With the source far from the surface the exact field is
  the two-ray sum, source plus its negative image, which pins the reflection
  coefficient and the impulse the marcher applies to the spreading at a
  boundary.

* **The ideal waveguide.** A pressure-release surface and bottom over
  isovelocity water is an image lattice, summed here to convergence. It is the
  worst case for a beam fan, because nothing but :math:`1/R` attenuates the
  steep multiple bounces, so the same image sum truncated to the fan's own
  half-angle is used to separate the fan's truncation from the method's error.

* **Beam width and wavefront curvature.** In free space Sect. 3.5.1 gives both
  in closed form, Eqs. (3.85)-(3.86), and the second of them is a real
  cross-check rather than a restatement: it is derived there from a complex
  source point, a different argument from the dynamic ray equations the solver
  integrates.

* **The caustic.** Located by driving the *geometric* spreading of Eq. (3.63) to
  zero by bisection on the launch angle, which is a construction the beam solver
  never performs, and which makes the classical amplitude of Eq. (3.65) diverge
  at a point of the field the beams then have to answer for.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry import PhonometryWarning
from phonometry._internal.rays import DynamicRays, march_rays
from phonometry.underwater.propagation.numerical import (
    GaussianBeamResult,
    _ocean_ray_derivative,
    gaussian_beams,
    ray_trace,
)

_C = 1500.0


def _free_space_column(max_range: float, max_angle_deg: float = 80.0) -> float:
    """A column deep enough that no beam of the fan can reach a boundary.

    The steepest ray climbs ``max_range * tan(max_angle)``, so putting the
    source in the middle of four times that leaves the whole fan in free water
    for the length of the run and the two boundaries out of the problem.
    """
    return 4.0 * max_range * np.tan(np.radians(max_angle_deg)) + 2000.0


# --- The absolute calibration -----------------------------------------------


def test_the_free_field_is_the_unit_point_source_in_magnitude_and_in_phase() -> None:
    """The beam sum against ``exp(ikR)/R``, which is the whole calibration."""
    f, rmax = 100.0, 2000.0
    depth = _free_space_column(rmax)
    zs = depth / 2.0
    r = np.array([500.0, 1000.0, 1500.0, 2000.0])
    z = zs + np.array([-400.0, -100.0, 0.0, 100.0, 400.0])
    res = gaussian_beams(f, [0.0, depth], [_C, _C], source_depth=zs,
                         max_range=rmax, ranges_m=r, receiver_depths_m=z,
                         range_step=10.0)
    assert isinstance(res, GaussianBeamResult)
    slant = np.hypot(r[None, :], z[:, None] - zs)
    exact = np.exp(1j * 2.0 * np.pi * f / _C * slant) / slant
    ratio = res.pressure / exact
    # 1.3e-4 in amplitude and 1.8e-3 rad in phase. A missing 4*pi would be a
    # factor of 12.6, the printed e^(+i pi/4) exactly pi/2, and a missing
    # conjugation twice the phase of the ratio, which grows with range.
    assert np.abs(np.abs(ratio) - 1.0).max() < 5e-4
    assert np.abs(np.angle(ratio)).max() < 5e-3
    assert np.abs(res.propagation_loss - 20.0 * np.log10(slant)).max() < 0.01


def test_one_reflection_is_the_two_ray_field() -> None:
    """Source and its negative image, with the source many beams off the surface.

    The beam's own half-width has to be small against the source depth for this
    to be the two-ray field at all: the fan splits at the grazing angle into
    beams that reflect and beams that do not, so a beam as wide as the source is
    deep gets only half of the launch-angle integral each arrival needs. A
    kilometre of water over a 150 m beam is comfortably clear of that.
    """
    f, depth, zs, zr = 100.0, 20_000.0, 1000.0, 1050.0
    r = np.array([1000.0, 2000.0, 4000.0])
    k = 2.0 * np.pi * f / _C
    direct, image = np.hypot(r, zr - zs), np.hypot(r, zr + zs)
    exact = -20.0 * np.log10(np.abs(np.exp(1j * k * direct) / direct
                                    - np.exp(1j * k * image) / image))
    res = gaussian_beams(f, [0.0, depth], [_C, _C], source_depth=zs,
                         max_range=4200.0, ranges_m=r,
                         receiver_depths_m=np.array([zr]), range_step=10.0)
    assert np.abs(res.propagation_loss[0] - exact).max() < 0.05


# --- The ideal waveguide ----------------------------------------------------
#
# Images of a point source between two pressure-release planes sit at
# 2 m D + z_s with strength +1 and at 2 m D - z_s with strength -1, for every
# integer m: the pair at m = 0 cancels at z = 0 and the whole lattice cancels at
# z = D. The sum converges slowly (each term falls off only as 1/R and it is the
# alternating signs that close it), hence the term count below.

_GUIDE = {"water_depth": 1000.0, "frequency": 300.0,
          "source_depth": 300.0, "receiver_depth": 600.0}


def _image_source_loss(
    ranges: np.ndarray, *, water_depth: float, frequency: float,
    source_depth: float, receiver_depth: float, max_angle_deg: float | None = None,
    n_images: int = 40_000,
) -> np.ndarray:
    """Propagation loss of the ideal guide, optionally cut to a fan half-angle."""
    k = 2.0 * np.pi * frequency / _C
    m = np.arange(-n_images, n_images + 1)[:, None]
    rr = np.asarray(ranges, dtype=np.float64)[None, :]
    up = receiver_depth - (2.0 * m * water_depth + source_depth)
    down = receiver_depth - (2.0 * m * water_depth - source_depth)
    r_up, r_down = np.hypot(rr, up), np.hypot(rr, down)
    take_up = take_down = 1.0
    if max_angle_deg is not None:
        slope = np.tan(np.radians(max_angle_deg))
        take_up = np.abs(up) < slope * rr
        take_down = np.abs(down) < slope * rr
    field = (np.exp(1j * k * r_up) / r_up * take_up
             - np.exp(1j * k * r_down) / r_down * take_down).sum(axis=0)
    return np.asarray(-20.0 * np.log10(np.abs(field)))


def test_the_ideal_waveguide_matches_the_image_source_sum() -> None:
    """With the fan opened wide enough, to a hundredth of a decibel.

    Everything is in this one number: the reflection coefficients of both
    boundaries, the impulse the spreading takes at each of them, the branch of
    the square root through a path that has bounced dozens of times, and the
    ladder of folded receiver images without which the answer is several
    decibels light.
    """
    r = np.array([2000.0])
    exact = _image_source_loss(r, **_GUIDE)
    res = gaussian_beams(
        _GUIDE["frequency"], [0.0, _GUIDE["water_depth"]], [_C, _C],
        source_depth=_GUIDE["source_depth"], max_range=2200.0, ranges_m=r,
        receiver_depths_m=np.array([_GUIDE["receiver_depth"]]),
        max_angle_deg=88.0, range_step=2.0, beam_width=100.0)
    assert np.abs(res.propagation_loss[0] - exact).max() < 0.02


def test_a_narrower_fan_costs_exactly_what_cutting_the_image_sum_costs() -> None:
    """The 80-degree error is the fan's truncation, not the method's.

    A guide with two perfect reflectors is the worst case a beam fan can be
    asked for, because nothing but 1/R attenuates the steep multiple bounces and
    the image lattice never dies out. Cutting the *oracle* to the same half
    angle shows where the difference comes from: the solver tracks the truncated
    lattice far more closely than either tracks the complete one.
    """
    r = np.array([2000.0])
    complete = _image_source_loss(r, **_GUIDE)
    truncated = _image_source_loss(r, max_angle_deg=80.0, **_GUIDE)
    res = gaussian_beams(
        _GUIDE["frequency"], [0.0, _GUIDE["water_depth"]], [_C, _C],
        source_depth=_GUIDE["source_depth"], max_range=2200.0, ranges_m=r,
        receiver_depths_m=np.array([_GUIDE["receiver_depth"]]),
        max_angle_deg=80.0, range_step=10.0, beam_width=100.0)
    missed = float(np.abs(truncated - complete).max())
    assert missed > 0.1, "the fan has to bite for this test to mean anything"
    assert np.abs(res.propagation_loss[0] - truncated).max() < 0.5 * missed


@pytest.mark.parametrize(
    ("bottom", "at_bottom"), [("pressure-release", 0.0), ("rigid", 2.0)])
def test_both_boundary_conditions_fall_out_of_the_folded_images(
    bottom: str, at_bottom: float,
) -> None:
    """Nothing imposes them; they are what the image ladder sums to.

    At the surface the two families of images coincide with opposite signs and
    cancel whatever the bottom is. At a rigid bottom they coincide with equal
    signs, so the field doubles against its mid-column scale and its depth
    derivative cancels, which is the Neumann condition; at a pressure-release
    bottom they cancel there too.
    """
    depth = _GUIDE["water_depth"]
    res = gaussian_beams(
        _GUIDE["frequency"], [0.0, depth], [_C, _C],
        source_depth=_GUIDE["source_depth"], max_range=2200.0,
        ranges_m=np.array([2000.0]),
        receiver_depths_m=np.array([0.0, 0.5 * depth, depth]),
        max_angle_deg=85.0, range_step=5.0, beam_width=100.0, bottom=bottom)
    scale = np.abs(res.pressure[1, 0])
    assert np.abs(res.pressure[0, 0]) / scale < 1e-3
    assert np.abs(res.pressure[2, 0]) / scale == pytest.approx(at_bottom, abs=0.2)


# --- What the beam itself does ----------------------------------------------

_FREE_BEAM_WIDTH = 120.0
_FREE_BEAM_FREQUENCY = 200.0


def _free_space_beam() -> tuple[GaussianBeamResult, np.ndarray, float]:
    """One free-space fan, with its arc lengths and its Rayleigh range.

    The run has to reach well past :math:`kW_0^2/2` for the hyperbola's scale to
    be measurable at all, so it is traced to twice that; the field itself is
    asked for at one point, since only the per-ray beam history is under test.
    """
    f, w0 = _FREE_BEAM_FREQUENCY, _FREE_BEAM_WIDTH
    rayleigh = (2.0 * np.pi * f / _C) * w0**2 / 2.0
    rmax = 2.0 * rayleigh * np.cos(np.radians(45.0))
    depth = _free_space_column(rmax, 45.0)
    res = gaussian_beams(f, [0.0, depth], [_C, _C], source_depth=depth / 2.0,
                         max_range=rmax, max_angle_deg=45.0, beam_width=w0,
                         n_beams=9, range_step=rmax / 400.0, n_depth_points=2,
                         ranges_m=np.array([rmax]))
    arc = res.ray_ranges / np.cos(np.radians(res.launch_angles))[:, None]
    return res, arc, rayleigh


def test_the_beam_half_width_is_the_free_space_hyperbola() -> None:
    r"""Eq. (3.86) with the waist at the source, to machine precision.

    In a homogeneous medium ``p`` stays at 1 and ``q(s) = q(0) + c_0 s``, so
    Eq. (3.89) collapses to :math:`W(s) = W_0\sqrt{1 + (2s/(kW_0^2))^2}`, a
    hyperbola with its waist at the source and the Rayleigh range
    :math:`kW_0^2/2` for its scale. The arc length is ``r/cos(theta_0)`` because
    the rays are straight.
    """
    res, arc, rayleigh = _free_space_beam()
    w0 = res.initial_beam_width
    exact = w0 * np.sqrt(1.0 + (arc / rayleigh) ** 2)
    assert res.beam_widths.shape == exact.shape
    assert np.allclose(res.beam_widths, exact, rtol=1e-12)
    # The waist is at the source, and one Rayleigh range out the beam has
    # widened by exactly sqrt(2), which is what fixes the scale of the
    # hyperbola rather than merely its shape.
    assert res.beam_widths[:, 0] == pytest.approx(w0, rel=1e-13)
    at_rayleigh = np.interp(rayleigh, arc[0], res.beam_widths[0])
    assert at_rayleigh == pytest.approx(w0 * np.sqrt(2.0), rel=1e-4)
    assert np.all(np.diff(res.beam_widths, axis=1) > 0.0)


def test_the_wavefront_curvature_is_the_complex_source_point_one() -> None:
    r"""Eq. (3.90) against Eq. (3.85), which is derived a different way.

    Sect. 3.5.1 gets :math:`K(x) = x/(x^2 + a^2)` by offsetting a point source
    into the complex plane, with :math:`a = kW_0^2/2` the same Rayleigh range;
    Sect. 3.5.2 gets its :math:`K` from the dynamic ray equations. They have to
    agree, and the sign is the one that belongs to the conjugated field this
    result exposes: positive for a beam spreading away from its waist.
    """
    res, arc, rayleigh = _free_space_beam()
    assert np.allclose(res.wavefront_curvatures, arc / (arc**2 + rayleigh**2),
                       rtol=1e-12, atol=1e-15)
    # Flat at the waist, and the curvature peaks one Rayleigh range out.
    assert np.all(res.wavefront_curvatures[:, 0] == 0.0)
    assert arc[0, int(np.argmax(res.wavefront_curvatures[0]))] == pytest.approx(
        rayleigh, rel=0.02)


def test_the_conserved_wronskian_is_why_the_spreading_never_vanishes() -> None:
    r"""``q_R p_I - q_I p_R`` holds at ``-omega W_0^2/2`` through everything.

    This is the structural claim the whole method rests on, so it is worth
    measuring rather than asserting in prose. Eq. (3.58) is linear with real
    coefficients, so the real and imaginary parts of :math:`(q, p)` are two real
    solutions of it and their Wronskian is constant; the impulse at a profile
    node and the one at a reflection are both shears of unit determinant, so
    they cannot change it either. A constant that starts non-zero means ``q``
    can never vanish, hence no caustic singularity, and it also gives
    :math:`\mathrm{Im}[p/q] = -\omega W_0^2/(2|q|^2)`, so Eq. (3.89) becomes
    :math:`W = 2|q|/(\omega W_0)`, which is what the solver reports.

    The path below crosses two hard gradient kinks and works both boundaries,
    which is every kind of event the marcher knows how to apply.
    """
    f, w0, depth = 300.0, 60.0, 1000.0
    z_prof = np.array([0.0, 100.0, 300.0, depth])
    c_prof = np.array([1500.0, 1510.0, 1488.0, 1512.0])
    omega = 2.0 * np.pi * f
    angles = np.radians(np.array([-25.0, -8.0, 3.0, 17.0, 40.0]))
    c0 = float(np.interp(200.0, z_prof, c_prof))
    xi = np.cos(angles) / c0
    march = march_rays(
        _ocean_ray_derivative(z_prof, c_prof, xi), xi=xi,
        z0=np.full(angles.size, 200.0), zeta0=np.sin(angles) / c0,
        range_step=10.0, n_steps=1201, lower=0.0, upper=depth,
        dynamic=DynamicRays(np.full(angles.size, 0.5j * omega * w0**2),
                            np.full(angles.size, 1.0 + 0.0j), z_prof, c_prof))
    assert march.reflections.sum() > 10  # the path really does work both walls
    q = np.asarray(march.spreadings)
    p = np.asarray(march.spreading_slopes)
    wronskian = q.real * p.imag - q.imag * p.real
    expected = -0.5 * omega * w0**2
    assert np.abs(wronskian / expected - 1.0).max() < 1e-12
    # Which is the same as saying q stays away from the origin, by a margin
    # that no reflection or kink can close.
    assert np.abs(q[:, 1:]).min() > 0.0
    ratio = np.imag(p / q) * np.abs(q) ** 2 / expected
    assert np.abs(ratio - 1.0).max() < 1e-11


# --- Caustics and shadow zones ----------------------------------------------
#
# The n^2-linear profile of Jensen Eq. (3.77), c(z) = c0/sqrt(1 + 2.4 z/c0) with
# c0 = 1550 m/s, which Sect. 3.4.1 uses for exactly these two illustrations: a
# deep source makes a caustic and a shallow one makes a shadow zone.

_N2_C0, _N2_DEPTH = 1550.0, 1000.0
_N2_DEPTHS = np.linspace(0.0, _N2_DEPTH, 201)
_N2_SPEEDS = _N2_C0 / np.sqrt(1.0 + 2.4 * _N2_DEPTHS / _N2_C0)


def _geometric_spreading(
    source_depth: float, angles_deg: np.ndarray | list[float], *,
    max_range: float, n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(q, z)`` for the *geometric* initial conditions of Eq. (3.63).

    With ``q(0) = 0`` and ``p(0) = 1/c(0)`` the pair is the ray-tube spreading
    itself, ``r q = J`` (Eq. 3.64), and the classical amplitude of Eq. (3.65)
    divides by it. That is the configuration with closed-form answers and the
    one that has caustics, so it is the oracle for where the caustics are; the
    beam solver never runs it.
    """
    th = np.radians(np.atleast_1d(np.asarray(angles_deg, dtype=np.float64)))
    c0 = float(np.interp(source_depth, _N2_DEPTHS, _N2_SPEEDS))
    xi = np.cos(th) / c0
    march = march_rays(
        _ocean_ray_derivative(_N2_DEPTHS, _N2_SPEEDS, xi), xi=xi,
        z0=np.full(th.size, source_depth), zeta0=np.sin(th) / c0,
        range_step=max_range / (n_steps - 1), n_steps=n_steps, lower=0.0,
        upper=_N2_DEPTH,
        dynamic=DynamicRays(np.zeros(th.size), np.full(th.size, 1.0 / c0),
                            _N2_DEPTHS, _N2_SPEEDS))
    return np.asarray(march.spreadings, dtype=np.float64), march.positions


def test_the_field_is_finite_where_the_classical_amplitude_is_not() -> None:
    """A caustic pinned by bisection, then asked of the beam field.

    The launch angle is refined until the geometric spreading at a chosen range
    is zero to the last bits, which is the definition of a caustic: the ray tube
    has closed and Eq. (3.65)'s ``1/sqrt(r q)`` is unbounded there. Ray theory
    answers infinity, and the whole point of the beams is that they answer a
    number.
    """
    zs, rmax, n_steps = 992.5, 4000.0, 2001
    column = n_steps // 2
    ranges = np.linspace(0.0, rmax, n_steps)
    scan = np.linspace(-25.0, -5.0, 41)
    q_scan, _ = _geometric_spreading(zs, scan, max_range=rmax, n_steps=n_steps)
    crossings = np.flatnonzero(np.diff(np.sign(q_scan[:, column])) != 0)
    assert crossings.size, "the scan has to straddle a caustic"
    lo, hi = scan[crossings[0]], scan[crossings[0] + 1]
    sign_lo = np.sign(_geometric_spreading(
        zs, [lo], max_range=rmax, n_steps=n_steps)[0][0, column])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q_mid, _z = _geometric_spreading(zs, [mid], max_range=rmax, n_steps=n_steps)
        lo, hi = (mid, hi) if np.sign(q_mid[0, column]) == sign_lo else (lo, mid)
    q_c, z_c = _geometric_spreading(zs, [0.5 * (lo + hi)], max_range=rmax,
                                    n_steps=n_steps)
    r_caustic, z_caustic = float(ranges[column]), float(z_c[0, column])
    tube = np.abs(ranges[1:] * q_c[0, 1:])
    # The classical amplitude is 1/sqrt(|r q|): seven orders of magnitude above
    # its own median along the very same ray, and unbounded under refinement.
    assert np.sqrt(np.median(tube) / abs(r_caustic * q_c[0, column])) > 1e6

    offsets = np.linspace(-120.0, 120.0, 121)
    res = gaussian_beams(600.0, _N2_DEPTHS, _N2_SPEEDS, source_depth=zs,
                         max_range=rmax, ranges_m=np.array([r_caustic]),
                         receiver_depths_m=z_caustic + offsets,
                         max_angle_deg=45.0, range_step=2.0)
    pl = res.propagation_loss[:, 0]
    assert np.all(np.isfinite(pl))
    assert pl.min() > 30.0  # finite, and not absurdly loud either
    # And it is a real focus rather than merely finite: the caustic stands well
    # above the interference either side of it.
    near = np.abs(offsets) < 20.0
    assert float(pl[near].min()) < float(np.median(pl)) - 5.0


def test_the_shadow_zone_decays_smoothly_without_going_silent() -> None:
    """Beyond the limiting ray, where ray theory has nothing at all to say.

    The shallow-source case of Sect. 3.4.1: with the source at 75 m in this
    profile no ray reaches that depth again beyond about 0.88 km, which
    :func:`ray_trace` is asked to confirm here rather than being taken on trust.
    Classical ray theory then returns silence; the exact solution decays
    gradually (Fig. 3.11) and so does this one.
    """
    zs, f = 77.5, 2000.0
    rays = ray_trace(_N2_DEPTHS, _N2_SPEEDS, source_depth=zs,
                     launch_angles_deg=np.linspace(-19.0, 19.0, 761),
                     max_range=1400.0, n_steps=1401)
    closest = np.abs(rays.depths - zs).min(axis=0)
    limiting = float(rays.ranges[0][np.flatnonzero(closest > 2.0)[0]])
    assert 800.0 < limiting < 950.0
    # Well inside the shadow, no ray comes within a hundred metres of the
    # receiver, so there is no eigenray to be found and no classical answer.
    assert closest[np.argmin(np.abs(rays.ranges[0] - 1300.0))] > 100.0

    r = np.linspace(600.0, 1350.0, 76)
    res = gaussian_beams(f, _N2_DEPTHS, _N2_SPEEDS, source_depth=zs,
                         max_range=1500.0, ranges_m=r,
                         receiver_depths_m=np.array([zs]),
                         max_angle_deg=30.0, range_step=2.0)
    pl = res.propagation_loss[0]
    assert np.all(np.isfinite(pl))
    # Nonzero everywhere: the beams' tails carry energy into the shadow, which
    # is the behaviour a wave solution has and a ray solution does not.
    assert pl.max() < 140.0
    # And decaying: incoherent averages over three successive windows, since the
    # shadow still carries the interference of whatever reached it.
    lit = r < limiting
    windows = [pl[lit], pl[(~lit) & (r < 1150.0)], pl[r >= 1150.0]]
    means = [-10.0 * np.log10(np.mean(10.0 ** (-w / 10.0))) for w in windows]
    assert means[0] < means[1] < means[2]
    assert means[2] - means[0] > 20.0


# --- Defaults, warnings and validation --------------------------------------


@pytest.mark.parametrize(
    ("frequency", "max_range", "water_depth", "expected"),
    [
        # sqrt(lambda r / pi) inside the book's 10-50 wavelength band.
        (100.0, 10_000.0, 5000.0, 218.6),
        (1000.0, 10_000.0, 5000.0, 69.1),
        # Too low a frequency for the optimum: the ten-wavelength floor lifts it.
        (20.0, 10_000.0, 5000.0, 750.0),
        # A shallow channel: the quarter-depth cap has the last word over both.
        (100.0, 10_000.0, 200.0, 50.0),
    ],
)
def test_the_default_beam_width_is_the_optimum_inside_its_clamps(
    frequency: float, max_range: float, water_depth: float, expected: float,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PhonometryWarning)
        res = gaussian_beams(
            frequency, [0.0, water_depth], [_C, _C],
            source_depth=0.5 * water_depth, max_range=max_range,
            ranges_m=np.array([max_range]), n_depth_points=2, n_beams=9,
            range_step=max_range / 4.0, max_angle_deg=20.0)
    assert res.initial_beam_width == pytest.approx(expected, rel=1e-3)


def test_the_answer_does_not_care_which_width_inside_the_band_is_used() -> None:
    """Doubling and halving the beam width moves the loss by hundredths of a dB.

    Which is what makes a defensible default cost nothing: the choice of
    :math:`W_0` is "a matter of current research" only outside the band the book
    recommends.
    """
    r = np.array([2000.0])
    losses = []
    for w0 in (60.0, 120.0, 240.0):
        res = gaussian_beams(
            _GUIDE["frequency"], [0.0, _GUIDE["water_depth"]], [_C, _C],
            source_depth=_GUIDE["source_depth"], max_range=2200.0, ranges_m=r,
            receiver_depths_m=np.array([_GUIDE["receiver_depth"]]),
            max_angle_deg=85.0, range_step=5.0, beam_width=w0)
        losses.append(float(res.propagation_loss[0, 0]))
    assert max(losses) - min(losses) < 0.2  # measured 0.13 dB


def test_a_source_on_a_profile_kink_is_warned_about() -> None:
    z = np.array([0.0, 100.0, 300.0, 1000.0])
    c = np.array([1500.0, 1510.0, 1488.0, 1512.0])
    with pytest.warns(PhonometryWarning, match="gradient discontinuity"):
        gaussian_beams(200.0, z, c, source_depth=100.0, max_range=1000.0,
                       ranges_m=np.array([1000.0]), n_depth_points=2, n_beams=9,
                       range_step=100.0, max_angle_deg=30.0)
    # A metre off the node is a different problem, and silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhonometryWarning)
        gaussian_beams(200.0, z, c, source_depth=101.0, max_range=1000.0,
                       ranges_m=np.array([1000.0]), n_depth_points=2, n_beams=9,
                       range_step=100.0, max_angle_deg=30.0)


def test_a_beam_wider_than_the_channel_is_warned_about() -> None:
    with pytest.warns(PhonometryWarning, match="quarter of the water depth"):
        gaussian_beams(200.0, [0.0, 400.0], [_C, _C], source_depth=200.0,
                       max_range=1000.0, ranges_m=np.array([1000.0]),
                       n_depth_points=2, n_beams=9, range_step=20.0,
                       max_angle_deg=30.0, beam_width=150.0)


def test_a_step_that_cannot_follow_the_steepest_beam_is_warned_about() -> None:
    with pytest.warns(PhonometryWarning, match="steepest beam"):
        gaussian_beams(200.0, [0.0, 400.0], [_C, _C], source_depth=200.0,
                       max_range=2000.0, ranges_m=np.array([2000.0]),
                       n_depth_points=2, n_beams=9, range_step=100.0,
                       max_angle_deg=85.0)


def test_invalid_inputs_rejected() -> None:
    iso = ([0.0, 1000.0], [_C, _C])
    with pytest.raises(ValueError, match="source_depth"):
        gaussian_beams(200.0, *iso, source_depth=1200.0)
    with pytest.raises(ValueError, match="max_angle_deg"):
        gaussian_beams(200.0, *iso, source_depth=500.0, max_angle_deg=90.0)
    with pytest.raises(ValueError, match="bottom"):
        gaussian_beams(200.0, *iso, source_depth=500.0, bottom="sandy")
    with pytest.raises(ValueError, match="range_step"):
        gaussian_beams(200.0, *iso, source_depth=500.0, max_range=100.0,
                       range_step=200.0)
    with pytest.raises(ValueError, match="n_beams"):
        gaussian_beams(200.0, *iso, source_depth=500.0, n_beams=1)
    with pytest.raises(ValueError, match="ranges_m"):
        gaussian_beams(200.0, *iso, source_depth=500.0, ranges_m=[-1.0])
    with pytest.raises(ValueError, match="receiver_depths_m"):
        gaussian_beams(200.0, *iso, source_depth=500.0,
                       ranges_m=[500.0], receiver_depths_m=[])
    with pytest.raises(ValueError, match="n_depth_points"):
        gaussian_beams(200.0, *iso, source_depth=500.0, ranges_m=[500.0],
                       n_depth_points=1)


def test_the_result_lines_up_with_the_parabolic_equation_grid_and_plots() -> None:
    """Same shape, same depths, and a ``.plot()`` in the sibling's own frame."""
    from phonometry.underwater.propagation.numerical import parabolic_equation

    iso = ([0.0, 1000.0], [_C, _C])
    beams = gaussian_beams(200.0, *iso, source_depth=500.0, max_range=2000.0,
                           range_step=100.0, n_depth_points=32, n_beams=101,
                           max_angle_deg=45.0)
    pe = parabolic_equation(200.0, *iso, source_depth=500.0, max_range=2000.0,
                            range_step=100.0, n_depth_points=32)
    assert np.allclose(beams.depths, pe.depths)
    assert beams.propagation_loss.shape == (32, beams.ranges.size)
    assert beams.propagation_loss.shape[0] == pe.propagation_loss.shape[0]
    with np.errstate(divide="ignore"):
        assert np.allclose(beams.propagation_loss,
                           -20.0 * np.log10(np.abs(beams.pressure)))
    assert beams.plot() is not None
    assert beams.plot(language="es") is not None
