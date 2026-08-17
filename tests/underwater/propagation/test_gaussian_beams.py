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

* **The branch of the square root**, in two steps, because no single comparison
  reaches it. Eq. (3.58) is linear with real coefficients and Eq. (3.91) starts
  :math:`q` on the imaginary axis with :math:`p` real, so :math:`\mathrm{Re}[q]`
  *is* the geometric spreading and every caustic puts :math:`q` on that axis.
  The unwrapped argument then has to read :math:`-(2k+1)\pi/2` at the
  :math:`k`-th caustic, a closed-form ladder which says :math:`\sqrt{q}` turns
  by :math:`-\pi/2` there: Jensen Eq. (3.79)'s :math:`(-i)^m` recovered instead
  of imposed. That pins the branch; a hand-built single beam handed to
  :func:`_beam_influence` with its argument set one turn below the principal
  value then pins that the sum spends it, since one turn in :math:`\arg q` is a
  factor of -1 in the field.

  It is worth saying what is *not* here, because it was looked for. The branch
  changes the summed field by 1.7 to 3.9 dB, and only at a caustic. That is
  under the beam method's own reproducibility in the only configurations that
  cross one: against the parabolic equation on a caustic cut the two disagree by
  5 dB in the mean, against :func:`normal_modes` by 5.4 dB, and against a
  closed-form two-ray sum with the KMAH factor put in by hand (exact parabolic
  rays for this profile, checked to millimetres) by 4.4 dB, the fringe periods
  agreeing while their positions slip. Doubling the beam width moves the same
  cut by 12 dB. So the two tests above are white box on purpose: a field-value
  oracle sharp enough to see the branch does not exist here.
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


def test_the_near_field_survives_a_beam_whose_foot_lands_on_the_source() -> None:
    r"""The one place Eq. (3.88) still divides by something unprotected.

    ``r q`` is the Jacobian of Eq. (3.46), range times ray-tube width. Complex
    initial data keeps the second factor off zero, which is the method; nothing
    keeps the *first* off zero, and it vanishes on the axis every ray starts
    from. The geometry that finds it is not exotic: the foot of the
    perpendicular from a receiver lands exactly on the source for whichever beam
    is launched perpendicular to the source-receiver line, so it is built here
    on purpose. A fan of 321 beams over +-80 degrees is spaced 0.5 degrees and
    therefore contains -45.0 exactly, and the receivers sit exactly 45 degrees
    below the source, well inside the fan so its truncation is not in the
    answer.

    Unfloored, the surviving cell had ``r`` at 1.4e-14 m and the field came out
    119 dB too loud, which is a *negative* propagation loss: louder at 71 m than
    at the 1 m the result is normalised to. That is the assertion below, and it
    is the one worth making, because it is the failure that cannot be mistaken
    for physics.
    """
    f, rmax = 100.0, 2000.0
    depth = _free_space_column(rmax)
    zs = depth / 2.0
    offs = np.array([50.0, 100.0, 200.0, 400.0])
    res = gaussian_beams(f, [0.0, depth], [_C, _C], source_depth=zs,
                         max_range=rmax, ranges_m=offs, receiver_depths_m=zs + offs,
                         max_angle_deg=80.0, n_beams=321)
    # Not asserted as exact equality: the fan is built in radians and read back
    # in degrees, so -45 lands on a rung to within a rounding rather than by
    # construction. A rounding of a degree is close enough to be the same test.
    assert np.abs(res.launch_angles + 45.0).min() < 1e-9, (
        "the singular beam has to be in the fan")
    slant = np.hypot(res.ranges[None, :], res.depths[:, None] - zs)
    err = res.propagation_loss - 20.0 * np.log10(slant)
    assert np.all(np.isfinite(res.propagation_loss))
    # Measured -82.09 dB before the floor, +37.52 after.
    assert res.propagation_loss.min() > 0.0, "no receiver is louder than the source"
    # On the 45-degree diagonal, worst 0.73 dB after and 119 dB before.
    diagonal = np.diag(err)
    assert np.abs(diagonal).max() < 1.5
    # By 400 m the floor is out of the answer entirely: 0.003 dB after, 2.7 before.
    assert abs(diagonal[-1]) < 0.05


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


def _beam_spreading(
    source_depth: float, angles_deg: np.ndarray | list[float], *,
    max_range: float, n_steps: int, omega: float, beam_width: float,
) -> np.ndarray:
    """``q`` for the *beam* initial conditions of Eq. (3.91), same marcher."""
    th = np.radians(np.atleast_1d(np.asarray(angles_deg, dtype=np.float64)))
    c0 = float(np.interp(source_depth, _N2_DEPTHS, _N2_SPEEDS))
    xi = np.cos(th) / c0
    march = march_rays(
        _ocean_ray_derivative(_N2_DEPTHS, _N2_SPEEDS, xi), xi=xi,
        z0=np.full(th.size, source_depth), zeta0=np.sin(th) / c0,
        range_step=max_range / (n_steps - 1), n_steps=n_steps, lower=0.0,
        upper=_N2_DEPTH,
        dynamic=DynamicRays(np.full(th.size, 0.5j * omega * beam_width**2),
                            np.full(th.size, 1.0 + 0.0j), _N2_DEPTHS, _N2_SPEEDS))
    return np.asarray(march.spreadings, dtype=np.complex128)


def test_the_tracked_branch_of_the_square_root_is_the_kmah_index() -> None:
    r"""Eq. (3.79)'s :math:`(-i)^m`, recovered from the branch rather than imposed.

    Ray theory has to count caustics and multiply by :math:`(-i)^m` by hand;
    Sect. 3.5's claim is that a complex :math:`q` carries that phase on its own,
    which is why this module has no KMAH index in it. The claim is exactly
    checkable, because Eq. (3.58) is linear with real coefficients and
    Eq. (3.91) starts :math:`q` at :math:`i\omega W_0^2/2` with :math:`p` real:
    the real and imaginary parts evolve independently, and
    :math:`\mathrm{Re}[q]` is then the *geometric* spreading of Eq. (3.63) to
    the last bits. So a caustic, where the geometric spreading vanishes, is
    exactly where the beam's :math:`q` is purely imaginary, and the question of
    the branch is the question of which way it goes round.

    It goes round one way, always, by :math:`\pi` per caustic. The unwrapped
    argument starts at :math:`+\pi/2` and reads :math:`-(2k+1)\pi/2` at the
    :math:`k`-th caustic, a descending ladder with no jumps in it, so
    :math:`\sqrt{q}` picks up :math:`e^{-i\pi/2} = -i` at each one. That is
    Eq. (3.79), arrived at from the other end.

    A principal-value square root taken sample by sample sees the same ladder
    folded into :math:`\pm\pi/2` and is wrong from the first caustic on, by
    :math:`2\pi` in the argument on rays that cross one and by :math:`6\pi` on
    the shallow rays that cross six. That difference is asserted too, so this
    stays a test of something the solver actually depends on.
    """
    zs, rmax, n_steps = 992.5, 4000.0, 4001
    omega, w0 = 2.0 * np.pi * 600.0, 35.9
    angles = np.array([-25.0, -20.0, -13.3886, -8.0, -3.0, 3.0, 10.0, 20.0, 30.0])
    q = _beam_spreading(zs, angles, max_range=rmax, n_steps=n_steps,
                        omega=omega, beam_width=w0)

    # Re[q] is the geometric spreading, which is what puts caustics on the axis.
    geometric, _z = _geometric_spreading(zs, angles, max_range=rmax, n_steps=n_steps)
    c0 = float(np.interp(zs, _N2_DEPTHS, _N2_SPEEDS))
    assert np.abs(q.real - c0 * geometric).max() / np.abs(q.real).max() < 1e-10

    crossed, deep = 0, 0
    for i in range(angles.size):
        real, phase = q[i].real, np.unwrap(np.angle(q[i]))
        assert phase[0] == pytest.approx(0.5 * np.pi)
        # Interpolate onto each zero of Re[q] rather than the sample beside it.
        zeros = np.flatnonzero(np.sign(real[1:-1]) != np.sign(real[2:])) + 1
        for k_th, k in enumerate(zeros):
            frac = real[k] / (real[k] - real[k + 1])
            at = phase[k] + frac * (phase[k + 1] - phase[k])
            assert at == pytest.approx(-(2 * k_th + 1) * 0.5 * np.pi, abs=5e-3)
        crossed += zeros.size
        # The whole point: past the second caustic the ladder has left the
        # principal strip, and a principal value is a different function.
        if zeros.size >= 2:
            assert np.abs(phase - np.angle(q[i])).max() > 6.0
            deep += 1
    assert crossed >= 20, "the fan has to cross plenty of caustics"
    assert deep >= 4, "and several rays have to get past the second one"

    # And the march resolves the winding rather than aliasing it: at a twentieth
    # of the sampling the ladder is the same one, so no rung was ever skipped.
    coarse = _beam_spreading(zs, angles, max_range=rmax, n_steps=201,
                             omega=omega, beam_width=w0)
    fine_turn = np.unwrap(np.angle(q), axis=1)[:, -1]
    coarse_turn = np.unwrap(np.angle(coarse), axis=1)[:, -1]
    assert np.abs(fine_turn - coarse_turn).max() < 0.5 * np.pi


def test_the_influence_sum_uses_the_tracked_branch_and_not_the_principal_one() -> None:
    r"""One beam, by hand, with a winding the principal value cannot see.

    The test above pins the branch as a property of :math:`q`; this one pins
    that the sum actually spends it. A single horizontal beam is handed to
    :func:`_beam_influence` with the receiver exactly on its axis, so
    :math:`n = 0`, the extrapolation along the ray is nothing, and Eq. (3.88)
    collapses to :math:`A\sqrt{c/(r q)}\,e^{-i\omega\tau}` with only the branch
    left to get wrong. Its tracked argument is set a full turn below the
    principal value, which is what a ray that has passed two caustics carries,
    and a full turn in :math:`\arg q` is half a turn in :math:`\sqrt{q}`: the
    two readings differ by a factor of exactly -1, not by a little.

    The water column is made enormous and the beam's reach tiny so the folded
    images of the receiver are all thrown out and the one beam is the whole
    answer.
    """
    from phonometry.underwater.propagation.numerical import (
        _beam_influence,
        _BeamSamples,
    )

    c, r_col, omega, w0 = 1500.0, 3000.0, 2.0 * np.pi * 100.0, 40.0
    q = 1.0e6 * (0.3 + 0.7j)
    p = 2.0e-4 * (1.0 - 0.5j)
    tau, z_ray = 2.0, 5.0e5
    one = np.array([[1.0]])
    samples = _BeamSamples(
        xi=np.array([[1.0 / c]]),
        column_range=np.array([[r_col]]),
        range_offset=np.array([[0.0]]),
        depth=np.array([[z_ray]]),
        vertical=np.array([[0.0]]),
        speed=one * c,
        spreading=np.array([[q]]),
        slope=np.array([[p]]),
        time=one * tau,
        # A ray two caustics along: one full turn under the principal value.
        phase=np.array([[np.angle(q) - 2.0 * np.pi]]),
        weight=np.array([[1.0 + 0.0j]]),
        reach=np.array([50.0]),
    )
    got = _beam_influence(samples, np.array([z_ray]), water_depth=1.0e6,
                          bottom_reflection=1.0, omega=omega, beam_width=w0)

    expected = (np.sqrt(c / r_col) / np.sqrt(abs(q))
                * np.exp(-0.5j * (np.angle(q) - 2.0 * np.pi) - 1j * omega * tau))
    assert got.shape == (1, 1)
    assert got[0, 0] == pytest.approx(expected, rel=1e-12)
    # The principal value is the same number with the opposite sign, so this is
    # not a tolerance that a rounding change could drift across.
    principal = (np.sqrt(c / r_col) / np.sqrt(abs(q))
                 * np.exp(-0.5j * np.angle(q) - 1j * omega * tau))
    assert principal == pytest.approx(-expected, rel=1e-12)
    assert abs(got[0, 0] - principal) == pytest.approx(2.0 * abs(expected), rel=1e-12)


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

    # A cut across the caustic, kept inside the water column: the solver has
    # nothing to say about a receiver under the seabed and refuses to pretend.
    reach = min(120.0, z_caustic - 1.0, _N2_DEPTH - z_caustic - 1.0)
    offsets = np.linspace(-reach, reach, 121)
    res = gaussian_beams(600.0, _N2_DEPTHS, _N2_SPEEDS, source_depth=zs,
                         max_range=rmax, ranges_m=np.array([r_caustic]),
                         receiver_depths_m=z_caustic + offsets,
                         max_angle_deg=45.0, range_step=2.0)
    pl = res.propagation_loss[:, 0]
    assert np.all(np.isfinite(pl))
    assert pl.min() > 30.0  # finite, and not absurdly loud either
    # And it is a real focus rather than merely finite: the caustic stands well
    # above the interference either side of it.
    near = np.abs(offsets) < 0.2 * reach
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
    with pytest.raises(ValueError, match="receiver_depths_m"):
        gaussian_beams(200.0, *iso, source_depth=500.0, ranges_m=[500.0],
                       receiver_depths_m=[1200.0])  # below the seabed
    with pytest.raises(ValueError, match="ranges_m"):
        # Past the march there is no column to read a beam off, and answering
        # by extrapolating the last one would be a silent wrong answer.
        gaussian_beams(200.0, *iso, source_depth=500.0, max_range=1000.0,
                       ranges_m=[2000.0])
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


def test_the_colour_window_lands_on_the_field_and_not_beside_it() -> None:
    """That a raster came out is not evidence that anything is visible in it.

    A loss field is bounded below by its strongest arrival and unbounded above,
    so hanging a fixed 50 dB window on a *high* percentile lets the empty part
    of the picture decide where the window sits. This is the case that shows it:
    the source sits 7.5 m off the bottom of an n^2-linear profile, the up-going
    fan turns into a caustic ladder, and 18 per cent of the field is the wedge
    no beam reaches at all. Anchored at the 95th percentile the window was
    [110, 160] dB and 85 per cent of the finite field fell below it and clipped
    to one flat colour, caustics and all.

    The assertion is on the fraction of the field the window actually resolves,
    which is what "the picture shows something" means in numbers.
    """
    c0 = 1550.0
    z = np.linspace(0.0, 1000.0, 201)
    c = c0 / np.sqrt(1.0 + 2.4 * z / c0)
    beams = gaussian_beams(600.0, z, c, source_depth=992.5, max_range=2500.0,
                           range_step=25.0, max_angle_deg=45.0, n_depth_points=80)
    pl = beams.propagation_loss
    finite = pl[np.isfinite(pl)]
    assert np.isinf(pl).mean() > 0.1, "the un-illuminated wedge has to be there"
    ax = beams.plot()
    vmin, vmax = ax.get_images()[0].get_clim()
    assert vmax - vmin == pytest.approx(50.0)
    resolved = float(((finite > vmin) & (finite < vmax)).mean())
    assert resolved > 0.75, f"only {resolved:.1%} of the field is inside the window"
    # The bright end is where the window is pinned, so only the tail saturates.
    assert float((finite < vmin).mean()) < 0.1
