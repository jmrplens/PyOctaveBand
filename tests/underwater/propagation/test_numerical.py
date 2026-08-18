#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the numerical underwater-propagation solvers.

Oracles are analytic and independent of the implementation: the ideal
(pressure-release) waveguide's exact normal modes, the circular-arc ray paths of
a linear sound-speed gradient and the published closed form for the travel time
along them (Medwin & Clay 1998, Eq. (3.3.20)), and free-field spherical
spreading for the parabolic equation.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.propagation.numerical import (
    NormalModeResult,
    ParabolicEquationResult,
    RayTraceResult,
    normal_modes,
    parabolic_equation,
    ray_trace,
)

_ISO = (np.array([0.0, 100.0]), np.array([1500.0, 1500.0]))  # isovelocity 100 m


# --- Normal modes -----------------------------------------------------------


def test_normal_modes_ideal_waveguide_wavenumbers() -> None:
    # Ideal waveguide krm = sqrt(k^2 - (m pi/D)^2); pressure-release bottom.
    D, c, f = 100.0, 1500.0, 20.0
    res = normal_modes(f, *_ISO, source_depth=36.0, receiver_depth=46.0,
                       bottom="pressure-release", n_depth_points=800)
    assert isinstance(res, NormalModeResult)
    k = 2 * np.pi * f / c
    m_max = int(np.floor(k * D / np.pi))
    kr_exact = np.sqrt(k**2 - (np.arange(1, m_max + 1) * np.pi / D) ** 2)
    assert res.wavenumbers.size == m_max
    assert np.allclose(res.wavenumbers[:m_max], kr_exact, atol=1e-4)


def test_normal_modes_pl_matches_analytic_modal_sum() -> None:
    D, c, f, rho = 100.0, 1500.0, 20.0, 1000.0
    zs, zr = 36.0, 46.0
    r = np.linspace(100.0, 10_000.0, 200)
    res = normal_modes(f, *_ISO, source_depth=zs, receiver_depth=zr, ranges_m=r,
                       density=rho, n_depth_points=800)
    k = 2 * np.pi * f / c
    m = np.arange(1, int(np.floor(k * D / np.pi)) + 1)
    kzm = m * np.pi / D
    kr = np.sqrt(k**2 - kzm**2)
    psis = np.sqrt(2 * rho / D) * np.sin(kzm * zs)
    psir = np.sqrt(2 * rho / D) * np.sin(kzm * zr)
    modal = (psis * psir / np.sqrt(kr))[:, None] * np.exp(1j * kr[:, None] * r[None, :])
    field = (1j / (rho * np.sqrt(8 * np.pi * r))) * np.exp(-1j * np.pi / 4) * modal.sum(0)
    pl_exact = -20 * np.log10(np.abs(field) / (1 / (4 * np.pi)))
    assert np.max(np.abs(res.propagation_loss - pl_exact)) < 0.05


def test_normal_modes_rigid_bottom_wavenumbers() -> None:
    # Rigid bottom (Neumann): analytic kzm = (2m-1)pi/(2D), so
    # krm = sqrt(k^2 - ((2m-1)pi/2D)^2).
    D, c, f = 100.0, 1500.0, 50.0
    res = normal_modes(f, [0.0, D], [c, c], source_depth=30.0, receiver_depth=60.0,
                       bottom="rigid", n_depth_points=2000)
    k = 2 * np.pi * f / c
    m_max = int(np.floor((k * 2 * D / np.pi + 1) / 2))
    kzm = (2 * np.arange(1, m_max + 1) - 1) * np.pi / (2 * D)
    kr_exact = np.sqrt(k**2 - kzm**2)
    assert res.wavenumbers.size == m_max
    assert np.allclose(res.wavenumbers[:m_max], kr_exact, atol=1e-3)


# --- Ray tracing ------------------------------------------------------------


def test_ray_trace_linear_gradient_turning_depth() -> None:
    # Linear gradient: a ray turns where c(z_t) = c(zs)/cos(theta0).
    c0, g, D = 1500.0, 0.05, 2000.0
    prof = (np.array([0.0, D]), np.array([c0, c0 + g * D]))
    theta0 = 10.0
    res = ray_trace(*prof, source_depth=0.0, launch_angles_deg=[theta0],
                    max_range=10_500.0, n_steps=20_000)
    xi = np.cos(np.radians(theta0)) / c0
    z_turn = (1.0 / xi - c0) / g
    assert isinstance(res, RayTraceResult)
    assert res.depths[0].max() == pytest.approx(z_turn, abs=1.0)


def test_ray_trace_linear_gradient_circular_arc() -> None:
    # The ray path is an arc of a circle of radius R = 1/(xi |g|).
    c0, g = 1500.0, 0.05
    prof = (np.array([0.0, 2000.0]), np.array([c0, c0 + g * 2000.0]))
    res = ray_trace(*prof, source_depth=0.0, launch_angles_deg=[10.0],
                    max_range=10_400.0, n_steps=20_000)
    r, z = res.ranges[0], res.depths[0]
    xi = np.cos(np.radians(10.0)) / c0
    r_oracle = 1.0 / (xi * g)
    a = np.c_[r, z, np.ones_like(r)]
    sol, *_ = np.linalg.lstsq(a, r**2 + z**2, rcond=None)
    r_fit = np.sqrt(sol[2] + (sol[0] / 2) ** 2 + (sol[1] / 2) ** 2)
    assert r_fit == pytest.approx(r_oracle, rel=1e-3)


def test_ray_trace_constant_speed_is_straight() -> None:
    res = ray_trace(*_ISO, source_depth=50.0, launch_angles_deg=[0.0],
                    max_range=1000.0, n_steps=2000)
    # A horizontal ray in isovelocity water stays at the source depth.
    assert np.allclose(res.depths[0], 50.0, atol=1e-6)


# --- Ray travel times -------------------------------------------------------
#
# Oracle: the closed-form travel time through a layer of constant sound-speed
# gradient, Medwin & Clay, *Fundamentals of Acoustical Oceanography* (Academic
# Press, 1998), Sec. 3.3.4, Eq. (3.3.20) printed on p. 88,
#
#     t(n+1) - t(n) = |1/b| ln{ w(n+1)[1 + (1 - a^2 b^2 w(n)^2)^(1/2)]
#                             / ( w(n) [1 + (1 - a^2 b^2 w(n+1)^2)^(1/2)] ) },
#
# in which b = dc/dz is the gradient, w = c/b (their Eqs. (3.3.16)-(3.3.17)) and
# a = sin(theta_v)/c is the Snell invariant with theta_v measured from the
# vertical. Since a*b*w = a*c = sin(theta_v) = cos(theta), with theta the angle
# from the HORIZONTAL that this library launches its rays at, the printed
# equation reads, in the library's own variables,
#
#     t = (1/g) ln[ (c2/c1) (1 + sin theta_1) / (1 + sin theta_2) ],
#
# and the companion Eq. (3.3.21) gives the range as
# r = (sin theta_1 - sin theta_2)/(a g), i.e. sin theta_2 = sin theta_1 - xi g r
# with xi = cos(theta_1)/c1. The numbers below were evaluated from those two
# printed equations at 40 decimal digits (mpmath) and rounded to 1e-11 s; no
# part of them comes from running the solver.
_MEDWIN_C0, _MEDWIN_G = 1500.0, 0.05  # c(z) = 1500 + 0.05 z, launch 10 deg down
#: (range in m, travel time in s) for a 10 deg ray launched at the surface.
_MEDWIN_TRAVEL_TIME = [
    (2_000.0, 1.34017403528),   # still descending
    (5_000.0, 3.31823980910),   # just short of the 5289.8 m turning range
    (8_000.0, 5.29257808047),   # climbing again, past the turn
    (10_000.0, 6.62594192637),  # back up at 96.4 m depth
]


def test_ray_travel_time_matches_iso_gradient_closed_form() -> None:
    """Travel time against Medwin & Clay (1998) Eq. (3.3.20), p. 88.

    The closed form for a layer of constant gradient, evaluated independently of
    this library (see the derivation above the table), pinned at four ranges
    that straddle the ray's turning point.
    """
    prof = (np.array([0.0, 2000.0]),
            np.array([_MEDWIN_C0, _MEDWIN_C0 + _MEDWIN_G * 2000.0]))
    res = ray_trace(*prof, source_depth=0.0, launch_angles_deg=[10.0],
                    max_range=10_000.0, n_steps=20_001)
    assert res.travel_times.shape == res.depths.shape
    assert res.travel_times[0, 0] == 0.0
    for rng, t_ref in _MEDWIN_TRAVEL_TIME:
        i = int(np.argmin(np.abs(res.ranges[0] - rng)))
        assert res.ranges[0, i] == pytest.approx(rng)
        assert res.travel_times[0, i] == pytest.approx(t_ref, abs=1e-9), rng


def test_ray_travel_time_deeper_turning_ray_takes_longer() -> None:
    """A ray that turns deeper spends longer getting to its turning point.

    Steeper launch means a deeper turn (Snell: c(z_t) = c(z_s)/cos theta_0), and
    Medwin & Clay Eq. (3.3.20) taken from the source to that turn collapses to
    t_turn = (1/g) ln[(1 + sin theta_0)/cos theta_0], which is strictly
    increasing in theta_0 and, notably, free of the sound speed itself.
    """
    g = _MEDWIN_G
    prof = (np.array([0.0, 2000.0]),
            np.array([_MEDWIN_C0, _MEDWIN_C0 + g * 2000.0]))
    angles = [4.0, 8.0, 12.0, 16.0]
    res = ray_trace(*prof, source_depth=0.0, launch_angles_deg=angles,
                    max_range=12_000.0, n_steps=24_001)
    turn_times, turn_depths = [], []
    for k, angle in enumerate(angles):
        th = np.radians(angle)
        xi = np.cos(th) / _MEDWIN_C0
        r_turn = np.sin(th) / (xi * g)  # Eq. (3.3.21) with sin theta_2 = 0
        t_turn = np.log((1.0 + np.sin(th)) / np.cos(th)) / g
        got = float(np.interp(r_turn, res.ranges[k], res.travel_times[k]))
        assert got == pytest.approx(t_turn, abs=1e-6), angle
        turn_times.append(got)
        turn_depths.append((1.0 / xi - _MEDWIN_C0) / g)
    assert np.all(np.diff(turn_depths) > 0.0)  # steeper turns deeper
    assert np.all(np.diff(turn_times) > 0.0)   # and takes longer to get there


def test_ray_travel_time_constant_profile_is_range_over_speed() -> None:
    # With no gradient the ray is straight, its arc is range/cos(theta0) long and
    # it is covered at c, so t = r/(c cos theta0); horizontally, exactly r/c.
    res = ray_trace(*_ISO, source_depth=50.0, launch_angles_deg=[0.0, 20.0],
                    max_range=1500.0, n_steps=2000)
    r, c = res.ranges[0], 1500.0
    assert np.allclose(res.travel_times[0], r / c, rtol=0.0, atol=1e-12)
    assert np.allclose(res.travel_times[1], r / (c * np.cos(np.radians(20.0))),
                       rtol=0.0, atol=1e-12)


# --- Ray arc lengths --------------------------------------------------------


def test_ray_arc_length_constant_profile_is_range_over_cosine() -> None:
    # The straight ray's own geometry: r/cos(theta0) of path to cover r of
    # range. The integrand 1/(xi c) is constant, so this is exact and the
    # tolerance is accumulation arithmetic.
    res = ray_trace(*_ISO, source_depth=50.0, launch_angles_deg=[0.0, 20.0],
                    max_range=1500.0, n_steps=2000)
    r = res.ranges[0]
    assert res.arc_lengths.shape == res.depths.shape
    assert res.arc_lengths[0, 0] == 0.0
    assert np.allclose(res.arc_lengths[0], r, rtol=0.0, atol=1e-9)
    assert np.allclose(res.arc_lengths[1], r / np.cos(np.radians(20.0)),
                       rtol=0.0, atol=1e-9)


def test_ray_arc_length_matches_the_circular_arc_through_the_turn() -> None:
    r"""Arc length against the circular-arc closed form, through a turning point.

    In a layer of constant gradient the ray is an arc of a circle of radius
    :math:`R = 1/(\xi g)` (the geometry test above pins that radius from the
    traced path), and Snell's law makes the sine of the local angle fall
    linearly in range, :math:`\sin\theta_2 = \sin\theta_1 - \xi g r` (Medwin &
    Clay 1998, Eq. (3.3.21) in this library's variables, as derived over the
    travel-time table). The length of a circular arc is its radius times the
    angle it subtends, so

    .. math::

        s(r) = \frac{\theta_1 - \arcsin(\sin\theta_1 - \xi g r)}{\xi g},

    valid straight through the turning point, where the arcsin argument changes
    sign. Every quantity on the right is launch data, so the oracle shares
    nothing with the marcher's fourth state.
    """
    prof = (np.array([0.0, 2000.0]),
            np.array([_MEDWIN_C0, _MEDWIN_C0 + _MEDWIN_G * 2000.0]))
    th0 = np.radians(10.0)
    res = ray_trace(*prof, source_depth=0.0, launch_angles_deg=[10.0],
                    max_range=10_000.0, n_steps=20_001)
    xi = np.cos(th0) / _MEDWIN_C0
    r = res.ranges[0]
    exact = (th0 - np.arcsin(np.sin(th0) - xi * _MEDWIN_G * r)) / (xi * _MEDWIN_G)
    # The turn is at 5289.8 m, well inside the run, so both branches are hit.
    assert exact[-1] > 10_000.0
    assert np.abs(res.arc_lengths[0] - exact).max() < 1e-6


# A ray steep enough to reach the bottom before it turns (it would only turn at
# 4098 m, five times the column), launched deep enough to work both boundaries.
_BOUNCE_C0, _BOUNCE_G = 1500.0, 0.06  # c(z) = 1500 + 0.06 z
_BOUNCE_DEPTH, _BOUNCE_SOURCE_Z = 800.0, 200.0
_BOUNCE_LAUNCH_DEG, _BOUNCE_RANGE = 30.0, 9000.0
_BOUNCE_MAX_LEGS = 200


def _arc_walk(rmax: float) -> tuple[float, float]:
    """Exact travel time and depth at ``rmax`` for the bouncing ray below.

    In a constant gradient a ray is an arc of a circle of radius R = 1/(xi g)
    centred at the depth where c extrapolates to zero, z_c = -c1/g, since
    Snell's cos(phi) = xi c(z) and c(z) = g (z - z_c) together give
    z = z_c + R cos(phi). Range and time along the arc follow from
    dr = -R cos(phi) dphi and dt = ds/c = dphi/(g cos(phi)):

        r = R [sin(phi_a) - sin(phi_b)],
        t = |ln((1 + sin phi_a)/cos phi_a) - ln((1 + sin phi_b)/cos phi_b)| / g,

    the second being Medwin & Clay Eq. (3.3.20) written with the angle alone. A
    specular reflection is exactly ``phi -> -phi`` at the boundary depth, so the
    whole bouncing path is these two formulae restarted at each boundary. None
    of it comes from the solver.
    """
    phi = math.radians(_BOUNCE_LAUNCH_DEG)
    z_c = -_BOUNCE_C0 / _BOUNCE_G
    xi = math.cos(phi) / (_BOUNCE_C0 + _BOUNCE_G * _BOUNCE_SOURCE_Z)
    radius = 1.0 / (xi * _BOUNCE_G)

    def arc(p: float) -> float:
        return math.log((1.0 + math.sin(p)) / math.cos(p))

    r = t = 0.0
    for _ in range(_BOUNCE_MAX_LEGS):
        target = _BOUNCE_DEPTH if phi > 0.0 else 0.0
        phi_b = math.copysign(math.acos((target - z_c) / radius), phi)
        leg = radius * (math.sin(phi) - math.sin(phi_b))
        if r + leg >= rmax:  # the partial leg the range ends inside
            p = math.asin(math.sin(phi) - (rmax - r) / radius)
            return t + abs(arc(phi) - arc(p)) / _BOUNCE_G, z_c + radius * math.cos(p)
        r += leg
        t += abs(arc(phi) - arc(phi_b)) / _BOUNCE_G
        phi = -phi_b  # the reflection itself
    raise AssertionError("the arc walk did not reach the range")


def test_bouncing_ray_matches_the_exact_arc_through_every_reflection() -> None:
    """A multi-bounce path against the closed form, at a sampled range.

    The comparison is at the last sample, where the marching grid lands exactly,
    because interpolating between samples is second order and would swamp what
    is being measured.
    """
    prof = (np.array([0.0, _BOUNCE_DEPTH]),
            np.array([_BOUNCE_C0, _BOUNCE_C0 + _BOUNCE_G * _BOUNCE_DEPTH]))
    res = ray_trace(*prof, source_depth=_BOUNCE_SOURCE_Z,
                    launch_angles_deg=[_BOUNCE_LAUNCH_DEG],
                    max_range=_BOUNCE_RANGE, n_steps=4001)
    t_exact, z_exact = _arc_walk(_BOUNCE_RANGE)
    assert res.ranges[0, -1] == pytest.approx(_BOUNCE_RANGE)
    # The ray really does work both boundaries over this range.
    assert res.depths[0].max() > _BOUNCE_DEPTH - 1.0
    assert res.depths[0].min() < 1.0
    assert float(res.travel_times[0, -1]) == pytest.approx(t_exact, abs=1e-12)
    assert float(res.depths[0, -1]) == pytest.approx(z_exact, abs=1e-8)


def test_reflection_costs_no_accuracy_in_a_gradient() -> None:
    """A bounce must not degrade the path, which is what splitting the step buys.

    A ray launched upward from the surface and the same ray launched downward
    are one trajectory: the first reflects immediately and continues as the
    second, so the two must agree to the accuracy of the integration itself.
    They only do if the range step is split at the crossing. Folding the
    overshoot back after a whole step instead integrates through water that is
    not there and applies a mid-step reflection at the step's end, which leaves
    a first-order error per bounce: metres of depth at these step counts, and
    visible in any figure that draws a reflected ray.

    Isovelocity water cannot see this (no gradient, nothing to get wrong outside
    the column), which is why a gradient is essential here.
    """
    prof = (np.array([0.0, 2000.0]), np.array([1500.0, 1600.0]))
    for n_steps in (201, 2001, 20_001):
        kw = {"source_depth": 0.0, "max_range": 10_000.0, "n_steps": n_steps}
        up = ray_trace(*prof, launch_angles_deg=[-10.0], **kw)
        down = ray_trace(*prof, launch_angles_deg=[10.0], **kw)
        # Nanometres over a 2 km column: roundoff, not discretisation. Before
        # the step was split this gap was 3.6 m at the middle step count.
        assert np.abs(up.depths[0] - down.depths[0]).max() < 1e-9, n_steps
        assert np.abs(up.travel_times[0] - down.travel_times[0]).max() < 1e-12


def test_ray_travel_time_survives_reflections() -> None:
    # Reflections cost no time and do not change the speed, so a ray bouncing
    # between surface and bottom in isovelocity water still covers r/(c cos th0);
    # this is the folding path, which a post-hoc integral over the drawn (folded)
    # geometry would get wrong.
    res = ray_trace(*_ISO, source_depth=100.0, launch_angles_deg=[60.0],
                    max_range=5000.0, n_steps=3000)
    assert res.travel_times[0, -1] == pytest.approx(
        5000.0 / (1500.0 * np.cos(np.radians(60.0))), rel=1e-9)


def test_ray_reflection_counts_match_the_folded_geometry() -> None:
    r"""The cumulative bounce counts against the unfolded straight line, exactly.

    In isovelocity water the unfolded ray is the straight line
    :math:`u(r) = z_\mathrm{s} + r\tan\theta_0`, and folding it back into the
    column touches the bottom wherever :math:`u` crosses an odd multiple of
    the depth :math:`D` and the surface wherever it crosses a nonzero even
    one, so both counts are floors of closed forms and the *whole history*
    can be asserted rather than a final total. An upward launch is the mirror
    :math:`z \mapsto D - z` of a downward one, which swaps the two boundaries
    and nothing else, so the same two formulas serve it with the source
    mirrored and the labels exchanged; that exchange is asserted too, because
    the labels are the point. The counts are what make a boundary coefficient
    usable downstream: every bottom touch of one ray arrives at one grazing
    angle (Snell's invariant fixes the crossing angle per depth), so any
    :math:`R(\theta)` enters a path's amplitude only as :math:`R^n` with the
    :math:`n` recorded here, per boundary because the surface's coefficient
    is not the seabed's.
    """
    zs, theta = 30.0, 35.0
    depth = float(_ISO[0][-1])
    res = ray_trace(*_ISO, source_depth=zs, launch_angles_deg=[theta, -theta],
                    max_range=1000.0, n_steps=2001)
    climb = res.ranges[0] * np.tan(np.radians(theta))
    down_bottom = np.floor((zs + climb + depth) / (2.0 * depth)).astype(int)
    down_surface = np.floor((zs + climb) / (2.0 * depth)).astype(int)
    up_surface = np.floor((depth - zs + climb + depth) / (2.0 * depth)).astype(int)
    up_bottom = np.floor((depth - zs + climb) / (2.0 * depth)).astype(int)

    assert res.surface_reflections.shape == res.depths.shape
    assert res.bottom_reflections.shape == res.depths.shape
    np.testing.assert_array_equal(res.bottom_reflections[0], down_bottom)
    np.testing.assert_array_equal(res.surface_reflections[0], down_surface)
    np.testing.assert_array_equal(res.surface_reflections[1], up_surface)
    np.testing.assert_array_equal(res.bottom_reflections[1], up_bottom)
    # Zero at the source, only ever growing, and a total that means the ray
    # really worked both walls: 4 + 3 down, 4 + 3 up with the labels swapped.
    assert res.bottom_reflections[0, 0] == 0
    assert res.surface_reflections[0, 0] == 0
    assert res.bottom_reflections[0, -1] == 4
    assert res.surface_reflections[0, -1] == 3
    assert res.surface_reflections[1, -1] == 4
    assert res.bottom_reflections[1, -1] == 3
    assert np.all(np.diff(res.bottom_reflections, axis=1) >= 0)
    assert np.all(np.diff(res.surface_reflections, axis=1) >= 0)


def test_ray_travel_time_increases_monotonically_along_every_ray() -> None:
    # Time only ever runs forward, whichever way the ray is going: a Munk-like
    # profile with rays that turn, reflect and cross each other.
    z = np.linspace(0.0, 5000.0, 60)
    eta = 2.0 * (z - 1300.0) / 1300.0
    c = 1500.0 * (1.0 + 0.00737 * (eta - 1.0 + np.exp(-eta)))
    res = ray_trace(z, c, source_depth=1000.0,
                    launch_angles_deg=np.linspace(-14.0, 14.0, 15),
                    max_range=60_000.0, n_steps=4000)
    assert np.all(res.travel_times[:, 0] == 0.0)
    assert np.all(np.diff(res.travel_times, axis=1) > 0.0)


# --- Parabolic equation -----------------------------------------------------


def test_parabolic_equation_free_field_spherical_spreading() -> None:
    # In a deep homogeneous medium the PE reproduces PL = 20 log10(r) at the
    # source depth (spherical spreading) before any boundary interaction.
    D, zs = 20_000.0, 10_000.0
    res = parabolic_equation(50.0, np.array([0.0, D]), np.array([1500.0, 1500.0]),
                             source_depth=zs, max_range=5000.0, range_step=2.0,
                             n_depth_points=8192)
    assert isinstance(res, ParabolicEquationResult)
    zi = int(np.argmin(np.abs(res.depths - zs)))
    r = res.ranges
    mask = (r > 500.0) & (r < 3000.0)
    d = res.propagation_loss[zi][mask] - 20 * np.log10(r[mask])
    assert np.max(np.abs(d)) < 0.05


def test_parabolic_equation_tracks_normal_modes_trend() -> None:
    # Smoothed (incoherent) PE and normal-mode PL agree in trend within a few dB
    # for a range-independent, depth-VARYING waveguide (exercises the PE's
    # refraction term n²≠1, not just free propagation).
    D, f = 200.0, 50.0
    zs, zr = 30.0, 120.0
    prof = (np.array([0.0, D]), np.array([1500.0, 1530.0]))  # downward gradient
    r = np.linspace(500.0, 9000.0, 200)
    nm = normal_modes(f, *prof, source_depth=zs, receiver_depth=zr, ranges_m=r,
                      n_depth_points=1000)
    pe = parabolic_equation(f, *prof, source_depth=zs, max_range=9500.0,
                            range_step=5.0, n_depth_points=1024)
    zi = int(np.argmin(np.abs(pe.depths - zr)))
    pe_pl = np.interp(r, pe.ranges, pe.propagation_loss[zi])

    def smooth(pl: np.ndarray) -> np.ndarray:
        w = np.ones(21) / 21
        return -10 * np.log10(np.convolve(10 ** (-pl / 10), w, mode="same"))

    diff = smooth(pe_pl) - smooth(nm.propagation_loss)
    core = diff[20:-20]
    # constant offset from the Gaussian starter's angular spectrum; low scatter.
    assert core.std() < 2.0


# --- Validation & plots -----------------------------------------------------


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="depths"):
        normal_modes(20.0, [10.0, 20.0], [1500.0, 1500.0], source_depth=15.0,
                     receiver_depth=15.0)  # does not start at z = 0
    with pytest.raises(ValueError, match="bottom"):
        normal_modes(20.0, *_ISO, source_depth=50.0, receiver_depth=50.0, bottom="soft")
    with pytest.raises(ValueError, match="source_depth"):
        parabolic_equation(50.0, *_ISO, source_depth=150.0)
    with pytest.raises(ValueError, match="launch_angles_deg"):
        ray_trace(*_ISO, source_depth=50.0, launch_angles_deg=[90.0])  # not forward
    with pytest.raises(ValueError, match="range_step"):
        parabolic_equation(50.0, *_ISO, source_depth=50.0, max_range=1000.0,
                           range_step=2000.0)


def test_parabolic_equation_covers_max_range_when_step_not_divisor() -> None:
    # A range step that does not divide max_range must still cover it (the grid
    # extends to at least max_range, not silently short).
    res = parabolic_equation(50.0, *_ISO, source_depth=50.0, max_range=1000.0,
                             range_step=300.0, n_depth_points=64)
    assert res.ranges[-1] >= 1000.0


def test_ray_trace_steep_angle_reaches_max_range() -> None:
    # Range-marching guarantees every valid ray spans [0, max_range], even steep
    # ones (arc-length marching would fall short here).
    res = ray_trace(*_ISO, source_depth=100.0, launch_angles_deg=[60.0],
                    max_range=5000.0, n_steps=3000)
    assert res.ranges[0, -1] == pytest.approx(5000.0)
    # Depth stays within the water column at all times (reflections folded).
    assert np.all((res.depths >= 0.0) & (res.depths <= _ISO[0][-1]))


def test_plots_smoke() -> None:
    nm = normal_modes(20.0, *_ISO, source_depth=36.0, receiver_depth=46.0, n_depth_points=400)
    rt = ray_trace(*_ISO, source_depth=50.0, launch_angles_deg=[-5.0, 0.0, 5.0],
                   max_range=5000.0, n_steps=2000)
    pe = parabolic_equation(50.0, *_ISO, source_depth=50.0, max_range=5000.0,
                            range_step=10.0, n_depth_points=256)
    assert nm.plot() is not None
    assert rt.plot() is not None
    assert pe.plot() is not None


# Independent absolute PL anchors: converged image-source sum (|m| <= 4000)
# for the ideal pressure-release waveguide D = 100 m, c = 1500 m/s, f = 20 Hz,
# zs = 36 m, zr = 46 m, rho = 1000 kg/m3, PL = -20*lg(4*pi*|p|). Unlike the
# analytic modal-sum test, this oracle does not share the Eq. 5.14 prefactor
# with the implementation, so it pins the absolute calibration.
_IMAGE_SOURCE_PL = [
    (200.0, 39.145), (500.0, 42.297), (1000.0, 48.238),
    (3000.0, 52.137), (5000.0, 53.165),
]


def test_normal_modes_match_image_source_oracle() -> None:
    ranges = [r for r, _ in _IMAGE_SOURCE_PL]
    res = normal_modes(20.0, [0.0, 100.0], [1500.0, 1500.0],
                       source_depth=36.0, receiver_depth=46.0,
                       ranges_m=ranges, n_depth_points=3000)
    for (rng, pl_ref), pl_mod in zip(_IMAGE_SOURCE_PL, res.propagation_loss):
        assert pl_mod == pytest.approx(pl_ref, abs=0.02), rng


def test_parabolic_equation_long_range_absolute_anchor() -> None:
    # The paraxial PE converges to the exact field at long range: pin the
    # absolute calibration against the 5 km image-source anchor.
    res = parabolic_equation(20.0, [0.0, 100.0], [1500.0, 1500.0],
                             source_depth=36.0, max_range=5000.0,
                             range_step=5.0, n_depth_points=1024)
    iz = int(np.argmin(np.abs(res.depths - 46.0)))
    pl = float(res.propagation_loss[iz, -1])
    assert pl == pytest.approx(53.165, abs=0.05)
