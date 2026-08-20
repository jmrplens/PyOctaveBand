#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Tests for the sloping bottom of ``ray_trace`` and ``gaussian_beams``.

Every oracle here is built in the test from pure geometry and shares nothing
with the solver.

* **The folded wedge trace.** In an isovelocity wedge every ray is a chain of
  straight segments, each reflection off the slope turning the ray by twice
  the slope angle (Jensen Eq. 3.121 is one line of vector algebra here), so
  the whole trajectory, its travel time and its arc length have closed forms
  by elementary folding. The marcher is held to them to a hundredth of a
  millimetre over four kilometres and twenty-plus reflections.

* **One facet bounce.** With the surface far out of reach the exact field is
  the two-path sum, source plus its mirror image about the *sloping* plane,
  which pins the reflection geometry, the dynamic pair across a sloping
  bounce and the first rung of the fan ladder at the tenth-of-a-decibel
  level, free of every other effect.

* **A bottom out of reach.** A tilted bottom no beam can touch must leave
  the field the free-water two-path sum it is, to a hundredth of a decibel.
  This is the test that pins the fan ladder's tail budget: uncapped
  analytic tails once rebuilt arrivals bouncing on wedge geometry twenty
  marched extents beyond anything the caller described, at up to 11 dB.

* **The ideal wedge.** An isovelocity wedge of angle :math:`\beta = \pi/n`
  under a pressure-release surface with a rigid bottom has an exact solution
  by images: reflecting the source alternately in the two faces unfolds the
  boundaries into a closed fan of :math:`2\pi/\beta` image sources on the
  circle of the source's radius about the apex. The signs follow from
  requiring a character of the dihedral group that is :math:`-1` on the
  surface mirror and :math:`+1` on the bottom mirror, which exists exactly
  when :math:`n` is even: the image rotated by :math:`2k\beta` carries
  :math:`(-1)^k` and its mirrored partner :math:`(-1)^{k+1}`. The test
  verifies the construction against both boundary conditions numerically
  before using it (the surface sum cancels to :math:`10^{-9}` relative, the
  bottom's normal derivative to the finite-difference floor), so the oracle
  stands on its own feet.

  What the comparison measures is worth stating, because it is the model
  and not the arithmetic. A thin wedge is a retroreflector: most arrivals at
  any cell went up the slope, steepened by :math:`2\beta` per bounce, turned
  past the vertical and came back, and a marcher whose independent variable
  is range cannot carry any of them on its axes. The beams recover most of
  it anyway, because the receiver-image fan evaluates each beam's *analytic*
  tail at rotated image points, and a wrapped rung of the fan is exactly an
  up-and-back path; what remains unpaid is the quadrature near each
  arrival's turning point, where the neighbouring beams have been terminated.
  Measured on the 2.8-degree wedge at 150 Hz over ten cells spanning three
  ranges: 1.85 dB worst, 0.67 dB mean, and the numbers are the model's, not
  the discretisation's -- halving the range step changes them in the fourth
  decimal, doubling the fan density and opening the aperture from 80 to 85
  degrees in the third, and the width scan (60 to 200 m) moves the mean
  between 0.52 and 1.07 dB. The bounds below are those measurements with
  headroom, not a target the solver was tuned to.

* **Slope zero.** The wedge machinery run over a constant polyline must
  reproduce the level-bottom solver bit for bit, geometry and field alike:
  the crossing search, the reflection rotation and the fan ladder are all
  written so their level limits collapse to the old arithmetic exactly, and
  these tests are what holds that promise.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.propagation.numerical import (
    BeamFan,
    FluidSeabed,
    eigenrays,
    gaussian_beams,
    ray_trace,
)

_C = 1500.0


# --- The exact folded geometry, straight lines and mirrors only -------------


def _fold_wedge_ray(
    r_grid: np.ndarray,
    *,
    z0: float,
    theta0: float,
    depth0: float,
    slope_angle: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Exact isovelocity wedge trajectory sampled on ``r_grid``.

    The bottom is ``z = depth0 - tan(slope_angle) r``; the surface is z = 0.
    Straight segments between reflections; a surface bounce negates the
    inclination, a bottom bounce maps ``theta`` to ``-(theta + 2 beta)``.
    Returns depths, travel times and the two bounce counts up to the sample
    where the ray would pass the vertical (the grid is expected to end
    before that).
    """
    beta = slope_angle
    tanb = np.tan(beta)
    z = np.empty_like(r_grid)
    t = np.empty_like(r_grid)
    r0, z_seg, th, t_acc = 0.0, z0, theta0, 0.0
    i = n_surf = n_bot = 0
    for _ in range(10_000):
        if abs(th) >= np.pi / 2:
            msg = "the sampled grid must end before a reversal"
            raise AssertionError(msg)
        if th > 0:
            r_hit = (depth0 - z_seg + np.tan(th) * r0) / (np.tan(th) + tanb)
        elif th < 0:
            r_hit = r0 - z_seg / np.tan(th)
        else:
            r_hit = r0 + (depth0 - z_seg) / tanb
        while i < r_grid.size and r_grid[i] <= r_hit + 1e-9:
            z[i] = z_seg + np.tan(th) * (r_grid[i] - r0)
            t[i] = t_acc + (r_grid[i] - r0) / (np.cos(th) * _C)
            i += 1
        if i >= r_grid.size:
            return z, t, n_surf, n_bot
        t_acc += (r_hit - r0) / (np.cos(th) * _C)
        z_seg += np.tan(th) * (r_hit - r0)
        r0 = r_hit
        if th >= 0:
            th = -(th + 2.0 * beta)
            n_bot += 1
        else:
            th = -th
            n_surf += 1
    msg = "the fold did not close"
    raise AssertionError(msg)


def test_the_wedge_trace_is_the_exact_folded_geometry() -> None:
    """Depths, times, arc lengths and bounce counts against pure folding.

    Twenty-plus reflections off a five-degree slope, every one of them
    steepening the ray by ten degrees: the crossing search against the
    interpolated polyline, the rotation of the slowness pair and both
    odometers are all inside this one comparison. Straight rays make RK4
    exact, so the tolerance is the crossing bisection's, not the step's.
    """
    beta = np.radians(5.0)
    depth0, zs, theta0 = 1000.0, 400.0, np.radians(20.0)
    rmax, ns = 4000.0, 1601
    trace = ray_trace(
        [0.0, depth0],
        [_C, _C],
        source_depth=zs,
        launch_angles_deg=[np.degrees(theta0)],
        max_range=rmax,
        n_steps=ns,
        bathymetry=([0.0, 2.0 * rmax], [depth0, depth0 - 2.0 * rmax * np.tan(beta)]),
    )
    r_grid = np.linspace(0.0, rmax, ns)
    z, t, n_surf, n_bot = _fold_wedge_ray(
        r_grid, z0=zs, theta0=theta0, depth0=depth0, slope_angle=beta
    )
    assert np.abs(trace.depths[0] - z).max() < 1e-5
    assert np.abs(trace.travel_times[0] - t).max() < 1e-10
    # Isovelocity: the arc length is exactly c times the time.
    assert np.allclose(
        trace.arc_lengths[0], _C * trace.travel_times[0], rtol=1e-12, atol=1e-8
    )
    assert trace.surface_reflections[0, -1] == n_surf
    assert trace.bottom_reflections[0, -1] == n_bot
    assert n_bot >= 2, "the wedge must actually be exercised"
    # A bounce off the slope turns the ray by twice the slope: after the
    # first bottom bounce the climb rate is tan(theta0 + 2 beta), read off
    # the trace between the first two reflections.
    dz = np.diff(trace.depths[0])
    dr = r_grid[1] - r_grid[0]
    first_bot = np.argmax(trace.bottom_reflections[0] > 0)
    first_surf = np.argmax(trace.surface_reflections[0] > 0)
    mid = slice(first_bot + 1, first_surf - 1)
    assert np.allclose(-dz[mid] / dr, np.tan(theta0 + 2.0 * beta), rtol=1e-9)


def test_slope_zero_reproduces_the_level_trace_bit_for_bit() -> None:
    """A constant polyline through the new path equals the old path exactly.

    The profile refracts and kinks, so the comparison covers the crossing
    search, the sub-stepping and the reflection arithmetic in the regime
    where all of them do real work; the polyline path is written so its
    level limit collapses to the same floating-point operations.
    """
    depths = [0.0, 100.0, 300.0, 1000.0]
    speeds = [1500.0, 1510.0, 1488.0, 1512.0]
    kw = {
        "source_depth": 30.0,
        "launch_angles_deg": [-40.0, -10.0, 5.0, 35.0],
        "max_range": 6000.0,
        "n_steps": 1201,
    }
    level = ray_trace(depths, speeds, **kw)
    poly = ray_trace(depths, speeds, **kw, bathymetry=([0.0, 6000.0], [1000.0, 1000.0]))
    assert np.array_equal(level.depths, poly.depths)
    assert np.array_equal(level.travel_times, poly.travel_times)
    assert np.array_equal(level.arc_lengths, poly.arc_lengths)
    assert np.array_equal(level.surface_reflections, poly.surface_reflections)
    assert np.array_equal(level.bottom_reflections, poly.bottom_reflections)
    assert poly.bathymetry_ranges is not None


def test_a_ray_reflected_past_the_vertical_ends_in_nan() -> None:
    """A steep ray on a steep upslope terminates where it turns.

    The samples from the terminating bounce on are NaN (the ray no longer
    exists to a range march), the integer bounce counts hold their last
    value, and the samples before the stop are finite and inside the water.
    """
    beta = np.radians(10.0)
    trace = ray_trace(
        [0.0, 200.0],
        [_C, _C],
        source_depth=100.0,
        launch_angles_deg=[60.0],
        max_range=1500.0,
        n_steps=601,
        bathymetry=([0.0, 1080.0], [200.0, 200.0 - 1080.0 * np.tan(beta)]),
    )
    z = trace.depths[0]
    dead = np.isnan(z)
    assert dead.any(), "the ray must terminate for this test to bite"
    first = int(np.argmax(dead))
    assert dead[first:].all(), "once terminated, terminated for good"
    assert np.isfinite(z[:first]).all()
    assert np.isnan(trace.travel_times[0, first:]).all()
    assert np.isnan(trace.arc_lengths[0, first:]).all()
    assert trace.bottom_reflections[0, -1] == trace.bottom_reflections[0, first]
    assert trace.bottom_reflections[0, -1] >= 1


def test_a_vertex_reflects_specularly_off_one_of_its_facets() -> None:
    r"""A ray aimed exactly at a polyline vertex still reflects cleanly.

    A vertex has no tangent of its own and the faceted model needs none: the
    reflection uses the facet holding the located crossing, whichever side
    the bisection's last halving landed on. What is guaranteed, and asserted,
    is that the bounce is specular off one of the two adjoining facets (here
    :math:`-\theta_0` off the level one or :math:`-(\theta_0 + 2\beta)`
    off the sloping one), counted once, with the march finite throughout;
    a wrong tangent, a double bounce or a corrupted crossing would all show.
    """
    beta = np.radians(5.0)
    r_v, d0 = 1000.0, 500.0
    d_v = d0 - r_v * np.tan(beta)
    zs = 300.0
    theta0 = np.arctan((d_v - zs) / r_v)  # aims exactly at the vertex
    trace = ray_trace(
        [0.0, d0],
        [_C, _C],
        source_depth=zs,
        launch_angles_deg=[np.degrees(theta0)],
        max_range=2000.0,
        n_steps=2001,
        bathymetry=([0.0, r_v], [d0, d_v]),
    )
    r = trace.ranges[0]
    z = trace.depths[0]
    assert np.isfinite(z).all()
    assert trace.bottom_reflections[0, -1] == 1
    after = (r > r_v + 1.0) & (r < r_v + 300.0)
    slope_back = np.diff(z[after]) / np.diff(r[after])
    level_facet = np.allclose(slope_back, -np.tan(theta0), atol=1e-9)
    sloping_facet = np.allclose(slope_back, -np.tan(theta0 + 2.0 * beta), atol=1e-9)
    assert level_facet or sloping_facet


def test_a_ray_nearly_parallel_to_the_slope_survives_the_graze() -> None:
    r"""A grazing contact with the slope neither spins nor corrupts the march.

    The ray is launched within a tenth of a degree of the downslope facet's
    own angle, meets it near-tangentially where the Newton polish has no
    slope difference to divide by, and must come out specular all the same:
    finite, forward, exactly one bottom touch, and leaving at
    :math:`2\beta - \theta_0`, a tenth of a degree on the *other* side of
    the facet, so it separates from the deepening bottom as slowly as it
    closed in on it.
    """
    beta = np.radians(3.0)
    theta0 = beta + np.radians(0.1)
    trace = ray_trace(
        [0.0, 900.0],
        [_C, _C],
        source_depth=395.0,
        launch_angles_deg=[np.degrees(theta0)],
        max_range=8000.0,
        n_steps=1601,
        bathymetry=([0.0, 8000.0], [400.0, 400.0 + 8000.0 * np.tan(beta)]),
    )
    r = trace.ranges[0]
    z = trace.depths[0]
    assert np.isfinite(z).all()
    assert trace.bottom_reflections[0, -1] == 1
    assert trace.surface_reflections[0, -1] == 0
    # Specular off the facet: the outgoing inclination is 2 beta - theta0.
    after = trace.bottom_reflections[0] > 0
    tail = np.flatnonzero(after)[1:]
    slope_out = np.diff(z[tail]) / np.diff(r[tail])
    assert np.allclose(slope_out, np.tan(2.0 * beta - theta0), atol=1e-9)
    # The ray stays inside the water and pulls away from the bottom after
    # the graze: still descending, but slower than the bottom deepens.
    bottom = 400.0 + r * np.tan(beta)
    assert np.all(z <= bottom + 1e-6)
    gap = (bottom - z)[tail]
    assert np.all(np.diff(gap) > 0.0)


# --- The ideal wedge oracle --------------------------------------------------


def _wedge_image_fan(
    n: int,
    depth0: float,
    zs: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """The closed image fan of the ideal wedge, from pure geometry.

    Wedge angle beta = pi/n (n even), pressure-release surface, rigid bottom
    sloping up from ``depth0`` at r = 0 to the apex. Returns the image
    angles, their signs, the source radius, the source angle and the apex
    range.
    """
    beta = np.pi / n
    r_apex = depth0 / np.tan(beta)
    gam_s = np.arctan2(zs, r_apex)
    rho_s = np.hypot(r_apex, zs)
    ks = np.arange(n)
    angles = np.concatenate([2 * ks * beta + gam_s, 2 * ks * beta - gam_s])
    signs = np.concatenate([(-1.0) ** ks, (-1.0) ** (ks + 1)])
    return angles, signs, rho_s, gam_s, r_apex


def _wedge_exact_field(
    rr: float,
    zz: float,
    *,
    n: int,
    depth0: float,
    zs: float,
    k: float,
) -> complex:
    angles, signs, rho_s, _gam_s, r_apex = _wedge_image_fan(n, depth0, zs)
    rho_r = np.hypot(r_apex - rr, zz)
    gam_r = np.arctan2(zz, r_apex - rr)
    dist = np.sqrt(rho_s**2 + rho_r**2 - 2.0 * rho_s * rho_r * np.cos(angles - gam_r))
    return complex((signs * np.exp(1j * k * dist) / dist).sum())


def test_the_wedge_image_fan_satisfies_both_boundary_conditions() -> None:
    """The oracle stands before it judges: Dirichlet above, Neumann below.

    Each term of the fan solves the Helmholtz equation exactly and the only
    singularity inside the wedge is the true source, so the boundary
    conditions are the entire proof: the sum must vanish on the surface and
    its normal derivative on the sloping bottom. The Neumann residual is
    limited by the finite difference used to read it, not by the fan.
    """
    n, depth0, zs, f = 64, 200.0, 60.0, 150.0
    k = 2.0 * np.pi * f / _C
    beta = np.pi / n

    def field(rr: float, zz: float) -> complex:
        return _wedge_exact_field(rr, zz, n=n, depth0=depth0, zs=zs, k=k)

    for rr in (800.0, 2500.0):
        mid = abs(field(rr, 0.5 * (depth0 - rr * np.tan(beta))))
        assert abs(field(rr, 1e-9)) / mid < 1e-6
    m = -np.tan(beta)
    nrm = np.array([m, -1.0]) / np.hypot(m, 1.0)
    for rr in (800.0, 2500.0):
        zb = depth0 + m * rr
        eps = 1e-4
        p1 = field(rr + eps * nrm[0], zb + eps * nrm[1])
        p2 = field(rr + 2 * eps * nrm[0], zb + 2 * eps * nrm[1])
        assert abs(p2 - p1) / eps / (k * abs(p2)) < 1e-2


def test_a_single_facet_bounce_matches_the_two_path_field() -> None:
    """Direct plus the mirror about the sloping plane, to a tenth of a dB.

    The surface stands 2.8 km above the deepest ray of the fan, so the exact
    field at these cells is two paths only: the source and its image
    reflected in the sloping bottom plane (rigid, coefficient +1). This is
    the sloping counterpart of the flat two-ray test, and it isolates the
    facet reflection of the central rays, the dynamic pair across it and
    the first fold of the fan ladder from everything the full wedge mixes
    in. The bound is the measured 0.05 dB doubled.
    """
    beta = np.pi / 64
    m = np.tan(beta)
    depth0, f, zs = 3000.0, 150.0, 2800.0
    k = 2.0 * np.pi * f / _C
    nrm = np.array([m, 1.0]) / np.hypot(m, 1.0)
    src = np.array([0.0, zs])
    d = (m * src[0] + src[1] - depth0) / np.hypot(m, 1.0)
    img = src - 2.0 * d * nrm

    def exact(rr: float, zz: float) -> float:
        r1 = np.hypot(rr - src[0], zz - src[1])
        r2 = np.hypot(rr - img[0], zz - img[1])
        p = np.exp(1j * k * r1) / r1 + np.exp(1j * k * r2) / r2
        return float(-20.0 * np.log10(abs(p)))

    rr = np.array([2500.0])
    zz = np.array([2300.0, 2500.0, 2650.0, 2750.0])
    res = gaussian_beams(
        f,
        [0.0, depth0],
        [_C, _C],
        source_depth=zs,
        max_range=3000.0,
        ranges_m=rr,
        receiver_depths_m=zz,
        fan=BeamFan(max_angle_deg=25.0, beam_width=150.0),
        range_step=2.0,
        bottom="rigid",
        bathymetry=([0.0, 3000.0], [depth0, depth0 - 3000.0 * m]),
    )
    for i, z in enumerate(zz):
        assert abs(res.propagation_loss[i, 0] - exact(2500.0, z)) < 0.1


def test_the_ideal_wedge_matches_its_closed_image_fan() -> None:
    """Upslope cross-sections against the exact wedge field, quantified.

    Ten cells at 1, 2 and 3 km up a 2.8-degree wedge (local depths 151, 102
    and 53 m), every one of them dense multipath and most of it arrivals
    that turned past the vertical and came back. The comparison is against
    the *complete* field, return legs included, because the fan ladder's
    wrapped rungs recover them analytically even though no marched axis can
    carry one; what remains is the quadrature near each arrival's turning
    point, where the neighbouring beams have been terminated. The bounds are
    the module-docstring measurements (1.85 dB worst, 0.67 dB mean, both
    insensitive to step, fan density and aperture) with headroom that covers
    the width scan's spread, not a fitted target.
    """
    n, depth0, zs, f = 64, 200.0, 60.0, 150.0
    beta = np.pi / n
    k = 2.0 * np.pi * f / _C
    ranges = np.array([1000.0, 2000.0, 3000.0])
    depths = np.array([20.0, 40.0, 60.0, 90.0, 120.0])
    res = gaussian_beams(
        f,
        [0.0, depth0],
        [_C, _C],
        source_depth=zs,
        max_range=3200.0,
        ranges_m=ranges,
        receiver_depths_m=depths,
        fan=BeamFan(max_angle_deg=80.0, beam_width=100.0),
        range_step=2.0,
        bottom="rigid",
        bathymetry=([0.0, 3200.0], [depth0, depth0 - 3200.0 * np.tan(beta)]),
    )
    errors = []
    for j, rr in enumerate(ranges):
        local = depth0 - rr * np.tan(beta)
        for i, zz in enumerate(depths):
            if zz > 0.85 * local:
                continue  # keep the receivers off the sloping boundary
            exact = -20.0 * np.log10(
                abs(
                    _wedge_exact_field(
                        float(rr), float(zz), n=n, depth0=depth0, zs=zs, k=k
                    )
                )
            )
            errors.append(res.propagation_loss[i, j] - exact)
    errors = np.asarray(errors)
    assert errors.size == 10
    assert np.abs(errors).max() < 2.5
    assert np.abs(errors).mean() < 1.0


def test_slope_zero_beams_reproduce_the_level_field_bit_for_bit() -> None:
    """The wedge machinery at slope zero is the level solver, exactly.

    Same configuration as the ideal-waveguide oracle; the constant polyline
    routes through the sloping marcher, the fan ladder and the termination
    bookkeeping, and must come out bit for bit the level run, which itself
    stays pinned to the image sum at the base branch's own bound.
    """
    guide = {
        "water_depth": 1000.0,
        "frequency": 300.0,
        "source_depth": 300.0,
        "receiver_depth": 600.0,
    }
    k = 2.0 * np.pi * guide["frequency"] / _C
    r = np.array([2000.0])
    m = np.arange(-40_000, 40_001)[:, None]
    up = guide["receiver_depth"] - (
        2.0 * m * guide["water_depth"] + guide["source_depth"]
    )
    down = guide["receiver_depth"] - (
        2.0 * m * guide["water_depth"] - guide["source_depth"]
    )
    r_up, r_down = np.hypot(r[None, :], up), np.hypot(r[None, :], down)
    exact = -20.0 * np.log10(
        np.abs(
            (np.exp(1j * k * r_up) / r_up - np.exp(1j * k * r_down) / r_down).sum(
                axis=0
            )
        )
    )
    kw = {
        "source_depth": guide["source_depth"],
        "max_range": 2200.0,
        "ranges_m": r,
        "receiver_depths_m": np.array([guide["receiver_depth"]]),
        "fan": BeamFan(max_angle_deg=88.0, beam_width=100.0),
        "range_step": 2.0,
    }
    level = gaussian_beams(
        guide["frequency"], [0.0, guide["water_depth"]], [_C, _C], **kw
    )
    poly = gaussian_beams(
        guide["frequency"],
        [0.0, guide["water_depth"]],
        [_C, _C],
        **kw,
        bathymetry=([0.0, 2200.0], [guide["water_depth"], guide["water_depth"]]),
    )
    assert np.array_equal(level.pressure, poly.pressure)
    assert np.abs(poly.propagation_loss[0] - exact).max() < 5e-4
    assert poly.bathymetry_ranges is not None
    assert level.bathymetry_ranges is None


def test_slope_zero_impulse_matches_the_flat_impulse_through_a_gradient() -> None:
    """The facet reflection impulse at slope zero is the flat impulse.

    The isovelocity slope-zero test cannot see the dynamic pair's reflection
    impulse, because a zero gradient zeroes both formulas. This one bounces
    beams off the bottom of the n^2-linear guide, where the sound-speed
    gradient is real and the flat machinery is already pinned to the Airy
    modes, and holds the constant-polyline run to the level run. The two
    compute Jensen Eq. (3.123) through different arithmetic (the flat path
    from the folded image medium's precomputed ladder, the facet path from
    the incident slownesses with the facet slope at zero), so the comparison
    is a closed-form identity checked in floating point: measured 7.7e-13
    relative, bounded at 1e-10 with headroom.
    """
    c0 = 1550.0
    z = np.linspace(0.0, 200.0, 81)
    c = c0 / np.sqrt(1.0 + 2.4 * z / c0)
    kw = {
        "source_depth": 26.0,
        "max_range": 3000.0,
        "ranges_m": np.array([1000.0, 2000.0, 2900.0]),
        "receiver_depths_m": np.array([40.0, 100.0, 160.0]),
        "fan": BeamFan(max_angle_deg=60.0, beam_width=80.0),
        "range_step": 2.0,
    }
    level = gaussian_beams(200.0, z, c, **kw)
    poly = gaussian_beams(200.0, z, c, **kw, bathymetry=([0.0, 3000.0], [200.0, 200.0]))
    assert level.ray_depths.shape == poly.ray_depths.shape
    rel = np.abs(poly.pressure - level.pressure) / np.abs(level.pressure)
    assert rel.max() < 1e-10


def test_terminated_beams_leave_a_finite_field() -> None:
    """Beams the slope turns back retire cleanly from the sum.

    On a steep wedge most of an 80-degree fan reflects past the vertical
    before the far end. The per-beam histories say so with NaN from each
    stop on; the field itself must stay free of NaN everywhere, because a
    terminated beam's weight is zero rather than a frozen sample shining
    from a point the ray never passed.
    """
    beta = np.radians(8.0)
    res = gaussian_beams(
        200.0,
        [0.0, 300.0],
        [_C, _C],
        source_depth=100.0,
        max_range=1800.0,
        receiver_depths_m=np.array([50.0, 100.0, 150.0]),
        ranges_m=np.array([500.0, 1000.0, 1500.0]),
        fan=BeamFan(max_angle_deg=80.0, beam_width=50.0),
        range_step=2.0,
        bottom="rigid",
        bathymetry=([0.0, 1800.0], [300.0, 300.0 - 1800.0 * np.tan(beta)]),
    )
    assert np.isnan(res.ray_depths).any(), "beams must terminate here"
    assert not np.isnan(res.propagation_loss).any()
    assert not np.isnan(res.pressure).any()
    assert np.isfinite(res.propagation_loss).any()


def test_a_sloping_bottom_out_of_reach_leaves_the_field_alone() -> None:
    """A tilted bottom no beam can touch must not change the field.

    Source and receivers sit midwater with the bottom 2.4 km below and every
    beam's reach far short of it, so the exact field is the free-water
    two-path sum (direct plus surface image) whether that unreachable bottom
    tilts or not. The fan ladder's wrapped rungs are exactly what could
    break this: their images circle the local facet's apex, which here
    stands twenty extents beyond the march, and an uncapped analytic tail
    evaluated at them once measured up to 11 dB of pollution on this very
    configuration, against 0.001 dB with the tail budget in place. The
    bound is that measurement with an order of magnitude of headroom.
    """
    beta = np.pi / 64
    depth0, f, zs = 3000.0, 150.0, 300.0
    k = 2.0 * np.pi * f / _C
    zz = np.linspace(200.0, 400.0, 21)
    res = gaussian_beams(
        f,
        [0.0, depth0],
        [_C, _C],
        source_depth=zs,
        max_range=3000.0,
        ranges_m=np.array([2500.0]),
        receiver_depths_m=zz,
        fan=BeamFan(max_angle_deg=25.0, beam_width=150.0),
        range_step=2.0,
        bottom="rigid",
        bathymetry=([0.0, 3000.0], [depth0, depth0 - 3000.0 * np.tan(beta)]),
    )
    r1 = np.hypot(2500.0, zz - zs)
    r2 = np.hypot(2500.0, zz + zs)
    exact = -20.0 * np.log10(
        np.abs(np.exp(1j * k * r1) / r1 - np.exp(1j * k * r2) / r2)
    )
    assert np.abs(res.propagation_loss[:, 0] - exact).max() < 0.01


# --- Wiring: rejections, records, plots --------------------------------------


def test_eigenrays_declines_a_sloping_trace() -> None:
    """The arrival search prices flat-bottom geometry only, and says so."""
    trace = ray_trace(
        [0.0, 200.0],
        [_C, _C],
        source_depth=60.0,
        launch_angles_deg=np.linspace(-30.0, 30.0, 21),
        max_range=2000.0,
        n_steps=401,
        bathymetry=([0.0, 2000.0], [200.0, 150.0]),
    )
    with pytest.raises(ValueError, match="sloping bottom|Snell"):
        eigenrays(trace, receiver_range=1500.0, receiver_depth=80.0)


def test_the_seabed_pair_and_a_slope_are_rejected_together() -> None:
    """One grazing angle per beam is a level-bottom fact, and the solver
    refuses to pretend otherwise rather than quietly mis-charging bounces.
    """
    seabed = FluidSeabed(density=1800.0, sound_speed=1700.0)
    with pytest.raises(ValueError, match="lossy fluid seabed"):
        gaussian_beams(
            150.0,
            [0.0, 200.0],
            [_C, _C],
            source_depth=60.0,
            max_range=2000.0,
            bottom=seabed,
            bathymetry=([0.0, 2000.0], [200.0, 150.0]),
        )


def test_invalid_bathymetry_is_rejected() -> None:
    """Each way a polyline can be malformed has its own message."""
    good = {"source_depth": 60.0, "launch_angles_deg": [10.0], "max_range": 1000.0}
    with pytest.raises(ValueError, match="pair"):
        ray_trace([0.0, 200.0], [_C, _C], **good, bathymetry=([0.0, 1000.0],))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="equal length"):
        ray_trace([0.0, 200.0], [_C, _C], **good, bathymetry=([0.0, 1000.0], [200.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        ray_trace(
            [0.0, 200.0],
            [_C, _C],
            **good,
            bathymetry=([0.0, 500.0, 500.0], [200.0, 180.0, 160.0]),
        )
    with pytest.raises(ValueError, match="start at the source"):
        ray_trace(
            [0.0, 200.0], [_C, _C], **good, bathymetry=([100.0, 1000.0], [200.0, 150.0])
        )
    with pytest.raises(ValueError, match="strictly positive"):
        ray_trace(
            [0.0, 200.0], [_C, _C], **good, bathymetry=([0.0, 1000.0], [200.0, 0.0])
        )
    with pytest.raises(ValueError, match="profile is the medium"):
        ray_trace(
            [0.0, 200.0], [_C, _C], **good, bathymetry=([0.0, 1000.0], [200.0, 250.0])
        )
    with pytest.raises(ValueError, match="must be finite"):
        ray_trace(
            [0.0, 200.0], [_C, _C], **good, bathymetry=([0.0, np.nan], [200.0, 150.0])
        )
    # A source below the local water column at r = 0 is outside the medium.
    with pytest.raises(ValueError, match="water column"):
        ray_trace(
            [0.0, 200.0],
            [_C, _C],
            source_depth=180.0,
            launch_angles_deg=[10.0],
            max_range=1000.0,
            bathymetry=([0.0, 1000.0], [150.0, 100.0]),
        )


def test_bathymetry_plots_draw_the_seabed() -> None:
    """Both renderers overlay the polyline, continued level to the far end."""
    trace = ray_trace(
        [0.0, 300.0],
        [_C, _C],
        source_depth=100.0,
        launch_angles_deg=[-20.0, 0.0, 20.0, 60.0],
        max_range=2500.0,
        n_steps=501,
        bathymetry=([0.0, 2000.0], [300.0, 100.0]),
    )
    ax = trace.plot()
    assert ax is not None
    ax_es = trace.plot(language="es")
    assert any("Fondo marino" in t.get_text() for t in ax_es.get_legend().get_texts())
    beams = gaussian_beams(
        150.0,
        [0.0, 300.0],
        [_C, _C],
        source_depth=100.0,
        max_range=2500.0,
        ranges_m=np.array([500.0, 1500.0, 2400.0]),
        receiver_depths_m=np.array([50.0, 150.0]),
        fan=BeamFan(max_angle_deg=30.0, beam_width=50.0),
        range_step=5.0,
        bathymetry=([0.0, 2000.0], [300.0, 100.0]),
    )
    assert beams.plot() is not None
    import matplotlib.pyplot as plt

    plt.close("all")
