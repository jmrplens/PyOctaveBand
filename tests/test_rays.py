#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Tests for the shared range-marching ray core, ``phonometry._internal.rays``.

The oracles for the ray-tube spreading are geometry and nothing else. Two are
closed forms derived below; the third is the Jacobian of the ray family itself,
finite-differenced from traced ray paths, which is what :math:`q` is *defined*
to be (Jensen Eq. 3.64) and so is independent of every line the marcher runs;
the fourth integrates the smooth form of the equations with SciPy on a profile
the marcher can only approximate, which is what pins the impulsive treatment of
a piecewise-linear profile's second derivative; and the fifth is that same
Jacobian read off a family written in closed form, circular arcs inside a
gradient layer and straight lines above the homogeneous cap on top of it, which
carries no discretisation error of its own and so can be aimed at the crossings
the marcher resolves least well.

The closed forms. In a range-independent medium the pair obeys
:math:`dq/dr = p/\xi` and :math:`dp/dr = -\xi\,c''(z)\,q/c(z)`, so wherever the
profile has no curvature, :math:`p` keeps its launch value and :math:`q` is a
straight line in range. With the geometric initial conditions of Eq. (3.63),
:math:`q(0) = 0` and :math:`p(0) = 1/c(z_\mathrm{s})`, that is

.. math::

    q(r) = \frac{r\,p(0)}{\xi} = \frac{r}{c(z_\mathrm{s})\,\xi}
         = \frac{r}{\cos\theta_0},

for an isovelocity medium and, less obviously, for a constant gradient too:
the rays there are circular arcs of radius :math:`1/(|g|\xi)`, yet
:math:`c'' = 0` all the same, so the spreading in *range* is the same straight
line and the gradient never enters. Working it out from the arc confirms it.
Writing :math:`a = c(z_\mathrm{s})/g` and putting the arc through the source,

.. math::

    z(r) = z_c + \sqrt{a^2\sec^2\theta_0 - (r - a\tan\theta_0)^2},

so :math:`\partial z/\partial\theta_0 = a\sec^2\theta_0\,r/(z - z_c)` at fixed
range, and with :math:`z - z_c = c(z)/g` the Jacobian relation
:math:`q = c\,\xi\,\partial z/\partial\theta_0` returns :math:`r\sec\theta_0`
exactly, gradient and all cancelled.

That Jacobian relation is the third oracle and worth deriving once. The
Jacobian of Eq. (3.64) is the three-dimensional one, :math:`r\,q = J`, whose
azimuthal factor is the :math:`r`; what is left is
:math:`J_{2D} = \det\partial(r,z)/\partial(s,\theta_0)`, and expanding it with
:math:`\partial r/\partial s = c\xi` and :math:`\partial z/\partial s = c\zeta`
the two terms in :math:`\partial r/\partial\theta_0` cancel, leaving

.. math::

    q = c(z)\,\xi\,\left.\frac{\partial z}{\partial\theta_0}\right|_r .

In a homogeneous medium that is :math:`\cos\theta_0 \cdot r\sec^2\theta_0 = s`,
which is the arclength, as it must be. A reflection flips the sign of the
derivative of the *folded* depth without touching :math:`q` (Eq. 3.122), so the
comparison carries a factor :math:`(-1)^{\text{bounces}}`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from phonometry._internal.rays import DynamicRays, RayMarch, march_rays

#: Isovelocity 1000 m column.
_ISO = (np.array([0.0, 1000.0]), np.array([1500.0, 1500.0]))
#: Constant gradient, 0.02 s^-1.
_GRAD = (np.array([0.0, 1000.0]), np.array([1500.0, 1520.0]))
#: A thermocline: three segments, so two nodes where dc/dz jumps hard.
_KINK = (np.array([0.0, 100.0, 300.0, 1000.0]),
         np.array([1500.0, 1510.0, 1488.0, 1512.0]))
#: A downwind atmosphere sampled to 200 m, 0.05 s^-1, and homogeneous above it:
#: the shape the marcher meets with no boundary over it, where the last node of
#: the profile is an ordinary kink rather than the top of the medium.
_CAPPED_GRADIENT = 0.05
_CAPPED_HEIGHT = 200.0
_CAPPED_SOURCE = 5.0
_CAPPED = (np.array([0.0, _CAPPED_HEIGHT]),
           np.array([340.0, 340.0 + _CAPPED_GRADIENT * _CAPPED_HEIGHT]))


def _march(
    profile: tuple[np.ndarray, np.ndarray], source_depth: float,
    angles_deg: list[float], *, max_range: float, n_steps: int,
    water_depth: float | None = 1000.0,
    spreading: complex | np.ndarray | None = None,
    slope: complex | np.ndarray | None = None, dynamic: bool = True,
) -> tuple[RayMarch, np.ndarray]:
    """Trace a fan through ``profile``, optionally carrying (q, p).

    This is the ocean derivative of ``ray_trace``, with one addition the
    dynamic march asks for: sub-steps land exactly on the profile nodes, so the
    segment lookup resolves the tie by the direction of travel rather than by
    rounding. Without it one Runge-Kutta stage per crossing reads the gradient
    on the far side of the kink.

    ``water_depth=None`` takes the boundary above away, which is the atmospheric
    tracer's geometry: above the last node :func:`numpy.interp` holds the speed
    flat, so the derivative holds the gradient flat with it, exactly as
    :func:`atmospheric_ray_paths` does.
    """
    z_prof, c_prof = profile
    seg_grad = np.diff(c_prof) / np.diff(z_prof)
    top = float(z_prof[-1])
    open_above = water_depth is None
    c0 = float(np.interp(source_depth, z_prof, c_prof))
    th = np.radians(np.asarray(angles_deg, dtype=np.float64))
    xi = np.cos(th) / c0

    def deriv(
        z_arr: np.ndarray, zeta_arr: np.ndarray, /,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cc = np.interp(z_arr, z_prof, c_prof)
        seg = np.where(zeta_arr >= 0.0,
                       np.searchsorted(z_prof, z_arr, side="right") - 1,
                       np.searchsorted(z_prof, z_arr, side="left") - 1)
        grad = seg_grad[np.clip(seg, 0, seg_grad.size - 1)]
        if open_above:
            grad = np.where(z_arr > top, 0.0, grad)
        return (zeta_arr / xi, -grad / (cc**3 * xi), 1.0 / (xi * cc**2))

    dyn = None
    if dynamic:
        q0 = np.zeros(th.size) if spreading is None else np.full(th.size, spreading)
        p0 = np.full(th.size, 1.0 / c0) if slope is None else np.full(th.size, slope)
        dyn = DynamicRays(q0, p0, z_prof, c_prof)
    march = march_rays(deriv, xi=xi, z0=np.full(th.size, float(source_depth)),
                       zeta0=np.sin(th) / c0, range_step=max_range / (n_steps - 1),
                       n_steps=n_steps, lower=0.0, upper=water_depth, dynamic=dyn)
    return march, xi


def _jacobian_spreading(
    profile: tuple[np.ndarray, np.ndarray], source_depth: float, angle_deg: float,
    *, max_range: float, n_steps: int, water_depth: float | None = 1000.0,
    delta_deg: float = 1e-4,
) -> np.ndarray:
    """``q`` read off the ray family: ``c xi dz/dtheta0`` at fixed range.

    The two neighbouring rays are traced by the same marcher *with* the dynamic
    states switched on, not because they need a spreading but because that is
    what makes them split their steps at the profile nodes: comparing a
    node-split ray against two unsplit ones measures the difference between the
    two integrations, amplified by the 1/(2 delta) of the difference quotient,
    and swamps the quantity under test.
    """
    z_prof, c_prof = profile
    kw = {"max_range": max_range, "n_steps": n_steps,
          "water_depth": water_depth}
    mid, xi = _march(profile, source_depth, [angle_deg], **kw)
    hi, _ = _march(profile, source_depth, [angle_deg + delta_deg], **kw)
    lo, _ = _march(profile, source_depth, [angle_deg - delta_deg], **kw)
    dz_dtheta = ((hi.positions[0] - lo.positions[0])
                 / (2.0 * np.radians(delta_deg)))
    parity = (-1.0) ** np.cumsum(mid.reflections[0])
    return parity * np.interp(mid.positions[0], z_prof, c_prof) * xi[0] * dz_dtheta


# --- The closed forms -------------------------------------------------------


@pytest.mark.parametrize("profile", [_ISO, _GRAD], ids=["isovelocity", "gradient"])
@pytest.mark.parametrize("angle", [-7.0, -3.0, 0.0, 4.0, 8.0])
def test_spreading_is_range_over_cosine_where_the_profile_has_no_curvature(
    profile: tuple[np.ndarray, np.ndarray], angle: float,
) -> None:
    # Both profiles have c'' = 0 everywhere, so p never moves and q is exactly
    # r / cos(theta0) (see the module docstring). The angles are shallow enough
    # that no ray reaches a boundary, where p would jump.
    rmax, ns = 2000.0, 201
    march, _ = _march(profile, 500.0, [angle], max_range=rmax, n_steps=ns)
    assert march.reflections.sum() == 0
    assert march.spreadings is not None and march.spreading_slopes is not None
    exact = np.linspace(0.0, rmax, ns) / np.cos(np.radians(angle))
    # Machine precision, not a discretisation tolerance: q is a polynomial of
    # degree one in range here, which the step reproduces to the last bit.
    assert np.allclose(march.spreadings[0], exact, rtol=1e-13, atol=1e-9)
    assert np.ptp(march.spreading_slopes[0]) == 0.0


@pytest.mark.parametrize("angle", [-6.0, 5.0])
def test_the_closed_form_really_is_the_jacobian_of_the_ray_family(
    angle: float,
) -> None:
    # The gradient profile bends the rays into circular arcs, so agreeing with
    # r / cos(theta0) could be a coincidence of the integrator; differencing the
    # traced family shows it is the ray-tube spreading.
    rmax, ns = 2000.0, 201
    march, _ = _march(_GRAD, 500.0, [angle], max_range=rmax, n_steps=ns)
    oracle = _jacobian_spreading(_GRAD, 500.0, angle, max_range=rmax, n_steps=ns)
    assert march.spreadings is not None
    # The floor here is the difference quotient's own truncation, not the march.
    assert np.allclose(march.spreadings[0][1:], oracle[1:], rtol=2e-7)


# --- Boundaries -------------------------------------------------------------


@pytest.mark.parametrize("angle", [-25.0, -10.0, 12.0, 28.0])
def test_a_reflection_bends_the_spreading_by_the_boundary_gradient(
    angle: float,
) -> None:
    # A specular reflection off a boundary standing in a sound-speed gradient
    # leaves q alone and jumps p (Jensen Eqs. 3.122-3.123). Ignoring that jump
    # is not a rounding error: it is what the last assertion measures.
    rmax, ns = 6000.0, 601
    march, _ = _march(_GRAD, 500.0, [angle], max_range=rmax, n_steps=ns)
    oracle = _jacobian_spreading(_GRAD, 500.0, angle, max_range=rmax, n_steps=ns)
    assert march.reflections.sum() > 0
    assert march.spreadings is not None
    assert np.allclose(march.spreadings[0][1:], oracle[1:], rtol=1e-6)

    ignored = np.linspace(0.0, rmax, ns) / np.cos(np.radians(angle))
    off = np.max(np.abs(ignored - oracle)) / np.max(np.abs(oracle))
    assert off > 1e-2, "the boundary jump must matter for this test to bite"


@pytest.mark.parametrize("angle", [-20.0, 15.0])
def test_reflections_leave_the_spreading_alone_in_an_isovelocity_channel(
    angle: float,
) -> None:
    # The image-source construction says a bounce in an isovelocity channel is
    # just a longer straight ray, so q must stay r / cos(theta0) through it. The
    # boundary rule delivers that on its own: its strength is the boundary
    # gradient, which is zero here.
    rmax, ns = 6000.0, 601
    march, _ = _march(_ISO, 500.0, [angle], max_range=rmax, n_steps=ns)
    assert march.reflections.sum() >= 2
    assert march.spreadings is not None
    exact = np.linspace(0.0, rmax, ns) / np.cos(np.radians(angle))
    assert np.allclose(march.spreadings[0], exact, rtol=1e-13, atol=1e-9)


def test_the_two_boundaries_are_counted_apart_and_add_up_to_the_total() -> None:
    """Which boundary a bounce happened at, against the geometry it left behind.

    A field solver has to tell them apart, because a pressure-release sea
    surface inverts the pressure and a seabed need not, so the running parity
    that multiplies the amplitude is not the one the total gives. The oracle
    here is the trajectory itself: a range step that ends a bounce puts the ray
    on its way back from whichever boundary it touched, so the sign of the
    vertical slowness at the end of the step names the boundary it came off,
    downward after the surface and upward after the seabed. That reading is
    independent of the counter under test, and it holds because the geometry
    below rules out two bounces inside one step: 40 m of range at 20 degrees
    crosses 15 m of a 1000 m column.
    """
    rmax, ns = 24_000.0, 601
    march, _ = _march(_ISO, 500.0, [20.0], max_range=rmax, n_steps=ns)
    total, upper = march.reflections[0], march.upper_reflections[0]
    assert np.all(total <= 1) and np.all(upper <= total)
    assert upper.sum() > 0 and (total - upper).sum() > 0
    off_seabed = (total == 1) & (march.verticals[0] < 0.0)
    off_surface = (total == 1) & (march.verticals[0] > 0.0)
    assert np.array_equal(upper.astype(bool), off_seabed)
    assert np.array_equal((total - upper).astype(bool), off_surface)


def test_a_medium_open_above_never_reports_an_upper_reflection() -> None:
    # The atmosphere has no boundary over it, so every bounce is the ground's
    # and the upper counter stays empty however far the rays climb.
    march, _ = _march(_CAPPED, _CAPPED_SOURCE, [-6.0, 3.0, 30.0],
                      max_range=8000.0, n_steps=801, water_depth=None)
    assert march.reflections.sum() > 0
    assert march.upper_reflections.sum() == 0


# --- Profile nodes ----------------------------------------------------------


def _smooth_reference(
    c0: float, a: float, b: float, source_depth: float, angle_deg: float,
    max_range: float, n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the *smooth* equations for ``c(z) = c0 + a z + b z^2``.

    Nothing here is impulsive: ``c'' = 2b`` is an honest function, which is what
    makes this an oracle for the marcher's delta-train treatment of a
    piecewise-linear profile rather than a restatement of it.
    """
    th = np.radians(angle_deg)
    c_s = c0 + a * source_depth + b * source_depth**2
    xi = np.cos(th) / c_s

    def rhs(_r: float, y: np.ndarray) -> list[float]:
        z, zeta, q, p = y
        cc = c0 + a * z + b * z * z
        return [zeta / xi, -(a + 2 * b * z) / (cc**3 * xi), p / xi,
                -xi * (2 * b) * q / cc]

    ranges = np.linspace(0.0, max_range, n_steps)
    sol = solve_ivp(rhs, (0.0, max_range),
                    [source_depth, np.sin(th) / c_s, 0.0, 1.0 / c_s],
                    t_eval=ranges, rtol=1e-12, atol=1e-14, method="DOP853")
    return sol.y[0], sol.y[2]


def test_the_node_impulse_converges_to_a_resolved_second_derivative() -> None:
    # The whole point of the impulsive p-jump: a piecewise-linear profile has
    # c'' = 0 inside every segment and a delta at every node, so sampling c''
    # in a Runge-Kutta stage returns zero and the refraction drops out of the
    # amplitude without a word. Sampling a curved profile more and more finely
    # must drive the staircase towards the curve it approximates.
    c0, a, b = 1500.0, 0.02, -3e-5
    zs, angle, rmax, ns = 200.0, 5.0, 2000.0, 201
    _, q_ref = _smooth_reference(c0, a, b, zs, angle, rmax, ns)

    errors = []
    for n_nodes in (11, 161):
        z_prof = np.linspace(0.0, 1000.0, n_nodes)
        profile = (z_prof, c0 + a * z_prof + b * z_prof**2)
        march, _ = _march(profile, zs, [angle], max_range=rmax, n_steps=ns)
        assert march.spreadings is not None
        errors.append(abs(march.spreadings[0][-1] - q_ref[-1]) / abs(q_ref[-1]))

    assert errors[1] < 1e-4
    assert errors[1] < errors[0] / 20.0

    # And the failure mode it guards: dropping the impulse leaves p at its
    # launch value, i.e. the straight-line spreading of a medium with no
    # curvature at all, which is three orders of magnitude further out.
    ignored = rmax / np.cos(np.radians(angle))
    assert abs(ignored - q_ref[-1]) / abs(q_ref[-1]) > 300.0 * errors[1]


def test_the_spreading_is_continuous_across_a_node_and_its_slope_is_not() -> None:
    # Jensen Eq. (3.129): q is continuous across a weak interface and p is not.
    # This ray leaves at 6 degrees from 50 m and crosses both nodes of the
    # thermocline on its way down, reaching neither boundary, so every impulse
    # in it comes from a kink in the profile.
    rmax, ns = 4000.0, 4001
    march, _ = _march(_KINK, 50.0, [6.0], max_range=rmax, n_steps=ns)
    assert march.reflections.sum() == 0
    assert march.spreadings is not None and march.spreading_slopes is not None
    q, p = march.spreadings[0], march.spreading_slopes[0]

    treads = np.flatnonzero(np.diff(p) != 0.0)
    assert treads.size == 2, "one impulse per node crossed, and no more"
    # Between them p does not move at all: c'' is identically zero inside a
    # segment, so nothing but a node may touch it.
    assert np.ptp(p[treads[0] + 1:treads[1] + 1]) == 0.0
    # q meanwhile has no seam: a jump in it would stand out by three orders of
    # magnitude against the per-step increment, which is all the slope change
    # amounts to.
    increments = np.abs(np.diff(q))
    assert np.max(increments) < 3.0 * float(np.median(increments))
    assert np.all(np.diff(q) > 0.0)


def _capped_layer_paths(
    angle_deg: float, ranges: np.ndarray,
) -> tuple[np.ndarray, float]:
    """``z(r)`` in closed form for a gradient layer under a homogeneous cap.

    In the layer the ray is the usual circular arc, and the angle from the
    horizontal is what writes it down without solving anything: differentiating
    Snell's law :math:`\\cos\\theta = c\\,\\xi` along the ray gives
    :math:`d\\theta/dr = -\\xi g/\\cos\\theta`, hence
    :math:`\\sin\\theta(r) = \\sin\\theta_0 - \\xi g r`, and Snell's law itself
    then gives :math:`z - z_\\mathrm{s} = (\\cos\\theta - \\cos\\theta_0)/
    (\\xi g)`. Above the cap the sound speed is constant and the ray is the
    straight line it arrives as. No integrator and no ray-tube quantity anywhere
    in it, which is what makes the family below an oracle rather than a
    restatement.
    """
    g, height = _CAPPED_GRADIENT, _CAPPED_HEIGHT
    c_ground = float(_CAPPED[1][0])
    c_s = c_ground + g * _CAPPED_SOURCE
    th0 = np.radians(angle_deg)
    xi = float(np.cos(th0) / c_s)
    cos_top = (c_ground + g * height) * xi
    assert cos_top <= 1.0, "the ray must clear the cap for this form to hold"
    th_top = np.arccos(cos_top)
    r_top = (np.sin(th0) - np.sin(th_top)) / (xi * g)
    sin_th = np.sin(th0) - xi * g * np.minimum(ranges, r_top)
    in_layer = _CAPPED_SOURCE + (np.sqrt(1.0 - sin_th**2) - np.cos(th0)) / (xi * g)
    above = height + (ranges - r_top) * np.tan(th_top)
    return np.where(ranges <= r_top, in_layer, above), xi


def _capped_layer_spreading(
    angle_deg: float, ranges: np.ndarray, delta_deg: float = 1e-3,
) -> np.ndarray:
    """``q = c(z) xi dz/dtheta0`` at fixed range, off the closed-form family.

    The paths carry no discretisation error, so the difference quotient has only
    its own truncation in it and the answer is the same to six figures whether
    the neighbours are a thousandth of a degree away or a hundredth.
    """
    z_mid, xi = _capped_layer_paths(angle_deg, ranges)
    hi, _ = _capped_layer_paths(angle_deg + delta_deg, ranges)
    lo, _ = _capped_layer_paths(angle_deg - delta_deg, ranges)
    c_at = np.minimum(_CAPPED[1][0] + _CAPPED_GRADIENT * z_mid, _CAPPED[1][1])
    return c_at * xi * (hi - lo) / (2.0 * np.radians(delta_deg))


def test_the_last_node_of_an_open_medium_kinks_the_spreading_like_any_other(
) -> None:
    # A medium with no boundary above ends its profile in mid air: the caller
    # reads the sound speed off a clamped interpolation, so past the last node
    # the medium is homogeneous and that node is a discontinuity of dc/dz the
    # rays climb through. It is not an end of the medium and there is no image
    # medium at it, so it takes the plain gradient jump of an interface. This
    # ray leaves the profile at 25 degrees and meets exactly one, which is what
    # lets the closed form above stand in for the whole of q.
    rmax = 2000.0
    errors = []
    for ns in (201, 1601):
        ranges = np.linspace(0.0, rmax, ns)
        march, _ = _march(_CAPPED, _CAPPED_SOURCE, [25.0], max_range=rmax,
                          n_steps=ns, water_depth=None)
        oracle = _capped_layer_spreading(25.0, ranges)
        assert march.positions.max() > _CAPPED_HEIGHT, "the ray must leave it"
        assert march.reflections.sum() == 0
        assert march.spreadings is not None
        errors.append(float(np.max(np.abs(march.spreadings[0][1:] / oracle[1:]
                                          - 1.0))))
    # Refining the march drives it onto the family, which is the signature of a
    # crossing resolved at the right range rather than one merely accounted for.
    assert errors[0] < 1e-4
    assert errors[1] < errors[0] / 4.0

    # And the failure mode, which is the reason for the test: the layer below
    # the cap has no curvature and the half space above it none either, so with
    # that one impulse missing q is the straight line r / cos(theta0) of a
    # medium that never refracted at all, eleven per cent adrift and silent.
    ignored = rmax / np.cos(np.radians(25.0))
    exact = _capped_layer_spreading(25.0, np.array([rmax]))[0]
    assert abs(ignored - exact) / exact > 1e-1


def _capped_layer_error(angle_deg: float, n_steps: int, max_range: float) -> float:
    """How far the marched ``q`` sits from the closed-form family, relatively."""
    ranges = np.linspace(0.0, max_range, n_steps)
    march, _ = _march(_CAPPED, _CAPPED_SOURCE, [angle_deg], max_range=max_range,
                      n_steps=n_steps, water_depth=None)
    oracle = _capped_layer_spreading(angle_deg, ranges)
    assert march.positions.max() > _CAPPED_HEIGHT, "the ray must cross the cap"
    assert march.spreadings is not None
    return float(np.max(np.abs(march.spreadings[0] - oracle))
                 / np.max(np.abs(oracle)))


def test_a_near_tangential_crossing_costs_what_a_square_one_does_not() -> None:
    # The one stage of first-order error a crossing still costs lands in zeta,
    # and the impulse divides by |zeta| (module docstring), so the same range
    # step buys wildly different accuracy depending on how squarely the ray
    # meets the kink. Both rays here clear the same cap, the first by a
    # hundredth of a degree and the second at 25 degrees, and the first pays
    # more than two orders of magnitude for it. Pinning that here is what keeps
    # the numbers in the module docstring honest, and what would catch a
    # sub-step that stopped landing on the node at all.
    rmax, coarse, fine = 4000.0, 401, 1601
    grazing = np.degrees(np.arccos(
        (_CAPPED[1][0] + _CAPPED_GRADIENT * _CAPPED_SOURCE) / _CAPPED[1][1]))
    tangential = _capped_layer_error(grazing + 0.01, coarse, rmax)
    square = _capped_layer_error(25.0, coarse, rmax)
    assert square < 2e-4
    assert tangential > 100.0 * square
    # An accuracy limit and not a wrong answer: the range step buys it back.
    assert _capped_layer_error(grazing + 0.01, fine, rmax) < tangential / 4.0


# --- Complex initial conditions ---------------------------------------------


def test_complex_initial_conditions_ride_the_same_linear_march() -> None:
    # The pair is linear in (q, p) with real coefficients, impulses included, so
    # a Gaussian beam's complex start (Jensen Eq. 3.91, p(0) = 1 and
    # q(0) = i omega W^2 / 2) is the same superposition of two real marches. If
    # it were not, carrying beams in this marcher would need a second solver.
    kw = {"max_range": 4000.0, "n_steps": 401}
    from_slope, _ = _march(_KINK, 200.0, [6.0], spreading=0.0, slope=1.0, **kw)
    from_spread, _ = _march(_KINK, 200.0, [6.0], spreading=1.0, slope=0.0, **kw)
    q0 = 0.5j * 2.0 * np.pi * 200.0 * 60.0**2 / 1490.0
    beam, _ = _march(_KINK, 200.0, [6.0], spreading=q0, slope=1.0, **kw)

    assert beam.spreadings is not None and beam.spreadings.dtype == np.complex128
    assert from_slope.spreadings is not None and from_spread.spreadings is not None
    superposed = from_slope.spreadings[0] + q0 * from_spread.spreadings[0]
    assert np.allclose(beam.spreadings[0], superposed, rtol=1e-12, atol=1e-9)
    # The trajectory does not care which of them was launched.
    assert np.array_equal(beam.positions, from_slope.positions)

    # And here is why a beam has no caustic, which is the whole reason for the
    # complex start. The real and imaginary parts are two independent real
    # solutions, so their Wronskian is conserved (an impulse adds N q to both
    # slopes and cancels out of it); q could only vanish if both parts vanished
    # at once, which would make the Wronskian nought. Im[p conj(q)] is that
    # invariant, and holding it below zero is what keeps the beam half-width of
    # Jensen Eq. (3.89) real at every range.
    assert beam.spreading_slopes is not None
    invariant = np.imag(beam.spreading_slopes[0] * np.conj(beam.spreadings[0]))
    assert np.allclose(invariant, -np.imag(q0), rtol=1e-12)
    assert np.all(np.abs(beam.spreadings[0]) > 0.0)
    assert np.all(np.imag(beam.spreading_slopes[0] / beam.spreadings[0]) < 0.0)


def test_a_march_without_dynamics_reports_no_spreading() -> None:
    march, _ = _march(_KINK, 200.0, [3.0], max_range=1000.0, n_steps=51,
                      dynamic=False)
    assert march.spreadings is None
    assert march.spreading_slopes is None


# --- The atmospheric caller -------------------------------------------------

#: Heights and travel times of :func:`atmospheric_ray_paths`, every fiftieth
#: range sample, recorded before the marcher learnt to carry a ray-tube
#: spreading. The dynamic states are opt-in precisely so this stays put: the
#: atmosphere has no use for an amplitude, and asking for one is what makes the
#: march split its steps at the profile nodes. The run was verified equal
#: bit for bit at the time; the tolerance here is for the last-place libm
#: differences between machines that the figure gate already allows for.
_ATMOSPHERIC_HEIGHTS = np.array([
    [20.0, 75.37324473692166, 141.82670785541063, 196.3062831902225,
     245.5334411479058],
    [20.0, 48.98105414984245, 100.9594954419497, 136.19175149790775,
     159.48089515475627],
    [20.0, 46.16660103940749, 33.340746708386284, 40.777097915517544,
     115.13013126048429],
    [20.0, 163.50943267713717, 296.6995242873955, 429.5415876471756,
     562.3836510069547],
])
_ATMOSPHERIC_TIMES = np.array([
    [0.0, 2.190434863741247, 4.362559980742843, 6.529099115036047,
     8.693532228182598],
    [0.0, 2.1763756856459073, 4.333177746652069, 6.484261180638386,
     8.632625345886305],
    [0.0, 2.1699828083316524, 4.33778562968088, 6.517628075077829,
     8.678149616330794],
    [0.0, 2.2024794486058665, 4.39406281050967, 6.585306571214145,
     8.776550331918637],
])


def test_atmospheric_ray_paths_are_untouched_by_the_dynamic_states() -> None:
    from phonometry.environment.propagation.refraction import (
        atmospheric_ray_paths,
        log_linear_sound_speed_profile,
    )

    profile = log_linear_sound_speed_profile(1.0, max_height=200.0, n_points=64)
    res = atmospheric_ray_paths(profile, source_height=20.0,
                                launch_angles_deg=[-8.0, -2.0, 4.0, 12.0],
                                max_range=3000.0, n_steps=201)
    assert np.allclose(res.heights[:, ::50], _ATMOSPHERIC_HEIGHTS, rtol=1e-12)
    assert np.allclose(res.travel_times[:, ::50], _ATMOSPHERIC_TIMES, rtol=1e-12)
    assert np.array_equal(res.ground_reflections, [1, 1, 1, 0])
    assert np.array_equal(res.turning_points, [0, 0, 1, 0])
