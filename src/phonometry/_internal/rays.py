#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Range-marching ray integration with specular boundaries (private).

In a range-independent medium the horizontal slowness
:math:`\xi = \cos\theta_0 / c(z_0)` is invariant along a ray, so with the
vertical slowness :math:`\zeta = \sin\theta / c` the trajectory marches in
range as

.. math::

    \frac{dz}{dr} = \frac{\zeta}{\xi}, \qquad
    \frac{d\zeta}{dr} = -\frac{dc/dz}{c^3 \xi}, \qquad
    \frac{dt}{dr} = \frac{1}{\xi c^2}, \qquad
    \frac{ds}{dr} = \frac{1}{\xi c},

the travel time being a third state of the very step that places the ray rather
than a quadrature run over the finished path, and the arc length :math:`s` a
fourth, for the same reason and at the same order: both odometers share the
sound speed the geometric stages already evaluate, so carrying them costs a
division per stage and cannot drift from the trajectory actually returned. The
arc length is what volume absorption multiplies on. Jensen, Kuperman, Porter &
Schmidt, *Computational Ocean Acoustics* (2nd ed., Springer 2011), Sect. 3.6.2
carries a loss :math:`\alpha` in nepers/m into the eikonal by perturbation and
lands on the factor :math:`e^{-\int_0^s \alpha(s')\,ds'}` of Eq. (3.116), an
integral along the *ray path*, not along the range axis: the two agree only for
a horizontal ray, and a steep or many-times-reflected path is longer than the
range it spans by exactly the factor the marcher integrates here. A reflection
is instantaneous and adds no path, so the accumulated arc length stays
continuous across it, like the time.

Those are the equations behind
:func:`phonometry.underwater.propagation.numerical.ray_trace`, and the same ones
behind the atmospheric
:func:`phonometry.environment.propagation.refraction.atmospheric_ray_paths`;
only the profile and the reflecting boundaries differ, the ocean being a slab
between the sea surface and the bottom and the atmosphere a half space bounded
by the ground alone. This module owns the marcher, and takes both as arguments:
the caller supplies its own derivative (hence its own profile) and its own
boundaries.

``z`` is whichever coordinate the caller measures across the medium, positive
depth downward for the ocean and positive height upward for the atmosphere; the
marcher only needs the boundaries to be given in the same one.

Dynamic ray tracing
-------------------

A ray path on its own carries no amplitude: the amplitude lives in how fast the
tube between neighbouring rays opens, which is a second pair of states obeying
the *dynamic* ray equations (Jensen Eq. 3.58)

.. math::

    \frac{dq}{ds} = c\,p, \qquad \frac{dp}{ds} = -\frac{c_{nn}}{c^2}\,q,

with :math:`q` the ray-tube spreading, tied to the Jacobian of the ray
coordinates by :math:`r\,q(s) = J(s)` (Eq. 3.64), and :math:`c_{nn}` the
curvature of the sound speed across the ray (Eq. 3.62). In a range-independent
medium the two cross terms of Eq. (3.62) vanish and it collapses to
:math:`c_{nn} = c^2 \xi^2 c''(z)`; dividing by :math:`dr/ds = c\,\xi` puts the
pair in the marcher's own variable next to the three above,

.. math::

    \frac{dq}{dr} = \frac{p}{\xi}, \qquad
    \frac{dp}{dr} = -\xi\,\frac{c''(z)}{c(z)}\,q .

Both callers interpolate their profile piecewise linearly, so :math:`c''` is
zero throughout every segment and a Dirac delta at every node. That is not a
detail to gloss over: run a Runge-Kutta step over such a profile and every
stage samples :math:`c'' = 0`, so ``p`` comes out constant and ``q`` linear, and
the field one would build from it is free-space spreading with the refraction
quietly deleted from the amplitude. The deltas have to be integrated as what
they are, an impulsive change in ``p`` each time the ray crosses a node
(Jensen Sect. 3.6.4). Between nodes the pair is then exact rather than
fourth-order accurate: ``p`` is constant and ``q`` is a straight line in range,
which is what a Runge-Kutta step would return anyway, to the last bit.

The impulse comes from the weak-interface formulas of Eqs. (3.129)-(3.130),
:math:`p^T = p^I + q^I N`, :math:`q^T = q^I` with
:math:`N = -M(2[c_\mathrm{n}] - M[c_\mathrm{s}])/c^2` and
:math:`M = \beta/\alpha`. With a flat interface and a range-independent
profile, the ray tangent and normal of Eqs. (3.25)-(3.26) give
:math:`c_\mathrm{n} = c\,c_z\,\xi`, :math:`c_\mathrm{s} = c\,c_z\,\zeta` and
:math:`M = \xi/\zeta`, and the whole of Eq. (3.130) collapses to one scalar:

.. math::

    q \mapsto q, \qquad
    p \mapsto p - \xi^2\,\frac{[dc/dz]}{|\zeta|\,c(z_k)}\,q ,

with :math:`[dc/dz]` the jump in the gradient across the node, the segment
below minus the segment above. Taking the jump in that fixed sense and dividing
by :math:`|\zeta|` rather than :math:`\zeta` makes the rule reciprocal, the same
impulse for a ray crossing upward as downward, which it must be. Integrating
the delta directly, :math:`\int c''\,dr = (\xi/|\zeta|)\,[dc/dz]` across the
crossing, reproduces it independently.

A specular reflection is the same formula with a curvature term, Eqs.
(3.122)-(3.123), and for the flat surface and flat bottom here the curvature
vanishes and it reduces to :math:`p \mapsto p + 2\xi^2 (dc/dz)\,q/(\zeta\,c)`
with :math:`\zeta` the *signed* incident vertical slowness. That is the rule
above with an effective jump of :math:`+2\,dc/dz` at the lower boundary and
:math:`-2\,dc/dz` at the upper one, which is exactly what the mirrored profile
of the image medium presents to a ray crossing the boundary, so the marcher
applies one rule at all three kinds of level. In particular an isovelocity
layer against a boundary leaves the pair untouched, as the image-source
construction requires.

Which of the two a level is depends on the medium and not on where the profile
happens to stop. The ocean's profile ends on its two boundaries, so its end
nodes are reflections; the atmosphere's ends in mid air with nothing above it
but the homogeneous half space the clamped interpolation of the sound speed
implies, so its last node is an interface, and a ray that climbs out through it
takes the plain gradient jump on the way.

Carrying the pair is opt-in (``dynamic=`` on :func:`march_rays`), for two
reasons worth stating rather than leaving to be discovered. It costs the
atmospheric tracer, which has no use for an amplitude, nothing at all. And
landing sub-steps on the profile nodes, which the impulses require, subdivides
steps that were previously taken whole, so a run with the pair returns a
slightly different (better resolved) trajectory than one without; a solver that
wants an amplitude has to trace geometry and amplitude together, in one call,
so that the amplitude is the Jacobian of the rays it is handed alongside.

One thing the splitting does not buy back, and it is worth putting numbers to.
The last Runge-Kutta stage of the sub-step that *arrives* at a node is evaluated
at the node itself, on one side or the other of the kink, and whichever side the
caller resolves the tie to, that stage is integrated with a gradient that does
not hold over all of the sub-step it closes: one stage of first-order error per
crossing, inside an otherwise fourth-order step. It is much the smaller half of
the problem, the alternative being a whole step integrated with the gradient of
one segment while the ray spends part of it in the next. But it lands in
``zeta`` at the crossing, and by an absolute amount that barely depends on how
steeply the ray meets the kink, while the impulse divides by ``|zeta|``: what
reaches the amplitude is therefore that fixed error measured against a vertical
slowness which vanishes as the crossing turns tangential.

The size of it, on a thermocline kinking from +0.10 to -0.11 s^-1 over a 4 km
path, with ``q`` measured against the ray family of an independently integrated
fan. A ray meeting that node a hundredth of a degree past grazing, ``|zeta|`` =
4.3e-6 s/m, is out by 2.5e-2 at a 10 m range step, 7.3e-3 at 2.5 m and 2.9e-4 at
0.6 m; one meeting the same node at 20 degrees, ``|zeta|`` fifty times larger, is
out by 7.6e-5 at that same 10 m step. The two absolute errors in
``zeta`` differ by less than a factor of two, and in the near-tangential case
the relative error in ``q`` tracks the relative error in ``zeta`` to within a few
per cent of itself, that one impulse being large enough to swamp everything else
along the path. So it converges, at first order and with a scatter set by where
in the step the crossing happens to fall, which makes it an accuracy limit
rather than a wrong answer; but it is largest exactly where a beam solver most
needs ``q``, at the edge of a duct or on the approach to a caustic, and the
range step is the only handle a caller has on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    #: The dynamic pair is linear in (q, p) with real coefficients, so the same
    #: code integrates the real initial conditions of geometric ray tracing
    #: (Jensen Eq. 3.63) and the complex ones of a Gaussian beam (Eq. 3.91); the
    #: history is allocated in whatever dtype the initial values come in.
    type DynamicArray = NDArray[np.float64] | NDArray[np.complex128]

#: Bisections used to locate a boundary crossing inside a ray step. The bracket
#: is the unit interval, so 40 halvings pin the crossing far below the accuracy
#: of the step it subdivides, at the cost of arithmetic only.
_BOUNDARY_BISECTIONS = 40
#: Newton polishes of the interpolated crossing, against real Runge-Kutta steps.
#: Two is convergence to the step's own residual from a start already close.
_BOUNDARY_NEWTON_STEPS = 2
#: Below this |dz/dr| a ray meets the boundary tangentially and Newton has no
#: slope to divide by; the interpolated crossing stands.
_GRAZING_SLOPE = 1e-12
#: Cap on reflections resolved within one range step. A step spans dr*tan(theta)
#: across the medium, so reaching this many crossings needs a ray within a hair
#: of vertical in a layer thinner than one step; the cap only stops a
#: pathological input from spinning.
_MAX_CROSSINGS_PER_STEP = 16
#: Floor on the |zeta| the impulsive change in ``p`` divides by. A ray meeting a
#: gradient discontinuity this close to tangentially runs along the kink instead
#: of crossing it, and its tube really does collapse, so the floor is there to
#: keep the arithmetic finite rather than to hide the physics: it sits far below
#: the |zeta| = sin(theta)/c of any launch angle a caller can pass.
_GRAZING_VERTICAL = 1e-12


class RayDerivative(Protocol):
    """The right-hand side of the ray equations, vectorised over rays.

    Returns ``(dz/dr, dzeta/dr, dt/dr, ds/dr)`` at the given state. The
    implementation closes over its own sound-speed profile and over
    :math:`\\xi`.

    ``zeta`` is passed as well as ``z`` because a caller that asks for dynamic
    ray tracing gets sub-steps landing *exactly* on its profile nodes, where a
    naive lookup would resolve the segment by rounding rather than by the
    direction the ray travels; the sign of ``zeta`` is what settles it, and one
    stage evaluated on the wrong side of a node is a first-order error inside a
    fourth-order step.
    """

    def __call__(
        self, z: NDArray[np.float64], zeta: NDArray[np.float64], /
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64]]: ...


class DynamicRays(NamedTuple):
    """The dynamic ray equations to integrate alongside the trajectory.

    :ivar spreading: Per-ray :math:`q(0)`. With :math:`q(0) = 0` and
        :math:`p(0) = 1/c(0)` (Jensen Eq. 3.63) the pair is the geometric
        spreading, :math:`r\\,q = J`; with :math:`p(0) = 1` and
        :math:`q(0) = i\\omega W_0^2/2` (Eq. 3.91) it is a Gaussian beam of
        initial half-width :math:`W_0` and flat wavefront.
    :ivar slope: Per-ray :math:`p(0)`, in the same dtype as ``spreading``.
    :ivar profile_depths: Node coordinates of the sound-speed profile,
        ascending, in the marcher's own ``z``. This must be the profile the
        ``deriv`` closes over, in full: the impulses live at its nodes, and
        beyond its ends the medium is taken to be homogeneous, which is what
        the :func:`numpy.interp` both callers read their sound speed with
        already says. So a node that ends the profile away from a reflecting
        boundary is an interface like any other, and the ``deriv`` has to hold
        its gradient flat past it for the two to agree.
    :ivar profile_speeds: Sound speed at each node, in m/s.
    """

    spreading: DynamicArray
    slope: DynamicArray
    profile_depths: NDArray[np.float64]
    profile_speeds: NDArray[np.float64]


class RayMarch(NamedTuple):
    """Per-ray history of a range march, all of shape ``(n_rays, n_steps)``.

    :ivar positions: ``z`` at each range sample.
    :ivar times: Cumulative travel time at each range sample, zero at the start.
    :ivar arc_lengths: Cumulative arc length along the ray at each range sample,
        zero at the start. :math:`ds/dr = 1/(\\xi c) = 1/\\cos\\theta \\ge 1`
        with :math:`\\theta` the local ray angle, so it never falls below the
        range spanned and exceeds it by exactly the obliquity of the path; a
        reflection adds no path, so it stays continuous across one, like the
        time. This is the measure a volume absorption
        :math:`e^{-\\int \\alpha\\,ds}` multiplies on (Jensen Sect. 3.6.2,
        Eq. 3.116).
    :ivar verticals: Vertical slowness :math:`\\zeta` at each range sample.
    :ivar reflections: Boundary reflections resolved inside each range step
        (zero in the first column, which is the launch point). Crossings of a
        profile node are not reflections and are not counted here.
    :ivar upper_reflections: The subset of ``reflections`` taken at the
        ``upper`` boundary, so ``reflections - upper_reflections`` are the ones
        taken at ``lower``. The two carry different reflection coefficients (a
        pressure-release sea surface inverts the pressure, a seabed need not),
        so a caller building a field has to tell them apart; a caller wanting
        only the geometry can ignore this and read ``reflections``. Always zero
        for a medium with no upper boundary.
    :ivar spreadings: Ray-tube spreading :math:`q`, or ``None`` when the march
        was not asked to carry it.
    :ivar spreading_slopes: Its conjugate :math:`p`, :math:`p = \\xi\\,dq/dr`,
        or ``None``. The beam half-width and wavefront curvature follow from the
        pair as Jensen Eqs. (3.89)-(3.90),
        :math:`W = \\sqrt{-2/(\\omega\\,\\mathrm{Im}[p/q])}` and
        :math:`K = -c\\,\\mathrm{Re}[p/q]`.
    """

    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    arc_lengths: NDArray[np.float64]
    verticals: NDArray[np.float64]
    reflections: NDArray[np.int_]
    upper_reflections: NDArray[np.int_]
    spreadings: DynamicArray | None = None
    spreading_slopes: DynamicArray | None = None


class _Impulses(NamedTuple):
    """Where ``p`` jumps along a ray, and by how much.

    :ivar interfaces: Depths of the profile nodes strictly inside the medium at
        which ``dc/dz`` actually jumps, ascending. Nodes a straight profile runs
        through are dropped: they would split steps for a zero impulse.
    :ivar levels: Those depths together with the reflecting boundaries,
        ascending; every depth a sub-step may be made to land on.
    :ivar strengths: The effective :math:`[dc/dz]/c` at each level, in 1/m: the
        gradient jump itself at an interface, and twice the boundary gradient
        (signed outward, so ``+2 dc/dz`` below and ``-2 dc/dz`` above) at a
        reflecting boundary, which is the jump the image medium presents there.
    """

    interfaces: NDArray[np.float64]
    levels: NDArray[np.float64]
    strengths: NDArray[np.float64]


class _Dynamic(NamedTuple):
    """The dynamic pair in flight, with the impulses it will meet."""

    spreading: DynamicArray
    slope: DynamicArray
    impulses: _Impulses


class _Step(NamedTuple):
    """What one range step of the march produced, per ray."""

    position: NDArray[np.float64]
    vertical: NDArray[np.float64]
    time: NDArray[np.float64]
    path: NDArray[np.float64]
    reflections: NDArray[np.int_]
    upper_reflections: NDArray[np.int_]
    dynamic: _Dynamic | None = None


def _rk4(
    deriv: RayDerivative, z: NDArray[np.float64], zeta: NDArray[np.float64],
    h: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64]]:
    """One Runge-Kutta step of per-ray range size ``h``.

    The time and the arc length come back as *increments*: they are quadratures
    riding on the geometric stages, they feed nothing back into them, and the
    caller accumulates them itself.
    """
    k1z, k1zeta, k1t, k1s = deriv(z, zeta)
    k2z, k2zeta, k2t, k2s = deriv(z + 0.5 * h * k1z, zeta + 0.5 * h * k1zeta)
    k3z, k3zeta, k3t, k3s = deriv(z + 0.5 * h * k2z, zeta + 0.5 * h * k2zeta)
    k4z, k4zeta, k4t, k4s = deriv(z + h * k3z, zeta + h * k3zeta)
    return (z + h / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z),
            zeta + h / 6.0 * (k1zeta + 2 * k2zeta + 2 * k3zeta + k4zeta),
            h / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t),
            h / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s))


def _gradients_with_half_spaces(
    z_prof: NDArray[np.float64], c_prof: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-segment ``dc/dz``, with the half spaces beyond the profile's ends.

    The profile arrays *are* the medium: both callers read the sound speed off
    them with :func:`numpy.interp`, which clamps, so past either end the medium
    is homogeneous and its gradient is zero there. Padding the segment gradients
    with that zero at both ends covers the whole line, one entry per gap between
    consecutive nodes plus one for each half space, which buys two things at
    once. A :func:`numpy.searchsorted` over the nodes then indexes it without
    clipping, so a boundary standing outside the sampled profile sees the
    homogeneous half space it is really in rather than an extrapolation of the
    last segment; and one :func:`numpy.diff` gives the gradient jump at *every*
    node, the two end ones included, which is what an end node needs when the
    medium is open on that side and it is a discontinuity like any other.
    """
    return np.concatenate(([0.0], np.diff(c_prof) / np.diff(z_prof), [0.0]))


def _segment_gradient(
    z_prof: NDArray[np.float64], grad: NDArray[np.float64], level: float, *,
    inward: float,
) -> float:
    """``dc/dz`` of the profile segment on the medium side of ``level``.

    ``grad`` is the padded array above, so the segment picked is the one holding
    ``level`` plus an infinitesimal step in the ``inward`` direction, the sign
    that leads into the medium, whether that lands inside the profile or in a
    half space beyond it.
    """
    seg = int(np.searchsorted(
        z_prof, level, side="right" if inward > 0.0 else "left"))
    return float(grad[seg])


def _prepare_impulses(
    dynamic: DynamicRays, lower: float, upper: float | None,
) -> _Impulses:
    """Locate every discontinuity of ``dc/dz`` a ray can meet, and size it.

    See the module docstring for where the three strengths come from: the raw
    gradient jump at an interior node (Jensen Eq. 3.130) and twice the boundary
    gradient at a reflection (Eq. 3.123), which is the jump the mirrored profile
    of the image medium shows at that boundary. Everything here depends only on
    depth, so it is computed once for the whole march.

    What makes a node an interface is that ``dc/dz`` jumps there and that the
    ray can reach it, i.e. that it lies strictly inside the medium; being an end
    of the sampled profile has nothing to do with it. For the ocean the two ends
    are the sea surface and the bottom, so they are boundaries and get the image
    medium's doubled strength instead. For a medium open on one side they are
    not: the atmosphere ends its profile in mid air, above which the caller's
    clamped interpolation makes the medium homogeneous, and the last node is
    then a kink the rays climb through like any other. Dropping it would put
    them back on the free-space spreading this whole apparatus exists to avoid,
    silently, and in proportion to the gradient the profile ends on.
    """
    z_prof = np.asarray(dynamic.profile_depths, dtype=np.float64).ravel()
    c_prof = np.asarray(dynamic.profile_speeds, dtype=np.float64).ravel()
    grad = _gradients_with_half_spaces(z_prof, c_prof)
    jumps = np.diff(grad)  # per node: segment below minus segment above
    # A jump of exactly zero is a node the profile runs straight through, and
    # `astype(bool)` is that test: no tolerance is wanted here, because a jump
    # of any size at all is an impulse the spreading has to take, while a
    # tolerance would silently delete the small ones a finely sampled profile
    # is made of.
    keep = (z_prof > lower) & jumps.astype(bool)
    if upper is not None:
        keep &= z_prof < upper
    interfaces = z_prof[keep]
    speeds = c_prof[keep]

    levels = [np.array([lower])]
    strengths = [np.array([2.0 * _segment_gradient(z_prof, grad, lower, inward=1.0)
                           / float(np.interp(lower, z_prof, c_prof))])]
    levels.append(interfaces)
    strengths.append(jumps[keep] / speeds)
    if upper is not None:
        levels.append(np.array([float(upper)]))
        strengths.append(
            np.array([-2.0 * _segment_gradient(z_prof, grad, upper, inward=-1.0)
                      / float(np.interp(upper, z_prof, c_prof))])
        )
    return _Impulses(interfaces, np.concatenate(levels), np.concatenate(strengths))


def _next_interface(
    z: NDArray[np.float64], z_end: NDArray[np.float64],
    interfaces: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.int_]]:
    """First interface the step reaches, in the direction it travels.

    A ray resting exactly on an interface (the previous sub-step landed there)
    must not find that same one again, hence the strict search on the near side
    and the inclusive test on the far side. The step is read as monotone in
    ``z``, the same assumption the boundary interpolant already makes.
    """
    down = z_end > z
    ahead = np.where(down, np.searchsorted(interfaces, z, side="right"),
                     np.searchsorted(interfaces, z, side="left") - 1)
    within = (ahead >= 0) & (ahead < interfaces.size)
    idx = np.clip(ahead, 0, interfaces.size - 1)
    depth = interfaces[idx]
    reached = np.where(down, depth <= z_end, depth >= z_end)
    return within & reached, idx


def _advance_spreading(
    dyn: _Dynamic, h: NDArray[np.float64], xi: NDArray[np.float64],
) -> _Dynamic:
    """Carry ``(q, p)`` across a range increment inside one profile segment.

    ``c''`` vanishes there, so ``dp/dr = 0`` and ``dq/dr = p/xi`` with ``p``
    constant: the pair is a straight line in range and the closed form below is
    what the four Runge-Kutta stages that place the ray would return, digit for
    digit. It is applied on the very increments those stages are taken over,
    the sub-steps included, so the pair never sees a step the geometry did not.
    """
    return dyn._replace(spreading=dyn.spreading + h * dyn.slope / xi)


def _apply_impulse(
    dyn: _Dynamic, xi: NDArray[np.float64], zeta: NDArray[np.float64],
    level: NDArray[np.float64], crossed: NDArray[np.bool_],
) -> _Dynamic:
    """Jump ``p`` where the sub-step landed on a gradient discontinuity.

    ``q`` is continuous across one (Jensen Eqs. 3.122, 3.129) and ``zeta`` is
    the incident value, taken before any reflection flips it; only its
    magnitude enters, which is what makes the rule reciprocal.
    """
    strength = dyn.impulses.strengths[np.searchsorted(dyn.impulses.levels, level)]
    jump = (-(xi**2) * strength * dyn.spreading
            / np.maximum(np.abs(zeta), _GRAZING_VERTICAL))
    return dyn._replace(slope=np.where(crossed, dyn.slope + jump, dyn.slope))


def _crossing_fraction(
    xi: NDArray[np.float64],
    za: NDArray[np.float64], zeta_a: NDArray[np.float64],
    zb: NDArray[np.float64], zeta_b: NDArray[np.float64],
    h: NDArray[np.float64], target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """How far into the step the ray first reaches ``target``, in [0, 1].

    The cubic Hermite through the two endpoint positions and their two slopes
    (dz/dr = zeta/xi) is the step's own interpolant, so bisecting it locates the
    crossing without a single further evaluation of the profile. Bisection
    rather than a closed-form cubic root because it is branch-free across rays
    and cannot pick the wrong one of three.
    """
    m0 = h * zeta_a / xi
    m1 = h * zeta_b / xi

    def offset_at(s: NDArray[np.float64]) -> NDArray[np.float64]:
        s2 = s * s
        s3 = s2 * s
        return ((2.0 * s3 - 3.0 * s2 + 1.0) * za + (s3 - 2.0 * s2 + s) * m0
                + (3.0 * s2 - 2.0 * s3) * zb + (s3 - s2) * m1 - target)

    lo = np.zeros_like(za)
    hi = np.ones_like(za)
    start_sign = np.sign(offset_at(lo))
    for _ in range(_BOUNDARY_BISECTIONS):
        mid = 0.5 * (lo + hi)
        keeps_sign = np.sign(offset_at(mid)) == start_sign
        lo = np.where(keeps_sign, mid, lo)
        hi = np.where(keeps_sign, hi, mid)
    return 0.5 * (lo + hi)


def _polish_fraction(
    deriv: RayDerivative, xi: NDArray[np.float64],
    z: NDArray[np.float64], zeta: NDArray[np.float64],
    h: NDArray[np.float64], target: NDArray[np.float64],
    frac: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Newton-refine the interpolated crossing against real Runge-Kutta steps.

    The interpolant places the crossing to its own order, which is one short of
    the step's. Polishing costs two evaluations per bounce, rare enough to be
    free, and buys back the order: the residual is then RK4's own. A ray tangent
    to the boundary has no slope to divide by, so it keeps the interpolated
    value, which is exactly the case where the interpolant is best.
    """
    for _ in range(_BOUNDARY_NEWTON_STEPS):
        z_try, zeta_try, _unused_t, _unused_s = _rk4(deriv, z, zeta, frac * h)
        slope = h * zeta_try / xi
        usable = np.abs(slope) > _GRAZING_SLOPE
        # Both branches of np.where are evaluated, so the divisor has to be
        # finite even where the result is discarded.
        step = np.where(usable, (z_try - target) / np.where(usable, slope, 1.0), 0.0)
        frac = np.clip(frac - step, 0.0, 1.0)
    return frac


class _Levels(NamedTuple):
    """Which level each ray meets inside the sub-step, and what kind it is.

    :ivar target: The depth the sub-step is aimed at, per ray. Meaningless
        where ``crossed`` is false, and never read there.
    :ivar crossed: Rays that meet a level of either kind and so have their
        step split.
    :ivar reflects: The subset of them that meet a reflecting boundary. A ray
        that meets a profile node instead is in ``crossed`` and not here: it
        goes straight on through, and only its spreading feels the kink.
    :ivar at_upper: The subset of ``reflects`` that leave through ``upper``.
    """

    target: NDArray[np.float64]
    crossed: NDArray[np.bool_]
    reflects: NDArray[np.bool_]
    at_upper: NDArray[np.bool_]


def _levels_reached(
    z: NDArray[np.float64], z_end: NDArray[np.float64],
    moving: NDArray[np.bool_], *, lower: float, upper: float | None,
    dyn: _Dynamic | None,
) -> _Levels:
    """The nearest level ahead of each ray, boundaries and profile nodes alike.

    An interface always lies strictly inside the medium, so a step that would
    cross both meets the interface first and the boundary is left to the
    sub-step after it; that ordering is what the unconditional overwrite of
    ``target`` below expresses.
    """
    out = z_end < lower if upper is None else (z_end < lower) | (z_end > upper)
    reflects = out & moving
    crossed = reflects
    target = (np.full(z.size, lower) if upper is None
              else np.where(z_end < lower, lower, float(upper)))
    if dyn is not None and dyn.impulses.interfaces.size:
        kink, idx = _next_interface(z, z_end, dyn.impulses.interfaces)
        kink &= moving
        target = np.where(kink, dyn.impulses.interfaces[idx], target)
        reflects = reflects & ~kink
        crossed = crossed | kink
    # Which boundary a reflection happened at is settled by the same test the
    # target was: everything that left the medium and is not below the lower
    # boundary went out through the upper one.
    return _Levels(target, crossed, reflects, reflects & ~(z_end < lower))


def _advance_one_step(
    deriv: RayDerivative, *, xi: NDArray[np.float64],
    z: NDArray[np.float64], zeta: NDArray[np.float64],
    range_step: float, lower: float, upper: float | None,
    dyn: _Dynamic | None = None,
) -> _Step:
    """Advance every ray by one whole range step, splitting it at each level.

    A reflection is handled by ending a sub-step exactly on the boundary rather
    than by taking the whole step and folding whatever came out of the medium
    back into it. Folding does two wrong things at once: it integrates the step
    through medium that is not there (the profile saturates outside it) and it
    applies at the step's end a reflection that happened somewhere inside the
    step, which leaves a first-order error at every bounce sitting inside a
    fourth-order integration. Splitting keeps a reflected ray at the order the
    rest of the path is integrated with.

    With ``dyn`` the same splitting is asked of the profile nodes, because the
    impulse in ``p`` has to be applied where the ray actually meets the kink and
    not at the end of whatever step happened to contain it. Both kinds of level
    are found together by :func:`_levels_reached`.
    """
    h = np.full(z.size, float(range_step))
    elapsed = np.zeros(z.size)
    travelled = np.zeros(z.size)
    bounces = np.zeros(z.size, dtype=np.int_)
    bounces_upper = np.zeros(z.size, dtype=np.int_)
    # Crossing a profile node is not pathological the way sixteen bounces in
    # one step would be: a steep ray through a finely sampled profile meets
    # them legitimately, so the budget grows by one per node it could meet.
    budget = _MAX_CROSSINGS_PER_STEP + (0 if dyn is None
                                        else dyn.impulses.interfaces.size)
    for _ in range(budget):
        # A ray whose step is spent has h = 0: its stage evaluations reproduce
        # its own state and add no time, and `moving` masks it out of every
        # update, so it costs arithmetic and nothing else. Once every ray is
        # spent nothing crosses and the loop leaves below.
        moving = h > 0.0
        z_end, zeta_end, dt, ds = _rk4(deriv, z, zeta, h)
        target, crossed, reflects, at_upper = _levels_reached(
            z, z_end, moving, lower=lower, upper=upper, dyn=dyn)
        if not crossed.any():
            z = np.where(moving, z_end, z)
            zeta = np.where(moving, zeta_end, zeta)
            elapsed += np.where(moving, dt, 0.0)
            travelled += np.where(moving, ds, 0.0)
            if dyn is not None:
                dyn = _advance_spreading(dyn, h, xi)
            break
        frac = np.clip(
            _crossing_fraction(xi, z, zeta, z_end, zeta_end, h, target), 0.0, 1.0
        )
        frac = _polish_fraction(deriv, xi, z, zeta, h, target, frac)
        h_sub = np.where(crossed, frac, 1.0) * h
        z_sub, zeta_sub, dt_sub, ds_sub = _rk4(deriv, z, zeta, h_sub)
        if dyn is not None:
            dyn = _apply_impulse(_advance_spreading(dyn, h_sub, xi), xi, zeta_sub,
                                 target, crossed)
        # A reflection is specular and instantaneous: zeta changes sign and
        # neither time nor path is added, so only the sub-step before it is
        # charged. Crossing a profile node changes neither: the ray goes
        # straight on through, only its spreading feels the kink.
        z = np.where(moving, np.where(crossed, target, z_sub), z)
        zeta = np.where(moving, np.where(reflects, -zeta_sub, zeta_sub), zeta)
        elapsed += np.where(moving, dt_sub, 0.0)
        travelled += np.where(moving, ds_sub, 0.0)
        h = np.where(moving, h - h_sub, 0.0)
        bounces += reflects.astype(np.int_)
        bounces_upper += at_upper.astype(np.int_)
    return _Step(z, zeta, elapsed, travelled, bounces, bounces_upper, dyn)


def march_rays(
    deriv: RayDerivative,
    *,
    xi: NDArray[np.float64],
    z0: NDArray[np.float64],
    zeta0: NDArray[np.float64],
    range_step: float,
    n_steps: int,
    lower: float,
    upper: float | None = None,
    dynamic: DynamicRays | None = None,
) -> RayMarch:
    """March rays in range, splitting every step at the levels it crosses.

    :param deriv: The ray equations, vectorised over rays.
    :param xi: Per-ray horizontal slowness (> 0), which the step's own
        interpolant needs for ``dz/dr = zeta/xi``.
    :param z0: Per-ray launch position across the medium.
    :param zeta0: Per-ray launch vertical slowness.
    :param range_step: Range increment between samples.
    :param n_steps: Number of range samples, launch point included (>= 2).
    :param lower: Reflecting boundary below (the sea surface, the ground).
    :param upper: Reflecting boundary above, or ``None`` for a medium open on
        that side (the ocean bottom; the atmosphere has none).
    :param dynamic: Ray-tube spreading to carry alongside the trajectory, or
        ``None`` to march the geometry alone. Asking for it also makes the
        march split its steps at the profile nodes, so the trajectory it
        returns is not bit-for-bit the one it returns without (see the module
        docstring).
    :return: A :class:`RayMarch`.
    """
    ns = int(n_steps)
    zeta = np.array(zeta0, dtype=np.float64)
    z = np.array(np.broadcast_to(np.asarray(z0, dtype=np.float64), zeta.shape))

    positions = np.zeros((z.size, ns))
    times = np.zeros((z.size, ns))
    arcs = np.zeros((z.size, ns))
    verticals = np.zeros((z.size, ns))
    reflections = np.zeros((z.size, ns), dtype=np.int_)
    upper_reflections = np.zeros((z.size, ns), dtype=np.int_)
    positions[:, 0] = z
    verticals[:, 0] = zeta

    dyn: _Dynamic | None = None
    dyn_dtype = np.float64 if dynamic is None else np.result_type(
        np.asarray(dynamic.spreading), np.asarray(dynamic.slope), np.float64)
    spreadings = np.zeros((z.size, ns if dynamic is not None else 0), dtype=dyn_dtype)
    slopes = np.zeros_like(spreadings)
    if dynamic is not None:
        dyn = _Dynamic(
            np.array(np.broadcast_to(dynamic.spreading, zeta.shape), dtype=dyn_dtype),
            np.array(np.broadcast_to(dynamic.slope, zeta.shape), dtype=dyn_dtype),
            _prepare_impulses(dynamic, lower, upper),
        )
        spreadings[:, 0] = dyn.spreading
        slopes[:, 0] = dyn.slope

    for s in range(1, ns):
        step = _advance_one_step(deriv, xi=xi, z=z, zeta=zeta, dyn=dyn,
                                 range_step=range_step, lower=lower, upper=upper)
        z, zeta = step.position, step.vertical
        positions[:, s] = z
        verticals[:, s] = zeta
        times[:, s] = times[:, s - 1] + step.time
        arcs[:, s] = arcs[:, s - 1] + step.path
        reflections[:, s] = step.reflections
        upper_reflections[:, s] = step.upper_reflections
        dyn = step.dynamic
        if dyn is not None:
            spreadings[:, s] = dyn.spreading
            slopes[:, s] = dyn.slope

    if dynamic is None:
        return RayMarch(positions, times, arcs, verticals, reflections,
                        upper_reflections)
    return RayMarch(positions, times, arcs, verticals, reflections,
                    upper_reflections, spreadings, slopes)
