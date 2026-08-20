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

A sloping boundary
------------------

The ``upper`` boundary may be a piecewise-linear polyline ``z(r)`` instead of a
level, which is the faceted boundary model of Jensen Fig. 3.20 and the first
range dependence this core admits; the profile the ``deriv`` closes over stays
range independent. Three things change at such a boundary and nothing changes
anywhere else.

**The crossing search.** The bracketing function becomes the step's own cubic
Hermite interpolant *minus the interpolated boundary*, a difference that is
continuous across the polyline's vertices, so the same bisection that pins a
level crossing pins a sloping one, vertices included; the Newton polish divides
by the difference of slopes, ray minus facet, and a ray running nearly parallel
to the facet has no slope difference to divide by, so it keeps the bisected
value exactly as a boundary-tangent ray does at a level. A vertex has no
tangent of its own and the faceted model needs none: the reflection uses the
facet holding the located crossing, which for a ray aimed at a vertex is
whichever side the bisection's last halving landed on, deterministic for a
given input and specular off a real facet either way; a crossing landing
*exactly* on the node resolves to the facet ahead in range, the same
rounding-free rule the profile-node lookup uses.

**The reflection.** Specular about the local facet, Jensen Eq. (3.121):
:math:`\mathbf{t}^R = \mathbf{t}^I - 2\alpha\,\mathbf{n}_\mathrm{bdry}` with
:math:`\alpha = \mathbf{t}^I \cdot \mathbf{n}_\mathrm{bdry}` (Eq. 3.118). In
the slowness components the marcher carries, with :math:`m = dz_b/dr` the
facet's slope, that is

.. math::

    d = \frac{2\,(\xi m - \zeta)}{1 + m^2}, \qquad
    \xi \mapsto \xi - d\,m, \qquad \zeta \mapsto \zeta + d ,

a rotation, so the slowness magnitude :math:`1/c` is conserved; at
:math:`m = 0` it collapses to the level rule :math:`\zeta \mapsto -\zeta` with
:math:`\xi` untouched, to the last bit. That is also why :math:`\xi` is an
argument of the derivative rather than a closure constant: it is invariant
along a ray of a range-independent profile *between* such bounces, and changes
at each of them by twice the local slope's worth of rotation.

**The dynamic pair.** Eq. (3.122) still reads :math:`q^R = q^I`,
:math:`p^R = p^I + q^I N`, and for a piecewise-linear facet the curvature
:math:`\kappa` of Eq. (3.123) vanishes, leaving
:math:`N = M(4c_\mathrm{n} - 2Mc_\mathrm{s})/c^2` with :math:`M = \beta/\alpha`
and :math:`c_\mathrm{n}, c_\mathrm{s}` the sound-speed derivatives along the
*ray's* normal and tangent (Eq. 3.124), which for a profile in depth alone are
:math:`c_\mathrm{n} = c\,c_z\,\xi` and :math:`c_\mathrm{s} = c\,c_z\,\zeta`
whatever the facet does. What the slope enters is only the ratio
:math:`M = \beta/\alpha` of the incident tangent's boundary-parallel to
boundary-normal components: with the inward normal
:math:`(m, -1)/\sqrt{1+m^2}` and the tangent signed right-handed,
:math:`\alpha = c\,(\xi m - \zeta)/\sqrt{1+m^2}` and
:math:`\beta = -c\,(\xi + \zeta m)/\sqrt{1+m^2}`, so

.. math::

    M = \frac{\xi + \zeta m}{\zeta - \xi m}, \qquad
    N = \frac{2\,c_z}{c}\, M\,(2\xi - M\zeta) ,

with :math:`\zeta` the signed incident vertical slowness. This is the same
formula the level case already validated, reached by rotating the frame until
the facet is level: the rotation re-mixes :math:`(\xi, \zeta)` into
:math:`\alpha, \beta` while leaving :math:`c_\mathrm{n}, c_\mathrm{s}`
untouched, because those differentiate along the ray, not along the boundary.
At :math:`m = 0`, :math:`M = \xi/\zeta` and :math:`N = 2\xi^2 c_z/(\zeta c)`,
the flat-bottom impulse above. The denominator of :math:`M` is
:math:`\propto \alpha`, the normal component of the incident tangent, so a ray
meeting the facet tangentially blows it up exactly as a grazing ray does at a
level, and it is floored the same way and for the same reason.

**What a range march cannot represent.** A bounce off an upslope steepens the
ray by twice the slope, and enough of them carry it past the vertical, where
the reflected ray runs *backward* in range. A marcher whose independent
variable is range has no way to carry it (the same one-way surgery the
parabolic equation performs on the elliptic problem), so the ray is terminated
at that bounce: it keeps its state, stops advancing, and the step at which it
stopped is reported per ray in ``stopped_columns``, so an amplitude carrier can
retire its contribution rather than sum a ray frozen in place. The threshold is
not exactly the vertical: a reflected ray within a milliradian of it
(:math:`\cos\theta' \le 10^{-3}`) climbs a thousand depths per unit range,
which the march would spend its whole crossing budget failing to resolve, so it
is terminated there, 0.06 degrees before it would have turned anyway. A ray
that exhausts the crossing budget of one step with range still to cover is
terminated the same way rather than silently handed the rest of the step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

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
#: Cosine of the local ray angle below which a reflected ray is terminated: a
#: range march cannot carry a ray past the vertical, and within a milliradian
#: of it the ray climbs a thousand depths per unit range, which no crossing
#: budget resolves. See the module docstring on what a range march cannot
#: represent.
_MIN_FORWARD_COSINE = 1e-3


class RayDerivative(Protocol):
    """The right-hand side of the ray equations, vectorised over rays.

    Returns ``(dz/dr, dzeta/dr, dt/dr, ds/dr)`` at the given state. The
    implementation closes over its own sound-speed profile.

    ``zeta`` is passed as well as ``z`` because a caller that asks for dynamic
    ray tracing gets sub-steps landing *exactly* on its profile nodes, where a
    naive lookup would resolve the segment by rounding rather than by the
    direction the ray travels; the sign of ``zeta`` is what settles it, and one
    stage evaluated on the wrong side of a node is a first-order error inside a
    fourth-order step.

    ``xi`` is passed rather than closed over because it is per-ray *state*, not
    a constant of the march: it is invariant along a ray of a range-independent
    profile only between reflections off a sloping boundary, each of which
    rotates the slowness pair (see the module docstring). A march over level
    boundaries passes the launch values unchanged, so a derivative written for
    one serves the other.
    """

    def __call__(
        self,
        z: NDArray[np.float64],
        zeta: NDArray[np.float64],
        xi: NDArray[np.float64],
        /,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]: ...


class SlopingBoundary(NamedTuple):
    """A piecewise-linear reflecting boundary ``z(r)`` for the ``upper`` side.

    :ivar ranges: Node ranges, ascending. Outside them the boundary holds the
        end value level (the same clamp :func:`numpy.interp` applies to the
        profile), so a polyline ending mid-run continues flat, with a vertex
        where it stops sloping.
    :ivar positions: Boundary coordinate at each node, in the marcher's own
        ``z``, all strictly beyond ``lower`` so a water (or air) column exists
        everywhere.
    """

    ranges: NDArray[np.float64]
    positions: NDArray[np.float64]


class _Sloping(NamedTuple):
    """The polyline with its per-facet slopes, resolved once for the march.

    ``slopes`` is padded with the zero slope of the level continuations beyond
    both ends, one entry per facet plus one per end, so
    ``slopes[searchsorted(ranges, r, side="right")]`` is the facet holding
    ``r`` with a vertex resolved to the facet ahead in range.
    """

    ranges: NDArray[np.float64]
    positions: NDArray[np.float64]
    slopes: NDArray[np.float64]


def _prepare_sloping(boundary: SlopingBoundary) -> _Sloping:
    r = np.asarray(boundary.ranges, dtype=np.float64).ravel()
    z = np.asarray(boundary.positions, dtype=np.float64).ravel()
    return _Sloping(r, z, np.concatenate(([0.0], np.diff(z) / np.diff(r), [0.0])))


def _sloping_position(
    boundary: _Sloping, r: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray(np.interp(r, boundary.ranges, boundary.positions))


def _sloping_slope(boundary: _Sloping, r: NDArray[np.float64]) -> NDArray[np.float64]:
    return boundary.slopes[np.searchsorted(boundary.ranges, r, side="right")]


class DynamicRays(NamedTuple):
    r"""The dynamic ray equations to integrate alongside the trajectory.

    :ivar spreading: Per-ray :math:`q(0)`. With :math:`q(0) = 0` and
        :math:`p(0) = 1/c(0)` (Jensen Eq. 3.63) the pair is the geometric
        spreading, :math:`r\,q = J`; with :math:`p(0) = 1` and
        :math:`q(0) = i\omega W_0^2/2` (Eq. 3.91) it is a Gaussian beam of
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
    r"""Per-ray history of a range march, all of shape ``(n_rays, n_steps)``.

    :ivar positions: ``z`` at each range sample.
    :ivar times: Cumulative travel time at each range sample, zero at the start.
    :ivar arc_lengths: Cumulative arc length along the ray at each range sample,
        zero at the start. :math:`ds/dr = 1/(\xi c) = 1/\cos\theta \ge 1`
        with :math:`\theta` the local ray angle, so it never falls below the
        range spanned and exceeds it by exactly the obliquity of the path; a
        reflection adds no path, so it stays continuous across one, like the
        time. This is the measure a volume absorption
        :math:`e^{-\int \alpha\,ds}` multiplies on (Jensen Sect. 3.6.2,
        Eq. 3.116).
    :ivar verticals: Vertical slowness :math:`\zeta` at each range sample.
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
    :ivar spreading_slopes: Its conjugate :math:`p`, :math:`p = \xi\,dq/dr`,
        or ``None``. The beam half-width and wavefront curvature follow from the
        pair as Jensen Eqs. (3.89)-(3.90),
        :math:`W = \sqrt{-2/(\omega\,\mathrm{Im}[p/q])}` and
        :math:`K = -c\,\mathrm{Re}[p/q]`.
    :ivar horizontals: Horizontal slowness :math:`\xi` at each range sample.
        Over level boundaries it is the launch column repeated, Snell's
        invariant; a reflection off a sloping boundary rotates it (module
        docstring), so a consumer forming ray-centred coordinates must read it
        here per sample rather than keep the launch value.
    :ivar stopped_columns: Per ray, the first range-sample index at which the
        march could no longer advance it (a reflection past the vertical, or a
        crossing budget exhausted with range still to cover; both need a
        sloping boundary to happen), or ``n_steps`` for a ray marched to the
        end. From that column on the recorded state is frozen at the point of
        the terminating bounce, which is *not* on the column's own range, so a
        consumer must treat columns at or past this index as absent.
    """

    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    arc_lengths: NDArray[np.float64]
    verticals: NDArray[np.float64]
    reflections: NDArray[np.int_]
    upper_reflections: NDArray[np.int_]
    spreadings: DynamicArray | None = None
    spreading_slopes: DynamicArray | None = None
    horizontals: NDArray[np.float64] | None = None
    stopped_columns: NDArray[np.int_] | None = None


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
    deriv: RayDerivative,
    z: NDArray[np.float64],
    zeta: NDArray[np.float64],
    h: NDArray[np.float64],
    xi: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """One Runge-Kutta step of per-ray range size ``h``.

    The time and the arc length come back as *increments*: they are quadratures
    riding on the geometric stages, they feed nothing back into them, and the
    caller accumulates them itself.
    """
    k1z, k1zeta, k1t, k1s = deriv(z, zeta, xi)
    k2z, k2zeta, k2t, k2s = deriv(z + 0.5 * h * k1z, zeta + 0.5 * h * k1zeta, xi)
    k3z, k3zeta, k3t, k3s = deriv(z + 0.5 * h * k2z, zeta + 0.5 * h * k2zeta, xi)
    k4z, k4zeta, k4t, k4s = deriv(z + h * k3z, zeta + h * k3zeta, xi)
    return (
        z + h / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z),
        zeta + h / 6.0 * (k1zeta + 2 * k2zeta + 2 * k3zeta + k4zeta),
        h / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t),
        h / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s),
    )


def _gradients_with_half_spaces(
    z_prof: NDArray[np.float64],
    c_prof: NDArray[np.float64],
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
    z_prof: NDArray[np.float64],
    grad: NDArray[np.float64],
    level: float,
    *,
    inward: float,
) -> float:
    """``dc/dz`` of the profile segment on the medium side of ``level``.

    ``grad`` is the padded array above, so the segment picked is the one holding
    ``level`` plus an infinitesimal step in the ``inward`` direction, the sign
    that leads into the medium, whether that lands inside the profile or in a
    half space beyond it.
    """
    seg = int(np.searchsorted(z_prof, level, side="right" if inward > 0.0 else "left"))
    return float(grad[seg])


def _prepare_impulses(
    dynamic: DynamicRays,
    lower: float,
    upper: float | None,
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
    strengths = [
        np.array(
            [
                2.0
                * _segment_gradient(z_prof, grad, lower, inward=1.0)
                / float(np.interp(lower, z_prof, c_prof))
            ]
        )
    ]
    levels.append(interfaces)
    strengths.append(jumps[keep] / speeds)
    if upper is not None:
        levels.append(np.array([float(upper)]))
        strengths.append(
            np.array(
                [
                    -2.0
                    * _segment_gradient(z_prof, grad, upper, inward=-1.0)
                    / float(np.interp(upper, z_prof, c_prof))
                ]
            )
        )
    return _Impulses(interfaces, np.concatenate(levels), np.concatenate(strengths))


def _next_interface(
    z: NDArray[np.float64],
    z_end: NDArray[np.float64],
    interfaces: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.int_]]:
    """First interface the step reaches, in the direction it travels.

    A ray resting exactly on an interface (the previous sub-step landed there)
    must not find that same one again, hence the strict search on the near side
    and the inclusive test on the far side. The step is read as monotone in
    ``z``, the same assumption the boundary interpolant already makes.
    """
    down = z_end > z
    ahead = np.where(
        down,
        np.searchsorted(interfaces, z, side="right"),
        np.searchsorted(interfaces, z, side="left") - 1,
    )
    within = (ahead >= 0) & (ahead < interfaces.size)
    idx = np.clip(ahead, 0, interfaces.size - 1)
    depth = interfaces[idx]
    reached = np.where(down, depth <= z_end, depth >= z_end)
    return within & reached, idx


def _advance_spreading(
    dyn: _Dynamic,
    h: NDArray[np.float64],
    xi: NDArray[np.float64],
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
    dyn: _Dynamic,
    xi: NDArray[np.float64],
    zeta: NDArray[np.float64],
    level: NDArray[np.float64],
    crossed: NDArray[np.bool_],
) -> _Dynamic:
    """Jump ``p`` where the sub-step landed on a gradient discontinuity.

    ``q`` is continuous across one (Jensen Eqs. 3.122, 3.129) and ``zeta`` is
    the incident value, taken before any reflection flips it; only its
    magnitude enters, which is what makes the rule reciprocal.
    """
    strength = dyn.impulses.strengths[np.searchsorted(dyn.impulses.levels, level)]
    jump = (
        -(xi**2)
        * strength
        * dyn.spreading
        / np.maximum(np.abs(zeta), _GRAZING_VERTICAL)
    )
    return dyn._replace(slope=np.where(crossed, dyn.slope + jump, dyn.slope))


def _hermite_position(
    s: NDArray[np.float64],
    za: NDArray[np.float64],
    m0: NDArray[np.float64],
    zb: NDArray[np.float64],
    m1: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The step's own cubic Hermite interpolant of ``z``, at fraction ``s``."""
    s2 = s * s
    s3 = s2 * s
    return (
        (2.0 * s3 - 3.0 * s2 + 1.0) * za
        + (s3 - 2.0 * s2 + s) * m0
        + (3.0 * s2 - 2.0 * s3) * zb
        + (s3 - s2) * m1
    )


def _bisect_offset(
    offset_at: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    za: NDArray[np.float64],
) -> NDArray[np.float64]:
    """First sign change of ``offset_at`` on the unit interval, by bisection.

    Bisection rather than a closed-form cubic root because it is branch-free
    across rays and cannot pick the wrong one of three; and because the offset
    need not be a cubic at all, only continuous, which is what lets the same
    loop bracket a crossing of a piecewise-linear boundary.
    """
    lo = np.zeros_like(za)
    hi = np.ones_like(za)
    start_sign = np.sign(offset_at(lo))
    for _ in range(_BOUNDARY_BISECTIONS):
        mid = 0.5 * (lo + hi)
        keeps_sign = np.sign(offset_at(mid)) == start_sign
        lo = np.where(keeps_sign, mid, lo)
        hi = np.where(keeps_sign, hi, mid)
    return 0.5 * (lo + hi)


def _crossing_fraction(
    xi: NDArray[np.float64],
    za: NDArray[np.float64],
    zeta_a: NDArray[np.float64],
    zb: NDArray[np.float64],
    zeta_b: NDArray[np.float64],
    h: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """How far into the step the ray first reaches ``target``, in [0, 1].

    The cubic Hermite through the two endpoint positions and their two slopes
    (dz/dr = zeta/xi) is the step's own interpolant, so bisecting it locates the
    crossing without a single further evaluation of the profile.
    """
    m0 = h * zeta_a / xi
    m1 = h * zeta_b / xi

    def offset_at(s: NDArray[np.float64]) -> NDArray[np.float64]:
        return _hermite_position(s, za, m0, zb, m1) - target

    return _bisect_offset(offset_at, za)


def _crossing_fraction_sloping(
    xi: NDArray[np.float64],
    za: NDArray[np.float64],
    zeta_a: NDArray[np.float64],
    zb: NDArray[np.float64],
    zeta_b: NDArray[np.float64],
    h: NDArray[np.float64],
    r_start: NDArray[np.float64],
    boundary: _Sloping,
) -> NDArray[np.float64]:
    """The same search against a piecewise-linear boundary.

    The offset is the Hermite interpolant minus the boundary interpolated at
    the same fraction, continuous across the polyline's vertices, so one
    bisection locates the first crossing whether it lands on a facet or at a
    vertex. With a level polyline it is bit for bit :func:`_crossing_fraction`.
    """
    m0 = h * zeta_a / xi
    m1 = h * zeta_b / xi

    def offset_at(s: NDArray[np.float64]) -> NDArray[np.float64]:
        return _hermite_position(s, za, m0, zb, m1) - _sloping_position(
            boundary, r_start + s * h
        )

    return _bisect_offset(offset_at, za)


def _polish_fraction(
    deriv: RayDerivative,
    xi: NDArray[np.float64],
    z: NDArray[np.float64],
    zeta: NDArray[np.float64],
    h: NDArray[np.float64],
    target: NDArray[np.float64],
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
        z_try, zeta_try, _unused_t, _unused_s = _rk4(deriv, z, zeta, frac * h, xi)
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
    z: NDArray[np.float64],
    z_end: NDArray[np.float64],
    moving: NDArray[np.bool_],
    *,
    lower: float,
    upper: float | None,
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
    target = (
        np.full(z.size, lower)
        if upper is None
        else np.where(z_end < lower, lower, float(upper))
    )
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


def _polish_fraction_mixed(
    deriv: RayDerivative,
    xi: NDArray[np.float64],
    z: NDArray[np.float64],
    zeta: NDArray[np.float64],
    h: NDArray[np.float64],
    level: NDArray[np.float64],
    on_boundary: NDArray[np.bool_],
    boundary: _Sloping,
    r_start: NDArray[np.float64],
    frac: NDArray[np.float64],
) -> NDArray[np.float64]:
    """:func:`_polish_fraction` with a per-ray choice of target.

    Rays flagged ``on_boundary`` are polished against the sloping polyline, the
    rest against their ``level`` (an interface node or the level boundary
    below); one loop serves both so each Newton pass costs the same two step
    evaluations however the events split. The slope Newton divides by is the
    *difference* of ray and facet slopes, which is what vanishes for a ray
    running nearly parallel to a slope: such a ray keeps the bisected value,
    exactly as a boundary-tangent ray does at a level, and for the same reason.
    """
    for _ in range(_BOUNDARY_NEWTON_STEPS):
        z_try, zeta_try, _unused_t, _unused_s = _rk4(deriv, z, zeta, frac * h, xi)
        r_try = r_start + frac * h
        facet = np.where(on_boundary, _sloping_slope(boundary, r_try), 0.0)
        slope = h * zeta_try / xi - h * facet
        target = np.where(on_boundary, _sloping_position(boundary, r_try), level)
        usable = np.abs(slope) > _GRAZING_SLOPE
        step = np.where(usable, (z_try - target) / np.where(usable, slope, 1.0), 0.0)
        frac = np.clip(frac - step, 0.0, 1.0)
    return frac


def _advance_one_step(
    deriv: RayDerivative,
    *,
    xi: NDArray[np.float64],
    z: NDArray[np.float64],
    zeta: NDArray[np.float64],
    range_step: float,
    lower: float,
    upper: float | None,
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
    budget = _MAX_CROSSINGS_PER_STEP + (
        0 if dyn is None else dyn.impulses.interfaces.size
    )
    for _ in range(budget):
        # A ray whose step is spent has h = 0: its stage evaluations reproduce
        # its own state and add no time, and `moving` masks it out of every
        # update, so it costs arithmetic and nothing else. Once every ray is
        # spent nothing crosses and the loop leaves below.
        moving = h > 0.0
        z_end, zeta_end, dt, ds = _rk4(deriv, z, zeta, h, xi)
        target, crossed, reflects, at_upper = _levels_reached(
            z, z_end, moving, lower=lower, upper=upper, dyn=dyn
        )
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
        z_sub, zeta_sub, dt_sub, ds_sub = _rk4(deriv, z, zeta, h_sub, xi)
        if dyn is not None:
            dyn = _apply_impulse(
                _advance_spreading(dyn, h_sub, xi), xi, zeta_sub, target, crossed
            )
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


class _SlopingStep(NamedTuple):
    """What one range step of a sloping march produced, per ray."""

    position: NDArray[np.float64]
    vertical: NDArray[np.float64]
    horizontal: NDArray[np.float64]
    time: NDArray[np.float64]
    path: NDArray[np.float64]
    reflections: NDArray[np.int_]
    upper_reflections: NDArray[np.int_]
    stopped: NDArray[np.bool_]
    dynamic: _Dynamic | None = None


def _reflect_off_facet(
    xi: NDArray[np.float64],
    zeta: NDArray[np.float64],
    m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Specular reflection of the slowness pair about a facet of slope ``m``.

    Jensen Eq. (3.121) in components: subtract twice the boundary-normal
    component. Written so a level facet reproduces the level rule to the last
    bit: with ``m = 0`` the update is ``xi - 0`` and ``zeta - 2 zeta``, both
    exact in IEEE arithmetic.
    """
    d = 2.0 * (xi * m - zeta) / (1.0 + m * m)
    return xi - d * m, zeta + d


def _facet_spreading_jump(
    xi: NDArray[np.float64],
    zeta: NDArray[np.float64],
    m: NDArray[np.float64],
    z_b: NDArray[np.float64],
    z_prof: NDArray[np.float64],
    c_prof: NDArray[np.float64],
    grad: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The ``p`` impulse of a reflection off a sloping facet, Eq. (3.123).

    ``kappa = 0`` on a piecewise-linear facet, so
    ``N = M (4 c_n - 2 M c_s)/c^2`` with ``M`` built from the incident
    slownesses and the facet slope; see the module docstring for the closed
    form and its flat-limit check. ``grad`` is the padded per-segment gradient
    array of :func:`_gradients_with_half_spaces`; the water side of the
    boundary is above it, so the segment is resolved upward, whatever
    direction the ray arrived from (a rising facet can catch a climbing ray
    from below).
    """
    seg = np.searchsorted(z_prof, z_b, side="left")
    c_z = grad[seg]
    c_b = np.interp(z_b, z_prof, c_prof)
    denom = zeta - xi * m
    denom = np.where(
        np.abs(denom) < _GRAZING_VERTICAL, np.copysign(_GRAZING_VERTICAL, denom), denom
    )
    ratio = (xi + zeta * m) / denom
    return np.asarray((2.0 * c_z / c_b) * ratio * (2.0 * xi - ratio * zeta))


def _advance_one_step_sloping(
    deriv: RayDerivative,
    *,
    xi: NDArray[np.float64],
    z: NDArray[np.float64],
    zeta: NDArray[np.float64],
    r0: float,
    range_step: float,
    lower: float,
    boundary: _Sloping,
    stopped: NDArray[np.bool_],
    dyn: _Dynamic | None = None,
    profile: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    | None = None,
) -> _SlopingStep:
    """One whole range step against a piecewise-linear upper boundary.

    The structure is :func:`_advance_one_step`'s; what differs is confined to
    the boundary. The bottom crossing is bisected against the interpolated
    polyline; which event a step meets first is settled by comparing located
    fractions rather than by the flat case's standing order, because a facet
    can climb above a profile node and put the reflection *before* the
    interface a level bottom would always meet second (a tie goes to the
    reflection, which is what a bottom touching a node depth means). The
    reflection rotates the slowness pair about the local facet, the dynamic
    impulse is Eq. (3.123) on that facet, and a ray the rotation carries past
    the vertical is terminated, since a range march cannot bring it back.
    """
    h = np.where(stopped, 0.0, float(range_step))
    elapsed = np.zeros(z.size)
    travelled = np.zeros(z.size)
    bounces = np.zeros(z.size, dtype=np.int_)
    bounces_upper = np.zeros(z.size, dtype=np.int_)
    stopped = stopped.copy()
    xi = xi.copy()
    budget = _MAX_CROSSINGS_PER_STEP + (
        0 if dyn is None else dyn.impulses.interfaces.size
    )
    for _ in range(budget):
        moving = h > 0.0
        z_end, zeta_end, dt, ds = _rk4(deriv, z, zeta, h, xi)
        r_now = r0 + (float(range_step) - h)
        depth_end = _sloping_position(boundary, r_now + h)
        out_upper = (z_end > depth_end) & moving
        out_lower = (z_end < lower) & moving
        level = np.full(z.size, lower)
        kink = np.zeros(z.size, dtype=bool)
        if dyn is not None and dyn.impulses.interfaces.size:
            kink, idx = _next_interface(z, z_end, dyn.impulses.interfaces)
            kink &= moving
            level = np.where(kink, dyn.impulses.interfaces[idx], level)
        if not (out_upper | out_lower | kink).any():
            z = np.where(moving, z_end, z)
            zeta = np.where(moving, zeta_end, zeta)
            elapsed += np.where(moving, dt, 0.0)
            travelled += np.where(moving, ds, 0.0)
            if dyn is not None:
                dyn = _advance_spreading(dyn, h, xi)
            h = np.where(moving, 0.0, h)
            break
        frac_level = _crossing_fraction(xi, z, zeta, z_end, zeta_end, h, level)
        frac_upper = _crossing_fraction_sloping(
            xi, z, zeta, z_end, zeta_end, h, r_now, boundary
        )
        # A step flagged for both the sloping boundary and a level event takes
        # whichever its own interpolant meets first; the flat case's standing
        # order (interface before boundary) is the special case of this in
        # which the bottom can never stand above a node.
        upper_ev = out_upper & (~kink | (frac_upper <= frac_level))
        kink_ev = kink & ~upper_ev
        lower_ev = out_lower & ~kink
        reflects = upper_ev | lower_ev
        crossed = reflects | kink_ev
        frac = np.clip(np.where(upper_ev, frac_upper, frac_level), 0.0, 1.0)
        frac = _polish_fraction_mixed(
            deriv, xi, z, zeta, h, level, upper_ev, boundary, r_now, frac
        )
        h_sub = np.where(crossed, frac, 1.0) * h
        z_sub, zeta_sub, dt_sub, ds_sub = _rk4(deriv, z, zeta, h_sub, xi)
        r_cross = r_now + h_sub
        z_upper = _sloping_position(boundary, r_cross)
        target = np.where(upper_ev, z_upper, level)
        m = np.where(upper_ev, _sloping_slope(boundary, r_cross), 0.0)
        if dyn is not None:
            dyn = _advance_spreading(dyn, h_sub, xi)
            dyn = _apply_impulse(dyn, xi, zeta_sub, level, kink_ev | lower_ev)
            if profile is not None:
                jump = _facet_spreading_jump(xi, zeta_sub, m, z_upper, *profile)
                dyn = dyn._replace(
                    slope=np.where(
                        upper_ev, dyn.slope + jump * dyn.spreading, dyn.slope
                    )
                )
        xi_new, zeta_new = _reflect_off_facet(xi, zeta_sub, m)
        # A reflection within a milliradian of the vertical, or past it, is a
        # ray this march cannot carry: it is terminated at the bounce, state
        # frozen there. See the module docstring.
        dying = upper_ev & (xi_new <= _MIN_FORWARD_COSINE * np.hypot(xi_new, zeta_new))
        z = np.where(moving, np.where(crossed, target, z_sub), z)
        zeta = np.where(moving, np.where(reflects, zeta_new, zeta_sub), zeta)
        xi = np.where(moving & reflects & ~dying, xi_new, xi)
        elapsed += np.where(moving, dt_sub, 0.0)
        travelled += np.where(moving, ds_sub, 0.0)
        h = np.where(moving & ~dying, h - h_sub, 0.0)
        stopped = stopped | dying
        bounces += reflects.astype(np.int_)
        bounces_upper += upper_ev.astype(np.int_)
    # Leaving the loop with range still to cover means the crossing budget was
    # exhausted while events kept coming, which only a ray within a hair of
    # the vertical in a thinning column can do: terminate it rather than hand
    # it the rest of the step untraced.
    stopped = stopped | (h > 0.0)
    return _SlopingStep(
        z, zeta, xi, elapsed, travelled, bounces, bounces_upper, stopped, dyn
    )


def march_rays(
    deriv: RayDerivative,
    *,
    xi: NDArray[np.float64],
    z0: NDArray[np.float64],
    zeta0: NDArray[np.float64],
    range_step: float,
    n_steps: int,
    lower: float,
    upper: float | SlopingBoundary | None = None,
    dynamic: DynamicRays | None = None,
) -> RayMarch:
    """March rays in range, splitting every step at the levels it crosses.

    :param deriv: The ray equations, vectorised over rays.
    :param xi: Per-ray launch horizontal slowness (> 0). Over level boundaries
        it is invariant and the march never touches it; a sloping ``upper``
        rotates it at each of its reflections, and the running value is what
        every step's interpolant and derivative then use.
    :param z0: Per-ray launch position across the medium.
    :param zeta0: Per-ray launch vertical slowness.
    :param range_step: Range increment between samples.
    :param n_steps: Number of range samples, launch point included (>= 2).
    :param lower: Reflecting boundary below (the sea surface, the ground).
    :param upper: Reflecting boundary above: a level, a
        :class:`SlopingBoundary` polyline over range (the first range
        dependence this core admits; the profile stays range independent), or
        ``None`` for a medium open on that side (the ocean bottom; the
        atmosphere has none).
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
    xi_state = np.array(np.broadcast_to(np.asarray(xi, dtype=np.float64), zeta.shape))
    sloping = _prepare_sloping(upper) if isinstance(upper, SlopingBoundary) else None
    level_upper = None if isinstance(upper, SlopingBoundary) else upper
    stopped = np.zeros(z.size, dtype=bool)
    stopped_at = np.full(z.size, ns, dtype=np.int_)

    positions = np.zeros((z.size, ns))
    times = np.zeros((z.size, ns))
    arcs = np.zeros((z.size, ns))
    verticals = np.zeros((z.size, ns))
    horizontals = np.zeros((z.size, ns))
    reflections = np.zeros((z.size, ns), dtype=np.int_)
    upper_reflections = np.zeros((z.size, ns), dtype=np.int_)
    positions[:, 0] = z
    verticals[:, 0] = zeta
    horizontals[:, 0] = xi_state

    dyn: _Dynamic | None = None
    profile: (
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    ) = None
    dyn_dtype = (
        np.float64
        if dynamic is None
        else np.result_type(
            np.asarray(dynamic.spreading), np.asarray(dynamic.slope), np.float64
        )
    )
    spreadings = np.zeros((z.size, ns if dynamic is not None else 0), dtype=dyn_dtype)
    slopes = np.zeros_like(spreadings)
    if dynamic is not None:
        # A sloping boundary has no single level to fold the image medium at:
        # its reflection impulse is computed on the local facet instead, so
        # the static ladder carries the lower boundary and the interfaces only.
        dyn = _Dynamic(
            np.array(np.broadcast_to(dynamic.spreading, zeta.shape), dtype=dyn_dtype),
            np.array(np.broadcast_to(dynamic.slope, zeta.shape), dtype=dyn_dtype),
            _prepare_impulses(dynamic, lower, level_upper),
        )
        spreadings[:, 0] = dyn.spreading
        slopes[:, 0] = dyn.slope
        if sloping is not None:
            z_prof = np.asarray(dynamic.profile_depths, dtype=np.float64).ravel()
            c_prof = np.asarray(dynamic.profile_speeds, dtype=np.float64).ravel()
            profile = (z_prof, c_prof, _gradients_with_half_spaces(z_prof, c_prof))

    for s in range(1, ns):
        if sloping is None:
            step = _advance_one_step(
                deriv,
                xi=xi_state,
                z=z,
                zeta=zeta,
                dyn=dyn,
                range_step=range_step,
                lower=lower,
                upper=level_upper,
            )
            z, zeta = step.position, step.vertical
        else:
            sstep = _advance_one_step_sloping(
                deriv,
                xi=xi_state,
                z=z,
                zeta=zeta,
                dyn=dyn,
                r0=(s - 1) * range_step,
                range_step=range_step,
                lower=lower,
                boundary=sloping,
                stopped=stopped,
                profile=profile,
            )
            z, zeta, xi_state = sstep.position, sstep.vertical, sstep.horizontal
            newly = sstep.stopped & ~stopped
            stopped_at[newly] = s
            stopped = sstep.stopped
            step = _Step(
                sstep.position,
                sstep.vertical,
                sstep.time,
                sstep.path,
                sstep.reflections,
                sstep.upper_reflections,
                sstep.dynamic,
            )
        positions[:, s] = z
        verticals[:, s] = zeta
        horizontals[:, s] = xi_state
        times[:, s] = times[:, s - 1] + step.time
        arcs[:, s] = arcs[:, s - 1] + step.path
        reflections[:, s] = step.reflections
        upper_reflections[:, s] = step.upper_reflections
        dyn = step.dynamic
        if dyn is not None:
            spreadings[:, s] = dyn.spreading
            slopes[:, s] = dyn.slope

    if dynamic is None:
        return RayMarch(
            positions,
            times,
            arcs,
            verticals,
            reflections,
            upper_reflections,
            horizontals=horizontals,
            stopped_columns=stopped_at,
        )
    return RayMarch(
        positions,
        times,
        arcs,
        verticals,
        reflections,
        upper_reflections,
        spreadings,
        slopes,
        horizontals=horizontals,
        stopped_columns=stopped_at,
    )
