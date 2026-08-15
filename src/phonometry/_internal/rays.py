#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Range-marching ray integration with specular boundaries (private).

In a range-independent medium the horizontal slowness
:math:`\xi = \cos\theta_0 / c(z_0)` is invariant along a ray, so with the
vertical slowness :math:`\zeta = \sin\theta / c` the trajectory marches in
range as

.. math::

    \frac{dz}{dr} = \frac{\zeta}{\xi}, \qquad
    \frac{d\zeta}{dr} = -\frac{dc/dz}{c^3 \xi}, \qquad
    \frac{dt}{dr} = \frac{1}{\xi c^2},

the travel time being a third state of the very step that places the ray rather
than a quadrature run over the finished path.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

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


class RayDerivative(Protocol):
    """The right-hand side of the ray equations, vectorised over rays.

    Returns ``(dz/dr, dzeta/dr, dt/dr)`` at the given state. The implementation
    closes over its own sound-speed profile and over :math:`\\xi`.
    """

    def __call__(
        self, z: NDArray[np.float64], zeta: NDArray[np.float64], /
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


class RayMarch(NamedTuple):
    """Per-ray history of a range march, all of shape ``(n_rays, n_steps)``.

    :ivar positions: ``z`` at each range sample.
    :ivar times: Cumulative travel time at each range sample, zero at the start.
    :ivar verticals: Vertical slowness :math:`\\zeta` at each range sample.
    :ivar reflections: Boundary reflections resolved inside each range step
        (zero in the first column, which is the launch point).
    """

    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    verticals: NDArray[np.float64]
    reflections: NDArray[np.int_]


class _Step(NamedTuple):
    """What one range step of the march produced, per ray."""

    position: NDArray[np.float64]
    vertical: NDArray[np.float64]
    time: NDArray[np.float64]
    reflections: NDArray[np.int_]


def _rk4(
    deriv: RayDerivative, z: NDArray[np.float64], zeta: NDArray[np.float64],
    h: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """One Runge-Kutta step of per-ray range size ``h``; the time as increment."""
    k1z, k1zeta, k1t = deriv(z, zeta)
    k2z, k2zeta, k2t = deriv(z + 0.5 * h * k1z, zeta + 0.5 * h * k1zeta)
    k3z, k3zeta, k3t = deriv(z + 0.5 * h * k2z, zeta + 0.5 * h * k2zeta)
    k4z, k4zeta, k4t = deriv(z + h * k3z, zeta + h * k3zeta)
    return (z + h / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z),
            zeta + h / 6.0 * (k1zeta + 2 * k2zeta + 2 * k3zeta + k4zeta),
            h / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t))


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
        z_try, zeta_try, _unused = _rk4(deriv, z, zeta, frac * h)
        slope = h * zeta_try / xi
        usable = np.abs(slope) > _GRAZING_SLOPE
        # Both branches of np.where are evaluated, so the divisor has to be
        # finite even where the result is discarded.
        step = np.where(usable, (z_try - target) / np.where(usable, slope, 1.0), 0.0)
        frac = np.clip(frac - step, 0.0, 1.0)
    return frac


def _advance_one_step(
    deriv: RayDerivative, *, xi: NDArray[np.float64],
    z: NDArray[np.float64], zeta: NDArray[np.float64],
    range_step: float, lower: float, upper: float | None,
) -> _Step:
    """Advance every ray by one whole range step, splitting it at each boundary.

    A reflection is handled by ending a sub-step exactly on the boundary rather
    than by taking the whole step and folding whatever came out of the medium
    back into it. Folding does two wrong things at once: it integrates the step
    through medium that is not there (the profile saturates outside it) and it
    applies at the step's end a reflection that happened somewhere inside the
    step, which leaves a first-order error at every bounce sitting inside a
    fourth-order integration. Splitting keeps a reflected ray at the order the
    rest of the path is integrated with.
    """
    h = np.full(z.size, float(range_step))
    elapsed = np.zeros(z.size)
    bounces = np.zeros(z.size, dtype=np.int_)
    for _ in range(_MAX_CROSSINGS_PER_STEP):
        # A ray whose step is spent has h = 0: its stage evaluations reproduce
        # its own state and add no time, and `moving` masks it out of every
        # update, so it costs arithmetic and nothing else. Once every ray is
        # spent nothing crosses and the loop leaves below.
        moving = h > 0.0
        z_end, zeta_end, dt = _rk4(deriv, z, zeta, h)
        out = z_end < lower if upper is None else (z_end < lower) | (z_end > upper)
        crossed = out & moving
        if not crossed.any():
            z = np.where(moving, z_end, z)
            zeta = np.where(moving, zeta_end, zeta)
            elapsed += np.where(moving, dt, 0.0)
            break
        target = (np.full(z.size, lower) if upper is None
                  else np.where(z_end < lower, lower, float(upper)))
        frac = np.clip(
            _crossing_fraction(xi, z, zeta, z_end, zeta_end, h, target), 0.0, 1.0
        )
        frac = _polish_fraction(deriv, xi, z, zeta, h, target, frac)
        h_sub = np.where(crossed, frac, 1.0) * h
        z_sub, zeta_sub, dt_sub = _rk4(deriv, z, zeta, h_sub)
        # A reflection is specular and instantaneous: zeta changes sign and no
        # time is added, so only the sub-step before it is charged.
        z = np.where(moving, np.where(crossed, target, z_sub), z)
        zeta = np.where(moving, np.where(crossed, -zeta_sub, zeta_sub), zeta)
        elapsed += np.where(moving, dt_sub, 0.0)
        h = np.where(moving, h - h_sub, 0.0)
        bounces += crossed.astype(np.int_)
    return _Step(z, zeta, elapsed, bounces)


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
) -> RayMarch:
    """March rays in range, splitting every step at the boundaries it crosses.

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
    :return: A :class:`RayMarch`.
    """
    ns = int(n_steps)
    zeta = np.array(zeta0, dtype=np.float64)
    z = np.array(np.broadcast_to(np.asarray(z0, dtype=np.float64), zeta.shape))

    positions = np.zeros((z.size, ns))
    times = np.zeros((z.size, ns))
    verticals = np.zeros((z.size, ns))
    reflections = np.zeros((z.size, ns), dtype=np.int_)
    positions[:, 0] = z
    verticals[:, 0] = zeta

    for s in range(1, ns):
        step = _advance_one_step(deriv, xi=xi, z=z, zeta=zeta,
                                 range_step=range_step, lower=lower, upper=upper)
        z, zeta = step.position, step.vertical
        positions[:, s] = z
        verticals[:, s] = zeta
        times[:, s] = times[:, s - 1] + step.time
        reflections[:, s] = step.reflections

    return RayMarch(positions, times, verticals, reflections)
