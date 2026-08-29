#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Realise an analog weighting prototype as biquads at the sampling rate.

The bilinear transform is exact in magnitude but wrong in frequency: it maps
the digital frequency :math:`f` onto the analog frequency
:math:`\Omega = 2 f_s \tan(\pi f / f_s)`, so a filter designed by transforming
the printed prototype has the prototype's response *at the warped frequency*
and not at the frequency the standard names. The error grows quadratically
with :math:`f / f_s` and is what the module this one serves used to hide by
running the sections at an oversampled rate.

This module removes the warp instead of outrunning it. It searches for an
analog prototype :math:`\tilde{H}` -- same number of zeros at the origin, same
count of stable quadratic and linear factors -- whose response *at the warped
frequencies* is the printed prototype's response at the true ones:

.. math::

   \bigl| \tilde{H}(j \Omega(f)) \bigr|
   \;\simeq\; \bigl| H(j\,2\pi f) \bigr|,
   \qquad \Omega(f) = 2 f_s \tan(\pi f / f_s).

Bilinear-transforming :math:`\tilde{H}` then gives a digital filter whose
magnitude is the printed curve's own, at the input rate, with no resampling
around it. What the caller runs is a plain ``sosfilt``, so block processing is
bit-identical to a single call and the whole path is one cascade of biquads.

Four properties make the search a design routine rather than an optimiser's
output:

* **Every parameter is a frequency.** Each factor is
  :math:`s^2 + b_1 s + b_0` with :math:`b_1 = e^{\theta_1}` and
  :math:`b_0 = e^{\theta_0}`, which is the Routh-Hurwitz condition for a stable
  quadratic: any real :math:`\theta` puts both roots strictly in the left half
  plane, so no step of the search can produce an unstable filter or a
  non-minimum-phase numerator, and no projection back into a feasible set is
  ever needed. What comes out is still a prototype rather than a coefficient
  array, and :func:`fitted_prototype_hz` hands it back in hertz, so what the
  fit did is legible. The corners the standard places low, where the warp is
  negligible, come back where they were printed: at 48 kHz the A weighting's
  double real pole at 20.598997 Hz comes back as a pair at
  :math:`-20.5981 \pm 0.1905\,\mathrm{j}` Hz -- the same corner to five
  digits, with a whisker of damping -- beside 107.657 for the printed
  107.65265 and 737.268 for 737.86223. The corners near the top of the band
  move much further, and moving them is the whole point: that displacement is
  the warp being cancelled.
* **The residual is in decibels.** The response enters only as
  :math:`\log |\tilde{H}|^2`, which turns a product of factors into a sum and
  makes the Jacobian one term per factor, bounded and analytic. A relative
  (voltage-ratio) residual cannot see an error that is far too *small* -- it
  saturates at -1 -- and a fit driven by one silently abandons the skirts,
  which is exactly where these curves live.
* **The design is pinned where the rate can reach.** Both the residual and
  the realised cascade are referred to one anchor frequency, and
  :func:`_anchor_hz` chooses it so that it always exists: the standard's own
  reference frequency at every rate that can carry it, and the middle of the
  fit band below ``2 * f_ref``, where the reference frequency is past the
  Nyquist frequency and asking the warp for it returns its alias rather than
  an error. What the pin fixes is the *level*, not the frequency: the cascade
  is set to the printed curve's own decibels at the anchor, so a tone reads
  the same number at 903 Hz sampling as at 48 kHz.
* **The work is fixed.** :data:`_LAWSON_ROUNDS` reweighting rounds of
  :data:`_LM_STEPS` Levenberg-Marquardt steps each, on a
  :data:`_GRID_POINTS`-point grid: no convergence test, no random state, no
  dependence on wall clock or thread count. The one thing that is not fixed is
  how many *starts* get that budget: a fit that comes back more than
  :data:`_FIT_FAILED_DB` from the printed curve has not produced a filter at
  all, and the remaining entries of :func:`_spare_placements` are tried before
  the best is kept. That branch is on the achieved residual, so it is still
  deterministic, and no rate this library is used at comes near it -- the
  loosest first fit in the corpus is three and a half times inside. See
  :func:`_lawson_fit` for what reproducibility that does and does not buy.

The reweighting is Lawson's algorithm: least squares whose weights are
multiplied by the current residual magnitude each round drives the fit from
least-squares toward minimax, which is the criterion that matters when the
result is graded against a tolerance mask rather than integrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

#: Decibels per natural log unit of a squared magnitude: ``10 / ln 10``.
_DB_PER_LOG = 10.0 / np.log(10.0)

#: Grid points, logarithmically spaced in the *warped* frequency
#: :math:`\Omega`. Spacing the grid in :math:`\Omega` rather than in :math:`f`
#: is what keeps the top of the band resolved: a decade of :math:`\Omega` near
#: the top maps to a tenfold smaller distance from the Nyquist frequency, so
#: the region where the fit is hardest gets the density it needs. Measured on
#: this corpus, a grid uniform in :math:`\log f` left 0.25 dB of ripple between
#: its points for the 468 curve at 44.1 kHz while reporting 0.05 dB on its own
#: samples; the same count spaced in :math:`\log \Omega` reports and delivers
#: 0.05 dB.
_GRID_POINTS = 257

#: Levenberg-Marquardt steps per reweighting round, and reweighting rounds.
#: Both are fixed budgets rather than convergence tests, so the routine always
#: does the same arithmetic, and they are set where the accuracy curve
#: flattens: measured over this corpus, raising them to 40 by 12 -- 2.4 times
#: the work, about 190 ms per design instead of 80 -- moves the worst
#: deviation of any (curve, rate) by at most 0.007 dB, and that one case (A at
#: 32 kHz, 0.008 dB to 0.001 dB) is already two orders of magnitude inside the
#: 0.05 dB its table is printed to.
_LM_STEPS = 25
_LAWSON_ROUNDS = 8

#: Lawson weight update exponent: ``w <- w * |residual|**p``. ``p = 1`` is
#: Lawson's own choice and is what drives weighted least squares to the
#: minimax solution.
_LAWSON_EXPONENT = 1.0

#: Levenberg-Marquardt damping: initial value, the factors applied on an
#: accepted and a rejected step, and the range it is held in. Ordinary
#: Marquardt values; nothing in the standards fixes them.
_LM_INITIAL_DAMPING = 1e-3
_LM_DAMPING_DOWN = 0.3
_LM_DAMPING_UP = 10.0
_LM_DAMPING_FLOOR = 1e-14
_LM_DAMPING_CEILING = 1e8

#: A step is accepted only if it lowers the cost by more than this relative
#: amount. The margin exists so that a step whose gain is at the level of
#: floating-point rounding is rejected on every machine rather than accepted on
#: some, which keeps the sequence of accepted steps -- and so the design -- from
#: depending on the last bits of a BLAS reduction.
_LM_ACCEPT_MARGIN = 1e-12

#: Decades a fitted corner frequency may sit outside the fit grid before it is
#: clamped. A factor this far out is flat over the whole grid to a part in
#: :math:`10^{24}`: it has stopped being a filter and become a constant, so
#: nothing measurable is given up by refusing to let the search push it
#: further. What is bought is that the roots stay small enough to multiply.
#: The number is not tuned to a result -- anything from about three decades
#: (flat to a part in a million) up is equally defensible -- and twelve is the
#: middle of that range on a log scale. Measured over the rates from 2 kHz up,
#: the clamp separates two behaviours rather than splitting one: the furthest
#: any corner travels and stays under it is 9.1 decades, and the only two that
#: pass it are at 49 and 81, which is the search abandoning a spare factor
#: rather than placing it.
_CORNER_DECADES = 12.0

#: Headroom, in natural log units, kept between the largest root the search may
#: reach and the point where a product of a whole cascade of them overflows.
#: :func:`scipy.signal.bilinear_zpk` forms exactly that product to carry the
#: gain across the transform, so the bound has to hold for it and not just for
#: the individual roots. One decade of margin, which at every rate this library
#: is used at is slack: the physical limit above binds first, and this one is
#: what keeps the guarantee from depending on that.
_OVERFLOW_MARGIN = np.log(10.0)

#: Floor under the residual magnitude in the Lawson update, so a point that
#: happens to sit on the target does not zero its own weight for good.
_LAWSON_RESIDUAL_FLOOR = 1e-14

#: Peak deviation, in decibels, at which a fit is treated as having failed
#: rather than as having come out loose. It is read off the standards, not
#: tuned: 0.05 dB is the finest quantum any of these tables is printed to and
#: +/-0.7 dB is the narrowest tolerance any of these curves is graded against,
#: so a design four quanta from the printed curve has not converged onto it.
#: What it has usually done is flatten -- pushed every corner outside the grid
#: and returned a constant, whose residual is then the printed curve's own
#: excursion, tens of decibels. :func:`fit_prototype` treats that as a start it
#: should not have used and works through :func:`_spare_placements`.
#:
#: The threshold decides how much work is done, not what is returned. Above it,
#: the fits that trip it are not marginal: the ones measured at 0.24 to 1.00 dB
#: come back between 0.00002 and 0.0005 dB from another placement, three to
#: four orders of magnitude better, which is what a threshold separating
#: "converged" from "gave up" should look like. Below it, the loosest first fit
#: at any of the thirteen standard rates is 0.060 dB (BS.468-4 at 48 kHz),
#: three and a half times inside, and the next loosest is seventeen times
#: inside; a BLAS reduction reassociating its last bits moves that 0.060 by
#: about 5 %, and it would have to move by 235 % to change an answer there.
#:
#: That separation is a property of the standard rates, not of the whole axis.
#: Swept at 500 Hz steps, BS.468-4's first fit is 0.28 dB at 46 500 Hz, so 3 %
#: away from a rate the library is graded at the threshold is straddled and a
#: perturbation of that size could flip which design comes back. What it flips
#: between is bounded: 0.28 dB against 0.033 dB, both far inside every mask in
#: the corpus. So the cost of being on the wrong side here is a slower design
#: and a looser but still conforming filter, never a wrong one.
_FIT_FAILED_DB = 0.2


@dataclass(frozen=True)
class _Layout:
    """How many factors of each kind the parameter vector carries.

    :ivar zeros_at_origin: Zeros pinned at :math:`s = 0`, one per zero the
        printed prototype has there. They carry no parameters and become exact
        zeros at ``z = 1`` under the bilinear transform, so the realised filter
        blocks dc exactly, as the analog network does.
    :ivar num_quadratic: Free quadratic numerator factors.
    :ivar num_linear: Free linear numerator factors (0 or 1).
    :ivar den_quadratic: Quadratic denominator factors.
    :ivar den_linear: Linear denominator factors (0 or 1).
    """

    zeros_at_origin: int
    num_quadratic: int
    num_linear: int
    den_quadratic: int
    den_linear: int

    def blocks(self, theta: np.ndarray) -> tuple[np.ndarray, ...]:
        """Split *theta* into (numerator quads, num linear, den quads, den linear)."""
        cut1 = 2 * self.num_quadratic
        cut2 = cut1 + self.num_linear
        cut3 = cut2 + 2 * self.den_quadratic
        return (
            theta[:cut1].reshape(self.num_quadratic, 2),
            theta[cut1:cut2],
            theta[cut2:cut3].reshape(self.den_quadratic, 2),
            theta[cut3:],
        )


def _corner_bounds(
    layout: _Layout, span: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    r"""Lower and upper limits on *theta*, entry by entry, in log units.

    The limits are read off the fit grid rather than pulled from the air: a
    corner may wander :data:`_CORNER_DECADES` decades below the grid's lowest
    frequency or that far above its highest, and no further. The two
    coefficients of a quadratic are bounded differently because they are not
    the same quantity: :math:`b_0` is the square of a corner frequency and
    :math:`b_1` is twice one, so their logs get twice and one times the log of
    the frequency limits.

    The ceiling is whichever is lower: that physical limit, or the largest
    corner whose whole cascade of roots can still be multiplied together in
    double precision. The second one is what makes "the design is finite" a
    proof rather than a measurement -- it holds at any sampling rate and for
    any number of sections, without anyone having swept for it. The factor of
    two in it is not slack: a quadratic whose roots have split apart carries
    them both in :math:`b_1`, so the largest *root* a corner limit of
    :math:`c` admits is :math:`2c`, not :math:`c`.

    :param layout: How many factors of each kind *theta* carries.
    :param span: ``(lowest, highest)`` grid frequency in rad/s.
    :return: ``(lower, upper)``, both shaped like *theta*.
    """
    cascade = max(
        layout.zeros_at_origin + 2 * layout.num_quadratic + layout.num_linear,
        2 * layout.den_quadratic + layout.den_linear,
        1,
    )
    largest_root = (np.log(np.finfo(np.float64).max) - _OVERFLOW_MARGIN) / cascade
    decades = _CORNER_DECADES * np.log(10.0)
    lowest = np.log(span[0]) - decades
    highest = min(np.log(span[1]) + decades, largest_root - np.log(2.0))
    quadratic = np.array(
        [[lowest, 2.0 * lowest], [highest + np.log(2.0), 2.0 * highest]]
    )
    linear = np.array([[lowest], [highest]])
    pieces = (
        (layout.num_quadratic, quadratic),
        (layout.num_linear, linear),
        (layout.den_quadratic, quadratic),
        (layout.den_linear, linear),
    )
    limits = np.hstack([np.tile(kind, count) for count, kind in pieces if count])
    return limits[0], limits[1]


def _anchor_hz(fs: float, band: tuple[float, float], f_ref: float) -> float:
    r"""The frequency at which the design is pinned to the printed curve.

    The standard's own reference frequency whenever the rate can carry it,
    which is what makes the realised response exactly 0 dB where the standard
    prints a zero tolerance.

    Below ``2 * f_ref`` the rate cannot carry it, and the bilinear warp
    :math:`2 f_s \tan(\pi f / f_s)` does not merely fail there, it answers:
    it leaves the positive axis at the Nyquist frequency and comes back as the
    alias, or as :math:`\Omega \simeq 0` whenever ``fs`` divides ``f_ref``.
    Anchoring on that answer is what used to hand the search an objective with
    no solution -- at ``fs = 1000`` the B weighting's residual could not fall
    below 424 dB, because the fit was being asked to read 0 dB at dc, where
    that curve has three zeros -- and the parameters ran to their clamps
    trying. So below that the anchor moves to the geometric mean of the fit
    band: the midpoint of the interval on the log-frequency axis the standard
    states the curve over, always representable, always inside the band, and as
    far as a single frequency can be from both the steep skirt at the bottom
    and the flattening at the Nyquist frequency at the top. The realised
    response is then pinned to the printed curve's own
    level *there* rather than to 0 dB, so the absolute gain is still the
    standard's; see :func:`design_sos`.

    :param fs: Sampling rate in Hz.
    :param band: The fit band in Hz, already clipped to the Nyquist frequency.
    :param f_ref: The standard's reference frequency in Hz.
    :return: The anchor frequency in Hz, strictly inside *band*.
    """
    if 2.0 * f_ref < fs:
        return f_ref
    low, high = band
    return float(np.sqrt(low * high))


def _quadratic_group(
    block: np.ndarray, omega: np.ndarray, omega2: np.ndarray, grad: bool
) -> tuple[np.ndarray, np.ndarray]:
    r"""``sum log|s^2 + b1 s + b0|^2`` at ``s = j omega``, and its gradient.

    With :math:`b_1 = e^{\theta_1}` and :math:`b_0 = e^{\theta_0}` the squared
    magnitude is :math:`(b_0 - \Omega^2)^2 + (b_1 \Omega)^2`, and the
    derivatives with respect to the *logs* of the coefficients are
    :math:`2 b_1^2 \Omega^2 / D` and :math:`2 b_0 (b_0 - \Omega^2) / D`.
    """
    coefficients = np.exp(block)
    b1 = coefficients[:, 0][None, :]
    b0 = coefficients[:, 1][None, :]
    offset = b0 - omega2[:, None]
    denominator = offset * offset + (b1 * omega[:, None]) ** 2
    total = np.log(denominator).sum(axis=1)
    if not grad:
        return total, np.zeros((omega.size, 0))
    d_b1 = 2.0 * b1 * b1 * omega2[:, None] / denominator
    d_b0 = 2.0 * b0 * offset / denominator
    return total, np.stack([d_b1, d_b0], axis=2).reshape(omega.size, -1)


def _linear_group(
    block: np.ndarray, omega2: np.ndarray, grad: bool
) -> tuple[np.ndarray, np.ndarray]:
    """``sum log|s + b1|^2`` at ``s = j omega``, and its gradient in ``log b1``."""
    b1 = np.exp(block)[None, :]
    denominator = omega2[:, None] + b1 * b1
    total = np.log(denominator).sum(axis=1)
    if not grad:
        return total, np.zeros((omega2.size, 0))
    return total, 2.0 * b1 * b1 / denominator


def _log_mag2(
    theta: np.ndarray, layout: _Layout, omega: np.ndarray, grad: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    r""":math:`\log |\tilde{H}(j\Omega)|^2` up to a constant, and its Jacobian.

    The overall gain is left out: the caller normalises the realised filter at
    the standard's reference frequency, so only the shape is fitted and the
    residual at that frequency is identically zero by construction rather than
    by luck.
    """
    omega2 = omega * omega
    num_quad, num_lin, den_quad, den_lin = layout.blocks(theta)
    total = layout.zeros_at_origin * np.log(omega2)
    columns = []
    groups = (
        (num_quad, 1.0, True),
        (num_lin, 1.0, False),
        (den_quad, -1.0, True),
        (den_lin, -1.0, False),
    )
    for block, sign, quadratic in groups:
        if block.size == 0:
            continue
        if quadratic:
            value, jacobian = _quadratic_group(block, omega, omega2, grad)
        else:
            value, jacobian = _linear_group(block, omega2, grad)
        total = total + sign * value
        columns.append(sign * jacobian)
    if not grad:
        return total, np.zeros((omega.size, 0))
    return total, np.hstack(columns)


def _residual_db(
    theta: np.ndarray,
    layout: _Layout,
    omega: np.ndarray,
    omega_ref: np.ndarray,
    target_db: np.ndarray,
) -> np.ndarray:
    """Deviation of the trial prototype from the printed one, in decibels."""
    value, _ = _log_mag2(theta, layout, omega)
    at_ref, _ = _log_mag2(theta, layout, omega_ref)
    return np.asarray(_DB_PER_LOG * (value - at_ref[0]) - target_db)


def _damped_step(
    jacobian: np.ndarray, residual: np.ndarray, weights: np.ndarray, damping: float
) -> np.ndarray:
    """One Levenberg-Marquardt step of the weighted normal equations.

    The normal equations are scaled by the square roots of their own diagonal
    before the damping is added, so the damping means the same thing for a
    corner frequency at 20 Hz and one at 20 kHz.
    """
    weighted = jacobian * weights[:, None]
    hessian = jacobian.T @ weighted
    gradient = -(weighted.T @ residual)
    diagonal = np.sqrt(np.diag(hessian))
    diagonal = np.where(diagonal > 0.0, diagonal, 1.0)
    scaled = hessian / np.outer(diagonal, diagonal)
    scaled = scaled + damping * np.eye(scaled.shape[0])
    return np.asarray(np.linalg.solve(scaled, gradient / diagonal) / diagonal)


def _lm_round(
    theta: np.ndarray,
    layout: _Layout,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """``_LM_STEPS`` Levenberg-Marquardt steps at fixed Lawson weights."""
    omega, omega_ref, target_db = grid
    damping = _LM_INITIAL_DAMPING
    residual = _residual_db(theta, layout, omega, omega_ref, target_db)
    cost = float(weights @ (residual * residual))
    for _ in range(_LM_STEPS):
        _, jacobian = _log_mag2(theta, layout, omega, grad=True)
        _, at_ref = _log_mag2(theta, layout, omega_ref, grad=True)
        step = _damped_step(
            _DB_PER_LOG * (jacobian - at_ref[0][None, :]), residual, weights, damping
        )
        trial = np.clip(theta + step, *bounds)
        trial_residual = _residual_db(trial, layout, omega, omega_ref, target_db)
        trial_cost = float(weights @ (trial_residual * trial_residual))
        if trial_cost < cost * (1.0 - _LM_ACCEPT_MARGIN):
            theta, residual, cost = trial, trial_residual, trial_cost
            damping = max(damping * _LM_DAMPING_DOWN, _LM_DAMPING_FLOOR)
        else:
            damping = min(damping * _LM_DAMPING_UP, _LM_DAMPING_CEILING)
    return theta


def _lawson_fit(
    theta: np.ndarray,
    layout: _Layout,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Drive the least-squares fit toward minimax, and keep the best round.

    Lawson's algorithm is not monotone -- a round can overshoot and come back
    worse -- so the round with the smallest peak residual is what is returned,
    which makes the routine's output a minimum over a fixed set of candidates
    rather than whatever the last round happened to leave behind.

    What that buys, exactly: the routine has no random state, reads no clock,
    and does the same arithmetic on every input, so two runs in one process,
    two processes, or two thread counts give bit-identical coefficients. It
    does *not* claim bit-identical results across BLAS builds, because
    ``numpy.linalg.solve`` does not; what is claimed there is the pinned
    residual bound of the test suite, and the accept margin of
    :data:`_LM_ACCEPT_MARGIN` keeps a step whose gain is at the level of
    rounding from being accepted on one machine and rejected on another.
    """
    omega, omega_ref, target_db = grid
    weights = np.ones(omega.size)
    best = theta
    best_peak = np.inf
    for _ in range(_LAWSON_ROUNDS):
        theta = _lm_round(theta, layout, grid, weights, bounds)
        residual = _residual_db(theta, layout, omega, omega_ref, target_db)
        peak = float(np.max(np.abs(residual)))
        if peak < best_peak:
            best_peak, best = peak, theta
        magnitude = np.maximum(np.abs(residual), _LAWSON_RESIDUAL_FLOOR)
        weights = weights * magnitude**_LAWSON_EXPONENT
        weights = weights / np.max(weights)
    return best


def _stable_factors(
    roots: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[float]]]:
    """Group left-half-plane roots into quadratic and linear factors.

    Conjugate pairs stay together, which is what keeps the fitted factors
    comparable with the printed ones; the remaining real roots are paired in
    order of magnitude, and an odd one out becomes a linear factor.
    """
    values = [complex(v) for v in roots]
    quadratic: list[tuple[float, float]] = []
    taken = [False] * len(values)
    for index, value in enumerate(values):
        if taken[index] or value.imag <= 0.0:
            continue
        # Distances materialised rather than closed over: a lambda here would
        # capture the loop variable, and its late binding is a hazard even
        # where, as here, it is consumed before the next iteration.
        wanted = value.conjugate()
        distance = [abs(other - wanted) for other in values]
        partner = min(
            (k for k in range(len(values)) if not taken[k] and k != index),
            key=distance.__getitem__,
        )
        taken[index] = taken[partner] = True
        quadratic.append((-2.0 * value.real, abs(value) ** 2))
    real = sorted(abs(values[k].real) for k in range(len(values)) if not taken[k])
    pairs, odd = divmod(len(real), 2)
    for index in range(pairs):
        first, second = real[2 * index], real[2 * index + 1]
        quadratic.append((first + second, first * second))
    return quadratic, [(real[-1],)] if odd else []


def _padded_factors(
    roots: np.ndarray,
    degree: int,
    placement: tuple[float, float],
    pinned: int = 0,
) -> tuple[list[tuple[float, ...]], int, int]:
    """The prototype's own factors, padded up to *degree* with spare ones.

    :param placement: ``(first, ratio)``. Spare factor *k* is a repeated real
        root at ``first * ratio**k`` rad/s. ``ratio = 1`` stacks them all on
        one corner, which is what the plain bilinear design does; that is the
        first placement tried and the only one at any rate this library is used
        at, but it is not the only one available, because stacking has a cost
        the closed-form design never pays. Identical factors give the search
        identical Jacobian columns, and identical columns move together under
        every step it can take: four stacked spares are one free factor wearing
        four hats, and the symmetry has nothing to break it. A ratio away from
        one hands the search four factors instead. See
        :func:`_spare_placements`.
    """
    quadratic, linear = _stable_factors(roots)
    spare = degree - pinned - 2 * len(quadratic) - len(linear)
    if spare < 0:
        msg = (
            "'sections' is too small for this prototype: "
            f"degree {degree} cannot hold {pinned + 2 * len(quadratic) + len(linear)}."
        )
        raise ValueError(msg)
    first, ratio = placement
    corners = [first * ratio**k for k in range(spare // 2 + spare % 2)]
    quadratic = [*quadratic, *[(2.0 * c, c * c) for c in corners[: spare // 2]]]
    linear = [*linear, *[(c,) for c in corners[spare // 2 :]]]
    return [*quadratic, *linear], len(quadratic), len(linear)


def _start_parameters(
    zeros: np.ndarray,
    poles: np.ndarray,
    degree: int,
    placement: tuple[float, float],
) -> tuple[np.ndarray, _Layout]:
    """Parameter vector and layout of the prototype the search starts from.

    :param placement: Where the spare factors start; see
        :func:`_spare_placements`.
    """
    away_from_origin = np.abs(np.asarray(zeros)) > 0.0
    at_origin = int(np.count_nonzero(~away_from_origin))
    finite = np.asarray(zeros)[away_from_origin]
    num, num_quad, num_lin = _padded_factors(finite, degree, placement, at_origin)
    den, den_quad, den_lin = _padded_factors(np.asarray(poles), degree, placement)
    theta = np.log(np.concatenate([np.asarray(f, dtype=float) for f in num + den]))
    return theta, _Layout(at_origin, num_quad, num_lin, den_quad, den_lin)


def _spare_placements(
    fs: float, span: tuple[float, float]
) -> tuple[tuple[float, float], ...]:
    r"""``(first, ratio)`` starts for the spare factors, in the order tried.

    A prototype with fewer finite zeros than poles -- every curve here -- is
    padded up to the requested degree, and where those spare factors start
    decides what the search has to undo. ``(2 * fs, 1)`` is the plain bilinear
    design's own answer: each spare is a root at ``2 * fs`` whose bilinear
    image is a root at :math:`z = 0`, a pure delay, so the *digital* filter
    that start describes is exactly the closed-form design. It is the first
    placement tried, and :func:`fit_prototype` reaches for the rest only when
    it does not land close enough. That is the usual case far below the audio
    band, and it also happens at scattered rates inside it: over every rate
    from 2 Hz to 200 kHz, 52 at or above 2 kHz take the retry and five of those
    are at or above 8 kHz. None of the thirteen standard rates does, and the
    retry always improves the fit where it fires.

    It fails at rates far below the audio band, for two reasons that compound.
    A root at ``2 * fs`` is neutral in the digital domain but not in the one
    the residual is measured in: the fit compares analog responses at the
    warped frequency, where ``2 * fs`` sits inside the grid and tilts the start
    by up to :math:`160\,\mathrm{dB}` per decade. At audio rates that tilt is
    small against the curve's own and the search absorbs it in the first
    steps; three decades lower the start is hundreds of decibels out, and the
    search answers a start like that by flattening -- pushing every corner
    outside the grid, returning a constant, and stopping there, because a
    constant has no gradient left to leave by. And stacking makes the recovery
    harder than it needs to be: identical factors have identical Jacobian
    columns and move together forever, so the AU prototype's four spare pairs
    are one free factor, not four.

    The fallbacks vary both knobs and nothing else -- the corner the first
    spare sits on, and the ratio between successive spares -- because those are
    the two things stacking at ``2 * fs`` gets wrong. Anchoring them on the
    grid's top edge rather than on the rate is what makes the second column
    worth trying: the top of the grid is where the warp does all its work, so
    it is where a spare factor has the most shape to lend.

    :param fs: Sampling rate in Hz.
    :param span: ``(lowest, highest)`` grid frequency in rad/s.
    :return: The placements to try, in order.
    """
    ratios = (1.0, 10.0, 0.1, 100.0)
    return tuple((first, ratio) for first in (2.0 * fs, span[1]) for ratio in ratios)


def _factor_roots(quadratic: np.ndarray, linear: np.ndarray) -> list[complex]:
    """Roots of the fitted factors, in rad/s, all strictly in the left half plane."""
    roots: list[complex] = []
    for b1, b0 in np.exp(quadratic):
        discriminant = b1 * b1 - 4.0 * b0
        radius = float(np.sqrt(abs(discriminant)))
        if discriminant >= 0.0:
            roots += [complex((-b1 + radius) / 2.0), complex((-b1 - radius) / 2.0)]
        else:
            roots += [
                complex(-b1 / 2.0, radius / 2.0),
                complex(-b1 / 2.0, -radius / 2.0),
            ]
    roots += [complex(-value) for value in np.exp(linear)]
    return roots


def _fitted_zpk(theta: np.ndarray, layout: _Layout) -> tuple[np.ndarray, np.ndarray]:
    """Zeros and poles, in rad/s, of the fitted analog prototype."""
    num_quad, num_lin, den_quad, den_lin = layout.blocks(theta)
    zeros = [complex(0.0)] * layout.zeros_at_origin + _factor_roots(num_quad, num_lin)
    return np.array(zeros), np.array(_factor_roots(den_quad, den_lin))


def _warped(frequencies: np.ndarray, fs: float) -> np.ndarray:
    r"""Analog frequencies the bilinear transform sends *frequencies* to, in rad/s."""
    return 2.0 * fs * np.tan(np.pi * np.asarray(frequencies, dtype=float) / fs)


def _analog_db(
    zeros: np.ndarray, poles: np.ndarray, frequencies: np.ndarray, f_ref: float
) -> np.ndarray:
    """The prototype's magnitude in dB, relative to its value at *f_ref*."""
    s = 2j * np.pi * np.asarray(frequencies, dtype=float)
    s_ref = 2j * np.pi * f_ref

    def magnitude(argument: np.ndarray) -> np.ndarray:
        numerator = (
            np.prod(argument[:, None] - zeros[None, :], axis=1)
            if zeros.size
            else np.ones_like(argument)
        )
        return np.abs(numerator / np.prod(argument[:, None] - poles[None, :], axis=1))

    reference = magnitude(np.array([s_ref]))[0]
    return np.asarray(20.0 * np.log10(magnitude(s) / reference))


def _fit_from(
    zeros: np.ndarray,
    poles: np.ndarray,
    degree: int,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    span: tuple[float, float],
    placement: tuple[float, float],
) -> tuple[float, np.ndarray, _Layout]:
    """One whole fit, from the start *placement* describes.

    :return: ``(peak residual in dB, parameters, layout)``. The peak is what
        :func:`fit_prototype` compares attempts on, and it is measured on the
        same grid the fit minimised over, so it says how well the search did
        and not how well the design will be graded.
    """
    theta, layout = _start_parameters(zeros, poles, degree, placement)
    bounds = _corner_bounds(layout, span)
    fitted = _lawson_fit(np.clip(theta, *bounds), layout, grid, bounds)
    residual = _residual_db(fitted, layout, grid[0], grid[1], grid[2])
    return float(np.max(np.abs(residual))), fitted, layout


def fit_prototype(
    zeros: np.ndarray,
    poles: np.ndarray,
    fs: float,
    band: tuple[float, float],
    sections: int,
    f_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Analog prototype whose bilinear image tracks *zeros*/*poles* at *fs*.

    :param zeros: Zeros of the printed prototype, in rad/s. Zeros at the origin
        are held there.
    :param poles: Poles of the printed prototype, in rad/s.
    :param fs: Sampling rate in Hz.
    :param band: Frequencies in Hz between which the response is fitted. The
        response outside it is not claimed: see
        :func:`phonometry.filters.weighting._fit_band`.
    :param sections: Biquad sections, i.e. half the degree of the realised
        filter.
    :param f_ref: The standard's reference frequency. The residual is measured
        relative to the response at :func:`_anchor_hz`, which is *f_ref* itself
        at every rate that can carry it.
    :return: ``(zeros, poles)`` of the fitted prototype in rad/s, ready for
        :func:`scipy.signal.bilinear_zpk`.
    :raises ValueError: if *band* is empty or *sections* cannot hold the
        prototype's own degree.
    """
    low, high = band
    # Spelled out rather than as ``not low < high``: the two differ only on
    # NaN, which the negated comparison happens to refuse, and a guard should
    # say what it refuses instead of relying on that.
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        msg = (
            f"'band' must span a positive frequency interval; got ({low:g}, {high:g}) "
            "Hz, which the sampling rate has closed."
        )
        raise ValueError(msg)
    anchor = _anchor_hz(fs, band, f_ref)
    edges = _warped(np.array([low, high]), fs)
    span = (float(edges[0]), float(edges[1]))
    omega = np.geomspace(span[0], span[1], _GRID_POINTS)
    omega_ref = _warped(np.array([anchor]), fs)
    frequencies = (fs / np.pi) * np.arctan(omega / (2.0 * fs))
    target_db = _analog_db(np.asarray(zeros), np.asarray(poles), frequencies, anchor)
    grid = (omega, omega_ref, target_db)
    placements = _spare_placements(fs, span)
    best = _fit_from(zeros, poles, 2 * sections, grid, span, placements[0])
    if best[0] > _FIT_FAILED_DB:
        # The closed-form design's own start did not produce a filter, so the
        # question is no longer which start is quickest but which one works:
        # every remaining placement is tried and the best is kept.
        for placement in placements[1:]:
            attempt = _fit_from(zeros, poles, 2 * sections, grid, span, placement)
            if attempt[0] < best[0]:
                best = attempt
    return _fitted_zpk(best[1], best[2])


def design_sos(
    zeros: np.ndarray,
    poles: np.ndarray,
    fs: float,
    band: tuple[float, float],
    sections: int,
    f_ref: float,
) -> np.ndarray:
    """Second-order sections realising the prototype at *fs*, 0 dB at *f_ref*.

    The gain is folded into the first section so that the realised response
    reads the printed curve's own level at :func:`_anchor_hz`. At every rate
    that can carry *f_ref* the anchor is *f_ref*, that level is exactly 0 dB,
    and the response is exactly 0 dB where the standard prints a zero
    tolerance. Below ``2 * f_ref`` no sampled signal carries that frequency, so
    the anchor moves into the band and the printed level there -- which is not
    0 dB -- is what the cascade is pinned to. The absolute gain is the
    standard's either way: a 200 Hz tone reads the same number of decibels at
    ``fs = 903`` as it does at 48 kHz.

    Both the pin and the fit's own residual use the same anchor, so the
    realised response inherits the fit's accuracy at the point it is pinned at
    instead of being pinned somewhere the fit never looked.

    :return: The cascade.
    """
    fitted_zeros, fitted_poles = fit_prototype(zeros, poles, fs, band, sections, f_ref)
    anchor = _anchor_hz(fs, band, f_ref)
    digital = signal.bilinear_zpk(fitted_zeros, fitted_poles, 1.0, fs)
    sos = signal.zpk2sos(*digital)
    _, at_anchor = signal.sosfreqz(sos, worN=np.array([anchor]), fs=fs)
    printed = _analog_db(
        np.asarray(zeros), np.asarray(poles), np.array([anchor]), f_ref
    )
    sos[0, :3] /= abs(at_anchor[0]) / 10.0 ** (printed[0] / 20.0)
    return np.asarray(sos, dtype=np.float64)


def fitted_prototype_hz(
    zeros: np.ndarray,
    poles: np.ndarray,
    fs: float,
    band: tuple[float, float],
    sections: int,
    f_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The fitted prototype's zeros and poles in Hz, for reading beside a table.

    The point of fitting a *prototype* rather than digital coefficients is that
    the answer stays comparable with the standard: the poles come back near the
    printed corner frequencies, moved by the warp the fit is undoing.
    """
    fitted_zeros, fitted_poles = fit_prototype(zeros, poles, fs, band, sections, f_ref)
    return fitted_zeros / (2.0 * np.pi), fitted_poles / (2.0 * np.pi)
