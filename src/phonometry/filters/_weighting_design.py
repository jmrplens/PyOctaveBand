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
* **The work is fixed, and so is the arithmetic.** :data:`_LAWSON_ROUNDS`
  reweighting rounds of :data:`_LM_STEPS` Levenberg-Marquardt steps each, on a
  :data:`_GRID_POINTS`-point grid: no convergence test, no random state, no
  clock. That makes the sequence of operations the same everywhere, which is
  not the same thing as making the answer the same, and for a while this
  routine did not make the answer the same. The reductions inside the fit --
  the normal equations, their right-hand side, the cost the accept test
  compares -- were library calls, and a BLAS reassociates a reduction to suit
  the vector registers it finds at run time. Levenberg-Marquardt in a shallow
  valley amplifies whatever that costs in the last bit: measured on the A
  weighting at 48 kHz, two OpenBLAS kernel sets agreed on every accept
  decision for thirty-nine consecutive steps while the costs they were
  comparing drifted from 3e-13 to 1e-4 apart, and the two runs then landed on
  different filters -- 5e-5 apart in the leading coefficient, 0.002 dB apart
  in response, and 0.0225 pt apart in a plotted curve. So the reductions are
  done here instead, by :func:`_ordered_sum` and :func:`_solve_dense`, in an
  order this module fixes rather than one a library picks, and the two places
  the path used to hand a *complex* expression to numpy -- the printed curve
  the fit aims at, and the one read-back that sets the shipped gain -- are
  written out in real arithmetic by :func:`_log_distance2` and
  :func:`_cascade_magnitude`, because complex multiplication and complex
  magnitude are dispatched on what the CPU can do in exactly the same way a
  BLAS reduction is. Every design in the corpus is then bit-identical under
  every OpenBLAS kernel set that can be selected here, over all 91 (curve,
  rate) pairs, and under every thread count and hash seed.

  What that does **not** yet cover, and this is measured rather than assumed:
  a machine whose numpy dispatches to AVX512. Continuous integration prints
  two different digests for the A weighting at 48 kHz on one such runner,
  depending only on whether numpy's dispatchable kernels are enabled, and the
  disabled one is byte for byte what a machine without AVX512 produces. So the
  claim above is bounded by the kernels a developer machine can reach, and the
  AVX512 case is open. It is what closed the last two library-ordered
  reductions in :func:`_quadratic_group` and :func:`_linear_group`, which had
  been left outside :func:`_ordered_sum`; whether that was the whole of it is
  a question only that runner can answer.

  The one thing that is not fixed is how many *starts*
  get that budget: a fit that comes back more than :data:`_FIT_FAILED_DB` from
  the printed curve has not produced a filter at all, and the remaining
  entries of :func:`_spare_placements` are tried before the best is kept. That
  branch is on the achieved residual, so it is still deterministic, and no
  rate this library is used at comes near it -- the loosest first fit in the
  corpus is 3.2 times inside. See :func:`_lawson_fit` for what
  reproducibility that does and does not buy, and :func:`fit_prototype` for
  why the cheaper rule is also the better one.

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
#: the work, about 310 ms per design instead of 130 -- moves the worst
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
#: amount, so a step whose gain is at the level of floating-point rounding is
#: rejected rather than taken and the iteration stops churning once it has
#: nothing left to gain. That is all it is worth. It was introduced to keep
#: the sequence of accepted steps from depending on the last bits of a BLAS
#: reduction, and measurement says it cannot do that: across two OpenBLAS
#: kernel sets the accept decisions agreed for thirty-nine consecutive steps
#: while the costs being compared drifted four orders of magnitude apart, so
#: by the time a decision differed the two runs had long since parted. A test
#: at any tolerance would have behaved the same way, because what it is
#: comparing has already moved. Machine independence comes from
#: :func:`_ordered_sum` and :func:`_solve_dense`, which leave no reduction
#: order for a library to choose in the first place.
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
#: the fits that trip it are not marginal: at the ten rates below 2 kHz, 32 of
#: the 70 designs trip it, with first fits from 0.27 dB to 107 dB, and a later
#: placement improves every one of them and lands all of them inside the
#: 0.44 dB the suite allows at these rates. The improvement runs from a factor
#: of 300 000 (AU at 1 kHz, 89 dB to 0.0003) down to 1.001 at the two G rows,
#: where what is left is the zero slope a real-coefficient filter has at the
#: Nyquist frequency and not where the search stopped. Below the threshold,
#: the loosest first fit at any of the
#: thirteen standard rates is 0.062 dB (BS.468-4 at 48 kHz), 3.2 times inside,
#: and the next loosest is BS.468-4 at 44.1 kHz at 3.9 times; perturbing every
#: solve in the fit by a relative 1e-12 moves that 0.062 by at most 3 %, and it
#: would have to move by 221 % to change an answer there.
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
    total = _ordered_sum(np.log(denominator).T)
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
    total = _ordered_sum(np.log(denominator).T)
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


def _ordered_sum(terms: np.ndarray) -> np.ndarray:
    r"""Add *terms* along their leading axis in an order fixed here.

    This module's machine independence rests on this function and on
    :func:`_solve_dense`, so it is worth saying exactly what it buys and why
    nothing in the library can supply it.

    IEEE 754 makes a single addition of two doubles exactly rounded: given the
    two operands, the result is the same bit pattern on every conforming
    machine. What it does not make reproducible is a *reduction*, because
    addition is not associative in floating point and a reduction is free to
    choose its order. A BLAS ``ddot`` chooses one that suits the vector
    registers it found at run time, so ``a @ b`` returns a different last bit
    on a kernel with four lanes than on one with two, and
    :func:`numpy.linalg.solve` inherits the same freedom from LAPACK. That is
    not a defect in either: they are permitted to reassociate, and they are
    faster for it.

    So the reduction is done here instead, by repeatedly folding the leading
    axis in half and adding the halves *elementwise*; an odd count leaves one
    term over, which goes into the first row of the fold. Every operation is
    then an elementwise addition of two arrays, which IEEE 754 pins entry by
    entry no matter how many of them a vector unit does at once, and the shape
    of the fold depends only on ``terms.shape[0]``. The result is a function of
    the operands alone: the same bits under any BLAS kernel, any thread count
    and any lane width.

    Folding in half rather than accumulating left to right is also the more
    accurate of the two. Pairwise summation carries
    :math:`O(\varepsilon \log n)` of rounding against the
    :math:`O(\varepsilon n)` of a running total, so what replaces ``ddot``
    here is not a slower, less exact reduction bought for reproducibility; on
    a 257-point grid it is the better one.

    :param terms: The addends, stacked along axis 0.
    :return: Their sum, shaped like a single term.
    """
    block = np.asarray(terms, dtype=np.float64)
    if block.shape[0] < 1:
        return np.zeros(block.shape[1:])
    while block.shape[0] > 1:
        half, odd = divmod(block.shape[0], 2)
        folded = block[:half] + block[half : 2 * half]
        if odd:
            folded[0] += block[2 * half]
        block = folded
    return np.asarray(block[0])


def _solve_dense(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Solve ``matrix @ x == vector`` by Gaussian elimination done here.

    The system is the damped normal equations: one row and column per fitted
    coefficient, so a dozen or two at the sizes this module builds. LAPACK
    exists for systems three orders of magnitude larger than that, and at this
    one its only contribution is to pick a reduction order this module cannot
    see; see :func:`_ordered_sum`. Elimination with partial pivoting is the
    same algorithm LAPACK's own ``dgesv`` runs, written out so that every
    reduction in it is one of ours.

    The one reduction elimination needs that is not elementwise is the inner
    product of the back substitution, and that one is :func:`math.fsum`
    rather than :func:`_ordered_sum`: over a handful of terms it is quicker
    than folding an array, and it is the stronger guarantee of the two, since
    it returns the correctly rounded sum and so does not depend on the order
    of its own addends either.

    A pivot that is not positive in magnitude is stepped over rather than
    divided by, and the unknown it would have determined is left at zero.
    Spelled that way round, and not as a comparison with zero, because the
    two differ on NaN: a Hessian that has gone to NaN must reach the caller as
    a step of zeros, which the accept test then rejects, and not as a division
    that quietly propagates it into the parameters.

    :param matrix: The square coefficient matrix.
    :param vector: The right-hand side.
    :return: The solution, shaped like *vector*.
    """
    rows = np.array(matrix, dtype=np.float64)
    right = np.array(vector, dtype=np.float64)
    order = right.size
    for column in range(order - 1):
        pivot = column + int(np.argmax(np.abs(rows[column:, column])))
        if pivot != column:
            rows[[column, pivot]] = rows[[pivot, column]]
            right[[column, pivot]] = right[[pivot, column]]
        head = rows[column, column]
        if not abs(head) > 0.0:
            continue
        factors = rows[column + 1 :, column] / head
        rows[column + 1 :, column + 1 :] -= (
            factors[:, None] * rows[column, column + 1 :]
        )
        right[column + 1 :] -= factors * right[column]
    solution = np.zeros(order)
    for column in range(order - 1, -1, -1):
        head = rows[column, column]
        if not abs(head) > 0.0:
            continue
        behind = math.fsum(rows[column, column + 1 :] * solution[column + 1 :])
        solution[column] = (right[column] - behind) / head
    return solution


def _damped_step(
    jacobian: np.ndarray, residual: np.ndarray, weights: np.ndarray, damping: float
) -> np.ndarray:
    """One Levenberg-Marquardt step of the weighted normal equations.

    The normal equations are scaled by the square roots of their own diagonal
    before the damping is added, so the damping means the same thing for a
    corner frequency at 20 Hz and one at 20 kHz.

    The Hessian is accumulated from the products themselves rather than as
    ``jacobian.T @ (jacobian * weights)``, which is the same sum with the
    reduction handed to a library. Only its upper triangle is formed and the
    lower one is mirrored from it, which halves the work and makes the matrix
    exactly symmetric rather than symmetric to within a rounding; the
    elimination that follows reads both halves.
    """
    rows, columns = np.triu_indices(jacobian.shape[1])
    products = jacobian[:, rows] * jacobian[:, columns]
    products *= weights[:, None]
    upper = _ordered_sum(products)
    hessian = np.empty((jacobian.shape[1], jacobian.shape[1]))
    hessian[rows, columns] = upper
    hessian[columns, rows] = upper
    gradient = -_ordered_sum(jacobian * (weights * residual)[:, None])
    diagonal = np.sqrt(np.diag(hessian))
    diagonal = np.where(diagonal > 0.0, diagonal, 1.0)
    scaled = hessian / np.outer(diagonal, diagonal)
    scaled = scaled + damping * np.eye(scaled.shape[0])
    return np.asarray(_solve_dense(scaled, gradient / diagonal) / diagonal)


def _weighted_cost(weights: np.ndarray, residual: np.ndarray) -> float:
    """The weighted sum of squares the search descends.

    Named rather than inlined because it is what the accept test compares, so
    the two costs it puts either side of an inequality have to be reduced the
    same way and by :func:`_ordered_sum` rather than by a ``ddot``.
    """
    return float(_ordered_sum(weights * residual * residual))


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
    cost = _weighted_cost(weights, residual)
    for _ in range(_LM_STEPS):
        _, jacobian = _log_mag2(theta, layout, omega, grad=True)
        _, at_ref = _log_mag2(theta, layout, omega_ref, grad=True)
        step = _damped_step(
            _DB_PER_LOG * (jacobian - at_ref[0][None, :]), residual, weights, damping
        )
        trial = np.clip(theta + step, *bounds)
        trial_residual = _residual_db(trial, layout, omega, omega_ref, target_db)
        trial_cost = _weighted_cost(weights, trial_residual)
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
    two processes, two thread counts, or two machines whose libraries dispatch
    to different kernels give bit-identical coefficients. The last of those is
    the one that is not free, and it is not the libraries' doing: they are
    entitled to reassociate a reduction and to fuse a multiply into an add,
    and for one release this module let them, in three places -- the BLAS
    reductions replaced by :func:`_ordered_sum` and :func:`_solve_dense`, and
    the two complex expressions replaced by :func:`_log_distance2` and
    :func:`_cascade_magnitude`.

    What is left rests on IEEE 754 and on numpy's transcendentals. Every
    arithmetic step on the path is now one addition, subtraction,
    multiplication, division or square root of two doubles, each of which the
    standard rounds to a single answer, or a :func:`math.fsum` that is
    correctly rounded whatever order it is handed. What the standard does not
    pin is ``exp``, ``log``, ``tan``, ``arctan``, ``sin``, ``cos`` and ``**``,
    and those are numpy's own results: measured here they do not move between
    numpy's SIMD kernel sets, but a numpy that retunes one of them, or a
    numpy on an architecture whose kernels are written separately, can move a
    last bit and with it -- through the amplification above -- the design.
    That is a property of the whole numeric stack rather than of this routine,
    and it is the one claim in this docstring that is argued rather than
    measured.
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


def _log_distance2(roots: np.ndarray, omega: np.ndarray) -> np.ndarray:
    r""":math:`\sum \log |j\Omega - r|^2` over *roots*, at each of *omega*.

    Written in real arithmetic on purpose. The argument is on the imaginary
    axis, so :math:`|j\Omega - r|^2` is
    :math:`(\Re r)^2 + (\Omega - \Im r)^2` exactly, and every operation in that
    is a single multiplication or addition of two doubles, which IEEE 754
    rounds to one answer. The complex spelling of the same quantity is not:
    ``numpy``'s complex multiply and complex absolute value are dispatched
    kernels that fuse a multiply and an add where the CPU offers the
    instruction, so they return a different last bit on a machine with FMA than
    on one without. Measured on this corpus, that one bit reaches every design:
    with ``NPY_DISABLE_CPU_FEATURES=X86_V3`` the complex form moved all 91 of
    them. The reduction over the roots is :func:`_ordered_sum` for the same
    reason the ones inside the fit are.
    """
    real = np.real(np.asarray(roots))[:, None]
    offset = (
        np.asarray(omega, dtype=np.float64)[None, :]
        - np.imag(np.asarray(roots))[:, None]
    )
    return _ordered_sum(np.log(real * real + offset * offset))


def _analog_db(
    zeros: np.ndarray, poles: np.ndarray, frequencies: np.ndarray, f_ref: float
) -> np.ndarray:
    """The prototype's magnitude in dB, relative to its value at *f_ref*.

    The same quantity :func:`_log_mag2` computes for a trial prototype, for the
    printed one, and deliberately in the same shape: a sum of logarithms rather
    than a logarithm of a product. That keeps it in real arithmetic (see
    :func:`_log_distance2`) and it also stops a curve with a dozen roots from
    overflowing a product of factors at the top of the grid, which the complex
    spelling was one decade of sampling rate away from doing.
    """
    omega = 2.0 * np.pi * np.asarray(frequencies, dtype=float)
    omega_ref = np.array([2.0 * np.pi * f_ref])

    def shape(at: np.ndarray) -> np.ndarray:
        top = _log_distance2(zeros, at) if np.asarray(zeros).size else np.zeros(at.size)
        return np.asarray(top - _log_distance2(poles, at))

    return np.asarray(_DB_PER_LOG * (shape(omega) - shape(omega_ref)[0]))


def _cascade_magnitude(sections: np.ndarray, frequency: float, fs: float) -> float:
    r"""``|H|`` of a biquad cascade at one frequency, in real arithmetic.

    :func:`scipy.signal.sosfreqz` computes the same number and is the obvious
    call, but it evaluates the sections as complex polynomials, and complex
    multiplication is one of the kernels ``numpy`` dispatches on what the CPU
    can do; see :func:`_log_distance2`. This is the only place the realised
    cascade is read back, and what is read back sets the shipped gain, so it is
    spelled out.

    A common factor of :math:`z^{-1}` does not change a magnitude, so a
    section's numerator is evaluated as
    :math:`b_1 + (b_0 + b_2)\cos\omega + \mathrm{j}(b_0 - b_2)\sin\omega`, and
    the real part of that as
    :math:`(b_0 + b_1 + b_2) - 2 (b_0 + b_2) \sin^2(\omega/2)`. The second form
    is the first one with :math:`\cos\omega` written through the half angle,
    and it is the one that survives dc: every curve here has zeros at the
    origin, which the bilinear transform puts at :math:`z = 1`, so
    :math:`b_0 + b_1 + b_2` is the small quantity the answer is made of, and
    computing it as a sum of the three coefficients keeps it while
    :math:`b_0 + b_1\cos\omega + b_2\cos 2\omega` throws it away. Measured on
    the G weighting at its 10 Hz reference frequency and 48 kHz, where the
    difference is at its worst in this corpus, that is 1.6e-14 relative against
    :func:`scipy.signal.sosfreqz`'s 1.3e-10; over the whole fit band it is
    6.4e-11 against 1.9e-7.

    The sections are multiplied in the order they are stored, one at a time, so
    the product is a sequence of individually rounded doubles rather than a
    reduction anything is free to reorder.

    :param sections: The cascade, one second-order section per row.
    :param frequency: Where to evaluate it, in Hz.
    :param fs: Sampling rate in Hz.
    :return: The magnitude, a plain float.
    """
    angle = 2.0 * np.pi * frequency / fs
    sine = float(np.sin(angle))
    half = float(np.sin(0.5 * angle))
    versine = 2.0 * half * half

    def squared(triplet: np.ndarray) -> float:
        c0, c1, c2 = (float(value) for value in triplet)
        real = (c0 + c1 + c2) - (c0 + c2) * versine
        imaginary = (c0 - c2) * sine
        return real * real + imaginary * imaginary

    magnitude = 1.0
    for section in np.asarray(sections, dtype=np.float64):
        magnitude *= math.sqrt(squared(section[:3]) / squared(section[3:]))
    return magnitude


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

    The first entry of :func:`_spare_placements` gets the budget, and the rest
    are tried only if what comes back is further than :data:`_FIT_FAILED_DB`
    from the printed curve. Trying every start and keeping the lowest peak was
    measured and rejected, which is worth recording because on the face of it
    it should win: the first start is beaten somewhere by a later one in 72 of
    the 91 (curve, rate) pairs, and taking the best of all eight takes the peak
    deviation over the fit band from 0.062 dB to 0.035 across the corpus.

    What that criterion minimises is the wrong thing. The peak over the band
    weights every frequency alike; the masks these curves are graded against do
    not. BS.468-4's 6.3 kHz row is allowed 0.05 dB where its neighbours are
    allowed ten to forty times as much, and at 44.1 kHz the lower-peak design
    spends 62 % of that row against the 50 % this library holds itself to,
    where the design kept here spends 13 % of it and 23 % of its worst row
    anywhere. A criterion that cannot see the mask does not get the last word
    over one that was chosen against it. It costs eight times the work as well,
    but that is not why it was refused.

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
    printed = _analog_db(
        np.asarray(zeros), np.asarray(poles), np.array([anchor]), f_ref
    )
    sos[0, :3] /= _cascade_magnitude(sos, anchor, fs) / 10.0 ** (printed[0] / 20.0)
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
