#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The routine that realises a weighting prototype at the sampling rate.

What ships is not a table of coefficients but
:mod:`phonometry.filters._weighting_design`, and this module holds it to the
three things that makes it a design rather than an optimiser's leftovers:

* it re-derives every filter the library ships, from the printed prototype,
  and gets back the same numbers the library carries;
* the response it produces is pinned against the analog prototype, per curve
  and per rate, far tighter than any standard's mask; and
* it is deterministic -- the same call gives the same bits, in this process, in
  another process, under another BLAS thread count, and on another CPU.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
from reference_data import IEC61672_TABLE3 as TABLE3
from scipy import signal as sg

from phonometry import filters
from phonometry.filters._weighting_design import (
    _CORNER_DECADES,
    _cascade_magnitude,
    _ordered_sum,
    _solve_dense,
    design_sos,
    fit_prototype,
    fitted_prototype_hz,
)
from phonometry.filters._weighting_design import (
    _analog_db as _shipped_analog_db,
)
from phonometry.filters.weighting import (
    _FIT_NYQUIST_FRACTION,
    _FIT_SECTIONS,
    _PROTOTYPE_CURVES,
    _REFERENCE_HZ,
    _STANDARD_BAND_HZ,
    _analog_weighting_zpk,
    _cached_weighting_sos,
    _fit_band,
)

#: Sample rates the accuracy table below is measured at: the two broadcast
#: families (32/44.1/48 kHz), their doubles, and 192 kHz.
_RATES = (32000, 44100, 48000, 88200, 96000, 192000)

#: Worst deviation from the analog prototype, in dB, that each curve is allowed
#: anywhere in its fit band at any rate in :data:`_RATES`. These are measured
#: values, not tolerances from any standard: the finest quantum any of these
#: tables is printed to is 0.05 dB, the 468 bound is a sixteenth of the
#: +/-1.6 dB its own Table 1 allows at 16 kHz, and every other row is at least
#: an order of magnitude inside its curve's mask. The 468 row is five times the
#: next largest and that is not an accuracy shortfall of the fit: the magnitude
#: of a real-coefficient filter has zero slope at the Nyquist frequency, and
#: that curve is still falling at about -30 dB/octave where the fit band ends,
#: half a percent below it.
#:
#: What they are *not* is the measurement on this machine rounded up. That is
#: the natural way to write a table like this, and what it does not survive is
#: the next change to the arithmetic underneath it. The design ends in a
#: sequence of small dense solves, and Levenberg-Marquardt in a shallow valley
#: amplifies a last-bit difference in one of them by four orders of magnitude
#: before the budget runs out. Nothing in the library reassociates those
#: reductions any more, and nothing on the path fuses a multiply into an add
#: either, which is what makes the design the same on every CPU; see
#: :func:`~phonometry.filters._weighting_design._ordered_sum` and
#: :func:`~phonometry.filters._weighting_design._log_distance2`. What is left
#: is numpy's own ``exp`` and ``log``, and a release that retunes one of them
#: still moves the last bit, so this table has to outlive that. Perturbing
#: every solve by a relative 1e-12 moves the C row from 0.00220 dB to 0.00324
#: and the AU row from 0.00590 to 0.00666. Rounded up, those two rows read
#: 0.003 and 0.006, so a gate written that way is already failing on a release
#: that has not happened yet.
#:
#: Each bound is therefore at least half again above what the routine reaches
#: under that perturbation, and still inside 2.5 times the unperturbed
#: measurement so that :func:`test_the_bound_is_not_slack` keeps biting from
#: the other side. Measured against both, the tightest row on the first count
#: is C at 1.54 times its perturbed worst and the loosest on the second is C
#: again at 2.27 times its unperturbed one, so every row is inside the band
#: and no row is at its edge.
#:
#: The bounds are left where they were rather than pulled down onto
#: measurements that have just moved. Several rows did move: the fit is
#: multi-modal, a change to the arithmetic underneath it lands it in a
#: different mode, and the 468 row went from 0.0597 dB to 0.0624 without
#: anything about the search changing. That is the whole reason a bound here is
#: a band and not a pinned number, and a bound retuned every time the fit lands
#: somewhere else would not be a guard at all.
_WORST_DEVIATION_DB = {
    "A": 0.013,
    "B": 0.0035,
    "C": 0.005,
    "D": 0.018,
    "G": 0.0001,
    "AU": 0.011,
    "468": 0.100,
}


#: Sample rates at or below twice the 1 kHz reference frequency, where the
#: frequency every curve but G is normalised at is past the Nyquist frequency
#: and there is no digital frequency to read 0 dB at. The rates the module's
#: own table is measured at start at 32 kHz and the IEC Table 3 test at 16 kHz,
#: so nothing in this suite used to look here at all. Includes the seven pairs
#: two separate faults were first seen at -- A at 903 Hz, B at 1 kHz, and AU at
#: 1 kHz, 250, 200, 125 and 100 -- each of which came back as a record of NaN
#: through the public entry point. G, whose reference frequency is 10 Hz, is
#: not in that region at any of these rates; it is swept here anyway, because
#: what these rates have in common is that no test looked at them.
_LOW_RATES = (100, 125, 200, 250, 400, 500, 903, 1000, 1500, 2000)

#: Worst deviation from the printed prototype, in dB, allowed anywhere in the
#: fit band at the rates in :data:`_LOW_RATES`. Those ten rates, and not the
#: whole region below them: swept at 1 Hz, the low-rate surface is rough, and
#: 96 rates under 1 kHz exceed this bound, up to 6.2 dB. What the fit band is
#: at these rates makes that unsurprising rather than alarming, and the
#: paragraph below says why, but the guard pins the sampled rates and no more.
#: Looser than the audio-rate table
#: above, for a reason that is not the fit's: the fit band at these rates is a
#: sliver whose top edge sits half a percent under a Nyquist frequency the
#: curve is still moving at, and a real-coefficient filter's magnitude has zero
#: slope there. The measurement is 0.307 dB, at the G weighting at fs = 500 and
#: again at 400, where the standard's own 315 Hz upper limit is past the
#: Nyquist frequency; eight of the seventy pairs are over 0.1 dB and the median
#: is 0.0013. The bound is set well clear of that rather than just above it,
#: because these are the designs that reach for a second start, and which start
#: wins is decided on an achieved residual that the last bit of the arithmetic
#: can still move. What the number has to beat is the design this branch
#: replaced: the oversampled path reached 0.446 dB over the same seventy
#: pairs, at the same G row.
#:
#: That last worry turns out not to apply to the row that sets the number. Under
#: the same perturbation of every linear solve the table above is sized against,
#: the seventy-pair maximum does not move at all: 0.30706 dB at G, fs = 500, to
#: five decimals, at 1e-12 and at 1e-10 and across seeds. It is set by the
#: zero slope a real-coefficient filter has at the Nyquist frequency and not by
#: where the search stopped, so there is nothing in it for a perturbed
#: reduction to decide. The bound therefore sits at 1.43 times a measurement
#: that does not wander, which is tighter than any row of the audio table above
#: and is not the coin flip that ratio would be if the measurement did.
_WORST_LOW_RATE_DB = 0.44

#: Sample rates at which the library claims IEC 61672-1 class 1 for A and C:
#: the two broadcast families and their halves and doubles. The claim is made
#: in prose in three places (the ``high_accuracy`` paragraph of
#: :class:`~phonometry.filters.WeightingFilter`, the module docstring and the
#: changelog) and until this table existed it was checked at four of these
#: eight rates. The four that nothing looked at were 8 kHz -- the lowest rate
#: the claim covers, and the one the oversampled path this branch replaced
#: failed hardest, missing every class by 5.86 dB -- 22.05 kHz, 44.1 kHz, and
#: 192 kHz.
_CLASS_1_RATES = (8000, 16000, 22050, 32000, 44100, 48000, 96000, 192000)

#: Worst deviation, in dB, between a tone measured through
#: :meth:`~phonometry.filters.WeightingFilter.filter` and the printed
#: prototype's own value at that frequency, over the rates in
#: :data:`_CLASS_1_RATES` and every IEC 61672-1 Table 3 row below the Nyquist
#: frequency at each of them.
#:
#: This measures a different thing from :data:`_WORST_DEVIATION_DB`, and the
#: difference is the whole of what this branch changed. That table reads the
#: response out of ``WeightingFilter.sos`` with ``sosfreqz``; this one puts a
#: signal in at one end and measures what comes out of the other. The two agree
#: only because the path is now one cascade of sections and nothing else, which
#: is a property of the library and not a law of nature: the path this branch
#: replaced reached its sections through an interpolation and a decimation
#: stage, and a ``sosfreqz`` of its sections said nothing whatever about what a
#: tone would read.
#:
#: The measurement is 0.0081 dB (A at 32 kHz, the 316.2 Hz row), of which at
#: most 0.0015 dB anywhere in the sweep is the RMS estimator rather than the
#: filter. Sized like the table above: at least half again above the 0.0079 dB
#: the same sweep reaches when every linear solve is perturbed by a relative
#: 1e-10, and inside 2.5 times the unperturbed measurement so that
#: :func:`test_the_tone_bound_is_not_slack` keeps biting from the other side.
_WORST_TONE_DB = 0.013


def _analog_db(
    zeros: np.ndarray, poles: np.ndarray, frequencies: np.ndarray, f_ref: float
) -> np.ndarray:
    """The prototype's magnitude in dB re its value at *f_ref*, computed here.

    Deliberately a second implementation rather than an import: a residual
    measured with the same helper the fit minimises would only prove the fit
    converged, not that it converged onto the printed curve.
    """
    s = 2j * np.pi * np.asarray(frequencies, dtype=float)

    def magnitude(argument: np.ndarray) -> np.ndarray:
        top = (
            np.prod(argument[:, None] - zeros[None, :], axis=1)
            if zeros.size
            else np.ones_like(argument)
        )
        return np.abs(top / np.prod(argument[:, None] - poles[None, :], axis=1))

    return np.asarray(
        20.0 * np.log10(magnitude(s) / magnitude(np.array([2j * np.pi * f_ref]))[0])
    )


def _deviation_db(curve: str, fs: int, points: int = 20001) -> np.ndarray:
    """Realised response minus the printed prototype, over the fit band."""
    zeros, poles, _ = _analog_weighting_zpk(curve)
    low, high = _fit_band(curve, fs)
    grid = np.geomspace(low, high, points)
    _, response = sg.sosfreqz(filters.WeightingFilter(fs, curve).sos, worN=grid, fs=fs)
    reference = _REFERENCE_HZ.get(curve, 1000.0)
    return 20.0 * np.log10(np.abs(response)) - _analog_db(zeros, poles, grid, reference)


def _tone_gain_db(curve: str, fs: int, f0: float) -> float:
    """Gain a steady tone at *f0* really sees, end to end through ``filter()``.

    The one measurement in this module that puts a signal through the public
    entry point rather than reading a transfer function off the coefficients,
    which is what makes it able to see a stage that ``sosfreqz`` cannot.

    The window closes on a whole number of periods after the transient: input
    and output differ in phase, so a part cycle biases the two RMS values by
    different amounts, which at the lowest rows is worth more than the quantity
    being measured. What is left of the estimator is at most 0.0015 dB anywhere
    in the sweep below, measured against ``sosfreqz`` at the same frequency.
    """
    samples = int(fs * max(0.5, 12.0 / f0))
    time = np.arange(samples) / fs
    tone = np.sin(2.0 * np.pi * f0 * time)
    weighted = np.asarray(filters.WeightingFilter(fs, curve).filter(tone))
    start = int(0.2 * fs)
    period = fs / f0
    whole = int((samples - start) / period)
    stop = start + max(int(round(whole * period)), 1)
    return float(
        20.0 * np.log10(np.std(weighted[start:stop]) / np.std(tone[start:stop]))
    )


def _table3_frequencies(fs: int) -> list[float]:
    """Exact base-10 frequencies of the Table 3 rows below the Nyquist frequency.

    Table 3's NOTE puts the design goals at the exact base-10 frequency behind
    each nominal label, so "16 k" is 15 848.9 Hz. The 20 kHz label's own
    frequency is added alongside its exact one because that is where the
    published high-frequency reading is quoted; see
    :func:`test_the_20_khz_reading_at_44_1_khz_is_the_published_one`.
    """
    exact = {10.0 ** (round(10.0 * np.log10(row[0])) / 10.0) for row in TABLE3}
    return sorted(f for f in exact | {20000.0} if f < 0.5 * fs)


def _cascade_magnitude_at_higher_precision(
    sos: np.ndarray, frequency: float, fs: float
) -> np.longdouble:
    """``|H|`` again, in whatever precision above double this machine has."""
    angle = np.longdouble(2.0) * np.longdouble(np.pi) * np.longdouble(frequency)
    z = np.exp(-1j * np.clongdouble(angle / np.longdouble(fs)))
    magnitude = np.longdouble(1.0)
    for section in sos:
        coefficients = [np.longdouble(value) for value in section]
        top = coefficients[0] + coefficients[1] * z + coefficients[2] * z * z
        bottom = coefficients[3] + coefficients[4] * z + coefficients[5] * z * z
        magnitude *= abs(top) / abs(bottom)
    return magnitude


def _worst_tone_deviation_db(curve: str, fs: int) -> tuple[float, float]:
    """Worst ``(deviation, frequency)`` of a tone from the printed curve at *fs*."""
    zeros, poles, _ = _analog_weighting_zpk(curve)
    worst, where = 0.0, 0.0
    for f0 in _table3_frequencies(fs):
        printed = _analog_db(zeros, poles, np.array([f0]), 1000.0)[0]
        deviation = _tone_gain_db(curve, fs, f0) - printed
        if abs(deviation) > abs(worst):
            worst, where = deviation, f0
    return worst, where


@pytest.mark.parametrize("fs", _RATES)
@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_realised_filter_tracks_its_printed_prototype(curve: str, fs: int) -> None:
    """Every shipped design, against the curve the standard defines."""
    worst = float(np.max(np.abs(_deviation_db(curve, fs))))
    assert worst <= _WORST_DEVIATION_DB[curve], f"{curve} at fs={fs}: {worst:.5f} dB"


@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_the_bound_is_not_slack(curve: str) -> None:
    """A bound ten times looser than the measurement pins nothing.

    Each entry of :data:`_WORST_DEVIATION_DB` has to stay close to what the
    routine actually delivers, or the table above stops being a regression
    guard and becomes decoration.

    Written as a floor under the measurement, which with
    :func:`test_realised_filter_tracks_its_printed_prototype` above it makes
    the pair a range: each curve has to land between 0.4 and 1.0 of its bound.
    It is worth saying why the floor stays, because it is the half that used to
    fail. Before the design was made machine independent, AU came back at
    0.00413 dB on four of the OpenBLAS kernel sets against the 0.0044 this
    floor needs, so ``tests/filters`` failed on eleven of the twenty-one
    selectable kernels -- for landing on a *better* filter. The answer to that
    is not to drop the floor. A design that quietly improves and leaves its
    bound where it was is precisely when the bound stops measuring anything,
    and the floor is the only thing that asks for it to be retightened. What
    was wrong was that the measurement was not a function of the source, and
    that is what has been fixed; see
    :func:`test_the_design_does_not_depend_on_the_blas_kernel`.
    """
    worst = max(float(np.max(np.abs(_deviation_db(curve, fs)))) for fs in _RATES)
    assert worst >= 0.4 * _WORST_DEVIATION_DB[curve], (
        f"{curve}: measured {worst:.5f} dB against a bound of "
        f"{_WORST_DEVIATION_DB[curve]}; tighten the bound"
    )


@pytest.mark.parametrize("curve", ["A", "C", "468", "G"])
def test_the_library_carries_no_coefficients_of_its_own(curve: str) -> None:
    """Re-derive the shipped filter from the printed prototype, from scratch.

    This is what makes the routine the deliverable: nothing between the
    standard's constants and the sections a caller runs is stored, so the
    numbers in the library cannot drift from the numbers the routine produces.
    """
    fs = 48000
    zeros, poles, _ = _analog_weighting_zpk(curve)
    rebuilt = design_sos(
        zeros,
        poles,
        float(fs),
        _fit_band(curve, fs),
        _FIT_SECTIONS[curve],
        _REFERENCE_HZ.get(curve, 1000.0),
    )
    np.testing.assert_array_equal(rebuilt, filters.WeightingFilter(fs, curve).sos)


@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_the_design_is_stable_and_minimum_phase(curve: str) -> None:
    """No root of any factor leaves the left half plane, at any rate.

    The parameterisation makes this true by construction -- every factor is
    ``s^2 + exp(t1) s + exp(t0)``, whose Routh-Hurwitz condition is satisfied
    for any real ``t`` -- so this test is checking that the construction is
    what the routine actually uses.
    """
    zeros, poles, _ = _analog_weighting_zpk(curve)
    for fs in (32000, 48000, 192000):
        fitted_zeros, fitted_poles = fit_prototype(
            zeros,
            poles,
            float(fs),
            _fit_band(curve, fs),
            _FIT_SECTIONS[curve],
            _REFERENCE_HZ.get(curve, 1000.0),
        )
        assert np.all(fitted_poles.real < 0.0)
        moved = fitted_zeros[np.abs(fitted_zeros) > 0.0]
        assert np.all(moved.real < 0.0)
        # The zeros the prototype puts at the origin stay exactly there, so
        # the realised filter blocks dc exactly, as the analog network does.
        at_origin = fitted_zeros.size - moved.size
        assert at_origin == int(np.count_nonzero(np.abs(zeros) <= 0.0))


@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_the_reference_frequency_carries_no_error(curve: str) -> None:
    """0 dB at the reference, exactly: every one of these standards demands it.

    IEC 61012 Table 1 prints a zero tolerance there outright, and the others
    define their curve as a level relative to it, so an error at the reference
    frequency is an error at every other one.

    Read back at a precision above the one the cascade is stored in, and not
    with :func:`scipy.signal.sosfreqz`, which is not accurate enough to see
    whether this holds. G's reference frequency is 10 Hz, which at 48 kHz is
    2.6e-6 of the Nyquist frequency, and evaluating a numerator with zeros at
    ``z = 1`` there costs ``sosfreqz`` ten significant digits to cancellation:
    it reads this filter 1.1e-9 dB from 0 where the filter is 1.4e-13 dB from
    it. That mattered less when the cascade was normalised by ``sosfreqz`` too,
    because then the same error appeared on both sides and cancelled, and this
    assertion measured only that a number had been divided by itself. It does
    not cancel now (see
    :func:`~phonometry.filters._weighting_design._cascade_magnitude`), which is
    what makes the pin worth asserting rather than tautological.
    """
    reference = _REFERENCE_HZ.get(curve, 1000.0)
    for fs in (44100, 48000):
        sos = filters.WeightingFilter(fs, curve).sos
        wider = _cascade_magnitude_at_higher_precision(sos, reference, float(fs))
        assert 20.0 * float(np.log10(wider)) == pytest.approx(0.0, abs=1e-12)


def test_the_fit_band_covers_every_row_the_standards_state() -> None:
    """The clip fraction is chosen for the row that comes closest to Nyquist.

    IEC 61672-1 Table 3's "16 kHz" row is measured at the exact base-10
    frequency 15 848.9 Hz, which is 0.9906 of the Nyquist frequency at
    fs = 32 kHz -- the closest approach anything in this corpus makes. If the
    clip fraction ever drops below that, the design stops controlling a
    frequency the standard grades.
    """
    exact_16k = 10.0 ** (round(10.0 * np.log10(16000.0)) / 10.0)
    assert exact_16k / 16000.0 < _FIT_NYQUIST_FRACTION
    assert _fit_band("A", 32000)[1] > exact_16k
    # And at a rate that does not clip, the band is the standard's own.
    assert _fit_band("A", 192000) == _STANDARD_BAND_HZ["A"]


def test_the_reduction_is_pairwise_and_not_a_running_total() -> None:
    r"""What replaces ``ddot`` has to be at least as accurate, not just as fixed.

    A running total accumulates :math:`O(\varepsilon n)` of rounding and
    folding in half accumulates :math:`O(\varepsilon \log n)`, so on a
    conditioned sum the fold has to land nearer the exactly rounded answer
    than the loop does. Measured against :func:`math.fsum`, which is that
    answer.
    """
    rng = np.random.default_rng(20260830)
    terms = rng.normal(size=4096) * rng.lognormal(sigma=8.0, size=4096)
    exact = math.fsum(terms.tolist())
    folded = abs(float(_ordered_sum(terms)) - exact)
    running = abs(float(np.cumsum(terms)[-1]) - exact)
    assert folded <= running

    # The leftover of an odd count is carried, not dropped.
    for count in (1, 2, 3, 5, 257, 1000):
        assert float(_ordered_sum(np.ones(count))) == pytest.approx(count, abs=0)
    # And an empty stack sums to a zero of the right shape.
    assert _ordered_sum(np.zeros((0, 4))).shape == (4,)


def test_the_solver_matches_lapack_and_refuses_rather_than_diverges() -> None:
    """The elimination done here, against the one it replaced.

    The point of writing it out is the reduction order, not a different
    answer, so on systems that have one it has to agree with
    :func:`numpy.linalg.solve` to the conditioning and no worse. The two
    degenerate inputs are the reason the pivot test is spelled as a magnitude:
    a singular system and one that has gone to NaN both have to come back as a
    step of zeros, which the accept test then rejects, rather than as an
    exception or as NaN in the parameters.
    """
    rng = np.random.default_rng(20260830)
    worst = 0.0
    for _ in range(200):
        order = int(rng.integers(1, 25))
        square = rng.normal(size=(order, order))
        spd = square @ square.T + 1e-3 * np.eye(order)
        right = rng.normal(size=order)
        mine = _solve_dense(spd, right)
        theirs = np.linalg.solve(spd, right)
        worst = max(worst, float(np.max(np.abs(mine - theirs))))
    assert worst < 1e-6, f"worst disagreement with LAPACK: {worst:.3e}"

    np.testing.assert_array_equal(_solve_dense(np.ones((5, 5)), np.arange(5.0)), 0.0)
    np.testing.assert_array_equal(
        _solve_dense(np.full((4, 4), np.nan), np.ones(4)), 0.0
    )


@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_the_printed_curve_is_the_same_curve_written_in_real_arithmetic(
    curve: str,
) -> None:
    r"""The target the fit aims at, against the complex spelling it replaced.

    :func:`~phonometry.filters._weighting_design._analog_db` sums logarithms of
    :math:`(\Re r)^2 + (\Omega - \Im r)^2` where it used to take the
    magnitude of a product of complex factors, because the complex kernels are
    dispatched on what the CPU can do and the real ones are not. That is a
    change of arithmetic, not of function, so it has to agree with the thing it
    replaced over the whole band -- and be the better conditioned of the two,
    since a product of a dozen factors at 384 kHz is within a decade of
    overflowing where a sum of logarithms is nowhere near it.
    """
    zeros, poles, _ = _analog_weighting_zpk(curve)
    reference = _REFERENCE_HZ.get(curve, 1000.0)
    grid = np.geomspace(1.0, 200000.0, 4001)

    s = 2j * np.pi * grid
    s_ref = 2j * np.pi * reference

    def complex_magnitude(argument: np.ndarray) -> np.ndarray:
        top = (
            np.prod(argument[:, None] - zeros[None, :], axis=1)
            if zeros.size
            else np.ones_like(argument)
        )
        return np.abs(top / np.prod(argument[:, None] - poles[None, :], axis=1))

    expected = 20.0 * np.log10(
        complex_magnitude(s) / complex_magnitude(np.array([s_ref]))[0]
    )
    np.testing.assert_allclose(
        _shipped_analog_db(zeros, poles, grid, reference),
        expected,
        atol=1e-11,
        rtol=0.0,
    )


def test_the_cascade_is_read_back_no_worse_than_sosfreqz_reads_it() -> None:
    """The gain the shipped filter is normalised by, against scipy's own.

    Same argument as the curve above: writing the read-back out in real
    arithmetic is worth doing only if it is the same number. Checked on every
    curve at a rate where the anchor is the reference frequency and at one
    where it is not, and away from the anchor as well, so that what is compared
    is a magnitude and not one coincidence at 1 kHz.

    Graded against a wider-precision evaluation of the same cascade rather than
    against ``sosfreqz`` directly, because the two disagree by 3e-7 relative at
    the bottom of the G weighting's band and neither is wrong there: at 0.25 Hz
    and 48 kHz the numerator of a section is a difference of ``O(1)``
    coefficients that comes out at 4e-5, so five significant digits are lost to
    cancellation before either method starts. What the test has to establish is
    that the arithmetic written out here does not lose more of them than the
    library call did, and at the worst point in the sweep it loses fewer
    (1.3e-7 against 1.9e-7).
    """
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        pytest.skip("no precision above double to grade against here")
    mine_worst, theirs_worst, agreement = 0.0, 0.0, 0.0
    for curve in _PROTOTYPE_CURVES:
        for fs in (903, 48000):
            sos = filters.WeightingFilter(fs, curve).sos
            probes = np.geomspace(*_fit_band(curve, fs), 40)
            _, response = sg.sosfreqz(sos, worN=probes, fs=fs)
            for probe, theirs in zip(probes, np.abs(response), strict=True):
                mine = _cascade_magnitude(sos, float(probe), float(fs))
                wider = _cascade_magnitude_at_higher_precision(
                    sos, float(probe), float(fs)
                )
                mine_worst = max(mine_worst, float(abs(mine - wider) / wider))
                theirs_worst = max(theirs_worst, float(abs(theirs - wider) / wider))
                agreement = max(agreement, float(abs(mine - theirs) / wider))
    assert agreement < 1e-6, f"disagreement with sosfreqz: {agreement:.3e}"
    assert mine_worst <= theirs_worst, (
        f"written out: {mine_worst:.3e}, sosfreqz: {theirs_worst:.3e}"
    )


def test_repeated_designs_are_bit_identical() -> None:
    """Same inputs, same bits: there is no random state and no clock here."""
    zeros, poles, _ = _analog_weighting_zpk("A")
    args = (zeros, poles, 44100.0, _fit_band("A", 44100), 4, 1000.0)
    np.testing.assert_array_equal(design_sos(*args), design_sos(*args))


def test_the_design_does_not_depend_on_the_blas_thread_count() -> None:
    """A filter that changes with ``OMP_NUM_THREADS`` is a filter that changes.

    The routine is a fixed sequence of small dense solves, so a threaded BLAS
    must not be able to reorder its way to a different answer. Run in a
    subprocess because the thread count is read when the library loads.
    """
    script = textwrap.dedent(
        """
        import numpy as np
        from phonometry import filters
        sos = filters.WeightingFilter(44100, "A").sos
        print(sos.tobytes().hex())
        """
    )

    def digest(threads: str) -> str:
        environment = dict(os.environ)
        environment.update(
            OMP_NUM_THREADS=threads,
            OPENBLAS_NUM_THREADS=threads,
            MKL_NUM_THREADS=threads,
        )
        return subprocess.run(  # noqa: S603  # fixed argv, no shell, no user input
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()

    assert digest("1") == digest("4")


#: Kernel sets ``OPENBLAS_CORETYPE`` can select that reach different code in a
#: ``DYNAMIC_ARCH`` build without needing instructions a 64-bit x86 machine
#: might not have. Deliberately not the AVX-512 sets: those fault outright on a
#: CPU that does not implement them, and nothing here is worth a SIGILL.
_KERNEL_SETS = ("Prescott", "Nehalem", "Sandybridge", "Haswell")


def test_the_design_does_not_depend_on_the_blas_kernel() -> None:
    """The same filter on every CPU, which is the harder half of determinism.

    The thread count above is the easy half. This is the one that bit: numpy
    here sits on an OpenBLAS built ``DYNAMIC_ARCH``, which picks its kernels
    from what the CPU reports at load time, and each set is free to reassociate
    a reduction differently. When the fit's reductions were BLAS calls, that
    was enough to hand back a different filter: 0.5904875 against 0.5905175
    against 0.5904705 in the leading coefficient of A at 48 kHz, from the same
    source on the same machine, with only this variable changed.

    ``OPENBLAS_CORETYPE`` is how a single machine can be asked for another
    machine's kernels. It is honoured only by a ``DYNAMIC_ARCH`` OpenBLAS, so
    on a build that ignores it every subprocess agrees for the wrong reason and
    this test passes without having looked at anything -- which is why the
    determinism it is guarding is a property of the routine (see
    :func:`~phonometry.filters._weighting_design._ordered_sum`) rather than of
    what this test can see. A kernel set the CPU cannot run is skipped rather
    than failed, because refusing to start is not disagreement.
    """
    script = textwrap.dedent(
        """
        from phonometry import filters
        print(filters.WeightingFilter(48000, "A").sos.tobytes().hex())
        """
    )
    digests = {}
    for kernel in _KERNEL_SETS:
        environment = dict(os.environ, OPENBLAS_CORETYPE=kernel)
        finished = subprocess.run(  # noqa: S603  # fixed argv, no shell, no user input
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if finished.returncode == 0:
            digests[kernel] = finished.stdout.strip()
    if len(digests) < 2:
        pytest.skip("no two OpenBLAS kernel sets are selectable here")
    assert len(set(digests.values())) == 1, {k: v[:32] for k, v in digests.items()}


def test_the_design_does_not_depend_on_the_numpy_kernel() -> None:
    """The other half of the same question, and the half that lasted longer.

    ``OPENBLAS_CORETYPE`` above varies the library numpy calls. This varies
    numpy's own: it is built with a baseline every x86-64 machine has and a set
    of feature groups it dispatches to at import, and ``NPY_DISABLE_CPU_FEATURES``
    turns a group off, which is what a CPU that does not implement it does by
    itself. The kernels that moved were the complex ones -- complex multiply
    and complex absolute value fuse a multiply into an add where the
    instruction exists -- and with the fit's reductions already fixed they were
    still enough to move all ninety-one designs in the corpus, because the
    curve the fit aims at was computed through them. Hence
    :func:`~phonometry.filters._weighting_design._log_distance2` and
    :func:`~phonometry.filters._weighting_design._cascade_magnitude`.

    The whole dispatchable set is turned off at once rather than one group at a
    time: what matters is that the baseline answer and the tuned answer agree,
    not which group carried the difference. As with the kernel test above, a
    build with nothing to disable passes without having looked at anything.
    """
    script = textwrap.dedent(
        """
        from phonometry import filters
        print(filters.WeightingFilter(48000, "A").sos.tobytes().hex())
        """
    )
    digests = {}
    for disabled in ("", "X86_V3 X86_V4 AVX512_ICL AVX512_SPR"):
        environment = dict(os.environ, NPY_DISABLE_CPU_FEATURES=disabled)
        finished = subprocess.run(  # noqa: S603  # fixed argv, no shell, no user input
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if finished.returncode == 0:
            digests[disabled or "everything"] = finished.stdout.strip()
    if len(digests) < 2:
        pytest.skip("numpy's dispatched kernels are not selectable here")
    assert len(set(digests.values())) == 1, {k: v[:32] for k, v in digests.items()}


@pytest.mark.parametrize("fs", _LOW_RATES)
@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_a_rate_that_cannot_carry_the_reference_frequency_still_filters(
    curve: str, fs: int
) -> None:
    """Below ``2 * f_ref`` the design still has to be a filter, not a NaN.

    Two separate faults lived in this region and neither raised anything: the
    realised cascade was normalised at the alias of the reference frequency,
    which is dc whenever the rate divides it, and the fit was anchored there
    too, which handed the search a residual it could not drive below 424 dB
    for the B weighting at fs = 1000. The parameters ran to their clamps, the
    fitted corners came back at :math:`10^{86}` rad/s, their product
    overflowed, and
    ``weighting_filter`` returned a record of NaN with nothing but a numpy
    RuntimeWarning to show for it. So the warnings are errors here: a design
    that has to warn to produce a number has not produced one.
    """
    samples = np.sin(2.0 * np.pi * 7.0 * np.arange(512) / fs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        weighted = filters.weighting_filter(samples, fs, curve)
    assert np.isfinite(np.asarray(weighted)).all()


@pytest.mark.parametrize("fs", _LOW_RATES)
@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_a_rate_that_cannot_carry_the_reference_frequency_still_tracks_the_curve(
    curve: str, fs: int
) -> None:
    """Finite is not the bar; the printed curve is.

    A filter that comes back finite and 30 dB from the standard's response is
    the same defect wearing a different face, and both of the faults above had
    that form at some rates and the NaN form at others -- the same A weighting
    that returned NaN at 903 Hz returned a finite 70 dB error at 1 kHz. So
    these rates are held to :data:`_WORST_LOW_RATE_DB` as well as to being
    finite. They are a sample of the region, not a cover of it: see that
    constant for what the rest of it does.
    """
    worst = float(np.max(np.abs(_deviation_db(curve, fs, points=4001))))
    assert worst <= _WORST_LOW_RATE_DB, f"{curve} at fs={fs}: {worst:.5f} dB"


def test_the_level_below_the_reference_frequency_is_the_printed_one() -> None:
    """The gain below ``2 * f_ref`` is still the standard's own.

    There is no 0 dB point to normalise at when the reference frequency is past
    the Nyquist frequency, and the two wrong ways to cope both leave a filter
    that is finite and wrong: normalising at the alias of 1 kHz, and
    normalising the analog prototype at 1 kHz, which the transform then puts
    somewhere else entirely. What has to survive is the absolute level -- a
    200 Hz tone at fs = 903 reads the A weighting's own -10.9 dB, the same
    number it reads at 48 kHz -- so this checks the level and not the shape.
    """
    tone = 200.0
    zeros, poles, _ = _analog_weighting_zpk("A")
    printed = _analog_db(zeros, poles, np.array([tone]), 1000.0)[0]
    assert printed == pytest.approx(-10.9, abs=0.1)
    levels = []
    for fs in (903, 48000):
        _, response = sg.sosfreqz(
            filters.WeightingFilter(fs, "A").sos, worN=np.array([tone]), fs=fs
        )
        levels.append(20.0 * np.log10(abs(response[0])))
    assert levels[0] == pytest.approx(printed, abs=0.01)
    assert levels[1] == pytest.approx(printed, abs=0.01)


@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_no_fitted_corner_leaves_the_representable_band(curve: str) -> None:
    """Every corner the fit returns is a frequency, at every rate.

    The clamp of :data:`~phonometry.filters._weighting_design._CORNER_DECADES`
    is what makes the finiteness above a property of the routine rather than an
    observation about the rates that happen to have been tried: a corner far
    enough outside the fit grid is flat over all of it, so pushing it further
    buys nothing and costs the ability to multiply a cascade of them together
    without overflowing. Without the clamp the search reached 10**86 rad/s.

    The bound checked here is a decade looser than the clamp, and the slack is
    for two things rather than for comfort: a quadratic that has split its two
    roots apart carries both of them in its ``b1``, so a corner limit of ``c``
    admits a root at ``2 c``; and a factor sitting exactly on a clamp stated in
    logs lands a rounding step either side of a limit recomputed here. Neither
    is worth widening the assertion for beyond a decade, because what it is
    there to catch is seventy decades away.
    """
    zeros, poles, _ = _analog_weighting_zpk(curve)
    for fs in (100, 903, 2000, 48000):
        low, high = _fit_band(curve, fs)
        fitted_zeros, fitted_poles = fit_prototype(
            zeros,
            poles,
            float(fs),
            (low, high),
            _FIT_SECTIONS[curve],
            _REFERENCE_HZ.get(curve, 1000.0),
        )
        corners = np.abs(np.concatenate([fitted_zeros, fitted_poles]))
        top = 2.0 * fs * np.tan(np.pi * high / fs)
        assert np.all(np.isfinite(corners))
        assert np.max(corners) < top * 10.0 ** (_CORNER_DECADES + 1.0)


def test_an_empty_band_is_refused() -> None:
    """A rate whose Nyquist frequency falls under the band's own floor.

    IEC 61672-1 Table 3 starts at 10 Hz, so below about 20 Hz of sample rate
    there is no interval left to fit over. The failure has to name the
    interval rather than come back as a numpy error from inside the fit.
    """
    zeros, poles, _ = _analog_weighting_zpk("A")
    band = _fit_band("A", 16)
    with pytest.raises(ValueError, match=r"'band' must span a positive frequency"):
        fit_prototype(zeros, poles, 16.0, band, 4, 1000.0)


def test_too_few_sections_is_refused() -> None:
    """A section count that cannot hold the prototype's own degree.

    The AU prototype has twelve poles; asking for two biquads would silently
    drop ten of them, so it is refused with the two degrees named.
    """
    zeros, poles, _ = _analog_weighting_zpk("AU")
    band = _fit_band("AU", 48000)
    with pytest.raises(ValueError, match=r"'sections' is too small"):
        fit_prototype(zeros, poles, 48000.0, band, 2, 1000.0)


def test_the_fitted_prototype_is_readable_in_hertz() -> None:
    """The fit stays a prototype, so a reader can look at where it landed.

    The low-frequency corners are the ones the printed table is really about,
    and the warp barely touches them: IEC 61672-1's ``f1 = 20.598997`` Hz comes
    back as a pole pair whose corner frequency agrees to five digits, and
    ``f2 = 107.65265`` as a real pole to four. The corners near the top of the
    band move much further, by design -- moving them is how the warp is
    cancelled -- so this checks the recognisable end, not all of it.
    """
    zeros, poles, _ = _analog_weighting_zpk("A")
    fitted_zeros, fitted_poles = fitted_prototype_hz(
        zeros, poles, 48000.0, _fit_band("A", 48000), 4, 1000.0
    )
    corners = np.sort(np.abs(fitted_poles))
    assert corners[0] == pytest.approx(20.598997, rel=1e-4)
    assert corners[1] == pytest.approx(20.598997, rel=1e-4)
    assert corners[2] == pytest.approx(107.65265, rel=1e-3)
    assert np.count_nonzero(np.abs(fitted_zeros) <= 0.0) == 4


def test_the_low_rate_bound_is_not_slack() -> None:
    """The other half of the guard the audio-rate table already had.

    :data:`_WORST_DEVIATION_DB` is held from both sides, so nobody can quietly
    widen a row to make something pass. :data:`_WORST_LOW_RATE_DB` was held
    from one, and a single number covering seventy (curve, rate) pairs is
    exactly the kind that gets widened: raising it from 0.44 to 3.0, a factor
    of seven, used to leave the whole suite green.

    Same trip point as :func:`test_the_bound_is_not_slack`, for the same
    reason, against a measurement that happens to be steadier than any row of
    the audio table -- see :data:`_WORST_LOW_RATE_DB`.
    """
    worst = max(
        float(np.max(np.abs(_deviation_db(curve, fs, points=4001))))
        for curve in _PROTOTYPE_CURVES
        for fs in _LOW_RATES
    )
    assert worst >= 0.4 * _WORST_LOW_RATE_DB, (
        f"measured {worst:.5f} dB against a bound of {_WORST_LOW_RATE_DB}; "
        "tighten the bound"
    )


@pytest.mark.parametrize("fs", [8000, 44100, 192000])
@pytest.mark.parametrize("curve", _PROTOTYPE_CURVES)
def test_the_whole_path_is_the_sections_and_nothing_else(curve: str, fs: int) -> None:
    """What ``filter()`` does is ``sosfilt``, to the bit, with nothing around it.

    Every accuracy figure in this module except the tone sweep below is read
    out of ``WeightingFilter.sos`` with ``sosfreqz``, and that is only a
    statement about the filter a caller runs while this holds. The path this
    branch replaced interpolated by three to eight, filtered, and decimated
    back, and its anti-alias FIR had its transition band sitting on the input
    Nyquist frequency: ``sosfreqz`` of those sections was 0.08 dB from the
    design goal at the top row of a 16 kHz system where a tone through the same
    object measured 12.0 dB from it.

    So this is the assumption the rest of the module rests on, written down.
    Restoring the resampler around the *fitted* sections -- which is the
    cheapest way to lose everything this branch bought, because the design
    still looks perfect on paper -- leaves every other test in this file green
    and fails this one.

    It is also the durable half of the speed claim. One ``sosfilt`` at the
    input rate is why a minute of 44.1 kHz audio costs about 18 ms instead of
    377; the wall-clock number is deliberately not asserted, because a loaded
    CI machine is entitled to be slow, but the structure that makes it true is.
    """
    samples = np.random.default_rng(4).standard_normal(2048)
    filtered = filters.WeightingFilter(fs, curve).filter(samples)
    bare = sg.sosfilt(filters.WeightingFilter(fs, curve).sos, samples)
    np.testing.assert_array_equal(np.asarray(filtered), bare)


@pytest.mark.parametrize("fs", _CLASS_1_RATES)
@pytest.mark.parametrize("curve", ["A", "C"])
def test_a_tone_reads_the_printed_curve_through_the_filter(curve: str, fs: int) -> None:
    """A signal through the public entry point, against the analog prototype.

    The end-to-end counterpart of
    :func:`test_realised_filter_tracks_its_printed_prototype`, and the only
    thing in this file that would survive somebody putting a stage back in
    front of the sections.

    It is graded against the prototype and not against the printed Table 3
    goal, which is what makes it an accuracy test rather than a rounding test.
    Table 3 prints to 0.1 dB, so a filter reproducing the analytic curve
    exactly still reads up to 0.05 dB from the printed number -- the 158.5 Hz
    row's goal is -13.4 dB where the curve is at -13.3504 -- and a bound drawn
    around the printed goal is spent on that rounding before the filter is
    measured at all. Against the curve itself the whole sweep is inside
    0.0081 dB.
    """
    worst, where = _worst_tone_deviation_db(curve, fs)
    assert abs(worst) <= _WORST_TONE_DB, (
        f"{curve} at fs={fs}: {worst:+.5f} dB at {where:.1f} Hz"
    )


def test_the_tone_bound_is_not_slack() -> None:
    """:data:`_WORST_TONE_DB` has to stay close to what a tone really reads."""
    worst, _ = _worst_tone_deviation_db("A", 32000)
    assert abs(worst) >= 0.4 * _WORST_TONE_DB, (
        f"measured {abs(worst):.5f} dB against a bound of {_WORST_TONE_DB}; "
        "tighten the bound"
    )


def test_the_20_khz_reading_at_44_1_khz_is_the_published_one() -> None:
    """The one high-frequency number the changelog quotes, measured.

    A weighting at the commonest consumer rate, at the top of the audio band:
    a 20 kHz tone through a 44.1 kHz filter reads -9.3443 dB against the
    prototype's own -9.3469, 0.0026 dB out. Every part of that is load
    bearing. The frequency is 0.907 of the Nyquist frequency, where the warp
    this design exists to undo does its worst; 44.1 kHz is not one of the rates
    the Table 3 files sweep; and the oversampled path this branch replaced read
    -11.586 dB here, 2.24 dB low, which is the size of the thing being claimed.
    The closed-form design that still ships as ``high_accuracy=False`` reads
    -33.884 dB, which is the size of the thing the fit is for.

    Pinned to 0.005 dB rather than to the four decimals it is quoted at. The
    first three are a property of the design and hold on every machine; the
    fourth is a property of the arithmetic the fit ran on, which is fixed
    across CPUs but not across numpy releases.
    """
    reading = _tone_gain_db("A", 44100, 20000.0)
    zeros, poles, _ = _analog_weighting_zpk("A")
    printed = _analog_db(zeros, poles, np.array([20000.0]), 1000.0)[0]
    assert printed == pytest.approx(-9.3469, abs=1e-4)
    assert reading == pytest.approx(-9.3443, abs=0.005)
    assert reading == pytest.approx(printed, abs=0.005)


@pytest.mark.parametrize("fs", _CLASS_1_RATES)
@pytest.mark.parametrize("curve", ["A", "C"])
def test_a_and_c_earn_class_1_at_every_rate_the_library_claims(
    curve: str, fs: int
) -> None:
    """The verdict the prose promises, at all eight rates rather than four.

    The claim is a verdict, so a verdict is what is checked, but the margin is
    checked too and it is the more useful of the two. Every band clears its
    class 1 limit by 0.7 dB or better at every rate here, and 0.7 dB is not a
    near miss of anything: it is the *whole* of the 1 kHz row's budget, which
    the design spends none of because it is exact at the reference frequency.
    A verdict alone would still read class 1 with 0.69 dB of that gone.
    """
    result = filters.verify_weighting_class(filters.WeightingFilter(fs, curve))
    assert result["overall_class"] == 1, f"{curve} at fs={fs}"
    worst = min(band["margin_class1_db"] for band in result["bands"])
    assert worst == pytest.approx(0.7, abs=0.01), f"{curve} at fs={fs}: {worst:+.4f} dB"
    assert result["between_nominals"]["margin_class1_db"] > 0.9


def test_the_design_is_paid_for_once_per_curve_rate_and_mode() -> None:
    """The fit runs once and is shared, which is what makes it affordable.

    It costs about 150 ms, against 0.007 ms to hand back a cached copy, so a
    caller that constructs a filter per block would pay four orders of
    magnitude more for the same coefficients. A rate no other test uses, so
    the counters mean what they say however this file is sharded.
    """
    before = _cached_weighting_sos.cache_info()
    first = filters.WeightingFilter(37000, "A")
    second = filters.WeightingFilter(37000, "A")
    after = _cached_weighting_sos.cache_info()
    assert after.misses - before.misses == 1
    assert after.hits - before.hits == 1
    np.testing.assert_array_equal(first.sos, second.sos)
    # Shared, but not the same array: ``sos`` is public and callers edit it.
    assert first.sos is not second.sos
