#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the sonar equation (passive and active).

Oracles: a hand-worked textbook term balance (Urick via Etter, Table 10.2) —
pure arithmetic, independent of the implementation.
"""

from __future__ import annotations

import dataclasses
import math

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.sonar_equation import (
    SonarEquationResult,
    active_sonar_equation,
    array_directivity_index,
    detection_range_from_curve,
    detection_threshold,
    passive_sonar_equation,
)


def test_passive_signal_excess_and_fom() -> None:
    # SE = SL - PL - (NL - DI) - DT ; hand values.
    res = passive_sonar_equation(
        140.0, 80.0, 60.0, directivity_index=10.0, detection_threshold=5.0
    )
    assert isinstance(res, SonarEquationResult)
    assert res.mode == "passive"
    # SE = 140 - 80 - (60 - 10) - 5 = 5
    assert res.signal_excess[0] == pytest.approx(5.0)
    # SNR = SE + DT = 10
    assert res.snr[0] == pytest.approx(10.0)
    # FOM = SL - (NL - DI) - DT = 140 - 50 - 5 = 85
    assert res.figure_of_merit == pytest.approx(85.0)


def test_passive_detection_at_fom() -> None:
    # At PL = FOM the signal excess is exactly zero (detection limit).
    res = passive_sonar_equation(
        140.0, [85.0], 60.0, directivity_index=10.0, detection_threshold=5.0
    )
    assert res.signal_excess[0] == pytest.approx(0.0, abs=1e-9)


def test_active_noise_limited() -> None:
    # SE = SL - 2 PL + TS - (NL - DI) - DT
    res = active_sonar_equation(
        220.0, 70.0, 15.0, 60.0, directivity_index=20.0, detection_threshold=10.0
    )
    assert res.mode == "active"
    # SE = 220 - 140 + 15 - (60 - 20) - 10 = 45
    assert res.signal_excess[0] == pytest.approx(45.0)
    # FOM = (SL + TS - (NL - DI) - DT)/2 = (220 + 15 - 40 - 10)/2 = 92.5
    assert res.figure_of_merit == pytest.approx(92.5)


def test_active_reverberation_limited_ignores_di() -> None:
    # With RL given, masking is RL (DI does not apply to reverberation).
    res = active_sonar_equation(
        220.0,
        70.0,
        15.0,
        60.0,
        directivity_index=20.0,
        detection_threshold=10.0,
        reverberation_level=55.0,
    )
    assert res.reverberation_limited is True
    # SE = 220 - 140 + 15 - 55 - 10 = 30
    assert res.signal_excess[0] == pytest.approx(30.0)
    assert res.figure_of_merit == pytest.approx((220.0 + 15.0 - 55.0 - 10.0) / 2.0)


def test_signal_excess_decreases_with_propagation_loss() -> None:
    pl = np.linspace(50.0, 120.0, 8)
    res = passive_sonar_equation(150.0, pl, 55.0)
    assert np.all(np.diff(res.signal_excess) < 0.0)
    # Passive SE decreases 1 dB per dB of one-way PL.
    np.testing.assert_allclose(np.diff(res.signal_excess), -np.diff(pl))


def test_active_two_way_loss() -> None:
    # Active SE loses 2 dB per dB of one-way PL.
    res = active_sonar_equation(200.0, [60.0, 61.0], 10.0, 50.0)
    assert res.signal_excess[1] - res.signal_excess[0] == pytest.approx(-2.0)


def test_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match=r"'source_level'.*finite"):
        passive_sonar_equation(float("nan"), 80.0, 60.0)


def test_plot_smoke() -> None:
    res = passive_sonar_equation(
        150.0, np.linspace(40.0, 110.0, 40), 55.0, detection_threshold=8.0
    )
    assert res.plot() is not None


_PER_LOSS = "one value per propagation loss"


def test_solution_columns_must_run_over_one_loss_axis() -> None:
    """A solution off its own loss axis is refused when built, not when read.

    ``.plot()`` sorts by propagation loss and reads the signal excess through
    that sort order, so a short ``signal_excess`` surfaces only as numpy's
    "index 3 is out of bounds for axis 0 with size 3" -- an axis and a size,
    naming neither field. A long one is silent: the sort order holds one entry
    per loss, so the tail is dropped without a word, and the tail is the half
    that carries the negative excesses. ``snr`` reaches no figure at all and
    is silent in both directions, yet it is read entry by entry beside
    ``propagation_loss``. An extra axis is quieter still: an ``(n, 2)`` column
    carries one value per loss by every count, which is why the three are held
    to one shape rather than to one length.
    """
    good = passive_sonar_equation(
        150.0, np.linspace(60.0, 110.0, 6), 57.0, detection_threshold=8.0
    )
    cases = (
        ("propagation_loss", good.propagation_loss[:-1], _PER_LOSS),
        ("signal_excess", good.signal_excess[:-1], _PER_LOSS),
        ("signal_excess", np.append(good.signal_excess, -20.0), _PER_LOSS),
        ("snr", good.snr[:-1], _PER_LOSS),
        ("snr", np.append(good.snr, -12.0), _PER_LOSS),
        ("propagation_loss", np.column_stack([good.propagation_loss] * 2), _PER_LOSS),
        ("signal_excess", np.column_stack([good.signal_excess] * 2), _PER_LOSS),
        ("snr", np.column_stack([good.snr] * 2), _PER_LOSS),
    )
    for field, value, fragment in cases:
        with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
            dataclasses.replace(good, **{field: value})


def test_entry_points_carry_a_grid_of_losses_through() -> None:
    """A loss field of two axes is a detection map, and it survives intact.

    ``gaussian_beams`` and ``parabolic_equation`` hand back a loss over depth
    and range, and feeding that to the sonar equation is how a detection
    footprint is drawn. The equation is elementwise, so every derived quantity
    keeps the grid's shape and each cell is the scalar result of the cell it
    came from. This is why the three columns are held to one shape and not
    merely to one length: pinning a single axis would refuse the grid the
    library itself produces.
    """
    grid = np.array([[60.0, 70.0], [80.0, 90.0], [100.0, 110.0]])
    for result in (
        passive_sonar_equation(150.0, grid, 57.0),
        active_sonar_equation(200.0, grid, 10.0, 50.0),
    ):
        assert result.propagation_loss.shape == grid.shape
        assert result.signal_excess.shape == grid.shape
        assert result.snr.shape == grid.shape
    # Passive: SNR = SL - PL - (NL - DI), cell by cell.
    passive = passive_sonar_equation(150.0, grid, 57.0)
    np.testing.assert_allclose(passive.snr, 150.0 - grid - 57.0)


def test_a_grid_that_disagrees_with_its_solutions_is_refused() -> None:
    """Two grids of the same height can still disagree everywhere else.

    A length check counts first axes alone, so a ``(3, 2)`` loss beside a
    ``(3, 4)`` excess passes it while agreeing about nothing. The shape is what
    the elementwise equation actually needs.
    """
    good = passive_sonar_equation(
        150.0, np.array([[60.0, 70.0], [80.0, 90.0], [100.0, 110.0]]), 57.0
    )
    with pytest.raises(ValueError, match=rf"'signal_excess'.*{_PER_LOSS}"):
        dataclasses.replace(good, signal_excess=np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# Ainslie (2010) worked-example figure-of-merit oracles
# ---------------------------------------------------------------------------

# Ainslie, Principles of Sonar Performance Modelling (Springer, 2010),
# publishes complete term tables for his sonar-equation worked examples. The
# term balances below re-combine those printed values through the library's
# equations; every expected number is the book's, so these anchor the sign
# conventions of the SL/NL/DI/DT/TS combination against an independent source.


def test_ainslie_passive_narrowband_fom() -> None:
    # Table 3.1 (p. 76), passive narrowband example of Sec. 3.2.3.8:
    # SL = 133.9 dB re uPa2 m2, NLf = 59.7 dB re uPa2/Hz, AG = 11.5 dB,
    # DT = 13.8 dB, analysis bandwidth 0.25 Hz (the printed "BW = -6.0 dB"
    # row) -> FOM = 78.0 dB re m2. The bandwidth term folds into the masking
    # noise as NL = NLf + 10 lg(0.25 Hz) = 59.7 - 6.0 dB. Five printed
    # terms, each rounded to 0.1 dB, give a 0.15 dB accumulation allowance.
    res = passive_sonar_equation(
        133.9,
        0.0,
        59.7 - 6.0,
        directivity_index=11.5,
        detection_threshold=13.8,
    )
    assert res.figure_of_merit == pytest.approx(78.0, abs=0.15)


def test_ainslie_passive_broadband_fom() -> None:
    # Table 3.2 (p. 90), passive broadband example of Sec. 3.2.4.8 (spectral
    # density form, no bandwidth term): SLf = 100.9 dB re uPa2 m2/Hz,
    # NLf = 53.2 dB re uPa2/Hz, AGm = 12.8 dB, DT = -18.6 dB ->
    # FOM = 79.0 dB re m2 (four printed 0.1 dB terms -> 0.15 dB allowance).
    res = passive_sonar_equation(
        100.9,
        0.0,
        53.2,
        directivity_index=12.8,
        detection_threshold=-18.6,
    )
    assert res.figure_of_merit == pytest.approx(79.0, abs=0.15)


def test_ainslie_active_orca_noise_limited_fom() -> None:
    # Sec. 11.4.6 (orca vs salmon, Tables 11.6-11.7 pp. 620-624): SL(RMS) =
    # 198.2 dB re uPa2 m2, TS(salmon, 0.8 m) = -29.0 dB re m2, wind noise
    # NL = 75.0 dB re uPa2, AG = 16.5 dB, DT = 8.7 dB -> noise-limited
    # FOM_NL = (SL + TS - (NL - AG) - DT)/2 = 51.0 dB re m2.
    res = active_sonar_equation(
        198.2,
        0.0,
        -29.0,
        75.0,
        directivity_index=16.5,
        detection_threshold=8.7,
    )
    assert res.figure_of_merit == pytest.approx(51.0, abs=0.05)


def test_ainslie_active_orca_hearing_threshold_fom() -> None:
    # Same example: against the orca's hearing threshold at 50 kHz,
    # HT = 51.2 dB re uPa2 (audiogram Eq. 11.159), the book's
    # FOM_HT = (SL + TS - HT)/2 = 59.0 dB re m2; the threshold acts as the
    # masking level with no array gain and no detection threshold.
    res = active_sonar_equation(198.2, 0.0, -29.0, 51.2)
    assert res.figure_of_merit == pytest.approx(59.0, abs=0.05)


def test_a_detection_curve_of_two_axes_is_refused_by_name() -> None:
    """A curve is one axis, and the crossing search cannot answer for a grid.

    Two equal two-dimensional inputs agree on their shape, so only the rank
    sees them. Without it the interpolation of the bracketing pair ended in
    numpy's "only 0-dimensional arrays can be converted to Python scalars",
    which names neither this function nor the argument that was wrong.
    """
    ranges = np.array([100.0, 200.0, 300.0, 400.0])
    losses = np.array([40.0, 50.0, 60.0, 70.0])
    assert detection_range_from_curve(55.0, ranges, losses) == pytest.approx(250.0)
    grid_r, grid_pl = ranges.reshape(2, 2), losses.reshape(2, 2)
    with pytest.raises(ValueError, match=r"'range_m' must have one axis"):
        detection_range_from_curve(55.0, grid_r, grid_pl)


# ---------------------------------------------------------------------------
# Directivity index and detection threshold (Ainslie 2010).
#
# The oracle for the directivity index is the book's own closed-form
# approximation, Equation (11.20) plotted as the dashed line of Figure 11.1
# against the full Chapter 6 integral:
#
#     G ~ 1 + G0 tanh(pi^2 G0 / 36),   G0 = 2L/lambda
#
# It is a different expression from the one implemented, fitted rather than
# derived, so agreeing with it to within the spread the figure shows is an
# independent check rather than a restatement. The three stated limits of
# Section 6.1.2.1 are checked separately, since they are exact.
# ---------------------------------------------------------------------------
def _di_approximation_db(g0: float) -> float:
    """Ainslie Eq. (11.20), the dashed curve of Figure 11.1."""
    return 10.0 * math.log10(1.0 + g0 * math.tanh(math.pi**2 * g0 / 36.0))


@pytest.mark.parametrize("g0", [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
def test_directivity_index_tracks_the_books_own_approximation(g0: float) -> None:
    """The integral and Eq. (11.20) agree over the range Figure 11.1 plots."""
    exact = array_directivity_index(array_length_m=g0 / 2.0, wavelength_m=1.0)
    assert exact == pytest.approx(_di_approximation_db(g0), abs=0.5)


def test_directivity_index_reaches_its_three_printed_limits() -> None:
    """Section 6.1.2.1 states all three, and they are exact rather than fitted.

    High frequency gives 2L/lambda for every steer direction but endfire, where
    the footprint halves and the factor doubles to 4L/lambda; low frequency
    gives unity, since an array much shorter than a wavelength resolves
    nothing at all.
    """
    length, wavelength = 100.0, 1.0
    broadside = array_directivity_index(length, wavelength)
    endfire = array_directivity_index(length, wavelength, steer_angle_rad=math.pi / 2.0)
    assert broadside == pytest.approx(
        10.0 * math.log10(2.0 * length / wavelength), abs=0.01
    )
    assert endfire == pytest.approx(
        10.0 * math.log10(4.0 * length / wavelength), abs=0.01
    )
    # Endfire buys exactly the 3 dB the halved footprint is worth.
    assert endfire - broadside == pytest.approx(10.0 * math.log10(2.0), abs=0.01)
    # The low-frequency limit is approached, not reached at any finite length,
    # so it is checked as the convergence it is.
    short = [array_directivity_index(length, 1.0) for length in (1e-2, 1e-3, 1e-4)]
    assert short == sorted(short, reverse=True)
    assert 0.0 < short[-1] < 1.0e-7


@pytest.mark.parametrize(
    ("length", "wavelength"),
    [
        (5e-324, 1.0),
        (1.0, 1.7976931348623157e308),
        (5e-324, 1.7976931348623157e308),
        (1e-200, 1e100),
    ],
)
def test_an_array_too_short_to_measure_returns_its_limit(
    length: float, wavelength: float
) -> None:
    """The dimensions are valid, so the answer is the limit and not an error.

    ``L/lambda`` underflows to nought for these, and the closed form divides by
    it: the first two used to raise a `math` domain error and the third a
    `ZeroDivisionError`, from a length and a wavelength that are both positive
    and finite. Cancelling that factor by hand leaves the low-frequency limit,
    which is what an array this short is worth.
    """
    assert array_directivity_index(length, wavelength) == 0.0


def test_the_low_frequency_limit_is_not_a_cutoff_at_one_wavelength() -> None:
    """An array a wavelength long still resolves something.

    The docstring used to read "0 dB below a wavelength", which is the limit
    described as a threshold. A finite array of one wavelength returns
    3.45 dB and half a wavelength 1.11 dB, both far from nought, and the
    approach to nought is smooth rather than a step.
    """
    assert array_directivity_index(1.0, 1.0) == pytest.approx(3.4543, abs=1e-4)
    assert array_directivity_index(0.5, 1.0) == pytest.approx(1.1143, abs=1e-4)
    ratios = [1.0, 0.5, 0.2, 0.1, 0.01]
    values = [array_directivity_index(r, 1.0) for r in ratios]
    assert values == sorted(values, reverse=True)
    assert all(value > 0.0 for value in values)


def test_directivity_index_is_symmetric_about_broadside() -> None:
    """Only the sine of the steer angle enters, so the two sides agree."""
    for psi in (0.2, 0.7, 1.3):
        assert array_directivity_index(5.0, 1.0, steer_angle_rad=psi) == pytest.approx(
            array_directivity_index(5.0, 1.0, steer_angle_rad=-psi), rel=1e-12
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"array_length_m": 0.0}, "'array_length_m' must be positive"),
        ({"array_length_m": float("nan")}, "'array_length_m' must be positive"),
        ({"wavelength_m": 0.0}, "'wavelength_m' must be positive"),
        ({"wavelength_m": -1.0}, "'wavelength_m' must be positive"),
        ({"steer_angle_rad": float("inf")}, "'steer_angle_rad' must be"),
    ],
)
def test_directivity_index_refuses_what_is_not_an_array(
    kwargs: dict[str, float], match: str
) -> None:
    base = {"array_length_m": 5.0, "wavelength_m": 1.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        array_directivity_index(**base)  # type: ignore[arg-type]


def test_detection_threshold_reproduces_equation_11_22() -> None:
    """DT50 = 10 log10(log2(1/(2 pfa))) - 0,8 dB, printed on folio 581.

    The inner logarithm is base two, not a square: the two are typeset alike
    and the distinction is worth a test of its own, since taking it as a
    square would give 10,5 dB where the book gives 10,1 at pfa = 1e-4.
    """
    for p_fa in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6):
        expected = 10.0 * math.log10(math.log2(1.0 / (2.0 * p_fa))) - 0.8
        assert detection_threshold(p_fa) == pytest.approx(expected, rel=1e-12)
    # And it is not the squared-logarithm reading.
    squared = 10.0 * math.log10(math.log10(1.0 / (2.0e-4)) ** 2) - 0.8
    assert detection_threshold(1.0e-4) != pytest.approx(squared, abs=0.05)


def test_detection_threshold_rises_as_false_alarms_get_rarer() -> None:
    """Demanding fewer false alarms costs signal-to-noise ratio."""
    values = [detection_threshold(p) for p in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6)]
    assert values == sorted(values)


@pytest.mark.parametrize("bad", [0.0, 0.5, 0.7, 1.0, -0.1, float("nan")])
def test_detection_threshold_refuses_an_impossible_false_alarm_rate(
    bad: float,
) -> None:
    """At one half the inner logarithm is zero and the threshold diverges."""
    with pytest.raises(ValueError, match="'false_alarm_probability' must"):
        detection_threshold(bad)
