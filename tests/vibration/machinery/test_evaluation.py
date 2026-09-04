#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the evaluation of machine vibration (ISO 20816-1:2016).

Oracle: ISO 20816-1:2016, *Mechanical vibration - Measurement and evaluation
of machine vibration - Part 1: General guidelines*. The numbers come from
Table C.1 on printed folio 29, Annex C.2 on printed folio 30 and the worked
case of Annex D.2 on printed folio 31.

The series states its shape in this part and its numbers in the others: the
four evaluation zones, the frequency-shaped criterion of Figure 9 and the
vector reading of a change are all here, while the boundaries that grade a
particular machine are in Parts 2 to 9. What Part 1 does print are the
typical ranges of Table C.1, for machines no other part covers, and one
worked vector change.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import vibration

#: Annex D.2: the initial state, 3 mm/s at 40 degrees.
INITIAL = (3.0, 40.0)
#: Annex D.2: the later state, 2,5 mm/s at 180 degrees.
FINAL = (2.5, 180.0)
#: Annex D.2: "the true change of vibration is represented by the vector
#: A2 - A1, which has an r.m.s. magnitude of 5,2 mm/s".
PRINTED_VECTOR_CHANGE_MM_S = 5.2
#: The annex prints one decimal.
TOLERANCE_MM_S = 0.05


# --------------------------------------------------------------------------- #
# The four zones (6.3.2.3) and their boundaries
# --------------------------------------------------------------------------- #
def test_a_magnitude_is_graded_into_one_of_the_four_zones() -> None:
    """Zone A up to the first boundary, D above the last."""
    boundaries = vibration.ZoneBoundaries(2.8, 7.1, 11.2)
    graded = vibration.evaluation_zone([0.5, 3.0, 8.0, 20.0], boundaries)
    assert list(graded) == ["A", "B", "C", "D"]


def test_a_boundary_belongs_to_the_zone_below_it() -> None:
    """A limit is the top of its own zone, which is how the parts print them.

    Table 3 of a machine-specific part reads "zone A/B: 2,8 mm/s", and a
    machine measured at exactly 2,8 meets zone A rather than falling out of
    it.
    """
    boundaries = vibration.ZoneBoundaries(2.8, 7.1, 11.2)
    assert vibration.evaluation_zone(2.8, boundaries) == "A"
    assert vibration.evaluation_zone(7.1, boundaries) == "B"
    assert vibration.evaluation_zone(11.2, boundaries) == "C"
    assert vibration.evaluation_zone(11.20001, boundaries) == "D"


def test_a_scalar_is_graded_to_a_string_and_an_array_to_an_array() -> None:
    """The shape of the answer follows the shape of the question."""
    boundaries = vibration.ZoneBoundaries(1.0, 2.0, 3.0)
    assert isinstance(vibration.evaluation_zone(0.5, boundaries), str)
    graded = vibration.evaluation_zone([0.5, 2.5], boundaries)
    assert isinstance(graded, np.ndarray)
    assert graded.shape == (2,)


@pytest.mark.parametrize(
    ("a_b", "b_c", "c_d"),
    [(7.1, 2.8, 11.2), (2.8, 11.2, 7.1), (2.8, 2.8, 7.1)],
)
def test_boundaries_that_do_not_rise_are_refused(
    a_b: float, b_c: float, c_d: float
) -> None:
    """Out of order they would grade a good machine as a bad one, silently."""
    with pytest.raises(ValueError, match="must increase through the zones"):
        vibration.ZoneBoundaries(a_b, b_c, c_d)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_a_boundary_that_is_not_a_magnitude_is_refused(bad: float) -> None:
    """Every boundary is a positive, finite magnitude.

    A nought or a negative is refused as not positive; a NaN or an infinity is
    refused as not finite, and by the field guard rather than the ordering one,
    since NaN compares false against everything it is asked about.
    """
    with pytest.raises(ValueError, match="a_b|must be positive|must be finite"):
        vibration.ZoneBoundaries(bad, 7.1, 11.2)


def test_a_magnitude_that_is_not_a_magnitude_is_refused() -> None:
    """Negative and non-finite alike: neither is a vibration severity."""
    boundaries = vibration.ZoneBoundaries(2.8, 7.1, 11.2)
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="non-negative and finite"):
            vibration.evaluation_zone(bad, boundaries)


# --------------------------------------------------------------------------- #
# Table C.1: the ladder and the three typical ranges
# --------------------------------------------------------------------------- #
def test_table_c1_prints_the_ladder_it_draws_its_ranges_from() -> None:
    """Fourteen rungs from 0,28 to 45 mm/s, rising and each one printed."""
    ladder = vibration.TYPICAL_BOUNDARY_LADDER_MM_S
    assert ladder[0] == pytest.approx(0.28)
    assert ladder[-1] == pytest.approx(45.0)
    assert list(ladder) == sorted(ladder)
    assert len(ladder) == 14


def test_the_three_typical_ranges_overlap_the_way_the_table_draws_them() -> None:
    """A/B 0,71 to 4,5; B/C 1,8 to 9,3; C/D 4,5 to 14,7.

    The ranges overlap on purpose: a large machine's A/B boundary can sit
    above a small machine's B/C one, which is why a range is not a limit.
    """
    ranges = vibration.TYPICAL_ZONE_BOUNDARY_RANGES_MM_S
    assert ranges["A/B"] == (0.71, 4.5)
    assert ranges["B/C"] == (1.8, 9.3)
    assert ranges["C/D"] == (4.5, 14.7)
    ladder = set(vibration.TYPICAL_BOUNDARY_LADDER_MM_S)
    for low, high in ranges.values():
        assert low in ladder
        assert high in ladder
    assert ranges["A/B"][1] > ranges["B/C"][0]
    assert ranges["B/C"][1] > ranges["C/D"][0]


def test_the_low_end_of_each_range_grades_a_small_machine() -> None:
    """Note 2: a 15 kW motor sits at the low end, a large one at the high end."""
    small = vibration.ZoneBoundaries(0.71, 1.8, 4.5)
    large = vibration.ZoneBoundaries(4.5, 9.3, 14.7)
    assert vibration.evaluation_zone(3.0, small) == "C"
    assert vibration.evaluation_zone(3.0, large) == "A"


# --------------------------------------------------------------------------- #
# Formula (C.1): the frequency-shaped criterion of Figure 9
# --------------------------------------------------------------------------- #
def test_the_criterion_is_flat_between_the_two_corners() -> None:
    """Both bracketed factors are unity there, so the curve is the plain vA."""
    v = vibration.allowable_velocity(
        [10.0, 50.0, 200.0, 1000.0],
        constant_velocity_mm_s=1.12,
        corner_low_hz=10.0,
        corner_high_hz=1000.0,
    )
    assert v == pytest.approx([1.12] * 4)


def test_below_the_lower_corner_the_criterion_holds_displacement_constant() -> None:
    """With k = 1 the allowable velocity falls with frequency, halving per octave."""
    v = np.asarray(
        vibration.allowable_velocity(
            [2.5, 5.0, 10.0],
            constant_velocity_mm_s=1.12,
            corner_low_hz=10.0,
            corner_high_hz=1000.0,
        )
    )
    assert v == pytest.approx([0.28, 0.56, 1.12])


def test_above_the_upper_corner_the_criterion_holds_acceleration_constant() -> None:
    """With m = 1 it falls with the reciprocal, halving per octave the other way."""
    v = np.asarray(
        vibration.allowable_velocity(
            [1000.0, 2000.0, 4000.0],
            constant_velocity_mm_s=1.12,
            corner_low_hz=10.0,
            corner_high_hz=1000.0,
        )
    )
    assert v == pytest.approx([1.12, 0.56, 0.28])


def test_the_printed_zone_factors_move_the_one_curve_onto_the_three_limits() -> None:
    """Annex C.2 prints 1, 2,56 and 6,4, and they scale the whole curve.

    The factors are 1,6 squared and 1,6 to the fourth, near enough, and the
    Table C.1 ladder steps by about 1,6 as well: from a zone A of 1,12 mm/s
    the B and C limits come out at 2,87 and 7,17, within 3 % of the 2,8 and
    7,1 rungs. Near, and not the same number, which is why the factors are
    kept as the standard prints them rather than snapped to the ladder.
    """
    factors = vibration.ZONE_LIMIT_FACTORS
    assert (factors["A"], factors["B"], factors["C"]) == (1.0, 2.56, 6.4)
    common = {
        "constant_velocity_mm_s": 1.12,
        "corner_low_hz": 10.0,
        "corner_high_hz": 1000.0,
    }
    limits = [
        float(vibration.allowable_velocity(100.0, zone_factor=z, **common))  # type: ignore[arg-type]
        for z in (factors["A"], factors["B"], factors["C"])
    ]
    assert limits == pytest.approx([1.12, 2.8672, 7.168])
    for limit, rung in ((limits[1], 2.8), (limits[2], 7.1)):
        assert rung in vibration.TYPICAL_BOUNDARY_LADDER_MM_S
        assert abs(limit - rung) / rung < 0.03


def test_the_criterion_never_rises_above_its_plateau() -> None:
    """Both slopes point down, so the plateau is the whole curve's maximum."""
    freq = np.logspace(0.0, 4.0, 200)
    v = np.asarray(
        vibration.allowable_velocity(
            freq,
            constant_velocity_mm_s=1.12,
            corner_low_hz=10.0,
            corner_high_hz=1000.0,
        )
    )
    assert float(v.max()) == pytest.approx(1.12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"constant_velocity_mm_s": 0.0}, "constant_velocity_mm_s"),
        ({"zone_factor": -1.0}, "zone_factor"),
        ({"corner_low_hz": 0.0}, "corner_low_hz"),
        ({"corner_high_hz": 5.0}, "must exceed"),
        ({"exponent_low": math.nan}, "exponent_low"),
    ],
)
def test_the_criterion_refuses_a_shape_it_cannot_draw(
    kwargs: dict[str, float], match: str
) -> None:
    """Each guard names the argument it is about."""
    common = {
        "constant_velocity_mm_s": 1.12,
        "corner_low_hz": 10.0,
        "corner_high_hz": 1000.0,
    }
    with pytest.raises(ValueError, match=match):
        vibration.allowable_velocity(100.0, **{**common, **kwargs})  # type: ignore[arg-type]


def test_a_frequency_that_is_not_a_frequency_is_refused() -> None:
    """Nought and negatives have no place on a logarithmic criterion."""
    with pytest.raises(ValueError, match="'frequency' must be positive"):
        vibration.allowable_velocity(
            [10.0, 0.0],
            constant_velocity_mm_s=1.12,
            corner_low_hz=10.0,
            corner_high_hz=1000.0,
        )


# --------------------------------------------------------------------------- #
# Annex D: the change is a vector
# --------------------------------------------------------------------------- #
def test_annex_d_reproduces_its_printed_vector_change() -> None:
    """3 mm/s at 40 deg becoming 2,5 mm/s at 180 deg is a change of 5,2 mm/s."""
    result = vibration.vibration_vector_change(*INITIAL, *FINAL)
    assert result.magnitude == pytest.approx(
        PRINTED_VECTOR_CHANGE_MM_S, abs=TOLERANCE_MM_S
    )


def test_the_magnitude_comparison_the_annex_warns_against() -> None:
    """It reports half a millimetre per second, and a fall at that.

    "although the vibration magnitude has decreased by 0,5 mm/s ... the true
    change ... is over ten times that indicated by comparing the vibration
    magnitude alone."
    """
    result = vibration.vibration_vector_change(*INITIAL, *FINAL)
    assert result.magnitude_change == pytest.approx(-0.5)
    assert result.magnitude > 10.0 * abs(result.magnitude_change)


def test_a_pure_change_of_phase_moves_the_vibration_without_moving_its_size() -> None:
    """Two states of equal magnitude in antiphase differ by twice that magnitude."""
    result = vibration.vibration_vector_change(4.0, 0.0, 4.0, 180.0)
    assert result.magnitude_change == pytest.approx(0.0)
    assert result.magnitude == pytest.approx(8.0)


def test_the_change_is_reported_with_a_direction_of_its_own() -> None:
    """The phase of the difference, wrapped into [0, 360)."""
    result = vibration.vibration_vector_change(1.0, 0.0, 2.0, 0.0)
    assert result.magnitude == pytest.approx(1.0)
    assert result.phase_deg == pytest.approx(0.0)
    back = vibration.vibration_vector_change(2.0, 0.0, 1.0, 0.0)
    assert back.magnitude == pytest.approx(1.0)
    assert back.phase_deg == pytest.approx(180.0)


def test_the_two_states_are_carried_on_the_result() -> None:
    """A plot needs the states, not only their difference."""
    result = vibration.vibration_vector_change(*INITIAL, *FINAL)
    assert result.initial == pytest.approx(INITIAL)
    assert result.final == pytest.approx(FINAL)


@pytest.mark.parametrize(
    "args",
    [
        (-1.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, -1.0, 0.0),
        (math.nan, 0.0, 1.0, 0.0),
        (1.0, math.inf, 1.0, 0.0),
        (1.0, 0.0, 1.0, math.nan),
    ],
)
def test_a_state_that_is_not_a_state_is_refused(args: tuple[float, ...]) -> None:
    """A magnitude is non-negative and every value is finite."""
    with pytest.raises(ValueError, match="magnitude|phase_deg"):
        vibration.vibration_vector_change(*args)
