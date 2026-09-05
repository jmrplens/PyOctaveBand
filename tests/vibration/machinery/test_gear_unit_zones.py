#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Gear-unit evaluation by ISO 20816-9:2020.

The three boundary tables, the classification that picks a row of each, and
the two rating curves of Annex A. The oracle is the printed page: Tables 2, 3
and 4 on printed folios 7 and 8, Table 5 on printed folio 9, and the notes
under Figures A.1 and A.2 on printed folios 12 and 13.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import vibration

#: Tables 2, 3 and 4, transcribed a second time so a change to the library's
#: copy has to be made twice to pass.
PRINTED_ZONES: dict[str, dict[float, tuple[float, float, float]]] = {
    "displacement": {
        31.5: (20.0, 31.5, 50.0),
        50.0: (31.5, 50.0, 80.0),
        80.0: (50.0, 80.0, 125.0),
        125.0: (80.0, 125.0, 200.0),
        200.0: (125.0, 200.0, 315.0),
    },
    "velocity": {
        3.15: (2.0, 3.15, 5.0),
        5.0: (3.15, 5.0, 8.0),
        8.0: (5.0, 8.0, 12.5),
        12.5: (8.0, 12.5, 20.0),
        20.0: (12.5, 20.0, 31.5),
    },
    "acceleration": {
        5.0: (3.15, 5.0, 8.0),
        8.0: (5.0, 8.0, 12.5),
        12.5: (8.0, 12.5, 20.0),
        20.0: (12.5, 20.0, 31.5),
        31.5: (20.0, 31.5, 50.0),
        50.0: (31.5, 50.0, 80.0),
        80.0: (50.0, 80.0, 125.0),
        125.0: (80.0, 125.0, 200.0),
        200.0: (125.0, 200.0, 315.0),
    },
}

#: Table 5, class and subclass to (DR, VR, AR).
PRINTED_CLASSES: dict[tuple[str, str], tuple[float, float, float | None]] = {
    ("I", "a"): (31.5, 3.15, 50.0),
    ("I", "b_low"): (31.5, 3.15, None),
    ("I", "b_high"): (50.0, 5.0, None),
    ("II", "a"): (50.0, 5.0, 80.0),
    ("II", "b_low"): (50.0, 5.0, None),
    ("II", "b_high"): (80.0, 8.0, None),
    ("III", "a"): (80.0, 8.0, 125.0),
    ("III", "b_low"): (80.0, 8.0, None),
    ("III", "b_high"): (125.0, 12.5, None),
    ("IV", "a"): (125.0, 20.0, 125.0),
    ("IV", "b_low"): (125.0, 12.5, None),
    ("IV", "b_high"): (200.0, 20.0, None),
}


@pytest.mark.parametrize("quantity", sorted(PRINTED_ZONES))
def test_every_printed_row_of_the_three_tables(quantity: str) -> None:
    for rating, printed in PRINTED_ZONES[quantity].items():
        assert vibration.gear_unit_zone_boundaries(quantity, rating).as_tuple == printed


def test_the_tables_hold_no_row_the_document_does_not_print() -> None:
    for quantity, printed in PRINTED_ZONES.items():
        assert set(vibration.GEAR_UNIT_ZONES[quantity]) == set(printed)


def test_a_rating_between_two_rows_is_refused_rather_than_interpolated() -> None:
    with pytest.raises(ValueError, match="prints no velocity row"):
        vibration.gear_unit_zone_boundaries("velocity", 4.0)


def test_an_unknown_quantity_is_refused() -> None:
    with pytest.raises(ValueError, match="'quantity' must be one of"):
        vibration.gear_unit_zone_boundaries("jerk", 5.0)


def test_a_rating_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="'rating'"):
        vibration.gear_unit_zone_boundaries("velocity", 0.0)


def test_every_row_is_three_consecutive_rungs_of_one_ladder() -> None:
    """The rating is the B/C boundary, and its neighbours are the other two.

    Stated by the tables rather than by the text, and worth pinning: it is
    what makes a rating a choice on a ladder instead of three numbers to
    look up, and a transcription slip in any of the 57 printed cells breaks
    it.
    """
    ladder = [2.0, 3.15, 5.0, 8.0, 12.5, 20.0, 31.5, 50.0, 80.0, 125.0, 200.0, 315.0]
    for table in vibration.GEAR_UNIT_ZONES.values():
        for rating, boundaries in table.items():
            index = ladder.index(rating)
            assert boundaries.as_tuple == (
                ladder[index - 1],
                ladder[index],
                ladder[index + 1],
            )


@pytest.mark.parametrize(("key", "printed"), sorted(PRINTED_CLASSES.items()))
def test_every_row_of_the_classification(
    key: tuple[str, str], printed: tuple[float, float, float | None]
) -> None:
    ratings = vibration.GEAR_UNIT_CLASSES[key]
    assert (ratings.displacement, ratings.velocity, ratings.acceleration) == printed


def test_every_class_rating_indexes_a_row_of_its_table() -> None:
    """A classification that pointed at a rating with no row would be useless."""
    for ratings in vibration.GEAR_UNIT_CLASSES.values():
        assert vibration.gear_unit_zone_boundaries("displacement", ratings.displacement)
        assert vibration.gear_unit_zone_boundaries("velocity", ratings.velocity)
        if ratings.acceleration is not None:
            assert vibration.gear_unit_zone_boundaries(
                "acceleration", ratings.acceleration
            )


def test_only_the_subclass_a_rows_carry_an_acceleration_rating() -> None:
    """Table 5 prints "no information available at this time" for every b) row."""
    for (_gear_class, subclass), ratings in vibration.GEAR_UNIT_CLASSES.items():
        assert (ratings.acceleration is None) is subclass.startswith("b")


def test_the_displacement_curve_is_flat_to_its_corner_then_falls() -> None:
    rating = 80.0
    for frequency in (1.0, 10.0, 50.0):
        assert vibration.gear_shaft_displacement_limit(
            frequency, rating=rating
        ) == pytest.approx(rating)
    # 10 dB per decade on an amplitude is a factor of sqrt(10) per decade.
    assert vibration.gear_shaft_displacement_limit(500.0, rating=rating) == (
        pytest.approx(rating / 10.0**0.5)
    )
    assert vibration.gear_shaft_displacement_limit(5000.0, rating=rating) == (
        pytest.approx(rating / 10.0)
    )


def test_the_velocity_curve_is_flat_between_its_corners_then_falls() -> None:
    rating = 5.0
    for frequency in (45.0, 300.0, 1590.0):
        assert vibration.gear_housing_velocity_limit(
            frequency, rating=rating
        ) == pytest.approx(rating)
    # 14 dB per decade, below the lower corner and above the upper one.
    assert vibration.gear_housing_velocity_limit(4.5, rating=rating) == pytest.approx(
        rating * 10.0 ** (-14.0 / 20.0)
    )
    assert vibration.gear_housing_velocity_limit(15900.0, rating=rating) == (
        pytest.approx(rating * 10.0 ** (-14.0 / 20.0))
    )


def test_the_two_curves_take_arrays_and_keep_their_shape() -> None:
    freq = np.array([10.0, 50.0, 500.0])
    displacement = vibration.gear_shaft_displacement_limit(freq, rating=125.0)
    velocity = vibration.gear_housing_velocity_limit(freq, rating=12.5)
    assert isinstance(displacement, np.ndarray)
    assert isinstance(velocity, np.ndarray)
    assert displacement.shape == freq.shape
    assert velocity.shape == freq.shape


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.inf, np.nan])
def test_a_frequency_that_is_not_positive_and_finite_is_refused(
    frequency: float,
) -> None:
    with pytest.raises(ValueError, match="'frequency'"):
        vibration.gear_shaft_displacement_limit(frequency, rating=80.0)


def test_grading_a_gear_unit_against_the_row_its_class_picks() -> None:
    """Class III a), an epicyclic unit: VR 8, so zone B ends at 8 mm/s."""
    ratings = vibration.GEAR_UNIT_CLASSES["III", "a"]
    boundaries = vibration.gear_unit_zone_boundaries("velocity", ratings.velocity)
    assert vibration.evaluation_zone(4.0, boundaries) == "A"
    assert vibration.evaluation_zone(6.0, boundaries) == "B"
    assert vibration.evaluation_zone(10.0, boundaries) == "C"
    assert vibration.evaluation_zone(20.0, boundaries) == "D"


def test_an_acceptance_criterion_sits_at_most_a_quarter_above_the_a_b_boundary() -> (
    None
):
    """8.3: normally not more than 1,25 times the A/B boundary."""
    boundaries = vibration.gear_unit_zone_boundaries("velocity", 5.0)
    ceiling = vibration.GEAR_ACCEPTANCE_HEADROOM * boundaries.a_b
    assert ceiling == pytest.approx(3.9375)
    assert vibration.evaluation_zone(ceiling, boundaries) == "B"
