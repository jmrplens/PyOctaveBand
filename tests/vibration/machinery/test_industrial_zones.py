#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the industrial-machine zone boundaries (ISO 10816-3:2009).

Oracle: ISO 10816-3:2009, *Mechanical vibration - Evaluation of machine
vibration by measurements on non-rotating parts - Part 3: Industrial machines
with nominal power above 15 kW and nominal speeds between 120 r/min and
15 000 r/min when measured in situ*, Annex A (normative), Tables A.1 and A.2
on printed folio 11, and the operational limits of 5.3 and 5.4 on printed
folios 7 and 8.

The edition is superseded, by ISO 20816-3, which is not held. It is the
direct predecessor of that part rather than a competing document, and the
ISO 20816-1 framework these boundaries are used with names the series as the
place its numbers live.
"""

from __future__ import annotations

import math

import pytest

from phonometry import vibration

#: Table A.1, group 1: displacement in micrometres, velocity in mm/s.
PRINTED_GROUP_1 = {
    "rigid": ((29.0, 57.0, 90.0), (2.3, 4.5, 7.1)),
    "flexible": ((45.0, 90.0, 140.0), (3.5, 7.1, 11.0)),
}
#: Table A.2, group 2.
PRINTED_GROUP_2 = {
    "rigid": ((22.0, 45.0, 71.0), (1.4, 2.8, 4.5)),
    "flexible": ((37.0, 71.0, 113.0), (2.3, 4.5, 7.1)),
}


@pytest.mark.parametrize(
    ("group", "printed"), [("group_1", PRINTED_GROUP_1), ("group_2", PRINTED_GROUP_2)]
)
def test_every_printed_cell_of_the_two_tables(
    group: str, printed: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]
) -> None:
    """Twelve cells per table, in both quantities and both support classes."""
    for support, (displacement, velocity) in printed.items():
        limits = vibration.INDUSTRIAL_MACHINE_ZONES[group, support]
        assert limits.displacement_um.as_tuple == pytest.approx(displacement)
        assert limits.velocity_mm_s.as_tuple == pytest.approx(velocity)


def test_a_flexible_support_is_allowed_more_than_a_rigid_one() -> None:
    """Every boundary of the flexible row sits above the rigid one beside it.

    A flexible support moves more for the same dynamic load at the bearing, so
    the same machine reads higher on it without being in worse condition.
    """
    for group in ("group_1", "group_2"):
        rigid = vibration.INDUSTRIAL_MACHINE_ZONES[group, "rigid"]
        flexible = vibration.INDUSTRIAL_MACHINE_ZONES[group, "flexible"]
        for stiff, soft in (
            (rigid.displacement_um, flexible.displacement_um),
            (rigid.velocity_mm_s, flexible.velocity_mm_s),
        ):
            assert all(
                a < b for a, b in zip(stiff.as_tuple, soft.as_tuple, strict=True)
            )


def test_a_large_machine_is_allowed_more_than_a_medium_one() -> None:
    """Group 1 sits above group 2 throughout, on the same support class."""
    for support in ("rigid", "flexible"):
        large = vibration.INDUSTRIAL_MACHINE_ZONES["group_1", support]
        medium = vibration.INDUSTRIAL_MACHINE_ZONES["group_2", support]
        for big, small in (
            (large.displacement_um, medium.displacement_um),
            (large.velocity_mm_s, medium.velocity_mm_s),
        ):
            assert all(a > b for a, b in zip(big.as_tuple, small.as_tuple, strict=True))


def test_grading_a_machine_on_one_measured_quantity() -> None:
    """A medium machine on a rigid support at 3,0 mm/s is in zone C.

    Its boundaries are 1,4, 2,8 and 4,5 mm/s, so 3,0 has passed the limit of
    unrestricted operation and is not yet doing damage.
    """
    assert (
        vibration.industrial_machine_zone("group_2", "rigid", velocity_mm_s=3.0) == "C"
    )
    # The same reading on a large machine with flexible supports is zone A.
    assert (
        vibration.industrial_machine_zone("group_1", "flexible", velocity_mm_s=3.0)
        == "A"
    )


def test_the_more_restrictive_of_the_two_quantities_wins() -> None:
    """5.2.3: with both measured, the worse grading is the grading.

    This is the case the pair of tables exists for. A large rigid machine at
    2,0 mm/s is comfortably inside zone A on velocity, and at 95 micrometres
    it is past the C/D boundary on displacement: a slow machine whose
    once-per-revolution component is large reads well on velocity and badly on
    displacement, and the standard takes the bad one.
    """
    assert (
        vibration.industrial_machine_zone(
            "group_1", "rigid", displacement_um=95.0, velocity_mm_s=2.0
        )
        == "D"
    )
    assert (
        vibration.industrial_machine_zone("group_1", "rigid", velocity_mm_s=2.0) == "A"
    )
    assert (
        vibration.industrial_machine_zone("group_1", "rigid", displacement_um=95.0)
        == "D"
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"group": "group_3", "support": "rigid", "velocity_mm_s": 1.0}, "group"),
        ({"group": "group_1", "support": "soft", "velocity_mm_s": 1.0}, "support"),
        ({"group": "group_1", "support": "rigid"}, "at least one of"),
    ],
)
def test_a_machine_it_cannot_place_is_refused(
    kwargs: dict[str, object], match: str
) -> None:
    """The group, the support class and at least one measured quantity."""
    with pytest.raises(ValueError, match=match):
        vibration.industrial_machine_zone(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The operational limits (5.3 and 5.4)
# --------------------------------------------------------------------------- #
def test_a_change_is_significant_past_a_quarter_of_zone_b() -> None:
    """5.3, and the sign does not matter: a fall counts as much as a rise."""
    upper = 4.5
    assert vibration.is_significant_change(1.2, upper)
    assert vibration.is_significant_change(-1.2, upper)
    assert not vibration.is_significant_change(1.0, upper)
    # Exactly a quarter is not "exceeding" it.
    assert not vibration.is_significant_change(0.25 * upper, upper)


def test_the_alarm_sits_a_quarter_of_zone_b_above_the_baseline() -> None:
    """5.4.1, and with a low baseline it lands below zone C, as the clause says."""
    upper = 4.5
    assert vibration.alarm_limit(2.0, upper) == pytest.approx(3.125)
    assert vibration.alarm_limit(2.0, upper) < upper


def test_the_alarm_is_capped_however_high_the_baseline_climbs() -> None:
    """5.4.1 again: not normally above 1,25 times the upper limit of zone B."""
    upper = 4.5
    capped = vibration.alarm_limit(100.0, upper)
    assert capped == pytest.approx(1.25 * upper)
    assert capped == pytest.approx(5.625)


def test_the_trip_ceiling_is_a_quarter_above_zone_c() -> None:
    """5.4.2 declines to give absolute values and gives a ceiling instead."""
    assert vibration.trip_limit(7.1) == pytest.approx(8.875)


def test_the_two_published_fractions_are_the_ones_the_clauses_print() -> None:
    """A quarter for a significant change, and a quarter of headroom."""
    assert vibration.SIGNIFICANT_CHANGE_FRACTION == pytest.approx(0.25)
    assert vibration.OPERATIONAL_LIMIT_HEADROOM == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: vibration.is_significant_change(math.nan, 4.5), "must be finite"),
        (lambda: vibration.is_significant_change(1.0, 0.0), "must be positive"),
        (lambda: vibration.alarm_limit(-1.0, 4.5), "non-negative"),
        (lambda: vibration.alarm_limit(1.0, -4.5), "must be positive"),
        (lambda: vibration.trip_limit(0.0), "must be positive"),
    ],
)
def test_the_operational_limits_refuse_what_they_cannot_set(
    call: object, match: str
) -> None:
    """Each guard names the argument it is about."""
    with pytest.raises(ValueError, match=match):
        call()  # type: ignore[operator]


def test_an_alarm_and_a_trip_bracket_the_zones_they_come_from() -> None:
    """The recommended settings sit where the clauses put them.

    For a medium machine on a rigid support: the ALARM lands inside zone C at
    worst, and the TRIP ceiling above the C/D boundary but not by much, which
    is the shape 5.4 describes without ever printing an absolute value.
    """
    limits = vibration.INDUSTRIAL_MACHINE_ZONES["group_2", "rigid"].velocity_mm_s
    alarm = vibration.alarm_limit(1.0, limits.b_c)
    trip = vibration.trip_limit(limits.c_d)
    assert alarm < limits.c_d
    assert trip > limits.c_d
    assert trip < 2.0 * limits.c_d
