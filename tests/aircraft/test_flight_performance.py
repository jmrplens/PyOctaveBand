#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ECAC Doc 29 Vol. 2 Appendix B flight performance against its own reference cases.

The 26 reference cases of Doc 29 Volume 3 Part 2 -- 9 arrivals over 124 profile
points and 17 departures over 190 -- are reproduced point by point and column by
column: distance, height, true airspeed and corrected net thrust. An endpoint
check would pass a profile that reached 10 000 ft by the wrong route, and the
route is the model.

The rest of the file pins the guards: every refusal the model can raise, with
the identifier it names.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from doc29_appendix_b_data import (
    AERODYNAMIC_COEFFICIENTS,
    AIRCRAFT,
    APPROACH_STEPS,
    ARRIVAL_CASES,
    ARRIVAL_RESULTS,
    DEFAULT_WEIGHTS,
    DEPARTURE_CASES,
    DEPARTURE_RESULTS,
    DEPARTURE_STEPS,
    JET_COEFFICIENTS,
    PRINTED_TOLERANCE,
    PROP_COEFFICIENTS,
)

from phonometry.aircraft.flight_performance import (
    Aerodrome,
    AerodynamicCoefficients,
    ApproachStep,
    DepartureStep,
    FlightProfile,
    JetEngineCoefficients,
    PerformanceAircraft,
    ProfilePoint,
    PropellerEngineCoefficients,
    approach_profile,
    departure_profile,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# --------------------------------------------------------------------------
# Comparison tolerances
# --------------------------------------------------------------------------
#: Slack added to a closed tolerance bound, in the unit of whichever column is
#: being compared.
#:
#: Doc 29 Volume 3 prints every result to one decimal, so a value that is
#: exactly half a digit away is exactly on the bound and has to pass: the last
#: transition point of every approach sits at half of a 49.9 ft screen height,
#: that is 24.95 ft, against a printed 24.9. Binary floating point puts that
#: subtraction a few ulps *outside* 0.05, so the bound is widened by far less
#: than the last printed digit and far more than the arithmetic can be wrong by.
_TIE_SLACK = 1e-9

#: Largest deviation from the printed reference value each column is allowed,
#: in that column's own unit, per operation.
#:
#: Six of the eight are :data:`PRINTED_TOLERANCE`, half of the last printed
#: digit, which is all the workbook's own precision can support. The three that
#: are looser are looser because the workbook cannot support that figure there,
#: each for a reason its own numbers demonstrate:
#:
#: * **Departure distance, 0.15 ft.** The Accelerate step is the one step of
#:   Appendix B that iterates, and B6.1.3 stops it as soon as two successive
#:   height estimates agree within a foot rather than at the fixed point, which
#:   leaves the segment length good to a few tenths. The workbook then disagrees
#:   with itself: case 6 and case 74 fly the identical Accelerate step from the
#:   identical state, and print segment lengths of 4678.3 and 4678.5 ft. Their
#:   rounding intervals do not overlap, so no single implementation can be
#:   within 0.05 ft of both. This model's largest departure over all 190 points
#:   is 0.145 ft, at exactly that point.
#: * **Departure thrust, 0.1 lb.** The same two cases print 16529.2 lb and
#:   16529.1 lb for one identical profile point, for the same reason.
#: * **Arrival thrust, 0.25 lb.** A Descend-Decel thrust is a difference of
#:   squared speeds (Eq. B-41) and moves 127 lb for every knot of start
#:   calibrated airspeed, which Doc 29 Volume 3 publishes to 0.1 kt. Half of
#:   that last digit is already 6 lb of thrust; the model lands within 0.22 lb
#:   of every one of the 124 arrival points, which bounds the input agreement at
#:   0.002 kt.
_MAX_DEVIATION = {
    ("A", "distance_ft"): PRINTED_TOLERANCE,
    ("A", "altitude_ft"): PRINTED_TOLERANCE,
    ("A", "true_airspeed_kt"): PRINTED_TOLERANCE,
    ("A", "corrected_net_thrust_lb"): 0.25,
    ("D", "distance_ft"): 0.15,
    ("D", "altitude_ft"): PRINTED_TOLERANCE,
    ("D", "true_airspeed_kt"): PRINTED_TOLERANCE,
    ("D", "corrected_net_thrust_lb"): 0.1,
}

_COLUMNS = ("distance_ft", "altitude_ft", "true_airspeed_kt", "corrected_net_thrust_lb")


def _round_printed(value: float | None) -> float | None:
    """One approach step input as Doc 29 Volume 3 prints it, to one decimal.

    The workbook's ``C-6.2`` sheet stores several step parameters to full
    double precision -- 49.86876640419947 ft is 15.2 m, 2460.6299212598424 ft is
    750 m -- while folio C-8 prints them to one decimal, and **the reference
    results were computed from the printed values**. The results say so
    themselves, four times over: an approach flown at 3 degrees from the stored
    1640.4199475065616 ft reaches touchdown 31301.08 ft away and the workbook
    tabulates 31300.7, which is 1640.4 ft; the stored 2460.6299 ft gives
    46951.62 against a tabulated 46951.0, which is 2460.6; and likewise for
    1968.5039 and 49.8688 ft. Substituting the stored values moves four arrival
    profiles by up to 0.6 ft, twelve times the tolerance, and moves the
    Level-Decel thrusts, which read the start speeds, by up to 0.9 lb.

    The stored values are what the transcription carries, because that is what
    the sheet holds; this is where the test says which of the two the reference
    implementation was fed.
    """
    return None if value is None else round(value, 1)


def _aircraft(aircraft_id: str) -> PerformanceAircraft:
    """The reference aeroplane's Appendix B coefficient set, from sheets C-1 to C-4."""
    _engine_type, engines, _mtow, landing_weight, _distance, static_thrust = AIRCRAFT[
        aircraft_id
    ]
    return PerformanceAircraft(
        aircraft_id=aircraft_id,
        engines=engines,
        max_static_thrust_lb=static_thrust,
        max_landing_weight_lb=landing_weight,
        jet_coefficients={
            rating: JetEngineCoefficients(*values)
            for (acft, rating), values in JET_COEFFICIENTS.items()
            if acft == aircraft_id
        },
        propeller_coefficients={
            rating: PropellerEngineCoefficients(*values)
            for (acft, rating), values in PROP_COEFFICIENTS.items()
            if acft == aircraft_id
        },
        aerodynamic_coefficients={
            (operation, flap): AerodynamicCoefficients(
                drag_ratio=drag,
                ground_roll_coefficient=roll,
                speed_coefficient=speed,
            )
            for (acft, operation, flap), (
                roll,
                speed,
                drag,
            ) in AERODYNAMIC_COEFFICIENTS.items()
            if acft == aircraft_id
        },
    )


def _departure_steps(aircraft_id: str, procedure: str) -> list[DepartureStep]:
    """Sheet C-6.1's rows for one procedure, as departure steps."""
    return [
        DepartureStep(
            step_type=step_type,
            thrust_rating=rating,
            flap_id=flap,
            end_altitude_ft=altitude,
            rate_of_climb_ft_per_min=rate,
            end_calibrated_airspeed_kt=airspeed,
            energy_share_percent=share,
            distance_ft=distance,
        )
        for (
            _number,
            step_type,
            rating,
            flap,
            altitude,
            rate,
            airspeed,
            share,
            distance,
        ) in DEPARTURE_STEPS[(aircraft_id, procedure)]
    ]


def _approach_steps(aircraft_id: str, procedure: str) -> list[ApproachStep]:
    """Sheet C-6.2's rows for one procedure, as approach steps.

    Two corrections on the way in, both of the sheet rather than of the model.
    A Level-Idle step keeps its ground distance in the *descent angle* column
    with the distance column empty, which is a defect of that sheet and not of
    the format -- the ANP release puts the same length in the distance column --
    so the length is moved to where it belongs. And every parameter is taken to
    one decimal, for the reason :func:`_round_printed` sets out.
    """
    steps = []
    for (
        _number,
        step_type,
        flap,
        altitude,
        airspeed,
        angle,
        touchdown_roll,
        distance,
        thrust,
    ) in APPROACH_STEPS[(aircraft_id, procedure)]:
        misfiled_length = step_type == "Level-Idle"
        steps.append(
            ApproachStep(
                step_type=step_type,
                flap_id=flap,
                start_altitude_ft=_round_printed(altitude),
                start_calibrated_airspeed_kt=_round_printed(airspeed),
                descent_angle_deg=None if misfiled_length else angle,
                touchdown_roll_ft=_round_printed(touchdown_roll),
                distance_ft=_round_printed(angle if misfiled_length else distance),
                start_thrust_percent=thrust,
            )
        )
    return steps


def _assert_matches(
    profile: FlightProfile,
    expected: Sequence[tuple[int, float, float, float, float]],
    case: object,
) -> None:
    """Every point of *profile* against every printed column of *expected*."""
    assert len(profile.points) == len(expected), (
        f"case {case}: the model produced {len(profile.points)} profile points "
        f"where Doc 29 Volume 3 tabulates {len(expected)}"
    )
    for row, point in zip(expected, profile.points, strict=True):
        number = row[0]
        for column, value in zip(_COLUMNS, row[1:], strict=True):
            got = getattr(point, column)
            allowed = _MAX_DEVIATION[(profile.operation, column)] + _TIE_SLACK
            assert abs(got - value) <= allowed, (
                f"case {case}, point {number}, {column}: {got!r} against the "
                f"tabulated {value!r}, off by {got - value:+.4f}"
            )


# --------------------------------------------------------------------------
# The reference cases
# --------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(DEPARTURE_CASES))
def test_departure_case_reproduces_every_profile_point(case: int) -> None:
    """Each Doc 29 Volume 3 departure case, point by point and column by column."""
    aircraft_id, procedure, elevation, temperature, headwind, pressure = (
        DEPARTURE_CASES[case]
    )
    profile = departure_profile(
        _aircraft(aircraft_id),
        _departure_steps(aircraft_id, procedure),
        weight_lb=DEFAULT_WEIGHTS[(aircraft_id, "D", "1")],
        aerodrome=Aerodrome(
            elevation_ft=elevation,
            temperature_c=temperature,
            sea_level_pressure_inhg=pressure,
            headwind_kt=headwind,
        ),
        procedure_id=procedure,
    )
    _assert_matches(profile, DEPARTURE_RESULTS[case], case)


@pytest.mark.parametrize("case", sorted(ARRIVAL_CASES))
def test_arrival_case_reproduces_every_profile_point(case: str) -> None:
    """Each Doc 29 Volume 3 arrival case, point by point and column by column."""
    aircraft_id, procedure, elevation, temperature, headwind, pressure, _local = (
        ARRIVAL_CASES[case]
    )
    profile = approach_profile(
        _aircraft(aircraft_id),
        _approach_steps(aircraft_id, procedure),
        aerodrome=Aerodrome(
            elevation_ft=elevation,
            temperature_c=temperature,
            sea_level_pressure_inhg=pressure,
            headwind_kt=headwind,
        ),
        procedure_id=procedure,
    )
    _assert_matches(profile, ARRIVAL_RESULTS[case], case)


def test_arrival_is_solved_backwards_from_touchdown() -> None:
    """The anchor is the Land step: airborne distances are negative, the roll positive.

    The direction of the sweep is the structural fact about an approach profile
    (folio B-5), and it is what a sign error in Eq. B-42 or Eq. B-64 breaks
    without changing any single point's height or speed.
    """
    profile = approach_profile(
        _aircraft("JETW"),
        _approach_steps("JETW", "Descend"),
        aerodrome=Aerodrome(elevation_ft=0.0, temperature_c=15.0, headwind_kt=0.0),
    )
    airborne = [p for p in profile.points if p.altitude_ft > 0.0]
    on_runway = [p for p in profile.points if p.altitude_ft == 0.0]
    assert all(p.distance_ft < 0.0 for p in airborne)
    assert [p.distance_ft for p in on_runway] == sorted(
        p.distance_ft for p in on_runway
    )
    assert min(abs(p.distance_ft) for p in on_runway) == 0.0


def test_idle_descent_thrust_can_be_negative() -> None:
    """An idle segment's corrected net thrust is drag, and the model may report it.

    Case 2C's third point is ``1100 - 6.5(250) + 0.17(3000) - 1e-5(3000)^2``,
    which is -105 lb: the reference aeroplane's own published idle coefficients
    at 3000 ft and 250 kt. A result type that required a positive thrust would
    refuse the reference data itself.
    """
    profile = approach_profile(
        _aircraft("JETW"),
        _approach_steps("JETW", "Level_Idle"),
        aerodrome=Aerodrome(elevation_ft=0.0, temperature_c=15.0, headwind_kt=0.0),
    )
    assert min(p.corrected_net_thrust_lb for p in profile.points) == pytest.approx(
        -105.0, abs=PRINTED_TOLERANCE
    )


def test_climb_step_below_an_accelerate_step_is_skipped() -> None:
    """B6.1.3: an Accelerate step may overfly a later Climb step's own altitude.

    Case 54 climbs past 5500 ft on its third Accelerate step, so the Climb step
    that aims at 5500 ft is dropped and the profile is one point shorter than
    the procedure has steps. Without the rule the profile descends.
    """
    aircraft_id, procedure, elevation, temperature, headwind, pressure = (
        DEPARTURE_CASES[54]
    )
    steps = _departure_steps(aircraft_id, procedure)
    profile = departure_profile(
        _aircraft(aircraft_id),
        steps,
        weight_lb=DEFAULT_WEIGHTS[(aircraft_id, "D", "1")],
        aerodrome=Aerodrome(
            elevation_ft=elevation,
            temperature_c=temperature,
            sea_level_pressure_inhg=pressure,
            headwind_kt=headwind,
        ),
    )
    skipped = [s for s in steps if s.kind == "climb" and s.end_altitude_ft == 5500.0]
    assert skipped, "case 54 should carry a Climb step to 5500 ft"
    assert 5500.0 not in [p.altitude_ft for p in profile.points]


def test_thrust_rating_change_inserts_a_transition_point() -> None:
    """B6.1.6: 1000 ft into the step that follows the change, and only then.

    Case 82 changes flap twice at unchanged ``MaxClimb`` and gets no transition
    point for either, which is the asymmetry with an arrival, where a flap
    change alone calls for one.
    """
    aircraft_id, procedure, elevation, temperature, headwind, pressure = (
        DEPARTURE_CASES[82]
    )
    profile = departure_profile(
        _aircraft(aircraft_id),
        _departure_steps(aircraft_id, procedure),
        weight_lb=DEFAULT_WEIGHTS[(aircraft_id, "D", "1")],
        aerodrome=Aerodrome(
            elevation_ft=elevation,
            temperature_c=temperature,
            sea_level_pressure_inhg=pressure,
            headwind_kt=headwind,
        ),
    )
    distances = [p.distance_ft for p in profile.points]
    gaps = [round(b - a, 1) for a, b in zip(distances, distances[1:], strict=False)]
    assert gaps.count(1000.0) == 1, (
        "case 82 changes thrust rating once, from MaxTakeoff to MaxClimb, and "
        "changes flap twice without one"
    )


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
def test_profile_point_refuses_a_non_finite_column() -> None:
    with pytest.raises(ValueError, match=r"ProfilePoint: 'altitude_ft'"):
        ProfilePoint(
            distance_ft=0.0,
            altitude_ft=math.nan,
            true_airspeed_kt=150.0,
            corrected_net_thrust_lb=10000.0,
        )


def test_profile_point_refuses_a_negative_height() -> None:
    with pytest.raises(ValueError, match=r"ProfilePoint: 'altitude_ft'"):
        ProfilePoint(
            distance_ft=0.0,
            altitude_ft=-1.0,
            true_airspeed_kt=150.0,
            corrected_net_thrust_lb=10000.0,
        )


def test_profile_point_refuses_a_negative_airspeed() -> None:
    with pytest.raises(ValueError, match=r"ProfilePoint: 'true_airspeed_kt'"):
        ProfilePoint(
            distance_ft=0.0,
            altitude_ft=0.0,
            true_airspeed_kt=-1.0,
            corrected_net_thrust_lb=10000.0,
        )


def _point(distance_ft: float) -> ProfilePoint:
    return ProfilePoint(
        distance_ft=distance_ft,
        altitude_ft=0.0,
        true_airspeed_kt=100.0,
        corrected_net_thrust_lb=1000.0,
    )


def test_flight_profile_refuses_an_unknown_operation() -> None:
    points = (_point(0.0), _point(1.0))
    with pytest.raises(ValueError, match=r"FlightProfile: 'operation'"):
        FlightProfile(aircraft_id="JETW", operation="X", procedure_id="", points=points)


def test_flight_profile_refuses_a_single_point() -> None:
    points = (_point(0.0),)
    with pytest.raises(ValueError, match=r"FlightProfile: 'points'"):
        FlightProfile(aircraft_id="JETW", operation="D", procedure_id="", points=points)


def test_flight_profile_refuses_points_that_double_back() -> None:
    points = (_point(0.0), _point(100.0), _point(50.0))
    with pytest.raises(ValueError, match=r"FlightProfile: 'points'"):
        FlightProfile(aircraft_id="JETW", operation="D", procedure_id="", points=points)


def test_flight_profile_columns_are_the_points_in_order() -> None:
    """The array views a caller reads a profile through are the points themselves."""
    profile = FlightProfile(
        aircraft_id="JETW",
        operation="D",
        procedure_id="",
        points=(_point(0.0), _point(100.0)),
    )
    assert list(profile.distance_ft) == [0.0, 100.0]
    assert list(profile.altitude_ft) == [0.0, 0.0]
    assert list(profile.true_airspeed_kt) == [100.0, 100.0]
    assert list(profile.corrected_net_thrust_lb) == [1000.0, 1000.0]


# --------------------------------------------------------------------------
# Aerodrome and coefficients
# --------------------------------------------------------------------------
def test_aerodrome_refuses_a_non_finite_field() -> None:
    with pytest.raises(ValueError, match=r"Aerodrome: 'elevation_ft'"):
        Aerodrome(elevation_ft=math.nan)


def test_aerodrome_refuses_a_pressure_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match=r"Aerodrome: 'sea_level_pressure_inhg'"):
        Aerodrome(elevation_ft=0.0, sea_level_pressure_inhg=0.0)


def test_aerodrome_refuses_a_runway_gradient_of_one() -> None:
    with pytest.raises(ValueError, match=r"Aerodrome: 'runway_gradient'"):
        Aerodrome(elevation_ft=0.0, runway_gradient=1.0)


def test_approach_step_refuses_a_negative_length() -> None:
    """A step is flown forwards, so a length below zero is not a short one.

    It matters beyond tidiness: the rollout skips a step with no length to
    travel, and it can only spell that ``length <= 0`` once nothing negative can
    reach it. Left open, a negative length would be silently skipped instead of
    reported.
    """
    with pytest.raises(ValueError, match=r"ApproachStep: 'distance_ft' must not"):
        ApproachStep(
            "Decelerate",
            "-NONE-",
            start_calibrated_airspeed_kt=130.0,
            distance_ft=-5.0,
            start_thrust_percent=40.0,
        )


def test_aerodrome_refuses_an_altitude_the_atmosphere_does_not_reach() -> None:
    """Above 145 448 ft Eq. B-4's bracket turns negative and leaves the reals."""
    aerodrome = Aerodrome(elevation_ft=0.0)
    with pytest.raises(ValueError, match=r"Eq\. B-4"):
        aerodrome.pressure_ratio(200000.0)


def test_atmosphere_lapses_from_the_aerodrome_not_from_sea_level() -> None:
    """Eq. B-3 puts the field temperature at field elevation, however high it is.

    Taking the lapse from sea level instead makes the take-off roll of the
    5000 ft reference cases 3.3 % short, and every case at a sea-level
    aerodrome agrees with either reading.
    """
    high = Aerodrome(elevation_ft=5000.0, temperature_c=40.0)
    assert high.temperature_c_at(5000.0) == pytest.approx(40.0)
    assert high.temperature_ratio(5000.0) == pytest.approx(
        (459.67 + 104.0) / 518.67, rel=1e-12
    )


def test_pressure_altitude_differs_from_the_geometric_one_off_standard_pressure() -> (
    None
):
    """Eq. B-9 reads the pressure altitude, which a non-standard QNH moves.

    Reference case 42 sits at sea level under 30.71 inHg and flies at a pressure
    altitude of -723 ft, and Eq. B-9's ``Ga h`` term reads that, not the zero.
    """
    aerodrome = Aerodrome(elevation_ft=0.0, sea_level_pressure_inhg=30.71)
    assert aerodrome.pressure_altitude_ft(0.0) == pytest.approx(-723.0, abs=0.5)


def test_jet_coefficients_refuse_a_non_finite_entry() -> None:
    with pytest.raises(ValueError, match=r"JetEngineCoefficients: 'ga'"):
        JetEngineCoefficients(e=25000.0, f=-25.0, ga=math.nan, gb=1e-5, h=0.0)


def test_propeller_coefficients_refuse_a_power_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match=r"PropellerEngineCoefficients: 'power_hp'"):
        PropellerEngineCoefficients(efficiency=0.85, power_hp=0.0)


def test_propeller_thrust_refuses_the_airspeed_it_divides_by() -> None:
    """Eq. B-12 is singular at rest, which is why B4.2 pins a floor for it."""
    coefficients = PropellerEngineCoefficients(efficiency=0.85, power_hp=9500.0)
    with pytest.raises(
        ValueError, match=r"PropellerEngineCoefficients: 'true_airspeed_kt'"
    ):
        coefficients.corrected_net_thrust_lb(true_airspeed_kt=0.0, pressure_ratio=1.0)


def test_aerodynamic_coefficients_refuse_a_non_finite_drag_ratio() -> None:
    with pytest.raises(ValueError, match=r"AerodynamicCoefficients: 'drag_ratio'"):
        AerodynamicCoefficients(drag_ratio=math.inf)


def test_performance_aircraft_refuses_an_engineless_aeroplane() -> None:
    with pytest.raises(ValueError, match=r"PerformanceAircraft: 'engines'"):
        PerformanceAircraft(
            aircraft_id="JETW",
            engines=0,
            max_static_thrust_lb=25000.0,
            max_landing_weight_lb=159222.0,
        )


def test_performance_aircraft_refuses_a_landing_weight_of_zero() -> None:
    with pytest.raises(ValueError, match=r"'max_landing_weight_lb'"):
        PerformanceAircraft(
            aircraft_id="JETW",
            engines=2,
            max_static_thrust_lb=25000.0,
            max_landing_weight_lb=0.0,
        )


def test_approach_weight_is_ninety_per_cent_of_the_landing_weight() -> None:
    """Folio B-31, not the ANP arrival weight row, which differs for most types."""
    assert _aircraft("JETW").approach_weight_lb == pytest.approx(0.9 * 159222.0)


def test_flap_lookup_names_the_configuration_it_could_not_find() -> None:
    aircraft = _aircraft("JETW")
    with pytest.raises(KeyError, match=r"flap '40'"):
        aircraft.flap("D", "40")


def test_flap_lookup_folds_case_and_padding() -> None:
    """The tables disagree with themselves about both (open item O-10)."""
    assert _aircraft("JETW").flap("d", " zero ").drag_ratio == pytest.approx(0.055)


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def test_departure_step_refuses_an_unknown_step_type() -> None:
    with pytest.raises(ValueError, match=r"DepartureStep: 'step_type'"):
        DepartureStep(step_type="Cruise", thrust_rating="MaxClimb", flap_id="ZERO")


def test_climb_step_refuses_a_missing_end_altitude() -> None:
    with pytest.raises(ValueError, match=r"DepartureStep: 'end_altitude_ft'"):
        DepartureStep(step_type="Climb", thrust_rating="MaxClimb", flap_id="ZERO")


def test_departure_level_step_refuses_a_missing_length() -> None:
    with pytest.raises(ValueError, match=r"DepartureStep: 'distance_ft'"):
        DepartureStep(step_type="Level", thrust_rating="AdaptedThrust", flap_id="5")


def test_accelerate_step_refuses_a_missing_end_airspeed() -> None:
    with pytest.raises(
        ValueError, match=r"DepartureStep: 'end_calibrated_airspeed_kt'"
    ):
        DepartureStep(
            step_type="Accelerate",
            thrust_rating="MaxClimb",
            flap_id="5",
            rate_of_climb_ft_per_min=900.0,
        )


def test_accelerate_step_refuses_carrying_neither_gradient_input() -> None:
    with pytest.raises(ValueError, match=r"'rate_of_climb_ft_per_min'"):
        DepartureStep(
            step_type="Accelerate",
            thrust_rating="MaxClimb",
            flap_id="5",
            end_calibrated_airspeed_kt=210.6,
        )


def test_step_type_spelling_folds_onto_the_standard_vocabulary() -> None:
    """The prose writes "Take-off", every table writes "Takeoff"."""
    assert (
        DepartureStep(
            step_type="Take-off", thrust_rating="MaxTakeoff", flap_id="5"
        ).kind
        == "takeoff"
    )


def test_approach_step_refuses_an_unknown_step_type() -> None:
    with pytest.raises(ValueError, match=r"ApproachStep: 'step_type'"):
        ApproachStep(step_type="Hold", flap_id="30")


def test_land_step_refuses_a_missing_touchdown_roll() -> None:
    with pytest.raises(ValueError, match=r"ApproachStep: 'touchdown_roll_ft'"):
        ApproachStep(step_type="Land", flap_id="30")


def test_decelerate_step_refuses_a_missing_start_thrust() -> None:
    with pytest.raises(ValueError, match=r"ApproachStep: 'distance_ft'"):
        ApproachStep(step_type="Decelerate", flap_id="-NONE-", distance_ft=3937.0)


def test_airborne_step_refuses_a_missing_start_altitude() -> None:
    with pytest.raises(ValueError, match=r"ApproachStep: 'start_altitude_ft'"):
        ApproachStep(
            step_type="Descend",
            flap_id="30",
            start_calibrated_airspeed_kt=135.0,
            descent_angle_deg=3.0,
        )


def test_descend_step_refuses_a_missing_start_airspeed() -> None:
    with pytest.raises(
        ValueError, match=r"ApproachStep: 'start_calibrated_airspeed_kt'"
    ):
        ApproachStep(
            step_type="Descend",
            flap_id="30",
            start_altitude_ft=1000.0,
            descent_angle_deg=3.0,
        )


def test_level_step_may_leave_its_airspeed_to_the_step_below_it() -> None:
    """Several ANP entries do, and a Level step holds whatever speed it is given."""
    step = ApproachStep(
        step_type="Level",
        flap_id="A_1+F",
        start_altitude_ft=3000.0,
        distance_ft=11893.0,
    )
    assert step.start_calibrated_airspeed_kt is None


def test_descend_step_refuses_a_missing_descent_angle() -> None:
    with pytest.raises(ValueError, match=r"ApproachStep: 'descent_angle_deg'"):
        ApproachStep(
            step_type="Descend",
            flap_id="30",
            start_altitude_ft=1000.0,
            start_calibrated_airspeed_kt=135.0,
        )


def test_approach_level_step_refuses_a_missing_length() -> None:
    with pytest.raises(ValueError, match=r"ApproachStep: 'distance_ft'"):
        ApproachStep(
            step_type="Level-Decel",
            flap_id="ZERO",
            start_altitude_ft=3000.0,
            start_calibrated_airspeed_kt=250.0,
        )


# --------------------------------------------------------------------------
# The model's own refusals
# --------------------------------------------------------------------------
_SEA_LEVEL = Aerodrome(elevation_ft=0.0, temperature_c=15.0)
_TAKEOFF = DepartureStep(step_type="Takeoff", thrust_rating="MaxTakeoff", flap_id="5")


def test_departure_refuses_a_procedure_that_does_not_start_on_the_runway() -> None:
    aircraft = _aircraft("JETW")
    climb = DepartureStep(
        step_type="Climb",
        thrust_rating="MaxTakeoff",
        flap_id="5",
        end_altitude_ft=1000.0,
    )
    with pytest.raises(ValueError, match=r"must start with a Take-off step"):
        departure_profile(aircraft, [climb], weight_lb=165347.0, aerodrome=_SEA_LEVEL)


def test_departure_refuses_a_weight_of_zero() -> None:
    aircraft = _aircraft("JETW")
    with pytest.raises(ValueError, match=r"'weight_lb'"):
        departure_profile(aircraft, [_TAKEOFF], weight_lb=0.0, aerodrome=_SEA_LEVEL)


def test_takeoff_refuses_a_flap_setting_with_no_ground_roll_coefficient() -> None:
    """Eq. B-15 and Eq. B-16 both need a coefficient flap ZERO does not carry."""
    aircraft = _aircraft("JETW")
    step = DepartureStep(
        step_type="Takeoff", thrust_rating="MaxTakeoff", flap_id="ZERO"
    )
    with pytest.raises(ValueError, match=r"Eq\. B-15 and Eq\. B-16"):
        departure_profile(aircraft, [step], weight_lb=165347.0, aerodrome=_SEA_LEVEL)


def test_takeoff_refuses_a_headwind_that_reaches_the_rotation_speed() -> None:
    """Eq. B-17's ``(V_C - 8)^2`` changes sign for a strong enough tailwind."""
    aircraft = _aircraft("JETW")
    gale = Aerodrome(elevation_ft=0.0, temperature_c=15.0, headwind_kt=200.0)
    with pytest.raises(ValueError, match=r"Eq\. B-17"):
        departure_profile(aircraft, [_TAKEOFF], weight_lb=165347.0, aerodrome=gale)


def test_takeoff_refuses_a_runway_too_steep_to_accelerate_along() -> None:
    """Eq. B-18's divisor ``a - g GR`` goes negative and shortens the roll."""
    aircraft = _aircraft("JETW")
    cliff = Aerodrome(elevation_ft=0.0, temperature_c=15.0, runway_gradient=0.9)
    with pytest.raises(ValueError, match=r"Eq\. B-18"):
        departure_profile(aircraft, [_TAKEOFF], weight_lb=165347.0, aerodrome=cliff)


def test_climb_refuses_a_step_the_aeroplane_has_no_thrust_for() -> None:
    """Eq. B-21 returns a sine at or below zero: that is not a climb."""
    aircraft = _aircraft("JETW")
    steps = [
        _TAKEOFF,
        DepartureStep(
            step_type="Climb",
            thrust_rating="IdleApproach",
            flap_id="5",
            end_altitude_ft=1000.0,
        ),
    ]
    with pytest.raises(ValueError, match=r"Eq\. B-21"):
        departure_profile(aircraft, steps, weight_lb=165347.0, aerodrome=_SEA_LEVEL)


def test_accelerate_refuses_a_step_with_too_little_thrust_to_climb_at_all() -> None:
    """B6.1.3's own abort: below a gradient of 0.01 the steps have to be revised."""
    aircraft = _aircraft("JETW")
    steps = [
        _TAKEOFF,
        DepartureStep(
            step_type="Accelerate",
            thrust_rating="IdleApproach",
            flap_id="5",
            end_calibrated_airspeed_kt=250.0,
            rate_of_climb_ft_per_min=2000.0,
        ),
    ]
    with pytest.raises(ValueError, match=r"climb gradient of"):
        departure_profile(aircraft, steps, weight_lb=165347.0, aerodrome=_SEA_LEVEL)


def test_accelerate_refuses_a_step_whose_height_never_settles() -> None:
    """B6.1.3's iteration is bounded, and the bound is a refusal rather than a guess.

    The Accelerate step is solved by iteration because its end height, end true
    airspeed and segment length each depend on the other two, and nothing in
    Doc 29 proves that loop contracts. On this aeroplane it does not: an
    Accelerate to 420 kt at a 10 % energy share settles into a two-cycle,
    alternating between 22 936 ft and 37 373 ft and returning to the same pair
    to the foot, still 14 437 ft apart after six hundred passes. It is not a
    divergence to infinity and no larger iteration count reaches it, so the
    only honest answer is to stop and say so.

    Nothing is forced here: the coefficients are the reference aeroplane's own,
    and the step is a legal one the manufacturer never publishes. The climb
    gradient never falls below 0.017, above the 0.01 floor, so it is this
    refusal that fires and not the one above it.
    """
    aircraft = _aircraft("JETW")
    steps = [
        _TAKEOFF,
        DepartureStep(
            step_type="Accelerate",
            thrust_rating="MaxClimb",
            flap_id="5",
            end_calibrated_airspeed_kt=420.0,
            energy_share_percent=10.0,
        ),
    ]
    with pytest.raises(
        ValueError,
        match=r"did not converge within \d+ iterations \(B6\.1\.3\); step 'Accelerate'",
    ):
        departure_profile(aircraft, steps, weight_lb=165347.0, aerodrome=_SEA_LEVEL)


def test_level_accelerate_refuses_a_step_with_no_thrust_to_spare() -> None:
    aircraft = _aircraft("JETW")
    steps = [
        _TAKEOFF,
        DepartureStep(
            step_type="Level-Accelerate",
            thrust_rating="IdleApproach",
            flap_id="5",
            end_calibrated_airspeed_kt=250.0,
        ),
    ]
    with pytest.raises(ValueError, match=r"Eq\. B-31"):
        departure_profile(aircraft, steps, weight_lb=165347.0, aerodrome=_SEA_LEVEL)


def test_thrust_lookup_names_the_rating_it_could_not_find() -> None:
    aircraft = _aircraft("JETW")
    steps = [
        DepartureStep(step_type="Takeoff", thrust_rating="Afterburner", flap_id="5")
    ]
    with pytest.raises(KeyError, match=r"thrust rating 'Afterburner'"):
        departure_profile(aircraft, steps, weight_lb=165347.0, aerodrome=_SEA_LEVEL)


def test_minimum_reduced_thrust_refuses_a_single_engine_aeroplane() -> None:
    """Eq. B-13 divides by ``N - 1`` and the engine-out case does not exist."""
    single = PerformanceAircraft(
        aircraft_id="ONE",
        engines=1,
        max_static_thrust_lb=25000.0,
        max_landing_weight_lb=159222.0,
        jet_coefficients=_aircraft("JETW").jet_coefficients,
        aerodynamic_coefficients=_aircraft("JETW").aerodynamic_coefficients,
    )
    steps = [
        _TAKEOFF,
        DepartureStep(
            step_type="Climb",
            thrust_rating="MinimumThrust",
            flap_id="5",
            end_altitude_ft=1000.0,
        ),
    ]
    with pytest.raises(ValueError, match=r"Eq\. B-13"):
        departure_profile(single, steps, weight_lb=165347.0, aerodrome=_SEA_LEVEL)


def test_approach_refuses_a_procedure_with_no_land_step() -> None:
    """Without a Land step nothing sits at distance zero (open item O-13)."""
    aircraft = _aircraft("JETW")
    steps = _approach_steps("JETW", "Descend")[:5]
    with pytest.raises(ValueError, match=r"exactly one Land step"):
        approach_profile(aircraft, steps, aerodrome=_SEA_LEVEL)


def test_approach_refuses_a_land_step_with_nothing_to_decelerate_into() -> None:
    """Eq. B-78 and Eq. B-79 read the Land step's Point2 from the next step."""
    aircraft = _aircraft("JETW")
    steps = _approach_steps("JETW", "Descend")[:6]
    with pytest.raises(ValueError, match=r"Eq\. B-78"):
        approach_profile(aircraft, steps, aerodrome=_SEA_LEVEL)


def test_approach_refuses_a_descend_step_that_starts_below_the_step_under_it() -> None:
    aircraft = _aircraft("JETW")
    steps = _approach_steps("JETW", "Descend")
    steps[0] = ApproachStep(
        step_type="Descend",
        flap_id="ZERO",
        start_altitude_ft=100.0,
        start_calibrated_airspeed_kt=250.0,
        descent_angle_deg=2.8,
    )
    with pytest.raises(ValueError, match=r"must start above the step below it"):
        approach_profile(aircraft, steps, aerodrome=_SEA_LEVEL)


def test_land_step_refuses_a_procedure_with_no_slope_above_it() -> None:
    """Eq. B-76 takes its angle from the last descending step before touchdown."""
    aircraft = _aircraft("JETW")
    steps = [
        ApproachStep(
            step_type="Level",
            flap_id="ZERO",
            start_altitude_ft=0.0,
            start_calibrated_airspeed_kt=135.0,
            distance_ft=1000.0,
        ),
        *_approach_steps("JETW", "Descend")[5:],
    ]
    with pytest.raises(ValueError, match=r"Eq\. B-76"):
        approach_profile(aircraft, steps, aerodrome=_SEA_LEVEL)


def test_land_step_refuses_a_flap_setting_with_no_landing_speed_coefficient() -> None:
    aircraft = _aircraft("JETW")
    steps = _approach_steps("JETW", "Descend")
    steps[5] = ApproachStep(step_type="Land", flap_id="5", touchdown_roll_ft=304.1)
    with pytest.raises(ValueError, match=r"Eq\. B-75"):
        approach_profile(aircraft, steps, aerodrome=_SEA_LEVEL)


def test_consecutive_level_idle_steps_at_different_altitudes_warn() -> None:
    """B7.1.4 asks the system to warn rather than repair the manufacturer's table."""
    aircraft = _aircraft("JETW")
    steps = _approach_steps("JETW", "Level_Idle")
    steps[1] = ApproachStep(
        step_type="Level-Idle",
        flap_id="ZERO",
        start_altitude_ft=4000.0,
        start_calibrated_airspeed_kt=250.0,
        distance_ft=16500.0,
    )
    with pytest.warns(UserWarning, match=r"one altitude"):
        approach_profile(aircraft, steps, aerodrome=_SEA_LEVEL)


def test_a_level_idle_step_that_accelerates_warns() -> None:
    aircraft = _aircraft("JETW")
    steps = _approach_steps("JETW", "Level_Idle")
    steps[1] = ApproachStep(
        step_type="Level-Idle",
        flap_id="ZERO",
        start_altitude_ft=3000.0,
        start_calibrated_airspeed_kt=100.0,
        distance_ft=16500.0,
    )
    with pytest.warns(UserWarning, match=r"start CAS"):
        approach_profile(aircraft, steps, aerodrome=_SEA_LEVEL)


def test_non_isa_adjustment_is_an_identity_at_sea_level_isa() -> None:
    """The idle steps' tabulated parameters were derived there (folio B-36).

    Doc 29's own reference cases are all flown at sea-level ISA, so they cannot
    exercise the adjustment; what they do pin is that it changes nothing when
    the conditions are the reference ones, which is what makes the reference
    cases evidence about the rest of the model at all.
    """
    steps = _approach_steps("JETW", "Level_Idle")
    at_isa = approach_profile(_aircraft("JETW"), steps, aerodrome=_SEA_LEVEL)
    # Points 3 to 5 of the profile are the two Level-Idle steps' own Point1s and
    # the Descend-Idle Point1 below them, so the gaps are the two steps'
    # tabulated lengths, unchanged.
    gaps = [
        at_isa.points[index + 1].distance_ft - at_isa.points[index].distance_ft
        for index in (2, 3)
    ]
    assert gaps == pytest.approx([16500.0, 5000.0], abs=PRINTED_TOLERANCE)
    elsewhere = approach_profile(
        _aircraft("JETW"),
        steps,
        aerodrome=Aerodrome(elevation_ft=5000.0, temperature_c=40.0),
    )
    moved = [
        elsewhere.points[index + 1].distance_ft - elsewhere.points[index].distance_ft
        for index in (2, 3)
    ]
    assert moved != pytest.approx(gaps, abs=1.0), (
        "away from sea-level ISA the tabulated length is what gives, so that "
        "the deceleration the manufacturer derived is what is held (Eq. B-74)"
    )


def test_an_idle_descent_onto_a_level_step_keeps_its_height_and_moves_its_speed() -> (
    None
):
    """Eq. B-58, the branch of B7.1.2 chosen by the step *below* the current one.

    "When the step following the current Descend-Idle step is a Level,
    Level-Idle, or Land step, Point1_Height is maintained and Point1_TAS is
    re-calculated (for non-ISA conditions)" (folio B-37). The alternative,
    Eq. B-59 and Eq. B-60, does the opposite: it holds the speed and moves the
    step's own top. Every Doc 29 Volume 3 idle procedure descends into another
    descent, so the reference cases only ever take that second branch; the
    procedure's first step is made an idle descent here, over the Level-Idle
    the manufacturer already tabulates beneath it, and everything else is
    untouched.

    At a 5000 ft aerodrome at 40 degC the step is entered at 246.6 kt rather
    than the tabulated 250, and its top stays at 6000 ft: had the model taken
    the other branch the top would have moved and the speed would not.
    """
    steps = _approach_steps("JETW", "Level_Idle")
    steps[0] = ApproachStep(
        step_type="Descend-Idle",
        flap_id="ZERO",
        start_altitude_ft=6000.0,
        start_calibrated_airspeed_kt=250.0,
        descent_angle_deg=2.8,
    )
    assert steps[1].kind == "level-idle", "the step below is what selects Eq. B-58"
    at_isa = approach_profile(_aircraft("JETW"), steps, aerodrome=_SEA_LEVEL)
    top = at_isa.points[0]
    assert top.altitude_ft == 6000.0
    assert _SEA_LEVEL.calibrated_airspeed_kt(
        top.true_airspeed_kt, top.altitude_ft
    ) == pytest.approx(250.0)
    hot_and_high = Aerodrome(elevation_ft=5000.0, temperature_c=40.0)
    elsewhere = approach_profile(_aircraft("JETW"), steps, aerodrome=hot_and_high)
    top = elsewhere.points[0]
    assert top.altitude_ft == 6000.0, "Eq. B-58 maintains the height"
    moved_kt = hot_and_high.calibrated_airspeed_kt(
        top.true_airspeed_kt, hot_and_high.elevation_ft + top.altitude_ft
    )
    assert abs(moved_kt - 250.0) > 1.0, (
        "away from sea-level ISA the tabulated start speed is what gives, so "
        "that the deceleration the manufacturer derived is what is held"
    )


def test_a_turn_only_ever_costs_thrust() -> None:
    """Eq. B-14's bank angle enters as ``R/cos(eps)``, which only inflates drag.

    No reference case turns, so this pins the sign and the reduction to the
    straight case rather than a number: banking cannot shorten a climb.
    """
    straight = _departure_steps("JETW", "ICAO_A")
    banked = [
        DepartureStep(
            step_type=s.step_type,
            thrust_rating=s.thrust_rating,
            flap_id=s.flap_id,
            end_altitude_ft=s.end_altitude_ft,
            rate_of_climb_ft_per_min=s.rate_of_climb_ft_per_min,
            end_calibrated_airspeed_kt=s.end_calibrated_airspeed_kt,
            energy_share_percent=s.energy_share_percent,
            distance_ft=s.distance_ft,
            bank_angle_deg=20.0,
        )
        for s in straight
    ]
    aerodrome = Aerodrome(elevation_ft=0.0, temperature_c=15.0)
    flat = departure_profile(
        _aircraft("JETW"), straight, weight_lb=165347.0, aerodrome=aerodrome
    )
    turning = departure_profile(
        _aircraft("JETW"), banked, weight_lb=165347.0, aerodrome=aerodrome
    )
    assert turning.points[-1].distance_ft > flat.points[-1].distance_ft


def test_an_aeroplane_in_both_engine_tables_is_refused() -> None:
    """Eq. B-9 and Eq. B-12 would disagree, and Doc 29 breaks no tie (O-11)."""
    both = PerformanceAircraft(
        aircraft_id="BOTH",
        engines=2,
        max_static_thrust_lb=25000.0,
        max_landing_weight_lb=159222.0,
        jet_coefficients=_aircraft("JETW").jet_coefficients,
        propeller_coefficients={
            "MaxTakeoff": PropellerEngineCoefficients(efficiency=0.85, power_hp=9500.0)
        },
        aerodynamic_coefficients=_aircraft("JETW").aerodynamic_coefficients,
    )
    with pytest.raises(ValueError, match=r"both jet and propeller coefficients"):
        departure_profile(both, [_TAKEOFF], weight_lb=165347.0, aerodrome=_SEA_LEVEL)
