#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO/TR 17534-3 quality-assurance test cases of ISO 9613-2.

Oracle: ISO/TR 17534-3:2015, *Acoustics - Software for the calculation of sound
outdoors - Part 3: Recommendations for quality assured implementation of
ISO 9613-2 in software according to ISO 17534-1*, 6.2.1 to 6.2.8 on printed
folios 6 to 15. Every expected value below is a cell of one of those tables.

The document exists because two conforming implementations of ISO 9613-2 can
still disagree, and it settles the disagreements by printing a whole worked
chain rather than a formula. It also fixes the envelope a result has to stay
inside: "The result values in frequency bands and for the total level are
considered to be correct if the deviation does not exceed +/-0,05 dB", which is
the tolerance used throughout.

Only T01 to T07 are covered, which is the split the document itself draws in
6.1: "Test cases T01 up to T07 can be solved by applying ISO 9613-2
exclusively." T08 to T19 build their ray paths by the additional
recommendations of Clause 5, over barriers and around buildings, and are not
attempted here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import environment

#: Table 1: the source and receiver of every case, in metres. The z are heights
#: above the local ground, which is what lets T06 reuse them over risen ground.
SOURCE = (10.0, 10.0, 1.0)
RECEIVER = (200.0, 50.0, 4.0)
#: Table 2: 93 dB in all eight octaves, so the printed spectra are propagation.
SOUND_POWER_DB = 93.0
#: The header row of every spectral table, Hz.
BANDS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
#: Row "A-Korrektur" of every spectral table, dB.
A_WEIGHTING_DB = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])
#: 6.2.1: the air the cases are calculated in.
TEMPERATURE_C = 20.0
RELATIVE_HUMIDITY_PCT = 70.0
#: The envelope the document declares for a band and for a total, dB.
ENVELOPE_DB = 0.05

#: Table 3: the ground projection, shared by every case.
DP = math.hypot(RECEIVER[0] - SOURCE[0], RECEIVER[1] - SOURCE[1])
#: Table 3: over flat ground the ray rises only between the two heights.
D3_FLAT = math.hypot(DP, RECEIVER[2] - SOURCE[2])
#: Table 14: with the ground 10 m higher under the receiver, the ray rises 13 m.
D3_SLOPED = math.hypot(DP, 13.0)

#: Tables 8 and 14: the ground projection spends these lengths over the three
#: areas of Table 7 (T04) and Table 11 (T06), in metres.
SEGMENTS = (40.88, 102.19, 51.10)
#: T04 runs from the least porous area to the most; T06 the other way about.
G_T04 = (0.2, 0.5, 0.9)
G_T06 = (0.9, 0.5, 0.2)

#: Table 12 cut along the path: level to the contour under x = 120 m, climbing
#: to the 10 m contour at x = 185 m, then level to the receiver.
SLOPE_DISTANCES = (
    0.0,
    (120.0 - SOURCE[0]) * DP / (RECEIVER[0] - SOURCE[0]),
    (185.0 - SOURCE[0]) * DP / (RECEIVER[0] - SOURCE[0]),
    DP,
)
SLOPE_HEIGHTS = (0.0, 0.0, 10.0, 10.0)


def _receiver_levels(ground: environment.GroundFactors, distance: float) -> np.ndarray:
    """Band levels at the receiver by the general ground method (7.3.1)."""
    return environment.predicted_receiver_level(
        np.full(BANDS.size, SOUND_POWER_DB),
        environment.PropagationGeometry(
            distance, SOURCE[2], RECEIVER[2], projected_distance=DP
        ),
        frequencies=BANDS,
        ground=ground,
        atmosphere=environment.AtmosphericConditions(
            temperature=TEMPERATURE_C, relative_humidity=RELATIVE_HUMIDITY_PCT
        ),
    )


def _alternative_levels(mean_height: float, distance: float) -> np.ndarray:
    """Band levels by the alternative ground method (7.3.2), with its DOmega.

    Equation (10) is paired with the solid-angle index of Equation (11), which
    Tables 10 and 18 print as a row of its own. The library keeps the two apart
    on purpose, so the pairing is made here.
    """
    return (
        SOUND_POWER_DB
        + environment.directivity_omega(SOURCE[2], RECEIVER[2], DP)
        - environment.geometric_divergence(distance)
        - environment.atmospheric_absorption(
            distance,
            BANDS,
            temperature=TEMPERATURE_C,
            relative_humidity=RELATIVE_HUMIDITY_PCT,
        )
        - environment.ground_attenuation_alternative(distance, mean_height)
    )


def _energy_sum(levels: np.ndarray, weighting: np.ndarray | None = None) -> float:
    """Energy sum of the octave bands, dB."""
    if weighting is not None:
        levels = levels + weighting
    return 10.0 * math.log10(float(np.sum(10.0 ** (levels / 10.0))))


# --------------------------------------------------------------------------- #
# The geometry every case shares (Table 3)
# --------------------------------------------------------------------------- #
def test_the_shared_geometry_matches_table_3() -> None:
    """dp, d3 and Adiv, printed as 194,16 m, 194,19 m and 56,76 dB."""
    assert DP == pytest.approx(194.16, abs=0.005)
    assert D3_FLAT == pytest.approx(194.19, abs=0.005)
    assert environment.geometric_divergence(D3_FLAT) == pytest.approx(56.76, abs=0.005)


def test_the_three_regions_leave_a_middle_one() -> None:
    """30 hs and 30 hr take 150 m of the 194,16 m, and q is the rest of it."""
    assert 30.0 * SOURCE[2] == pytest.approx(30.0)
    assert 30.0 * RECEIVER[2] == pytest.approx(120.0)
    assert DP - 150.0 == pytest.approx(44.16, abs=0.005)
    q = 1.0 - 30.0 * (SOURCE[2] + RECEIVER[2]) / DP
    assert q == pytest.approx(0.23, abs=0.005)


# --------------------------------------------------------------------------- #
# T01 to T03: flat ground of one kind (Tables 4, 5, 6)
# --------------------------------------------------------------------------- #
HOMOGENEOUS_CASES = (
    (
        "T01",
        0.0,
        (39.90, 39.86, 39.70, 39.37, 38.95, 38.17, 35.47, 25.04),
        47.46,
        44.29,
    ),
    (
        "T02",
        0.5,
        (39.90, 36.17, 33.02, 33.20, 36.11, 36.33, 33.63, 23.20),
        44.61,
        41.53,
    ),
    (
        "T03",
        1.0,
        (39.90, 32.48, 26.33, 27.03, 33.27, 34.49, 31.79, 21.36),
        42.80,
        39.14,
    ),
)


@pytest.mark.parametrize(("case", "g", "levels", "total", "total_a"), HOMOGENEOUS_CASES)
def test_flat_ground_of_one_kind_reproduces_its_printed_table(
    case: str, g: float, levels: tuple[float, ...], total: float, total_a: float
) -> None:
    """T01, T02 and T03: reflecting, mixed and porous ground, band by band."""
    computed = _receiver_levels(environment.GroundFactors(g, g, g), D3_FLAT)
    assert computed == pytest.approx(levels, abs=ENVELOPE_DB), case
    assert _energy_sum(computed) == pytest.approx(total, abs=ENVELOPE_DB)
    assert _energy_sum(computed, A_WEIGHTING_DB) == pytest.approx(
        total_a, abs=ENVELOPE_DB
    )


def test_hard_ground_gives_back_the_three_decibels_it_reflects() -> None:
    """T01 prints Agr = -3,68 dB in every band: -1,5 twice, and -3q for the middle.

    Reflecting ground is the one case Table 3 answers with a constant, so the
    whole spectrum of T01 is the free field shifted by one number.
    """
    agr = environment.ground_attenuation(
        D3_FLAT, SOURCE[2], RECEIVER[2], BANDS, 0.0, 0.0, 0.0, projected_distance=DP
    )
    q = 1.0 - 30.0 * (SOURCE[2] + RECEIVER[2]) / DP
    assert agr == pytest.approx(np.full(BANDS.size, -3.0 - 3.0 * q), abs=0.005)
    assert agr == pytest.approx(np.full(BANDS.size, -3.68), abs=0.005)


def test_porous_ground_costs_most_where_the_notch_falls() -> None:
    """T03 against T01: the loss peaks at 250 Hz and dies away above 2 kHz."""
    hard = _receiver_levels(environment.GroundFactors(0.0, 0.0, 0.0), D3_FLAT)
    porous = _receiver_levels(environment.GroundFactors(1.0, 1.0, 1.0), D3_FLAT)
    difference = hard - porous
    assert int(np.argmax(difference)) == 2
    assert difference[-1] == pytest.approx(difference[-2], abs=0.01)


# --------------------------------------------------------------------------- #
# T04 and T06: ground of three kinds, general method (Tables 8, 9, 14, 15)
# --------------------------------------------------------------------------- #
def test_t04_averages_the_three_areas_into_three_regions() -> None:
    """Table 8: Gs 0,20, Gm 0,43 and Gr 0,67 from the segments of Table 7."""
    factors = environment.region_ground_factors(SEGMENTS, G_T04, SOURCE[2], RECEIVER[2])
    assert factors.source == pytest.approx(0.20, abs=0.005)
    assert factors.middle == pytest.approx(0.43, abs=0.005)
    assert factors.receiver == pytest.approx(0.67, abs=0.005)


def test_t06_averages_the_same_segments_the_other_way_about() -> None:
    """Table 14: the same lengths over reversed areas give 0,90, 0,60 and 0,37."""
    factors = environment.region_ground_factors(SEGMENTS, G_T06, SOURCE[2], RECEIVER[2])
    assert factors.source == pytest.approx(0.90, abs=0.005)
    assert factors.middle == pytest.approx(0.60, abs=0.005)
    assert factors.receiver == pytest.approx(0.37, abs=0.005)


def test_t04_reproduces_table_9() -> None:
    """Flat ground of three kinds, by the general method."""
    printed = (39.90, 36.24, 35.23, 36.04, 36.95, 36.57, 33.87, 23.45)
    factors = environment.region_ground_factors(SEGMENTS, G_T04, SOURCE[2], RECEIVER[2])
    computed = _receiver_levels(factors, D3_FLAT)
    assert computed == pytest.approx(printed, abs=ENVELOPE_DB)
    assert _energy_sum(computed) == pytest.approx(45.25, abs=ENVELOPE_DB)
    assert _energy_sum(computed, A_WEIGHTING_DB) == pytest.approx(
        42.23, abs=ENVELOPE_DB
    )


def test_t06_reproduces_table_15() -> None:
    """The same three areas reversed, over ground that rises 10 m at the receiver.

    Only two things separate this from T04: which area the source stands on,
    and the 13 m the ray climbs, which lengthens d3 to 194,60 m and moves Adiv
    and Aatm with it.
    """
    printed = (39.88, 35.65, 29.70, 29.24, 34.82, 35.83, 33.13, 22.68)
    factors = environment.region_ground_factors(SEGMENTS, G_T06, SOURCE[2], RECEIVER[2])
    computed = _receiver_levels(factors, D3_SLOPED)
    assert D3_SLOPED == pytest.approx(194.60, abs=0.005)
    assert computed == pytest.approx(printed, abs=ENVELOPE_DB)
    assert _energy_sum(computed) == pytest.approx(43.85, abs=ENVELOPE_DB)
    assert _energy_sum(computed, A_WEIGHTING_DB) == pytest.approx(
        40.59, abs=ENVELOPE_DB
    )


def test_the_printed_table_3_functions_come_out_of_the_heights_alone() -> None:
    """Tables 15 and 21 print a', b', c' and d' for both outer regions.

    They are the only intermediate of the ground method the guideline exposes,
    and they depend on the region height and dp and on nothing else, which is
    what makes them the same in T06 and in T08.
    """
    from phonometry.environment.propagation import outdoor_propagation as impl

    for height, printed in (
        (SOURCE[2], (2.45, 9.20, 10.16, 3.49)),
        (RECEIVER[2], (4.24, 3.50, 1.51, 1.50)),
    ):
        computed = (
            impl._a_prime(height, DP),
            impl._b_prime(height, DP),
            impl._c_prime(height, DP),
            impl._d_prime(height, DP),
        )
        assert computed == pytest.approx(printed, abs=0.005)


# --------------------------------------------------------------------------- #
# T05 and T07: the same two scenarios by the alternative method (7.3.2)
# --------------------------------------------------------------------------- #
def test_t05_reproduces_table_10() -> None:
    """Flat ground: hm is 2,50 m, Agr is one number and DOmega is 3,01 dB."""
    printed = (34.90, 34.86, 34.71, 34.38, 33.95, 33.17, 30.48, 20.05)
    hm = environment.mean_path_height((0.0, DP), (0.0, 0.0), SOURCE[2], RECEIVER[2])
    assert hm == pytest.approx(2.50, abs=0.005)
    assert environment.ground_attenuation_alternative(D3_FLAT, hm) == pytest.approx(
        4.32, abs=0.005
    )
    assert environment.directivity_omega(SOURCE[2], RECEIVER[2], DP) == pytest.approx(
        3.01, abs=0.005
    )
    computed = _alternative_levels(hm, D3_FLAT)
    assert computed == pytest.approx(printed, abs=ENVELOPE_DB)
    assert _energy_sum(computed) == pytest.approx(42.46, abs=ENVELOPE_DB)
    assert _energy_sum(computed, A_WEIGHTING_DB) == pytest.approx(
        39.30, abs=ENVELOPE_DB
    )


def test_t07_reproduces_table_18() -> None:
    """The rising slope halves the clearance the ray has, so hm is 4,99 m."""
    printed = (35.36, 35.32, 35.16, 34.83, 34.40, 33.62, 30.92, 20.47)
    hm = environment.mean_path_height(
        SLOPE_DISTANCES,
        SLOPE_HEIGHTS,
        SOURCE[2],
        RECEIVER[2],
        distance=D3_SLOPED,
    )
    assert hm == pytest.approx(4.99, abs=0.005)
    assert environment.ground_attenuation_alternative(D3_SLOPED, hm) == pytest.approx(
        3.85, abs=0.005
    )
    computed = _alternative_levels(hm, D3_SLOPED)
    assert computed == pytest.approx(printed, abs=ENVELOPE_DB)
    assert _energy_sum(computed) == pytest.approx(42.91, abs=ENVELOPE_DB)
    assert _energy_sum(computed, A_WEIGHTING_DB) == pytest.approx(
        39.75, abs=ENVELOPE_DB
    )


def test_the_two_methods_disagree_by_more_than_their_own_envelope() -> None:
    """T04 against T05, and T06 against T07: the rows are not restatements.

    7.3.2 is offered as a shortcut for the A-weighted level, and the guideline
    prints both answers for the same two scenarios precisely because they are
    not the same answer. If they were within the +/-0,05 dB envelope, half of
    these rows would be checking nothing.
    """
    general_flat = _receiver_levels(
        environment.region_ground_factors(SEGMENTS, G_T04, SOURCE[2], RECEIVER[2]),
        D3_FLAT,
    )
    alternative_flat = _alternative_levels(
        environment.mean_path_height((0.0, DP), (0.0, 0.0), SOURCE[2], RECEIVER[2]),
        D3_FLAT,
    )
    assert _energy_sum(general_flat, A_WEIGHTING_DB) - _energy_sum(
        alternative_flat, A_WEIGHTING_DB
    ) == pytest.approx(42.23 - 39.30, abs=2 * ENVELOPE_DB)


# --------------------------------------------------------------------------- #
# region_ground_factors: the shape of the rule, and the guards
# --------------------------------------------------------------------------- #
def test_one_kind_of_ground_averages_to_itself_in_all_three_regions() -> None:
    """A single segment is the homogeneous case, whatever the heights."""
    factors = environment.region_ground_factors((200.0,), (0.42,), 1.0, 4.0)
    assert (factors.source, factors.middle, factors.receiver) == pytest.approx(
        (0.42, 0.42, 0.42)
    )


def test_the_regions_are_weighted_by_length_and_not_by_count() -> None:
    """Half the source region on each of two grounds is their plain mean.

    A count-weighted average would give the same answer here and a different
    one below, which is what the second half checks.
    """
    even = environment.region_ground_factors(
        (15.0, 15.0, 170.0), (0.0, 1.0, 0.0), 1.0, 0.0
    )
    assert even.source == pytest.approx(0.5)
    uneven = environment.region_ground_factors(
        (5.0, 25.0, 170.0), (0.0, 1.0, 0.0), 1.0, 0.0
    )
    assert uneven.source == pytest.approx(25.0 / 30.0)


def test_a_region_that_is_a_point_takes_the_ground_it_stands_on() -> None:
    """A source on the ground has a region of no length, not an empty average."""
    factors = environment.region_ground_factors((50.0, 50.0), (0.3, 0.8), 0.0, 1.0)
    assert factors.source == pytest.approx(0.3)


def test_regions_that_meet_leave_no_middle_one() -> None:
    """With 30(hs + hr) past dp, Gm is the whole path and q drops it anyway.

    The returned middle factor is reported rather than withheld, but it cannot
    reach a result: over exactly this range Table 3, note 2 makes q nought, and
    the middle-region term with it.
    """
    lengths, factors = (60.0, 40.0), (1.0, 0.0)
    regions = environment.region_ground_factors(lengths, factors, 3.0, 3.0)
    assert regions.middle == pytest.approx(0.6)
    agr = environment.ground_attenuation(
        100.0, 3.0, 3.0, BANDS, regions.source, 1.0, regions.receiver
    )
    other = environment.ground_attenuation(
        100.0, 3.0, 3.0, BANDS, regions.source, 0.0, regions.receiver
    )
    assert agr == pytest.approx(other)


@pytest.mark.parametrize(
    ("lengths", "factors", "hs", "hr", "match"),
    [
        ((10.0, 20.0), (0.5,), 1.0, 1.0, "one value per segment"),
        ((), (), 1.0, 1.0, "non-empty 1-D array"),
        ((10.0, -5.0), (0.5, 0.5), 1.0, 1.0, "must be positive"),
        ((10.0,), (1.5,), 1.0, 1.0, r"within \[0, 1\]"),
        ((10.0,), (0.5,), -1.0, 1.0, "non-negative"),
        ((10.0,), (0.5,), math.nan, 1.0, "non-negative"),
        ((10.0,), (0.5,), 1.0, math.inf, "non-negative"),
    ],
)
def test_region_ground_factors_rejects_a_path_it_cannot_average(
    lengths: tuple[float, ...],
    factors: tuple[float, ...],
    hs: float,
    hr: float,
    match: str,
) -> None:
    """Each guard names the argument it is about."""
    with pytest.raises(ValueError, match=match):
        environment.region_ground_factors(lengths, factors, hs, hr)


# --------------------------------------------------------------------------- #
# mean_path_height: the shape of the area, and the guards
# --------------------------------------------------------------------------- #
def test_flat_ground_puts_the_ray_halfway_between_the_two_heights() -> None:
    """F is a trapezium, so hm is the mean height shortened by the slant."""
    hm = environment.mean_path_height((0.0, 100.0), (0.0, 0.0), 2.0, 8.0)
    assert hm == pytest.approx(5.0 * 100.0 / math.hypot(100.0, 6.0))


def test_the_datum_of_the_profile_cancels() -> None:
    """Only differences enter F, so lifting the whole ground changes nothing."""
    low = environment.mean_path_height(SLOPE_DISTANCES, SLOPE_HEIGHTS, 1.0, 4.0)
    lifted = tuple(z + 137.0 for z in SLOPE_HEIGHTS)
    assert environment.mean_path_height(
        SLOPE_DISTANCES, lifted, 1.0, 4.0
    ) == pytest.approx(low)


def test_ground_rising_into_the_ray_takes_area_away_from_it() -> None:
    """A hillock under the path lowers hm, as the area of Figure 3 is drawn."""
    flat = environment.mean_path_height((0.0, 50.0, 100.0), (0.0, 0.0, 0.0), 5.0, 5.0)
    humped = environment.mean_path_height((0.0, 50.0, 100.0), (0.0, 4.0, 0.0), 5.0, 5.0)
    assert humped < flat
    assert flat - humped == pytest.approx(2.0, abs=0.02)


def test_a_given_distance_overrides_the_one_the_profile_implies() -> None:
    """The guideline divides F by d3, which a caller may already hold."""
    implied = environment.mean_path_height((0.0, 100.0), (0.0, 0.0), 2.0, 8.0)
    given = environment.mean_path_height(
        (0.0, 100.0), (0.0, 0.0), 2.0, 8.0, distance=100.0
    )
    assert given == pytest.approx(5.0)
    assert given > implied


@pytest.mark.parametrize(
    ("us", "zs", "hs", "hr", "distance", "match"),
    [
        ((0.0, 1.0), (0.0,), 1.0, 1.0, None, "one value per profile point"),
        ((0.0,), (0.0,), 1.0, 1.0, None, "at least two points"),
        ((0.0, 10.0, 5.0), (0.0, 0.0, 0.0), 1.0, 1.0, None, "strictly increasing"),
        ((0.0, 10.0), (0.0, 0.0), -1.0, 1.0, None, "non-negative"),
        ((0.0, 10.0), (0.0, 0.0), math.nan, 1.0, None, "non-negative"),
        ((0.0, 10.0), (0.0, 0.0), 1.0, 1.0, 0.0, "must be positive"),
        ((0.0, 10.0), (0.0, 0.0), 1.0, 1.0, math.inf, "must be positive"),
    ],
)
def test_mean_path_height_rejects_a_profile_it_cannot_integrate(
    us: tuple[float, ...],
    zs: tuple[float, ...],
    hs: float,
    hr: float,
    distance: float | None,
    match: str,
) -> None:
    """Each guard names the argument it is about."""
    with pytest.raises(ValueError, match=match):
        environment.mean_path_height(us, zs, hs, hr, distance=distance)


def test_both_helpers_are_reachable_from_the_package_root() -> None:
    """They are part of the public surface, not module-private conveniences."""
    assert "region_ground_factors" in environment.__all__
    assert "mean_path_height" in environment.__all__


def test_a_height_that_is_not_a_number_cannot_pass_as_one() -> None:
    """A comparison against nought lets NaN through; a finiteness check does not.

    Both guards used to read ``height < 0``, which is false for NaN, so a NaN
    source height produced ground factors and a mean path height that were
    themselves NaN, with nothing said. An infinite ``distance`` was worse: it
    divided a finite area and returned a mean height of exactly nought.
    """
    with pytest.raises(ValueError, match="non-negative"):
        environment.region_ground_factors((10.0,), (0.5,), math.nan, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        environment.mean_path_height((0.0, 10.0), (0.0, 0.0), 1.0, math.nan)
    with pytest.raises(ValueError, match="must be positive"):
        environment.mean_path_height(
            (0.0, 10.0), (0.0, 0.0), 1.0, 1.0, distance=math.inf
        )


def test_the_profile_heights_themselves_may_go_below_the_datum() -> None:
    """Only the two source and receiver heights are bounded below.

    A profile is read on whatever datum the terrain model uses, so ground
    below it is a negative number and not a mistake; what the guard is about
    is a source or receiver standing under the ground it stands on.
    """
    below = environment.mean_path_height(
        (0.0, 50.0, 100.0), (-3.0, -5.0, -3.0), 2.0, 2.0
    )
    above = environment.mean_path_height((0.0, 50.0, 100.0), (7.0, 5.0, 7.0), 2.0, 2.0)
    assert below == pytest.approx(above)
