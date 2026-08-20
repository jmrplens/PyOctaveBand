#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the heavy and soft impact sources (rubber ball and bang machine).

Every expected value below is transcribed from a primary printed source, with
the clause or table reference in the test docstring:

* **ISO 16283-2:2020 Table A.1** (printed p. 23) and **JIS A 1418-2:2019
  Tables A.1 and A.2** (printed pp. 6-7) for the impact force exposure level
  specification of both sources. ISO 10140-5:2010 Table F.1 prints the same
  rubber-ball column a third time.
* **ISO 717-2:2020 Table D.4** (printed p. 22) for the A-weighted rating,
  including the deliberately unrounded intermediate ``55,350 66...``, and
  **Table D.3** (printed p. 21) for the A-weighting corrections.
* **ISO 16283-2:2020 Formulae (4), (5) and (6)** (printed p. 4) for the
  standardization, checked against its own ``T = T0`` identity and against a
  published 25-band reproduction of the formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import building

# ---------------------------------------------------------------------------
# Printed oracles
# ---------------------------------------------------------------------------

#: ISO 16283-2:2020 Table A.1 = ISO 10140-5:2010 Table F.1 = JIS A 1418-2:2019
#: Table A.2 (impact force characteristic 2), rubber ball: (LFE, tolerance) in
#: dB re 1 N at 31,5 / 63 / 125 / 250 / 500 Hz.
RUBBER_BALL_LFE = ((39.0, 1.0), (31.0, 1.5), (23.0, 1.5), (17.0, 2.0), (12.5, 2.0))

#: JIS A 1418-2:2019 Table A.1 (impact force characteristic 1), bang machine.
BANG_MACHINE_LFE = ((47.0, 1.0), (40.0, 1.5), (22.0, 1.5), (11.5, 2.0), (5.5, 2.0))

#: ISO 717-2:2020 Table D.3, A-weighting correction A in dB for the
#: one-third-octave bands 50 Hz to 630 Hz.
ISO717_2_TABLE_D3_THIRD = (
    -30.3,
    -26.2,
    -22.4,
    -19.1,
    -16.2,
    -13.2,
    -10.8,
    -8.7,
    -6.6,
    -4.8,
    -3.2,
    -1.9,
)

#: ISO 717-2:2020 Table D.3, octave bands 63 Hz to 500 Hz.
ISO717_2_TABLE_D3_OCTAVE = (-26.2, -16.2, -8.7, -3.2)

#: ISO 717-2:2020 Table D.4: the printed worked example, a field measurement in
#: octave bands. Columns: Li,Fmax, A, corrected value; the printed result is
#: LiA,Fmax = 55,350 66... = 55 dB.
ISO717_2_TABLE_D4_LEVELS = (65.3, 64.5, 58.0, 55.8)
ISO717_2_TABLE_D4_CORRECTED = (39.1, 48.3, 49.3, 52.6)
ISO717_2_TABLE_D4_RATING = 55

#: Reproduction of ISO 16283-2:2020 Formula (4) over 25 one-third-octave bands,
#: from Gurtner, "Impact Noise Study using the ISO 16283-2 / ISO 140-5 Rubber
#: Ball", Edition 3 (2019-10-30), p. 19: a mid-sized room, V = 41,4 m3. This is
#: a reproduction check of the formula from grey literature, not accredited
#: reference measurement data; the standard prints no worked example of its own.
GURTNER_VOLUME = 41.4
GURTNER_REVERBERATION_TIME = (
    1.3,
    1.3,
    1.3,
    1.3,
    0.42,
    1.43,
    2.06,
    1.663,
    3.7,
    3.22,
    3.09,
    3.1,
    2.87,
    2.53,
    2.38,
    2.37,
    2.03,
    1.92,
    1.65,
    1.38,
    1.27,
    1.11,
    0.97,
    0.86,
    0.74,
)
GURTNER_MEASURED = (
    60.13,
    71.09,
    73.7,
    76.71,
    76.51,
    60.79,
    51.62,
    48.74,
    43.61,
    42.27,
    40.54,
    35.9,
    29.6,
    27.0,
    21.5,
    19.8,
    17.6,
    15.96,
    13.52,
    12.15,
    11.22,
    12.02,
    15.43,
    15.15,
    11.74,
)
GURTNER_STANDARDIZED = (
    56.7,
    67.7,
    70.3,
    73.3,
    76.3,
    57.2,
    47.2,
    44.8,
    38.1,
    37.0,
    35.3,
    30.7,
    24.5,
    22.2,
    16.8,
    15.1,
    13.2,
    11.7,
    9.6,
    8.6,
    7.9,
    9.0,
    12.8,
    12.8,
    9.8,
)


# ---------------------------------------------------------------------------
# Source specification (ISO 16283-2 Table A.1, JIS A 1418-2 Tables A.1 / A.2)
# ---------------------------------------------------------------------------


def test_rubber_ball_force_exposure_table() -> None:
    """ISO 16283-2:2020 Table A.1 / JIS A 1418-2:2019 Table A.2, digit by digit."""
    assert building.HEAVY_IMPACT_SOURCES["rubber_ball"] == RUBBER_BALL_LFE


def test_bang_machine_force_exposure_table() -> None:
    """JIS A 1418-2:2019 Table A.1 (impact force characteristic 1)."""
    assert building.HEAVY_IMPACT_SOURCES["bang_machine"] == BANG_MACHINE_LFE


def test_octave_bands_of_the_specification() -> None:
    """Both tables are printed over the five octaves 31,5 Hz to 500 Hz."""
    assert building.HEAVY_IMPACT_OCTAVE_BANDS == (
        31.5,
        63.0,
        125.0,
        250.0,
        500.0,
    )


@pytest.mark.parametrize(
    ("source", "table"),
    [("rubber_ball", RUBBER_BALL_LFE), ("bang_machine", BANG_MACHINE_LFE)],
)
def test_source_limits_bracket_the_printed_nominal(
    source: str, table: tuple[tuple[float, float], ...]
) -> None:
    """The tolerance band is the printed nominal +/- the printed tolerance."""
    freqs, lower, upper = building.heavy_impact_source_limits(source)
    np.testing.assert_allclose(freqs, building.HEAVY_IMPACT_OCTAVE_BANDS)
    np.testing.assert_allclose(lower, [v - t for v, t in table])
    np.testing.assert_allclose(upper, [v + t for v, t in table])


def test_rubber_ball_construction_example() -> None:
    """ISO 16283-2:2020 A.2.1/A.2.2 and JIS A 1418-2:2019 B.3.

    Drop height (100 +/- 1) cm from the bottom of the ball, effective mass
    (2,5 +/- 0,1) kg, coefficient of restitution 0,8 +/- 0,1, single-peak force
    pulse of 20 +/- 2 ms (JIS A.2 b) 2)).
    """
    spec = building.heavy_impact_source_specification("rubber_ball")
    assert spec.drop_height == pytest.approx(1.00)
    assert spec.drop_height_tolerance == pytest.approx(0.01)
    assert spec.effective_mass == pytest.approx(2.5)
    assert spec.effective_mass_tolerance == pytest.approx(0.1)
    assert spec.restitution == pytest.approx(0.8)
    assert spec.contact_time == pytest.approx(0.020)
    assert spec.contact_time_tolerance == pytest.approx(0.002)


def test_bang_machine_construction_example() -> None:
    """JIS A 1418-2:2019 B.2: tyre dropped from 85 cm, effective mass 7,3 +/- 0,2 kg."""
    spec = building.heavy_impact_source_specification("bang_machine")
    assert spec.drop_height == pytest.approx(0.85)
    assert spec.effective_mass == pytest.approx(7.3)
    assert spec.effective_mass_tolerance == pytest.approx(0.2)


def test_check_source_accepts_the_nominal_spectrum() -> None:
    """A source exactly on the printed nominal conforms in every band."""
    check = building.check_heavy_impact_source([v for v, _ in RUBBER_BALL_LFE])
    assert check.passed
    assert bool(np.all(check.within_tolerance))
    np.testing.assert_allclose(check.deviation, 0.0)


def test_check_source_rejects_a_band_outside_the_tolerance() -> None:
    """500 Hz has a +/- 2,0 dB tolerance, so 12,5 + 2,1 dB must fail there only."""
    measured = [v for v, _ in RUBBER_BALL_LFE]
    measured[-1] += 2.1
    check = building.check_heavy_impact_source(measured)
    assert not check.passed
    assert list(check.within_tolerance) == [True, True, True, True, False]


def test_check_source_accepts_the_tolerance_edge() -> None:
    """The printed tolerance is inclusive: 31,0 - 1,5 dB at 63 Hz still conforms."""
    measured = [v for v, _ in RUBBER_BALL_LFE]
    measured[1] -= 1.5
    assert building.check_heavy_impact_source(measured).passed


def test_bang_machine_spectrum_fails_the_rubber_ball_check() -> None:
    """The two characteristics differ by far more than their tolerances."""
    bang = [v for v, _ in BANG_MACHINE_LFE]
    assert building.check_heavy_impact_source(bang, "bang_machine").passed
    assert not building.check_heavy_impact_source(bang, "rubber_ball").passed


def test_check_source_rejects_a_wrong_band_count() -> None:
    with pytest.raises(ValueError, match="5 octave-band values"):
        building.check_heavy_impact_source([39.0, 31.0, 23.0])


def test_unknown_source_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="source"):
        building.heavy_impact_source_specification("tapping_machine")


# ---------------------------------------------------------------------------
# Impact force exposure level (Formula (A.1))
# ---------------------------------------------------------------------------


def test_force_exposure_level_of_a_rectangular_pulse() -> None:
    """Closed form: a pulse of constant F over t gives LFE = 10 lg(F**2 t).

    ISO 16283-2:2020 Formula (A.1) with F0 = 1 N and Tref = 1 s.
    """
    fs = 200_000.0
    duration = 0.020
    amplitude = 1500.0
    force = np.full(int(fs * duration) + 1, amplitude)
    expected = 10.0 * np.log10(amplitude**2 * duration)
    assert building.impact_force_exposure_level(force, fs) == pytest.approx(
        expected, abs=1e-6
    )


def test_force_exposure_level_of_a_half_sine_pulse() -> None:
    """Closed form: a half-sine of peak Fp over t integrates to Fp**2 t / 2.

    The 20 ms single-peak pulse shape that JIS A 1418-2:2019 A.2 b) requires of
    both heavy sources.
    """
    fs = 200_000.0
    duration = 0.020
    peak = 1500.0
    t = np.arange(0.0, duration, 1.0 / fs)
    force = peak * np.sin(np.pi * t / duration)
    expected = 10.0 * np.log10(peak**2 * duration / 2.0)
    assert building.impact_force_exposure_level(force, fs) == pytest.approx(
        expected, abs=1e-3
    )


def test_force_exposure_level_scales_6_db_per_force_doubling() -> None:
    """LFE is a squared-force integral, so doubling F adds 20 lg 2 = 6,02 dB."""
    fs = 100_000.0
    force = np.full(2001, 100.0)
    single = building.impact_force_exposure_level(force, fs)
    double = building.impact_force_exposure_level(2.0 * force, fs)
    assert double - single == pytest.approx(20.0 * np.log10(2.0), abs=1e-9)


def test_force_exposure_level_rejects_a_silent_record() -> None:
    silent = np.zeros(100)
    with pytest.raises(ValueError, match="non-zero energy"):
        building.impact_force_exposure_level(silent, 48_000.0)


# ---------------------------------------------------------------------------
# Standardization (ISO 16283-2 Formulae (4), (5), (6))
# ---------------------------------------------------------------------------


def test_reverberation_correction_vanishes_at_the_reference_time() -> None:
    """Formula (4) reduces to 10 lg(V/V0) when T = T0, since C = C0."""
    assert building.fast_reverberation_correction([0.5])[0] == pytest.approx(
        0.0, abs=1e-12
    )


def test_standardization_at_the_reference_time_is_the_volume_term_only() -> None:
    """T = T0 = 0,5 s: L' = Li,Fmax + 10 lg(V/V0) with V0 = 50 m3."""
    res = building.standardized_maximum_impact_level([70.0, 65.0], 100.0, 0.5)
    np.testing.assert_allclose(res.reverberation_correction, 0.0, atol=1e-12)
    assert res.volume_term == pytest.approx(10.0 * np.log10(2.0))
    np.testing.assert_allclose(
        res.standardized, [70.0 + res.volume_term, 65.0 + res.volume_term]
    )


def test_standardization_at_the_reference_volume_is_the_time_term_only() -> None:
    """V = V0 = 50 m3 leaves only the Fast reverberation correction."""
    res = building.standardized_maximum_impact_level([70.0], 50.0, 2.0)
    assert res.volume_term == pytest.approx(0.0)
    np.testing.assert_allclose(res.standardized, 70.0 - res.reverberation_correction)


def test_reverberation_correction_grows_with_reverberation_time() -> None:
    """A livelier room needs a larger correction: g(C) increases with C."""
    correction = building.fast_reverberation_correction([0.5, 1.0, 2.0, 4.0])
    assert bool(np.all(np.diff(correction) > 0.0))


def test_reverberation_correction_is_smooth_across_the_c_equals_one_pole() -> None:
    """C = 1 at T = 1,7275 s is a removable singularity of g(C) (limit 1/e).

    The correction there must sit between its neighbours, not blow up.
    """
    times = np.array([1.7274, 1.7275, 1.7276])
    correction = building.fast_reverberation_correction(times)
    assert np.all(np.isfinite(correction))
    assert correction[0] < correction[1] < correction[2]
    assert correction[1] == pytest.approx(
        0.5 * (correction[0] + correction[2]), abs=1e-6
    )


def test_reverberation_correction_is_accurate_inside_the_pole_guard() -> None:
    """The closed form for g(C) loses its digits within about 1e-6 of C = 1.

    Evaluated in double precision against a 60-digit reference, the printed
    expression is already 3,9e-4 dB out at |C - 1| = 5e-7 and 3,5e-3 dB out at
    1e-7, because the two powers cancel. Inside the guard band the removable
    limit 1/e is used instead, so the correction is flat to the last bit across
    the pole rather than wandering.
    """
    times = 1.7275 * np.array([1.0 - 5e-7, 1.0 - 1e-7, 1.0, 1.0 + 1e-7, 1.0 + 5e-7])
    correction = building.fast_reverberation_correction(times)
    assert float(np.ptp(correction)) < 1e-6
    # The exact value at C = 1: g(1) = 1/e, so the correction is
    # 10 lg[(1/e) / g(C0)] with C0 = 0,5/1,7275.
    assert float(correction[2]) == pytest.approx(3.234806783, abs=1e-6)


def test_standardization_reproduces_a_published_25_band_example() -> None:
    """Reproduction check of ISO 16283-2:2020 Formula (4) over 25 bands.

    Gurtner, "Impact Noise Study using the ISO 16283-2 / ISO 140-5 Rubber
    Ball", Edition 3 (2019-10-30), p. 19: V = 41,4 m3, per-band T and Li,Fmax
    printed with the resulting L'i,Fmax,V,T. The printed output carries one
    decimal, and its 50 Hz entry is affected by the author rounding C to 0,24,
    so the bound is 0,1 dB.
    """
    res = building.standardized_maximum_impact_level(
        GURTNER_MEASURED, GURTNER_VOLUME, GURTNER_REVERBERATION_TIME
    )
    np.testing.assert_allclose(res.standardized, GURTNER_STANDARDIZED, atol=0.1)


def test_standardization_rejects_a_mismatched_reverberation_time() -> None:
    with pytest.raises(ValueError, match="reverberation_time"):
        building.standardized_maximum_impact_level([70.0, 65.0], 50.0, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Octave conversion (ISO 16283-2 Formula (20))
# ---------------------------------------------------------------------------


def test_octave_conversion_sums_three_equal_thirds_to_plus_4_77_db() -> None:
    """Formula (20) is an energy sum: three equal thirds give +10 lg 3 dB."""
    octaves = building.heavy_impact_octave_levels([60.0] * 6)
    np.testing.assert_allclose(octaves, 60.0 + 10.0 * np.log10(3.0))


def test_octave_conversion_groups_consecutive_thirds() -> None:
    """Formula (20) sums the three thirds *inside* each octave, in band order.

    A flat spectrum cannot tell that apart from striding across the array, so
    this uses a falling spectrum: with 50 to 630 Hz in 12 thirds, grouping
    consecutively gives the four octaves below, while striding would put
    50/125/315 Hz in the first one and land 21,1 dB away.
    """
    thirds = np.array(
        [76.5, 60.8, 51.6, 48.7, 43.6, 42.3, 40.5, 35.9, 29.6, 27.0, 21.5, 19.8]
    )
    octaves = building.heavy_impact_octave_levels(thirds)
    assert octaves.size == 4
    expected = [
        10.0 * np.log10(np.sum(10.0 ** (thirds[3 * i : 3 * i + 3] / 10.0)))
        for i in range(4)
    ]
    np.testing.assert_allclose(octaves, expected, atol=1e-12)
    # And explicitly: the top octave is 400 + 500 + 630 Hz. Striding instead
    # would put 160 + 400 + 630 Hz there and read 48,9 dB, 20,2 dB too high.
    np.testing.assert_allclose(octaves, [76.629, 50.570, 42.047, 28.680], atol=0.01)


def test_octave_conversion_rejects_a_non_multiple_of_three() -> None:
    with pytest.raises(ValueError, match="multiple of 3"):
        building.heavy_impact_octave_levels([60.0, 61.0, 62.0, 63.0])


# ---------------------------------------------------------------------------
# A-weighted rating (ISO 717-2:2020 Annex D)
# ---------------------------------------------------------------------------


def test_a_weighting_table_third_octave() -> None:
    """ISO 717-2:2020 Table D.3, one-third-octave row, 50 Hz to 630 Hz."""
    table = building.HEAVY_IMPACT_A_WEIGHTING["third"]
    values = tuple(table[f] for f in sorted(table))
    assert values == ISO717_2_TABLE_D3_THIRD
    assert sorted(table) == [
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
    ]


def test_a_weighting_table_octave() -> None:
    """ISO 717-2:2020 Table D.3, octave row, 63 Hz to 500 Hz."""
    table = building.HEAVY_IMPACT_A_WEIGHTING["octave"]
    values = tuple(table[f] for f in sorted(table))
    assert values == ISO717_2_TABLE_D3_OCTAVE
    assert sorted(table) == [63.0, 125.0, 250.0, 500.0]


def test_octave_a_weighting_matches_the_third_octave_row() -> None:
    """The octave row of Table D.3 repeats the mid-band third-octave values."""
    third = building.HEAVY_IMPACT_A_WEIGHTING["third"]
    for band, value in building.HEAVY_IMPACT_A_WEIGHTING["octave"].items():
        assert third[band] == value


def test_iso717_2_table_d4_worked_example() -> None:
    """ISO 717-2:2020 Table D.4 (printed p. 22), digit for digit.

    A field measurement in octave bands: Li,Fmax = 65,3 / 64,5 / 58,0 / 55,8 dB
    at 63 / 125 / 250 / 500 Hz gives the corrected values 39,1 / 48,3 / 49,3 /
    52,6 dB and LiA,Fmax = 55,350 66... = 55 dB.
    """
    res = building.a_weighted_maximum_impact_level(ISO717_2_TABLE_D4_LEVELS)
    assert res.band == "octave"
    np.testing.assert_allclose(res.a_weighting, ISO717_2_TABLE_D3_OCTAVE)
    np.testing.assert_allclose(res.corrected, ISO717_2_TABLE_D4_CORRECTED, atol=5e-2)
    assert res.unrounded == pytest.approx(55.35066, abs=5e-5)
    assert res.rating == ISO717_2_TABLE_D4_RATING


def test_rating_rounds_halves_up() -> None:
    """Annex D note a): XX,5 rounds to XX + 1.

    Four equal octave levels L give XiA,Fmax = 10 lg(sum 10**((L+Aj)/10)); the
    level that lands the sum exactly on x,5 is solved for and must round up.
    """
    a = np.asarray(ISO717_2_TABLE_D3_OCTAVE)
    offset = 10.0 * np.log10(np.sum(10.0 ** (a / 10.0)))
    level = 55.5 - offset
    res = building.a_weighted_maximum_impact_level([level] * 4)
    assert res.unrounded == pytest.approx(55.5, abs=1e-9)
    assert res.rating == 56


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (54.5, 55),  # half-to-even would give 54
        (55.5, 56),
        (53.5, 54),
        (54.4999, 54),
        (-0.5, 0),  # half-away-from-zero would give -1
        (-1.5, -1),
    ],
)
def test_annex_d_rounding_is_half_up(value: float, expected: int) -> None:
    """ISO 717-2:2020 Annex D note a: XX,Y rounds to XX + 1 once Y reaches 5.

    That is half-up toward positive infinity. It differs from Python's
    round-half-to-even at 54,5 and from half-away-from-zero at -0,5, and the
    formula's own note warns that a rounding slip here costs a whole decibel
    in the reported single number.
    """
    from phonometry.building.measurement.heavy_impact import _round_half_up

    assert _round_half_up(value) == expected


def test_rating_is_dominated_by_the_low_bands() -> None:
    """The A-weighting is 23 dB steeper at 63 Hz than at 500 Hz.

    Adding 1 dB at 500 Hz therefore moves the rating far more than 1 dB at
    63 Hz for the Table D.4 spectrum.
    """
    base = building.a_weighted_maximum_impact_level(ISO717_2_TABLE_D4_LEVELS).unrounded
    low = list(ISO717_2_TABLE_D4_LEVELS)
    low[0] += 1.0
    high = list(ISO717_2_TABLE_D4_LEVELS)
    high[-1] += 1.0
    delta_low = building.a_weighted_maximum_impact_level(low).unrounded - base
    delta_high = building.a_weighted_maximum_impact_level(high).unrounded - base
    assert delta_high > delta_low


def test_third_octave_rating_uses_twelve_bands() -> None:
    """A one-third-octave measurement is rated in one-third octaves (Clause D.3)."""
    res = building.a_weighted_maximum_impact_level([60.0] * 12)
    assert res.band == "third"
    assert res.frequencies.size == 12
    expected = 10.0 * np.log10(
        np.sum(10.0 ** ((60.0 + np.asarray(ISO717_2_TABLE_D3_THIRD)) / 10.0))
    )
    assert res.unrounded == pytest.approx(expected)


def test_rating_rejects_an_unsupported_band_count() -> None:
    with pytest.raises(ValueError, match="12 one-third-octave values"):
        building.a_weighted_maximum_impact_level([60.0] * 7)


def test_rating_rejects_mismatched_frequencies() -> None:
    with pytest.raises(ValueError, match="rating bands"):
        building.a_weighted_maximum_impact_level(
            ISO717_2_TABLE_D4_LEVELS, [125.0, 250.0, 500.0, 1000.0]
        )


def test_rating_and_standardization_compose() -> None:
    """L'iA,Fmax,V,T is the Annex D rating of the standardized spectrum.

    Table D.2 of ISO 717-2:2020 pairs the two: Formula (D.1) is applied to the
    ISO 16283-2 Formula (4) output, so a pure volume gain of 10 lg(V/V0) must
    pass straight through to the single number.
    """
    standardized = building.standardized_maximum_impact_level(
        ISO717_2_TABLE_D4_LEVELS, 100.0, 0.5
    )
    rated = building.a_weighted_maximum_impact_level(standardized.standardized)
    plain = building.a_weighted_maximum_impact_level(ISO717_2_TABLE_D4_LEVELS)
    assert rated.unrounded - plain.unrounded == pytest.approx(10.0 * np.log10(2.0))
