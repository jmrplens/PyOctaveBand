#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.environment.assessment.spain` (RD 1367/2007).

Two independent oracle families anchor this module.

**The regulation itself** (Real Decreto 1367/2007, BOE-A-2007-18397,
consolidated text): the printed limit tables of Annex II (quality objectives)
and Annex III (emitter immission limits), the ``Kt``/``Kf``/``Ki`` threshold
tables of Annex IV A.3.3, the evaluation periods of Annex I A.1 and the
compliance criteria of Article 25. Every table value below is transcribed from
that text.

**Aviles Lopez & Perera Martin, Manual de acustica ambiental y arquitectonica
(Paraninfo), Ejemplos 3.1 to 3.3** (printed pages 171 to 175): a complete
worked assessment of one activity, from the per-phase corrections through the
period integration and the annual average to the Article 25 verdict. The
published values are ``LKeq,d`` = 57 dB, ``LKeq,e`` = 51 dB, ``LK,d`` = 56 dB,
``LK,e`` = 50 dB, and a verdict that fails for a new activity (``LK,d`` exceeds
the 55 dB of Annex III Table B1 for area type a) but passes for an activity
already in operation, judged under Article 25.2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.environment.assessment import spain as rd

# --------------------------------------------------------------------------- #
# Manual Ejemplo 3.1: the two measured noise phases
# --------------------------------------------------------------------------- #
#: Ejemplo 3.1, printed page 171: with the noisy machine running the
#: background-corrected LAeq,5s is 50 dB with Kt 6, Kf 3, Ki 0, so K = 9 and
#: LKeq,5s = 59 dB; for the rest of the opening hours LAeq,5s is 48 dB with
#: Kt 3, Kf 3, Ki 0, so K = 6 and LKeq,5s = 54 dB.
_NOISY = {"laeq": 50.0, "kt": 6.0, "kf": 3.0, "ki": 0.0, "k": 9.0, "lkeq": 59.0}
_REST = {"laeq": 48.0, "kt": 3.0, "kf": 3.0, "ki": 0.0, "k": 6.0, "lkeq": 54.0}


def _example_periods() -> dict[str, list[rd.NoisePhase]]:
    """The day and evening phase decomposition of Manual Ejemplo 3.1.

    The day period (07:00-19:00) splits into 2 h with the activity shut down,
    6 h with the noisy machine running and 4 h with the remaining sources; the
    evening period (19:00-23:00) into 2 h with the remaining sources and 2 h
    shut down. A shut-down phase carries 0 dB, as the book enters it.
    """
    return {
        "day": [
            rd.NoisePhase(2.0, 0.0, label="closed"),
            rd.NoisePhase(6.0, _NOISY["laeq"], kt=_NOISY["kt"], kf=_NOISY["kf"]),
            rd.NoisePhase(4.0, _REST["laeq"], kt=_REST["kt"], kf=_REST["kf"]),
        ],
        "evening": [
            rd.NoisePhase(2.0, _REST["laeq"], kt=_REST["kt"], kf=_REST["kf"]),
            rd.NoisePhase(2.0, 0.0, label="closed"),
        ],
    }


# --------------------------------------------------------------------------- #
# Corrections Kt, Kf, Ki (Annex IV A.3.3)
# --------------------------------------------------------------------------- #
def test_phase_corrections_match_ejemplo_3_1() -> None:
    """The per-phase K and LKeq of Manual Ejemplo 3.1, printed page 171."""
    for spec in (_NOISY, _REST):
        assert rd.total_correction(spec["kt"], spec["kf"], spec["ki"]) == spec["k"]
        level = rd.corrected_level(
            spec["laeq"], kt=spec["kt"], kf=spec["kf"], ki=spec["ki"]
        )
        assert level == pytest.approx(spec["lkeq"], abs=1e-12)


def test_total_correction_is_capped_at_nine_decibels() -> None:
    """Annex IV A.3.3: "El valor maximo de la suma Kt + Kf + Ki no sera superior a 9 dB"."""
    assert rd.total_correction(6.0, 6.0, 6.0) == rd.RD1367_MAX_CORRECTION == 9.0
    assert rd.total_correction(6.0, 3.0, 0.0) == 9.0
    assert rd.corrected_level(50.0, kt=6.0, kf=6.0, ki=6.0) == pytest.approx(59.0)


@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        (0.0, 0.0),
        (9.9, 0.0),
        (10.0, 0.0),  # "Si Lf <= 10" -> 0 dB, the boundary belongs to the row above
        (10.1, 3.0),
        (15.0, 3.0),  # "10 < Lf <= 15" -> 3 dB, upper boundary included
        (15.1, 6.0),
        (30.0, 6.0),
    ],
)
def test_low_frequency_and_impulsive_thresholds(
    difference: float, expected: float
) -> None:
    """Annex IV A.3.3 Kf and Ki tables, including both boundary rows.

    The printed 3 dB row reads "Si 10 > Lf <= 15", an unsatisfiable condition
    and a typeset inversion of "10 < Lf <= 15"; the bracketing rows leave no
    other consistent reading (see docs/ERRATA.md).
    """
    assert rd.low_frequency_correction(50.0 + difference, 50.0) == expected
    assert rd.impulsive_correction(50.0 + difference, 50.0) == expected


@pytest.mark.parametrize(
    ("centre", "lower", "upper"),
    [
        (20.0, 8.0, 12.0),
        (125.0, 8.0, 12.0),
        (160.0, 5.0, 8.0),
        (400.0, 5.0, 8.0),
        (500.0, 3.0, 5.0),
        (10000.0, 3.0, 5.0),
    ],
)
def test_tonal_correction_thresholds_per_band_range(
    centre: float, lower: float, upper: float
) -> None:
    """The three band ranges of the Annex IV A.3.3 Kt table and their thresholds.

    ``Lt < lower`` gives 0 dB, ``lower <= Lt <= upper`` gives 3 dB and
    ``Lt > upper`` gives 6 dB, with the ranges 20-125 Hz (8/12 dB),
    160-400 Hz (5/8 dB) and 500 Hz-10 kHz (3/5 dB).
    """
    ratio = 10.0 ** (1.0 / 10.0)
    freqs = [centre / ratio, centre, centre * ratio]
    for emergence, expected in (
        (lower - 0.5, 0.0),
        (lower, 3.0),
        (upper, 3.0),
        (upper + 0.5, 6.0),
    ):
        # Neighbours both at 60 dB, so Ls = 60 dB and Lt = level - 60.
        levels = [60.0, 60.0 + emergence, 60.0]
        result = rd.tonal_correction(levels, freqs)
        assert result.correction == expected
        assert result.differences[1] == pytest.approx(emergence, abs=1e-12)


def test_tonal_correction_uses_arithmetic_mean_of_neighbours() -> None:
    """Annex IV A.3.3 b: Ls is the arithmetic mean of the two adjacent bands.

    With neighbours at 50 dB and 70 dB the mean is 60 dB, so a 66 dB band in
    the 500 Hz-10 kHz range emerges by 6 dB and earns the 6 dB Kt.
    """
    result = rd.tonal_correction([50.0, 66.0, 70.0], [800.0, 1000.0, 1250.0])
    assert result.differences[1] == pytest.approx(6.0)
    assert result.correction == 6.0
    assert result.governing_frequency == 1000.0


def test_tonal_correction_takes_the_largest_of_several_tones() -> None:
    """Annex IV A.3.3 d: with more than one emergent tone the largest Kt governs."""
    freqs = [400.0, 500.0, 630.0, 800.0, 1000.0]
    # 500 Hz emerges by 4 dB (3 dB row), 1000 Hz is an end band and ignored;
    # 800 Hz emerges by 9 dB, above the 5 dB upper threshold, so 6 dB governs.
    levels = [60.0, 64.0, 60.0, 69.0, 60.0]
    result = rd.tonal_correction(levels, freqs)
    assert result.band_corrections[1] == 3.0
    assert result.band_corrections[3] == 6.0
    assert result.correction == 6.0
    assert result.governing_frequency == 800.0


def test_tonal_correction_ignores_bands_outside_the_table() -> None:
    """The Kt table only covers 20 Hz to 10 kHz; bands outside it get no Kt."""
    result = rd.tonal_correction([60.0, 90.0, 60.0], [10000.0, 12500.0, 16000.0])
    assert result.correction == 0.0
    assert result.governing_frequency is None
    assert math.isnan(float(result.differences[1]))


def test_tonal_correction_end_bands_are_not_evaluated() -> None:
    """The first and last band have no pair of neighbours, so they carry NaN."""
    result = rd.tonal_correction([90.0, 60.0, 90.0], [500.0, 630.0, 800.0])
    assert math.isnan(float(result.differences[0]))
    assert math.isnan(float(result.differences[-1]))


# --------------------------------------------------------------------------- #
# Rounding and period integration (Annex IV A.3.4.2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [(56.82, 57), (50.99, 51), (56.19, 56), (56.5, 57), (56.49, 56), (0.0, 0)],
)
def test_round_reported_level(value: float, expected: int) -> None:
    """Annex IV A.3.4.2: add 0,5 dB and take the integer part."""
    assert rd.round_reported_level(value) == expected


def test_evaluation_period_levels_match_ejemplo_3_1() -> None:
    """Manual Ejemplo 3.1: LKeq,d = 57 dB and LKeq,e = 51 dB (printed page 172).

    The period level is the duration-weighted energy mean of the phase levels
    (Annex IV A.3.4.2 b), rounded as the regulation prescribes.
    """
    periods = _example_periods()
    day = rd.evaluation_period_level(periods["day"], hours=12.0)
    evening = rd.evaluation_period_level(periods["evening"], hours=4.0)
    assert rd.round_reported_level(day) == 57
    assert rd.round_reported_level(evening) == 51
    # Independent recomputation of the same energy means.
    assert day == pytest.approx(
        10.0 * math.log10((2 * 10**0.0 + 6 * 10**5.9 + 4 * 10**5.4) / 12.0), abs=1e-9
    )
    assert evening == pytest.approx(
        10.0 * math.log10((2 * 10**5.4 + 2 * 10**0.0) / 4.0), abs=1e-9
    )


def test_long_term_levels_match_ejemplo_3_2() -> None:
    """Manual Ejemplo 3.2: LK,d = 56 dB and LK,e = 50 dB (printed page 172).

    The activity runs 303 days a year and is closed 62 (July and August), and
    the annual index is the energy mean of the daily levels over the year.
    """
    day = rd.long_term_corrected_level([57.0, 0.0], weights=[303.0, 62.0])
    evening = rd.long_term_corrected_level([51.0, 0.0], weights=[303.0, 62.0])
    assert rd.round_reported_level(day) == 56
    assert rd.round_reported_level(evening) == 50
    assert day == pytest.approx(
        10.0 * math.log10((303 * 10**5.7 + 62 * 10**0.0) / 365.0), abs=1e-9
    )


def test_long_term_level_unweighted_is_the_plain_energy_mean() -> None:
    """Annex I A.2 d with one sample per day: LK,x is the energy mean."""
    value = rd.long_term_corrected_level([60.0, 60.0, 60.0])
    assert value == pytest.approx(60.0, abs=1e-12)


def test_evaluation_period_rejects_durations_that_do_not_sum_to_the_period() -> None:
    """Annex IV A.3.4.2 b requires "La suma de los Ti = T"."""
    phases = [rd.NoisePhase(2.0, 50.0)]
    with pytest.raises(ValueError, match="sum Ti = T"):
        rd.evaluation_period_level(phases, hours=12.0)


# --------------------------------------------------------------------------- #
# Evaluation periods (Annex I A.1)
# --------------------------------------------------------------------------- #
def test_evaluation_periods_and_clock_limits() -> None:
    """Annex I A.1: 12 h day, 4 h evening, 8 h night at 07:00, 19:00 and 23:00."""
    assert rd.RD1367_EVALUATION_PERIODS == ("day", "evening", "night")
    assert rd.RD1367_PERIOD_HOURS == {"day": 12.0, "evening": 4.0, "night": 8.0}
    assert sum(rd.RD1367_PERIOD_HOURS.values()) == 24.0
    assert rd.RD1367_PERIOD_CLOCK_LIMITS == {
        "day": (7, 19),
        "evening": (19, 23),
        "night": (23, 7),
    }


# --------------------------------------------------------------------------- #
# Limit tables (Annexes II and III)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("area", "expected"),
    [
        ("e", (60.0, 60.0, 50.0)),
        ("a", (65.0, 65.0, 55.0)),
        ("d", (70.0, 70.0, 65.0)),
        ("c", (73.0, 73.0, 63.0)),
        ("b", (75.0, 75.0, 65.0)),
    ],
)
def test_outdoor_quality_objectives_annex_ii_table_a(
    area: str, expected: tuple[float, float, float]
) -> None:
    """Annex II Table A as amended by RD 1038/2012, every area type."""
    limits = rd.outdoor_quality_objectives(area)
    day, evening, night = expected
    assert limits.day == day
    assert limits.evening == evening
    assert limits.night == night
    assert limits.index == "Lx"


def test_outdoor_objectives_of_newly_urbanised_areas_are_five_decibels_stricter() -> (
    None
):
    """Article 14.2: for the rest of the urbanised areas, Table A minus 5 dB."""
    existing = rd.outdoor_quality_objectives("a")
    new = rd.outdoor_quality_objectives("a", urbanisation="new")
    assert (new.day, new.evening, new.night) == (60.0, 60.0, 50.0)
    for period in rd.RD1367_EVALUATION_PERIODS:
        assert new[period] == existing[period] - 5.0


@pytest.mark.parametrize(
    ("use", "room", "expected"),
    [
        ("residential", "living", (45.0, 45.0, 35.0)),
        ("residential", "bedrooms", (40.0, 40.0, 30.0)),
        ("sanitary", "living", (45.0, 45.0, 35.0)),
        ("sanitary", "bedrooms", (40.0, 40.0, 30.0)),
        ("educational", "classrooms", (40.0, 40.0, 40.0)),
        ("educational", "reading_rooms", (35.0, 35.0, 35.0)),
    ],
)
def test_indoor_quality_objectives_annex_ii_table_b(
    use: str, room: str, expected: tuple[float, float, float]
) -> None:
    """Annex II Table B, every row of the indoor quality objectives."""
    limits = rd.indoor_quality_objectives(use, room)
    day, evening, night = expected
    assert limits.day == day
    assert limits.evening == evening
    assert limits.night == night


@pytest.mark.parametrize(
    ("use", "expected"),
    [("residential", 75.0), ("sanitary", 72.0), ("educational", 72.0)],
)
def test_vibration_quality_objectives_annex_ii_table_c(
    use: str, expected: float
) -> None:
    """Annex II Table C: the Law vibration objective by building use."""
    assert rd.vibration_quality_objective(use) == expected


@pytest.mark.parametrize(
    ("area", "expected"),
    [
        ("e", (55.0, 55.0, 45.0)),
        ("a", (60.0, 60.0, 50.0)),
        ("d", (65.0, 65.0, 55.0)),
        ("c", (68.0, 68.0, 58.0)),
        ("b", (70.0, 70.0, 60.0)),
    ],
)
def test_infrastructure_limits_annex_iii_table_a1(
    area: str, expected: tuple[float, float, float]
) -> None:
    """Annex III Table A1: new road, rail and airport infrastructure."""
    limits = rd.infrastructure_limits(area)
    day, evening, night = expected
    assert limits.day == day
    assert limits.evening == evening
    assert limits.night == night


@pytest.mark.parametrize(
    ("area", "expected"),
    [("e", 80.0), ("a", 85.0), ("d", 88.0), ("c", 90.0), ("b", 90.0)],
)
def test_max_infrastructure_limits_annex_iii_table_a2(
    area: str, expected: float
) -> None:
    """Annex III Table A2: the LAmax limit for rail and airport infrastructure."""
    assert rd.max_infrastructure_limit(area) == expected


@pytest.mark.parametrize(
    ("area", "expected"),
    [
        ("e", (50.0, 50.0, 40.0)),
        ("a", (55.0, 55.0, 45.0)),
        ("d", (60.0, 60.0, 50.0)),
        ("c", (63.0, 63.0, 53.0)),
        ("b", (65.0, 65.0, 55.0)),
    ],
)
def test_activity_limits_annex_iii_table_b1(
    area: str, expected: tuple[float, float, float]
) -> None:
    """Annex III Table B1: port infrastructure and activities.

    The area type a row (55/55/45 dB) is the one Manual Ejemplo 3.3 assesses
    its activity against (its Tabla 3.7, printed page 173).
    """
    limits = rd.activity_limits(area)
    day, evening, night = expected
    assert limits.day == day
    assert limits.evening == evening
    assert limits.night == night
    assert limits.index == "LK,x"


@pytest.mark.parametrize(
    ("use", "room", "expected"),
    [
        ("residential", "living", (40.0, 40.0, 30.0)),
        ("residential", "bedrooms", (35.0, 35.0, 25.0)),
        ("office", "professional_offices", (35.0, 35.0, 35.0)),
        ("office", "offices", (40.0, 40.0, 40.0)),
        ("sanitary", "living", (40.0, 40.0, 30.0)),
        ("sanitary", "bedrooms", (35.0, 35.0, 25.0)),
        ("educational", "classrooms", (35.0, 35.0, 35.0)),
        ("educational", "reading_rooms", (30.0, 30.0, 30.0)),
    ],
)
def test_adjacent_premises_limits_annex_iii_table_b2(
    use: str, room: str, expected: tuple[float, float, float]
) -> None:
    """Annex III Table B2, every row of the noise transmitted to adjacent premises."""
    limits = rd.adjacent_premises_limits(use, room)
    day, evening, night = expected
    assert limits.day == day
    assert limits.evening == evening
    assert limits.night == night


def test_area_type_aliases_and_catalogue() -> None:
    """The letter codes of Article 7 of Ley 37/2003 and their spelled-out aliases."""
    assert set(rd.ACOUSTIC_AREA_TYPES) == {"e", "a", "d", "c", "b", "f"}
    assert (
        rd.activity_limits("residential").as_dict() == rd.activity_limits("a").as_dict()
    )
    assert rd.activity_limits("INDUSTRIAL").day == rd.activity_limits("b").day
    assert rd.outdoor_quality_objectives("sanitary").night == 50.0


def test_regulation_limits_lookup_by_period() -> None:
    """A limit row is indexable by evaluation period and exportable as a dict."""
    limits = rd.activity_limits("a")
    assert limits["day"] == 55.0
    assert limits["night"] == 45.0
    assert limits.as_dict() == {"day": 55.0, "evening": 55.0, "night": 45.0}


# --------------------------------------------------------------------------- #
# Compliance assessment (Article 25) - Manual Ejemplo 3.3
# --------------------------------------------------------------------------- #
def test_assess_activity_reproduces_ejemplo_3_3_for_a_new_activity() -> None:
    """Manual Ejemplo 3.3, printed page 175: a new activity that does not comply.

    Against Annex III Table B1 area type a (55 dB day and evening), the book
    prints the three criteria as LKeq,5s <= 60 (= 55 + 5), LKeq,T <= 58
    (= 55 + 3) and LK,T <= 55. The phase and daily criteria pass, but
    LK,d = 56 dB exceeds 55 dB, so the activity does not comply.
    """
    verdict = rd.assess_activity(
        _example_periods(), rd.activity_limits("a"), operating_days=303
    )
    day, evening = verdict.periods
    assert day.period == "day"
    assert day.reported_level == 57
    assert day.reported_long_term == 56
    assert day.max_phase_level == 59.0
    assert (day.limit, day.daily_limit, day.phase_limit) == (55.0, 58.0, 60.0)
    assert day.phase_pass is True
    assert day.daily_pass is True
    assert day.long_term_pass is False
    assert day.complies is False

    assert evening.reported_level == 51
    assert evening.reported_long_term == 50
    assert evening.complies is True

    assert verdict.complies is False


def test_assess_activity_reproduces_ejemplo_3_3_for_an_operating_activity() -> None:
    """Manual Ejemplo 3.3: the same activity complies if it is not new.

    Article 25.2 subjects an activity already in operation only to the daily
    and phase criteria, so the annual LK,d = 56 dB no longer disqualifies it.
    """
    verdict = rd.assess_activity(
        _example_periods(),
        rd.activity_limits("a"),
        operating_days=303,
        new_activity=False,
    )
    assert all(p.long_term_pass is None for p in verdict.periods)
    assert verdict.complies is True


def test_assess_activity_accepts_annual_levels_directly() -> None:
    """The annual index may be supplied instead of derived from operating days."""
    verdict = rd.assess_activity(
        _example_periods(),
        rd.activity_limits("a"),
        long_term_levels={"day": 56.19, "evening": 50.19},
    )
    assert verdict.periods[0].reported_long_term == 56
    assert verdict.periods[0].long_term_pass is False


def test_new_activity_without_annual_information_is_refused() -> None:
    """Article 25.1 b i is mandatory for a new activity, so it cannot be skipped.

    Returning a compliant verdict while never evaluating the annual criterion
    would assert something the assessment has not shown, so the missing input
    is an error rather than a silently omitted check.
    """
    periods, limits = _example_periods(), rd.activity_limits("a")
    with pytest.raises(ValueError, match="annual index"):
        rd.assess_activity(periods, limits)


def test_operating_activity_without_annual_information_is_allowed() -> None:
    """Article 25.2 subjects an activity in operation only to the other two.

    That inspection needs no annual index, so it is assessed without one and
    the annual criterion is reported as not evaluated.
    """
    verdict = rd.assess_activity(
        _example_periods(), rd.activity_limits("a"), new_activity=False
    )
    day = verdict.periods[0]
    assert day.long_term_corrected_level is None
    assert day.reported_long_term is None
    assert day.long_term_pass is None
    assert day.complies is True
    assert verdict.complies is True


def test_assess_activity_flags_a_phase_that_exceeds_the_five_decibel_allowance() -> (
    None
):
    """Article 25.1 b iii: no measured LKeq,Ti more than 5 dB above the limit."""
    phases = {"night": [rd.NoisePhase(8.0, 51.0)]}  # limit 45 dB, allowance 50 dB
    verdict = rd.assess_activity(phases, rd.activity_limits("a"), new_activity=False)
    night = verdict.periods[0]
    assert night.phase_limit == 50.0
    assert night.phase_pass is False
    assert night.complies is False


def test_assess_activity_reports_only_the_periods_supplied() -> None:
    """A period absent from the measurements is not assessed (the night here)."""
    verdict = rd.assess_activity(
        _example_periods(), rd.activity_limits("a"), operating_days=303
    )
    assert [p.period for p in verdict.periods] == ["day", "evening"]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_noise_phase_validation() -> None:
    """A phase needs a positive duration, a finite level and non-negative corrections."""
    with pytest.raises(ValueError, match="positive"):
        rd.NoisePhase(0.0, 50.0)
    with pytest.raises(ValueError, match="positive"):
        rd.NoisePhase(-1.0, 50.0)
    with pytest.raises(ValueError, match="finite"):
        rd.NoisePhase(1.0, math.nan)
    with pytest.raises(ValueError, match="0, 3 or 6"):
        rd.NoisePhase(1.0, 50.0, kt=-3.0)


def test_corrections_are_graded_zero_three_or_six() -> None:
    """Annex IV A.3.3 grades each correction 0, 3 or 6 dB and nothing else.

    An intermediate reading such as 4,5 dB is not a value any of the three
    tables can produce, so it is rejected rather than silently accepted.
    """
    assert rd.RD1367_CORRECTION_VALUES == (0.0, 3.0, 6.0)
    for step in rd.RD1367_CORRECTION_VALUES:
        assert rd.total_correction(step, step, 0.0) == min(2 * step, 9.0)
    for bad in (1.0, 4.5, 7.0, -3.0):
        with pytest.raises(ValueError, match="0, 3 or 6"):
            rd.total_correction(bad)
        with pytest.raises(ValueError, match="0, 3 or 6"):
            rd.corrected_level(50.0, kf=bad)


def test_published_constant_tables_are_read_only() -> None:
    """The evaluation periods and area catalogue are published constants."""
    for table in (
        rd.RD1367_PERIOD_HOURS,
        rd.RD1367_PERIOD_CLOCK_LIMITS,
        rd.ACOUSTIC_AREA_TYPES,
    ):
        with pytest.raises(TypeError):
            table["day"] = 0  # type: ignore[index]


def test_tonal_result_does_not_alias_the_caller_s_arrays() -> None:
    """Mutating the input spectrum afterwards must not rewrite the result."""
    levels = np.array([50.0, 66.0, 70.0])
    freqs = np.array([800.0, 1000.0, 1250.0])
    result = rd.tonal_correction(levels, freqs)
    levels[1] = 0.0
    freqs[1] = 1.0
    assert result.levels[1] == pytest.approx(66.0)
    assert result.frequencies[1] == pytest.approx(1000.0)


def test_tonal_correction_validation() -> None:
    """The Kt spectrum must be one-dimensional, aligned, finite and ascending."""
    with pytest.raises(ValueError, match="same length"):
        rd.tonal_correction([60.0, 60.0], [100.0, 125.0, 160.0])
    with pytest.raises(ValueError, match="At least three"):
        rd.tonal_correction([60.0, 60.0], [100.0, 125.0])
    with pytest.raises(ValueError, match="finite"):
        rd.tonal_correction([60.0, math.nan, 60.0], [100.0, 125.0, 160.0])
    with pytest.raises(ValueError, match="strictly ascending"):
        rd.tonal_correction([60.0, 60.0, 60.0], [160.0, 125.0, 100.0])
    with pytest.raises(ValueError, match="positive"):
        rd.tonal_correction([60.0, 60.0, 60.0], [0.0, 125.0, 160.0])
    two_dimensional = np.zeros((2, 3))
    with pytest.raises(ValueError, match="one-dimensional"):
        rd.tonal_correction(two_dimensional, two_dimensional)


def test_limit_lookup_validation() -> None:
    """Unknown area types, room combinations and periods are rejected."""
    with pytest.raises(ValueError, match="area_type"):
        rd.activity_limits("z")
    with pytest.raises(ValueError, match="building_use"):
        rd.vibration_quality_objective("industrial")
    with pytest.raises(ValueError, match="Unknown"):
        rd.indoor_quality_objectives("residential", "classrooms")
    with pytest.raises(ValueError, match="urbanisation"):
        rd.outdoor_quality_objectives("a", urbanisation="future")
    with pytest.raises(ValueError, match="period"):
        rd.activity_limits("a")["afternoon"]


def test_assess_activity_validation() -> None:
    """Empty inputs, unknown periods and impossible year fractions are rejected."""
    limits = rd.activity_limits("a")
    with pytest.raises(ValueError, match="At least one evaluation period"):
        rd.assess_activity({}, limits)
    empty_day: dict[str, list[rd.NoisePhase]] = {"day": []}
    with pytest.raises(ValueError, match="no noise phases"):
        rd.assess_activity(empty_day, limits)
    afternoon = {"afternoon": [rd.NoisePhase(1.0, 50.0)]}
    with pytest.raises(ValueError, match="evaluation period"):
        rd.assess_activity(afternoon, limits)
    day = {"day": [rd.NoisePhase(12.0, 50.0)]}
    with pytest.raises(ValueError, match="operating_days"):
        rd.assess_activity(day, limits, operating_days=400)
    with pytest.raises(ValueError, match="year_days"):
        rd.assess_activity(day, limits, operating_days=10, year_days=0)
    # 'year_days' is validated even when it is the only annual input given,
    # and a fractional count of days is rejected rather than truncated.
    with pytest.raises(ValueError, match="year_days"):
        rd.assess_activity(
            day,
            limits,
            long_term_levels={"day": 50.0},
            year_days=365.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="whole number"):
        rd.assess_activity(
            day,
            limits,
            operating_days=10.5,  # type: ignore[arg-type]
        )


def test_long_term_level_validation() -> None:
    """The annual average needs at least one finite level and matching weights."""
    with pytest.raises(ValueError, match="At least one"):
        rd.long_term_corrected_level([])
    with pytest.raises(ValueError, match="finite"):
        rd.long_term_corrected_level([math.nan])
    with pytest.raises(ValueError, match="one value per"):
        rd.long_term_corrected_level([50.0, 50.0], weights=[1.0])
    with pytest.raises(ValueError, match="positive"):
        rd.long_term_corrected_level([50.0], weights=[0.0])


def test_round_reported_level_rejects_non_finite() -> None:
    """Rounding a non-finite level is an error rather than a silent NaN."""
    with pytest.raises(ValueError, match="finite"):
        rd.round_reported_level(math.inf)
