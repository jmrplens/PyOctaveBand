#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the CNOSSOS-EU road traffic source (Annex II 2.2 + Appendix F).

Two independent kinds of oracle are used, and they are kept strictly apart.

**The equations** are pinned by the official European Commission test workbook
``CNOSSOS_ROAD_EMISSION_TEST.xlsx`` published on CIRCABC with the CNOSSOS-EU
source module, whose 4 875 cases were produced twice, once by the reference
emission DLL of DGMR and once by the independent implementation of Stapelfeldt
Ingenieurgesellschaft (see the *Testing of Emission DLL's for CNOSSOS-EU Road,
Rail and Industry Noise Sources* report, 7 July 2014). Sixty of those cases are
committed under ``tests/data/cnossos/`` together with the coefficient tables
they were computed with, which are the ones published in Commission Directive
(EU) 2015/996 and later superseded by (EU) 2021/1226. Feeding the shipped
equations that superseded coefficient set has to reproduce the workbook, and
that is what ``test_workbook_*`` asserts.

**The coefficient tables** cannot be pinned by the workbook, which predates the
2021 amendment, so they are pinned against a second, machine-made transcription
of the Official Journal text held in ``tests/reference_data/``.

Everything the workbook does not reach - the studded-tyre speed branches above
50 km/h, the gradient breakpoints, the 20 km/h floor - is anchored on the exact
closed forms of the published equations.
"""

from __future__ import annotations

import functools
import math
import warnings

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest
import reference_data as ref

from phonometry.environment.sources.cnossos_road import (
    CNOSSOS_A_WEIGHTING,
    ROAD_COEFFICIENTS,
    ROAD_OCTAVE_BANDS,
    ROAD_REFERENCE_SPEED,
    JunctionType,
    RoadEmissionCoefficients,
    RoadSurface,
    RoadSurfaceCoefficients,
    RoadTraffic,
    RoadVehicleCategory,
    line_source_segment_power,
    road_propulsion_noise,
    road_rolling_noise,
    road_source_power,
    road_surface_coefficients,
    road_vehicle_sound_power,
)

#: The workbook prints levels rounded to 0,01 dB, so that is the whole budget.
WORKBOOK_TOLERANCE = 0.01
CATEGORIES = ("1", "2", "3", "4a", "4b")


from cnossos_road_oracle import coefficients_2015 as _coefficients_2015
from cnossos_road_oracle import surfaces_2015 as _surfaces_2015


@functools.cache
def _workbook_cases() -> list[dict[str, str]]:
    return ref.cnossos_road_workbook_cases()


def _run_case(case: dict[str, str]):
    surfaces = _surfaces_2015()
    traffic = [
        RoadTraffic(
            RoadVehicleCategory(c),
            float(case[f"q_{c}"]),
            float(case[f"v_{c}"]),
            studded_fraction=0.5 if c == "1" else 0.0,
        )
        for c in CATEGORIES
    ]
    return road_source_power(
        traffic,
        surface=surfaces[case["surface"]],
        temperature=float(case["temperature_c"]),
        gradient=float(case["gradient_pct"]),
        studded_months=float(case["studded_months"]),
        junction_distance=float(case["junction_distance_m"]),
        junction_type=JunctionType(int(case["junction_type"])),
        coefficients=_coefficients_2015(),
    )


def test_workbook_data_is_complete() -> None:
    """The committed subset spans every surface, correction and category."""
    cases = _workbook_cases()
    assert len(cases) == 60
    assert len({c["surface"] for c in cases}) == 13
    assert len({(c["temperature_c"], c["studded_months"]) for c in cases}) == 4
    assert len({(c["gradient_pct"], c["junction_type"]) for c in cases}) == 5
    speeds = {float(c[f"v_{k}"]) for c in cases for k in CATEGORIES}
    assert speeds == {
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        130.0,
    }


@pytest.mark.parametrize("case", _workbook_cases(), ids=lambda c: c["case"])
def test_workbook_band_levels(case: dict[str, str]) -> None:
    """Per-band line power against the CIRCABC road emission test workbook.

    Oracle: ``CNOSSOS_ROAD_EMISSION_TEST.xlsx`` (European Commission, CIRCABC,
    CNOSSOS-EU source module test set, 2014), evaluated with the Appendix F
    tables of Commission Directive (EU) 2015/996.
    """
    result = _run_case(case)
    expected = [float(case[f"lw_{band}"]) for band in ref.CNOSSOS_ROAD_BANDS]
    assert result.total_line_power == pytest.approx(expected, abs=WORKBOOK_TOLERANCE)


@pytest.mark.parametrize("case", _workbook_cases(), ids=lambda c: c["case"])
def test_workbook_total_level(case: dict[str, str]) -> None:
    """Unweighted energy sum over the eight bands, same workbook oracle."""
    result = _run_case(case)
    total = 10.0 * np.log10(np.sum(10.0 ** (result.total_line_power / 10.0)))
    assert total == pytest.approx(float(case["lw_total"]), abs=WORKBOOK_TOLERANCE)


# ---------------------------------------------------------------------------
# The shipped Appendix F tables against the Official Journal transcription.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", CATEGORIES)
def test_table_f1_matches_official_journal(category: str) -> None:
    """Table F-1 as replaced by (EU) 2021/1226 Annex point (19)(a)."""
    expected = ref.CNOSSOS_ROAD_TABLE_F1[category]
    assert ROAD_COEFFICIENTS.rolling_a[category] == expected["AR"]
    assert ROAD_COEFFICIENTS.rolling_b[category] == expected["BR"]
    assert ROAD_COEFFICIENTS.propulsion_a[category] == expected["AP"]
    assert ROAD_COEFFICIENTS.propulsion_b[category] == expected["BP"]


def test_table_f2_matches_official_journal() -> None:
    """Table F-2 studded-tyre coefficients, unchanged since (EU) 2015/996."""
    assert ROAD_COEFFICIENTS.studded_a == ref.CNOSSOS_ROAD_TABLE_F2["ai"]
    assert ROAD_COEFFICIENTS.studded_b == ref.CNOSSOS_ROAD_TABLE_F2["bi"]


@pytest.mark.parametrize("category", CATEGORIES)
def test_table_f3_matches_official_journal(category: str) -> None:
    """Table F-3 junction coefficients, unchanged since (EU) 2015/996."""
    expected = ref.CNOSSOS_ROAD_TABLE_F3[category]
    assert ROAD_COEFFICIENTS.junction_c[category] == (expected[1], expected[2])


@pytest.mark.parametrize("surface", list(RoadSurface))
def test_table_f4_matches_official_journal(surface: RoadSurface) -> None:
    """Table F-4 as replaced by (EU) 2021/1226 Annex point (19)(b)."""
    row = road_surface_coefficients(surface)
    expected = ref.CNOSSOS_ROAD_TABLE_F4[surface.value]
    for category in CATEGORIES:
        key = category if category in expected else "4a/4b"
        assert row.alpha[category] == expected[key][0], category
        assert row.beta[category] == expected[key][1], category


def test_a_weighting_matches_official_journal() -> None:
    """The octave-band A-weighting printed by (EU) 2021/1226 point (8)(b)."""
    assert CNOSSOS_A_WEIGHTING == ref.CNOSSOS_A_WEIGHTING_TABLE


def test_octave_bands_are_the_corrected_range() -> None:
    """The corrigendum of OJ L 5, 10.1.2018 sets 2.2.1 to 63 Hz - 8 kHz."""
    assert ROAD_OCTAVE_BANDS == (
        63.0,
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
    )


@pytest.mark.parametrize("surface", list(RoadSurface))
def test_table_f4_speed_ranges(surface: RoadSurface) -> None:
    """The validity ranges printed in the two leading columns of Table F-4."""
    expected = ref.CNOSSOS_ROAD_TABLE_F4_SPEED_RANGE[surface.value]
    assert road_surface_coefficients(surface).speed_range == expected


def test_temperature_coefficients_match_the_directive() -> None:
    """``K_1 = 0,08`` and ``K_2 = K_3 = 0,04 dB/degC`` (Annex II 2.2.3).

    Category 4 has no rolling noise, so the Directive prints no ``K_m`` for it
    and the correction is identically zero; the stored zeros are asserted here
    because no computed level can reach them.
    """
    assert ROAD_COEFFICIENTS.temperature_k == ref.CNOSSOS_ROAD_TEMPERATURE_K


# ---------------------------------------------------------------------------
# Exact identity anchors: at the reference conditions every correction of 2.2
# vanishes identically, so the sound power *is* the Appendix F coefficient.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", CATEGORIES)
def test_reference_conditions_reduce_to_table_f1(category: str) -> None:
    """(2.2.4)/(2.2.11) at v_ref, reference surface, 20 degC, no junction."""
    rolling = road_rolling_noise(category, ROAD_REFERENCE_SPEED)
    propulsion = road_propulsion_noise(category, ROAD_REFERENCE_SPEED)
    assert np.array_equal(rolling, np.asarray(ROAD_COEFFICIENTS.rolling_a[category]))
    assert np.array_equal(
        propulsion, np.asarray(ROAD_COEFFICIENTS.propulsion_a[category])
    )


@pytest.mark.parametrize("category", ("1", "2", "3"))
def test_rolling_speed_law_is_exactly_logarithmic(category: str) -> None:
    """Doubling the speed adds exactly ``B_R lg 2`` to the rolling noise."""
    low = road_rolling_noise(category, 45.0)
    high = road_rolling_noise(category, 90.0)
    expected = np.asarray(ROAD_COEFFICIENTS.rolling_b[category]) * math.log10(2.0)
    assert high - low == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("category", CATEGORIES)
def test_propulsion_speed_law_is_exactly_linear(category: str) -> None:
    """(2.2.11) is affine in ``v``: equal speed steps give equal level steps."""
    step = np.asarray(ROAD_COEFFICIENTS.propulsion_b[category]) * (
        10.0 / ROAD_REFERENCE_SPEED
    )
    for v in (30.0, 60.0, 90.0, 120.0):
        delta = road_propulsion_noise(category, v + 10.0) - road_propulsion_noise(
            category, v
        )
        assert delta == pytest.approx(step, abs=1e-12)


def test_two_wheelers_have_no_rolling_noise() -> None:
    """(2.2.3): for category 4 the total power is the propulsion term alone."""
    for category in ("4a", "4b"):
        total = road_vehicle_sound_power(category, 60.0)
        assert np.array_equal(total, road_propulsion_noise(category, 60.0))
        assert np.array_equal(road_rolling_noise(category, 60.0), np.zeros(8))


def test_speed_floor_freezes_the_power_below_20_km_h() -> None:
    """2.2.1: below 20 km/h the sound power is the one at 20 km/h."""
    frozen = road_vehicle_sound_power("1", 20.0)
    for v in (5.0, 10.0, 19.999):
        assert np.array_equal(road_vehicle_sound_power("1", v), frozen)


def test_flow_term_is_exactly_the_published_logarithm() -> None:
    """(2.2.1): doubling the hourly flow adds exactly ``10 lg 2``."""
    single = road_source_power(RoadTraffic(RoadVehicleCategory.HEAVY, 500.0, 80.0))
    double = road_source_power(RoadTraffic(RoadVehicleCategory.HEAVY, 1000.0, 80.0))
    assert double.total_line_power - single.total_line_power == pytest.approx(
        10.0 * math.log10(2.0), abs=1e-12
    )
    vehicle = road_vehicle_sound_power("3", 80.0)
    assert single.total_line_power == pytest.approx(
        vehicle + 10.0 * math.log10(500.0 / (1000.0 * 80.0)), abs=1e-12
    )


def test_flow_term_uses_the_true_speed_below_the_floor() -> None:
    """The 20 km/h floor freezes the power, never the ``Q/(1000 v)`` term."""
    result = road_source_power(RoadTraffic(RoadVehicleCategory.LIGHT, 600.0, 10.0))
    expected = road_vehicle_sound_power("1", 20.0) + 10.0 * math.log10(600.0 / 10000.0)
    assert result.total_line_power == pytest.approx(expected, abs=1e-12)


def test_zero_flow_contributes_nothing() -> None:
    """An empty category leaves the line power of the rest untouched."""
    busy = RoadTraffic(RoadVehicleCategory.LIGHT, 800.0, 60.0)
    empty = RoadTraffic(RoadVehicleCategory.HEAVY, 0.0, 60.0)
    alone = road_source_power(busy)
    both = road_source_power([busy, empty])
    assert both.total_line_power == pytest.approx(alone.total_line_power, abs=1e-12)
    # An empty road is a legitimate input: no source, no warning, -inf power.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deserted = road_source_power(empty)
    assert np.all(np.isneginf(deserted.total_line_power))


# ---------------------------------------------------------------------------
# Correction terms, each against its own closed form.
# ---------------------------------------------------------------------------


def test_temperature_correction_is_linear_and_zero_at_20_degrees() -> None:
    """(2.2.10): ``K_m (tau_ref - tau)``, equal in every octave band."""
    base = road_rolling_noise("1", 50.0, temperature=20.0)
    warm = road_rolling_noise("1", 50.0, temperature=30.0)
    cold = road_rolling_noise("1", 50.0, temperature=-5.0)
    assert warm - base == pytest.approx(np.full(8, -0.8), abs=1e-12)
    assert cold - base == pytest.approx(np.full(8, 2.0), abs=1e-12)
    heavy = road_rolling_noise("3", 50.0, temperature=0.0) - road_rolling_noise(
        "3", 50.0
    )
    assert heavy == pytest.approx(np.full(8, 0.8), abs=1e-12)


def test_studded_tyres_vanish_without_a_season_or_a_fleet() -> None:
    """(2.2.7)/(2.2.8): ``p_s = 0`` leaves ``10 lg 1 = 0``."""
    base = road_rolling_noise("1", 60.0)
    assert np.array_equal(
        road_rolling_noise("1", 60.0, studded_fraction=0.0, studded_months=4.0), base
    )
    assert np.array_equal(
        road_rolling_noise("1", 60.0, studded_fraction=0.5, studded_months=0.0), base
    )


def test_studded_tyres_at_full_penetration_give_exactly_d_stud() -> None:
    """(2.2.8) with ``p_s = 1`` collapses to ``D_stud,i(v)`` of (2.2.6)."""
    v = 70.0
    d_stud = np.asarray(ROAD_COEFFICIENTS.studded_a) + np.asarray(
        ROAD_COEFFICIENTS.studded_b
    ) * math.log10(v / ROAD_REFERENCE_SPEED)
    delta = road_rolling_noise(
        "1", v, studded_fraction=1.0, studded_months=12.0
    ) - road_rolling_noise("1", v)
    assert delta == pytest.approx(d_stud, abs=1e-12)


def test_studded_tyre_speed_branches_saturate() -> None:
    """(2.2.6) freezes below 50 km/h and above 90 km/h."""
    kwargs = {"studded_fraction": 1.0, "studded_months": 12.0}
    at_50 = road_rolling_noise("1", 50.0, **kwargs) - road_rolling_noise("1", 50.0)
    at_20 = road_rolling_noise("1", 20.0, **kwargs) - road_rolling_noise("1", 20.0)
    at_90 = road_rolling_noise("1", 90.0, **kwargs) - road_rolling_noise("1", 90.0)
    at_130 = road_rolling_noise("1", 130.0, **kwargs) - road_rolling_noise("1", 130.0)
    assert at_20 == pytest.approx(at_50, abs=1e-12)
    assert at_130 == pytest.approx(at_90, abs=1e-12)
    b = np.asarray(ROAD_COEFFICIENTS.studded_b)
    assert at_50 == pytest.approx(
        np.asarray(ROAD_COEFFICIENTS.studded_a) + b * math.log10(50.0 / 70.0), abs=1e-12
    )
    assert at_90 == pytest.approx(
        np.asarray(ROAD_COEFFICIENTS.studded_a) + b * math.log10(90.0 / 70.0), abs=1e-12
    )


def test_studded_tyres_are_a_light_vehicle_effect_only() -> None:
    """(2.2.9): every category other than 1 takes zero."""
    for category in ("2", "3", "4a", "4b"):
        base = road_rolling_noise(category, 60.0)
        studded = road_rolling_noise(
            category, 60.0, studded_fraction=1.0, studded_months=12.0
        )
        assert np.array_equal(studded, base)


def test_junction_correction_breakpoints() -> None:
    """(2.2.17)/(2.2.18): full ``C`` at the junction, zero from 100 m out."""
    for category in ("1", "2", "3"):
        c_r, c_p = ROAD_COEFFICIENTS.junction_c[category][0]
        at_zero = road_rolling_noise(
            category, 50.0, junction_distance=0.0, junction_type=JunctionType.CROSSING
        ) - road_rolling_noise(category, 50.0)
        assert at_zero == pytest.approx(np.full(8, c_r), abs=1e-12)
        prop_zero = road_propulsion_noise(
            category, 50.0, junction_distance=0.0, junction_type=JunctionType.CROSSING
        ) - road_propulsion_noise(category, 50.0)
        assert prop_zero == pytest.approx(np.full(8, c_p), abs=1e-12)
        half = road_rolling_noise(
            category,
            50.0,
            junction_distance=-50.0,
            junction_type=JunctionType.ROUNDABOUT,
        ) - road_rolling_noise(category, 50.0)
        assert half == pytest.approx(
            np.full(8, ROAD_COEFFICIENTS.junction_c[category][1][0] * 0.5), abs=1e-12
        )
        for distance in (100.0, 150.0):
            far = road_propulsion_noise(
                category,
                50.0,
                junction_distance=distance,
                junction_type=JunctionType.CROSSING,
            )
            assert np.array_equal(far, road_propulsion_noise(category, 50.0))


@pytest.mark.parametrize(
    ("category", "flat_low", "flat_high"),
    [("1", -6.0, 2.0), ("2", -4.0, 0.0), ("3", -4.0, 0.0)],
)
def test_gradient_correction_is_zero_inside_the_flat_band(
    category: str, flat_low: float, flat_high: float
) -> None:
    """(2.2.13)-(2.2.15): no correction between the two breakpoints."""
    base = road_propulsion_noise(category, 70.0)
    for slope in (flat_low, 0.5 * (flat_low + flat_high), flat_high):
        assert np.array_equal(
            road_propulsion_noise(category, 70.0, gradient=slope), base
        )


def test_gradient_correction_downhill_branches() -> None:
    """The published downhill branches, including the missing speed factor.

    Category 1 has no speed factor downhill, categories 2 and 3 do, and their
    speed offsets differ (20 km/h and 10 km/h).
    """
    v = 70.0
    base_1 = road_propulsion_noise("1", v)
    assert road_propulsion_noise("1", v, gradient=-10.0) - base_1 == pytest.approx(
        np.full(8, (10.0 - 6.0) / 1.0), abs=1e-12
    )
    base_2 = road_propulsion_noise("2", v)
    assert road_propulsion_noise("2", v, gradient=-10.0) - base_2 == pytest.approx(
        np.full(8, (10.0 - 4.0) / 0.7 * (v - 20.0) / 100.0), abs=1e-12
    )
    base_3 = road_propulsion_noise("3", v)
    assert road_propulsion_noise("3", v, gradient=-10.0) - base_3 == pytest.approx(
        np.full(8, (10.0 - 4.0) / 0.5 * (v - 10.0) / 100.0), abs=1e-12
    )


def test_gradient_correction_uphill_branches() -> None:
    """The published uphill branches, all three with a speed factor."""
    v = 80.0
    assert road_propulsion_noise("1", v, gradient=6.0) - road_propulsion_noise(
        "1", v
    ) == pytest.approx(np.full(8, (6.0 - 2.0) / 1.5 * v / 100.0), abs=1e-12)
    assert road_propulsion_noise("2", v, gradient=6.0) - road_propulsion_noise(
        "2", v
    ) == pytest.approx(np.full(8, 6.0 / 1.0 * v / 100.0), abs=1e-12)
    assert road_propulsion_noise("3", v, gradient=6.0) - road_propulsion_noise(
        "3", v
    ) == pytest.approx(np.full(8, 6.0 / 0.8 * v / 100.0), abs=1e-12)


@pytest.mark.parametrize(
    ("category", "inside", "just_outside"),
    [
        ("1", (-5.5, -0.5, 1.0), (-6.5, 2.5)),
        ("2", (-3.5, -0.5), (-4.5, 0.5)),
        ("3", (-3.5, -0.5), (-4.5, 0.5)),
    ],
)
def test_gradient_dead_band_is_pinned_on_both_sides_of_every_breakpoint(
    category: str, inside: tuple[float, ...], just_outside: tuple[float, ...]
) -> None:
    """(2.2.13)-(2.2.15): the dead band ends exactly at the printed breakpoints.

    The correction is continuous, so it evaluates to zero *at* every breakpoint
    and a test that samples only the edges cannot tell a widened branch from
    the printed one: at s = 2 % the category 1 uphill branch returns
    (2 - 2)/1,5 = 0 whether it opens at 0 % or at 2 %. These gradients sit
    strictly inside the band, where a breakpoint moved outwards makes the
    correction negative, and strictly outside it, where one moved inwards
    would leave it at zero.
    """
    base = road_propulsion_noise(category, 70.0)
    for slope in inside:
        assert np.array_equal(
            road_propulsion_noise(category, 70.0, gradient=slope), base
        )
    for slope in just_outside:
        assert not np.array_equal(
            road_propulsion_noise(category, 70.0, gradient=slope), base
        )


def test_gradient_correction_never_subtracts() -> None:
    """The published correction only ever adds propulsion noise.

    Every branch of (2.2.13)-(2.2.15) is non-negative by construction, so a
    misplaced breakpoint shows up as a negative correction on the gentle
    slopes that make up most of a real network. Swept rather than sampled,
    because that is the property the breakpoints have to preserve.
    """
    for category in ("1", "2", "3", "4a", "4b"):
        base = road_propulsion_noise(category, 70.0)
        for slope in np.arange(-15.0, 15.001, 0.25):
            assert np.all(
                road_propulsion_noise(category, 70.0, gradient=float(slope))
                >= base - 1e-12
            )


def test_gradient_saturates_at_twelve_per_cent() -> None:
    """``Min(12 %, s)`` caps the correction in both directions."""
    for category in ("1", "2", "3"):
        for sign in (1.0, -1.0):
            capped = road_propulsion_noise(category, 70.0, gradient=sign * 12.0)
            for slope in (sign * 20.0, sign * 100.0):
                assert np.array_equal(
                    road_propulsion_noise(category, 70.0, gradient=slope), capped
                )


def test_gradient_does_not_touch_powered_two_wheelers() -> None:
    """(2.2.16): the correction is identically zero for category 4."""
    for category in ("4a", "4b"):
        base = road_propulsion_noise(category, 70.0)
        for slope in (-12.0, -5.0, 5.0, 12.0):
            assert np.array_equal(
                road_propulsion_noise(category, 70.0, gradient=slope), base
            )


def test_reference_surface_is_transparent() -> None:
    """Table F-4's first row is all zeros, so it changes nothing."""
    for category in CATEGORIES:
        assert np.array_equal(
            road_rolling_noise(category, 90.0, surface=RoadSurface.REFERENCE),
            road_rolling_noise(category, 90.0),
        )


def test_surface_effect_on_rolling_and_propulsion_differ() -> None:
    """(2.2.19) takes ``alpha + beta lg(v/v_ref)``, (2.2.20) ``min(alpha, 0)``."""
    surface = RoadSurface.HARD_ELEMENTS_NOT_HERRINGBONE
    row = road_surface_coefficients(surface)
    v = 40.0
    rolling = road_rolling_noise("1", v, surface=surface) - road_rolling_noise("1", v)
    expected = np.asarray(row.alpha["1"]) + row.beta["1"] * math.log10(
        v / ROAD_REFERENCE_SPEED
    )
    assert rolling == pytest.approx(expected, abs=1e-12)
    propulsion = road_propulsion_noise("1", v, surface=surface) - road_propulsion_noise(
        "1", v
    )
    assert propulsion == pytest.approx(
        np.minimum(np.asarray(row.alpha["1"]), 0.0), abs=1e-12
    )
    # Every alpha of this loud surface is positive, so propulsion is untouched.
    assert np.array_equal(propulsion, np.zeros(8))


def test_absorbing_surface_reduces_propulsion_noise() -> None:
    """A negative ``alpha`` passes straight through (2.2.20)."""
    surface = RoadSurface.TWO_LAYER_ZOAB_FINE
    row = road_surface_coefficients(surface)
    delta = road_propulsion_noise("1", 100.0, surface=surface) - road_propulsion_noise(
        "1", 100.0
    )
    assert delta == pytest.approx(
        np.minimum(np.asarray(row.alpha["1"]), 0.0), abs=1e-12
    )
    assert float(np.min(delta)) < 0.0


def test_surface_accepts_a_description_string() -> None:
    by_name = road_rolling_noise("1", 60.0, surface="thin layer A")
    by_enum = road_rolling_noise("1", 60.0, surface=RoadSurface.THIN_LAYER_A)
    assert np.array_equal(by_name, by_enum)


# ---------------------------------------------------------------------------
# Assembly, A-weighting and the hand-off to the propagation stage.
# ---------------------------------------------------------------------------


def test_categories_are_summed_energetically() -> None:
    """(2.3.1)-style energy sum of the per-category line powers."""
    flows = [
        RoadTraffic(RoadVehicleCategory.LIGHT, 1200.0, 90.0),
        RoadTraffic(RoadVehicleCategory.HEAVY, 150.0, 80.0),
    ]
    result = road_source_power(flows)
    expected = 10.0 * np.log10(np.sum(10.0 ** (result.line_power / 10.0), axis=0))
    assert result.total_line_power == pytest.approx(expected, abs=1e-12)
    assert result.line_power.shape == (2, 8)
    assert result.categories == (RoadVehicleCategory.LIGHT, RoadVehicleCategory.HEAVY)


def test_result_arrays_are_consistent() -> None:
    flows = [RoadTraffic(RoadVehicleCategory.MEDIUM_HEAVY, 400.0, 70.0)]
    result = road_source_power(flows)
    assert result.frequencies.tolist() == list(ROAD_OCTAVE_BANDS)
    assert result.source_height == 0.05
    assert result.vehicle_power[0] == pytest.approx(
        10.0
        * np.log10(
            10.0 ** (result.rolling[0] / 10.0) + 10.0 ** (result.propulsion[0] / 10.0)
        ),
        abs=1e-12,
    )


def test_a_weighted_line_power_uses_the_directive_table() -> None:
    result = road_source_power(RoadTraffic(RoadVehicleCategory.LIGHT, 1000.0, 70.0))
    weighted = result.total_line_power + np.asarray(CNOSSOS_A_WEIGHTING)
    assert result.a_weighted_line_power == pytest.approx(
        float(10.0 * np.log10(np.sum(10.0 ** (weighted / 10.0)))), abs=1e-12
    )


def test_segment_power_is_the_line_power_plus_ten_log_length() -> None:
    line = np.array([90.0, 92.0, 94.0])
    assert line_source_segment_power(line, 1.0) == pytest.approx(line, abs=1e-12)
    assert line_source_segment_power(line, 10.0) == pytest.approx(
        line + 10.0, abs=1e-12
    )
    assert line_source_segment_power(90.0, 25.0) == pytest.approx(
        np.array([90.0 + 10.0 * math.log10(25.0)]), abs=1e-12
    )


def test_segment_power_feeds_the_propagation_stage() -> None:
    """The emission output is an octave-band ``L_W`` on the 63 Hz - 8 kHz grid."""
    from phonometry.environment.propagation.outdoor_propagation import (
        PropagationGeometry,
        predicted_receiver_level,
    )

    result = road_source_power(
        [
            RoadTraffic(RoadVehicleCategory.LIGHT, 1000.0, 90.0),
            RoadTraffic(RoadVehicleCategory.HEAVY, 120.0, 80.0),
        ],
        surface=RoadSurface.THIN_LAYER_A,
    )
    segment = line_source_segment_power(result.total_line_power, 20.0)
    levels = predicted_receiver_level(
        segment,
        PropagationGeometry(100.0, result.source_height, 4.0),
        frequencies=result.frequencies,
    )
    assert levels.shape == (8,)
    assert np.all(levels < segment)


# ---------------------------------------------------------------------------
# Validation and plotting.
# ---------------------------------------------------------------------------


def test_unknown_category_and_surface_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown vehicle category"):
        road_rolling_noise("7", 50.0)
    with pytest.raises(ValueError, match="Unknown road surface"):
        road_surface_coefficients("gravel")


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive number of km/h"):
        road_rolling_noise("1", 0.0)
    with pytest.raises(ValueError, match="finite"):
        road_rolling_noise("1", 50.0, temperature=float("nan"))
    with pytest.raises(ValueError, match="at least one vehicle category"):
        road_source_power([])
    repeated = [
        RoadTraffic(RoadVehicleCategory.LIGHT, 100.0, 50.0),
        RoadTraffic(RoadVehicleCategory.LIGHT, 200.0, 60.0),
    ]
    with pytest.raises(ValueError, match="at most one flow per vehicle category"):
        road_source_power(repeated)
    negative = RoadTraffic(RoadVehicleCategory.LIGHT, -1.0, 50.0)
    with pytest.raises(ValueError, match="non-negative"):
        road_source_power(negative)
    power = np.array([90.0])
    with pytest.raises(ValueError, match="positive number of metres"):
        line_source_segment_power(power, 0.0)


def test_substituted_database_may_replace_one_row() -> None:
    """A national Appendix F row overrides only the coefficients it carries."""
    import dataclasses

    row = (83.4, 89.0, 88.1, 93.6, 100.4, 96.9, 87.0, 76.5)
    national = dataclasses.replace(
        ROAD_COEFFICIENTS,
        rolling_a={**ROAD_COEFFICIENTS.rolling_a, "1": row},
    )
    assert road_rolling_noise("1", 70.0, coefficients=national) == pytest.approx(
        np.asarray(row), abs=1e-12
    )
    assert road_rolling_noise("3", 70.0, coefficients=national) == pytest.approx(
        np.asarray(ROAD_COEFFICIENTS.rolling_a["3"]), abs=1e-12
    )


def test_incomplete_database_is_rejected_as_an_invalid_input() -> None:
    """A database missing the requested category fails as a ValueError.

    ``_category_key`` only checks Table [2.2.a]; the coefficients actually in
    use may cover fewer categories, and that has to surface as an invalid input
    rather than as a bare lookup error deeper in the calculation.
    """
    partial = RoadEmissionCoefficients(
        rolling_a={"1": ROAD_COEFFICIENTS.rolling_a["1"]},
        rolling_b={"1": ROAD_COEFFICIENTS.rolling_b["1"]},
        propulsion_a={"1": ROAD_COEFFICIENTS.propulsion_a["1"]},
        propulsion_b={"1": ROAD_COEFFICIENTS.propulsion_b["1"]},
        studded_a=ROAD_COEFFICIENTS.studded_a,
        studded_b=ROAD_COEFFICIENTS.studded_b,
        junction_c={"1": ROAD_COEFFICIENTS.junction_c["1"]},
        temperature_k={"1": ROAD_COEFFICIENTS.temperature_k["1"]},
    )
    assert road_rolling_noise("1", 70.0, coefficients=partial).shape == (8,)
    for call in (road_rolling_noise, road_propulsion_noise):
        with pytest.raises(ValueError, match="no category '3'"):
            call("3", 70.0, coefficients=partial)
    with pytest.raises(ValueError, match="no category '3'"):
        road_vehicle_sound_power("3", 70.0, coefficients=partial)
    heavy = RoadTraffic(RoadVehicleCategory.HEAVY, 100.0, 70.0)
    with pytest.raises(ValueError, match="no category '3'"):
        road_source_power(heavy, coefficients=partial)


def test_incomplete_surface_row_is_rejected_as_an_invalid_input() -> None:
    """A Table F-4 row missing the requested category fails as a ValueError."""
    published = road_surface_coefficients(RoadSurface.THIN_LAYER_A)
    partial = RoadSurfaceCoefficients(
        name="light vehicles only",
        alpha={"1": published.alpha["1"]},
        beta={"1": published.beta["1"]},
        speed_range=published.speed_range,
    )
    assert road_rolling_noise("1", 70.0, surface=partial).shape == (8,)
    with pytest.raises(ValueError, match="no coefficients for category '2'"):
        road_rolling_noise("2", 70.0, surface=partial)
    with pytest.raises(ValueError, match="no coefficients for category '2'"):
        road_propulsion_noise("2", 70.0, surface=partial)


def test_studded_tyre_inputs_are_range_checked() -> None:
    """``Q_stud,ratio`` is a fraction and ``T_s`` a share of a 12-month year.

    Outside those ranges ``p_s`` of (2.2.7) leaves [0, 1] and the logarithm of
    (2.2.8) takes a negative argument, which would silently return NaN.
    """
    for fraction in (-0.1, 1.5):
        with pytest.raises(ValueError, match="fraction in "):
            road_rolling_noise("1", 60.0, studded_fraction=fraction, studded_months=3.0)
    for months in (-1.0, 13.0):
        with pytest.raises(ValueError, match="months in "):
            road_rolling_noise("1", 60.0, studded_fraction=0.5, studded_months=months)
    # The check runs for every category, not only the one it can affect.
    with pytest.raises(ValueError, match="fraction in "):
        road_rolling_noise("3", 60.0, studded_fraction=2.0)
    # The two extremes themselves are valid.
    assert road_rolling_noise(
        "1", 60.0, studded_fraction=1.0, studded_months=12.0
    ).shape == (8,)


def test_plot_draws_the_total_and_every_category() -> None:
    import matplotlib.pyplot as plt

    result = road_source_power(
        [
            RoadTraffic(RoadVehicleCategory.LIGHT, 1000.0, 90.0),
            RoadTraffic(RoadVehicleCategory.HEAVY, 200.0, 80.0),
            RoadTraffic(RoadVehicleCategory.MOTORCYCLES, 50.0, 70.0),
        ]
    )
    ax = result.plot()
    assert len(ax.lines) == 3
    assert len(ax.containers) == 1
    assert ax.get_ylabel()
    ax_es = result.plot(language="es")
    assert "CNOSSOS" in ax_es.get_title()
    plt.close("all")
