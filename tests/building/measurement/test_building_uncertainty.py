#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 12999-1:2020 measurement uncertainty (building_uncertainty).

The tabulated standard uncertainties are the oracle: every band value and every
single-number value is asserted digit-for-digit against ISO 12999-1:2020(E)
Tables 1-8 and Annex D. Combination examples reproduce Annexes A and B.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry.building.measurement.uncertainty import (
    UncertainValue,
    band_uncertainty,
    combine_uncertainties,
    insulation_coverage_factor,
    insulation_expanded_uncertainty,
    maximum_repeatability_standard_deviation,
    prediction_input_uncertainty,
    reduce_by_independent_measurements,
    satisfies_lower_requirement,
    satisfies_upper_requirement,
    single_number_uncertainty,
    single_number_uncertainty_uncorrelated,
    uncertain_value,
)

# Frequency axes from the standard.
FREQ_FULL = [
    50,
    63,
    80,
    100,
    125,
    160,
    200,
    250,
    315,
    400,
    500,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
    4000,
    5000,
]
FREQ_IMPACT = [
    50,
    63,
    80,
    100,
    125,
    160,
    200,
    250,
    315,
    400,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
    4000,
    5000,
]


# --------------------------------------------------------------------------- #
# Table 2 — airborne one-third-octave (Clause 7.2), every band digit-exact.
# --------------------------------------------------------------------------- #
TABLE2 = {
    50: (6.8, 4.0, 2.0),
    63: (4.6, 3.6, 1.8),
    80: (3.8, 3.2, 1.6),
    100: (3.0, 2.8, 1.4),
    125: (2.7, 2.4, 1.2),
    160: (2.4, 2.0, 1.0),
    200: (2.1, 1.8, 0.9),
    250: (1.8, 1.6, 0.8),
    315: (1.8, 1.4, 0.7),
    400: (1.8, 1.2, 0.6),
    500: (1.8, 1.1, 0.6),
    630: (1.8, 1.0, 0.6),
    800: (1.8, 1.0, 0.6),
    1000: (1.8, 1.0, 0.6),
    1250: (1.8, 1.0, 0.6),
    1600: (1.8, 1.0, 0.6),
    2000: (1.8, 1.0, 0.6),
    2500: (1.9, 1.3, 0.6),
    3150: (2.0, 1.6, 0.6),
    4000: (2.4, 1.9, 0.6),
    5000: (2.8, 2.2, 0.6),
}


@pytest.mark.parametrize(("situation", "col"), [("A", 0), ("B", 1), ("C", 2)])
def test_table2_airborne_every_band(situation, col):
    result = band_uncertainty("airborne", situation)
    assert list(result.frequencies) == FREQ_FULL
    for f, u in zip(result.frequencies, result.uncertainties, strict=True):
        assert u == pytest.approx(TABLE2[int(f)][col]), f"{situation} @ {f} Hz"


# --------------------------------------------------------------------------- #
# Table 4 — impact one-third-octave (Clause 7.3). No 500 Hz in the 2020 edition.
# --------------------------------------------------------------------------- #
TABLE4 = {
    50: (3.2, 1.5),
    63: (2.8, 1.4),
    80: (2.4, 1.3),
    100: (2.0, 1.2),
    125: (1.6, 1.1),
    160: (1.4, 1.0),
    200: (1.3, 0.9),
    250: (1.2, 0.8),
    315: (1.2, 0.8),
    400: (1.2, 0.8),
    630: (1.2, 0.8),
    800: (1.2, 0.8),
    1000: (1.2, 0.8),
    1250: (1.3, 0.8),
    1600: (1.4, 0.8),
    2000: (1.5, 0.8),
    2500: (1.7, 1.0),
    3150: (1.9, 1.2),
    4000: (2.1, 1.4),
    5000: (2.3, 1.6),
}


@pytest.mark.parametrize(("situation", "col"), [("B", 0), ("C", 1)])
def test_table4_impact_every_band(situation, col):
    result = band_uncertainty("impact", situation)
    assert list(result.frequencies) == FREQ_IMPACT
    assert 500 not in result.frequencies  # 2020 edition drops 500 Hz
    for f, u in zip(result.frequencies, result.uncertainties, strict=True):
        assert u == pytest.approx(TABLE4[int(f)][col]), f"{situation} @ {f} Hz"


def test_impact_has_no_situation_a():
    with pytest.raises(ValueError, match="not tabulated"):
        band_uncertainty("impact", "A")


# --------------------------------------------------------------------------- #
# Table 6 — reduction ΔL one-third-octave (Clause 7.4), situation A only.
# --------------------------------------------------------------------------- #
TABLE6_A = [
    1.4,
    1.3,
    1.2,
    1.1,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.1,
    1.2,
    1.3,
    1.6,
    1.9,
    2.2,
    2.5,
    2.8,
    3.2,
    3.6,
    4.0,
    4.4,
]


def test_table6_reduction_every_band():
    result = band_uncertainty("impact_reduction", "A")
    assert list(result.frequencies) == FREQ_FULL
    for u, expected in zip(result.uncertainties, TABLE6_A, strict=True):
        assert u == pytest.approx(expected)


@pytest.mark.parametrize("situation", ["B", "C"])
def test_reduction_only_situation_a(situation):
    with pytest.raises(ValueError, match="not tabulated"):
        band_uncertainty("impact_reduction", situation)


# --------------------------------------------------------------------------- #
# Annex D Table D.1 — σR95 airborne (situation A upper limit), digit-exact.
# --------------------------------------------------------------------------- #
TABLED1 = [
    11.7,
    6.7,
    5.9,
    5.0,
    5.0,
    3.8,
    3.3,
    3.3,
    3.3,
    3.3,
    3.3,
    3.3,
    3.3,
    3.3,
    3.4,
    3.4,
    3.4,
    3.5,
    3.6,
    4.0,
    4.7,
]


def test_tabled1_sigma_r95_bands():
    result = band_uncertainty("airborne", "A", upper_limit=True)
    assert result.upper_limit is True
    assert list(result.frequencies) == FREQ_FULL
    for u, expected in zip(result.uncertainties, TABLED1, strict=True):
        assert u == pytest.approx(expected)


def test_sigma_r95_only_airborne():
    with pytest.raises(ValueError, match="σR95"):
        band_uncertainty("impact", "B", upper_limit=True)


# --------------------------------------------------------------------------- #
# Table 1 — maximum repeatability standard deviation (Clause 5.8).
# --------------------------------------------------------------------------- #
def test_table1_maximum_repeatability():
    result = maximum_repeatability_standard_deviation()
    expected = [
        4.0,
        3.5,
        3.0,
        2.6,
        2.2,
        1.9,
        1.7,
        1.5,
        1.4,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
        1.3,
    ]
    assert list(result.frequencies) == FREQ_FULL
    for u, e in zip(result.uncertainties, expected, strict=True):
        assert u == pytest.approx(e)


# --------------------------------------------------------------------------- #
# Table 3 — airborne single-number (ISO 717-1), every row / situation.
# --------------------------------------------------------------------------- #
TABLE3 = {
    "r_w": (1.2, 0.9, 0.4),
    "r_w+c_100_3150": (1.3, 0.9, 0.5),
    "r_w+c_100_5000": (1.3, 1.1, 0.5),
    "r_w+c_50_3150": (1.3, 1.0, 0.7),
    "r_w+c_50_5000": (1.3, 1.1, 0.7),
    "r_w+ctr_100_3150": (1.5, 1.1, 0.7),
    "r_w+ctr_100_5000": (1.5, 1.1, 0.7),
    "r_w+ctr_50_3150": (1.5, 1.3, 1.0),
    "r_w+ctr_50_5000": (1.5, 1.0, 1.0),
}


@pytest.mark.parametrize(("quantity", "values"), list(TABLE3.items()))
@pytest.mark.parametrize(("situation", "idx"), [("A", 0), ("B", 1), ("C", 2)])
def test_table3_single_number(quantity, values, situation, idx):
    assert single_number_uncertainty(quantity, situation) == pytest.approx(values[idx])


def test_airborne_aliases_share_row():
    for alias in ("R_w", "Rprime_w", "Dn_w", "DnT_w"):
        assert single_number_uncertainty(alias, "A") == pytest.approx(1.2)
        assert single_number_uncertainty(alias, "B") == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Table 5 — impact single-number (ISO 717-2).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("quantity", "expected"),
    [("ln_w", (1.5, 1.0, 0.5)), ("ln_w+ci", (1.5, 1.0, 0.6))],
)
def test_table5_impact_single_number(quantity, expected):
    for situation, e in zip("ABC", expected, strict=True):
        assert single_number_uncertainty(quantity, situation) == pytest.approx(e)


def test_impact_single_number_aliases():
    for alias in ("Lprime_n_w", "LnT_w"):
        assert single_number_uncertainty(alias, "C") == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Table 7 — reduction single-number ΔLw (situation A only).
# --------------------------------------------------------------------------- #
def test_table7_delta_lw():
    assert single_number_uncertainty("delta_lw", "A") == pytest.approx(1.1)


@pytest.mark.parametrize("situation", ["B", "C"])
def test_delta_lw_only_situation_a(situation):
    with pytest.raises(ValueError, match="not tabulated"):
        single_number_uncertainty("delta_lw", situation)


# --------------------------------------------------------------------------- #
# Annex D Table D.2 — σR95 single-number (situation A).
# --------------------------------------------------------------------------- #
TABLED2 = {
    "r_w": 2.0,
    "r_w+c_100_3150": 2.1,
    "r_w+c_100_5000": 2.1,
    "r_w+c_50_3150": 2.1,
    "r_w+c_50_5000": 2.1,
    "r_w+ctr_100_3150": 2.4,
    "r_w+ctr_100_5000": 2.4,
    "r_w+ctr_50_3150": 2.4,
    "r_w+ctr_50_5000": 2.4,
}


@pytest.mark.parametrize(("quantity", "expected"), list(TABLED2.items()))
def test_tabled2_sigma_r95_single_number(quantity, expected):
    assert single_number_uncertainty(quantity, "A", upper_limit=True) == pytest.approx(
        expected
    )


def test_sigma_r95_single_number_requires_situation_a():
    with pytest.raises(ValueError, match="situation A only"):
        single_number_uncertainty("r_w", "B", upper_limit=True)


def test_sigma_r95_single_number_impact_absent():
    with pytest.raises(ValueError, match="No σR95"):
        single_number_uncertainty("ln_w", "A", upper_limit=True)


# --------------------------------------------------------------------------- #
# Table 8 — coverage factors (Clause 8), every row.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("confidence", "k"),
    [
        (0.68, 1.00),
        (0.80, 1.28),
        (0.90, 1.65),
        (0.95, 1.96),
        (0.99, 2.58),
        (0.999, 3.29),
    ],
)
def test_table8_two_sided(confidence, k):
    assert insulation_coverage_factor(confidence, one_sided=False) == pytest.approx(k)


@pytest.mark.parametrize(
    ("confidence", "k"),
    [
        (0.84, 1.00),
        (0.90, 1.28),
        (0.95, 1.65),
        (0.975, 1.96),
        (0.995, 2.58),
        (0.9995, 3.29),
    ],
)
def test_table8_one_sided(confidence, k):
    assert insulation_coverage_factor(confidence, one_sided=True) == pytest.approx(k)


def test_coverage_factor_unknown_confidence():
    with pytest.raises(ValueError, match="not tabulated"):
        insulation_coverage_factor(0.925)


def test_coverage_factors_table_is_public():
    import phonometry
    from phonometry.building.measurement.uncertainty import COVERAGE_FACTORS

    assert phonometry.building.COVERAGE_FACTORS is COVERAGE_FACTORS
    # Keyed by (confidence, one_sided); matches the functional lookup.
    assert COVERAGE_FACTORS[(0.95, False)] == pytest.approx(1.96)
    assert COVERAGE_FACTORS[(0.95, True)] == pytest.approx(1.65)


def test_coverage_factors_table_is_read_only():
    from phonometry.building.measurement.uncertainty import COVERAGE_FACTORS

    with pytest.raises(TypeError):
        COVERAGE_FACTORS[(0.95, False)] = 9.9  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Expansion U = k·u (Formula 2) and the k >= 1 minimum.
# --------------------------------------------------------------------------- #
def test_expanded_uncertainty_two_sided():
    # Rw situation A, u = 1.2 dB, 95 % two-sided -> k = 1.96.
    assert insulation_expanded_uncertainty(1.2, coverage=0.95) == pytest.approx(
        1.96 * 1.2
    )


def test_expanded_uncertainty_one_sided():
    # Conformity check at 95 % one-sided -> k = 1.65.
    assert insulation_expanded_uncertainty(
        1.2, coverage=0.95, one_sided=True
    ) == pytest.approx(1.65 * 1.2)


def test_coverage_minimum_k_is_one():
    # 68 % two-sided is exactly k = 1; U == u.
    assert insulation_expanded_uncertainty(0.9, coverage=0.68) == pytest.approx(0.9)


def test_expanded_uncertainty_rejects_negative():
    with pytest.raises(ValueError, match="Standard uncertainty u must be non-negative"):
        insulation_expanded_uncertainty(-0.1)


# --------------------------------------------------------------------------- #
# UncertainValue convenience (value ± U) — the reporting form Y = y ± U.
# --------------------------------------------------------------------------- #
def test_uncertain_value_two_sided_interval():
    # Standard's example: R = (35.1 ± 1.2) dB at k = 1 (two-sided 68 %).
    uv = uncertain_value(35.1, "r_w", "A", coverage=0.68)
    assert isinstance(uv, UncertainValue)
    assert uv.standard_uncertainty == pytest.approx(1.2)
    assert uv.coverage_factor == pytest.approx(1.0)
    assert uv.expanded_uncertainty == pytest.approx(1.2)
    assert uv.lower == pytest.approx(33.9)
    assert uv.upper == pytest.approx(36.3)


def test_uncertain_value_one_sided_for_conformity():
    # Annex A.3: in-situ R'w, u = 0.9 dB, 84 % one-sided -> k = 1 -> U = 0.9.
    uv = uncertain_value(52.0, "rprime_w", "B", coverage=0.84, one_sided=True)
    assert uv.standard_uncertainty == pytest.approx(0.9)
    assert uv.expanded_uncertainty == pytest.approx(0.9)
    assert uv.one_sided is True


# --------------------------------------------------------------------------- #
# Combination rules — Annexes A/B/C with hand-computed oracles.
# --------------------------------------------------------------------------- #
def test_combine_uncertainties_quadrature():
    assert combine_uncertainties(3.0, 4.0) == pytest.approx(5.0)


def test_prediction_input_uncertainty_annex_a_example():
    # Annex A: sigma_R = 1.2, sigma_product = 1.0, n = 1 -> u_input = sqrt(3.44) ~ 1.9.
    u_input = prediction_input_uncertainty(1.2, 1.0, 1)
    assert u_input == pytest.approx(math.sqrt(3.44))
    assert round(u_input, 1) == 1.9


def test_predicted_uncertainty_annex_a_example():
    # Annex A: u_calc = u_input (single element), u_reality = 0.8 -> u_pred ~ 2.0.
    u_input = prediction_input_uncertainty(1.2, 1.0, 1)
    u_pred = combine_uncertainties(u_input, 0.8)
    assert u_pred == pytest.approx(math.sqrt(4.08))
    assert round(u_pred, 1) == 2.0


def test_reduce_by_independent_measurements():
    # Formula A.7: u = 0.9 / sqrt(m).
    assert reduce_by_independent_measurements(0.9, 4) == pytest.approx(0.45)
    assert reduce_by_independent_measurements(0.9, 1) == pytest.approx(0.9)


def test_single_number_uncorrelated_equal_weights():
    # Two bands with equal (L_i - R_i) => equal weights 0.5; u_i = 2.0 each.
    # u = sqrt((0.5*2)^2 + (0.5*2)^2) = sqrt(2).
    u = single_number_uncertainty_uncorrelated([2.0, 2.0], [0.0, 0.0])
    assert u == pytest.approx(math.sqrt(2.0))


def test_single_number_uncorrelated_dominant_band():
    # One band dominates the reference energy (much smaller L-R gap) -> its weight
    # -> 1, so the combined uncertainty approaches that band's u.
    u = single_number_uncertainty_uncorrelated([1.0, 5.0], [0.0, -100.0])
    assert u == pytest.approx(1.0, abs=1e-6)


def test_single_number_uncorrelated_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        single_number_uncertainty_uncorrelated([1.0, 2.0], [0.0])


# --------------------------------------------------------------------------- #
# Conformity with a requirement (Formulae 4/5).
# --------------------------------------------------------------------------- #
def test_satisfies_lower_requirement():
    # R'w = 54, U = 1.5 -> 52.5 > 52 required: pass.
    assert satisfies_lower_requirement(54.0, 1.5, 52.0) is True
    assert satisfies_lower_requirement(53.0, 1.5, 52.0) is False


def test_satisfies_upper_requirement():
    # L'n,w = 50, U = 1.0 -> 51 < 53 required: pass.
    assert satisfies_upper_requirement(50.0, 1.0, 53.0) is True
    assert satisfies_upper_requirement(52.5, 1.0, 53.0) is False


# --------------------------------------------------------------------------- #
# Validation of unknown quantities / situations.
# --------------------------------------------------------------------------- #
def test_unknown_quantity():
    with pytest.raises(ValueError, match="Unknown single-number quantity"):
        single_number_uncertainty("nonsense", "A")


def test_unknown_measurand():
    with pytest.raises(ValueError, match="Unknown measurand"):
        band_uncertainty("magic", "A")  # type: ignore[arg-type]


def test_unknown_situation_single_number():
    with pytest.raises(ValueError, match="Unknown situation"):
        single_number_uncertainty("r_w", "Z")  # type: ignore[arg-type]


def test_band_uncertainty_to_arrays_roundtrip():
    result = band_uncertainty("airborne", "A")
    freqs, u = result.to_arrays()
    assert isinstance(freqs, np.ndarray)
    assert isinstance(u, np.ndarray)
    assert freqs.shape == u.shape == (21,)
    assert u[0] == pytest.approx(6.8)


def test_prediction_input_rejects_bad_n():
    with pytest.raises(ValueError, match="n must be a positive integer"):
        prediction_input_uncertainty(1.2, 1.0, 0)


def test_the_bare_names_are_gone():
    # The bare names shadowed the GUM pair at the package root. They were
    # deprecated in 3.1 and removed in 4.0; only the insulation_* names remain.
    import phonometry.building.measurement.uncertainty as bu

    for name in ("coverage_factor", "expanded_uncertainty"):
        with pytest.raises(AttributeError):
            getattr(bu, name)
    assert bu.insulation_coverage_factor(0.95) == insulation_coverage_factor(0.95)
    assert bu.insulation_expanded_uncertainty(1.2) == insulation_expanded_uncertainty(
        1.2
    )


# ---------------------------------------------------------------------------
# Annex B worked example (Tables B.1/B.2)
# ---------------------------------------------------------------------------


def test_annex_b_uncorrelated_single_number_uncertainties() -> None:
    """Formula (B.2) on the Table B.1 spectrum gives u(Rw+C50-5000) = 0,6 dB
    and u(Rw+Ctr,50-5000) = 0,8 dB.
    """
    import reference_data as ref

    from phonometry.building.measurement.insulation import (
        _SPECTRUM1_50_5000,
        _SPECTRUM2_50_5000,
    )

    ri = np.asarray(ref.ISO12999_1_ANNEX_B_RI)
    ui = np.asarray(ref.ISO12999_1_ANNEX_B_UI)
    u_c = single_number_uncertainty_uncorrelated(
        ui, np.asarray(_SPECTRUM1_50_5000) - ri
    )
    u_ctr = single_number_uncertainty_uncorrelated(
        ui, np.asarray(_SPECTRUM2_50_5000) - ri
    )
    assert u_c == pytest.approx(0.6034, abs=1e-3)
    assert round(u_c, 1) == ref.ISO12999_1_ANNEX_B_U_UNCORR_C
    assert u_ctr == pytest.approx(0.7924, abs=1e-3)
    assert round(u_ctr, 1) == ref.ISO12999_1_ANNEX_B_U_UNCORR_CTR


def test_annex_b_correlated_rw_uncertainty() -> None:
    """Formula (B.6) with the 0,1 dB shift gives u(Rw) = 1,9 dB."""
    import reference_data as ref

    from phonometry import building

    ri = np.asarray(ref.ISO12999_1_ANNEX_B_RI)
    ui = np.asarray(ref.ISO12999_1_ANNEX_B_UI)
    up = building.weighted_rating_extended(
        ri + ui, ref.ISO12999_1_ANNEX_B_FREQ, one_decimal=True
    ).rating
    down = building.weighted_rating_extended(
        ri - ui, ref.ISO12999_1_ANNEX_B_FREQ, one_decimal=True
    ).rating
    assert (up - down) / 2.0 == pytest.approx(ref.ISO12999_1_ANNEX_B_U_CORR_RW)


def test_annex_b_correlated_adaptation_sum_uncertainties() -> None:
    """Formulae (B.3)-(B.5): u(Rw+C50-5000) = 2,1 dB and
    u(Rw+Ctr,50-5000) = 2,6 dB (unrounded 2,05 / 2,63).
    """
    import reference_data as ref

    from phonometry.building.measurement.insulation import (
        _SPECTRUM1_50_5000,
        _SPECTRUM2_50_5000,
    )

    ri = np.asarray(ref.ISO12999_1_ANNEX_B_RI)
    ui = np.asarray(ref.ISO12999_1_ANNEX_B_UI)

    def rw_plus_c(spectrum: np.ndarray, shift: np.ndarray) -> float:
        return float(-10.0 * np.log10(np.sum(10.0 ** ((spectrum - ri + shift) / 10.0))))

    for spectrum, printed, unrounded in (
        (
            np.asarray(_SPECTRUM1_50_5000, dtype=float),
            ref.ISO12999_1_ANNEX_B_U_CORR_C,
            2.05,
        ),
        (
            np.asarray(_SPECTRUM2_50_5000, dtype=float),
            ref.ISO12999_1_ANNEX_B_U_CORR_CTR,
            2.63,
        ),
    ):
        u = (rw_plus_c(spectrum, -ui) - rw_plus_c(spectrum, +ui)) / 2.0
        assert u == pytest.approx(unrounded, abs=0.01)
        assert u == pytest.approx(printed, abs=0.06)


def test_uncorrelated_rejects_non_finite_reference_differences() -> None:
    nan_reference = [0.0, float("nan")]
    inf_uncertainty = [1.0, float("inf")]
    with pytest.raises(ValueError, match="finite"):
        single_number_uncertainty_uncorrelated([1.0, 1.0], nan_reference)
    with pytest.raises(ValueError, match="finite"):
        single_number_uncertainty_uncorrelated(inf_uncertainty, [0.0, 0.0])
