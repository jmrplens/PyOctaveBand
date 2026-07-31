#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 12999-2:2020 sound-absorption measurement uncertainty."""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    absorption_coverage_factor,
    equivalent_area_uncertainty,
    practical_coefficient_uncertainty,
    single_number_rating_uncertainty,
    sound_absorption_coefficient_uncertainty,
    weighted_coefficient_uncertainty,
)


# ---------------------------------------------------------------------------
# Coverage factors (Table 3)
# ---------------------------------------------------------------------------
def test_coverage_factor_table3() -> None:
    expected = {0.68: 1.0, 0.80: 1.3, 0.90: 1.6, 0.95: 2.0, 0.99: 2.6, 0.999: 3.3}
    for confidence, k in expected.items():
        assert absorption_coverage_factor(confidence) == k


def test_coverage_factor_rejects_untabulated() -> None:
    with pytest.raises(ValueError, match="not tabulated in Table 3"):
        absorption_coverage_factor(0.975)


def test_coverage_factors_differ_from_iso12999_1() -> None:
    """Table 3 uses rounded factors (2.0, 2.6), not the Gaussian-exact ones."""
    assert absorption_coverage_factor(0.95) == 2.0
    assert absorption_coverage_factor(0.99) == 2.6


# ---------------------------------------------------------------------------
# Clause 5 - one-third-octave bands, worked Table 4
# ---------------------------------------------------------------------------
def test_absorption_coefficient_reproduces_table4() -> None:
    res = sound_absorption_coefficient_uncertainty(
        ref.ISO12999_2_TABLE4_ALPHA_S,
        ref.ISO12999_2_TABLE4_FREQ,
        condition="reproducibility",
        confidence=0.95,
    )
    assert res.coverage_factor == 2.0
    np.testing.assert_allclose(
        res.reported_expanded_uncertainty, ref.ISO12999_2_TABLE4_U_K2
    )


def test_absorption_coefficient_formula_1() -> None:
    # 1000 Hz: m=0.040, n=0.015 => sigma_R = 0.040*0.68 + 0.015 = 0.0422.
    res = sound_absorption_coefficient_uncertainty([0.68], [1000])
    np.testing.assert_allclose(res.standard_uncertainty, [0.0422])
    np.testing.assert_allclose(res.expanded_uncertainty, [0.0844])  # k=2, exact


def test_scalar_inputs_are_promoted_to_one_band() -> None:
    # The type hints allow a bare scalar; it must behave like a 1-band array.
    res = sound_absorption_coefficient_uncertainty(0.68, 1000)
    np.testing.assert_allclose(res.standard_uncertainty, [0.0422])
    assert res.frequencies.shape == (1,)
    np.testing.assert_allclose(
        equivalent_area_uncertainty(8.0, 500).standard_uncertainty,
        [0.050 * 8.0 + 0.015 * 10.0],
    )


def test_repeatability_is_0_6_of_reproducibility() -> None:
    rep = sound_absorption_coefficient_uncertainty([0.5], [500], condition="repeatability")
    repro = sound_absorption_coefficient_uncertainty([0.5], [500])
    np.testing.assert_allclose(
        rep.standard_uncertainty, 0.6 * repro.standard_uncertainty
    )


def test_equivalent_area_formula_2() -> None:
    # sigma_R = m*A_T + n*S, S = 10 m². At 500 Hz: m=0.050, n=0.015.
    res = equivalent_area_uncertainty([8.0], [500])
    np.testing.assert_allclose(res.standard_uncertainty, [0.050 * 8.0 + 0.015 * 10.0])


def test_equivalent_area_ylabel_is_area() -> None:
    res = equivalent_area_uncertainty([5.0, 6.0], [500, 1000])
    assert res.quantity == "equivalent_area"
    # Reported to one decimal (not a coefficient).
    np.testing.assert_allclose(res.reported_expanded_uncertainty, np.round(
        res.expanded_uncertainty, 1))


# ---------------------------------------------------------------------------
# Clause 6 - practical coefficient, worked Table 5
# ---------------------------------------------------------------------------
def test_practical_coefficient_reproduces_table5() -> None:
    res = practical_coefficient_uncertainty(
        ref.ISO12999_2_TABLE5_ALPHA_P, ref.ISO12999_2_TABLE5_FREQ
    )
    np.testing.assert_allclose(
        res.reported_expanded_uncertainty, ref.ISO12999_2_TABLE5_U_K2
    )


def test_practical_coefficient_500hz_is_constant() -> None:
    # Table 2 at 500/1000/2000 Hz has m=0 => sigma_R = 0.040 regardless of alpha.
    for alpha in (0.1, 0.9):
        res = practical_coefficient_uncertainty([alpha], [500])
        np.testing.assert_allclose(res.standard_uncertainty, [0.040])


# ---------------------------------------------------------------------------
# Clause 7 - single numbers
# ---------------------------------------------------------------------------
def test_weighted_coefficient_example_1() -> None:
    res = weighted_coefficient_uncertainty(ref.ISO12999_2_ALPHA_W_EXAMPLE)
    np.testing.assert_allclose(res.standard_uncertainty, [0.035])
    assert float(res.reported_expanded_uncertainty[0]) == ref.ISO12999_2_ALPHA_W_U_K2


def test_weighted_coefficient_repeatability() -> None:
    res = weighted_coefficient_uncertainty(0.7, condition="repeatability")
    np.testing.assert_allclose(res.standard_uncertainty, [0.020])


def test_single_number_rating_example_2() -> None:
    res = single_number_rating_uncertainty(ref.ISO12999_2_DLALPHA_EXAMPLE)
    np.testing.assert_allclose(res.standard_uncertainty, [0.10 * 8.1])
    assert float(res.reported_expanded_uncertainty[0]) == ref.ISO12999_2_DLALPHA_U_K2


def test_single_number_rating_repeatability() -> None:
    res = single_number_rating_uncertainty(8.1, condition="repeatability")
    np.testing.assert_allclose(res.standard_uncertainty, [0.02 * 8.1])


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def test_interval_bounds() -> None:
    res = sound_absorption_coefficient_uncertainty([0.5], [1000])
    u = res.expanded_uncertainty
    np.testing.assert_allclose(res.lower, res.values - u)
    np.testing.assert_allclose(res.upper, res.values + u)


def test_reported_rounding_rule() -> None:
    # Coefficients -> 2 decimals; area and DLalpha -> 1 decimal.
    coeff = sound_absorption_coefficient_uncertainty([0.33], [63])  # U~0.327
    assert float(coeff.reported_expanded_uncertainty[0]) == 0.33
    rating = single_number_rating_uncertainty(8.1)  # U=1.62 -> 1.6
    assert float(rating.reported_expanded_uncertainty[0]) == 1.6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_frequency_must_be_tabulated() -> None:
    with pytest.raises(ValueError, match="not a tabulated one-third-octave"):
        sound_absorption_coefficient_uncertainty([0.5], [440])
    with pytest.raises(ValueError, match="not a tabulated octave"):
        practical_coefficient_uncertainty([0.5], [630])


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        sound_absorption_coefficient_uncertainty([0.5, 0.6], [1000])
    with pytest.raises(ValueError, match="same shape"):
        equivalent_area_uncertainty([5.0, 6.0], [500])


def test_unknown_condition_raises() -> None:
    with pytest.raises(ValueError, match="'condition' must be one of"):
        sound_absorption_coefficient_uncertainty([0.5], [1000], condition="typical")


def test_negative_rating_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        single_number_rating_uncertainty(-1.0)


def test_non_finite_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        sound_absorption_coefficient_uncertainty([np.nan], [1000])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_single_number_raises() -> None:
    res = weighted_coefficient_uncertainty(0.7)
    with pytest.raises(ValueError, match="single-number"):
        res.plot()


def test_plot_band_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    res = sound_absorption_coefficient_uncertainty(
        ref.ISO12999_2_TABLE4_ALPHA_S, ref.ISO12999_2_TABLE4_FREQ
    )
    ax = res.plot()
    assert ax.get_title().startswith("ISO 12999-2")
