#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 12999-2:2020 sound-absorption measurement uncertainty."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import reference_data as ref

from phonometry import materials


# ---------------------------------------------------------------------------
# Coverage factors (Table 3)
# ---------------------------------------------------------------------------
def test_coverage_factor_table3() -> None:
    expected = {0.68: 1.0, 0.80: 1.3, 0.90: 1.6, 0.95: 2.0, 0.99: 2.6, 0.999: 3.3}
    for confidence, k in expected.items():
        assert materials.absorption_coverage_factor(confidence) == k


def test_coverage_factor_rejects_untabulated() -> None:
    with pytest.raises(ValueError, match="not tabulated in Table 3"):
        materials.absorption_coverage_factor(0.975)


def test_coverage_factors_differ_from_iso12999_1() -> None:
    """Table 3 uses rounded factors (2.0, 2.6), not the Gaussian-exact ones."""
    assert materials.absorption_coverage_factor(0.95) == 2.0
    assert materials.absorption_coverage_factor(0.99) == 2.6


# ---------------------------------------------------------------------------
# Clause 5 - one-third-octave bands, worked Table 4
# ---------------------------------------------------------------------------
def test_absorption_coefficient_reproduces_table4() -> None:
    res = materials.sound_absorption_coefficient_uncertainty(
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
    res = materials.sound_absorption_coefficient_uncertainty([0.68], [1000])
    np.testing.assert_allclose(res.standard_uncertainty, [0.0422])
    np.testing.assert_allclose(res.expanded_uncertainty, [0.0844])  # k=2, exact


def test_scalar_inputs_are_promoted_to_one_band() -> None:
    # The type hints allow a bare scalar; it must behave like a 1-band array.
    res = materials.sound_absorption_coefficient_uncertainty(0.68, 1000)
    np.testing.assert_allclose(res.standard_uncertainty, [0.0422])
    assert res.frequencies.shape == (1,)
    np.testing.assert_allclose(
        materials.equivalent_area_uncertainty(8.0, 500).standard_uncertainty,
        [0.050 * 8.0 + 0.015 * 10.0],
    )


def test_repeatability_is_0_6_of_reproducibility() -> None:
    rep = materials.sound_absorption_coefficient_uncertainty(
        [0.5], [500], condition="repeatability"
    )
    repro = materials.sound_absorption_coefficient_uncertainty([0.5], [500])
    np.testing.assert_allclose(
        rep.standard_uncertainty, 0.6 * repro.standard_uncertainty
    )


def test_equivalent_area_formula_2() -> None:
    # sigma_R = m*A_T + n*S, S = 10 m². At 500 Hz: m=0.050, n=0.015.
    res = materials.equivalent_area_uncertainty([8.0], [500])
    np.testing.assert_allclose(res.standard_uncertainty, [0.050 * 8.0 + 0.015 * 10.0])


def test_equivalent_area_ylabel_is_area() -> None:
    res = materials.equivalent_area_uncertainty([5.0, 6.0], [500, 1000])
    assert res.quantity == "equivalent_area"
    # Reported to one decimal (not a coefficient).
    np.testing.assert_allclose(
        res.reported_expanded_uncertainty, np.round(res.expanded_uncertainty, 1)
    )


# ---------------------------------------------------------------------------
# Clause 6 - practical coefficient, worked Table 5
# ---------------------------------------------------------------------------
def test_practical_coefficient_reproduces_table5() -> None:
    res = materials.practical_coefficient_uncertainty(
        ref.ISO12999_2_TABLE5_ALPHA_P, ref.ISO12999_2_TABLE5_FREQ
    )
    np.testing.assert_allclose(
        res.reported_expanded_uncertainty, ref.ISO12999_2_TABLE5_U_K2
    )


def test_practical_coefficient_500hz_is_constant() -> None:
    # Table 2 at 500/1000/2000 Hz has m=0 => sigma_R = 0.040 regardless of alpha.
    for alpha in (0.1, 0.9):
        res = materials.practical_coefficient_uncertainty([alpha], [500])
        np.testing.assert_allclose(res.standard_uncertainty, [0.040])


# ---------------------------------------------------------------------------
# Clause 7 - single numbers
# ---------------------------------------------------------------------------
def test_weighted_coefficient_example_1() -> None:
    res = materials.weighted_coefficient_uncertainty(ref.ISO12999_2_ALPHA_W_EXAMPLE)
    np.testing.assert_allclose(res.standard_uncertainty, [0.035])
    assert float(res.reported_expanded_uncertainty[0]) == ref.ISO12999_2_ALPHA_W_U_K2


def test_weighted_coefficient_repeatability() -> None:
    res = materials.weighted_coefficient_uncertainty(0.7, condition="repeatability")
    np.testing.assert_allclose(res.standard_uncertainty, [0.020])


def test_single_number_rating_example_2() -> None:
    res = materials.single_number_rating_uncertainty(ref.ISO12999_2_DLALPHA_EXAMPLE)
    np.testing.assert_allclose(res.standard_uncertainty, [0.10 * 8.1])
    assert float(res.reported_expanded_uncertainty[0]) == ref.ISO12999_2_DLALPHA_U_K2


def test_single_number_rating_repeatability() -> None:
    res = materials.single_number_rating_uncertainty(8.1, condition="repeatability")
    np.testing.assert_allclose(res.standard_uncertainty, [0.02 * 8.1])


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def test_interval_bounds() -> None:
    res = materials.sound_absorption_coefficient_uncertainty([0.5], [1000])
    u = res.expanded_uncertainty
    np.testing.assert_allclose(res.lower, res.values - u)
    np.testing.assert_allclose(res.upper, res.values + u)


def test_reported_rounding_rule() -> None:
    # Coefficients -> 2 decimals; area and DLalpha -> 1 decimal.
    coeff = materials.sound_absorption_coefficient_uncertainty([0.33], [63])  # U~0.327
    assert float(coeff.reported_expanded_uncertainty[0]) == 0.33
    rating = materials.single_number_rating_uncertainty(8.1)  # U=1.62 -> 1.6
    assert float(rating.reported_expanded_uncertainty[0]) == 1.6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_frequency_must_be_tabulated() -> None:
    with pytest.raises(ValueError, match="not a tabulated one-third-octave"):
        materials.sound_absorption_coefficient_uncertainty([0.5], [440])
    with pytest.raises(ValueError, match="not a tabulated octave"):
        materials.practical_coefficient_uncertainty([0.5], [630])


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"'alpha'.*must all have the same shape"):
        materials.sound_absorption_coefficient_uncertainty([0.5, 0.6], [1000])
    with pytest.raises(ValueError, match=r"'alpha_p'.*must all have the same shape"):
        materials.practical_coefficient_uncertainty([0.5, 0.6], [1000])
    with pytest.raises(ValueError, match=r"'area'.*must all have the same shape"):
        materials.equivalent_area_uncertainty([5.0, 6.0], [500])


def test_unknown_condition_raises() -> None:
    with pytest.raises(ValueError, match="'condition' must be one of"):
        materials.sound_absorption_coefficient_uncertainty(
            [0.5], [1000], condition="typical"
        )


def test_negative_rating_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        materials.single_number_rating_uncertainty(-1.0)


def test_non_finite_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        materials.sound_absorption_coefficient_uncertainty([np.nan], [1000])


@pytest.mark.parametrize(
    "field", ["values", "standard_uncertainty", "expanded_uncertainty", "frequencies"]
)
def test_spectra_of_unequal_length_are_refused(field: str) -> None:
    """A band result cannot hold one spectrum shorter than the rest.

    ``lower`` and ``upper`` are subtractions of two of these arrays, so a
    one-element column would be stretched over the whole spectrum by numpy
    and the interval would still come back the right length for the table
    that prints it, stating a coverage the measurement never had.
    """
    res = materials.sound_absorption_coefficient_uncertainty(
        ref.ISO12999_2_TABLE4_ALPHA_S, ref.ISO12999_2_TABLE4_FREQ
    )
    one_short = getattr(res, field)[:-1]
    with pytest.raises(ValueError, match=f"'{field}'"):
        dataclasses.replace(res, **{field: one_short})


def test_single_number_carries_a_value_without_a_spectrum() -> None:
    """The Clause 7 single numbers are exempt from the band-axis check.

    ``frequencies`` is empty there while ``values`` holds the one number, so
    the frequency axis is pinned only where the result has one; folding it
    into the unconditional group would make ``αw`` unconstructible.
    """
    for res in (
        materials.weighted_coefficient_uncertainty(0.7),
        materials.single_number_rating_uncertainty(8.1),
    ):
        assert res.frequencies.shape == (0,)
        assert res.values.shape == (1,)
        assert res.standard_uncertainty.shape == (1,)
        assert res.expanded_uncertainty.shape == (1,)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_single_number_raises() -> None:
    res = materials.weighted_coefficient_uncertainty(0.7)
    with pytest.raises(ValueError, match="single-number"):
        res.plot()


def test_plot_band_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    res = materials.sound_absorption_coefficient_uncertainty(
        ref.ISO12999_2_TABLE4_ALPHA_S, ref.ISO12999_2_TABLE4_FREQ
    )
    ax = res.plot()
    assert ax.get_title().startswith("ISO 12999-2")
