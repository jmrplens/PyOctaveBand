#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 1996-2:2017 environmental-noise determination.

Anchored on the Annex C.5 tonal-audibility worked examples and the Annex G.2
uncertainty budget (genuine numeric oracles), plus closed-form checks of the
survey method, the residual correction and the uncertainty combination.
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

from phonometry import environment
from phonometry.environment.assessment.measurement import (
    EnvironmentalMeasurementWarning,
)


# ---------------------------------------------------------------------------
# Tonal audibility -- Annex C.5 oracles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("lpt", "lpn", "fc", "delta_expected", "kt"), ref.ISO1996_2_TONAL_EXAMPLES
)
def test_tonal_audibility_annex_c5(
    lpt: float, lpn: float, fc: float, delta_expected: float, kt: float
) -> None:
    """Formula (C.3) reproduces Annex C.5 examples 1/2/4 within 0.05 dB."""
    delta = environment.tonal_audibility(lpt, lpn, fc)
    assert delta == pytest.approx(delta_expected, abs=0.05)
    assert environment.tonal_adjustment(delta) == pytest.approx(kt)


def test_tonal_audibility_example3_loose() -> None:
    """Example 3 is within 0.7 dB (printed value rounded in the figure)."""
    lpt, lpn, fc, delta_expected, kt = ref.ISO1996_2_TONAL_EXAMPLE3
    delta = environment.tonal_audibility(lpt, lpn, fc)
    assert delta == pytest.approx(delta_expected, abs=0.7)
    assert environment.tonal_adjustment(delta) == pytest.approx(kt)


def test_tonal_adjustment_piecewise() -> None:
    """Kt piecewise (Formulae C.4-C.6): 0 below 4, sloped 4-10, 6 above 10."""
    assert environment.tonal_adjustment(3.0) == 0.0
    assert environment.tonal_adjustment(4.0) == 0.0
    assert environment.tonal_adjustment(7.0) == pytest.approx(3.0)
    assert environment.tonal_adjustment(10.0) == pytest.approx(6.0)
    assert environment.tonal_adjustment(15.0) == 6.0


def test_tonal_adjustment_is_continuous_and_fractional() -> None:
    """Kt is not restricted to integers on the sloped branch."""
    assert environment.tonal_adjustment(6.4) == pytest.approx(2.4)


def test_assess_tonal_audibility_bundles() -> None:
    lpt, lpn, fc, delta_expected, kt = ref.ISO1996_2_TONAL_EXAMPLES[0]
    res = environment.assess_tonal_audibility(lpt, lpn, fc)
    assert res.audibility == pytest.approx(delta_expected, abs=0.05)
    assert res.adjustment == pytest.approx(kt)
    assert res.critical_bandwidth == pytest.approx(0.2 * fc)


# ---------------------------------------------------------------------------
# Critical bandwidth (Table C.1)
# ---------------------------------------------------------------------------
def test_critical_bandwidth_table_c1() -> None:
    assert environment.critical_bandwidth(200.0) == 100.0
    assert environment.critical_bandwidth(500.0) == 100.0
    assert environment.critical_bandwidth(430.0) == 100.0  # example 2 band width
    assert environment.critical_bandwidth(4000.0) == 800.0  # example 1 band width
    assert environment.critical_bandwidth(755.0) == pytest.approx(151.0)  # example 4


# ---------------------------------------------------------------------------
# Mean-audibility route (Table J.1)
# ---------------------------------------------------------------------------
def test_table_j1_mapping() -> None:
    cases = {-1.0: 0, 0.0: 0, 1.0: 1, 3.0: 2, 5.0: 3, 8.0: 4, 11.0: 5, 13.0: 6}
    for delta_l, kt in cases.items():
        assert environment.tonal_adjustment_from_mean_audibility(delta_l) == kt


def test_table_j1_coarse() -> None:
    assert environment.tonal_adjustment_from_mean_audibility(2.0, coarse=True) == 0
    assert environment.tonal_adjustment_from_mean_audibility(5.0, coarse=True) == 3
    assert environment.tonal_adjustment_from_mean_audibility(10.0, coarse=True) == 6


# ---------------------------------------------------------------------------
# Survey method (Annex K)
# ---------------------------------------------------------------------------
def test_survey_thresholds() -> None:
    """A band flagged only when it exceeds both neighbours by the threshold."""
    freqs = [80.0, 100.0, 125.0, 160.0, 200.0, 500.0, 630.0, 800.0]
    # 125 Hz band +20 over neighbours (>15 low-freq threshold) -> flagged
    # 630 Hz band +6 over neighbours (>5 high-freq threshold) -> flagged
    levels = [40.0, 40.0, 60.0, 40.0, 40.0, 50.0, 56.0, 50.0]
    flags = environment.tonal_seeking_survey(levels, freqs)
    assert flags[2]  # 125 Hz
    assert flags[6]  # 630 Hz
    assert not flags[0]  # end bands never flagged
    assert not flags[-1]


def test_survey_below_threshold_not_flagged() -> None:
    freqs = [160.0, 200.0, 250.0]
    levels = [40.0, 45.0, 40.0]  # +5 < 8 dB mid-freq threshold
    assert not environment.tonal_seeking_survey(levels, freqs)[1]


def test_survey_names_both_band_counts() -> None:
    """A level without a band centre is refused, naming what each input holds."""
    with pytest.raises(
        ValueError,
        match=r"tonal_seeking_survey: 'levels' .* must each carry one value per band",
    ):
        environment.tonal_seeking_survey(
            [40.0, 60.0, 40.0, 41.0], [100.0, 125.0, 160.0]
        )


# ---------------------------------------------------------------------------
# Residual-noise correction (Formula 16, Annex I)
# ---------------------------------------------------------------------------
def test_residual_correction_formula16() -> None:
    # Independent hand value: a source 10 dB above the residual (L'=60, Lres=50)
    # gives 10*lg(10^6 - 10^5) = 10*lg(9e5) = 59.5424 dB.
    res = environment.residual_sound_correction(60.0, 50.0)
    assert res.corrected_level == pytest.approx(59.5424, abs=1e-3)
    assert res.margin == pytest.approx(10.0)
    assert res.reliable
    # a second point cross-checked by hand: 58 over 50 -> 57.2506 dB
    assert environment.residual_sound_correction(
        58.0, 50.0
    ).corrected_level == pytest.approx(57.2506, abs=1e-3)


def test_residual_correction_unreliable_warns() -> None:
    with pytest.warns(
        EnvironmentalMeasurementWarning,
        match=r"Use reportable_upper_bound, not corrected_level",
    ):
        res = environment.residual_sound_correction(50.0, 48.0)  # margin 2 dB <= 3
    assert not res.reliable
    # 10.4: with a margin <= 3 dB no correction is allowed; the reportable
    # value is the UNCORRECTED measured level, as an upper bound (the
    # corrected value estimates from below).
    assert res.reportable_upper_bound == pytest.approx(50.0)
    assert res.corrected_level < res.reportable_upper_bound


def test_residual_correction_reports_measured_as_bound() -> None:
    res = environment.residual_sound_correction(60.0, 50.0)
    assert res.reportable_upper_bound == pytest.approx(60.0)


def test_residual_not_below_raises() -> None:
    with pytest.raises(
        ValueError, match=r"'residual_level' must be below 'measured_level'"
    ):
        environment.residual_sound_correction(50.0, 50.0)


def test_gaussian_residual_i1_i2() -> None:
    # Independent hand values: (I.1) 50 + 0.115*(10/1.28)^2 = 57.019;
    # (I.2) 50 + 0.115*(12/1.65)^2 = 56.0826.
    assert environment.gaussian_residual_level(50.0, l90=40.0) == pytest.approx(
        57.019, abs=1e-3
    )
    assert environment.gaussian_residual_level(50.0, l95=38.0) == pytest.approx(
        56.0826, abs=1e-3
    )


def test_gaussian_residual_needs_one_percentile() -> None:
    with pytest.raises(ValueError, match=r"Supply exactly one of 'l90' or 'l95'"):
        environment.gaussian_residual_level(50.0, l90=40.0, l95=38.0)
    with pytest.raises(ValueError, match=r"Supply exactly one of 'l90' or 'l95'"):
        environment.gaussian_residual_level(50.0)


def test_gaussian_residual_rejects_inverted_percentiles() -> None:
    # L90 > L50 is impossible (exceedance ordering): almost certainly swapped
    # arguments; the squared spread would otherwise hide the mistake.
    with pytest.raises(ValueError, match=r"'l90' exceeds 'l50'.*look swapped"):
        environment.gaussian_residual_level(40.0, l90=50.0)
    with pytest.raises(ValueError, match=r"'l95' exceeds 'l50'.*look swapped"):
        environment.gaussian_residual_level(40.0, l95=41.0)


# ---------------------------------------------------------------------------
# Measurement uncertainty (Clause 4, Annex F, Annex G oracle)
# ---------------------------------------------------------------------------
def test_combined_uncertainty_g2() -> None:
    """Annex G.2 component products combine to u = 2.18 dB."""
    u = environment.combined_standard_uncertainty(ref.ISO1996_2_G2_CONTRIBUTIONS)
    assert u == pytest.approx(ref.ISO1996_2_G2_COMBINED, abs=0.01)


def test_expanded_uncertainty_g2() -> None:
    """k = 2 (95 %) expansion of the G.2 uncertainty -> 4.36 dB."""
    u = environment.combined_standard_uncertainty(ref.ISO1996_2_G2_CONTRIBUTIONS)
    assert environment.environmental_expanded_uncertainty(u) == pytest.approx(
        ref.ISO1996_2_G2_EXPANDED, abs=0.01
    )
    assert environment.environmental_expanded_uncertainty(u, confidence=0.80) == pytest.approx(
        1.3 * u
    )


def test_combined_uncertainty_accepts_pairs() -> None:
    """(uncertainty, sensitivity) pairs give the same result as the products."""
    pairs = [(1.0, 2.0), (3.0, 1.0)]
    assert environment.combined_standard_uncertainty(pairs) == pytest.approx(
        np.hypot(2.0, 3.0)
    )


def test_expanded_uncertainty_bad_confidence() -> None:
    with pytest.raises(ValueError, match=r"'confidence' must be one of"):
        environment.environmental_expanded_uncertainty(1.0, confidence=0.90)


def test_residual_correction_uncertainty_f9() -> None:
    """Formulae (F.7)/(F.8)/(F.9): sensitivity-weighted quadrature.

    Independent hand value for (L'=58, Lres=50, uL'=0.5, ures=2.0): m=10^-0.8,
    cL'=1.18834, cres=-0.18834 -> uL = hypot(0.59417, 0.37667) = 0.7035 dB.
    """
    assert environment.residual_correction_uncertainty(
        58.0, 50.0, 0.5, 2.0
    ) == pytest.approx(0.7035, abs=1e-3)


def test_repeated_measurements() -> None:
    """Formulae (17)-(20): energy mean, primary uncertainty, approximation."""
    levels = [58.0, 59.0, 57.0, 60.0]
    res = environment.uncertainty_from_repeated_measurements(levels)
    assert isinstance(res, environment.RepeatedMeasurementResult)
    # Independent hand values for [58, 59, 57, 60]: the energy mean (Formula
    # 18) is 58.6428 dB; the primary route (Formula 17 energy-domain sk =
    # 215423, Formula 19) gives uk = 10 lg(731618 + 215423) - 58.6428 =
    # 1.1205 dB; the Note 2 approximation (Formula 20) gives 1.3015 dB.
    assert res.mean_level == pytest.approx(58.6428, abs=1e-3)
    assert res.standard_uncertainty == pytest.approx(1.121, abs=2e-3)
    assert res.approximate_uncertainty == pytest.approx(1.3015, abs=1e-3)
    assert res.n == 4


def test_repeated_measurements_spread_levels() -> None:
    """Spread levels: the primary Formulae (17)+(19) route stays sane
    (3.944 dB for [50, 60, 70]) while the Note 2 approximation inflates to
    12.183 dB and triggers the spread warning.
    """
    with pytest.warns(
        EnvironmentalMeasurementWarning,
        match=r"Formula \(20\) approximation \(approximate_uncertainty\)",
    ):
        res = environment.uncertainty_from_repeated_measurements([50.0, 60.0, 70.0])
    assert res.standard_uncertainty == pytest.approx(3.944, abs=2e-3)
    assert res.approximate_uncertainty == pytest.approx(12.183, abs=2e-3)


def test_repeated_measurements_needs_two() -> None:
    with pytest.raises(
        ValueError, match="'levels' must be a 1-D array of at least two"
    ):
        environment.uncertainty_from_repeated_measurements([58.0])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    res = environment.assess_tonal_audibility(54.1, 45.2, 430.0)
    assert res.plot() is not None
