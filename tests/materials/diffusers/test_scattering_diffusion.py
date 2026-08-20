#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Scattering (ISO 17497-1) and diffusion (ISO 17497-2) coefficient tests.

Neither part of ISO 17497 has a numeric worked example, so the tests anchor
on algebraic and physical identities, on a committed model-generated arc for
a published diffuser geometry (an arithmetic oracle for Formulas (5)/(7)),
and on the published Cox & D'Antonio Appendix B BEM table as the external
third-party anchor:

- diffusion ``d = 0`` when all energy reaches one receiver and ``d = 1`` when
  all ``n`` receivers are equal (proving the ``(n - 1)`` autocorrelation form);
  the normalisation of Formula (7) maps ``d_r -> 0`` and ``1 -> 1``.
- scattering ``s = 0`` when ``alpha_spec == alpha_s`` and ``s = 1`` when
  ``alpha_spec == 1``; a full synthetic end-to-end pass with a hand-computed
  expected value; negative truncation; ``s > 1`` preserved.
- Table 1 base-plate limits reproduced exactly and the over-limit warning.
- hand-computed Annex A (A.1)/(A.3)/(A.5) uncertainties.
- input-validation guards raise ``ValueError``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from reference_data import (
    ISO17497_1_CHAIN_ALPHA_S,
    ISO17497_1_CHAIN_ALPHA_SPEC,
    ISO17497_1_CHAIN_SCATTERING,
    ISO17497_2_FLAT_DIFFUSION,
    ISO17497_2_FLAT_LEVELS,
    ISO17497_2_NORMALIZED_DIFFUSION,
    ISO17497_2_QRD_DIFFUSION,
    ISO17497_2_QRD_LEVELS,
)

from phonometry.materials.diffusers.reverberation_room_scattering import (
    BASE_PLATE_BANDS,
    BASE_PLATE_MAX_SCATTERING,
    ScatteringDiffusionWarning,
    ScatteringResult,
    ScatteringUncertainty,
    absorption_coefficient_uncertainty,
    air_attenuation_coefficient,
    base_plate_scattering,
    check_base_plate_scattering,
    random_incidence_absorption,
    reverberation_time_uncertainty,
    scattering_coefficient,
    scattering_coefficient_spectrum,
    scattering_coefficient_uncertainty,
    specular_absorption_coefficient,
    speed_of_sound,
)
from phonometry.materials.diffusers.scattering_diffusion import (
    TWO_DIMENSIONAL_SOURCE_WEIGHTS,
    DiffusionResult,
    DiffusionSpectrum,
    area_factors,
    diffusion_spectrum,
    directional_diffusion,
    directional_diffusion_coefficient,
    normalized_diffusion_coefficient,
    random_incidence_diffusion,
)

# Fixed synthetic geometry for the scattering end-to-end oracle.
V = 200.0
S = 10.0
C = 343.2
K = 55.3  # ISO 17497-1 Sabine constant (Eqs. (1), (4), (6)).


# ---------------------------------------------------------------------------
# ISO 17497-1 air-property helpers (Eqs. (2)/(3)).
# ---------------------------------------------------------------------------
def test_speed_of_sound_20c_is_reference() -> None:
    # Eq. (2): c = 343.2 * sqrt((273.15 + 20) / 293.15) = 343.2 exactly.
    assert float(speed_of_sound(20.0)) == pytest.approx(343.2, abs=1e-9)


def test_speed_of_sound_monotonic_and_array() -> None:
    c = speed_of_sound([0.0, 20.0, 40.0])
    assert c[0] < c[1] < c[2]


def test_air_attenuation_uses_ten_lg_e() -> None:
    # Eq. (3): m = alpha / (10 lg e).
    alpha = 4.343
    assert float(air_attenuation_coefficient(alpha)) == pytest.approx(
        alpha / (10.0 * math.log10(math.e))
    )


# ---------------------------------------------------------------------------
# ISO 17497-1 scattering: identities and a synthetic end-to-end oracle.
# ---------------------------------------------------------------------------
def test_scattering_zero_when_spec_equals_diffuse() -> None:
    # Eq. (5): s = (alpha_spec - alpha_s) / (1 - alpha_s) = 0 when equal.
    s = scattering_coefficient(0.3, 0.3)
    assert float(s) == pytest.approx(0.0)


def test_scattering_one_when_spec_is_one() -> None:
    # alpha_spec = 1 => s = (1 - alpha_s) / (1 - alpha_s) = 1 for any alpha_s.
    for alpha_s in (0.0, 0.25, 0.5):
        assert float(scattering_coefficient(1.0, alpha_s)) == pytest.approx(1.0)


def test_scattering_negative_truncated_to_zero() -> None:
    # alpha_spec < alpha_s gives a negative raw s; Clause 8.3 truncates to 0.
    assert float(scattering_coefficient(0.2, 0.5)) == 0.0
    # ... but the untruncated value is available and is negative.
    raw = float(scattering_coefficient(0.2, 0.5, truncate_negative=False))
    assert raw < 0.0


def test_scattering_above_one_preserved() -> None:
    # Edge effects (Clause 6.3.2) can push s > 1; it must not be clipped.
    s = float(scattering_coefficient(1.2, 0.3))
    assert s > 1.0
    assert s == pytest.approx((1.2 - 0.3) / (1.0 - 0.3))


def test_scattering_end_to_end_synthetic() -> None:
    # Independent re-derivation of Eqs. (1), (4), (5) with plain arithmetic.
    T1, T2, T3, T4 = 8.0, 6.0, 7.5, 5.0
    expected_alpha_s = K * (V / S) * (1 / (C * T2) - 1 / (C * T1))
    expected_alpha_spec = K * (V / S) * (1 / (C * T4) - 1 / (C * T3))
    expected_s = (expected_alpha_spec - expected_alpha_s) / (1.0 - expected_alpha_s)

    alpha_s = random_incidence_absorption(V, S, c1=C, t1=T1, c2=C, t2=T2)
    alpha_spec = specular_absorption_coefficient(V, S, c3=C, t3=T3, c4=C, t4=T4)
    s = scattering_coefficient(alpha_spec, alpha_s)

    # Shared oracles from tests/reference_data/ (used by the CI report too).
    assert float(alpha_s) == pytest.approx(ISO17497_1_CHAIN_ALPHA_S)
    assert float(alpha_spec) == pytest.approx(ISO17497_1_CHAIN_ALPHA_SPEC)
    assert float(s) == pytest.approx(ISO17497_1_CHAIN_SCATTERING)
    # And it matches the independent re-derivation.
    assert float(alpha_s) == pytest.approx(expected_alpha_s)
    assert float(alpha_spec) == pytest.approx(expected_alpha_spec)
    assert float(s) == pytest.approx(expected_s)
    assert 0.0 <= float(s) <= 1.0


def test_scattering_end_to_end_above_one_reported() -> None:
    # A very short T4 makes alpha_spec > 1, so s > 1 and is reported as-is.
    T1, T3, T4 = 8.0, 7.5, 2.0
    alpha_s = random_incidence_absorption(V, S, c1=C, t1=T1, c2=C, t2=6.0)
    alpha_spec = specular_absorption_coefficient(V, S, c3=C, t3=T3, c4=C, t4=T4)
    s = float(scattering_coefficient(alpha_spec, alpha_s))
    assert float(alpha_spec) > 1.0
    assert s == pytest.approx(1.2097941324956527)
    assert s > 1.0


def test_air_attenuation_term_reduces_absorption() -> None:
    # The -(4 V / S)(m2 - m1) term lowers alpha_s when m2 > m1.
    base = random_incidence_absorption(V, S, c1=C, t1=8.0, c2=C, t2=6.0)
    with_air = random_incidence_absorption(
        V, S, c1=C, t1=8.0, c2=C, t2=6.0, m1=0.001, m2=0.002
    )
    assert float(with_air) < float(base)
    assert float(base) - float(with_air) == pytest.approx(4.0 * V / S * (0.002 - 0.001))


def test_base_plate_scattering_zero_when_t1_equals_t3() -> None:
    # Eq. (6): a perfectly symmetrical base plate has T1 == T3 => s_base = 0.
    s_base = base_plate_scattering(V, S, c1=C, t1=7.5, c3=C, t3=7.5)
    assert float(s_base) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ISO 17497-1 Table 1 base-plate limits and the over-limit warning.
# ---------------------------------------------------------------------------
def test_table1_exact_values_spot_bands() -> None:
    assert BASE_PLATE_BANDS == (
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
    )
    assert tuple(BASE_PLATE_MAX_SCATTERING) == BASE_PLATE_BANDS
    assert BASE_PLATE_MAX_SCATTERING[100] == 0.05
    assert BASE_PLATE_MAX_SCATTERING[500] == 0.05
    assert BASE_PLATE_MAX_SCATTERING[630] == 0.10
    assert BASE_PLATE_MAX_SCATTERING[1000] == 0.10
    assert BASE_PLATE_MAX_SCATTERING[1250] == 0.15
    assert BASE_PLATE_MAX_SCATTERING[2000] == 0.15
    assert BASE_PLATE_MAX_SCATTERING[2500] == 0.20
    assert BASE_PLATE_MAX_SCATTERING[4000] == 0.20
    assert BASE_PLATE_MAX_SCATTERING[5000] == 0.25


def test_base_plate_within_limits_no_warning() -> None:
    values = {b: BASE_PLATE_MAX_SCATTERING[b] for b in BASE_PLATE_BANDS}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exceeded = check_base_plate_scattering(values)
    assert exceeded == ()


def test_base_plate_over_limit_warns_and_lists_bands() -> None:
    values = dict.fromkeys(BASE_PLATE_BANDS, 0.0)
    values[100] = 0.06  # limit 0.05
    values[5000] = 0.30  # limit 0.25
    with pytest.warns(ScatteringDiffusionWarning):
        exceeded = check_base_plate_scattering(values)
    assert exceeded == (100, 5000)


def test_base_plate_checker_accepts_sequence() -> None:
    seq = [BASE_PLATE_MAX_SCATTERING[b] for b in BASE_PLATE_BANDS]
    assert check_base_plate_scattering(seq) == ()


# ---------------------------------------------------------------------------
# ISO 17497-1 Annex A uncertainty (Eqs. (A.1)/(A.3)/(A.5)).
# ---------------------------------------------------------------------------
def test_reverberation_time_uncertainty_a1() -> None:
    times = [6.0, 6.1, 5.9, 6.05]
    n = len(times)
    mean = sum(times) / n
    expected = math.sqrt(sum((t - mean) ** 2 for t in times) / (n * (n - 1)))
    u = reverberation_time_uncertainty(times)
    assert float(u) == pytest.approx(expected)
    assert float(u) == pytest.approx(0.04269562819149817)


def test_absorption_uncertainty_a3() -> None:
    ua, ub, Ta, Tb = 0.02, 0.03, 8.0, 6.0
    expected = K * V / (C * S) * math.sqrt((ub / Tb**2) ** 2 + (ua / Ta**2) ** 2)
    u = absorption_coefficient_uncertainty(V, S, c=C, t_a=Ta, u_a=ua, t_b=Tb, u_b=ub)
    assert float(u) == pytest.approx(expected)
    assert float(u) == pytest.approx(0.0028681248003840053)


def test_scattering_uncertainty_a5_and_expansion() -> None:
    alpha_s, alpha_spec = 0.3, 0.6
    u_alpha_s, u_alpha_spec = 0.01, 0.02
    expected = abs((alpha_spec - 1) / (1 - alpha_s)) * math.sqrt(
        (u_alpha_spec / (alpha_spec - 1)) ** 2 + (u_alpha_s / (1 - alpha_s)) ** 2
    )
    result = scattering_coefficient_uncertainty(
        alpha_spec, alpha_s, u_alpha_spec, u_alpha_s
    )
    assert isinstance(result, ScatteringUncertainty)
    assert float(result.u_scattering) == pytest.approx(expected)
    assert float(result.u_scattering) == pytest.approx(0.0297147342419613)
    # Expanded uncertainty at 95 % is 2 u_s (Annex A).
    assert float(result.expanded) == pytest.approx(2.0 * float(result.u_scattering))


# ---------------------------------------------------------------------------
# ISO 17497-2 directional diffusion (Formulas (5)/(6)).
# ---------------------------------------------------------------------------
def test_diffusion_zero_single_receiver_energy() -> None:
    # All energy at one receiver; a -inf dB level is zero energy.
    levels = [80.0, -np.inf, -np.inf, -np.inf]
    assert directional_diffusion_coefficient(levels) == pytest.approx(0.0)


def test_diffusion_one_when_all_receivers_equal() -> None:
    # Equal levels => numerator n(n-1)x^2, denominator (n-1) n x^2 => 1.
    for n in (2, 5, 13):
        levels = [72.0] * n
        assert directional_diffusion_coefficient(levels) == pytest.approx(1.0)


def test_diffusion_matches_formula_5_by_hand() -> None:
    levels = np.array([70.0, 74.0, 68.0, 72.0])
    p = 10.0 ** (levels / 10.0)
    n = levels.size
    expected = (p.sum() ** 2 - (p**2).sum()) / ((n - 1) * (p**2).sum())
    assert directional_diffusion_coefficient(levels) == pytest.approx(expected)
    assert 0.0 <= directional_diffusion_coefficient(levels) <= 1.0


# ---------------------------------------------------------------------------
# ISO 17497-2 arithmetic oracle: single-plane semicircular arc (37 receivers,
# 5 deg) generated by the library's own Fraunhofer far-field model
# (materials/diffusers/design.py) for a published geometry - the N = 7 QRD,
# 6 periods, 3.6 m wide, 0.2 m deep row of Cox & D'Antonio 3e Appendix B
# (the commercial N = 7 QRD of Hargreaves et al. 2000, Table I) - at 1000 Hz,
# normal incidence, with the equal-footprint zero-depth flat reference. The
# committed levels are model output, not third-party data; the checks below
# confirm Formula (5) and Formula (7) reproduce the committed coefficients
# from the levels alone (exact arithmetic, 1e-6). The independent external
# anchor against published third-party BEM data is
# test_predicted_band_dn_matches_cox_appendix_b below.
# ---------------------------------------------------------------------------
def test_diffusion_qrd_arc_matches_committed_oracle() -> None:
    d = directional_diffusion_coefficient(list(ISO17497_2_QRD_LEVELS))
    assert len(ISO17497_2_QRD_LEVELS) == 37
    assert d == pytest.approx(ISO17497_2_QRD_DIFFUSION, abs=1e-6)
    assert 0.0 <= d <= 1.0


def test_diffusion_flat_arc_matches_committed_oracle() -> None:
    d = directional_diffusion_coefficient(list(ISO17497_2_FLAT_LEVELS))
    assert len(ISO17497_2_FLAT_LEVELS) == 37
    assert d == pytest.approx(ISO17497_2_FLAT_DIFFUSION, abs=1e-6)
    # The flat reference collapses into the specular direction, so its
    # directional diffusion coefficient is far below the QRD's.
    assert d < directional_diffusion_coefficient(list(ISO17497_2_QRD_LEVELS))


def test_diffusion_qrd_normalized_matches_committed_oracle() -> None:
    d_qrd = directional_diffusion_coefficient(list(ISO17497_2_QRD_LEVELS))
    d_flat = directional_diffusion_coefficient(list(ISO17497_2_FLAT_LEVELS))
    d_n = float(normalized_diffusion_coefficient(d_qrd, d_flat))
    assert d_n == pytest.approx(ISO17497_2_NORMALIZED_DIFFUSION, abs=1e-6)
    # The QRD diffuses more than its flat reference, so removing the reference
    # baseline leaves a positive, bounded normalised coefficient.
    assert 0.0 < d_n < 1.0
    assert d_n > d_flat


def test_committed_arc_levels_regenerate_from_the_model() -> None:
    # Regression guard: the committed reference levels are exactly what the
    # library's Fraunhofer model predicts for the published geometry (N = 7
    # QRD, 6 periods, 3.6 m wide, 0.2 m deep; Cox & D'Antonio 3e Appendix B
    # section 7 / Hargreaves et al. 2000 Table I) at 1000 Hz, normal
    # incidence, rounded to 1e-3 dB.
    from diffuser_prediction import predicted_arc
    from reference_data import ISO17497_2_PREDICTION_FREQUENCY

    qrd = predicted_arc(ISO17497_2_PREDICTION_FREQUENCY)
    flat = predicted_arc(ISO17497_2_PREDICTION_FREQUENCY, flat=True)
    assert np.asarray(qrd.levels) == pytest.approx(
        np.asarray(ISO17497_2_QRD_LEVELS), abs=5.01e-4
    )
    assert np.asarray(flat.levels) == pytest.approx(
        np.asarray(ISO17497_2_FLAT_LEVELS), abs=5.01e-4
    )


def test_predicted_band_dn_matches_cox_appendix_b() -> None:
    # External anchor against published third-party data: Cox & D'Antonio,
    # "Acoustic Absorbers and Diffusers", 3rd ed., Appendix B (pp. 481-485),
    # section 7, row "N = 7 QRD, 6 periods, 0.2 m deep", normal incidence:
    # 2D BEM normalised diffusion coefficients in one-third-octave bands.
    # The library's Fraunhofer model reproduces the published 200-400 Hz
    # bands within 0.01 (asserted at +/-0.015). Low-band anchor only: across
    # the full published 100-5000 Hz range the model-vs-BEM mean absolute
    # deviation is ~0.09 (edge diffraction is outside the Fraunhofer model).
    from diffuser_prediction import predicted_band_normalized_diffusion
    from reference_data import (
        COX3E_APPENDIX_B_QRD_BANDS,
        COX3E_APPENDIX_B_QRD_DN,
        COX3E_APPENDIX_B_TOLERANCE,
    )

    for band, published in zip(
        COX3E_APPENDIX_B_QRD_BANDS, COX3E_APPENDIX_B_QRD_DN, strict=True
    ):
        predicted = predicted_band_normalized_diffusion(band)
        assert predicted == pytest.approx(published, abs=COX3E_APPENDIX_B_TOLERANCE), (
            f"{band} Hz band"
        )


def test_diffusion_qrd_arc_independent_energy_recompute() -> None:
    # Independent re-derivation of Formula (5) with plain energy arithmetic,
    # not calling the library kernel through any shared helper.
    levels = np.asarray(ISO17497_2_QRD_LEVELS, dtype=float)
    p = 10.0 ** (levels / 10.0)
    n = levels.size
    expected = (p.sum() ** 2 - (p**2).sum()) / ((n - 1) * (p**2).sum())
    assert expected == pytest.approx(ISO17497_2_QRD_DIFFUSION, abs=1e-6)


def test_diffusion_formula_6_reduces_to_5_for_uniform_weights() -> None:
    levels = [70.0, 74.0, 68.0, 72.0]
    d5 = directional_diffusion_coefficient(levels)
    d6 = directional_diffusion_coefficient(levels, area_weights=[1.0, 1.0, 1.0, 1.0])
    assert d5 == pytest.approx(d6)


def test_diffusion_formula_6_area_weighted_by_hand() -> None:
    levels = np.array([70.0, 74.0, 68.0, 72.0])
    weights = np.array([1.0, 1.5, 1.7, 1.0])
    p = 10.0 ** (levels / 10.0)
    num = (p * weights).sum() ** 2 - (weights * p**2).sum()
    den = (weights.sum() - 1.0) * (weights * p**2).sum()
    expected = num / den
    got = directional_diffusion_coefficient(levels, area_weights=weights)
    assert got == pytest.approx(expected)


def test_diffusion_one_with_weights_when_equal() -> None:
    levels = [65.0, 65.0, 65.0]
    weights = [1.0, 2.0, 3.0]
    assert directional_diffusion_coefficient(
        levels, area_weights=weights
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ISO 17497-2 normalisation (Formula (7)).
# ---------------------------------------------------------------------------
def test_normalization_maps_reference_to_zero() -> None:
    assert float(normalized_diffusion_coefficient(0.4, 0.4)) == pytest.approx(0.0)


def test_normalization_maps_one_to_one() -> None:
    assert float(normalized_diffusion_coefficient(1.0, 0.4)) == pytest.approx(1.0)


def test_normalization_formula() -> None:
    d, d_ref = 0.6, 0.2
    assert float(normalized_diffusion_coefficient(d, d_ref)) == pytest.approx(
        (d - d_ref) / (1.0 - d_ref)
    )


# ---------------------------------------------------------------------------
# ISO 17497-2 area factors (Formula (8)) - the radians convention.
# ---------------------------------------------------------------------------
def test_area_factors_zenith_uses_radians() -> None:
    # The theta = 0 form (4 pi / dphi) sin^2(dtheta / 4) needs radians.
    n = area_factors([0.0, 30.0, 60.0, 90.0], delta_theta=5.0, delta_phi=5.0)
    assert n[0] == pytest.approx(1.571045588794762)  # radians convention
    # sqrt(3) relationship between 60 deg and 30 deg receivers (physics).
    assert n[2] / n[1] == pytest.approx(math.sqrt(3.0))
    # Smallest factor is normalised to 1.
    assert n.min() == pytest.approx(1.0)


def test_area_factors_default_delta_phi_equals_delta_theta() -> None:
    n_default = area_factors([0.0, 45.0], delta_theta=5.0)
    n_explicit = area_factors([0.0, 45.0], delta_theta=5.0, delta_phi=5.0)
    assert np.allclose(n_default, n_explicit)


# ---------------------------------------------------------------------------
# ISO 17497-2 random-incidence average (Clause 8.4).
# ---------------------------------------------------------------------------
def test_random_incidence_equal_weight_is_mean() -> None:
    d = [0.2, 0.4, 0.6]
    assert random_incidence_diffusion(d) == pytest.approx(0.4)


def test_random_incidence_two_dimensional_weighting() -> None:
    # 0 deg weight 1, four other sources weight 3 each; total weight 13.
    d = [0.5, 0.2, 0.2, 0.2, 0.2]
    expected = (1 * 0.5 + 3 * (0.2 + 0.2 + 0.2 + 0.2)) / 13.0
    got = random_incidence_diffusion(d, weights=TWO_DIMENSIONAL_SOURCE_WEIGHTS)
    assert got == pytest.approx(expected)
    assert sum(TWO_DIMENSIONAL_SOURCE_WEIGHTS) == 13


# ---------------------------------------------------------------------------
# Input-validation guards.
# ---------------------------------------------------------------------------
def test_diffusion_requires_two_receivers() -> None:
    with pytest.raises(ValueError, match="'levels' needs at least two receivers"):
        directional_diffusion_coefficient([80.0])


def test_diffusion_weight_length_mismatch() -> None:
    with pytest.raises(
        ValueError, match="'area_weights' must match the number of receivers"
    ):
        directional_diffusion_coefficient([70.0, 72.0], area_weights=[1.0])


def test_reverberation_uncertainty_requires_two() -> None:
    with pytest.raises(ValueError, match="'times' needs at least two measurements"):
        reverberation_time_uncertainty([6.0])


def test_absorption_rejects_nonpositive_geometry() -> None:
    with pytest.raises(ValueError, match="'volume' must be a positive, finite number"):
        random_incidence_absorption(0.0, S, c1=C, t1=8.0, c2=C, t2=6.0)
    with pytest.raises(ValueError, match="'area' must be a positive, finite number"):
        random_incidence_absorption(V, -1.0, c1=C, t1=8.0, c2=C, t2=6.0)


def test_absorption_rejects_nonpositive_time_and_speed() -> None:
    with pytest.raises(ValueError, match="'T' values must be positive"):
        random_incidence_absorption(V, S, c1=C, t1=0.0, c2=C, t2=6.0)
    with pytest.raises(ValueError, match="'c' values must be positive"):
        random_incidence_absorption(V, S, c1=-1.0, t1=8.0, c2=C, t2=6.0)


def test_scattering_rejects_alpha_s_equal_one() -> None:
    with pytest.raises(ValueError, match="'alpha_s' must not equal 1"):
        scattering_coefficient(0.5, 1.0)


def test_normalization_rejects_reference_one() -> None:
    with pytest.raises(ValueError, match="'d_theta_reference' must not equal 1"):
        normalized_diffusion_coefficient(0.5, 1.0)


def test_area_factors_rejects_nonpositive_spacing() -> None:
    with pytest.raises(
        ValueError, match="'delta_theta' must be a positive, finite number"
    ):
        area_factors([0.0, 30.0], delta_theta=0.0)


def test_area_factors_rejects_empty_elevations() -> None:
    with pytest.raises(
        ValueError, match="'elevations' must be a non-empty 1-D sequence"
    ):
        area_factors([], delta_theta=5.0)


def test_diffusion_coefficient_rejects_zero_energy() -> None:
    # All -inf levels means zero energy everywhere; the coefficient is undefined.
    with pytest.raises(ValueError, match="The polar response carries no energy"):
        directional_diffusion_coefficient([float("-inf"), float("-inf")])


def test_base_plate_checker_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="values for bands"):
        check_base_plate_scattering([0.1, 0.2, 0.3])


def test_base_plate_checker_rejects_missing_band() -> None:
    incomplete = {b: 0.0 for b in BASE_PLATE_BANDS if b != 500}
    with pytest.raises(ValueError, match="missing band 500 Hz"):
        check_base_plate_scattering(incomplete)


# ---------------------------------------------------------------------------
# Module surface (package __init__ wiring is done separately).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Plottable result objects: scattering_coefficient_spectrum / directional_diffusion.
# ---------------------------------------------------------------------------
def test_scattering_spectrum_recomputes_s_per_band() -> None:
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0])
    alpha_spec = np.array([0.12, 0.25, 0.40, 0.60, 0.80])
    alpha_s = np.array([0.10, 0.11, 0.12, 0.13, 0.14])
    result = scattering_coefficient_spectrum(freqs, alpha_spec, alpha_s)

    # Independent re-derivation of Eq. (5) per band.
    expected = (alpha_spec - alpha_s) / (1.0 - alpha_s)
    assert isinstance(result, ScatteringResult)
    np.testing.assert_allclose(result.scattering, expected)
    np.testing.assert_allclose(result.frequencies, freqs)
    np.testing.assert_allclose(result.specular, alpha_spec)
    np.testing.assert_allclose(result.random_incidence, alpha_s)


def test_scattering_spectrum_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="non-empty, 1-D and equal-length"):
        scattering_coefficient_spectrum([250.0, 500.0], [0.2], [0.1])


def test_scattering_spectrum_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty, 1-D and equal-length"):
        scattering_coefficient_spectrum([], [], [])


def test_scattering_spectrum_rejects_2d_input() -> None:
    # frequencies is documented 1-D; equal-shaped 2-D arrays must be rejected.
    two_d = [[250.0, 500.0], [1000.0, 2000.0]]
    with pytest.raises(ValueError, match="non-empty, 1-D and equal-length"):
        scattering_coefficient_spectrum(two_d, two_d, two_d)


def test_scattering_spectrum_plot_returns_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = scattering_coefficient_spectrum(
        [250.0, 500.0, 1000.0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.1]
    )
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_directional_diffusion_coefficient_matches_scalar() -> None:
    angles = np.arange(-90.0, 90.5, 5.0)
    rng = np.random.default_rng(3)
    levels = (
        70.0
        + 2.0 * np.sin(np.radians(angles) * 3.0)
        + rng.normal(0.0, 1.0, angles.size)
    )
    result = directional_diffusion(angles, levels)

    assert isinstance(result, DiffusionResult)
    assert result.coefficient == pytest.approx(
        directional_diffusion_coefficient(levels)
    )
    np.testing.assert_allclose(result.angles, angles)
    np.testing.assert_allclose(result.levels, levels)


def test_directional_diffusion_length_mismatch_raises() -> None:
    with pytest.raises(
        ValueError, match="'angles' and 'levels' must have the same length"
    ):
        directional_diffusion([-30.0, 0.0, 30.0], [70.0, 72.0])


def test_directional_diffusion_plot_returns_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = directional_diffusion([-30.0, 0.0, 30.0], [70.0, 72.0, 69.0])
    ax = result.plot()
    assert ax.name == "polar"
    plt.close("all")


def test_diffusion_spectrum_builds_and_carries_fields() -> None:
    freqs = [250.0, 500.0, 1000.0]
    d = [0.3, 0.5, 0.7]
    d_n = [0.2, 0.4, 0.6]
    result = diffusion_spectrum(freqs, d, normalized=d_n)
    assert isinstance(result, DiffusionSpectrum)
    np.testing.assert_allclose(result.frequencies, freqs)
    np.testing.assert_allclose(result.diffusion, d)
    np.testing.assert_allclose(result.normalized, d_n)


def test_diffusion_spectrum_optional_fields_default_none() -> None:
    result = diffusion_spectrum([250.0, 500.0], [0.3, 0.5])
    assert result.normalized is None


def test_diffusion_spectrum_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="non-empty, 1-D and equal-length"):
        diffusion_spectrum([250.0, 500.0], [0.3])


def test_diffusion_spectrum_normalized_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="'normalized' must match"):
        diffusion_spectrum([250.0, 500.0], [0.3, 0.5], normalized=[0.2])


def test_diffusion_spectrum_plot_returns_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = diffusion_spectrum(
        [250.0, 500.0, 1000.0], [0.3, 0.5, 0.7], normalized=[0.2, 0.4, 0.6]
    )
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_public_names_in_module_all() -> None:
    import phonometry.materials.diffusers.reverberation_room_scattering as part1
    import phonometry.materials.diffusers.scattering_diffusion as part2

    for name in (
        "random_incidence_absorption",
        "specular_absorption_coefficient",
        "scattering_coefficient",
        "base_plate_scattering",
        "BASE_PLATE_MAX_SCATTERING",
        "ScatteringDiffusionWarning",
    ):
        assert name in part1.__all__
    for name in (
        "directional_diffusion_coefficient",
        "normalized_diffusion_coefficient",
        "area_factors",
        "random_incidence_diffusion",
    ):
        assert name in part2.__all__


def test_public_exports() -> None:
    import phonometry.materials.diffusers.reverberation_room_scattering as part1
    import phonometry.materials.diffusers.scattering_diffusion as part2
    from phonometry import materials

    for name in (*part1.__all__, *part2.__all__):
        assert hasattr(materials, name), name
