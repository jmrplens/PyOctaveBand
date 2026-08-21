#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power by sound-intensity scanning: ISO 9614-2:1996.

Physics / standard anchors:
- Partial power Pi = <In,i>*Si (Eq. 5/12); total P = sum Pi (Eq. 6);
  LW = 10*lg(P/P0), P0 = 1e-12 W (Eq. 13). A source of true power W enclosed by
  segments with sum(In,i*Si) = W recovers LW = 10*lg(W/P0) exactly.
- FpI = [Lp] - LW + 10*lg(S/S0) (Eq. A.1); F+/- = 10*lg(sum|Pi|/|sum Pi|)
  (Eq. A.2).
- Criteria: 1) Ld > FpI, Ld = dpI0 - K (K = 10 eng / 7 survey, Table 1);
  2) F+/- <= 3 dB (grade 2); 3) |LWi(1)-LWi(2)| <= s (Table 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import emission

_P0 = 1.0e-12


# --------------------------------------------------------------------------
# Exact power recovery over an enclosing surface (Eq. 12-13)
# --------------------------------------------------------------------------
def test_uniform_enclosed_source_recovers_power_exactly() -> None:
    """Uniform In = W/S over 4 equal segments recovers LW = 10*lg(W/P0)."""
    w = 1.0e-3  # 90 dB
    areas = np.array([0.5, 0.5, 0.5, 0.5])  # S = 2 m^2
    s_total = areas.sum()
    intensity = np.full((4, 1), w / s_total)  # (N_seg, 1 band)
    res = emission.sound_power_intensity(intensity, areas)
    assert isinstance(res, emission.SoundPowerIntensityResult)
    assert res.sound_power[0] == pytest.approx(w)
    assert res.sound_power_level[0] == pytest.approx(90.0)
    assert res.surface_area == pytest.approx(2.0)
    assert not res.negative_band[0]


def test_non_uniform_segmentation_recovers_power() -> None:
    """Different segment areas/intensities but sum(In*Si) = W recovers LW."""
    areas = np.array([0.2, 0.3, 0.5, 1.0])  # S = 2 m^2
    w = 5.0e-4
    # In,i chosen so partial powers are 1e-4, 1e-4, 1.5e-4, 1.5e-4 -> sum 5e-4.
    partial = np.array([1.0e-4, 1.0e-4, 1.5e-4, 1.5e-4])
    intensity = (partial / areas).reshape(4, 1)
    res = emission.sound_power_intensity(intensity, areas)
    assert res.sound_power[0] == pytest.approx(w)
    assert res.sound_power_level[0] == pytest.approx(10.0 * np.log10(w / _P0))
    np.testing.assert_allclose(res.partial_power[:, 0], partial)


def test_multiband_independent_recovery() -> None:
    """Two bands are handled independently."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    # band 0: uniform 5e-4 -> P 1e-3 (90 dB); band 1: uniform 5e-5 -> P 1e-4 (80).
    intensity = np.column_stack([np.full(4, 5.0e-4), np.full(4, 5.0e-5)])
    res = emission.sound_power_intensity(intensity, areas)
    np.testing.assert_allclose(res.sound_power, [1.0e-3, 1.0e-4])
    np.testing.assert_allclose(res.sound_power_level, [90.0, 80.0])


# --------------------------------------------------------------------------
# Negative partial power (external source) -> indicator + warning (Eq. A.2)
# --------------------------------------------------------------------------
def test_negative_partial_power_indicator_and_warning() -> None:
    """One inward-flowing segment: F+/- > 0, criterion 2 fails, warn."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    # partial powers 2.5e-4 *[1,1,1,-2]: sum positive, strong negative flow.
    intensity = (np.array([1.0, 1.0, 1.0, -2.0]) * 5.0e-4).reshape(4, 1)
    with pytest.warns(emission.SoundPowerWarning):
        res = emission.sound_power_intensity(intensity, areas)
    pi = intensity[:, 0] * areas
    expected = 10.0 * np.log10(np.sum(np.abs(pi)) / abs(np.sum(pi)))
    assert res.negative_partial_power_index[0] == pytest.approx(expected)
    assert res.negative_partial_power_index[0] > 3.0  # criterion 2 fails
    # Total still positive -> band determinable.
    assert res.sound_power[0] > 0.0
    assert not res.negative_band[0]


def test_small_negative_partial_power_no_warning() -> None:
    """Modest inward flow keeping F+/- <= 3 dB does not warn."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = (np.array([1.0, 1.0, 1.0, -0.5]) * 1.0e-3).reshape(4, 1)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = emission.sound_power_intensity(intensity, areas)
    assert res.negative_partial_power_index[0] <= 3.0


def test_f_plus_minus_warning_suppressed_under_survey_grade() -> None:
    """Criterion 2 is optional for the survey grade (B.1.2): F+/- > 3 dB must
    not emit the 'engineering grade not achieved' warning when grade='survey',
    while the engineering grade still warns for the same data.
    """
    import warnings

    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = (np.array([1.0, 1.0, 1.0, -2.0]) * 5.0e-4).reshape(4, 1)
    # Survey grade: no F+/- warning (total power still positive -> no other
    # warning source either).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = emission.sound_power_intensity(intensity, areas, grade="survey")
    assert res.negative_partial_power_index[0] > 3.0
    # Engineering grade: the same data still warns.
    with pytest.warns(emission.SoundPowerWarning):
        emission.sound_power_intensity(intensity, areas, grade="engineering")


def test_negative_total_power_band_not_determinable() -> None:
    """Sum Pi < 0: method not applicable to the band (clause 9.2), warn, NaN."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = (np.array([1.0, 1.0, 1.0, -4.0]) * 5.0e-4).reshape(4, 1)
    with pytest.warns(emission.SoundPowerWarning):
        res = emission.sound_power_intensity(intensity, areas)
    assert res.sound_power[0] < 0.0
    assert res.negative_band[0]
    assert np.isnan(res.sound_power_level[0])


# --------------------------------------------------------------------------
# Field indicators FpI (Eq. A.1)
# --------------------------------------------------------------------------
def test_fpi_matches_definition() -> None:
    """FpI = [Lp] - LW + 10*lg(S/S0) with area-weighted [Lp]."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = np.full((4, 1), 5.0e-4)  # P = 1e-3 -> LW = 90
    lp = np.full((4, 1), 95.0)
    res = emission.sound_power_intensity(intensity, areas, pressure_levels=lp)
    s_total = areas.sum()
    lp_surface = 10.0 * np.log10(np.sum(areas * 10.0 ** (0.1 * lp[:, 0])) / s_total)
    expected = lp_surface - 90.0 + 10.0 * np.log10(s_total)
    assert res.surface_pressure_intensity_index[0] == pytest.approx(expected)


def test_fpi_none_without_pressure_levels() -> None:
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = np.full((4, 1), 5.0e-4)
    res = emission.sound_power_intensity(intensity, areas)
    assert res.surface_pressure_intensity_index is None


# --------------------------------------------------------------------------
# Class assignment per band (Annex B criteria; clause 8.4)
# --------------------------------------------------------------------------
def _base_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4 equal segments, uniform In (P=1e-3, LW=90), [Lp]=95 -> FpI≈8.0103."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = np.full((4, 1), 5.0e-4)
    lp = np.full((4, 1), 95.0)
    return areas, intensity, lp


def test_class_engineering_when_all_criteria_pass() -> None:
    areas, intensity, lp = _base_case()
    # FpI ≈ 8.0103; dpI0 = 20 -> Ld_eng = 10 > FpI, Ld_survey = 13 > FpI.
    res = emission.sound_power_intensity(
        intensity,
        areas,
        normal_intensity_2=intensity,  # identical scans -> repeatability 0
        pressure_levels=lp,
        pressure_residual_index=20.0,
        frequencies=np.array([1000.0]),
        band_type="octave",
    )
    assert res.achieved_grade[0] == "engineering"


def test_class_survey_when_c1_fails_for_engineering() -> None:
    areas, intensity, lp = _base_case()
    # dpI0 = 17 -> Ld_eng = 7 < 8.0103 (fail), Ld_survey = 10 > 8.0103 (pass).
    res = emission.sound_power_intensity(
        intensity,
        areas,
        normal_intensity_2=intensity,
        pressure_levels=lp,
        pressure_residual_index=17.0,
        frequencies=np.array([1000.0]),
        band_type="octave",
    )
    assert res.achieved_grade[0] == "survey"


def test_class_none_when_c1_fails_for_both() -> None:
    areas, intensity, lp = _base_case()
    # dpI0 = 14 -> Ld_survey = 7 < 8.0103 (fail).
    res = emission.sound_power_intensity(
        intensity,
        areas,
        normal_intensity_2=intensity,
        pressure_levels=lp,
        pressure_residual_index=14.0,
        frequencies=np.array([1000.0]),
        band_type="octave",
    )
    assert res.achieved_grade[0] == "none"


def test_class_survey_when_c2_fails_only() -> None:
    """F+/- > 3 dB fails engineering (c2) but survey ignores c2 -> survey."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    # partial powers *[1,1,1,-1.2916]*1e-3 -> F+/- ≈ 4 dB, sum positive.
    intensity = (np.array([1.0, 1.0, 1.0, -1.2916]) * 1.0e-3).reshape(4, 1)
    pi = intensity[:, 0] * areas
    lw = 10.0 * np.log10(np.sum(pi) / _P0)
    lp = np.full((4, 1), lw)  # -> FpI ≈ 10*lg(S) ≈ 3.0103, small
    with pytest.warns(emission.SoundPowerWarning):
        res = emission.sound_power_intensity(
            intensity,
            areas,
            normal_intensity_2=intensity,
            pressure_levels=lp,
            pressure_residual_index=30.0,  # Ld large, c1 passes both
            frequencies=np.array([1000.0]),
            band_type="octave",
        )
    assert res.negative_partial_power_index[0] > 3.0
    assert res.achieved_grade[0] == "survey"


def test_class_downgrade_by_repeatability() -> None:
    """|LWi(1)-LWi(2)| = 3 dB: fails eng s=1.5 but passes survey s=4."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    scan1 = np.full((4, 1), 5.0e-4)
    # LWi(1) = 10*lg(2.5e-4/P0); choose scan2 so each |dLWi| = 3 dB.
    lwi1 = 10.0 * np.log10(2.5e-4 / _P0)
    pi2 = _P0 * 10.0 ** (0.1 * (lwi1 - 3.0))
    scan2 = np.full((4, 1), pi2 / 0.5)
    lp = np.full((4, 1), 90.0)
    res = emission.sound_power_intensity(
        scan1,
        areas,
        normal_intensity_2=scan2,
        pressure_levels=lp,
        pressure_residual_index=40.0,
        frequencies=np.array([1000.0]),
        band_type="octave",
    )
    np.testing.assert_allclose(res.repeatability[:, 0], 3.0, atol=1e-9)
    assert res.achieved_grade[0] == "survey"


def test_sweep_reversal_fails_criterion_3() -> None:
    """An equal-magnitude opposite-sign segment between the two sweeps has
    |ΔL| ~ 0 from the magnitudes alone, yet is grossly non-repeatable. With the
    band still determinable (positive total power) and criteria 1/2 satisfied,
    the reversed segment must force criterion 3 (and thus the grade) to fail.
    """
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    scan1 = np.full((4, 1), 5.0e-4)
    scan2 = scan1.copy()
    scan2[0, 0] = -5.0e-4  # equal magnitude, opposite sign on segment 0
    lp = np.full((4, 1), 90.0)
    res = emission.sound_power_intensity(
        scan1,
        areas,
        normal_intensity_2=scan2,
        pressure_levels=lp,
        pressure_residual_index=40.0,  # Ld high -> criterion 1 passes
        frequencies=np.array([1000.0]),
        band_type="octave",
    )
    assert not res.negative_band[0]  # band remains determinable
    assert np.isinf(res.repeatability[0, 0])
    # criterion 3 fails on segment 0 -> neither engineering nor survey qualifies.
    assert res.achieved_grade[0] == "none"
    # Without the sign guard, the magnitude-only |ΔL| would have been 0 and the
    # band would have qualified; confirm an all-aligned pair does qualify.
    ok = emission.sound_power_intensity(
        scan1,
        areas,
        normal_intensity_2=scan1,
        pressure_levels=lp,
        pressure_residual_index=40.0,
        frequencies=np.array([1000.0]),
        band_type="octave",
    )
    assert ok.achieved_grade[0] == "engineering"


def test_partial_sweep_reversal_only_affects_flipped_segment() -> None:
    """Only the segment whose flow reverses gets +inf repeatability; the aligned
    segments keep their finite |ΔL|.
    """
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    scan1 = np.full((4, 1), 5.0e-4)
    scan2 = scan1.copy()
    scan2[2, 0] = -5.0e-4  # reverse only segment 2
    res = emission.sound_power_intensity(scan1, areas, normal_intensity_2=scan2)
    rep = res.repeatability[:, 0]
    assert np.isinf(rep[2])
    assert np.allclose(rep[[0, 1, 3]], 0.0, atol=1e-9)


def test_exact_zero_partial_power_does_not_trigger_reversal() -> None:
    """An exact-zero partial power carries no direction and must not be treated
    as a reversal (no spurious +inf repeatability).
    """
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    scan1 = np.full((4, 1), 5.0e-4)
    scan2 = scan1.copy()
    scan2[1, 0] = 0.0  # zero, not a sign flip
    res = emission.sound_power_intensity(scan1, areas, normal_intensity_2=scan2)
    assert np.all(np.isfinite(res.repeatability[:, 0]))


def test_short_frequencies_raises_value_error_not_index_error() -> None:
    """A 'frequencies' array shorter than the band count raises the public
    ValueError up front, not an IndexError during classification.
    """
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = np.column_stack([np.full(4, 5.0e-4), np.full(4, 5.0e-4)])
    levels = np.full((4, 2), 90.0)
    short_frequencies = np.array([1000.0])  # one freq, two bands
    with pytest.raises(ValueError, match="'frequencies' length must match"):
        emission.sound_power_intensity(
            intensity,
            areas,
            normal_intensity_2=intensity,
            pressure_levels=levels,
            pressure_residual_index=40.0,
            frequencies=short_frequencies,
        )


def test_repeatability_uses_mean_of_two_scans_for_power() -> None:
    """Partial power uses the mean of the two scan intensities (Eq. 12)."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    scan1 = np.full((4, 1), 6.0e-4)
    scan2 = np.full((4, 1), 4.0e-4)
    res = emission.sound_power_intensity(scan1, areas, normal_intensity_2=scan2)
    # mean In = 5e-4 -> P = 1e-3.
    assert res.sound_power[0] == pytest.approx(1.0e-3)


def test_achieved_grade_none_without_inputs() -> None:
    """Without dpI0/two scans the achieved grade cannot be determined."""
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = np.full((4, 1), 5.0e-4)
    res = emission.sound_power_intensity(intensity, areas)
    assert res.achieved_grade is None


# --------------------------------------------------------------------------
# A-weighted total (bands with negative total excluded)
# --------------------------------------------------------------------------
def test_a_weighted_total_excludes_non_determinable_band() -> None:
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    # band0 determinable (P=1e-3), band1 negative total (excluded).
    intensity = np.column_stack(
        [np.full(4, 5.0e-4), np.array([1.0, 1.0, 1.0, -4.0]) * 5.0e-5]
    )
    with pytest.warns(emission.SoundPowerWarning):
        res = emission.sound_power_intensity(
            intensity,
            areas,
            frequencies=np.array([1000.0, 2000.0]),
            band_type="octave",
        )
    # Only the 1000 Hz band contributes; Ck(1000)=0 -> LWA = 90.
    assert res.sound_power_level_a == pytest.approx(90.0)


def test_single_band_a_weighted_equals_lw() -> None:
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = np.full((4, 1), 5.0e-4)
    res = emission.sound_power_intensity(intensity, areas)
    assert res.sound_power_level_a == pytest.approx(90.0)


# --------------------------------------------------------------------------
# A-weighted total omits bands failing criteria 1 and/or 2 (clause 10.6 b)
# --------------------------------------------------------------------------
def test_a_weighted_total_omits_band_failing_criterion_1() -> None:
    """Clause 10.6 b: a band with Ld <= FpI (criterion 1 fails) is omitted
    from the A-weighted determination.

    Hand oracle: four 1 m^2 segments (S = 4 m^2), uniform In = 1e-3 W/m^2 at
    500 Hz (P = 4e-3 W, LW = 96,0206 dB) and 5e-4 W/m^2 at 1 kHz
    (LW = 93,0103 dB). With dpI0 = 16 dB (Ld = 6 dB) and surface SPLs of
    92 dB / 95 dB, FpI = 2,00 dB at 500 Hz (passes) and 8,01 dB at 1 kHz
    (fails). LWA keeps only 500 Hz: 96,0206 - 3,2 = 92,8206 dB, not the
    two-band 95,9268 dB.
    """
    areas = np.ones(4)
    intensity = np.column_stack([np.full(4, 1.0e-3), np.full(4, 5.0e-4)])
    res = emission.sound_power_intensity(
        intensity,
        areas,
        pressure_levels=np.column_stack([np.full(4, 92.0), np.full(4, 95.0)]),
        pressure_residual_index=16.0,
        frequencies=np.array([500.0, 1000.0]),
    )
    assert res.a_weighting_omitted_bands is not None
    assert res.a_weighting_omitted_bands.tolist() == [False, True]
    assert res.sound_power_level_a == pytest.approx(92.8206, abs=1e-4)
    # The per-band levels themselves stay reported (only LWA omits the band).
    assert res.sound_power_level[1] == pytest.approx(93.0103, abs=1e-4)


def test_a_weighted_total_omits_band_failing_criterion_2_engineering() -> None:
    """Clause 10.6 b: for an engineering determination a band with
    F+/- > 3 dB (criterion 2 fails) is omitted from LWA.

    Hand oracle: band 2 partial powers 5/-1/1/-1 mW give P = 4 mW
    (LW = 96,0206 dB) with F+/- = 10*lg(8/4) = 3,0103 dB > 3 dB; band 1 is
    uniform 1 mW per segment (same LW, F+/- = 0). LWA keeps only band 1:
    96,0206 - 3,2 = 92,8206 dB.
    """
    areas = np.ones(4)
    intensity = np.column_stack(
        [np.full(4, 1.0e-3), np.array([5.0e-3, -1.0e-3, 1.0e-3, -1.0e-3])]
    )
    lp = np.full((4, 2), 92.0)  # FpI ~ 2 dB in both bands: criterion 1 passes
    with pytest.warns(emission.SoundPowerWarning, match="criterion 2"):
        res = emission.sound_power_intensity(
            intensity,
            areas,
            pressure_levels=lp,
            pressure_residual_index=16.0,
            frequencies=np.array([500.0, 1000.0]),
        )
    assert res.a_weighting_omitted_bands is not None
    assert res.a_weighting_omitted_bands.tolist() == [False, True]
    assert res.sound_power_level_a == pytest.approx(92.8206, abs=1e-4)


def test_survey_grade_does_not_omit_on_criterion_2() -> None:
    """Criterion 2 is optional for grade 3 (Note 16), so a survey
    determination keeps a band with F+/- > 3 dB in LWA:
    10*lg(10^9,28206 + 10^9,60206) = 97,7192 dB.
    """
    areas = np.ones(4)
    intensity = np.column_stack(
        [np.full(4, 1.0e-3), np.array([5.0e-3, -1.0e-3, 1.0e-3, -1.0e-3])]
    )
    lp = np.full((4, 2), 92.0)
    res = emission.sound_power_intensity(
        intensity,
        areas,
        pressure_levels=lp,
        pressure_residual_index=16.0,
        frequencies=np.array([500.0, 1000.0]),
        grade="survey",
    )
    assert res.a_weighting_omitted_bands is not None
    assert res.a_weighting_omitted_bands.tolist() == [False, False]
    assert res.sound_power_level_a == pytest.approx(97.719195, abs=1e-4)


def test_a_weighting_screening_unavailable_warns_and_sums_all() -> None:
    """Without FpI/Ld the clause 10.6 b screening cannot run: every
    determinable band is summed and a warning notes the missing screening.
    """
    areas = np.ones(4)
    intensity = np.column_stack([np.full(4, 1.0e-3), np.full(4, 5.0e-4)])
    with pytest.warns(emission.SoundPowerWarning, match="10.6 b"):
        res = emission.sound_power_intensity(
            intensity, areas, frequencies=np.array([500.0, 1000.0])
        )
    assert res.a_weighting_omitted_bands is None
    assert res.sound_power_level_a == pytest.approx(95.9268, abs=1e-4)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def test_area_length_mismatch_raises() -> None:
    intensity = np.full((4, 1), 1e-4)
    three_areas = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="'normal_intensity' first axis"):
        emission.sound_power_intensity(intensity, three_areas)


def test_pressure_levels_shape_mismatch_raises() -> None:
    intensity = np.full((4, 1), 1e-4)
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    two_band_levels = np.full((4, 2), 90.0)
    with pytest.raises(ValueError, match="'pressure_levels' must have shape"):
        emission.sound_power_intensity(
            intensity, areas, pressure_levels=two_band_levels
        )


def test_non_positive_area_raises() -> None:
    intensity = np.full((4, 1), 1e-4)
    zero_area = np.array([0.5, 0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="segment 'areas' must be positive"):
        emission.sound_power_intensity(intensity, zero_area)


def test_fewer_than_four_segments_warns() -> None:
    """Clause 8.2 requires at least 4 segments."""
    intensity = np.full((3, 1), 5e-4)
    areas = np.array([0.5, 0.5, 0.5])
    with pytest.warns(emission.SoundPowerWarning):
        emission.sound_power_intensity(intensity, areas)


def test_bad_grade_raises() -> None:
    intensity = np.full((4, 1), 1e-4)
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="'grade' must be"):
        emission.sound_power_intensity(intensity, areas, grade="bogus")


# --------------------------------------------------------------------------
# Quantities that do not run over the band axis
# --------------------------------------------------------------------------
def _five_band_determination() -> object:
    """A four-segment, five-band determination with every indicator populated."""
    intensity = np.tile(np.array([6e-4, 5e-4, 4e-4, 3e-4, 2e-4]), (4, 1))
    return emission.sound_power_intensity(
        intensity,
        np.full(4, 0.5),
        frequencies=np.array([125.0, 250.0, 500.0, 1000.0, 2000.0]),
        normal_intensity_2=intensity * 0.9,
        pressure_levels=np.tile(np.full(5, 80.0), (4, 1)),
        pressure_residual_index=12.0,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "sound_power_level",
        "negative_band",
        "surface_pressure_intensity_index",
        "negative_partial_power_index",
        "dynamic_capability_index",
        "a_weighting_omitted_bands",
    ],
)
@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_a_per_band_quantity_off_the_band_axis_is_refused(
    field_name: str, trim: bool
) -> None:
    """Every per-band quantity is pinned to the band axis at construction.

    The field indicators reach the fiche as a range rather than as a column,
    so an array of the wrong length prints a range taken over the wrong set of
    bands and nothing downstream can object: a range is well formed whatever
    it was taken over. That is why the long direction matters as much as the
    short one.
    """
    import dataclasses

    result = _five_band_determination()
    values = np.asarray(getattr(result, field_name))
    wrong = values[:-1] if trim else np.append(values, values[-1])
    with pytest.raises(ValueError, match=f"'{field_name}'"):
        dataclasses.replace(result, **{field_name: wrong})


def test_a_per_segment_quantity_off_the_segment_axis_is_refused() -> None:
    """The segment axis is its own axis: repeatability is per segment and band."""
    import dataclasses

    result = _five_band_determination()
    with pytest.raises(ValueError, match="measurement segment"):
        dataclasses.replace(result, repeatability=result.repeatability[:-1])
