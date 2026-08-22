#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Precision-grade sound power: ISO 3745:2012 (anechoic / hemi-anechoic) and
ISO 9614-3:2002 (sound-intensity scanning, precision).

Physics / standard anchors:
- ISO 3745 K1 (Eq. 11): dLpi = 6 dB -> 1,256 dB; dLpi = 10 dB -> 0,458 dB.
- ISO 3745 areas: sphere S1 = 4*pi*r^2, hemisphere S2 = 2*pi*r^2.
- ISO 3745 C1/C2 (Eq. 14): at 23 deg C, 101,325 kPa -> C2 = 0, C1 = -0,128 dB.
- ISO 3745 uncertainty (Eq. 24/25): U = k*sqrt(sigma_R0^2 + sigma_omc^2).
- ISO 9614-3 LW = 10*lg(P/P0) (Eq. 9); LW0 = LW at 23 deg C / 101 325 Pa (Eq. 10).
- ISO 9614-3 Annex B/C field indicators and five acceptance criteria.
"""

import numpy as np
import pytest

from phonometry.emission.sound_power import SoundPowerWarning
from phonometry.emission.sound_power_anechoic import (
    _TABLE_D1,
    _TABLE_E1,
    _TABLE_E2,
    MeteorologicalCorrection,
    PrecisionSoundPowerResult,
    meteorological_corrections,
    precision_background_correction,
    precision_positions,
    precision_uncertainty,
    sound_power_anechoic,
)
from phonometry.emission.sound_power_intensity import (
    PrecisionCriteria,
    PrecisionFieldIndicators,
    PrecisionIntensityResult,
    precision_field_indicators,
    precision_qualification,
    sound_power_intensity_precision,
)

_P0 = 1.0e-12


# ==========================================================================
# ISO 3745 - coordinate tables (Annex D/E), digit-exact and unit-norm
# ==========================================================================
@pytest.mark.parametrize("table", [_TABLE_D1, _TABLE_E1, _TABLE_E2])
def test_precision_tables_are_unit_vectors_to_3dp(table: np.ndarray) -> None:
    """Every (x/r,y/r,z/r) row of D.1/E.1/E.2 is a unit vector to 3 decimals."""
    assert table.shape == (40, 3)
    norms = np.linalg.norm(table, axis=1)
    assert np.all(np.abs(norms - 1.0) < 2.0e-3)


def test_precision_positions_sphere_shape_and_scale() -> None:
    """Sphere returns Table D.1 scaled by r; full array is 40 positions."""
    pos = precision_positions("sphere", radius=2.0)
    assert pos.shape == (40, 3)
    assert np.allclose(np.linalg.norm(pos, axis=1), 2.0, atol=2.0 * 2.0e-3)
    # Position 2 of D.1 is exactly (0,494, -0,856, 0,150)*r.
    assert np.allclose(pos[1], np.array([0.494, -0.856, 0.150]) * 2.0)


def test_precision_positions_primary_array_is_20() -> None:
    """count=20 returns the primary array (positions 1-20)."""
    pos = precision_positions("hemisphere", radius=1.0, count=20)
    assert pos.shape == (20, 3)


def test_precision_positions_hemisphere_general_pos7_z_is_0320() -> None:
    """Table E.1 pos 7 z/r = 0,320 (not 0,325) - do not regularise it."""
    pos = precision_positions("hemisphere", radius=1.0, array="general")
    assert pos[6, 2] == pytest.approx(0.320)
    assert np.allclose(pos[6], [0.000, 0.947, 0.320])


def test_precision_positions_hemisphere_broadband_pos19_x_is_neg0380() -> None:
    """Table E.2 pos 19 x/r = -0,380 (a normal negative)."""
    pos = precision_positions("hemisphere", radius=1.0, array="broadband")
    assert pos[18, 0] == pytest.approx(-0.380)


def test_precision_positions_hemisphere_z_nonnegative() -> None:
    """Hemisphere coordinates all sit on/above the reflecting plane z = 0."""
    for arr in ("general", "broadband"):
        pos = precision_positions("hemisphere", radius=3.0, array=arr)
        assert np.all(pos[:, 2] >= 0.0)


def test_precision_positions_invalid_surface_raises() -> None:
    with pytest.raises(ValueError, match="'surface' must be 'sphere' or 'hemisphere'"):
        precision_positions("box", radius=1.0)  # type: ignore[arg-type]


def test_precision_positions_requires_a_radius() -> None:
    """Keyword-only and required, so Python enforces it before the body runs."""
    with pytest.raises(TypeError, match="required keyword-only argument"):
        precision_positions("sphere")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="A positive 'radius' is required"):
        precision_positions("sphere", radius=0.0)


def test_precision_positions_bad_count_raises() -> None:
    with pytest.raises(ValueError, match="'count' must be 20"):
        precision_positions("sphere", radius=1.0, count=30)


# ==========================================================================
# ISO 3745 - per-position background correction K1i (Eq. 11) with floors
# ==========================================================================
def test_k1_floor_edge_band_6db() -> None:
    """dLpi = 6 dB in a <=200 Hz band -> K1 = -10*lg(1-10^-0,6) = 1,256 dB."""
    k1 = precision_background_correction(
        np.array([[56.0]]), np.array([[50.0]]), np.array([200.0])
    )
    assert k1[0, 0] == pytest.approx(1.25628, abs=1e-4)


def test_k1_floor_mid_band_10db() -> None:
    """dLpi = 10 dB in a 250-5000 Hz band -> K1 = -10*lg(1-10^-1) = 0,458 dB."""
    k1 = precision_background_correction(
        np.array([[60.0]]), np.array([[50.0]]), np.array([1000.0])
    )
    assert k1[0, 0] == pytest.approx(0.45757, abs=1e-4)


def test_k1_zero_above_15db() -> None:
    """dLpi >= 15 dB -> background negligible -> K1 = 0."""
    k1 = precision_background_correction(
        np.array([[80.0]]), np.array([[60.0]]), np.array([1000.0])
    )
    assert k1[0, 0] == 0.0


def test_k1_edge_band_below_6db_clamps_and_warns() -> None:
    """dLpi < 6 dB in an edge band clamps to the 6 dB value (1,26 dB), warns."""
    with pytest.warns(SoundPowerWarning):
        k1 = precision_background_correction(
            np.array([[53.0]]), np.array([[50.0]]), np.array([6300.0])
        )
    assert k1[0, 0] == pytest.approx(1.25628, abs=1e-4)


def test_k1_mid_band_below_10db_clamps_to_0_46() -> None:
    """dLpi < 10 dB in a mid band clamps to the 10 dB value (0,46 dB), warns."""
    with pytest.warns(SoundPowerWarning):
        k1 = precision_background_correction(
            np.array([[57.0]]), np.array([[50.0]]), np.array([1000.0])
        )
    assert k1[0, 0] == pytest.approx(0.45757, abs=1e-4)


def test_k1_is_per_position() -> None:
    """K1 is applied per microphone position, not to the energy mean."""
    src = np.array([[80.0, 63.0], [70.0, 62.0]])  # 2 positions, 2 bands
    bg = np.array([[60.0, 55.0], [60.0, 55.0]])  # band-1 dLpi = 8/7 dB (>= 6)
    k1 = precision_background_correction(src, bg, np.array([1000.0, 200.0]))
    assert k1.shape == (2, 2)
    # position 0 band 0 dLpi = 20 dB -> 0; position 1 band 0 dLpi = 10 dB -> 0,458
    assert k1[0, 0] == 0.0
    assert k1[1, 0] == pytest.approx(0.45757, abs=1e-4)


def test_k1_frequency_length_mismatch_raises() -> None:
    source_2_bands = np.array([[56.0, 57.0]])
    background_2_bands = np.array([[50.0, 50.0]])
    one_frequency = np.array([200.0])
    with pytest.raises(ValueError, match="must match the number of 'frequencies'"):
        precision_background_correction(
            source_2_bands, background_2_bands, one_frequency
        )


# ==========================================================================
# ISO 3745 - meteorological corrections C1, C2, C3 (Eq. 14)
# ==========================================================================
def test_c2_zero_at_reference() -> None:
    """At 23 deg C, 101,325 kPa the ratio (273+theta)/theta1 = 1 -> C2 = 0."""
    mc = meteorological_corrections(23.0, 101.325)
    assert isinstance(mc, MeteorologicalCorrection)
    assert mc.c2 == pytest.approx(0.0, abs=1e-12)


def test_c1_reference_value_is_minus_0128() -> None:
    """At 23 deg C, ps = ps0 -> C1 = 5*lg(296/314) = -0,128 dB."""
    mc = meteorological_corrections(23.0, 101.325)
    assert mc.c1 == pytest.approx(-0.12819, abs=1e-4)


def test_c1_zero_at_theta0() -> None:
    """C1 temperature term vanishes at 273+theta = theta0 = 314 K (theta=41)."""
    mc = meteorological_corrections(41.0, 101.325)
    assert mc.c1 == pytest.approx(0.0, abs=1e-9)


def test_c3_zero_without_air_absorption() -> None:
    """No attenuation coefficient supplied -> C3 = 0."""
    mc = meteorological_corrections(23.0, 101.325)
    assert mc.c3 == 0.0


def test_c3_from_air_absorption() -> None:
    """C3 = A0*(1,005 3 - 0,001 2*A0)^1,6, A0 = a(f)*r."""
    a, r = 0.02, 3.0
    mc = meteorological_corrections(
        23.0, 101.325, air_absorption_coefficient=a, radius=r
    )
    a0 = a * r
    expected = a0 * (1.0053 - 0.0012 * a0) ** 1.6
    assert mc.c3 == pytest.approx(expected)


def test_meteorological_invalid_pressure_raises() -> None:
    with pytest.raises(ValueError, match="'static_pressure' must be positive"):
        meteorological_corrections(23.0, 0.0)


# ==========================================================================
# ISO 3745 - uncertainty (Eq. 24/25)
# ==========================================================================
def test_uncertainty_worked_example_k2() -> None:
    """Clause 10.5 EXAMPLE: sigma_omc=2,0, sigma_R0=0,5, k=2 -> U = 4,1 dB."""
    u = precision_uncertainty(0.5, 2.0, 2.0)
    assert u == pytest.approx(4.123, abs=1e-3)


def test_uncertainty_one_sided_k_1p6() -> None:
    """k = 1,6 (one-sided) scales sigma_tot: U = 1,6*sqrt(0,25+4) = 3,299 dB."""
    u = precision_uncertainty(0.5, 2.0, 1.6)
    assert u == pytest.approx(1.6 * np.sqrt(4.25), abs=1e-6)


def test_uncertainty_bad_coverage_factor_raises() -> None:
    with pytest.raises(ValueError, match="'coverage_factor' must be positive"):
        precision_uncertainty(0.5, 2.0, 0.0)


# ==========================================================================
# ISO 3745 - surface averaging and sound power (Eq. 12/13/14/15)
# ==========================================================================
def test_surface_average_of_equal_levels_returns_that_level() -> None:
    """A uniform field: Lp_bar equals the common position level (Eq. 12)."""
    res = sound_power_anechoic(np.full((40, 1), 74.0), "hemisphere", radius=1.0)
    assert res.surface_pressure_level[0] == pytest.approx(74.0, abs=1e-12)


def test_hemisphere_uniform_field_lw() -> None:
    """Hemi uniform field: LW = Lp_bar + 10*lg(2*pi*r^2) + C1 + C2 (C3 = 0)."""
    r = 2.0
    lp = 70.0
    res = sound_power_anechoic(np.full((40, 1), lp), "hemisphere", radius=r)
    mc = meteorological_corrections(23.0, 101.325)
    expected = lp + 10.0 * np.log10(2.0 * np.pi * r**2) + mc.c1 + mc.c2
    assert res.surface_area == pytest.approx(2.0 * np.pi * r**2)
    assert res.sound_power_level[0] == pytest.approx(expected, abs=1e-9)
    assert isinstance(res, PrecisionSoundPowerResult)


def test_sphere_uniform_field_lw_uses_4pi() -> None:
    """Sphere uniform field uses S1 = 4*pi*r^2 (anechoic, Eq. 14)."""
    r = 1.0
    lp = 65.0
    res = sound_power_anechoic(np.full((40, 1), lp), "sphere", radius=r)
    mc = meteorological_corrections(23.0, 101.325)
    expected = lp + 10.0 * np.log10(4.0 * np.pi * r**2) + mc.c1 + mc.c2
    assert res.surface_area == pytest.approx(4.0 * np.pi * r**2)
    assert res.sound_power_level[0] == pytest.approx(expected, abs=1e-9)


def test_area_weighted_equals_equal_area_for_equal_areas() -> None:
    """Eq. 13 with equal areas reduces to the Eq. 12 equal-area average."""
    levels = 70.0 + np.random.default_rng(1).uniform(-4.0, 4.0, size=(40, 3))
    res_eq = sound_power_anechoic(
        levels, "hemisphere", radius=1.5, frequencies=np.array([250.0, 500.0, 1000.0])
    )
    res_ar = sound_power_anechoic(
        levels,
        "hemisphere",
        radius=1.5,
        areas=np.full(40, 0.7),
        frequencies=np.array([250.0, 500.0, 1000.0]),
    )
    assert np.allclose(res_eq.surface_pressure_level, res_ar.surface_pressure_level)


def test_anechoic_a_weighted_uncertainty_is_half_db_base() -> None:
    """The A-weighted sigma_R0 is 0,5 dB in both rooms; U = 2*0,5 = 1,0 dB
    with no operating uncertainty (Eq. 24/25).
    """
    res = sound_power_anechoic(np.full((40, 1), 70.0), "sphere", radius=1.0)
    assert res.uncertainty == pytest.approx(1.0, abs=1e-9)


def test_anechoic_per_band_uncertainty_uses_table3() -> None:
    """Sphere -> Table 3 (anechoic): 1000 Hz sigma_R0 = 0,5 -> U = 1,0 dB."""
    res = sound_power_anechoic(
        np.full((40, 1), 70.0), "sphere", radius=1.0, frequencies=np.array([1000.0])
    )
    assert res.uncertainty_bands[0] == pytest.approx(1.0, abs=1e-9)


def test_hemisphere_per_band_uncertainty_uses_table2() -> None:
    """Hemisphere -> Table 2 (hemi-anechoic): 1000 Hz sigma_R0 = 1,0 -> U = 2,0."""
    res = sound_power_anechoic(
        np.full((40, 1), 70.0), "hemisphere", radius=1.0, frequencies=np.array([1000.0])
    )
    assert res.uncertainty_bands[0] == pytest.approx(2.0, abs=1e-9)


def test_anechoic_background_requires_frequencies() -> None:
    """K1 needs the band centres to pick the 6/10 dB floor."""
    levels = np.full((40, 1), 80.0)
    background = np.full((40, 1), 60.0)
    with pytest.raises(
        ValueError,
        match="'frequencies' are required with 'background_levels'",
    ):
        sound_power_anechoic(
            levels,
            "hemisphere",
            radius=1.0,
            background_levels=background,
        )


def test_anechoic_full_chain_with_k1() -> None:
    """End-to-end: LW = Lp_bar(after K1) + 10*lg(S) + C1 + C2."""
    r = 1.0
    src = np.full((40, 1), 80.0)
    bg = np.full((40, 1), 70.0)  # dLpi = 10 dB (mid band) -> K1 = 0,458
    freqs = np.array([1000.0])
    res = sound_power_anechoic(
        src, "hemisphere", radius=r, background_levels=bg, frequencies=freqs
    )
    k1 = precision_background_correction(src, bg, freqs)
    lp_bar = 80.0 - k1[0, 0]
    mc = meteorological_corrections(23.0, 101.325)
    expected = lp_bar + 10.0 * np.log10(2.0 * np.pi * r**2) + mc.c1 + mc.c2
    assert res.sound_power_level[0] == pytest.approx(expected, abs=1e-9)
    assert np.allclose(res.background_correction, k1)


def test_anechoic_bands_above_10khz_survive_without_lwa() -> None:
    """ISO 3745 covers one-third octaves up to 20 kHz (sigma_R0 Tables 2/3),
    but the ISO 3744 Annex E A-weighting table stops at 10 kHz: a 12,5 kHz
    band must not abort the determination -- the per-band LW and uncertainty
    stay defined and only the A-weighted total is NaN.
    """
    freqs = np.array([10000.0, 12500.0])
    res = sound_power_anechoic(
        np.full((40, freqs.size), 70.0), "sphere", radius=1.0, frequencies=freqs
    )
    assert np.all(np.isfinite(res.sound_power_level))
    assert np.all(np.isfinite(res.uncertainty_bands))
    assert np.isnan(res.sound_power_level_a)


def test_anechoic_invalid_surface_raises() -> None:
    levels = np.full((40, 1), 70.0)
    with pytest.raises(ValueError, match="'surface' must be 'sphere' or 'hemisphere'"):
        sound_power_anechoic(levels, "box", radius=1.0)  # type: ignore[arg-type]


def test_anechoic_requires_a_radius() -> None:
    levels = np.full((40, 1), 70.0)
    with pytest.raises(TypeError, match="required keyword-only argument"):
        sound_power_anechoic(levels, "sphere")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="A positive 'radius' is required"):
        sound_power_anechoic(levels, "sphere", radius=0.0)


def test_anechoic_areas_wrong_length_raises() -> None:
    levels = np.full((40, 1), 70.0)
    short_areas = np.full(10, 1.0)
    with pytest.raises(
        ValueError,
        match="'areas' must have one value per microphone position",
    ):
        sound_power_anechoic(levels, "sphere", radius=1.0, areas=short_areas)


# --------------------------------------------------------------------------
# Per-band and per-position quantities that do not run over their own axis
# --------------------------------------------------------------------------
def _forty_position_determination() -> PrecisionSoundPowerResult:
    """A forty-position, four-band ISO 3745 hemi-anechoic determination."""
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    return sound_power_anechoic(
        np.full((40, freqs.size), 74.0), "hemisphere", radius=2.0, frequencies=freqs
    )


@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_a_per_band_column_off_the_band_axis_is_refused(trim: bool) -> None:
    """``uncertainty_bands`` is per band and nothing downstream says so.

    The fiche never opens the column, so a length other than the band axis
    reaches no reader that could complain: the sheet renders complete with
    the surplus value nowhere on it.
    """
    import dataclasses

    result = _forty_position_determination()
    values = result.uncertainty_bands
    wrong = values[:-1] if trim else np.append(values, values[-1])
    with pytest.raises(ValueError, match="'uncertainty_bands'"):
        dataclasses.replace(result, uncertainty_bands=wrong)


def test_the_background_correction_is_pinned_on_its_band_axis() -> None:
    """``background_correction`` is positions by bands: axis 1 carries them."""
    import dataclasses

    result = _forty_position_determination()
    with pytest.raises(ValueError, match=r"'background_correction \(axis 1\)'"):
        dataclasses.replace(
            result, background_correction=result.background_correction[:, :-1]
        )


def test_the_microphone_position_axis_is_pinned() -> None:
    """K1i (Eq. 11) and DIi (Eq. 21) both describe the same position i.

    Two position counts leave no position to read them side by side, and the
    fiche states neither, so unguarded the mismatch reaches the sheet as no
    difference at all.
    """
    import dataclasses

    result = _forty_position_determination()
    with pytest.raises(ValueError, match=r"'directivity_index'.*microphone position"):
        dataclasses.replace(result, directivity_index=result.directivity_index[:-1])


# ==========================================================================
# ISO 9614-3 - sound power by intensity scanning (Eq. 5/8/9/10)
# ==========================================================================
def test_uniform_intensity_recovers_lw_exact() -> None:
    """Fully enclosed source, uniform In: sum Pi = W -> LW = 10*lg(W/P0)."""
    w = 1.0e-4  # 100 uW -> LW = 80 dB
    areas = np.array([0.5, 1.0, 0.25, 2.0])
    s = float(np.sum(areas))
    i_n = np.full(areas.shape, w / s)  # In = W/S on every surface
    res = sound_power_intensity_precision(i_n, areas)
    assert isinstance(res, PrecisionIntensityResult)
    assert res.sound_power[0] == pytest.approx(w)
    assert res.sound_power_level[0] == pytest.approx(10.0 * np.log10(w / _P0), abs=1e-9)


def test_lw_independent_of_segmentation() -> None:
    """LW depends only on the total power, not on the area split."""
    w = 5.0e-5
    for areas in (np.array([1.0, 1.0]), np.array([0.2, 0.5, 1.3, 3.0])):
        s = float(np.sum(areas))
        res = sound_power_intensity_precision(np.full(areas.shape, w / s), areas)
        assert res.sound_power_level[0] == pytest.approx(
            10.0 * np.log10(w / _P0), abs=1e-9
        )


def test_net_negative_band_flagged_not_applicable() -> None:
    """A dominant external source makes net P < 0 -> band not applicable."""
    i_n = np.array([[1.0e-6], [-5.0e-6], [1.0e-6]])
    areas = np.array([1.0, 1.0, 1.0])
    with pytest.warns(SoundPowerWarning):
        res = sound_power_intensity_precision(i_n, areas)
    assert bool(res.not_applicable_band[0]) is True
    assert np.isnan(res.sound_power_level[0])
    assert res.sound_power[0] < 0.0


def test_lw0_equals_lw_at_reference() -> None:
    """At 23 deg C, 101 325 Pa the Eq. 10 argument is 1 -> LW0 = LW."""
    areas = np.array([1.0, 1.0])
    i_n = np.full((2,), 1.0e-5)
    res = sound_power_intensity_precision(
        i_n, areas, temperature=23.0, barometric_pressure=101325.0
    )
    assert res.sound_power_level_normalized[0] == pytest.approx(
        res.sound_power_level[0], abs=1e-12
    )


def test_lw0_shift_off_reference() -> None:
    """B=100 000 Pa, theta=20 deg C -> LW0 = LW + 0,0194 dB (Eq. 10)."""
    areas = np.array([1.0, 1.0])
    i_n = np.full((2,), 1.0e-5)
    res = sound_power_intensity_precision(
        i_n, areas, temperature=20.0, barometric_pressure=100000.0
    )
    shift = res.sound_power_level_normalized[0] - res.sound_power_level[0]
    expected = -15.0 * np.log10((100000.0 / 101325.0) * (296.15 / 293.15))
    assert shift == pytest.approx(expected, abs=1e-9)
    assert shift == pytest.approx(0.0194, abs=1e-3)


def test_intensity_areas_mismatch_raises() -> None:
    intensity_3_segments = np.full((3,), 1e-5)
    areas_2_segments = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="'partial_intensity' first axis"):
        sound_power_intensity_precision(intensity_3_segments, areas_2_segments)


def test_intensity_nonpositive_area_raises() -> None:
    intensity = np.full((2,), 1e-5)
    areas_with_zero = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="All 'areas' must be positive"):
        sound_power_intensity_precision(intensity, areas_with_zero)


def test_intensity_single_segment_2d_input_not_transposed() -> None:
    # A genuine (1, N) single-segment, N-band array must NOT be read as N
    # segments, even when N equals a plausible segment count. One area -> one
    # segment; the N bands are preserved.
    i_n = np.array([[1e-5, 2e-5, 3e-5]])  # 1 segment, 3 bands
    res = sound_power_intensity_precision(i_n, np.array([2.0]))
    assert res.sound_power.shape == (3,)  # 3 bands, not 3 segments
    np.testing.assert_allclose(res.partial_power[0], np.array([2e-5, 4e-5, 6e-5]))


# ==========================================================================
# ISO 9614-3 - field indicators (Annex B) and criteria (Annex C)
# ==========================================================================
def test_indicators_uniform_field_fs_zero() -> None:
    """A uniform in-phase field: FS = 0 and F_pIn(signed) = F_pIn(unsigned)."""
    i_n = np.full((6, 1), 2.0e-6)
    lp = np.full((6, 1), 60.0)
    ind = precision_field_indicators(i_n, lp)
    assert isinstance(ind, PrecisionFieldIndicators)
    assert ind.fs[0] == pytest.approx(0.0, abs=1e-12)
    assert ind.f_pi_signed[0] == pytest.approx(ind.f_pi_unsigned[0], abs=1e-12)


def test_indicators_signed_geq_unsigned() -> None:
    """With sign changes, |sum In| < sum|In| -> F_pIn(signed) > F_pIn(unsigned)."""
    i_n = np.array([[3.0e-6], [-1.0e-6], [2.0e-6], [-0.5e-6]])
    lp = np.full((4, 1), 62.0)
    ind = precision_field_indicators(i_n, lp)
    assert ind.f_pi_signed[0] > ind.f_pi_unsigned[0]


def test_indicators_ft_zero_for_constant_time_series() -> None:
    """A constant time series has zero temporal variability FT."""
    i_n = np.full((4, 1), 1.0e-6)
    lp = np.full((4, 1), 60.0)
    windows = np.full((10, 1), 1.0e-6)
    ind = precision_field_indicators(i_n, lp, time_window_intensity=windows)
    assert ind.ft is not None
    assert ind.ft[0] == pytest.approx(0.0, abs=1e-12)


def test_criteria_uniform_field_passes_3_and_4() -> None:
    """Uniform field: criterion 3 (0 <= 3) and criterion 4 (FS = 0 <= 2) pass."""
    i_n = np.full((6, 1), 2.0e-6)
    lp = np.full((6, 1), 60.0)
    ind = precision_field_indicators(i_n, lp)
    crit = precision_qualification(ind)
    assert isinstance(crit, PrecisionCriteria)
    assert bool(crit.criterion_3[0]) is True
    assert bool(crit.criterion_4[0]) is True


def test_criterion_3_fails_on_strong_sign_cancellation() -> None:
    """A near-reactive field (net ~0) makes F_signed - F_unsigned > 3 dB."""
    i_n = np.array([[1.0e-5], [-0.999e-5], [1.0e-5], [-0.999e-5]])
    lp = np.full((4, 1), 70.0)
    ind = precision_field_indicators(i_n, lp)
    crit = precision_qualification(ind)
    assert bool(crit.criterion_3[0]) is False


def test_criterion_4_fails_on_highly_nonuniform_field() -> None:
    """A strongly non-uniform positive field makes FS > 2."""
    i_n = np.array([[1.0e-8], [1.0e-8], [1.0e-8], [1.0e-8], [1.0e-8], [1.0e-4]])
    lp = np.full((6, 1), 70.0)
    ind = precision_field_indicators(i_n, lp)
    crit = precision_qualification(ind)
    assert ind.fs[0] > 2.0
    assert bool(crit.criterion_4[0]) is False


def test_criterion_2_dynamic_capability() -> None:
    """delta_pI0 = 14 dB, K = 10 -> Ld = 4; criterion 2 passes iff
    F_pIn(signed) <= 4 dB.
    """
    # Build a field with a small, controllable F_pIn(signed).
    i_n = np.full((4, 1), 2.0e-6)
    # Choose Lp so that F_pIn(signed) = Lp_bar - 10*lg(In/I0) is ~2 dB (< 4).
    li = 10.0 * np.log10(2.0e-6 / _P0)
    lp = np.full((4, 1), li + 2.0)
    ind = precision_field_indicators(i_n, lp)
    assert ind.f_pi_signed[0] == pytest.approx(2.0, abs=1e-6)
    crit_pass = precision_qualification(ind, pressure_residual_index=14.0)
    assert crit_pass.criterion_2 is not None
    assert bool(crit_pass.criterion_2[0]) is True
    # A weaker probe (delta_pI0 = 11 -> Ld = 1 < 2) fails criterion 2.
    crit_fail = precision_qualification(ind, pressure_residual_index=11.0)
    assert crit_fail.criterion_2 is not None
    assert bool(crit_fail.criterion_2[0]) is False


def test_criterion_1_repeatability_uses_half_s() -> None:
    """Criterion 1 |LIn(1)-LIn(2)| <= s/2, s from Table 1 (1000 Hz -> s = 1)."""
    i_n = np.full((4, 1), 2.0e-6)
    lp = np.full((4, 1), 60.0)
    ind = precision_field_indicators(i_n, lp)
    freqs = np.array([1000.0])  # s = 1,0 -> s/2 = 0,5 dB
    crit_pass = precision_qualification(
        ind,
        scan_intensity_level_1=np.array([70.0]),
        scan_intensity_level_2=np.array([70.4]),
        frequencies=freqs,
    )
    assert crit_pass.criterion_1 is not None
    assert bool(crit_pass.criterion_1[0]) is True
    crit_fail = precision_qualification(
        ind,
        scan_intensity_level_1=np.array([70.0]),
        scan_intensity_level_2=np.array([70.8]),
        frequencies=freqs,
    )
    assert crit_fail.criterion_1 is not None
    assert bool(crit_fail.criterion_1[0]) is False


def test_criterion_5_scan_density_ratio() -> None:
    """Criterion 5: 0,83 <= FS(1)/FS(2) <= 1,2."""
    i_n = np.full((4, 1), 2.0e-6)
    lp = np.full((4, 1), 60.0)
    ind = precision_field_indicators(i_n, lp)
    crit = precision_qualification(
        ind,
        field_nonuniformity_1=np.array([1.0]),
        field_nonuniformity_2=np.array([1.0]),
    )
    assert crit.criterion_5 is not None
    assert bool(crit.criterion_5[0]) is True
    crit2 = precision_qualification(
        ind,
        field_nonuniformity_1=np.array([1.0]),
        field_nonuniformity_2=np.array([2.0]),  # ratio 0,5 < 0,83
    )
    assert crit2.criterion_5 is not None
    assert bool(crit2.criterion_5[0]) is False


def test_qualified_combines_criteria_1_to_4() -> None:
    """'qualified' is the AND of criteria 1-4 when 1 and 2 are evaluable."""
    i_n = np.full((5, 1), 2.0e-6)
    li = 10.0 * np.log10(2.0e-6 / _P0)
    lp = np.full((5, 1), li + 1.0)  # F_pIn(signed) ~ 1 dB
    ind = precision_field_indicators(i_n, lp)
    crit = precision_qualification(
        ind,
        scan_intensity_level_1=np.array([70.0]),
        scan_intensity_level_2=np.array([70.2]),
        pressure_residual_index=14.0,
        frequencies=np.array([1000.0]),
    )
    assert crit.qualified is not None
    assert bool(crit.qualified[0]) is True


def test_criterion_5_qualifies_band_despite_failed_criterion_4() -> None:
    """C.1.6.2: if criterion 5 holds, the band is qualified as a final result
    even if FS(2) >= 2 (criterion 4 fails).
    """
    # A strongly non-uniform positive field: FS > 2 (criterion 4 fails).
    i_n = np.array([[1.0e-8], [1.0e-8], [1.0e-8], [1.0e-8], [1.0e-8], [1.0e-4]])
    li = 10.0 * np.log10(np.mean(i_n) / _P0)
    lp = np.full((6, 1), li + 1.0)  # small F_pIn(signed): criteria 2/3 pass
    ind = precision_field_indicators(i_n, lp)
    common: dict = {
        "scan_intensity_level_1": np.array([70.0]),
        "scan_intensity_level_2": np.array([70.2]),
        "pressure_residual_index": 14.0,
        "frequencies": np.array([1000.0]),
    }
    # Doubled scan density converges (ratio 1,0): criterion 5 rescues the band.
    crit = precision_qualification(
        ind,
        field_nonuniformity_1=ind.fs.copy(),
        field_nonuniformity_2=ind.fs.copy(),
        **common,
    )
    assert bool(crit.criterion_4[0]) is False
    assert crit.criterion_5 is not None
    assert bool(crit.criterion_5[0]) is True
    assert crit.qualified is not None
    assert bool(crit.qualified[0]) is True
    # Without criterion 5 the failed criterion 4 still disqualifies the band.
    crit_no5 = precision_qualification(ind, **common)
    assert crit_no5.qualified is not None
    assert bool(crit_no5.qualified[0]) is False
    # A non-converged density (ratio 0,5) does not rescue it either.
    crit_bad5 = precision_qualification(
        ind,
        field_nonuniformity_1=ind.fs.copy(),
        field_nonuniformity_2=2.0 * ind.fs,
        **common,
    )
    assert crit_bad5.qualified is not None
    assert bool(crit_bad5.qualified[0]) is False


def test_criterion_1_without_limit_raises() -> None:
    """Criterion 1 needs s (frequencies or an explicit limit)."""
    i_n = np.full((4, 1), 2.0e-6)
    lp = np.full((4, 1), 60.0)
    ind = precision_field_indicators(i_n, lp)
    scan_1, scan_2 = np.array([70.0]), np.array([70.2])
    with pytest.raises(ValueError, match="Criterion 1 needs the limit s"):
        precision_qualification(
            ind,
            scan_intensity_level_1=scan_1,
            scan_intensity_level_2=scan_2,
        )


def test_field_indicators_shape_mismatch_raises() -> None:
    intensity = np.full((4, 1), 1e-6)
    mismatched_levels = np.full((3, 1), 60.0)
    with pytest.raises(
        ValueError,
        match=(
            r"precision_field_indicators: 'segment_intensity' "
            r".*'segment_pressure_levels' .*same shape"
        ),
    ):
        precision_field_indicators(intensity, mismatched_levels)


# ==========================================================================
# ISO 9614-3 - quantities pinned to their own axis at construction
# ==========================================================================
def _four_band_determination() -> PrecisionIntensityResult:
    """A four-segment, four-band determination from the public entry point."""
    areas = np.array([0.5, 1.0, 0.25, 2.0])
    intensity = np.full((areas.size, 4), 1.0e-4 / float(np.sum(areas)))
    return sound_power_intensity_precision(
        intensity, areas, frequencies=np.array([250.0, 500.0, 1000.0, 2000.0])
    )


def _four_band_indicators() -> PrecisionFieldIndicators:
    """Annex B indicators over four bands, from the public entry point."""
    i_n = np.tile(np.array([[1.0e-6], [2.0e-6], [1.5e-6], [1.2e-6]]), (1, 4))
    lp = np.full((4, 4), 62.0)
    return precision_field_indicators(i_n, lp)


@pytest.mark.parametrize(
    "field_name",
    [
        "frequencies",
        "sound_power",
        "sound_power_level",
        "sound_power_level_normalized",
        "not_applicable_band",
    ],
)
@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_a_determination_column_off_the_band_axis_is_refused(
    field_name: str, trim: bool
) -> None:
    """Every per-band quantity of the determination is pinned at construction.

    ``sound_power`` is why the long direction matters as much as the short
    one: the sheet prints the level, and the signed band total in watts is
    the one column no reader in the library opens, so a band too many rides
    all the way through a complete fiche.
    """
    import dataclasses

    result = _four_band_determination()
    values = np.asarray(getattr(result, field_name))
    wrong = values[:-1] if trim else np.append(values, values[-1])
    with pytest.raises(ValueError, match=f"'{field_name}'"):
        dataclasses.replace(result, **{field_name: wrong})


def test_partial_power_off_the_band_axis_is_refused() -> None:
    """Partial powers run over surfaces on axis 0 and over bands on axis 1."""
    import dataclasses

    result = _four_band_determination()
    three_bands = result.partial_power[:, :-1]
    with pytest.raises(ValueError, match=r"'partial_power \(axis 1\)'"):
        dataclasses.replace(result, partial_power=three_bands)


def test_a_field_indicator_off_the_band_axis_is_refused() -> None:
    """FS read on one band cannot stand in for a four-band spectrum (Eq. B.8)."""
    import dataclasses

    indicators = _four_band_indicators()
    one_band_fs = indicators.fs[:1]
    with pytest.raises(ValueError, match="'fs'"):
        dataclasses.replace(indicators, fs=one_band_fs)


def test_criteria_from_a_one_band_scan_pair_are_refused() -> None:
    """Criterion 1 comes from the scan levels, the others from the indicators.

    Nothing compares the two, so a repeatability pair read on a single band
    (Eq. C.1) would otherwise decide 'qualified' for all four of them.
    """
    indicators = _four_band_indicators()
    one_band_scan = np.array([70.0])
    with pytest.raises(ValueError, match="'criterion_1'"):
        precision_qualification(
            indicators,
            scan_intensity_level_1=one_band_scan,
            scan_intensity_level_2=one_band_scan,
            pressure_residual_index=14.0,
            frequencies=np.array([1000.0]),
        )


def test_criterion_5_off_the_band_axis_is_refused() -> None:
    """Criterion 5 rescues a band from criterion 4 (C.1.6.2) band by band."""
    import dataclasses

    indicators = _four_band_indicators()
    criteria = precision_qualification(
        indicators,
        field_nonuniformity_1=indicators.fs.copy(),
        field_nonuniformity_2=indicators.fs.copy(),
    )
    assert criteria.criterion_5 is not None
    one_band_criterion_5 = criteria.criterion_5[:1]
    with pytest.raises(ValueError, match="'criterion_5'"):
        dataclasses.replace(criteria, criterion_5=one_band_criterion_5)


def test_anechoic_result_plot_returns_axes() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    levels = np.full((40, freqs.size), 74.0)
    result = sound_power_anechoic(levels, "hemisphere", radius=1.0, frequencies=freqs)
    assert isinstance(result, PrecisionSoundPowerResult)
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_intensity_result_plot_returns_axes() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    freqs = np.array([250.0, 500.0, 1000.0])
    areas = np.array([0.5, 1.0, 0.75])
    intensity = np.full((3, freqs.size), 1.0e-5)
    result = sound_power_intensity_precision(intensity, areas, frequencies=freqs)
    assert isinstance(result, PrecisionIntensityResult)
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_public_exports() -> None:
    from phonometry import emission

    for name in (
        "sound_power_anechoic",
        "PrecisionSoundPowerResult",
        "precision_positions",
        "precision_background_correction",
        "meteorological_corrections",
        "MeteorologicalCorrection",
        "precision_uncertainty",
        "sound_power_intensity_precision",
        "PrecisionIntensityResult",
        "precision_field_indicators",
        "PrecisionFieldIndicators",
        "precision_qualification",
        "PrecisionCriteria",
    ):
        assert hasattr(emission, name), name
