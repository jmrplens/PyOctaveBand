#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power from pressure over an enveloping surface: ISO 3744:2010
(engineering) and ISO 3746:2010 (survey).

Physics anchors:
- Monopole over a rigid plane radiates into a hemisphere (Q = 2); on a
  radius-r hemisphere the field is uniform: Lp = LW - 10*lg(2*pi*r^2),
  so LW = Lp + 10*lg(S) is recovered exactly (ISO 3744 Eq. 18, S = 2*pi*r^2).
- K1 = -10*lg(1 - 10^(-0,1*dL)) (ISO 3744 Eq. 16). dL = 6 dB -> 1,256 dB.
- K2 = 10*lg(1 + 4*S/A) (ISO 3744 Eq. A.2).
"""

import numpy as np
import pytest

from phonometry import emission


# --------------------------------------------------------------------------
# measurement_positions — normative coordinate tables (ISO 3744 Annex B)
# --------------------------------------------------------------------------
def test_hemisphere_key_positions_count_and_radius() -> None:
    """Engineering, one plane -> 10 key positions (Table B.1), scaled by r."""
    pos = emission.measurement_positions("hemisphere", radius=4.0)
    assert pos.shape == (10, 3)
    # z >= 0 (upper hemisphere) and radius ~ 4 m (unit coords * r).
    assert np.all(pos[:, 2] >= 0.0)
    norms = np.linalg.norm(pos, axis=1)
    assert np.allclose(norms, 4.0, atol=4.0 * 0.03)


def test_hemisphere_survey_four_positions() -> None:
    """Survey, one plane -> 4 positions 4,5,6,10 (ISO 3746 clause 8.2.1)."""
    pos = emission.measurement_positions("hemisphere", radius=1.0, grade="survey")
    assert pos.shape == (4, 3)


def test_hemisphere_exact_table_b1_position_one() -> None:
    """First key coordinate is exactly (0,16, -0,96, 0,22)*r (Table B.1)."""
    pos = emission.measurement_positions("hemisphere", radius=1.0)
    assert np.allclose(pos[0], [0.16, -0.96, 0.22])


def test_hemisphere_two_planes_uses_table_b2() -> None:
    """Two planes -> 5 positions 2,3,6,7,9 from Table B.2 (ISO 3744 8.1.1)."""
    pos = emission.measurement_positions("hemisphere", radius=1.0, reflecting_planes=2)
    assert pos.shape == (5, 3)
    assert np.allclose(pos[0], [0.50, -0.86, 0.15])  # position 2 of Table B.2


def test_hemisphere_three_planes_uses_table_b3() -> None:
    """Three planes -> 3 positions 1,2,3 from Table B.3 (ISO 3744 8.1.1)."""
    pos = emission.measurement_positions("hemisphere", radius=1.0, reflecting_planes=3)
    assert pos.shape == (3, 3)
    assert np.allclose(pos[0], [0.86, -0.50, 0.15])  # position 1 of Table B.3


def test_survey_three_planes_not_implemented() -> None:
    """Survey 3-plane needs untranscribed extended Table B.2 positions."""
    with pytest.raises(NotImplementedError):
        emission.measurement_positions(
            "hemisphere", radius=1.0, reflecting_planes=3, grade="survey"
        )


# --------------------------------------------------------------------------
# K1 background-noise correction (ISO 3744 Eq. 16, ISO 3746 Eq. 12)
# --------------------------------------------------------------------------
def test_k1_exact_value_at_6db() -> None:
    """dL = 6 dB -> K1 = -10*lg(1-10^-0,6) = 1,2560... dB."""
    k1 = emission.background_noise_correction(np.array([56.0]), np.array([50.0]))
    assert k1[0] == pytest.approx(1.25628, abs=1e-4)


def test_k1_zero_above_15db() -> None:
    """dL > 15 dB -> K1 = 0 (no correction)."""
    k1 = emission.background_noise_correction(np.array([80.0]), np.array([60.0]))
    assert k1[0] == 0.0


def test_k1_applies_equation_16_at_exactly_15db() -> None:
    """dL == 15 dB stays inside '6 dB <= dLp <= 15 dB' (ISO 3744:2010, 8.2.3),
    so Eq. (16) applies: K1 = -10*lg(1-10^-1,5) = 0,1395... dB; K1 = 0 only
    for dLp strictly above 15 dB.
    """
    k1 = emission.background_noise_correction(np.array([75.0]), np.array([60.0]))
    assert k1[0] == pytest.approx(-10.0 * np.log10(1.0 - 10.0**-1.5), abs=1e-9)
    assert k1[0] == pytest.approx(0.13955, abs=1e-4)


def test_k1_survey_applies_equation_at_exactly_10db() -> None:
    """Survey: dL == 10 dB stays inside '3 dB <= dLp <= 10 dB' (ISO 3746:2010,
    8.3.3), so the correction applies: K1 = -10*lg(1-10^-1) = 0,4576 dB.
    """
    k1 = emission.background_noise_correction(
        np.array([70.0]), np.array([60.0]), grade="survey"
    )
    assert k1[0] == pytest.approx(0.45757, abs=1e-4)
    assert (
        emission.background_noise_correction(
            np.array([70.1]), np.array([60.0]), grade="survey"
        )[0]
        == 0.0
    )


def test_k1_below_criterion_clamps_and_warns() -> None:
    """dL < 6 dB (engineering) -> clamp to the 6 dB value, warn."""
    with pytest.warns(emission.SoundPowerWarning):
        k1 = emission.background_noise_correction(np.array([53.0]), np.array([50.0]))
    assert k1[0] == pytest.approx(1.25628, abs=1e-4)


def test_k1_survey_criterion_3db() -> None:
    """Survey clamps below 3 dB to -10*lg(1-10^-0,3) = 3,0206 dB."""
    with pytest.warns(emission.SoundPowerWarning):
        k1 = emission.background_noise_correction(
            np.array([52.0]), np.array([50.0]), grade="survey"
        )
    assert k1[0] == pytest.approx(3.0206, abs=1e-4)


# --------------------------------------------------------------------------
# K2 environmental correction (ISO 3744 Eq. A.2/A.3)
# --------------------------------------------------------------------------
def test_k2_free_field_is_zero() -> None:
    """A -> infinity (ideal free field) -> K2 = 0."""
    k2 = emission.environmental_correction(100.0, absorption_area=1e12)
    assert k2 == pytest.approx(0.0, abs=1e-6)


def test_k2_closed_form_from_absorption() -> None:
    """S = 50, A = 200 -> K2 = 10*lg(1 + 4*50/200) = 10*lg(2) = 3,0103 dB."""
    k2 = emission.environmental_correction(50.0, absorption_area=200.0)
    assert k2 == pytest.approx(3.0103, abs=1e-4)


def test_k2_from_reverberation_time_sabine() -> None:
    """A = 0,16*V/T; V=300, T=1,2 -> A=40; S=40 -> K2=10*lg(1+4)=6,9897 dB."""
    k2 = emission.environmental_correction(40.0, reverberation_time=1.2, volume=300.0)
    assert k2 == pytest.approx(6.9897, abs=1e-4)


def test_k2_no_room_data_is_free_field_zero() -> None:
    """All room inputs omitted is the legitimate free-field path: K2 = 0."""
    k2 = emission.environmental_correction(100.0)
    assert k2 == pytest.approx(0.0, abs=1e-12)


def test_k2_reverberation_time_without_volume_raises() -> None:
    """Half-specified room (T without V) must not silently give K2 = 0."""
    with pytest.raises(ValueError, match="volume"):
        emission.environmental_correction(40.0, reverberation_time=1.2)


def test_k2_volume_without_reverberation_time_raises() -> None:
    with pytest.raises(ValueError, match="reverberation_time"):
        emission.environmental_correction(40.0, volume=300.0)


def test_k2_mean_absorption_without_room_surface_raises() -> None:
    with pytest.raises(ValueError, match="room_surface"):
        emission.environmental_correction(50.0, mean_absorption_coefficient=0.2)


def test_k2_room_surface_without_mean_absorption_raises() -> None:
    with pytest.raises(ValueError, match="mean_absorption_coefficient"):
        emission.environmental_correction(50.0, room_surface=500.0)


def test_partial_room_data_raises_via_sound_power_pressure() -> None:
    """The partial-pair guard is enforced through sound_power_pressure too."""
    levels = np.full((10, 1), 90.0)
    half_specified_room = emission.RoomEnvironment(reverberation_time=1.2)
    with pytest.raises(ValueError, match="volume"):
        emission.sound_power_pressure(
            levels,
            "hemisphere",
            radius=2.0,
            room=half_specified_room,
        )


# --------------------------------------------------------------------------
# sound_power_pressure — monopole recovery (ISO 3744 Eq. 18)
# --------------------------------------------------------------------------
def test_monopole_recovers_lw_exact() -> None:
    """Uniform hemisphere field Lp = LW - 10*lg(2*pi*r^2) recovers LW."""
    lw_true = 95.0
    r = 4.0
    s = 2.0 * np.pi * r**2
    lp = lw_true - 10.0 * np.log10(s)
    levels = np.full((10, 1), lp)
    res = emission.sound_power_pressure(levels, "hemisphere", radius=r)
    assert isinstance(res, emission.SoundPowerResult)
    assert res.surface_area == pytest.approx(s)
    assert res.sound_power_level[0] == pytest.approx(lw_true, abs=1e-9)


def test_monopole_radius_independence() -> None:
    """The recovered LW is independent of the hemisphere radius."""
    lw_true = 88.0
    for r in (1.0, 2.5, 7.0):
        s = 2.0 * np.pi * r**2
        lp = lw_true - 10.0 * np.log10(s)
        res = emission.sound_power_pressure(
            np.full((10, 1), lp), "hemisphere", radius=r
        )
        assert res.sound_power_level[0] == pytest.approx(lw_true, abs=1e-9)


def test_box_surface_area_formula() -> None:
    """One plane: S = 4(ab+bc+ca), a=0,5l1+d, b=0,5l2+d, c=l3+d (Eq. 9)."""
    l1, l2, l3, d = 2.0, 1.0, 1.5, 1.0
    a, b, c = 0.5 * l1 + d, 0.5 * l2 + d, l3 + d
    s_exp = 4.0 * (a * b + b * c + c * a)
    lp = 70.0 - 10.0 * np.log10(s_exp)
    res = emission.sound_power_pressure(
        np.full((9, 1), lp), "box", dimensions=(l1, l2, l3), distance=d
    )
    assert res.surface_area == pytest.approx(s_exp)
    assert res.sound_power_level[0] == pytest.approx(70.0, abs=1e-9)


def test_full_chain_with_k1_and_k2() -> None:
    """LW = Lp'(ST) - K1 - K2 + 10*lg(S/S0) applied end to end."""
    r = 2.0
    s = 2.0 * np.pi * r**2
    lp = np.full((10, 1), 80.0)
    bg = np.full((10, 1), 71.0)  # dL = 9 dB -> apply K1
    dl = 9.0
    k1 = -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * dl))
    k2 = emission.environmental_correction(s, absorption_area=300.0)
    res = emission.sound_power_pressure(
        lp,
        "hemisphere",
        radius=r,
        background_levels=bg,
        room=emission.RoomEnvironment(absorption_area=300.0),
    )
    expected = 80.0 - k1 - k2 + 10.0 * np.log10(s)
    assert res.sound_power_level[0] == pytest.approx(expected, abs=1e-9)
    assert res.background_correction[0] == pytest.approx(k1, abs=1e-9)
    assert res.environmental_correction[0] == pytest.approx(k2, abs=1e-9)


def test_a_weighted_sound_power_from_bands() -> None:
    """LWA = 10*lg(sum 10^(0,1(LWk+Ck))) using Annex E octave Ck."""
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    ck = np.array([-8.6, -3.2, 0.0, 1.2])
    r = 2.0
    s = 2.0 * np.pi * r**2
    lw_bands = np.array([90.0, 92.0, 95.0, 93.0])
    lp_bands = lw_bands - 10.0 * np.log10(s)
    levels = np.tile(lp_bands, (10, 1))  # uniform field, 10 positions x 4 bands
    res = emission.sound_power_pressure(
        levels, "hemisphere", radius=r, frequencies=freqs
    )
    assert np.allclose(res.sound_power_level, lw_bands, atol=1e-9)
    lwa_exp = 10.0 * np.log10(np.sum(10.0 ** (0.1 * (lw_bands + ck))))
    assert res.sound_power_level_a == pytest.approx(lwa_exp, abs=1e-6)


def test_directivity_index_uniform_field_is_zero() -> None:
    """A perfectly uniform field has DI = 0 at every position (Eq. 7)."""
    res = emission.sound_power_pressure(
        np.full((10, 1), 60.0), "hemisphere", radius=2.0
    )
    assert np.allclose(res.directivity_index, 0.0, atol=1e-9)


def test_directivity_index_uniform_field_with_background_is_zero() -> None:
    """With background correction and a uniform field (large dL so K1 ~ 0),
    the apparent DI stays ~0 at every position; it must NOT be inflated by the
    surface-area term 10*lg(S/S0) (~22 dB at r = 5 m). ISO 3744 Eq. 7 / Eq. 16.
    """
    src = np.full((10, 1), 80.0)
    bg = np.full((10, 1), 40.0)  # dL = 40 dB -> K1 ~ 4e-4 dB
    res = emission.sound_power_pressure(
        src, "hemisphere", radius=5.0, background_levels=bg
    )
    assert np.allclose(res.directivity_index, 0.0, atol=1e-3)


def test_directivity_index_background_corrects_each_position() -> None:
    """Per Eq. 7 both the per-position level and the surface mean are
    background-corrected by the same broadband K1, which cancels in the
    difference: DI_i = (Lp_i - K1) - (mean - K1) = Lp_i - mean. The DI must
    NOT carry a residual +K1 offset (ISO 3744 Eq. 7, notes sec. 9).
    """
    src = np.array(
        [[82.0], [78.0], [80.0], [79.0], [81.0], [80.0], [83.0], [77.0], [80.0], [80.0]]
    )
    bg = np.full((10, 1), 72.0)  # dL ~ 8.4 dB -> above the 6 dB criterion
    res = emission.sound_power_pressure(
        src, "hemisphere", radius=5.0, background_levels=bg
    )
    mean_level = 10.0 * np.log10(np.mean(10.0 ** (0.1 * src[:, 0])))
    expected = (src[:, 0] - mean_level)[:, np.newaxis]  # DI is (NM, NB)
    assert res.directivity_index.shape == (10, 1)
    assert np.allclose(res.directivity_index, expected, atol=1e-9)
    # And the DI differences are invariant to the background correction: the
    # same source with negligible background gives an identical DI.
    res_nobg = emission.sound_power_pressure(src, "hemisphere", radius=5.0)
    assert np.allclose(res.directivity_index, res_nobg.directivity_index, atol=1e-9)


def test_directivity_index_per_band_localises_a_hot_band() -> None:
    """ISO 3744 clause 8.6: DI is per band, shape (NM, NB). A position that is
    hotter only in one band must show a positive DI in that band and ~0 in the
    others; the other positions stay ~0 outside that band.
    """
    n_pos, n_bands = 10, 4
    levels = np.full((n_pos, n_bands), 70.0)
    levels[3, 2] += 12.0  # position 3 hot only in band 2
    res = emission.sound_power_pressure(levels, "hemisphere", radius=2.0)
    di = res.directivity_index
    assert di.shape == (n_pos, n_bands)
    # Band 2: the hot position stands out, others dip slightly below the mean.
    assert di[3, 2] > 5.0
    assert np.all(di[np.arange(n_pos) != 3, 2] < 0.0)
    # Every other band is perfectly uniform -> DI == 0 everywhere.
    for b in (0, 1, 3):
        assert np.allclose(di[:, b], 0.0, atol=1e-9)


def test_directivity_index_per_band_equals_level_minus_energy_mean() -> None:
    """DIi*(k) reduces to the raw per-band level minus the per-band energy mean
    (the uniform per-band K1 cancels; ISO 3744 Eq. 7).
    """
    rng = np.random.default_rng(0)
    levels = 70.0 + rng.uniform(-3.0, 3.0, size=(10, 3))
    res = emission.sound_power_pressure(levels, "hemisphere", radius=2.0)
    mean = 10.0 * np.log10(np.mean(10.0 ** (0.1 * levels), axis=0))
    assert np.allclose(res.directivity_index, levels - mean[np.newaxis, :], atol=1e-9)


def test_background_levels_single_spectrum_broadcasts() -> None:
    """A single background spectrum (NB,) or (1, NB) is broadcast to every
    position and gives the same K1 as the equivalent full (NM, NB) array.
    """
    levels = np.tile(np.array([90.0, 92.0, 95.0]), (10, 1))
    bg_spectrum = np.array([70.0, 71.0, 72.0])
    res_1d = emission.sound_power_pressure(
        levels, "hemisphere", radius=2.0, background_levels=bg_spectrum
    )
    res_row = emission.sound_power_pressure(
        levels, "hemisphere", radius=2.0, background_levels=bg_spectrum[np.newaxis, :]
    )
    res_full = emission.sound_power_pressure(
        levels,
        "hemisphere",
        radius=2.0,
        background_levels=np.tile(bg_spectrum, (10, 1)),
    )
    assert np.allclose(res_1d.background_correction, res_full.background_correction)
    assert np.allclose(res_row.background_correction, res_full.background_correction)
    assert np.allclose(res_1d.sound_power_level, res_full.sound_power_level)


def test_background_levels_wrong_length_raises() -> None:
    levels = np.tile(np.array([90.0, 92.0, 95.0]), (10, 1))
    background_two_bands = np.array([70.0, 71.0])
    with pytest.raises(ValueError, match="background_levels"):
        emission.sound_power_pressure(
            levels, "hemisphere", radius=2.0, background_levels=background_two_bands
        )


def test_k2_per_band_from_absorption_area() -> None:
    """A per-band absorption area yields a per-band K2 (ISO 3744 Eq. A.2)."""
    a = np.array([100.0, 200.0, 400.0])
    k2 = emission.environmental_correction(50.0, absorption_area=a)
    expected = 10.0 * np.log10(1.0 + 4.0 * 50.0 / a)
    assert isinstance(k2, np.ndarray)
    assert np.allclose(k2, expected)


def test_k2_per_band_eq_a7_from_mean_absorption() -> None:
    """Eq. A.7: A = alpha*Sv per band -> K2 per band, and matches the scalar
    branch band by band.
    """
    alpha = np.array([0.1, 0.2, 0.4])
    sv = 500.0
    k2 = emission.environmental_correction(
        50.0, mean_absorption_coefficient=alpha, room_surface=sv
    )
    expected = np.array(
        [
            emission.environmental_correction(50.0, absorption_area=float(al) * sv)
            for al in alpha
        ]
    )
    assert isinstance(k2, np.ndarray)
    assert np.allclose(k2, expected)


def test_k2_per_band_from_reverberation_time() -> None:
    """Eq. A.3: A = 0,16*V/T per band gives a per-band K2."""
    t = np.array([1.0, 2.0, 4.0])
    v = 300.0
    k2 = emission.environmental_correction(40.0, reverberation_time=t, volume=v)
    expected = 10.0 * np.log10(1.0 + 4.0 * 40.0 / (0.16 * v / t))
    assert np.allclose(k2, expected)


def test_k2_scalar_still_returns_float() -> None:
    """Scalar inputs keep the original scalar (float) return type."""
    k2 = emission.environmental_correction(50.0, absorption_area=200.0)
    assert isinstance(k2, float)


def test_per_band_k2_flows_into_sound_power() -> None:
    """A per-band K2 (from per-band reverberation time) is applied band by band
    in the full determination.
    """
    levels = np.tile(np.array([90.0, 92.0, 95.0]), (10, 1))
    t = np.array([1.0, 2.0, 4.0])
    volume = 2000.0  # large enough that per-band K2 stays under the 4 dB limit
    res = emission.sound_power_pressure(
        levels,
        "hemisphere",
        radius=2.0,
        room=emission.RoomEnvironment(reverberation_time=t, volume=volume),
    )
    a = 0.16 * volume / t
    expected_k2 = 10.0 * np.log10(1.0 + 4.0 * res.surface_area / a)
    assert np.allclose(res.environmental_correction, expected_k2)
    # The three K2 values differ band to band (per-band absorption).
    assert res.environmental_correction[0] != res.environmental_correction[2]


def test_sound_power_level_a_multiband_without_frequencies_is_nan() -> None:
    """Multi-band input without 'frequencies' cannot be A-weighted, so the
    A-weighted field is NaN (A-weighting needs the band centre frequencies).
    """
    levels = np.tile(np.array([90.0, 92.0, 95.0]), (10, 1))
    res = emission.sound_power_pressure(levels, "hemisphere", radius=2.0)
    assert np.isnan(res.sound_power_level_a)


def test_sound_power_level_a_single_band_equals_level() -> None:
    """A single band without 'frequencies' keeps LWA = LW (no weighting)."""
    res = emission.sound_power_pressure(
        np.full((10, 1), 60.0), "hemisphere", radius=2.0
    )
    assert res.sound_power_level_a == pytest.approx(float(res.sound_power_level[0]))


# --------------------------------------------------------------------------
# validations
# --------------------------------------------------------------------------
def test_too_few_positions_raises() -> None:
    """Engineering hemisphere needs >= 10 positions (ISO 3744 clause 8.1.1)."""
    five_positions = np.full((5, 1), 60.0)
    with pytest.raises(ValueError, match="microphone positions"):
        emission.sound_power_pressure(five_positions, "hemisphere", radius=2.0)


def test_survey_allows_four_positions() -> None:
    """Survey hemisphere accepts 4 positions (ISO 3746 clause 8.2.1)."""
    res = emission.sound_power_pressure(
        np.full((4, 1), 60.0), "hemisphere", radius=2.0, grade="survey"
    )
    assert res.grade == "survey"


def test_missing_radius_raises() -> None:
    levels = np.full((10, 1), 60.0)
    with pytest.raises(ValueError, match="'radius' is required"):
        emission.sound_power_pressure(levels, "hemisphere")


def test_invalid_surface_raises() -> None:
    levels = np.full((10, 1), 60.0)
    with pytest.raises(ValueError, match="'surface' must be"):
        emission.sound_power_pressure(levels, "sphere", radius=2.0)


def test_k2_over_validity_limit_warns() -> None:
    """K2 above 4 dB (engineering) exceeds the ISO 3744 validity limit."""
    r = 2.0
    with pytest.warns(emission.SoundPowerWarning):
        emission.sound_power_pressure(
            np.full((10, 1), 60.0),
            "hemisphere",
            radius=r,
            room=emission.RoomEnvironment(absorption_area=10.0),
        )


def test_plot_language_spanish_and_validation() -> None:
    # The sound-power .plot() renderer localises to Spanish and rejects an
    # unknown language code; English remains the unchanged default.
    levels = np.full((10, 1), 70.0)
    res = emission.sound_power_pressure(levels, "hemisphere", radius=2.0)
    ax = res.plot(language="es")
    assert "potencia acústica" in ax.get_ylabel()
    assert "espectro de potencia acústica" in ax.get_title()
    ax_en = res.plot()
    assert ax_en.get_ylabel() == "Sound power level $L_W$ [dB]"
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="xx")


# --------------------------------------------------------------------------
# Per-band quantities that do not run over the band axis
# --------------------------------------------------------------------------
def _ten_position_determination() -> object:
    """A ten-position, four-band ISO 3744 determination."""
    return emission.sound_power_pressure(
        np.tile(np.array([80.0, 78.0, 76.0, 74.0]), (10, 1)),
        "hemisphere",
        radius=2.0,
        frequencies=np.array([250.0, 500.0, 1000.0, 2000.0]),
    )


@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_a_per_band_quantity_off_the_band_axis_is_refused(trim: bool) -> None:
    """The boxed A-weighted total sums every band of ``sound_power_level``.

    A spectrum of another length than the band axis gives a sheet whose
    headline number sums bands its own table does not print.
    """
    import dataclasses

    result = _ten_position_determination()
    values = np.asarray(result.sound_power_level)
    wrong = values[:-1] if trim else np.append(values, values[-1])
    with pytest.raises(ValueError, match="'sound_power_level'"):
        dataclasses.replace(result, sound_power_level=wrong)


def test_the_directivity_index_is_pinned_on_its_band_axis() -> None:
    """``directivity_index`` is positions by bands: axis 1 carries the bands."""
    import dataclasses

    result = _ten_position_determination()
    with pytest.raises(ValueError, match=r"'directivity_index \(axis 1\)'"):
        dataclasses.replace(result, directivity_index=result.directivity_index[:, :-1])
