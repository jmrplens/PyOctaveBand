#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
In-situ road-surface sound absorption (ISO 13472-1 / ISO 13472-2).

Neither standard gives a computable narrow-band worked example, so the tests
anchor on physics identities and the numeric oracles captured in the Step-0
transcription:

- Geometry: Kr = (ds - dm)/(ds + dm) = 2/3 for the mandatory ds = 1.25 m,
  dm = 0.25 m; Kr,theta reduces to Kr at theta = 0 (Annex F).
- Reflection round trips: hr = Kr * delayed(hi) -> |r| = 1, alpha = 0 (perfect
  reflector); hr = Kr * r0 * delayed(hi) -> alpha = 1 - r0^2 exactly, via both
  the reflection-factor route and the direct energy route (Clause 4.1).
- Reference correction (Annex B): dividing road by reference reflection cancels
  Kr and the chain error.
- Adrienne window (Clause 6.4): total length, unit flat top, ~0 endpoints,
  sharp (short) leading edge, most energy in the flat portion.
- MSA radius (Annex A): r ~ 1.34 m for ds/dm/c = 1.25/0.25/340 and Tw = 5 ms.
- Annex E one-third-octave spectrum: 13 bands 250-4000 Hz.
- Spot method (Part 2): f_u = 0.58 c0/d for a 100 mm tube; spacing bounds
  85/77 mm; internal-loss subtraction; 250-1600 Hz range guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import phonometry.materials.surfaces.road_absorption as road
from phonometry.materials.surfaces.road_absorption import (
    DEFAULT_MIC_HEIGHT,
    DEFAULT_SOURCE_HEIGHT,
    PART1_FREQUENCY_RANGE,
    SPOT_FREQUENCY_RANGE,
    SPOT_NARROW_BAND_RANGE,
    InsituAbsorptionResult,
    RoadAbsorptionWarning,
    absorption_reference_corrected,
    adrienne_window,
    check_spot_frequency_range,
    geometric_spreading_factor,
    geometric_spreading_factor_angle,
    insitu_absorption_coefficient,
    insitu_absorption_from_reflection,
    insitu_absorption_spectrum,
    insitu_reflection_factor,
    max_sampled_area_radius,
    msa_major_axis,
    one_third_octave_absorption,
    power_reflection_coefficient,
    reflected_path_delay,
    spot_internal_loss_correction,
    spot_microphone_spacing_bounds,
    spot_tube_upper_frequency,
)

FS = 48_000.0


def _incident_ir(n: int = 4096) -> np.ndarray:
    """A short band-limited direct-path impulse response (arbitrary shape)."""
    rng = np.random.default_rng(1234)
    t = np.arange(n) / FS
    ir = np.zeros(n)
    ir[:64] = np.hanning(64) * np.cos(2.0 * np.pi * 1500.0 * t[:64])
    ir += 1e-6 * rng.standard_normal(n)
    return ir


# --------------------------------------------------------------------------- #
# __all__ / public API
# --------------------------------------------------------------------------- #
def test_module_exports_match_all() -> None:
    for name in road.__all__:
        assert hasattr(road, name), name
    # Every documented public callable/constant is exported.
    for name in (
        "adrienne_window",
        "insitu_reflection_factor",
        "insitu_absorption_coefficient",
        "absorption_reference_corrected",
        "geometric_spreading_factor_angle",
        "max_sampled_area_radius",
        "spot_tube_upper_frequency",
        "spot_internal_loss_correction",
    ):
        assert name in road.__all__


# --------------------------------------------------------------------------- #
# Geometry (Clause 4.1 / Annex F)
# --------------------------------------------------------------------------- #
def test_default_kr_is_two_thirds() -> None:
    kr = geometric_spreading_factor(DEFAULT_SOURCE_HEIGHT, DEFAULT_MIC_HEIGHT)
    assert kr == pytest.approx(2.0 / 3.0)
    assert kr == pytest.approx(0.6666666666, abs=1e-9)


def test_kr_angle_reduces_to_kr_at_normal_incidence() -> None:
    kr = geometric_spreading_factor()
    kr_theta = geometric_spreading_factor_angle(0.0)
    assert kr_theta == pytest.approx(kr)


def test_kr_angle_grows_toward_unity_at_grazing() -> None:
    kr = geometric_spreading_factor()
    kr_60 = geometric_spreading_factor_angle(np.pi / 3.0)
    # Kr,theta^2 = 1 - cos^2 (1 - Kr^2); check against the closed form.
    expected = np.sqrt(1.0 - 0.25 * (1.0 - kr**2))
    assert kr_60 == pytest.approx(expected)
    assert kr < kr_60 < 1.0


def test_reflected_path_delay() -> None:
    assert reflected_path_delay(0.25, 340.0) == pytest.approx(2.0 * 0.25 / 340.0)


def test_geometry_guards() -> None:
    with pytest.raises(
        ValueError, match="'source_height' and 'mic_height' must be positive"
    ):
        geometric_spreading_factor(0.0, 0.25)
    with pytest.raises(
        ValueError, match="'source_height' must exceed 'mic_height'"
    ):
        geometric_spreading_factor(0.25, 1.25)  # source must exceed mic


# --------------------------------------------------------------------------- #
# Reflection factor / absorption round trips (Clause 4.1 / Annex C)
# --------------------------------------------------------------------------- #
def test_perfect_reflector_gives_unit_reflection_zero_absorption() -> None:
    hi = _incident_ir()
    kr = geometric_spreading_factor()
    shift = 96
    hr = kr * np.roll(hi, shift)  # circular shift preserves |spectrum|
    r = insitu_reflection_factor(hi, hr)
    np.testing.assert_allclose(np.abs(r)[1:], 1.0, atol=1e-9)
    alpha = insitu_absorption_coefficient(hi, hr)
    np.testing.assert_allclose(alpha[1:], 0.0, atol=1e-9)


def test_synthetic_absorber_round_trip() -> None:
    hi = _incident_ir()
    kr = geometric_spreading_factor()
    r0 = 0.4
    hr = kr * r0 * np.roll(hi, 96)
    # Reflection-factor route.
    r = insitu_reflection_factor(hi, hr)
    np.testing.assert_allclose(np.abs(r)[1:], r0, atol=1e-9)
    alpha_refl = insitu_absorption_from_reflection(r)
    # Direct energy route.
    alpha_energy = insitu_absorption_coefficient(hi, hr)
    np.testing.assert_allclose(alpha_refl, alpha_energy, atol=1e-12)
    np.testing.assert_allclose(alpha_energy[1:], 1.0 - r0**2, atol=1e-9)


def test_power_reflection_matches_reflection_magnitude() -> None:
    hi = _incident_ir()
    kr = geometric_spreading_factor()
    hr = kr * 0.5 * np.roll(hi, 80)
    qw = power_reflection_coefficient(hi, hr)
    r = insitu_reflection_factor(hi, hr)
    np.testing.assert_allclose(qw, np.abs(r) ** 2, atol=1e-12)


def test_phase_restoration_recovers_real_reflection() -> None:
    hi = _incident_ir()
    kr = geometric_spreading_factor()
    shift = 96
    delay = shift / FS
    r0 = 0.6
    hr = kr * r0 * np.roll(hi, shift)
    r = insitu_reflection_factor(hi, hr, fs=FS, delay=delay)
    # After undoing exp(-j2pi f tau) the reflection factor is ~ real r0.
    np.testing.assert_allclose(r.real[1:], r0, atol=1e-6)
    np.testing.assert_allclose(r.imag[1:], 0.0, atol=1e-6)


def test_phase_restoration_recovers_real_reflection_odd_length() -> None:
    # Odd-length inputs: rfftfreq must use the true time-domain length, not the
    # even length reconstructed from the bin count, or the restored phase drifts.
    hi = _incident_ir(4097)
    kr = geometric_spreading_factor()
    shift = 96
    delay = shift / FS
    r0 = 0.6
    hr = kr * r0 * np.roll(hi, shift)
    r = insitu_reflection_factor(hi, hr, fs=FS, delay=delay)
    np.testing.assert_allclose(r.real[1:], r0, atol=1e-6)
    np.testing.assert_allclose(r.imag[1:], 0.0, atol=1e-6)


def test_oblique_incidence_uses_kr_theta() -> None:
    hi = _incident_ir()
    theta = np.pi / 4.0
    kr_theta = geometric_spreading_factor_angle(theta)
    hr = kr_theta * 0.3 * np.roll(hi, 64)
    alpha = insitu_absorption_coefficient(hi, hr, incidence_angle=theta)
    np.testing.assert_allclose(alpha[1:], 1.0 - 0.3**2, atol=1e-9)


def test_reference_correction_cancels_kr_and_chain(recwarn: pytest.WarningsRecorder) -> None:
    # Road measured reflection = chain error e(f) * Kr-scaled road reflection;
    # reference measured = same e(f). Ratio recovers |Qp,road|^2.
    freq = np.linspace(250.0, 1600.0, 64)
    e = (0.9 + 0.1j) * np.exp(1j * freq / 400.0)  # arbitrary chain error
    q_road_true = 0.3 * np.exp(1j * freq / 900.0)
    q_ref_meas = e  # totally reflecting reference, Qp,ref = 1
    q_road_meas = q_road_true * e
    alpha = absorption_reference_corrected(q_road_meas, q_ref_meas)
    np.testing.assert_allclose(alpha, 1.0 - np.abs(q_road_true) ** 2, atol=1e-12)


def test_reflection_input_guards() -> None:
    with pytest.raises(ValueError, match="Impulse responses must be non-empty"):
        insitu_reflection_factor([], [1.0, 2.0])
    with pytest.raises(ValueError, match="'fs' is required to apply 'delay'"):
        insitu_reflection_factor([1.0, 2.0], [1.0, 2.0], delay=1e-3)  # no fs


# --------------------------------------------------------------------------- #
# Adrienne temporal window (Clause 6.4)
# --------------------------------------------------------------------------- #
def test_adrienne_window_length_and_flat_top() -> None:
    w = adrienne_window(
        FS, flat_duration=5e-3, leading_duration=0.5e-3, trailing_duration=5e-3
    )
    n_lead = round(0.5e-3 * FS)
    n_flat = round(5e-3 * FS)
    n_trail = round(5e-3 * FS)
    assert w.shape == (n_lead + n_flat + n_trail,)
    # Flat portion is exactly unity.
    np.testing.assert_allclose(w[n_lead:n_lead + n_flat], 1.0, atol=1e-12)


def test_adrienne_window_endpoints_and_sharp_leading_edge() -> None:
    w = adrienne_window(
        FS, flat_duration=5e-3, leading_duration=0.5e-3, trailing_duration=5e-3
    )
    n_lead = round(0.5e-3 * FS)
    n_trail = round(5e-3 * FS)
    # Cosine-squared/Blackman-Harris edges start and end near zero.
    assert w[0] < 1e-3
    assert w[-1] < 1e-3
    # Trailing edge falls monotonically 1 -> 0.
    trailing = w[-n_trail:]
    assert np.all(np.diff(trailing) <= 1e-9)
    # Leading edge is sharp: far fewer samples than the trailing edge.
    assert n_lead < n_trail
    # It rises monotonically 0 -> 1.
    leading = w[:n_lead]
    assert np.all(np.diff(leading) >= -1e-9)


def test_adrienne_window_energy_in_flat_region() -> None:
    w = adrienne_window(FS)
    n_lead = round(0.5e-3 * FS)
    n_flat = round(5e-3 * FS)
    flat_energy = np.sum(w[n_lead:n_lead + n_flat] ** 2)
    assert flat_energy > 0.5 * np.sum(w**2)


def test_adrienne_window_cosine_squared_shape() -> None:
    w = adrienne_window(
        FS,
        leading_duration=0.0,
        trailing_duration=2e-3,
        trailing_edge="cosine-squared",
    )
    n_flat = round(5e-3 * FS)
    # No leading edge -> starts flat at unity.
    assert w[0] == pytest.approx(1.0)
    # cos^2 trailing edge reaches exactly zero at the final sample.
    assert w[-1] == pytest.approx(0.0, abs=1e-12)
    assert w.shape[0] == n_flat + round(2e-3 * FS)


def test_adrienne_blackman_harris_edges_meet_flat_at_unity() -> None:
    # The Blackman-Harris half-tapers must reach exactly 1 where they join the
    # flat top, so the flat-to-edge transition is continuous (no ~0.26 % step).
    w = adrienne_window(
        FS,
        leading_duration=1e-3,
        flat_duration=5e-3,
        trailing_duration=3e-3,
        leading_edge="blackman-harris",
        trailing_edge="blackman-harris",
    )
    n_lead = round(1e-3 * FS)
    n_trail = round(3e-3 * FS)
    assert w[n_lead - 1] == pytest.approx(1.0, abs=1e-12)  # end of rising edge
    assert w[-n_trail] == pytest.approx(1.0, abs=1e-12)  # start of falling edge
    # Edges still start/end near zero and stay within [0, 1].
    assert w[0] < 1e-3
    assert w[-1] < 1e-3
    assert np.all(w <= 1.0 + 1e-12)


def test_adrienne_window_guards() -> None:
    with pytest.raises(ValueError, match="'fs' must be positive"):
        adrienne_window(0.0)
    with pytest.raises(ValueError, match="'flat_duration' must be positive"):
        adrienne_window(FS, flat_duration=0.0)
    with pytest.raises(ValueError, match="Edge durations must be non-negative"):
        adrienne_window(FS, leading_duration=-1e-3)
    with pytest.raises(ValueError, match="Edge shapes must be one of"):
        adrienne_window(FS, trailing_edge="triangular")


# --------------------------------------------------------------------------- #
# One-third-octave presentation (Clause 4.1 / Clause 6.6)
# --------------------------------------------------------------------------- #
def test_one_third_octave_band_count_part1() -> None:
    freq = np.linspace(100.0, 6000.0, 8000)
    alpha = np.full_like(freq, 0.3)
    centres, band = one_third_octave_absorption(freq, alpha)
    assert centres.shape == (13,)  # 250..4000 Hz, Annex E
    assert centres[0] == 250.0 and centres[-1] == 4000.0
    np.testing.assert_allclose(band, 0.3, atol=1e-9)


def test_one_third_octave_band_count_spot() -> None:
    freq = np.linspace(100.0, 3000.0, 6000)
    alpha = np.full_like(freq, 0.05)
    centres, _ = one_third_octave_absorption(
        freq, alpha, f_max=SPOT_FREQUENCY_RANGE[1]
    )
    assert centres.shape == (9,)  # 250..1600 Hz (Annex D of Part 2)


def test_one_third_octave_clipping() -> None:
    freq = np.linspace(200.0, 4500.0, 6000)
    alpha = np.full_like(freq, -0.02)
    _, clipped = one_third_octave_absorption(freq, alpha, clip_negative=True)
    _, raw = one_third_octave_absorption(freq, alpha, clip_negative=False)
    np.testing.assert_allclose(clipped, 0.0, atol=1e-12)
    assert np.all(raw < 0.0)


def test_one_third_octave_guards() -> None:
    with pytest.raises(
        ValueError, match="must be non-empty and equal-length"
    ):
        one_third_octave_absorption([250.0, 500.0], [0.1])
    with pytest.raises(
        ValueError, match="must be non-empty and equal-length"
    ):
        one_third_octave_absorption([], [])


# --------------------------------------------------------------------------- #
# Plottable end-to-end spectrum (insitu_absorption_spectrum)
# --------------------------------------------------------------------------- #
def test_insitu_absorption_spectrum_mid_bands_are_one_minus_r0_sq() -> None:
    hi = _incident_ir()
    kr = geometric_spreading_factor()
    r0 = 0.5
    # hr = Kr * r0 * delayed(hi): a flat reflection of magnitude r0, so
    # |Hr/Hi| = Kr*r0 and alpha = 1 - (1/Kr^2)(Kr*r0)^2 = 1 - r0^2 in every band.
    hr = kr * r0 * np.roll(hi, 96)
    result = insitu_absorption_spectrum(hi, hr, FS)

    assert isinstance(result, InsituAbsorptionResult)
    assert result.frequencies.shape == (13,)  # 250..4000 Hz
    np.testing.assert_allclose(result.absorption, 1.0 - r0**2, atol=1e-6)


def test_insitu_absorption_spectrum_rejects_nonpositive_fs() -> None:
    hi = _incident_ir()
    hr = 0.4 * np.roll(hi, 96)
    with pytest.raises(ValueError, match="'fs' must be positive"):
        insitu_absorption_spectrum(hi, hr, 0.0)
    with pytest.raises(ValueError, match="'fs' must be positive"):
        insitu_absorption_spectrum(hi, hr, -48000.0)


def test_insitu_absorption_spectrum_plot_returns_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi = _incident_ir()
    hr = geometric_spreading_factor() * 0.5 * np.roll(hi, 96)
    result = insitu_absorption_spectrum(hi, hr, FS)
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


# --------------------------------------------------------------------------- #
# Maximum sampled area (Annex A / Annex F)
# --------------------------------------------------------------------------- #
def test_msa_radius_annex_a_oracle() -> None:
    # ds=1.25, dm=0.25, c=340, Tw=5 ms -> r ~ 1.34 m (Annex A worked example).
    r = max_sampled_area_radius(
        5e-3,
        source_height=1.25,
        mic_height=0.25,
        speed_of_sound=340.0,
    )
    assert r == pytest.approx(1.34, abs=0.005)


def test_msa_major_axis_reduces_to_normal_at_zero_projection() -> None:
    tw, c = 5e-3, 340.0
    a = msa_major_axis(tw, 0.0, source_height=1.25, mic_height=0.25, speed_of_sound=c)
    # dp = 0 -> a = c*Tw + (ds + dm).
    assert a == pytest.approx(c * tw + 1.5)


def test_msa_guards() -> None:
    with pytest.raises(ValueError, match="'window_width' must be positive"):
        max_sampled_area_radius(0.0)
    with pytest.raises(
        ValueError, match="'projected_distance' must be non-negative"
    ):
        msa_major_axis(5e-3, -1.0)


# --------------------------------------------------------------------------- #
# ISO 13472-2 spot method helpers (Clause 5.4 / Annex A)
# --------------------------------------------------------------------------- #
def test_spot_upper_frequency_100mm_tube() -> None:
    fu = spot_tube_upper_frequency(0.100, 343.0)
    assert fu == pytest.approx(1989.4, abs=0.1)  # 0.58*343/0.1


def test_spot_spacing_bounds() -> None:
    s_min, s_max = spot_microphone_spacing_bounds(
        340.0, f_min=220.0, f_max=1800.0
    )
    assert s_max == pytest.approx(0.085, abs=1e-4)  # 85 mm
    assert s_min == pytest.approx(0.07727, abs=1e-4)  # 77 mm
    assert s_min < 0.081 < s_max  # brackets nominal 81 mm spacing


def test_spot_spacing_bounds_warns_when_interval_empty() -> None:
    # A range far wider than the narrow band leaves no valid spacing (s_min>=s_max).
    with pytest.warns(RoadAbsorptionWarning):
        s_min, s_max = spot_microphone_spacing_bounds(
            340.0, f_min=220.0, f_max=4000.0
        )
    assert s_min >= s_max


def test_spot_frequency_range_guard_warns() -> None:
    with pytest.warns(RoadAbsorptionWarning):
        check_spot_frequency_range([200.0, 500.0, 2000.0])
    # In-range frequencies do not warn.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_spot_frequency_range([250.0, 1000.0, 1600.0])


def test_spot_frequency_range_constants() -> None:
    assert SPOT_FREQUENCY_RANGE == (250.0, 1600.0)
    assert SPOT_NARROW_BAND_RANGE == (220.0, 1800.0)
    assert PART1_FREQUENCY_RANGE == (250.0, 4000.0)


def test_spot_internal_loss_correction() -> None:
    measured = np.array([0.06, 0.04, 0.03, 0.05])
    system = np.array([0.02, 0.01, 0.02, 0.01])
    corrected = spot_internal_loss_correction(measured, system)
    np.testing.assert_allclose(corrected, measured - system, atol=1e-12)


def test_spot_internal_loss_clips_negative() -> None:
    corrected = spot_internal_loss_correction([0.01, 0.02], [0.03, 0.005])
    np.testing.assert_allclose(corrected, [0.0, 0.015], atol=1e-12)
    raw = spot_internal_loss_correction(
        [0.01, 0.02], [0.03, 0.005], clip_negative=False
    )
    assert raw[0] < 0.0


def test_spot_guards() -> None:
    with pytest.raises(ValueError, match="'diameter' must be positive"):
        spot_tube_upper_frequency(0.0)
    with pytest.raises(
        ValueError, match="'f_min' must be less than 'f_max'"
    ):
        spot_microphone_spacing_bounds(340.0, f_min=1800.0, f_max=220.0)
    with pytest.raises(ValueError, match="must share a shape"):
        spot_internal_loss_correction([0.1, 0.2], [0.1])


def test_public_exports() -> None:
    import phonometry

    for name in (
        "adrienne_window", "geometric_spreading_factor",
        "geometric_spreading_factor_angle", "reflected_path_delay",
        "insitu_reflection_factor", "insitu_absorption_from_reflection",
        "power_reflection_coefficient", "insitu_absorption_coefficient",
        "absorption_reference_corrected", "one_third_octave_absorption",
        "max_sampled_area_radius", "msa_major_axis", "spot_tube_upper_frequency",
        "spot_microphone_spacing_bounds", "check_spot_frequency_range",
        "spot_internal_loss_correction", "RoadAbsorptionWarning",
    ):
        assert hasattr(phonometry, name), name
