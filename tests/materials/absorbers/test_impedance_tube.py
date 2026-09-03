#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Impedance-tube material characterisation (ISO 10534-1/-2, ASTM E2611-19).

No worked numeric example exists in either standard, so the tests anchor on
physics identities:

- Rigid wall (r = 1) -> alpha = 0, infinite impedance; perfect absorber
  (r = 0) -> alpha = 1, Z = rho c; half-absorbing (r = 0.5) -> alpha = 0.75.
- ISO 10534-2 round trip: synthesise H12 from a known r via the Annex D field
  equations (D.7) and recover r from reflection_factor().
- ASTM E2611 analytic air layer T = [[cos kd, j rho c sin kd],
  [j sin kd/(rho c), cos kd]]: det(T) = 1, T11 = T22, TL ~ 0 dB; and a full
  synthesise-then-recover round trip of the four-microphone reduction (this
  also locks down the Eq. (17)-(20) microphone-numbering / l1,l2,s1,s2
  geometry convention).
- Speed of sound ~ 343.2 m/s at 20 degC by both the ISO (kelvin) and ASTM
  (Celsius) formulas.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from phonometry.materials.absorbers.four_microphone import (
    TransferMatrix,
    air_density_astm,
    air_layer_transfer_matrix,
    face_quantities,
    plane_wave_frequency_range_astm,
    speed_of_sound_astm,
    transfer_matrix_one_load,
    transfer_matrix_two_load,
    wave_decomposition,
)
from phonometry.materials.absorbers.impedance_tube import (
    ImpedanceTubeResult,
    ImpedanceTubeWarning,
    absorption_from_reflection,
    air_density_iso10534,
    apply_mic_calibration,
    characteristic_impedance,
    hydraulic_diameter,
    mic_calibration_factor,
    normalized_surface_admittance,
    normalized_surface_impedance,
    plane_wave_frequency_range,
    reflection_factor,
    speed_of_sound_iso10534,
    surface_impedance,
    tube_attenuation_constant,
    tube_wavenumber,
    two_microphone_impedance,
)
from phonometry.materials.absorbers.standing_wave import (
    standing_wave_absorption,
    standing_wave_normalized_impedance,
    standing_wave_ratio_from_level,
    standing_wave_reflection,
    standing_wave_reflection_magnitude,
)

RHO = 1.204
C0 = 343.24
RC = RHO * C0


# ---------------------------------------------------------------------------
# Air properties.
# ---------------------------------------------------------------------------
def test_speed_of_sound_iso_20c() -> None:
    # ISO Eq. (5) c0 = 343.2*sqrt(T/293), with T = 273.15 + t.
    c20 = float(speed_of_sound_iso10534(temperature_c=20.0))
    assert c20 == pytest.approx(343.2, abs=0.1)
    # 19,85 degC is the reference 293 K exactly, so the constant comes back.
    c_ref = float(speed_of_sound_iso10534(temperature_c=19.85))
    assert c_ref == pytest.approx(343.2, abs=1e-9)


def test_speed_of_sound_astm_20c() -> None:
    # ASTM Eq. (4) c = 20.047*sqrt(273.15 + T), T in degC.
    assert float(speed_of_sound_astm(temperature_c=20.0)) == pytest.approx(
        343.24, abs=0.02
    )


def test_both_speed_formulas_agree_at_20c() -> None:
    iso = float(speed_of_sound_iso10534(temperature_c=20.0))
    astm = float(speed_of_sound_astm(temperature_c=20.0))
    assert iso == pytest.approx(astm, abs=0.2)


def test_air_density_reference_values() -> None:
    # ISO Eq. (7): rho0 at the reference state (19,85 degC is 293 K exactly).
    rho_iso = air_density_iso10534(
        temperature_c=19.85, atmospheric_pressure_kpa=101.325
    )
    assert float(rho_iso) == pytest.approx(1.186, abs=1e-9)
    # ASTM Eq. (5) at 20 degC, 101.325 kPa -> 1.2020 kg/m3 (notes 2.11).
    rho_astm = air_density_astm(temperature_c=20.0, atmospheric_pressure_kpa=101.325)
    assert float(rho_astm) == pytest.approx(1.2020, abs=1e-3)


def test_air_property_domain_errors() -> None:
    with pytest.raises(ValueError, match="'temperature_c' must exceed"):
        speed_of_sound_iso10534(temperature_c=-300.0)
    with pytest.raises(ValueError, match="'temperature_c' must exceed"):
        speed_of_sound_astm(temperature_c=-300.0)
    with pytest.raises(ValueError, match="'atmospheric_pressure_kpa' must be positive"):
        air_density_iso10534(temperature_c=19.85, atmospheric_pressure_kpa=-1.0)
    with pytest.raises(ValueError, match="'atmospheric_pressure_kpa' must be positive"):
        air_density_astm(temperature_c=20.0, atmospheric_pressure_kpa=-1.0)


def test_air_density_astm_rejects_temperature_below_absolute_zero() -> None:
    """ASTM Eq. (5) scales the reference density by 273,15/(273,15 + T).

    Below absolute zero that ratio is negative, so rho would come back
    negative and carry the sign into every rho c the four-microphone
    reduction builds on. The temperature is refused at the door, exactly as
    ``speed_of_sound_astm`` refuses it above.
    """
    with pytest.raises(ValueError, match="'temperature_c' must exceed"):
        air_density_astm(temperature_c=-300.0, atmospheric_pressure_kpa=101.325)


def test_air_density_mismatched_arrays_raise() -> None:
    # Two arrays of different lengths must be refused by name, not left to
    # numpy's two-shapes broadcast error.
    with pytest.raises(ValueError, match="'temperature_c'.*'atmospheric_pressure_kpa'"):
        air_density_iso10534(
            temperature_c=[19.85, 26.85],
            atmospheric_pressure_kpa=[101.0, 100.0, 99.0],
        )


_AIR_HELPERS = (
    (speed_of_sound_iso10534, {}),
    (air_density_iso10534, {}),
    (speed_of_sound_astm, {}),
    (air_density_astm, {}),
)


@pytest.mark.parametrize(("helper", "extra"), _AIR_HELPERS)
def test_air_helpers_refuse_a_positional_temperature(
    helper: object, extra: dict[str, float]
) -> None:
    """The unit lives in the keyword, so it cannot be dropped at the call site.

    Before these were keyword-only, ``speed_of_sound_iso(20.0)`` read as a room
    temperature, was taken as 20 K and returned 89,7 m/s with no exception.
    """
    with pytest.raises(TypeError, match="positional"):
        helper(20.0, **extra)  # type: ignore[operator]


@pytest.mark.parametrize(("helper", "extra"), _AIR_HELPERS)
def test_air_helpers_bound_only_absolute_zero(
    helper: object, extra: dict[str, float]
) -> None:
    """Nothing but the physical bound is refused.

    Neither ISO 10534-2 Eq. (5)/(7) nor ASTM E2611-19 Eq. (4)/(5) states a
    temperature range, so the helpers must accept any temperature that can
    exist. A magnitude guard drawn at laboratory bounds would reject an
    extreme but conforming call, and would still not catch a caller who means
    a different unit.
    """
    for temperature_c in (-272.0, -50.0, 0.0, 100.0, 500.0):
        assert np.isfinite(float(helper(temperature_c=temperature_c, **extra)))  # type: ignore[operator]
    with pytest.raises(ValueError, match="'temperature_c' must exceed"):
        helper(temperature_c=-273.15, **extra)  # type: ignore[operator]


def test_air_density_scalar_beside_array_still_broadcasts() -> None:
    # One shared temperature against several pressures stays legal.
    rho = np.asarray(
        air_density_iso10534(
            temperature_c=19.85, atmospheric_pressure_kpa=[101.325, 100.0]
        )
    )
    assert rho.shape == (2,)
    assert rho[0] == pytest.approx(1.186, abs=1e-9)


def test_characteristic_impedance_is_real_product() -> None:
    assert characteristic_impedance(RHO, C0) == pytest.approx(RC)
    with pytest.raises(
        ValueError, match="'density' and 'speed_of_sound' must be positive"
    ):
        characteristic_impedance(-1.0, C0)


# ---------------------------------------------------------------------------
# ISO 10534-2: reflection / impedance / absorption identities.
# ---------------------------------------------------------------------------
def test_rigid_wall_absorption_zero() -> None:
    assert float(absorption_from_reflection(1.0 + 0.0j)) == pytest.approx(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = normalized_surface_impedance(1.0 + 0.0j)
    assert not np.isfinite(z)


def test_perfect_absorber_absorption_one_and_impedance_rhoc() -> None:
    assert float(absorption_from_reflection(0.0 + 0.0j)) == pytest.approx(1.0)
    assert normalized_surface_impedance(0.0 + 0.0j) == pytest.approx(1.0 + 0.0j)
    assert surface_impedance(0.0 + 0.0j, RC) == pytest.approx(RC + 0.0j)


def test_half_absorbing_absorption_075() -> None:
    assert float(absorption_from_reflection(0.5 + 0.0j)) == pytest.approx(0.75)


def test_admittance_is_reciprocal_of_impedance() -> None:
    r = 0.3 - 0.2j
    z = normalized_surface_impedance(r)
    y = normalized_surface_admittance(r)
    assert complex(z * y) == pytest.approx(1.0 + 0.0j)


def _synth_h12(
    reflection: complex, k0: np.ndarray, x1: float, spacing: float
) -> np.ndarray:
    """H12 from a known r via ISO 10534-2 Annex D, Eq. (D.7)."""
    x2 = x1 - spacing
    num = np.exp(1j * k0 * x2) + reflection * np.exp(-1j * k0 * x2)
    den = np.exp(1j * k0 * x1) + reflection * np.exp(-1j * k0 * x1)
    return np.asarray(num / den, dtype=np.complex128)


@pytest.mark.parametrize(
    "reflection", [0.0 + 0.0j, 0.5 + 0.0j, 0.3 - 0.4j, 0.8 * np.exp(1j * 1.1)]
)
def test_reflection_factor_round_trip(reflection: complex) -> None:
    f = np.array([500.0, 1000.0, 1800.0])
    x1, spacing = 0.12, 0.03
    k0 = tube_wavenumber(f, C0)
    h12 = _synth_h12(reflection, np.asarray(k0), x1, spacing)
    r = reflection_factor(h12, spacing=spacing, x1=x1, wavenumber=k0)
    assert np.allclose(r, reflection, atol=1e-12)


def test_reflection_factor_round_trip_with_attenuation() -> None:
    f = np.array([500.0, 1200.0])
    x1, spacing = 0.12, 0.03
    att = tube_attenuation_constant(f, C0, diameter=0.05)
    k0 = tube_wavenumber(f, C0, attenuation=att)
    reflection = 0.6 - 0.2j
    h12 = _synth_h12(reflection, np.asarray(k0), x1, spacing)
    r = reflection_factor(h12, spacing=spacing, x1=x1, wavenumber=k0)
    assert np.allclose(r, reflection, atol=1e-12)


def test_tube_wavenumber_rejects_negative_frequency() -> None:
    with pytest.raises(ValueError, match="'frequency' must be non-negative"):
        tube_wavenumber(np.array([-1000.0, 500.0]), C0)


def test_tube_wavenumber_rejects_nan_frequency() -> None:
    with pytest.raises(ValueError, match="'frequency' must be non-negative"):
        tube_wavenumber(np.array([500.0, np.nan]), C0)


def test_tube_wavenumber_rejects_mismatched_attenuation() -> None:
    # A two-band attenuation against three frequencies used to die as
    # numpy's two-shapes broadcast error, naming neither argument.
    with pytest.raises(ValueError, match="'attenuation' must be a scalar or carry"):
        tube_wavenumber(np.array([100.0, 200.0, 300.0]), C0, attenuation=[0.1, 0.2])


def test_reflection_factor_rejects_mismatched_h12() -> None:
    with pytest.raises(ValueError, match="'h12' must be a scalar or carry"):
        reflection_factor(
            np.array([0.5 + 0.1j, 0.4 + 0.2j]),
            spacing=0.03,
            x1=0.12,
            wavenumber=np.array([1.0, 2.0, 3.0]),
        )


def test_two_microphone_rejects_mismatched_h12() -> None:
    # Through the full reduction the mismatch is reported against
    # 'frequency', the parameter name the caller actually typed.
    with pytest.raises(ValueError, match="'h12'.*band.*'frequency'"):
        two_microphone_impedance(
            np.array([0.5 + 0.1j, 0.4 + 0.2j]),
            frequency=np.array([500.0, 1000.0, 1500.0]),
            spacing=0.03,
            x1=0.12,
            speed_of_sound=C0,
            characteristic_impedance=RC,
        )


def test_two_microphone_rejects_nan_frequency() -> None:
    # A NaN frequency used to come back as a silently NaN absorption.
    with pytest.raises(ValueError, match="'frequency' must be non-negative"):
        two_microphone_impedance(
            np.array([0.5 + 0.1j, 0.4 + 0.2j]),
            frequency=np.array([-1000.0, np.nan]),
            spacing=0.03,
            x1=0.12,
            speed_of_sound=C0,
            characteristic_impedance=RC,
        )


def test_two_microphone_impedance_result() -> None:
    f = np.array([500.0, 1000.0, 1500.0])
    x1, spacing = 0.12, 0.03
    reflection = 0.4 - 0.3j
    k0 = tube_wavenumber(f, C0)
    h12 = _synth_h12(reflection, np.asarray(k0), x1, spacing)
    res = two_microphone_impedance(
        h12,
        frequency=f,
        spacing=spacing,
        x1=x1,
        speed_of_sound=C0,
        characteristic_impedance=RC,
    )
    assert isinstance(res, ImpedanceTubeResult)
    assert np.allclose(res.reflection, reflection, atol=1e-12)
    assert np.allclose(res.absorption, 1.0 - abs(reflection) ** 2)
    assert np.allclose(res.surface_impedance, RC * res.normalized_impedance)


def test_result_is_frozen() -> None:
    res = two_microphone_impedance(
        np.array([0.5 + 0.1j]),
        frequency=np.array([1000.0]),
        spacing=0.03,
        x1=0.12,
        speed_of_sound=C0,
        characteristic_impedance=RC,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.reflection = np.array([0.0])  # type: ignore[misc]


def test_mic_calibration_identity() -> None:
    # A perfectly matched pair (config I == config II) gives Hc = 1.
    h = np.array([0.7 + 0.3j, 0.5 - 0.2j])
    hc = mic_calibration_factor(h, h)
    assert np.allclose(hc, 1.0 + 0.0j)
    assert np.allclose(apply_mic_calibration(h, hc), h)


def test_mic_calibration_removes_known_mismatch() -> None:
    # Channel mismatch m: config I measures true*m, config II (mics swapped)
    # measures true/m, so Hc = sqrt(H12^I / H12^II) = m and dividing it out of
    # the config-I measurement recovers the true transfer function (Eqs. (10)/(13)).
    true = np.array([0.6 + 0.2j])
    mismatch = 1.05 * np.exp(1j * 0.03)
    h_config1 = true * mismatch
    h_config2 = true / mismatch
    hc = mic_calibration_factor(h_config1, h_config2)
    assert np.allclose(hc, mismatch, atol=1e-12)
    corrected = apply_mic_calibration(h_config1, hc)
    assert np.allclose(corrected, true, atol=1e-12)


# ---------------------------------------------------------------------------
# ISO 10534-2: frequency-range warnings (Eqs. (1)-(4)).
# ---------------------------------------------------------------------------
def test_plane_wave_range_bounds() -> None:
    f_lower, f_upper = plane_wave_frequency_range(
        0.03, C0, diameter=0.05, shape="circular"
    )
    # Upper is min(spacing bound 0.45 c/s, tube bound 0.58 c/d).
    assert f_upper == pytest.approx(min(0.45 * C0 / 0.03, 0.58 * C0 / 0.05))
    assert f_lower == pytest.approx(C0 / (20.0 * 0.03))


def test_frequency_range_warning_fires() -> None:
    f_lower, f_upper = plane_wave_frequency_range(0.03, C0, diameter=0.05)
    f = np.array([f_lower * 0.5, 1000.0, f_upper * 1.5])
    h12 = _synth_h12(0.3 + 0.1j, np.asarray(tube_wavenumber(f, C0)), 0.12, 0.03)
    with pytest.warns(ImpedanceTubeWarning):
        two_microphone_impedance(
            h12,
            frequency=f,
            spacing=0.03,
            x1=0.12,
            speed_of_sound=C0,
            characteristic_impedance=RC,
            diameter=0.05,
        )


def test_plane_wave_range_rectangular_bound() -> None:
    # ISO 10534-2 Eq. (3): a rectangular tube cuts on at 0,50 c0 / d with d the
    # maximum side length (stricter than the 0,58 circular factor of Eq. (2)).
    f_lower, f_upper = plane_wave_frequency_range(
        0.03, C0, diameter=0.05, shape="rectangular"
    )
    assert f_upper == pytest.approx(min(0.45 * C0 / 0.03, 0.50 * C0 / 0.05))
    assert f_lower == pytest.approx(C0 / (20.0 * 0.03))


def test_plane_wave_range_square_alias() -> None:
    # A square tube is the rectangular case with d the side length, in both
    # the ISO 10534-2 and the ASTM E2611-19 bounds.
    for fn in (plane_wave_frequency_range, plane_wave_frequency_range_astm):
        square = fn(0.03, C0, diameter=0.05, shape="square")
        rectangular = fn(0.03, C0, diameter=0.05, shape="rectangular")
        assert square == rectangular


def test_plane_wave_range_invalid_shape() -> None:
    with pytest.raises(ValueError, match=r"'shape' must be"):
        plane_wave_frequency_range(0.03, C0, diameter=0.05, shape="oval")
    with pytest.raises(ValueError, match=r"'shape' must be"):
        plane_wave_frequency_range_astm(0.03, C0, diameter=0.05, shape="oval")


def test_plane_wave_range_astm_bounds() -> None:
    # ASTM E2611-19: upper = min(0,40 c/s (6.5.4), 0,586 c/d circular
    # (6.2.4.1) or 0,500 c/d rectangular (6.2.5)); lower = c / (100 s), the
    # spacing greater than 1 % of the wavelength (6.2.3).
    f_lower, f_upper = plane_wave_frequency_range_astm(
        0.03, C0, diameter=0.05, shape="circular"
    )
    assert f_upper == pytest.approx(min(0.40 * C0 / 0.03, 0.586 * C0 / 0.05))
    assert f_lower == pytest.approx(C0 / (100.0 * 0.03))
    _, f_upper_rect = plane_wave_frequency_range_astm(
        0.03, C0, diameter=0.05, shape="rectangular"
    )
    assert f_upper_rect == pytest.approx(min(0.40 * C0 / 0.03, 0.50 * C0 / 0.05))


def test_hydraulic_diameter_values() -> None:
    # d_h = 4 A / P = 2 w h / (w + h) (ISO 10534-2 A.2.1.5); a square tube
    # gives the side length back.
    assert hydraulic_diameter(0.05, 0.05) == pytest.approx(0.05)
    assert hydraulic_diameter(0.08, 0.04) == pytest.approx(
        4.0 * (0.08 * 0.04) / (2.0 * (0.08 + 0.04))
    )
    with pytest.raises(ValueError, match=r"'width' and 'height' must be positive"):
        hydraulic_diameter(0.0, 0.05)


def test_result_retains_tube_geometry() -> None:
    f = np.array([800.0, 1200.0])
    h12 = _synth_h12(0.3 + 0.1j, np.asarray(tube_wavenumber(f, C0)), 0.12, 0.03)
    res = two_microphone_impedance(
        h12,
        frequency=f,
        spacing=0.03,
        x1=0.12,
        speed_of_sound=C0,
        characteristic_impedance=RC,
        diameter=0.05,
        shape="square",
    )
    assert res.spacing == pytest.approx(0.03)
    assert res.x1 == pytest.approx(0.12)
    assert res.diameter == pytest.approx(0.05)
    # "square" is stored canonically as the rectangular case.
    assert res.shape == "rectangular"
    # Without a diameter no cross-section claim is retained.
    res_no_d = two_microphone_impedance(
        h12,
        frequency=f,
        spacing=0.03,
        x1=0.12,
        speed_of_sound=C0,
        characteristic_impedance=RC,
    )
    assert res_no_d.diameter is None
    assert res_no_d.shape is None


def test_frequency_range_no_warning_in_band() -> None:
    f = np.array([800.0, 1200.0])
    h12 = _synth_h12(0.3 + 0.1j, np.asarray(tube_wavenumber(f, C0)), 0.12, 0.03)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        two_microphone_impedance(
            h12,
            frequency=f,
            spacing=0.03,
            x1=0.12,
            speed_of_sound=C0,
            characteristic_impedance=RC,
            diameter=0.05,
        )


# ---------------------------------------------------------------------------
# ISO 10534-1: standing-wave-ratio method.
# ---------------------------------------------------------------------------
def test_swr_from_level() -> None:
    assert float(standing_wave_ratio_from_level(0.0)) == pytest.approx(1.0)
    assert float(standing_wave_ratio_from_level(20.0)) == pytest.approx(10.0)


def test_swr_reflection_and_absorption() -> None:
    # s = 3 -> |r| = 0.5 -> alpha = 4*3/16 = 0.75.
    assert float(standing_wave_reflection_magnitude(3.0)) == pytest.approx(0.5)
    assert float(standing_wave_absorption(3.0)) == pytest.approx(0.75)
    # s = 1 -> perfect absorber; s -> inf -> rigid.
    assert float(standing_wave_absorption(1.0)) == pytest.approx(1.0)
    assert float(standing_wave_reflection_magnitude(1e12)) == pytest.approx(
        1.0, abs=1e-9
    )


def test_swr_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="Standing-wave ratio 's' must be"):
        standing_wave_reflection_magnitude(0.5)


def test_swr_phase_rigid_wall_first_minimum() -> None:
    # A (near) rigid wall has its first pressure minimum at lambda/4, giving
    # phase phi = pi(4*(lambda/4)/lambda - 1) = 0, i.e. r ~ +|r|.
    wavelength = 0.343
    r = standing_wave_reflection(50.0, wavelength / 4.0, wavelength)
    assert complex(r).imag == pytest.approx(0.0, abs=1e-12)
    assert complex(r).real > 0.0


def test_swr_normalized_impedance_matches_reflection() -> None:
    swr, x_min, lam = 4.0, 0.05, 0.343
    r = standing_wave_reflection(swr, x_min, lam)
    z = standing_wave_normalized_impedance(swr, x_min, lam)
    assert complex(z) == pytest.approx((1.0 + r) / (1.0 - r))


# ---------------------------------------------------------------------------
# ASTM E2611-19: transfer matrix.
# ---------------------------------------------------------------------------
def test_air_layer_matrix_reciprocal_symmetric() -> None:
    f = np.array([500.0, 1000.0, 2000.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    d = 0.05
    tm = air_layer_transfer_matrix(k, d, RC)
    assert np.allclose(tm.determinant(), 1.0 + 0.0j, atol=1e-12)
    assert np.allclose(tm.t11, tm.t22)


def test_air_layer_transmission_loss_zero() -> None:
    f = np.array([500.0, 1000.0, 2000.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, 0.05, RC)
    assert np.allclose(tm.transmission_loss(RC), 0.0, atol=1e-9)


def test_air_layer_material_properties_recovered() -> None:
    f = np.array([500.0, 1500.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    d = 0.04
    tm = air_layer_transfer_matrix(k, d, RC)
    assert np.allclose(tm.material_wavenumber(d), k, atol=1e-10)
    assert np.allclose(tm.characteristic_impedance_material(), RC, atol=1e-9)


def test_hard_backed_rigid_from_air_gap() -> None:
    # An air layer whose T21 -> 0 behaves like a rigid boundary in the limit
    # of a very thin layer: R -> 1, alpha -> 0.
    f = np.array([1000.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, 1e-6, RC)
    assert np.allclose(np.abs(tm.reflection_hard_backed(RC)), 1.0, atol=1e-6)
    assert np.allclose(tm.absorption_hard_backed(RC), 0.0, atol=1e-6)


# Geometry (single origin at the front face x = 0).
GEOM = {"l1": 0.10, "s1": 0.03, "l2": 0.15, "s2": 0.03}
THICKNESS = 0.05


def _matmul(a: TransferMatrix, b: TransferMatrix) -> TransferMatrix:
    return TransferMatrix(
        t11=a.t11 * b.t11 + a.t12 * b.t21,
        t12=a.t11 * b.t12 + a.t12 * b.t22,
        t21=a.t21 * b.t11 + a.t22 * b.t21,
        t22=a.t21 * b.t12 + a.t22 * b.t22,
    )


def _synth_four_mics(
    tm: TransferMatrix,
    k: np.ndarray,
    down_c: complex,
    down_d: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise (H1, H2, H3, H4) for one termination from a known T-matrix.

    Physical single-origin field (front face at x = 0, forward wave e^{-j k x}):
    downstream amplitudes (down_c, down_d) fix the back-face pressure/velocity
    via Eq. (21); the specimen T-matrix maps them to the front face; the two
    upstream amplitudes then follow, and the microphone pressures are sampled at
    their physical positions.
    """
    d = THICKNESS
    ep = np.exp(-1j * k * d)
    em = np.exp(1j * k * d)
    pd = down_c * ep + down_d * em
    ud = (down_c * ep - down_d * em) / RC
    p0 = tm.t11 * pd + tm.t12 * ud
    u0 = tm.t21 * pd + tm.t22 * ud
    a = (p0 + RC * u0) / 2.0
    b = (p0 - RC * u0) / 2.0
    l1, s1, l2, s2 = GEOM["l1"], GEOM["s1"], GEOM["l2"], GEOM["s2"]
    # Upstream mics at x = -l1 (nearer, H2) and x = -(l1+s1) (farther, H1).
    h2 = a * np.exp(1j * k * l1) + b * np.exp(-1j * k * l1)
    h1 = a * np.exp(1j * k * (l1 + s1)) + b * np.exp(-1j * k * (l1 + s1))
    # Downstream mics at x = l2 (nearer, H3) and x = l2+s2 (farther, H4).
    h3 = down_c * np.exp(-1j * k * l2) + down_d * np.exp(1j * k * l2)
    h4 = down_c * np.exp(-1j * k * (l2 + s2)) + down_d * np.exp(1j * k * (l2 + s2))
    return (
        np.asarray(h1, dtype=np.complex128),
        np.asarray(h2, dtype=np.complex128),
        np.asarray(h3, dtype=np.complex128),
        np.asarray(h4, dtype=np.complex128),
    )


def test_wave_decomposition_recovers_amplitudes_up_to_sign() -> None:
    # Locks the Eq. (17)-(20) exponent convention: the decomposition of the
    # synthesised physical field returns exactly the (A, B, C, D) that built it,
    # up to a common global -1 (the standard's ansatz vs. the physical pressure)
    # that cancels in every transfer-matrix-derived quantity.
    f = np.array([600.0, 1400.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, THICKNESS, RC)
    down_c, down_d = 1.0 + 0.0j, 0.3 - 0.1j
    # Ground-truth upstream amplitudes for this field.
    ep, em = np.exp(-1j * k * THICKNESS), np.exp(1j * k * THICKNESS)
    pd = down_c * ep + down_d * em
    ud = (down_c * ep - down_d * em) / RC
    p0 = tm.t11 * pd + tm.t12 * ud
    u0 = tm.t21 * pd + tm.t22 * ud
    a_true = (p0 + RC * u0) / 2.0
    b_true = (p0 - RC * u0) / 2.0
    h1, h2, h3, h4 = _synth_four_mics(tm, np.asarray(k), down_c, down_d)
    a, b, c, d = wave_decomposition(h1, h2, h3, h4, wavenumber=k, **GEOM)
    assert np.allclose(a, -a_true, atol=1e-9)
    assert np.allclose(b, -b_true, atol=1e-9)
    assert np.allclose(c, -down_c, atol=1e-9)
    assert np.allclose(d, -down_d, atol=1e-9)


def test_face_quantities_progressive_wave_impedance() -> None:
    # ASTM E2611 Eq. (21): a single forward-travelling wave has the specific
    # acoustic impedance p/u = +rho*c at every plane, on both faces, whatever
    # the frequency or thickness (the propagation phase cancels in the ratio).
    f = np.array([500.0, 1500.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    p0, pd, u0, ud = face_quantities(
        1.0,
        0.0,
        1.0,
        0.0,
        wavenumber=k,
        thickness=THICKNESS,
        characteristic_impedance=RC,
    )
    assert np.allclose(p0 / u0, RC, atol=1e-9)
    assert np.allclose(pd / ud, RC, atol=1e-9)


def test_face_quantities_backward_wave_impedance() -> None:
    # A single backward-travelling wave reverses the sign: p/u = -rho*c.
    f = np.array([500.0, 1500.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    p0, pd, u0, ud = face_quantities(
        0.0,
        1.0,
        0.0,
        1.0,
        wavenumber=k,
        thickness=THICKNESS,
        characteristic_impedance=RC,
    )
    assert np.allclose(p0 / u0, -RC, atol=1e-9)
    assert np.allclose(pd / ud, -RC, atol=1e-9)


def test_two_load_warns_on_singular_load_pair() -> None:
    # Two identical loads make DEN = 0 (ASTM E2611); flag it rather than
    # silently returning inf/nan (CodeRabbit).
    f = np.array([500.0, 1000.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    load = (1.0 + 0j, 0.9 + 0j, 0.8 + 0j, 0.7 + 0j)
    with pytest.warns(
        ImpedanceTubeWarning,
        match=r"transfer_matrix_two_load: the solve denominator is near-singular",
    ):
        transfer_matrix_two_load(
            load,
            load,
            thickness=THICKNESS,
            wavenumber=k,
            characteristic_impedance=RC,
            **GEOM,
        )


def test_two_load_rejects_short_load() -> None:
    # A three-element load used to die in tuple indexing (IndexError),
    # naming neither load_a nor load_b.
    load = (1.0 + 0j, 0.9 + 0j, 0.8 + 0j, 0.7 + 0j)
    with pytest.raises(ValueError, match="'load_a' must be the four transfer"):
        transfer_matrix_two_load(
            load[:3],
            load,
            thickness=THICKNESS,
            wavenumber=np.array([10.0]),
            characteristic_impedance=RC,
            **GEOM,
        )


def test_one_load_rejects_long_load() -> None:
    # A five-element load used to be accepted with the extra entry
    # silently dropped.
    load = (1.0 + 0j, 0.9 + 0j, 0.8 + 0j, 0.7 + 0j, 0.6 + 0j)
    with pytest.raises(ValueError, match="'load' must be the four transfer"):
        transfer_matrix_one_load(
            load,
            thickness=THICKNESS,
            wavenumber=np.array([10.0]),
            characteristic_impedance=RC,
            **GEOM,
        )


def test_face_quantities_rejects_non_positive_thickness() -> None:
    # The same rule the module already applies to 'thickness' elsewhere;
    # a zero-depth specimen was accepted silently.
    with pytest.raises(ValueError, match="'thickness' must be positive"):
        face_quantities(
            1.0,
            0.5,
            0.7,
            0.2,
            wavenumber=10.0,
            thickness=0.0,
            characteristic_impedance=RC,
        )


def test_one_load_rejects_zero_thickness() -> None:
    # Both public solvers inherit the face_quantities guard.
    load = (1.0 + 0j, 0.9 + 0j, 0.8 + 0j, 0.7 + 0j)
    with pytest.raises(ValueError, match="'thickness' must be positive"):
        transfer_matrix_one_load(
            load,
            thickness=0.0,
            wavenumber=np.array([10.0]),
            characteristic_impedance=RC,
            **GEOM,
        )


def test_two_load_no_warning_when_well_conditioned() -> None:
    # A genuine two-load measurement (distinct terminations) must not warn.
    import warnings as _w

    f = np.array([500.0, 1000.0, 1800.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, THICKNESS, RC)
    h_a = _synth_four_mics(tm, np.asarray(k), 1.0 + 0j, 0.2 - 0.3j)
    h_b = _synth_four_mics(tm, np.asarray(k), 0.4 + 0.5j, -0.6 + 0.1j)
    with _w.catch_warnings():
        _w.simplefilter("error", ImpedanceTubeWarning)
        transfer_matrix_two_load(
            h_a,
            h_b,
            thickness=THICKNESS,
            wavenumber=k,
            characteristic_impedance=RC,
            **GEOM,
        )


def test_two_load_recovers_air_layer_matrix() -> None:
    # End-to-end validation of Eqs. (17)-(22) AND the geometry convention.
    f = np.array([500.0, 1000.0, 1800.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, THICKNESS, RC)
    load_a = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, 0.3 + 0.0j)
    load_b = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, -0.5 + 0.2j)
    rec = transfer_matrix_two_load(
        load_a,
        load_b,
        thickness=THICKNESS,
        wavenumber=k,
        characteristic_impedance=RC,
        **GEOM,
    )
    assert np.allclose(rec.t11, tm.t11, atol=1e-9)
    assert np.allclose(rec.t12, tm.t12, atol=1e-9)
    assert np.allclose(rec.t21, tm.t21, atol=1e-9)
    assert np.allclose(rec.t22, tm.t22, atol=1e-9)


def test_two_load_recovers_asymmetric_specimen() -> None:
    # A mass layer followed by an air gap: reciprocal (det = 1) but NOT
    # symmetric (T11 != T22), so it exercises the full two-load solve.
    f = np.array([500.0, 1200.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    omega = 2.0 * np.pi * f
    mass = 0.5  # kg/m2 limp mass
    zeros = np.zeros_like(k)
    ones = np.ones_like(k)
    t_mass = TransferMatrix(t11=ones, t12=1j * omega * mass, t21=zeros, t22=ones)
    t_air = air_layer_transfer_matrix(k, THICKNESS, RC)
    tm = _matmul(t_mass, t_air)
    assert not np.allclose(tm.t11, tm.t22)
    load_a = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, 0.2 + 0.0j)
    load_b = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, -0.4 + 0.1j)
    rec = transfer_matrix_two_load(
        load_a,
        load_b,
        thickness=THICKNESS,
        wavenumber=k,
        characteristic_impedance=RC,
        **GEOM,
    )
    assert np.allclose(rec.determinant(), 1.0 + 0.0j, atol=1e-9)
    assert np.allclose(rec.t11, tm.t11, atol=1e-9)
    assert np.allclose(rec.t12, tm.t12, atol=1e-9)
    assert np.allclose(rec.t21, tm.t21, atol=1e-9)
    assert np.allclose(rec.t22, tm.t22, atol=1e-9)


def test_one_load_recovers_symmetric_specimen() -> None:
    # A pure air layer is reciprocal and symmetric -> a single load suffices.
    f = np.array([500.0, 1000.0, 1500.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, THICKNESS, RC)
    load = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, 0.25 + 0.1j)
    rec = transfer_matrix_one_load(
        load,
        thickness=THICKNESS,
        wavenumber=k,
        characteristic_impedance=RC,
        **GEOM,
    )
    assert np.allclose(rec.t11, tm.t11, atol=1e-9)
    assert np.allclose(rec.t12, tm.t12, atol=1e-9)
    assert np.allclose(rec.t21, tm.t21, atol=1e-9)
    assert np.allclose(rec.t22, tm.t22, atol=1e-9)
    assert np.allclose(rec.t11, rec.t22)


_OUTSIDE_ASTM_RANGE = r"Wavenumbers outside the ASTM E2611-19 plane-wave working range"


def test_astm_solvers_warn_outside_plane_wave_range() -> None:
    # With the tube diameter given, frequencies past the 6.2.4.1 cut-on
    # (0,586 c/d for this 10 cm circular tube ~ 2011 Hz) must be flagged.
    f = np.array([500.0, 2500.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, THICKNESS, RC)
    load_a = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, 0.3 + 0.0j)
    load_b = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, -0.5 + 0.2j)
    with pytest.warns(ImpedanceTubeWarning, match=_OUTSIDE_ASTM_RANGE):
        transfer_matrix_two_load(
            load_a,
            load_b,
            thickness=THICKNESS,
            wavenumber=k,
            characteristic_impedance=RC,
            diameter=0.10,
            **GEOM,
        )
    with pytest.warns(ImpedanceTubeWarning, match=_OUTSIDE_ASTM_RANGE):
        transfer_matrix_one_load(
            load_a,
            thickness=THICKNESS,
            wavenumber=k,
            characteristic_impedance=RC,
            diameter=0.10,
            **GEOM,
        )
    with pytest.warns(ImpedanceTubeWarning, match=_OUTSIDE_ASTM_RANGE):
        wave_decomposition(
            *load_a,
            wavenumber=k,
            diameter=0.10,
            **GEOM,
        )


def test_astm_solvers_warn_below_lower_bound() -> None:
    # The 6.2.3 lower bound: spacing greater than 1 % of the wavelength. For
    # s = 3 cm that is ~114 Hz; 50 Hz must be flagged.
    f = np.array([50.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, THICKNESS, RC)
    load = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, 0.3 + 0.0j)
    with pytest.warns(ImpedanceTubeWarning, match=_OUTSIDE_ASTM_RANGE):
        wave_decomposition(*load, wavenumber=k, diameter=0.10, **GEOM)


def test_astm_solvers_silent_in_band_and_without_diameter() -> None:
    import warnings as _w

    f = np.array([500.0, 1000.0, 1800.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, THICKNESS, RC)
    load_a = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, 0.3 + 0.0j)
    load_b = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, -0.5 + 0.2j)
    with _w.catch_warnings():
        _w.simplefilter("error", ImpedanceTubeWarning)
        # In band with the diameter given (10 cm tube: 114 Hz .. 2011 Hz).
        transfer_matrix_two_load(
            load_a,
            load_b,
            thickness=THICKNESS,
            wavenumber=k,
            characteristic_impedance=RC,
            diameter=0.10,
            **GEOM,
        )
        # No diameter -> no range check, as in the ISO branch.
        k_wide = np.asarray(tube_wavenumber(np.array([50.0, 2500.0]), C0)).real.astype(
            np.complex128
        )
        tm_wide = air_layer_transfer_matrix(k_wide, THICKNESS, RC)
        wide_a = _synth_four_mics(tm_wide, np.asarray(k_wide), 1.0 + 0.0j, 0.3 + 0.0j)
        wide_b = _synth_four_mics(tm_wide, np.asarray(k_wide), 0.4 + 0.5j, -0.6 + 0.1j)
        transfer_matrix_two_load(
            wide_a,
            wide_b,
            thickness=THICKNESS,
            wavenumber=k_wide,
            characteristic_impedance=RC,
            **GEOM,
        )


def test_transfer_matrix_retains_measurement_context() -> None:
    # 500..1500 Hz sits inside the square-tube working range (114..1716 Hz).
    f = np.array([500.0, 1000.0, 1500.0])
    k = np.asarray(tube_wavenumber(f, C0)).real.astype(np.complex128)
    tm = air_layer_transfer_matrix(k, THICKNESS, RC)
    load_a = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, 0.3 + 0.0j)
    load_b = _synth_four_mics(tm, np.asarray(k), 1.0 + 0.0j, -0.5 + 0.2j)
    rec = transfer_matrix_two_load(
        load_a,
        load_b,
        thickness=THICKNESS,
        wavenumber=k,
        characteristic_impedance=RC,
        frequency=f,
        diameter=0.10,
        shape="square",
        **GEOM,
    )
    assert rec.l1 == pytest.approx(GEOM["l1"])
    assert rec.s1 == pytest.approx(GEOM["s1"])
    assert rec.l2 == pytest.approx(GEOM["l2"])
    assert rec.s2 == pytest.approx(GEOM["s2"])
    assert rec.thickness == pytest.approx(THICKNESS)
    assert rec.diameter == pytest.approx(0.10)
    assert rec.shape == "rectangular"
    assert rec.frequency is not None
    assert np.allclose(rec.frequency, f)
    assert rec.air_characteristic_impedance == pytest.approx(RC)
    # A hand-built matrix retains nothing and plot() demands the arguments.
    bare = air_layer_transfer_matrix(k, THICKNESS, RC)
    assert bare.frequency is None
    assert bare.air_characteristic_impedance is None
    with pytest.raises(
        ValueError,
        match=r"'frequency' and 'characteristic_impedance' must be supplied",
    ):
        bare.plot()


def test_public_exports() -> None:
    from phonometry import materials

    for name in (
        "ImpedanceTubeResult",
        "ImpedanceTubeWarning",
        "TransferMatrix",
        "reflection_factor",
        "surface_impedance",
        "absorption_from_reflection",
        "normalized_surface_impedance",
        "normalized_surface_admittance",
        "characteristic_impedance",
        "speed_of_sound_iso10534",
        "speed_of_sound_astm",
        "air_density_iso10534",
        "air_density_astm",
        "tube_wavenumber",
        "tube_attenuation_constant",
        "mic_calibration_factor",
        "apply_mic_calibration",
        "two_microphone_impedance",
        "plane_wave_frequency_range",
        "plane_wave_frequency_range_astm",
        "hydraulic_diameter",
        "standing_wave_absorption",
        "standing_wave_reflection",
        "standing_wave_reflection_magnitude",
        "standing_wave_ratio_from_level",
        "standing_wave_normalized_impedance",
        "wave_decomposition",
        "face_quantities",
        "transfer_matrix_two_load",
        "transfer_matrix_one_load",
        "air_layer_transfer_matrix",
    ):
        assert hasattr(materials, name), name
