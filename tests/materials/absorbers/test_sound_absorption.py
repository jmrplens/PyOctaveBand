#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for BS EN ISO 354:2003 sound absorption in a reverberation room.

Normative anchors (ISO 354:2003):
- Eq. (5)/(7): A = 55,3*V/(c*T) - 4*V*m  (empty room A1 / with specimen A2).
- Eq. (6):     c = (331 + 0,6*t/degC) m/s, valid 15..30 degC.
- Eq. (8):     AT = A2 - A1 = 55,3*V*(1/(c2*T2) - 1/(c1*T1)) - 4*V*(m2 - m1).
- Eq. (9):     alpha_s = AT / S  (may exceed 1,0; clause 3.7 NOTE 2).
- m from ISO 9613-1 alpha (dB/m): m = alpha / (10*lg e).

Validation strategy: closed-form inversion of the Sabine equation, not
self-consistency. All tolerances are far below the standard's reporting
rounding (AT to 0,1 m2; alpha_s to 0,01; clause 8.3).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phonometry import materials
from phonometry.materials.absorbers.sound_absorption import _speed_of_sound

# --- Eq. (6): speed of sound ------------------------------------------------


def test_speed_of_sound_reference_20c() -> None:
    # c = 331 + 0,6*20 = 343 m/s.
    assert _speed_of_sound(20.0) == pytest.approx(343.0)


def test_default_temperature_is_20c() -> None:
    # absorption_area default temperature must give c = 343 m/s.
    a_default = materials.absorption_area(5.0, 200.0)
    a_explicit = materials.absorption_area(5.0, 200.0, speed_of_sound=343.0)
    assert np.asarray(a_default) == pytest.approx(np.asarray(a_explicit))


# --- Eq. (5)/(7): absorption area & exact T->A->T inversion -----------------


def test_absorption_area_formula() -> None:
    v, c, t = 200.0, 343.0, 3.5
    expected = 55.3 * v / (c * t)  # m = 0
    assert materials.absorption_area(t, v, speed_of_sound=c) == pytest.approx(expected)


def test_exact_inversion_t_to_a_to_t() -> None:
    # A = 55,3 V/(c T)  =>  T = 55,3 V/(c A), exact round trip (m = 0).
    v, c = 200.0, 343.0
    t60 = np.array([1.5, 2.0, 3.7, 5.0, 8.2])
    a = materials.absorption_area(t60, v, speed_of_sound=c)
    t_back = 55.3 * v / (c * np.asarray(a))
    np.testing.assert_allclose(t_back, t60, rtol=0, atol=1e-12)


def test_array_input_shape_preserved() -> None:
    a = materials.absorption_area([2.0, 4.0, 6.0], 200.0)
    assert isinstance(a, np.ndarray)
    assert a.shape == (3,)


# --- Air-attenuation term ---------------------------------------------------


def test_air_correction_zero_when_m_zero() -> None:
    # -4 V m term vanishes at m = 0 (reference/zero condition).
    v, c, t = 200.0, 343.0, 4.0
    assert materials.absorption_area(t, v, speed_of_sound=c, m=0.0) == pytest.approx(
        55.3 * v / (c * t)
    )


def test_air_correction_subtracts_4vm() -> None:
    v, c, t, m = 200.0, 343.0, 4.0, 0.003
    a = materials.absorption_area(t, v, speed_of_sound=c, m=m)
    assert a == pytest.approx(55.3 * v / (c * t) - 4.0 * v * m)


def test_air_correction_per_band_array() -> None:
    v, c = 200.0, 343.0
    t = np.array([3.0, 3.0, 3.0])
    m = np.array([0.0, 0.001, 0.005])
    a = materials.absorption_area(t, v, speed_of_sound=c, m=m)
    expected = 55.3 * v / (c * t) - 4.0 * v * m
    np.testing.assert_allclose(np.asarray(a), expected)


# --- m from ISO 9613-1 alpha (dB/m) -----------------------------------------


def test_attenuation_from_alpha_conversion() -> None:
    # m = alpha / (10 lg e); 10*lg(e) = 4,342944819...
    alpha = 4.342944819032518
    assert materials.attenuation_from_alpha(alpha) == pytest.approx(1.0, rel=1e-9)
    assert materials.attenuation_from_alpha(0.0) == 0.0


def test_attenuation_from_alpha_array() -> None:
    alpha = np.array([0.0, 4.342944819032518, 8.685889638065036])
    m = materials.attenuation_from_alpha(alpha)
    np.testing.assert_allclose(np.asarray(m), [0.0, 1.0, 2.0], atol=1e-9)


# --- Eq. (8)/(9): absorption coefficient of a synthetic sample --------------


def test_recover_known_alpha() -> None:
    v, c, s, alpha_true = 200.0, 343.0, 10.0, 0.80
    t1 = 5.0
    a1 = 55.3 * v / (c * t1)
    a2 = a1 + alpha_true * s  # AT = alpha*S
    t2 = 55.3 * v / (c * a2)  # invert Eq. (7)
    alpha = materials.absorption_coefficient(
        t1, t2, v, s, speed_of_sound1=c, speed_of_sound2=c
    )
    assert float(np.asarray(alpha)) == pytest.approx(alpha_true, abs=1e-9)


def test_recover_known_alpha_per_band() -> None:
    v, c, s = 200.0, 343.0, 10.0
    alpha_true = np.array([0.15, 0.45, 0.9, 1.15])
    t1 = np.full(4, 5.0)
    a1 = 55.3 * v / (c * t1)
    a2 = a1 + alpha_true * s
    t2 = 55.3 * v / (c * a2)
    alpha = materials.absorption_coefficient(
        t1, t2, v, s, speed_of_sound1=c, speed_of_sound2=c
    )
    np.testing.assert_allclose(np.asarray(alpha), alpha_true, atol=1e-9)


def test_alpha_can_exceed_one_no_warning() -> None:
    # Clause 3.7 NOTE 2: alpha_s > 1 is valid and must not be clamped/warned.
    v, c, s = 200.0, 343.0, 10.0
    t1 = 6.0
    a1 = 55.3 * v / (c * t1)
    a2 = a1 + 1.3 * s
    t2 = 55.3 * v / (c * a2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alpha = materials.absorption_coefficient(
            t1, t2, v, s, speed_of_sound1=c, speed_of_sound2=c
        )
    assert float(np.asarray(alpha)) == pytest.approx(1.3, abs=1e-9)


def test_two_temperature_correction_eq8() -> None:
    # c1 != c2: AT built with per-measurement speeds of sound.
    v, s = 200.0, 10.0
    t1, t2 = 5.0, 3.0
    c1, c2 = _speed_of_sound(18.0), _speed_of_sound(22.0)
    expected_at = 55.3 * v * (1.0 / (c2 * t2) - 1.0 / (c1 * t1))
    alpha = materials.absorption_coefficient(
        t1, t2, v, s, temperature1=18.0, temperature2=22.0
    )
    assert float(np.asarray(alpha)) == pytest.approx(expected_at / s, abs=1e-9)


def test_temperature2_defaults_to_temperature1() -> None:
    v, s = 200.0, 10.0
    a = materials.absorption_coefficient(5.0, 3.0, v, s, temperature1=25.0)
    b = materials.absorption_coefficient(
        5.0, 3.0, v, s, temperature1=25.0, temperature2=25.0
    )
    assert float(np.asarray(a)) == pytest.approx(float(np.asarray(b)))


def test_speed_of_sound2_defaults_to_speed_of_sound1() -> None:
    """Overriding only c1 must apply the same speed to the second measurement
    (c2 defaults to c1), mirroring the temperature2 -> temperature1 default.
    """
    v, s = 200.0, 10.0
    c = 340.0
    only_c1 = materials.absorption_coefficient(5.0, 3.0, v, s, speed_of_sound1=c)
    both = materials.absorption_coefficient(
        5.0, 3.0, v, s, speed_of_sound1=c, speed_of_sound2=c
    )
    assert float(np.asarray(only_c1)) == pytest.approx(float(np.asarray(both)))
    # And it must differ from letting c2 fall back to the 20 degC default.
    default_c2 = materials.absorption_coefficient(5.0, 3.0, v, s, speed_of_sound1=c)
    explicit_343 = materials.absorption_coefficient(
        5.0, 3.0, v, s, speed_of_sound1=c, speed_of_sound2=_speed_of_sound(20.0)
    )
    assert float(np.asarray(default_c2)) != pytest.approx(
        float(np.asarray(explicit_343))
    )


# --- Non-physical result warning (A2 <= A1, i.e. T2 >= T1) ------------------


def test_warns_when_alpha_non_positive() -> None:
    # Adding an absorber must reduce T; T2 >= T1 gives alpha_s <= 0.
    with pytest.warns(materials.AbsorptionWarning):
        materials.absorption_coefficient(3.0, 3.5, 200.0, 10.0)


def test_no_warning_for_positive_alpha() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        materials.absorption_coefficient(5.0, 3.0, 200.0, 10.0)


# --- Setup advisories: room volume (6.1.1) & sample area (6.2.1.1) ----------


def test_warns_small_room_absorption_area() -> None:
    # Clause 6.1.1: V < 150 m3 is advisory (result still returned).
    with pytest.warns(materials.AbsorptionWarning):
        a = materials.absorption_area(3.0, 120.0)
    assert np.isfinite(np.asarray(a)).all()


def test_warns_small_room_absorption_coefficient() -> None:
    with pytest.warns(materials.AbsorptionWarning):
        materials.absorption_coefficient(5.0, 3.0, 120.0, 10.0)


def test_small_room_warning_not_duplicated_in_coefficient(
    recwarn: pytest.WarningsRecorder,
) -> None:
    # The internal absorption_area calls must not re-emit the volume advisory.
    materials.absorption_coefficient(5.0, 3.0, 120.0, 10.0)
    volume_warnings = [
        w
        for w in recwarn.list
        if issubclass(w.category, materials.AbsorptionWarning)
        and "Room volume" in str(w.message)
    ]
    assert len(volume_warnings) == 1


def test_no_room_warning_at_minimum_volume() -> None:
    # V = 150 m3 is exactly at the clause 6.1.1 minimum (>= 150): no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        materials.absorption_area(3.0, 150.0)


def test_warns_sample_area_below_range() -> None:
    # Clause 6.2.1.1: S < 10 m2 is advisory.
    with pytest.warns(materials.AbsorptionWarning):
        materials.absorption_coefficient(5.0, 3.0, 200.0, 8.0)


def test_warns_sample_area_above_range_no_scaling() -> None:
    # V <= 200 m3: upper limit is 12 m2; S = 13 m2 is out of range.
    with pytest.warns(materials.AbsorptionWarning):
        materials.absorption_coefficient(5.0, 3.0, 200.0, 13.0)


def test_warns_sample_area_above_scaled_range() -> None:
    # V > 200 m3: upper limit is 12*(V/200)^(2/3). For V = 400, ~19,05 m2;
    # S = 20 m2 exceeds it and must warn.
    v = 400.0
    upper = 12.0 * (v / 200.0) ** (2.0 / 3.0)
    assert 19.0 < upper < 19.1
    with pytest.warns(materials.AbsorptionWarning):
        materials.absorption_coefficient(5.0, 3.0, v, 20.0)


def test_no_sample_area_warning_within_scaled_range() -> None:
    # V = 400 m3 lifts the upper limit to ~19,05 m2; S = 15 m2 is now in range.
    v = 400.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        materials.absorption_coefficient(5.0, 3.0, v, 15.0)


def test_no_sample_area_warning_at_limits() -> None:
    # S = 10 and S = 12 m2 are the inclusive clause 6.2.1.1 bounds (V <= 200).
    for s in (10.0, 12.0):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            materials.absorption_coefficient(5.0, 3.0, 200.0, s)


# --- Range / input validation ----------------------------------------------


def test_negative_volume_raises() -> None:
    with pytest.raises(ValueError, match="'volume' must be positive"):
        materials.absorption_area(3.0, -1.0)


def test_zero_reverberation_time_raises() -> None:
    with pytest.raises(ValueError, match="'t60' must be positive"):
        materials.absorption_area(0.0, 200.0)


def test_negative_m_raises() -> None:
    with pytest.raises(ValueError, match="'m' must be non-negative"):
        materials.absorption_area(3.0, 200.0, m=-0.001)


def test_m_shape_mismatch_raises() -> None:
    """A per-band 'm' whose shape differs from 't60' is a contract violation."""
    t60 = np.array([3.0, 2.5, 2.0])  # 3 bands
    m_two_values = np.array([0.001, 0.002])
    with pytest.raises(ValueError, match="'m' must be a scalar"):
        materials.absorption_area(t60, 200.0, m=m_two_values)


def test_m_matching_shape_and_scalar_allowed() -> None:
    """A scalar 'm' or one matching 't60' is accepted."""
    t60 = np.array([3.0, 2.5, 2.0])
    a_scalar = materials.absorption_area(t60, 200.0, m=0.001)
    a_perband = materials.absorption_area(t60, 200.0, m=np.array([0.001, 0.001, 0.001]))
    assert a_scalar.shape == (3,)
    assert np.allclose(a_scalar, a_perband)


def test_negative_sample_area_raises() -> None:
    with pytest.raises(ValueError, match="'sample_area' must be positive"):
        materials.absorption_coefficient(5.0, 3.0, 200.0, -10.0)


def test_negative_t1_raises() -> None:
    # `t1`, not `t60`: the message names the argument this function takes.
    with pytest.raises(ValueError, match="'t1' must be positive"):
        materials.absorption_coefficient(-5.0, 3.0, 200.0, 10.0)


def test_temperature_outside_eq6_range_warns() -> None:
    # Eq. (6) is valid 15..30 degC; outside that range we still compute but warn.
    with pytest.warns(materials.AbsorptionWarning):
        materials.absorption_area(3.0, 200.0, temperature=5.0)


def test_temperature_below_absolute_zero_raises() -> None:
    # Below -273,15 degC Eq. (6) would hand back a non-positive speed of
    # sound; the refusal is a hard error, not the out-of-range advisory.
    with pytest.raises(
        ValueError, match="'temperature' must be a finite temperature above"
    ):
        materials.absorption_area(3.0, 200.0, temperature=-600.0)


def test_nan_temperature_raises() -> None:
    with pytest.raises(
        ValueError, match="'temperature' must be a finite temperature above"
    ):
        materials.absorption_area(3.0, 200.0, temperature=float("nan"))


def test_nan_speed_of_sound_raises() -> None:
    # NaN passes a bare <= 0 comparison and would reach every derived quantity.
    with pytest.raises(
        ValueError, match="'speed_of_sound' must be finite and positive"
    ):
        materials.absorption_area(3.0, 200.0, speed_of_sound=float("nan"))


def test_infinite_speed_of_sound_raises() -> None:
    # Infinity is positive but zeroes the speed-dependent terms of Eq. (5).
    with pytest.raises(
        ValueError, match="'speed_of_sound' must be finite and positive"
    ):
        materials.absorption_area(3.0, 200.0, speed_of_sound=float("inf"))


def test_non_positive_speed_of_sound_raises() -> None:
    with pytest.raises(
        ValueError, match="'speed_of_sound' must be finite and positive"
    ):
        materials.absorption_area(3.0, 200.0, speed_of_sound=-343.0)


def test_no_temperature_warning_when_speed_supplied() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        materials.absorption_area(3.0, 200.0, temperature=5.0, speed_of_sound=340.0)


# --- Synergy with room_parameters -------------------------------------------


def test_synergy_with_room_parameters() -> None:
    from phonometry import room

    fs = 48000
    # Synthesise a decaying reverberant impulse response so T is well-defined.
    rng = np.random.default_rng(0)
    t60_target = 1.2
    n = int(fs * 2.5)
    decay = np.exp(-3.0 * np.log(10.0) * np.arange(n) / (t60_target * fs))
    ir = rng.standard_normal(n) * decay
    res = room.room_parameters(ir, fs, limits=(250.0, 2000.0), fraction=1)
    a = materials.absorption_area(res.t20, 200.0)
    assert isinstance(a, np.ndarray)
    assert a.shape == res.t20.shape
    # Finite where T20 is finite and positive.
    finite = np.isfinite(res.t20) & (res.t20 > 0)
    assert np.all(np.isfinite(np.asarray(a)[finite]))


# --- SoundAbsorptionMeasurement result type (Clause 8) ----------------------


#: One-third-octave bands and a documented empty/with-specimen reverberation
#: pair (V = 200 m3, S = 10.8 m2, t = 20 degC -> c = 343 m/s, m = 0). The
#: closed-form Eq. (5)/(7)/(8)/(9) values of two bands are asserted below.
_FREQS = np.array(
    [
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
    ],
    dtype=float,
)
_T1 = np.array(
    [
        9.0,
        9.0,
        8.8,
        8.6,
        8.4,
        8.2,
        8.0,
        7.8,
        7.5,
        7.2,
        6.9,
        6.6,
        6.2,
        5.8,
        5.4,
        5.0,
        4.6,
        4.2,
    ]
)
_T2 = np.array(
    [
        8.4,
        8.2,
        7.7,
        7.2,
        6.5,
        5.7,
        4.9,
        4.2,
        3.6,
        3.15,
        2.85,
        2.65,
        2.55,
        2.5,
        2.55,
        2.6,
        2.7,
        2.85,
    ]
)


def _measurement() -> materials.SoundAbsorptionMeasurement:
    return materials.measure_sound_absorption(
        _FREQS,
        _T1,
        _T2,
        volume=200.0,
        area=10.8,
        temperature=20.0,
        humidity=54.0,
    )


def test_measurement_speed_of_sound_from_eq6() -> None:
    # Eq. (6) at 20 degC gives c = 343 m/s.
    assert _measurement().speed_of_sound == pytest.approx(343.0)


def test_measurement_alpha_s_matches_absorption_coefficient() -> None:
    # The result must reuse the validated Eq. (8)/(9) path, not re-derive it.
    res = _measurement()
    ref = materials.absorption_coefficient(_T1, _T2, 200.0, 10.8, temperature1=20.0)
    np.testing.assert_allclose(res.alpha_s, np.asarray(ref), rtol=0, atol=1e-12)


def test_measurement_closed_form_500hz_1000hz() -> None:
    # Independent closed-form ISO 354 Eq. (5)/(7)/(8)/(9), K = 55.3*200/343.
    k = 55.3 * 200.0 / 343.0
    res = _measurement()
    # 500 Hz (index 7): T1 = 7.80 s, T2 = 4.20 s.
    a1 = k / 7.80
    a2 = k / 4.20
    assert res.absorption_area_empty[7] == pytest.approx(a1)
    assert res.absorption_area_with_specimen[7] == pytest.approx(a2)
    assert res.alpha_s[7] == pytest.approx((a2 - a1) / 10.8)
    assert round(float(res.alpha_s[7]), 2) == 0.33
    # 1000 Hz (index 10): T1 = 6.90 s, T2 = 2.85 s.
    assert res.alpha_s[10] == pytest.approx((k / 2.85 - k / 6.90) / 10.8)
    assert round(float(res.alpha_s[10]), 2) == 0.61


def test_measurement_equivalent_absorption_area_is_a2_minus_a1() -> None:
    res = _measurement()
    np.testing.assert_allclose(
        res.equivalent_absorption_area,
        np.asarray(res.absorption_area_with_specimen)
        - np.asarray(res.absorption_area_empty),
    )


def test_measurement_carries_conditions() -> None:
    res = _measurement()
    assert res.volume == 200.0
    assert res.area == 10.8
    assert res.temperature == 20.0
    assert res.humidity == 54.0
    np.testing.assert_array_equal(res.frequencies, _FREQS)
    np.testing.assert_allclose(res.air_attenuation, np.zeros_like(_FREQS))


def test_measurement_negative_humidity_raises() -> None:
    with pytest.raises(ValueError, match=r"'humidity' must be within \[0, 100\]"):
        materials.measure_sound_absorption(
            _FREQS, _T1, _T2, volume=200.0, area=10.8, humidity=-40.0
        )


def test_measurement_humidity_above_saturation_raises() -> None:
    with pytest.raises(ValueError, match=r"'humidity' must be within \[0, 100\]"):
        materials.measure_sound_absorption(
            _FREQS, _T1, _T2, volume=200.0, area=10.8, humidity=150.0
        )


def test_measurement_non_numeric_humidity_raises_by_name() -> None:
    # Not float()'s anonymous "could not convert string to float".
    with pytest.raises(ValueError, match=r"'humidity' must be within \[0, 100\]"):
        materials.measure_sound_absorption(
            _FREQS, _T1, _T2, volume=200.0, area=10.8, humidity="wet"
        )


def test_measurement_one_element_array_humidity_raises_by_name() -> None:
    # float() refuses a 1-d array with TypeError, not ValueError; the guard
    # renames that failure too instead of dying inside float().
    with pytest.raises(ValueError, match=r"'humidity' must be within \[0, 100\]"):
        materials.measure_sound_absorption(
            _FREQS, _T1, _T2, volume=200.0, area=10.8, humidity=np.array([54.0])
        )


def test_measurement_air_attenuation_reduces_area() -> None:
    # A non-zero m subtracts the 4 V m term (Eq. (5)/(7)).
    m = 1e-3
    res = materials.measure_sound_absorption(
        _FREQS, _T1, _T2, volume=200.0, area=10.8, temperature=20.0, m=m
    )
    expected_a1 = 55.3 * 200.0 / (343.0 * _T1) - 4.0 * 200.0 * m
    np.testing.assert_allclose(res.absorption_area_empty, expected_a1)


def test_measurement_shape_mismatch_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"measure_sound_absorption: .*'t_empty' .* must all have the same shape",
    ):
        materials.measure_sound_absorption(
            _FREQS, _T1[:-1], _T2, volume=200.0, area=10.8
        )


def test_measurement_frozen() -> None:
    import dataclasses

    measurement = _measurement()
    replacement = np.zeros(18)
    with pytest.raises(dataclasses.FrozenInstanceError):
        measurement.alpha_s = replacement  # type: ignore[misc]


def test_measurement_small_room_warns_once() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        materials.measure_sound_absorption(
            _FREQS, _T1, _T2, volume=100.0, area=10.8, temperature=20.0
        )
    volume_warnings = [
        w
        for w in caught
        if issubclass(w.category, materials.AbsorptionWarning)
        and "volume" in str(w.message)
    ]
    assert len(volume_warnings) == 1


def test_measurement_temperature_out_of_range_warns_once() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        materials.measure_sound_absorption(
            _FREQS, _T1, _T2, volume=200.0, area=10.8, temperature=5.0
        )
    temp_warnings = [
        w
        for w in caught
        if issubclass(w.category, materials.AbsorptionWarning)
        and "Eq. (6)" in str(w.message)
    ]
    assert len(temp_warnings) == 1
