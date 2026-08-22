#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for EN 12354-6:2003 sound absorption in enclosed spaces.

Validated against the three worked cases of the standard's Annex E (a
4,54 x 2,73 x 2,40 = 29,75 m3 room, 1000 Hz octave band).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from reference_data import (
    EN12354_6_A_BARE,
    EN12354_6_A_OBJECTS,
    EN12354_6_ANNEX_E_BARE_SURFACES,
    EN12354_6_ANNEX_E_VOLUME,
    EN12354_6_T_BARE,
)

from phonometry.room import enclosed_space_absorption as m

_VOLUME = EN12354_6_ANNEX_E_VOLUME
# The six surfaces of the bare room (area, absorption coefficient at 1000 Hz),
# shared with the CI conformance report via tests/reference_data/.
_BARE_SURFACES = EN12354_6_ANNEX_E_BARE_SURFACES


# ---------------------------------------------------------------------------
# Annex E worked cases.
# ---------------------------------------------------------------------------


def test_annex_e_case1_bare_room() -> None:
    a = m.equivalent_absorption_area(_BARE_SURFACES)
    assert a == pytest.approx(EN12354_6_A_BARE, abs=0.01)
    t = m.reverberation_time(a, _VOLUME)
    assert t == pytest.approx(EN12354_6_T_BARE, abs=0.05)


def test_annex_e_case1_air_note() -> None:
    # With air absorption Aair = 0.12 m2 the reverberation time becomes 2.0 s.
    a = m.equivalent_absorption_area(_BARE_SURFACES)
    aair = m.air_absorption_area(m.AIR_ATTENUATION["20C_50-70"][3], _VOLUME)
    assert aair == pytest.approx(0.12, abs=0.005)
    assert m.reverberation_time(a + aair, _VOLUME) == pytest.approx(2.0, abs=0.05)


def test_annex_e_case2_with_objects() -> None:
    volumes = [0.15, 0.60, 0.05, 0.05, 0.65, 0.65]
    aobj = m.hard_object_absorption(volumes)
    assert float(np.sum(aobj)) == pytest.approx(2.77, abs=0.01)
    psi = m.object_fraction(volumes, _VOLUME)
    assert psi == pytest.approx(0.072, abs=0.001)
    a = m.equivalent_absorption_area(_BARE_SURFACES, objects=aobj)
    assert a == pytest.approx(EN12354_6_A_OBJECTS, abs=0.01)
    t = m.reverberation_time(a, _VOLUME, object_fraction=psi)
    assert t == pytest.approx(0.9, abs=0.05)


def test_annex_e_case3_absorbing_wall() -> None:
    # One long wall lined for 90 % of its area with alpha_s = 0.85.
    surfaces = [
        (12.39, 0.05),
        (12.39, 0.02),
        (1.09, 0.04),
        (9.81, 0.85),
        (10.90, 0.04),
        (6.55, 0.04),
        (6.55, 0.04),
    ]
    a = m.equivalent_absorption_area(surfaces)
    assert a == pytest.approx(10.21, abs=0.01)
    assert m.reverberation_time(a, _VOLUME) == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# Formula checks.
# ---------------------------------------------------------------------------


def test_hard_object_absorption_formula_4() -> None:
    assert float(m.hard_object_absorption(0.65)) == pytest.approx(0.65 ** (2.0 / 3.0))
    assert float(m.hard_object_absorption(1.0)) == pytest.approx(1.0)


def test_air_absorption_area_formula_2() -> None:
    # Aair = 4*m*V*(1 - psi).
    assert m.air_absorption_area(1e-3, 100.0, 0.1) == pytest.approx(
        4.0 * 1e-3 * 100.0 * 0.9
    )


def test_reverberation_factor_is_016() -> None:
    # 55.3 / c0 = 0.16 for the standard's c0 = 345.6 m/s.
    assert m._RT_CONSTANT / m.SPEED_OF_SOUND == pytest.approx(0.16, abs=1e-4)


def test_object_fraction_formula_3() -> None:
    assert m.object_fraction([1.0, 2.0], 30.0) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Per-band spectrum and validation.
# ---------------------------------------------------------------------------


def test_enclosed_space_reverberation_spectrum() -> None:
    alpha = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    result = m.enclosed_space_reverberation(
        [(20.0, alpha)], 50.0, air_condition="20C_50-70"
    )
    assert result.frequencies.shape == (7,)
    assert result.reverberation_time.shape == (7,)
    # More absorption at high frequency shortens the reverberation time.
    assert result.reverberation_time[0] > result.reverberation_time[-1]
    # Cross-check the 1000 Hz band against the primitives.
    aair = m.air_absorption_area(m.AIR_ATTENUATION["20C_50-70"][3], 50.0)
    a_1k = m.equivalent_absorption_area([(20.0, 0.40)], air_area=aair)
    assert result.absorption_area[3] == pytest.approx(a_1k)


def test_equivalent_area_accepts_single_object() -> None:
    # A room with a single hard object (a 0-d array from hard_object_absorption).
    aobj = m.hard_object_absorption(0.65)
    a = m.equivalent_absorption_area([(12.39, 0.05)], objects=aobj)
    assert a == pytest.approx(12.39 * 0.05 + 0.65 ** (2.0 / 3.0))


def test_equivalent_area_per_band_objects() -> None:
    # Objects with per-band areas come as a (n_objects, n_bands) array.
    objects = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    a = m.equivalent_absorption_area([(10.0, [0.1, 0.2, 0.3])], objects=objects)
    np.testing.assert_allclose(a, [2.5, 4.5, 6.5])


def test_air_condition_none_neglects_air() -> None:
    result = m.enclosed_space_reverberation([(20.0, 0.5)], 50.0, air_condition=None)
    assert result.absorption_area[0] == pytest.approx(10.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="'volume' must be positive"):
        m.object_fraction([1.0], 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        m.hard_object_absorption([-1.0])
    with pytest.raises(ValueError, match="absorption_area must be positive"):
        m.reverberation_time(0.0, 30.0)
    with pytest.raises(ValueError, match="'volume' must be positive"):
        m.reverberation_time(2.0, 0.0)
    with pytest.raises(ValueError, match="'speed_of_sound' must be positive"):
        m.reverberation_time(2.0, 30.0, speed_of_sound=0.0)
    with pytest.raises(ValueError, match="'volume' must be positive"):
        m.reverberation_time(2.0, float("nan"))
    with pytest.raises(ValueError, match="air_condition must be"):
        m.enclosed_space_reverberation([(20.0, 0.5)], 50.0, air_condition="hot")


def test_physical_range_validation() -> None:
    # Object volumes cannot be negative nor exceed the room volume.
    with pytest.raises(ValueError, match="non-negative"):
        m.object_fraction([-1.0], 30.0)
    with pytest.raises(ValueError, match="cannot exceed the room volume"):
        m.object_fraction([40.0], 30.0)
    # The object fraction is a fraction in [0, 1).
    with pytest.raises(ValueError, match=r"range \[0, 1\)"):
        m.air_absorption_area(1e-3, 100.0, 1.5)
    with pytest.raises(ValueError, match=r"range \[0, 1\)"):
        m.reverberation_time(2.0, 30.0, object_fraction=1.0)
    # Negative attenuation, absorption coefficients, object areas or air area.
    with pytest.raises(ValueError, match="must be non-negative"):
        m.air_absorption_area(-1e-3, 100.0)
    with pytest.raises(ValueError, match="absorption coefficients"):
        m.equivalent_absorption_area([(10.0, -0.1)])
    with pytest.raises(ValueError, match="object absorption areas"):
        m.equivalent_absorption_area([(10.0, 0.1)], objects=[-1.0])
    with pytest.raises(ValueError, match="air_area"):
        m.equivalent_absorption_area([(10.0, 0.1)], air_area=-0.5)


def _prediction() -> m.ReverberationResult:
    """A seven-band prediction straight out of the public entry point."""
    alpha = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    return m.enclosed_space_reverberation(
        [(20.0, alpha)], 50.0, air_condition="20C_50-70"
    )


def test_absorption_area_off_the_band_axis_is_refused() -> None:
    # The silent one: the plot beside the table never opens absorption_area,
    # so an extra entry would only shift the printed A column and the boxed
    # mid-frequency area, with nothing raised.
    result = _prediction()
    shifted = np.concatenate([[3.10], result.absorption_area])
    with pytest.raises(ValueError, match=r"'absorption_area'.+one value per band"):
        dataclasses.replace(result, absorption_area=shifted)


def test_reverberation_time_off_the_band_axis_is_refused() -> None:
    result = _prediction()
    clipped = result.reverberation_time[:-1]
    with pytest.raises(ValueError, match=r"'reverberation_time'.+one value per band"):
        dataclasses.replace(result, reverberation_time=clipped)


def test_absorption_area_with_an_extra_axis_is_refused() -> None:
    # A (bands, 2) area counts the right number of bands on its first axis.
    result = _prediction()
    stacked = np.column_stack([result.absorption_area, result.absorption_area])
    with pytest.raises(ValueError, match=r"'absorption_area' must have one axis"):
        dataclasses.replace(result, absorption_area=stacked)


def test_air_condition_requires_standard_bands() -> None:
    # The built-in Table 1 profiles cover the standard octave bands only.
    with pytest.raises(ValueError, match="OCTAVE_BANDS"):
        m.enclosed_space_reverberation(
            [(20.0, 0.5)],
            50.0,
            air_condition="20C_50-70",
            frequencies=[500.0, 1000.0],
        )
