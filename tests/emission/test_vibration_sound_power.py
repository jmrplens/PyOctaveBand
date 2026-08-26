#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO/TS 7849 airborne sound power from surface vibration.

Anchored on the standard's own worked calibration example (Part 1, Eq. 8:
a_peak = 9,81 m/s^2 at 100 Hz -> L_v = 106,9 dB), on the closed-form velocity
and sound-power-level relations (Eqs. 3, 12/15) and on the exact round-trip
between the radiation factor (Eq. 8: eps = P/(Z_c v^2 S)) and the sound power
level L_W = 10 lg(P/P0). The K1A extraneous-velocity correction reproduces
Table 2 exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import emission

V0 = 5.0e-8
ZCN, ZC0, P0 = 411.0, 400.0, 1.0e-12


# ---------------------------------------------------------------------------
# Velocity level (Eq. 3) and calibration (Eq. 8)
# ---------------------------------------------------------------------------
def test_velocity_level_reference() -> None:
    assert emission.velocity_level(V0) == pytest.approx(0.0)
    assert emission.velocity_level(5.0e-5) == pytest.approx(60.0)  # 20 lg(1000)


def test_calibration_example_from_standard() -> None:
    """ISO/TS 7849-1 worked EXAMPLE: 9,81 m/s^2 at 100 Hz -> 106,9 dB."""
    lv = float(emission.velocity_level_from_acceleration(9.81, 100.0))
    assert lv == pytest.approx(106.9, abs=0.05)


def test_calibration_matches_hand_formula() -> None:
    a, f = 4.0, 250.0
    expected = 20.0 * math.log10(a / (2.0 * math.pi * f * V0 * math.sqrt(2.0)))
    assert emission.velocity_level_from_acceleration(a, f) == pytest.approx(expected)


def test_calibration_rejects_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        emission.velocity_level_from_acceleration(9.81, 0.0)


# ---------------------------------------------------------------------------
# Mean velocity level (Eq. 10/11)
# ---------------------------------------------------------------------------
def test_mean_velocity_level_energetic() -> None:
    levels = np.array([60.0, 60.0, 66.0])
    expected = 10.0 * math.log10(np.mean(10.0 ** (0.1 * levels)))
    assert emission.mean_velocity_level(levels) == pytest.approx(expected)


def test_mean_velocity_level_area_weighted() -> None:
    levels = np.array([60.0, 66.0])
    areas = np.array([1.0, 3.0])
    expected = 10.0 * math.log10(np.sum(areas * 10.0 ** (0.1 * levels)) / np.sum(areas))
    assert emission.mean_velocity_level(levels, areas=areas) == pytest.approx(expected)


def test_mean_velocity_level_area_shape_mismatch() -> None:
    with pytest.raises(
        ValueError, match=r"mean_velocity_level: 'levels' .*'areas' .*same shape"
    ):
        emission.mean_velocity_level([60.0, 66.0], areas=[1.0])


# ---------------------------------------------------------------------------
# Radiation factor (Eq. 4/8) and the round-trip to the sound power level
# ---------------------------------------------------------------------------
def test_radiation_factor_definition() -> None:
    p, s, v2 = 3.0e-4, 2.0, (1.0e-3) ** 2
    assert emission.radiation_factor(p, s, v2) == pytest.approx(p / (ZCN * v2 * s))


def test_power_level_round_trip_through_radiation_factor() -> None:
    """eps from Eq. 8, fed into Eq. 15, recovers L_W = 10 lg(P/P0) exactly."""
    p, s, v2 = 3.0e-4, 2.0, (1.0e-3) ** 2
    eps = float(emission.radiation_factor(p, s, v2))
    lv = float(emission.velocity_level(math.sqrt(v2)))
    lw = float(emission.radiated_sound_power_level(lv, s, radiation_factor=eps))
    assert lw == pytest.approx(10.0 * math.log10(p / P0))


def test_power_level_impedance_constant() -> None:
    # eps = 1, S = S0 = 1 -> L_W = L_v + 10 lg(411/400)
    lv = 80.0
    lw = float(emission.radiated_sound_power_level(lv, 1.0))
    assert lw == pytest.approx(lv + 10.0 * math.log10(ZCN / ZC0))


def test_upper_limit_is_largest() -> None:
    # Part 1 (eps = 1) is an upper limit for any eps < 1.
    lv, s = 75.0, 2.0
    upper = float(emission.radiated_sound_power_level(lv, s))
    measured = float(emission.radiated_sound_power_level(lv, s, radiation_factor=0.4))
    assert upper > measured


def test_power_level_rejects_bad_area() -> None:
    with pytest.raises(ValueError, match="area"):
        emission.radiated_sound_power_level(80.0, 0.0)


# ---------------------------------------------------------------------------
# Extraneous-velocity correction (Table 2)
# ---------------------------------------------------------------------------
def test_k1a_table_values() -> None:
    table = {3: 3.0, 4: 2.0, 5: 2.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 0.0}
    for dlv, k in table.items():
        assert emission.extraneous_velocity_correction(float(dlv)) == pytest.approx(k)


def test_k1a_boundaries() -> None:
    assert emission.extraneous_velocity_correction(2.0) == 3.0  # dLv < 3 -> 3 dB
    assert emission.extraneous_velocity_correction(15.0) == 0.0  # dLv >= 10 -> 0


# ---------------------------------------------------------------------------
# Result bundle
# ---------------------------------------------------------------------------
def test_result_bundle_and_total() -> None:
    lv = np.array([70.0, 75.0, 72.0])
    eps = np.array([0.5, 0.8, 1.0])
    f = np.array([250.0, 500.0, 1000.0])
    res = emission.sound_power_from_vibration(
        lv, 1.5, radiation_factor=eps, frequencies=f
    )
    assert isinstance(res, emission.VibrationSoundPowerResult)
    assert res.area == 1.5
    # per-band level matches the standalone function
    assert np.allclose(
        res.sound_power_level,
        emission.radiated_sound_power_level(lv, 1.5, radiation_factor=eps),
    )
    # total is the energetic sum of the bands
    expected_total = 10.0 * math.log10(np.sum(10.0 ** (0.1 * res.sound_power_level)))
    assert res.total_level == pytest.approx(expected_total)


def test_result_frequencies_shape_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"sound_power_from_vibration: 'velocity_level' .*"
        r"'frequencies' .*same shape",
    ):
        emission.sound_power_from_vibration(
            [70.0, 75.0, 72.0], 1.5, frequencies=[250.0, 500.0]
        )


def test_result_scalar_frequency_is_coerced() -> None:
    # a single-band call with a scalar frequency plots without error
    res = emission.sound_power_from_vibration(80.0, 2.0, frequencies=1000.0)
    assert res.frequencies is not None
    assert res.frequencies.shape == (1,)


def test_result_scalar_radiation_factor_broadcasts() -> None:
    res = emission.sound_power_from_vibration(np.array([70.0, 72.0]), 2.0)
    assert res.radiation_factor.shape == (2,)
    assert np.all(res.radiation_factor == 1.0)


def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    res = emission.sound_power_from_vibration(
        np.array([70.0, 75.0, 72.0]), 1.5, frequencies=np.array([250.0, 500.0, 1000.0])
    )
    assert res.plot() is not None


# ---------------------------------------------------------------------------
# Per-band quantities that do not run over the band axis
# ---------------------------------------------------------------------------
def _four_band_determination() -> emission.VibrationSoundPowerResult:
    """A four-band ISO/TS 7849-2 determination with a determined epsilon."""
    return emission.sound_power_from_vibration(
        [100.0, 102.0, 104.0, 101.0],
        1.5,
        radiation_factor=[0.8, 0.9, 1.0, 1.0],
        frequencies=[250.0, 500.0, 1000.0, 2000.0],
    )


@pytest.mark.parametrize("field_name", ["velocity_level", "radiation_factor"])
@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_a_vibration_band_quantity_off_the_band_axis_is_refused(
    field_name: str, trim: bool
) -> None:
    """The fiche reads these two columns at the row indices of ``LW``.

    The table takes its row count from ``sound_power_level``, so one entry
    too many is dropped from the sheet without a word (the surplus band is
    nowhere on the rendered page, under a header that says nothing is
    missing) and one too few stops the render with a bare ``IndexError``.
    """
    import dataclasses

    result = _four_band_determination()
    values = np.asarray(getattr(result, field_name))
    wrong = values[:-1] if trim else np.append(values, values[-1])
    with pytest.raises(ValueError, match=f"'{field_name}'"):
        dataclasses.replace(result, **{field_name: wrong})


def test_a_stray_single_frequency_is_refused() -> None:
    """One frequency beside four bands is read by ``sound_power_level_a``.

    The property picks an A-weighting correction per band and adds it to the
    band levels, so a lone frequency stretches its single correction over the
    whole spectrum: these bands then report L_WA = 109,7 dB where their own
    centres give 108,8 dB, and nothing on the way says which one is meant.
    """
    import dataclasses

    result = _four_band_determination()
    one_band = np.asarray(result.frequencies)[2:3]
    with pytest.raises(ValueError, match="'frequencies'"):
        dataclasses.replace(result, frequencies=one_band)


@pytest.mark.parametrize(
    "field_name",
    ["velocity_level", "sound_power_level", "radiation_factor", "frequencies"],
)
def test_a_non_finite_band_quantity_is_refused(field_name: str) -> None:
    """A NaN band changes which quantity the fiche boxes and compares.

    ``sound_power_level_a`` sums through every band, so one NaN ``LW`` turns
    L_WA into NaN and the sheet falls back to the unweighted total, whose
    energy sum skips the NaN band: the box then carries a partial ``LW`` and
    the verdict compares *that* against the declared A-weighted limit, which
    flips the printed PASS/FAIL on unchanged physical data. An all-NaN
    ``radiation_factor`` fails the ``epsilon = 1`` survey test, so the basis
    line asserts an engineering-method determination whose every epsilon cell
    is an em dash. :func:`sound_power_from_vibration` computes finite levels
    from finite inputs, so none of these is ever the library's own output.
    """
    import dataclasses

    result = _four_band_determination()
    values = np.asarray(getattr(result, field_name), dtype=float).copy()
    values[1] = float("nan")
    with pytest.raises(ValueError, match=f"VibrationSoundPowerResult: '{field_name}'"):
        dataclasses.replace(result, **{field_name: values})


def test_a_non_finite_radiating_area_is_refused() -> None:
    """A NaN area prints ``S = nan m2`` beside the boxed sound-power result.

    The area is a scalar of the measurement chain, fixed by the test setup and
    pinned positive by :func:`sound_power_from_vibration`, so no producer can
    leave it undetermined; the fiche prints it unconditionally in the extended
    terms of a fully rendered page.
    """
    import dataclasses

    result = _four_band_determination()
    with pytest.raises(ValueError, match="'area' must be positive"):
        dataclasses.replace(result, area=float("nan"))


def test_a_determination_covering_no_band_is_refused() -> None:
    """Four empty columns agree, and the sheet comes out complete and blank.

    Length-0 arrays satisfy every rank and count, so the fiche rendered a
    full accredited sound-power test sheet: the ISO/TS 7849-1 basis line, an
    empty per-band table, a boxed "Sound power level LW = — dB re 1 pW" over
    the em dash an energy sum of no bands leaves behind, and the disclaimer
    under it. It is reachable straight from the entry point, which read an
    empty velocity level as a determination of nothing.
    """
    empty = np.array([], dtype=float)
    with pytest.raises(
        ValueError, match="'sound_power_level' must carry at least one band"
    ):
        emission.sound_power_from_vibration(empty, 1.5, frequencies=empty)


def test_a_broadband_determination_is_not_read_as_empty() -> None:
    """One level and no band axis is one band's worth of data, not none."""
    result = emission.sound_power_from_vibration(100.0, 1.5)
    assert result.frequencies is None
    levels = np.atleast_1d(result.sound_power_level)
    assert levels.size == 1
    assert result.total_level == pytest.approx(float(levels[0]))


def test_radiated_power_norton_diesel_engine_example() -> None:
    # Norton & Karczub, Fundamentals of Noise and Vibration Analysis for
    # Engineers 2e (CUP, 2003), problem 3.9 (p. 580) with the published
    # answer (p. 611): a diesel engine approximated by a 1.2 m cube (five
    # radiating faces, S = 7.2 m2) with v_rms = 12.2 mm/s in the 500 Hz
    # octave band and radiation ratio sigma = 0.25 radiates W = 0.1112 W,
    # i.e. Lw = 110.5 dB re 1 pW — the ISO/TS 7849-style vibration-velocity
    # method. The book uses rho*c ~ 415 N.s/m3 against the standard's
    # normalized 411 N.s/m3 (0.04 dB) and rounds to 0.1 dB, so 0.15 dB
    # covers both.
    lv = 20.0 * math.log10(12.2e-3 / 5e-8)  # dB re 5e-8 m/s
    lw = emission.radiated_sound_power_level(lv, 7.2, radiation_factor=0.25)
    assert float(lw) == pytest.approx(110.5, abs=0.15)
