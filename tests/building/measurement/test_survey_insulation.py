#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 10052 field survey method module."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from phonometry import building

_OCTAVE = 5  # ISO 10052 octave-band count (125-2000 Hz)


def _norm(volume: float) -> float:
    """The 10 lg(A0 T0 / (0,16 V)) offset, recomputed independently."""
    return 10.0 * np.log10(10.0 * 0.5 / (0.16 * volume))


# ---------------------------------------------------------------------------
# Reverberation index k (Formula (3))
# ---------------------------------------------------------------------------


def test_reverberation_index_formula_3() -> None:
    """k = 10 lg(T/T0), T0 = 0,5 s."""
    k = building.reverberation_index([0.5, 1.0, 0.25])
    np.testing.assert_allclose(k, [0.0, 10.0 * np.log10(2.0), 10.0 * np.log10(0.5)])


def test_reverberation_index_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        building.reverberation_index([0.5, 0.0])
    with pytest.raises(ValueError, match="t0"):
        building.reverberation_index([0.5], t0=0.0)


# ---------------------------------------------------------------------------
# Airborne (Formulas (2), (4), (5), (7))
# ---------------------------------------------------------------------------


def test_airborne_d_dnt_dn_rprime() -> None:
    """D, DnT = D + k, Dn and R' band by band."""
    k = np.full(_OCTAVE, 3.0)
    res = building.survey_airborne_insulation(
        np.full(_OCTAVE, 80.0), np.full(_OCTAVE, 45.0), k, volume=50.0, area=12.0
    )
    assert res.d[0] == pytest.approx(35.0)
    assert res.d_nt[0] == pytest.approx(38.0)  # D + k
    assert res.d_n[0] == pytest.approx(35.0 + 3.0 + _norm(50.0))
    # S_eff = max(12, 50/7.5 = 6.67) = 12
    assert res.r_prime[0] == pytest.approx(
        35.0 + 3.0 + 10.0 * np.log10(12.0 * 0.5 / (0.16 * 50.0))
    )


def test_airborne_v_over_7_5_rule() -> None:
    """Where V/7,5 > S, the larger V/7,5 replaces S in R' (Clause 3.6)."""
    k = np.zeros(_OCTAVE)
    # V/7.5 = 120/7.5 = 16 > S = 5, so S_eff = 16.
    res = building.survey_airborne_insulation(
        np.full(_OCTAVE, 70.0), np.full(_OCTAVE, 40.0), k, volume=120.0, area=5.0
    )
    s_eff = 120.0 / 7.5
    assert res.r_prime[0] == pytest.approx(
        30.0 + 10.0 * np.log10(s_eff * 0.5 / (0.16 * 120.0))
    )


def test_airborne_without_volume_has_only_d_dnt() -> None:
    res = building.survey_airborne_insulation(
        np.full(_OCTAVE, 80.0), np.full(_OCTAVE, 45.0), np.zeros(_OCTAVE)
    )
    assert res.d_n is None
    assert res.r_prime is None
    assert res.rating is not None  # DnT,w still available (5 octave bands)
    assert res.r_prime_rating is None


def test_airborne_area_without_volume_raises() -> None:
    source, receiving = np.full(_OCTAVE, 80.0), np.full(_OCTAVE, 45.0)
    index = np.zeros(_OCTAVE)
    with pytest.raises(ValueError, match="requires 'volume'"):
        building.survey_airborne_insulation(source, receiving, index, area=12.0)


def test_airborne_energy_averages_positions() -> None:
    """A 2-D (positions, bands) source input is energy-averaged first."""
    l1 = np.array([[80.0, 80.0], [86.0, 84.0]])
    avg = 10.0 * np.log10(np.mean(10.0 ** (0.1 * l1), axis=0))
    multi = building.survey_airborne_insulation(l1, np.full(2, 40.0), np.zeros(2))
    single = building.survey_airborne_insulation(avg, np.full(2, 40.0), np.zeros(2))
    np.testing.assert_allclose(multi.d, single.d)


# ---------------------------------------------------------------------------
# Impact (Formulas (8), (9), (10))
# ---------------------------------------------------------------------------


def test_impact_li_lnt_ln() -> None:
    """Li energy-average, L'nT = Li - k, L'n."""
    li = np.array([[60.0] * _OCTAVE, [62.0] * _OCTAVE])
    expected_li = 10.0 * np.log10((10.0**6.0 + 10.0**6.2) / 2.0)
    k = np.full(_OCTAVE, 3.0)
    res = building.survey_impact_insulation(li, k, volume=50.0)
    assert res.l_i[0] == pytest.approx(expected_li)
    assert res.l_nt[0] == pytest.approx(expected_li - 3.0)
    assert res.l_n[0] == pytest.approx(expected_li - 3.0 - _norm(50.0))
    assert res.rating is not None


# ---------------------------------------------------------------------------
# Façade (Formulas (11), (12), (13))
# ---------------------------------------------------------------------------


def test_facade_d2m_family() -> None:
    k = np.full(_OCTAVE, 2.0)
    res = building.survey_facade_insulation(
        np.full(_OCTAVE, 75.0), np.full(_OCTAVE, 40.0), k, volume=40.0
    )
    assert res.d_2m[0] == pytest.approx(35.0)
    assert res.d_2m_nt[0] == pytest.approx(37.0)  # D2m + k
    assert res.d_2m_n[0] == pytest.approx(35.0 + 2.0 + _norm(40.0))
    assert res.rating is not None


# ---------------------------------------------------------------------------
# Service equipment (Formulas (14), (15), (16))
# ---------------------------------------------------------------------------


def test_service_equipment_lxy() -> None:
    """LXY is the energy average of the three positions; LXY,nT / LXY,n follow."""
    meas = [35.0, 30.0, 32.0]
    expected = 10.0 * np.log10((10.0**3.5 + 10.0**3.0 + 10.0**3.2) / 3.0)
    res = building.survey_service_equipment_level(meas, 3.0, volume=50.0)
    assert float(res.l_xy) == pytest.approx(expected)
    assert float(res.l_xy_nt) == pytest.approx(expected - 3.0)
    assert float(res.l_xy_n) == pytest.approx(expected - 3.0 - _norm(50.0))


def test_service_equipment_banded() -> None:
    """A (3, bands) input with a per-band k gives a per-band LXY."""
    meas = np.array([[35.0, 33.0], [30.0, 31.0], [32.0, 30.0]])
    k = np.array([3.0, 2.0])
    res = building.survey_service_equipment_level(meas, k)
    assert res.l_xy.shape == (2,)
    assert res.l_xy_n is None  # no volume


def test_service_equipment_requires_three_positions() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        building.survey_service_equipment_level([35.0, 30.0], 3.0)


def test_service_equipment_index_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="match the shape"):
        # Scalar average but a per-band k.
        building.survey_service_equipment_level([35.0, 30.0, 32.0], [3.0, 2.0])


def test_service_equipment_scalar_measurements_raises_cleanly() -> None:
    # A 0-d input must give a clean ValueError, not an IndexError.
    with pytest.raises(ValueError, match="exactly three"):
        building.survey_service_equipment_level(35.0, 3.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reverberation-index shape validation shared by the measurement functions
# ---------------------------------------------------------------------------


def test_index_band_count_must_match_levels() -> None:
    source, receiving = np.full(_OCTAVE, 80.0), np.full(_OCTAVE, 45.0)
    short_index = np.zeros(3)
    with pytest.raises(ValueError, match="one value per band"):
        building.survey_airborne_insulation(source, receiving, short_index)


def test_rating_none_off_band_count() -> None:
    """Neither 5 nor 16 bands => no automatic rating, and plot() raises."""
    res = building.survey_airborne_insulation(
        np.full(4, 80.0), np.full(4, 45.0), np.zeros(4)
    )
    assert res.rating is None
    with pytest.raises(ValueError, match="5 octave"):
        res.plot()


# ---------------------------------------------------------------------------
# One band set per result (the companion quantities nobody reads again)
# ---------------------------------------------------------------------------


def test_airborne_result_refuses_a_normalized_curve_of_another_length() -> None:
    """Dn is a column of the airborne result that nothing downstream measures.

    plot() draws from the ISO 717-1 rating and the field fiche reads exactly
    one of DnT and R', whichever report() was asked for, so D, Dn and the
    unreported quantity reach the reader and are never looked at again. With
    the check off, a Dn one band longer plots and reports without a word.
    """
    res = building.survey_airborne_insulation(
        np.full(_OCTAVE, 80.0), np.full(_OCTAVE, 50.0), np.zeros(_OCTAVE), volume=50.0
    )
    longer = np.append(res.d_n, 99.9)
    with pytest.raises(ValueError, match=r"'d_n' \(6\).*per band"):
        dataclasses.replace(res, d_n=longer)


def test_impact_result_refuses_a_normalized_level_of_another_length() -> None:
    """L'n is carried to the reader; only L'nT is drawn and reported.

    The impact fiche reads L'nT alone, and plot() draws from the ISO 717-2
    rating, so Li and L'n running over a different band set is a disagreement
    nothing downstream would ever raise.
    """
    res = building.survey_impact_insulation(
        np.full(_OCTAVE, 70.0), np.zeros(_OCTAVE), volume=50.0
    )
    longer = np.append(res.l_n, 99.9)
    with pytest.raises(ValueError, match=r"'l_n' \(6\).*per band"):
        dataclasses.replace(res, l_n=longer)


def test_facade_result_refuses_a_normalized_curve_of_another_length() -> None:
    """D2m,n reaches neither the facade sheet nor the plot.

    Both take the reported D2m,nT, so a normalized form covering a different
    span of bands is handed over unchallenged unless the result refuses it at
    construction.
    """
    res = building.survey_facade_insulation(
        np.full(_OCTAVE, 90.0), np.full(_OCTAVE, 50.0), np.zeros(_OCTAVE), volume=50.0
    )
    longer = np.append(res.d_2m_n, 99.9)
    with pytest.raises(ValueError, match=r"'d_2m_n' \(6\).*per band"):
        dataclasses.replace(res, d_2m_n=longer)


# ---------------------------------------------------------------------------
# Reverberation-index estimation (ISO 10052:2021 Table 4)
# ---------------------------------------------------------------------------

# Independent transcription of representative Table 4 cells (125/250/500/1000/
# 2000 Hz), for a dual-source check against the module's own table.
_TABLE4_SPOT = {
    (10.0, "kitchen"): [0.0, 0.0, 0.0, 0.0, 0.0],
    (10.0, "bathroom"): [1.0, 1.0, 0.0, 0.0, -0.5],
    (10.0, "furnished"): [0.0, 0.0, -0.5, -0.5, -1.0],
    (10.0, "b"): [1.0, 2.5, 3.0, 2.5, 2.0],
    (10.0, "h"): [4.0, 4.5, 5.0, 5.0, 4.5],
    (10.0, "c+g"): [2.0, 3.5, 4.0, 4.5, 4.5],
    (25.0, "kitchen"): [0.0, 0.5, 0.0, 0.0, 0.0],
    (25.0, "g"): [4.0, 5.0, 5.0, 5.0, 5.0],
    (50.0, "furnished"): [0.5, 0.5, 0.5, 0.0, 0.0],
    (50.0, "c+g"): [3.0, 4.5, 5.0, 5.5, 5.0],
    (50.0, "h"): [5.0, 5.5, 6.0, 5.0, 5.5],
    (100.0, "furnished"): [0.5, 0.5, 0.5, 0.5, 0.0],
    (100.0, "a"): [1.0, 2.5, 2.5, 2.0, 1.5],
    (100.0, "g"): [5.0, 5.5, 6.0, 6.0, 6.0],
    (100.0, "d+h"): [4.0, 5.0, 5.5, 5.5, 6.0],
}

# A-/C-weighted column anchors for the same spots.
_TABLE4_AC = {
    (10.0, "furnished"): -0.5,
    (25.0, "g"): 5.5,
    (50.0, "c+g"): 5.5,
    (100.0, "h"): 6.0,
}


def test_estimate_matches_table4_spot_cells() -> None:
    """The module table reproduces the independent Table 4 transcription."""
    for (volume, room), expected in _TABLE4_SPOT.items():
        got = building.estimate_reverberation_index(volume, room)
        np.testing.assert_allclose(got, expected, err_msg=f"{volume} {room}")


def test_estimate_weighted_column() -> None:
    for (volume, room), expected in _TABLE4_AC.items():
        got = building.estimate_reverberation_index(volume, room, weighted=True)
        assert got == pytest.approx(expected)


def test_estimate_furnished_aliases() -> None:
    """'other'/'others' map to the general furnished row."""
    base = building.estimate_reverberation_index(10.0, "furnished")
    np.testing.assert_allclose(
        building.estimate_reverberation_index(10.0, "other"), base
    )
    np.testing.assert_allclose(
        building.estimate_reverberation_index(10.0, "others"), base
    )


def test_estimate_room_is_case_and_space_insensitive() -> None:
    """Room labels are normalized (trimmed + lower-cased) before lookup."""
    base = building.estimate_reverberation_index(10.0, "kitchen")
    np.testing.assert_allclose(
        building.estimate_reverberation_index(10.0, " Kitchen "), base
    )
    np.testing.assert_allclose(
        building.estimate_reverberation_index(10.0, "KITCHEN"), base
    )
    furn = building.estimate_reverberation_index(10.0, "furnished")
    np.testing.assert_allclose(
        building.estimate_reverberation_index(10.0, " OTHER "), furn
    )


def test_estimate_volume_range_boundaries() -> None:
    """Boundaries follow V<15, 15<=V<35, 35<=V<60, 60<=V<=150."""
    # type a straddling the 15 m³ boundary: <15 vs 15-35 rows differ.
    np.testing.assert_allclose(
        building.estimate_reverberation_index(14.9, "a"),
        [0.0, 1.0, 1.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        building.estimate_reverberation_index(15.0, "a"),
        [1.0, 1.5, 1.5, 1.0, 0.5],
    )
    # 150 m³ is the inclusive upper limit of the last range.
    assert building.estimate_reverberation_index(150.0, "a").shape == (5,)


def test_estimate_feeds_survey_functions() -> None:
    """An estimated k is a drop-in for the measurement functions."""
    k = building.estimate_reverberation_index(50.0, "g")
    res = building.survey_airborne_insulation(np.full(5, 80.0), np.full(5, 45.0), k)
    np.testing.assert_allclose(res.d_nt, 35.0 + np.asarray(k))


def test_estimate_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at most 150"):
        building.estimate_reverberation_index(200.0, "a")
    # The error names the actual volume range, not an internal index.
    with pytest.raises(ValueError, match=r"60 <= V <= 150"):
        building.estimate_reverberation_index(100.0, "kitchen")  # only V < 35
    with pytest.raises(ValueError, match="not tabulated"):
        building.estimate_reverberation_index(10.0, "z")
    with pytest.raises(ValueError, match="positive"):
        building.estimate_reverberation_index(0.0, "a")


def test_estimate_class_ordering() -> None:
    """Heavier construction (h) never has a lower index than lighter (a)."""
    for volume in (10.0, 25.0, 50.0, 100.0):
        a = building.estimate_reverberation_index(volume, "a")
        h = building.estimate_reverberation_index(volume, "h")
        assert np.all(np.asarray(h) >= np.asarray(a))
