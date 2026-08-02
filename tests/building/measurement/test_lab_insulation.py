#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for ISO 10140 laboratory sound insulation.

Validation strategy: closed-form identities from the standards' own
formulae, and consistency with the verified ISO 717-1/2 rating engine.

- Airborne ``R = L1 - L2 + 10 lg(S/A)`` (ISO 10140-2 Formula (2)) with
  ``A = 0,16 V / T`` (ISO 10140-4 Formula (5)): reduces to ``L1 - L2`` when
  ``S = A``, and adds ``10 lg(S/A)`` exactly for a known ratio.
- Impact ``Ln = Li + 10 lg(A/A0)`` (ISO 10140-3 Formula (1)) with
  ``A0 = 10 m²``: reduces to ``Li`` when ``A = A0 = 10``.
- The automatic single-number ratings match direct calls to
  :func:`weighted_rating` / :func:`weighted_impact_rating` on the per-band
  quantity, and reproduce a curve laid on the ISO 717 reference.
- Background correction (ISO 10140-4 Clause 4.3, Formula (4)): the 6/15 dB
  criteria and the fixed 1,3 dB limit-of-measurement cap.
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    LabAirborneInsulationResult,
    LabImpactInsulationResult,
    background_correction,
    lab_airborne_insulation,
    lab_impact_insulation,
    weighted_impact_rating,
    weighted_rating,
)
from phonometry.building.measurement.lab_insulation import LabInsulationWarning

# ISO 717-1 Table 3 airborne reference (100-3150 Hz, 16 bands).
_REF_AIRBORNE = np.array(
    [33, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 56, 56, 56, 56],
    dtype=np.float64,
)
# ISO 717-2 Table 3 impact reference (100-3150 Hz, 16 bands).
_REF_IMPACT = np.array(
    [62, 62, 62, 62, 62, 62, 61, 60, 59, 58, 57, 54, 51, 48, 45, 42],
    dtype=np.float64,
)


# --- Airborne R (ISO 10140-2) --------------------------------------------


def test_r_reduces_to_level_difference_when_s_equals_a() -> None:
    # A = 0,16 * 50 / 0,8 = 10 m² per band; S = 10 => 10 lg(S/A) = 0.
    l1 = np.full(16, 90.0)
    l2 = np.full(16, 40.0)
    t2 = np.full(16, 0.8)
    res = lab_airborne_insulation(l1, l2, t2, area=10.0, volume=50.0)
    assert np.allclose(res.absorption, 10.0)
    assert np.allclose(res.r, 50.0)


def test_r_adds_ten_lg_s_over_a() -> None:
    # A = 10 m², S = 20 => 10 lg(2) added to (L1 - L2) = 50.
    l1 = np.full(16, 90.0)
    l2 = np.full(16, 40.0)
    t2 = np.full(16, 0.8)
    res = lab_airborne_insulation(l1, l2, t2, area=20.0, volume=50.0)
    assert np.allclose(res.r, 50.0 + 10.0 * np.log10(2.0))


def test_r_absorption_follows_sabine_per_band() -> None:
    t2 = np.linspace(0.5, 1.5, 16)
    res = lab_airborne_insulation(
        np.full(16, 80.0), np.full(16, 30.0), t2, area=12.0, volume=60.0
    )
    assert np.allclose(res.absorption, 0.16 * 60.0 / t2)


def test_r_energy_averages_positions() -> None:
    # Two source positions 90 and 96 dB energy-average above their mean.
    l1 = np.vstack([np.full(16, 90.0), np.full(16, 96.0)])
    l2 = np.full(16, 40.0)
    t2 = np.full(16, 0.8)
    res = lab_airborne_insulation(l1, l2, t2, area=10.0, volume=50.0)
    expected_l1 = 10.0 * np.log10((10 ** 9.0 + 10 ** 9.6) / 2.0)
    assert np.allclose(res.r, expected_l1 - 40.0)


def test_airborne_rating_matches_direct_engine() -> None:
    l1 = np.full(16, 90.0)
    l2 = 90.0 - _REF_AIRBORNE  # R equals the reference curve (S = A).
    t2 = np.full(16, 0.8)
    res = lab_airborne_insulation(l1, l2, t2, area=10.0, volume=50.0)
    assert res.rating is not None
    # R exactly equal to the ISO 717-1 reference curve shape => Rw = 54 dB: the
    # 32 dB unfavourable-deviation allowance permits a 2 dB upward shift of the
    # reference (32 dB / 16 bands), i.e. curve@500 Hz (52) + 2. Independent
    # anchor of the engine-consistency check below.
    assert res.rating.rating == 54
    direct = weighted_rating(res.r)
    assert res.rating.rating == direct.rating
    assert res.rating.c == direct.c
    assert res.rating.ctr == direct.ctr


def test_airborne_octave_bands_rate() -> None:
    l1 = np.full(5, 80.0)
    l2 = np.full(5, 30.0)
    t2 = np.full(5, 0.8)
    res = lab_airborne_insulation(l1, l2, t2, area=10.0, volume=50.0)
    assert res.rating is not None
    assert res.rating.rating == weighted_rating(res.r).rating


def test_airborne_extended_range_has_no_rating() -> None:
    # 18 bands (100-5000 Hz) cannot be rated by ISO 717-1 directly.
    res = lab_airborne_insulation(
        np.full(18, 80.0), np.full(18, 30.0), np.full(18, 0.8),
        area=10.0, volume=50.0,
    )
    assert res.rating is None
    assert res.r.shape == (18,)


# --- Impact Ln (ISO 10140-3) ---------------------------------------------


def test_ln_reduces_to_li_when_a_equals_a0() -> None:
    # A = 0,16 * 50 / 0,8 = 10 = A0 => Ln = Li.
    li = np.full(16, 60.0)
    t2 = np.full(16, 0.8)
    res = lab_impact_insulation(li, t2, volume=50.0)
    assert np.allclose(res.absorption, 10.0)
    assert np.allclose(res.l_n, 60.0)


def test_ln_adds_ten_lg_a_over_a0() -> None:
    # A = 0,16 * 100 / 0,8 = 20 => Ln = Li + 10 lg(20/10).
    li = np.full(16, 55.0)
    t2 = np.full(16, 0.8)
    res = lab_impact_insulation(li, t2, volume=100.0)
    assert np.allclose(res.l_n, 55.0 + 10.0 * np.log10(2.0))


def test_impact_rating_matches_direct_engine() -> None:
    # Ln equals the ISO 717-2 reference curve (A = A0 => Ln = Li).
    li = _REF_IMPACT.copy()
    t2 = np.full(16, 0.8)
    res = lab_impact_insulation(li, t2, volume=50.0)
    assert res.rating is not None
    # Ln exactly equal to the ISO 717-2 reference curve shape => Ln,w = 58 dB:
    # the 32 dB unfavourable-deviation allowance permits a 2 dB downward shift of
    # the reference (32 dB / 16 bands), i.e. curve@500 Hz (60) − 2. Independent
    # anchor of the engine-consistency check below.
    assert res.rating.rating == 58
    direct = weighted_impact_rating(res.l_n)
    assert res.rating.rating == direct.rating
    assert res.rating.ci == direct.ci


def test_impact_energy_averages_positions() -> None:
    li = np.vstack([np.full(16, 60.0), np.full(16, 66.0)])
    t2 = np.full(16, 0.8)
    res = lab_impact_insulation(li, t2, volume=50.0)  # A = A0 => Ln = Li_avg
    expected = 10.0 * np.log10((10 ** 6.0 + 10 ** 6.6) / 2.0)
    assert np.allclose(res.l_n, expected)


# --- Background correction (ISO 10140-4, Clause 4.3) ---------------------


def test_background_correction_formula_mid_margin() -> None:
    # Margin 10 dB (6 < 10 < 15): Formula (4).
    corrected = background_correction([60.0], [50.0])
    expected = 10.0 * np.log10(10 ** 6.0 - 10 ** 5.0)
    assert np.allclose(corrected, expected)
    assert np.allclose(corrected, 59.542425)


def test_background_correction_high_margin_unchanged() -> None:
    # Margin 20 dB (>= 15): no correction.
    corrected = background_correction([70.0], [50.0])
    assert np.allclose(corrected, 70.0)


def test_background_correction_exactly_15_unchanged() -> None:
    corrected = background_correction([65.0], [50.0])
    assert np.allclose(corrected, 65.0)


def test_background_correction_low_margin_capped_and_warns() -> None:
    # Margin 3 dB (<= 6): fixed 1,3 dB cap, warning.
    with pytest.warns(LabInsulationWarning):
        corrected = background_correction([53.0], [50.0])
    assert np.allclose(corrected, 53.0 - 1.3)


def test_background_correction_exactly_6_capped() -> None:
    with pytest.warns(LabInsulationWarning):
        corrected = background_correction([56.0], [50.0])
    assert np.allclose(corrected, 56.0 - 1.3)


def test_background_correction_per_band_mixed() -> None:
    lsb = np.array([70.0, 60.0, 53.0])  # margins 20, 10, 3 dB
    lb = np.array([50.0, 50.0, 50.0])
    with pytest.warns(LabInsulationWarning):
        corrected = background_correction(lsb, lb)
    assert np.allclose(corrected[0], 70.0)
    assert np.allclose(corrected[1], 10.0 * np.log10(10 ** 6.0 - 10 ** 5.0))
    assert np.allclose(corrected[2], 53.0 - 1.3)


def test_background_correction_feeds_r() -> None:
    # End-to-end: correct L2 then form R.
    l1 = np.full(16, 90.0)
    l2_raw = np.full(16, 41.0)  # combined signal+background
    lb = np.full(16, 31.0)  # margin 10 dB
    l2 = background_correction(l2_raw, lb)
    res = lab_airborne_insulation(l1, l2, np.full(16, 0.8), area=10.0, volume=50.0)
    expected_l2 = 10.0 * np.log10(10 ** 4.1 - 10 ** 3.1)
    assert np.allclose(res.r, 90.0 - expected_l2)


# --- Validation ----------------------------------------------------------


def test_airborne_band_count_mismatch() -> None:
    with pytest.raises(ValueError, match="band count"):
        lab_airborne_insulation(
            np.full(16, 80.0), np.full(5, 30.0), np.full(16, 0.8),
            area=10.0, volume=50.0,
        )


def test_airborne_t2_band_mismatch() -> None:
    with pytest.raises(ValueError, match="band count"):
        lab_airborne_insulation(
            np.full(16, 80.0), np.full(16, 30.0), np.full(5, 0.8),
            area=10.0, volume=50.0,
        )


@pytest.mark.parametrize("area,volume", [(0.0, 50.0), (-1.0, 50.0)])
def test_airborne_bad_area(area: float, volume: float) -> None:
    with pytest.raises(ValueError, match="area"):
        lab_airborne_insulation(
            np.full(16, 80.0), np.full(16, 30.0), np.full(16, 0.8),
            area=area, volume=volume,
        )


def test_airborne_bad_volume() -> None:
    with pytest.raises(ValueError, match="volume"):
        lab_airborne_insulation(
            np.full(16, 80.0), np.full(16, 30.0), np.full(16, 0.8),
            area=10.0, volume=-5.0,
        )


def test_airborne_bad_t2() -> None:
    t2 = np.full(16, 0.8)
    t2[3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        lab_airborne_insulation(
            np.full(16, 80.0), np.full(16, 30.0), t2, area=10.0, volume=50.0
        )


def test_impact_t2_band_mismatch() -> None:
    with pytest.raises(ValueError, match="band count"):
        lab_impact_insulation(np.full(16, 60.0), np.full(5, 0.8), volume=50.0)


def test_impact_bad_volume() -> None:
    with pytest.raises(ValueError, match="volume"):
        lab_impact_insulation(np.full(16, 60.0), np.full(16, 0.8), volume=0.0)


def test_background_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        background_correction([60.0, 50.0], [50.0])


def test_result_types() -> None:
    a = lab_airborne_insulation(
        np.full(16, 80.0), np.full(16, 30.0), np.full(16, 0.8),
        area=10.0, volume=50.0,
    )
    i = lab_impact_insulation(np.full(16, 60.0), np.full(16, 0.8), volume=50.0)
    assert isinstance(a, LabAirborneInsulationResult)
    assert isinstance(i, LabImpactInsulationResult)


def test_plot_without_rating_raises() -> None:
    res = lab_airborne_insulation(
        np.full(18, 80.0), np.full(18, 30.0), np.full(18, 0.8),
        area=10.0, volume=50.0,
    )
    with pytest.raises(ValueError, match="rating"):
        res.plot()


# ---------------------------------------------------------------------------
# ISO 10140-5:2010+A1 reference elements (Tables B.1 / C.1) - printed anchors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r_name,rating_name", [
    ("ISO10140_5_B1_HEAVY_WALL_R", "ISO10140_5_B1_HEAVY_WALL_RATING"),
    ("ISO10140_5_B1_HEAVY_FLOOR_R", "ISO10140_5_B1_HEAVY_FLOOR_RATING"),
    ("ISO10140_5_B1_LIGHT_WALL_R", "ISO10140_5_B1_LIGHT_WALL_RATING"),
])
def test_reference_element_airborne_end_to_end(
    r_name: str, rating_name: str
) -> None:
    """Table B.1 reference elements reproduce their printed Rw (C; Ctr).

    End to end: with S = A (S = 10 m2, A = 0,16*50/0,8 = 10 m2) the level
    difference equals the tabulated R, so the whole ISO 10140-2 Formula (2)
    -> ISO 717-1 chain must return the printed single numbers.
    """
    r = np.asarray(getattr(ref, r_name), dtype=float)
    rw, c, ctr = getattr(ref, rating_name)
    res = lab_airborne_insulation(
        np.full(16, 90.0), 90.0 - r, np.full(16, 0.8), area=10.0, volume=50.0
    )
    np.testing.assert_allclose(res.r, r, atol=1e-9)
    assert res.rating is not None
    assert (res.rating.rating, res.rating.c, res.rating.ctr) == (rw, c, ctr)


@pytest.mark.parametrize("ln_name,rating_name", [
    ("ISO10140_5_C1_FLOOR_C1C2_LN", "ISO10140_5_C1_FLOOR_C1C2_RATING"),
    ("ISO10140_5_C1_FLOOR_C3_LN", "ISO10140_5_C1_FLOOR_C3_RATING"),
])
def test_reference_floor_impact_end_to_end(
    ln_name: str, rating_name: str
) -> None:
    """Table C.1 reference floors reproduce their printed Ln,t,r,0,w (CI).

    End to end: with A = A0 (V = 31,25 m3, T = 0,5 s -> A = 10 m2) the
    receiving level equals the tabulated Ln, so the ISO 10140-3 Formula (1)
    -> ISO 717-2 chain must return the printed single numbers.
    """
    ln = np.asarray(getattr(ref, ln_name), dtype=float)
    lnw, ci = getattr(ref, rating_name)
    res = lab_impact_insulation(ln, np.full(16, 0.5), volume=31.25)
    np.testing.assert_allclose(res.l_n, ln, atol=1e-9)
    assert res.rating is not None
    assert (res.rating.rating, res.rating.ci) == (lnw, ci)
