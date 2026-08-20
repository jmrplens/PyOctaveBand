#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for ISO 16283-2 field impact sound insulation and ISO 717-2
weighted impact ratings (CI).

Validation strategy: the standards' own numbers, not self-consistency.

- The weighted impact rating, CI and the unfavourable-deviation sum are
  checked against the ISO 717-2 Annex C worked examples: Table C.1
  (laboratory, one-third octave: Ln,w = 79, CI = -11, sum 28,0) and
  Table C.3 (in situ, octave: Ln,w = 54, CI = 0, sum 7,8 -- exercising
  the octave-band -5 dB rule).
- The 32,0 dB (16 thirds) / 10,0 dB (5 octaves) bound is exercised with a
  curve that hits it exactly and one that tips over.
- L'nT (Formula (1)) reduces to Li when T = T0 = 0,5 s; L'n (Formula (2)
  with A = 0,16 V / T, A0 = 10 m2) is checked against hand values.
- The shared shift engine is re-verified against an independent brute-force
  impact-shift search on 10 000 random curves for both band sets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from reference_data import ISO717_2_ANNEX_C1_LN

from phonometry import building

# ISO 717-2 Table 3 reference values.
_REF_IMPACT_THIRD = [
    62,
    62,
    62,
    62,
    62,
    62,
    61,
    60,
    59,
    58,
    57,
    54,
    51,
    48,
    45,
    42,
]
_REF_IMPACT_OCTAVE = [67, 67, 65, 62, 49]
_INDEX_500_THIRD = 7
_INDEX_500_OCTAVE = 2

# ISO 717-2 Annex C, Table C.1 measured Ln (one-third octave, 100-3150).
# Shared oracle with the CI conformance report (tests/reference_data/).
_ANNEX_C1_LN = list(ISO717_2_ANNEX_C1_LN)
# ISO 717-2 Annex C, Table C.3 measured Ln (octave, 125-2000).
_ANNEX_C3_LN = [65.3, 64.5, 58.0, 55.8, 43.0]


def _round_half_up_tenths(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) * 10.0 + 0.5) / 10.0


def _brute_force_impact_rating(
    measured: list[float],
    reference: list[float],
    limit: float,
    index_500: int,
    octave_offset: int,
) -> tuple[int, float]:
    """Independent brute-force impact shift search (ISO 717-2 Clause 4.3).

    Unfavourable deviation = measured above reference. Find the smallest
    integer shift k with the deviation sum <= limit; rating is the shifted
    reference at 500 Hz (minus 5 dB for octave bands).
    """
    meas = _round_half_up_tenths(np.asarray(measured, dtype=np.float64))
    ref = np.asarray(reference, dtype=np.float64)
    best_k = None
    for k in range(-200, 201):
        dev = float(np.sum(np.maximum(0.0, meas - (ref + k))))
        if dev <= limit + 1e-6:
            best_k = k
            break
    assert best_k is not None
    dev = float(np.sum(np.maximum(0.0, meas - (ref + best_k))))
    rating = int(reference[index_500]) + best_k + octave_offset
    return rating, dev


def _brute_force_ci(measured: list[float], rating: int, n_bands: int) -> int:
    """Independent CI (ISO 717-2 A.2.1): energetic sum 100-2500 - 15 - Ln,w."""
    meas = _round_half_up_tenths(np.asarray(measured, dtype=np.float64))
    subset = meas[:n_bands]
    l_sum = 10.0 * np.log10(np.sum(10.0 ** (subset / 10.0)))
    return math.floor(l_sum + 0.5) - 15 - rating


# --------------------------------------------------------------------------
# ISO 717-2 weighted impact rating and CI
# --------------------------------------------------------------------------


def test_annex_c1_worked_example_third_octave() -> None:
    """ISO 717-2 Annex C Table C.1: Ln,w = 79, CI = -11, sum 28,0."""
    res = building.weighted_impact_rating(_ANNEX_C1_LN)
    assert isinstance(res, building.ImpactRatingResult)
    assert res.rating == 79
    assert res.ci == -11
    assert res.unfavourable_sum == pytest.approx(28.0, abs=1e-9)


def test_annex_c3_worked_example_octave() -> None:
    """ISO 717-2 Annex C Table C.3 (octave, -5 dB rule): Ln,w = 54, CI = 0."""
    res = building.weighted_impact_rating(_ANNEX_C3_LN)
    assert res.rating == 54
    assert res.ci == 0
    assert res.unfavourable_sum == pytest.approx(7.8, abs=1e-9)


def test_octave_minus_five_rule_vs_third_octave() -> None:
    """The octave -5 dB reduction (4.3.2) is applied only to octave bands.

    With measured == reference, the reference must shift down 2 dB so the
    deviation sum reaches the bound (16 * 2 = 32,0 / 5 * 2 = 10,0).
    """
    # One-third octave: shift down 2 dB, read at 500 Hz => 60 - 2 = 58.
    third = building.weighted_impact_rating(_REF_IMPACT_THIRD)
    assert third.rating == _REF_IMPACT_THIRD[_INDEX_500_THIRD] - 2  # 58
    assert third.unfavourable_sum == pytest.approx(32.0, abs=1e-9)
    # Octave: shift down 2 dB then -5 dB => 65 - 2 - 5 = 58.
    octave = building.weighted_impact_rating(_REF_IMPACT_OCTAVE)
    assert octave.rating == _REF_IMPACT_OCTAVE[_INDEX_500_OCTAVE] - 2 - 5  # 58
    assert octave.unfavourable_sum == pytest.approx(10.0, abs=1e-9)


def test_unfavourable_sum_exactly_at_bound_third_octave() -> None:
    """Measured = reference + 2 everywhere => sum = 32,0 exactly at k = 0."""
    measured = [r + 2.0 for r in _REF_IMPACT_THIRD]
    res = building.weighted_impact_rating(measured)
    # The unshifted reference already sits 2 dB below measured (sum 32,0),
    # so no shift is needed; the rating is read at 500 Hz unchanged.
    assert res.rating == _REF_IMPACT_THIRD[_INDEX_500_THIRD]  # 60
    assert res.unfavourable_sum == pytest.approx(32.0, abs=1e-9)


def test_unfavourable_sum_tips_over_forces_one_more_db() -> None:
    """0,1 dB over the 32,0 bound forces one more decibel of upward shift."""
    measured = [r + 2.0 for r in _REF_IMPACT_THIRD]
    measured[0] += 0.1  # sum would be 32,1 at k = 0
    res = building.weighted_impact_rating(measured)
    assert res.rating == _REF_IMPACT_THIRD[_INDEX_500_THIRD] + 1  # 61
    # At +1 dB: 15 bands contribute 1,0 dB each and the bumped band 1,1 dB.
    assert res.unfavourable_sum == pytest.approx(16.1, abs=1e-9)


def test_explicit_band_set_override() -> None:
    res = building.weighted_impact_rating(_ANNEX_C1_LN, bands="third-octave")
    assert res.rating == 79
    res_oct = building.weighted_impact_rating(_ANNEX_C3_LN, bands="octave")
    assert res_oct.rating == 54


def test_measured_data_rounded_to_one_decimal() -> None:
    """Clause 4.3.1 footnote 1: inputs reduced to 0,1 dB (round half up)."""
    res = building.weighted_impact_rating(_ANNEX_C1_LN)
    perturbed = [v + 0.049 for v in _ANNEX_C1_LN]  # rounds back to originals
    assert building.weighted_impact_rating(perturbed).rating == res.rating
    assert building.weighted_impact_rating(perturbed).ci == res.ci


def test_weighted_impact_rating_rejects_bad_length() -> None:
    with pytest.raises(ValueError, match="Expected 16 one-third-octave"):
        building.weighted_impact_rating([1.0, 2.0, 3.0])


def test_weighted_impact_rating_rejects_nan() -> None:
    bad = list(_ANNEX_C1_LN)
    bad[0] = float("nan")
    with pytest.raises(
        ValueError, match="'values_by_band' must contain only finite values"
    ):
        building.weighted_impact_rating(bad)


def test_engine_matches_brute_force_third_octave() -> None:
    """10 000 random third-octave curves: shared engine == brute force."""
    rng = np.random.default_rng(20262)
    for _ in range(10_000):
        curve = rng.uniform(20.0, 90.0, size=16)
        res = building.weighted_impact_rating(curve)
        rating, dev = _brute_force_impact_rating(
            list(curve), _REF_IMPACT_THIRD, 32.0, _INDEX_500_THIRD, 0
        )
        ci = _brute_force_ci(list(curve), rating, 15)
        assert res.rating == rating
        assert res.ci == ci
        assert res.unfavourable_sum == pytest.approx(dev, abs=1e-9)


def test_engine_matches_brute_force_octave() -> None:
    """10 000 random octave curves: shared engine == brute force (+ -5 dB)."""
    rng = np.random.default_rng(20263)
    for _ in range(10_000):
        curve = rng.uniform(20.0, 90.0, size=5)
        res = building.weighted_impact_rating(curve)
        rating, dev = _brute_force_impact_rating(
            list(curve), _REF_IMPACT_OCTAVE, 10.0, _INDEX_500_OCTAVE, -5
        )
        ci = _brute_force_ci(list(curve), rating, 5)
        assert res.rating == rating
        assert res.ci == ci
        assert res.unfavourable_sum == pytest.approx(dev, abs=1e-9)


# --------------------------------------------------------------------------
# ISO 16283-2 field quantities
# --------------------------------------------------------------------------


def test_lnt_equals_li_when_t_is_half_second() -> None:
    """Formula (1): T = T0 = 0,5 s => L'nT = Li exactly."""
    li = np.array([60.0, 62.0, 55.0])
    t2 = np.full(3, 0.5)
    res = building.impact_insulation(li, t2)
    assert isinstance(res, building.ImpactInsulationResult)
    np.testing.assert_allclose(res.l_n_t, li)


def test_lnt_decreases_with_reverberation_time() -> None:
    """Formula (1) minus sign: T = 5 s, T0 = 0,5 s => L'nT = Li - 10 dB."""
    li = np.array([60.0])
    res = building.impact_insulation(li, np.array([5.0]))
    np.testing.assert_allclose(res.l_n_t, [50.0])


def test_normalized_level_formula2() -> None:
    """Formula (2): A = 0,16 V / T = A0 = 10 => 10 lg(A/A0) = 0 => L'n = Li."""
    li = np.array([60.0, 62.0])
    t2 = np.array([1.0, 1.0])
    # A = 0,16 * V / T = 10 for V = 62,5, T = 1 => A/A0 = 1.
    res = building.impact_insulation(li, t2, volume=62.5)
    assert res.l_n is not None
    np.testing.assert_allclose(res.l_n, li)


def test_normalized_level_ten_db_offset() -> None:
    """A = 100 m2 (V = 625, T = 1) => A/A0 = 10 => L'n = Li + 10."""
    li = np.array([60.0])
    res = building.impact_insulation(li, np.array([1.0]), volume=625.0)
    assert res.l_n is not None
    np.testing.assert_allclose(res.l_n, [70.0])


def test_l_n_none_without_volume() -> None:
    res = building.impact_insulation(np.array([60.0]), np.array([0.5]))
    assert res.l_n is None


def test_impact_energy_averages_positions() -> None:
    """2-D li (positions x bands) is energy-averaged (Formula (10))."""
    li = np.array([[60.0, 50.0], [50.0, 60.0]])  # two positions, two bands
    res = building.impact_insulation(li, np.array([0.5, 0.5]))
    avg = 10.0 * np.log10((10**6 + 10**5) / 2.0)
    np.testing.assert_allclose(res.l_n_t, [avg, avg])


def test_impact_rejects_length_mismatch() -> None:
    li_2_bands = np.array([60.0, 60.0])
    t2_1_band = np.array([0.5])
    with pytest.raises(
        ValueError, match="'li' and 't2' must share the same band count"
    ):
        building.impact_insulation(li_2_bands, t2_1_band)


def test_impact_rejects_bad_reverberation_time() -> None:
    li = np.array([60.0])
    zero_t2 = np.array([0.0])
    with pytest.raises(ValueError, match="'t2' must contain positive, finite values"):
        building.impact_insulation(li, zero_t2)


def test_field_rating_pipeline_lnt_w() -> None:
    """L'nT per band (T = 0,5 s) fed to weighted_impact_rating gives L'nT,w."""
    li = np.array([float(r) for r in _REF_IMPACT_THIRD])  # Li == reference
    t2 = np.full(16, 0.5)
    res = building.impact_insulation(li, t2)
    rating = building.weighted_impact_rating(res.l_n_t)
    # Li == reference => shift down 2 dB (sum 32,0), read at 500 => 60 - 2.
    assert rating.rating == 58
