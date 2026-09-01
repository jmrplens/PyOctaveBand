#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 16283-1:2014 field airborne sound insulation and
ISO 717-1 weighted ratings (C / Ctr).

Validation strategy: the standards' own numbers, not self-consistency.

- The weighted rating and C / Ctr are checked against the worked example
  of ISO 717-1 Annex C, Table C.1 (measured R gives Rw = 30 with an
  unfavourable-deviation sum of 31,8 dB, C = -2, Ctr = -3).
- The unfavourable-deviation bound (Clause 4.4: <= 32,0 dB for 16
  one-third-octave bands, <= 10,0 dB for 5 octave bands) is exercised
  with a curve that hits the bound exactly and one that tips over,
  forcing one more decibel of shift.
- DnT (Formula (2)) reduces to D when T = T0 = 0,5 s; R' (Formula (4)
  with A = 0,16 V / T, Formula (5)) reduces to D for S T = 0,16 V.
- The energy-average level (Formula (9)) is checked against hand values.
- The shared shift engine is re-verified against an independent brute-force
  airborne-shift search on 10 000 random curves for both band sets, pinning
  the shared-engine float-tolerance behaviour on the airborne path too.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from reference_data import ISO717_1_ANNEX_C_R as _ANNEX_C_R
from reference_data import ISO717_2_ANNEX_C1_LN as _ANNEX_C1_LN

from phonometry import building

# One-third-octave reference values, ISO 717-1 Table 3 (100 Hz to 3150 Hz).
_REF_THIRD = [33, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 56, 56, 56, 56]
# Octave reference values, ISO 717-1 Table 3 (125 Hz to 2000 Hz).
_REF_OCTAVE = [36, 45, 52, 55, 56]
_INDEX_500_THIRD = 7
_INDEX_500_OCTAVE = 2

# ISO 717-1 Table 4 spectra (A-weighted, normalized to 0 dB).
_SPECTRUM1_THIRD = [
    -29,
    -26,
    -23,
    -21,
    -19,
    -17,
    -15,
    -13,
    -12,
    -11,
    -10,
    -9,
    -9,
    -9,
    -9,
    -9,
]
_SPECTRUM2_THIRD = [
    -20,
    -20,
    -18,
    -16,
    -15,
    -14,
    -13,
    -12,
    -11,
    -9,
    -8,
    -9,
    -10,
    -11,
    -13,
    -15,
]
_SPECTRUM1_OCTAVE = [-21, -14, -8, -5, -4]
_SPECTRUM2_OCTAVE = [-14, -10, -7, -4, -6]


def _round_half_up_tenths(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) * 10.0 + 0.5) / 10.0


def _brute_force_airborne_rating(
    measured: list[float],
    reference: list[float],
    limit: float,
    index_500: int,
) -> tuple[int, float]:
    """Independent brute-force airborne shift search (ISO 717-1 Clause 4.4).

    Unfavourable deviation = measured below reference. Find the largest
    integer shift k with the deviation sum <= limit (the sum grows with k);
    the rating is the shifted reference read at 500 Hz.
    """
    meas = _round_half_up_tenths(np.asarray(measured, dtype=np.float64))
    ref = np.asarray(reference, dtype=np.float64)
    best_k = None
    for k in range(200, -201, -1):
        dev = float(np.sum(np.maximum(0.0, ref + k - meas)))
        if dev <= limit + 1e-6:
            best_k = k
            break
    assert best_k is not None
    dev = float(np.sum(np.maximum(0.0, ref + best_k - meas)))
    rating = int(reference[index_500]) + best_k
    return rating, dev


def _brute_force_adaptation(
    measured: list[float], spectrum: list[int], rating: int
) -> int:
    """Independent adaptation term Xaj - rating (ISO 717-1 Clause 4.5)."""
    meas = _round_half_up_tenths(np.asarray(measured, dtype=np.float64))
    spec = np.asarray(spectrum, dtype=np.float64)
    x_aj = -10.0 * np.log10(np.sum(10.0 ** ((spec - meas) / 10.0)))
    return math.floor(x_aj + 0.5) - rating


# ISO 717-1 Annex C, Table C.1 measured sound reduction index R (100-3150)
# is imported from reference_data (shared with the CI conformance report).


# --------------------------------------------------------------------------
# ISO 717-1 weighted rating and spectrum adaptation terms
# --------------------------------------------------------------------------


def test_annex_c_worked_example_third_octave() -> None:
    """ISO 717-1 Annex C Table C.1: Rw(C;Ctr) = 30(-2;-3) dB."""
    res = building.weighted_rating(_ANNEX_C_R)
    assert isinstance(res, building.WeightedRatingResult)
    assert res.rating == 30
    assert res.c == -2
    assert res.ctr == -3
    # Sum of unfavourable deviations at the final shift: 31,8 dB (< 32,0).
    assert res.unfavourable_sum == pytest.approx(31.8, abs=1e-9)


def test_reference_curve_rates_itself() -> None:
    """Measured == reference => shift up by 2 dB (16 * 2 = 32,0), Rw = 54."""
    res = building.weighted_rating(_REF_THIRD)
    assert res.rating == 54
    assert res.unfavourable_sum == pytest.approx(32.0, abs=1e-9)


def test_unfavourable_sum_exactly_at_bound_third_octave() -> None:
    """Measured = reference - 2 everywhere => sum = 32,0 exactly, Rw = 52."""
    measured = [r - 2.0 for r in _REF_THIRD]
    res = building.weighted_rating(measured)
    assert res.rating == 52
    assert res.unfavourable_sum == pytest.approx(32.0, abs=1e-9)


def test_unfavourable_sum_tips_over_forces_one_more_db() -> None:
    """0,1 dB over the 32,0 bound forces one more decibel of shift."""
    measured = [r - 2.0 for r in _REF_THIRD]
    measured[0] -= 0.1  # sum would be 32,1 at the previous shift
    res = building.weighted_rating(measured)
    assert res.rating == 51
    # At Rw = 51 the 15 untouched bands contribute 1,0 dB each and the
    # tipped band 1,1 dB => 16,1 dB (<= 32,0).
    assert res.unfavourable_sum == pytest.approx(16.1, abs=1e-9)


def test_octave_band_rating_bound() -> None:
    """5 octave bands: bound is 10,0 dB; measured = ref - 2 => Rw = 52."""
    measured = [r - 2.0 for r in _REF_OCTAVE]
    res = building.weighted_rating(measured)
    assert res.rating == 52
    assert res.unfavourable_sum == pytest.approx(10.0, abs=1e-9)


def test_octave_reference_rates_itself() -> None:
    """Octave reference == measured => 5 * 2 = 10,0 => Rw = 54."""
    res = building.weighted_rating(_REF_OCTAVE)
    assert res.rating == 54
    assert res.unfavourable_sum == pytest.approx(10.0, abs=1e-9)


def test_explicit_band_set_override() -> None:
    """Band count 16/5 is inferred but can be stated explicitly."""
    res = building.weighted_rating(_ANNEX_C_R, bands="third-octave")
    assert res.rating == 30


def test_measured_data_rounded_to_one_decimal() -> None:
    """Clause 4.4 footnote 1: inputs reduced to 0,1 dB (round half up)."""
    # 30,04 -> 30,0 and 30,05 -> 30,1 must not change already-tenths data.
    res = building.weighted_rating(_ANNEX_C_R)
    perturbed = [v + 0.049 for v in _ANNEX_C_R]  # rounds back to originals
    assert building.weighted_rating(perturbed).rating == res.rating


def test_weighted_rating_rejects_bad_length() -> None:
    with pytest.raises(ValueError, match="Expected 16 one-third-octave"):
        building.weighted_rating([1.0, 2.0, 3.0])


def test_weighted_rating_rejects_nan() -> None:
    bad = list(_ANNEX_C_R)
    bad[0] = float("nan")
    with pytest.raises(
        ValueError, match="'values_by_band' must contain only finite values"
    ):
        building.weighted_rating(bad)


def test_engine_matches_brute_force_third_octave() -> None:
    """10 000 random third-octave curves: shared engine == brute force."""
    rng = np.random.default_rng(20264)
    for _ in range(10_000):
        curve = rng.uniform(10.0, 80.0, size=16)
        res = building.weighted_rating(curve)
        rating, dev = _brute_force_airborne_rating(
            list(curve), _REF_THIRD, 32.0, _INDEX_500_THIRD
        )
        c = _brute_force_adaptation(list(curve), _SPECTRUM1_THIRD, rating)
        ctr = _brute_force_adaptation(list(curve), _SPECTRUM2_THIRD, rating)
        assert res.rating == rating
        assert res.c == c
        assert res.ctr == ctr
        assert res.unfavourable_sum == pytest.approx(dev, abs=1e-9)


def test_engine_matches_brute_force_octave() -> None:
    """10 000 random octave curves: shared engine == brute force."""
    rng = np.random.default_rng(20265)
    for _ in range(10_000):
        curve = rng.uniform(10.0, 80.0, size=5)
        res = building.weighted_rating(curve)
        rating, dev = _brute_force_airborne_rating(
            list(curve), _REF_OCTAVE, 10.0, _INDEX_500_OCTAVE
        )
        c = _brute_force_adaptation(list(curve), _SPECTRUM1_OCTAVE, rating)
        ctr = _brute_force_adaptation(list(curve), _SPECTRUM2_OCTAVE, rating)
        assert res.rating == rating
        assert res.c == c
        assert res.ctr == ctr
        assert res.unfavourable_sum == pytest.approx(dev, abs=1e-9)


# --------------------------------------------------------------------------
# ISO 16283-1 field quantities
# --------------------------------------------------------------------------


def test_energy_average_level_formula9() -> None:
    """Formula (9): equal levels average to themselves; 60 & 70 -> 67,4."""
    assert building.energy_average_level([60.0, 60.0, 60.0]) == pytest.approx(60.0)
    expected = 10.0 * np.log10((10**6 + 10**7) / 2.0)
    assert building.energy_average_level([60.0, 70.0]) == pytest.approx(expected)


def test_dnt_equals_d_when_t_is_half_second() -> None:
    """Formula (2): T = T0 = 0,5 s => DnT = D exactly."""
    l1 = np.array([80.0, 82.0, 85.0])
    l2 = np.array([40.0, 45.0, 50.0])
    t2 = np.full(3, 0.5)
    res = building.airborne_insulation(l1, l2, t2)
    assert isinstance(res, building.AirborneInsulationResult)
    np.testing.assert_allclose(res.d, l1 - l2)
    np.testing.assert_allclose(res.dnt, l1 - l2)


def test_dnt_scales_with_reverberation_time() -> None:
    """T = 5 s, T0 = 0,5 s => 10 lg(10) = 10 dB added to D."""
    l1 = np.array([80.0])
    l2 = np.array([40.0])
    res = building.airborne_insulation(l1, l2, np.array([5.0]))
    np.testing.assert_allclose(res.dnt, [50.0])


def test_apparent_reduction_index_formula4() -> None:
    """Formula (4)+(5): S*T = 0,16*V => 10 lg(S/A) = 0 => R' = D."""
    l1 = np.array([80.0, 82.0])
    l2 = np.array([40.0, 45.0])
    t2 = np.array([1.0, 1.0])
    # A = 0,16 * V / T = 0,16; S = 0,16 => S/A = 1.
    res = building.airborne_insulation(l1, l2, t2, area=0.16, volume=1.0)
    assert res.r_prime is not None
    np.testing.assert_allclose(res.r_prime, l1 - l2)


def test_apparent_reduction_index_ten_db_offset() -> None:
    """S = 1,6, V = 1, T = 1 => A = 0,16, S/A = 10 => R' = D + 10."""
    l1 = np.array([80.0])
    l2 = np.array([40.0])
    res = building.airborne_insulation(l1, l2, np.array([1.0]), area=1.6, volume=1.0)
    assert res.r_prime is not None
    np.testing.assert_allclose(res.r_prime, [50.0])


def test_r_prime_none_without_geometry() -> None:
    res = building.airborne_insulation(
        np.array([80.0]), np.array([40.0]), np.array([0.5])
    )
    assert res.r_prime is None


def test_airborne_energy_averages_positions() -> None:
    """2-D inputs (positions x bands) are energy-averaged (Formula (9))."""
    l1 = np.array([[80.0, 80.0], [80.0, 80.0]])  # two positions, two bands
    l2 = np.array([[40.0, 50.0], [50.0, 40.0]])
    res = building.airborne_insulation(l1, l2, np.array([0.5, 0.5]))
    l2_avg = 10.0 * np.log10((10**4 + 10**5) / 2.0)
    np.testing.assert_allclose(res.d, 80.0 - l2_avg)


def test_airborne_rejects_length_mismatch() -> None:
    two_bands = np.array([80.0, 80.0])
    one_band = np.array([40.0])
    one_time = np.array([0.5])
    with pytest.raises(
        ValueError,
        match=r"airborne_insulation: 'l1'.*'l2'.*'t2'.*one value per band",
    ):
        building.airborne_insulation(two_bands, one_band, one_time)


def test_airborne_rejects_reverberation_time_with_an_extra_axis() -> None:
    """A `t2` carrying an extra axis is named for what is wrong with it.

    The band counts match here, so a message about counts would be false.
    """
    two_bands = np.array([80.0, 80.0])
    other_two_bands = np.array([40.0, 40.0])
    with pytest.raises(ValueError, match="'t2' must be one-dimensional"):
        building.airborne_insulation(two_bands, other_two_bands, np.full((1, 2), 0.5))


def test_a_chain_column_of_another_length_is_refused() -> None:
    """A source-room column one band long prints the chain out of step.

    The verbose ISO 16283-1 fiche pairs ``L1``, ``L2``, ``T`` and ``DnT`` by
    position and reads only as far as the frequency header, so a longer
    ``L1`` sets every source level beside the receiving level of the next
    band, drops the 3150 Hz value and leaves ``DnT`` untouched: the sheet
    documents a derivation that never took place. The plot does not read the
    chain at all, so nothing downstream complains.
    """
    result = building.airborne_insulation(
        np.linspace(90.0, 96.0, 16),
        np.linspace(50.0, 40.0, 16),
        np.full(16, 0.6),
        area=10.0,
        volume=50.0,
    )
    longer = np.insert(result.l1, 0, 99.9)
    with pytest.raises(ValueError, match="'l1' \\(17\\).*per band"):
        dataclasses.replace(result, l1=longer)


@pytest.mark.parametrize("field", ["l1", "l2", "t2"])
def test_a_non_finite_chain_column_is_refused(field: str) -> None:
    """The verbose ISO 16283-1 table forwards the chain raw, so it is pinned.

    ``airborne_insulation`` refuses non-finite levels and reverberation times
    on the way in, so a NaN can only reach a hand-built result; without the
    pin it renders a literal ``nan`` cell inside the accredited field-report
    chain table while the DnT,w rating box beside it prints normally.
    """
    result = building.airborne_insulation(
        np.linspace(90.0, 96.0, 16),
        np.linspace(50.0, 40.0, 16),
        np.full(16, 0.6),
        area=10.0,
        volume=50.0,
    )
    column = np.asarray(getattr(result, field), dtype=np.float64).copy()
    column[4] = np.nan
    with pytest.raises(
        ValueError,
        match=f"AirborneInsulationResult: '{field}' must contain only finite",
    ):
        dataclasses.replace(result, **{field: column})


def test_airborne_requires_both_area_and_volume() -> None:
    l1 = np.array([80.0])
    l2 = np.array([40.0])
    t2 = np.array([0.5])
    with pytest.raises(ValueError, match="'area' and 'volume' must be given together"):
        building.airborne_insulation(l1, l2, t2, area=10.0)


@pytest.mark.parametrize("name", ["area", "volume"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_airborne_rejects_a_geometry_that_is_not_positive(
    name: str, bad: float
) -> None:
    """Zero, negative and NaN all fail the same positivity gate.

    NaN answers False to ``> 0`` just as zero does, so the ``not x > 0``
    spelling catches all three without a separate finiteness question.
    """
    geometry = {"area": 10.0, "volume": 50.0, name: bad}
    with pytest.raises(ValueError, match="'area' and 'volume' must be positive"):
        building.airborne_insulation(
            np.array([80.0]), np.array([40.0]), np.array([0.5]), **geometry
        )


@pytest.mark.parametrize("name", ["area", "volume"])
def test_airborne_rejects_an_infinite_geometry(name: str) -> None:
    """Infinity passes the positivity gate and is refused by name."""
    geometry = {"area": 10.0, "volume": 50.0, name: float("inf")}
    with pytest.raises(ValueError, match="'area' and 'volume' must be finite"):
        building.airborne_insulation(
            np.array([80.0]), np.array([40.0]), np.array([0.5]), **geometry
        )


def test_field_rating_pipeline_dnt_w() -> None:
    """DnT per band (T = 0,5 s) fed to weighted_rating gives DnT,w."""
    l2 = np.array([float(80 - r) for r in _REF_THIRD])  # D = ref
    l1 = np.full(16, 80.0)
    t2 = np.full(16, 0.5)
    res = building.airborne_insulation(l1, l2, t2)
    rating = building.weighted_rating(res.dnt)
    # D == reference curve => rating 54 (2 dB up, sum 32,0).
    assert rating.rating == 54


# --------------------------------------------------------------------------
# Enlarged frequency ranges (ISO 717-1 Annex B) and one-decimal ratings
# --------------------------------------------------------------------------


def test_extended_annex_c2_enlarged_range() -> None:
    """ISO 717-1:2020 Annex C Table C.2: Rw(C;Ctr;C50-5000;Ctr,50-5000)
    = 30 (-2; -3; -2; -4) dB.
    """
    import reference_data as ref

    freqs = [
        50,
        63,
        80,
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
    ]
    res = building.weighted_rating_extended(ref.ISO717_1_ANNEX_C2_R_50_5000, freqs)
    exp = ref.ISO717_1_ANNEX_C2_EXPECTED
    assert res.rating == exp["rw"]
    assert res.c == exp["c"]
    assert res.ctr == exp["ctr"]
    assert res.c_50_5000 == exp["c_50_5000"]
    assert res.ctr_50_5000 == exp["ctr_50_5000"]
    # The 50-3150 and 100-5000 ranges are also covered by a 21-band input.
    assert res.c_50_3150 is not None
    assert res.ctr_100_5000 is not None
    # The core result matches the plain 16-band rating.
    assert res.core.rating == exp["rw"]
    assert res.core.c == exp["c"]
    assert res.core.ctr == exp["ctr"]


def test_extended_core_only_input() -> None:
    """A bare 16-band input yields the core terms; extended ones are None."""
    import reference_data as ref

    res = building.weighted_rating_extended(ref.ISO717_1_ANNEX_C_R)
    assert res.rating == ref.ISO717_1_ANNEX_C_EXPECTED["rw"]
    assert res.c == ref.ISO717_1_ANNEX_C_EXPECTED["c"]
    assert res.ctr == ref.ISO717_1_ANNEX_C_EXPECTED["ctr"]
    assert res.c_50_3150 is None
    assert res.c_50_5000 is None
    assert res.c_100_5000 is None
    assert res.ctr_50_5000 is None


def test_extended_18_band_100_5000_range() -> None:
    """An 18-band 100-5000 Hz input yields C100-5000 but not the 50 Hz terms."""
    import reference_data as ref

    freqs = [
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
    ]
    values = [*ref.ISO717_1_ANNEX_C_R, 26.8, 29.2]
    res = building.weighted_rating_extended(values, freqs)
    assert res.rating == 30
    assert res.c_100_5000 is not None
    assert res.ctr_100_5000 is not None
    assert res.c_50_5000 is None
    assert res.c_50_3150 is None


def test_extended_requires_core_bands() -> None:

    with pytest.raises(
        ValueError, match=r"input must contain the \d+ core one-third-octave bands"
    ):
        building.weighted_rating_extended(
            [40.0] * 10, [50, 63, 80, 100, 125, 160, 200, 250, 315, 400]
        )
    with pytest.raises(
        ValueError,
        match=r"Without 'frequencies' the input must be the \d+ core one-third-octave bands",
    ):
        building.weighted_rating_extended([40.0] * 18)


def test_extended_mismatched_frequencies_name_the_entry_point() -> None:
    """Both entry points share one validator; each names itself, not it."""
    freqs = [100.0 * 2.0 ** (k / 3.0) for k in range(16)]
    with pytest.raises(
        ValueError,
        match=r"weighted_rating_extended: 'values_by_band'.*'frequencies'.*same shape",
    ):
        building.weighted_rating_extended([40.0] * 18, freqs)
    with pytest.raises(
        ValueError,
        match=r"weighted_impact_rating_extended: 'values_by_band'.*'frequencies'",
    ):
        building.weighted_impact_rating_extended([40.0] * 18, freqs)


def test_one_decimal_rating_annex_b() -> None:
    """ISO 12999-1:2020 Annex B: the 0,1 dB shift yields Rw = 57,4 dB and the
    one-decimal sums Rw + C50-5000 = 56,4 / Rw + Ctr,50-5000 = 51,1 dB.
    """
    import reference_data as ref

    res = building.weighted_rating_extended(
        ref.ISO12999_1_ANNEX_B_RI,
        ref.ISO12999_1_ANNEX_B_FREQ,
        one_decimal=True,
    )
    assert res.rating == pytest.approx(ref.ISO12999_1_ANNEX_B_RW)
    assert res.c_50_5000 is not None
    assert res.ctr_50_5000 is not None
    assert res.rating + res.c_50_5000 == pytest.approx(
        ref.ISO12999_1_ANNEX_B_RW_C50_5000
    )
    assert res.rating + res.ctr_50_5000 == pytest.approx(
        ref.ISO12999_1_ANNEX_B_RW_CTR50_5000
    )
    # The integer-mode rating of the same spectrum stays an integer.
    integer = building.weighted_rating_extended(
        ref.ISO12999_1_ANNEX_B_RI, ref.ISO12999_1_ANNEX_B_FREQ
    )
    assert integer.rating == 57


def test_impact_extended_ci_50_2500() -> None:
    """CI,50-2500 sums 50-2500 Hz (A.2.1 NOTE); flat extensions with low
    energy leave it equal to the core CI.
    """
    import reference_data as ref

    freqs = [
        50,
        63,
        80,
        *[int(f) for f in np.asarray(ref.ISO717_2_REFERENCE_FLOOR_FREQ, dtype=float)],
    ]
    ln = [30.0, 30.0, 30.0, *ref.ISO717_2_REFERENCE_FLOOR_LN_R0]
    res = building.weighted_impact_rating_extended(ln, freqs)
    assert res.rating == 78
    assert res.ci == -11
    # 30 dB extension bands are ~40 dB below the sum: CI unchanged.
    assert res.ci_50_2500 == -11
    # Strong low-frequency content raises the enlarged-range term.
    ln_low = [75.0, 75.0, 75.0, *ref.ISO717_2_REFERENCE_FLOOR_LN_R0]
    boosted = building.weighted_impact_rating_extended(ln_low, freqs)
    assert boosted.ci_50_2500 is not None
    assert boosted.ci_50_2500 > -11


def test_impact_one_decimal_reference_floor() -> None:
    """The 0,1 dB variant reproduces the printed uncertainty constants of
    ISO 717-2:2020 A.2.2: Ln,r,0,w = 77,6 dB and CI,r,0 = -10,3 dB.
    """
    import reference_data as ref

    res = building.weighted_impact_rating_extended(
        ref.ISO717_2_REFERENCE_FLOOR_LN_R0, one_decimal=True
    )
    assert res.rating == pytest.approx(77.6)
    assert res.ci == pytest.approx(-10.3)
    assert res.core.rating == 78
    assert res.core.ci == -11


# ---------------------------------------------------------------------------
# ISO 717-1 rating of a published Spanish field test report (CTE DB-HR chain)
# ---------------------------------------------------------------------------

# Aviles Lopez & Perera Martin, "Manual de acustica ambiental y
# arquitectonica" (Paraninfo, ISBN 978-84-283-3814-1), Ejemplo 7.2
# (pp. 394-395): apparent sound reduction index R' of a separating wall from a
# real field test report, one-third-octave bands 100 Hz - 3150 Hz (the report
# extends to 5 kHz; the 16 ISO 717-1 rating bands are used here).
_MANUAL_ES_R_PRIME = (
    36.2,
    41.5,
    36.9,
    40.4,
    44.7,
    42.4,
    45.7,
    46.1,
    47.1,
    52.3,
    54.3,
    57.5,
    57.8,
    57.3,
    59.0,
    62.8,
)

# Ejercicio 7.1 (p. 395): standardized facade level difference D2m,nT of the
# same building's facade, same band layout.
_MANUAL_ES_D_2M_NT = (
    28.5,
    28.5,
    18.9,
    23.7,
    30.7,
    31.3,
    37.8,
    35.2,
    34.7,
    38.5,
    37.7,
    43.1,
    42.3,
    44.2,
    41.9,
    37.5,
)


def test_weighted_rating_manual_es_field_report() -> None:
    # Ejemplo 7.1 (pp. 391-392) publishes the report's ISO 717-1 statement for
    # this curve: R'w = 52 dB with C = -1 and Ctr = -5, hence the CTE DB-HR
    # global indices R'A = R'w + C = 51 dBA (pink noise) and
    # R'A,tr = R'w + Ctr = 47 dBA (traffic). All integers by definition.
    res = building.weighted_rating(_MANUAL_ES_R_PRIME)
    assert res.rating == 52
    assert res.c == -1
    assert res.ctr == -5
    # The published global indices 51 dBA and 47 dBA follow as 52 - 1 and
    # 52 - 5 from the three values asserted above.


def test_weighted_rating_manual_es_facade_traffic_index() -> None:
    # Ejercicio 7.1 evaluates the CTE DB-HR facade index directly with the
    # one-decimal formula D2m,nT,Atr = -10 lg sum 10^((LAtr,i - Di)/10) and
    # publishes 32.8 dBA. The ISO 717-1 route implemented here yields the
    # integer pair D2m,nT,w + Ctr; the two definitions agree to within the
    # C-term's integer rounding, so 1.0 dB is the definitional tolerance
    # (0.5 dB rounding of Ctr plus 0.5 dB of the printed one-decimal value).
    res = building.weighted_rating(_MANUAL_ES_D_2M_NT)
    assert res.rating + res.ctr == pytest.approx(32.8, abs=1.0)


@pytest.mark.parametrize("field", ["band_centers", "measured", "shifted_reference"])
def test_rating_band_curves_must_be_finite(field: str) -> None:
    """The ISO 717 fiche reads the stored curves raw, so they are pinned.

    No constructor can emit a non-finite band: ``weighted_rating`` refuses
    non-finite input by name, the centres come from the Table 1 band sets and
    the shifted reference is the Table 3 curve plus an integer shift. Reading
    one anyway, a NaN centre died in a bare ``ValueError: cannot convert float
    NaN to integer`` from ``round`` inside the band table, and a NaN measured
    value printed a literal ``nan`` cell beside an em dash in the Annex C
    deviation column whose defined meaning is "deviation below 0,05 dB".
    """
    result = building.weighted_rating(_ANNEX_C_R)
    curve = np.asarray(getattr(result, field), dtype=np.float64).copy()
    curve[5] = np.nan  # 315 Hz band
    with pytest.raises(
        ValueError, match=f"WeightedRatingResult: '{field}' must contain only finite"
    ):
        dataclasses.replace(result, **{field: curve})


def test_impact_rating_band_curves_must_be_finite() -> None:
    """The ISO 717-2 rating carries the same curves and the same refusal."""
    result = building.weighted_impact_rating([60.0 - k for k in range(16)])
    measured = np.asarray(result.measured, dtype=np.float64).copy()
    measured[5] = np.inf
    with pytest.raises(
        ValueError, match="ImpactRatingResult: 'measured' must contain only finite"
    ):
        dataclasses.replace(result, measured=measured)


def test_rating_unfavourable_sum_must_restate_its_own_curves() -> None:
    """The sum is the two curves added up, so it is pinned to them.

    ISO 717-1:2020 Table C.1 rates its example at Rw = 30 dB with the
    unfavourable deviations summing to 31,8 dB. Swapping ``measured`` for a
    curve lying on the shifted reference leaves no unfavourable deviation at
    all, and the stale 31,8 dB used to survive the swap: the verbose Annex C
    fiche then printed an em dash in every deviation row and closed the
    table with "sum 0,0 dB", beside a plot titled
    ``... = 30 dB (Sigma unfav. = 31.8 dB)``.
    """
    result = building.weighted_rating(_ANNEX_C_R)
    on_the_reference = np.asarray(result.shifted_reference, dtype=np.float64)
    with pytest.raises(
        ValueError,
        match=(
            r"WeightedRatingResult: 'unfavourable_sum' must be the sum of the "
            r"unfavourable deviations of its own curves"
        ),
    ):
        dataclasses.replace(result, measured=on_the_reference)


def test_impact_unfavourable_sum_must_restate_its_own_curves() -> None:
    """The ISO 717-2 sum is pinned the same way, with the opposite sign.

    ISO 717-2:2020 Table C.1 rates the bare heavy floor at Ln,w = 79 dB with
    the deviations -- here where the measurement *exceeds* the reference --
    summing to 28,0 dB. A ``measured`` curve laid on the shifted reference
    leaves none of them, and the refusal must name the impact sense rather
    than the airborne one.
    """
    result = building.weighted_impact_rating(_ANNEX_C1_LN)
    on_the_reference = np.asarray(result.shifted_reference, dtype=np.float64)
    with pytest.raises(
        ValueError,
        match=(
            r"ImpactRatingResult: 'unfavourable_sum' must be the sum of the "
            r"unfavourable deviations of its own curves, 'measured' above "
            r"'shifted_reference'"
        ),
    ):
        dataclasses.replace(result, measured=on_the_reference)


def test_unfavourable_sum_accepts_a_variant_that_keeps_its_curves() -> None:
    """A coherent variant is not refused over floating-point slack.

    Raising both curves by the same 3 dB leaves every deviation, and so the
    sum, exactly as it was; recomputing the sum along a different path (band
    by band in plain Python rather than one numpy reduction) must still land
    inside the guard's tolerance.
    """
    result = building.weighted_rating(_ANNEX_C_R)
    measured = np.asarray(result.measured, dtype=np.float64) + 3.0
    shifted = np.asarray(result.shifted_reference, dtype=np.float64) + 3.0
    recomputed = math.fsum(
        max(0.0, r - m) for m, r in zip(measured, shifted, strict=True)
    )
    moved = dataclasses.replace(
        result,
        measured=measured,
        shifted_reference=shifted,
        unfavourable_sum=recomputed,
    )
    assert moved.unfavourable_sum == pytest.approx(31.8, abs=1e-9)
