#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What ISO 4869-2 leaves at the ear, against its own four worked annexes.

Annexes A to D of ISO 4869-2:2018 are one example built on one 16-subject
attenuation grid, carried through all three methods. Every printed number of
that example is pinned here: the per-band mean and spread, the octave-band
result, all sixteen HML triples with their statistics, all sixteen SNR values,
and the effective levels each method reports for the same noise.
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

from phonometry import hearing

# ---------------------------------------------------------------------------
# Clause 5: the assumed protection value (Annex A)
# ---------------------------------------------------------------------------


def test_annex_a_mean_and_spread() -> None:
    """m_f and s_f reproduce Table A.1 band for band."""
    result = hearing.assumed_protection_value(ref.ISO4869_2_ATTENUATION)
    assert np.round(result.mean_attenuation, 1).tolist() == ref.ISO4869_2_MEAN
    assert (
        np.round(result.standard_deviation, 1).tolist()
        == ref.ISO4869_2_STANDARD_DEVIATION
    )
    assert result.subjects == 16
    assert result.alpha == 1.0
    assert result.performance == 84


def test_annex_a_printed_apv_subtracts_the_rounded_intermediates() -> None:
    """Formula (1) on the data and the annex's own row differ by 0,1 dB.

    Table A.1 displays m_f and s_f to one decimal and prints their difference,
    which is not Formula (1) applied to the underlying attenuations. Three of
    the eight bands land a tenth apart, so the library returns the formula and
    the annex is reproduced by rounding first.
    """
    result = hearing.assumed_protection_value(ref.ISO4869_2_ATTENUATION)
    as_displayed = np.round(
        np.round(result.mean_attenuation, 1)
        - result.alpha * np.round(result.standard_deviation, 1),
        1,
    )
    assert as_displayed.tolist() == ref.ISO4869_2_APV84_PRINTED
    from_the_data = np.round(result.apv, 1)
    differing = np.flatnonzero(from_the_data != as_displayed)
    assert differing.tolist() == [2, 3, 6]
    assert np.abs(from_the_data - as_displayed).max() == pytest.approx(0.1)


def test_alpha_comes_from_table_1() -> None:
    """Every tabulated performance, and nothing else."""
    assert hearing.PROTECTION_PERFORMANCES == {
        50: 0.00,
        75: 0.67,
        80: 0.84,
        84: 1.00,
        90: 1.28,
        95: 1.64,
        98: 2.00,
    }
    strict = hearing.assumed_protection_value(ref.ISO4869_2_ATTENUATION, performance=98)
    lenient = hearing.assumed_protection_value(
        ref.ISO4869_2_ATTENUATION, performance=50
    )
    # A larger performance subtracts more of the spread, never less.
    assert np.all(strict.apv < lenient.apv)
    # x = 50 % is the mean itself, alpha being zero.
    assert lenient.apv == pytest.approx(lenient.mean_attenuation)


@pytest.mark.parametrize("bad", [85, 0, -1, "84", None, 84.5])
def test_a_performance_outside_table_1_is_refused(bad: object) -> None:
    """The seven rows of Table 1 are the whole of the choice."""
    with pytest.raises(ValueError, match="protection performances Table 1"):
        hearing.assumed_protection_value(ref.ISO4869_2_ATTENUATION, performance=bad)


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        ([1.0, 2.0, 3.0], "subjects, bands"),
        ([[1.0, 2.0]], "at least two subjects"),
        ([[1.0, np.nan], [2.0, 3.0]], "finite"),
    ],
)
def test_the_attenuation_grid_is_checked(grid: object, match: str) -> None:
    """A distribution needs two dimensions and more than one subject."""
    with pytest.raises(ValueError, match=match):
        hearing.assumed_protection_value(grid, frequencies=[125.0, 250.0])


def test_the_a_weighting_matches_the_librarys_own_iec_table() -> None:
    """The eight values are IEC 61672-1 Table 3, not a second copy of it."""
    from phonometry.filters.weighting_compliance import _WEIGHTING_TABLE3

    printed = {row[0]: row[1] for row in _WEIGHTING_TABLE3}
    assert [printed[f] for f in hearing.PROTECTOR_OCTAVE_BANDS] == list(
        hearing.PROTECTOR_A_WEIGHTING
    )
    # And they are the row Annex B prints beside its own spectrum.
    assert list(hearing.PROTECTOR_A_WEIGHTING) == ref.ISO4869_2_ANNEX_B_A_WEIGHTING


# ---------------------------------------------------------------------------
# Clause 6: the octave-band method (Annex B)
# ---------------------------------------------------------------------------


def test_annex_b_octave_band_method() -> None:
    """Every row of Table B.1 and the level it sums to."""
    result = hearing.octave_band_protected_level(
        ref.ISO4869_2_ANNEX_B_NOISE, ref.ISO4869_2_APV84_PRINTED
    )
    assert result.band_levels is not None
    assert np.round(result.band_levels, 1).tolist() == ref.ISO4869_2_ANNEX_B_NET
    assert result.effective_level == pytest.approx(
        ref.ISO4869_2_ANNEX_B_EFFECTIVE, abs=0.05
    )
    assert result.reported_level == ref.ISO4869_2_ANNEX_B_REPORTED
    assert result.method == "octave-band"


def test_the_octave_band_method_recovers_the_unprotected_level() -> None:
    """PNR is the difference the same spectrum makes without the protector."""
    result = hearing.octave_band_protected_level(
        ref.ISO4869_2_ANNEX_B_NOISE, ref.ISO4869_2_APV84_PRINTED
    )
    unprotected = result.effective_level + result.noise_reduction
    assert unprotected == pytest.approx(ref.ISO4869_2_ANNEX_B_LPA, abs=0.05)


def test_the_octave_band_method_takes_a_result_or_an_array() -> None:
    """Passing the result carries the performance through to the report."""
    apv = hearing.assumed_protection_value(ref.ISO4869_2_ATTENUATION)
    from_result = hearing.octave_band_protected_level(ref.ISO4869_2_ANNEX_B_NOISE, apv)
    from_array = hearing.octave_band_protected_level(
        ref.ISO4869_2_ANNEX_B_NOISE, apv.apv
    )
    assert from_result.effective_level == pytest.approx(from_array.effective_level)
    assert from_result.performance == 84
    assert from_array.performance is None


def test_the_octave_band_method_drops_63_hz_when_asked_to() -> None:
    """Clause 6 starts at 125 Hz when either input lacks the 63 Hz band."""
    seven = hearing.octave_band_protected_level(
        ref.ISO4869_2_ANNEX_B_NOISE[1:], ref.ISO4869_2_APV84_PRINTED[1:]
    )
    assert seven.frequencies is not None
    assert seven.frequencies.tolist() == list(hearing.PROTECTOR_OCTAVE_BANDS[1:])
    # 63 Hz contributes 44,7 dB against a total of 81,4 dB, so dropping it
    # moves the answer by less than the reported resolution.
    assert seven.reported_level == ref.ISO4869_2_ANNEX_B_REPORTED


@pytest.mark.parametrize(
    ("noise", "apv", "match"),
    [
        ([80.0, 81.0], [1.0], "same octave bands"),
        ([80.0, np.inf], [1.0, 2.0], "finite"),
        ([[80.0, 81.0]], [1.0, 2.0], "one-dimensional"),
    ],
)
def test_the_octave_band_inputs_are_checked(
    noise: object, apv: object, match: str
) -> None:
    """Formula (2) subtracts two arrays over one band set."""
    with pytest.raises(ValueError, match=match):
        hearing.octave_band_protected_level(
            noise, apv, frequencies=[125.0, 250.0], a_weighting=[-16.1, -8.6]
        )


# ---------------------------------------------------------------------------
# Clause 7: the HML method (Annex C)
# ---------------------------------------------------------------------------


def test_annex_c_per_subject_hml() -> None:
    """All sixteen H, M and L triples of Table C.2."""
    rating = hearing.hml_rating(ref.ISO4869_2_ATTENUATION)
    assert np.round(rating.subject_h, 1).tolist() == ref.ISO4869_2_ANNEX_C_H
    assert np.round(rating.subject_m, 1).tolist() == ref.ISO4869_2_ANNEX_C_M
    assert np.round(rating.subject_l, 1).tolist() == ref.ISO4869_2_ANNEX_C_L


def test_annex_c_statistics_and_rating() -> None:
    """The three means, the three spreads and H84 / M84 / L84."""
    rating = hearing.hml_rating(ref.ISO4869_2_ATTENUATION)
    means = tuple(
        round(float(v.mean()), 1)
        for v in (rating.subject_h, rating.subject_m, rating.subject_l)
    )
    spreads = tuple(
        round(float(v.std(ddof=1)), 1)
        for v in (rating.subject_h, rating.subject_m, rating.subject_l)
    )
    assert means == ref.ISO4869_2_ANNEX_C_MEANS
    assert spreads == ref.ISO4869_2_ANNEX_C_DEVIATIONS
    assert rating.reported == ref.ISO4869_2_ANNEX_C_HML84


def test_the_hml_fit_uses_table_2_and_not_its_reprint() -> None:
    """Table C.1 disagrees with Table 2, and Annex C follows Table 2.

    Table C.1 says it reprints Table 2 but prints 89,4 dB and 93,5 dB at
    250 Hz and 500 Hz of the sixth reference noise where Table 2 prints 89,3
    and 93,3. The sixth row of Table C.2 tells them apart: Table 2 reproduces
    all sixteen values, the reprint misses thirteen of them by 0,1 dB.
    """
    rating = hearing.hml_rating(ref.ISO4869_2_ATTENUATION)
    from_table_2 = np.round(rating.predicted_reduction[:, 5], 1)
    assert from_table_2.tolist() == ref.ISO4869_2_ANNEX_C_PNR_NOISE6

    attenuation = np.asarray(ref.ISO4869_2_ATTENUATION, dtype=float)[:, 1:]
    reprint = np.asarray(ref.ISO4869_2_TABLE_C1_NOISE6, dtype=float)
    from_reprint = 100.0 - 10.0 * np.log10(
        np.sum(10.0 ** (0.1 * (reprint[None, :] - attenuation)), axis=1)
    )
    missed = np.flatnonzero(
        np.round(from_reprint, 1) != np.asarray(ref.ISO4869_2_ANNEX_C_PNR_NOISE6)
    )
    assert missed.size == 13


def test_annex_c_application() -> None:
    """PNR84 = 22,5 dB and the level it leaves, from the Annex B noise."""
    rating = hearing.hml_rating(ref.ISO4869_2_ATTENUATION)
    result = hearing.hml_protected_level(
        ref.ISO4869_2_ANNEX_B_LPA, ref.ISO4869_2_ANNEX_B_LPC, rating
    )
    assert result.noise_reduction == pytest.approx(ref.ISO4869_2_ANNEX_C_PNR84)
    assert result.effective_level == pytest.approx(ref.ISO4869_2_ANNEX_C_EFFECTIVE)
    assert result.reported_level == ref.ISO4869_2_ANNEX_C_REPORTED
    assert result.method == "HML"


def test_the_hml_application_consumes_the_rounded_triple() -> None:
    """Clause 7.2 rounds H, M and L before Formulas (16) and (17) see them."""
    rating = hearing.hml_rating(ref.ISO4869_2_ATTENUATION)
    high, medium, low = rating.reported
    # Both branches meet at the +2 dB anchor, which is M itself.
    at_the_break = hearing.hml_protected_level(100.0, 102.0, rating)
    assert at_the_break.noise_reduction == pytest.approx(medium)
    # And each anchor is recovered at its own defining difference.
    at_high = hearing.hml_protected_level(100.0, 98.0, rating)
    at_low = hearing.hml_protected_level(100.0, 110.0, rating)
    assert at_high.noise_reduction == pytest.approx(high)
    assert at_low.noise_reduction == pytest.approx(low)


def test_the_hml_branches_meet_and_only_meet_at_two_decibels() -> None:
    """Formula (16) below the break, Formula (17) above it, no jump."""
    rating = hearing.hml_rating(ref.ISO4869_2_ATTENUATION)
    high, medium, low = rating.reported
    just_below = hearing.hml_protected_level(100.0, 101.999, rating)
    just_above = hearing.hml_protected_level(100.0, 102.001, rating)
    # The two slopes are (H - M)/4 and (M - L)/8, so a step of 0,002 dB either
    # side of the break separates them by their sum times that step and no
    # more: continuous, with a corner.
    slopes = (high - medium) / 4.0 + (medium - low) / 8.0
    assert just_below.noise_reduction == pytest.approx(
        just_above.noise_reduction, abs=abs(slopes) * 0.002 + 1e-9
    )
    # The two segments have different slopes, so they are not one line.
    far_below = hearing.hml_protected_level(100.0, 96.0, rating)
    far_above = hearing.hml_protected_level(100.0, 108.0, rating)
    below_slope = (far_below.noise_reduction - just_below.noise_reduction) / -6.0
    above_slope = (far_above.noise_reduction - just_above.noise_reduction) / 6.0
    assert below_slope != pytest.approx(above_slope)


# ---------------------------------------------------------------------------
# Clause 8: the SNR method (Annex D)
# ---------------------------------------------------------------------------


def test_annex_d_per_subject_and_rating() -> None:
    """All sixteen SNRj of Table D.2, their statistics and SNR84."""
    rating = hearing.snr_rating(ref.ISO4869_2_ATTENUATION)
    assert np.round(rating.subject_snr, 1).tolist() == ref.ISO4869_2_ANNEX_D_SNR
    assert round(rating.mean, 1) == ref.ISO4869_2_ANNEX_D_MEAN
    assert round(rating.standard_deviation, 1) == ref.ISO4869_2_ANNEX_D_DEVIATION
    assert rating.reported == ref.ISO4869_2_ANNEX_D_SNR84


def test_annex_d_both_applications_land_together() -> None:
    """Formula (24) is Formula (23) with the C-weighted level reassembled."""
    rating = hearing.snr_rating(ref.ISO4869_2_ATTENUATION)
    by_c = hearing.snr_protected_level(rating, l_p_c=ref.ISO4869_2_ANNEX_B_LPC)
    by_a = hearing.snr_protected_level(
        rating,
        l_p_a=ref.ISO4869_2_ANNEX_B_LPA,
        c_minus_a=ref.ISO4869_2_ANNEX_B_LPC - ref.ISO4869_2_ANNEX_B_LPA,
    )
    assert by_c.reported_level == ref.ISO4869_2_ANNEX_D_REPORTED
    assert by_a.reported_level == ref.ISO4869_2_ANNEX_D_REPORTED
    assert by_c.effective_level == pytest.approx(by_a.effective_level)
    assert by_c.method == "SNR"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"l_p_c": 103.0, "l_p_a": 104.0, "c_minus_a": -1.0},
        {"l_p_a": 104.0},
        {"c_minus_a": -1.0},
    ],
)
def test_the_snr_application_needs_exactly_one_pairing(
    kwargs: dict[str, float],
) -> None:
    """Either the C-weighted level, or the A-weighted one and the difference."""
    rating = hearing.snr_rating(ref.ISO4869_2_ATTENUATION)
    with pytest.raises(ValueError, match="Formula"):
        hearing.snr_protected_level(rating, **kwargs)


# ---------------------------------------------------------------------------
# Across the three methods
# ---------------------------------------------------------------------------


def test_the_three_methods_agree_within_the_standards_own_tolerance() -> None:
    """81 dB, 82 dB and 82 dB for one protector in one noise.

    Clause 1's NOTE calls differences of 3 dB or less insignificant, and the
    three methods of the worked example sit inside 1 dB of each other.
    """
    attenuation = ref.ISO4869_2_ATTENUATION
    apv = hearing.assumed_protection_value(attenuation)
    octave = hearing.octave_band_protected_level(ref.ISO4869_2_ANNEX_B_NOISE, apv)
    hml = hearing.hml_protected_level(
        ref.ISO4869_2_ANNEX_B_LPA,
        ref.ISO4869_2_ANNEX_B_LPC,
        hearing.hml_rating(attenuation),
    )
    snr = hearing.snr_protected_level(
        hearing.snr_rating(attenuation), l_p_c=ref.ISO4869_2_ANNEX_B_LPC
    )
    reported = [octave.reported_level, hml.reported_level, snr.reported_level]
    assert reported == [81, 82, 82]
    assert max(reported) - min(reported) <= 3


def test_a_better_protector_leaves_a_lower_level_by_every_method() -> None:
    """Ten decibels more attenuation in every band, three lower answers."""
    attenuation = np.asarray(ref.ISO4869_2_ATTENUATION, dtype=float)
    better = attenuation + 10.0
    for grid in (attenuation, better):
        rating = hearing.hml_rating(grid)
        assert rating.high > 0.0
    base = hearing.snr_rating(attenuation).snr
    improved = hearing.snr_rating(better).snr
    assert improved == pytest.approx(base + 10.0)


@pytest.mark.parametrize("bands", [3, 6, 9])
def test_the_ratings_refuse_a_band_set_the_reference_spectra_cannot_meet(
    bands: int,
) -> None:
    """Formulas (15) and (22) read seven bands from 125 Hz."""
    grid = np.zeros((16, bands))
    for call in (hearing.hml_rating, hearing.snr_rating):
        with pytest.raises(ValueError, match="eight octave bands|seven from 125"):
            call(grid)


def test_the_reported_levels_round_halves_away_from_zero() -> None:
    """ "Rounded to the nearest integer" is not Python's round()."""
    rating = hearing.snr_rating(ref.ISO4869_2_ATTENUATION)
    # SNR84 = 21 dB, so a C-weighted level of 103,5 dB lands exactly on .5.
    half = hearing.snr_protected_level(rating, l_p_c=103.5)
    assert half.effective_level == pytest.approx(82.5)
    assert half.reported_level == 83
    assert round(82.5) == 82
