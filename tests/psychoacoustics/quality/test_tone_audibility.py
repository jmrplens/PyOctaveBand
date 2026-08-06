#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO/PAS 20065:2016 tonal audibility (engineering method).

Anchored on the Annex E combustion-engine worked example: the critical band
(Formulae (2)-(5)), the critical-band level LG (Formula (12)), the masking
index av (Formula (13)), the per-tone audibility ΔL = LT − LG − av
(Formula (14)) and the energy-mean mean audibility (Formula (20)).
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    HANNING_BANDWIDTH_FACTOR,
    NO_TONE_AUDIBILITY,
    ToneAudibilityResult,
    analyze_spectrum,
    assess_tones,
    audibility_from_levels,
    audibility_uncertainty,
    combined_tone_level,
    critical_band_corners,
    critical_band_level,
    critical_bandwidth_engineering,
    energy_sum_level,
    masking_index,
    mean_audibility,
    mean_audibility_uncertainty,
    mean_narrowband_level,
    resolve_tones_separately,
    tone_audibility,
    tone_level,
    two_tone_separation_frequency,
)


# ---------------------------------------------------------------------------
# Critical band about the tone (Formulae (2)-(5))
# ---------------------------------------------------------------------------
def test_critical_bandwidth_closed_form() -> None:
    ft = 137.3
    expected = 25.0 + 75.0 * (1.0 + 1.4 * (ft / 1000.0) ** 2) ** 0.69
    assert critical_bandwidth_engineering(ft) == pytest.approx(expected)


def test_critical_bandwidth_grows_with_frequency() -> None:
    assert critical_bandwidth_engineering(2000.0) > critical_bandwidth_engineering(
        100.0
    )


def test_critical_band_corners_geometry() -> None:
    # f1·f2 = fT² and f2 − f1 = Δfc (geometric placement, Formulae (3)-(5)).
    for ft in (55.0, 137.3, 592.2, 1582.7):
        f1, f2 = critical_band_corners(ft)
        assert math.sqrt(f1 * f2) == pytest.approx(ft, rel=1e-12)
        assert f2 - f1 == pytest.approx(critical_bandwidth_engineering(ft), rel=1e-12)
        assert 0.0 < f1 < ft < f2


# ---------------------------------------------------------------------------
# Masking index (Formula (13)) and critical-band level (Formula (12))
# ---------------------------------------------------------------------------
def test_masking_index_closed_form() -> None:
    f = 592.2
    expected = -2.0 - math.log10(1.0 + (f / 502.0) ** 2.5)
    assert masking_index(f) == pytest.approx(expected)


def test_masking_index_matches_annex_e() -> None:
    assert masking_index(137.3) == pytest.approx(ref.ISO20065_AV_137, abs=0.005)
    assert masking_index(592.2) == pytest.approx(ref.ISO20065_AV_592, abs=0.005)


def test_masking_index_decreases_with_frequency() -> None:
    assert masking_index(2000.0) < masking_index(100.0) < 0.0


def test_critical_band_level_matches_annex_e() -> None:
    value = critical_band_level(49.22, 137.3, ref.ISO20065_LINE_SPACING)
    assert value == pytest.approx(ref.ISO20065_LG_137, abs=0.05)


def test_critical_band_level_bandwidth_ratio() -> None:
    ls, ft, df = 50.0, 300.0, 2.7
    dfc = critical_bandwidth_engineering(ft)
    assert critical_band_level(ls, ft, df) == pytest.approx(
        ls + 10.0 * math.log10(dfc / df)
    )


# ---------------------------------------------------------------------------
# Audibility (Formula (14))
# ---------------------------------------------------------------------------
def test_audibility_from_levels_identity() -> None:
    assert audibility_from_levels(67.96, 64.98, -2.02) == pytest.approx(5.0)


def test_tone_audibility_matches_annex_e() -> None:
    for ft, ls, lt, expected in ref.ISO20065_ANNEX_E_TONES:
        value = tone_audibility(lt, ls, ft, ref.ISO20065_LINE_SPACING)
        assert value == pytest.approx(expected, abs=0.03)


# ---------------------------------------------------------------------------
# Energy-sum level with the window correction (Formulae (6)/(8))
# ---------------------------------------------------------------------------
def test_energy_sum_hanning_correction() -> None:
    # Two equal 80 dB lines: 10 lg(2·10^8) = 83.01 dB, minus 1.76 dB Hanning.
    assert energy_sum_level([80.0, 80.0]) == pytest.approx(
        83.0103 - 10.0 * math.log10(1.5), abs=1e-4
    )


def test_energy_sum_rectangular_is_plain_sum() -> None:
    value = energy_sum_level([80.0, 80.0], effective_bandwidth_factor=1.0)
    assert value == pytest.approx(10.0 * math.log10(2.0e8), abs=1e-6)


def test_hanning_factor_constant() -> None:
    assert HANNING_BANDWIDTH_FACTOR == 1.5


# ---------------------------------------------------------------------------
# From-spectrum front-end: LS (Formula (6)) and LT (Formula (8))
# ---------------------------------------------------------------------------
def test_mean_narrowband_level_matches_annex_e() -> None:
    ls = mean_narrowband_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3
    )
    assert ls == pytest.approx(ref.ISO20065_E1_LS, abs=0.02)


def test_tone_level_matches_annex_e() -> None:
    ls = mean_narrowband_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3
    )
    lt = tone_level(ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3, ls)
    assert lt == pytest.approx(ref.ISO20065_E1_LT, abs=0.02)


def test_from_spectrum_full_chain_matches_annex_e() -> None:
    # Spectrum -> LS -> LT -> ΔL reproduces the 137.3 Hz tone (ΔL ≈ 4.99 dB).
    ls = mean_narrowband_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3
    )
    lt = tone_level(ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3, ls)
    assert tone_audibility(lt, ls, 137.3, 2.7) == pytest.approx(4.99, abs=0.03)


def test_mean_narrowband_level_rectangular_higher() -> None:
    # Dropping the Hanning correction (factor 1.0) raises LS by at least the
    # 1.76 dB correction; the iterative exclusion also then keeps more lines,
    # so the gap is not a constant offset.
    ls_h = mean_narrowband_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3
    )
    ls_r = mean_narrowband_level(
        ref.ISO20065_E1_LEVELS,
        ref.ISO20065_E1_FREQUENCIES,
        137.3,
        effective_bandwidth_factor=1.0,
    )
    assert ls_r > ls_h + 10.0 * math.log10(1.5) - 1e-6


def test_tone_level_single_line_when_isolated() -> None:
    # A lone peak far above its neighbours yields just itself: Formula (7)
    # applies no bandwidth correction to a K = 1 run (DIN 45681 Annex J:
    # "If l = 1 Then LT = 10*Log(LT)/Log(10)", no -1.76).
    freqs = [100.0, 102.7, 105.4, 108.1, 110.8]
    levels = [40.0, 41.0, 70.0, 41.0, 40.0]
    ls = 45.0
    lt = tone_level(levels, freqs, 105.4, ls)
    assert lt == pytest.approx(70.0, abs=1e-9)


def test_energy_sum_single_line_has_no_bandwidth_correction() -> None:
    # Formula (7): LT = L1 for K = 1; the Hanning correction is K > 1 only.
    assert energy_sum_level([53.0]) == pytest.approx(53.0, abs=1e-12)
    assert energy_sum_level([53.0], effective_bandwidth_factor=1.0) == (
        pytest.approx(53.0, abs=1e-12)
    )


def test_single_line_tone_audibility_flip_regression() -> None:
    # H1 regression (Formula (7) vs (8)): a single-line tone 13 dB above a
    # flat 40 dB noise floor (line spacing 2.6917 Hz, tone near 301.5 Hz).
    # Formula (7) gives LT = 53.0 dB and dL = +0.90 dB (audible); wrongly
    # applying the K > 1 Hanning correction (-1.76 dB) flips the verdict to
    # "no audible tone".
    df = 2.6917
    freqs = df * np.arange(20, 201)          # 53.8 Hz .. 538.3 Hz
    levels = np.full(freqs.size, 40.0)
    tone_index = int(np.argmin(np.abs(freqs - 301.5)))
    levels[tone_index] = 53.0
    result = analyze_spectrum(levels, freqs, df)
    assert result.tone_frequencies.size == 1
    assert result.tone_frequencies[0] == pytest.approx(freqs[tone_index])
    assert result.tone_levels[0] == pytest.approx(53.0, abs=1e-9)
    assert result.audibilities[0] == pytest.approx(0.90, abs=0.02)
    assert bool(result.audible[0])


def test_mean_narrowband_level_empty_critical_band_raises() -> None:
    # Lines far below the critical band of a 1 kHz tone (≈922-1085 Hz).
    with pytest.raises(ValueError, match="critical band"):
        mean_narrowband_level([50.0, 51.0, 52.0], [100.0, 110.0, 120.0], 1000.0)


def test_mean_narrowband_level_few_lines_returns_unpruned_mean() -> None:
    # Fewer than five lines on one side of the tone: the iteration stops on the
    # first pass and returns the pre-prune mean (Annex D fallback).
    freqs = [930.0, 950.0, 970.0, 990.0, 1010.0, 1030.0, 1050.0, 1070.0]
    levels = [50.0, 50.0, 50.0, 50.0, 60.0, 50.0, 50.0, 50.0]
    ls = mean_narrowband_level(levels, freqs, 1000.0)
    # Line under investigation = nearest to 1000 Hz (990 Hz); only three lines
    # sit below it, so no pruning happens and LS is the raw mean of the rest.
    band = [50.0, 50.0, 50.0, 60.0, 50.0, 50.0, 50.0]
    expected = 10.0 * math.log10(
        np.mean([10.0 ** (x / 10.0) for x in band])
    ) - 10.0 * math.log10(1.5)
    assert ls == pytest.approx(expected)


def test_mean_narrowband_level_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="share their length"):
        mean_narrowband_level([50.0, 51.0], [100.0], 100.0)


def test_mean_narrowband_level_rejects_unsorted_frequencies() -> None:
    with pytest.raises(ValueError, match="increasing"):
        mean_narrowband_level([50.0, 51.0, 52.0], [100.0, 99.0, 101.0], 100.0)


def test_tone_level_rejects_non_finite_ls() -> None:
    with pytest.raises(ValueError):
        tone_level([50.0, 60.0, 50.0], [100.0, 102.7, 105.4], 102.7, float("nan"))


# ---------------------------------------------------------------------------
# Full-spectrum detection (Clause 5.3.8) and FG combination (Formula (17))
# ---------------------------------------------------------------------------
def test_analyze_spectrum_detects_annex_e_tones() -> None:
    result = analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert result.group_sizes is not None
    singles = result.group_sizes == 1
    found = sorted(round(float(f), 1) for f in result.tone_frequencies[singles])
    assert found == sorted(ref.ISO20065_E1_TONE_FREQUENCIES)


def test_analyze_spectrum_step3_combines_annex_e_band() -> None:
    # Clause 5.3.8 Step 3 inside analyze_spectrum: the three tones share the
    # critical band, so a combined FG entry appears with the Table E.2 "2 FG"
    # tone level LT = 72.15 dB (Formula (17), shared lines counted once).
    result = analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert result.group_sizes is not None
    fg = result.group_sizes > 1
    assert int(np.sum(fg)) == 1
    assert int(result.group_sizes[fg][0]) == 3
    assert result.tone_levels[fg][0] == pytest.approx(
        ref.ISO20065_E1_LT_FG, abs=0.02
    )
    # Step 4: the decisive audibility is the FG entry (Table E.2 rates the
    # group at the most audible member; on the *truncated* E.1 spectrum the
    # 158.8 Hz tone's masking level is underestimated, so the anchor differs
    # from the full-spectrum printed one, but the decisive entry is the FG).
    assert result.decisive_audibility == pytest.approx(
        float(np.max(result.audibilities[fg])), abs=1e-12
    )


def test_analyze_spectrum_137hz_tone_audibility() -> None:
    # The 137.3 Hz tone's critical band is fully inside Table E.1, so its
    # detected audibility matches the oracle (ΔL ≈ 4.99 dB).
    result = analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    i = int(np.argmin(np.abs(result.tone_frequencies - 137.3)))
    assert result.audibilities[i] == pytest.approx(4.99, abs=0.03)


def test_detect_tone_at_first_line_not_dropped() -> None:
    # Regression: a peak on the very first line must not be skipped because the
    # left-neighbour test wrapped around to the last line (lev[-1]).
    from phonometry.psychoacoustics.quality.tone_audibility import _detect_tones

    freqs = np.arange(100.0, 100.0 + 2.7 * 40, 2.7)
    levels = np.full(freqs.size, 50.0)
    levels[0] = 90.0     # strong peak on the first line
    levels[-1] = 95.0    # even stronger last line (would trigger the wrap bug)
    detected = _detect_tones(levels, freqs, 2.7, 1.5)
    assert 0 in [peak for peak, _lo, _hi, _ls in detected]


def test_analyze_spectrum_rejects_inaudible_peak() -> None:
    # A single weak peak (below LS+6) yields no audible tone.
    freqs = list(np.arange(900.0, 1100.0, 2.7))
    levels = [50.0] * len(freqs)
    levels[len(freqs) // 2] = 52.0
    with pytest.raises(ValueError, match="No audible tone"):
        analyze_spectrum(levels, freqs, 2.7)


def test_combined_tone_level_matches_annex_e() -> None:
    value = combined_tone_level(
        ref.ISO20065_E1_LEVELS,
        ref.ISO20065_E1_FREQUENCIES,
        ref.ISO20065_E1_TONE_FREQUENCIES,
        ref.ISO20065_E1_TONE_LS,
    )
    assert value == pytest.approx(ref.ISO20065_E1_LT_FG, abs=0.02)


def test_combined_tone_level_single_tone_equals_tone_level() -> None:
    # With one tone the FG combination reduces to that tone's level.
    fg = combined_tone_level(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, [137.3], [49.22]
    )
    lt = tone_level(ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, 137.3, 49.22)
    assert fg == pytest.approx(lt)


def _e1_line_energy(indices: list[int]) -> float:
    """Hanning-corrected energy sum of the Table E.1 lines at *indices*.

    Computed here from the printed narrow-band levels with plain arithmetic,
    so the expectations below do not come from the code under test.
    """
    total = sum(10.0 ** (ref.ISO20065_E1_LEVELS[i] / 10.0) for i in indices)
    return 10.0 * math.log10(total) - 10.0 * math.log10(HANNING_BANDWIDTH_FACTOR)


def test_combined_tone_level_counts_a_shared_line_once() -> None:
    """Formula (17) with Anmerkung 2: overlapping tonal runs share their lines.

    The Annex E "2 FG" oracle cannot test this. Its three tones at
    118.4 / 137.3 / 158.8 Hz occupy the disjoint line runs 115.7-121.1,
    129.2-140.0 and 156.1-161.5 Hz, so counting every line once and summing
    the three tone levels outright give the same 72.15 dB: the printed value
    does not discriminate the deduplication from a plain union sum.

    These two cases do. Both pairs lie inside the same committed Table E.1
    spectrum, and the expected values are built from the printed line levels
    rather than from ``combined_tone_level``.
    """
    lev, freq = ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES
    ls = ref.ISO20065_E1_LS

    # (a) Identical runs. 134.6 Hz and 137.3 Hz both sit inside the run
    # 129.2-140.0 Hz (Table E.1 indices 12-16), so the union IS that run and
    # the combined level must equal the single-tone level of Formula (8)
    # exactly -- not approximately, since the same lines are summed.
    same_run = _e1_line_energy([12, 13, 14, 15, 16])
    fg_same = combined_tone_level(lev, freq, [134.6, 137.3], [ls, ls])
    assert fg_same == pytest.approx(same_run, abs=1e-12)
    assert fg_same == tone_level(lev, freq, 137.3, ls)
    # An implementation that summed the two tone levels instead would land
    # 10 lg 2 = 3.01 dB higher; assert that the gap is real and its exact size.
    naive_same = energy_sum_level(
        [tone_level(lev, freq, 134.6, ls), tone_level(lev, freq, 137.3, ls)],
        effective_bandwidth_factor=1.0,
    )
    assert naive_same - fg_same == pytest.approx(10.0 * math.log10(2.0), abs=1e-9)

    # (b) Nested runs. At 156.1 Hz the run is 156.1-164.2 Hz (indices 22-25)
    # and at 158.8 Hz it is 156.1-161.5 Hz (indices 22-24), the second strictly
    # inside the first, so the union is the wider run alone.
    nested = _e1_line_energy([22, 23, 24, 25])
    fg_nested = combined_tone_level(lev, freq, [156.1, 158.8], [ls, ls])
    assert fg_nested == pytest.approx(nested, abs=1e-12)
    assert fg_nested == tone_level(lev, freq, 156.1, ls)
    naive_nested = energy_sum_level(
        [tone_level(lev, freq, 156.1, ls), tone_level(lev, freq, 158.8, ls)],
        effective_bandwidth_factor=1.0,
    )
    assert naive_nested - fg_nested == pytest.approx(2.9104, abs=1e-3)
    assert naive_nested > fg_nested


def test_combined_tone_level_no_double_counting() -> None:
    # Regression guard on the ordering of the deduplicated sum: the combined
    # level of a pair whose runs overlap can never exceed the plain sum of the
    # two tone levels, and can never fall below the louder of them.
    fg = combined_tone_level(
        ref.ISO20065_E1_LEVELS,
        ref.ISO20065_E1_FREQUENCIES,
        [134.6, 137.3],  # adjacent tones sharing lines
        [49.22, 49.22],
    )
    singles = [
        tone_level(ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, f, 49.22)
        for f in (134.6, 137.3)
    ]
    assert np.isfinite(fg)
    assert max(singles) <= fg <= energy_sum_level(
        singles, effective_bandwidth_factor=1.0
    )


def test_combined_tone_level_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="share their length"):
        combined_tone_level(
            ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, [137.3], [49.0, 50.0]
        )


# ---------------------------------------------------------------------------
# Separate evaluation of two tones below 1000 Hz (Formulae (18)/(19))
#
# No numeric worked example exercises this branch (the Annex E band groups three
# tones, so the "exactly two tones" rule never fires there). These tests fix the
# closed-form Formula (19) and the decision logic against the DIN 45681 Annex J
# reference program; they are logic/consistency checks, not a numeric oracle.
# ---------------------------------------------------------------------------
def test_separation_frequency_minimum_at_reference() -> None:
    # fD = 21·10^(1.2·|lg(fT/212)|^1.8): the |lg| makes fT = 212 Hz the minimum,
    # where the exponent vanishes and fD = 21 Hz exactly.
    assert two_tone_separation_frequency(212.0) == pytest.approx(21.0)


def test_separation_frequency_matches_reference_program() -> None:
    # Reproduce the DIN 45681 Annex J expression verbatim:
    #   fD = 21 * 10 ^ (1.2 * Abs(Log(fT / 212) / Log(10)) ^ 1.8)
    for ft in (88.0, 137.3, 300.0, 500.0, 999.0):
        expected = 21.0 * 10.0 ** (1.2 * abs(math.log10(ft / 212.0)) ** 1.8)
        assert two_tone_separation_frequency(ft) == pytest.approx(expected)


def test_separation_frequency_grows_either_side_of_reference() -> None:
    # The absolute value gives a symmetric-in-log minimum at 212 Hz.
    base = two_tone_separation_frequency(212.0)
    assert two_tone_separation_frequency(120.0) > base
    assert two_tone_separation_frequency(600.0) > base


def test_separation_frequency_rejects_bad_frequency() -> None:
    with pytest.raises(ValueError, match="tone_frequency"):
        two_tone_separation_frequency(0.0)


def test_resolve_separately_true_when_far_apart() -> None:
    # 200/260 Hz, both < 1000 Hz, |Δf| = 60 Hz ≫ fD(200) ≈ 21 Hz → separate.
    assert resolve_tones_separately(200.0, 260.0, 3.0, 2.0) is True


def test_resolve_separately_false_when_close() -> None:
    # Annex E tones 118.4/137.3 Hz: |Δf| = 18.9 Hz < fD(137.3) ≈ 24.1 Hz →
    # combined. This is consistent with the Annex E FG grouping.
    assert resolve_tones_separately(118.4, 137.3, 4.0, 5.0) is False


def test_resolve_separately_false_at_or_above_1000hz() -> None:
    # The rule only applies when BOTH tones lie below 1000 Hz.
    assert resolve_tones_separately(1200.0, 1400.0, 3.0, 2.0) is False
    assert resolve_tones_separately(900.0, 1100.0, 3.0, 2.0) is False


def test_resolve_separately_uses_more_prominent_tone() -> None:
    # fT is the tone with the larger ΔL; which tone dominates picks the threshold
    # and, at a spacing that straddles the two thresholds, flips the decision.
    f1, f2 = 300.0, 323.1  # |Δf| = 23.1 Hz
    d_lo = two_tone_separation_frequency(f1)  # ≈ 23.02 Hz
    d_hi = two_tone_separation_frequency(f2)  # ≈ 23.91 Hz
    assert d_lo < abs(f2 - f1) < d_hi
    assert resolve_tones_separately(f1, f2, 5.0, 3.0) is True  # fT = 300, thr low
    assert resolve_tones_separately(f1, f2, 3.0, 5.0) is False  # fT = 323.1, thr hi


def test_resolve_separately_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        resolve_tones_separately(-1.0, 200.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        resolve_tones_separately(200.0, 260.0, math.nan, 1.0)


# ---------------------------------------------------------------------------
# Mean audibility (Formula (20)) and the no-tone convention (Formula (21))
# ---------------------------------------------------------------------------
def test_mean_audibility_matches_annex_e() -> None:
    value = mean_audibility(ref.ISO20065_DECISIVE_AUDIBILITIES)
    assert value == pytest.approx(ref.ISO20065_MEAN_AUDIBILITY, abs=0.05)


def test_mean_audibility_is_energy_mean() -> None:
    deltas = [4.0, 6.0, NO_TONE_AUDIBILITY]
    expected = 10.0 * math.log10(np.mean([10.0 ** (d / 10.0) for d in deltas]))
    assert mean_audibility(deltas) == pytest.approx(expected)


def test_mean_audibility_between_min_and_max() -> None:
    deltas = ref.ISO20065_DECISIVE_AUDIBILITIES
    value = mean_audibility(deltas)
    assert min(deltas) <= value <= max(deltas)


# ---------------------------------------------------------------------------
# assess_tones result
# ---------------------------------------------------------------------------
def _annex_e_result() -> ToneAudibilityResult:
    freqs = [t[0] for t in ref.ISO20065_ANNEX_E_TONES]
    ls = [t[1] for t in ref.ISO20065_ANNEX_E_TONES]
    lt = [t[2] for t in ref.ISO20065_ANNEX_E_TONES]
    return assess_tones(freqs, lt, ls, ref.ISO20065_LINE_SPACING)


def test_assess_tones_reproduces_annex_e_column() -> None:
    result = _annex_e_result()
    expected = np.array([t[3] for t in ref.ISO20065_ANNEX_E_TONES])
    assert np.max(np.abs(result.audibilities - expected)) < 0.03


def test_assess_tones_decisive() -> None:
    result = _annex_e_result()
    # Per-tone decisive of spectrum 1 is the 137.3 Hz tone (ΔL ≈ 4.99 dB).
    assert result.decisive_frequency == pytest.approx(137.3)
    assert result.decisive_audibility == pytest.approx(4.99, abs=0.03)
    assert result.decisive_audibility == float(np.max(result.audibilities))


def test_assess_tones_audible_mask() -> None:
    result = _annex_e_result()
    assert result.audible.all()  # all nine Annex E tones have ΔL > 0
    quiet = assess_tones([200.0], [40.0], [50.0], 2.7)
    assert not quiet.audible.any()


def test_assess_tones_fields_consistent() -> None:
    result = _annex_e_result()
    assert result.lower_corners.shape == result.tone_frequencies.shape
    # LG = LS + 10 lg(Δfc/Δf) elementwise.
    expected_lg = result.mean_narrowband_levels + 10.0 * np.log10(
        result.critical_bandwidths / result.line_spacing
    )
    assert np.allclose(result.critical_band_levels, expected_lg)
    # ΔL = LT − LG − av elementwise.
    assert np.allclose(
        result.audibilities,
        result.tone_levels - result.critical_band_levels - result.masking_indices,
    )


def test_plot_smoke() -> None:
    ax = _annex_e_result().plot()
    assert ax.get_ylabel()
    import matplotlib.pyplot as plt

    plt.close("all")


def test_plot_levels_view_draws_tone_levels_over_frequency() -> None:
    """view='levels' draws Lpt above the critical-band masking noise Lpn."""
    import matplotlib.pyplot as plt

    result = _annex_e_result()
    ax = result.plot(view="levels")
    assert ax.get_xscale() == "log"          # continuous frequency axis
    assert "level" in ax.get_ylabel().lower()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("L_\\mathrm{pt}" in label for label in labels)
    plt.close("all")


def test_plot_rejects_unknown_view() -> None:
    result = _annex_e_result()
    with pytest.raises(ValueError, match="Unknown view"):
        result.plot(view="spectrum")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_critical_bandwidth_rejects_bad_frequency(bad: float) -> None:
    with pytest.raises(ValueError):
        critical_bandwidth_engineering(bad)


def test_critical_band_level_rejects_bad_spacing() -> None:
    with pytest.raises(ValueError):
        critical_band_level(50.0, 137.3, 0.0)


def test_audibility_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        audibility_from_levels(float("nan"), 60.0, -2.0)


def test_energy_sum_rejects_empty() -> None:
    with pytest.raises(ValueError):
        energy_sum_level([])


def test_mean_audibility_rejects_empty() -> None:
    with pytest.raises(ValueError):
        mean_audibility([])


def test_assess_tones_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="share their length"):
        assess_tones([100.0, 200.0], [60.0], [50.0], 2.7)


def test_assess_tones_rejects_empty() -> None:
    with pytest.raises(ValueError):
        assess_tones([], [], [], 2.7)


def test_assess_tones_rejects_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="positive"):
        assess_tones([0.0], [60.0], [50.0], 2.7)


# ---------------------------------------------------------------------------
# Table E.2 full columns: LG, av, band limits and uncertainty U
# ---------------------------------------------------------------------------
def test_table_e2_lg_and_av_columns() -> None:
    """Every Table E.2 LG (Formula (12)) and av (Formula (13)) value chains
    from the printed LS and fT to <= 0.03 dB (2-decimal print rounding)."""
    for (ft, ls, _lt, _dl), lg_p, av_p in zip(
        ref.ISO20065_ANNEX_E_TONES, ref.ISO20065_E2_LG, ref.ISO20065_E2_AV
    ):
        assert critical_band_level(ls, ft, ref.ISO20065_LINE_SPACING) == pytest.approx(
            lg_p, abs=0.03
        )
        assert masking_index(ft) == pytest.approx(av_p, abs=0.01)


def test_table_e2_band_limits_are_line_snapped() -> None:
    """The printed Table E.2 f1/f2 are the first/last FFT lines inside the
    analytic Formula (4)/(5) band: each printed limit lies within one line
    spacing inside the analytic corner."""
    df = ref.ISO20065_LINE_SPACING
    for (ft, _ls, _lt, _dl), (f1_p, f2_p) in zip(
        ref.ISO20065_ANNEX_E_TONES, ref.ISO20065_E2_BAND_LIMITS
    ):
        f1_a, f2_a = critical_band_corners(ft)
        assert f1_a < f1_p <= f1_a + df + 0.01, f"{ft} Hz lower limit"
        assert f2_a - df - 0.01 <= f2_p < f2_a, f"{ft} Hz upper limit"


def test_table_e2_uncertainty_of_the_137hz_tone() -> None:
    """analyze_spectrum reports the Clause 6 extended uncertainty; on the
    E.1 spectrum the decisive 137.3 Hz tone reproduces the printed
    U = 2.79 dB. The flanking tones' critical bands extend beyond the
    truncated E.1 table, so their U reproduces within 0.1 dB only."""
    res = analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )
    assert res.extended_uncertainties is not None
    assert res.group_sizes is not None
    singles = res.group_sizes == 1
    by_freq = dict(
        zip(res.tone_frequencies[singles], res.extended_uncertainties[singles])
    )
    assert by_freq[137.3] == pytest.approx(2.79, abs=0.02)
    assert by_freq[118.4] == pytest.approx(3.66, abs=0.1)
    assert by_freq[158.8] == pytest.approx(3.51, abs=0.1)


def test_table_e2_fg_uncertainty() -> None:
    """The '2 FG' row of Table E.2: with the N summated tone levels as the K
    summands (the Clause 6 reading for combined tones) and the decisive
    tone's noise lines, U reproduces the printed 3.21 dB."""
    from phonometry.psychoacoustics.quality.tone_audibility import (
        _mean_narrowband_level_lines,
    )

    lev = np.asarray(ref.ISO20065_E1_LEVELS)
    _, kept = _mean_narrowband_level_lines(
        lev, np.asarray(ref.ISO20065_E1_FREQUENCIES), 137.3
    )
    u = audibility_uncertainty(
        ref.ISO20065_E2_FG_TONE_LEVELS, lev[kept], 137.3, ref.ISO20065_LINE_SPACING
    )
    assert u == pytest.approx(ref.ISO20065_E2_FG_U, abs=0.01)


def test_uncertainty_constants_and_validation() -> None:
    from phonometry.psychoacoustics.quality.tone_audibility import (
        COVERAGE_FACTOR_90,
        SIGMA_NARROWBAND_LEVEL,
    )

    # Clause 6 constants: uniform 3 dB level sigma, k = 1.645 (90 % bilateral).
    assert SIGMA_NARROWBAND_LEVEL == 3.0
    assert COVERAGE_FACTOR_90 == 1.645
    with pytest.raises(ValueError, match="at least one line"):
        audibility_uncertainty([], [50.0], 137.3, 2.7)
    with pytest.raises(ValueError, match="finite"):
        audibility_uncertainty([60.0, np.nan], [50.0], 137.3, 2.7)
    with pytest.raises(ValueError, match="share their length"):
        mean_audibility_uncertainty([9.18, 6.04], [3.21])


# ---------------------------------------------------------------------------
# Table E.4: decisive tones of the five spectra, and the mean uncertainty
# ---------------------------------------------------------------------------
def test_table_e4_decisive_chain() -> None:
    """Each Table E.4 decisive row chains LS/LT -> LG, av, dL to the printed
    values (<= 0.03 dB residual from the 2-decimal intermediates)."""
    for ft, dl_p, ls, lt, lg_p, av_p, _u in ref.ISO20065_E4_DECISIVE_ROWS:
        assert critical_band_level(ls, ft, ref.ISO20065_LINE_SPACING) == pytest.approx(
            lg_p, abs=0.03
        )
        assert masking_index(ft) == pytest.approx(av_p, abs=0.01)
        assert tone_audibility(
            lt, ls, ft, ref.ISO20065_LINE_SPACING
        ) == pytest.approx(dl_p, abs=0.03)


def test_table_e4_mean_audibility_uncertainty() -> None:
    """Annex E Step 4: the extended uncertainty of the mean audibility over
    the five spectra (Formulae (28)/(29)) reproduces the printed 1.38 dB and
    respects the printed 1.4 dB check margin for fewer than 12 spectra."""
    u_j = [row[6] for row in ref.ISO20065_E4_DECISIVE_ROWS]
    u_mean = mean_audibility_uncertainty(ref.ISO20065_DECISIVE_AUDIBILITIES, u_j)
    assert u_mean == pytest.approx(ref.ISO20065_E4_MEAN_UNCERTAINTY, abs=0.01)
    assert u_mean <= 1.4


# ---------------------------------------------------------------------------
# Table E.3: all tonal components of the five spectra (data consistency)
# ---------------------------------------------------------------------------
def test_table_e3_decisive_audibilities_and_mean() -> None:
    """The decisive audibility of each spectrum is the maximum over its tone
    and FG audibilities (Clause 5.3.8 Step 4) and matches the printed bold
    values; their Formula (20) energy mean is the printed 6.96 dB. (The
    narrow-band lines of spectra 2-5 are not printed, so E.3 cannot be
    chained from levels; this locks the printed record's consistency.)"""
    decisives = []
    for j in sorted(ref.ISO20065_E3_TONES):
        candidates = [dl for _f, dl in ref.ISO20065_E3_TONES[j]]
        candidates += [dl for _f, dl in ref.ISO20065_E3_FG[j]]
        decisives.append(max(candidates))
    assert decisives == pytest.approx(ref.ISO20065_DECISIVE_AUDIBILITIES)
    # 0.05 dB: the print carries 2-decimal dLj, and the standard's own mean
    # lands at 6.96 vs the 6.98 recomputed from the rounded decisives.
    assert mean_audibility(decisives) == pytest.approx(
        ref.ISO20065_MEAN_AUDIBILITY, abs=0.05
    )


# ---------------------------------------------------------------------------
# DIN 45681:2005-03 Anhang I, Beispiel I.3 (wind-energy plant): the parent
# standard's second end-to-end oracle, independent of the ISO/PAS 20065
# Annex E example. Line spacing 2.6917 Hz.
# ---------------------------------------------------------------------------
def test_din_anhang_i_decisive_tone_from_spectrum() -> None:
    """Tabelle I.9 -> I.10 row k = 2 (j = 24): the from-levels chain
    reproduces the printed LS = 41.71, LT = 68.10, LG = 57.68, av = -2.10
    and decisive dL = 12.52 dB (2-decimal print rounding)."""
    ft, dl_p, ls_p, lt_p, lg_p, av_p, _u = ref.DIN45681_I10_DECISIVE
    df = ref.DIN45681_LINE_SPACING
    ls = mean_narrowband_level(
        ref.DIN45681_I9_LEVELS, ref.DIN45681_I9_FREQUENCIES, ft
    )
    lt = tone_level(ref.DIN45681_I9_LEVELS, ref.DIN45681_I9_FREQUENCIES, ft, ls)
    assert ls == pytest.approx(ls_p, abs=0.02)
    assert lt == pytest.approx(lt_p, abs=0.02)
    assert critical_band_level(ls, ft, df) == pytest.approx(lg_p, abs=0.02)
    assert masking_index(ft) == pytest.approx(av_p, abs=0.01)
    assert tone_audibility(lt, ls, ft, df) == pytest.approx(dl_p, abs=0.03)


def test_din_anhang_i_decisive_tone_uncertainty() -> None:
    """Tabelle I.10 row k = 2: the Clause 6 / Anhang G extended uncertainty of
    the decisive tone reproduces the printed u = 3.18 dB from the K = 4 tone
    lines and the M noise lines of the final Formula (6) iteration."""
    from phonometry.psychoacoustics.quality.tone_audibility import (
        _mean_narrowband_level_lines,
    )

    ft = ref.DIN45681_I10_DECISIVE[0]
    lev = np.asarray(ref.DIN45681_I9_LEVELS)
    freqs = np.asarray(ref.DIN45681_I9_FREQUENCIES)
    _, kept = _mean_narrowband_level_lines(lev, freqs, ft)
    # Bold tone lines of Tabelle I.9: 296.1-304.2 Hz (indices 16..19).
    u = audibility_uncertainty(lev[16:20], lev[kept], ft, ref.DIN45681_LINE_SPACING)
    assert u == pytest.approx(ref.DIN45681_I10_DECISIVE[6], abs=0.02)


def test_din_anhang_i_analyze_spectrum_end_to_end() -> None:
    """analyze_spectrum on the Tabelle I.9 lines finds exactly the decisive
    298.8 Hz tone with the printed audibility and uncertainty."""
    res = analyze_spectrum(
        ref.DIN45681_I9_LEVELS,
        ref.DIN45681_I9_FREQUENCIES,
        ref.DIN45681_LINE_SPACING,
    )
    assert res.tone_frequencies.size == 1
    assert res.decisive_frequency == pytest.approx(298.8, abs=0.05)
    assert res.decisive_audibility == pytest.approx(
        ref.DIN45681_I10_DECISIVE[1], abs=0.03
    )
    assert res.extended_uncertainties is not None
    assert res.extended_uncertainties[0] == pytest.approx(
        ref.DIN45681_I10_DECISIVE[6], abs=0.02
    )


def test_din_anhang_i_two_tone_rule_matches_5fg_row() -> None:
    """Tabelle I.10 rows k = 4/5 (705.2 / 732.1 Hz, 26.9 Hz apart, both below
    1000 Hz) are printed combined ("5 FG"): the Formula (18)/(19) decision
    must NOT separate them -- the only printed outcome of the two-tone rule.
    The FG chain then reproduces the printed dL = 3.22 dB from the printed
    combined LT = 55.95 dB at the more audible member (732.1 Hz)."""
    ft4, dl4, _ls4, _lt4 = ref.DIN45681_I10_K4
    ft5, dl5, ls5, _lt5 = ref.DIN45681_I10_K5
    assert resolve_tones_separately(ft4, ft5, dl4, dl5) is False
    ft, dl_p, ls_p, lt_fg, lg_p, av_p, _u = ref.DIN45681_I10_5FG
    assert ls_p == ls5
    df = ref.DIN45681_LINE_SPACING
    assert critical_band_level(ls_p, ft, df) == pytest.approx(lg_p, abs=0.02)
    assert masking_index(ft) == pytest.approx(av_p, abs=0.01)
    assert tone_audibility(lt_fg, ls_p, ft, df) == pytest.approx(dl_p, abs=0.03)


def test_din_anhang_i11_rows_j45_j48_chain() -> None:
    """Tabelle I.11 rows j = 45 / j = 48: the Formulae (12)-(14) chain from
    the printed LS/LT reproduces the printed LG, av and dL columns."""
    df = ref.DIN45681_LINE_SPACING
    for ft, dl_p, ls_p, lt_p, lg_p, av_p, _u in (
        ref.DIN45681_I11_J45,
        ref.DIN45681_I11_J48,
    ):
        assert critical_band_level(ls_p, ft, df) == pytest.approx(lg_p, abs=0.02)
        assert masking_index(ft) == pytest.approx(av_p, abs=0.01)
        assert tone_audibility(lt_p, ls_p, ft, df) == pytest.approx(dl_p, abs=0.03)


def test_din_tabelle_i6_6fg_plain_sum_reproduces_printed_audibility() -> None:
    """Tabelle I.6 row "6 FG": the printed dL = 9.12 dB reproduces from the
    plain Formula (17) energy sum of the three tone levels (82.87 dB) through
    the chain at 592.2 Hz. (The row's printed LT cell, 81.11 dB, follows the
    Anmerkung-2 shared-line dedupe instead and contradicts the printed dL;
    only the dL chain is pinned, see reference_data.)"""
    ft, dl_p, ls_p, _av_p = ref.DIN45681_I6_6FG
    lt = energy_sum_level(
        ref.DIN45681_I6_6FG_TONE_LEVELS, effective_bandwidth_factor=1.0
    )
    assert lt == pytest.approx(82.87, abs=0.01)
    assert tone_audibility(lt, ls_p, ft, ref.DIN45681_LINE_SPACING) == (
        pytest.approx(dl_p, abs=0.03)
    )


def test_din_anhang_i3_mean_audibility_maps_to_kt_4() -> None:
    """Anhang I.3 Steps 3/5: the printed mean audibility 6.38 dB maps to
    KT = 4 dB (DIN Abschnitt 6 Tabelle 1 == ISO 1996-2:2017 Table J.1)."""
    from phonometry import tonal_adjustment_from_mean_audibility

    assert tonal_adjustment_from_mean_audibility(
        ref.DIN45681_I3_MEAN_AUDIBILITY
    ) == ref.DIN45681_I3_KT


def test_din_tabelle_a1_critical_bandwidths() -> None:
    """Tabelle A.1: Formula (2) reproduces every printed (integer-rounded)
    critical bandwidth from 100 Hz to 13.5 kHz, except the 250 Hz print
    quirk recorded in reference_data (printed 105 vs Formula (2) 104.47)."""
    for ft, dfc_printed in ref.DIN45681_A1_BANDWIDTHS:
        computed = critical_bandwidth_engineering(ft)
        if ft == 250.0:
            assert computed == pytest.approx(104.47, abs=0.01)
            assert dfc_printed == 105.0
        else:
            # abs 0.5 covers the integer print; the informative table drifts
            # from Formula (2) by up to ~0.03 % at the top rows (3469 printed
            # vs 3467.9 computed at 13.5 kHz), covered by the rel term.
            assert computed == pytest.approx(dfc_printed, abs=0.5, rel=5e-4)


def test_analyze_spectrum_step3_merges_chained_clusters() -> None:
    # The critical bandwidth grows with frequency, so a chain of tones at
    # 900/950/1000 Hz produces three DIFFERENT own-band candidate sets
    # ({900,950}, {900,950,1000}, {950,1000}); Step 3 must merge them into
    # exactly one FG entry, not three overlapping combinations.
    df = 2.5
    freq = np.arange(200.0, 1600.0, df)
    lev = np.full(freq.size, 40.0)
    for ft in (900.0, 950.0, 1000.0):
        lev[int(np.argmin(np.abs(freq - ft)))] = 65.0
    result = analyze_spectrum(lev, freq, df)
    assert result.group_sizes is not None
    fg = result.group_sizes > 1
    assert int(np.sum(fg)) == 1
    assert int(result.group_sizes[fg][0]) == 3
