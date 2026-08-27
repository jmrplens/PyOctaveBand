#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 16251-1:2014 floor-covering impact-sound improvement."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import reference_data as ref

from phonometry import building

#: The clause 6.3 measurement range: 18 one-third-octave bands 100-5000 Hz.
_CLAUSE_63_FREQS = [
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
]


# ---------------------------------------------------------------------------
# ISO 717-2 reference floor anchor (the numeric oracle)
# ---------------------------------------------------------------------------
def test_reference_floor_rating_is_78() -> None:
    """weighted_impact_rating of the ISO 717-2 Table 4 reference floor is 78 dB."""
    res = building.weighted_impact_rating(ref.ISO717_2_REFERENCE_FLOOR_LN_R0)
    assert res.rating == ref.ISO717_2_REFERENCE_FLOOR_LN_R0_W
    assert res.ci == ref.ISO717_2_REFERENCE_FLOOR_CI


def test_zero_improvement_gives_zero_delta_lw() -> None:
    assert building.weighted_impact_improvement(np.zeros(16)) == 0


# ---------------------------------------------------------------------------
# Real measured oracle: textile carpet (Foret, Chene & Guigou-Carter, Forum
# Acusticum 2011, Figure 4). The Delta-L spectrum was digitized from the vector
# chart to +/- 0,5 dB per band; the ISO 717-2 rating must reproduce the paper's
# published ΔLw = 29 dB. See tests/reference_data/ for the full provenance.
# ---------------------------------------------------------------------------
def test_foret2011_carpet_rates_to_29() -> None:
    """The measured carpet spectrum rates to the paper's ΔLw = 29 dB."""
    bare = np.full(len(ref.FORET2011_CARPET_FREQ), 100.0)
    delta_l = np.asarray(ref.FORET2011_CARPET_ISO16251_DELTA_L)
    res = building.impact_improvement(bare, bare - delta_l, ref.FORET2011_CARPET_FREQ)
    assert res.delta_lw == ref.FORET2011_CARPET_ISO16251_DELTA_LW


def test_foret2011_carpet_rating_robust_to_half_db() -> None:
    """The single-number ΔLw is stable to +/- 0,5 dB band perturbations.

    This justifies the figure-digitization tolerance: the published rating does
    not depend on reading each band better than about half a decibel.
    """
    base = np.asarray(ref.FORET2011_CARPET_ISO16251_DELTA_L)
    # Independent per-band reading error (a uniform shift is excluded on
    # purpose: it moves the single number one-for-one, unlike independent
    # errors that cancel over the sixteen bands). Include the closed-interval
    # endpoints the open random draw never reaches: each band set to exactly
    # base +/- 0,5 by an alternating sign pattern, then seeded interior draws.
    signs = np.where(np.arange(base.size) % 2 == 0, 0.5, -0.5)
    for pattern in (signs, -signs):
        assert (
            building.weighted_impact_improvement((base + pattern)[:16])
            == ref.FORET2011_CARPET_ISO16251_DELTA_LW
        )
    rng = np.random.default_rng(0)
    for _ in range(64):
        perturbed = base + rng.uniform(-0.5, 0.5, size=base.shape)
        assert (
            building.weighted_impact_improvement(perturbed[:16])
            == ref.FORET2011_CARPET_ISO16251_DELTA_LW
        )


def test_flat_improvement_shifts_delta_lw_one_for_one() -> None:
    # A uniform ΔL lowers Ln,r uniformly, so ΔLw equals the flat improvement.
    assert building.weighted_impact_improvement(np.full(16, 10.0)) == 10


def test_weighted_improvement_requires_16_bands() -> None:
    five_bands = np.zeros(5)
    with pytest.raises(
        ValueError, match=r"'delta_l' must give the 16 one-third-octave values"
    ):
        building.weighted_impact_improvement(five_bands)


def test_weighted_improvement_rejects_non_finite() -> None:
    bad = np.zeros(16)
    bad[0] = np.inf
    with pytest.raises(ValueError, match=r"'delta_l' must contain only finite values"):
        building.weighted_impact_improvement(bad)


# ---------------------------------------------------------------------------
# Formula (1) - acceleration level
# ---------------------------------------------------------------------------
def test_acceleration_level_formula_1() -> None:
    # 20 lg(1e-3 / 1e-6) = 20 * 3 = 60 dB.
    np.testing.assert_allclose(building.acceleration_level(1e-3), [60.0])
    np.testing.assert_allclose(building.acceleration_level([1e-6, 1e-5]), [0.0, 20.0])


def test_acceleration_level_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match=r"'acceleration' must be positive"):
        building.acceleration_level(0.0)
    with pytest.raises(ValueError, match="'reference' must be positive"):
        building.acceleration_level(1e-3, reference=0.0)


# ---------------------------------------------------------------------------
# Formula (2) - background correction
# ---------------------------------------------------------------------------
def test_background_correction_three_branches() -> None:
    # margin 30 (>=15, unchanged), 10 (6..15, energy subtraction), 3 (<6, limit).
    lp = np.array([80.0, 80.0, 65.0])
    lb = np.array([50.0, 70.0, 62.0])
    corrected, limited = building.background_corrected_level(lp, lb)
    expected_mid = 10.0 * np.log10(10.0**8 - 10.0**7)  # ~79.54
    np.testing.assert_allclose(corrected, [80.0, expected_mid, 65.0 - 1.3])
    np.testing.assert_array_equal(limited, [False, False, True])


def test_background_correction_boundary_at_6_subtracts() -> None:
    # ISO 16251-1 (unlike ISO 10140-4) energy-subtracts at exactly margin = 6 dB.
    corrected, limited = building.background_corrected_level([56.0], [50.0])
    np.testing.assert_allclose(corrected, [10.0 * np.log10(10.0**5.6 - 10.0**5.0)])
    assert not bool(limited[0])


def test_background_correction_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"'signal_and_background'.*same shape"):
        building.background_corrected_level([80.0, 70.0], [50.0])


# ---------------------------------------------------------------------------
# Formulae (3)/(4) - improvement
# ---------------------------------------------------------------------------
def test_impact_improvement_difference_and_rating() -> None:
    freqs = ref.ISO717_2_REFERENCE_FLOOR_FREQ
    bare = np.full(16, 75.0)
    cov = bare - np.array(
        [0, 0, 1, 2, 4, 7, 11, 15, 18, 21, 23, 25, 27, 28, 29, 30], dtype=float
    )
    res = building.impact_improvement(bare, cov, freqs)
    np.testing.assert_allclose(
        res.improvement, [0, 0, 1, 2, 4, 7, 11, 15, 18, 21, 23, 25, 27, 28, 29, 30]
    )
    # ΔLw computed automatically for the 16 rating bands.
    assert res.delta_lw == building.weighted_impact_improvement(res.improvement)
    assert not bool(np.any(res.limited))


def test_annex_c2_improvement_oracle() -> None:
    """ISO 717-2 Annex C Table C.2: ΔLw = 15 dB and CI,Δ = -9 dB."""
    dl = np.asarray(ref.ISO717_2_ANNEX_C2_DELTA_L)
    assert building.weighted_impact_improvement(dl) == ref.ISO717_2_ANNEX_C2_DELTA_LW
    assert (
        building.impact_improvement_adaptation_term(dl)
        == ref.ISO717_2_ANNEX_C2_CI_DELTA
    )
    # End-to-end through the ISO 16251-1 front-end.
    bare = np.full(16, 75.0)
    res = building.impact_improvement(
        bare, bare - dl, ref.ISO717_2_REFERENCE_FLOOR_FREQ
    )
    assert res.delta_lw == ref.ISO717_2_ANNEX_C2_DELTA_LW
    assert res.ci_delta == ref.ISO717_2_ANNEX_C2_CI_DELTA


def test_baruch_2018_published_spectrum_crosscheck() -> None:
    """Published cross-check: Baruch et al. 2018 (Applied Acoustics 142, 18-28).

    Their Table 2 gives the improvement spectrum ΔL of a 10 mm soft covering on
    a 140 mm heavyweight floor from the full (Eq. 2) and simplified (Eq. 8)
    engineering models, and reports ΔLw = 16 dB (full) and 15 dB (simplified).
    Feeding those two spectra through the ISO 717-2 rating here gives 15 dB for
    both. The 1 dB gap on the full spectrum is the well-known rounding
    sensitivity of the ISO 717-2 shift near the 32,0 dB boundary (the two
    published spectra are nearly identical yet the paper itself splits them
    16 vs 15); our rating is the value validated against the ISO 717-2 Annex C
    worked examples, so the published figures corroborate it to within 1 dB.
    """
    freqs = np.asarray(ref.ISO717_2_REFERENCE_FLOOR_FREQ, dtype=float)  # 100-3150 Hz
    baruch_full = np.array(
        [
            0.25,
            0.39,
            0.64,
            0.98,
            1.48,
            2.25,
            3.38,
            4.84,
            6.79,
            9.28,
            12.02,
            15.09,
            18.79,
            22.32,
            25.99,
            29.89,
        ]
    )
    baruch_simplified = np.array(
        [
            0.25,
            0.39,
            0.64,
            0.97,
            1.48,
            2.24,
            3.38,
            4.82,
            6.76,
            9.25,
            11.97,
            15.02,
            18.69,
            22.19,
            25.82,
            29.67,
        ]
    )
    dlw_full = building.weighted_impact_improvement(baruch_full)
    dlw_simplified = building.weighted_impact_improvement(baruch_simplified)
    assert dlw_simplified == 15  # matches the paper exactly
    assert dlw_full == 15  # paper prints 16; ISO 717-2 rounding boundary
    assert abs(dlw_full - 16) <= 1
    assert abs(dlw_simplified - 15) <= 1
    # Both spectra reproduce the same rating through the ISO 16251-1 front-end.
    bare = np.full(16, 75.0)
    res_full = building.impact_improvement(bare, bare - baruch_full, freqs)
    assert res_full.delta_lw == dlw_full
    res_simplified = building.impact_improvement(bare, bare - baruch_simplified, freqs)
    assert res_simplified.delta_lw == dlw_simplified


def test_ci_delta_zero_improvement_is_zero() -> None:
    # ΔL = 0 leaves the reference floor unchanged: CI,r = CI,r,0 -> CI,Δ = 0.
    assert building.impact_improvement_adaptation_term(np.zeros(16)) == 0


def test_ci_delta_validation() -> None:
    short = np.zeros(5)
    with pytest.raises(
        ValueError, match=r"'delta_l' must give the 16 one-third-octave values"
    ):
        building.impact_improvement_adaptation_term(short)
    bad = np.zeros(16)
    bad[3] = np.nan
    with pytest.raises(ValueError, match=r"'delta_l' must contain only finite values"):
        building.impact_improvement_adaptation_term(bad)


def test_clause_63_18_band_spectrum_rates_on_sub_range() -> None:
    """A clause 6.3 spectrum (18 bands 100-5000 Hz) now yields ΔLw and CI,Δ
    from its 100-3150 Hz sub-range.
    """
    dl16 = np.asarray(ref.ISO717_2_ANNEX_C2_DELTA_L)
    dl18 = np.concatenate([dl16, [22.0, 21.0]])  # 4 k / 5 kHz extensions
    bare = np.full(18, 75.0)
    res = building.impact_improvement(bare, bare - dl18, _CLAUSE_63_FREQS)
    assert res.delta_lw == ref.ISO717_2_ANNEX_C2_DELTA_LW
    assert res.ci_delta == ref.ISO717_2_ANNEX_C2_CI_DELTA


def test_extended_21_band_spectrum_rates_on_sub_range() -> None:
    """The optional 50/63/80 Hz extension also rates on 100-3150 Hz."""
    freqs = [50.0, 63.0, 80.0, *_CLAUSE_63_FREQS]
    dl = np.concatenate([[1.0, 1.5, 2.0], ref.ISO717_2_ANNEX_C2_DELTA_L, [22.0, 21.0]])
    bare = np.full(21, 75.0)
    res = building.impact_improvement(bare, bare - dl, freqs)
    assert res.delta_lw == ref.ISO717_2_ANNEX_C2_DELTA_LW
    assert res.ci_delta == ref.ISO717_2_ANNEX_C2_CI_DELTA


def test_spectrum_missing_rating_bands_stays_unrated() -> None:
    # Without the full 100-3150 Hz sub-range no rating is formed.
    freqs = _CLAUSE_63_FREQS[:15]  # stops at 2500 Hz
    bare = np.full(15, 75.0)
    res = building.impact_improvement(bare, bare - 5.0, freqs)
    assert res.delta_lw is None
    assert res.ci_delta is None


def test_impact_improvement_with_background_flags_limited() -> None:
    freqs = [500.0, 1000.0]
    bare = np.array([80.0, 80.0])
    cov = np.array([70.0, 79.0])
    bg = np.array([50.0, 79.5])  # 2nd band: signal within 6 dB of background
    res = building.impact_improvement(bare, cov, freqs, background=bg)
    assert bool(res.limited[1])
    assert res.delta_lw is None  # not the 16 rating bands


def test_impact_improvement_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"'with_covering'.*same shape"):
        building.impact_improvement([75.0, 75.0], [70.0], [500.0, 1000.0])


def test_impact_improvement_per_position_averages_after_difference() -> None:
    # Two tapping positions, two bands; no background => ΔL is the mean over
    # positions of (L0 - L1).
    freqs = [500.0, 1000.0]
    l0 = np.array([[80.0, 78.0], [82.0, 76.0]])
    l1 = np.array([[70.0, 70.0], [74.0, 66.0]])
    res = building.impact_improvement(l0, l1, freqs)
    np.testing.assert_allclose(res.improvement, [(10 + 8) / 2, (8 + 10) / 2])
    assert res.improvement.shape == (2,)


def test_background_correction_precedes_averaging() -> None:
    # Formula (2) is non-linear, so per-position correction then averaging must
    # differ from averaging then correcting when margins are heterogeneous.
    freqs = [500.0]
    l0 = np.array([[80.0], [80.0]])
    l1 = np.array([[62.0], [80.0]])  # pos 1 close to background, pos 2 not
    bg = np.array([60.0])  # margins: L1 pos1 = 2 dB (<6 -> limit)
    res = building.impact_improvement(l0, l1, freqs, background=bg)
    assert bool(res.limited[0])  # a position hit the 1.3 dB limit
    # Averaging-then-correcting would miss that per-position limit flag.


def test_impact_improvement_background_shape_validation() -> None:
    bare = np.full((2, 2), 80.0)
    covered = np.full((2, 2), 70.0)
    mismatched_background = np.full((3, 2), 50.0)  # 3 rows against 2 positions
    with pytest.raises(ValueError, match="'background' must be"):
        building.impact_improvement(
            bare,
            covered,
            [500.0, 1000.0],
            background=mismatched_background,
        )


# ---------------------------------------------------------------------------
# Formula (5) - octave conversion
# ---------------------------------------------------------------------------
def test_octave_conversion_formula_5() -> None:
    freqs = [400.0, 500.0, 630.0]  # the 500 Hz octave triplet
    dl = np.array([10.0, 10.0, 10.0])
    oct_f, oct_dl = building.improvement_octave_bands(dl, freqs)
    np.testing.assert_allclose(oct_f, [500.0])
    # Equal thirds => octave equals the common value.
    np.testing.assert_allclose(oct_dl, [10.0])


def test_octave_conversion_energy_mean() -> None:
    freqs = [400.0, 500.0, 630.0]
    dl = np.array([6.0, 10.0, 14.0])
    _, oct_dl = building.improvement_octave_bands(dl, freqs)
    expected = -10.0 * np.log10(
        np.mean([10.0 ** (-6 / 10), 10.0 ** (-10 / 10), 10.0 ** (-14 / 10)])
    )
    np.testing.assert_allclose(oct_dl, [expected])


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["delta_lw", "ci_delta"])
def test_a_fractional_rating_is_refused_by_name(field: str) -> None:
    """ISO 717-2 rounds both ratings to whole decibels, and the fiche prints them so.

    A float smuggled into a hand-built result crashes the boxed statement's
    ``{ci_delta:+d}`` format with a message naming no field, no result and no
    fiche, or prints an unlocalised decimal point on a comma-decimal sheet.
    """
    freqs = ref.ISO717_2_REFERENCE_FLOOR_FREQ
    bare = np.full(16, 75.0)
    res = building.impact_improvement(bare, bare - 5.0, freqs)
    with pytest.raises(ValueError, match=rf"'{field}' must be an integer number"):
        dataclasses.replace(res, **{field: -1.0})


def test_an_adaptation_term_without_its_rating_is_refused() -> None:
    """``CI,delta`` and ``delta_lw`` are two readings of one rating, so both or neither.

    Formula (A.4) takes ``CI,r`` from the same weighted rating of the same
    ``Ln,r = Ln,r,0 - delta-L`` over the same 16 bands 100-3150 Hz that yields
    ``delta_lw``, so a spectrum that leaves the rating undetermined leaves the
    adaptation term undetermined too, and ``impact_improvement`` returns them
    together. A hand-built result pairing them the other way used to render a
    fiche that dropped the term without a word: the boxed statement of results
    fell through to the unrated characterisation headline.
    """
    unrated = building.impact_improvement(
        [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [500.0, 1000.0, 2000.0]
    )
    assert unrated.delta_lw is None
    assert unrated.ci_delta is None
    with pytest.raises(
        ValueError, match=r"'ci_delta' is the adaptation term of 'delta_lw'"
    ):
        dataclasses.replace(unrated, ci_delta=-11)


def test_a_rating_quoted_without_its_adaptation_term_is_allowed() -> None:
    """The other half of the pair is a legitimate omission, not a broken relation.

    ISO 16251-1 Clause 8 e) asks for ``delta-Lw (CI,delta)``, but a result
    carrying only the rating is a rating quoted without the term, and the fiche
    prints the bare ``delta-Lw`` instead of fabricating a ``(+0)``.
    """
    freqs = ref.ISO717_2_REFERENCE_FLOOR_FREQ
    bare = np.full(16, 75.0)
    res = building.impact_improvement(bare, bare - 5.0, freqs)
    assert dataclasses.replace(res, ci_delta=None).delta_lw == res.delta_lw


# ---------------------------------------------------------------------------
# Result helpers / plotting
# ---------------------------------------------------------------------------
def test_result_octave_bands_method() -> None:
    freqs = ref.ISO717_2_REFERENCE_FLOOR_FREQ
    bare = np.full(16, 75.0)
    res = building.impact_improvement(bare, bare - 5.0, freqs)
    oct_f, oct_dl = res.octave_bands()
    assert oct_f.tolist() == [125.0, 250.0, 500.0, 1000.0, 2000.0]
    np.testing.assert_allclose(oct_dl, np.full(5, 5.0))


def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    freqs = ref.ISO717_2_REFERENCE_FLOOR_FREQ
    bare = np.full(16, 75.0)
    res = building.impact_improvement(bare, bare - 8.0, freqs)
    ax = res.plot()
    assert ax.get_title().startswith("ISO 16251-1")
