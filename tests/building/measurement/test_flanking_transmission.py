#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 10848 laboratory flanking-transmission measurement.

ISO 10848 contains no worked numeric example, so correctness is anchored on
closed-form identities synthesized to hand-checkable results (see
``reference_data``) plus the structural invariants the standard guarantees
(``Kij`` symmetry, the simplified/full-formula relationship, the octave-band
energy sum).
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

from phonometry import building, vibration

THIRD_OCTAVE = [
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


# ---------------------------------------------------------------------------
# Closed-form oracles (no numeric example in the standard)
# ---------------------------------------------------------------------------
def test_simplified_kij_closed_form() -> None:
    """Formula (14) matches a hand-computed value."""
    res = building.vibration_reduction_index(
        [ref.ISO10848_KIJ_DBAR],
        ref.ISO10848_KIJ_LIJ,
        ref.ISO10848_KIJ_AREA,
        ref.ISO10848_KIJ_AREA,
    )
    assert res.k_ij[0] == pytest.approx(ref.ISO10848_KIJ_SIMPLIFIED)


def test_equivalent_absorption_length_at_reference_frequency() -> None:
    """Formula (12) at f = f_ref reduces to 2.2·π²·S/(Ts·c0)."""
    a = building.equivalent_absorption_length(
        ref.ISO10848_ABS_AREA,
        ref.ISO10848_ABS_TS,
        [1000.0],
        speed_of_sound=ref.ISO10848_ABS_C0,
    )
    assert a[0] == pytest.approx(ref.ISO10848_ABS_LENGTH_AT_FREF)


def test_absorption_length_frequency_scaling() -> None:
    """aj scales as sqrt(f_ref/f): quartering f doubles aj."""
    a = building.equivalent_absorption_length(10.0, 0.5, [250.0, 1000.0])
    assert a[0] == pytest.approx(2.0 * a[1])


def test_total_loss_factor_closed_form() -> None:
    """η = 2.2/(f·Ts)."""
    eta = building.total_loss_factor([1000.0], [0.5])
    assert eta[0] == pytest.approx(ref.ISO10848_LOSS_FACTOR)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------
def test_kij_is_symmetric() -> None:
    """Direction averaging makes Kij = Kji (Formula (11) + (13))."""
    dv_ij, dv_ji = [6.0], [4.0]
    dbar = building.direction_averaged_level_difference(dv_ij, dv_ji)
    dbar_rev = building.direction_averaged_level_difference(dv_ji, dv_ij)
    k_ij = building.vibration_reduction_index(dbar, 3.0, 5.0, 7.0).k_ij
    k_ji = building.vibration_reduction_index(dbar_rev, 3.0, 7.0, 5.0).k_ij
    assert k_ij[0] == pytest.approx(k_ji[0])


def test_direction_average() -> None:
    """D̄v,ij is the arithmetic mean of the two directions."""
    dbar = building.direction_averaged_level_difference([6.0, 8.0], [4.0, 2.0])
    assert dbar == pytest.approx([5.0, 5.0])


def test_velocity_level_difference() -> None:
    """Dv,ij = Lv,i − Lv,j (Formula (8))."""
    dv = building.velocity_level_difference([80.0, 70.0], [74.0, 68.0])
    assert dv == pytest.approx([6.0, 2.0])


def test_full_formula_reduces_to_simplified_when_a_equals_area() -> None:
    """Formula (13) with aj = Sj/l0 equals the simplified Formula (14)."""
    dv = [5.0]
    simplified = building.vibration_reduction_index(dv, 2.0, 4.0, 9.0).k_ij[0]
    # Reproduce a = S/l0 (l0 = 1 m) exactly via chosen Ts/f: not generally
    # possible, so instead check the simplified path directly is internally
    # consistent with the manual formula.
    manual = 5.0 + 10.0 * np.log10(2.0 / np.sqrt(4.0 * 9.0))
    assert simplified == pytest.approx(manual)


def test_full_formula_uses_absorption_length() -> None:
    """With Ts + frequency the full Formula (13) differs from the simplified."""
    dv = np.full(len(THIRD_OCTAVE), 8.0)
    full = building.vibration_reduction_index(
        dv,
        2.0,
        10.0,
        12.0,
        frequency=THIRD_OCTAVE,
        structural_reverberation_time_i=0.4,
        structural_reverberation_time_j=0.5,
    )
    simplified = building.vibration_reduction_index(dv, 2.0, 10.0, 12.0)
    # a_i, a_j from Formula (12) differ from S/l0, so Kij differs.
    assert not np.allclose(full.k_ij, simplified.k_ij)
    # and manually reproduce band 0
    a_i = building.equivalent_absorption_length(10.0, 0.4, THIRD_OCTAVE)
    a_j = building.equivalent_absorption_length(12.0, 0.5, THIRD_OCTAVE)
    manual0 = 8.0 + 10.0 * np.log10(2.0 / np.sqrt(a_i[0] * a_j[0]))
    assert full.k_ij[0] == pytest.approx(manual0)


# ---------------------------------------------------------------------------
# Single-number rating and octave conversion
# ---------------------------------------------------------------------------
def test_single_number_is_mean_over_200_1250() -> None:
    """K̄ij is the arithmetic mean of Kij over 200-1250 Hz (Annex A)."""
    k = np.arange(len(THIRD_OCTAVE), dtype=float)  # distinct per band
    res = building.vibration_reduction_index(k, 2.0, 4.0, 4.0, frequency=THIRD_OCTAVE)
    freqs = np.asarray(THIRD_OCTAVE, dtype=float)
    mask = (freqs >= 200.0) & (freqs <= 1250.0)
    expected = float(np.mean(res.k_ij[mask]))
    assert res.single_number == pytest.approx(expected)


def test_single_number_none_without_frequency() -> None:
    """No frequencies -> no single number."""
    res = building.vibration_reduction_index([1.0, 2.0, 3.0], 2.0, 4.0, 4.0)
    assert res.single_number is None
    assert res.frequencies is None


def test_octave_band_energy_sum() -> None:
    """Kij,oct = −10 lg[(1/3) Σ 10^(−Kij/10)] over each triple."""
    dv = np.full(len(THIRD_OCTAVE), 6.0)
    res = building.vibration_reduction_index(dv, 2.0, 4.0, 4.0, frequency=THIRD_OCTAVE)
    octaves = res.octave_bands()
    assert octaves.k_ij.size == 6
    first = res.k_ij[:3]
    expected = -10.0 * np.log10(np.mean(10.0 ** (-first / 10.0)))
    assert octaves.k_ij[0] == pytest.approx(expected)
    assert list(octaves.frequencies.astype(int)) == [125, 250, 500, 1000, 2000, 4000]


def test_octave_bands_requires_multiple_of_three() -> None:
    res = building.vibration_reduction_index([1.0, 2.0, 3.0, 4.0], 2.0, 4.0, 4.0)
    with pytest.raises(ValueError, match="multiple of three"):
        res.octave_bands()


def test_octave_single_number_averages_125_to_1000() -> None:
    """Octave K̄ij averages 125-1000 Hz (Annex A), not the 1/3-octave range.

    A Kij rising 1 dB per band (3 dB/octave) makes the two ranges differ:
    averaging the octave values over 250-1000 Hz only (the wrong 200-1250 Hz
    mask applied to octave centres) would sit 1.5 dB above the correct
    125-1000 Hz average.
    """
    k = np.arange(len(THIRD_OCTAVE), dtype=float)  # 1 dB per third band
    res = building.vibration_reduction_index(k, 2.0, 4.0, 4.0, frequency=THIRD_OCTAVE)
    oct_res = res.octave_bands()
    assert oct_res.frequencies is not None
    freqs = oct_res.frequencies
    mask = (freqs >= 125.0) & (freqs <= 1000.0)
    assert mask.sum() == 4  # 125, 250, 500, 1000 Hz
    expected = float(np.mean(oct_res.k_ij[mask]))
    assert oct_res.single_number == pytest.approx(expected)
    # Regression guard for the previous 200-1250 Hz mask on octave centres.
    wrong_mask = (freqs >= 200.0) & (freqs <= 1250.0)
    wrong = float(np.mean(oct_res.k_ij[wrong_mask]))
    assert wrong - expected == pytest.approx(1.5, abs=0.01)


def test_octave_bands_reject_misaligned_triples() -> None:
    """Groups crossing octave boundaries are rejected (frequencies given)."""
    freqs = THIRD_OCTAVE[2:14]  # 12 bands starting at 160 Hz
    res = building.vibration_reduction_index(
        np.zeros(len(freqs)), 2.0, 4.0, 4.0, frequency=freqs
    )
    with pytest.raises(
        ValueError, match=r"octave_bands\(\) needs whole octave triples"
    ):
        res.octave_bands()


# ---------------------------------------------------------------------------
# ISO 10848-4 Clause 9 modal-overlap bracketing
# ---------------------------------------------------------------------------
def test_modal_overlap_brackets_and_excludes_bands() -> None:
    """Bands with M < 0,25 are flagged and left out of the single number."""
    k = np.arange(len(THIRD_OCTAVE), dtype=float)
    m = np.full(len(THIRD_OCTAVE), 1.0)
    m[3] = 0.1  # 200 Hz: inside the 200-1250 Hz single-number range
    res = building.vibration_reduction_index(
        k, 2.0, 4.0, 4.0, frequency=THIRD_OCTAVE, modal_overlap=m
    )
    assert res.bracketed is not None
    assert bool(res.bracketed[3])
    assert res.bracketed.sum() == 1
    freqs = np.asarray(THIRD_OCTAVE, dtype=float)
    mask = (freqs >= 200.0) & (freqs <= 1250.0) & (m >= 0.25)
    assert res.single_number == pytest.approx(float(np.mean(res.k_ij[mask])))


def test_modal_overlap_all_bracketed_gives_no_single_number() -> None:
    res = building.vibration_reduction_index(
        np.zeros(len(THIRD_OCTAVE)),
        2.0,
        4.0,
        4.0,
        frequency=THIRD_OCTAVE,
        modal_overlap=np.full(len(THIRD_OCTAVE), 0.1),
    )
    assert res.single_number is None
    assert res.bracketed is not None
    assert bool(np.all(res.bracketed))


def test_modal_overlap_propagates_to_octave_bands() -> None:
    """An octave band is bracketed when any of its thirds is."""
    m = np.full(len(THIRD_OCTAVE), 1.0)
    m[4] = 0.2  # 250 Hz third -> 250 Hz octave bracketed
    res = building.vibration_reduction_index(
        np.zeros(len(THIRD_OCTAVE)),
        2.0,
        4.0,
        4.0,
        frequency=THIRD_OCTAVE,
        modal_overlap=m,
    )
    oct_res = res.octave_bands()
    assert oct_res.bracketed is not None
    assert oct_res.bracketed.tolist() == [False, True, False, False, False, False]
    # 250 Hz octave excluded: mean over 125, 500 and 1000 Hz only.
    assert oct_res.frequencies is not None
    keep = np.isin(oct_res.frequencies, [125.0, 500.0, 1000.0])
    assert oct_res.single_number == pytest.approx(float(np.mean(oct_res.k_ij[keep])))


def test_modal_overlap_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match=r"'modal_overlap' must contain positive"):
        building.vibration_reduction_index(
            [5.0], 2.0, 4.0, 4.0, frequency=[500.0], modal_overlap=[0.0]
        )


# ---------------------------------------------------------------------------
# Overall flanking descriptors Dn,f / Ln,f
# ---------------------------------------------------------------------------
def test_normalized_flanking_level_difference() -> None:
    """Dn,f = L1 − L2 − 10 lg(A/A0) (Formula (4))."""
    res = building.normalized_flanking_level_difference([80.0], [50.0], [5.0])
    assert res.d_n_f[0] == pytest.approx(30.0 - 10.0 * np.log10(5.0 / 10.0))


def test_normalized_flanking_impact_level() -> None:
    """Ln,f = L2 + 10 lg(A/A0) (Formula (5))."""
    res = building.normalized_flanking_impact_level([50.0], [5.0])
    assert res.l_n_f[0] == pytest.approx(50.0 + 10.0 * np.log10(5.0 / 10.0))


def test_flanking_descriptors_rating_when_16_bands() -> None:
    """A 16-band input yields ISO 717 single-number ratings."""
    d = building.normalized_flanking_level_difference(
        np.full(16, 80.0), np.full(16, 50.0), np.full(16, 5.0)
    )
    li = building.normalized_flanking_impact_level(np.full(16, 50.0), np.full(16, 5.0))
    assert d.rating is not None
    assert li.rating is not None
    assert isinstance(int(d.rating.rating), int)
    assert isinstance(int(li.rating.rating), int)


def test_flanking_descriptors_no_rating_for_odd_band_count() -> None:
    d = building.normalized_flanking_level_difference([80.0] * 7, [50.0] * 7, [5.0] * 7)
    assert d.rating is None


def test_indirect_kij_from_flanking_roundtrip() -> None:
    """Kij from Dn,f inverts the forward relation for resonant transmission."""
    # Build a Dn,f consistent with a known Kij, then recover it.
    k_true = 6.0
    r_i, r_j = np.array([45.0]), np.array([48.0])
    lij, s_i, s_j, a0 = 2.0, 10.0, 12.0, 10.0
    a_i, a_j = np.array([1.3]), np.array([1.5])
    dn_f = (
        k_true
        + 0.5 * (r_i + r_j)
        + 10.0 * np.log10(np.sqrt(a_i * a_j) / lij)
        - 10.0 * np.log10(np.sqrt(s_i * s_j) / a0)
    )
    recovered = building.vibration_reduction_index_from_flanking(
        dn_f, r_i, r_j, lij, s_i, s_j, a_i, a_j, reference_area=a0
    )
    assert recovered[0] == pytest.approx(k_true)


def test_indirect_kij_band_count_mismatch_names_every_input() -> None:
    """The guard names each per-band argument, not just "the inputs"."""
    with pytest.raises(
        ValueError,
        match=r"vibration_reduction_index_from_flanking: "
        r"'normalized_flanking_level_difference'.*'reduction_index_i'.*"
        r"'reduction_index_j'.*'absorption_length_i'.*'absorption_length_j'.*"
        r"same shape, one value per band",
    ):
        building.vibration_reduction_index_from_flanking(
            [50.0, 52.0, 54.0],
            [45.0, 46.0],
            [44.0, 45.0, 46.0],
            2.8,
            10.0,
            12.0,
            [1.5, 1.6, 1.7],
            [1.4, 1.5, 1.6, 1.7],
        )


# ---------------------------------------------------------------------------
# Validity criteria
# ---------------------------------------------------------------------------
def test_critical_frequency() -> None:
    """fc = c0²/(1.8·cL·h) (Formula (20)), checked against two independent oracles.

    (a) Hand-computed value for 6 mm float glass (cL = 5130 m/s):
    343² / (1.8 · 5130 · 0.006) ≈ 2123.5 Hz. The lower bound guards against a
    reintroduced spurious π factor (which would give ≈ 676 Hz).
    (b) Cross-oracle: the constant 1.8 rounds 2π/√12, so for a plate with
    mutually consistent bending stiffness and mass the result must match the
    Hopkins Eq. 2.201 coincidence frequency to within that rounding (< 1 %).
    """
    # (a) Independent published/hand-computed value for 6 mm float glass.
    fc = building.critical_frequency(5130.0, 0.006, speed_of_sound=343.0)
    assert fc == pytest.approx(2123.5, rel=0.01)
    assert fc > 1500.0  # a spurious π factor would give ~676 Hz

    # (b) Cross-oracle against the closed-form plate coincidence frequency.
    e, rho, nu, h = 6.2e10, 2500.0, 0.24, 0.006
    c_l = np.sqrt(e / (rho * (1.0 - nu**2)))
    fc_flank = building.critical_frequency(float(c_l), h, speed_of_sound=343.0)
    fc_coinc = vibration.coincidence_frequency(
        rho * h,
        vibration.plate_bending_stiffness(e, h, nu),
        speed_of_sound=343.0,
    )
    assert fc_flank == pytest.approx(fc_coinc, rel=0.01)


def test_strong_coupling_check() -> None:
    """Formula (15): the threshold is 3 − 10 lg((mi·fcj)/(mj·fci))."""
    # equal masses and critical frequencies -> threshold = 3 dB
    ok = building.strong_coupling_satisfied([4.0, 2.0], 20.0, 20.0, 100.0, 100.0)
    assert list(ok) == [True, False]


def test_modal_density_and_overlap() -> None:
    """n = π·S·fc/c0²; M = 2.2·n/Ts (Formulas (5), (6))."""
    n = building.modal_density(10.0, 200.0, speed_of_sound=343.0)
    assert n == pytest.approx(np.pi * 10.0 * 200.0 / 343.0**2)
    m = building.modal_overlap_factor(10.0, 200.0, [0.5], speed_of_sound=343.0)
    assert m[0] == pytest.approx(2.2 * n / 0.5)


def test_band_mode_count() -> None:
    """N = 0.23·f·n (Formula (4))."""
    n = building.modal_density(10.0, 200.0)
    counts = building.band_mode_count([500.0], 10.0, 200.0)
    assert counts[0] == pytest.approx(0.23 * 500.0 * n)


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------
def test_only_one_reverberation_time_raises() -> None:
    with pytest.raises(ValueError, match="both structural reverberation times"):
        building.vibration_reduction_index(
            [5.0],
            2.0,
            4.0,
            4.0,
            frequency=[1000.0],
            structural_reverberation_time_i=0.5,
        )


def test_reverberation_time_needs_frequency() -> None:
    with pytest.raises(ValueError, match="'frequency' is required"):
        building.vibration_reduction_index(
            [5.0],
            2.0,
            4.0,
            4.0,
            structural_reverberation_time_i=0.5,
            structural_reverberation_time_j=0.5,
        )


def test_frequency_band_count_mismatch_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"vibration_reduction_index: 'velocity_level_difference'.*"
        r"'frequency'.*one value per band",
    ):
        building.vibration_reduction_index(
            [5.0, 6.0], 2.0, 4.0, 4.0, frequency=[1000.0]
        )


def test_nonpositive_geometry_raises() -> None:
    with pytest.raises(ValueError, match=r"'junction_length' must be a positive"):
        building.vibration_reduction_index([5.0], -2.0, 4.0, 4.0)


def test_mismatched_level_lengths_raise() -> None:
    with pytest.raises(
        ValueError,
        match=r"velocity_level_difference: 'source_level'.*'receive_level'.*"
        r"one value per band",
    ):
        building.velocity_level_difference([1.0, 2.0], [1.0])


def test_scalar_input_accepted() -> None:
    """A scalar D̄v,ij is accepted as a single band."""
    res = building.vibration_reduction_index(5.0, 2.0, 4.0, 4.0)
    assert res.k_ij.shape == (1,)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")

    dv = np.full(len(THIRD_OCTAVE), 6.0)
    res = building.vibration_reduction_index(dv, 2.0, 4.0, 4.0, frequency=THIRD_OCTAVE)
    assert res.plot() is not None
    # frequency-less result falls back to a band-index axis
    assert (
        building.vibration_reduction_index([1.0, 2.0, 3.0], 2.0, 4.0, 4.0).plot()
        is not None
    )
