#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 15186 sound-intensity insulation module."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import reference_data as ref

from phonometry import building


def _levels_for_target_ri(
    ri: list[float], lp1: float, sm: float, s: float
) -> np.ndarray:
    """Receiving-side LIn that make Formula (7) return exactly ``ri``."""
    ri = np.asarray(ri, dtype=np.float64)
    return lp1 - 6.0 - 10.0 * np.log10(sm / s) - ri


# ---------------------------------------------------------------------------
# Intensity sound reduction index RI (Formula (7))
# ---------------------------------------------------------------------------


def test_ri_reproduces_formula_7_scalar() -> None:
    """RI = Lp1 - 6 - [LIn + 10 lg(Sm/S)] band by band."""
    r = building.intensity_sound_reduction(
        [80.0], [40.0], measurement_area=10.0, area=10.0
    )
    assert r.r_i[0] == pytest.approx(34.0)  # 80 - 6 - 40 - 0
    r2 = building.intensity_sound_reduction(
        [80.0], [40.0], measurement_area=20.0, area=10.0
    )
    assert r2.r_i[0] == pytest.approx(34.0 - 10.0 * np.log10(2.0))


def test_ri_reproduces_iso717_rw_through_intensity_path() -> None:
    """Feeding LIn that yield the ISO 717-1 curve returns RI,w = 30 dB."""
    lin = _levels_for_target_ri(
        ref.ISO15186_1_REF_RI,
        ref.ISO15186_1_REF_LP1,
        ref.ISO15186_1_REF_SM,
        ref.ISO15186_1_REF_S,
    )
    result = building.intensity_sound_reduction(
        [ref.ISO15186_1_REF_LP1] * 16,
        lin,
        measurement_area=ref.ISO15186_1_REF_SM,
        area=ref.ISO15186_1_REF_S,
    )
    np.testing.assert_allclose(result.r_i, ref.ISO15186_1_REF_RI, atol=1e-9)
    assert result.rating is not None
    assert result.rating.rating == ref.ISO15186_1_REF_RIW


def test_ri_energy_averages_positions() -> None:
    """A 2-D (positions, bands) input is energy-averaged before Formula (7)."""
    positions = np.array([[40.0, 42.0], [46.0, 44.0]])
    avg = 10.0 * np.log10(np.mean(10.0 ** (0.1 * positions), axis=0))
    r_multi = building.intensity_sound_reduction(
        [[80.0, 80.0], [80.0, 80.0]], positions, measurement_area=10.0, area=10.0
    )
    r_avg = building.intensity_sound_reduction(
        [80.0, 80.0], avg, measurement_area=10.0, area=10.0
    )
    np.testing.assert_allclose(r_multi.r_i, r_avg.r_i)


def test_ri_modified_adds_kc() -> None:
    """RI,M = RI + Kc, and its rating is formed independently."""
    lin = _levels_for_target_ri(ref.ISO15186_1_REF_RI, 85.0, 12.0, 10.0)
    freq = np.array(
        [
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
        ],
        dtype=float,
    )
    kc = building.adaptation_term_kc(freq)
    result = building.intensity_sound_reduction(
        [85.0] * 16, lin, measurement_area=12.0, area=10.0, kc=kc
    )
    assert result.r_i_modified is not None
    np.testing.assert_allclose(result.r_i_modified, result.r_i + kc)
    assert result.rating_modified is not None
    # Kc > 0 everywhere, so the modified rating cannot be lower.
    assert result.rating_modified.rating >= result.rating.rating


def test_ri_rating_none_off_band_count() -> None:
    """No automatic rating when the band count is neither 16 nor 5."""
    r = building.intensity_sound_reduction(
        [80.0] * 18, [40.0] * 18, measurement_area=10.0, area=10.0
    )
    assert r.rating is None
    with pytest.raises(ValueError, match="16"):
        r.plot()


# ---------------------------------------------------------------------------
# Adaptation term Kc (Annex B)
# ---------------------------------------------------------------------------


def test_kc_reproduces_printed_table_b1() -> None:
    """Kc reproduces all 21 printed Table B.1 rows at one decimal place."""
    kc = building.adaptation_term_kc(ref.ISO15186_1_KC_BANDS)
    np.testing.assert_allclose(kc, ref.ISO15186_1_KC_B1_PRINTED, atol=0.05)


def test_kc_b1_reference_room_reduces_to_b2() -> None:
    """Formula (B.1) with the reference room equals (B.2) within 0,001 dB."""
    b2 = building.adaptation_term_kc(ref.ISO15186_1_KC_BANDS)
    b1 = building.adaptation_term_kc(
        ref.ISO15186_1_KC_BANDS, boundary_area=117.0, volume=81.0
    )
    np.testing.assert_allclose(b1, b2, atol=1e-3)


def test_kc_decreases_with_frequency() -> None:
    """Kc is strictly monotone decreasing (the 61,4/f term shrinks)."""
    kc = building.adaptation_term_kc([100.0, 200.0, 400.0, 800.0, 1600.0])
    assert np.all(np.diff(kc) < 0.0)


def test_kc_requires_both_room_parameters() -> None:
    with pytest.raises(ValueError, match="both"):
        building.adaptation_term_kc([500.0], boundary_area=117.0)
    with pytest.raises(ValueError, match="both"):
        building.adaptation_term_kc([500.0], volume=81.0)


# ---------------------------------------------------------------------------
# Element normalized level difference DI,n,e (Formula (8))
# ---------------------------------------------------------------------------


def test_element_normalized_difference_formula_8() -> None:
    """DI,n,e = Lp1 - 6 - (LIn + 10 lg(Sm/A0)) + 10 lg N (corrected sign).

    The printed Formula (8) subtracts its 10 lg(N) term, which contradicts
    ISO 10140-2:2010 Formula (6) and ISO 15186-2:2010 Formula (12) (see
    docs/ERRATA.md); the per-unit value adds it, and n > 1 warns about the
    deviation from the print.
    """
    d = building.intensity_element_normalized_difference(
        [80.0], [40.0], measurement_area=10.0, n=1
    )
    assert d.d_i_n_e[0] == pytest.approx(34.0)  # Sm = A0, N = 1
    with pytest.warns(UserWarning, match="Formula \\(8\\)"):
        d2 = building.intensity_element_normalized_difference(
            [80.0], [40.0], measurement_area=10.0, n=2
        )
    assert d2.d_i_n_e[0] == pytest.approx(34.0 + 10.0 * np.log10(2.0))
    assert d2.n == 2
    assert d2.measurement_area == pytest.approx(10.0)


def test_element_normalized_rejects_bad_n() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        building.intensity_element_normalized_difference(
            [80.0], [40.0], measurement_area=10.0, n=0
        )


# ---------------------------------------------------------------------------
# Surface pressure-intensity indicator FpI (Formula (10))
# ---------------------------------------------------------------------------


def test_fpi_is_lp_minus_lin() -> None:
    fpi = building.surface_pressure_intensity_indicator([60.0, 58.0], [55.0, 54.0])
    np.testing.assert_allclose(fpi, [5.0, 4.0])


def test_fpi_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"'lp'.*'l_in'.*same shape"):
        building.surface_pressure_intensity_indicator([60.0, 58.0], [55.0])


# ---------------------------------------------------------------------------
# Subarea combination (Formulas (11)-(12))
# ---------------------------------------------------------------------------


def test_combine_subareas_energy_average() -> None:
    """Equal-level subareas average to the same level; Sm is the total."""
    lin, sm = building.combine_subareas([[40.0, 42.0], [40.0, 42.0]], [5.0, 5.0])
    np.testing.assert_allclose(lin, [40.0, 42.0])
    assert sm == pytest.approx(10.0)


def test_combine_subareas_area_weighting() -> None:
    """The larger subarea dominates the energy average."""
    lin, sm = building.combine_subareas([[50.0], [40.0]], [9.0, 1.0])
    expected = 10.0 * np.log10((9.0 * 10**5.0 + 1.0 * 10**4.0) / 10.0)
    assert lin[0] == pytest.approx(expected)
    assert sm == pytest.approx(10.0)


def test_combine_subareas_validation() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        building.combine_subareas([40.0, 42.0], [5.0])
    with pytest.raises(ValueError, match=r"'measurement_area'.*one value per subarea"):
        building.combine_subareas([[40.0, 42.0], [40.0, 42.0]], [5.0])
    with pytest.raises(ValueError, match=r"'measurement_area' must be a one-dim"):
        building.combine_subareas([[40.0], [42.0]], [[5.0], [5.0]])
    with pytest.raises(ValueError, match="non-zero"):
        building.combine_subareas([[40.0], [42.0]], [5.0, 0.0])


def test_combine_subareas_negative_direction_rule() -> None:
    """Clause 6.4.6: a reverse-flow subarea enters Formula (11) with -Smi
    while Sm keeps the unsigned area sum (Formula (12)).
    """
    # Forward 9 m2 at 50 dB, reverse 1 m2 at 40 dB (~10 % reverse power).
    lin, sm = building.combine_subareas([[50.0], [40.0]], [9.0, -1.0])
    expected = 10.0 * np.log10((9.0 * 10**5.0 - 1.0 * 10**4.0) / 10.0)
    assert lin[0] == pytest.approx(expected)
    assert sm == pytest.approx(10.0)  # Sm = sum(|Smi|)
    # The unsigned sum would overestimate LIn by 10 lg(9,1/8,9) ~ 0,1 dB of
    # numerator energy for this case; check the exact signed/unsigned gap.
    unsigned, _ = building.combine_subareas([[50.0], [40.0]], [9.0, 1.0])
    assert unsigned[0] - lin[0] == pytest.approx(10.0 * np.log10(9.1 / 8.9))


def test_combine_subareas_reverse_flow_dominating_raises() -> None:
    # Reverse energy equal to (or exceeding) the forward flow leaves no level.
    with pytest.raises(ValueError, match="not positive"):
        building.combine_subareas([[50.0], [50.0]], [5.0, -5.0])
    with pytest.raises(ValueError, match="not positive"):
        building.combine_subareas([[50.0], [53.0]], [5.0, -5.0])


# ---------------------------------------------------------------------------
# Shared input validation
# ---------------------------------------------------------------------------


def test_reduction_rejects_nonpositive_areas() -> None:
    with pytest.raises(ValueError, match="measurement_area"):
        building.intensity_sound_reduction(
            [80.0], [40.0], measurement_area=0.0, area=10.0
        )
    with pytest.raises(ValueError, match="area"):
        building.intensity_sound_reduction(
            [80.0], [40.0], measurement_area=10.0, area=-1.0
        )


def test_reduction_band_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"'lp1'.*'l_in'.*same shape"):
        building.intensity_sound_reduction(
            [80.0, 80.0], [40.0], measurement_area=10.0, area=10.0
        )


def test_element_normalized_band_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"'lp1'.*'l_in'.*same shape"):
        building.intensity_element_normalized_difference(
            [80.0, 80.0], [40.0], measurement_area=10.0
        )


def test_kc_band_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"'lp1'.*'kc'.*same shape"):
        building.intensity_sound_reduction(
            [80.0], [40.0], measurement_area=10.0, area=10.0, kc=[1.0, 2.0]
        )


def test_reduction_rejects_modified_index_of_another_band_count() -> None:
    """``r_i_modified`` is the column the verbose fiche never measures.

    ``FpI`` and ``δpI0`` are arguments of ``report()`` and are measured there
    against the reported band count; ``RI,M`` arrives on the result and is
    admitted unmeasured. A column one entry too long prints beside ``RI``
    shifted by a band -- the 100 Hz row shows the surplus value and the
    3150 Hz row the 2500 Hz one -- and the last entry is dropped by the table
    without a word.
    """
    result = building.intensity_sound_reduction(
        [80.0] * 16, [40.0] * 16, measurement_area=12.0, area=10.0, kc=[1.0] * 16
    )
    assert result.r_i_modified is not None
    stretched = np.append(result.r_i_modified, 0.0)
    with pytest.raises(ValueError, match=r"'r_i_modified'.*one value per band"):
        dataclasses.replace(result, r_i_modified=stretched)
