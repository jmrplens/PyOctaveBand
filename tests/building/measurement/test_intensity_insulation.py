#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 15186 sound-intensity insulation module."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest
import reference_data as ref

from phonometry import building

if TYPE_CHECKING:
    from collections.abc import Callable


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
    with pytest.raises(
        ValueError, match=r"No single-number rating is available to plot"
    ):
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
    with pytest.raises(ValueError, match=r"Supply both 'boundary_area' and 'volume'"):
        building.adaptation_term_kc([500.0], boundary_area=117.0)
    with pytest.raises(ValueError, match=r"Supply both 'boundary_area' and 'volume'"):
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
    with pytest.warns(
        UserWarning, match=r"Formula \(8\) as printed subtracts 10 lg\(N\)"
    ):
        d2 = building.intensity_element_normalized_difference(
            [80.0], [40.0], measurement_area=10.0, n=2
        )
    assert d2.d_i_n_e[0] == pytest.approx(34.0 + 10.0 * np.log10(2.0))
    assert d2.n == 2
    assert d2.measurement_area == pytest.approx(10.0)


def test_element_normalized_rejects_bad_n() -> None:
    with pytest.raises(ValueError, match=r"'n' must be a positive integer"):
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
    with pytest.raises(ValueError, match=r"'l_in' must be a two-dimensional"):
        building.combine_subareas([40.0, 42.0], [5.0])
    with pytest.raises(ValueError, match=r"'measurement_area'.*one value per subarea"):
        building.combine_subareas([[40.0, 42.0], [40.0, 42.0]], [5.0])
    with pytest.raises(ValueError, match=r"'measurement_area' must be a one-dim"):
        building.combine_subareas([[40.0], [42.0]], [[5.0], [5.0]])
    with pytest.raises(
        ValueError, match=r"'measurement_area' must contain non-zero, finite areas"
    ):
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
    with pytest.raises(
        ValueError, match=r"signed subarea energy sum of Formula \(11\) is not positive"
    ):
        building.combine_subareas([[50.0], [50.0]], [5.0, -5.0])
    with pytest.raises(
        ValueError, match=r"signed subarea energy sum of Formula \(11\) is not positive"
    ):
        building.combine_subareas([[50.0], [53.0]], [5.0, -5.0])


# ---------------------------------------------------------------------------
# Shared input validation
# ---------------------------------------------------------------------------


def test_reduction_rejects_nonpositive_areas() -> None:
    with pytest.raises(ValueError, match=r"'measurement_area' must be positive"):
        building.intensity_sound_reduction(
            [80.0], [40.0], measurement_area=0.0, area=10.0
        )
    with pytest.raises(ValueError, match=r"'area' must be positive"):
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


# ---------------------------------------------------------------------------
# ISO 15186-3:2002 - low-frequency intensity method
# ---------------------------------------------------------------------------


def test_limp_panel_reproduces_printed_table_a1_plaster_column() -> None:
    """Annex A reproduces its own plaster-board column of Table A.1."""
    r = building.limp_panel_reduction_index(
        ref.ISO15186_3_ANNEX_A_BANDS,
        surface_mass=ref.ISO15186_3_PLASTER_SURFACE_MASS,
        area=ref.ISO15186_3_PLASTER_AREA,
        temperature=ref.ISO15186_3_ANNEX_A_TEMPERATURE,
        static_pressure=ref.ISO15186_3_ANNEX_A_PRESSURE,
    )
    # The table prints one decimal, so agreement is asserted at that
    # resolution: every band rounds to the published value.
    assert np.round(r, 1).tolist() == ref.ISO15186_3_PLASTER_TABLE_A1


def test_limp_panel_climate_enters_through_a4_and_a5() -> None:
    """Colder air is denser, so rho c rises and the mass law reads lower."""
    warm = building.limp_panel_reduction_index(
        [100.0], surface_mass=10.0, area=10.0, temperature=23.0
    )
    cold = building.limp_panel_reduction_index(
        [100.0], surface_mass=10.0, area=10.0, temperature=0.0
    )
    assert cold[0] < warm[0]
    # Formula (A.4) is linear in B, and (A.2) takes 20 lg of its reciprocal.
    half = building.limp_panel_reduction_index(
        [100.0], surface_mass=10.0, area=10.0, static_pressure=101300.0 / 2.0
    )
    assert half[0] - warm[0] == pytest.approx(20.0 * np.log10(2.0))


def test_limp_panel_doubling_mass_adds_six_decibels() -> None:
    """R0 = 20 lg(pi f m / rho c), so doubling m adds 20 lg 2."""
    single = building.limp_panel_reduction_index(
        ref.ISO15186_3_ANNEX_A_BANDS, surface_mass=10.0, area=10.0
    )
    double = building.limp_panel_reduction_index(
        ref.ISO15186_3_ANNEX_A_BANDS, surface_mass=20.0, area=10.0
    )
    assert np.allclose(double - single, 20.0 * np.log10(2.0))


def test_limp_panel_refuses_area_below_one_square_metre() -> None:
    """A.1 states Formula (A.3) for a panel of at least 1 m2."""
    with pytest.raises(ValueError, match="at least 1 m2"):
        building.limp_panel_reduction_index([100.0], surface_mass=10.0, area=0.5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"surface_mass": 0.0, "area": 10.0}, "surface_mass"),
        ({"surface_mass": 10.0, "area": 10.0, "static_pressure": -1.0}, "pressure"),
        ({"surface_mass": 10.0, "area": 10.0, "temperature": -300.0}, "temperature"),
        ({"surface_mass": np.nan, "area": 10.0}, "surface_mass"),
    ],
)
def test_limp_panel_rejects_impossible_inputs(
    kwargs: dict[str, float], match: str
) -> None:
    """Annex A refuses inputs its formulas cannot be evaluated at."""
    with pytest.raises(ValueError, match=match):
        building.limp_panel_reduction_index([100.0], **kwargs)


def test_limp_panel_rejects_nonpositive_frequency() -> None:
    """Formula (A.2) takes the logarithm of the frequency."""
    with pytest.raises(ValueError, match="'frequencies' must be positive"):
        building.limp_panel_reduction_index([0.0], surface_mass=10.0, area=10.0)


def test_low_frequency_ri_reproduces_formula_7() -> None:
    """RI = LpS - 9 - [LIn + 10 lg(Sm/S)], nine decibels and not six."""
    r = building.low_frequency_intensity_reduction(
        [84.0, 86.0, 88.0],
        [60.0, 61.0, 62.0],
        measurement_area=12.0,
        area=10.0,
        frequencies=[50.0, 63.0, 80.0],
    )
    expected = [
        lp - 9.0 - (lin + 10.0 * np.log10(12.0 / 10.0))
        for lp, lin in ((84.0, 60.0), (86.0, 61.0), (88.0, 62.0))
    ]
    assert r.r_i == pytest.approx(expected)


def test_low_frequency_ri_is_three_decibels_below_its_part_one_sibling() -> None:
    """The surface measurement sees the pressure the wall doubles."""
    kwargs = {"measurement_area": 12.0, "area": 10.0}
    part1 = building.intensity_sound_reduction([84.0], [60.0], **kwargs)
    part3 = building.low_frequency_intensity_reduction([84.0], [60.0], **kwargs)
    assert part1.r_i[0] - part3.r_i[0] == pytest.approx(3.0)


def test_low_frequency_indicator_is_formula_5() -> None:
    """FpI = Lp - LIn, both read on the receiving-side measurement surface."""
    r = building.low_frequency_intensity_reduction(
        [84.0, 86.0],
        [70.0, 79.0],
        measurement_area=10.0,
        area=10.0,
        l_p=[84.0, 86.0],
    )
    assert r.surface_pressure_intensity == pytest.approx([14.0, 7.0])


def test_low_frequency_indicator_is_not_the_source_room_level() -> None:
    """Formula (5) never touches LpS: without l_p there is no indicator."""
    r = building.low_frequency_intensity_reduction(
        [84.0, 86.0], [70.0, 79.0], measurement_area=10.0, area=10.0
    )
    assert r.surface_pressure_intensity is None
    assert r.qualified is None
    # Same source-room levels, two different receiving-side pressures: the
    # index cannot tell them apart, the indicator does.
    quiet = building.low_frequency_intensity_reduction(
        [84.0], [70.0], measurement_area=10.0, area=10.0, l_p=[75.0]
    )
    loud = building.low_frequency_intensity_reduction(
        [84.0], [70.0], measurement_area=10.0, area=10.0, l_p=[85.0]
    )
    assert quiet.r_i == pytest.approx(loud.r_i)
    assert quiet.qualified is not None
    assert loud.qualified is not None
    assert quiet.qualified[0]
    assert not loud.qualified[0]


@pytest.mark.parametrize(
    ("absorbing", "limit", "qualified"),
    [(False, 10.0, [False, True, True]), (True, 6.0, [False, False, True])],
)
def test_low_frequency_qualification_follows_clause_6_4_2(
    absorbing: bool, limit: float, qualified: list[bool]
) -> None:
    """FpI above 10 dB (or 6 dB for an absorbing specimen) is not qualified."""
    r = building.low_frequency_intensity_reduction(
        [90.0, 90.0, 90.0],
        [70.0, 70.0, 70.0],
        measurement_area=10.0,
        area=10.0,
        l_p=[81.0, 79.0, 76.0],
        absorbing_specimen_surface=absorbing,
    )
    assert r.indicator_limit == limit
    # 11, 9 and 6 dB: the limit itself is satisfactory, the clause refuses
    # only what exceeds it.
    assert r.surface_pressure_intensity == pytest.approx([11.0, 9.0, 6.0])
    assert r.qualified is not None
    assert r.qualified.tolist() == qualified


def test_low_frequency_keeps_the_index_of_a_refused_band() -> None:
    """A refused band is flagged, not dropped: 6.4.2 asks for a better setup."""
    r = building.low_frequency_intensity_reduction(
        [90.0], [70.0], measurement_area=10.0, area=10.0, l_p=[90.0]
    )
    assert r.qualified is not None
    assert not r.qualified[0]
    assert r.r_i[0] == pytest.approx(11.0)


@pytest.mark.parametrize("band", [40.0, 200.0, 1000.0])
def test_low_frequency_refuses_bands_outside_clause_6_6(band: float) -> None:
    """Clause 6.6 defines this method from 50 Hz to 160 Hz only."""
    with pytest.raises(ValueError, match="50 Hz to 160 Hz"):
        building.low_frequency_intensity_reduction(
            [80.0], [60.0], measurement_area=10.0, area=10.0, frequencies=[band]
        )


def test_low_frequency_accepts_every_band_clause_6_6_allows() -> None:
    """The three mandatory bands and the three optional ones."""
    bands = ref.ISO15186_3_ANNEX_A_BANDS
    r = building.low_frequency_intensity_reduction(
        [80.0] * len(bands),
        [60.0] * len(bands),
        measurement_area=10.0,
        area=10.0,
        frequencies=bands,
    )
    assert r.frequencies is not None
    assert r.frequencies.tolist() == list(bands)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"measurement_area": 0.0, "area": 10.0}, "measurement_area"),
        ({"measurement_area": 10.0, "area": -1.0}, "area"),
        ({"measurement_area": np.inf, "area": 10.0}, "measurement_area"),
    ],
)
def test_low_frequency_rejects_nonpositive_areas(
    kwargs: dict[str, float], match: str
) -> None:
    """Formula (7) takes 10 lg(Sm/S), which needs both areas positive."""
    with pytest.raises(ValueError, match=match):
        building.low_frequency_intensity_reduction([80.0], [60.0], **kwargs)


def test_low_frequency_band_count_mismatch_raises() -> None:
    """The two level arrays index the same bands."""
    with pytest.raises(ValueError, match="l_in"):
        building.low_frequency_intensity_reduction(
            [80.0, 81.0], [60.0], measurement_area=10.0, area=10.0
        )


def test_low_frequency_frequency_count_mismatch_raises() -> None:
    """So do the frequencies, when they are given."""
    with pytest.raises(ValueError, match="frequencies"):
        building.low_frequency_intensity_reduction(
            [80.0, 81.0],
            [60.0, 61.0],
            measurement_area=10.0,
            area=10.0,
            frequencies=[50.0],
        )


def test_low_frequency_result_rejects_arrays_that_disagree() -> None:
    """The result pins the shapes where they are built."""
    r = building.low_frequency_intensity_reduction(
        [80.0, 81.0],
        [60.0, 61.0],
        measurement_area=10.0,
        area=10.0,
        l_p=[70.0, 71.0],
    )
    with pytest.raises(ValueError, match="qualified"):
        dataclasses.replace(r, qualified=np.array([True]))


def test_low_frequency_result_rejects_half_an_indicator() -> None:
    """The indicator and the qualification are one answer, given together."""
    r = building.low_frequency_intensity_reduction(
        [80.0], [60.0], measurement_area=10.0, area=10.0, l_p=[70.0]
    )
    with pytest.raises(ValueError, match="two\\s+halves"):
        dataclasses.replace(r, qualified=None)


def test_low_frequency_l_p_band_count_mismatch_raises() -> None:
    """Formula (5) subtracts two arrays that index the same bands."""
    with pytest.raises(ValueError, match="l_p"):
        building.low_frequency_intensity_reduction(
            [80.0, 81.0],
            [60.0, 61.0],
            measurement_area=10.0,
            area=10.0,
            l_p=[70.0],
        )


def test_low_frequency_element_reproduces_formula_8() -> None:
    """DI,n,e = LpS - 9 - [LIn - 10 lg(A0/Sm) - 10 lg N], A0 = 10 m2."""
    r = building.low_frequency_element_normalized_difference(
        [90.0], [60.0], measurement_area=10.0
    )
    # Sm = A0 and N = 1, so both bracket terms vanish: 90 - 9 - 60.
    assert r.d_i_n_e[0] == pytest.approx(21.0)
    scaled = building.low_frequency_element_normalized_difference(
        [90.0], [60.0], measurement_area=20.0
    )
    assert scaled.d_i_n_e[0] == pytest.approx(21.0 - 10.0 * np.log10(2.0))


def test_low_frequency_element_adds_the_unit_count() -> None:
    """This part prints +10 lg N, the sign its part 1 sibling gets wrong."""
    single = building.low_frequency_element_normalized_difference(
        [90.0], [60.0], measurement_area=10.0
    )
    four = building.low_frequency_element_normalized_difference(
        [90.0], [60.0], measurement_area=10.0, elements=4
    )
    assert four.d_i_n_e[0] - single.d_i_n_e[0] == pytest.approx(10.0 * np.log10(4.0))


def test_low_frequency_element_is_three_decibels_below_part_one() -> None:
    """The 9 dB of the surface measurement against part 1's 6 dB."""
    part3 = building.low_frequency_element_normalized_difference(
        [90.0], [60.0], measurement_area=10.0
    )
    part1 = building.intensity_element_normalized_difference(
        [90.0], [60.0], measurement_area=10.0
    )
    assert part1.d_i_n_e[0] - part3.d_i_n_e[0] == pytest.approx(3.0)


def test_low_frequency_element_carries_the_clause_6_4_2_verdict() -> None:
    """The same qualification as the index, from the same Formula (5)."""
    r = building.low_frequency_element_normalized_difference(
        [90.0, 90.0],
        [60.0, 60.0],
        measurement_area=10.0,
        l_p=[68.0, 75.0],
        absorbing_specimen_surface=False,
    )
    assert r.indicator_limit == 10.0
    assert r.surface_pressure_intensity == pytest.approx([8.0, 15.0])
    assert r.qualified is not None
    assert r.qualified.tolist() == [True, False]
    bare = building.low_frequency_element_normalized_difference(
        [90.0], [60.0], measurement_area=10.0
    )
    assert bare.surface_pressure_intensity is None
    assert bare.qualified is None


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_low_frequency_element_rejects_a_bad_unit_count(bad: object) -> None:
    """N counts installed units, so it is a positive integer and not a bool."""
    with pytest.raises(ValueError, match="'elements' must be a positive integer"):
        building.low_frequency_element_normalized_difference(
            [90.0], [60.0], measurement_area=10.0, elements=bad
        )


def test_low_frequency_element_refuses_bands_outside_clause_6_6() -> None:
    """Clause 6.6 binds Formula (8) exactly as it binds Formula (7)."""
    with pytest.raises(ValueError, match="50 Hz to 160 Hz"):
        building.low_frequency_element_normalized_difference(
            [90.0], [60.0], measurement_area=10.0, frequencies=[250.0]
        )


def test_low_frequency_element_rejects_nonpositive_measurement_area() -> None:
    """Formula (8) takes 10 lg(A0/Sm)."""
    with pytest.raises(ValueError, match="measurement_area"):
        building.low_frequency_element_normalized_difference(
            [90.0], [60.0], measurement_area=0.0
        )


def test_low_frequency_element_result_rejects_arrays_that_disagree() -> None:
    """The result pins the shapes where they are built."""
    r = building.low_frequency_element_normalized_difference(
        [90.0, 91.0], [60.0, 61.0], measurement_area=10.0, l_p=[68.0, 69.0]
    )
    with pytest.raises(ValueError, match="qualified"):
        dataclasses.replace(r, qualified=np.array([True]))
    with pytest.raises(ValueError, match="two\\s+halves"):
        dataclasses.replace(r, qualified=None)


def test_low_frequency_accepts_the_exact_band_series() -> None:
    """The same band is written 63 Hz on a filter and 63,096 Hz exactly."""
    from phonometry.filters import nominal_frequencies

    exact = nominal_frequencies(fraction=3, limits=[50, 160])[0]
    assert len(exact) == len(ref.ISO15186_3_ANNEX_A_BANDS)
    nominal = building.low_frequency_intensity_reduction(
        [80.0] * 6,
        [60.0] * 6,
        measurement_area=10.0,
        area=10.0,
        frequencies=ref.ISO15186_3_ANNEX_A_BANDS,
    )
    series = building.low_frequency_intensity_reduction(
        [80.0] * 6, [60.0] * 6, measurement_area=10.0, area=10.0, frequencies=exact
    )
    assert series.r_i == pytest.approx(nominal.r_i)


@pytest.mark.parametrize(
    "call",
    [
        lambda grid: building.low_frequency_intensity_reduction(
            [80.0, 81.0],
            [60.0, 61.0],
            measurement_area=10.0,
            area=10.0,
            frequencies=grid,
        ),
        lambda grid: building.low_frequency_element_normalized_difference(
            [80.0, 81.0], [60.0, 61.0], measurement_area=10.0, frequencies=grid
        ),
        lambda grid: building.limp_panel_reduction_index(
            grid, surface_mass=10.0, area=10.0
        ),
    ],
)
def test_low_frequency_refuses_a_grid_of_frequencies(
    call: Callable[[np.ndarray], object],
) -> None:
    """Hertz are not levels: a (positions, bands) grid must not be averaged."""
    with pytest.raises(ValueError, match="one-dimensional"):
        call(np.array([[50.0, 63.0], [80.0, 100.0]]))


@pytest.mark.parametrize("bad", ["reflecting", 1, 0, None])
def test_low_frequency_refuses_a_non_boolean_clause_6_4_2_case(bad: object) -> None:
    """Every non-empty string is truthy and would pick the tighter limit."""
    with pytest.raises(ValueError, match="absorbing_specimen_surface"):
        building.low_frequency_intensity_reduction(
            [80.0],
            [60.0],
            measurement_area=10.0,
            area=10.0,
            l_p=[70.0],
            absorbing_specimen_surface=bad,
        )
    with pytest.raises(ValueError, match="absorbing_specimen_surface"):
        building.low_frequency_element_normalized_difference(
            [80.0], [60.0], measurement_area=10.0, absorbing_specimen_surface=bad
        )


def test_low_frequency_accepts_numpy_booleans_and_integers() -> None:
    """A numpy scalar is the ordinary way these arrive from a dataframe."""
    absorbing = building.low_frequency_intensity_reduction(
        [80.0],
        [60.0],
        measurement_area=10.0,
        area=10.0,
        l_p=[70.0],
        absorbing_specimen_surface=np.True_,
    )
    assert absorbing.indicator_limit == 6.0
    counted = building.low_frequency_element_normalized_difference(
        [90.0], [60.0], measurement_area=10.0, elements=np.int64(4)
    )
    assert counted.elements == 4
    # np.bool_ is not a bool subclass, so the unit-count guard has to name it.
    with pytest.raises(ValueError, match="'elements' must be a positive integer"):
        building.low_frequency_element_normalized_difference(
            [90.0], [60.0], measurement_area=10.0, elements=np.True_
        )


def test_low_frequency_element_rejects_a_non_finite_indicator() -> None:
    """The indicator is checked as closely as the index it travels with."""
    r = building.low_frequency_element_normalized_difference(
        [90.0], [60.0], measurement_area=10.0, l_p=[70.0]
    )
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(r, surface_pressure_intensity=np.array([np.nan]))


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf]])
def test_limp_panel_rejects_frequencies_it_cannot_evaluate(bad: list[float]) -> None:
    """Formula (A.2) takes the logarithm of every one of them."""
    with pytest.raises(ValueError, match="'frequencies' must be positive and finite"):
        building.limp_panel_reduction_index(bad, surface_mass=10.0, area=10.0)
