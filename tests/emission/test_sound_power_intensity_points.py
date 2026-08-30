#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power by sound intensity at discrete points: ISO 9614-1:1993.

Standard anchors:
- Partial power Pi = Ini*Si (Eq. 11); LW = 10*lg(sum Pi / P0), P0 = 1e-12 W
  (Eq. 12), the method not applicable to a band whose sum is negative (9.2).
- The signed level convention of clauses 3.5, 9.1 and A.2.3: a level printed
  "XX dB" is Ini = +I0*10^(XX/10), one printed "(-) XX dB" is -I0*10^(XX/10).
- Criterion 1, Ld > F2 (Eq. B.1), with Ld = dpI0 - K and K from Table 1;
  Figure B.1's unnumbered (F3 - F2) <= 3 dB gate; criterion 2, N > C*F4^2
  (Eq. B.2), with C from Table B.2.
- Tables B.1 (Delta), B.2 (C) and 2 (s) are exact printed tables and are the
  oracle here, transcribed in ``tests/reference_data/`` and parametrised cell
  by cell below, blanks included: grade 3 has no per-band column in any of the
  three, so a per-band grade-3 lookup must raise.
- Table B.3's five action codes, each reached through the case that triggers
  it in Figure B.1's order.
- Equation (B.3), the 95 % confidence interval 10*lg(1 +/- 2*F4/sqrt(N)).
- Equation (B.4) and the optional procedure of clause 8.3.2 / B.1.3.
"""

from __future__ import annotations

import math
import unittest.mock
import warnings

import numpy as np
import pytest
import reference_data as ref

from phonometry import emission
from phonometry.emission.sound_power_intensity_points import ActionCode

_P0 = 1.0e-12
_I0 = 1.0e-12

#: The grade names in the column order Tables 1, 2, B.1 and B.2 print them in:
#: precision (grade 1), peritaje (grade 2), control (grade 3).
_GRADES = ("precision", "engineering", "survey")

#: Nominal one-third-octave centres of ISO 9614-1 Tables B.2 and 2, in Hz.
_THIRD_BANDS = (
    50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
    5000.0, 6300.0,
)  # fmt: skip

#: Nominal octave centres of the same tables, in Hz.
_OCTAVE_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)


def _bands_of(span: tuple[float, float], band_type: str) -> list[float]:
    """The nominal centres a printed frequency range covers."""
    series = _OCTAVE_BANDS if band_type == "octave" else _THIRD_BANDS
    return [f for f in series if span[0] <= f <= span[1]]


def _table_cells(
    table: list[tuple[float | None, float | None, float | None]],
) -> list[tuple[str, str, float, float | None]]:
    """Every printed per-band cell as (band_type, grade, frequency, value)."""
    cells: list[tuple[str, str, float, float | None]] = []
    for row, values in zip(ref.ISO9614_1_BAND_ROWS, table, strict=True):
        for band_type, span in (("octave", row[0]), ("third", row[1])):
            if span is None:
                continue
            for frequency in _bands_of(span, band_type):
                for grade, value in zip(_GRADES, values, strict=True):
                    cells.append((band_type, grade, frequency, value))
    return cells


_C_CELLS = _table_cells(ref.ISO9614_1_TABLE_B2_C)
_S_CELLS = _table_cells(ref.ISO9614_1_TABLE_2_S)


def _uniform_surface(
    positions: int = 10, power: float = 1.0e-3, area: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """A surface of equal segments carrying one known sound power."""
    areas = np.full(positions, area)
    intensity = np.full((positions, 1), power / float(areas.sum()))
    return intensity, areas


def _qualified_case(
    intensity: np.ndarray,
    *,
    residual_index: float = 30.0,
    levels: float = 80.0,
    temporal: np.ndarray | None = None,
    grade: str = "precision",
) -> emission.DiscretePointIntensityResult:
    """One-band determination over unit segments, qualified at ``grade``."""
    areas = np.full(intensity.shape[0], 1.0)
    return emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full(intensity.shape[0], levels),
        pressure_residual_index=residual_index,
        temporal_intensity=temporal,
        frequencies=[1000.0],
        grade=grade,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Table B.2 (C) and Table 2 (s), cell by cell, and the grade-3 asymmetry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("band_type", "grade", "frequency", "expected"), _C_CELLS)
def test_table_b2_reproduces_every_printed_cell(
    band_type: str, grade: str, frequency: float, expected: float | None
) -> None:
    """Every per-band cell of Table B.2, blanks included (Eq. B.2)."""
    if expected is None:
        with pytest.raises(ValueError, match="no per-band value"):
            emission.position_count_factor(grade, frequency, band_type=band_type)  # type: ignore[arg-type]
        return
    factor = emission.position_count_factor(grade, frequency, band_type=band_type)  # type: ignore[arg-type]
    assert factor == pytest.approx(expected)


@pytest.mark.parametrize(("band_type", "grade", "frequency", "expected"), _S_CELLS)
def test_table_2_reproduces_every_printed_cell(
    band_type: str, grade: str, frequency: float, expected: float | None
) -> None:
    """Every per-band cell of Table 2, blanks included (footnote 1: +/- 2s)."""
    if expected is None:
        with pytest.raises(ValueError, match="no per-band value"):
            emission.determination_standard_deviation(
                grade,  # type: ignore[arg-type]
                frequency,
                band_type=band_type,
            )
        return
    s = emission.determination_standard_deviation(grade, frequency, band_type=band_type)  # type: ignore[arg-type]
    assert s == pytest.approx(expected)


@pytest.mark.parametrize(
    ("grade", "expected"),
    list(zip(_GRADES, ref.ISO9614_1_TABLE_B2_C_A_WEIGHTED, strict=True)),
)
def test_table_b2_a_weighted_row(grade: str, expected: float | None) -> None:
    """The A-weighted row of Table B.2 holds the survey grade and nothing else."""
    if expected is None:
        with pytest.raises(ValueError, match="A-weighted C only for the survey"):
            emission.position_count_factor(grade)  # type: ignore[arg-type]
        return
    assert emission.position_count_factor(grade) == pytest.approx(expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("grade", "expected"),
    list(zip(_GRADES, ref.ISO9614_1_TABLE_2_S_A_WEIGHTED, strict=True)),
)
def test_table_2_a_weighted_row(grade: str, expected: float | None) -> None:
    """The A-weighted row of Table 2 likewise: grade 3 only, tentative (note 3)."""
    if expected is None:
        with pytest.raises(ValueError, match="A-weighted standard deviation"):
            emission.determination_standard_deviation(grade)  # type: ignore[arg-type]
        return
    value = emission.determination_standard_deviation(grade)  # type: ignore[arg-type]
    assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    ("grade", "expected"),
    list(zip(_GRADES, ref.ISO9614_1_TABLE_B1_ALL_BANDS, strict=True)),
)
def test_table_b1_all_bands_row(grade: str, expected: float | None) -> None:
    """The "all bands" row of Table B.1: 0,20 at grade 1, 0,29 at grade 2."""
    if expected is None:
        with pytest.raises(ValueError, match="leaves the 'all bands' cell"):
            emission.error_factor(grade)  # type: ignore[arg-type]
        return
    assert emission.error_factor(grade) == pytest.approx(expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("grade", "expected"),
    list(zip(_GRADES, ref.ISO9614_1_TABLE_B1_A_WEIGHTED, strict=True)),
)
def test_table_b1_a_weighted_row(grade: str, expected: float | None) -> None:
    """The A-weighted row of Table B.1: 0,60 at grade 3 and blank elsewhere."""
    if expected is None:
        with pytest.raises(ValueError, match="leaves the 'A-weighted' cell"):
            emission.error_factor(grade, a_weighted=True)  # type: ignore[arg-type]
        return
    value = emission.error_factor(grade, a_weighted=True)  # type: ignore[arg-type]
    assert value == pytest.approx(expected)


def test_grade_3_has_no_per_band_column_in_any_of_the_three_tables() -> None:
    """The asymmetry stated once: grade 3 is an A-weighted determination.

    Table B.2 (C), Table B.1 (Delta) and Table 2 (s) all tabulate grades 1 and
    2 band by band and grade 3 A-weighted only, so a per-band grade-3 figure
    exists in none of them and must not be invented from the A-weighted one.
    """
    for lookup in (
        lambda: emission.position_count_factor("survey", 1000.0),
        lambda: emission.determination_standard_deviation("survey", 1000.0),
        lambda: emission.error_factor("survey"),
    ):
        with pytest.raises(ValueError, match="survey"):
            lookup()


def test_a_weighted_lookups_are_closed_to_grades_1_and_2() -> None:
    """The other half of the asymmetry: no A-weighted C, s or Delta below grade 3."""
    for lookup in (
        lambda: emission.position_count_factor("precision"),
        lambda: emission.determination_standard_deviation("engineering"),
        lambda: emission.error_factor("precision", a_weighted=True),
    ):
        with pytest.raises(ValueError, match="A-weighted"):
            lookup()


def test_table_1_bias_error_factor_drives_the_dynamic_capability() -> None:
    """Table 1: K = 10 dB at grades 1 and 2, 7 dB at grade 3, so Ld differs."""
    intensity, areas = _uniform_surface()
    levels = np.full((10, 1), 80.0)
    for grade, k in ref.ISO9614_1_TABLE_1_K.items():
        result = emission.sound_power_intensity_points(
            intensity,
            areas,
            pressure_levels=levels,
            pressure_residual_index=18.0,
            grade=grade,  # type: ignore[arg-type]
        )
        assert result.dynamic_capability_index is not None
        assert result.dynamic_capability_index[0] == pytest.approx(18.0 - k)


@pytest.mark.parametrize("band_type", ["octave", "third"])
def test_untabulated_band_is_refused(band_type: str) -> None:
    """A band outside Tables B.2 and 2 is refused rather than extrapolated."""
    with pytest.raises(ValueError, match="not a band of ISO 9614-1"):
        emission.position_count_factor(
            "engineering",
            8000.0,
            band_type=band_type,  # type: ignore[arg-type]
        )


def test_6300_hz_row_has_no_octave_counterpart() -> None:
    """The 6 300 Hz row of Tables B.2 and 2 is printed for thirds only."""
    assert emission.position_count_factor(
        "precision", 6300.0, band_type="third"
    ) == pytest.approx(19.0)
    with pytest.raises(ValueError, match="not a band of ISO 9614-1"):
        emission.position_count_factor("precision", 6300.0, band_type="octave")


def test_exact_base_ten_centre_designates_its_nominal_label() -> None:
    """1000*10**(0.1) Hz is the 1 250 Hz band, as in the IEC 61043 lookups."""
    exact = 1000.0 * 10.0 ** (1.0 / 10.0)
    assert emission.position_count_factor(
        "engineering", exact, band_type="third"
    ) == pytest.approx(29.0)


def test_a_frequency_five_percent_off_a_centre_matches_no_band() -> None:
    """The tolerance recognises a rounded centre, not a nearby measurement.

    The lookups accept 1000*10**(1/10) Hz as the 1 250 Hz band because that is
    the exact centre its nominal label rounds, so the match runs to a few per
    cent. Widen it and a frequency nobody tabulated starts returning a
    plausible C: 1 050 Hz is 5 % above the 1 000 Hz third and belongs to no row
    of Tables B.2 and 2.
    """
    with pytest.raises(ValueError, match="not a band of ISO 9614-1"):
        emission.position_count_factor("engineering", 1050.0, band_type="third")


def test_a_frequency_of_zero_is_refused_by_name() -> None:
    """Zero is where the nominal-band arithmetic takes a logarithm of nothing.

    The band a frequency belongs to is found by comparing logarithms, so the
    guard has to stop zero and every negative frequency before that, and say
    which input was wrong instead of surfacing a domain error from inside.
    """
    with pytest.raises(ValueError, match="finite and positive"):
        emission.position_count_factor("engineering", 0.0, band_type="third")
    with pytest.raises(ValueError, match="finite and positive"):
        emission.position_count_factor("engineering", -1000.0, band_type="third")


def test_a_frequency_that_is_not_a_number_is_refused_by_name() -> None:
    """NaN passes a positivity test unaided, so finiteness is checked beside it.

    NaN compares False against every bound, so a guard written on the sign
    alone would let it through to a lookup that would then match no row and
    blame the band table for an input that was never a frequency.
    """
    with pytest.raises(ValueError, match="finite and positive"):
        emission.determination_standard_deviation(
            "precision", float("nan"), band_type="third"
        )


def test_unknown_grade_and_band_type_are_refused() -> None:
    """The vocabulary of grades and band types is closed."""
    with pytest.raises(ValueError, match="'grade' must be"):
        emission.position_count_factor("class 1", 1000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'band_type' must be"):
        emission.position_count_factor("engineering", 1000.0, band_type="decade")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The signed-intensity convention (clauses 3.5, 9.1 and A.2.3)
# ---------------------------------------------------------------------------
def test_printed_level_without_the_minus_sign_is_outward_flow() -> None:
    """ "XX dB" is Ini = +I0*10^(XX/10)."""
    assert emission.normal_intensity_from_levels(70.0) == pytest.approx(1.0e-5)


def test_printed_level_with_the_minus_sign_is_inward_flow() -> None:
    """ "(-) XX dB" is Ini = -I0*10^(XX/10), the same magnitude, flowing in."""
    assert emission.normal_intensity_from_levels(70.0, negative=True) == pytest.approx(
        -1.0e-5
    )


def test_the_minus_sign_is_per_position() -> None:
    """One position of a surface may flow inward while the rest flow outward."""
    levels = [70.0, 70.0, 60.0, 70.0]
    intensity = emission.normal_intensity_from_levels(
        levels, negative=[False, False, True, False]
    )
    np.testing.assert_allclose(intensity, [1.0e-5, 1.0e-5, -1.0e-6, 1.0e-5])


def test_levels_convert_band_by_band_over_a_grid() -> None:
    """A (positions, bands) grid of levels keeps its shape and its signs."""
    levels = np.array([[70.0, 60.0], [65.0, 55.0]])
    negative = np.array([[False, True], [False, False]])
    intensity = emission.normal_intensity_from_levels(levels, negative=negative)
    expected = _I0 * 10.0 ** (levels / 10.0) * np.where(negative, -1.0, 1.0)
    np.testing.assert_allclose(intensity, expected)


def test_level_conversion_refuses_a_non_finite_level() -> None:
    """A NaN level would turn a whole partial power into a silent NaN."""
    with pytest.raises(ValueError, match="'levels' must contain only finite"):
        emission.normal_intensity_from_levels([70.0, math.nan])


def test_level_conversion_refuses_a_sign_mask_of_another_shape() -> None:
    """A sign per position cannot be handed to a surface of another size."""
    with pytest.raises(ValueError, match="does not broadcast"):
        emission.normal_intensity_from_levels([70.0, 65.0], negative=[True] * 3)


def test_the_sign_mask_never_widens_the_result() -> None:
    """The result has the shape of the levels, not of the two broadcast together.

    ``negative`` carries the "(-)" of one printed level, so it is broadcast
    onto the levels and never the other way round: three levels under a (2, 1)
    mask are not six intensities, they are a mask that does not fit.
    """
    levels = np.array([70.0, 66.0, 63.0])
    values = emission.normal_intensity_from_levels(
        levels, negative=np.array([True, False, True])
    )
    assert values.shape == levels.shape
    with pytest.raises(ValueError, match="does not broadcast"):
        emission.normal_intensity_from_levels(
            levels, negative=np.array([[True], [False]])
        )


def test_a_genuinely_negative_partial_power_is_kept_and_summed() -> None:
    """Negative partial power is normal; only the sum going negative is fatal."""
    areas = np.full(10, 1.0)
    intensity = emission.normal_intensity_from_levels(
        [70.0] * 9 + [66.0], negative=[False] * 9 + [True]
    )
    result = emission.sound_power_intensity_points(intensity, areas)
    assert result.partial_power[-1, 0] < 0.0
    expected = 9.0 * 1.0e-5 - _I0 * 10.0 ** (66.0 / 10.0)
    assert result.sound_power[0] == pytest.approx(expected)
    assert not bool(result.not_applicable_band[0])
    assert result.sound_power_level[0] == pytest.approx(
        10.0 * math.log10(expected / _P0)
    )


# ---------------------------------------------------------------------------
# The power sum (Eqs. 11 and 12) and clause 9.2
# ---------------------------------------------------------------------------
def test_uniform_enclosing_surface_recovers_the_source_power() -> None:
    """Ini = W/S over equal segments gives LW = 10*lg(W/P0) exactly (Eq. 12)."""
    intensity, areas = _uniform_surface(power=1.0e-3)
    result = emission.sound_power_intensity_points(intensity, areas)
    assert isinstance(result, emission.DiscretePointIntensityResult)
    assert result.sound_power[0] == pytest.approx(1.0e-3)
    assert result.sound_power_level[0] == pytest.approx(90.0)
    assert result.surface_area == pytest.approx(10.0)
    assert result.positions == 10


def test_unequal_segments_weight_by_their_own_area() -> None:
    """Each position stands for its segment, so Pi = Ini*Si (Eq. 11)."""
    areas = np.array([0.2, 0.3, 0.5, 1.0, 2.0, 1.0, 0.5, 0.3, 0.2, 4.0])
    partial = np.array([1e-4, 1e-4, 1.5e-4, 1.5e-4, 2e-5, 3e-5, 1e-5, 1e-5, 1e-5, 2e-5])
    intensity = partial / areas
    result = emission.sound_power_intensity_points(intensity, areas)
    np.testing.assert_allclose(result.partial_power[:, 0], partial)
    assert result.sound_power[0] == pytest.approx(partial.sum())


def test_bands_are_determined_independently() -> None:
    """Two bands of one surface do not mix (Eq. 12 is per band)."""
    areas = np.full(10, 1.0)
    intensity = np.column_stack([np.full(10, 1.0e-4), np.full(10, 1.0e-5)])
    result = emission.sound_power_intensity_points(intensity, areas)
    np.testing.assert_allclose(result.sound_power, [1.0e-3, 1.0e-4])
    np.testing.assert_allclose(result.sound_power_level, [90.0, 80.0])


def test_net_negative_band_is_outside_the_method() -> None:
    """Clause 9.2: a band whose sum of partial powers is negative is dropped."""
    areas = np.full(10, 1.0)
    intensity = np.column_stack([np.full(10, 1.0e-4), np.full(10, -1.0e-5)])
    with pytest.warns(emission.SoundPowerWarning, match="clause 9.2"):
        result = emission.sound_power_intensity_points(intensity, areas)
    np.testing.assert_array_equal(result.not_applicable_band, [False, True])
    assert math.isnan(float(result.sound_power_level[1]))
    assert result.f4 is not None
    assert math.isnan(float(result.f4[1]))


def test_a_band_of_exactly_zero_net_power_is_outside_the_method_too() -> None:
    """A net flow of zero is no determinable power either, and is not -inf dB."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 1), 1.0e-5)
    intensity[:5, 0] = -1.0e-5
    with pytest.warns(emission.SoundPowerWarning, match="clause 9.2"):
        result = emission.sound_power_intensity_points(intensity, areas)
    assert bool(result.not_applicable_band[0])
    assert math.isnan(float(result.sound_power_level[0]))


# ---------------------------------------------------------------------------
# The ISO 9614-2 round trip
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:The A-weighted total sums every")
def test_discrete_points_and_a_scan_of_the_same_surface_agree() -> None:
    """The same tiling, the same intensities, the same LW.

    ISO 9614-2 sums <In,i>*Si over scanned segments and ISO 9614-1 sums Ini*Si
    over the segments discrete positions stand for. Where the two describe the
    same surface with the same intensities the determinations are the same
    arithmetic, and the two implementations must agree band for band.
    """
    areas = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 0.5, 0.75, 1.0, 1.25, 1.5])
    rng = np.random.default_rng(9614)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.2, (areas.size, 4)))
    frequencies = np.array([250.0, 500.0, 1000.0, 2000.0])

    points = emission.sound_power_intensity_points(
        intensity, areas, frequencies=frequencies, band_type="octave"
    )
    scan = emission.sound_power_intensity(
        intensity, areas, frequencies=frequencies, band_type="octave"
    )
    np.testing.assert_allclose(points.sound_power, scan.sound_power)
    np.testing.assert_allclose(points.sound_power_level, scan.sound_power_level)
    np.testing.assert_allclose(points.partial_power, scan.partial_power)
    assert points.surface_area == pytest.approx(scan.surface_area)
    assert points.sound_power_level_a == pytest.approx(scan.sound_power_level_a)


# ---------------------------------------------------------------------------
# Annex A indicators as this module reads them
# ---------------------------------------------------------------------------
def test_indicators_match_the_annex_a_entry_point() -> None:
    """F2, F3 and F4 are the library's own Annex A indicators, band by band."""
    areas = np.full(10, 1.0)
    rng = np.random.default_rng(4)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.25, (10, 2)))
    levels = 80.0 + rng.normal(0.0, 0.5, (10, 2))
    result = emission.sound_power_intensity_points(
        intensity, areas, pressure_levels=levels
    )
    expected = emission.field_indicators(levels, intensity)
    assert result.f2 is not None
    assert result.f3 is not None
    assert result.f4 is not None
    np.testing.assert_allclose(result.f2, expected.f2)
    np.testing.assert_allclose(result.f3, expected.f3)
    np.testing.assert_allclose(result.f4, expected.f4)


def test_f4_is_available_without_the_pressure_levels() -> None:
    """Criterion 2 needs only F4, which needs only the intensities (A.8)."""
    areas = np.full(10, 1.0)
    rng = np.random.default_rng(5)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.25, (10, 1)))
    result = emission.sound_power_intensity_points(intensity, areas)
    assert result.f2 is None
    assert result.f3 is None
    assert result.f4 is not None
    reference = emission.field_indicators(np.full((10, 1), 80.0), intensity)
    np.testing.assert_allclose(result.f4, reference.f4)


def _unequal_segments_with_a_negative_mean() -> tuple[np.ndarray, np.ndarray]:
    """Ten segments whose first band is positive by power and negative by mean.

    One large segment flowing outward against nine small ones flowing inward:
    the area-weighted sum of clause 9.2 is positive, so the band is inside the
    method, while the unweighted mean over the positions that A.2.3 conditions
    the indicators on is not. The second band is ordinary, and is there to be
    watched for collateral damage.
    """
    areas = np.array([5.0] + [0.5] * 9)
    inward = np.array([1.0e-4] + [-5.0e-5] * 9)
    return np.column_stack([inward, np.full(10, 1.0e-5)]), areas


#: The opening words of each of the two per-band refusals. A.2.3's message
#: names clause 9.2 in order to say it is not that one, so telling the two
#: apart in a test takes the opening and not a search for the clause number.
_CLAUSE_9_2_WARNING = "The total sound power is not positive"
_A_2_3_WARNING = "The algebraic mean normal intensity is not positive"


def test_a_band_of_negative_mean_intensity_goes_nan_on_its_own() -> None:
    """A.2.3 refuses "en esa banda de frecuencia", not in the determination."""
    intensity, areas = _unequal_segments_with_a_negative_mean()
    assert float(np.sum(intensity[:, 0] * areas)) > 0.0
    assert float(np.mean(intensity[:, 0])) < 0.0

    with pytest.warns(emission.SoundPowerWarning, match="A.2.3"):
        result = emission.sound_power_intensity_points(intensity, areas)
    assert result.f4 is not None
    assert math.isnan(float(result.f4[0]))
    assert float(result.f4[1]) == pytest.approx(0.0)
    np.testing.assert_array_equal(result.not_applicable_band, [False, False])
    assert np.all(np.isfinite(result.sound_power_level))


def test_the_pressure_indicators_go_nan_in_that_band_too() -> None:
    """F2 and F3 divide by the same mean, and take the same route out."""
    intensity, areas = _unequal_segments_with_a_negative_mean()
    with pytest.warns(emission.SoundPowerWarning, match="A.2.3"):
        result = emission.sound_power_intensity_points(
            intensity, areas, pressure_levels=np.full((10, 2), 80.0)
        )
    assert result.f2 is not None
    assert result.f3 is not None
    assert result.f4 is not None
    for indicator in (result.f2, result.f3, result.f4):
        assert math.isnan(float(indicator[0]))
        assert math.isfinite(float(indicator[1]))


def test_the_band_a_2_3_refuses_is_announced_where_clause_9_2_keeps_it() -> None:
    """The stronger refusal must not be the quieter one.

    Clause 9.2 says only that the method "no es aplicable a esta banda" and
    warns when it does. A.2.3 says the test conditions do not satisfy the
    requirements of ISO 9614-1 in that band, which is more than that, and here
    it is the only one of the two that refuses: the level stays finite, the
    band stays applicable, and the criteria were never asked for. Without a
    warning nothing in the call says the band failed at all.
    """
    intensity, areas = _unequal_segments_with_a_negative_mean()
    with pytest.warns(emission.SoundPowerWarning, match="A.2.3") as caught:
        result = emission.sound_power_intensity_points(intensity, areas)
    assert not any(str(w.message).startswith(_CLAUSE_9_2_WARNING) for w in caught)
    np.testing.assert_array_equal(result.not_applicable_band, [False, False])
    assert np.all(np.isfinite(result.sound_power_level))


def test_a_band_both_clauses_refuse_is_announced_by_both() -> None:
    """Equal segments make the two quantities agree, and both statements true.

    The unweighted mean of A.2.3 and the area-weighted sum of clause 9.2 share
    a sign whenever the segments share an area, so the ordinary inward-flowing
    band fails both. Each clause says its own thing about it: one that the
    method does not apply to the band, the other that the test conditions were
    not met in it.
    """
    areas = np.full(10, 1.0)
    intensity = np.column_stack([np.full(10, 1.0e-4), np.full(10, -1.0e-5)])
    with pytest.warns(emission.SoundPowerWarning) as caught:
        emission.sound_power_intensity_points(intensity, areas)
    messages = [str(w.message) for w in caught]
    assert any(message.startswith(_CLAUSE_9_2_WARNING) for message in messages)
    assert any(message.startswith(_A_2_3_WARNING) for message in messages)


def test_the_a_2_3_guard_is_on_the_arithmetic_mean_and_not_the_median() -> None:
    """A.2.3 sums the intensities, and a sum is not a middle value.

    "Si sum Ini/I0 es negativo": the sign of a sum over N positions is the sign
    of their arithmetic mean and of nothing else. A field where one position
    carries almost all the outward flow has a negative median and a positive
    mean, and it is exactly the field the concentration procedure of clause
    8.3.2 exists for, so refusing it would refuse the standard's own case.
    """
    intensity = np.array([[9.0e-5]] + [[-5.0e-6]] * 9)
    assert float(np.mean(intensity)) > 0.0 > float(np.median(intensity))
    result = emission.sound_power_intensity_points(intensity, np.full(10, 1.0))
    assert result.f4 is not None
    assert math.isfinite(float(result.f4[0]))


def test_a_determination_that_meets_a_2_3_everywhere_stays_quiet() -> None:
    """The announcement is of a failure, so a sound surface must not draw one."""
    intensity, areas = _uniform_surface()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emission.sound_power_intensity_points(intensity, areas)


def test_f1_comes_from_the_initial_test_and_is_optional() -> None:
    """F1 is the coefficient of variation of the M short-time samples (A.1)."""
    intensity, areas = _uniform_surface()
    samples = [1.2e-5, 0.9e-5, 1.5e-5, 1.1e-5, 1.3e-5, 1.0e-5]
    result = emission.sound_power_intensity_points(
        intensity, areas, temporal_intensity=samples
    )
    assert result.f1 is not None
    assert result.f1[0] == pytest.approx(
        emission.temporal_variability_indicator(samples)
    )
    bare = emission.sound_power_intensity_points(intensity, areas)
    assert bare.f1 is None


def test_short_time_samples_must_span_the_same_bands() -> None:
    """A one-band F1 cannot qualify a two-band determination."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 2), 1.0e-5)
    samples = np.full((6, 1), 1.0e-5)
    with pytest.raises(ValueError, match="one column per band"):
        emission.sound_power_intensity_points(
            intensity, areas, temporal_intensity=samples
        )


# ---------------------------------------------------------------------------
# Criterion 1 (Eq. B.1) and criterion 2 (Eq. B.2)
# ---------------------------------------------------------------------------
def test_criterion_1_is_ld_strictly_greater_than_f2() -> None:
    """Eq. (B.1): the instrument is adequate when Ld > F2, not when it ties."""
    areas = np.full(10, 1.0)
    rng = np.random.default_rng(6)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.1, (10, 1)))
    levels = np.full((10, 1), 80.0)
    probe = emission.sound_power_intensity_points(
        intensity, areas, pressure_levels=levels, pressure_residual_index=40.0
    )
    assert probe.f2 is not None
    f2 = float(probe.f2[0])
    at_limit = emission.sound_power_intensity_points(
        intensity, areas, pressure_levels=levels, pressure_residual_index=f2 + 10.0
    )
    assert at_limit.criterion_1 is not None
    assert not bool(at_limit.criterion_1[0])


def test_criterion_2_compares_the_position_count_with_c_times_f4_squared() -> None:
    """Eq. (B.2): N > C*F4^2, with C read from Table B.2 at the band centre."""
    areas = np.full(12, 1.0)
    rng = np.random.default_rng(7)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.3, (12, 1)))
    result = emission.sound_power_intensity_points(
        intensity, areas, frequencies=[1000.0], grade="engineering"
    )
    assert result.f4 is not None
    assert result.minimum_positions is not None
    assert result.minimum_positions[0] == pytest.approx(29.0 * result.f4[0] ** 2)
    assert result.criterion_2 is not None
    assert bool(result.criterion_2[0]) is bool(12 > result.minimum_positions[0])


def test_criterion_2_needs_the_band_centres() -> None:
    """Without a frequency there is no Table B.2 row, so no criterion 2."""
    intensity, areas = _uniform_surface()
    result = emission.sound_power_intensity_points(intensity, areas)
    assert result.criterion_2 is None
    assert result.minimum_positions is None


def test_criterion_2_has_no_per_band_form_at_the_survey_grade() -> None:
    """Table B.2 gives grade 3 no per-band C, so the criterion is not evaluated."""
    intensity, areas = _uniform_surface()
    result = emission.sound_power_intensity_points(
        intensity, areas, frequencies=[1000.0], grade="survey"
    )
    assert result.criterion_2 is None
    assert result.expanded_uncertainty is None


def test_grade_3_never_appears_in_the_per_band_verdict() -> None:
    """A band reaches grade 1 or grade 2 or nothing; grade 3 is A-weighted."""
    areas = np.full(30, 1.0)
    rng = np.random.default_rng(8)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.5, (30, 1)))
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full((30, 1), 80.0),
        pressure_residual_index=40.0,
        frequencies=[1000.0],
        grade="survey",
    )
    assert result.achieved_grade is not None
    assert set(result.achieved_grade) <= {"precision", "engineering", "none"}


def test_a_uniform_field_over_many_positions_reaches_the_precision_grade() -> None:
    """F4 near zero, a wide dynamic margin: criterion 2 is met at grade 1."""
    intensity, areas = _uniform_surface(positions=20)
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full((20, 1), 80.0),
        pressure_residual_index=40.0,
        frequencies=[1000.0],
        grade="precision",
    )
    assert result.achieved_grade is not None
    assert result.achieved_grade[0] == "precision"


def test_too_much_inward_flow_leaves_the_band_ungraded() -> None:
    """Figure B.1's third gate stops the grade, not only the action code.

    Forty positions, thirty flowing outward at 0,94 and ten inward at 1,0
    (arbitrary units), which puts F3 - F2 at 3,2 dB, just past the gate, while
    holding F4 low enough that criterion 2 is comfortably satisfied at
    grade 2: 11 x F4^2 is 38,5 against the forty positions measured. So the
    only thing standing between this band and a grade is the inward flow, and
    the verdict has to be "none" all the same.
    """
    areas = np.full(40, 1.0)
    intensity = np.concatenate([np.full(30, 0.9394e-5), np.full(10, -1.0e-5)])
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full(40, 80.0),
        pressure_residual_index=40.0,
        frequencies=[125.0],
        band_type="octave",
        grade="precision",
    )
    assert result.f2 is not None
    assert result.f3 is not None
    assert result.f4 is not None
    assert result.achieved_grade is not None
    assert float(result.f3[0] - result.f2[0]) > 3.0
    assert result.criterion_1 is not None
    assert bool(result.criterion_1[0])
    assert result.positions > 11.0 * float(result.f4[0]) ** 2
    assert result.achieved_grade[0] == "none"


def _a_little_inward_flow() -> emission.DiscretePointIntensityResult:
    """Thirty-six segments outward and four inward: F2 = 10, F3 = 10,97 dB.

    Ld is put at 10,5 dB, between the two, which is the only arrangement that
    can tell equation (B.1) apart from the same comparison written on F3. A
    field with no inward flow at all has F2 and F3 equal, so it says nothing
    about which of the two indicators criterion 1 reads.
    """
    pattern = np.array([1.0] * 36 + [-1.0] * 4)
    return _qualified_case(1.0e-5 * pattern, residual_index=20.5)


def test_criterion_1_is_read_against_f2_and_not_against_f3() -> None:
    """Equation (B.1) is Ld > F2, the unsigned indicator of equation (A.5)."""
    result = _a_little_inward_flow()
    assert result.f2 is not None
    assert result.f3 is not None
    assert float(result.f2[0]) < 10.5 < float(result.f3[0])
    assert result.criterion_1 is not None
    assert bool(result.criterion_1[0])


def test_the_per_band_grade_reads_the_same_f2() -> None:
    """The grade walks Figure B.1's own gates, so it reads (B.1) as (B.1) is."""
    result = _a_little_inward_flow()
    assert result.achieved_grade is not None
    assert list(result.achieved_grade) == ["precision"]


def test_the_a_weighted_grade_reads_the_same_f2() -> None:
    """And so does the A-weighted determination of B.1.2, gate for gate."""
    assert _a_little_inward_flow().achieved_grade_a == "precision"


def test_the_per_band_grade_reads_ld_at_the_grade_1_and_2_bias_factor() -> None:
    """A per-band grade may never borrow the survey grade's looser K.

    Table 1 allows grade 3 a K of 7 dB against the 10 dB of grades 1 and 2, so
    the same delta_pI0 buys 3 dB more dynamic capability there. Here
    delta_pI0 = 18,5 dB puts Ld at 8,5 dB with K = 10 and at 11,5 dB with
    K = 7, against an F2 of 10 dB: read at the survey K the band would clear
    criterion 1 and be graded, which Table B.2 gives no per-band column to do.
    """
    result = _qualified_case(np.full(20, 1.0e-5), residual_index=18.5)
    assert result.f2 is not None
    assert float(result.f2[0]) == pytest.approx(10.0, abs=1e-9)
    assert result.achieved_grade is not None
    assert list(result.achieved_grade) == ["none"]


def test_the_per_band_grade_needs_ld_strictly_above_f2() -> None:
    """Equation (B.1) prints ">", and a tie is not a margin.

    Built the way the criterion-1 test builds one, by measuring F2 first and
    handing back a residual index of exactly F2 + K, so that Ld lands on F2
    without any float being compared for equality in the module.
    """
    rng = np.random.default_rng(6)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.1, (20, 1)))
    probe = _qualified_case(intensity, residual_index=40.0)
    assert probe.f2 is not None
    f2 = float(probe.f2[0])
    at_limit = _qualified_case(intensity, residual_index=f2 + 10.0)
    assert at_limit.dynamic_capability_index is not None
    assert float(at_limit.dynamic_capability_index[0]) == pytest.approx(f2)
    assert at_limit.achieved_grade is not None
    assert list(at_limit.achieved_grade) == ["none"]


def test_an_unsteady_field_reaches_no_grade_either() -> None:
    """Figure B.1's first gate is a gate on the grade and not only on an action.

    F1 > 0,6 is the initial test failing, and Table B.3 sends it to action (e)
    before any other question is asked. A band whose field would otherwise
    qualify at grade 1 must therefore still come back ungraded.
    """
    wandering = np.array([1.0e-5, 5.0e-5, 2.0e-6, 9.0e-5, 1.0e-6])
    result = _qualified_case(
        np.full(20, 1.0e-5), residual_index=40.0, temporal=wandering
    )
    assert result.f1 is not None
    assert float(result.f1[0]) > emission.TEMPORAL_VARIABILITY_LIMIT
    assert result.achieved_grade is not None
    assert list(result.achieved_grade) == ["none"]


def test_a_band_outside_the_method_reaches_no_grade() -> None:
    """Clause 9.2's refusal is the mirror of A.2.3's, and it gates the grade.

    One large segment flowing inward against many small ones flowing outward:
    the unweighted mean of A.2.3 is positive, so the indicators all exist and
    criterion 2 is satisfied, while the area-weighted sum of clause 9.2 is
    negative and the method does not apply to the band. There is no level to
    put a grade on, so the verdict must be "none" however good the indicators.
    """
    n = 200
    areas = np.array([42.0] + [1.0] * (n - 1))
    intensity = np.array([-5.0e-6] + [1.0e-6] * (n - 1)).reshape(-1, 1)
    with pytest.warns(emission.SoundPowerWarning, match="clause 9.2"):
        result = emission.sound_power_intensity_points(
            intensity,
            areas,
            pressure_levels=np.full((n, 1), 80.0),
            pressure_residual_index=40.0,
            frequencies=[1000.0],
            grade="precision",
        )
    assert result.criterion_2 is not None
    assert bool(result.criterion_2[0])
    assert bool(result.not_applicable_band[0])
    assert result.f4 is not None
    assert math.isfinite(float(result.f4[0]))
    assert result.achieved_grade is not None
    assert list(result.achieved_grade) == ["none"]


# ---------------------------------------------------------------------------
# Table B.3 action codes, one case per code (Figure B.1's order)
# ---------------------------------------------------------------------------
def test_action_e_when_the_field_is_not_stationary() -> None:
    """Table B.3 row 1: F1 > 0,6 asks for action (e) before anything else."""
    wandering = np.array([1.0e-5, 5.0e-5, 2.0e-6, 9.0e-5, 1.0e-6])
    result = _qualified_case(np.full(10, 1.0e-5), temporal=wandering)
    assert result.f1 is not None
    assert float(result.f1[0]) > emission.TEMPORAL_VARIABILITY_LIMIT
    assert result.required_actions()[0] == (ActionCode.REDUCE_TEMPORAL_VARIABILITY,)


def test_actions_a_or_b_when_criterion_1_fails() -> None:
    """Table B.3 row 2: F2 above Ld offers the choice of (a) or (b)."""
    result = _qualified_case(np.full(10, 1.0e-5), residual_index=10.0)
    assert result.criterion_1 is not None
    assert not bool(result.criterion_1[0])
    assert result.required_actions()[0] == (
        ActionCode.ADJUST_MEASUREMENT_DISTANCE,
        ActionCode.SHIELD_OR_REDUCE_REFLECTIONS,
    )


def test_actions_a_or_b_when_too_much_power_flows_inward() -> None:
    """Figure B.1's third gate: (F3 - F2) above 3 dB, same two actions."""
    intensity = np.full(10, 1.0e-5)
    # (F3 - F2) = 10*lg[(9p + x)/(9p - x)] with one inward position of -x, so
    # x = 3.2p puts the excess at about 3,2 dB, just past the Figure B.1 gate.
    intensity[0] = -3.2e-5
    result = _qualified_case(intensity)
    assert result.f2 is not None
    assert result.f3 is not None
    assert float(result.f3[0] - result.f2[0]) > 3.0
    assert result.criterion_1 is not None
    assert bool(result.criterion_1[0])
    assert result.required_actions()[0] == (
        ActionCode.ADJUST_MEASUREMENT_DISTANCE,
        ActionCode.SHIELD_OR_REDUCE_REFLECTIONS,
    )


def test_action_c_when_criterion_2_fails_with_moderate_inward_flow() -> None:
    """Table B.3 row 3: criterion 2 unmet and 1 dB <= (F3 - F2) <= 3 dB."""
    intensity = np.full(10, 1.0e-5)
    intensity[0] = -2.04e-5  # tuned to put (F3 - F2) at about 2 dB
    result = _qualified_case(intensity)
    assert result.f2 is not None
    assert result.f3 is not None
    excess = float(result.f3[0] - result.f2[0])
    assert 1.0 <= excess <= 3.0
    assert result.criterion_2 is not None
    assert not bool(result.criterion_2[0])
    assert result.required_actions()[0] == (ActionCode.INCREASE_POSITION_DENSITY,)


def test_action_c_still_holds_halfway_between_the_two_rows() -> None:
    """The row-c and row-d split is at 1 dB, and 1,5 dB is still row c.

    The case above sits at about 2 dB, which leaves the whole of 1 dB to 2 dB
    untested and a limit anywhere in it indistinguishable from the printed one.
    One inward position of -1,54 against nine outward of 1,0 puts F3 - F2 at
    1,5 dB, halfway across that gap.
    """
    intensity = np.full(10, 1.0e-5)
    intensity[0] = -1.5388e-5
    result = _qualified_case(intensity)
    assert result.f2 is not None
    assert result.f3 is not None
    excess = float(result.f3[0] - result.f2[0])
    assert 1.4 < excess < 1.6
    assert result.criterion_2 is not None
    assert not bool(result.criterion_2[0])
    assert result.required_actions()[0] == (ActionCode.INCREASE_POSITION_DENSITY,)


def test_action_d_when_criterion_2_fails_with_little_inward_flow() -> None:
    """Table B.3 row 4: criterion 2 unmet, (F3 - F2) <= 1 dB, 8.3.2 not taken."""
    intensity = np.array([4e-5, 3e-5, 2e-5, 1e-5, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6])
    result = _qualified_case(intensity)
    assert result.f2 is not None
    assert result.f3 is not None
    assert float(result.f3[0] - result.f2[0]) <= 1.0
    assert result.criterion_2 is not None
    assert not bool(result.criterion_2[0])
    assert result.required_actions()[0] == (ActionCode.INCREASE_DISTANCE_OR_POSITIONS,)


def test_action_d_is_the_action_at_exactly_one_decibel() -> None:
    """Table B.3's rows c and d overlap at F3 - F2 = 1 dB; Figure B.1 gives it to d.

    The action-c row is conditioned on ``1 dB <= (F3 - F2) <= 3 dB`` and the
    action-d row on ``(F3 - F2) <= 1 dB``, so the printed table prescribes two
    different actions for that one state (see ``docs/ERRATA.md``). Figure B.1
    settles it: the "(F3 - F2) <= 1 dB ?" diamond sends its Yes branch to the
    optional procedure and to action d. The boundary is set on the result
    rather than tuned into the intensities, because 1 dB has to be exactly
    1 dB for the case to be the one the two rows disagree about.
    """
    import dataclasses

    intensity = np.array([4e-5, 3e-5, 2e-5, 1e-5, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6])
    result = _qualified_case(intensity)
    assert result.criterion_2 is not None
    assert not bool(result.criterion_2[0])
    on_the_boundary = dataclasses.replace(
        result, f2=np.array([5.0]), f3=np.array([6.0])
    )
    assert on_the_boundary.required_actions()[0] == (
        ActionCode.INCREASE_DISTANCE_OR_POSITIONS,
    )


def test_an_f1_that_is_not_a_number_asks_for_action_e() -> None:
    """A field indicator that is not a number did not qualify the field.

    F1 is compared against 0,6, and NaN answers that comparison False just as a
    small F1 does. Written as "F1 > 0,6" the first gate of Figure B.1 would
    wave through an initial test that produced no usable indicator at all, so
    it is written as "not F1 <= 0,6" instead and the two part company here.
    """
    import dataclasses

    intensity, areas = _uniform_surface(positions=10)
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full(10, 80.0),
        pressure_residual_index=30.0,
        frequencies=[1000.0],
    )
    unsteady = dataclasses.replace(result, f1=np.array([math.nan]))
    assert unsteady.required_actions()[0] == (ActionCode.REDUCE_TEMPORAL_VARIABILITY,)


def test_a_criterion_2_that_was_never_evaluated_is_skipped_not_failed() -> None:
    """An absent gate has nothing to act on, which is not the same as failing.

    Without frequencies there is no Table B.2 row, so criterion 2 was never
    put. Reading its absence as a failure would send a band that cleared every
    gate it could be judged on to action (c) or (d), prescribing new positions
    for a surface nothing has found fault with.
    """
    intensity, areas = _uniform_surface(positions=10)
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full(10, 80.0),
        pressure_residual_index=30.0,
    )
    assert result.criterion_2 is None
    assert result.required_actions()[0] == ()


def test_a_qualified_band_calls_for_no_action() -> None:
    """Figure B.1's "final result" branch: every gate passed, nothing to change."""
    intensity, areas = _uniform_surface(positions=20)
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full((20, 1), 80.0),
        pressure_residual_index=40.0,
        frequencies=[1000.0],
        grade="precision",
    )
    assert result.required_actions() == ((),)


def test_only_the_first_failing_gate_is_acted_on() -> None:
    """Figure B.1 returns to the next measurement, so one action set per band."""
    intensity = np.array([4e-5, 3e-5, 2e-5, 1e-5, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6])
    wandering = np.array([1.0e-5, 5.0e-5, 2.0e-6, 9.0e-5, 1.0e-6])
    result = _qualified_case(intensity, residual_index=10.0, temporal=wandering)
    # F1, criterion 1 and criterion 2 all fail; only the first gate is reported.
    assert result.required_actions()[0] == (ActionCode.REDUCE_TEMPORAL_VARIABILITY,)


def test_an_unqualified_determination_has_nothing_to_act_on() -> None:
    """Table B.3 presupposes the criteria were evaluated."""
    intensity, areas = _uniform_surface()
    result = emission.sound_power_intensity_points(intensity, areas)
    with pytest.raises(ValueError, match="was not qualified"):
        result.required_actions()


@pytest.mark.parametrize(
    ("criterion", "codes"),
    [(row[0], row[1]) for row in ref.ISO9614_1_TABLE_B3],
)
def test_every_printed_action_code_carries_its_row(
    criterion: str, codes: tuple[str, ...]
) -> None:
    """Each Table B.3 code exists, is a letter, and explains itself."""
    del criterion
    for code in codes:
        action = ActionCode(code)
        assert action.value == code
        assert action.criterion
        assert action.action.endswith(".")


def test_the_five_action_codes_are_the_printed_five() -> None:
    """Table B.3 lists a, b, c, d and e, and nothing else."""
    printed = {code for _, codes in ref.ISO9614_1_TABLE_B3 for code in codes}
    assert {member.value for member in ActionCode} == printed


# ---------------------------------------------------------------------------
# Equation (B.3) and the Table 2 uncertainty
# ---------------------------------------------------------------------------
def test_confidence_interval_follows_equation_b3() -> None:
    """10*lg(1 +/- 2*F4/sqrt(N)) dB, the interval clause 10.5 c) asks for."""
    areas = np.full(10, 1.0)
    rng = np.random.default_rng(11)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.3, (10, 1)))
    result = emission.sound_power_intensity_points(intensity, areas)
    assert result.f4 is not None
    assert result.confidence_interval is not None
    spread = 2.0 * float(result.f4[0]) / math.sqrt(10)
    assert result.confidence_interval[0, 0] == pytest.approx(
        10.0 * math.log10(1.0 - spread)
    )
    assert result.confidence_interval[0, 1] == pytest.approx(
        10.0 * math.log10(1.0 + spread)
    )


def test_confidence_interval_lower_end_is_nan_without_an_argument() -> None:
    """2*F4/sqrt(N) at or above 1 leaves nothing to take the logarithm of."""
    areas = np.full(4, 1.0)
    intensity = np.array([1.0e-4, 1.0e-6, 1.0e-6, 1.0e-6])
    with pytest.warns(emission.SoundPowerWarning, match="at least 10"):
        result = emission.sound_power_intensity_points(intensity, areas)
    assert result.f4 is not None
    assert result.confidence_interval is not None
    assert 2.0 * float(result.f4[0]) / 2.0 >= 1.0
    assert math.isnan(float(result.confidence_interval[0, 0]))
    assert float(result.confidence_interval[0, 1]) > 0.0


def _three_bands_of_three_different_grades() -> emission.DiscretePointIntensityResult:
    """One determination whose bands reach grade 1, grade 2 and no grade.

    Twenty unit segments over three octave bands, asked for at grade 1. The
    250 Hz band alternates about its mean at F4 = 0,90, so 29 x F4^2 is 23,5
    against twenty positions and criterion 2 fails at grade 1 while 19 x F4^2
    is 15,4 and it holds at grade 2. The 1 kHz band is uniform and reaches
    grade 1. The 2 kHz band is uniform too, but is measured in a pressure
    field 15 dB louder, which puts F2 at 25 dB over an Ld of 20 dB and fails
    criterion 1.
    """
    areas = np.full(20, 1.0)
    alternating = 1.0e-5 * (1.0 + 0.8772 * np.array([1.0, -1.0] * 10))
    intensity = np.column_stack([alternating, np.full(20, 1.0e-5), np.full(20, 1.0e-5)])
    levels = np.full((20, 3), 80.0)
    levels[:, 2] = 95.0
    return emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=levels,
        pressure_residual_index=30.0,
        frequencies=[250.0, 1000.0, 2000.0],
        band_type="octave",
        grade="precision",
    )


def test_expanded_uncertainty_is_twice_the_table_2_standard_deviation() -> None:
    """Table 2 footnote 1: the true level lies within +/- 2s at 95 %."""
    result = _three_bands_of_three_different_grades()
    assert result.achieved_grade is not None
    assert list(result.achieved_grade) == ["engineering", "precision", "none"]
    assert result.expanded_uncertainty is not None
    assert float(result.expanded_uncertainty[0]) == pytest.approx(
        2.0
        * emission.determination_standard_deviation(
            "engineering", 250.0, band_type="octave"
        )
    )
    assert float(result.expanded_uncertainty[1]) == pytest.approx(
        2.0
        * emission.determination_standard_deviation(
            "precision", 1000.0, band_type="octave"
        )
    )
    np.testing.assert_allclose(result.expanded_uncertainty[:2], [4.0, 2.0])


def test_the_uncertainty_is_read_at_the_grade_the_band_achieved() -> None:
    """Clause 10.6 states the grade *achieved*, so Table 2 is read in its row.

    Grade 1 was asked for and only the 1 kHz band reached it. Reading Table 2
    at the grade requested would give the whole spectrum the grade-1 figures
    of 3 dB, 2 dB and 2 dB, and would understate by a decibel exactly the band
    that fell short.
    """
    result = _three_bands_of_three_different_grades()
    assert result.grade == "precision"
    assert result.expanded_uncertainty is not None
    requested = [
        2.0
        * emission.determination_standard_deviation(
            "precision", frequency, band_type="octave"
        )
        for frequency in (250.0, 1000.0, 2000.0)
    ]
    assert float(result.expanded_uncertainty[0]) > requested[0]
    assert float(result.expanded_uncertainty[1]) == pytest.approx(requested[1])


def test_a_band_that_reached_no_grade_carries_no_uncertainty() -> None:
    """Table 2 has no row for a determination that qualified as nothing.

    Clause 10.5 c) gives such a band the confidence interval of Formula (B.3)
    instead, which the result carries beside this and which stays finite.
    """
    result = _three_bands_of_three_different_grades()
    assert result.achieved_grade is not None
    assert result.achieved_grade[2] == "none"
    assert result.expanded_uncertainty is not None
    assert math.isnan(float(result.expanded_uncertainty[2]))
    assert result.confidence_interval is not None
    assert np.all(np.isfinite(result.confidence_interval[2]))


@pytest.mark.filterwarnings("ignore:The A-weighted total sums every")
def test_an_unqualified_determination_states_no_uncertainty_either() -> None:
    """No criteria inputs, no achieved grade, so nothing clause 10.6 can state."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 3), 1.0e-5)
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        frequencies=[125.0, 500.0, 2000.0],
        band_type="octave",
        grade="engineering",
    )
    assert result.achieved_grade is None
    assert result.expanded_uncertainty is None


# ---------------------------------------------------------------------------
# The A-weighted determination (B.1.2 and clause 10.5 b)
# ---------------------------------------------------------------------------
def test_a_weighted_total_omits_the_bands_that_failed_the_criteria() -> None:
    """Clause 10.5 b): the bands failing criteria 1 and/or 2 leave the sum."""
    areas = np.full(20, 1.0)
    frequencies = np.array([125.0, 250.0, 500.0, 1000.0])
    intensity = np.full((20, 4), 1.0e-5)
    intensity[:, 0] = 1.0e-5 * np.linspace(0.2, 1.8, 20)  # non-uniform low band
    levels = np.full((20, 4), 80.0)
    levels[:, 0] = 95.0  # a reactive low band: F2 above Ld
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=levels,
        pressure_residual_index=25.0,
        frequencies=frequencies,
        band_type="octave",
    )
    assert result.a_weighting_omitted_bands is not None
    assert bool(result.a_weighting_omitted_bands[0])
    kept = ~result.a_weighting_omitted_bands & ~result.not_applicable_band
    corrections = np.array([-16.1, -8.6, -3.2, 0.0])
    expected = 10.0 * math.log10(
        float(
            np.sum(10.0 ** (0.1 * (result.sound_power_level[kept] + corrections[kept])))
        )
    )
    assert result.sound_power_level_a == pytest.approx(expected)


def test_a_band_that_fails_only_criterion_2_leaves_the_sum_as_well() -> None:
    """Clause 10.5 b) says "criteria 1 and/or 2", and the "or" half is real.

    The case above omits a band on criterion 1 alone, which a screening that
    read criterion 1 only would omit just the same. Here the instrument is
    ample in both bands and criterion 1 holds in both, and what the low band
    fails is criterion 2: its field alternates between 2,0 and 0,2, and
    29 x F4^2 is 21,6 against the ten positions measured.
    """
    areas = np.full(10, 1.0)
    intensity = np.column_stack(
        [np.concatenate([np.full(5, 2.0e-5), np.full(5, 0.2e-5)]), np.full(10, 1.0e-5)]
    )
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full((10, 2), 80.0),
        pressure_residual_index=40.0,
        frequencies=[1000.0, 2000.0],
        band_type="octave",
    )
    assert result.criterion_1 is not None
    assert result.criterion_2 is not None
    assert result.a_weighting_omitted_bands is not None
    np.testing.assert_array_equal(result.criterion_1, [True, True])
    np.testing.assert_array_equal(result.criterion_2, [False, True])
    np.testing.assert_array_equal(result.a_weighting_omitted_bands, [True, False])
    corrections = np.array([0.0, 1.2])
    expected = float(result.sound_power_level[1]) + corrections[1]
    assert result.sound_power_level_a == pytest.approx(expected)


def test_note_11_reads_c_in_the_mid_row_when_the_top_bands_are_quiet() -> None:
    """Note 11: below half the A-weighted power up top, C comes from 200-630 Hz.

    B.1.2 otherwise takes the largest C over the summed range, which the 1 kHz
    octave sets at 57 for grade 1. Here the 1 kHz band carries 7,5 % of the
    A-weighted power, so Note 11 sends the lookup to the row covering the
    250 Hz and 500 Hz octaves instead, where C is 29. Twenty positions against
    an A-weighted F4 of 0,664 clear 29 x F4^2 = 12,8 and do not clear
    57 x F4^2 = 25,2, so the two readings hand back different grades.
    """
    areas = np.full(20, 1.0)
    alternating = 1.0e-5 * (1.0 + 0.7 * np.array([1.0, -1.0] * 10))
    intensity = np.column_stack([alternating, alternating, np.full(20, 0.05e-5)])
    frequencies = np.array([250.0, 500.0, 1000.0])
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full((20, 3), 80.0),
        pressure_residual_index=40.0,
        frequencies=frequencies,
        band_type="octave",
        grade="precision",
    )
    assert result.a_weighting_omitted_bands is not None
    assert not np.any(result.a_weighting_omitted_bands)

    corrections = np.array([-8.6, -3.2, 0.0])
    contributions = 10.0 ** (0.1 * (result.sound_power_level + corrections))
    assert float(contributions[2]) < 0.5 * float(np.sum(contributions))

    required = result.positions / result.field_nonuniformity_a**2
    assert (
        emission.position_count_factor("precision", 250.0, band_type="octave")
        < required
        < emission.position_count_factor("precision", 1000.0, band_type="octave")
    )
    assert result.achieved_grade_a == "precision"


#: Twenty positions alternating +/- 70 % about their mean. Every band built
#: from it has the same shape, so the A-weighted F4 is 0,718 whatever the mix
#: of band powers is, and N/F4a^2 is 38,8: between Table B.2's 29 and 57 at
#: grade 1 and clear of 11 and 19 at grade 2. That is what lets the Note 11
#: probes below move the power between bands and watch only the C lookup move.
_ALTERNATING_20 = 1.0 + 0.7 * np.array([1.0, -1.0] * 10)


def _three_octaves(top: float) -> emission.DiscretePointIntensityResult:
    """250, 500 and 1 000 Hz octaves of the same shape, the top one scaled."""
    return emission.sound_power_intensity_points(
        np.column_stack(
            [1.0e-5 * _ALTERNATING_20, 1.0e-5 * _ALTERNATING_20, top * _ALTERNATING_20]
        ),
        np.full(20, 1.0),
        pressure_levels=np.full((20, 3), 80.0),
        pressure_residual_index=40.0,
        frequencies=[250.0, 500.0, 1000.0],
        band_type="octave",
        grade="engineering",
    )


def test_note_11_measures_the_top_bands_against_half_the_power() -> None:
    """Note 11 prints "half", and the half is what decides which row is read.

    Here the 1 kHz octave carries 35,0 % of the A-weighted power: below half,
    so the Note fires and C at grade 1 is the mid row's 29, asking 15,0 of the
    twenty positions measured, which the sum has. Read against a quarter the
    Note would stand down, the largest C over the range would be the 1 kHz
    row's 57, asking 29,4 positions, and the same sum would settle for grade 2.
    """
    result = _three_octaves(0.332e-5)
    assert result.a_weighting_omitted_bands is not None
    assert not np.any(result.a_weighting_omitted_bands)
    assert result.achieved_grade_a == "precision"


def test_above_half_the_largest_c_over_the_summed_range_is_taken() -> None:
    """B.1.2 asks for the largest C in the range, which is its worst case.

    With the 1 kHz octave carrying 60,0 % of the A-weighted power Note 11
    stands down and B.1.2's own rule applies: "el mayor valor de C en la banda
    de frecuencia comprendida por esta suma". That largest value is the 1 kHz
    row's 57 at grade 1, which asks 29,4 of the twenty positions measured, so
    the sum settles for grade 2; the smallest C over the same range, the mid
    row's 29, would have asked 15,0 and granted grade 1. The 1 kHz octave is
    the first band of the high row, so the range is read inclusively at both
    ends: drop it and the Note fires on a top row that carries nothing.
    """
    result = _three_octaves(0.925e-5)
    assert result.a_weighting_omitted_bands is not None
    assert not np.any(result.a_weighting_omitted_bands)
    assert result.achieved_grade_a == "engineering"


def test_the_800_hz_one_third_octave_belongs_to_note_11s_top_row() -> None:
    """Note 11's high row is 800 Hz to 5 kHz in thirds and 1 to 4 kHz in octaves.

    The two spans are the same row of Table B.2 read in its two columns, so
    reading the octave span for a one-third-octave determination moves the
    800 Hz band out of the top row and Note 11 fires when it should not. Here
    the 800 Hz third carries 85,8 % of the A-weighted power, so the Note stands
    down and C at grade 1 is that row's 57, asking 29,4 of the twenty positions
    measured; read out of the row, the Note would fire and the mid row's 29
    would ask only 15,0 and grant grade 1.
    """
    result = emission.sound_power_intensity_points(
        np.column_stack([1.0e-5 * _ALTERNATING_20, 1.0e-5 * _ALTERNATING_20]),
        np.full(20, 1.0),
        pressure_levels=np.full((20, 2), 80.0),
        pressure_residual_index=40.0,
        frequencies=[250.0, 800.0],
        band_type="third",
        grade="engineering",
    )
    assert result.achieved_grade_a == "engineering"


def test_note_11s_row_is_read_at_the_grade_being_tried() -> None:
    """The substituted C is still a C, so it comes from the grade's own column.

    Note 11 changes which row of Table B.2 is read and not which column. The
    1 kHz octave carries 7,5 % of the A-weighted power, so the Note fires; with
    an A-weighted F4 of 0,854 over twenty positions, the mid row's 29 at
    grade 1 asks for 21,2 positions and its 11 at grade 2 asks for 8,0, so the
    sum reaches grade 2. Read at the wrong grade it would be handed grade 1.
    """
    alternating = 1.0 + 0.9 * np.array([1.0, -1.0] * 10)
    result = emission.sound_power_intensity_points(
        np.column_stack(
            [1.0e-5 * alternating, 1.0e-5 * alternating, np.full(20, 0.05e-5)]
        ),
        np.full(20, 1.0),
        pressure_levels=np.full((20, 3), 80.0),
        pressure_residual_index=40.0,
        frequencies=[250.0, 500.0, 1000.0],
        band_type="octave",
        grade="engineering",
    )
    assert result.a_weighting_omitted_bands is not None
    assert not np.any(result.a_weighting_omitted_bands)
    assert result.field_nonuniformity_a == pytest.approx(0.854, abs=5e-4)
    assert result.achieved_grade_a == "engineering"


def test_one_band_of_inward_flow_stops_the_a_weighted_grade() -> None:
    """The gates before criterion 2 must hold in every band the sum covers.

    An A-weighted level built on a band whose surface carries too much inward
    flow is no better than that band, so B.1.2's per-band gates are quantified
    over all of them. Here the 2 kHz octave has F3 - F2 at 3,98 dB while the
    1 kHz one is uniform and outward at 0 dB, and a gate read over "any band",
    or one reading the difference the other way round, would grade the sum on
    the clean band alone.
    """
    n = 40
    pattern = np.array([1.0] * 28 + [-1.0] * 12)
    result = emission.sound_power_intensity_points(
        np.column_stack([np.full(n, 1.0e-5), 1.0e-5 * pattern]),
        np.full(n, 1.0),
        pressure_levels=np.full((n, 2), 80.0),
        pressure_residual_index=30.0,
        frequencies=[1000.0, 2000.0],
        band_type="octave",
        grade="survey",
    )
    assert result.f2 is not None
    assert result.f3 is not None
    assert float(result.f3[1] - result.f2[1]) > 3.0
    assert result.a_weighting_omitted_bands is not None
    assert not np.any(result.a_weighting_omitted_bands)
    assert result.achieved_grade_a == "none"


def test_one_unsteady_band_stops_the_a_weighted_grade_too() -> None:
    """The same quantifier over F1: one band's initial test failing is enough."""
    n = 40
    temporal = np.column_stack(
        [np.full(5, 1.0e-5), np.array([1.0e-5, 5.0e-5, 2.0e-6, 9.0e-5, 1.0e-6])]
    )
    result = emission.sound_power_intensity_points(
        np.full((n, 2), 1.0e-5),
        np.full(n, 1.0),
        pressure_levels=np.full((n, 2), 80.0),
        pressure_residual_index=30.0,
        temporal_intensity=temporal,
        frequencies=[1000.0, 2000.0],
        band_type="octave",
        grade="precision",
    )
    assert result.f1 is not None
    assert float(result.f1[0]) < emission.TEMPORAL_VARIABILITY_LIMIT
    assert float(result.f1[1]) > emission.TEMPORAL_VARIABILITY_LIMIT
    assert result.achieved_grade_a == "none"


def test_criterion_1_must_hold_in_every_summed_band_at_the_grade_tried() -> None:
    """And over criterion 1, whose Ld moves with the grade being tried.

    delta_pI0 = 17 dB gives Ld = 7 dB at the K = 10 dB of grades 1 and 2 and
    Ld = 10 dB at the K = 7 dB of grade 3. Against F2 of 5,0 dB in one band and
    9,5 dB in the other, only grade 3 clears both, so the search walks down the
    grades and stops at survey. Quantified over "any band" the 5 dB band alone
    would have carried the sum to grade 1.
    """
    n = 40
    levels = np.column_stack([np.full(n, 75.0), np.full(n, 79.5)])
    result = emission.sound_power_intensity_points(
        np.full((n, 2), 1.0e-5),
        np.full(n, 1.0),
        pressure_levels=levels,
        pressure_residual_index=17.0,
        frequencies=[1000.0, 2000.0],
        band_type="octave",
        grade="survey",
    )
    assert result.criterion_1 is not None
    assert bool(np.all(result.criterion_1))
    assert result.achieved_grade_a == "survey"


def test_the_a_weighted_indicator_goes_nan_when_a_2_3_refuses_it() -> None:
    """B.1.2's own F4 meets A.2.3 too, and takes the same way out as a band.

    The A-weighted determination sums the weighted band intensities into one
    intensity per position and applies (A.8) and (A.9) to those, so its mean
    can be non-positive exactly as a band's can. Letting the coefficient of
    variation raise there would take the whole determination down over a
    quantity the standard scopes to the A-weighted figure alone: the level and
    every per-band result stand, and only ``achieved_grade_a`` is withheld.
    """
    areas = np.array([5.0] + [0.5] * 9)
    intensity = np.array([1.0e-4] + [-5.0e-5] * 9).reshape(-1, 1)
    with pytest.warns(emission.SoundPowerWarning, match="A.2.3"):
        result = emission.sound_power_intensity_points(
            intensity, areas, frequencies=[1000.0]
        )
    assert not bool(result.not_applicable_band[0])
    assert math.isnan(result.field_nonuniformity_a)
    assert result.achieved_grade_a is None
    assert math.isfinite(result.sound_power_level_a)


def test_two_positions_carry_an_a_weighted_field_indicator() -> None:
    """Equation (A.8) divides by N - 1, so two positions are one spread.

    The A-weighted route has its own position count check, and it has to admit
    the same minimum the per-band one does: at two positions (A.8) is defined
    and the indicator is a number, not the NaN of a determination too small to
    have a spread at all.
    """
    with pytest.warns(emission.SoundPowerWarning, match="at least 10"):
        result = emission.sound_power_intensity_points(
            np.array([[1.0e-5], [2.0e-5]]),
            np.full(2, 1.0),
            frequencies=[1000.0],
        )
    assert math.isfinite(result.field_nonuniformity_a)
    assert result.field_nonuniformity_a > 0.0


def test_a_weighted_screening_is_announced_when_it_cannot_be_done() -> None:
    """Without the criteria inputs every applicable band is summed, with a warning."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 2), 1.0e-5)
    with pytest.warns(emission.SoundPowerWarning, match="clause 10.5 b"):
        result = emission.sound_power_intensity_points(
            intensity, areas, frequencies=[500.0, 1000.0], band_type="octave"
        )
    assert result.a_weighting_omitted_bands is None


def test_the_a_weighted_determination_has_its_own_field_non_uniformity() -> None:
    """B.1.2: F4 of the A-weighted band intensities summed per position."""
    areas = np.full(20, 1.0)
    frequencies = np.array([500.0, 1000.0, 2000.0])
    rng = np.random.default_rng(12)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.15, (20, 3)))
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full((20, 3), 80.0),
        pressure_residual_index=40.0,
        frequencies=frequencies,
        band_type="octave",
    )
    corrections = np.array([-3.2, 0.0, 1.2])
    per_position = np.sum(intensity * 10.0 ** (0.1 * corrections), axis=1)
    mean = float(np.mean(per_position))
    expected = (
        math.sqrt(float(np.sum((per_position - mean) ** 2)) / (per_position.size - 1))
        / mean
    )
    assert result.field_nonuniformity_a == pytest.approx(expected)


def test_the_survey_grade_is_reached_only_on_the_a_weighted_sum() -> None:
    """Grade 3 uses the single A-weighted C = 8, which no band column offers."""
    areas = np.full(10, 1.0)
    frequencies = np.array([500.0, 1000.0, 2000.0])
    # One position four times the rest gives F4 = 0,730 over ten positions, so
    # C F4^2 is 30,4 at grade 1 and 15,4 at grade 2, both above N = 10, while
    # the A-weighted C = 8 of grade 3 asks for only 4,3.
    weights = np.array([4.0] + [1.0] * 9)
    intensity = 1.0e-5 * np.repeat(weights[:, None], 3, axis=1)
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full((10, 3), 80.0),
        pressure_residual_index=40.0,
        frequencies=frequencies,
        band_type="octave",
        grade="survey",
    )
    assert result.achieved_grade_a == "survey"
    assert result.positions > 8.0 * result.field_nonuniformity_a**2
    assert result.positions < 29.0 * result.field_nonuniformity_a**2


def test_a_single_band_without_frequencies_is_its_own_a_weighted_total() -> None:
    """There is nothing to weight, so the band level stands as the total."""
    intensity, areas = _uniform_surface()
    result = emission.sound_power_intensity_points(intensity, areas)
    assert result.sound_power_level_a == pytest.approx(90.0)
    assert math.isnan(result.field_nonuniformity_a)
    assert result.achieved_grade_a is None


def test_a_single_band_needs_no_clause_10_5_b_warning() -> None:
    """Clause 10.5 b) screens a sum, and one band is not summed with anything.

    The warning exists because an A-weighted total assembled without the
    criteria has not had the failing bands taken out of it. With a single band
    there is nothing to take out and nothing to state, so warning would train
    the reader to ignore the message that matters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emission.sound_power_intensity_points(
            np.full((10, 1), 1.0e-5), np.full(10, 1.0), frequencies=[1000.0]
        )


# ---------------------------------------------------------------------------
# The optional procedure of clause 8.3.2 / B.1.3 (Eq. B.4)
# ---------------------------------------------------------------------------
def test_partial_power_concentration_follows_equation_b4() -> None:
    """N* = ceil(4*[F4(alpha)/Delta_alpha]^2), with Delta from Table B.1."""
    areas = np.full(12, 1.0)
    intensity = np.array([3.0e-5, 2.4e-5, 1.8e-5, 1.4e-5] + [8.0e-6] * 8)
    outcome = emission.partial_power_concentration(
        intensity, areas, grade="engineering"
    )

    total = float(np.sum(intensity * areas))
    assert outcome.subset_positions == 4
    assert outcome.subset_positions < 0.5 * outcome.positions
    assert outcome.power_fraction == pytest.approx(float(np.sum(intensity[:4])) / total)
    assert outcome.power_fraction > 0.5
    assert outcome.subset_area == pytest.approx(4.0)
    assert outcome.error_factor == pytest.approx(0.29)

    subset, remainder = intensity[:4], intensity[4:]
    f4_subset = float(np.std(subset, ddof=1) / np.mean(subset))
    f4_remainder = float(np.std(remainder, ddof=1) / np.mean(remainder))
    alpha = outcome.power_fraction
    delta_alpha = (
        0.29 - (1.0 - alpha) * (2.0 / math.sqrt(remainder.size)) * f4_remainder
    ) / alpha
    assert outcome.subset_nonuniformity == pytest.approx(f4_subset)
    assert outcome.remainder_nonuniformity == pytest.approx(f4_remainder)
    assert outcome.subset_error_factor == pytest.approx(delta_alpha)
    assert outcome.additional_positions == math.ceil(
        4.0 * (f4_subset / delta_alpha) ** 2
    )


def test_the_concentration_error_factor_follows_the_grade() -> None:
    """Table B.1 spends 0,20 at grade 1, 0,29 at grade 2 and 0,60 at grade 3."""
    areas = np.full(12, 1.0)
    intensity = np.array([3.0e-5, 2.4e-5, 1.8e-5, 1.4e-5] + [8.0e-6] * 8)
    for grade, delta in (
        ("precision", 0.20),
        ("engineering", 0.29),
        ("survey", 0.60),
    ):
        outcome = emission.partial_power_concentration(intensity, areas, grade=grade)  # type: ignore[arg-type]
        assert outcome.error_factor == pytest.approx(delta)


def test_a_stricter_grade_asks_for_more_new_positions() -> None:
    """A smaller Delta leaves less error budget, so N* rises."""
    areas = np.full(12, 1.0)
    intensity = np.array([3.0e-5, 2.4e-5, 1.8e-5, 1.4e-5] + [8.0e-6] * 8)
    precision = emission.partial_power_concentration(
        intensity, areas, grade="precision"
    )
    engineering = emission.partial_power_concentration(
        intensity, areas, grade="engineering"
    )
    assert precision.additional_positions >= engineering.additional_positions


def test_a_non_uniform_remainder_spends_part_of_the_error_budget() -> None:
    """Eq. (B.4) with every term of Delta_alpha alive.

    A perfectly uniform remainder has F4(1 - alpha) = 0, which cancels the
    whole ``(1 - alpha) * 2/sqrt(N_(1-alpha)) * F4(1 - alpha)`` term and leaves
    Delta_alpha = Delta/alpha, so neither the 2 nor the square root in it is
    pinned by such a case. Here the remainder falls away steadily instead, and
    every factor of both formulae carries its own weight in the answer.
    """
    areas = np.full(12, 1.0)
    intensity = np.array(
        [3.0e-5, 2.4e-5, 1.8e-5, 1.4e-5]
        + [1.2e-5, 1.0e-5, 9.0e-6, 8.0e-6, 7.0e-6, 6.0e-6, 5.0e-6, 4.0e-6]
    )
    outcome = emission.partial_power_concentration(
        intensity, areas, grade="engineering"
    )

    subset, remainder = intensity[:4], intensity[4:]
    alpha = float(np.sum(subset)) / float(np.sum(intensity))
    f4_subset = float(np.std(subset, ddof=1) / np.mean(subset))
    f4_remainder = float(np.std(remainder, ddof=1) / np.mean(remainder))
    spent = (1.0 - alpha) * (2.0 / math.sqrt(remainder.size)) * f4_remainder
    delta_alpha = (0.29 - spent) / alpha

    assert outcome.subset_positions == 4
    assert outcome.remainder_nonuniformity == pytest.approx(f4_remainder)
    assert outcome.remainder_nonuniformity > 0.0
    assert spent == pytest.approx(0.1027, abs=5.0e-4)
    assert outcome.subset_error_factor == pytest.approx(delta_alpha)
    assert outcome.additional_positions == 5
    assert outcome.additional_positions == math.ceil(
        4.0 * (f4_subset / delta_alpha) ** 2
    )
    # What the degenerate case cannot tell apart: a remainder that spends
    # nothing asks for 2 new positions, a factor of 3 in place of the 4 asks
    # for 4, and dropping the square root asks for 3.
    assert outcome.additional_positions != math.ceil(
        4.0 * (f4_subset * alpha / 0.29) ** 2
    )
    assert outcome.additional_positions != math.ceil(
        3.0 * (f4_subset / delta_alpha) ** 2
    )
    assert outcome.additional_positions != math.ceil(
        4.0
        * (
            f4_subset
            * alpha
            / (0.29 - (1.0 - alpha) * (2.0 / remainder.size) * f4_remainder)
        )
        ** 2
    )


def test_a_spread_out_field_has_no_concentration_to_exploit() -> None:
    """B.1.3 requires the subset to be fewer than half the segments."""
    areas = np.full(10, 1.0)
    intensity = np.full(10, 1.0e-5)
    with pytest.raises(ValueError, match="carries more than half the sound power"):
        emission.partial_power_concentration(intensity, areas)


def test_a_subset_of_exactly_half_the_segments_is_not_a_concentration() -> None:
    """B.1.3 asks for fewer than half, and half is not fewer than half.

    Four segments of eight at 1,9 and four at 1,0 (arbitrary units): the top
    three carry 5,7 of the 11,6 total and so do not reach half, and it takes
    the fourth to pass it. That is a subset of exactly N/2, which the standard
    does not allow, with the power comparison itself well clear of the
    boundary.
    """
    areas = np.full(8, 1.0)
    intensity = np.array([1.9e-5] * 4 + [1.0e-5] * 4)
    with pytest.raises(ValueError, match="carries more than half the sound power"):
        emission.partial_power_concentration(intensity, areas)


def test_a_subset_carrying_exactly_half_the_power_is_not_yet_the_subset() -> None:
    """B.1.3 prints "más de la mitad", and exactly half is not more than half.

    Eight segments of powers 2, 2, 1, 1, 1, 1, 1, 1 in units of the smallest,
    all of them exact in binary so that the cumulative sum lands on half the
    total and not near it: the top three carry 5 of the 10 and the subset is
    not closed until the fourth joins. Four of eight is not fewer than half of
    N, so the concentration test fails, whereas a subset closed at three would
    have passed it.
    """
    intensity = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    areas = np.full(8, 1.0)
    with pytest.raises(ValueError, match="carries more than half the sound power"):
        emission.partial_power_concentration(intensity, areas)


def test_fewer_than_half_of_an_odd_segment_count_is_half_of_n() -> None:
    """The half B.1.3 bounds N_alpha by is half of N, not half of N - 1.

    Five of eleven segments is fewer than 5,5 and so is a legal subset; taken
    against half of N - 1 it would be exactly 5 and refused. Odd counts are the
    only ones that can tell the two readings apart, and B.1.3 says "la mitad
    del número total de segmentos N".
    """
    intensity = np.array([1.5] * 5 + [1.0] * 6)
    areas = np.full(11, 1.0)
    outcome = emission.partial_power_concentration(intensity, areas)
    assert outcome.positions == 11
    assert outcome.subset_positions == 5


def test_one_position_cannot_be_split_into_a_subset_and_a_remainder() -> None:
    """The procedure needs both halves, and one segment cannot make two.

    Refusing it up front says which input was wrong. Left to fall through, a
    single position reaches the "fewer than half" refusal instead and blames
    the concentration test for a surface that was never divisible.
    """
    with pytest.raises(ValueError, match="At least two measurement positions"):
        emission.partial_power_concentration([1.0e-5], [1.0])


def test_the_half_is_half_of_the_net_sound_power() -> None:
    """The total B.1.3 halves is the sound power, inward flow included.

    The ranking runs over the positive partial powers, so it is tempting to
    halve their sum too, but the sound power of the band is the net one of
    equation (12) and the inward-flowing segments are part of it. Here they
    make the difference between a subset of two and a subset of three: the top
    two carry 6,4e-5 W of a net 1,1e-4 W, over half, while the positive powers
    alone come to 1,32e-4 W and would take a third segment to pass.
    """
    intensity = np.array(
        [-2e-6, 2e-6, 4e-6, 8e-6, 1e-6, -2e-6, 4e-6, -2e-6, 1.6e-5, 8e-6]
    )
    areas = np.array([4.0, 2.0, 4.0, 2.0, 4.0, 4.0, 2.0, 8.0, 1.0, 1.0])
    outcome = emission.partial_power_concentration(intensity, areas)
    assert outcome.subset_positions == 2
    assert outcome.subset_area == pytest.approx(6.0)


def test_the_ranking_is_of_partial_powers_and_not_of_intensities() -> None:
    """B.1.3 ranks "las potencias acústicas ... parciales", which carry the area.

    A small segment of high intensity passes less power than a large one of
    low intensity, so on unequal segments the two orders differ, and every
    figure the procedure returns differs with them. Here ranking by intensity
    picks a different subset of the same size, an area of 16 m^2 in place of
    20 m^2, and equation (B.4) then asks for no new positions at all instead
    of 91.
    """
    intensity = np.array(
        [2e-6, 8e-6, 8e-6, 1.6e-5, 8e-6, 4e-6, 8e-6, 1.6e-5, 2e-6, 1.6e-5, 4e-6, -2e-6]
    )
    areas = np.array([8.0, 4.0, 1.0, 4.0, 2.0, 4.0, 8.0, 8.0, 8.0, 4.0, 2.0, 4.0])
    outcome = emission.partial_power_concentration(intensity, areas)
    assert outcome.subset_positions == 3
    assert outcome.subset_area == pytest.approx(20.0)
    assert outcome.additional_positions == 91


def test_a_source_concentrated_in_one_segment_is_refused_by_name() -> None:
    """Eq. (A.8) has no spread over one position, so Eq. (B.4) is undefined.

    B.1.3 bounds N_alpha from above only, so one segment of twelve carrying
    more than half the power satisfies the condition it states and is the
    archetypal concentrated source. The procedure still cannot be run, and
    saying so beats dividing by ``N_alpha - 1`` and rounding up the NaN.
    """
    areas = np.full(12, 1.0)
    intensity = np.array([2.0e-4] + [1.0e-5] * 11)
    assert intensity[0] > np.sum(intensity[1:])
    with pytest.raises(ValueError, match="concentrated in a single segment"):
        emission.partial_power_concentration(intensity, areas)


def test_a_non_uniform_remainder_exhausts_the_error_budget() -> None:
    """Delta_alpha at or below zero means no number of new positions helps."""
    areas = np.full(12, 1.0)
    intensity = np.array([4.0e-5] * 4 + [2.0e-6] * 8)
    with pytest.raises(ValueError, match="exhaust the ISO 9614-1"):
        emission.partial_power_concentration(intensity, areas)


def test_a_remainder_flowing_inward_has_no_field_indicator() -> None:
    """A.2.3 again, on the remainder: the subset took all the outward flow.

    Three large segments carry the whole power, and the subset that passes
    half of it takes only two of them; what is left is one of those three
    against nine small segments flowing inward, whose algebraic mean normal
    intensity is negative. F4(1 - alpha) is then not defined, and (B.4) has no
    remainder term to subtract.
    """
    areas = np.array([10.0, 10.0, 10.0] + [0.01] * 9)
    intensity = np.array([1.0e-4] * 3 + [-1.2e-5] * 9)
    assert float(np.sum(intensity * areas)) > 0.0
    with pytest.raises(ValueError, match="algebraic mean normal intensity"):
        emission.partial_power_concentration(intensity, areas)


def test_the_concentration_procedure_needs_a_determinable_band() -> None:
    """Clause 9.2 first: a net-negative band has no power to concentrate."""
    areas = np.full(12, 1.0)
    intensity = np.full(12, -1.0e-6)
    with pytest.raises(ValueError, match="not positive"):
        emission.partial_power_concentration(intensity, areas)


def test_a_band_of_exactly_zero_net_power_has_nothing_to_concentrate() -> None:
    """A surface in balance is outside the method too, not merely on its edge."""
    areas = np.full(12, 1.0)
    intensity = np.array([1.0e-5, -1.0e-5] * 6)
    assert float(np.sum(intensity * areas)) == 0.0
    with pytest.raises(ValueError, match="not positive"):
        emission.partial_power_concentration(intensity, areas)


def test_the_concentration_procedure_works_on_one_band() -> None:
    """A per-band procedure refuses a whole spectrum rather than picking a band."""
    areas = np.full(12, 1.0)
    intensity = np.full((12, 2), 1.0e-5)
    with pytest.raises(ValueError, match="one frequency band"):
        emission.partial_power_concentration(intensity, areas)


# ---------------------------------------------------------------------------
# Clause 8.2 sampling warnings
# ---------------------------------------------------------------------------
def _sound_power_warnings(recwarn: pytest.WarningsRecorder) -> list[str]:
    """Every clause 8.2 warning a determination raised, as its text."""
    return [
        str(record.message)
        for record in recwarn
        if issubclass(record.category, emission.SoundPowerWarning)
    ]


def test_fewer_than_ten_positions_warns() -> None:
    """Clause 8.2 asks for a minimum of 10 positions."""
    intensity, areas = _uniform_surface(positions=6)
    with pytest.warns(emission.SoundPowerWarning, match="at least 10"):
        emission.sound_power_intensity_points(intensity, areas)


def test_nine_positions_is_one_short_and_ten_is_not(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """The printed minimum is ten, so the warning stops at ten and not at nine."""
    nine, nine_areas = _uniform_surface(positions=9)
    with pytest.warns(emission.SoundPowerWarning, match="at least 10"):
        emission.sound_power_intensity_points(nine, nine_areas)
    recwarn.clear()
    ten, ten_areas = _uniform_surface(positions=10)
    emission.sound_power_intensity_points(ten, ten_areas)
    assert not _sound_power_warnings(recwarn)


def test_a_density_below_one_position_per_square_metre_warns() -> None:
    """Clause 8.2 asks for at least one position per square metre."""
    areas = np.full(10, 4.0)
    intensity = np.full((10, 1), 1.0e-5)
    with pytest.warns(emission.SoundPowerWarning, match="per square metre"):
        emission.sound_power_intensity_points(intensity, areas)


def test_one_position_per_square_metre_exactly_is_dense_enough(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Clause 8.2 asks for one per square metre, which parity meets."""
    intensity, areas = _uniform_surface(positions=10, area=1.0)
    assert float(np.sum(areas)) == 10.0
    emission.sound_power_intensity_points(intensity, areas)
    assert not _sound_power_warnings(recwarn)


def test_forty_nine_positions_are_short_of_the_clause_8_2_relaxations() -> None:
    """Both relaxations need fifty, so forty-nine over 196 m2 still warns."""
    areas = np.full(49, 4.0)
    intensity = np.full((49, 1), 1.0e-5)
    with pytest.warns(emission.SoundPowerWarning, match="per square metre"):
        emission.sound_power_intensity_points(intensity, areas)


def test_fifty_positions_reach_the_clause_8_2_relaxations(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Both relaxations of clause 8.2 end at 50 positions, so 50 is left alone."""
    areas = np.full(50, 4.0)
    intensity = np.full((50, 1), 1.0e-5)
    emission.sound_power_intensity_points(intensity, areas)
    assert not _sound_power_warnings(recwarn)


def test_two_positions_are_enough_for_the_bessel_corrected_spread() -> None:
    """(A.1) and (A.8) divide by N - 1, so two positions is the fewest that works.

    Clause 8.2 wants ten and the determination says so, but the arithmetic of
    Annex A is defined from two, and a two-position surface must come back with
    an F4 rather than with nothing.
    """
    areas = np.full(2, 1.0)
    intensity = np.array([[1.2e-5], [0.8e-5]])
    with pytest.warns(emission.SoundPowerWarning, match="at least 10"):
        result = emission.sound_power_intensity_points(intensity, areas)
    assert result.f4 is not None
    assert float(result.f4[0]) == pytest.approx(0.2 * math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_an_empty_position_set_is_refused() -> None:
    """A surface with no segments determines nothing."""
    with pytest.raises(ValueError, match="At least one measurement position"):
        emission.sound_power_intensity_points(np.zeros((0, 1)), np.zeros(0))


def test_mismatched_positions_and_areas_are_refused() -> None:
    """One segment per position (3.8), so the two lengths must agree."""
    intensity = np.full((10, 1), 1.0e-5)
    with pytest.raises(ValueError, match="must match the number of segment"):
        emission.sound_power_intensity_points(intensity, np.full(8, 1.0))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_area_is_refused(bad: float) -> None:
    """A segment of zero or negative area is no segment."""
    areas = np.full(10, 1.0)
    areas[3] = bad
    intensity = np.full((10, 1), 1.0e-5)
    with pytest.raises(ValueError, match="must be positive"):
        emission.sound_power_intensity_points(intensity, areas)


def test_a_non_finite_area_is_refused() -> None:
    """NaN passes every bound, and would make the surface area not a number.

    Matched on the whole sentence and not on "must be finite" alone: the
    result's own invariant refuses a non-finite ``surface_area`` in almost the
    same words further downstream, so a looser match would be satisfied by the
    determination running to completion and failing at the end, naming a
    derived quantity rather than the input that was wrong.
    """
    areas = np.full(10, 1.0)
    areas[0] = math.nan
    intensity = np.full((10, 1), 1.0e-5)
    with pytest.raises(ValueError, match="segment 'areas' must be finite"):
        emission.sound_power_intensity_points(intensity, areas)


def test_a_non_finite_intensity_is_refused() -> None:
    """A NaN intensity would turn its partial power into a silent NaN."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 1), 1.0e-5)
    intensity[2, 0] = math.inf
    with pytest.raises(ValueError, match="only finite values"):
        emission.sound_power_intensity_points(intensity, areas)


def test_a_non_finite_pressure_level_is_refused() -> None:
    """The same for the levels F2 and F3 are built from."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 1), 1.0e-5)
    levels = np.full((10, 1), 80.0)
    levels[1, 0] = math.nan
    with pytest.raises(ValueError, match="'pressure_levels' must contain only finite"):
        emission.sound_power_intensity_points(intensity, areas, pressure_levels=levels)


def test_a_one_position_determination_checks_its_levels_itself() -> None:
    """The determination owns this check, and cannot lean on Annex A for it.

    With two positions or more the same sentence comes back from
    :func:`~phonometry.emission.field_indicators`, so removing the check here
    would look harmless. A one-position surface has no spread for Annex A to
    measure and never reaches it, and the levels would go through unexamined.
    """
    with pytest.raises(ValueError, match="'pressure_levels' must contain only"):
        emission.sound_power_intensity_points(
            np.full((1, 1), 1.0e-5),
            np.full(1, 1.0),
            pressure_levels=np.full((1, 1), math.inf),
        )


def test_more_short_time_columns_than_bands_are_refused_too() -> None:
    """F1 is one number per band, so a surplus column is as wrong as a missing one.

    The neighbour above passes fewer columns than there are bands, which any
    length check catches. A surplus is the direction that slips through a check
    written as "too few": the extra F1 would be dropped or, worse, misaligned
    against the bands it is compared with.
    """
    with pytest.raises(ValueError, match="one column per band"):
        emission.sound_power_intensity_points(
            np.full((10, 1), 1.0e-5),
            np.full(10, 1.0),
            temporal_intensity=np.full((5, 2), 1.0e-5),
        )


def test_a_non_finite_residual_index_is_refused() -> None:
    """A NaN delta_pI0 would make criterion 1 quietly false everywhere."""
    intensity, areas = _uniform_surface()
    with pytest.raises(ValueError, match="'pressure_residual_index' must be finite"):
        emission.sound_power_intensity_points(
            intensity,
            areas,
            pressure_levels=np.full((10, 1), 80.0),
            pressure_residual_index=math.nan,
        )


def test_pressure_levels_of_another_shape_are_refused() -> None:
    """The levels are measured at the same positions and bands as the intensity."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 2), 1.0e-5)
    with pytest.raises(ValueError, match="must have shape"):
        emission.sound_power_intensity_points(
            intensity, areas, pressure_levels=np.full((10, 3), 80.0)
        )


def test_frequencies_of_another_length_are_refused() -> None:
    """One band centre per band, or the tables are read at the wrong row."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 2), 1.0e-5)
    with pytest.raises(ValueError, match="one value per band"):
        emission.sound_power_intensity_points(
            intensity, areas, frequencies=[500.0, 1000.0, 2000.0]
        )


def test_a_three_dimensional_intensity_is_refused() -> None:
    """A determination is positions by bands and nothing more."""
    with pytest.raises(ValueError, match="must be 1D"):
        emission.sound_power_intensity_points(np.ones((2, 2, 2)), np.full(2, 1.0))


def test_areas_must_be_one_dimensional() -> None:
    """A segment has one area."""
    intensity = np.full((10, 1), 1.0e-5)
    with pytest.raises(ValueError, match="'areas' must be a 1D array"):
        emission.sound_power_intensity_points(intensity, np.ones((10, 1)))


# ---------------------------------------------------------------------------
# The result's own invariants
# ---------------------------------------------------------------------------
def test_a_result_whose_bands_disagree_is_refused() -> None:
    """A column of the wrong length would be broadcast over the whole spectrum."""
    intensity, areas = _uniform_surface()
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        pressure_levels=np.full((10, 1), 80.0),
        pressure_residual_index=30.0,
    )
    import dataclasses

    with pytest.raises(ValueError, match="one value per band"):
        dataclasses.replace(result, criterion_1=np.array([True, False]))


def test_a_result_with_a_non_finite_surface_area_is_refused() -> None:
    """The surface area is what a report prints beside the boxed level."""
    intensity, areas = _uniform_surface()
    result = emission.sound_power_intensity_points(intensity, areas)
    import dataclasses

    with pytest.raises(ValueError, match="surface_area"):
        dataclasses.replace(result, surface_area=math.nan)


@pytest.mark.filterwarnings("ignore:The A-weighted total sums every")
def test_the_result_plots_as_a_sound_power_spectrum() -> None:
    """The .plot() of the emission results: LW per band, ISO 9614-1 in the title."""
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    areas = np.full(10, 1.0)
    intensity = np.full((10, 3), 1.0e-5)
    result = emission.sound_power_intensity_points(
        intensity, areas, frequencies=[500.0, 1000.0, 2000.0], band_type="octave"
    )
    ax = result.plot()
    assert "ISO 9614-1" in ax.get_title()
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# What A.2.3's refusal is, and is not, about
# ---------------------------------------------------------------------------
#
# Both tests below pin a sentence rather than a number, because both sentences
# were wrong before them and neither error could be caught by a value check.


def test_the_temporal_indicator_survives_a_band_a_2_3_refuses() -> None:
    """F1 is not undefined in a band whose surface mean is not positive.

    Equation (A.1) is the spread of the M short-time samples at one position
    over time. A.2.3 conditions the test on the sign of the *surface* mean and
    says nothing about the temporal samples, so a band it refuses still has a
    perfectly good temporal variability while the three spatial indicators
    have none. The docstring said all four went NaN, and only three do.
    """
    rng = np.random.default_rng(3)
    areas = np.array([5.0] + [0.5] * 9)
    intensity = np.column_stack(
        [np.array([1.0e-4] + [-5.0e-5] * 9), np.full(10, 1.0e-5)]
    )
    temporal = np.tile(intensity[0], (6, 1)) * (
        1.0 + 0.05 * rng.standard_normal((6, 2))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", emission.SoundPowerWarning)
        result = emission.sound_power_intensity_points(
            intensity,
            areas,
            pressure_levels=np.full((10, 2), 70.0),
            pressure_residual_index=12.0,
            temporal_intensity=temporal,
        )
    assert np.all(np.isfinite(np.asarray(result.f1)))
    for indicator in (result.f2, result.f3, result.f4):
        assert math.isnan(float(np.asarray(indicator)[0]))
        assert math.isfinite(float(np.asarray(indicator)[1]))


def test_the_remainder_refusal_does_not_blame_the_whole_surface() -> None:
    """A.2.3 over the remainder is not A.2.3 over the measurement surface.

    When the selected subset carries all of the outward flow, the remainder's
    mean goes negative while the surface's stays positive. That is the
    ordinary shape of a concentrated source, not a failed measurement, so a
    message naming the measurement surface sends the reader to check a
    quantity that is fine. Here the surface mean is +1.6e-5 W/m2.
    """
    areas = np.array([10.0, 10.0, 10.0] + [0.01] * 9)
    intensity = np.array([1.0e-4] * 3 + [-1.2e-5] * 9)
    assert float(np.mean(intensity)) > 0.0
    with pytest.raises(ValueError, match=r"outside the selected subset"):
        emission.partial_power_concentration(intensity, areas)


def test_every_intensity_result_is_captioned_with_its_own_part() -> None:
    """The caption is the only place a figure says which method it came from.

    ``PrecisionIntensityResult`` does not inherit from the Part 2 result, so
    with no test of its own it fell past every branch of the designation
    helper to the enveloping-surface fallback, and an ISO 9614-3 determination
    was captioned with the pressure methods it exists to outrank.
    """
    from phonometry._plot.common import _sound_power_designation
    from phonometry.emission.sound_power_intensity import (
        PrecisionIntensityResult,
        SoundPowerIntensityResult,
    )

    scanning = unittest.mock.Mock(spec=SoundPowerIntensityResult)
    precision = unittest.mock.Mock(spec=PrecisionIntensityResult)
    points = unittest.mock.Mock(spec=emission.DiscretePointIntensityResult)
    assert _sound_power_designation(points) == "ISO 9614-1"
    assert _sound_power_designation(scanning) == "ISO 9614"
    assert _sound_power_designation(precision) == "ISO 9614"
