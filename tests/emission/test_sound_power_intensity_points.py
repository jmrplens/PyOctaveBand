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


# ---------------------------------------------------------------------------
# Table B.3 action codes, one case per code (Figure B.1's order)
# ---------------------------------------------------------------------------
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


@pytest.mark.filterwarnings("ignore:The A-weighted total sums every")
def test_expanded_uncertainty_is_twice_the_table_2_standard_deviation() -> None:
    """Table 2 footnote 1: the true level lies within +/- 2s at 95 %."""
    areas = np.full(10, 1.0)
    intensity = np.full((10, 3), 1.0e-5)
    frequencies = [125.0, 500.0, 2000.0]
    result = emission.sound_power_intensity_points(
        intensity,
        areas,
        frequencies=frequencies,
        band_type="octave",
        grade="engineering",
    )
    assert result.expanded_uncertainty is not None
    np.testing.assert_allclose(result.expanded_uncertainty, [6.0, 4.0, 3.0])


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


def test_a_spread_out_field_has_no_concentration_to_exploit() -> None:
    """B.1.3 requires the subset to be fewer than half the segments."""
    areas = np.full(10, 1.0)
    intensity = np.full(10, 1.0e-5)
    with pytest.raises(ValueError, match="carries more than half the sound power"):
        emission.partial_power_concentration(intensity, areas)


def test_a_non_uniform_remainder_exhausts_the_error_budget() -> None:
    """Delta_alpha at or below zero means no number of new positions helps."""
    areas = np.full(12, 1.0)
    intensity = np.array([4.0e-5] * 4 + [2.0e-6] * 8)
    with pytest.raises(ValueError, match="exhaust the ISO 9614-1"):
        emission.partial_power_concentration(intensity, areas)


def test_the_concentration_procedure_needs_a_determinable_band() -> None:
    """Clause 9.2 first: a net-negative band has no power to concentrate."""
    areas = np.full(12, 1.0)
    intensity = np.full(12, -1.0e-6)
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
def test_fewer_than_ten_positions_warns() -> None:
    """Clause 8.2 asks for a minimum of 10 positions."""
    intensity, areas = _uniform_surface(positions=6)
    with pytest.warns(emission.SoundPowerWarning, match="at least 10"):
        emission.sound_power_intensity_points(intensity, areas)


def test_a_density_below_one_position_per_square_metre_warns() -> None:
    """Clause 8.2 asks for at least one position per square metre."""
    areas = np.full(10, 4.0)
    intensity = np.full((10, 1), 1.0e-5)
    with pytest.warns(emission.SoundPowerWarning, match="per square metre"):
        emission.sound_power_intensity_points(intensity, areas)


def test_fifty_positions_reach_the_clause_8_2_relaxations(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Both relaxations of clause 8.2 end at 50 positions, so 50 is left alone."""
    areas = np.full(50, 4.0)
    intensity = np.full((50, 1), 1.0e-5)
    emission.sound_power_intensity_points(intensity, areas)
    assert not [
        record
        for record in recwarn
        if issubclass(record.category, emission.SoundPowerWarning)
    ]


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
    """NaN passes every bound, and would make the surface area not a number."""
    areas = np.full(10, 1.0)
    areas[0] = math.nan
    intensity = np.full((10, 1), 1.0e-5)
    with pytest.raises(ValueError, match="must be finite"):
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
