#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 16283 low-frequency procedure and its three consumers.

**There is no numeric oracle.** ISO 16283-1:2014, ISO 16283-2:2020 and
ISO 16283-3:2016 publish no worked example of this procedure: Annex B of Part 1
and Annex C of Part 2 are blank recording forms and the "Examples" of Annexes D
and E are loudspeaker-position drawings. What stands in for one, and is what
these tests hold the code to:

- **Exact printed numbers.** The trigger is "smaller than 25 m³ (calculated to
  the nearest cubic metre)" and the bands are 50 Hz, 63 Hz and 80 Hz. Both are
  characters on a page, so both are pinned exactly, including the boundary
  case where the tree's round-half-away-from-zero and Python's built-in
  round-half-to-even disagree.
- **Closed forms of Formula (13).** The combination degenerates to ``L`` when
  the corner level equals it, is strictly increasing in the corner level, and
  is bounded below by ``10 lg(2/3) + L``. Its value is also recomputed here
  from the printed expression on a hand case.
- **Formula (12) against its own two readings.** The energy mean over ``q``
  source positions reduces to the maximum over corners at ``q = 1``, and the
  maximum is taken per band, which is the NOTE under the formula.
- **The 63 Hz octave substitution against the tree's own reverberation
  machinery.** NOTE 1 of Clause 10.4 says one-third-octave decays below 100 Hz
  are prone to error for want of modes; run through
  :func:`phonometry.room.room_parameters`, the 63 Hz octave band recovers a
  known single-sloped decay measurably better than the three one-third-octave
  bands do, which is the reason the clause gives, measured rather than
  asserted.
- **One implementation, three parts.** The airborne, impact and facade entry
  points are shown to produce identical low-frequency records from identical
  inputs, which is the whole point of the shared helper.

Nothing here computes an expected value by calling the code under test.
"""

from __future__ import annotations

import functools
import math
import warnings
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

import phonometry as ph
from phonometry import building
from phonometry.building.measurement.low_frequency import (
    LOW_FREQUENCY_BANDS,
    LOW_FREQUENCY_VOLUME_LIMIT,
    LowFrequencyProcedure,
    LowFrequencyWarning,
    apply_low_frequency_procedure,
    corner_level,
    low_frequency_level,
    low_frequency_procedure_applies,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# A five-band measurement: the three low-frequency bands the procedure owns,
# then two the default procedure keeps to itself.
_FREQS = np.array([50.0, 63.0, 80.0, 100.0, 125.0])

# Four corners by three bands. The highest corner is a different one in each
# band (corner 0 at 50 Hz, corner 1 at 63 Hz, corner 2 at 80 Hz), which is the
# case the NOTE under Formula (12) exists for.
_CORNERS = np.array(
    [
        [56.0, 58.0, 54.0],
        [55.0, 60.0, 53.0],
        [54.0, 57.0, 56.0],
        [53.0, 56.0, 55.0],
    ]
)
_CORNER_MAXIMA = np.array([56.0, 60.0, 56.0])

_L2 = np.array([50.0, 52.0, 49.0, 45.0, 44.0])
_L1 = np.array([80.0, 82.0, 79.0, 75.0, 74.0])
_T2 = np.array([0.60, 0.55, 0.50, 0.45, 0.40])

#: A room that triggers, and a 63 Hz octave reverberation time for it.
_SMALL_VOLUME = 18.0
_T63_OCTAVE = 0.72


def _receiving_procedure() -> LowFrequencyProcedure:
    """The receiving-room measurements the three entry points share."""
    return LowFrequencyProcedure(
        volume=_SMALL_VOLUME,
        corner_levels=_CORNERS,
        reverberation_63_octave=_T63_OCTAVE,
    )


def _printed_formula_13(default: float, corner: float) -> float:
    """Formula (13) transcribed from the page, independently of the module."""
    return 10.0 * math.log10(
        (10.0 ** (0.1 * corner) + (2.0 * 10.0 ** (0.1 * default))) / 3.0
    )


# --- The printed trigger (Clause 8.1 / 7.3.1 and Clause 10.4 / 8.4) --------


def test_trigger_limit_is_the_printed_25_cubic_metres() -> None:
    """The constant is the number every clause prints, not a rounded stand-in."""
    assert LOW_FREQUENCY_VOLUME_LIMIT == 25.0


def test_bands_are_the_printed_three() -> None:
    """50 Hz, 63 Hz and 80 Hz, in that order, and nothing else."""
    assert LOW_FREQUENCY_BANDS == (50.0, 63.0, 80.0)


@pytest.mark.parametrize(
    ("volume", "applies"),
    [
        (8.0, True),
        (24.0, True),
        (24.4, True),
        # floor(24,49 + 0,5) = 24, the last volume that still triggers.
        (24.49, True),
        # floor(24,5 + 0,5) = 25: rounds up to the limit, so it does not.
        (24.5, False),
        (24.6, False),
        # "smaller than", strictly: 25 m³ exactly is outside the procedure.
        (25.0, False),
        (25.4, False),
        (40.0, False),
    ],
)
def test_trigger_is_strict_below_25_after_rounding(
    volume: float, applies: bool
) -> None:
    """The printed condition, at every boundary it has."""
    assert low_frequency_procedure_applies(volume) is applies


def test_half_way_volume_diverges_from_the_builtin_round() -> None:
    """24,5 m³ is where the tie rule shows, so the tie rule is pinned here.

    The standards give none, so the tree's own rule applies: half away from
    zero, ``floor(V + 0,5)``, which answers 25 m³ and does not trigger. Python's
    built-in :func:`round` is half-to-even, answers 24 m³, and would.
    """
    assert math.floor(24.5 + 0.5) == 25
    assert round(24.5) == 24
    assert low_frequency_procedure_applies(24.5) is False


@pytest.mark.parametrize("volume", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_volume_is_refused_before_the_rounding(volume: float) -> None:
    """``math.floor`` raises on NaN, so the finiteness test has to run first."""
    with pytest.raises(ValueError, match="positive, finite room volume"):
        low_frequency_procedure_applies(volume)


@pytest.mark.parametrize("volume", [0.0, -1.0])
def test_non_positive_volume_is_refused(volume: float) -> None:
    """A room has a positive volume; zero and negative are not small rooms."""
    with pytest.raises(ValueError, match="positive, finite room volume"):
        low_frequency_procedure_applies(volume)


def test_procedure_refuses_a_room_that_does_not_trigger() -> None:
    """Above the line the standards say nothing, so neither does the library."""
    with pytest.raises(ValueError, match="smaller than 25"):
        LowFrequencyProcedure(volume=30.0, corner_levels=_CORNERS)


# --- Formula (12): the corner sound pressure level ------------------------


def test_corner_level_takes_the_maximum_per_band() -> None:
    """One source position: the highest corner, band by band.

    The maximum lands on a different corner in each of the three bands here,
    which is what the NOTE under Formula (12) allows.
    """
    assert np.array_equal(corner_level(_CORNERS), _CORNER_MAXIMA)


def test_corner_level_is_not_one_corner_for_all_bands() -> None:
    """No single corner reproduces the answer, so the per-band choice is real."""
    for row in _CORNERS:
        assert not np.array_equal(row, _CORNER_MAXIMA)


def test_one_position_is_the_q_equals_one_case_of_formula_12() -> None:
    """A ``(corners, 3)`` sheet and a one-position ``(1, corners, 3)`` agree."""
    single = corner_level(_CORNERS)
    stacked = corner_level(_CORNERS[np.newaxis, ...])
    assert np.allclose(single, stacked, rtol=0.0, atol=1e-12)


def test_corner_level_energy_averages_over_source_positions() -> None:
    """Formula (12) over q = 2 positions, against the printed expression."""
    positions = np.stack([_CORNERS, _CORNERS + 6.0])
    computed = corner_level(positions)
    expected = np.array(
        [
            10.0 * math.log10((10.0 ** (0.1 * m) + 10.0 ** (0.1 * (m + 6.0))) / 2.0)
            for m in _CORNER_MAXIMA
        ]
    )
    assert np.allclose(computed, expected, rtol=0.0, atol=1e-12)


def test_corner_level_rejects_a_rank_it_cannot_read() -> None:
    """One vector of corners with no band axis is not a corner sheet."""
    with pytest.raises(ValueError, match="corner_levels"):
        corner_level(np.array([50.0, 51.0, 52.0, 53.0]))


@pytest.mark.parametrize(
    "shape", [(3,), (2, 2, 4, 3)], ids=["one-vector", "four-dimensional"]
)
def test_the_procedure_refuses_a_rank_it_will_not_be_able_to_read(
    shape: tuple[int, ...],
) -> None:
    """The refusal belongs where the caller built the object, not later."""
    with pytest.raises(ValueError, match="corners x bands"):
        LowFrequencyProcedure(volume=23.0, corner_levels=np.zeros(shape))


def test_corner_level_rejects_an_empty_sheet() -> None:
    """No corner was measured, so there is no maximum to take per band."""
    with pytest.raises(ValueError, match="must not be empty"):
        corner_level(np.empty((0, 3)))


def test_corner_level_rejects_non_finite_values() -> None:
    """A NaN corner would propagate silently into the reported band."""
    corners = _CORNERS.copy()
    corners[0, 0] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        corner_level(corners)


def test_three_corners_warn_about_the_printed_minimum() -> None:
    """Every part asks for four corners; three still computes, and says so."""
    with pytest.warns(LowFrequencyWarning, match="minimum of 4"):
        corner_level(_CORNERS[:3])


def test_four_corners_is_the_printed_minimum_and_says_nothing() -> None:
    """Four corners satisfy "a minimum of four", so a conforming set is quiet.

    The complaint is pinned at three and its wording pins the number, but
    neither says which side of four the comparison falls on. A four-corner set
    is what the clauses ask for, and warning about it would teach the reader to
    filter the warning away.
    """
    assert _CORNERS.shape[0] == 4
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        computed = corner_level(_CORNERS)
    assert np.array_equal(computed, _CORNER_MAXIMA)


# --- Formula (13): the combination ----------------------------------------


def test_combination_matches_the_printed_formula() -> None:
    """A hand case, recomputed from the transcription rather than the module."""
    computed = low_frequency_level(_L2[:3], _CORNER_MAXIMA)
    expected = np.array(
        [
            _printed_formula_13(d, c)
            for d, c in zip(_L2[:3], _CORNER_MAXIMA, strict=True)
        ]
    )
    assert np.allclose(computed, expected, rtol=0.0, atol=1e-12)


def test_combination_degenerates_when_the_corner_equals_the_default() -> None:
    """Weights of 1/3 and 2/3 sum to one, so equal inputs come back unchanged."""
    level = np.array([44.0, 51.5, 62.25])
    assert np.allclose(low_frequency_level(level, level), level, rtol=0.0, atol=1e-12)


def test_combination_is_strictly_increasing_in_the_corner_level() -> None:
    """The corner level can only raise the reported band, never lower it twice."""
    default = np.full(41, 50.0)
    corners = np.linspace(20.0, 90.0, 41)
    combined = low_frequency_level(default, corners)
    assert np.all(np.diff(combined) > 0.0)


def test_combination_is_bounded_below_by_two_thirds_of_the_default() -> None:
    """Even a silent corner leaves 2/3 of the default energy in place."""
    default = 50.0
    floor_db = 10.0 * math.log10(2.0 / 3.0) + default
    combined = low_frequency_level(
        np.full(5, default), np.array([-40.0, -10.0, 0.0, 10.0, 20.0])
    )
    assert np.all(combined > floor_db)
    # And it approaches that floor as the corner level falls away.
    assert low_frequency_level(np.array([default]), np.array([-200.0]))[
        0
    ] == pytest.approx(floor_db, abs=1e-9)


def test_combination_rejects_empty_band_axes() -> None:
    """Formula 13 combines a band with its corner; with no bands there is none."""
    with pytest.raises(ValueError, match="must not be empty"):
        low_frequency_level(np.empty(0), np.empty(0))


@pytest.mark.parametrize("which", ["level", "corner"])
def test_combination_rejects_non_finite_values(which: str) -> None:
    """A NaN on either side would survive the logarithm as a NaN level."""
    default = np.zeros(3)
    highest = np.zeros(3)
    (default if which == "level" else highest)[1] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        low_frequency_level(default, highest)


def test_combination_rejects_mismatched_band_axes() -> None:
    """Pairing three default bands with two corner bands is not a measurement."""
    with pytest.raises(ValueError, match="same bands"):
        low_frequency_level(np.zeros(3), np.zeros(2))


# --- Applying the procedure to a whole band vector ------------------------


def test_only_the_three_low_frequency_bands_change() -> None:
    """100 Hz and up keep the default procedure, untouched."""
    result = apply_low_frequency_procedure(
        _L2, _FREQS, _receiving_procedure(), reverberation_time=_T2
    )
    assert np.array_equal(result.level[3:], _L2[3:])
    assert result.reverberation_time is not None
    assert np.array_equal(result.reverberation_time[3:], _T2[3:])


def test_the_63_hz_octave_value_replaces_all_three_bands() -> None:
    """Clause 10.4: one measured value stands for 50 Hz, 63 Hz and 80 Hz."""
    result = apply_low_frequency_procedure(
        _L2, _FREQS, _receiving_procedure(), reverberation_time=_T2
    )
    assert result.reverberation_time is not None
    assert np.array_equal(result.reverberation_time[:3], np.full(3, _T63_OCTAVE))


def test_exact_band_centres_are_recognised_as_well_as_nominal_ones() -> None:
    """49,6 / 62,5 / 79,4 Hz is the same measurement as 50 / 63 / 80 Hz."""
    exact = np.array([49.6, 62.5, 79.4, 100.0, 125.0])
    nominal = apply_low_frequency_procedure(
        _L2, _FREQS, _receiving_procedure(), reverberation_time=_T2
    )
    measured = apply_low_frequency_procedure(
        _L2, exact, _receiving_procedure(), reverberation_time=_T2
    )
    assert np.allclose(nominal.l_lf, measured.l_lf, rtol=0.0, atol=1e-12)
    assert np.allclose(measured.low_frequency_bands, exact[:3], rtol=0.0, atol=1e-12)


def test_the_three_bands_are_read_where_they_actually_sit() -> None:
    """An analyser sheet that starts below 50 Hz still gets the right columns.

    Every other case here puts 50 / 63 / 80 Hz in the first three columns, so
    reading the leading three would answer them all. A real sheet from a
    two-thirds-octave-wider analyser starts at 31,5 Hz, and then the leading
    three columns are 31,5 / 40 / 50 Hz: the wrong bands, and up to 6 dB out.
    """
    freqs = np.array([31.5, 40.0, 50.0, 63.0, 80.0, 100.0])
    levels = np.array([62.0, 58.0, 50.0, 52.0, 49.0, 45.0])
    times = np.array([0.80, 0.70, 0.60, 0.55, 0.50, 0.45])
    result = apply_low_frequency_procedure(
        levels, freqs, _receiving_procedure(), reverberation_time=times
    )
    assert np.array_equal(result.low_frequency_bands, freqs[2:5])
    assert np.array_equal(result.l_default, levels[2:5])
    expected = np.array(
        [
            _printed_formula_13(d, c)
            for d, c in zip(levels[2:5], _CORNER_MAXIMA, strict=True)
        ]
    )
    assert np.allclose(result.level[2:5], expected, rtol=0.0, atol=1e-12)
    # 31,5 Hz and 40 Hz are below the procedure and 100 Hz above it: the
    # substitution touches neither end.
    assert np.array_equal(result.level[:2], levels[:2])
    assert np.array_equal(result.level[5:], levels[5:])
    assert result.reverberation_time is not None
    assert np.array_equal(result.reverberation_time[:2], times[:2])
    assert np.array_equal(result.reverberation_time[2:5], np.full(3, _T63_OCTAVE))
    assert np.array_equal(result.reverberation_time[5:], times[5:])


def test_the_callers_own_arrays_come_back_unchanged() -> None:
    """The substitution writes into copies, so a reused sheet stays measured.

    ``level`` and ``reverberation_time`` are the caller's own arrays and the
    procedure overwrites three columns of each. Writing in place would edit the
    measurement under a caller who goes on to compute the default-procedure
    spectrum from the same vectors for comparison, which is exactly what the
    documented small-room example does.

    Every array here is built from literals and snapshotted on the spot, not
    copied from the module-level fixtures. An implementation that wrote in
    place would have overwritten those in an earlier test, and comparing
    against them would then pass against the corruption.
    """
    levels = np.array([50.0, 52.0, 49.0, 45.0, 44.0])
    times = np.array([0.60, 0.55, 0.50, 0.45, 0.40])
    freqs = np.array([50.0, 63.0, 80.0, 100.0, 125.0])
    levels_before, times_before = levels.copy(), times.copy()
    apply_low_frequency_procedure(
        levels, freqs, _receiving_procedure(), reverberation_time=times
    )
    assert np.array_equal(levels, levels_before)
    assert np.array_equal(times, times_before)

    source = np.array([80.0, 82.0, 79.0, 75.0, 74.0])
    source_before = source.copy()
    building.airborne_insulation(
        source,
        levels,
        times,
        frequencies=freqs,
        receiver_low_frequency=_receiving_procedure(),
    )
    assert np.array_equal(source, source_before)
    assert np.array_equal(levels, levels_before)
    assert np.array_equal(times, times_before)


def test_a_level_axis_with_more_than_one_dimension_is_refused() -> None:
    """One level per band is the contract; a spectrogram is not that."""
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_low_frequency_procedure(
            _L2[np.newaxis, :], _FREQS, procedure, reverberation_time=_T2
        )


def test_a_missing_low_frequency_band_is_refused() -> None:
    """The procedure is stated for the three together; two is not a reading."""
    freqs = np.array([50.0, 63.0, 100.0, 125.0])
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="missing 80 Hz"):
        apply_low_frequency_procedure(
            _L2[:4], freqs, procedure, reverberation_time=_T2[:4]
        )


def test_a_duplicated_band_centre_is_refused() -> None:
    """Two columns answering to 63 Hz leave the target column undecidable."""
    freqs = np.array([50.0, 62.5, 63.0, 80.0])
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="cannot be identified"):
        apply_low_frequency_procedure(
            _L2[:4], freqs, procedure, reverberation_time=_T2[:4]
        )


def test_level_and_frequency_band_counts_must_agree() -> None:
    """The band axis is what places the procedure, so it has to line up."""
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="band"):
        apply_low_frequency_procedure(
            _L2[:4], _FREQS, procedure, reverberation_time=_T2[:4]
        )


def test_receiving_room_without_the_63_hz_octave_time_is_refused() -> None:
    """Clause 10.4 is a "shall" under the same trigger as Clause 8.1."""
    partial = LowFrequencyProcedure(volume=_SMALL_VOLUME, corner_levels=_CORNERS)
    with pytest.raises(ValueError, match="63 Hz octave"):
        apply_low_frequency_procedure(
            _L2, _FREQS, partial, reverberation_time=_T2, room="receiving"
        )


def test_source_room_carrying_a_63_hz_octave_time_is_refused() -> None:
    """Clause 10.4 speaks about the receiving room and about no other."""
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="source-room call takes neither"):
        apply_low_frequency_procedure(_L2, _FREQS, procedure, room="source")


def test_reverberation_times_must_cover_the_same_bands_as_the_levels() -> None:
    """Substituting by band index needs the two arrays on one band axis."""
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="they must match"):
        apply_low_frequency_procedure(
            _L2, _FREQS, procedure, reverberation_time=_T2[:-1]
        )


def test_receiving_room_without_the_measured_times_is_refused() -> None:
    """Clause 10.4 replaces three values and leaves the rest as measured."""
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="needs 'reverberation_time'"):
        apply_low_frequency_procedure(_L2, _FREQS, procedure)


def test_an_unknown_room_is_refused() -> None:
    """Only two rooms exist in ISO 16283, and only one of them takes 10.4."""
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="'room' must be"):
        apply_low_frequency_procedure(
            _L2, _FREQS, procedure, reverberation_time=_T2, room="kitchen"
        )


def test_corner_sheet_must_carry_exactly_the_three_bands() -> None:
    """Corners are measured at 50 / 63 / 80 Hz only, so the sheet is 3 wide."""
    with pytest.raises(ValueError, match="exactly the three low-frequency bands"):
        LowFrequencyProcedure(volume=_SMALL_VOLUME, corner_levels=np.zeros((4, 5)))


@pytest.mark.parametrize("t63", [0.0, -0.1, float("nan"), float("inf")])
def test_non_positive_or_non_finite_63_hz_octave_time_is_refused(t63: float) -> None:
    """A reverberation time divides and takes a logarithm downstream."""
    with pytest.raises(ValueError, match="positive, finite"):
        LowFrequencyProcedure(
            volume=_SMALL_VOLUME,
            corner_levels=_CORNERS,
            reverberation_63_octave=t63,
        )


# --- Fed to airborne, impact and facade -----------------------------------


def test_all_three_parts_reach_the_same_low_frequency_code() -> None:
    """Identical inputs give identical records through all three entry points.

    Part 1 Formula (13), Part 2 Formula (16) and Part 3 Formula (5) are one
    expression under three sets of subscripts, so one implementation answers
    them; this is the test that says so.
    """
    procedure = _receiving_procedure()
    airborne = building.airborne_insulation(
        _L1, _L2, _T2, frequencies=_FREQS, receiver_low_frequency=procedure
    )
    impact = building.impact_insulation(
        _L2, _T2, frequencies=_FREQS, low_frequency=procedure
    )
    facade = building.facade_insulation(
        _L1, _L2, _T2, frequencies=_FREQS, low_frequency=procedure
    )
    records = [
        airborne.receiver_low_frequency,
        impact.low_frequency,
        facade.low_frequency,
    ]
    assert all(record is not None for record in records)
    reference = records[0]
    assert reference is not None
    for record in records[1:]:
        assert record is not None
        assert np.array_equal(record.l_corner, reference.l_corner)
        assert np.array_equal(record.l_lf, reference.l_lf)
        assert np.array_equal(record.level, reference.level)
        assert record.reverberation_time is not None
        assert reference.reverberation_time is not None
        assert np.array_equal(record.reverberation_time, reference.reverberation_time)


def test_airborne_uses_the_combined_level_and_the_substituted_time() -> None:
    """D and DnT are formed from L_LF and the 63 Hz octave T, band by band."""
    result = building.airborne_insulation(
        _L1, _L2, _T2, frequencies=_FREQS, receiver_low_frequency=_receiving_procedure()
    )
    expected_l2 = np.array(
        [
            _printed_formula_13(d, c)
            for d, c in zip(_L2[:3], _CORNER_MAXIMA, strict=True)
        ]
    )
    assert result.l2 is not None
    assert np.allclose(result.l2[:3], expected_l2, rtol=0.0, atol=1e-12)
    expected_d = _L1[:3] - expected_l2
    assert np.allclose(result.d[:3], expected_d, rtol=0.0, atol=1e-12)
    expected_dnt = expected_d + 10.0 * np.log10(_T63_OCTAVE / 0.5)
    assert np.allclose(result.dnt[:3], expected_dnt, rtol=0.0, atol=1e-12)


def test_airborne_source_room_alone_leaves_the_receiving_side_untouched() -> None:
    """Clause 8.1 tests the two rooms separately; Clause 10.4 only the receiver.

    A small source room beside a large receiving room therefore moves ``L1``
    and leaves both ``L2`` and the one-third-octave reverberation times exactly
    as measured. That asymmetry is printed, not inferred.
    """
    source = LowFrequencyProcedure(volume=_SMALL_VOLUME, corner_levels=_CORNERS + 20.0)
    result = building.airborne_insulation(
        _L1, _L2, _T2, frequencies=_FREQS, source_low_frequency=source
    )
    assert result.receiver_low_frequency is None
    assert result.l2 is not None
    assert np.array_equal(result.l2, _L2)
    assert result.t2 is not None
    assert np.array_equal(result.t2, _T2)
    assert result.l1 is not None
    assert not np.array_equal(result.l1[:3], _L1[:3])
    assert np.array_equal(result.l1[3:], _L1[3:])


def test_airborne_treats_both_rooms_when_both_are_small() -> None:
    """Two procedures, two records, and only the receiver one touches ``t2``."""
    source = LowFrequencyProcedure(volume=20.0, corner_levels=_CORNERS + 20.0)
    result = building.airborne_insulation(
        _L1,
        _L2,
        _T2,
        frequencies=_FREQS,
        source_low_frequency=source,
        receiver_low_frequency=_receiving_procedure(),
    )
    assert result.source_low_frequency is not None
    assert result.receiver_low_frequency is not None
    assert result.source_low_frequency.reverberation_time is None
    assert result.receiver_low_frequency.reverberation_time is not None
    assert result.t2 is not None
    assert np.array_equal(result.t2[:3], np.full(3, _T63_OCTAVE))


def test_impact_level_uses_the_substituted_reverberation_time() -> None:
    """L'nT at 50 / 63 / 80 Hz is formed with the 63 Hz octave value."""
    result = building.impact_insulation(
        _L2, _T2, frequencies=_FREQS, low_frequency=_receiving_procedure()
    )
    expected_li = np.array(
        [
            _printed_formula_13(d, c)
            for d, c in zip(_L2[:3], _CORNER_MAXIMA, strict=True)
        ]
    )
    expected = expected_li - 10.0 * np.log10(_T63_OCTAVE / 0.5)
    assert np.allclose(result.l_n_t[:3], expected, rtol=0.0, atol=1e-12)


def test_facade_refuses_the_procedure_with_a_traffic_source() -> None:
    """ISO 16283-3 Clause 6 confines it to the loudspeaker methods."""
    procedure = _receiving_procedure()
    with pytest.raises(ValueError, match="loudspeaker methods"):
        building.facade_insulation(
            _L1,
            _L2,
            _T2,
            frequencies=_FREQS,
            method="road_traffic",
            low_frequency=procedure,
        )


def test_facade_accepts_the_procedure_with_a_loudspeaker_source() -> None:
    """The method Clause 7.3 names in its own heading."""
    result = building.facade_insulation(
        _L1, _L2, _T2, frequencies=_FREQS, low_frequency=_receiving_procedure()
    )
    assert result.low_frequency is not None


def test_facade_level_difference_uses_the_combined_level() -> None:
    """D2m at 50 / 63 / 80 Hz is formed from L_LF, not from the default level.

    The airborne and impact entry points each have a test that follows the
    combined level all the way into the reported quantity; this is the facade
    one. Without it, dropping the assignment that feeds ``l2_bands`` back from
    the record leaves the whole suite green while the facade is reported up to
    4 dB better insulated than ISO 16283-3 says it is.
    """
    result = building.facade_insulation(
        _L1, _L2, _T2, frequencies=_FREQS, low_frequency=_receiving_procedure()
    )
    expected_l2 = np.array(
        [
            _printed_formula_13(d, c)
            for d, c in zip(_L2[:3], _CORNER_MAXIMA, strict=True)
        ]
    )
    expected_d = _L1[:3] - expected_l2
    assert np.allclose(result.d_2m[:3], expected_d, rtol=0.0, atol=1e-12)
    expected_nt = expected_d + 10.0 * np.log10(_T63_OCTAVE / 0.5)
    assert np.allclose(result.d_2m_nt[:3], expected_nt, rtol=0.0, atol=1e-12)
    # The bands above the procedure keep the measured level and the measured
    # reverberation time, so the substitution is confined to the three.
    assert np.allclose(result.d_2m[3:], _L1[3:] - _L2[3:], rtol=0.0, atol=1e-12)


def _without_frequencies(part: str) -> Callable[[], object]:
    """One entry point called with a procedure and no band centres."""
    procedure = _receiving_procedure()
    calls: dict[str, Callable[[], object]] = {
        "airborne": functools.partial(
            building.airborne_insulation,
            _L1,
            _L2,
            _T2,
            receiver_low_frequency=procedure,
        ),
        "impact": functools.partial(
            building.impact_insulation, _L2, _T2, low_frequency=procedure
        ),
        "facade": functools.partial(
            building.facade_insulation, _L1, _L2, _T2, low_frequency=procedure
        ),
    }
    return calls[part]


def _with_disagreeing_volume(part: str) -> Callable[[], object]:
    """One entry point given 20 m³ beside a procedure that says 18 m³."""
    procedure = _receiving_procedure()
    calls: dict[str, Callable[[], object]] = {
        "airborne": functools.partial(
            building.airborne_insulation,
            _L1,
            _L2,
            _T2,
            area=10.0,
            volume=20.0,
            frequencies=_FREQS,
            receiver_low_frequency=procedure,
        ),
        "impact": functools.partial(
            building.impact_insulation,
            _L2,
            _T2,
            volume=20.0,
            frequencies=_FREQS,
            low_frequency=procedure,
        ),
        "facade": functools.partial(
            building.facade_insulation,
            _L1,
            _L2,
            _T2,
            volume=20.0,
            frequencies=_FREQS,
            low_frequency=procedure,
        ),
    }
    return calls[part]


@pytest.mark.parametrize("part", ["airborne", "impact", "facade"])
def test_frequencies_are_required_with_the_procedure(part: str) -> None:
    """Without band centres the three columns to rewrite cannot be found."""
    call = _without_frequencies(part)
    with pytest.raises(ValueError, match="'frequencies' must be given"):
        call()


@pytest.mark.parametrize("part", ["airborne", "impact", "facade"])
def test_a_disagreeing_volume_is_refused(part: str) -> None:
    """The Sabine area and the trigger describe the same receiving room."""
    call = _with_disagreeing_volume(part)
    with pytest.raises(ValueError, match="must agree"):
        call()


def test_matching_volume_is_accepted_and_normalises_against_it() -> None:
    """The same number twice is not a conflict."""
    result = building.impact_insulation(
        _L2,
        _T2,
        volume=_SMALL_VOLUME,
        frequencies=_FREQS,
        low_frequency=_receiving_procedure(),
    )
    assert result.l_n is not None
    absorption = 0.16 * _SMALL_VOLUME / _T63_OCTAVE
    assert result.li is not None
    expected = result.li[:3] + 10.0 * np.log10(absorption / 10.0)
    assert np.allclose(result.l_n[:3], expected, rtol=0.0, atol=1e-12)


def test_without_the_procedure_nothing_changes() -> None:
    """The default path is untouched, so the new arguments are additions only."""
    plain = building.airborne_insulation(_L1, _L2, _T2)
    with_freqs = building.airborne_insulation(_L1, _L2, _T2, frequencies=_FREQS)
    assert np.array_equal(plain.dnt, with_freqs.dnt)
    assert plain.source_low_frequency is None
    assert plain.receiver_low_frequency is None


# --- The room that triggers and the caller who has not noticed ------------
#
# Clause 8.1 (Part 3: 7.3.1) is a "shall", so a room under the line answered
# from the default procedure alone is not the ISO 16283 quantity at 50, 63 and
# 80 Hz. When the volume and the band centres are both in hand the library can
# see that for itself, and says so.

#: The bedroom of the low-frequency-procedure page: 3,6 m by 2,7 m by 2,4 m,
#: which is 23 m³ to the nearest cubic metre, measured over the 16 core bands
#: and the optional low range of Clause 5.
_BEDROOM_VOLUME = 23.328
_BEDROOM_FREQS = np.array(
    [50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
     500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0]
)  # fmt: skip
_BEDROOM_L1 = np.array(
    [88.6, 90.4, 89.1, 87.3, 88.0, 87.4, 86.9, 86.5, 86.2, 85.8,
     85.5, 85.1, 84.7, 84.2, 83.6, 82.9, 82.1, 81.2, 80.1]
)  # fmt: skip
_BEDROOM_L2 = np.array(
    [54.7, 57.9, 53.2, 49.6, 47.1, 44.3, 41.0, 38.2, 35.6, 33.1,
     31.0, 29.2, 27.6, 26.1, 24.9, 23.8, 23.0, 22.4, 22.1]
)  # fmt: skip
_BEDROOM_T2 = np.array(
    [0.74, 0.69, 0.63, 0.58, 0.55, 0.53, 0.51, 0.50, 0.49, 0.48,
     0.47, 0.46, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40]
)  # fmt: skip
#: Eight corners of that bedroom, four at each of two loudspeaker positions.
_BEDROOM_CORNERS = np.array(
    [
        [[60.2, 63.8, 58.4], [58.9, 65.1, 57.2],
         [61.4, 62.6, 60.1], [57.8, 64.2, 59.3]],
        [[59.6, 64.9, 59.8], [60.8, 63.4, 58.1],
         [58.3, 65.6, 60.7], [61.1, 62.9, 57.6]],
    ]
)  # fmt: skip
_BEDROOM_T63_OCTAVE = 0.66


def test_the_documented_bedroom_does_not_answer_the_three_bands_silently() -> None:
    """The 23 m³ room of the guide page, called the way a reader would call it.

    Volume and band centres are both supplied, both for reasons that have
    nothing to do with the low-frequency procedure: the volume sizes the Sabine
    absorption area for ``R'`` and the band centres label the spectrum. Between
    them they are enough to know that Clause 8.1 is in force.

    The size of the gap is what makes the silence unacceptable, so it is pinned
    here too. Answering those three bands from the default procedure alone puts
    ``DnT`` nearly 4 dB above the ISO 16283 quantity, which moves the
    enlarged-range adaptation term the whole way from -1 dB to -2 dB. The
    weighted rating does not move, because ISO 717-1 reads it from 100 Hz up
    and never sees these bands.
    """
    with pytest.warns(LowFrequencyWarning, match="low-frequency procedure"):
        plain = building.airborne_insulation(
            _BEDROOM_L1,
            _BEDROOM_L2,
            _BEDROOM_T2,
            area=6.48,
            volume=_BEDROOM_VOLUME,
            frequencies=_BEDROOM_FREQS,
        )
    procedure = LowFrequencyProcedure(
        volume=_BEDROOM_VOLUME,
        corner_levels=_BEDROOM_CORNERS,
        reverberation_63_octave=_BEDROOM_T63_OCTAVE,
    )
    field = building.airborne_insulation(
        _BEDROOM_L1,
        _BEDROOM_L2,
        _BEDROOM_T2,
        area=6.48,
        volume=_BEDROOM_VOLUME,
        frequencies=_BEDROOM_FREQS,
        receiver_low_frequency=procedure,
    )
    gap = plain.dnt[:3] - field.dnt[:3]
    assert np.all(gap > 3.5)
    assert np.all(gap < 4.5)
    assert np.array_equal(plain.dnt[3:], field.dnt[3:])

    plain_rating = building.weighted_rating_extended(plain.dnt, _BEDROOM_FREQS)
    field_rating = building.weighted_rating_extended(field.dnt, _BEDROOM_FREQS)
    assert plain_rating.rating == field_rating.rating
    assert plain_rating.c_50_3150 == -1
    assert field_rating.c_50_3150 == -2


def _room_that_triggers(part: str) -> Callable[[], object]:
    """One entry point handed a small room, the three bands, and no procedure."""
    calls: dict[str, Callable[[], object]] = {
        "airborne": functools.partial(
            building.airborne_insulation,
            _L1,
            _L2,
            _T2,
            area=10.0,
            volume=_SMALL_VOLUME,
            frequencies=_FREQS,
        ),
        "impact": functools.partial(
            building.impact_insulation,
            _L2,
            _T2,
            volume=_SMALL_VOLUME,
            frequencies=_FREQS,
        ),
        "facade": functools.partial(
            building.facade_insulation,
            _L1,
            _L2,
            _T2,
            volume=_SMALL_VOLUME,
            frequencies=_FREQS,
        ),
    }
    return calls[part]


@pytest.mark.parametrize("part", ["airborne", "impact", "facade"])
def test_all_three_parts_say_the_procedure_is_required(part: str) -> None:
    """One clause, three entry points, one warning."""
    call = _room_that_triggers(part)
    with pytest.warns(LowFrequencyWarning, match="rounds to 18 m³"):
        call()


def test_a_partial_low_range_is_warned_about_by_its_own_bands() -> None:
    """The warning says what was measured, and advice that would be refused.

    With only 50 Hz and 63 Hz named, the room is as much outside ISO 16283 as
    with all three, so the warning fires; but a message claiming all three
    bands were named would be false, and telling the caller to pass a
    ``LowFrequencyProcedure`` would send them straight into the refusal of the
    band-axis check, which runs on the three together. The message names the
    two bands and says to complete the set first.
    """
    freqs = np.array([50.0, 63.0, 100.0, 125.0, 160.0])
    with pytest.warns(
        LowFrequencyWarning, match=r"names the 50 Hz and 63 Hz bands"
    ) as caught:
        building.impact_insulation(_L2, _T2, volume=_SMALL_VOLUME, frequencies=freqs)
    assert "complete the low range to 50 Hz, 63 Hz and 80 Hz" in str(caught[0].message)


def test_a_full_low_range_is_told_to_pass_the_procedure_directly() -> None:
    """With all three bands named, the advice is the procedure itself."""
    with pytest.warns(LowFrequencyWarning) as caught:
        building.impact_insulation(_L2, _T2, volume=_SMALL_VOLUME, frequencies=_FREQS)
    message = str(caught[0].message)
    assert "names the 50 Hz, 63 Hz and 80 Hz bands" in message
    assert "complete the low range" not in message


def test_a_wrong_length_band_axis_is_refused_before_the_warning() -> None:
    """A vector that does not describe the measured bands decides nothing.

    Five measured bands and a three-entry vector naming the low range: without
    the entry check the warning would fire keyed on columns the measurement
    does not have.
    """
    with pytest.raises(
        ValueError, match=r"'frequencies' must carry one band centre per"
    ):
        building.impact_insulation(
            _L2, _T2, volume=_SMALL_VOLUME, frequencies=[50.0, 63.0, 80.0]
        )


def _room_that_triggers_with_the_procedure(part: str) -> Callable[[], object]:
    """The same three calls, with the procedure the warning asks for."""
    procedure = _receiving_procedure()
    calls: dict[str, Callable[[], object]] = {
        "airborne": functools.partial(
            building.airborne_insulation,
            _L1,
            _L2,
            _T2,
            area=10.0,
            volume=_SMALL_VOLUME,
            frequencies=_FREQS,
            receiver_low_frequency=procedure,
        ),
        "impact": functools.partial(
            building.impact_insulation,
            _L2,
            _T2,
            volume=_SMALL_VOLUME,
            frequencies=_FREQS,
            low_frequency=procedure,
        ),
        "facade": functools.partial(
            building.facade_insulation,
            _L1,
            _L2,
            _T2,
            volume=_SMALL_VOLUME,
            frequencies=_FREQS,
            low_frequency=procedure,
        ),
    }
    return calls[part]


@pytest.mark.parametrize("part", ["airborne", "impact", "facade"])
def test_the_caller_who_ran_the_procedure_is_not_warned(part: str) -> None:
    """The warning exists to be answerable; answering it silences it."""
    call = _room_that_triggers_with_the_procedure(part)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        call()


def test_a_volume_the_library_was_not_given_raises_nothing() -> None:
    """Without a volume there is no trigger to test, so there is nothing to say.

    ``volume`` is optional on all three entry points, and a caller who omits it
    has not told the library which side of 25 m³ the room is on. Guessing would
    warn about every measurement ever made.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        building.airborne_insulation(_L1, _L2, _T2, frequencies=_FREQS)
        building.impact_insulation(_L2, _T2, frequencies=_FREQS)
        building.facade_insulation(_L1, _L2, _T2, frequencies=_FREQS)


def test_a_room_at_or_above_the_trigger_raises_nothing() -> None:
    """25 m³ takes the default procedure, and the default procedure is right."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        building.impact_insulation(
            _L2, _T2, volume=LOW_FREQUENCY_VOLUME_LIMIT, frequencies=_FREQS
        )


def test_a_measurement_without_the_low_range_raises_nothing() -> None:
    """The 16 core bands alone are a complete ISO 16283 measurement.

    Clause 5 (Part 2: 5.1) makes 50 Hz to 80 Hz optional. A caller who did not
    measure them has nothing for the procedure to rewrite, however small the
    room, so the trigger never fires.
    """
    core = np.array([100.0, 125.0, 160.0, 200.0, 250.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        building.impact_insulation(_L2, _T2, volume=_SMALL_VOLUME, frequencies=core)


def test_an_ambiguous_band_axis_raises_nothing() -> None:
    """Two columns answering to 63 Hz are a different complaint, not this one.

    A procedure passed with this band vector is refused by name, because the
    column to rewrite cannot be identified. Warning that one is missing would
    send the caller straight at that error.
    """
    freqs = np.array([50.0, 62.5, 63.0, 80.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        building.impact_insulation(
            _L2[:4], _T2[:4], volume=_SMALL_VOLUME, frequencies=freqs
        )


def test_unnamed_bands_raise_nothing() -> None:
    """Without ``frequencies`` no column can be said to be 50, 63 or 80 Hz."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        building.impact_insulation(_L2, _T2, volume=_SMALL_VOLUME)


def test_a_band_axis_with_an_extra_dimension_is_refused_on_entry() -> None:
    """A band vector has to be one, and a ``(1, bands)`` array is not.

    This used to pass in silence: the warning's rank test kept it from firing,
    and nothing else looked at the axis. Silence was the wrong answer, because
    the vector exists to describe the measured bands and this one does not, so
    the entry points refuse it by name before either use.
    """
    freqs = np.array([[50.0, 63.0, 80.0, 100.0, 125.0]])
    with pytest.raises(
        ValueError, match=r"'frequencies' must carry one band centre per"
    ):
        building.impact_insulation(_L2, _T2, volume=_SMALL_VOLUME, frequencies=freqs)


def test_a_traffic_facade_measurement_raises_nothing() -> None:
    """ISO 16283-3 Clause 6: with traffic, the default procedure is the whole of it.

    "For the element and global road traffic methods, only the default
    procedure shall be used", so a small room measured against traffic is
    conforming without corners, and warning would be wrong rather than noisy.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        building.facade_insulation(
            _L1,
            _L2,
            _T2,
            volume=_SMALL_VOLUME,
            frequencies=_FREQS,
            method="road_traffic",
        )


def test_a_small_source_room_alone_still_leaves_the_receiver_warned() -> None:
    """The two rooms are tested separately, and only the receiver's volume is known.

    ``airborne_insulation`` takes one volume, the receiving room's, so that is
    the room it can speak about. Treating the source room does not answer
    Clause 8.1 for the receiving one.
    """
    source = LowFrequencyProcedure(volume=_SMALL_VOLUME, corner_levels=_CORNERS + 20.0)
    with pytest.warns(LowFrequencyWarning, match="receiver_low_frequency"):
        building.airborne_insulation(
            _L1,
            _L2,
            _T2,
            area=10.0,
            volume=_SMALL_VOLUME,
            frequencies=_FREQS,
            source_low_frequency=source,
        )


def test_a_volume_that_is_not_a_volume_leaves_the_complaint_to_its_owner() -> None:
    """A non-finite volume is somebody else's error, and never this warning.

    ``low_frequency_procedure_applies`` raises on NaN rather than answering, so
    the trigger test has to be reached only for a volume that is one. The
    facade entry point accepts a non-finite volume today and reports NaN; that
    is a separate defect and this check must not turn it into a crash.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowFrequencyWarning)
        result = building.facade_insulation(
            _L1, _L2, _T2, volume=float("nan"), frequencies=_FREQS
        )
    assert result.d_2m_n is not None


# --- Why the 63 Hz octave band, measured rather than asserted -------------


def test_63_hz_octave_recovers_a_known_decay_better_than_the_thirds() -> None:
    """NOTE 1 of Clause 10.4, reproduced with the tree's own decay analysis.

    The clause says one-third-octave decay curves below 100 Hz are prone to
    error because a single-slope decay needs many modes, and that the wider
    63 Hz octave filter partly resolves it. Filtering a synthetic
    single-sloped decay of known ``T60`` through
    :func:`phonometry.room.room_parameters` shows the same thing as a number:
    over a set of realisations the octave-band ``T20`` sits closer to the truth
    than the 50 Hz, 63 Hz and 80 Hz one-third-octave estimates, both on average
    and at worst. This is not an oracle for the substitution, which is a
    prescription about what to measure; it is the reason the clause gives,
    measured on this tree.
    """
    fs = 48000
    t60 = 0.8
    decay = 6.0 * math.log(10.0)
    n = int(4.0 * t60 * fs)
    envelope = np.exp(-0.5 * decay * np.arange(n) / fs / t60)

    octave_error: list[float] = []
    third_error: list[float] = []
    for seed in range(8):
        ir = np.random.default_rng(seed).standard_normal(n) * envelope
        # The bank needs two centres to bracket a range, so the 125 Hz band
        # comes along and is discarded; index 0 is the 63 Hz octave.
        octave = ph.room.room_parameters(ir, fs, limits=(63.0, 125.0), fraction=1)
        thirds = ph.room.room_parameters(ir, fs, limits=(50.0, 80.0), fraction=3)
        octave_error.append(abs(float(octave.t20[0]) - t60))
        third_error.extend(abs(float(v) - t60) for v in thirds.t20)

    assert float(np.mean(octave_error)) < float(np.mean(third_error))
    assert max(octave_error) < max(third_error)


# --- The result object ----------------------------------------------------


def test_result_carries_the_three_band_chain() -> None:
    """The record keeps what it was built from, not only what it produced."""
    result = apply_low_frequency_procedure(
        _L2, _FREQS, _receiving_procedure(), reverberation_time=_T2
    )
    assert np.array_equal(result.l_default, _L2[:3])
    assert np.array_equal(result.l_corner, _CORNER_MAXIMA)
    assert np.array_equal(result.low_frequency_bands, _FREQS[:3])
    assert result.volume == pytest.approx(_SMALL_VOLUME)
    assert result.reverberation_63_octave == pytest.approx(_T63_OCTAVE)


def test_result_plots() -> None:
    """Every result class exposes ``.plot()``; this one draws three curves."""
    result = apply_low_frequency_procedure(
        _L2, _FREQS, _receiving_procedure(), reverberation_time=_T2
    )
    ax = result.plot()
    assert len(ax.lines) == 3
    ax.figure.clf()


def test_result_plots_in_spanish() -> None:
    """The Spanish twin is a peer, so the renderer answers to it too."""
    result = apply_low_frequency_procedure(
        _L2, _FREQS, _receiving_procedure(), reverberation_time=_T2
    )
    ax = result.plot(language="es")
    assert "Nivel de presión" in ax.get_ylabel()
    ax.figure.clf()
