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


def test_a_missing_low_frequency_band_is_refused() -> None:
    """The procedure is stated for the three together; two is not a reading."""
    freqs = np.array([50.0, 63.0, 100.0, 125.0])
    with pytest.raises(ValueError, match="missing 80 Hz"):
        apply_low_frequency_procedure(
            _L2[:4], freqs, _receiving_procedure(), reverberation_time=_T2[:4]
        )


def test_a_duplicated_band_centre_is_refused() -> None:
    """Two columns answering to 63 Hz leave the target column undecidable."""
    freqs = np.array([50.0, 62.5, 63.0, 80.0])
    with pytest.raises(ValueError, match="cannot be identified"):
        apply_low_frequency_procedure(
            _L2[:4], freqs, _receiving_procedure(), reverberation_time=_T2[:4]
        )


def test_level_and_frequency_band_counts_must_agree() -> None:
    """The band axis is what places the procedure, so it has to line up."""
    with pytest.raises(ValueError, match="band"):
        apply_low_frequency_procedure(
            _L2[:4], _FREQS, _receiving_procedure(), reverberation_time=_T2[:4]
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


def test_receiving_room_without_the_measured_times_is_refused() -> None:
    """Clause 10.4 replaces three values and leaves the rest as measured."""
    with pytest.raises(ValueError, match="needs 'reverberation_time'"):
        apply_low_frequency_procedure(_L2, _FREQS, _receiving_procedure())


def test_an_unknown_room_is_refused() -> None:
    """Only two rooms exist in ISO 16283, and only one of them takes 10.4."""
    with pytest.raises(ValueError, match="'room' must be"):
        apply_low_frequency_procedure(
            _L2, _FREQS, _receiving_procedure(), reverberation_time=_T2, room="kitchen"
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
    with pytest.raises(ValueError, match="loudspeaker methods"):
        building.facade_insulation(
            _L1,
            _L2,
            _T2,
            frequencies=_FREQS,
            method="road_traffic",
            low_frequency=_receiving_procedure(),
        )


def test_facade_accepts_the_procedure_with_a_loudspeaker_source() -> None:
    """The method Clause 7.3 names in its own heading."""
    result = building.facade_insulation(
        _L1, _L2, _T2, frequencies=_FREQS, low_frequency=_receiving_procedure()
    )
    assert result.low_frequency is not None


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
