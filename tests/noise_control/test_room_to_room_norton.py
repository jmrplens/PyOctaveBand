#  Copyright (c) 2026. Jose Manuel Requena Plens
"""End-to-end tests of the room-to-room chain against Norton & Karczub's answers.

Oracle: Norton, M. P. & Karczub, D. G., *Fundamentals of Noise and Vibration
Analysis for Engineers*, 2nd ed. (Cambridge University Press, 2003), the
problems of Chapter 4 with their published answers:

* **problem 4.16** (printed pp. 584-585, answer p. 613): the transmission loss
  a lined enclosure over a refrigeration compressor needs so the reverberant
  level in the room drops to an NC-45 curve, octave by octave from 63 Hz to
  8 kHz;
* **problem 4.18** (printed pp. 585-586, answer p. 613): a blower in a plant
  room, through the separating wall, into an adjacent operator room, as octave
  band sound pressure levels and their A-weighted values;
* **problem 4.21** (printed pp. 586-587, answer p. 614): the octave-band noise
  reduction three different partitions deliver into the same receiving room.

Every expected value here is a number printed in the book's answer section, and
every input is the one printed in the book's problem statement. Nothing is read
back from the library.

The chain exercised is the composition the library did not have: the reverberant
level in the source room (``steady_state_spl`` over ``room_constant``), the
transmission loss of the partition, the equivalent absorption area of the
receiving room (``equivalent_absorption_area``) and the room criterion
(``noise_criterion``).
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import noise_control, room

# ---------------------------------------------------------------------------
# Problem 4.21 - noise reduction of three partitions (printed pp. 586-587).
# ---------------------------------------------------------------------------

#: Octave bands of problems 4.18 and 4.21 (the printed 4.21 header carries a
#: typographical "5000" where the band is 500 Hz).
_BANDS_6 = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])

#: Receiving room of problem 4.21: 8 m wide, 9 m long, 3 m high, with the
#: printed absorption coefficients of the walls, the floor and the ceiling.
_R421_WALLS = np.array([0.04, 0.04, 0.09, 0.15, 0.17, 0.23])
_R421_FLOOR = np.array([0.02, 0.06, 0.14, 0.37, 0.60, 0.66])
_R421_CEILING = np.array([0.30, 0.20, 0.15, 0.05, 0.05, 0.05])

#: Printed transmission loss and printed noise reduction of the three
#: partitions of problem 4.21.
_R421_CASES = (
    (
        "two 13 mm wallboards separated by a 64 mm air gap",
        [18.0, 27.0, 37.0, 45.0, 43.0, 39.0],
        [18.5, 26.8, 38.0, 47.8, 47.3, 43.9],
    ),
    (
        "125 mm plastered brick wall",
        [36.0, 36.0, 40.0, 46.0, 54.0, 57.0],
        [36.5, 35.8, 41.0, 48.8, 58.3, 61.9],
    ),
    (
        "double brick wall, 50 mm cavity, 100 mm plastered bricks",
        [37.0, 41.0, 48.0, 60.0, 61.0, 61.0],
        [37.5, 40.8, 49.0, 62.8, 65.3, 65.9],
    ),
)


def _receiving_absorption_421() -> np.ndarray:
    """Equivalent absorption area of the 8 x 9 x 3 m receiving room, m2."""
    walls = 2.0 * (8.0 * 3.0) + 2.0 * (9.0 * 3.0)
    return np.asarray(
        room.equivalent_absorption_area(
            [
                (walls, _R421_WALLS),
                (8.0 * 9.0, _R421_FLOOR),
                (8.0 * 9.0, _R421_CEILING),
            ]
        ),
        dtype=np.float64,
    )


@pytest.mark.parametrize(("name", "tl", "printed"), _R421_CASES)
def test_problem_4_21_noise_reduction(
    name: str, tl: list[float], printed: list[float]
) -> None:
    """Problem 4.21 (answer p. 614): NR of three partitions, six octave bands.

    ``NR = TL - 10 lg[S_w / (S_2 alpha_2)]`` with the partition the 8 m x 3 m
    wall of the receiving room. The printed answers carry one decimal, and the
    chain reproduces all eighteen values to 0.05 dB.
    """
    result = noise_control.room_to_room_transmission(
        _BANDS_6,
        tl,
        8.0 * 3.0,
        _receiving_absorption_421(),
        source=noise_control.SourceRoom(level=90.0),
        label=name,
    )
    assert np.allclose(result.noise_reduction, printed, atol=0.05)
    # The noise reduction is not the transmission loss: it is larger wherever
    # the receiving room absorbs more than the partition area, smaller where it
    # absorbs less. Both happen in these three columns.
    assert np.any(result.noise_reduction > np.asarray(tl))
    assert np.any(result.noise_reduction < np.asarray(tl))
    # L_p2 = L_p1 - NR, exactly.
    assert np.allclose(result.received_level, 90.0 - result.noise_reduction)


def test_problem_4_21_partition_transmission_term_is_negligible() -> None:
    """The ``tau S_w`` term of Equation (4.101) shifts nothing at these TLs.

    Norton prints Equation (4.101) with the power the partition passes back
    included in the receiving-room absorption. His own answers to problem 4.21
    omit it, and at a transmission loss of 18 dB or more the term is worth less
    than 0.1 dB, which is why the library leaves it off by default.
    """
    _, tl, printed = _R421_CASES[0]
    with_term = noise_control.room_to_room_transmission(
        _BANDS_6,
        tl,
        8.0 * 3.0,
        _receiving_absorption_421(),
        source=noise_control.SourceRoom(level=90.0),
        include_partition_transmission=True,
    )
    assert np.allclose(with_term.noise_reduction, printed, atol=0.1)
    assert np.max(np.abs(with_term.noise_reduction - np.asarray(printed))) > 0.05


# ---------------------------------------------------------------------------
# Problem 4.18 - plant room to operator room (printed pp. 585-586).
# ---------------------------------------------------------------------------

#: Printed data of problem 4.18: blower sound power level, the ceiling, floor
#: and wall absorption of the plant room, the transmission loss of the
#: separating wall and the absorption of the carpeted operator-room floor.
_R418_LW = np.array([105.0, 103.0, 98.0, 108.0, 107.0, 109.0])
_R418_CEILING = np.array([0.07, 0.20, 0.40, 0.52, 0.60, 0.67])
_R418_PLANT_FLOOR = np.array([0.01, 0.01, 0.015, 0.02, 0.02, 0.02])
_R418_WALLS = np.array([0.03, 0.03, 0.03, 0.04, 0.05, 0.07])
_R418_TL = np.array([39.0, 42.0, 50.0, 58.0, 63.0, 67.0])
_R418_CARPET = np.array([0.08, 0.24, 0.57, 0.69, 0.71, 0.73])

#: Printed answers (p. 613): octave-band levels in the operator room and their
#: A-weighted values.
_R418_LP2 = np.array([72.3, 60.4, 41.4, 41.0, 33.8, 30.7])
_R418_LPA = np.array([56.2, 51.5, 38.2, 41.0, 35.0, 31.5])


def _problem_4_18(source_model: str = "constant_volume"):  # type: ignore[no-untyped-def]
    """The whole chain of problem 4.18, from the blower to the operator room."""
    # Plant room 8 x 10 x 3 m: floor and ceiling 80 m2 each, walls 108 m2.
    plant = [
        (80.0, _R418_PLANT_FLOOR),
        (80.0, _R418_CEILING),
        (2.0 * (8.0 * 3.0) + 2.0 * (10.0 * 3.0), _R418_WALLS),
    ]
    plant_area = sum(area for area, _ in plant)
    r1 = room.room_constant(plant_area, room.mean_absorption(plant))
    # Operator room 5 x 5 x 3 m, carpeted floor, plant-room ceiling and walls.
    operator = [
        (25.0, _R418_CARPET),
        (25.0, _R418_CEILING),
        (4.0 * (5.0 * 3.0), _R418_WALLS),
    ]
    return noise_control.room_to_room_transmission(
        _BANDS_6,
        _R418_TL,
        5.0 * 3.0,
        room.equivalent_absorption_area(operator),
        source=noise_control.SourceRoom(
            power_level=_R418_LW,
            room_constant=r1,
            # The blower stands on the floor along the middle of a wall, i.e.
            # in the intersection of two large flat surfaces (Q = 4), and the
            # problem asks for a *conservative* estimate, which is Norton's
            # constant-volume model (Table 4.5): the radiated power rises by
            # 10 lg Q = 6 dB.
            directivity=4.0,
            model=source_model,
        ),
        criterion=noise_control.DesignCriterion(family="NC", target=45.0),
        label="Plant room to operator room",
    )


def test_problem_4_18_operator_room_levels() -> None:
    """Problem 4.18 (answer p. 613): 72.3/60.4/41.4/41.0/33.8/30.7 dB.

    The full chain: the blower sound power level into the reverberant field of
    the 8 x 10 x 3 m plant room as a conservative constant-volume source in a
    floor-wall edge, across the separating 5 m x 3 m wall, into the 5 x 5 x 3 m
    operator room. All six printed levels are reproduced to 0.1 dB.
    """
    result = _problem_4_18()
    assert np.allclose(result.received_level, _R418_LP2, atol=0.1)
    assert np.allclose(result.received_level, result.source_level - result.noise_reduction)


def test_problem_4_18_source_power_model_is_worth_six_decibels() -> None:
    """The constant-volume model of Table 4.5 is what makes 4.18 conservative.

    A source in the intersection of a floor and a wall has ``Q = 4``, so the
    constant-volume model raises the radiated power by ``10 lg 4 = 6.02 dB``
    over the free-space sound power level and the constant-power model leaves it
    alone. Without that step the printed answers come out 6 dB low.
    """
    conservative = _problem_4_18()
    plain = _problem_4_18(source_model="constant_power")
    gap = 10.0 * np.log10(4.0)
    assert np.allclose(conservative.source_level - plain.source_level, gap)
    assert np.allclose(conservative.received_level - plain.received_level, gap)
    assert np.max(np.abs(plain.received_level - _R418_LP2)) > 5.9


def test_problem_4_18_a_weighted_levels() -> None:
    """Problem 4.18 (answer p. 613): 56.2/51.5/38.2/41.0/35.0/31.5 dB(A).

    A-weighted with the library's own octave-band corrections (ISO 3744 Annex E
    Table E.2). Two known differences keep this from matching to a tenth: the
    book weights with its own Table 4.3, whose 250 Hz value is -8.9 dB against
    the -8.6 dB of the standard, and its printed 4 kHz answer of 31.5 dB(A)
    is 0.2 dB below the 30.7 dB level plus the +1.0 dB weighting it states.
    Hence 0.3 dB here against the 0.1 dB of the unweighted spectrum.
    """
    from phonometry.emission.sound_power import _a_weighting_corrections

    result = _problem_4_18()
    weighted = result.received_level + _a_weighting_corrections(_BANDS_6)
    assert np.allclose(weighted, _R418_LPA, atol=0.3)


def test_problem_4_18_rating_and_verdict() -> None:
    """The operator-room spectrum of problem 4.18 fails an NC 45 design goal.

    The book does not rate the spectrum, so this pins only what the printed
    levels imply on their own: the 125 Hz band at 72.3 dB stands 12.3 dB above
    the 60 dB of the NC-45 curve at 125 Hz (ANSI/ASA S12.2-2019 Table 1), so no
    NC 45 verdict is possible and the low-frequency band governs.
    """
    result = _problem_4_18()
    curve = result.criterion_curve
    assert curve is not None
    assert curve[0] == pytest.approx(60.0)
    assert result.meets_target is False
    excess = result.exceedance
    assert excess is not None
    assert int(np.argmax(excess)) == 0
    assert float(excess[0]) == pytest.approx(12.3, abs=0.1)
    # Required TL of the separating wall to reach NC 45 in every band.
    required = result.required_transmission_loss
    assert required is not None
    assert np.all(required[excess > 0.0] > result.transmission_loss[excess > 0.0])
    assert np.allclose(required - result.transmission_loss, excess, atol=1e-9)
    assert result.rating.rating > 45.0


def test_problem_4_18_table_rows() -> None:
    """``table()`` prints the rows of the hand calculation, in order."""
    rows = _problem_4_18().table()
    assert [row["kind"] for row in rows] == [
        "source_power",
        "source",
        "transmission_loss",
        "absorption",
        "noise_reduction",
        "received",
        "criterion",
        "required",
    ]
    assert np.allclose(rows[0]["values"], _R418_LW)
    assert np.allclose(rows[2]["values"], _R418_TL)
    assert np.allclose(rows[5]["values"], _R418_LP2, atol=0.1)
    assert np.allclose(rows[6]["values"][0], 60.0)


# ---------------------------------------------------------------------------
# Problem 4.16 - required enclosure TL against NC-45 (printed pp. 584-585).
# ---------------------------------------------------------------------------

_BANDS_8 = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])

#: Printed data of problem 4.16: the absorption of the concrete surfaces and of
#: the 50 mm mineral wool lining, the reverberant level in the room with the
#: compressor unenclosed and the NC-45 target column.
_R416_CONCRETE = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03])
_R416_WOOL = np.array([0.10, 0.20, 0.45, 0.65, 0.75, 0.80, 0.80, 0.80])
_R416_LP1 = np.array([72.0, 79.0, 81.0, 84.0, 83.0, 81.0, 80.0, 75.0])
_R416_LP2 = np.array([67.0, 60.0, 54.0, 49.0, 46.0, 44.0, 43.0, 41.0])

#: Printed answer (p. 613): the required transmission loss of the walls and roof.
_R416_TL = np.array([14.4, 25.2, 28.9, 34.4, 35.2, 34.7, 34.7, 31.6])


def _enclosure_4_16() -> tuple[float, float, np.ndarray]:
    """External area, internal area and mean interior absorption of 4.16.

    Enclosure 2.5 x 3.5 x 2.5 m standing on the concrete floor of the room: the
    walls and the roof radiate and are the surfaces whose transmission loss the
    problem asks for. Their inner faces carry the 50 mm mineral wool lining;
    the rest of the interior is the strip of concrete floor the 1.5 x 2.5 m
    machine does not cover plus the five exposed faces of the machine, which
    the problem says have the absorption of concrete.
    """
    external_area = 2.0 * (2.5 * 2.5) + 2.0 * (3.5 * 2.5) + 2.5 * 3.5
    machine = 2.0 * (1.5 * 1.5) + 2.0 * (2.5 * 1.5) + 1.5 * 2.5
    bare_floor = 2.5 * 3.5 - 1.5 * 2.5
    absorption = np.asarray(
        room.mean_absorption(
            [(external_area, _R416_WOOL), (bare_floor + machine, _R416_CONCRETE)]
        ),
        dtype=np.float64,
    )
    return external_area, external_area + bare_floor + machine, absorption


def _problem_4_16(model: str = "norton"):  # type: ignore[no-untyped-def]
    """Required enclosure transmission loss of problem 4.16.

    ``model="norton"`` is Equation (4.115), which has no ``0.3`` floor inside
    the logarithm; that is the model the printed answer was computed with.
    """
    external_area, internal_area, absorption = _enclosure_4_16()
    return noise_control.enclosure_required_transmission_loss(
        _R416_LP1 - _R416_LP2,
        external_area,
        internal_area,
        absorption,
        frequencies=_BANDS_8,
        model=model,
    )


def test_problem_4_16_required_transmission_loss() -> None:
    """Problem 4.16 (answer p. 613): 14.4/25.2/28.9/34.4/35.2/34.7/34.7/31.6 dB.

    ``R = IL + 10 lg(S_E / R_i)`` with the required insertion loss the gap
    between the printed reverberant level with the compressor unenclosed and the
    printed NC-45 column, the external radiating area the walls and roof of the
    2.5 x 3.5 x 2.5 m enclosure, and the interior absorption of the mineral
    wool lining, the uncovered concrete floor and the machine surface. The
    printed answers are reproduced to 0.15 dB, the resolution of the book's own
    one-decimal rounding.
    """
    result = _problem_4_16()
    assert result.external_area == pytest.approx(38.75)
    assert result.internal_area == pytest.approx(59.5)
    assert np.allclose(result.panel_transmission_loss, _R416_TL, atol=0.15)
    # The requested insertion loss is carried through unchanged, so the result
    # reads like a forward calculation.
    assert np.allclose(result.insertion_loss, _R416_LP1 - _R416_LP2)


def test_problem_4_16_target_column_is_the_book_nc45_curve() -> None:
    """The book's NC-45 column differs from ANSI S12.2 Table 1 at 8 kHz only.

    Problem 4.16 tabulates the NC-45 target as 67/60/54/49/46/44/43/**41** dB
    from 63 Hz to 8 kHz. ANSI/ASA S12.2-2019 Table 1, which the library
    implements, reads 42 dB in the 8 kHz band and agrees everywhere else. The
    test of the required transmission loss therefore uses the book's printed
    column, not the library curve, so the oracle stays the published one.
    """
    from phonometry.room.noise_criteria import _criterion_curve_at

    ansi = _criterion_curve_at("NC", 45.0, _BANDS_8)
    assert np.allclose(ansi[:-1], _R416_LP2[:-1])
    assert ansi[-1] == pytest.approx(42.0)
    assert _R416_LP2[-1] == pytest.approx(41.0)


def test_bies_and_norton_enclosure_models_differ_by_the_floor() -> None:
    """``model`` selects the ``0.3`` of Bies Eq. (7.111) or Norton Eq. (4.115).

    The two corrections are ``10 lg(0.3 + S_E/R_i)`` and ``10 lg(S_E/R_i)``, so
    Bies always asks for more transmission loss, and the gap grows as the
    lining takes the interior room constant up. Checked against the closed form
    with the geometry of problem 4.16.
    """
    norton = _problem_4_16()
    bies = _problem_4_16(model="bies")
    ratio = norton.external_area / norton.room_constant
    expected_gap = 10.0 * np.log10((0.3 + ratio) / ratio)
    assert np.all(bies.panel_transmission_loss > norton.panel_transmission_loss)
    assert np.allclose(
        bies.panel_transmission_loss - norton.panel_transmission_loss,
        expected_gap,
    )
    # The gap grows with the lining: smallest in the hard 63 Hz band, largest
    # where the mineral wool is at its most absorbing.
    assert expected_gap[0] < expected_gap[-1]
