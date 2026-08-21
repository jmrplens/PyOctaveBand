#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the end-to-end duct-borne noise cascade.

Oracle: Long, *Architectural Acoustics* 2nd ed. (Academic Press, 2014),
Table 14.9, "HVAC System Loss Calculations" (printed pages 556-558): the
complete worked sheet for a forward-curved centrifugal fan at 5000 cfm and
2 in w.g. feeding a 20 x 20 x 8 ft room through a supply and a return path, and
compared with NC 30. Every intermediate row of that sheet is printed (element
attenuation, *Sum*, *Self-Noise*, *Combined*), so the cascade can be checked row
for row.

The sheet's element rows are fed in as published, because Long states that the
numbers were produced by a computer program whose element data differ in places
from the tables printed alongside them (see the warning in
:mod:`phonometry.noise_control.hvac`). What is under test here is the cascade
*arithmetic*: subtract the attenuation, apply the 0 dB self-noise floor the
program uses, combine the regenerated noise on an energy basis, apply the room
effect, add the two paths and compare with the criterion curve.

The sheet is printed to whole decibels and its own rounding is not always
self-consistent (e.g. row 3 of the supply path prints a *Sum* of 49 dB at 500 Hz
where 76 - 28 = 48, then a *Combined* consistent with 48), so the comparison
tolerance is the 1 dB the sheet itself carries.
"""

from __future__ import annotations

import dataclasses
import re
import warnings

import numpy as np
import pytest

from phonometry.noise_control.duct_modes import PlaneWaveWarning
from phonometry.noise_control.duct_path import (
    DuctElement,
    DuctPathResult,
    combine_duct_paths,
    duct_path,
)
from phonometry.noise_control.hvac import OCTAVE_BANDS

#: Long Table 14.9, row 1 of both paths: the fan sound power level, dB re 1 pW.
FAN_LW = [90.0, 86.0, 82.0, 79.0, 77.0, 75.0, 71.0, 61.0]

#: Long Table 14.9, supply path: (label, attenuation, self-noise) per element,
#: attenuations entered as the positive losses the models return.
SUPPLY_ELEMENTS = (
    (
        "Elbow, 36 x 24 in, unlined",
        [0, 1, 2, 3, 3, 3, 3, 3],
        [41, 39, 36, 29, 20, 6, 0, 0],
    ),
    (
        "Silencer, standard pressure drop, 3 ft, 36 x 24 in",
        [7, 12, 16, 28, 35, 35, 28, 17],
        [49, 43, 44, 42, 42, 45, 35, 24],
    ),
    (
        "Duct, rectangular, 36 x 24 in, 5 ft, 1 in lining",
        [2, 2, 3, 7, 15, 12, 11, 9],
        None,
    ),
    ("Split, 25 per cent", 6.0, None),
    (
        "Duct, rectangular, 18 x 12 in, 6 ft, 1 in lining",
        [3, 3, 5, 11, 25, 22, 16, 13],
        None,
    ),
    (
        "Duct, round flexible, 12 in diameter, 6 ft",
        [14, 14, 16, 15, 17, 22, 16, 13],
        None,
    ),
    ("Rectangular diffuser, 312 cfm, 0.05 in pd", 0.0, [33, 32, 29, 23, 15, 4, 0, 0]),
)
SUPPLY_ROOM_EFFECT = [6.0, 6.0, 5.0, 5.0, 6.0, 7.0, 6.0, 6.0]

#: Long Table 14.9, supply path, the printed *Self-Noise* rows. Rows the sheet
#: prints as a line of zeros are the elements whose ``self_noise`` is left
#: unset above: they check the 0 dB floor of the sheet, not an input.
SUPPLY_SELF_NOISE = (
    [41, 39, 36, 29, 20, 6, 0, 0],
    [49, 43, 44, 42, 42, 45, 35, 24],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [33, 32, 29, 23, 15, 4, 0, 0],
)

#: Long Table 14.9, supply path, the printed *Sum* and *Combined* rows.
SUPPLY_ROWS = (
    ([90, 85, 80, 76, 74, 72, 68, 58], [90, 85, 80, 76, 74, 72, 68, 58]),
    ([83, 73, 64, 49, 39, 37, 40, 41], [83, 73, 64, 49, 44, 45, 41, 41]),
    ([81, 71, 61, 42, 29, 33, 30, 32], [81, 71, 61, 42, 29, 33, 30, 32]),
    ([75, 65, 55, 36, 23, 27, 24, 26], [75, 65, 55, 36, 23, 27, 24, 26]),
    ([72, 62, 50, 25, -2, 5, 8, 13], [72, 62, 50, 25, 2, 6, 9, 13]),
    ([58, 48, 34, 10, -15, -16, -7, 0], [58, 48, 34, 10, 0, 0, 1, 3]),
    ([58, 48, 34, 10, 0, 0, 1, 3], [58, 48, 35, 23, 15, 5, 4, 5]),
)
SUPPLY_RECEIVED = [52, 42, 30, 18, 9, -2, -2, -1]

#: Long Table 14.9, return path.
RETURN_ELEMENTS = (
    (
        "Elbow, 36 x 24 in, unlined",
        [0, 1, 2, 3, 3, 3, 3, 3],
        [43, 42, 39, 33, 24, 12, 0, 0],
    ),
    (
        "Silencer, low-frequency standard pressure drop, 5 ft, 36 x 24 in",
        [16, 21, 35, 41, 41, 28, 21, 15],
        [51, 49, 53, 56, 56, 59, 60, 53],
    ),
    (
        "Elbow, 36 x 24 in, lined, 1 in",
        [1, 2, 3, 4, 5, 6, 8, 10],
        [39, 38, 34, 28, 18, 4, 0, 0],
    ),
    ("Plenum, 800 ft2, 50 per cent lined", [12, 13, 19, 20, 20, 20, 21, 21], None),
    (
        "Rectangular grille, 24 x 24 in, 563 cfm, 0.05 in pd",
        0.0,
        [30, 29, 26, 20, 12, 1, 0, 0],
    ),
)
RETURN_ROOM_EFFECT = [9.0, 8.0, 6.0, 8.0, 8.0, 8.0, 9.0, 10.0]

#: Long Table 14.9, return path, the printed *Self-Noise* rows.
RETURN_SELF_NOISE = (
    [43, 42, 39, 33, 24, 12, 0, 0],
    [51, 49, 53, 56, 56, 59, 60, 53],
    [39, 38, 34, 28, 18, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [30, 29, 26, 20, 12, 1, 0, 0],
)
RETURN_ROWS = (
    ([90, 85, 80, 76, 74, 72, 68, 58], [90, 85, 80, 76, 74, 72, 68, 58]),
    ([74, 64, 45, 35, 33, 44, 47, 43], [74, 64, 54, 56, 56, 59, 60, 53]),
    ([73, 62, 51, 52, 51, 53, 52, 43], [73, 62, 51, 52, 51, 53, 52, 43]),
    ([61, 49, 32, 32, 31, 33, 31, 22], [61, 49, 32, 32, 31, 33, 31, 22]),
    ([61, 49, 32, 32, 31, 33, 31, 22], [61, 49, 33, 33, 31, 33, 31, 22]),
)
RETURN_RECEIVED = [52, 41, 27, 25, 23, 25, 22, 12]

#: Long Table 14.9, closing rows: the two paths combined.
COMBINED_RECEIVED = [55, 45, 32, 26, 23, 25, 22, 12]

#: The sheet's own rounding, dB (see the module docstring).
SHEET_TOLERANCE = 1.0


def _build(
    elements: tuple[tuple[str, object, object], ...],
    room_effect: list[float],
    label: str,
) -> DuctPathResult:
    """Cascade one path of Long Table 14.9 from its published element rows."""
    return duct_path(
        OCTAVE_BANDS,
        FAN_LW,
        [
            DuctElement(name, attenuation, self_noise, code=str(i))
            for i, (name, attenuation, self_noise) in enumerate(elements, start=2)
        ],
        room_effect=room_effect,
        source_label="Fan, centrifugal, forward-curved, 5000 cfm, 2 in w.g.",
        criterion="NC",
        target=30.0,
        label=label,
    )


def _assert_sheet(got: np.ndarray, printed: list[int], what: str) -> None:
    """Compare a computed row with a printed one at the sheet's own rounding."""
    rounded = np.round(np.asarray(got, dtype=np.float64))
    deviation = np.abs(rounded - np.asarray(printed, dtype=np.float64))
    assert np.all(deviation <= SHEET_TOLERANCE), (
        f"{what}: {rounded.tolist()} vs printed {printed}"
    )


def _row(value: object) -> list[int]:
    """The eight printed columns of a sheet row given as a scalar or a list."""
    return [int(v) for v in np.broadcast_to(np.asarray(value, dtype=np.float64), (8,))]


def _assert_path(
    result: DuctPathResult,
    elements: tuple[tuple[str, object, object], ...],
    self_noise_rows: tuple[list[int], ...],
    level_rows: tuple[tuple[list[int], list[int]], ...],
    room_effect: list[float],
    received: list[int],
    what: str,
) -> None:
    """Check all four printed rows of every element, and the two foot rows.

    The attenuation rows are the published ones fed back, so they check that
    the cascade carries each element's loss into the row it belongs to; the
    *Self-Noise* rows the sheet prints as zeros check its 0 dB floor; the
    *Sum* and *Combined* rows check the cascade arithmetic itself.
    """
    assert len(result.stages) == len(level_rows)
    printed = zip(elements, self_noise_rows, level_rows, strict=True)
    for stage, ((_, loss, _), self_noise, (row_sum, combined)) in zip(
        result.stages, printed, strict=True
    ):
        where = f"{what} {stage.code}"
        _assert_sheet(stage.attenuation, _row(loss), f"{where} attenuation")
        _assert_sheet(stage.attenuated, row_sum, f"{where} Sum")
        _assert_sheet(stage.self_noise, self_noise, f"{where} Self-Noise")
        _assert_sheet(stage.level, combined, f"{where} Combined")
    assert result.room_effect is not None
    _assert_sheet(result.room_effect, _row(room_effect), f"{what} room effect")
    _assert_sheet(result.received_level, received, f"{what} received")


def test_long_table_14_9_supply_path_row_for_row() -> None:
    _assert_path(
        _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply"),
        SUPPLY_ELEMENTS,
        SUPPLY_SELF_NOISE,
        SUPPLY_ROWS,
        SUPPLY_ROOM_EFFECT,
        SUPPLY_RECEIVED,
        "supply",
    )


def test_long_table_14_9_return_path_row_for_row() -> None:
    _assert_path(
        _build(RETURN_ELEMENTS, RETURN_ROOM_EFFECT, "Return"),
        RETURN_ELEMENTS,
        RETURN_SELF_NOISE,
        RETURN_ROWS,
        RETURN_ROOM_EFFECT,
        RETURN_RECEIVED,
        "return",
    )


def test_long_table_14_9_combined_meets_nc_30() -> None:
    supply = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    path = _build(RETURN_ELEMENTS, RETURN_ROOM_EFFECT, "Return")
    both = combine_duct_paths([supply, path], label="Supply + return")
    _assert_sheet(both.received_level, COMBINED_RECEIVED, "combined received")
    assert both.criterion == "NC"
    assert both.target == 30.0
    # Long's closing verdict: the design meets NC 30 (his printed curve is the
    # original Beranek NC-30, which differs by 1 dB at 1 kHz from the
    # ANSI/ASA S12.2-2019 Table 1 curve the library implements).
    assert both.meets_target is True
    assert np.allclose(
        both.criterion_curve, [57, 48, 41, 35, 32, 29, 28, 27], atol=1e-9
    )
    assert [name for name, _ in both.contributions] == ["Supply", "Return"]


def test_long_table_14_9_supply_row_5_is_the_split_loss_model() -> None:
    """The one supply row that follows the book's own printed equation.

    Long Eq. 14.17 for a 25 per cent split whose branch areas add up to the
    main duct area: the reflection term vanishes and the division term is
    ``-10 lg 0.25 = 6.02 dB``, printed as -6 dB in the sheet.
    """
    from phonometry.noise_control import hvac

    main = 36.0 * 24.0 * 0.0254**2
    loss = hvac.split_loss(main, [0.25 * main] * 4, branch=0)
    assert loss == pytest.approx(-10.0 * np.log10(0.25), abs=1e-12)
    assert round(loss) == 6


def test_table_rows_follow_the_published_sheet_layout() -> None:
    supply = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    rows = supply.table()
    kinds = [row["kind"] for row in rows]
    assert kinds[0] == "source"
    # One quartet per element: attenuation, Sum, Self-noise, Combined.
    assert kinds[1:5] == ["attenuation", "sum", "self_noise", "level"]
    assert kinds[-3:] == ["room_effect", "received", "criterion"]
    # Attenuation rows carry the worksheet sign.
    assert np.allclose(rows[1]["values"], -np.asarray(SUPPLY_ELEMENTS[0][1]))
    assert np.allclose(rows[-3]["values"], -np.asarray(SUPPLY_ROOM_EFFECT))
    assert rows[-1]["label"] == "NC 30"


def test_self_noise_floor_can_be_disabled() -> None:
    without = duct_path(
        OCTAVE_BANDS,
        FAN_LW,
        [DuctElement("Silencer", 60.0)],
        self_noise_floor=None,
    )
    assert np.all(np.isneginf(without.stages[0].self_noise))
    assert np.allclose(without.received_level, np.asarray(FAN_LW) - 60.0)
    # With the floor, the level bottoms out at the floor instead.
    with_floor = duct_path(OCTAVE_BANDS, FAN_LW, [DuctElement("Silencer", 60.0)])
    assert np.all(with_floor.received_level >= 0.0)


def test_no_room_effect_leaves_a_sound_power_level() -> None:
    res = duct_path(OCTAVE_BANDS, FAN_LW, [DuctElement("Duct", 3.0)])
    assert res.room_effect is None
    assert res.target is None
    assert res.criterion_curve is None
    assert res.exceedance is None
    assert res.meets_target is None
    assert np.allclose(res.received_level, np.asarray(FAN_LW) - 3.0)


def test_element_accepts_an_hvac_spectrum_result() -> None:
    from phonometry.noise_control import hvac

    spectrum = hvac.unlined_circular_duct_attenuation(None, 3.0)
    res = duct_path(OCTAVE_BANDS, FAN_LW, [DuctElement("Round duct", spectrum)])
    assert np.allclose(res.stages[0].attenuation, spectrum.values)


def test_rc_criterion_family() -> None:
    res = duct_path(
        OCTAVE_BANDS,
        FAN_LW,
        [DuctElement("Silencer", 50.0)],
        room_effect=6.0,
        criterion="RC",
        target=35.0,
    )
    assert res.criterion == "RC"
    assert res.rating.__class__.__name__ == "RCResult"
    assert res.criterion_curve is not None


def test_section_declares_the_plane_wave_limit() -> None:
    with pytest.warns(PlaneWaveWarning, match="Supply"):
        duct_path(
            OCTAVE_BANDS,
            FAN_LW,
            [DuctElement("Duct", 3.0)],
            section={"width": 0.9144, "height": 0.6096},
            flow_velocity=8.0,
            label="Supply",
        )


def test_no_section_means_no_validity_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        duct_path(OCTAVE_BANDS, FAN_LW, [DuctElement("Duct", 3.0)])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frequencies": [], "source_level": []}, "frequencies"),
        ({"criterion": "NR"}, "criterion"),
        ({"target": float("nan")}, "target"),
        ({"source_level": [1.0, 2.0]}, "source_level"),
        ({"room_effect": [1.0, 2.0]}, "room_effect"),
    ],
)
def test_duct_path_validation(kwargs: dict[str, object], match: str) -> None:
    call: dict[str, object] = {
        "frequencies": OCTAVE_BANDS,
        "source_level": FAN_LW,
        "elements": [DuctElement("Duct", 3.0)],
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        duct_path(**call)  # type: ignore[arg-type]


def test_elements_must_be_duct_elements() -> None:
    with pytest.raises(TypeError, match="DuctElement"):
        duct_path(OCTAVE_BANDS, FAN_LW, ["not an element"])  # type: ignore[list-item]


def test_element_spectrum_must_be_finite() -> None:
    elements = [DuctElement("Duct", [1, 2, 3, 4, 5, 6, 7, np.nan])]
    with pytest.raises(ValueError, match="finite"):
        duct_path(OCTAVE_BANDS, FAN_LW, elements)


def test_combine_requires_at_least_one_path() -> None:
    with pytest.raises(ValueError, match="at least one"):
        combine_duct_paths([])


def test_combine_requires_matching_bands() -> None:
    supply = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    other = duct_path([63.0, 125.0], [90.0, 86.0], [DuctElement("Duct", 3.0)])
    with pytest.raises(ValueError, match="same analysis bands"):
        combine_duct_paths([supply, other])


def test_combine_of_one_path_is_that_path() -> None:
    supply = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    same = combine_duct_paths([supply])
    assert np.allclose(same.received_level, supply.received_level)
    assert same.stages == ()


# --------------------------------------------------------------------------
# Rows that do not run over the analysis bands
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field_name",
    ["attenuation", "attenuated", "self_noise", "level"],
)
@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_a_stage_row_off_the_band_axis_is_refused(field_name: str, trim: bool) -> None:
    """A calculation sheet cannot carry a row of the wrong width.

    The sheet prints one row per element under a single band header with a
    running total beneath, so reportlab pads a short row out to the header and
    the total then sums a band that appears nowhere in the column above it. A
    row one *too long* is truncated by the same table, just as quietly.
    """
    result = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    stage = result.stages[0]
    row = np.asarray(getattr(stage, field_name))
    wrong = row[:-1] if trim else np.append(row, row[-1])
    # The stage itself carries no guard, so it is built outside the block:
    # what is under test is the sheet refusing to hold it.
    stages = (dataclasses.replace(stage, **{field_name: wrong}), *result.stages[1:])
    with pytest.raises(ValueError, match=f"'stages\\[0\\]\\.{field_name}'"):
        dataclasses.replace(result, stages=stages)


def test_a_room_effect_off_the_band_axis_is_refused() -> None:
    """The room effect is applied band by band, so it must have the bands."""
    result = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    one_band_short = np.zeros(len(OCTAVE_BANDS) - 1)
    with pytest.raises(ValueError, match="'room_effect'"):
        dataclasses.replace(result, room_effect=one_band_short)


def test_no_room_effect_at_all_is_still_allowed() -> None:
    """``None`` is an absent quantity, not a disagreement."""
    result = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    assert dataclasses.replace(result, room_effect=None).room_effect is None


@pytest.mark.parametrize(
    ("what", "build"),
    [
        (
            "stages[0].attenuation",
            lambda r: {"stages": (dataclasses.replace(r.stages[0], attenuation=3.0),)},
        ),
        ("contributions[0] (Supply)", lambda r: {"contributions": (("Supply", 3.0),)}),
        ("frequencies", lambda r: {"frequencies": 500.0}),
    ],
    ids=["stage-row", "contribution", "band-axis"],
)
def test_a_single_number_where_a_spectrum_belongs_says_so(what, build) -> None:
    """A scalar in a nested field is named, not left to numpy or to len().

    Reaching the nested rows means indexing a shape or taking a length, and
    both answer a scalar with an exception of their own that names neither the
    field nor the type the guard promises to name.
    """
    result = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    scalar_field = build(result)
    with pytest.raises(ValueError, match=f"'{re.escape(what)}'"):
        dataclasses.replace(result, **scalar_field)


@pytest.mark.parametrize(
    ("what", "build"),
    [
        (
            "stages[0].attenuation",
            lambda r, wrong: {
                "stages": (dataclasses.replace(r.stages[0], attenuation=wrong),)
            },
        ),
        (
            "contributions[0] (Supply)",
            lambda r, wrong: {"contributions": (("Supply", wrong),)},
        ),
    ],
    ids=["stage-row", "contribution"],
)
def test_a_nested_row_with_an_extra_axis_is_refused(what, build) -> None:
    """The nested rows are pinned by their axes, not only by their length.

    A row of shape ``(bands, 2)`` counts the right number of bands on its
    first axis, so counting alone lets it into the sheet with a second axis
    nobody asked for.
    """
    result = _build(SUPPLY_ELEMENTS, SUPPLY_ROOM_EFFECT, "Supply")
    row = np.asarray(result.stages[0].attenuation)
    two_dimensional = np.column_stack([row, row])
    replacement = build(result, two_dimensional)
    with pytest.raises(ValueError, match=f"'{re.escape(what)}' must have one axis"):
        dataclasses.replace(result, **replacement)
