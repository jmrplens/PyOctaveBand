#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the shared construction guards (``phonometry._internal.validation``).

Every result type in the library reaches these three helpers, so their coverage
used to be entirely indirect: a change of semantics here could leave a hundred
classes quietly wrong and the suite green. What is pinned below is the part
that is easy to get wrong and impossible to see from a call site, above all the
two kinds of field numpy answers badly for.

A **grid handed in as nested lists** is as two-dimensional as the array it would
become. Taking it for one axis refuses a correct grid and, worse, lets a list
carrying an extra axis walk past the very pin that exists to stop one.

A **sequence whose entries are arrays** is one axis of entries, whatever each
entry holds. A filter bank keeps one set of second-order sections per band and
the sets need not be the same length; asking numpy to stack them either raises
about an inhomogeneous shape or, when the orders happen to agree, invents an
axis nobody meant. Both have to answer the same, or a bank would be accepted or
refused according to a coincidence between its bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from phonometry._internal.validation import (
    require_ranks,
    require_same_length,
    require_same_shape,
)


@dataclass(frozen=True)
class _Result:
    """A stand-in for the result types, so the helpers are tested directly."""

    first: object = None
    second: object = None
    third: object = None


#: A filter bank whose bands carry filters of different orders: the ragged case
#: numpy cannot stack at all.
_RAGGED_BANK = (np.zeros((2, 6)), np.zeros((3, 6)))

#: The same bank where the orders happen to agree. Numpy *can* stack this one,
#: into a (2, 2, 6) array, which is the coincidence the guards must ignore.
_UNIFORM_BANK = (np.zeros((2, 6)), np.zeros((2, 6)))


@pytest.mark.parametrize(
    ("label", "bank"), [("ragged", _RAGGED_BANK), ("uniform", _UNIFORM_BANK)]
)
def test_a_bank_of_per_band_filters_is_one_axis_of_entries(
    label: str, bank: tuple[np.ndarray, ...]
) -> None:
    """Both banks are one axis of two entries, and neither reaches numpy raw.

    The uniform bank is the trap: stacking it succeeds and reports three axes,
    so a bank would be refused or accepted according to whether its bands
    happened to carry filters of the same order.
    """
    result = _Result(first=np.zeros(2), second=bank)
    require_ranks(result, second=1)
    require_same_length(result, "first", "second", axis="band")


def test_a_bank_is_measured_the_same_whichever_field_comes_first() -> None:
    """Argument order decides nothing.

    The scalar shortcut reads the fields in the order given, so a ragged entry
    listed first used to reach ``np.ndim`` before anything had established that
    it could not be stacked, and numpy answered about an inhomogeneous shape in
    its own words, naming neither the field nor the result.
    """
    result = _Result(first=np.zeros(2), second=_RAGGED_BANK)
    require_same_length(result, "first", "second", axis="band")
    require_same_length(result, "second", "first", axis="band")


def test_a_grid_of_nested_lists_carries_the_axes_it_looks_like() -> None:
    """Nested lists are measured, not taken for a single axis."""
    grid = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    require_ranks(_Result(first=grid), first=2)


def test_a_nested_list_cannot_smuggle_an_extra_axis_past_a_one_axis_pin() -> None:
    """The hole the rank pin exists to close, in the form that used to slip.

    A ``(bands, 2)`` list satisfies every count on its first axis, so only the
    rank sees it.
    """
    result = _Result(first=[[60.0, 1.0], [61.0, 1.0]])
    with pytest.raises(ValueError, match="'first'"):
        require_ranks(result, first=1)


def test_a_missing_axis_is_refused_and_not_described_as_an_extra_one() -> None:
    """A bare number where an axis was asked for is the opposite failure.

    It is refused too, and the message has to say which way round it went: a
    value short of an axis is spread over bands it was never measured on, and
    telling the reader an extra axis reached the reader would send them looking
    for the wrong bug.
    """
    result = _Result(first=np.float64(80.0), second=np.zeros(3))
    with pytest.raises(ValueError, match="'first' must have one axis; got 0") as caught:
        require_ranks(result, first=1, second=1)
    assert "never measured on" in str(caught.value)


def test_a_result_of_bare_numbers_is_passed_over_whole() -> None:
    """Several entry points take one frequency and answer in bare numbers.

    That is not an extra axis, it is no axis. The exemption is all or nothing:
    the test below pins the other half.
    """
    result = _Result(first=np.float64(1.0), second=np.float64(2.0))
    require_ranks(result, first=1, second=1)
    require_same_length(result, "first", "second")


def test_a_lone_number_beside_a_spectrum_is_still_refused() -> None:
    """The boundary of the exemption, and the reason it is all or nothing.

    Waiving the rank field by field would leave a scalar sitting beside real
    spectra wherever no length check happens to cover that field, and several
    fields in the library are pinned against an extra axis without being pinned
    against a missing one.
    """
    result = _Result(first=np.float64(1.0), second=np.zeros(4))
    with pytest.raises(ValueError, match="'first'"):
        require_ranks(result, first=1, second=1)


def test_none_is_an_absent_quantity_rather_than_a_disagreement() -> None:
    result = _Result(first=np.zeros(3), second=None)
    require_ranks(result, first=1, second=1)
    require_same_length(result, "first", "second")
    require_same_shape(result, "first", "second")


def test_same_shape_refuses_two_grids_that_agree_on_one_axis_only() -> None:
    """What a length check cannot see.

    Two grids of equal height agree on the only axis a count reads and disagree
    about every value in them.
    """
    result = _Result(first=np.zeros((3, 2)), second=np.zeros((3, 4)))
    require_same_length(result, "first", "second")
    with pytest.raises(ValueError, match="'second'"):
        require_same_shape(result, "first", "second", quantity="cell")


def test_same_shape_accepts_the_grid_the_library_produces() -> None:
    """A quantity derived elementwise keeps the shape it was derived from."""
    grid = np.zeros((3, 2))
    require_same_shape(_Result(first=grid, second=grid.copy()), "first", "second")


def test_an_axis_other_than_the_first_is_counted_where_it_lives() -> None:
    """``(name, i)`` reads axis ``i``, which is how a grid pins both of its."""
    result = _Result(first=np.zeros(2), second=np.zeros(5), third=np.zeros((2, 5)))
    require_same_length(result, "first", ("third", 0), axis="grid row")
    require_same_length(result, "second", ("third", 1), axis="grid column")


def test_an_axis_that_does_not_exist_is_named_rather_than_indexed() -> None:
    result = _Result(first=np.zeros(2), third=np.zeros(2))
    with pytest.raises(ValueError, match="'third' must have an axis 1"):
        require_same_length(result, "first", ("third", 1), axis="grid column")


def test_the_message_names_the_result_the_fields_and_the_counts() -> None:
    """What the readers could not say, and the whole point of checking here.

    Matplotlib reports two shapes, a table loop reports an index and a size,
    and neither names the field or the type it came from.
    """
    result = _Result(first=np.zeros(16), second=np.zeros(17))
    with pytest.raises(ValueError, match="'first'") as caught:
        require_same_length(result, "first", "second", axis="band")
    message = str(caught.value)
    assert "_Result" in message
    assert "'first' (16)" in message
    assert "'second' (17)" in message
    assert "one value per band" in message
