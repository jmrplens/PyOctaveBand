#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the shared fiche builders of :mod:`phonometry._report._layout`.

These are the pieces every accredited fiche is assembled from, so a defect in
one of them is latent in all of them whether or not a caller reaches it today.
Their contract is with the renderers rather than with a user, which is why it
is pinned here directly instead of through a ``report()`` call.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

from phonometry._report._layout import (  # noqa: E402
    _octave_grouping,
    _require_column_widths,
    band_table,
    compliance_table,
    grid_table,
)


# --------------------------------------------------------------------------
# A pair with no label still has a value
# --------------------------------------------------------------------------
def test_a_pair_with_no_label_keeps_its_value() -> None:
    """The right-hand value used to be dropped when its label was empty.

    The odd-length pad was inferred from an empty label, so a genuine
    label-less value on the right went with it while the same value on the
    left was printed. The pad is explicit now and both columns are alike.
    """
    table = grid_table([("Label A", "VALUE-A"), ("", "VALUE-B")])
    printed = [
        cell.text if hasattr(cell, "text") else cell
        for row in table._cellvalues
        for cell in row
    ]
    assert any("VALUE-B" in str(cell) for cell in printed)


def test_an_odd_number_of_pairs_still_pads_the_last_row() -> None:
    """The pad is the one case that ever wanted a blank cell."""
    table = grid_table([("Label A", "VALUE-A")])
    last_row = table._cellvalues[-1]
    assert last_row[2] == ""
    assert last_row[3] == ""


# --------------------------------------------------------------------------
# A verdict that is not one of the three the table knows
# --------------------------------------------------------------------------
def test_an_unrecognised_status_is_refused_rather_than_shown_as_no_verdict() -> None:
    """An unknown status used to fall through to the informational dash.

    That turned a mistyped verdict into no verdict at all: a row a caller
    meant to mark FAIL came out looking like a row nobody had judged.
    """
    with pytest.raises(ValueError, match="'status' must be one of"):
        compliance_table([("Metric", "1.0", "2.0", "FAIL")])


@pytest.mark.parametrize("status", ["pass", "fail", "info"])
def test_the_three_statuses_the_table_knows_are_drawn(status: str) -> None:
    """The dash is asked for by name, as ``info``."""
    assert compliance_table([("Metric", "1.0", "2.0", status)]) is not None


# --------------------------------------------------------------------------
# Column widths that do not match the columns
# --------------------------------------------------------------------------
def test_a_width_list_that_misses_a_column_is_refused() -> None:
    """reportlab pads a short list by repeating its last width, silently."""
    data = [["a", "b", "c", "d"], ["1", "2", "3", "4"]]
    with pytest.raises(ValueError, match="'col_widths' has 2 entries for 4"):
        _require_column_widths(data, [70.0, 26.0], "band_table")


def test_a_single_width_is_left_alone() -> None:
    """One number is the way to ask for an even table, and reportlab honours it."""
    data = [["a", "b", "c"], ["1", "2", "3"]]
    assert _require_column_widths(data, 26.0, "band_table") is None


# --------------------------------------------------------------------------
# Which rows carry octaves, measured rather than counted
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("centres", "expected"),
    [
        ([100, 125, 160, 200, 250, 315, 400, 500], 3),
        ([400, 500, 630, 800, 1000], 3),
        ([125, 250, 500, 1000, 2000], None),
        ([400, 450, 500, 550, 600, 650], None),
        ([-85, -75, -65, -55, -45, -35], None),
        (None, None),
    ],
    ids=[
        "third-octave",
        "third-octave-rounded",
        "octave",
        "linear",
        "angles",
        "absent",
    ],
)
def test_the_octave_grouping_is_measured_from_the_band_centres(
    centres, expected
) -> None:
    """The rules group a table by octave, so they need rows that are bands.

    Deciding by row count ruled nineteen goniometer angles into triplets and
    ruled a linear frequency sweep the same way, while an octave table of any
    length other than five got one-third-octave rules.
    """
    assert _octave_grouping(centres) == expected


# --------------------------------------------------------------------------
# A drawing wider than the width it declares
# --------------------------------------------------------------------------
def test_a_drawing_is_scaled_by_what_it_covers_not_by_what_it_declares() -> None:
    """A legend anchored outside the axes reaches past the figure box.

    svglib reports the declared box, so scaling by that let the overhang
    through at full size: the duct-path sheet asked for 96 mm, drew 106.6,
    and put its legend 10.7 mm beyond the right margin of the page, straight
    through the plot's own border.
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")

    from phonometry._report._layout import render_figure_drawing

    def wide_legend(ax=None, language="en", **kwargs):
        """A plot whose legend is far wider than its axes."""
        ax.plot([1.0, 2.0], [1.0, 2.0], label="a legend entry long enough to overhang")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005))
        return ax

    target = 100.0
    drawing = render_figure_drawing(wide_legend, target, y_top=None)
    assert drawing.getBounds()[2] <= target + 0.5


def test_band_centres_that_are_not_the_rows_are_refused() -> None:
    """The rules are counted off the rows and read off the centres.

    Nothing else relates the two, so a caller handing over a longer axis than
    it tabulated would rule the rows it printed by the structure of bands it
    did not.
    """
    rows = [["f", "value"], ["100", "1.0"], ["125", "2.0"], ["160", "3.0"]]
    centres = [100, 125, 160, 200, 250, 315]
    with pytest.raises(ValueError, match="carries 6 entries for 3 band rows"):
        band_table(rows, [28.0, 28.0], 3, band_centres=centres)
