#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The unreadable-annotation gate, measured on figures built to be measured.

``scripts/figure_annotation_audit.py`` answers two questions -- how many pixels
of a label's letters does a stroke paint under them, and how many never reach
the page at all -- and every interesting case is a way of getting one of them
wrong: a box a curve enters through the air above the letters, a curve that
stops on the edge of the box, a rule whose coordinates are 0.0 and 1.0 in a
blended transform, a marker series with nothing drawn between its markers, a
curve on the axes next door, a stem that is not a ``Line2D``, a chip the curve
is drawn over. Each figure below is one of those, small enough that the
expected answer is obvious by construction.

Then the ways an artist hides from a walk of the figure, which are the ways a
defect ships silently: an annotation holds its own arrow, ``inset_axes`` files
its result under the host, a figure draws its own text after every panel, and a
3-D plate projects its labels inside their own draw. Each has a synthetic here,
and the arrow has two, because an arrow pointing at its own curve is the
pattern the gate looks *for*.

And the property the whole design rests on: the measurement runs on the live
figure that is about to be written to disk, so it has to put every artist back.
That one is pinned on the rendered bytes.

``scripts/check_figure_annotations.py`` is then the arithmetic on top: the
gate, the advisory band, the exemptions and their staleness, and the command
line where the refusals live.
"""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_figure_annotations as check
import figure_annotation_audit as audit

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Iterator[None]:
    """Recording on, into this test's own directory, with an empty tally."""
    monkeypatch.setenv(audit.AUDIT_ENV, str(tmp_path / "recording"))
    monkeypatch.setattr(audit, "_FOUND", {})
    yield
    plt.close("all")


def _blank(dpi: float = audit.REFERENCE_DPI) -> tuple[Figure, Axes]:
    """A bare axes on the unit square, at the resolution the rule is pinned to."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def _measure(fig: Figure, stem: str = "probe") -> list[dict[str, Any]]:
    """Run the audit over *fig* and hand back what it recorded for *stem*."""
    audit.audit(fig, stem)
    return audit._FOUND[stem]


def _of_kind(hits: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """The recorded hits of one defect. A figure can be guilty of both at once."""
    return [hit for hit in hits if hit["kind"] == kind]


def _prefiltered(fig: Figure) -> list[audit._Candidate]:
    """The labels the geometric stage nominates, boxes captured as it draws them."""
    return audit._candidates(fig, audit._glyph_boxes(fig))


def test_a_curve_through_a_label_is_counted_and_named() -> None:
    """The plain defect: a line across the letters, and no chip anywhere."""
    fig, ax = _blank()
    ax.plot([0, 1], [0.5, 0.5], color="black", label="the curve")
    ax.text(0.5, 0.5, "struck", ha="center", va="center")

    (hit,) = _measure(fig)

    assert hit["text"] == "struck"
    assert hit["pixels"] > check.GATE_PX[audit.BEHIND]
    assert hit["struck_by"] == [["the curve", hit["pixels"]]]


def test_a_chipped_label_is_not_the_defect() -> None:
    """A label that already carries its backing is what the gate asks for."""
    fig, ax = _blank()
    ax.plot([0, 1], [0.5, 0.5], color="black")
    ax.text(
        0.5,
        0.5,
        "struck",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f0f2f5",
            "edgecolor": "#e0e0e0",
        },
    )

    assert _measure(fig) == []


def test_a_curve_that_stops_on_the_edge_of_the_box_stays_far_below_the_gate() -> None:
    """The inline curve-end label, which reads fine and must not fail.

    The label is placed at the curve's last point with ``ha="left"``, so the
    curve terminates exactly on the left edge of its box and geometry calls
    that a crossing. What it really leaves is the width of the line's own cap
    against the first letter: single pixels, an order of magnitude under the
    band the checker will even mention.
    """
    fig, ax = _blank()
    ax.plot([0.1, 0.5], [0.5, 0.5], color="black")
    ax.text(0.5, 0.5, "25", ha="left", va="center")

    (hit,) = _measure(fig)

    assert hit["pixels"] < check.ADVISORY_PX[audit.BEHIND]


def test_a_curve_through_the_air_above_the_letters_paints_nothing() -> None:
    """Why the box cannot be the rule.

    ``get_window_extent`` returns the font's line box, ascent to descent, so a
    row of lowercase letters with no ascender leaves the top third of its box
    empty and a curve can cross it there.
    """
    fig, ax = _blank()
    text = ax.text(0.5, 0.5, "verano", ha="center", va="center")
    fig.canvas.draw()
    box = text.get_window_extent(fig.canvas.get_renderer())
    ascender = ax.transData.inverted().transform((box.x0, box.y1 - 2))[1]
    ax.plot([0, 1], [ascender, ascender], color="black", linewidth=0.8)

    assert _prefiltered(fig), "the box is crossed, so the prefilter must fire"
    assert _measure(fig) == []


def test_an_arrow_reaching_the_curve_is_not_the_label_being_struck() -> None:
    """A label set aside with a pointer is the pattern this looks for.

    The arrow belongs to the same artist as the letters, so it is drawn in the
    render the glyph ink is read from; only the glyph box keeps it out, and
    without that the correct pattern fails the gate.
    """
    fig, ax = _blank()
    ax.plot([0, 1], [0.2, 0.2], color="black", label="the curve")
    ax.annotate(
        "set aside, pointing at it",
        xy=(0.5, 0.2),
        xytext=(0.35, 0.7),
        arrowprops={"arrowstyle": "->", "color": "black"},
    )

    assert _measure(fig) == []


def test_a_label_is_not_even_nominated_by_its_own_pointer() -> None:
    """An arrow leaves the edge of the box it is anchored to.

    So its bounding box meets that box on every annotation there is, and the
    generous prefilter would nominate all 266 un-chipped labels that carry one
    in the corpus for a measurement the count is guaranteed to return zero
    from.
    """
    fig, ax = _blank()
    ax.annotate(
        "set aside, pointing at it",
        xy=(0.9, 0.5),
        xytext=(0.1, 0.8),
        arrowprops={"arrowstyle": "->", "color": "black"},
    )

    assert audit._ink(fig), "the arrow is ink, which is the whole point"
    assert _prefiltered(fig) == []


def test_a_rule_is_tested_where_it_is_drawn() -> None:
    """``axvline`` carries a blended transform, and its own y values are 0 and 1.

    Pushed through ``transData`` instead, the rule would be tested between
    data y = 0 and y = 1 -- off the bottom of these axes, where it would miss
    the label entirely.
    """
    fig, ax = _blank()
    ax.set_ylim(10, 20)
    ax.axvline(0.5, color="black")
    ax.text(0.5, 15, "on the rule", ha="center", va="center")

    (hit,) = _measure(fig)

    assert hit["pixels"] > 0


def test_a_marker_series_is_ink_only_where_its_markers_are() -> None:
    """``linestyle="none"`` draws nothing between the points, so nothing crosses."""
    fig, ax = _blank()
    ax.plot([0.1, 0.9], [0.5, 0.5], linestyle="none", marker="o", color="black")
    ax.text(0.5, 0.5, "in the gap", ha="center", va="center")

    assert _prefiltered(fig) == []
    assert _measure(fig) == []


def test_the_ink_is_credited_to_the_line_that_paints_it() -> None:
    """Geometry names the wrong line, so attribution hides one line at a time.

    Both lines reach the label's box, and the one the prefilter meets first
    only stops on its edge while the other runs through the letters -- which
    is exactly the pair that appears on ``room_noise_criteria``. The report
    has to lead with the one doing the painting.
    """
    fig, ax = _blank()
    text = ax.text(0.5, 0.5, "which one", ha="center", va="center")
    fig.canvas.draw()
    box = text.get_window_extent(fig.canvas.get_renderer())
    edge = ax.transData.inverted().transform((box.x0, box.y0))[0]
    ax.plot([0.1, edge], [0.5, 0.5], color="black", label="ends at the box")
    ax.plot([0.1, 0.9], [0.51, 0.51], color="red", label="crosses it")

    (hit,) = _measure(fig)
    worst, rest = hit["struck_by"][0], dict(hit["struck_by"][1:])

    assert worst[0] == "crosses it"
    assert rest.get("ends at the box", 0) < check.ADVISORY_PX[audit.BEHIND]


def _menagerie() -> Figure:
    """A figure carrying one of every artist the measurement touches.

    A ``Line2D``, a hatched patch with no fill, a ``LineCollection`` added with
    ``add_collection``, a contour, a band whose edge is its own colour, a
    ``twinx`` sibling, a legend, and three labels: a bare one a curve runs
    through, a chipped one something is painted over, and an annotation with an
    arrow and a chip of its own. Between them they exercise every branch of
    ``_hidden``, ``_unstroked``, ``_raised``, ``_inked`` and ``_lettering_off``.
    """
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(6, 4), dpi=audit.REFERENCE_DPI)
    grid = np.linspace(0, 10, 200)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.5)
    ax.plot(grid, np.sin(grid), color="tab:blue", label="sine")
    ax.plot(grid, np.cos(grid), color="tab:red", linestyle="--", label="cosine")
    ax.axhspan(-0.2, 0.2, color="tab:green")
    ax.add_patch(
        plt.Rectangle((1, 0.6), 3, 0.6, facecolor="none", edgecolor="black", hatch="//")
    )
    ax.add_collection(
        LineCollection(
            [[(x, -1.0), (x, 1.0)] for x in range(1, 10)],
            colors="purple",
            linewidths=1.0,
        )
    )
    ax.contour(
        np.linspace(0, 10, 20),
        np.linspace(-1.5, 1.5, 20),
        np.outer(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20)),
        levels=3,
    )
    ax.text(5, 0.0, "a bare label over the curves", ha="center", va="center")
    ax.text(
        2,
        -0.8,
        "a chipped label under a rule",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f0f2f5",
            "edgecolor": "#e0e0e0",
        },
    )
    ax.plot([0, 10], [-0.8, -0.8], color="black", linewidth=6.0, zorder=9)
    ax.annotate(
        "set aside,\npointing at it",
        xy=(7, 0.9),
        xytext=(7.2, -1.2),
        fontsize=9,
        arrowprops={"arrowstyle": "->"},
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f0f2f5",
            "edgecolor": "#e0e0e0",
        },
    )
    twin = ax.twinx()
    twin.set_ylim(-4, 4)
    twin.plot(grid, 3 * np.sin(0.5 * grid), color="tab:brown", label="the sibling")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


#: What the measurement is known to change on an artist, and therefore what
#: has to come back: visibility and zorder, the line width and hatch that
#: ``_unstroked`` takes off a patch, and the colours that ``_inked`` and
#: ``_lettering_off`` repaint a label's letters in.
_MUTATED = (
    "get_visible",
    "get_zorder",
    "get_linewidth",
    "get_hatch",
    "get_color",
    "get_facecolor",
    "get_edgecolor",
    "get_alpha",
)


def _look(artist: Artist) -> tuple[str, ...]:
    """One artist's mutable state, written out as text.

    Text rather than the values, so a numpy colour array compares like
    anything else, and an artist that does not answer a question reads as
    ``None`` rather than raising.
    """
    told = []
    for name in _MUTATED:
        method = getattr(artist, name, None)
        told.append(repr(method() if callable(method) else None))
    return (type(artist).__name__, *told)


def _state(fig: Figure) -> list[tuple[str, ...]]:
    """The mutable state of every artist of *fig* the measurement can reach.

    The artists of every axes, plus the two an ``Annotation`` owns privately:
    its chip, which ``_lettering_off`` hides, and its arrow, which
    ``_unstroked`` does. Neither is in ``ax.patches``, so walking the axes
    alone would not notice either being left as the measurement found it
    convenient.
    """
    found = []
    for ax in fig.axes:
        for artist in (*ax.get_lines(), *ax.collections, *ax.patches, *ax.texts):
            found.append(_look(artist))
            chip = getattr(artist, "get_bbox_patch", lambda: None)()
            arrow = getattr(artist, "arrow_patch", None)
            found.extend(_look(owned) for owned in (chip, arrow) if owned is not None)
    return found


def _rendered(fig: Figure) -> bytes:
    """The figure written the way the corpus writes it: SVG, glyphs as outlines.

    Not a pixel buffer: the SVG is the artefact that gets committed and
    byte-compared, and it also records what a raster does not -- an element's
    id, its order in the tree, a stroke width, a hatch.
    """
    buffer = io.BytesIO()
    with mpl.rc_context({"svg.hashsalt": "phonometry", "svg.fonttype": "path"}):
        fig.savefig(buffer, format="svg", metadata={"Date": None})
    return buffer.getvalue()


def test_the_measurement_puts_every_artist_back() -> None:
    """The whole design rests on this: the figure is written straight after.

    Measuring means hiding artists, taking their strokes off, lifting a label's
    zorder and repainting its letters in another colour, on the live figure
    that is about to be saved. One artist left as the measurement found it
    convenient corrupts a committed SVG, and nothing else in the pipeline would
    say so: ``check_figures.py`` compares the file against a fresh render of
    the same code, so both sides would carry the same corruption.

    So this compares the bytes, and every attribute behind them, across a
    figure holding one of every artist family the measurement touches.
    """
    fig = _menagerie()
    fig.canvas.draw()
    before, state = _rendered(fig), _state(fig)

    hits = _measure(fig)

    assert {hit["kind"] for hit in hits} == {audit.BEHIND, audit.COVERED}, (
        "a figure that records nothing would pass this test without measuring"
    )
    assert _state(fig) == state
    assert _rendered(fig) == before


def test_measuring_twice_gives_the_same_answer() -> None:
    """The other half of leaving no trace: the second pass sees the same figure."""
    fig = _menagerie()
    fig.canvas.draw()

    first = [(hit["kind"], hit["text"], hit["pixels"]) for hit in _measure(fig, "one")]
    second = [(hit["kind"], hit["text"], hit["pixels"]) for hit in _measure(fig, "two")]

    assert first == second
    assert first


def test_a_clean_figure_still_records_that_it_was_drawn() -> None:
    """ "Measured and clean" and "not in this run" are different answers."""
    fig, ax = _blank()
    ax.plot([0, 1], [0.1, 0.1], color="black")
    ax.text(0.5, 0.8, "well clear", ha="center", va="center")

    assert _measure(fig) == []
    assert "probe" in audit._FOUND


def test_nothing_is_measured_when_recording_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain ``generate_graphs.py`` run pays none of this."""
    monkeypatch.delenv(audit.AUDIT_ENV)
    fig, ax = _blank()
    ax.plot([0, 1], [0.5, 0.5], color="black")
    ax.text(0.5, 0.5, "struck", ha="center", va="center")

    audit.audit(fig, "probe")

    assert audit._FOUND == {}


def test_a_canvas_with_no_pixels_to_read_says_so() -> None:
    """The measurement is pixels, so it needs a raster canvas and demands one."""
    from matplotlib.backends.backend_svg import FigureCanvasSVG

    fig, _ = _blank()
    FigureCanvasSVG(fig)

    with pytest.raises(RuntimeError, match="buffer_rgba"):
        audit._render(fig)


def test_the_fragments_of_two_runs_merge_on_the_worse_count(
    tmp_path: pathlib.Path,
) -> None:
    """Merging may under-report a fix; it may never invent one."""
    directory = tmp_path / "fragments"
    directory.mkdir()
    for pid, pixels in ((1, 40), (2, 90)):
        (directory / f"{pid}.json").write_text(
            json.dumps({"probe": [{"text": "a", "pixels": pixels, "dpi": 150.0}]}),
            encoding="utf-8",
        )

    assert audit.load(str(directory))["probe"][0]["pixels"] == 90


# --------------------------------------------------------------------------
# What the narrow rule could not see: another axes, another artist class,
# another language, and a chip that is itself painted over.


def test_a_curve_on_the_axes_next_door_still_runs_through_the_label() -> None:
    """The ink of a figure is not ``ax.get_lines()`` of one of its axes.

    A ``twinx`` sibling draws on the same rectangle, so its curve crosses an
    annotation on the host exactly as one of the host's own would. Reading
    only the label's own axes left ``sii_masking_chain`` (345 px, an orange
    twin-axes curve straight through a three-line annotation) recorded as
    clean.
    """
    fig, ax = _blank()
    ax.text(0.5, 0.5, "struck from next door", ha="center", va="center")
    twin = ax.twinx()
    twin.set_ylim(0, 1)
    twin.plot([0, 1], [0.5, 0.5], color="black", label="the sibling's curve")

    (hit,) = _of_kind(_measure(fig), audit.BEHIND)

    assert hit["pixels"] > check.GATE_PX[audit.BEHIND]
    assert hit["struck_by"][0][0] == "the sibling's curve"


def test_a_stem_that_is_not_a_line2d_is_ink_all_the_same() -> None:
    """``ax.get_lines()`` is one artist class of several that put strokes down.

    A reflectogram draws its stems as a ``LineCollection``, and a collection
    is also where a contour and a shaded outline live. It has to be seen by
    both stages: the prefilter has to nominate the label, and the pixel count
    has to include the collection's ink. The prefilter cannot get its box from
    ``get_window_extent`` either, which answers with the empty box for any
    collection added to an axes.
    """
    from matplotlib.collections import LineCollection

    fig, ax = _blank()
    stems = LineCollection(
        [[(x, 0.0), (x, 1.0)] for x in (0.44, 0.48, 0.52, 0.56)],
        colors="black",
        linewidths=1.5,
        label="the stems",
    )
    ax.add_collection(stems)
    ax.text(0.5, 0.5, "between the stems", ha="center", va="center")

    assert _prefiltered(fig), "the prefilter must nominate the label"
    (hit,) = _of_kind(_measure(fig), audit.BEHIND)

    assert hit["pixels"] > check.GATE_PX[audit.BEHIND]
    assert hit["struck_by"][0][0] == "the stems"


def test_a_label_on_a_filled_shape_is_backed_and_not_struck() -> None:
    """A fill under a label is a backing, which is the whole point of the chip.

    Counting a patch as ink and hiding it wholesale to find out what it paints
    makes every label inside a shaded shape fail: on ``metadiffuser_geometry``
    that scored the five slit numbers, black on solid pale blue and perfectly
    legible, at 32 to 47 px. Only what the patch *strokes* counts.
    """
    from matplotlib.patches import Rectangle

    fig, ax = _blank()
    ax.add_patch(
        Rectangle((0.2, 0.35), 0.6, 0.3, facecolor="#aaccee", edgecolor="none")
    )
    ax.text(0.5, 0.5, "on a panel", ha="center", va="center")

    assert _measure(fig) == []


def test_the_outline_of_that_same_shape_is_a_stroke() -> None:
    """The fill is a backing and the edge is not, on one artist."""
    from matplotlib.patches import Rectangle

    fig, ax = _blank()
    ax.add_patch(
        Rectangle(
            (0.2, 0.35),
            0.6,
            0.3,
            facecolor="#aaccee",
            edgecolor="black",
            linewidth=6.0,
            label="the panel",
        )
    )
    ax.text(0.5, 0.65, "on the edge", ha="center", va="center")

    (hit,) = _of_kind(_measure(fig), audit.BEHIND)

    assert hit["pixels"] > check.GATE_PX[audit.BEHIND]
    assert hit["struck_by"][0][0] == "the panel"


def test_a_band_edged_in_its_own_colour_strokes_nothing() -> None:
    """``axhspan(color=...)`` sets the edge and the face to the same shade.

    What that "stroke" paints is a one-pixel fringe of the band's own colour
    around the band: a step from the fill to the page, not a line across
    anything. On ``mobility_result_lines`` the note sits on the +-90 deg band
    with the top edge running through its first row of letters, and counting
    the fringe scored a perfectly legible label at 282 px.
    """
    fig, ax = _blank()
    ax.axhspan(0.0, 0.5, color="#c8ddf0")
    ax.text(0.5, 0.5, "along the band edge", ha="center", va="center")

    assert _measure(fig) == []


def test_a_chip_drawn_under_the_curve_is_not_a_fix() -> None:
    """ "Has a box" is not "is readable", and the behind measure cannot tell.

    The convention is a chip *and* a zorder above the curves; 122 of the 194
    calls in the corpus that pass a bbox set no zorder, and where the curve
    sets one the chip goes under it. ``coupling_term_regimes`` is the shipped
    case: a correctly chipped equation with the regime curve drawn at
    ``zorder=4`` straight over it.
    """
    fig, ax = _blank()
    ax.text(
        0.5,
        0.5,
        "boxed, and painted over",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f0f2f5",
            "edgecolor": "#e0e0e0",
        },
    )
    ax.plot([0, 1], [0.5, 0.5], color="black", linewidth=6.0, zorder=4, label="on top")

    hits = _measure(fig)

    assert _of_kind(hits, audit.BEHIND) == []
    (hit,) = _of_kind(hits, audit.COVERED)
    assert hit["pixels"] > check.GATE_PX[audit.COVERED]
    assert hit["struck_by"][0][0] == "on top"


def test_a_clipped_chip_with_its_letters_intact_is_not_the_defect() -> None:
    """The covered measure counts letters, and the chip is not a letter.

    ``Text.draw`` draws the chip, so the render that locates the label's ink
    raises the box along with the letters: taken that way the count of a
    chipped label is mostly fill. Measured both ways over the corpus the two
    diverged by up to seventy-seven times -- ``rd1367_vs_iso_tonal_es`` at
    237 px of lost chip against 0 px of lost letters, every character
    perfect, and ``ship_source_level`` at 384 against 5 -- and the clean
    population reached higher (384 px) than the real one began, so no
    threshold could separate them.
    """
    from matplotlib.patches import Rectangle

    fig, ax = _blank()
    ax.text(
        0.5,
        0.5,
        "letters",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "square,pad=2.0",
            "facecolor": "#f0f2f5",
            "edgecolor": "#e0e0e0",
        },
    )
    # Over one corner of the chip and nowhere near a character.
    ax.add_patch(Rectangle((0.0, 0.0), 0.36, 0.36, facecolor="black", zorder=4))

    assert _of_kind(_measure(fig), audit.COVERED) == []


def test_the_same_chip_with_its_letters_covered_is_the_defect() -> None:
    """The control for the one above: move the cover onto the words."""
    from matplotlib.patches import Rectangle

    fig, ax = _blank()
    ax.text(
        0.5,
        0.5,
        "letters",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "square,pad=2.0",
            "facecolor": "#f0f2f5",
            "edgecolor": "#e0e0e0",
        },
    )
    ax.add_patch(Rectangle((0.4, 0.4), 0.36, 0.36, facecolor="black", zorder=4))

    (hit,) = _of_kind(_measure(fig), audit.COVERED)

    assert hit["pixels"] > check.GATE_PX[audit.COVERED]


def test_a_chip_above_the_curve_is_the_fix() -> None:
    """The same figure done right records nothing, so the gate has a way out."""
    fig, ax = _blank()
    ax.text(
        0.5,
        0.5,
        "boxed, and on top",
        ha="center",
        va="center",
        zorder=5,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f0f2f5",
            "edgecolor": "#e0e0e0",
        },
    )
    ax.plot([0, 1], [0.5, 0.5], color="black", linewidth=6.0, zorder=4)

    assert _measure(fig) == []


def test_a_label_a_zorder_cannot_lift_is_measured_by_taking_the_cover_away() -> None:
    """A zorder orders artists inside one axes, and no further.

    The figure draws each axes whole, so a label on the earlier of two axes
    cannot be raised over the later one's curve at all: put at a zorder of a
    million it renders exactly as it shipped, and the measure would score it
    zero for the wrong reason. ``transfer_stiffness`` is the shipped case, its
    ``twinx`` label under the host curve the figure deliberately puts on top.
    """
    fig, ax = _blank()
    ax.text(0.5, 0.5, "painted over from next door", ha="center", va="center")
    twin = ax.twinx()
    twin.set_ylim(0, 1)
    twin.plot([0, 1], [0.5, 0.5], color="black", linewidth=6.0, label="from next door")

    (hit,) = _of_kind(_measure(fig), audit.COVERED)

    assert hit["pixels"] > check.GATE_PX[audit.COVERED]
    assert hit["struck_by"][0][0] == "from next door"


def test_one_label_drawn_over_another_is_named_by_what_it_says() -> None:
    """A label can be covered by a label, and "unnamed Text" is no help.

    ``decay_range_bias`` ships it: a rotated rule label swallowed whole by the
    chipped corner box added later, of which one row of pixels shows below the
    box. Both are ``ax.texts`` at the same zorder, so the one added last wins.
    """
    fig, ax = _blank()
    ax.text(0.5, 0.5, "underneath", ha="center", va="center")
    ax.text(
        0.5,
        0.5,
        "the corner box",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0f2f5"},
    )

    (hit,) = _of_kind(_measure(fig), audit.COVERED)

    assert hit["text"] == "underneath"
    assert hit["pixels"] > check.GATE_PX[audit.COVERED]
    assert hit["struck_by"][0][0] == "the label 'the corner box'"


def test_switching_the_words_off_paints_exactly_nothing() -> None:
    """What lets an annotation stay in the render while its words leave it.

    The ground the behind measure compares against is the figure with the
    labels out of it, and an annotation has to stay in that render or its arrow
    goes with it. So its letters are drawn in a transparent ink and its chip is
    hidden, which is only the same thing as taking the label away if the
    transparent ink changes not one pixel. With the arrow hidden as well, the
    two renders have to be identical.
    """
    fig, ax = _blank()
    ax.plot([0, 1], [0.5, 0.5], color="black")
    note = ax.annotate(
        "words on a chip",
        xy=(0.9, 0.5),
        xytext=(0.15, 0.8),
        arrowprops={"arrowstyle": "-"},
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f0f2f5"},
    )
    fig.canvas.draw()

    with audit._hidden([note]):
        gone = audit._render(fig)
    with audit._lettering_off([note]), audit._hidden([note.arrow_patch]):
        muted = audit._render(fig)

    assert np.array_equal(muted, gone)


def test_an_arrow_across_the_label_next_door_is_a_stroke_like_any_other() -> None:
    """The control for the arrow that points at its own curve, one figure over.

    An ``Annotation``'s arrow is not in ``ax.patches``: the annotation holds it
    and draws it itself. Hiding the labels to build the ground therefore took
    the arrow out of the render with them, and a black rule straight through
    somebody else's words scored nothing. Measured on this figure with the
    stroke drawn as a ``Line2D`` instead, the same victim scores 287 px.
    """
    fig, ax = _blank()
    ax.annotate(
        "the source",
        xy=(0.95, 0.5),
        xytext=(0.03, 0.5),
        arrowprops={"arrowstyle": "-", "lw": 1.5, "color": "black"},
    )
    ax.text(0.55, 0.5, "the victim", ha="center", va="center")

    (hit,) = _of_kind(_measure(fig), audit.BEHIND)

    assert hit["text"] == "the victim"
    assert hit["pixels"] > check.GATE_PX[audit.BEHIND]
    assert hit["struck_by"][0][0] == "the pointer from 'the source'"


def test_something_the_figure_draws_over_a_panel_is_seen() -> None:
    """``fig.text`` and friends are drawn after every axes, and are in no axes.

    Both measures walked ``fig.axes``, so an opaque figure-level box laid over
    an axes label destroyed the words and scored zero on each. A zorder cannot
    answer it either -- the label is inside an axes and the box is not -- which
    is why it joins the artists the covered measure takes away.
    """
    fig, ax = _blank()
    ax.text(0.5, 0.5, "cannot be read", ha="center", va="center")
    fig.text(
        0.5,
        0.5,
        "                    ",
        ha="center",
        va="center",
        bbox={"boxstyle": "square,pad=0.6", "facecolor": "black", "edgecolor": "black"},
    )

    (hit,) = _of_kind(_measure(fig), audit.COVERED)

    assert hit["text"] == "cannot be read"
    assert hit["pixels"] > check.GATE_PX[audit.COVERED]


def test_an_inset_is_an_axes_the_figure_does_not_list() -> None:
    """``inset_axes`` files its result under the host, not under the figure.

    So the inset's curves were not ink, a label on the inset was never
    enumerated, and a host label under the inset panel was not seen. All three
    follow from the same walk, and this pins the last of them: the inset is
    opaque and drawn inside the host, so nothing the host label does with its
    zorder gets it back.
    """
    fig, ax = _blank()
    ax.text(0.5, 0.5, "under the inset", ha="center", va="center")
    inset = ax.inset_axes((0.25, 0.25, 0.5, 0.5))
    inset.set_xticks([])
    inset.set_yticks([])

    assert inset not in fig.axes, "the figure cannot name it, which is the point"
    (hit,) = _of_kind(_measure(fig), audit.COVERED)

    assert hit["text"] == "under the inset"
    assert hit["pixels"] > check.GATE_PX[audit.COVERED]


def test_capturing_the_box_during_the_draw_moves_no_ordinary_label() -> None:
    """The 3-D repair has to leave the other 890 drawings' numbers alone.

    Every threshold in the checker was calibrated against boxes asked for after
    the draw, so the capture is only safe if it answers the same thing for a
    label whose position does not change inside its own draw -- which is every
    label that is not on a 3-D plate.
    """
    fig = _menagerie()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    after = {
        id(text): Text.get_window_extent(text, renderer)
        for ax in fig.axes
        for text in ax.texts
    }

    during = audit._glyph_boxes(fig)

    assert after
    for known, box in after.items():
        assert during[known].bounds == box.bounds


def test_a_label_on_a_three_dimensional_plate_is_measured_where_it_is_drawn() -> None:
    """A 3-D plate projects its text inside its own draw, and puts it back after.

    ``Text3D.draw`` sets ``_x`` and ``_y`` to the projected display coordinates
    for the duration of the draw and restores the data ones on the way out, so
    a window extent asked for afterwards runs three-dimensional data through
    ``transData``. On ``microphone_positions_hemisphere`` that answered
    ``x = 12264.9`` on a 1350 px canvas: every box missed the panel, every
    label of both plates was dropped, and the recording said the drawings were
    measured and clean when nothing had been measured. Capturing the box while
    the label draws is what makes the answer the same question the reader asks.
    """
    fig = plt.figure(figsize=(4, 3), dpi=audit.REFERENCE_DPI)
    ax = fig.add_subplot(projection="3d")
    ax.plot([0, 1], [0, 1], [0.5, 0.5], color="black", linewidth=6.0, label="the wire")
    text = ax.text(0.5, 0.5, 0.5, "on the wire", ha="center", va="center")

    stale = Text.get_window_extent(text, fig.canvas.get_renderer())
    assert not Bbox.intersection(stale, ax.bbox), (
        "the point of the capture is that the extent asked for afterwards is wrong"
    )

    (hit,) = _of_kind(_measure(fig), audit.BEHIND)

    assert hit["text"] == "on the wire"
    assert hit["pixels"] > check.GATE_PX[audit.BEHIND]
    assert hit["struck_by"][0][0] == "the wire"


def test_the_spanish_pass_is_measured_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Spanish prose is longer, and half the corpus is Spanish.

    The English label here stops well short of the curve and the Spanish one
    runs into it, which is the whole class: eight figures measured over the
    gate in Spanish alone. So the measurement has to run on both light passes
    and *after* the translation, and the record has to say which drawing it
    is talking about.
    """
    from figures import i18n, theme

    english = "brief"
    spanish = "una anotación considerablemente más larga que la inglesa"
    monkeypatch.setitem(i18n._ES_EXACT, english, spanish)

    saved = dict(mpl.rcParams)
    try:
        for lang, dark in (("en", False), ("es", False), ("es", True)):
            i18n.set_lang(lang)
            theme.set_theme(dark)
            fig, ax = _blank()
            ax.plot([0.45, 1.0], [0.5, 0.5], color="black", linewidth=6.0)
            ax.text(0.02, 0.5, english, ha="left", va="center")
            theme.save_figure(str(tmp_path), "probe.svg")
            plt.close(fig)
    finally:
        # ``set_lang`` and ``set_theme`` rebind module globals and the whole
        # rcParams; the next test must not inherit a dark Spanish figure.
        i18n.set_lang("en")
        theme.set_theme(False)
        mpl.rcParams.update(saved)

    assert audit._FOUND["probe"] == []
    (hit,) = _of_kind(audit._FOUND["probe_es"], audit.BEHIND)
    assert hit["text"] == spanish
    assert hit["pixels"] > check.GATE_PX[audit.BEHIND]
    assert "probe_es_dark" not in audit._FOUND, "the dark twin is the same drawing"


# --------------------------------------------------------------------------
# The checker: the arithmetic on top of the recording.


def _recording(
    pixels: int, text: str = "a label", kind: str = audit.BEHIND
) -> dict[str, list[dict[str, Any]]]:
    """One measured figure, losing *pixels* to *kind*."""
    return {
        "probe": [
            {
                "kind": kind,
                "text": text,
                "pixels": pixels,
                "dpi": check.REQUIRED_DPI,
                "struck_by": [["the curve", pixels]],
            }
        ]
    }


def test_the_gate_fails_at_the_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    """45 px is the rule for a curve behind the letters, and it is inclusive."""
    behind = check.GATE_PX[audit.BEHIND]

    assert check.report(_recording(behind), {}, partial=False) == 1
    assert "carry no chip" in capsys.readouterr().out


def test_the_band_below_the_gate_prints_and_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Between 32 and 44 px a struck label and a clean one cannot be told apart."""
    below = check.GATE_PX[audit.BEHIND] - 1

    assert check.report(_recording(below), {}, partial=False) == 0
    printed = capsys.readouterr().out
    assert "Advisory" in printed
    assert f"{below} px" in printed


def test_a_touch_below_the_advisory_floor_is_not_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stroke grazing one character is not something to tell anybody about."""
    quiet = check.ADVISORY_PX[audit.BEHIND] - 1

    assert check.report(_recording(quiet), {}, partial=False) == 0
    assert "Advisory" not in capsys.readouterr().out


def test_the_covered_gate_is_the_lower_one_and_has_no_advisory_band(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The two defects are two measurements, and each carries its own threshold.

    Covered pixels are ink the reader never receives, so they destroy a
    character outright; a struck pixel is a character still fully drawn and
    merely competing with a curve. Measured on the corpus the covered classes
    part where the behind ones interleave -- 18 px clean against 21 px
    unreadable -- so covered gates at 20 and says nothing at all below it.
    """
    covered = check.GATE_PX[audit.COVERED]
    assert covered < check.GATE_PX[audit.BEHIND]

    assert check.report(_recording(covered, kind=audit.COVERED), {}, partial=False) == 1
    assert "painted over" in capsys.readouterr().out

    under = _recording(covered - 1, kind=audit.COVERED)
    assert check.report(under, {}, partial=False) == 0
    assert "Advisory" not in capsys.readouterr().out


def test_the_two_gates_do_not_borrow_each_other_s_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A count that fails one measure passes the other, and that is the point."""
    covered = check.GATE_PX[audit.COVERED]

    assert check.report(_recording(covered, kind=audit.BEHIND), {}, partial=False) == 0
    assert "Advisory" not in capsys.readouterr().out


def test_an_exemption_covers_the_label_it_names() -> None:
    """Some annotations belong on the curve, and the answer is a recorded decision."""
    exempt = {("probe", "a label"): "the curve has to be seen continuing under it"}

    assert check.report(_recording(90), exempt, partial=False) == 0


def test_an_exemption_that_stopped_being_true_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Once the label is chipped the line covers nothing, and has to go."""
    exempt = {("probe", "a label"): "a reason that no longer applies"}

    assert check.report({"probe": []}, exempt, partial=False) == 1
    assert "no longer true" in capsys.readouterr().out


def test_an_exemption_for_a_figure_nobody_draws_any_more_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A full run covers every committed figure, so a missing stem is a gone one."""
    exempt = {("gone", "a label"): "a reason for a figure that no longer exists"}

    assert check.report(_recording(90), exempt, partial=False) == 1
    assert "no longer true" in capsys.readouterr().out


def test_a_partial_run_reports_no_exemption_as_stale() -> None:
    """A run that did not draw the figure cannot say anything about it."""
    exempt = {("probe", "a label"): "a reason", ("elsewhere", "x"): "another"}

    assert check.report({"probe": [], **_recording(90)}, exempt, partial=True) == 0


def test_a_recording_at_another_resolution_is_refused() -> None:
    """The count scales with dpi squared, so the number means nothing without it."""
    other = _recording(90)
    other["probe"][0]["dpi"] = check.REQUIRED_DPI * 2

    assert check.wrong_dpi(other) == {check.REQUIRED_DPI * 2}


def test_an_exemption_line_must_carry_a_reason(tmp_path: pathlib.Path) -> None:
    """A bare pair of names is not a decision, so it is not accepted as one."""
    path = tmp_path / "exemptions.txt"
    path.write_text('probe: "a label"\n', encoding="utf-8")

    with pytest.raises(SystemExit, match='stem: "label": reason'):
        check.read_exemptions(path)


def test_each_kind_is_told_what_to_do_about_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A chip answers one defect and does nothing at all for the other."""
    recorded = {
        "probe": [
            *_recording(90, "behind a curve", audit.BEHIND)["probe"],
            *_recording(90, "under a curve", audit.COVERED)["probe"],
        ]
    }

    assert check.report(recorded, {}, partial=False) == 1
    printed = capsys.readouterr().out
    assert check.HEADLINE[audit.BEHIND] in printed
    assert check.HEADLINE[audit.COVERED] in printed
    assert 'bbox={"boxstyle"' in printed
    assert "zorder higher" in printed


def test_the_two_kinds_of_one_label_are_merged_apart(tmp_path: pathlib.Path) -> None:
    """One label can be both, and the two counts are not the same number."""
    directory = tmp_path / "fragments"
    directory.mkdir()
    (directory / "1.json").write_text(
        json.dumps(
            {
                "probe": [
                    {"kind": audit.BEHIND, "text": "a", "pixels": 90, "dpi": 150.0},
                    {"kind": audit.COVERED, "text": "a", "pixels": 40, "dpi": 150.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    kept = {hit["kind"]: hit["pixels"] for hit in audit.load(str(directory))["probe"]}

    assert kept == {audit.BEHIND: 90, audit.COVERED: 40}


def test_a_full_run_owes_both_languages() -> None:
    """Half the corpus is Spanish, and it is a different drawing."""
    figures = check.committed_figures()

    assert "g_weighting_response" in figures
    assert "g_weighting_response_es" in figures
    assert not [name for name in figures if name.endswith("_dark")]


def test_an_exemption_line_is_read_back_whole(tmp_path: pathlib.Path) -> None:
    """A label with a colon, a comma and a newline in it still parses."""
    path = tmp_path / "exemptions.txt"
    path.write_text(
        '# a comment\nprobe: "first: second\\nthird": because it must be, here\n',
        encoding="utf-8",
    )

    assert check.read_exemptions(path) == {
        ("probe", "first: second\nthird"): "because it must be, here"
    }


# --------------------------------------------------------------------------
# The command line: the refusals, the exit codes and what gets printed. All of
# them live in ``main`` and are reachable only through it.


def _fragment(directory: pathlib.Path, **hits: list[dict[str, Any]]) -> pathlib.Path:
    """A recording directory holding one fragment, keyed by drawing."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "1.json").write_text(json.dumps(hits), encoding="utf-8")
    return directory


def _hit(
    pixels: int, kind: str = audit.BEHIND, text: str = "a label"
) -> dict[str, Any]:
    """One recorded label, as a fragment on disk holds it."""
    return {
        "kind": kind,
        "text": text,
        "pixels": pixels,
        "dpi": check.REQUIRED_DPI,
        "struck_by": [["the curve", pixels]],
    }


def test_no_recording_at_all_is_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """The check answers about a generation run, so with none it says so.

    A directory that is not there and a directory that is there and empty are
    the same answer: whoever ran this has not run `make graphs`, and printing
    "nothing is unreadable" would be a lie.
    """
    assert check.main(["--audit", str(tmp_path / "never-written")]) == 1
    assert "no figure-annotation recording" in capsys.readouterr().out

    (tmp_path / "empty").mkdir()
    assert check.main(["--audit", str(tmp_path / "empty")]) == 1
    assert "no figure-annotation recording" in capsys.readouterr().out


def test_a_recording_taken_at_another_resolution_is_refused_by_the_command(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """A struck-pixel count scales with dpi squared, so it is not rescaled."""
    other = _hit(90)
    other["dpi"] = check.REQUIRED_DPI * 2
    directory = _fragment(tmp_path / "rec", probe=[other])

    assert check.main(["--audit", str(directory)]) == 1
    printed = capsys.readouterr().out
    assert f"taken at [{check.REQUIRED_DPI * 2}] dpi" in printed
    assert "not comparable" in printed


def test_a_run_that_drew_only_some_figures_is_refused_without_the_flag(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """Reading a directory some other run filled answers about the wrong tree.

    The recording has to cover every committed drawing in both languages, or
    the pass is only a statement about whatever happened to be re-rendered.
    """
    directory = _fragment(tmp_path / "rec", g_weighting_response=[])

    assert check.main(["--audit", str(directory)]) == 1
    printed = capsys.readouterr().out
    assert "is not a full run" in printed
    assert "make graphs" in printed


def test_the_same_partial_run_is_accepted_with_the_flag(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """``--partial`` is for a targeted re-render: it checks what that run drew."""
    directory = _fragment(tmp_path / "rec", g_weighting_response=[])

    assert check.main(["--audit", str(directory), "--partial"]) == 0
    assert "No unreadable annotation" in capsys.readouterr().out


def test_the_command_fails_on_an_unreadable_label_and_names_it(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """The whole point, driven the way CI drives it."""
    directory = _fragment(tmp_path / "rec", probe=[_hit(check.GATE_PX[audit.BEHIND])])

    assert check.main(["--audit", str(directory), "--partial"]) == 1
    printed = capsys.readouterr().out
    assert "::error::an annotation on a figure cannot be read" in printed
    assert "probe" in printed
    assert "the curve" in printed


def test_the_command_prints_the_advisory_band_and_still_passes(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """A count in the band is reported for a person to judge, and does not fail."""
    band = check.GATE_PX[audit.BEHIND] - 1
    directory = _fragment(tmp_path / "rec", probe=[_hit(band)])

    assert check.main(["--audit", str(directory), "--partial"]) == 0
    printed = capsys.readouterr().out
    assert "Advisory" in printed
    assert f"{band} px" in printed


def test_the_command_reads_the_committed_exemption_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    """An exemption is a recorded decision, and the command is what honours it."""
    exemptions = tmp_path / "exemptions.txt"
    exemptions.write_text(
        'probe: "a label": the curve has to be seen continuing under it\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(check, "EXEMPTIONS", exemptions)
    directory = _fragment(tmp_path / "rec", probe=[_hit(90)])

    assert check.main(["--audit", str(directory), "--partial"]) == 0
    assert "1 exempt" in capsys.readouterr().out


def test_the_command_fails_on_an_exemption_that_stopped_being_true(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    """Only on a full run: a partial one cannot know the label is gone."""
    exemptions = tmp_path / "exemptions.txt"
    exemptions.write_text('probe: "a label": a reason that no longer applies\n')
    monkeypatch.setattr(check, "EXEMPTIONS", exemptions)
    monkeypatch.setattr(check, "committed_figures", lambda: {"probe"})
    directory = _fragment(tmp_path / "rec", probe=[])

    assert check.main(["--audit", str(directory)]) == 1
    assert "no longer true" in capsys.readouterr().out

    assert check.main(["--audit", str(directory), "--partial"]) == 0


def test_the_committed_exemption_file_parses() -> None:
    """The file CI reads is read here too, so a typo in it fails a test."""
    assert isinstance(check.read_exemptions(check.EXEMPTIONS), dict)


# --------------------------------------------------------------------------
# One process, one fragment -- including a process that arrived by fork.


def _record_in_child(key: str) -> None:
    """A forked child's whole job: measure one figure and exit normally."""
    audit.audit(plt.figure(), key)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the inherited state this pins exists only where os.fork does",
)
def test_a_child_forked_after_the_parent_recorded_still_writes_its_own_fragment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fork child records for itself, or the gate measures less than the corpus.

    ``os.fork`` copies the tally and the "handler is registered" flag into the
    child, but ``multiprocessing`` empties the child's ``atexit`` registry
    before the target runs. A child that trusted the inherited flag would
    therefore never register a handler of its own, and everything it measured
    would go unwritten -- a recording short of the corpus, which is a defect
    the gate never sees.

    The fragment holds the child's drawing and not the parent's: each process
    writes what it measured, so :func:`audit.load` merging the two cannot
    credit one process with the other's work.
    """
    directory = pathlib.Path(os.environ[audit.AUDIT_ENV])
    monkeypatch.setattr(audit, "_REGISTERED", False)
    audit.audit(plt.figure(), "drawn_in_the_parent")
    assert audit._REGISTERED, "the parent has to have registered for this to bite"

    child = multiprocessing.get_context("fork").Process(
        target=_record_in_child, args=("drawn_in_the_child",)
    )
    child.start()
    child.join()

    assert child.exitcode == 0
    fragment = directory / f"{child.pid}.json"
    assert fragment.exists(), "the child measured a drawing and wrote nothing down"
    assert json.loads(fragment.read_text(encoding="utf-8")) == {
        "drawn_in_the_child": []
    }


def test_a_recording_one_fragment_short_is_refused_rather_than_passed(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """The failure mode a lost fragment would produce, and what it costs.

    A process whose fragment never lands takes its drawings out of the
    recording, and the coverage rule is what stands between that and a green
    gate: the same directory passes whole and is refused with one fragment
    removed. So the worst a lost fragment can do is stop the run, not let a
    defect through.
    """
    directory = tmp_path / "rec"
    directory.mkdir()
    drawings = sorted(check.committed_figures())
    half = len(drawings) // 2
    for name, keys in (("1.json", drawings[:half]), ("2.json", drawings[half:])):
        (directory / name).write_text(
            json.dumps(dict.fromkeys(keys, [])), encoding="utf-8"
        )

    assert check.main(["--audit", str(directory)]) == 0
    assert "No unreadable annotation" in capsys.readouterr().out

    (directory / "2.json").unlink()

    assert check.main(["--audit", str(directory)]) == 1
    printed = capsys.readouterr().out
    assert "is not a full run" in printed
    assert "make graphs" in printed
