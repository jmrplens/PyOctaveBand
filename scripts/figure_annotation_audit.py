#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Record, while the figures are being generated, which annotations cannot be read.

A label the reader cannot pick out of the plot fails in one of two ways, and
they are not the same failure:

* **behind.** The letters are drawn on top of a curve, with nothing between
  them. Both strokes are the same weight, and on the dark page close to the
  same lightness. The answer is the opaque chip the corpus already uses
  (``bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL, "edgecolor":
  COLOR_GRID}``), and the defect is a label that should carry one and does not.
* **covered.** The letters are painted over by something drawn later. A chip
  does not answer this one -- the chip is being painted over too -- and the
  fix is a ``zorder`` that puts the label above whatever is on top of it.

Neither subsumes the other, and the second is easy to miss precisely because
matplotlib draws text at zorder 3 and lines at zorder 2: in the ordinary
"behind" defect every letter is fully visible, and merely illegible. So the
two are measured separately and reported as separate kinds, because the advice
they need is different.

Why it is measured here, during generation, rather than read off the shipped
files: ``svg.fonttype = "path"`` means text is written as glyph outlines, so
``.github/images`` contains no ``<text>`` element to find a label in and no
way to ask where its letters are. The matplotlib artists are the only place
the question can be put, and they exist only while the figure is being drawn.

The "behind" measure, in two stages, because the precise one is expensive
-------------------------------------------------------------------------

* **The prefilter** is geometry: for every un-chipped ``Text`` of the figure,
  does any visible artist that puts a stroke down enter its glyph box? Every
  axes counts, not only the one the label lives in -- a label on a ``twinx``
  host is drawn straight through by the curve on its sibling -- and every
  stroking artist counts, not only ``Line2D``: a reflectogram stem drawn as a
  ``LineCollection``, a contour, the outline or the hatching of a patch and
  the arrow of a neighbouring annotation are ink like any other. It runs on
  every figure, needs one draw, and is deliberately generous: it decides which
  figures are worth measuring, and nothing else, so for an artist whose exact
  vertices are not on offer a bounding-box overlap is enough and the pixel
  count settles it afterwards.
* **The measurement** is pixels, and it is what the gate compares. The figure
  is rendered with the labels' letters off and every stroke suppressed (the
  ground), then with one label alone, then with the strokes back; differencing
  gives the ink each paints on that same ground, and the count of pixels in
  both is the answer.

An annotation's own arrow is kept out of its own count twice over, because a
label set aside with a pointer that reaches down to touch the curve is the
pattern this looks *for*. The arrow is a stroke, so it is suppressed in the
render the label's letters are read from, exactly as a curve is; and that
render is masked to the glyph box, which comes from an unbound
``Text.get_window_extent`` and is the letters alone. What is *not* excused is
an arrow across the label next door, which is a black rule through the words
and reads no better than a curve would. It is why the letters are switched off
rather than the labels hidden: hiding an ``Annotation`` takes its arrow with
it, so the stroke to be measured would not be in the render at all. Measured on
two figures identical but for how the one stroke is drawn, the victim scored
287 px as a ``Line2D`` and 0 px as an ``annotate(arrowstyle="-")`` across the
same pixels.

What a stroke is, and what it is not, is load-bearing. A patch or a collection
is suppressed by taking its line width to zero and its hatching away, not by
hiding it, so its *fill* stays in the ground. A filled region behind a label
is a backing -- which is what the chip is for -- and the letters on it read
perfectly: measured on ``metadiffuser_geometry``, hiding the fills as well
scored the five legible slit numbers at 32 to 47 px, straight into the gate.
For a figure whose only ink is ``Line2D`` this is the same arithmetic as
hiding the lines, because a line has no fill to keep.

The geometric answer alone cannot gate: ``get_window_extent`` returns the
font's line box, ascent to descent, plus the leading between the lines of a
multi-line annotation, so a curve can cross the box through the ascender space
above a row of lowercase letters and touch nothing. Measured across the
corpus, a curve inside the box with zero ink on the letters is the *dominant*
class of geometric hit, which is why the box is a prefilter and the pixels are
the rule.

Two things fall out of measuring pixels rather than distances, and both are
wanted. A curve hidden under a filled band contributes nothing, because the
stroke mask is read against a render with every fill in place, so ink
something else paints over never enters it. And a curve that merely ends on
the edge of the box -- the inline curve-end label, which reads fine -- scores
the width of its own end cap against the first letter, a few pixels, an order
of magnitude below anything the checker reports.

The "covered" measure
---------------------

Four renders per label, and it asks the reader's question directly: of the
ink this label lays down, how much fails to survive into the figure as it
ships?

.. code-block:: text

    raised_dark   = render(fig)  label on top, its letters drawn in black
    raised_light  = render(fig)  label on top, its letters drawn in white
    shipped_dark  = render(fig)  label where it ships, its letters in black
    shipped_light = render(fig)  label where it ships, its letters in white
    glyphs  = (raised_dark  != raised_light)  & region
    visible = (shipped_dark != shipped_light) & region
    struck  = glyphs & ~visible

Rendering the label at the top of the stack is the whole trick. Locating it
in a figure that still has the curves drawn over it finds only the part that
already survives, which is the very thing being tested, and returns zero for
every figure in the corpus.

Nothing in it depends on the type of the covering artist, on its zorder, or on
whether the label carries a chip -- which is why it is applied to chipped
labels too. "Has a box" is not "is readable": the convention is a chip *and* a
zorder above the curves, and a chip drawn under the curve is exactly the case
that the behind measure passes and the reader fails. It is not free, so a
label is only measured when some artist that is drawn after it reaches its
box.

What is counted is letters, never chip. That distinction is what makes the
number gateable. The chip is drawn by ``Text.draw``, so raising the label
raises the box with it, and a glyph mask read straight off that render is the
chip's whole rectangle: the count would be mostly fill. Measured both ways
across the corpus the two diverge by up to seventy-seven times:
``rd1367_vs_iso_tonal_es`` scores 237 px of lost chip and 0 px of lost letters:
a legend frame trimming the top-right corner of the box, every character
perfect. The clean population reaches further up than the real one begins, so
no threshold separates the two, and a clipped chip corner is untidy rather
than unreadable. (A second example stood here, ``ship_source_level`` at 384 px
against 5. This branch then moved that formula box to the top right, where
nothing is drawn over it, and its chip now loses nothing. The measurement that
carried the argument is gone even though the argument holds, which is the
hazard of a worked example naming a figure the same branch is free to move.)

So the letters are found by drawing them twice, once in each of
:data:`_INKS`, and differencing the two renders. Everything else in the figure
is identical between them, the chip included, so what is left is the letters
and only the letters. It is also the only mask that survives a letter drawn
over a curve of the label's own colour: a mask read against a render of the
figure *without* the label scores that pixel as unpainted, and the character
the reader has lost goes uncounted.

One correction to the recipe, measured rather than assumed. A zorder orders
artists *inside* one axes, and the figure draws each axes whole, so a label on
the earlier of two axes cannot be lifted over the later one's curve at all: a
synthetic label under an opaque line on a ``twinx`` sibling renders identically
with its zorder at a million, and the bare recipe scores it zero. Where such
an artist reaches the box it is taken away for the pair of renders that
locates the label's ink, and the shipped render still decides what survives.
With no such artist -- every figure with one axes -- nothing is taken away and
this is the recipe unchanged. It is what finds ``transfer_stiffness``, whose
``twinx`` label is painted over by the transmissibility curve on the host that
the figure deliberately puts on top.

One thing it still cannot see, and it is worth writing down. An artist drawn
over a label with ``alpha`` below 1 lets the letters tint the result, so the
pixel does differ from the label-hidden render and counts as surviving: the
same synthetic control at ``alpha=0.9`` scores zero where at 1.0 it scores
346. Where that artist is a *stroke*, the other measure has it, which is why both
are kept. ``sii_masking_chain`` was the example: 345 px under behind and
silent here. It is silent under both now, because this branch handed the
annotation to the twin axes and chipped it, so ``_candidates`` no longer
nominates it at all. Where it is a translucent *fill*, neither does: the fill is a backing to
the behind measure by design, so a wash laid over a label scores 0 and 0
(measured, ``axhspan(alpha=0.55)`` over a label at a higher zorder). The corpus
draws its washes under the labels rather than over them, so it has no instance;
a wash that did move on top would need the third thing neither measure is, a
contrast measure.

Recording is off unless :data:`AUDIT_ENV` names a directory, so a plain
``python scripts/generate_graphs.py`` is untouched: the measurement costs
renders, and nobody generating one figure by hand asked for it. When it is
set, every process that draws figures -- the parent, each spawned pool worker
-- writes its own ``<pid>.json`` fragment there at exit, and
``scripts/check_figure_annotations.py`` merges them. Fragments accumulate:
whoever sets the variable empties the directory first, which is what ``make
graphs`` does.

The record is keyed by the *asset* name, language suffix included
(``foo`` and ``foo_es``), because the two languages are two different
drawings: Spanish prose is longer than English, so a label that clears a curve
in one grows into it in the other. Both light passes are measured; the dark
pass of each is the same geometry in other colours, and it is skipped because
the measurement costs renders and doubling them buys three pixels. Not zero
pixels: forcing the measurement on the dark pass of ten drawings recorded 44
labels, of which 25 came out differently, every one of them by 1 to 3 px
(``airport_sor`` "-12" 42 light against 39 dark). That is anti-aliasing against a different page, not a
different drawing: the behind counts all fall on the dark page and only the
covered ones can rise, by 2 px at the most. So the skip can hide a defect
sitting within three pixels of a threshold, and nothing further from one.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import pathlib
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.collections import Collection
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    #: What counts as ink: the four artist families that can put a stroke on
    #: a panel. They agree on the accessors the suppression below needs
    #: (``linewidth``, ``hatch``, ``edgecolor``), which is what lets one
    #: context manager handle all of them.
    _Ink = Line2D | Patch | Collection

#: Names the directory the fragments are written to. Unset means "do not
#: record", which is the default everywhere except the generation runs the
#: gate is wired into.
AUDIT_ENV = "PHONOMETRY_FIGURE_ANNOTATION_AUDIT"

#: Where ``make graphs`` points :data:`AUDIT_ENV`, and where the checker looks
#: when it is not told otherwise. Under ``build/``, which is gitignored.
DEFAULT_DIR = "build/figure-annotations"

#: The dpi every still figure is drawn at (``figure.dpi`` in the theme's
#: rcParams). A struck-pixel count scales with the square of it, so the number
#: the checker compares means nothing without it; it is recorded per figure
#: and the checker refuses a recording taken at any other resolution.
REFERENCE_DPI = 150.0

#: A label the reader sees whole, drawn straight over a curve with nothing
#: between the two. Answered by a chip.
BEHIND = "behind"

#: A label whose ink does not reach the reader at all, because something drawn
#: later paints over it. Answered by a zorder, not by a chip.
COVERED = "covered"

#: Where a label is put to render it above everything else in the figure, for
#: the covered measure. Far above any zorder a figure sets by hand.
_TOP_ZORDER = 1.0e6

#: The two inks a label's letters are re-drawn in to find out where they are.
#: A pixel the letters cover is a blend of the ink with whatever stands behind
#: it, so the two renders differ there whatever that is, and a pixel they do
#: not cover is the same in both. Differencing the pair therefore returns the
#: letters exactly -- including a letter drawn over a curve of the label's own
#: colour, which is invisible to a mask taken against a render of the figure
#: without the label. Black and white and not, say, the label's own colour
#: against one other: the further apart the two inks are, the fainter the
#: anti-aliased edge pixel the difference can still resolve, and these are as
#: far apart as a screen goes.
_INKS = ("#000000", "#ffffff")

#: The ink a label's letters are given when a render has to keep the
#: annotation alive for its arrow and lose the words. ``"none"`` is
#: ``(0, 0, 0, 0)``, and ``Text.draw`` hands the colour to the graphics context
#: as RGBA, so the glyphs are laid down and change not one pixel.
_NO_INK = "none"

#: Slack, in pixels, on a bounding box used as a prefilter. A box is the
#: coarse answer already; a degenerate one (a horizontal ``LineCollection`` has
#: zero height) would otherwise miss the very stroke it stands for.
_PREFILTER_PAD = 2.0

# key -> the labels of that drawing that cannot be read, worst first. A key
# with an empty list is a figure that was drawn and came out clean, which is
# what lets the checker tell "no longer struck" from "not generated in this
# run".
_FOUND: dict[str, list[dict[str, Any]]] = {}

_REGISTERED = False


class _Candidate(NamedTuple):
    """One un-chipped label, its glyph box, and the artists whose geometry reaches it."""

    text: Text
    box: Bbox
    artists: list[_Ink]


def audit_dir() -> str | None:
    """The directory fragments are written to, or ``None`` when recording is off."""
    return os.environ.get(AUDIT_ENV) or None


def audit(fig: Figure, key: str) -> None:
    """Measure *fig* and record what cannot be read on it, under the name *key*.

    Called once per language as the figure is saved -- on the light pass, since
    the dark one is the same drawing in other colours -- so a figure that comes
    out clean still leaves a record that it was generated at all. *key* is the
    asset name with the language suffix on it, because the two languages set
    different strings and a label that clears a curve in one may not in the
    other.
    """
    if audit_dir() is None:
        return
    _FOUND.setdefault(key, [])
    if not _REGISTERED:
        _register()
    labels = list(_labels(fig))
    if not labels:
        return
    boxes = _glyph_boxes(fig)
    hits = _behind(fig, boxes) + _covered(fig, labels, boxes)
    if hits:
        _FOUND[key] = sorted(hits, key=lambda hit: -hit["pixels"])


def _panels(fig: Figure) -> list[Axes]:
    """Every axes of *fig*, including the ones the figure does not list.

    ``Axes.inset_axes`` files its result under the host's ``child_axes`` and
    not under ``fig.axes``, so a figure with an inset on it holds an axes the
    figure cannot name: its curves would not be ink, a label on it would never
    be enumerated, and a host label the inset panel is laid over would not be
    seen. Five drawings in the corpus carry an inset and none of them is any of
    those three today, which is exactly the kind of thing that stops being true
    without anybody noticing.
    """
    inset = [child for ax in fig.axes for child in ax.child_axes]
    return [*fig.axes, *cast("list[Axes]", inset)]


def _figure_artists(fig: Figure) -> list[Artist]:
    """The visible artists the figure draws itself, outside any axes.

    ``figtext``, ``fig.legend``, a rule or a rectangle added to the figure: the
    figure draws them after its axes, so each can paint over any label in it,
    and none is reachable by walking the axes. No zorder on a label can lift it
    clear of one, for the same reason a zorder cannot lift it over a ``twinx``
    sibling, so they join the artists the covered measure has to take away
    rather than raise the label above.
    """
    here = [*fig.patches, *fig.lines, *fig.artists, *fig.texts, *fig.legends]
    return [artist for artist in here if artist.get_visible()]


def _labels(fig: Figure) -> Iterator[tuple[Axes, Text]]:
    """Every visible, non-empty ``ax.texts`` entry of every axes.

    Figure-level text (``suptitle``, ``figtext``) is left out: it is not drawn
    over the data. It can still be drawn over *another* label, which is why
    :func:`_figure_artists` puts it on the other side of the covered measure.
    """
    for ax in _panels(fig):
        for text in ax.texts:
            if text.get_visible() and text.get_text().strip():
                yield ax, text


def _unchipped(fig: Figure) -> Iterator[tuple[Axes, Text]]:
    """The labels that carry no chip, which is what the behind measure asks about.

    A ``Text`` with a ``bbox_patch`` already has its backing, whatever shape it
    is, so a curve *behind* it is not a defect. Whether that backing survives
    to the page is the covered measure's question, and that one does look at
    chipped labels.
    """
    for ax, text in _labels(fig):
        if text.get_bbox_patch() is None:
            yield ax, text


def _glyph_boxes(fig: Figure) -> dict[int, Bbox]:
    """Where every label of *fig* puts its letters, captured as it draws them.

    Asked during the draw rather than after it, because for one kind of label
    the two answers are not the same. ``Text3D.draw`` projects the label's
    three data coordinates into display coordinates, sets ``_x`` and ``_y`` to
    the result for the duration of the draw, and puts them back on the way
    out. Asked afterwards, ``Text.get_window_extent`` therefore runs the *data*
    coordinates through ``transData``: on ``microphone_positions_hemisphere``,
    a 1350 px canvas, it answered ``x = 12264.9`` for "Reflecting plane" and
    ``y = -9588.2`` for "1". Every box then missed ``ax.bbox``, every label of
    both 3-D plates was dropped, and the record said the four drawings were
    measured and clean when nothing about them had been measured at all.

    Wrapping the draw is what both kinds of label have in common: for an
    ordinary ``Text`` the position is the same at that moment as at any other,
    so this changes nothing about what it answers, and the draw the audit
    needs anyway is the one it is taken from.

    ``Text.get_window_extent`` is called unbound on purpose: ``Annotation``
    overrides it to include the arrow, and an arrow reaching down to touch the
    curve is the *correct* pattern, a label set aside with a pointer, not the
    defect. On ``filter_leakage_floor`` the override reported 50 px of glyphs
    as a 185 px box and flagged the figure for being drawn well. It is also
    the letters and not the chip: a ``Text``'s own extent has never included
    its ``bbox_patch``.
    """
    boxes: dict[int, Bbox] = {}
    drawn = Text.draw

    def draw(text: Text, renderer: RendererBase) -> None:
        if text.get_visible():
            with contextlib.suppress(Exception):
                boxes[id(text)] = Text.get_window_extent(text, renderer)
        drawn(text, renderer)

    Text.draw = draw  # type: ignore[method-assign,assignment]
    try:
        fig.canvas.draw()
    finally:
        Text.draw = drawn  # type: ignore[method-assign]
    return boxes


def _glyph_box(ax: Axes, text: Text, boxes: dict[int, Bbox]) -> Bbox | None:
    """*text*'s captured box, clipped to the axes it lives in.

    Every stage reads this -- the prefilters test artists against it, and both
    measurements count glyph pixels only inside it -- so a label the draw never
    reached is left out of both by one decision here.

    The clip is an intersection rather than a drop of the out-of-axes part,
    because a rule (``axvline``) is two points and dropping one of them leaves
    no segment to test at all.
    """
    box = boxes.get(id(text))
    return None if box is None else Bbox.intersection(box, ax.bbox)


def _line_points(line: Line2D) -> np.ndarray:
    """*line*'s vertices in display coordinates, non-finite ones dropped.

    Through the line's own transform, not the axes' ``transData``: a rule from
    ``axvline`` / ``axhline`` carries a *blended* transform -- data in one
    direction, axes fraction in the other -- and its stored coordinates are
    literally 0.0 and 1.0. Pushed through ``transData`` every rule in the
    corpus lands at data 0 and 1, which on a log axis is nowhere near the line
    the reader sees.
    """
    data = np.asarray(line.get_xydata(), dtype=float)
    if data.size == 0:
        return np.empty((0, 2))
    points = np.asarray(line.get_transform().transform(data), dtype=float)
    return points[np.isfinite(points).all(axis=1)]


def _segment_enters(box: Bbox, start: np.ndarray, end: np.ndarray) -> bool:
    """Whether the segment ``start -> end`` meets *box* (Liang-Barsky clip).

    Exact, where sampling the segment at a fixed number of points is not: a
    segment long compared with a glyph box steps straight over it between two
    samples, and a one-line annotation is a thin box.
    """
    delta = end - start
    low, high = 0.0, 1.0
    for axis, (lo, hi) in enumerate(((box.x0, box.x1), (box.y0, box.y1))):
        if delta[axis] == 0.0:
            if not (lo <= start[axis] <= hi):
                return False
            continue
        first = (lo - start[axis]) / delta[axis]
        second = (hi - start[axis]) / delta[axis]
        near, far = min(first, second), max(first, second)
        low, high = max(low, near), min(high, far)
        if low > high:
            return False
    return True


def _reaches(box: Bbox, line: Line2D) -> bool:
    """Whether any ink *line* draws could land inside *box*.

    A marker-only series (``linestyle="none"``) draws no segments, so only its
    marker points are ink; interpolating between them flags a label sitting in
    the gap between two markers.
    """
    points = _line_points(line)
    if len(points) == 0:
        return False
    inside = (
        (points[:, 0] >= box.x0)
        & (points[:, 0] <= box.x1)
        & (points[:, 1] >= box.y0)
        & (points[:, 1] <= box.y1)
    )
    if inside.any():
        return True
    if line.get_linestyle() in ("none", "None", " ", ""):
        return False
    return any(
        _segment_enters(box, points[i], points[i + 1]) for i in range(len(points) - 1)
    )


def _finite(box: Bbox) -> bool:
    """Whether *box* is a real rectangle rather than the empty one."""
    return bool(np.isfinite([box.x0, box.y0, box.x1, box.y1]).all())


def _display_extent(artist: Artist) -> Bbox | None:
    """*artist*'s bounding box in display pixels, or ``None`` if it has none.

    ``Collection.get_window_extent`` answers with the empty box for anything
    put on an axes with ``add_collection``: it asks for the data limits under
    an identity transform, and the collection's transform is ``transData``, so
    a ``LineCollection`` reads as nowhere at all. Its data limits *under the
    axes transform* are right, and pushing those back through the same
    transform gives the pixels. Patches answer ``get_window_extent`` properly
    and fall through to it.
    """
    axes = getattr(artist, "axes", None)
    box: Bbox | None = None
    if axes is not None and hasattr(artist, "get_datalim"):
        with contextlib.suppress(Exception):
            box = artist.get_datalim(axes.transData).transformed(axes.transData)
    if box is None or not _finite(box):
        box = None
        with contextlib.suppress(Exception):
            box = artist.get_window_extent()
    if box is None or not _finite(box):
        return None
    return box.padded(_PREFILTER_PAD)


def _overlaps(one: Bbox, other: Bbox) -> bool:
    """Whether two display boxes share any area (touching counts)."""
    return bool(
        one.x0 <= other.x1
        and other.x0 <= one.x1
        and one.y0 <= other.y1
        and other.y0 <= one.y1
    )


class _Extents:
    """Display bounding boxes, worked out once per artist.

    ``Collection.get_datalim`` walks every path the collection holds, and a
    contour or a reflectogram holds thousands; the prefilter asks for the same
    box once per label, so the answer is remembered for the pass.
    """

    def __init__(self) -> None:
        self._known: dict[int, Bbox | None] = {}

    def of(self, artist: Artist) -> Bbox | None:
        if id(artist) not in self._known:
            self._known[id(artist)] = _display_extent(artist)
        return self._known[id(artist)]

    def touches(self, box: Bbox, artist: Artist) -> bool:
        """Whether *artist*'s extent reaches *box*. An artist with none is kept."""
        extent = self.of(artist)
        return extent is None or _overlaps(box, extent)

    def might_stroke(self, box: Bbox, artist: _Ink) -> bool:
        """Whether *artist* could put a stroke inside *box*.

        Exact for anything that offers its vertices, which is every
        ``Line2D``: a bounding box would call a label inside the arc of a
        curve struck by it. For the rest the box is all there is, and that is
        fine here -- erring towards measuring costs one render and the pixel
        count settles it, while erring the other way is a defect that ships.
        """
        if isinstance(artist, Line2D):
            return _reaches(box, artist)
        return self.touches(box, artist)


def _paints_a_stroke(artist: _Ink) -> bool:
    """Whether *artist* draws a line, as opposed to filling an area.

    A patch with no outline paints an area and nothing else, so it can neither
    strike a letter nor make a label worth measuring, and neither can one whose
    outline is the colour of its own fill: ``axhspan(color=...)`` and
    ``fill_between(color=...)`` set the edge and the face together, so the
    "stroke" is a one-pixel fringe of the same shade around the band, a step
    from the fill to the page and not a line across anything. Measured, that
    fringe scored the legible ``mobility_result_lines`` note -- which sits on
    the ±90 deg band, its top edge running through the first row of letters --
    at 282 px, and saying so here also keeps the prefilter from nominating
    every label of every figure that has a shaded region on it. Anything that
    cannot answer is kept, because a wrong "no" is a defect that ships.

    A ``Line2D`` and an annotation's arrow are answered before any of that,
    because for them the question does not arise: both are a mark and nothing
    else. The arrow has to be said out loud, because ``annotate`` with a plain
    ``color`` in its ``arrowprops`` sets the arrow's face and edge to the same
    shade, which is the band test above and would drop it.

    The rule has one shape it is wrong about, and the corpus does not draw it:
    a band thinner than a letter. ``axhspan(0.500, 0.504)`` renders as a solid
    rule struck through every character and scores 0, where the same rule drawn
    as an ``axhline`` scores 549 px on the same synthetic. It stays anyway,
    because the shape it is right about is the one the corpus does draw, and
    the numbers above are what that costs.
    """
    if isinstance(artist, Line2D | FancyArrowPatch):
        return True
    try:
        if artist.get_hatch():
            return True
        widths = np.atleast_1d(artist.get_linewidth()).astype(float)
        if widths.size == 0 or not (widths > 0).any():
            return False
        edges = np.atleast_2d(artist.get_edgecolor())
        if edges.size == 0 or not (edges[:, 3] > 0).any():
            return False
        faces = np.atleast_2d(artist.get_facecolor())
        return faces.size == 0 or not bool(np.allclose(edges, faces))
    except Exception:  # noqa: BLE001 - an artist that cannot answer is measured
        return True


def _arrows(ax: Axes) -> list[FancyArrowPatch]:
    """The pointer of every annotation on *ax* that carries one.

    An arrow is not in ``ax.patches``: ``Annotation`` holds it in
    ``arrow_patch`` and draws it itself, so walking the axes' children finds
    the letters and misses the line they are attached to. Measured on two
    figures identical but for how one stroke is drawn, the victim label scored
    287 px when it was a ``Line2D`` and 0 px when it was an
    ``annotate(arrowstyle="-")`` across the same pixels.

    A hidden annotation draws no arrow (``Annotation.draw`` returns before it
    reaches ``arrow_patch``), which is why the owner's visibility is the test
    and not the arrow's.
    """
    return [
        text.arrow_patch
        for text in ax.texts
        if isinstance(text, Annotation)
        and text.arrow_patch is not None
        and text.get_visible()
    ]


def _pointers(fig: Figure) -> dict[int, Text]:
    """Which annotation each arrow of *fig* belongs to, for the report.

    An arrow has no back-pointer to its annotation and no label of its own, so
    "unnamed FancyArrowPatch" is all :func:`_artist_name` could say about it,
    and a figure with four arrows on it would say that four times.
    """
    return {
        id(text.arrow_patch): text
        for ax in fig.axes
        for text in ax.texts
        if isinstance(text, Annotation) and text.arrow_patch is not None
    }


def _ink(fig: Figure) -> list[_Ink]:
    """Every visible artist of every axes that can put a stroke on the panel.

    Every axes, because a ``twinx`` sibling draws over the host and its curve
    is what runs through the host's annotation. ``Line2D``, collections,
    patches and the arrows of :func:`_arrows`, because a stem drawn as a
    ``LineCollection``, a contour, a patch outline and an annotation's pointer
    are strokes exactly like a plotted line.

    What is deliberately *not* here is the furniture the axes draws around the
    data: the gridlines (which live on the ``Axis``, not in ``ax.get_lines()``),
    the spines, and the edge of a chip. They are excluded on the measurements,
    not by oversight. A major gridline reaches an un-chipped label on
    something over four hundred of the 894 drawings and something over nine
    hundred of its roughly 1390 un-chipped labels: two independent censuses
    of the same corpus gave 371 and 382 drawings, and neither could say where
    they parted, so the order is what this argument rests on and not the
    figure. Counting gridlines would move every threshold in the checker,
    and what they paint is an order of magnitude below the gate: putting the
    gridlines and the spines into this list and re-measuring the one corpus
    note a spine crosses, ``sweep_distortion_separation``'s "causal part: what
    impulse_response() returns", scores 9 px of spine and 8 px of gridline
    against a gate of 45. A chip edge is one pixel of ``COLOR_GRID``, and a
    chip over another label is what the covered measure is for, since the chip
    is opaque and takes the letters with it.
    """
    found: list[_Ink] = []
    for ax in _panels(fig):
        here: list[_Ink] = [
            *ax.get_lines(),
            *ax.collections,
            *ax.patches,
            *_arrows(ax),
        ]
        found.extend(a for a in here if a.get_visible() and _paints_a_stroke(a))
    over: list[_Ink] = [*fig.patches, *fig.lines]
    found.extend(a for a in over if a.get_visible() and _paints_a_stroke(a))
    return found


def _candidates(fig: Figure, boxes: dict[int, Bbox]) -> list[_Candidate]:
    """The un-chipped labels a stroke's geometry reaches, and which artists those are.

    The prefilter selects *labels*, not artists: a figure is worth rendering
    again because one of its labels might be struck, and which artist does the
    striking is settled afterwards by suppressing one at a time. On
    ``room_noise_criteria`` the two answers differ -- the label is selected
    because an NC curve ends on the edge of its box, and the ink on its digits
    comes from a different curve entirely.

    A label's own pointer is the one stroke that cannot select it. An arrow
    leaves the edge of the box it is anchored to, so its bounding box meets
    that box on every annotation in the corpus, and the measurement excludes it
    from that label's count by design; nominating on it would buy a render per
    arrow and a count of zero, and 266 un-chipped labels across the corpus's
    894 drawings carry one.
    """
    ink = _ink(fig)
    extents = _Extents()
    found: list[_Candidate] = []
    for ax, text in _unchipped(fig):
        box = _glyph_box(ax, text, boxes)
        if box is None:
            continue
        own = text.arrow_patch if isinstance(text, Annotation) else None
        reaching = [
            artist
            for artist in ink
            if artist is not own and extents.might_stroke(box, artist)
        ]
        if reaching:
            found.append(_Candidate(text, box, reaching))
    return found


@contextlib.contextmanager
def _hidden(artists: Sequence[Artist]) -> Iterator[None]:
    """Draw without *artists*, then put every one of them back as it was.

    The figure is about to be written to disk, so the measurement must leave
    no trace in it.
    """
    was = [artist.get_visible() for artist in artists]
    for artist in artists:
        artist.set_visible(False)
    try:
        yield
    finally:
        for artist, visible in zip(artists, was, strict=True):
            artist.set_visible(visible)


@contextlib.contextmanager
def _unstroked(artists: Sequence[_Ink]) -> Iterator[None]:
    """Draw *artists* without their strokes, keeping whatever they fill.

    A ``Line2D`` is nothing but stroke and markers, so it goes away entirely,
    and so does an arrow: it is a mark from one place to another with no
    interior anybody reads a letter on, and taking its line width to zero
    would leave a filled arrowhead behind.
    A patch or a collection loses its line width and its hatching and keeps
    its face, which is the distinction the whole behind measure rests on: a
    filled region under a label is a backing and the letters on it read
    perfectly, while its outline crossing them does not.
    """
    restore: list[tuple[_Ink, Any, Any]] = []
    for artist in artists:
        if isinstance(artist, Line2D | FancyArrowPatch):
            restore.append((artist, artist.get_visible(), None))
            artist.set_visible(False)
        else:
            restore.append((artist, artist.get_linewidth(), artist.get_hatch()))
            artist.set_linewidth(0)
            artist.set_hatch(None)  # type: ignore[arg-type]  # None clears it
    try:
        yield
    finally:
        for artist, first, second in restore:
            if isinstance(artist, Line2D | FancyArrowPatch):
                artist.set_visible(first)
            else:
                artist.set_linewidth(first)
                artist.set_hatch(second)


@contextlib.contextmanager
def _raised(artists: Sequence[Artist]) -> Iterator[None]:
    """Draw *artists* above everything else, then put their zorder back."""
    was = [artist.get_zorder() for artist in artists]
    for artist in artists:
        artist.set_zorder(_TOP_ZORDER)
    try:
        yield
    finally:
        for artist, zorder in zip(artists, was, strict=True):
            artist.set_zorder(zorder)


@contextlib.contextmanager
def _inked(text: Text, colour: str) -> Iterator[None]:
    """Draw *text*'s letters in *colour*, then give them their own back.

    Only the letters: the chip is the ``bbox_patch``, which keeps its own two
    colours, so a pair of renders taken this way differs in the letters and in
    nothing else.
    """
    was = text.get_color()
    text.set_color(colour)
    try:
        yield
    finally:
        text.set_color(was)


@contextlib.contextmanager
def _lettering_off(texts: Sequence[Text]) -> Iterator[None]:
    """Draw without the words of *texts*, keeping any arrow they carry.

    Taking the labels out of a render is how the behind measure builds the
    ground it compares against, and ``set_visible(False)`` on an
    ``Annotation`` takes its arrow out with them:
    :meth:`Annotation.draw` returns on ``get_visible()`` before it reaches
    ``arrow_patch``. The arrow has to stay in that render, because an arrow
    across *somebody else's* label is a stroke behind letters exactly like a
    curve, and it can only be counted where every label is out of the way.

    So an annotation with an arrow keeps being drawn and loses only what makes
    it a label: its letters go down in a fully transparent ink, which changes
    no pixel, and its chip is hidden. Everything else is hidden outright, as
    before.
    """
    with contextlib.ExitStack() as stack:
        for text in texts:
            arrow = text.arrow_patch if isinstance(text, Annotation) else None
            if arrow is None:
                stack.enter_context(_hidden([text]))
                continue
            stack.enter_context(_inked(text, _NO_INK))
            chip = text.get_bbox_patch()
            if chip is not None:
                stack.enter_context(_hidden([chip]))
        yield


def _render(fig: Figure) -> np.ndarray:
    """One RGBA snapshot of *fig* as it stands."""
    canvas = fig.canvas
    buffer = getattr(canvas, "buffer_rgba", None)
    if buffer is None:
        msg = (
            f"the annotation audit needs a raster canvas to count pixels on, and "
            f"{type(canvas).__name__} has no buffer_rgba; generate with "
            f"MPLBACKEND=Agg, or unset {AUDIT_ENV} to skip the measurement"
        )
        raise RuntimeError(msg)
    canvas.draw()
    return np.asarray(buffer()).copy()


def _painted(image: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """The pixels *image* differs from *ground* in: the ink the extra artists put down."""
    return np.asarray((image != ground).any(axis=2))


def _inside(shape: tuple[int, ...], box: Bbox) -> np.ndarray:
    """A mask of the pixels of a render that fall inside *box*.

    The row axis of the buffer runs down from the top and display coordinates
    run up from the bottom, so the rows are counted back from the height.
    """
    height, width = shape[0], shape[1]
    mask = np.zeros((height, width), dtype=bool)
    left, right = max(0, int(box.x0)), min(width, int(np.ceil(box.x1)) + 1)
    top, bottom = (
        max(0, height - int(np.ceil(box.y1)) - 1),
        min(height, height - int(box.y0) + 1),
    )
    mask[top:bottom, left:right] = True
    return mask


# --------------------------------------------------------------------------
# The behind measure: glyph pixels with a stroke painted under them.


def _behind(fig: Figure, boxes: dict[int, Bbox]) -> list[dict[str, Any]]:
    """Every un-chipped label of *fig* that a stroke is drawn through."""
    candidates = _candidates(fig, boxes)
    return _measure(fig, candidates) if candidates else []


def _measure(fig: Figure, candidates: list[_Candidate]) -> list[dict[str, Any]]:
    """Count, for each candidate label, the pixels of its glyphs a stroke also paints.

    Two renders answer it for the whole figure -- the ground, which is the
    figure with the labels hidden and every stroke suppressed, and the same
    thing with the strokes back -- plus one per candidate for its glyphs and
    one per artist to say which artist it was. Everything else the figure
    draws, its fills included, is present in all of them and cancels.
    """
    labels = [candidate.text for candidate in candidates]
    ink = _ink(fig)
    with _lettering_off(labels), _unstroked(ink):
        ground = _render(fig)
    with _lettering_off(labels):
        shipped = _render(fig)
    strokes = _painted(shipped, ground)
    hits = [
        hit
        for candidate in candidates
        if (hit := _strike(fig, candidate, labels, ink, ground, strokes)) is not None
    ]
    _attribute(fig, hits, labels, shipped, _pointers(fig))
    return hits


def _strike(
    fig: Figure,
    candidate: _Candidate,
    labels: list[Text],
    ink: list[_Ink],
    ground: np.ndarray,
    strokes: np.ndarray,
) -> dict[str, Any] | None:
    """Render *candidate* alone on *ground* and count its glyph pixels under *strokes*.

    A label set aside with an arrow that reaches down to touch the curve is
    the *correct* pattern, and two things keep it from being scored as the
    defect. The candidate's own arrow is ink, so ``_unstroked`` suppresses it
    in this render exactly as it does in the ground, and it lands in neither;
    and the render is masked to the glyph box, which comes from
    ``Text.get_window_extent`` and is the letters only. Somebody *else's*
    arrow is a different thing entirely: it stays in the shipped render, so it
    is in *strokes*, and it counts.
    """
    others = [text for text in labels if text is not candidate.text]
    with _lettering_off(others), _unstroked(ink):
        glyphs = _painted(_render(fig), ground) & _inside(ground.shape, candidate.box)
    pixels = int((glyphs & strokes).sum())
    if pixels == 0:
        return None
    return {
        "kind": BEHIND,
        "text": candidate.text.get_text(),
        "pixels": pixels,
        "dpi": float(fig.dpi),
        "struck_by": [],
        "_glyphs": glyphs,
        "_artists": candidate.artists,
    }


def _attribute(
    fig: Figure,
    hits: list[dict[str, Any]],
    labels: list[Text],
    shipped: np.ndarray,
    pointers: dict[int, Text],
) -> None:
    """Name the artist each hit's ink belongs to, by suppressing one at a time.

    Geometry cannot say it: the curve that reaches a label's box is often not
    the curve that paints its letters. Taking one artist's stroke away and
    differencing against the figure with every stroke in place leaves exactly
    the pixels where that stroke is the topmost thing the reader sees, so
    where two curves cross on a letter the pixel is credited to the one on
    top -- which is the one the reader is actually reading through.

    The renders here are taken the same way *shipped* was, letters off rather
    than labels hidden, so the one artist under test is the only difference
    between the two.
    """
    alone: dict[int, np.ndarray] = {}
    for hit in hits:
        for artist in hit.pop("_artists"):
            if id(artist) not in alone:
                with _lettering_off(labels), _unstroked([artist]):
                    alone[id(artist)] = _painted(shipped, _render(fig))
            pixels = int((hit["_glyphs"] & alone[id(artist)]).sum())
            if pixels:
                hit["struck_by"].append([_artist_name(artist, pointers), pixels])
        hit.pop("_glyphs")
        hit["struck_by"].sort(key=lambda named: -named[1])


def _artist_name(artist: Artist, pointers: dict[int, Text] | None = None) -> str:
    """How to point at *artist* in a report: its legend label, else its look.

    A ``Text`` is named by what it says, because "unnamed Text" tells whoever
    reads the report nothing and one label drawn over another is a case the
    covered measure finds. No corpus figure is that case today: the one that
    was, ``decay_range_bias``, had its corner box moved off the rotated rule
    label in this change, so the naming is kept for the class rather than for
    an instance. An arrow is named by the annotation it leaves, for the same
    reason and because it has no label of its own to fall back on; *pointers*
    is :func:`_pointers` of the figure it belongs to.
    """
    owner = (pointers or {}).get(id(artist))
    if owner is not None:
        first, _, _ = owner.get_text().partition("\n")
        return f"the pointer from {first[:40]!r}"
    if isinstance(artist, Text):
        first, _, _ = artist.get_text().partition("\n")
        return f"the label {first[:40]!r}"
    label = str(artist.get_label())
    if label and not label.startswith("_"):
        return label
    if isinstance(artist, Line2D):
        return f"unnamed {artist.get_color()} {artist.get_linestyle()!r} line"
    return f"unnamed {type(artist).__name__}"


# --------------------------------------------------------------------------
# The covered measure: label ink that does not reach the reader.


def _axes_order(fig: Figure) -> dict[int, int]:
    """Which axes of *fig* matplotlib draws first, as a rank per axes.

    An axes is one artist of the figure, drawn whole: everything on the second
    of two axes lands on top of everything on the first, whatever the two sets
    of artist zorders say. The figure sorts them by ``Axes.get_zorder()``, and
    Python's sort is stable, so ties keep the order they were created in --
    which is what puts a ``twinx`` sibling above its host. An inset is ranked
    the same way: ``inset_axes`` gives it a zorder above the default, so it
    sorts after the host it is drawn inside, which is where it lands.
    """
    ranked = sorted(_panels(fig), key=lambda axes: axes.get_zorder())
    return {id(axes): rank for rank, axes in enumerate(ranked)}


def _drawn_after(
    fig: Figure, ax: Axes, text: Text, order: dict[int, int]
) -> tuple[list[Artist], list[Artist]]:
    """What can be painted over *text*, split by whether a zorder can lift it clear.

    Two orderings decide it. Within one axes the artists are sorted by zorder,
    ties going to whichever was added last, and raising the label's zorder
    puts it above all of them. Across axes nothing of the sort is possible:
    ``Text.set_zorder`` orders artists inside their own axes, so a label on
    the host of a ``twinx`` pair cannot be lifted over the sibling's curve at
    all -- measured, a label under an opaque twin-axes line still renders
    identically with its zorder at a million. Those artists are returned
    separately so the measurement can take them away instead.

    What the figure draws itself is in the second list for the same reason and
    without asking its zorder: it is not in the label's axes, so no zorder the
    label carries can reach it.

    Ties are kept rather than resolved, on the same principle as the other
    prefilter: a render settles it.
    """
    here = order[id(ax)]
    same: list[Artist] = []
    later: list[Artist] = list(_figure_artists(fig))
    for axes in _panels(fig):
        after = order[id(axes)] > here
        drawn = [
            artist
            for artist in _drawables(axes)
            if artist is not text
            and (after or artist.get_zorder() >= text.get_zorder())
        ]
        (later if after else same).extend(drawn)
    return same, later


def _drawables(ax: Axes) -> list[Artist]:
    """The visible children of *ax* that put ink on the page.

    ``ax.patch``, the panel's own background, is one of them and is in none of
    the other lists. It is what an opaque inset lays over the host label
    underneath it, and it is harmless everywhere else: for the label's own axes
    it sits at zorder 1, below the caller's zorder test, and for two panels
    side by side it reaches no label but its own. A ``twinx`` sibling's patch
    is made invisible by ``twinx`` itself, so it never enters here.
    """
    legend = ax.get_legend()
    return [
        artist
        for artist in (
            ax.patch,
            *ax.get_lines(),
            *ax.collections,
            *ax.patches,
            *ax.images,
            *ax.texts,
            *([legend] if legend is not None else []),
        )
        if artist.get_visible()
    ]


def _covered(
    fig: Figure, labels: list[tuple[Axes, Text]], boxes: dict[int, Bbox]
) -> list[dict[str, Any]]:
    """Every label of *fig*, chipped or not, whose ink does not reach the page.

    The prefilter is the draw order: a label nothing is drawn after cannot be
    covered, and asking costs no render at all.
    """
    extents = _Extents()
    order = _axes_order(fig)
    targets = []
    for ax, text in labels:
        box = _glyph_box(ax, text, boxes)
        if box is None:
            continue
        same, later = _drawn_after(fig, ax, text, order)
        over = [artist for artist in same if extents.touches(box, artist)]
        above = [artist for artist in later if extents.touches(box, artist)]
        if over or above:
            targets.append((text, box, over, above))
    if not targets:
        return []
    full = _render(fig)
    return [
        hit
        for text, box, over, above in targets
        if (hit := _lost(fig, text, box, over, above, full)) is not None
    ]


def _lost(
    fig: Figure,
    text: Text,
    box: Bbox,
    over: list[Artist],
    above: list[Artist],
    full: np.ndarray,
) -> dict[str, Any] | None:
    """Count the pixels of *text*'s letters that never reach the shipped figure.

    Two pairs of renders, and each pair asks the same question of a different
    stack: where does this label's own ink land on the page? A pair is the
    label drawn in each of :data:`_INKS` with everything else held still, so
    the two differ exactly where a letter puts ink down and nowhere else.
    Taken with the label lifted to the top of the stack, that is every letter
    it would draw; taken as the figure ships, it is the letters the reader
    actually receives. The difference is what is lost.

    The obvious way to write either half -- does this pixel *change* when the
    label is taken away -- is blind wherever the label's ink and what lies
    against it are the same colour, and a black note over a black rule is not
    a rare figure. Written that way on both sides the two blind spots line up
    and a character the reader has genuinely lost is scored as never drawn;
    written that way on one side only, they no longer line up and a character
    that is perfectly visible is scored as lost. An ink difference has the
    blind spot on neither side, which is why both halves are one.

    Letters, and never the chip: the chip is drawn by ``Text.draw`` and so is
    lifted with the label, but it keeps its own colours in every render here
    and cancels. That is the difference between a number that can be gated
    and one that cannot; the module docstring has the measurements.

    *above* is what a zorder cannot lift the label over, because it lives on
    an axes drawn later. It is taken away for the pair that locates the
    letters -- otherwise the label would still be under it at any zorder, and
    the pair would find only the part that already survives -- and left in
    place for the pair that decides what survives.
    """
    raised, shipped = [], []
    for colour in _INKS:
        with _hidden(above), _raised([text]), _inked(text, colour):
            raised.append(_render(fig))
        with _inked(text, colour):
            shipped.append(_render(fig))
    region = _inside(full.shape, box)
    glyphs = _painted(*raised) & region
    struck = glyphs & ~(_painted(*shipped) & region)
    pixels = int(struck.sum())
    if pixels == 0:
        return None
    return {
        "kind": COVERED,
        "text": text.get_text(),
        "pixels": pixels,
        "dpi": float(fig.dpi),
        "struck_by": _covered_by(fig, [*over, *above], struck, full),
    }


def _covered_by(
    fig: Figure, over: list[Artist], struck: np.ndarray, full: np.ndarray
) -> list[list[Any]]:
    """Name the artists whose ink stands where the label's should have been.

    One render each, and only for a label already known to be losing pixels:
    hiding an artist and differencing against the shipped figure leaves the
    pixels where that artist is the topmost thing drawn, which is precisely
    what has to be moved (or the label lifted over).
    """
    named: list[list[Any]] = []
    for artist in over:
        with _hidden([artist]):
            pixels = int((struck & _painted(full, _render(fig))).sum())
        if pixels:
            named.append([_artist_name(artist), pixels])
    named.sort(key=lambda entry: -entry[1])
    return named


# --------------------------------------------------------------------------
# Writing the recording out, and reading it back.


def _register() -> None:
    global _REGISTERED
    atexit.register(_dump)
    _REGISTERED = True


def _dump() -> None:
    """Write this process's fragment. Registered on the first :func:`audit`."""
    directory = audit_dir()
    if directory is None or not _FOUND:
        return
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{os.getpid()}.json").write_text(
        json.dumps(_FOUND, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )


def load(directory: str) -> dict[str, list[dict[str, Any]]]:
    """Merge every fragment in *directory* into one ``key -> hits``.

    A key drawn by two fragments is the same drawing measured twice (a stale
    directory, or a run restarted): keep the worse count for each label, so
    merging can only ever under-report a fix, never invent one. The two kinds
    are kept apart, because one label can be both drawn over a curve and
    painted over by another, and the two want different fixes.
    """
    merged: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for fragment in sorted(pathlib.Path(directory).glob("*.json")):
        data = json.loads(fragment.read_text(encoding="utf-8"))
        for key, hits in data.items():
            by_label = merged.setdefault(key, {})
            for hit in hits:
                found = (hit.get("kind", BEHIND), hit["text"])
                seen = by_label.get(found)
                if seen is None or hit["pixels"] > seen["pixels"]:
                    by_label[found] = hit
    return {
        key: sorted(by_label.values(), key=lambda hit: -hit["pixels"])
        for key, by_label in merged.items()
    }
