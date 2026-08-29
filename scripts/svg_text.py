#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Type set as geometry, for SVGs that have to look the same everywhere.

GitHub renders a repository SVG inside an ``<img>``, which is a sandbox: no
stylesheet from the host page reaches into it and no font it names is
guaranteed to exist on the reader's machine. A live ``<text>`` element
therefore falls back to whatever the viewer happens to have, or to a serif, and
a lockup measured against one face reflows against another. Converting the
glyphs to outlines removes the question: the shapes travel with the file, no
font is embedded, and the same bytes draw the same picture in every renderer.
It is also the corpus rule for figures (``svg.fonttype='path'``).

Two generators need it and neither should own it: :mod:`scripts.generate_brand`
sets the wordmark and the cards, and :mod:`conformance.badges` sets the numbers
on the conformance banner. What they do not share is how a coordinate is
written - the brand works in a 256-unit viewBox at two decimals, the banner in
a ten-times-integer space - so the formatter is the caller's, passed in.

Determinism is the reason the conversion is worth isolating. The outline comes
from the committed font file's own control points, scaled by the requested
size; matplotlib is asked for the path without hinting, so nothing about the
machine's rasterizer enters, and rounding through the caller's formatter is the
only place precision is lost. Same tree, same bytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from matplotlib.path import Path as MplPath

#: An (x, y) pair in SVG coordinates, y growing downwards.
Point = tuple[float, float]


def two_decimals(value: float) -> str:
    """One coordinate at two decimals, trailing zeros gone, never ``-0``.

    The house convention for a viewBox measured in the tens or hundreds: a
    hundredth of a unit is finer than any renderer resolves and much coarser
    than any difference two machines could compute, so the bytes are stable.
    Negative zero is normalised because ``-0`` and ``0`` are the same point and
    a sign that appears only sometimes is a diff that means nothing.
    """
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def outline(text: str, font: Path, size: float, origin: Point) -> MplPath:
    """The glyph outlines of ``text``, in SVG coordinates.

    :param text: The string to set.
    :param font: A committed font file. Pinning the file is what stops the
        lockup shifting when a machine's font set changes.
    :param size: Em size, in the units of the target viewBox.
    :param origin: Baseline start, as in SVG.
    :return: The outlines as one matplotlib path.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    glyphs = TextPath((0, 0), text, size=size, prop=FontProperties(fname=str(font)))
    # matplotlib's y axis points up and SVG's points down.
    return Affine2D().scale(1, -1).translate(*origin).transform_path(glyphs)


def ink_width(text: str, font: Path, size: float) -> float:
    """How wide the set string draws, measured on the ink and not the advances.

    Optical rather than metric: the trailing sidebearing of the last glyph is
    not width the reader can see, so laying two runs out against the ink and an
    explicit gap spaces them the way a typesetter would, and a run ending in a
    narrow glyph does not look adrift from what follows it.

    :param text: The string to measure.
    :param font: The same font file it will be set in.
    :param size: The same em size it will be set at.
    :return: Width of the inked bounding box, zero for a string with no ink.
    """
    extents = outline(text, font, size, (0.0, 0.0)).get_extents()
    return float(extents.x1 - extents.x0)


def path_data(
    text: str,
    font: Path,
    size: float,
    origin: Point,
    fmt: Callable[[float], str],
) -> str:
    """The ``d`` attribute of one ``<path>`` holding ``text`` as outlines.

    :param text: The string to set.
    :param font: A committed font file.
    :param size: Em size, in the units of the target viewBox.
    :param origin: Baseline start, as in SVG.
    :param fmt: How the caller writes one coordinate. This is where precision
        is chosen, and it is the caller's choice because the two callers work
        in viewBoxes two orders of magnitude apart.
    :return: The path data, with no leading or trailing whitespace.
    """
    from matplotlib.path import Path as MplPath

    parts: list[str] = []
    for verts, code in outline(text, font, size, origin).iter_segments():
        if code == MplPath.MOVETO:
            parts.append(f"M{fmt(verts[0])} {fmt(verts[1])}")
        elif code == MplPath.LINETO:
            parts.append(f"L{fmt(verts[0])} {fmt(verts[1])}")
        elif code == MplPath.CURVE3:
            parts.append(
                f"Q{fmt(verts[0])} {fmt(verts[1])} {fmt(verts[2])} {fmt(verts[3])}"
            )
        elif code == MplPath.CURVE4:
            parts.append(
                f"C{fmt(verts[0])} {fmt(verts[1])}"
                f" {fmt(verts[2])} {fmt(verts[3])}"
                f" {fmt(verts[4])} {fmt(verts[5])}"
            )
        elif code == MplPath.CLOSEPOLY:
            parts.append("Z")
    return "".join(parts)
