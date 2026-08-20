#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Deterministic generator for the phonometry brand assets: the mark itself and
every icon derived from it.

The mark is a wavefront leaving a point source and crossing a measurement grid.
It is authored here as parametric geometry rather than stored as hand-edited
path data, so a change of proportion, palette or grid density is a one-line
edit and every derivative stays in step. Nothing is auto-traced: each ribbon is
one closed path built from a seven-point centreline carrying a width profile,
which keeps the whole file under 2 kB and every node meaningful.

Both emitters read the same geometry. ``_svg_d`` writes SVG path data;
``_mpl_path`` builds the equivalent matplotlib path for the raster icons, so
the PNGs cannot drift from the SVG.

Run via ``make brand``. Deliberately outside ``make graphs``: that target wipes
``.github/images`` before regenerating, and these are design assets rather than
computed figures.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parent.parent

# --- canvas ------------------------------------------------------------------
VB = 256.0  # viewBox is 0 0 256 256
GX = GY = 28.0  # grid top-left corner
GRID_W = 192.0  # the grid always spans 28..220, whatever the cell count
COLS = 4  # cells per side; the pitch follows from GRID_W
GRID_STROKE = 6.0  # heavy enough to survive a 16 px render, light at full size
SRC_C = (32.5, 32.5)  # the point source, in the grid's top-left cell
SRC_R = 14.0

# --- palette -----------------------------------------------------------------
INK = "#0a6f8c"  # the waves and the source: the mark's mass
GRID_INK = "#35b8d8"  # the measurement grid: secondary, lighter
INK_DARK = "#5ecbe6"  # same roles on a dark ground, where the mid teal muddies
GRID_INK_DARK = "#1c7fa0"
ICON_BG = "#064a5e"  # app icons sit on the darkest brand teal
ICON_INK = "#ffffff"
ICON_GRID = "#4ec4e0"

# --- ribbons -----------------------------------------------------------------
# Centrelines measured off the chosen design study, tracked wave by wave through
# its pixels and resampled. Stored as explicit waypoints rather than as a chord
# plus a deviation, because the shape does something a chord cannot express: the
# lowest ribbon drops 60 units in its first 14 of advance as it leaves the
# source, and only then flattens. That plunge is what reads as an undulation.
#
# Every ribbon starts at the centre of the source disc, so the union with it is
# seamless, and starts narrow so the three separate immediately instead of
# merging into a blob. Widths are perpendicular to the centreline.


@dataclass(frozen=True)
class Ribbon:
    """One wavefront: a centreline and the width carried at each of its points."""

    spine: tuple[tuple[float, float], ...]
    width: tuple[float, ...]


RIBBONS: tuple[Ribbon, ...] = (
    Ribbon(  # settles high, the shallowest of the three
        spine=(
            (32.5, 32.5),
            (46.8, 35.3),
            (71.2, 44.4),
            (111.2, 56.1),
            (163.5, 80.0),
            (217.0, 88.5),
            (232.0, 90.0),
        ),
        width=(7.0, 10.0, 12.0, 18.0, 21.0, 11.0, 0.0),
    ),
    Ribbon(  # through the middle of the grid
        spine=(
            (32.5, 32.5),
            (46.8, 61.3),
            (74.0, 74.5),
            (121.4, 101.0),
            (168.8, 139.2),
            (223.3, 155.1),
            (233.0, 157.0),
        ),
        width=(7.5, 11.0, 14.0, 23.5, 21.0, 10.0, 0.0),
    ),
    Ribbon(  # plunges out of the source, then runs out along the bottom
        spine=(
            (32.5, 32.5),
            (46.8, 92.6),
            (77.1, 109.1),
            (116.8, 154.8),
            (159.2, 208.2),
            (218.3, 234.4),
            (231.0, 236.5),
        ),
        width=(8.0, 12.0, 15.5, 21.0, 22.0, 12.0, 0.0),
    ),
)

Point = tuple[float, float]
Segment = tuple[Point, Point, Point]  # two control points and an end point


def _report(dest: Path) -> None:
    """Log a written file, repo-relative when it sits inside the repo."""
    try:
        shown = dest.resolve().relative_to(REPO)
    except ValueError:
        shown = dest
    print(f"  {shown}")


def _fmt(value: float) -> str:
    """Two decimals, trailing zeros stripped, so the path data stays readable."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


def _normals(spine: tuple[Point, ...]) -> list[Point]:
    """Unit normal at each control point, from the local centreline direction."""
    out: list[Point] = []
    for i, _ in enumerate(spine):
        prev = spine[max(i - 1, 0)]
        nxt = spine[min(i + 1, len(spine) - 1)]
        dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
        length = math.hypot(dx, dy) or 1.0
        out.append((-dy / length, dx / length))
    return out


def _catmull(points: list[Point], alpha: float = 0.5) -> list[Segment]:
    """Smooth cubic segments through the points, centripetal parameterisation.

    Centripetal rather than uniform: the waypoints are deliberately unevenly
    spaced, dense where a ribbon turns hard out of the source and sparse along
    the run, and uniform tangents put a visible kink at every transition.
    """
    pts = [points[0], *points, points[-1]]  # duplicate ends to pin the tangents
    knots = [0.0]
    for a, b in itertools.pairwise(pts):
        knots.append(knots[-1] + max(math.dist(a, b), 1e-6) ** alpha)

    out: list[Segment] = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        t0, t1, t2, t3 = knots[i - 1], knots[i], knots[i + 1], knots[i + 2]
        span = t2 - t1
        m1 = ((p2[0] - p0[0]) / (t2 - t0), (p2[1] - p0[1]) / (t2 - t0))
        m2 = ((p3[0] - p1[0]) / (t3 - t1), (p3[1] - p1[1]) / (t3 - t1))
        c1 = (p1[0] + span * m1[0] / 3, p1[1] + span * m1[1] / 3)
        c2 = (p2[0] - span * m2[0] / 3, p2[1] - span * m2[1] / 3)
        out.append((c1, c2, p2))
    return out


def ribbon_outline(ribbon: Ribbon) -> tuple[Point, list[Segment]]:
    """The closed outline of one ribbon: up the top edge, back along the bottom.

    The two halves meet without a joining segment only because the tip has zero
    width, which puts the last point of both edges on the centreline. A ribbon
    ending wide would emit a curve from the top edge's tip to the second-to-last
    point of the bottom one, and the malformed tip would show in both emitters.
    """
    if ribbon.width[-1] != 0.0:
        msg = "a ribbon has to taper to zero width at its tip"
        raise ValueError(msg)
    spine, width = ribbon.spine, ribbon.width
    nrm = _normals(spine)
    upper = [
        (p[0] + n[0] * w / 2, p[1] + n[1] * w / 2)
        for p, n, w in zip(spine, nrm, width, strict=True)
    ]
    lower = [
        (p[0] - n[0] * w / 2, p[1] - n[1] * w / 2)
        for p, n, w in zip(spine, nrm, width, strict=True)
    ]
    return upper[0], _catmull(upper) + _catmull(lower[::-1])


def _svg_d(start: Point, segments: list[Segment]) -> str:
    parts = [f"M{_fmt(start[0])} {_fmt(start[1])}"]
    parts += [
        f"C{_fmt(c1[0])} {_fmt(c1[1])} {_fmt(c2[0])} {_fmt(c2[1])}"
        f" {_fmt(end[0])} {_fmt(end[1])}"
        for c1, c2, end in segments
    ]
    parts.append("Z")
    return "".join(parts)


def _grid_d(cols: int = COLS) -> str:
    """The interior lattice lines only.

    The four outer lines are a ``<rect>``, which mitres its own corners; two
    crossing strokes would each stop at the other's centreline and leave a
    stepped notch half a stroke deep. The interior lines run to that centreline,
    where the border covers their ends, so the T-joins are clean too.
    """
    pitch = GRID_W / cols
    parts: list[str] = []
    for i in range(1, cols):
        x = GX + i * pitch
        y = GY + i * pitch
        parts.append(f"M{_fmt(x)} {_fmt(GY)}V{_fmt(GY + GRID_W)}")
        parts.append(f"M{_fmt(GX)} {_fmt(y)}H{_fmt(GX + GRID_W)}")
    return "".join(parts)


def _wave_shapes() -> str:
    shapes = "".join(f'<path d="{_svg_d(*ribbon_outline(r))}"/>' for r in RIBBONS)
    return shapes + (
        f'<circle cx="{_fmt(SRC_C[0])}" cy="{_fmt(SRC_C[1])}" r="{_fmt(SRC_R)}"/>'
    )


def mark_svg(
    ink: str = INK,
    grid_ink: str = GRID_INK,
    *,
    grid_opacity: float | None = None,
    theme_aware: bool = False,
) -> str:
    """The mark as a standalone SVG.

    With ``theme_aware`` the two roles carry the ``p-ink`` and ``p-grid`` class
    names and a ``prefers-color-scheme: dark`` rule lightens them against a dark
    ground, so one file covers both themes: the rule for the asset opened on its
    own, the classes for a page that inlines it and has its own theme switch,
    which can disagree with the operating system. The colours still
    travel as presentation attributes and the media query only overrides them:
    written as CSS custom properties instead, every renderer without variable
    support (cairosvg and svglib among them) falls back to black.
    """
    opacity = f' opacity="{grid_opacity}"' if grid_opacity is not None else ""
    style = ""
    if theme_aware:
        style = (
            "\n  <style>@media(prefers-color-scheme:dark){"
            f".p-ink{{fill:{INK_DARK}}}.p-grid{{stroke:{GRID_INK_DARK}}}"
            "}</style>"
        )

    grid_class = ' class="p-grid"' if theme_aware else ""
    ink_class = ' class="p-ink"' if theme_aware else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_fmt(VB)} {_fmt(VB)}"'
        ' role="img" aria-label="phonometry">\n'
        "  <title>phonometry</title>"
        f"{style}\n"
        f'  <g{grid_class} fill="none" stroke="{grid_ink}"'
        f' stroke-width="{_fmt(GRID_STROKE)}"{opacity}>\n'
        f'    <rect x="{_fmt(GX)}" y="{_fmt(GY)}"'
        f' width="{_fmt(GRID_W)}" height="{_fmt(GRID_W)}"/>\n'
        f'    <path d="{_grid_d()}"/>\n'
        "  </g>\n"
        f'  <g{ink_class} fill="{ink}">{_wave_shapes()}</g>\n'
        "</svg>\n"
    )


def _mpl_path(start: Point, segments: list[Segment]) -> MplPath:
    verts: list[Point] = [start]
    # The path codes are numpy uint8 constants; plain ints keep the list typed.
    codes: list[int] = [int(MplPath.MOVETO)]
    for c1, c2, end in segments:
        verts += [c1, c2, end]
        codes += [int(MplPath.CURVE4)] * 3
    verts.append(start)
    codes.append(int(MplPath.CLOSEPOLY))
    return MplPath(verts, codes)


def render_png(
    dest: Path,
    size: int,
    *,
    ink: str = INK,
    grid_ink: str = GRID_INK,
    background: str | None = None,
    margin: float = 0.0,
) -> None:
    """Rasterise the mark, from the same geometry the SVG emitters use.

    ``margin`` is the fraction of the canvas left empty on each side. Icons need
    it because platforms crop: a maskable Android icon may lose everything
    outside the middle 80 %, and iOS rounds the corners of what it is given.
    """
    fig = plt.figure(figsize=(1.0, 1.0), dpi=size)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    if background is not None:
        fig.patch.set_facecolor(background)
        ax.add_patch(
            Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes, facecolor=background, zorder=0
            )
        )
    else:
        fig.patch.set_alpha(0.0)

    span = VB / (1.0 - 2 * margin)
    inset = (span - VB) / 2
    ax.set_xlim(-inset, VB + inset)
    ax.set_ylim(VB + inset, -inset)  # SVG's y axis points down
    ax.set_aspect("equal")

    # The grid is drawn in points; at 1 inch square, 1 point is VB/72 units.
    lw = GRID_STROKE / span * 72.0
    ax.add_patch(
        Rectangle(
            (GX, GY),
            GRID_W,
            GRID_W,
            fill=False,
            edgecolor=grid_ink,
            linewidth=lw,
            joinstyle="miter",
            zorder=1,
        )
    )
    pitch = GRID_W / COLS
    for i in range(1, COLS):
        ax.plot(
            [GX + i * pitch] * 2,
            [GY, GY + GRID_W],
            color=grid_ink,
            linewidth=lw,
            solid_capstyle="butt",
            zorder=1,
        )
        ax.plot(
            [GX, GX + GRID_W],
            [GY + i * pitch] * 2,
            color=grid_ink,
            linewidth=lw,
            solid_capstyle="butt",
            zorder=1,
        )

    for ribbon in RIBBONS:
        ax.add_patch(
            PathPatch(
                _mpl_path(*ribbon_outline(ribbon)),
                facecolor=ink,
                edgecolor="none",
                zorder=2,
            )
        )
    ax.add_patch(Circle(SRC_C, SRC_R, facecolor=ink, edgecolor="none", zorder=2))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        # Software=None keeps matplotlib's version out of the PNG's tEXt
        # chunk, so a version bump does not rewrite every icon byte for byte.
        fig.savefig(
            dest,
            dpi=size,
            transparent=background is None,
            facecolor=fig.get_facecolor(),
            metadata={"Software": None},
        )
    plt.close(fig)
    _report(dest)


# --- wordmark and cards ------------------------------------------------------
# IBM Plex Sans, committed under .github/brand/fonts (SIL OFL 1.1). A grotesque
# drawn for technical documentation suits a metrology library, and pinning the
# file means the wordmark cannot shift when a machine's font set changes.
FONT_DIR = REPO / ".github" / "brand" / "fonts"
# Familjen Grotesk carries the wordmark and IBM Plex Sans the supporting copy.
# A display face distinct from the text face is what makes the wordmark read as
# a mark rather than as a caption; both are committed here (SIL OFL 1.1) so the
# lockup cannot shift when a machine's font set changes.
FONT_DISPLAY = FONT_DIR / "FamiljenGrotesk-Bold.ttf"
FONT_REGULAR = FONT_DIR / "IBMPlexSans-Regular.ttf"

WORDMARK = "phonometry"
TAGLINE = "Acoustic measurement toolkit for Python"
STRAPLINE = "Conformance-tested against the standards it implements"


def text_outline(text: str, font: Path, size: float, origin: Point) -> MplPath:
    """The glyph outlines of ``text``, in SVG coordinates.

    Outlines rather than a ``<text>`` element: GitHub renders README SVGs with
    no guarantee about available fonts, so live text falls back to whatever the
    viewer has, or to a serif. Converted glyphs look the same everywhere and
    need no font embedded. ``origin`` is the baseline start, as in SVG.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    tp = TextPath((0, 0), text, size=size, prop=FontProperties(fname=str(font)))
    # matplotlib's y axis points up and SVG's points down.
    return Affine2D().scale(1, -1).translate(*origin).transform_path(tp)


def text_svg_path(
    text: str,
    font: Path,
    size: float,
    origin: Point,
    colour: str,
    *,
    css_class: str = "",
) -> str:
    """One ``<path>`` element holding the text as outlines."""
    parts: list[str] = []
    for verts, code in text_outline(text, font, size, origin).iter_segments():
        if code == MplPath.MOVETO:
            parts.append(f"M{_fmt(verts[0])} {_fmt(verts[1])}")
        elif code == MplPath.LINETO:
            parts.append(f"L{_fmt(verts[0])} {_fmt(verts[1])}")
        elif code == MplPath.CURVE3:
            parts.append(
                f"Q{_fmt(verts[0])} {_fmt(verts[1])} {_fmt(verts[2])} {_fmt(verts[3])}"
            )
        elif code == MplPath.CURVE4:
            parts.append(
                f"C{_fmt(verts[0])} {_fmt(verts[1])}"
                f" {_fmt(verts[2])} {_fmt(verts[3])}"
                f" {_fmt(verts[4])} {_fmt(verts[5])}"
            )
        elif code == MplPath.CLOSEPOLY:
            parts.append("Z")
    attr = f' class="{css_class}"' if css_class else ""
    return f'<path{attr} fill="{colour}" d="{"".join(parts)}"/>'


def _mark_group(
    x: float,
    y: float,
    scale: float,
    ink: str,
    grid_ink: str,
    *,
    ink_class: str = "",
    grid_class: str = "",
) -> str:
    """The mark, translated and scaled into a larger composition."""
    ink_attr = f' class="{ink_class}"' if ink_class else ""
    grid_attr = f' class="{grid_class}"' if grid_class else ""
    return (
        f'<g transform="translate({_fmt(x)} {_fmt(y)}) scale({scale:.4f})">'
        f'<g{grid_attr} fill="none" stroke="{grid_ink}"'
        f' stroke-width="{_fmt(GRID_STROKE)}">'
        f'<rect x="{_fmt(GX)}" y="{_fmt(GY)}"'
        f' width="{_fmt(GRID_W)}" height="{_fmt(GRID_W)}"/>'
        f'<path d="{_grid_d()}"/></g>'
        f'<g{ink_attr} fill="{ink}">{_wave_shapes()}</g></g>'
    )


def _mark_ink_bounds() -> tuple[Point, Point]:
    """The mark's ink bounds in mark coordinates, as (top-left, bottom-right).

    Not the grid square: the outer rule is drawn on the 192-unit rectangle so
    half its stroke falls outside, and the wavefronts overshoot the grid on the
    right, so anything that trims the mark to its ink has to measure it.
    """
    half = GRID_STROKE / 2
    x0, y0 = GX - half, GY - half
    x1, y1 = GX + GRID_W + half, GY + GRID_W + half
    for ribbon in RIBBONS:
        box = _mpl_path(*ribbon_outline(ribbon)).get_extents()
        x0, y0 = min(x0, box.x0), min(y0, box.y0)
        x1, y1 = max(x1, box.x1), max(y1, box.y1)
    x0, y0 = min(x0, SRC_C[0] - SRC_R), min(y0, SRC_C[1] - SRC_R)
    x1, y1 = max(x1, SRC_C[0] + SRC_R), max(y1, SRC_C[1] + SRC_R)
    return (x0, y0), (x1, y1)


# Artwork the cards are composed over. Generated once and committed, then the
# mark and the type are drawn on top from the geometry above, so the wording can
# be revised without touching the art. Each is dark on the left, where the type
# sits, and carries its detail on the right.
ART_DIR = REPO / ".github" / "brand" / "art"


ART_QUALITY = 90  # see optimise_art for how this was chosen


def optimise_art(staging: Path, quality: int = ART_QUALITY) -> None:
    """Convert freshly generated artwork into the committed WebP masters.

    An authoring step, not part of ``make brand``: the art is regenerated only
    when the look changes, and the WebP is then the master. Measured against the
    1.3 MB source PNGs at 1920x640, WebP at this quality lands at 4.8 % of the
    original with a global SSIM of 0.983 and at most five levels of deviation in
    the flat dark areas where banding would show. It beat every alternative
    tried: JPEG needed 23 % more bytes for the same SSIM, palette quantisation
    stalled at 13 %, and the TinyPNG service returned either a PNG five times
    larger or a WebP visibly more aggressive than this one.
    """
    from PIL import Image

    ART_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(staging.glob("*.png")):
        dest = ART_DIR / f"{source.stem}.webp"
        with Image.open(source) as img:
            img.convert("RGB").save(dest, "WEBP", quality=quality, method=6)
        _report(dest)


def _background_uri(source: Path) -> str:
    """The committed artwork inlined verbatim as a data URI.

    Inlined rather than referenced: an SVG rendered by GitHub cannot fetch a
    second file, so a linked background would simply not appear. The bytes are
    passed through untouched rather than re-encoded, which would otherwise add a
    second generation of lossy compression on top of the master.
    """
    import base64

    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


HEADING = "#f2f8fa"
SUPPORT = "#a8d8e6"
SUPPORT_DIM = "#84bccf"


@dataclass(frozen=True)
class Line:
    """One line of copy in a card, positioned by its own baseline."""

    text: str
    font: Path
    size: float
    colour: str
    at: Point


@dataclass(frozen=True)
class Card:
    """A card's layout, shared by the SVG and the raster emitter."""

    width: float
    height: float
    mark_px: float
    mark_at: Point
    lines: tuple[Line, ...]


def banner_layout() -> Card:
    """The README lockup: mark on the left, wordmark and tagline beside it."""
    height = 320.0
    mark_px = 168.0
    text_x = 72.0 + mark_px + 48.0
    return Card(
        width=1200.0,
        height=height,
        mark_px=mark_px,
        mark_at=(72.0, (height - mark_px) / 2),
        lines=(
            Line(WORDMARK, FONT_DISPLAY, 68.0, HEADING, (text_x, height / 2 + 2)),
            Line(TAGLINE, FONT_REGULAR, 26.0, SUPPORT, (text_x + 2, height / 2 + 46)),
        ),
    )


def social_layout() -> Card:
    """The repository card: mark high on the left, three lines stacked below."""
    return Card(
        width=1280.0,
        height=640.0,
        mark_px=148.0,
        mark_at=(88.0, 96.0),
        lines=(
            Line(WORDMARK, FONT_DISPLAY, 74.0, HEADING, (88.0, 372.0)),
            Line(TAGLINE, FONT_REGULAR, 27.0, SUPPORT, (90.0, 424.0)),
            Line(STRAPLINE, FONT_REGULAR, 21.0, SUPPORT_DIM, (90.0, 468.0)),
        ),
    )


def card_svg(card: Card, art: Path) -> str:
    """A card as SVG: artwork inlined, then the mark and the type over it.

    One card rather than a light and a dark variant: the artwork brings its own
    dark ground, so it reads the same under either GitHub theme.
    """
    text = "\n  ".join(
        text_svg_path(line.text, line.font, line.size, line.at, line.colour)
        for line in card.lines
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_fmt(card.width)}'
        f' {_fmt(card.height)}" width="{_fmt(card.width)}"'
        f' height="{_fmt(card.height)}" role="img"'
        ' aria-label="phonometry, acoustic measurement toolkit for Python">\n'
        "  <title>phonometry</title>\n"
        f'  <image x="0" y="0" width="{_fmt(card.width)}"'
        f' height="{_fmt(card.height)}" preserveAspectRatio="xMidYMid slice"'
        f' href="{_background_uri(art)}"/>\n'
        f"  {_mark_group(*card.mark_at, card.mark_px / VB, ICON_INK, ICON_GRID)}\n"
        f"  {text}\n"
        "</svg>\n"
    )


def lockup_svg() -> str:
    """Mark and wordmark side by side, trimmed to their ink.

    The proportions are the banner's, read from ``banner_layout`` rather than
    restated, so the header of the documentation site and the card at the top of
    the README cannot drift apart. Two things differ from the card. The wordmark
    is emitted as outlines in ``currentColor``, so a page that inlines this
    asset colours it with the text around it and no web font has to load, which
    is also why the header cannot shift while it renders. And the viewBox is the
    ink itself rather than a padded canvas, so the height set in CSS is the
    height the reader sees, with the spacing left to the layout.

    The mark keeps the brand teals and carries the same class names as the
    favicon, so a dark ground can restyle it; the ``prefers-color-scheme`` rule
    is there for the file viewed on its own, and a site with its own theme
    switch overrides both classes from its stylesheet.
    """
    card = banner_layout()
    word = card.lines[0]
    scale = card.mark_px / VB
    (mx0, my0), (mx1, my1) = _mark_ink_bounds()
    mark = (
        card.mark_at[0] + mx0 * scale,
        card.mark_at[1] + my0 * scale,
        card.mark_at[0] + mx1 * scale,
        card.mark_at[1] + my1 * scale,
    )
    ink = text_outline(word.text, word.font, word.size, word.at).get_extents()
    x0, y0 = min(mark[0], ink.x0), min(mark[1], ink.y0)
    width = max(mark[2], ink.x1) - x0
    height = max(mark[3], ink.y1) - y0

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_fmt(width)}'
        f' {_fmt(height)}" role="img" aria-label="phonometry">\n'
        "  <title>phonometry</title>\n"
        "  <style>@media(prefers-color-scheme:dark){"
        f".p-ink{{fill:{INK_DARK}}}.p-grid{{stroke:{GRID_INK_DARK}}}"
        "}</style>\n"
        f'  <g transform="translate({_fmt(-x0)} {_fmt(-y0)})">\n'
        f"    {_mark_group(*card.mark_at, scale, INK, GRID_INK, ink_class='p-ink', grid_class='p-grid')}\n"
        f"    {text_svg_path(word.text, word.font, word.size, word.at, 'currentColor', css_class='p-word')}\n"
        "  </g>\n"
        "</svg>\n"
    )


def wordmark_svg() -> str:
    """The wordmark alone, as outlines in ``currentColor``, trimmed to its ink.

    For the landing hero's H1, the one place the name still rendered as live
    text: the theme set it in whatever sans the reader's system had, while the
    header lockup beside it carried the brand face. Outlines close that gap
    without fetching a web font, for the same reasons as the lockup. The word
    is set at the banner's size so the letterforms are sampled at the same
    fidelity, and the viewBox is the ink itself, so the height set in CSS is
    the height the reader sees.
    """
    word = banner_layout().lines[0]
    ink = text_outline(word.text, word.font, word.size, word.at).get_extents()
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0'
        f' {_fmt(ink.x1 - ink.x0)} {_fmt(ink.y1 - ink.y0)}" role="img"'
        ' aria-label="phonometry">\n'
        "  <title>phonometry</title>\n"
        f'  <g transform="translate({_fmt(-ink.x0)} {_fmt(-ink.y0)})">\n'
        f"    {text_svg_path(word.text, word.font, word.size, word.at, 'currentColor', css_class='p-word')}\n"
        "  </g>\n"
        "</svg>\n"
    )


def card_png(dest: Path, card: Card, art: Path, *, scale: int = 1) -> None:
    """The same card as a raster, composed from the same layout and geometry.

    Rasterised here rather than by converting the SVG: that would need an SVG
    renderer as a system dependency, and the icons already prove the two
    emitters agree because both read the geometry above.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Affine2D
    from PIL import Image

    width, height = round(card.width * scale), round(card.height * scale)
    dpi = 100.0
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.set_xlim(0, card.width)
    ax.set_ylim(card.height, 0)

    with Image.open(art) as img:
        # Cover, not stretch: crop the overhang so the artwork keeps its shapes.
        src = img.convert("RGB")
        target = card.width / card.height
        if src.width / src.height > target:
            keep = round(src.height * target)
            box = ((src.width - keep) // 2, 0, (src.width + keep) // 2, src.height)
        else:
            keep = round(src.width / target)
            box = (0, (src.height - keep) // 2, src.width, (src.height + keep) // 2)
        ax.imshow(
            src.resize((width, height), Image.Resampling.LANCZOS, box=box),
            extent=(0, card.width, card.height, 0),
            zorder=0,
        )

    unit = card.mark_px / VB
    trans = Affine2D().scale(unit).translate(*card.mark_at) + ax.transData
    lw = GRID_STROKE * unit * 72.0 / dpi * scale
    ax.add_patch(
        Rectangle(
            (GX, GY),
            GRID_W,
            GRID_W,
            fill=False,
            zorder=2,
            edgecolor=ICON_GRID,
            linewidth=lw,
            transform=trans,
            joinstyle="miter",
        )
    )
    pitch = GRID_W / COLS
    for i in range(1, COLS):
        ax.plot(
            [GX + i * pitch] * 2,
            [GY, GY + GRID_W],
            color=ICON_GRID,
            linewidth=lw,
            transform=trans,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            [GX, GX + GRID_W],
            [GY + i * pitch] * 2,
            color=ICON_GRID,
            linewidth=lw,
            transform=trans,
            solid_capstyle="butt",
            zorder=2,
        )
    for ribbon in RIBBONS:
        ax.add_patch(
            PathPatch(
                _mpl_path(*ribbon_outline(ribbon)),
                zorder=3,
                facecolor=ICON_INK,
                edgecolor="none",
                transform=trans,
            )
        )
    ax.add_patch(
        Circle(
            SRC_C,
            SRC_R,
            facecolor=ICON_INK,
            edgecolor="none",
            transform=trans,
            zorder=3,
        )
    )

    for line in card.lines:
        ax.text(
            line.at[0],
            line.at[1],
            line.text,
            color=line.colour,
            zorder=3,
            fontsize=line.size * 72.0 / dpi * scale,
            fontproperties=FontProperties(fname=str(line.font)),
            va="baseline",
        )

    import io

    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)

    # Re-encoded rather than written straight out: matplotlib's PNG of a
    # photographic card runs to hundreds of kilobytes, and these end up in a
    # README and in a social scraper's fetch.
    buf.seek(0)
    with Image.open(buf) as rendered:
        flat = rendered.convert("RGB")
        if dest.suffix == ".webp":
            flat.save(dest, "WEBP", quality=ART_QUALITY, method=6)
        elif dest.suffix in (".jpg", ".jpeg"):
            flat.save(dest, "JPEG", quality=92, optimize=True, progressive=True)
        else:
            flat.save(dest, "PNG", optimize=True)
    _report(dest)


def write_ico(dest: Path, source: Path) -> None:
    """Legacy .ico, for the browsers and tools that still ask for one by name."""
    from PIL import Image

    with Image.open(source) as img:
        img.save(dest, sizes=[(16, 16), (32, 32), (48, 48)])
    _report(dest)


def generate_all() -> None:
    """Write the mark and every derived asset to their committed locations."""
    brand = REPO / ".github" / "brand"
    public = REPO / "site" / "public"
    brand.mkdir(parents=True, exist_ok=True)

    print("Generating brand marks...")
    for path, svg in (
        # Theme-aware like the favicon and the lockup: the classes are what the
        # documentation site restyles per theme when it inlines this file (the
        # landing hero does), and the media rule keeps the asset legible when it
        # is opened on its own.
        (brand / "logo.svg", mark_svg(theme_aware=True)),
        # One flat colour inherited from the surrounding text, for badges,
        # stamps and anywhere the mark has to survive a single-ink reproduction.
        (
            brand / "logo-mono.svg",
            mark_svg("currentColor", "currentColor", grid_opacity=0.45),
        ),
        (public / "favicon.svg", mark_svg(theme_aware=True)),
        # Mark and wordmark as one asset, for the header of the documentation
        # site: the type is already outlines, so the header needs no web font
        # and cannot reflow while one loads.
        (brand / "lockup.svg", lockup_svg()),
        # The wordmark alone, for the landing hero's H1: same outlines as the
        # lockup's type, inheriting the text colour around it.
        (brand / "wordmark.svg", wordmark_svg()),
    ):
        path.write_text(svg, encoding="utf-8")
        _report(path)

    print("Generating app icons...")
    # Home-screen icons sit on the brand's darkest teal: a transparent or white
    # icon disappears against the light and dark wallpapers people actually use.
    icon = {"ink": ICON_INK, "grid_ink": ICON_GRID, "background": ICON_BG}
    render_png(public / "apple-touch-icon.png", 180, margin=0.10, **icon)
    render_png(public / "icon-192.png", 192, margin=0.08, **icon)
    render_png(public / "icon-512.png", 512, margin=0.08, **icon)
    # Android crops a maskable icon to an arbitrary shape and only guarantees
    # the middle 80 %, so this one keeps well clear of the edges.
    render_png(public / "icon-maskable-512.png", 512, margin=0.20, **icon)
    render_png(brand / "logo-1024.png", 1024, margin=0.04)
    # The mark reversed out for the cards: the site's Open Graph generator
    # composes it over the dark artwork and cannot recolour an SVG.
    render_png(
        brand / "logo-on-dark-512.png",
        512,
        margin=0.02,
        ink=ICON_INK,
        grid_ink=ICON_GRID,
    )

    print("Generating favicon fallback...")
    render_png(brand / "_favicon-48.png", 48, ink=INK, grid_ink=GRID_INK)
    write_ico(public / "favicon.ico", brand / "_favicon-48.png")
    (brand / "_favicon-48.png").unlink()

    print("Generating cards...")
    banner, social = banner_layout(), social_layout()
    banner_art, social_art = ART_DIR / "banner.webp", ART_DIR / "generic.webp"

    # The SVG is the editable master; the README points at the raster instead,
    # because GitHub serves README images through its own proxy and an SVG with
    # an embedded raster is not worth the risk on the page everybody sees first.
    for path, svg in ((brand / "banner.svg", card_svg(banner, banner_art)),):
        path.write_text(svg, encoding="utf-8")
        _report(path)
    card_png(brand / "banner.webp", banner, banner_art, scale=2)
    # The site-wide fallback, for the docstring-generated API pages that get no
    # card of their own. Same layout as the repository card, 1200x630.
    site_card = Card(
        width=1200.0,
        height=630.0,
        mark_px=148.0,
        mark_at=(88.0, 92.0),
        lines=social.lines,
    )
    card_png(public / "og-image.png", site_card, social_art)
    # GitHub's social-preview uploader takes PNG, JPG or GIF, so this one cannot
    # be WebP however much smaller that would be.
    card_png(brand / "social-preview.jpg", social, social_art)


if __name__ == "__main__":
    generate_all()
