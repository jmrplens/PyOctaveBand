#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Deterministic generator for the phonometry brand assets: the mark itself and
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

import matplotlib

matplotlib.use("Agg")
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
        spine=((32.5, 32.5), (46.8, 35.3), (71.2, 44.4), (111.2, 56.1),
               (163.5, 80.0), (217.0, 88.5), (232.0, 90.0)),
        width=(7.0, 10.0, 12.0, 18.0, 21.0, 11.0, 0.0),
    ),
    Ribbon(  # through the middle of the grid
        spine=((32.5, 32.5), (46.8, 61.3), (74.0, 74.5), (121.4, 101.0),
               (168.8, 139.2), (223.3, 155.1), (233.0, 157.0)),
        width=(7.5, 11.0, 14.0, 23.5, 21.0, 10.0, 0.0),
    ),
    Ribbon(  # plunges out of the source, then runs out along the bottom
        spine=((32.5, 32.5), (46.8, 92.6), (77.1, 109.1), (116.8, 154.8),
               (159.2, 208.2), (218.3, 234.4), (231.0, 236.5)),
        width=(8.0, 12.0, 15.5, 21.0, 22.0, 12.0, 0.0),
    ),
)

Point = tuple[float, float]
Segment = tuple[Point, Point, Point]  # two control points and an end point


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
    """The closed outline of one ribbon: up the top edge, back along the bottom."""
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

    With ``theme_aware`` a ``prefers-color-scheme: dark`` rule lightens the mark
    against a dark browser chrome, so one favicon covers both. The colours still
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
        f"  <g{grid_class} fill=\"none\" stroke=\"{grid_ink}\""
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
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=background, zorder=0))
    else:
        fig.patch.set_alpha(0.0)

    span = VB / (1.0 - 2 * margin)
    inset = (span - VB) / 2
    ax.set_xlim(-inset, VB + inset)
    ax.set_ylim(VB + inset, -inset)  # SVG's y axis points down
    ax.set_aspect("equal")

    # The grid is drawn in points; at 1 inch square, 1 point is VB/72 units.
    lw = GRID_STROKE / span * 72.0
    ax.add_patch(Rectangle((GX, GY), GRID_W, GRID_W, fill=False,
                           edgecolor=grid_ink, linewidth=lw, joinstyle="miter",
                           zorder=1))
    pitch = GRID_W / COLS
    for i in range(1, COLS):
        ax.plot([GX + i * pitch] * 2, [GY, GY + GRID_W],
                color=grid_ink, linewidth=lw, solid_capstyle="butt", zorder=1)
        ax.plot([GX, GX + GRID_W], [GY + i * pitch] * 2,
                color=grid_ink, linewidth=lw, solid_capstyle="butt", zorder=1)

    for ribbon in RIBBONS:
        ax.add_patch(PathPatch(_mpl_path(*ribbon_outline(ribbon)),
                               facecolor=ink, edgecolor="none", zorder=2))
    ax.add_patch(Circle(SRC_C, SRC_R, facecolor=ink, edgecolor="none", zorder=2))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(dest, dpi=size, transparent=background is None,
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {dest.relative_to(REPO)}")


def write_ico(dest: Path, source: Path) -> None:
    """Legacy .ico, for the browsers and tools that still ask for one by name."""
    from PIL import Image

    with Image.open(source) as img:
        img.save(dest, sizes=[(16, 16), (32, 32), (48, 48)])
    print(f"  {dest.relative_to(REPO)}")


def generate_all() -> None:
    """Write the mark and every derived asset to their committed locations."""
    brand = REPO / ".github" / "brand"
    public = REPO / "site" / "public"
    brand.mkdir(parents=True, exist_ok=True)

    print("Generating brand marks...")
    for path, svg in (
        (brand / "logo.svg", mark_svg()),
        # One flat colour inherited from the surrounding text, for badges,
        # stamps and anywhere the mark has to survive a single-ink reproduction.
        (brand / "logo-mono.svg", mark_svg("currentColor", "currentColor",
                                           grid_opacity=0.45)),
        (public / "favicon.svg", mark_svg(theme_aware=True)),
    ):
        path.write_text(svg, encoding="utf-8")
        print(f"  {path.relative_to(REPO)}")

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

    print("Generating favicon fallback...")
    render_png(brand / "_favicon-48.png", 48, ink=INK, grid_ink=GRID_INK)
    write_ico(public / "favicon.ico", brand / "_favicon-48.png")
    (brand / "_favicon-48.png").unlink()


if __name__ == "__main__":
    generate_all()
