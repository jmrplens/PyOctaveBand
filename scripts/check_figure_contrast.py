#  Copyright (c) 2026. Jose M. Requena-Plens
"""Perceptual contrast gate for the shaded regions of the documentation figures.

A filled region (an acceptance corridor, a period zone, a tolerance band) only
carries meaning when the reader can *see* it. Alpha-composited fills make that
easy to get wrong: ``alpha=0.10`` of a mid-tone hue reads as a pale tint over
the white page but collapses onto the near-black page of the dark theme, where
the same 10 % of the hue lands within a couple of levels of the background. The
figure still passes the staleness gate (:mod:`scripts.check_figures`), which
compares a regeneration against what is committed and knows nothing about how
the result looks.

This script closes that gap. It parses the committed SVG figures, reconstructs
every filled area region together with the background it is drawn on,
composites the fill with its own opacity exactly as a renderer does, and
measures the perceptual distance between the two.

Measures
--------
Two are reported for every region:

* **CIEDE2000** (``dE00``), the CIE's current colour-difference formula. It is
  the decision measure. ``dE00 ~ 1`` is the just-noticeable difference under
  ideal side-by-side conditions, so it is the right scale for "is this area a
  different colour from the page at all", and unlike a luminance ratio it also
  credits a difference carried by hue or chroma rather than lightness.
* **WCAG 2.2 relative-luminance contrast ratio**, reported for context. It is
  deliberately not the decision measure: it is blind to chroma (two colours of
  equal luminance score 1.00 however different they look), and its 3:1
  non-text-contrast requirement is written for the *boundary* of a graphical
  object, which in these figures is the limit line drawn on top of the fill,
  not the wash underneath it.

Threshold
---------
:data:`DELTA_E_MIN` is 10. The scale behind the number: ``dE00`` of 1 is a
just-noticeable difference, 2-3 is "visible when you look for it", and the
region where a difference is read at a glance without hunting for it starts
around 10. A documentation figure is skimmed, not inspected, so the shading has
to be in that last band. The value was also checked against the corpus: fills
the maintainer reads as fine in the light theme sit well above it, while every
region reported as invisible in the dark theme sits below 5.

Scope
-----
SVG only. The raster figures (``_RASTER_FIGURES`` in ``generate_graphs.py``)
are pcolormesh fields, spectrograms and dense time series: they have no
declarative fill regions to extract, only pixels, and their colour mapping is
already background-aware through ``_field_cmap``. Inside an SVG the check
looks at area regions (``fill_between`` polygons, ``axvspan``/``axhspan``
rectangles, bars, patches) above :data:`MIN_AREA_FRACTION` of the axes, which
excludes glyphs, markers and line joins without needing to special-case them.

Usage::

    python scripts/check_figure_contrast.py            # gate: exit 1 on any offender
    python scripts/check_figure_contrast.py --report   # full table, always exit 0
    python scripts/check_figure_contrast.py FILE ...   # restrict to given files

Exit status 0 when every fill clears the threshold, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from phonometry._plot.common import delta_e_2000

IMG_DIR = Path(".github/images")

#: Minimum CIEDE2000 distance between a filled region and its background.
DELTA_E_MIN = 10.0

#: Regions smaller than this fraction of the axes rectangle are ignored: they
#: are glyphs, markers, legend swatches and line joins rather than shaded
#: zones, and their visibility is carried by their own edge.
MIN_AREA_FRACTION = 0.004

_SVG_NS = "{http://www.w3.org/2000/svg}"
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

#: Coordinate pairs of an SVG path ``d`` attribute; command letters are dropped
#: because only the vertex ring is needed for a shoelace area.
_PATH_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: matplotlib names its groups after the artist that produced them. Only the
#: area artists are candidates; ``line2d`` (``fill: none``), ``text`` (glyphs),
#: ``legend`` (frame plus swatches) and ``pane3d`` (the walls of a 3-D axes,
#: which are page rather than content) never are.
_SKIP_GROUPS = (
    "text_", "legend_", "matplotlib.axis_", "line2d_", "pane3d_",
)


@dataclass(frozen=True)
class Region:
    """One filled area region measured against the background behind it."""

    figure: str
    group: str
    fill: tuple[float, float, float]
    opacity: float
    background: tuple[float, float, float]
    composited: tuple[float, float, float]
    area_fraction: float
    delta_e: float
    contrast_ratio: float

    @property
    def ok(self) -> bool:
        """True when the region clears :data:`DELTA_E_MIN`."""
        return self.delta_e >= DELTA_E_MIN

    def describe(self) -> str:
        """One-line report row."""
        return (
            f"{self.figure}: {self.group} "
            f"fill {_hex(self.fill)} @ {self.opacity:.2f} on {_hex(self.background)} "
            f"-> {_hex(self.composited)}  "
            f"dE00 {self.delta_e:5.2f}  WCAG {self.contrast_ratio:4.2f}:1  "
            f"area {self.area_fraction * 100:4.1f}%"
        )


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def _hex(rgb: tuple[float, float, float]) -> str:
    """``#rrggbb`` for a 0-1 sRGB triple."""
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)


def _parse_color(text: str) -> tuple[float, float, float] | None:
    """Parse an SVG colour token into a 0-1 sRGB triple (``None`` if not one)."""
    value = text.strip().lower()
    if value in ("none", "transparent", ""):
        return None
    if value == "white":
        return (1.0, 1.0, 1.0)
    if value == "black":
        return (0.0, 0.0, 0.0)
    if value.startswith("#"):
        digits = value[1:]
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        if len(digits) != 6:
            return None
        return tuple(int(digits[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    match = re.fullmatch(r"rgb\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", value)
    if match:
        return tuple(float(g) / 255.0 for g in match.groups())  # type: ignore[return-value]
    return None


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG 2.2 relative luminance of a 0-1 sRGB triple."""
    channels = np.asarray(rgb, dtype=np.float64)
    linear = np.where(
        channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4
    )
    return float(np.dot((0.2126, 0.7152, 0.0722), linear))


def contrast_ratio(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    """WCAG 2.2 contrast ratio between two sRGB colours (1.0 to 21.0)."""
    lum_a, lum_b = _relative_luminance(first), _relative_luminance(second)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def composite(
    fill: tuple[float, float, float],
    opacity: float,
    background: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Source-over composite of ``fill`` at ``opacity`` on ``background``."""
    return tuple(  # type: ignore[return-value]
        opacity * f + (1.0 - opacity) * b
        for f, b in zip(fill, background, strict=True)
    )


# ---------------------------------------------------------------------------
# SVG geometry
# ---------------------------------------------------------------------------


def _polygon_area(path_d: str) -> float:
    """Shoelace area of the vertex ring of an SVG path ``d`` attribute.

    Bezier control points are treated as vertices, which is exact for the
    straight-segment polygons matplotlib emits for fills and bars and a close
    approximation for the rounded corners of a legend or annotation box. The
    value is only used to decide whether a region is large enough to count as
    a shaded zone, so approximate is enough.
    """
    coords = [float(tok) for tok in _PATH_NUMBER.findall(path_d)]
    if len(coords) < 6:
        return 0.0
    points = np.asarray(coords[: len(coords) - len(coords) % 2]).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _parse_style(element: ElementTree.Element) -> dict[str, str]:
    """Declarations of an element's ``style`` attribute plus its presentation
    attributes, lowercased keys, presentation attributes taking lower priority.
    """
    declarations: dict[str, str] = {}
    for key in ("fill", "fill-opacity", "opacity", "stroke"):
        value = element.get(key)
        if value is not None:
            declarations[key] = value
    style = element.get("style", "")
    for chunk in style.split(";"):
        if ":" in chunk:
            key, _, value = chunk.partition(":")
            declarations[key.strip().lower()] = value.strip()
    return declarations


def _tag(element: ElementTree.Element) -> str:
    """Local name of an element's tag."""
    return element.tag.removeprefix(_SVG_NS)


def _element_area(
    element: ElementTree.Element, defs: dict[str, str]
) -> float:
    """Area of a drawable element in SVG user units (0 when not an area shape)."""
    tag = _tag(element)
    if tag == "path":
        return _polygon_area(element.get("d", ""))
    if tag == "rect":
        return float(element.get("width", 0)) * float(element.get("height", 0))
    if tag == "polygon":
        return _polygon_area(element.get("points", ""))
    if tag == "use":
        href = (element.get(_XLINK_HREF) or element.get("href") or "").lstrip("#")
        return _polygon_area(defs.get(href, ""))
    return 0.0


def _collect_defs(root: ElementTree.Element) -> dict[str, str]:
    """Map every ``id`` in the document to the path data it carries."""
    return {
        element.get("id", ""): element.get("d", "")
        for element in root.iter(f"{_SVG_NS}path")
        if element.get("id")
    }


def _collect_clip_areas(root: ElementTree.Element) -> dict[str, float]:
    """Map every ``clipPath`` id to the area of the rectangle it encloses.

    matplotlib clips the artists of an axes to the axes rectangle, so this is
    the reference area a drawn region is compared against, and it survives
    ``set_axis_off()`` (which suppresses the background rectangle entirely).
    """
    areas: dict[str, float] = {}
    for clip in root.iter(f"{_SVG_NS}clipPath"):
        for shape in clip:
            area = _element_area(shape, {})
            if area > 0.0:
                areas[clip.get("id", "")] = area
                break
    return areas


def _clip_id(element: ElementTree.Element) -> str:
    """Referenced ``clipPath`` id of an element, or ``""`` when unclipped."""
    match = re.fullmatch(r"url\(#(.+)\)", element.get("clip-path", "").strip())
    return match.group(1) if match else ""


def _axes_geometry(
    group: ElementTree.Element,
    clip_areas: dict[str, float],
    fallback: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float, str]:
    """Background colour, reference area and background group id of an axes.

    The background rectangle is the first ``patch_N`` child of the ``axes_N``
    group *and* is the only patch matplotlib leaves unclipped, which
    distinguishes it from a drawn region that happens to come first when the
    axes frame is switched off (schematic figures). Its ``fill`` declaration is
    omitted entirely when it is black, the SVG initial value and exactly what
    the dark theme uses. The reference area comes from the axes clip rectangle
    when the artists are clipped, so it is right even for an axes whose
    background rectangle is not drawn at all.
    """
    background, area, background_group = fallback, 0.0, ""
    for child in group:
        if _tag(child) != "g" or not child.get("id", "").startswith("patch_"):
            continue
        for shape in child.iter():
            if _tag(shape) not in ("path", "rect") or _clip_id(shape):
                continue
            declarations = _parse_style(shape)
            colour = _parse_color(declarations.get("fill", "black"))
            shape_area = _element_area(shape, {})
            # An unclipped patch that is *stroked* is a drawn outline (a polar
            # frame, a schematic box), not the page it sits on.
            if (
                colour is not None
                and shape_area > 0.0
                and _parse_color(declarations.get("stroke", "none")) is None
            ):
                background, area = colour, shape_area
                background_group = child.get("id", "")
            break
        break
    for element in group.iter():
        clip = _clip_id(element)
        if clip in clip_areas:
            area = clip_areas[clip]
            break
    return background, area, background_group


def _walk(
    node: ElementTree.Element,
    *,
    figure: str,
    defs: dict[str, str],
    clip_areas: dict[str, float],
    background: tuple[float, float, float],
    axes_area: float,
    group_id: str,
    background_group: str,
    regions: list[Region],
) -> None:
    """Recursively measure the filled area regions under ``node``."""
    for child in node:
        tag = _tag(child)
        if tag == "g":
            child_id = child.get("id", "") or group_id
            if child_id.startswith(_SKIP_GROUPS) or child_id == background_group:
                continue
            child_bg, child_area = background, axes_area
            child_bg_group = background_group
            if child_id.startswith(("axes_", "figure_")):
                child_bg, child_area, child_bg_group = _axes_geometry(
                    child, clip_areas, background
                )
            _walk(
                child,
                figure=figure,
                defs=defs,
                clip_areas=clip_areas,
                background=child_bg,
                axes_area=child_area,
                group_id=child_id,
                background_group=child_bg_group,
                regions=regions,
            )
            continue
        if axes_area <= 0.0:
            continue
        declarations = _parse_style(child)
        fill = _parse_color(declarations.get("fill", ""))
        if fill is None:
            continue
        opacity = float(declarations.get("fill-opacity", 1.0)) * float(
            declarations.get("opacity", 1.0)
        )
        area = _element_area(child, defs)
        fraction = area / axes_area
        if fraction < MIN_AREA_FRACTION:
            continue
        if opacity >= 1.0 and fill == background:
            continue  # an opaque plate in the page colour, drawn to mask
        blended = composite(fill, opacity, background)
        regions.append(
            Region(
                figure=figure,
                group=group_id,
                fill=fill,
                opacity=opacity,
                background=background,
                composited=blended,
                area_fraction=fraction,
                delta_e=delta_e_2000(blended, background),
                contrast_ratio=contrast_ratio(blended, background),
            )
        )


def measure(path: Path) -> list[Region]:
    """Every filled area region of one SVG figure, measured against its page."""
    root = ElementTree.parse(path).getroot()
    defs = _collect_defs(root)
    regions: list[Region] = []
    _walk(
        root,
        figure=path.name,
        defs=defs,
        clip_areas=_collect_clip_areas(root),
        background=(1.0, 1.0, 1.0),
        axes_area=0.0,
        group_id="",
        background_group="",
        regions=regions,
    )
    return regions


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "files", nargs="*", type=Path, help="SVG figures (default: all committed)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print every measured region, not just the offenders, and always "
        "exit 0",
    )
    parser.add_argument(
        "--min-delta-e",
        type=float,
        default=DELTA_E_MIN,
        metavar="DE",
        help=f"CIEDE2000 threshold (default: {DELTA_E_MIN:g})",
    )
    args = parser.parse_args(argv)

    files = args.files or sorted(IMG_DIR.glob("*.svg"))
    regions: list[Region] = []
    for path in files:
        regions.extend(measure(path))

    if args.report:
        for region in sorted(regions, key=lambda r: r.delta_e):
            print(region.describe())
        print(f"\n{len(regions)} filled regions in {len(files)} figures.")
        return 0

    offenders = [r for r in regions if r.delta_e < args.min_delta_e]
    if offenders:
        print(
            f"::error::{len(offenders)} filled region(s) below the "
            f"dE00 {args.min_delta_e:g} legibility threshold - see "
            "scripts/check_figure_contrast.py"
        )
        for region in sorted(offenders, key=lambda r: (r.figure, r.delta_e)):
            print(f"  - {region.describe()}")
        return 1

    print(
        f"All {len(regions)} filled regions in {len(files)} figures clear "
        f"dE00 {args.min_delta_e:g}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
