#!/usr/bin/env python3
"""Find strokes that composite to near-invisibility on the dark page.

The contrast checker measures filled regions. A stroke is not a fill, so a
half-opacity mid grey on a 0.4 pt line passed every gate and disappeared on the
dark theme, which is how the ISO 3745 wireframe shipped invisible.

Blends every stroke colour with its stroke-opacity over the dark page and
reports the ones that land close to it, weighted by how thin the line is: a
hairline needs more contrast than a thick one to read at all.

Two kinds of stroke are not measured, because for them the page is not what
the reader sees behind the line:

* **Grid and axis furniture**, which is drawn to recede.  In two dimensions it
  is recognised by colour (:data:`SCAFFOLDING`); in three, matplotlib names
  the group itself (:data:`SCAFFOLD_GROUPS`), which is how the 3-D grid of the
  microphone arrays is exempt without pinning the theme-derived grey it is
  drawn in.
* **Strokes inside an axes that an opaque field covers** -- a filled contour
  set, a mesh, an embedded raster.  The level lines of a noise contour are
  drawn on viridis and the ISO 1999 contours on YlOrRd; they are read against
  that field, and the field's own legibility is
  ``scripts/check_figure_contrast.py``'s business, not this one's.
"""

from __future__ import annotations

import pathlib
import re
from xml.etree import ElementTree as ET

#: The dark documentation page, from scripts/figures/theme.py.
PAGE = (0x1C, 0x21, 0x28)

SVG_NS = "{http://www.w3.org/2000/svg}"

#: Grid and axis furniture: drawn to recede, so the floor below does not apply.
#: Everything else with a stroke and no fill is a line somebody meant to be seen.
SCAFFOLDING = {(0x55, 0x55, 0x55), (0xE0, 0xE0, 0xE0), (0xB0, 0xB0, 0xB0)}

#: Groups matplotlib names for its own scaffolding. The 3-D grid is drawn in a
#: colour derived from the page (``theme_fill(foreground, ax, delta_e=24)``),
#: so it cannot be recognised from a list of literal greys the way the 2-D grid
#: is; the group name is what stays true when the theme moves.
SCAFFOLD_GROUPS = ("grid3d",)

#: Group names that mean the axes is covered by an opaque field rather than by
#: the page: a filled contour set, a quadrilateral mesh (``pcolormesh``, and
#: the colourbar of either).
FIELD_GROUPS = ("QuadContourSet", "QuadMesh")


def _srgb_to_lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = (_srgb_to_lin(v / 255.0) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _parse(style: str) -> tuple[tuple[int, int, int] | None, float, float]:
    colour = re.search(r"stroke:\s*#([0-9a-fA-F]{6})", style)
    opacity = re.search(r"stroke-opacity:\s*([0-9.]+)", style)
    width = re.search(r"stroke-width:\s*([0-9.]+)", style)
    if not colour:
        return None, 1.0, 1.0
    v = colour.group(1)
    rgb = (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
    return (
        rgb,
        float(opacity.group(1)) if opacity else 1.0,
        float(width.group(1)) if width else 1.0,
    )


def _group_id(element: ET.Element) -> str:
    """The ``id`` of a group element, without matplotlib's trailing counter."""
    if element.tag != f"{SVG_NS}g":
        return ""
    return element.get("id", "").rsplit("_", 1)[0]


def _is_field(element: ET.Element) -> bool:
    """Whether this subtree paints an opaque field over the axes it is in."""
    if element.tag == f"{SVG_NS}image":
        return True
    if _group_id(element) in FIELD_GROUPS:
        # A contour set drawn as *lines* is the thing being checked, not a
        # field: only the filled one hides the page.
        return any("fill: #" in (child.get("style") or "") for child in element.iter())
    return any(_is_field(child) for child in element)


def _strokes(
    element: ET.Element, on_field: bool = False
) -> list[tuple[tuple[int, int, int], float, float]]:
    """Every measurable stroke in a subtree, as ``(rgb, opacity, width)``."""
    group = _group_id(element)
    if group in SCAFFOLD_GROUPS:
        return []
    if group == "axes":
        on_field = any(_is_field(child) for child in element)
    found: list[tuple[tuple[int, int, int], float, float]] = []
    style = element.get("style") or ""
    if "stroke:" in style and "fill: none" in style and not on_field:
        rgb, alpha, width = _parse(style)
        if rgb is not None and rgb not in SCAFFOLDING:
            found.append((rgb, alpha, width))
    for child in element:
        found.extend(_strokes(child, on_field))
    return found


def main() -> int:
    images = pathlib.Path(".github/images")
    rows: list[tuple[float, str, str]] = []
    for svg in sorted(images.glob("*_dark.svg")):
        seen: set[tuple[tuple[int, int, int], float, float]] = set()
        root = ET.parse(svg).getroot()
        for rgb, alpha, width in _strokes(root):
            key = (rgb, alpha, width)
            if key in seen:
                continue
            seen.add(key)
            blended = tuple(alpha * c + (1 - alpha) * p for c, p in zip(rgb, PAGE))
            ratio = _ratio(blended, PAGE)  # type: ignore[arg-type]
            # A hairline has to work harder: below 0.6 pt demand 3:1, else 2:1.
            floor = 3.0 if width < 0.6 else 2.0
            if ratio < floor:
                detail = (
                    f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x} alpha={alpha} "
                    f"width={width} -> {ratio:.2f}:1 (min {floor})"
                )
                rows.append((ratio, svg.name, detail))
    rows.sort()
    print(f"{len(rows)} low-contrast strokes on the dark page\n")
    for _, name, detail in rows:
        print(f"  {name:<52} {detail}")
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
