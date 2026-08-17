#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The glyph-outlining engine of the diagram plates.

The plates bake every label to glyph outlines at generation time, so the
committed artwork renders identically on every machine instead of falling
back to whatever sans-serif the viewer owns. This module is the whole
engine, isolated from the canvas: which font files exist
(:data:`_FACES`), how a string splits across them (:func:`segment`), how
wide it is (:func:`advance`), where each glyph lands (:func:`shape`),
the per-document outline atlas (:class:`GlyphStore`) and the writer that
turns styled runs into ``<g>``/``<use>`` markup (:func:`emit_runs`).

Two decisions carry the whole design. **One typeface across the drawn
corpus**: every face in :data:`_FACES` is a DejaVu face out of the
pinned matplotlib wheel, the same family and the same pin that sets the
figures, so a plate and a figure sitting on one documentation page are
set in one typeface instead of two. Nothing else fixes the plates'
metrics -- the compositions are held against these advances by the
canvas's fit gate and by the census below. **Coverage is the chain**,
not the primary face: a glyph the primary lacks falls back per-cluster
to the next face, resolved deterministically at generation time --
which replaces the silent, machine-dependent fontconfig substitution
the live-text plates used to take. No face in the chain covering a
character is a hard error, never a blank.

Every face is loaded by file path; fontconfig is never consulted. All
measurement is the **pen advance** of the shaped string, never the ink
extents of a ``TextPath``: the ink box loses side bearings and eats the
boundary spaces that a fifth of the math runs begin or end with.

``python scripts/diagrams/outline.py --census`` measures every label of
every plate in both languages against its 900 px sheet and reports the
overhangs -- the permanent debugging tool behind the canvas's fit gate.
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import matplotlib
from matplotlib import (
    _text_helpers,
    ft2font,
)
from matplotlib.ft2font import FT2Font, GlyphIndexType, LoadFlags
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Combining diacritics a symbol may carry (T̂, L̄, x̃); each travels with
#: the base letter it modifies, so the pair shapes, styles and falls back
#: as one glyph cluster. Circumflex, macron, overline, dot above, tilde,
#: acute, grave, breve and dot below.
_COMBINING = "\u0302\u0304\u0305\u0307\u0303\u0301\u0300\u0306\u0323"

_MPL_TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

#: A face request: ``(mono, bold, italic)``.
FaceKey = tuple[bool, bool, bool]

#: (mono, bold, italic) -> the ordered chain of face files, every one of
#: them out of the pinned matplotlib wheel. The proportional faces stand
#: alone: one family leaves no second family to put under them, and what
#: the oblique cuts lack against the upright is Arabic, Tifinagh, N'Ko,
#: Ogham and the pre-italicised Mathematical Alphanumerics, none of which
#: a plate can set. The mono faces do fall back, to the proportional face
#: of the same weight, because they lack characters this corpus really
#: does set (``ℓ`` and ``∥``). There is no mono italic face on purpose:
#: no plate sets code in italic.
_FACES: dict[FaceKey, tuple[Path, ...]] = {
    (False, False, False): (_MPL_TTF / "DejaVuSans.ttf",),
    (False, True, False): (_MPL_TTF / "DejaVuSans-Bold.ttf",),
    (False, False, True): (_MPL_TTF / "DejaVuSans-Oblique.ttf",),
    (False, True, True): (_MPL_TTF / "DejaVuSans-BoldOblique.ttf",),
    (True, False, False): (_MPL_TTF / "DejaVuSansMono.ttf",
                           _MPL_TTF / "DejaVuSans.ttf"),
    (True, True, False): (_MPL_TTF / "DejaVuSansMono-Bold.ttf",
                          _MPL_TTF / "DejaVuSans-Bold.ttf"),
}

#: The fixed grid every face is sized at: glyph ids, outlines and per-glyph
#: positions are all expressed at 100 pt / 72 dpi and scaled into place by
#: the run group's transform, matplotlib's own scheme for outlined text.
_GRID_SIZE = 100.0
_GRID_DPI = 72


class Run(NamedTuple):
    """One styled run of a label: the glyphs and how they are set."""

    text: str
    face: FaceKey
    #: Baseline shift as a fraction of the *element* font size (positive
    #: drops, negative rises); 0.0 at the baseline.
    shift: float
    #: Glyph scale relative to the element font size (script runs shrink).
    scale: float


class FaceRun(NamedTuple):
    """A maximal substring covered by one font file of a chain."""

    text: str
    path: Path


_capabilities_checked = False


def assert_capabilities() -> None:
    """Fail fast when the environment cannot outline the plates faithfully.

    Without libraqm, combining marks get their own advance and ``T̂``
    silently becomes ``T ˆ``; a face missing from the wheel would drop
    its strings down the chain and quietly change every metric it sets.
    Checked once per process.
    """
    global _capabilities_checked
    if _capabilities_checked:
        return
    raqm = getattr(ft2font, "__libraqm_version__", None)
    if not raqm:
        raise RuntimeError(
            "matplotlib was built without libraqm: combining marks would "
            "shape with their own advance and the plates would mis-set "
            "every T̂/L̄/x̃ label"
        )
    for chain in _FACES.values():
        for path in chain:
            if not path.is_file():
                raise RuntimeError(f"font file missing from the chain: {path}")
    _capabilities_checked = True


@cache
def _font(path: Path) -> FT2Font:
    """The sized FT2Font for a file, cached: one face object per file."""
    font = FT2Font(str(path))
    font.set_size(_GRID_SIZE, _GRID_DPI)
    return font


@cache
def _coverage(path: Path) -> frozenset[int]:
    """The codepoints a font file maps to glyphs, via its best cmap."""
    from fontTools.ttLib import TTFont

    with TTFont(str(path), lazy=True) as ttf:
        return frozenset(ttf.getBestCmap())


def _clusters(text: str) -> Iterator[str]:
    """Split into glyph clusters: a base with its trailing combining marks."""
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and text[j] in _COMBINING:
            j += 1
        yield text[i:j]
        i = j


def segment(text: str, face_key: FaceKey) -> list[FaceRun]:
    """Split ``text`` into maximal substrings coverable by one chain file.

    The primary face takes every cluster it fully covers; what it lacks
    moves to the fallback. The split point never lands between a base
    character and a :data:`_COMBINING` mark: if any member of a cluster
    is missing from a face, the whole cluster moves down the chain, so a
    mark is always shaped against the base it modifies. A character no
    face draws is a hard error naming string and character -- never
    ``.notdef``, never a blank.
    """
    chain = _FACES[face_key]
    runs: list[FaceRun] = []
    for cluster in _clusters(text):
        for path in chain:
            if all(ord(ch) in _coverage(path) for ch in cluster):
                break
        else:
            ch = next(
                (ch for ch in cluster
                 if not any(ord(ch) in _coverage(p) for p in chain)),
                None,
            )
            if ch is not None:
                raise ValueError(
                    f"no face in the chain draws {ch!r} of {text!r}"
                )
            raise ValueError(
                f"no single face in the chain covers the cluster "
                f"{cluster!r} of {text!r}"
            )
        if runs and runs[-1].path == path:
            runs[-1] = FaceRun(runs[-1].text + cluster, path)
        else:
            runs.append(FaceRun(cluster, path))
    return runs


@cache
def _advance_grid(path: Path, text: str) -> float:
    """Pen advance of the shaped ``text`` at the 100 pt grid, in px."""
    font = _font(path)
    font.set_text(text, flags=LoadFlags.NO_HINTING)
    return font.get_width_height()[0] / 64


def advance(text: str, font_path: Path, size: float) -> float:
    """Pen advance of ``text`` at ``size`` px, shaped in one face file.

    The **pen advance**, never the ink extents: ``TextPath.get_extents``
    loses side bearings and eats boundary spaces, and a fifth of the math
    runs begin or end with a space that must keep its width.
    """
    return _advance_grid(font_path, text) * size / _GRID_SIZE


def shape(text: str, font_path: Path) -> list[tuple[GlyphIndexType, float, float]]:
    """Shape ``text`` in one face: ``(glyph_id, x, y)`` at the 100 pt grid.

    Shaping goes through matplotlib's libraqm layout helper, so GPOS
    kerning, ligatures and mark-to-base attachment survive; ``y`` is the
    mark-attachment offset (positive up, the grid's own axis), nonzero
    only for positioned marks. The helper is private API, guarded by a
    canary test so a wheel bump fails in the test suite, not in artwork.
    """
    font = _font(font_path)
    return [
        (item.glyph_index, item.x, item.y)
        for item in _text_helpers.layout(text, font)
    ]


def _fmt(value: float) -> str:
    """Two decimals, trailing zeros stripped, so path data stays readable."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


def _fmt_scale(value: float) -> str:
    """Four decimals for the group scale, which multiplies whole outlines."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


@cache
def _glyph_path_d(path: Path, glyph_index: GlyphIndexType) -> str:
    """The outline of one glyph as SVG path data at the 100 pt grid.

    Empty for a blank glyph (spaces): the caller counts its advance and
    emits nothing, so ``<path d=""/>`` never reaches the artwork.
    """
    font = _font(path)
    font.load_glyph(glyph_index, flags=LoadFlags.NO_HINTING)
    verts, codes = font.get_path()
    parts: list[str] = []
    for points, code in MplPath(verts, codes).iter_segments(simplify=False):
        if code == MplPath.MOVETO:
            parts.append(f"M{_fmt(points[0])} {_fmt(points[1])}")
        elif code == MplPath.LINETO:
            parts.append(f"L{_fmt(points[0])} {_fmt(points[1])}")
        elif code == MplPath.CURVE3:
            parts.append(f"Q{_fmt(points[0])} {_fmt(points[1])}"
                         f" {_fmt(points[2])} {_fmt(points[3])}")
        elif code == MplPath.CURVE4:
            parts.append(f"C{_fmt(points[0])} {_fmt(points[1])}"
                         f" {_fmt(points[2])} {_fmt(points[3])}"
                         f" {_fmt(points[4])} {_fmt(points[5])}")
        elif code == MplPath.CLOSEPOLY:
            parts.append("Z")
    return "".join(parts)


class GlyphStore:
    """The per-document outline atlas: each glyph's path stored once.

    Ids are content-derived -- ``{face file stem}-{glyph id:x}`` -- so
    they are stable across regenerations and never collide with the
    ``axes_``/``figure_`` namespaces the figure checkers key on. The
    dict is insertion-ordered, which makes ``defs()`` deterministic.
    """

    def __init__(self) -> None:
        self._paths: dict[str, str] = {}

    def use(self, font_path: Path, glyph_index: GlyphIndexType) -> str | None:
        """Register a glyph and return its id, or ``None`` for a blank."""
        d = _glyph_path_d(font_path, glyph_index)
        if not d:
            return None
        ref = f"{font_path.stem}-{glyph_index:x}"
        self._paths.setdefault(ref, d)
        return ref

    def defs(self) -> str:
        """The ``<defs>`` payload: one ``<path>`` per stored glyph."""
        return "".join(
            f'<path id="{ref}" d="{d}"/>' for ref, d in self._paths.items()
        )


def emit_runs(store: GlyphStore, runs: list[Run], x0: float, y: float,
              size: float, fill: str) -> str:
    """Write styled runs as ``<g>``/``<use>`` groups walking one pen.

    One group per (run x face) segment: ``translate`` places the pen at
    the run baseline (the script shift is a fraction of the *element*
    size, never the script size), ``scale`` sizes the 100 pt outlines and
    carries the y-flip, and each ``<use>`` offsets a glyph in grid units.
    Whitespace advances the pen and emits nothing.
    """
    parts: list[str] = []
    pen = x0
    for run in runs:
        run_size = size * run.scale
        run_y = y + run.shift * size
        for sub, face_path in segment(run.text, run.face):
            uses: list[str] = []
            for glyph_index, gx, gy in shape(sub, face_path):
                ref = store.use(face_path, glyph_index)
                if ref is None:
                    continue
                xa = f' x="{_fmt(gx)}"' if gx else ""
                ya = f' y="{_fmt(gy)}"' if gy else ""
                uses.append(f'<use href="#{ref}"{xa}{ya}/>')
            if uses:
                k = _fmt_scale(run_size / _GRID_SIZE)
                parts.append(
                    f'<g fill="{fill}" transform="translate({_fmt(pen)} '
                    f'{_fmt(run_y)}) scale({k} -{k})">' + "".join(uses) + "</g>"
                )
            pen += advance(sub, face_path, run_size)
    return "".join(parts)


def measure(runs: list[Run], size: float) -> float:
    """Total pen advance of a label's runs at the element ``size`` px."""
    return sum(
        advance(sub, face_path, size * run.scale)
        for run in runs
        for sub, face_path in segment(run.text, run.face)
    )


# --- the census tool ---------------------------------------------------------

class _Label(NamedTuple):
    plate: str
    text: str
    x0: float
    x1: float
    sheet: float


def _census_labels() -> list[_Label]:
    """Measure every label of every plate, both languages, writing nothing."""
    from . import canvas
    from .i18n import visit
    from .registry import DIAGRAMS

    assert_capabilities()
    records: list[_Label] = []

    class MeasuringSVG(canvas.SVG):
        plate = ""

        def text(self, x: float, y: float, s: str, size: int = 20,
                 fill: str = "", anchor: str = "middle", bold: bool = False,
                 mono: bool = False, italic: bool = False) -> None:
            s = self.tr(s)
            runs = canvas._label_runs(s, mono=mono, bold=bold, italic=italic)
            if not runs:
                return
            width = measure(runs, size)
            x0 = x - {"start": 0.0, "middle": width / 2, "end": width}[anchor]
            records.append(_Label(self.plate, s, x0, x0 + width, self.w))

        def render(self, title: str) -> str:
            self.text(self.w / 2, 30, title, canvas._TITLE_SIZE, bold=True)
            return ""

    for name, (builder, title, height) in DIAGRAMS.items():
        for lang, lang_suffix in (("en", ""), ("es", "_es")):
            visit(name, lang)
            svg = MeasuringSVG(900, height, canvas.LIGHT, lang)
            svg.plate = f"{name}{lang_suffix}"
            builder(svg, canvas.LIGHT)
            svg.render(title)
    return records


def main(argv: list[str] | None = None) -> int:
    """``--census``: list the labels overhanging their sheet (``--all``: all)."""
    args = sys.argv[1:] if argv is None else argv
    if "--census" not in args:
        print(__doc__)
        return 2
    show_all = "--all" in args
    labels = _census_labels()
    overhangs = 0
    for rec in labels:
        over = max(0.0, -rec.x0) + max(0.0, rec.x1 - rec.sheet)
        if over > 0.5:
            overhangs += 1
            print(f"{rec.plate}: {rec.text!r} spans "
                  f"{rec.x0:.0f}..{rec.x1:.0f} on a {rec.sheet:.0f} px sheet "
                  f"({over:.0f} px overhang)")
        elif show_all:
            print(f"{rec.plate}: {rec.text!r} spans "
                  f"{rec.x0:.0f}..{rec.x1:.0f} on a {rec.sheet:.0f} px sheet")
    print(f"{len(labels)} labels measured, {overhangs} overhang the sheet")
    return 1 if overhangs else 0


if __name__ == "__main__":
    # Run as a script: put the package root on the path and re-enter as
    # the package member so the relative imports of the census resolve.
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(importlib.import_module("diagrams.outline").main())
