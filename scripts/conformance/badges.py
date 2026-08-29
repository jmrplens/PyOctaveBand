#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The conformance report's visual indicators, as committed SVG.

A verdict used to be an emoji. An emoji is not a picture the project controls:
it is a font lookup, so it renders as a coloured glyph on one machine, as an
outline on another and as a tofu box on a third, it carries a screen-reader
announcement nobody here wrote, and there is no such glyph for "by design" or
"not applicable" at all. Words replaced it, which fixed the meaning; this
module puts a picture back beside the word, one the repository owns.

Four marks and one banner, all drawn here as geometry and committed under
``.github/badges``. That directory rather than ``.github/images`` because
``make graphs`` deletes every ``*.svg`` there before regenerating, so a badge
parked beside the figures would vanish on the next figure run.

**Light and dark, decided by measurement.** A per-row mark paints its own
opaque ground and uses one fill that clears 3:1 on GitHub light (#ffffff),
GitHub dark (#0d1117), GitHub dim (#22272e) and Starlight's #17181c alike, so
the same single file is correct in every theme. This is why a shields.io badge
has worked in both themes for a decade with no theme variant. A per-theme
pair would need a ``<picture>`` element per row, and 566 of those add 163 kB to
``docs/CONFORMANCE.md`` against 9.6 kB for reference-style images of a
single-file mark. The banner is the opposite case: once per page, so the pair
costs 278 characters, and a card wants a card-coloured ground that no single
fill gives on both. It therefore ships as ``…-summary.svg`` plus
``…-summary_dark.svg``, the ``_dark`` suffix the README's ``<picture>`` blocks
and ``site/src/components/ThemeImage.astro`` already pair on.

An in-SVG ``@media (prefers-color-scheme: dark)`` is deliberately not used.
GitHub loads an SVG through ``<img>``, where the query resolves against the
reader's operating system rather than against the GitHub theme toggle, so a
reader on a light GitHub page with a dark desktop gets the dark palette on
white.

**Shape carries the meaning, not only colour.** Filled circle, filled hexagon,
filled rounded square, hollow ring: four silhouettes that stay distinguishable
in greyscale and to a reader who cannot separate the hues. Colour agrees with
the shape; it never has to carry it alone.

**The word is the indicator; the mark illustrates it.** Every consumer states
the verdict in text - the Markdown alt, the ``<title>`` here - so a reader
whose images failed, or who is listening rather than looking, gets the verdict
and not a filename.

Determinism, because the output is committed and CI diffs the whole tree:
nothing here reads a clock, a version or the environment. The counts come from
``docs/conformance.json``, the palette and geometry are literals, the glyphs
are outlines of two committed fonts (see :mod:`svg_text`), and every coordinate
is rounded through one of two formatters before it is written.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any

import svg_text

from .artifact import load
from .registry import _ROOT, Verdict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Where the marks and the banner are committed. Not ``.github/images``: the
#: ``graphs`` target empties that directory of SVGs before every figure run.
BADGE_DIR = _ROOT / ".github" / "badges"

#: Prefix for the absolute URLs the documents will point at. GitHub serves a
#: raw URL from this repository directly; only third-party image hosts are
#: rewritten through ``camo.githubusercontent.com``. Kept here so the Markdown
#: renderer, the README and the pull-request comment all spell it one way.
RAW_BASE = "https://raw.githubusercontent.com/jmrplens/phonometry"

#: Path fragment under a ref, e.g. ``.../main/{RAW_PATH}/verdict-pass.svg``.
RAW_PATH = ".github/badges"

#: The two committed faces, as the brand assets use them: a display grotesque
#: for figures, a text grotesque for words.
_FONT_DIR = _ROOT / ".github" / "brand" / "fonts"
_FONT_DISPLAY = _FONT_DIR / "FamiljenGrotesk-Bold.ttf"
_FONT_REGULAR = _FONT_DIR / "IBMPlexSans-Regular.ttf"


@dataclass(frozen=True)
class Mark:
    """One verdict's indicator: the file, the word, and how to cite it.

    :param verdict: The verdict string the artefact stores.
    :param filename: The committed file, under :data:`BADGE_DIR`.
    :param alt: What a reader gets when the image does not arrive. It is the
        verdict in words, never the name of a picture.
    :param label: The Markdown link-reference label the tables use. Reference
        style rather than inline: a definition resolves document-wide, which
        turns a 117-character inline tag into a 16-character reference and is
        what keeps 566 illustrated rows to +9.6 kB.
    """

    verdict: str
    filename: str
    alt: str
    label: str


#: The four marks, in the order a legend should read them.
MARKS: tuple[Mark, ...] = (
    Mark(str(Verdict.PASS), "verdict-pass.svg", "Pass", "cv-pass"),
    Mark(str(Verdict.FAIL), "verdict-fail.svg", "Fail", "cv-fail"),
    Mark(str(Verdict.BY_DESIGN), "verdict-by-design.svg", "By design", "cv-by-design"),
    Mark(str(Verdict.NOT_APPLICABLE), "verdict-not-applicable.svg", "n/a", "cv-na"),
)

#: The summary banner's light variant, and the name the dark one is derived
#: from. ``BANNER_DARK`` is below, once the palettes exist to name it.
BANNER = "conformance-summary.svg"

# --- palette -----------------------------------------------------------------
# Verdict fills, chosen against every ground the marks land on rather than
# copied from a UI kit. Measured WCAG 2.2 contrast of each fill against the
# five grounds, and of the white glyph against the fill itself:
#
#                    #ffffff  #f6f8fa  #0d1117  #22272e  #17181c   white on it
#   green  #1f883d      4.52     4.24     4.19     3.33     3.93          4.52
#   red    #da3633      4.61     4.33     4.11     3.26     3.85          4.61
#   blue   #1f6feb      4.63     4.35     4.08     3.24     3.83          4.63
#   grey   #6e7781      4.55     4.27     4.16     3.30     3.90          4.55
#
# The grounds are GitHub light, GitHub's own table stripe, GitHub dark, GitHub
# dim and Starlight's page colour. All four fills clear 3:1 on every one of
# them and carry the white glyph at 4.5:1, which is what makes one file per
# mark correct in every theme and a per-row ``<picture>`` unnecessary.
#
# Dim is the ground that decides the blue. Primer's accent blue for light mode
# (#0969da) measures 2.89 there and is rejected for it, as is Primer's danger
# red (#cf222e, 2.81 on dim) and shields.io's green (#4c1, 2.12 under white).
# ``test_conformance_badges.py`` measures this table rather than trusting it.
_FILLS = {
    str(Verdict.PASS): "#1f883d",
    str(Verdict.FAIL): "#da3633",
    str(Verdict.BY_DESIGN): "#1f6feb",
    str(Verdict.NOT_APPLICABLE): "#6e7781",
}

#: The glyph cut out of a filled mark, and only ever that.
_GLYPH = "#ffffff"


@dataclass(frozen=True)
class Palette:
    """One theme's colours for the banner.

    The banner ships as a pair, so unlike a mark it may use a per-theme colour
    and does: a card needs a ground a hair off the page, which no single fill
    provides on both #ffffff and #0d1117.

    :param suffix: Appended before the extension, ``""`` or ``"_dark"``.
    :param card: The card's ground.
    :param border: Its hairline.
    :param ink: Figures and anything that must read first.
    :param muted: Supporting words.
    :param track: The unfilled part of the bar.
    :param passing: The bar's filled part, and nothing else.
    :param failing: The bar's remainder when a check fails.
    """

    suffix: str
    card: str
    border: str
    ink: str
    muted: str
    track: str
    passing: str
    failing: str


#: Primer's light tokens: canvas.subtle, borderColor.default, fg.default,
#: fg.muted, success.emphasis, danger.emphasis.
LIGHT = Palette(
    suffix="",
    card="#f6f8fa",
    border="#d1d9e0",
    ink="#1f2328",
    muted="#59636e",
    track="#e6eaef",
    passing="#1f883d",
    failing="#cf222e",
)

#: The same roles in Primer's dark tokens.
DARK = Palette(
    suffix="_dark",
    card="#151b23",
    border="#3d444d",
    ink="#f0f6fc",
    muted="#9198a1",
    track="#262c36",
    passing="#3fb950",
    failing="#f85149",
)


def banner_name(palette: Palette) -> str:
    """The file one palette's banner is committed as.

    The pairing rule is written once, here, because two other places already
    depend on it being exactly this substitution: the ``<picture>`` blocks in
    the README, and ``ThemeImage.astro``, which is handed only the light URL
    and derives the dark one by inserting ``_dark`` before the extension.
    """
    return BANNER.replace(".svg", f"{palette.suffix}.svg")


#: The dark twin of :data:`BANNER`, by that rule.
BANNER_DARK = banner_name(DARK)


def _svg(
    body: str, *, width: float, height: float, title: str, units: float = 1.0
) -> str:
    """Wrap drawn geometry in a root element.

    ``role="img"`` plus a ``<title>`` is the SVG-native text alternative. It
    reaches a reader who opens the file directly; the reader who meets it
    through GitHub gets the Markdown alt instead, because an SVG in an ``<img>``
    exposes nothing of its own to assistive technology. Both say the verdict.

    :param body: The drawn geometry.
    :param width: Intrinsic width, in CSS pixels.
    :param height: Intrinsic height, in CSS pixels.
    :param title: The text alternative.
    :param units: Drawing units per pixel. The banner is authored ten units to
        the pixel so every coordinate can be written as a whole number; the
        marks are authored one to one.
    """
    # Escaped even though every title today is literals and integers: the
    # title is the only place data reaches the markup, and a caption that
    # someday carries a standard's name with an ampersand in it should not
    # produce a file no parser accepts.
    safe = escape(title, quote=True)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_trim(width)}" '
        f'height="{_trim(height)}" '
        f'viewBox="0 0 {_trim(width * units)} {_trim(height * units)}" '
        f'role="img" aria-label="{safe}">'
        f"<title>{safe}</title>{body}</svg>\n"
    )


def _trim(value: float) -> str:
    """A mark's coordinate, on the house convention for SVG path data.

    A hundredth of a pixel on a 16-pixel mark: finer than any renderer
    resolves, coarser than any difference two machines could compute.
    """
    return svg_text.two_decimals(value)


def _polygon(points: Sequence[tuple[float, float]]) -> str:
    """Closed path data through ``points``."""
    head = f"M{_trim(points[0][0])} {_trim(points[0][1])}"
    rest = "".join(f"L{_trim(x)} {_trim(y)}" for x, y in points[1:])
    return f"{head}{rest}Z"


def _hexagon(centre: float, radius: float) -> str:
    """A pointy-top regular hexagon, the silhouette reserved for a failure.

    Pointy-top rather than flat-top so the outline differs from the circle at
    every angle, not only at the corners: at 16 pixels a shape is recognised by
    its profile long before its colour is judged.
    """
    points = [
        (
            centre + radius * math.cos(math.radians(90 + 60 * step)),
            centre - radius * math.sin(math.radians(90 + 60 * step)),
        )
        for step in range(6)
    ]
    return _polygon(points)


def _stroke(data: str, colour: str, width: float) -> str:
    """One stroked path with round ends, for a glyph drawn as strokes."""
    return (
        f'<path d="{data}" fill="none" stroke="{colour}" '
        f'stroke-width="{_trim(width)}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )


def _mark_body(verdict: str) -> str:
    """The geometry of one 16-pixel mark, without its root element."""
    if verdict == str(Verdict.PASS):
        # Filled disc, white tick. The settled convention for "this is fine",
        # and the only one of the four that needs no second reading.
        return f'<circle cx="8" cy="8" r="7.5" fill="{_FILLS[verdict]}"/>' + _stroke(
            "M4.4 8.35L6.9 10.85L11.6 5.5", _GLYPH, 1.9
        )
    if verdict == str(Verdict.FAIL):
        return f'<path d="{_hexagon(8.0, 8.0)}" fill="{_FILLS[verdict]}"/>' + _stroke(
            "M5.5 5.5L10.5 10.5M10.5 5.5L5.5 10.5", _GLYPH, 1.9
        )
    if verdict == str(Verdict.BY_DESIGN):
        # A bar, not a cross: the architecture was excluded from the mask on
        # purpose, so the mark has to say "out of scope" and not "wrong".
        return (
            f'<rect x="0.5" y="0.5" width="15" height="15" rx="4.5" '
            f'fill="{_FILLS[verdict]}"/>' + _stroke("M4.6 8L11.4 8", _GLYPH, 2.1)
        )
    # Hollow, and the only hollow one: nothing was filled in because the clause
    # does not reach this library. Read in greyscale it is still the odd one.
    return (
        f'<circle cx="8" cy="8" r="6.9" fill="none" '
        f'stroke="{_FILLS[verdict]}" stroke-width="1.8"/>'
        + _stroke("M5.1 10.9L10.9 5.1", _FILLS[verdict], 1.8)
    )


def render_marks() -> dict[str, str]:
    """The four verdict marks, filename to SVG source.

    :return: One entry per :data:`MARKS` member, in that order.
    """
    return {
        mark.filename: _svg(
            _mark_body(mark.verdict), width=16, height=16, title=mark.alt
        )
        for mark in MARKS
    }


# --- banner ------------------------------------------------------------------
# Drawn in a ten-times integer coordinate space: the viewBox is 7200x840 while
# the element is 720x84, so every coordinate written is a whole number at a
# tenth of a pixel. That is finer than any renderer resolves and it removes the
# last place a float could be formatted differently on two machines, at about a
# third of the bytes of the same drawing at full precision.
_SCALE = 10
_BANNER_W = 720.0
_BANNER_H = 84.0
_PAD = 22.0
_HEADLINE_SIZE = 30.0
_CAPTION_SIZE = 13.0
_STAT_FIGURE_SIZE = 14.0
_STAT_WORD_SIZE = 12.5
_BASELINE = 42.0
_BAR_Y = 59.0
_BAR_H = 7.0
_RUN_GAP = 6.0


def _units(value: float) -> str:
    """A value already in drawing units, as a whole number."""
    return str(round(value))


def _u(pixels: float) -> str:
    """A layout constant written in pixels, as a whole number of drawing units.

    The layout reads in pixels because that is the size the reader sees; the
    file is written in tenths of a pixel because a whole number cannot be
    formatted two ways.
    """
    return _units(pixels * _SCALE)


def _run(text: str, font: pathlib.Path, size: float, colour: str, x: float) -> str:
    """One left-to-right run of set text, as a filled ``<path>``."""
    origin = (x * _SCALE, _BASELINE * _SCALE)
    data = svg_text.path_data(text, font, size * _SCALE, origin, _units)
    return f'<path fill="{colour}" d="{data}"/>'


def _runs(pieces: Sequence[tuple[str, pathlib.Path, float, str]], x: float) -> str:
    """Several runs set one after another from ``x``, separated by a gap."""
    out: list[str] = []
    cursor = x
    for text, font, size, colour in pieces:
        out.append(_run(text, font, size, colour, cursor))
        cursor += svg_text.ink_width(text, font, size) + _RUN_GAP
    return "".join(out)


def _runs_width(pieces: Sequence[tuple[str, pathlib.Path, float, str]]) -> float:
    """How wide :func:`_runs` will draw ``pieces``."""
    inks = [svg_text.ink_width(text, font, size) for text, font, size, _ in pieces]
    return sum(inks) + _RUN_GAP * (len(inks) - 1)


def _plural(count: int, singular: str) -> str:
    """``singular`` agreeing with ``count``.

    Both nouns the banner names take a plain ``-s``. The banner is read by
    people, and a fixture with one domain in it should not be captioned
    "1 domains" in the sentence a screen reader gets.
    """
    return singular if count == 1 else f"{singular}s"


def _bar(passed: int, total: int, palette: Palette) -> str:
    """The pass-fraction bar.

    Plain rectangles inside a rounded clip, so the boundary between the passing
    and the failing part is a straight edge in the middle of the bar rather
    than a rounded cap that would read as the end of the bar.

    The clip's ``id`` needs no prefix. Every consumer loads this file through
    an ``<img>`` or a ``<picture>``, which makes it its own document; it is
    never inlined into a page where the name could meet another one.
    """
    width = _BANNER_W - 2 * _PAD
    filled = width * passed / total if total else width
    clip = (
        f'<clipPath id="bar"><rect x="{_u(_PAD)}" y="{_u(_BAR_Y)}" '
        f'width="{_u(width)}" height="{_u(_BAR_H)}" rx="{_u(_BAR_H / 2)}"/></clipPath>'
    )
    parts = [
        f'<rect x="{_u(_PAD)}" y="{_u(_BAR_Y)}" width="{_u(width)}" '
        f'height="{_u(_BAR_H)}" fill="{palette.track}"/>',
        f'<rect x="{_u(_PAD)}" y="{_u(_BAR_Y)}" width="{_u(filled)}" '
        f'height="{_u(_BAR_H)}" fill="{palette.passing}"/>',
    ]
    if passed < total:
        parts.append(
            f'<rect x="{_u(_PAD + filled)}" y="{_u(_BAR_Y)}" '
            f'width="{_u(width - filled)}" height="{_u(_BAR_H)}" '
            f'fill="{palette.failing}"/>'
        )
    return f'{clip}<g clip-path="url(#bar)">{"".join(parts)}</g>'


def banner_alt(counts: Mapping[str, Any]) -> str:
    """What the banner says, in words.

    The banner is a picture of four numbers; this is the same four numbers as a
    sentence, and it is what a reader gets when the image does not load. It is
    the text alternative for the ``<title>`` here and for the Markdown ``alt``
    wherever the banner is embedded, so the two cannot drift apart.
    """
    passed, total = counts["passing"], counts["checks"]
    failing = total - passed
    verdict = (
        f"{passed} of {total} conformance checks pass"
        if failing
        else f"All {total} conformance checks pass"
    )
    tail = f", {failing} failing" if failing else ""
    domains, standards = counts["domains"], counts["standards"]
    return (
        f"{verdict}{tail}, across {domains} {_plural(domains, 'domain')} "
        f"and {standards} {_plural(standards, 'standard')}"
    )


def render_banner(counts: Mapping[str, Any], palette: Palette) -> str:
    """The summary banner for one theme.

    :param counts: The artefact's ``counts`` block, which is where the live
        figures come from; nothing here is written down twice.
    :param palette: :data:`LIGHT` or :data:`DARK`.
    :return: The SVG source.
    """
    passed, total = counts["passing"], counts["checks"]
    failing = total - passed
    headline = f"{passed}/{total}"
    caption = "conformance checks pass"
    if failing:
        caption = f"conformance checks pass, {failing} failing"

    domains, standards = counts["domains"], counts["standards"]
    stats: tuple[tuple[str, pathlib.Path, float, str], ...] = (
        (str(domains), _FONT_DISPLAY, _STAT_FIGURE_SIZE, palette.ink),
        (_plural(domains, "domain"), _FONT_REGULAR, _STAT_WORD_SIZE, palette.muted),
        ("·", _FONT_REGULAR, _STAT_WORD_SIZE, palette.muted),
        (str(standards), _FONT_DISPLAY, _STAT_FIGURE_SIZE, palette.ink),
        (_plural(standards, "standard"), _FONT_REGULAR, _STAT_WORD_SIZE, palette.muted),
    )

    headline_width = svg_text.ink_width(headline, _FONT_DISPLAY, _HEADLINE_SIZE)
    body = (
        f'<rect x="{_u(0.5)}" y="{_u(0.5)}" width="{_u(_BANNER_W - 1)}" '
        f'height="{_u(_BANNER_H - 1)}" rx="{_u(12)}" fill="{palette.card}" '
        f'stroke="{palette.border}" stroke-width="{_u(1)}"/>'
        + _run(headline, _FONT_DISPLAY, _HEADLINE_SIZE, palette.ink, _PAD)
        + _run(
            caption,
            _FONT_REGULAR,
            _CAPTION_SIZE,
            palette.muted,
            _PAD + headline_width + 10.0,
        )
        + _runs(stats, _BANNER_W - _PAD - _runs_width(stats))
        + _bar(passed, total, palette)
    )
    return _svg(
        body,
        width=_BANNER_W,
        height=_BANNER_H,
        title=banner_alt(counts),
        units=_SCALE,
    )


def assets(document: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Every file this module owns, filename to SVG source.

    :param document: The artefact; read from disk when omitted.
    :return: The four marks then the two banner variants, in that order.
    """
    source = load() if document is None else document
    counts = source["counts"]
    written = render_marks()
    for palette in (LIGHT, DARK):
        written[banner_name(palette)] = render_banner(counts, palette)
    return written


def asset_names() -> tuple[str, ...]:
    """The filenames :func:`assets` writes, without drawing anything.

    Cheap enough for a gate to call: the manifest is data, so a check that the
    documents point at files that exist does not have to set type to find out
    what those files are called.
    """
    return (
        *(mark.filename for mark in MARKS),
        *(banner_name(palette) for palette in (LIGHT, DARK)),
    )


def write(
    document: Mapping[str, Any] | None = None,
    directory: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Write every asset, and remove any SVG there that is no longer one.

    The sweep is what makes a rename fail loudly. Without it a renamed badge
    leaves its old file in place, tracked and unchanged, so the tree stays
    clean, CI stays green, and the only symptom is a 404 in a rendered
    document. It is confined to ``*.svg`` directly inside the target directory,
    which this module is the sole author of.

    :param document: The artefact; read from disk when omitted.
    :param directory: Where to write; defaults to :data:`BADGE_DIR`.
    :return: The paths written, in manifest order.
    """
    target = BADGE_DIR if directory is None else directory
    target.mkdir(parents=True, exist_ok=True)
    written = assets(document)
    for stale in sorted(target.glob("*.svg")):
        if stale.name not in written:
            stale.unlink()
    paths: list[pathlib.Path] = []
    for name, source in written.items():
        path = target / name
        path.write_text(source, encoding="utf8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Regenerate the committed indicators from the committed artefact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help=f"directory to write into (default: {BADGE_DIR.relative_to(_ROOT)})",
    )
    args = parser.parse_args(argv)
    paths = write(directory=args.out)
    print(
        f"[conformance] {len(paths)} indicators at {args.out or BADGE_DIR}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
