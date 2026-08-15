#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The sheet the diagrams are drawn on: theme, canvas and the writing of files.

One subject because a builder never sees anything else. :class:`Theme`
fixes the palette, and the light and dark instances of it are the same
colours the matplotlib figures use, so the docs can theme-switch an SVG
exactly like a raster plot. :class:`SVG` is the element accumulator with
the technical-drawing helpers a setup diagram needs (dimension lines,
microphones on stands, people, hatched ground). :func:`_write` renders a
builder the four ways the documentation embeds it: English and Spanish,
each light and dark.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from .i18n import lookup, visit


@dataclass(frozen=True)
class Theme:
    suffix: str
    bg: str
    fg: str
    muted: str
    panel: str
    primary: str
    secondary: str
    accent: str


LIGHT = Theme(
    suffix="", bg="#ffffff", fg="#1a1a1a", muted="#666666", panel="#f0f2f5",
    primary="#1f77b4", secondary="#d62728", accent="#2ca02c",
)
DARK = Theme(
    suffix="_dark", bg="#0d1117", fg="#e6e6e6", muted="#9a9a9a", panel="#1c2128",
    primary="#4da3d8", secondary="#e46a6a", accent="#5abf5a",
)

_FONT = "Segoe UI, Helvetica, Arial, sans-serif"
_MONO = "Consolas, Menlo, monospace"

#: Combining diacritics a symbol may carry (T̂, L̄, x̃); each travels with
#: the base letter it modifies, so the pair styles as one glyph. Circumflex,
#: macron, overline, dot above, tilde, acute, grave, breve and dot below.
_COMBINING = "\u0302\u0304\u0305\u0307\u0303\u0301\u0300\u0306\u0323"

#: The letter runs a sub/superscript sets upright: the descriptive
#: subscripts of the corpus, printed in roman by the standards that define
#: them -- weightings, averages and exposure (Aeq, eq, EQ, EX), extremes
#: and bounds (max, MAX, min, upper, lower, low, high, limit), qualifiers
#: (ref, rms, tot, TOT, eff, mod, norm, spec, inst, cal, tab, cum, ss,
#: shadow, co, tr, diff, ff, ax, SN, CS, MS), the hand-arm/whole-body
#: vibration axes of ISO 5349 and ISO 2631 (hv, hwx, hwy, hwz, wx, wy, wz),
#: and the Spanish twins the i18n table sets beside them (sup for upper,
#: ef for eff). Every other letter run inside a script is an index and is
#: set in italic ($K_{ij}$, $η_{ij}$); extend this set only for a subscript
#: that abbreviates a word, never for letter-indices.
_ROMAN_SCRIPTS = frozenset((
    "Aeq", "eq", "EQ", "EX", "max", "MAX", "min", "upper", "lower", "sup",
    "low", "high", "limit", "ref", "rms", "tot", "TOT", "eff", "ef", "mod",
    "norm", "spec", "inst", "cal", "tab", "cum", "ss", "shadow", "co",
    "tr", "diff", "ff", "ax", "SN", "CS", "MS", "hv", "hwx", "hwy", "hwz",
    "wx", "wy", "wz",
))

#: Script metrics of the ``$...$`` composer, as fractions of the font size:
#: how far a subscript drops, how far a superscript rises, and the glyph
#: scale of both.
_SUB_DROP = 0.22
_SUP_RISE = -0.38
_SCRIPT_SCALE = 0.70

#: Font size of the title :meth:`SVG.render` sets across the top. The
#: ``$...$`` composer scales scripts against the size the ``<text>`` is
#: given, so the two must always be the same number.
_TITLE_SIZE = 26


def _esc(s: str) -> str:
    """Escape XML metacharacters so labels may contain <, > and & literally."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _math_tokens(run: str, s: str, script: bool = False) -> list[tuple[str, str]]:
    """Split one math run into ``(kind, text)`` chunks.

    ``var`` is set in italic: at the baseline a single letter -- Latin or
    lowercase Greek, with any combining marks -- and at script level
    (*script* true, tokenizing the payload of a ``_``/``^``) also letter
    runs, which there are indices (``$K_{ij}$``), unless the run is one of
    the descriptive subscripts of :data:`_ROMAN_SCRIPTS` (``$L_{Aeq}$``).
    ``up`` stays upright: digits, operators, primes, brackets, capital
    Greek letters at every level (``Δ``, ``Φ``: operators and descriptors
    per the roman Δ of ISO 80000-2, the ``Δ_SOR`` print of ECAC Doc 29 and
    the upright ``\\Delta`` mathtext sets in the matplotlib figures), and
    at the baseline runs of two or more Latin letters, which are operator
    names and acronyms (log, grad, CN), never products; a product of two
    symbols is written with an explicit space or middle dot between them. ``sub`` and ``sup``
    carry the payload of ``_``/``^``, braced or single character, to be
    tokenized again at script size.

    Malformed markup raises :class:`ValueError` naming the whole string *s*
    and the offending piece, so a typo breaks the generation instead of
    publishing a silently mis-set diagram: a multi-character script written
    without braces (``f_max`` sets only the f-m pair and pushes "ax" back to
    the baseline), a comma glued to an unbraced script (``L_p,s`` would push
    ",s" back to the baseline; spaced-off commas as in ``a_x , a_y`` stay
    legal), a script marker with an empty payload (``L_``, ``L_{}``), an
    unclosed script brace, and a script inside a script (``L_{p_1}``),
    which the composer cannot set.
    """
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(run):
        ch = run[i]
        if ch in "_^":
            if script:
                raise ValueError(
                    f"nested script {run[i:]!r} inside a script of {s!r}: "
                    "the composer sets a single script level"
                )
            kind = "sub" if ch == "_" else "sup"
            if i + 1 == len(run):
                raise ValueError(
                    f"empty script {ch!r} at the end of a math run in {s!r}"
                )
            if run[i + 1] == "{":
                end = run.find("}", i + 2)
                if end < 0:
                    raise ValueError(
                        f"unclosed script brace {run[i:]!r} in {s!r}"
                    )
                if end == i + 2:
                    raise ValueError(
                        f"empty script {run[i:end + 1]!r} in {s!r}"
                    )
                out.append((kind, run[i + 2:end]))
                i = end + 1
            else:
                nxt = run[i + 2:i + 3]
                if nxt == "(":
                    raise ValueError(
                        f"ambiguous script {run[i:i + 2]!r} before '(' in "
                        f"{s!r}: brace the script and keep the argument "
                        f"outside, as {ch}{{{run[i + 1]}}}(...)"
                    )
                if nxt.isalnum():
                    j = i + 2
                    while j < len(run) and run[j].isalnum():
                        j += 1
                    raise ValueError(
                        f"ambiguous script {run[i:j]!r} in {s!r}: only "
                        f"{run[i + 1]!r} would attach to {ch!r}, brace the "
                        f"whole script as {ch}{{...}}"
                    )
                if nxt == "," and i + 3 < len(run) and run[i + 3].isalnum():
                    j = i + 3
                    while j < len(run) and (run[j].isalnum()
                                            or run[j] == ","):
                        j += 1
                    raise ValueError(
                        f"ambiguous comma {run[i:j]!r} in {s!r}: glued to "
                        f"the script it reads as part of the subscript, "
                        f"write {ch}{{{run[i + 1]},...}} or space the "
                        "comma off"
                    )
                out.append((kind, run[i + 1:i + 2]))
                i += 2
        elif ch.isalpha():
            latin = ch.isascii()
            j = i + 1
            while j < len(run) and (
                run[j] in _COMBINING
                or (latin and run[j].isascii() and run[j].isalpha())
            ):
                j += 1
            if not latin and ch.isupper():
                # Capital Greek is upright at every level: in this corpus
                # it is an operator or a descriptor, never an index.
                kind = "up"
            elif script:
                kind = "up" if run[i:j] in _ROMAN_SCRIPTS else "var"
            else:
                letters = sum(1 for c in run[i:j] if c not in _COMBINING)
                kind = "var" if letters == 1 else "up"
            out.append((kind, run[i:j]))
            i = j
        else:
            j = i
            while j < len(run) and run[j] not in "_^" and not run[j].isalpha():
                j += 1
            out.append(("up", run[i:j]))
            i = j
    return out


def _math_runs(s: str) -> list[tuple[str, bool, float, float]]:
    """Chunk a translated ``$...$`` string into styled runs.

    Each run is ``(text, italic, shift, scale)``: the glyphs, whether they
    are italicised, how far the baseline drops (positive) or rises
    (negative) as a fraction of the font size, and the glyph scale
    (:data:`_SCRIPT_SCALE` inside a script). Adjacent chunks of identical
    style merge into one run -- that merge is a shaping-boundary decision,
    not an optimisation: ``CN = `` kerns and shapes as a single run only
    if it stays whole. Prose outside the ``$...$`` spans stays upright at
    the baseline; inside them variables are italicised and ``_``/``^``
    scripts are dropped or raised at reduced size. Radicals keep the house
    spelling ``√(...)``; there are no commands, every glyph is literal,
    and a backslash in a math run is an error -- the LaTeX commands of
    the matplotlib figures do not exist here.

    Script policy: inside a ``_``/``^`` script, letters are indices and are
    set in italic (``$K_{ij}$``, ``$η_{ij}$``), except the descriptive
    subscripts curated in :data:`_ROMAN_SCRIPTS`, which are abbreviations
    of words and stay upright as the standards print them (``$L_{Aeq}$``,
    ``$f_{max}$``). At the baseline the opposite rule holds: a run of two
    or more Latin letters is an operator name or an acronym (log, grad,
    CN, TL) and stays upright, single letters are italic variables. Greek
    letters split by case at every level: lowercase are italic variables
    (``$θ$``, ``$η_{ij}$``) and capitals are upright (``$ΔL_s$``,
    ``$Δ_{SOR}$``, ``$Φ$``) -- in this corpus a capital Greek letter is an
    operator or a descriptor, matching the roman Δ of difference of ISO
    80000-2, the ``Δ_SOR`` print of ECAC Doc 29 §4.5.7 and the upright
    ``\\Delta`` matplotlib's mathtext sets in the figures. The
    grid steps ``dx``/``dt`` follow that baseline rule inside a formula
    (upright, per the roman d of ISO 80000-2); in plain prose ("dt from
    the Courant number") they are not mathematics and take no ``$...$``.
    """
    segments = s.split("$")
    if len(segments) % 2 == 0:
        raise ValueError(f"unbalanced $ markup in {s!r}")
    chunks: list[tuple[str, bool, float, float]] = []

    def add(text: str, italic: bool = False, shift: float = 0.0,
            scale: float = 1.0) -> None:
        if not text:
            return
        if chunks and chunks[-1][1:] == (italic, shift, scale):
            chunks[-1] = (chunks[-1][0] + text, italic, shift, scale)
        else:
            chunks.append((text, italic, shift, scale))

    for k, segment in enumerate(segments):
        if k % 2 == 0:
            add(segment)
            continue
        if "\\" in segment:
            raise ValueError(
                f"backslash in math run {segment!r} of {s!r}: there are no "
                "commands here, write the glyph itself (θ, √, ·, …)"
            )
        for kind, payload in _math_tokens(segment, s):
            if kind in ("var", "up"):
                add(payload, italic=kind == "var")
            else:
                shift = _SUB_DROP if kind == "sub" else _SUP_RISE
                for kind2, payload2 in _math_tokens(payload, s, script=True):
                    add(payload2, italic=kind2 == "var", shift=shift,
                        scale=_SCRIPT_SCALE)
    return chunks


def _math_spans(s: str, size: int) -> str:
    """Serialise the runs of :func:`_math_runs` into ``<tspan>`` markup.

    Baseline shifts become cumulative rounded ``dy`` steps, script runs
    carry their reduced ``font-size`` and italic runs their
    ``font-style``. No tspan carries an ``x`` of its own, so the whole
    string remains one text chunk and ``text-anchor`` keeps working. The
    caller must put ``xml:space="preserve"`` on the wrapping ``<text>``:
    without it librsvg collapses every space at a tspan boundary, even
    NBSP.
    """
    parts: list[str] = []
    baseline = 0.0
    for text, italic, shift, scale in _math_runs(s):
        attrs: list[str] = []
        dy = (shift - baseline) * size
        baseline = shift
        if abs(dy) > 1e-9:
            attrs.append(f' dy="{dy:.1f}"')
        if scale != 1.0:
            attrs.append(f' font-size="{size * scale:.1f}"')
        if italic:
            attrs.append(' font-style="italic"')
        parts.append(f'<tspan{"".join(attrs)}>{_esc(text)}</tspan>')
    return "".join(parts)


class SVG:
    """Tiny element accumulator with technical-drawing helpers."""

    def __init__(self, width: int, height: int, th: Theme, lang: str = "en") -> None:
        self.w, self.h, self.th = width, height, th
        self.lang = lang
        self.parts: list[str] = []

    def tr(self, s: str) -> str:
        """Translate a user-visible string for the current language."""
        return lookup(s, translate=self.lang == "es")

    # -- primitives -------------------------------------------------------
    def add(self, fragment: str) -> None:
        self.parts.append(fragment)

    def rect(self, x: float, y: float, w: float, h: float, fill: str,
             stroke: str = "none", rx: float = 0.0, sw: float = 1.5,
             dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str,
             sw: float = 1.5, dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d} stroke-linecap="round"/>')

    def circle(self, cx: float, cy: float, r: float, fill: str,
               stroke: str = "none", sw: float = 1.5) -> None:
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def ellipse(self, cx: float, cy: float, rx: float, ry: float,
                fill: str = "none", stroke: str = "none", sw: float = 1.5,
                dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x: float, y: float, s: str, size: int = 20,
             fill: str = "", anchor: str = "middle", bold: bool = False,
             mono: bool = False, italic: bool = False) -> None:
        s = self.tr(s)
        self.add(self._emit_text(x, y, s, size, fill or self.th.fg, anchor,
                                 bold=bold, mono=mono, italic=italic))

    def _emit_text(self, x: float, y: float, s: str, size: int, fill: str,
                   anchor: str, *, bold: bool = False, mono: bool = False,
                   italic: bool = False) -> str:
        """The emission core of :meth:`text`: one already-translated label.

        The caller translates; this composes and serialises. Keeping the
        two apart is what lets the title of :meth:`render` share the exact
        same emission as every body label.
        """
        w = ' font-weight="600"' if bold else ""
        if "$" in s:
            if mono:
                raise ValueError(
                    f"mono cannot carry composed mathematics: {s!r} would "
                    "drop its $...$ styling; write the label without mono "
                    "or without markup"
                )
            # Composed mathematics: ``italic`` does not apply, the markup
            # styles every glyph run itself (see _math_runs).
            return (f'<text x="{x}" y="{y}" font-family="{_FONT}" '
                    f'font-size="{size}" fill="{fill}" '
                    f'text-anchor="{anchor}"{w} xml:space="preserve">'
                    f'{_math_spans(s, size)}</text>')
        i = ' font-style="italic"' if italic else ""
        fam = _MONO if mono else _FONT
        return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
                f'fill="{fill}" text-anchor="{anchor}"{w}{i}>{_esc(s)}</text>')

    def path(self, d: str, fill: str = "none", stroke: str = "none",
             sw: float = 1.5, dash: str = "") -> None:
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}" stroke-linejoin="round"{dd}/>')

    # -- technical helpers -------------------------------------------------
    def arrow(self, x1: float, y1: float, x2: float, y2: float, stroke: str,
              sw: float = 1.6) -> None:
        """Straight arrow with a filled head at (x2, y2)."""
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        L, W = 9.0, 3.6
        bx, by = x2 - L * math.cos(ang), y2 - L * math.sin(ang)
        px, py = -math.sin(ang), math.cos(ang)
        self.line(x1, y1, bx, by, stroke, sw)
        self.path(f"M {x2:.1f} {y2:.1f} L {bx + W * px:.1f} {by + W * py:.1f} "
                  f"L {bx - W * px:.1f} {by - W * py:.1f} Z", fill=stroke)

    def dim(self, x1: float, y1: float, x2: float, y2: float, label: str,
            offset: float = 0.0, size: int = 18, label_side: str = "left") -> None:
        """Dimension between two measured points, drafting style.

        The dimension line is placed ``offset`` px away (perpendicular);
        dashed witness lines connect it to the measured points. With
        ``offset=0`` the caller is responsible for any witness lines.
        """
        th = self.th
        horizontal = abs(y2 - y1) < abs(x2 - x1)
        if horizontal:
            y = y1 + offset
            if offset:
                self.line(x1, y1, x1, y, th.muted, 0.9, dash="3,3")
                self.line(x2, y2, x2, y, th.muted, 0.9, dash="3,3")
            mid = (x1 + x2) / 2
            self.arrow(mid - 4, y, x1, y, th.muted, 1.2)
            self.arrow(mid + 4, y, x2, y, th.muted, 1.2)
            self.text(mid, y - 7, label, size, th.fg, "middle")
        else:
            x = x1 + offset
            if offset:
                self.line(x1, y1, x, y1, th.muted, 0.9, dash="3,3")
                self.line(x2, y2, x, y2, th.muted, 0.9, dash="3,3")
            mid = (y1 + y2) / 2
            self.arrow(x, mid - 4, x, y1, th.muted, 1.2)
            self.arrow(x, mid + 4, x, y2, th.muted, 1.2)
            # Label beside the line, on whichever side is clear of the
            # measured object (masts, people, furniture).
            if label_side == "right":
                self.text(x + 9, mid + 6, label, size, th.fg, "start")
            else:
                self.text(x - 9, mid + 6, label, size, th.fg, "end")

    def mic(self, x: float, capsule_top: float, ground: float,
            scale: float = 1.0) -> None:
        """Measurement microphone on a stand that reaches the ground.

        ``capsule_top`` is the y of the capsule tip (the measurement point).
        """
        th, s = self.th, scale
        cap_h, body_h = 12 * s, 34 * s
        self.rect(x - 4 * s, capsule_top, 8 * s, cap_h, th.fg, rx=2.5 * s)
        self.rect(x - 6 * s, capsule_top + cap_h, 12 * s, body_h, th.primary, rx=4 * s)
        self.line(x, capsule_top + cap_h + body_h, x, ground, th.fg, 2.2)
        self.line(x - 16 * s, ground, x + 16 * s, ground, th.fg, 2.2)

    def person(self, x: float, y: float, h: float = 90.0, seated: bool = False) -> None:
        """Simple engineering-style human silhouette; (x, y) = feet."""
        th = self.th
        r = h * 0.10
        if not seated:
            self.circle(x, y - h + r, r, th.muted)
            self.line(x, y - h + 2 * r, x, y - h * 0.35, th.muted, 3)
            self.line(x, y - h * 0.75, x - h * 0.18, y - h * 0.5, th.muted, 2.4)
            self.line(x, y - h * 0.75, x + h * 0.18, y - h * 0.5, th.muted, 2.4)
            self.line(x, y - h * 0.35, x - h * 0.13, y, th.muted, 2.4)
            self.line(x, y - h * 0.35, x + h * 0.13, y, th.muted, 2.4)
        else:
            self.circle(x, y - h + r, r, th.muted)
            self.line(x, y - h + 2 * r, x, y - h * 0.45, th.muted, 3)       # torso
            self.line(x, y - h * 0.45, x + h * 0.30, y - h * 0.45, th.muted, 2.4)  # thigh
            self.line(x + h * 0.30, y - h * 0.45, x + h * 0.30, y, th.muted, 2.4)  # shin
            self.line(x, y - h * 0.70, x + h * 0.22, y - h * 0.55, th.muted, 2.4)  # arm

    def ground(self, y: float, x1: float, x2: float, hatch: int = 24) -> None:
        th = self.th
        self.line(x1, y, x2, y, th.fg, 2.2)
        x = x1
        while x < x2:
            self.line(x, y, x - 8, y + 9, th.muted, 1.1)
            x += hatch

    def render(self, title: str) -> str:
        th = self.th
        t = self.tr(title)
        body = (f' xml:space="preserve">{_math_spans(t, _TITLE_SIZE)}'
                if "$" in t else f'>{_esc(t)}')
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'<rect width="{self.w}" height="{self.h}" fill="{th.bg}"/>'
                f'<text x="{self.w / 2}" y="30" font-family="{_FONT}" '
                f'font-size="{_TITLE_SIZE}" font-weight="600" fill="{th.fg}" '
                f'text-anchor="middle"{body}</text>')
        return head + "".join(self.parts) + "</svg>"


def _write(output_dir: str, name: str, build: Callable[[SVG, Theme], None], title: str,
           height: int = 560) -> None:
    for lang, lang_suffix in (("en", ""), ("es", "_es")):
        for th in (LIGHT, DARK):
            visit(name, lang)
            svg = SVG(900, height, th, lang)
            build(svg, th)
            path = os.path.join(output_dir, f"{name}{lang_suffix}{th.suffix}.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg.render(title))
    print(f"Generated {name}.svg (+dark, +es, +es_dark)")
