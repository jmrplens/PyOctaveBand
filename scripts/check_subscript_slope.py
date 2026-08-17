#!/usr/bin/env python3
#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Gate for a subscript that means two things on one page.

ISO 80000-2 settles the slope of a subscript by what the subscript *is*: a
quantity symbol stays italic (:math:`L_p`, pressure; :math:`L_W`, power), a
word or an abbreviation is upright (:math:`R_\mathrm{w}`, weighted), a number
is upright by definition, and an index that runs over a sum is italic. The
corpus applies that rule, and CONTRIBUTING.md states it.

What the rule does not settle is the glyph that takes both slopes honestly.
:math:`D_z` is the ISO 9613-2 barrier screening, whose *z* is the difference
between the diffracted and direct path lengths -- a quantity, italic -- and it
is the ISO 2631-5 acceleration dose, whose *z* names a direction -- upright.
Both documents print ``D_z``. Neither spelling is ours to change, so no single
corpus-wide slope for that pair can be right: one of the two modules would be
wrong by construction. Twenty pairs behave this way here.

So the unit of decision is the file, not the glyph. Each module opens with the
standard it implements and each guide with the standard it explains, so inside
one file a subscript has one meaning and therefore one slope. That is the
invariant this checks: **no file writes the same (base, subscript) pair both
ways.** Across files it says nothing, deliberately -- that is where the
legitimate duals live.

What a failure means, in order of how often it is the answer
------------------------------------------------------------

1. **One of the two is an index and can be re-lettered.** A subscript that
   runs over a sum is a bound variable: the letter is ours and means nothing
   outside the formula, while the colliding quantity's letter belongs to the
   standard that defines it. ``insulation.py`` energy-averages positions with
   :math:`\sum_i 10^{L_i/10}` and states ISO 16283-2's impact sound pressure
   level as :math:`L_\mathrm{i}`, on the same page. The index moved to *j*;
   the quantity could not move at all.
2. **One of the two lags.** The file already writes the settled slope
   elsewhere and one occurrence -- typically inside a plotting snippet, where
   the mathematics is a string inside a string -- was missed. Set it the way
   its neighbours are set.
3. **The page really does carry both meanings**, which is what a glossary is
   for. Register it in :data:`DECLARED`, naming both meanings. That list is
   meant to stay short: it is the escape hatch for a page whose subject is the
   collision, not a way to land one.

What this cannot check
----------------------

Whether the slope a file chose is the *right* one for its standard. That is a
reading of a source document, and no script does it. This gate only keeps a
file from asserting two answers at once, which is the failure a reader can see
without leaving the page.

Exit status 0 when every file is single-valued, 1 otherwise.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

#: Where the prose lives: the module docstrings that become the API reference,
#: and the three editions of the guides.
DEFAULT_ROOTS = (
    "src/phonometry",
    "docs",
    "site/src/content/docs",
)

#: Suffixes carrying mathematics. ``.py`` for the docstrings, the rest for the
#: guides, whose plotting snippets carry labels of their own.
_SUFFIXES = {".py", ".md", ".mdx"}

#: What the file scope does not apply to.
#:
#: The errata registry transcribes what a published page prints, so its slopes
#: are the source's and not this project's to reconcile.
#:
#: The drawing modules are filed by domain rather than by standard -- one
#: ``_plot`` module holds the figures of a dozen of them -- so a file there is
#: not the scope in which a letter has one meaning; the guide that embeds the
#: figure is, and that is read here. The diagram plates are further out still:
#: ``scripts/diagrams/canvas.py`` derives the slope from the letter run alone,
#: so their sources never write one.
_EXCLUDED = re.compile(r"errata|/_plot/|/_report/", re.IGNORECASE)

#: Pages that carry both meanings on purpose, with the reason. Keyed on the
#: repository-relative path and the ``(base, subscript)`` pair; the path is
#: matched against the tail of the file's own, so the gate declares the same
#: page whether it is run from the repository root or given absolute roots.
DECLARED: dict[str, dict[tuple[str, str], str]] = {
    "docs/reference/glossary.md": {
        ("L", "N"): (
            "the ambiguity table's own subject: the percentile level, whose "
            "N is the percentage, against the sonar noise level of "
            "ISO 18405, whose N is the noise"
        ),
        ("L", "s"): (
            "likewise: the mean of the two bands adjacent to a candidate "
            "tone, whose s is unexpanded anywhere in this corpus, against "
            "the equivalent monopole source level of ISO 17208-2"
        ),
    },
}


def math_regions(text: str, suffix: str) -> list[tuple[int, str]]:
    r"""``(offset, snippet)`` for every mathematics region of *text*.

    ``$$...$$`` and ``$...$`` in every file kind, plus the reStructuredText
    ``:math:`...``` roles and ``.. math::`` blocks the docstrings use.
    """
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        out.append((m.start(1), m.group(1)))
    for m in re.finditer(r"(?<![$\\])\$(?!\$)([^$\n]+?)\$(?!\$)", text):
        out.append((m.start(1), m.group(1)))
    if suffix == ".py":
        for m in re.finditer(r":math:`(.+?)`", text, re.DOTALL):
            out.append((m.start(1), m.group(1)))
        for m in re.finditer(r"^([ \t]*)\.\. math::[ \t]*\n(.*?)(?=\n\s*\n|\Z)",
                             text, re.DOTALL | re.MULTILINE):
            out.append((m.start(2), m.group(2)))
    return out


#: A run that is a regex or its replacement rather than a label, by the same
#: shapes ``check_mathtext.py`` recognises: a doubled backslash means "a
#: literal backslash" in a pattern, and a capture group or a backreference is
#: not mathematics. The Spanish translation tables are made of these.
_PATTERN_SHAPE = re.compile(r"\\\\|\(\.\+\)|\(\\d|\(\.\*\)|\\\d")

#: ``base_subscript``: a single letter or a Greek command, an optional prime,
#: then the subscript, braced or bare. The prime stays with the base, so the
#: apparent quantity :math:`R'` and the quantity :math:`R` are two symbols.
_SUBSCRIPTED = re.compile(
    r"(\\[A-Za-z]+|[A-Za-z])('*)"
    r"_(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\\mathrm\{[^{}]*\}|\\text\{[^{}]*\}|.)"
)

#: A subscript, or one component of one, set upright the two ways this corpus
#: writes one. The wrapper may hold the whole comma-separated run
#: (``L_\mathrm{n,w,eq}``) or one component of it (``\alpha_{\mathrm{s},i}``).
_UPRIGHT = re.compile(r"^\\(?:mathrm|text)\{([^{}]*)\}$")

#: A bare letter, which inside mathematics is italic.
_ITALIC = re.compile(r"^[A-Za-z]$")

#: Where a symbol was written, slope by slope: ``{(base, sub): {slope: lines}}``.
Sightings = dict[tuple[str, str], dict[str, list[int]]]


def sightings(text: str, suffix: str) -> Sightings:
    """Every one-letter subscript of *text*, by symbol and by slope.

    A comma-separated subscript is read component by component, since that is
    how a reader reads it: :math:`\\alpha_{\\mathrm{s},i}` is the Sabine
    absorption of surface *i*, one abbreviation and one index, and the letter
    that matters is the one being compared. Where the upright command wraps the
    whole run, as ``L_\\mathrm{n,w,eq}`` does, every component of it is upright.
    Any component that is not a single letter is passed over: a digit, a
    weighting prefix and a word are not what this is about.
    """
    found: Sightings = collections.defaultdict(lambda: collections.defaultdict(list))
    for offset, region in math_regions(text, suffix):
        if _PATTERN_SHAPE.search(region):
            continue
        line = text.count("\n", 0, offset) + 1
        for match in _SUBSCRIPTED.finditer(region):
            base, prime, raw = match.group(1), match.group(2), match.group(3)
            sub = raw[1:-1] if raw.startswith("{") else raw
            whole = _UPRIGHT.match(sub.strip())
            body = whole.group(1) if whole else sub
            for part in (p.strip() for p in body.split(",")):
                upright = _UPRIGHT.match(part)
                if upright and _ITALIC.match(upright.group(1)):
                    found[(base + prime, upright.group(1))]["upright"].append(line)
                elif _ITALIC.match(part):
                    slope = "upright" if whole else "italic"
                    found[(base + prime, part)][slope].append(line)
    return found


def collect(roots: list[str]) -> list[pathlib.Path]:
    """The files to read, in a stable order."""
    paths: list[pathlib.Path] = []
    for root in roots:
        target = pathlib.Path(root)
        if target.is_file():
            paths.append(target)
            continue
        paths.extend(p for p in sorted(target.rglob("*"))
                     if p.suffix in _SUFFIXES and p.is_file())
    return [p for p in paths if not _EXCLUDED.search(str(p))]


def declared(path: pathlib.Path) -> dict[tuple[str, str], str]:
    """The symbols *path* is registered as carrying both ways, if any."""
    posix = path.as_posix()
    for key, symbols in DECLARED.items():
        if posix == key or posix.endswith("/" + key):
            return symbols
    return {}


def check(paths: list[pathlib.Path]) -> tuple[int, list[str]]:
    """``(files read, one report per undeclared collision)``."""
    failures: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        registered = declared(path)
        for (base, sub), slopes in sorted(sightings(text, path.suffix).items()):
            if len(slopes) < 2 or (base, sub) in registered:
                continue
            italic = ", ".join(str(n) for n in sorted(set(slopes["italic"])))
            upright = ", ".join(str(n) for n in sorted(set(slopes["upright"])))
            failures.append(
                f"  {path}\n"
                f"      {base}_{sub}: italic on line {italic}\n"
                f"      {' ' * len(base)}  upright on line {upright}"
            )
    return len(paths), failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS),
                        help="files or directories to scan (default: the prose)")
    args = parser.parse_args(argv)

    paths = collect(args.roots)
    read, failures = check(paths)
    if not failures:
        print(f"Every subscript is single-valued in each of {read} files.")
        return 0

    print(f"{len(failures)} subscripts carry both slopes inside one file:\n",
          file=sys.stderr)
    for failure in failures:
        print(failure, file=sys.stderr)
    print("\nRe-letter the index, set the lagging one the way its neighbours "
          "are set, or\ndeclare the page in DECLARED naming both meanings.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
