#!/usr/bin/env python3
"""Name the labels whose Spanish decimal the save-time comma pass will skip.

The Spanish edition of a figure is the English one with its text rewritten at
save time, and the last step of that rewrite is the decimal comma::

    if s and "$" not in s and re.search(r"\\d\\.\\d", s):
        artist.set_text(_decimal_comma(s))

The guard is deliberate -- a bare comma inside ``$...$`` sets as a maths
separator with the wrong spacing -- but it tests the WHOLE string. So a label
that carries mathematics ANYWHERE and a decimal number ANYWHERE ELSE keeps the
English point in its Spanish variant:

    "$L_\\mathrm{eq}$ = 52.4 dB"   ->  Spanish figure still prints 52.4

Nothing downstream can see it. The language gate compares what each pass could
not translate, and a number is not a word; the parity check sees a pair,
because the pair exists and is correct apart from one character; and a reader
only notices by opening the Spanish figure and looking at the digits.

This is the check that catches it, and it is a static one: it reads the drawn
string literals and reports any that mix a maths run with a decimal outside it.

Not every hit is a defect. A label whose number is interpolated through
``format_number(value, language)`` localises itself and needs no pass, so the
literal to look for is a HARD-CODED decimal: ``"= 52.4 dB"``, not ``"= {value}
dB"``. Placeholders are therefore ignored, and what is left is either a defect
or a decimal that is not a decimal (a clause number, a standard designation),
which is what the allowlist below is for.

Exit status 0 when nothing is at risk, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

DEFAULT_ROOTS = ("src/phonometry/_plot", "scripts/figures")

#: A decimal number: a digit, a point, a digit. Also the shape of a clause
#: number and of a standard designation, which is why the guards below exist.
_DECIMAL = re.compile(r"\d\.\d")

#: Numbers that are not measurements, and so keep their point in Spanish too: a
#: clause number ("5.3.3"), a standard's designation ("S3.5", "ISO 9613-2"), and
#: a pointer into a document -- an equation, a clause, a table, an example. The
#: last group is why this list is bilingual: a Spanish label writes "apartado
#: 8.2" and "Ec. 9.52", and neither number is a quantity.
_NOT_A_MEASUREMENT = re.compile(
    r"\d+\.\d+\.\d"                     # 5.3.3
    r"|[A-Z]\d*\.\d"                    # S3.5
    r"|(?:ISO|IEC|EN|UNE|ANSI|ASTM|AES|ITU|EBU|ECMA|DIN|BS)[\s\d.\-]*\d\.\d"
    r"|(?i:"
    r"Eq\.|Ec\.|Ecuaci[oó]n|Equation"
    r"|clause|apartado|cl[aá]usula|secci[oó]n|section"
    r"|cap[ií]tulo|chapter|annex|anexo"
    r"|table|tabla|figure|figura|fig\.|formula|f[oó]rmula"
    r"|example|ejemplo|note|nota"
    r")[\s\w&]*\d\.\d"
)

#: An f-string placeholder or a ``str.format`` field: the number is computed,
#: so the label's own text carries no decimal to lose.
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def maths_runs(text: str) -> list[tuple[int, int]]:
    """Index pairs of the ``$...$`` regions of *text*."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, char in enumerate(text):
        if char != "$" or (i and text[i - 1] == "\\"):
            continue
        if start is None:
            start = i
        else:
            runs.append((start, i))
            start = None
    return runs


def at_risk(text: str) -> str | None:
    """The decimal a Spanish figure would print with a point, or None."""
    runs = maths_runs(text)
    if not runs:
        return None  # no dollar: the comma pass runs normally
    # Blank out the maths and every placeholder, then look at what is left.
    outside = list(text)
    for lo, hi in runs:
        for i in range(lo, hi + 1):
            outside[i] = " "
    rest = "".join(outside)
    rest = _PLACEHOLDER.sub(" ", rest)
    for match in _DECIMAL.finditer(rest):
        window = rest[max(0, match.start() - 12):match.end() + 6]
        if _NOT_A_MEASUREMENT.search(window):
            continue
        return rest[max(0, match.start() - 20):match.end() + 12].strip()
    return None


def check(paths: list[pathlib.Path]) -> list[tuple[str, int, str, str]]:
    """Report a translation PAIR whose Spanish half still carries the point.

    Asking "does this string mix maths with a decimal" alone is too blunt: the
    answer is yes for every label whose number is fine because its Spanish twin
    already bakes the comma in, and yes again for an equation reference, whose
    number is not a measurement in either language.

    The defect has a sharper shape. The tables are ``{english: spanish}``
    literals, so the question is whether the SPANISH value keeps a decimal point
    the comma pass will not reach. That is a defect whatever the English says.
    """
    found: list[tuple[str, int, str, str]] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_bytes())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=False):
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    continue
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                risky = at_risk(value.value)
                if risky:
                    found.append((str(path), value.lineno, value.value[:80], risky))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    args = parser.parse_args(argv)

    paths: list[pathlib.Path] = []
    for root in args.roots:
        target = pathlib.Path(root)
        paths.extend([target] if target.is_file() else sorted(target.rglob("*.py")))

    found = check(paths)
    if not found:
        print(f"No label in {len(paths)} files mixes mathematics with a "
              f"hard-coded decimal.")
        return 0

    print(f"{len(found)} label(s) would keep an English decimal point in "
          f"Spanish:\n", file=sys.stderr)
    for path, line, text, risky in found:
        print(f"  {path}:{line}", file=sys.stderr)
        print(f"      {text!r}", file=sys.stderr)
        print(f"      the maths blocks the comma pass, so {risky!r} keeps its "
              f"point", file=sys.stderr)
    print("\nEither bake the comma into the Spanish value, or format the number "
          "through format_number(value, language), which localises itself.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
