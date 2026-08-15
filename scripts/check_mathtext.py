#!/usr/bin/env python3
"""Parse every ``$...$`` label the generators draw, and name what matplotlib refuses.

A label mathtext cannot parse does not degrade into a worse label: generation
raises and the figure is never written. In a run of four hundred figures that
traceback scrolls past, and the gate that notices is ``check_figures.py``, which
reports the figure as *stale* -- a description that sends the reader looking for
a data change rather than a typo three characters wide.

The reason this is worth its own check, rather than trusting the render: a
construct can be legal alone and illegal in combination. ``^{\\prime}`` is a
perfectly good prime, and ``^{\\prime}^{\\prime}`` is a double superscript
matplotlib rejects outright -- which is exactly what a mechanical substitution
of ``′`` produces from a double prime, and exactly what it did produce in three
labels of this tree.

It reads the sources rather than rendering, so it costs seconds and covers
labels no example happens to exercise.

What it deliberately does not judge
-----------------------------------

Three kinds of string carry a dollar without being a label:

* the **pattern table** (``_ES_PATTERNS``), whose entries are regexes that MATCH
  a drawn label; a doubled backslash means "a literal backslash" there, and a
  capture group is not mathematics;
* the **replacement** half of the same pairs, which carries backreferences;
* the shell text of the GPU ffmpeg wrapper in ``figures/media.py``, where a
  dollar is a shell expansion.

Each is recognised by its own shape rather than by an exclusion list, so a new
file gets the same treatment without being registered anywhere.

Exit status 0 when every run parses, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

#: Where labels are drawn from. The plates compose their own maths with the SVG
#: canvas rather than mathtext, so they are not parsed here.
DEFAULT_ROOTS = ("scripts/figures", "src/phonometry/_plot", "src/phonometry/_report")

#: A regex, not a label: ``\\`` is regex-speak for a literal backslash.
_ESCAPED_BACKSLASH = "\\\\"
#: A capture group or a backreference; neither is mathematics.
_REGEX_SHAPE = re.compile(r"\(\.\+\)|\(\\d|\(\.\*\)|\\\d")


def maths_runs(text: str) -> list[str]:
    """Every ``$...$`` region of *text*, dollars included.

    A dollar escaped as ``\\$`` opens nothing: that is how a pattern spells a
    literal dollar it means to match.
    """
    runs: list[str] = []
    start: int | None = None
    for i, char in enumerate(text):
        if char != "$" or (i and text[i - 1] == "\\"):
            continue
        if start is None:
            start = i
        else:
            runs.append(text[start:i + 1])
            start = None
    return runs


def is_label(run: str) -> bool:
    """False for a run that is a regex, a replacement or shell text."""
    if _ESCAPED_BACKSLASH in run or _REGEX_SHAPE.search(run):
        return False
    # Shell expansion: ``${...}``, ``$(...)``, ``$#``, ``$@``.
    if re.match(r"^\$[{(#@]", run):
        return False
    # A double quote inside the dollars means two shell expansions on one line
    # were read as one maths run (``"$rid" "$out"``). Nothing sets a quotation
    # mark as mathematics.
    return '"' not in run


def check(paths: list[pathlib.Path]) -> tuple[int, list[tuple[str, int, str, str]]]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import mathtext

    parser = mathtext.MathTextParser("agg")
    checked = 0
    failures: list[tuple[str, int, str, str]] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_bytes())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            for run in maths_runs(node.value):
                if not is_label(run):
                    continue
                checked += 1
                try:
                    parser.parse(run, dpi=100, prop=None)
                except Exception as exc:  # noqa: BLE001
                    lines = [line for line in str(exc).splitlines() if line.strip()]
                    reason = lines[0] if lines else type(exc).__name__
                    failures.append((str(path), node.lineno, run, reason))
    return checked, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS),
                        help="files or directories to scan (default: the drawing modules)")
    args = parser.parse_args(argv)

    paths: list[pathlib.Path] = []
    for root in args.roots:
        target = pathlib.Path(root)
        if target.is_file():
            paths.append(target)
        else:
            paths.extend(sorted(target.rglob("*.py")))

    checked, failures = check(paths)
    if not failures:
        print(f"All {checked} mathtext labels in {len(paths)} files parse.")
        return 0

    print(f"{len(failures)} of {checked} mathtext labels do not parse:\n", file=sys.stderr)
    for path, line, run, reason in failures:
        print(f"  {path}:{line}", file=sys.stderr)
        print(f"      {run}", file=sys.stderr)
        print(f"      {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
