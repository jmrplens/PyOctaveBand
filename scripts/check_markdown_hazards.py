#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Gate for markdown that does not render the way it reads.

The guides are hard-wrapped, so a sentence long enough to wrap can put a
``-``, a ``>`` or a digit-and-dot at the start of a line. CommonMark does not
read that as the middle of a sentence: it reads it as a list item, a block
quote or an ordered list, and it ends the paragraph there. The author sees a
paragraph; the reader gets two blocks, or worse.

Three checks. The first two are the same defect seen twice, and they are not
equally visible; the third is a different one with the same shape, a source
that looks right and renders wrong.

1. **Unclosed inline maths.** An inline ``$...$`` that wraps onto a block
   marker never closes: the marker ends the paragraph first. The maths is then
   published as literal text, and in MDX the subscript braces become a
   JavaScript expression, so the page does not render at all. This is what
   ``$L_{n,ij,w} = ... - \\Delta R_{j,w}`` followed by ``- K_{ij} ...`` did:
   the site build failed with ``ReferenceError: n is not defined``, ``n`` being
   the first subscript. A list marker is only reported when it interrupts an
   open ``$``, because a list that legitimately follows its introducing line
   without a blank line is ordinary in this corpus (3830 of them) and is not a
   defect.

2. **A wrapped comparison operator.** ``>`` at the start of a line, where the
   line before it is ordinary paragraph text, is a greater-than sign that
   wrapped, not a quotation: it turned the tail of a sentence into a quoted
   block. This one is silent. Nothing renders wrong, nothing fails to build,
   and it ships. Three were found this way, two of them the same sentence in
   two languages. A deliberate block quote in this corpus is preceded by a
   blank line, which is why the rule can be this simple.

3. **A same-page link to an accented heading.** Markdown percent-encodes
   non-ASCII in a URL, so ``[La medición](#la-medición-...)`` ships as
   ``href="#la-medici%C3%B3n-..."`` while the heading keeps its id unencoded.
   A browser resolves it; the accessibility audit compares the raw attribute
   against the ids and reports a dangling anchor, and it only samples 66 URLs,
   so five of the six the corpus had went unreported. The fix the corpus
   already used in two places is an ASCII ``<span id=...>`` above the heading.

Display maths (``$$``) is exempt: it is a block of its own and the delimiters
are on their own lines.

Usage::

    python scripts/check_markdown_hazards.py

Exit status 0 when every page is clean, 1 otherwise, with the file, the line
and the offending text.
"""

from __future__ import annotations

import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Trees of hand-written markdown. The API reference is generated from the
#: docstrings and the plans folder is local scratch, so neither is checked.
_ROOTS = (
    _ROOT / "site" / "src" / "content" / "docs",
    _ROOT / "docs",
)
_SKIP = ("reference/api/", "superpowers/")

#: A line that opens a new CommonMark block rather than continuing a paragraph.
_BLOCK = re.compile(r"^\s{0,3}(?:[-*+]\s|>|#{1,6}\s|\d{1,9}[.)]\s|\||```)")

#: ``>`` opening a line with content after it: a wrapped comparison operator,
#: unless the paragraph deliberately starts a quotation, which is always
#: preceded by a blank line.
_QUOTE = re.compile(r"^>\s*\S")

#: Text that continues a paragraph rather than starting a block of its own.
_PROSE = re.compile(r"^\s{0,3}[^\s\-*+>#|<:0-9]")

#: A same-page link whose target carries a character markdown will
#: percent-encode on the way out.
_ANCHOR = re.compile(r"\]\(#([^)]+)\)")


def _unescaped_dollars(line: str) -> int:
    """Inline ``$`` delimiters on a line, ignoring ``\\$`` and ``$$``."""
    return len(re.findall(r"(?<!\\)\$", re.sub(r"\$\$", "", line)))


def _check(path: pathlib.Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rel = path.relative_to(_ROOT).as_posix()
    problems: list[str] = []
    fenced = False
    open_math = False

    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            open_math = False
            continue
        if fenced:
            continue

        if open_math and _BLOCK.match(line):
            problems.append(
                f"{rel}:{number}: inline maths opened on line {number - 1} is cut off by a "
                f"block marker; rewrap so the line does not start with "
                f"{line.lstrip()[:2]!r}"
            )
            open_math = False

        for anchor in _ANCHOR.finditer(line):
            target = anchor.group(1)
            if not target.isascii():
                problems.append(
                    f"{rel}:{number}: the same-page link "
                    f"'#{target}' will be percent-encoded in the href, and the accessibility "
                    f"audit compares the raw attribute against the ids, so it reads as a "
                    f"dangling anchor. Put <span id=\"ascii-slug\"></span> above the heading "
                    f"and link to that, as the pages that already hit this do."
                )

        if number > 1 and _QUOTE.match(line) and _PROSE.match(lines[number - 2]):
            problems.append(
                f"{rel}:{number}: {line.strip()[:40]!r} continues the sentence above but "
                f"starts a block quote; rewrap so '>' does not start the line"
            )

        if not line.strip() or _BLOCK.match(line):
            open_math = False
        if _unescaped_dollars(line) % 2 == 1:
            open_math = not open_math

    return problems


def main() -> int:
    problems: list[str] = []
    checked = 0
    for root in _ROOTS:
        for path in sorted(root.rglob("*.md*")):
            if any(part in path.as_posix() for part in _SKIP):
                continue
            checked += 1
            problems.extend(_check(path))

    if problems:
        print("::error::markdown that does not render the way it reads", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"Markdown renders the way it reads: {checked} pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
