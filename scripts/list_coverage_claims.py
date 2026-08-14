#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Enumerate every scope claim the documentation makes.

Every guide ends its argument with a section declaring what the library
covers and what it deliberately does not: ``## What this guide covers`` on a
guide, ``## What this section does not cover`` on a section overview, and
their Spanish twins. Those sentences are promises. A promise nobody can list
is a promise nobody can check, and until the claims were marked up they were
ordinary prose: to answer "does the corpus still say we do not read WAV
files?" you had to read the corpus.

This walks ``site/src/content/docs/`` and prints one line per claim: the
file, the language, whether the claim is a *covered* or a *not-covered* one,
and the sentence it opens with. Two jobs follow from that list and neither is
this script's:

* **Audit.** Every ``covered`` claim asserts something about the library and
  can be held against the source, the tests and the conformance report. Every
  ``not-covered`` claim asserts an absence, which is the cheaper thing to
  check and the easier thing to get wrong: a boundary drawn once and quietly
  crossed two releases later is exactly the defect this list makes findable.
* **Backlog.** The ``not-covered`` half is a census of deliberate gaps, each
  with the reasoning for the gap attached. It is the most honest feature
  backlog the project has, because every entry was written by someone who had
  just decided not to do the work and said why.

THE MARKUP IS A CONTRACT, and this script is the proof that it parses. The
authored form is::

    <Scope>
    <ScopeClaim status="not-covered" label="Not covered">
    prose, links and all
    </ScopeClaim>
    </Scope>

and the rendered form of the same entry is an ``<li data-coverage=
"not-covered">`` inside a ``<ul data-scope="coverage">`` (see
site/src/components/Scope.astro). Both are read here, by the same parser: the
authored ``status`` attribute and the rendered ``data-coverage`` attribute
carry the same two values and neither may change without changing this file.
Reading the built HTML is what ``--html`` is for, and it is a stronger check
than reading the source, because it also proves the component emitted what
the page asked for.

Language comes from the path (``es/`` prefix) and not from the prose, which
is the whole reason ``status`` is an English keyword rather than the visible
legend: the Spanish pages are separate files carrying their own translated
label, and the keyword is what joins a claim to its twin.

Exit codes: 0 when the census was produced, 1 when a claim is malformed (a
missing or unknown ``status``, a claim outside a panel, an unclosed element),
2 for a usage error.

Usage::

    python scripts/list_coverage_claims.py
    python scripts/list_coverage_claims.py --status not-covered --lang en
    python scripts/list_coverage_claims.py --format json
    python scripts/list_coverage_claims.py --html site/dist
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "site" / "src" / "content" / "docs"

#: The two values `status` and `data-coverage` may take. Anything else is a
#: defect: a claim that does not answer "which side of the boundary" is not a
#: claim, and silently dropping it would make the census lie by omission.
STATUSES = ("covered", "not-covered")

#: How the panel and its entries are named in each of the two forms this
#: reads. The authored MDX names them after the components; the built HTML
#: names them after the elements those components emit.
SOURCE_MARKUP = {"panel": "scope", "claim": "scopeclaim", "status": "status"}
HTML_MARKUP = {"panel": "ul", "claim": "li", "status": "data-coverage"}

#: Enough of the opening prose to recognise a claim in a terminal. The whole
#: argument is in the file; this is an index, not a replacement for it.
SUMMARY_CHARS = 90

#: Markup that survives into the summary if it is not removed: the bold lead
#: an overview claim opens with, inline code and links. The underscore is
#: deliberately left alone -- in this corpus it is far more often part of an
#: identifier (`verify_filter_class`, $K_{ij}$) than an emphasis marker, and
#: stripping it turned function names into prose.
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_INLINE = re.compile(r"[*`]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Claim:
    """One scope declaration, as printed."""

    path: str
    lang: str
    status: str
    label: str
    summary: str


class _ScopeParser(HTMLParser):
    """Pull the claims out of one page.

    HTMLParser rather than a regex, and the same parser for both forms,
    because the authored MDX tags are spelled like HTML tags and the built
    HTML is HTML. It also gives the nesting for free, which is what catches
    the two structural defects worth catching: a claim outside a panel, and a
    panel that never closes.
    """

    def __init__(self, markup: dict[str, str], path: str) -> None:
        super().__init__(convert_charrefs=True)
        self._markup = markup
        self._path = path
        self._depth = 0
        self._open: dict[str, str] | None = None
        self._text: list[str] = []
        self._legend: list[str] | None = None
        self.claims: list[tuple[dict[str, str], str]] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: (value or "") for key, value in attrs}
        if tag == self._markup["panel"] and self._is_panel(values):
            self._depth += 1
            return
        # In the built form the legend is an element rather than an attribute,
        # so it has to be caught here or its words run into the declaration:
        # the census read "Not coveredParts 4 and 5 of the ISO 10846 series".
        if self._open is not None and "scope-legend" in values.get("class", "").split():
            self._legend = []
            return
        if tag != self._markup["claim"]:
            return
        if self._markup["status"] not in values:
            # An <li> of an ordinary list inside a claim's prose, in the HTML
            # form. Not a claim, and not a defect.
            if self._markup is HTML_MARKUP:
                return
            self.problems.append(f"{self._path}: <ScopeClaim> without a status")
            return
        if self._depth == 0:
            self.problems.append(
                f"{self._path}: a claim ({values[self._markup['status']]}) sits outside "
                f"any panel, so nothing says which page section declares it"
            )
        self._open = values
        self._text = []
        self._legend = None

    def _is_panel(self, values: dict[str, str]) -> bool:
        return self._markup is SOURCE_MARKUP or values.get("data-scope") == "coverage"

    def handle_endtag(self, tag: str) -> None:
        if self._legend is not None:
            assert self._open is not None
            self._open.setdefault("label", "".join(self._legend).strip())
            self._legend = None
            return
        if tag == self._markup["panel"] and self._depth:
            self._depth -= 1
        elif tag == self._markup["claim"] and self._open is not None:
            self.claims.append((self._open, "".join(self._text)))
            self._open = None

    def handle_data(self, data: str) -> None:
        if self._legend is not None:
            self._legend.append(data)
        elif self._open is not None:
            self._text.append(data)

    def finish(self) -> None:
        if self._open is not None:
            self.problems.append(f"{self._path}: a claim is never closed")
        if self._depth:
            self.problems.append(f"{self._path}: a scope panel is never closed")


def _summary(prose: str) -> str:
    """The opening of a claim, with the markdown taken off."""
    text = _WHITESPACE.sub(" ", _INLINE.sub("", _LINK.sub(r"\1", prose))).strip()
    if len(text) <= SUMMARY_CHARS:
        return text
    return text[: SUMMARY_CHARS - 1].rstrip() + "…"


def _language(relative: str) -> str:
    return "es" if relative.startswith("es/") else "en"


def _claims(path: Path, base: Path, markup: dict[str, str]) -> tuple[list[Claim], list[str]]:
    relative = path.relative_to(base).as_posix()
    parser = _ScopeParser(markup, relative)
    parser.feed(path.read_text(encoding="utf-8"))
    parser.finish()

    claims: list[Claim] = []
    for values, prose in parser.claims:
        status = values[markup["status"]]
        if status not in STATUSES:
            parser.problems.append(
                f"{relative}: {status!r} is not one of {', '.join(STATUSES)}"
            )
            continue
        claims.append(
            Claim(
                path=relative,
                lang=_language(relative),
                status=status,
                label=values.get("label", ""),
                summary=_summary(prose),
            )
        )
    return claims, parser.problems


def collect(base: Path, pattern: str, markup: dict[str, str]) -> tuple[list[Claim], list[str]]:
    """Every claim under a tree, in path order."""
    claims: list[Claim] = []
    problems: list[str] = []
    for path in sorted(base.rglob(pattern)):
        found, trouble = _claims(path, base, markup)
        claims.extend(found)
        problems.extend(trouble)
    return claims, problems


def _print_table(claims: list[Claim]) -> None:
    for claim in claims:
        label = f" [{claim.label}]" if claim.label else ""
        print(f"{claim.lang}\t{claim.status}\t{claim.path}{label}\t{claim.summary}")


def _print_census(claims: list[Claim]) -> None:
    pages = len({claim.path for claim in claims})
    covered = sum(claim.status == "covered" for claim in claims)
    english = sum(claim.lang == "en" for claim in claims)
    print(
        f"\n{len(claims)} coverage claims over {pages} pages: "
        f"{covered} covered, {len(claims) - covered} not covered; "
        f"{english} English, {len(claims) - english} Spanish."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--html",
        metavar="DIR",
        type=Path,
        help="read a built site (site/dist) instead of the authored MDX, which "
        "also proves the components emitted what the pages asked for",
    )
    parser.add_argument("--status", choices=STATUSES, help="only claims of this side")
    parser.add_argument("--lang", choices=("en", "es"), help="only claims in this language")
    parser.add_argument(
        "--format", choices=("table", "json"), default="table", help="output shape"
    )
    args = parser.parse_args()

    if args.html is not None:
        base, pattern, markup = args.html, "*.html", HTML_MARKUP
    else:
        base, pattern, markup = CONTENT, "*.md*", SOURCE_MARKUP
    if not base.is_dir():
        print(f"no such directory: {base}", file=sys.stderr)
        return 2

    claims, problems = collect(base, pattern, markup)
    if args.status:
        claims = [claim for claim in claims if claim.status == args.status]
    if args.lang:
        claims = [claim for claim in claims if claim.lang == args.lang]

    if args.format == "json":
        print(json.dumps([asdict(claim) for claim in claims], ensure_ascii=False, indent=2))
    else:
        _print_table(claims)
        _print_census(claims)

    if problems:
        print("::error::coverage claims that do not parse", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
