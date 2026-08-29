#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Render the committed evidence documents as Starlight pages.

The documents that carry the project's numerical evidence live under ``docs/``
and are the authority for their own content. This script owns one of them:

* ``docs/ERRATA.md`` - the hand-maintained registry of defects found in the
  published sources the library implements from.

It used to own the conformance report too. That one is now structured data,
``docs/conformance.json``, and its pages render from it through a component
rather than carrying a copy of its text, so a transplant would be copying a
rendering of the same rows into a page that already has the rows.

The registry used to be reachable only on GitHub: the site merely described it
and linked out. This script transplants its body into the Starlight pages so the
site renders the evidence itself, without anyone ever hand-copying a table.

Each target page keeps its hand-written introduction and carries a marker
line; everything below the marker is replaced by the transplanted body on
every run. The transformation is purely textual and deterministic (no
timestamps, no environment-dependent text), so CI can diff a fresh run against
the committed pages exactly like it does for the generated API reference (see
the ``site-reports`` job in .github/workflows/python-app.yml).

The Spanish page carries the same generated body: the registry is generated
English text, and the Spanish prose above the marker says so.

Stdlib only. Regenerate with ``make site-reports``.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import posixpath
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CONTENT = ROOT / "site" / "src" / "content" / "docs"

#: Blob root used to rewrite the source documents' repo-relative links, which
#: point at files (tests, modules) that have no page on the site.
BLOB = "https://github.com/jmrplens/phonometry/blob/main"

#: Everything below this line in a target page is generated. The marker is
#: written back on every run, so a page that lost it is simply re-marked.
BEGIN = (
    "<!-- BEGIN GENERATED BODY - transplanted from {source} by "
    "scripts/generate_site_reports.py (`make site-reports`). "
    "Edit the source document, never the text below. -->"
)
END = "<!-- END GENERATED BODY -->"

#: A markdown link target: ``](target)``. Bare enough for these two documents,
#: which contain no reference-style links and no parentheses inside targets.
_LINK_RE = re.compile(r"\]\((?P<target>[^)\s]+)\)")

#: Absolute or in-page targets that must be left alone.
_ABSOLUTE_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|#)")


def rewrite_links(body: str, *, relative_to: str) -> str:
    """Point repo-relative markdown links at the GitHub blob.

    ``docs/ERRATA.md`` cites source modules and regression tests with paths
    relative to ``docs/`` (``../tests/reference_data/``). Those files have no
    route on the site, so the site copy must reach them on GitHub.

    :param body: Markdown body to rewrite.
    :type body: str
    :param relative_to: Repo-relative directory the links resolve against.
    :type relative_to: str
    :return: The body with every repo-relative link turned into a blob URL.
    :rtype: str
    """

    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if _ABSOLUTE_RE.match(target):
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(relative_to, path))
        return f"]({BLOB}/{resolved}{'#' + anchor if anchor else ''})"

    return _LINK_RE.sub(replace, body)


def body_from(source: pathlib.Path, *, start: re.Pattern[str]) -> str:
    """Return the source document from its first ``start`` line onwards.

    Drops whatever precedes it: the "do not hand-edit" file banner, the
    back-link to the docs index and the H1 (the page frontmatter supplies the
    site's own title, and a second H1 would break the heading order).

    :param source: Document to read.
    :type source: pathlib.Path
    :param start: Pattern matching the first line to keep.
    :type start: re.Pattern[str]
    :return: The retained body, with trailing whitespace stripped.
    :rtype: str
    :raises SystemExit: If no line matches ``start``.
    """
    lines = source.read_text(encoding="utf8").splitlines()
    for index, line in enumerate(lines):
        if start.match(line):
            body = "\n".join(lines[index:]).rstrip()
            return rewrite_links(body, relative_to=source.parent.name)
    sys.exit(
        f"{source.relative_to(ROOT)}: no line matches {start.pattern!r}. "
        "The document layout changed; update scripts/generate_site_reports.py."
    )


@dataclasses.dataclass(frozen=True)
class Page:
    """One target page: where the body comes from and where it lands."""

    #: Document under ``docs/`` whose body is transplanted.
    source: pathlib.Path
    #: Target page, relative to ``site/src/content/docs``.
    target: str
    #: Pattern matching the first source line to keep.
    start: re.Pattern[str]


PAGES: tuple[Page, ...] = (
    # The conformance report is not here. It used to be transplanted into two
    # pages, 109 kB of English table each, the Spanish one differing only in its
    # first heading. Those pages now render from ``docs/conformance.json``
    # through ``src/components/Conformance.astro``, which can do what a
    # transplanted table could not: rank the checks by how much of their
    # published tolerance they consume, filter, and give every row an anchor.
    # ``docs/CONFORMANCE.md`` stays as the text copy a GitHub reader and an
    # agent get; it is simply no longer copied into the site.
    Page(
        source=DOCS / "ERRATA.md",
        target="reference/errata.md",
        # Keep the registry's own introduction and status legend: they explain
        # how to read every entry and exist only in the source document.
        start=re.compile(r"^During the clean-room implementation"),
    ),
    Page(
        source=DOCS / "ERRATA.md",
        target="es/reference/errata.md",
        start=re.compile(r"^During the clean-room implementation"),
    ),
)


def render(page: Page) -> str:
    """Build the full text of a target page (kept prose + generated body).

    :param page: Page to render.
    :type page: Page
    :return: The page text, newline-terminated.
    :rtype: str
    :raises SystemExit: If the target page does not exist yet.
    """
    path = CONTENT / page.target
    if not path.is_file():
        sys.exit(
            f"{path.relative_to(ROOT)}: missing. Create the page with its "
            "frontmatter and hand-written introduction first; this script only "
            "fills in the generated body below the marker."
        )
    marker = BEGIN.format(source=f"docs/{page.source.name}")
    kept = path.read_text(encoding="utf8")
    # Split on the marker's stable prefix so a reworded marker still matches.
    prefix = kept.split("<!-- BEGIN GENERATED BODY", 1)[0].rstrip()
    body = body_from(page.source, start=page.start)
    return f"{prefix}\n\n{marker}\n\n{body}\n\n{END}\n"


def main(argv: list[str] | None = None) -> int:
    """Write every target page, reporting what changed.

    :param argv: Command-line arguments (defaults to :data:`sys.argv`).
    :type argv: list[str] | None
    :return: ``0`` on success, ``1`` when ``--check`` finds a stale page.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if any page is out of date",
    )
    args = parser.parse_args(argv)

    stale: list[str] = []
    for page in PAGES:
        path = CONTENT / page.target
        rendered = render(page)
        if path.read_text(encoding="utf8") == rendered:
            continue
        stale.append(page.target)
        if not args.check:
            path.write_text(rendered, encoding="utf8")

    if args.check and stale:
        print(
            "Generated site pages are out of date; run `make site-reports`:",
            file=sys.stderr,
        )
        for target in stale:
            print(f"  site/src/content/docs/{target}", file=sys.stderr)
        return 1

    verb = "stale" if args.check else "written"
    print(
        f"[site-reports] {len(PAGES)} pages checked, {len(stale)} {verb}"
        + (f": {', '.join(stale)}" if stale else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
