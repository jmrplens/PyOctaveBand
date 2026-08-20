"""Mirror the topic and section overview pages into ``docs/``.

The guides have a plain-markdown edition under ``docs/`` so that a reader on
GitHub, and the llms artifacts built from that folder, get the same prose the
site publishes. The overview pages did not: every topic and every section has
an ``index.md`` on the site explaining what the area is and how its guides fit
together, and none of them had a mirror file. The consequence was quiet and
double. On GitHub, a folder of guides had no page saying what the folder is.
In the llms artifacts, a shard advertised as the full text of an area omitted
the one page that describes the area, and the branch of ``generate_llms.py``
that would have carried it was dead code.

The guides themselves are written twice on purpose: the mirror edition reads
differently (no frontmatter, no components, image-free). The overviews carry
no figures, and the one component they do use wraps prose rather than
generating any, so their mirror needs no second authorship: dropping the
``<Scope>`` and ``<ScopeClaim>`` tags leaves exactly the paragraphs they
present. And a page whose job is to stay in step with its folder is exactly
the kind that drifts when kept by hand. So these are generated: the
frontmatter title becomes the H1, the component wrappers and their imports
come off, site-absolute guide links become mirror-relative ones where the
target has a mirror page and absolute site URLs where it does not, and the
file opens with the same index backlink every mirror page carries.

Run with ``--check`` to compare against the committed files and fail on drift,
which is what CI does; run bare to rewrite them.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site" / "src" / "content" / "docs"
DOCS = ROOT / "docs"

#: Routes with no mirror file whose links must stay absolute: the pages under
#: these prefixes are generated or site-only, so a relative link would dangle.
SITE_ONLY_PREFIXES = ("reference/api",)

#: Site routes whose mirror counterpart is not a file of the same name. The
#: guides index is the clearest case: on the site it is ``start/guides``, in the
#: mirror it is ``docs/README.md`` itself, so the generated overviews used to
#: send a GitHub reader off to the website for a page sitting one directory up
#: in the same repository.
ROUTE_ALIASES = {"start/guides": "README.md"}

SITE_BASE = "https://jmrplens.github.io/phonometry"


def _frontmatter(text: str, page: Path) -> tuple[dict[str, str], str]:
    """The title of a page, and its body."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        msg = f"{page}: overview without frontmatter"
        raise SystemExit(msg)
    found = re.search(
        r"""^title:\s*(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|(\S.*?))\s*$""",
        match.group(1),
        re.MULTILINE,
    )
    if not found:
        msg = f"{page}: overview without a title"
        raise SystemExit(msg)
    title = (found.group(1) or found.group(2) or found.group(3)).replace('\\"', '"')
    return {"title": title}, text[match.end() :]


def _overviews() -> list[Path]:
    """Every topic and section overview on the English site, in path order."""
    # ``index.md*``, not ``index.md``: the overviews became .mdx when their
    # scope sections moved onto the <Scope> components, and a glob that only
    # matched .md did not fail, it matched nothing — the run reported "0 pages
    # up to date", CI's staleness diff saw no change, and every mirror would
    # have frozen at its pre-component prose in silence.
    pages = sorted(SITE.glob("*/index.md*")) + sorted(SITE.glob("*/*/index.md*"))
    # The mirror is the English edition only, like every other mirror page;
    # ``reference`` holds hand-written mirror content already (theory/,
    # bibliography.md) and the generated API tree, so its indexes are site
    # chrome rather than prose to carry.
    return [
        p
        for p in pages
        if not p.relative_to(SITE).as_posix().startswith(("es/", "reference/"))
    ]


def _mirror_path(route: str) -> Path:
    return DOCS / route / "index.md"


#: The mirror index files this run will produce, whether or not they exist
#: yet: resolving against the disk instead made the output depend on what a
#: previous run had left behind, so a cold tree and a warm tree disagreed.
PLANNED: set[Path] = set()


def _relative_link(from_route: str, to_route: str) -> str | None:
    """The mirror-relative path from one route's index to another page, if
    that page has, or will have, a mirror file.
    """
    alias = ROUTE_ALIASES.get(to_route)
    if alias is not None:
        target = DOCS / alias
        if not target.exists():
            return None
        here = (DOCS / from_route / "index.md").parent
        return os.path.relpath(target, here).replace(os.sep, "/")
    target = DOCS / f"{to_route}.md"
    if not target.exists():
        target = DOCS / to_route / "index.md"
    if not target.exists() and target not in PLANNED:
        return None
    here = (DOCS / from_route / "index.md").parent
    # The shortest relative path, so a link inside the folder reads as the
    # bare filename rather than climbing to docs/ and back down.
    return os.path.relpath(target, here).replace(os.sep, "/")


def _rewrite_links(body: str, route: str) -> str:
    """Site-absolute links to mirror-relative ones, where a mirror page exists.

    A link can carry a ``#fragment`` onto a heading of the target page — a
    See-also block pointing at one theory section rather than the whole page.
    The fragment has to survive onto whichever form the target link takes,
    mirror-relative or the absolute site URL, or it is silently dropped and
    the site-absolute path is left in the mirror file verbatim: neither a
    working mirror-relative link nor a working external one.
    """

    def swap(match: re.Match[str]) -> str:
        target = match.group(1).strip("/")
        fragment = match.group(2) or ""
        if target.startswith(SITE_ONLY_PREFIXES) or target.startswith("es/"):
            return f"({SITE_BASE}/{target}/{fragment})"
        rel = _relative_link(route, target)
        return f"({rel}{fragment})" if rel else f"({SITE_BASE}/{target}/{fragment})"

    return re.sub(r"\(/phonometry/([^)#]+?)/?(#[^)]*)?\)", swap, body)


#: The component wrappers an overview may use, and the import lines that bring
#: them in. The mirror is plain markdown for GitHub, which renders neither, so
#: both are removed and the children are kept: a <ScopeClaim> holds nothing but
#: the prose of one claim, so dropping the tag leaves the paragraph it wraps.
#: The claim's `label` is not recovered here — the overviews are the pages that
#: carry no label, because every claim under "What this section does not cover"
#: is on the same side of the boundary and the heading already says which.
#: ``[ \t]*`` and not ``\s*``: ``\s`` matches a newline, so a greedy trailing
#: ``\s*$`` swallows the blank line after the tag as well as the tag's own,
#: which welds two claims into one paragraph and glues the panel's last line
#: to the heading below it. Both were visible in the first generated mirror.
_IMPORT = re.compile(r"^import\s+\w+\s+from\s+'[^']*';[ \t]*$\n?", re.MULTILINE)
_COMPONENT = re.compile(
    r"^</?(?:Scope|ScopeClaim)(?:\s[^>]*)?>[ \t]*$\n?", re.MULTILINE
)
_BLANK_RUN = re.compile(r"\n{3,}")


def _strip_components(body: str) -> str:
    """The prose of an overview, without the MDX that presents it."""
    return _BLANK_RUN.sub("\n\n", _COMPONENT.sub("", _IMPORT.sub("", body))).strip()


def render(page: Path) -> tuple[Path, str]:
    route = page.parent.relative_to(SITE).as_posix()
    fields, body = _frontmatter(page.read_text(encoding="utf-8"), page)
    title = fields.get("title")
    if not title:
        msg = f"{page}: overview without a title"
        raise SystemExit(msg)
    up = "../" * len(Path(route).parts)
    head = f"← [Documentation index]({up}README.md)\n\n# {title}\n\n"
    body = _rewrite_links(_strip_components(body), route) + "\n"
    return _mirror_path(route), head + body


#: Mirror files that legitimately do not open with the index backlink: the
#: README is the index, and the other two are generated elsewhere and start
#: with an HTML comment.
_NO_BACKLINK = {"README.md", "CONFORMANCE.md", "ERRATA.md"}

_BACKLINK = "← [Documentation index]("


def _readme_links_every_overview() -> list[str]:
    """Overview mirrors ``docs/README.md`` does not link.

    ``generate_llms.py::_check_readme_covers_the_mirror`` exempts the index
    files "because the README's section structure is its own index of the
    folders". That rationale was false for as long as the README was one flat
    list of guides and linked none of the 35 generated overviews; this is the
    assertion that makes it true.
    """
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    return [
        path.relative_to(DOCS).as_posix()
        for path in sorted(PLANNED)
        if f"({path.relative_to(DOCS).as_posix()})" not in readme
    ]


def _pages_without_the_index_backlink() -> list[str]:
    """Mirror pages whose first line is not the documentation-index backlink.

    It is the mirror's only upward navigation — the guides do not link their
    section overview — so a page without it strands a reader who arrived from a
    search result. One page lost it and nothing noticed.
    """
    missing: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(DOCS)
        if rel.parts[0] == "superpowers" or rel.as_posix() in _NO_BACKLINK:
            continue
        first = path.read_text(encoding="utf-8").split("\n", 1)[0]
        if not first.startswith(_BACKLINK):
            missing.append(rel.as_posix())
    return missing


#: Punctuation GitHub strips before turning a heading into its automatic
#: anchor, mirrored from the character classes and literals ``github-slugger``
#: removes. This is what actually renders the ``#...`` next to a heading on
#: github.com, not a guess at one — validated against every fragment link
#: already in the mirror before this was wired into ``--check``. Built only
#: from ``chr()`` code points and ``re.escape()`` of a literal string, so
#: this is one flat character class with no alternation or repetition for a
#: ReDoS scanner to flag.
_SLUG_PUNCTUATION = re.compile(
    "["
    + chr(0x2000)
    + "-"
    + chr(0x206F)
    + chr(0x2E00)
    + "-"
    + chr(0x2E7F)
    + re.escape("""\'!"#$%&()*+,./:;<=>?@[]^`{|}~""")
    + "]"
)

_ATX_HEADING = re.compile(r"^#{1,6}(?:\s+(.*?))?\s*#*\s*$")
_LINK_LABEL = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EXTERNAL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_LINK_TITLE = re.compile(r"""\s+(?:"[^"]*"|'[^']*')\s*$""")


def _slug(heading: str) -> str:
    """The anchor GitHub assigns ``heading``, before duplicate-numbering.

    A heading that is itself a link — the generated overviews' subsections are
    written that way — anchors on its label, not its target:
    ``[Absorbers](i.md)`` slugs to ``absorbers``. Inline code needs no
    separate unwrapping: the backticks are in the punctuation class above and
    fall out either way, which is why ``Report metadata (`ReportMetadata`)``,
    the heading every report-fiche page's See-also block points at, slugs to
    ``report-metadata-reportmetadata`` rather than something that keeps the
    backticks.
    """
    text = _LINK_LABEL.sub(r"\1", heading).lower()
    text = _SLUG_PUNCTUATION.sub("", text)
    return text.replace(" ", "-")


def _unfenced_lines(text: str) -> list[tuple[int, str]]:
    """The ``(lineno, line)`` pairs of ``text`` outside fenced code blocks.

    A fence can hold anything that merely looks like markdown: a Python
    comment that starts a line with ``#``, as several reproduction blocks do,
    or a snippet demonstrating link syntax a guide is teaching rather than
    using. Shared by the heading scan and the anchor scan so both agree on
    what counts as fenced.
    """
    lines: list[tuple[int, str]] = []
    fence = ""
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped[:3] in ("```", "~~~"):
            if fence and stripped.startswith(fence):
                fence = ""
            elif not fence:
                fence = stripped[:3]
            continue
        if fence:
            continue
        lines.append((lineno, line))
    return lines


def _headings(path: Path) -> list[str]:
    """The ATX heading text of a markdown file, in document order.

    Skips fenced code blocks (see ``_unfenced_lines``), so a Python comment
    that starts a line with ``#`` inside a reproduction block is not mistaken
    for one.
    """
    headings: list[str] = []
    for _, line in _unfenced_lines(path.read_text(encoding="utf-8")):
        match = _ATX_HEADING.match(line)
        if match and match.group(1):
            headings.append(match.group(1))
    return headings


def _fragments(path: Path) -> set[str]:
    """The anchors ``path`` answers for.

    Includes the ``-1``, ``-2`` GitHub appends when the same slug recurs
    later on the page, in appearance order — the second "References" heading
    on a page is not the same anchor as the first.
    """
    seen: dict[str, int] = {}
    fragments: set[str] = set()
    for heading in _headings(path):
        slug = _slug(heading)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        fragments.add(slug if n == 0 else f"{slug}-{n}")
    return fragments


def _link_bodies(line: str) -> list[str]:
    """The raw text inside each markdown link's ``(...)`` on ``line``.

    Finds the closing parenthesis by tracking depth rather than stopping at
    the first ``)``: a link target can contain one, as some URLs do
    (``[x](url_(a))``). A quoted title can contain one too
    (``[x](f.md "t)")``), so parentheses inside an open quote do not count.
    """
    bodies: list[str] = []
    i = 0
    while True:
        start = line.find("](", i)
        if start == -1:
            break
        depth = 1
        quote = ""
        j = start + 2
        while j < len(line) and depth:
            char = line[j]
            if quote:
                if char == quote:
                    quote = ""
            elif char in "\"'":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            j += 1
        if depth != 0:
            break  # unterminated; nothing further on this line is reliable
        bodies.append(line[start + 2 : j - 1])
        i = j
    return bodies


def _link_target(body: str) -> str:
    """The destination inside a link body, an optional quoted title
    (``url "title"`` or ``url 'title'``) stripped off the end.

    Without this, a title's own punctuation — ``[x](f.md "See #3")`` — could
    be mistaken for a fragment, or a ``)`` inside it could have been mistaken
    for the link's closing parenthesis before ``_link_bodies`` balanced them.
    """
    title = _LINK_TITLE.search(body)
    return body[: title.start()] if title else body


def _broken_intra_mirror_anchors() -> list[str]:
    """Fragment links in the mirror whose target heading does not exist.

    A See-also block copied from the site carries the site's anchor with it.
    When a theory page's heading structure moved and the mirror's hand-kept
    copy of that heading did not move with it, seven such anchors pointed at
    text that no longer existed there and were repaired by hand, one at a
    time, because nothing else would have caught the next one.
    """
    broken: list[str] = []
    cache: dict[Path, set[str]] = {}

    def fragments_of(dest: Path) -> set[str]:
        if dest not in cache:
            cache[dest] = _fragments(dest)
        return cache[dest]

    for path in sorted(DOCS.rglob("*.md")):
        if path.relative_to(DOCS).parts[0] == "superpowers":
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in _unfenced_lines(text):
            for body in _link_bodies(line):
                target_part, sep, fragment = _link_target(body).partition("#")
                if not sep or not fragment:
                    continue
                where = f"{path.relative_to(ROOT)}:{lineno}"
                if _EXTERNAL_SCHEME.match(target_part):
                    continue
                if target_part.startswith("/"):
                    # A site-absolute path with a fragment: ``_rewrite_links``
                    # is the only producer of these, and it always resolves
                    # them to a mirror-relative link or a full site URL, so
                    # one reaching here is not a link this repository serves.
                    broken.append(
                        f"{where}: #{fragment} -> {target_part} "
                        "(site-absolute path, not a mirror-relative link)"
                    )
                    continue
                dest = (
                    path if not target_part else (path.parent / target_part).resolve()
                )
                try:
                    shown = dest.relative_to(ROOT)
                except ValueError:
                    shown = dest
                if not dest.exists():
                    broken.append(f"{where}: #{fragment} -> {shown} (file not found)")
                    continue
                if dest.suffix != ".md":
                    # A line anchor into a non-markdown file, such as
                    # GitHub's own ``script.py#L42``, is not a heading slug
                    # this check knows how to validate.
                    continue
                if fragment not in fragments_of(dest):
                    broken.append(
                        f"{where}: #{fragment} -> {shown} (heading not found)"
                    )
    return broken


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail on drift instead of writing"
    )
    args = parser.parse_args()

    PLANNED.update(
        _mirror_path(page.parent.relative_to(SITE).as_posix()) for page in _overviews()
    )
    stale: list[str] = []
    for page in _overviews():
        path, text = render(page)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    count = len(_overviews())
    if args.check:
        if stale:
            print(
                f"{len(stale)} stale overview mirror(s): {', '.join(stale)}. "
                "Run `python scripts/mirror_overviews.py`.",
                file=sys.stderr,
            )
            return 1
        unlinked = _readme_links_every_overview()
        if unlinked:
            print(
                "docs/README.md links none of these overview pages: "
                + ", ".join(unlinked),
                file=sys.stderr,
            )
            return 1
        stranded = _pages_without_the_index_backlink()
        if stranded:
            print(
                "mirror pages without the documentation-index backlink: "
                + ", ".join(stranded),
                file=sys.stderr,
            )
            return 1
        broken_anchors = _broken_intra_mirror_anchors()
        if broken_anchors:
            print(
                "broken intra-mirror anchors:\n  " + "\n  ".join(broken_anchors),
                file=sys.stderr,
            )
            return 1
        print(f"overview mirrors up to date: {count} pages.")
        return 0
    print(f"overview mirrors written: {count} pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
