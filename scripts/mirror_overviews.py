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
no figures and no components, so their mirror needs no second authorship, and
a page whose job is to stay in step with its folder is exactly the kind that
drifts when kept by hand. So these are generated: the frontmatter title
becomes the H1, site-absolute guide links become mirror-relative ones where the
target has a mirror page and absolute site URLs where it does not, and the file
opens with the same index backlink every mirror page carries.

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

SITE_BASE = "https://jmrplens.github.io/phonometry"


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    """The title and description of a page, and its body."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise SystemExit("overview without frontmatter")
    found = re.search(r'^title:\s*"((?:[^"\\]|\\.)*)"\s*$', match.group(1), re.MULTILINE)
    title = found.group(1).replace('\\"', '"') if found else ""
    return {"title": title}, text[match.end() :]


def _overviews() -> list[Path]:
    """Every topic and section overview on the English site, in path order."""
    pages = sorted(SITE.glob("*/index.md")) + sorted(SITE.glob("*/*/index.md"))
    # The mirror is the English edition only, like every other mirror page;
    # ``reference`` holds hand-written mirror content already (theory/,
    # bibliography.md) and the generated API tree, so its indexes are site
    # chrome rather than prose to carry.
    return [
        p
        for p in pages
        if not str(p.relative_to(SITE)).startswith(("es/", "reference/"))
    ]


def _mirror_path(route: str) -> Path:
    return DOCS / route / "index.md"


def _relative_link(from_route: str, to_route: str) -> str | None:
    """The mirror-relative path from one route's index to another page, if
    that page has a mirror file."""
    target = DOCS / f"{to_route}.md"
    if not target.exists():
        target = DOCS / to_route / "index.md"
    if not target.exists():
        return None
    here = (DOCS / from_route / "index.md").parent
    # The shortest relative path, so a link inside the folder reads as the
    # bare filename rather than climbing to docs/ and back down.
    return os.path.relpath(target, here).replace(os.sep, "/")


def _rewrite_links(body: str, route: str) -> str:
    """Site-absolute links to mirror-relative ones, where a mirror page exists."""

    def swap(match: re.Match[str]) -> str:
        target = match.group(1).strip("/")
        if target.startswith(SITE_ONLY_PREFIXES) or target.startswith("es/"):
            return f"({SITE_BASE}/{target}/)"
        rel = _relative_link(route, target)
        return f"({rel})" if rel else f"({SITE_BASE}/{target}/)"

    return re.sub(r"\(/phonometry/([^)#]+?)/?\)", swap, body)


def render(page: Path) -> tuple[Path, str]:
    route = str(page.parent.relative_to(SITE))
    fields, body = _frontmatter(page.read_text())
    title = fields.get("title")
    if not title:
        raise SystemExit(f"{page}: overview without a title")
    up = "../" * len(Path(route).parts)
    head = f"← [Documentation index]({up}README.md)\n\n# {title}\n\n"
    body = _rewrite_links(body.strip(), route) + "\n"
    return _mirror_path(route), head + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on drift instead of writing")
    args = parser.parse_args()

    stale: list[str] = []
    for page in _overviews():
        path, text = render(page)
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    count = len(_overviews())
    if args.check:
        if stale:
            print(
                f"{len(stale)} stale overview mirror(s): {', '.join(stale)}. "
                "Run `python scripts/mirror_overviews.py`.",
                file=sys.stderr,
            )
            return 1
        print(f"overview mirrors up to date: {count} pages.")
        return 0
    print(f"overview mirrors written: {count} pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
