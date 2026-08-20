"""Mirror the glossary into ``docs/reference/glossary.md``.

The glossary is the most reusable reference artefact the documentation has:
you hold a symbol from a report or a specification and you want its unit, the
document that defines it and the page that computes it. It existed only on the
website. ``find docs -iname '*glossar*'`` returned nothing, and the llms
artifacts carried a link to the page rather than its text, so the GitHub reader
and every full-text consumer were sent off-site for the one page that
disambiguates two hundred colliding symbols.

Like the overview mirrors, this file is generated rather than written twice.
The prose above the cards is lifted from ``reference/glossary.mdx`` itself
(site-absolute links become mirror-relative ones where the target has a mirror
page), and the cards come from ``site/src/data/glossary.mjs``, the same array
``Glossary.astro`` renders. A glossary that is copied by hand is exactly what
that data file was created to stop: it had already drifted four ways once.

The card blocks the site uses are the right shape for a phone and the wrong
shape for a plain-markdown file with no CSS, so the mirror is one five-column
table per group, which is what GitHub renders well.

Reading ``.mjs`` needs a JavaScript engine, so the data is dumped through
``node`` and consumed here as JSON. Node is present wherever this runs: the
site is built with pnpm and the docs workflow sets both up before this step.

Run with ``--check`` to compare against the committed file and fail on drift,
which is what CI does; run bare to rewrite it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site" / "src" / "content" / "docs"
DOCS = ROOT / "docs"
DATA = ROOT / "site" / "src" / "data" / "glossary.mjs"
PAGE = SITE / "reference" / "glossary.mdx"
MIRROR = DOCS / "reference" / "glossary.md"

SITE_BASE = "https://jmrplens.github.io/phonometry"

#: Routes that are generated or site-only: a relative link would dangle.
SITE_ONLY_PREFIXES = ("reference/api",)

#: The dump program, kept here rather than in a file of its own so the schema
#: and its reader stay in one place. It prints the array as JSON on stdout.
_DUMP = "import {glossary} from %s; process.stdout.write(JSON.stringify(glossary));"


def _load() -> list[dict[str, Any]]:
    """The glossary array, read through node."""
    node = shutil.which("node")
    if node is None:
        raise SystemExit(
            "scripts/mirror_glossary.py needs node to read site/src/data/glossary.mjs"
        )
    program = _DUMP % json.dumps(DATA.as_uri())
    # Fixed argv, no shell: the only variable is a path this repo owns.
    result = subprocess.run(
        [node, "--input-type=module", "-e", program],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"could not read {DATA}: {result.stderr.strip()}")
    data: list[dict[str, Any]] = json.loads(result.stdout)
    return data


def _localized(value: Any) -> str:
    """A field that is either shared by both languages or given per language."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value.get("en", ""))


def _relative_link(to_route: str) -> str | None:
    """The mirror-relative path from the glossary to another page, if that page
    has a mirror file.
    """
    target = DOCS / f"{to_route}.md"
    if not target.exists():
        target = DOCS / to_route / "index.md"
    if not target.exists():
        return None
    return os.path.relpath(target, MIRROR.parent).replace(os.sep, "/")


def _link(to_route: str) -> str:
    rel = _relative_link(to_route)
    return rel or f"{SITE_BASE}/{to_route}/"


def _rewrite_links(body: str) -> str:
    """Site-absolute links to mirror-relative ones, where a mirror page exists."""

    def swap(match: re.Match[str]) -> str:
        target = match.group(1).strip("/")
        if target.startswith(SITE_ONLY_PREFIXES) or target.startswith("es/"):
            return f"({SITE_BASE}/{target}/)"
        return f"({_link(target)})"

    return re.sub(r"\(/phonometry/([^)#]+?)/?\)", swap, body)


def _prose() -> str:
    """The page's own prose: everything between the frontmatter and the card
    component, with the component import dropped.
    """
    text = PAGE.read_text(encoding="utf-8")
    match = re.match(r"---\n.*?\n---\n", text, re.DOTALL)
    if not match:
        raise SystemExit(f"{PAGE}: page without frontmatter")
    body = text[match.end() :]
    body = re.sub(r"^import .*$", "", body, flags=re.MULTILINE)
    body = body.split("<Glossary")[0]
    return _rewrite_links(body.strip())


def _cell(text: str) -> str:
    """A table cell: pipes escaped, newlines flattened."""
    return text.replace("|", r"\|").replace("\n", " ").strip()


def _row(term: dict[str, Any], titles: dict[str, str]) -> str:
    symbol = term.get("symbol") or _localized(term.get("name")) or term["id"]
    qualifier = _localized(term.get("qualifier"))
    if qualifier:
        symbol = f"{symbol} ({qualifier})"
    where = ", ".join(
        part
        for part in (_localized(term.get("standard")), _localized(term.get("clause")))
        if part
    )
    guide = term.get("guide")
    computed = f"[{titles.get(guide, guide)}]({_link(guide)})" if guide else ""
    cells = (
        symbol,
        _localized(term.get("definition")),
        _localized(term.get("unit")),
        where,
        computed,
    )
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


def _titles() -> dict[str, str]:
    """The title of every English page, by route, so a link carries the name the
    site gives the page instead of a hand-kept copy of it.
    """
    found: dict[str, str] = {}
    for page in SITE.rglob("*.md*"):
        route = page.relative_to(SITE).with_suffix("").as_posix()
        if route.startswith("es/"):
            continue
        head = re.match(
            r"---\n(.*?)\n---\n", page.read_text(encoding="utf-8"), re.DOTALL
        )
        if not head:
            continue
        title = re.search(
            r"""^title:\s*(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|(\S.*?))\s*$""",
            head.group(1),
            re.MULTILINE,
        )
        if title:
            name = (title.group(1) or title.group(2) or title.group(3)).replace(
                '\\"', '"'
            )
            found[route] = name
    return found


def render(glossary: list[dict[str, Any]]) -> str:
    titles = _titles()
    parts = [
        "← [Documentation index](../README.md)\n",
        "# Glossary\n",
        _prose(),
        "",
    ]
    for group in glossary:
        parts.append(f"## {_localized(group['label'])}\n")
        parts.append("| Symbol | Definition | Unit | Defined in | Computed in |")
        parts.append("| :--- | :--- | :--- | :--- | :--- |")
        parts.extend(_row(term, titles) for term in group["terms"])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail on drift instead of writing"
    )
    args = parser.parse_args()

    glossary = _load()
    text = render(glossary)
    terms = sum(len(group["terms"]) for group in glossary)
    if args.check:
        if not MIRROR.exists() or MIRROR.read_text(encoding="utf-8") != text:
            print(
                f"{MIRROR.relative_to(ROOT)} is stale. "
                "Run `python scripts/mirror_glossary.py`.",
                file=sys.stderr,
            )
            return 1
        print(f"glossary mirror up to date: {terms} quantities.")
        return 0
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    MIRROR.write_text(text, encoding="utf-8")
    print(f"glossary mirror written: {terms} quantities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
