#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Guard the reading order of every Python fence in the documentation.

The code blocks of a page form one sequential example: a later fence may use
names an earlier fence defined, and 200-odd pages lean on that to build a
result step by step without repeating a prelude. What the convention cannot
survive is the *reverse* reference. One shipped page used ``ir`` and ``fs``
defined only in the figure block further down -- and an ``ir`` from an
earlier, different room was in scope, so reading the page top to bottom
produced numbers that were not the annotated ones, with no visible error.
The llms.txt shards inherit the fences verbatim, so a machine reader pays for
the same slip more readily than a human.

Hence the rule this script enforces, over every documentation page that
carries Python fences: **a fence may only use names defined by an earlier
fence of the same page** (or by itself, or by Python's builtins). Never a
later fence, never another page.

Two kinds of name are exempt, both deliberately:

* **Placeholders.** Some pages address the reader's own data -- "your
  measured ``spl``", "the ``audio_blocks`` of your capture" -- and never
  define it, because defining it would replace the reader's measurement with
  an invented one. Those names are declared in :data:`PLACEHOLDERS`, keyed by
  the page's route with the ``es/`` prefix stripped, so the Spanish twin is
  held to exactly the same set as the English page. A new placeholder means a
  new registry line, which is the point: it puts the choice in review instead
  of letting a missing definition pass as one. The page must introduce the
  name as the reader's own, in prose or in the fence's own marker comment
  (``# audio_blocks: successive frames of your microphone recording``); the
  comment form travels with the code wherever the fence is copied, which is
  what the llms shards do to it. A page may instead bind the name to
  ``...`` (Ellipsis) with the same comment, which is the older idiom some
  pages carry: that defines the name, so no registry line is needed, at the
  price of a fence that cannot run.
* **Builtins**, plus ``_`` as the conventional throwaway.

A companion rule is editorial and not enforced here, because it is not
mechanisable: when a name's *value* carries an annotated output
(``# 0.60 (D)``), the fence that annotates should define it itself or stand
immediately after the fence that does, not a section away.

Usage::

    python scripts/check_fence_names.py

Exit status 0 when every fence reads in order, 1 otherwise, with one line per
offence naming the page, the fence and the names.
"""

from __future__ import annotations

import ast
import builtins
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The documentation trees that carry sequential examples. The generated API
#: reference is excluded: its fences are signatures rendered from docstrings,
#: not a narrative.
CONTENT = ROOT / "site" / "src" / "content" / "docs"
DOCS = ROOT / "docs"

#: A Python fence, opening attributes tolerated, non-greedy to the closer.
_FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)

#: Names a fence may always use.
_BUILTINS = frozenset(dir(builtins)) | {"_"}

#: Reader-owned names, per page route (``es/`` stripped, suffix kept off), so
#: the English page and its Spanish twin carry the same set. The prose of the
#: page introduces each one as the reader's own data; defining it in code
#: would replace the reader's measurement with an invented one.
PLACEHOLDERS: dict[str, frozenset[str]] = {
    "aircraft/aircraft-noise": frozenset({"spl"}),
    "aircraft/rotorcraft-noise": frozenset(
        {
            "h_50_level",
            "h_60_climb",
            "h_70_level",
            "hemispheres",
            "measured_levels",
            "positions",
            "times",
            "tx",
            "ty",
            "tz",
        }
    ),
    "devices/electroacoustics/electroacoustics": frozenset(
        {"fs", "idle", "idle_output", "output", "signal", "x", "y"}
    ),
    "docs/aircraft/aircraft-noise": frozenset({"spl"}),
    "docs/aircraft/rotorcraft-noise": frozenset(
        {
            "h_50_level",
            "h_60_climb",
            "h_70_level",
            "hemispheres",
            "measured_levels",
            "positions",
            "times",
            "tx",
            "ty",
            "tz",
        }
    ),
    "docs/devices/electroacoustics/electroacoustics": frozenset(
        {"fs", "idle_output", "output", "signal", "x", "y"}
    ),
    "docs/signals/filters/block-processing": frozenset(
        {"audio_blocks", "audio_stream"}
    ),
    "docs/signals/levels/time-weighting": frozenset({"audio_blocks"}),
    "docs/signals/spectra/spectral-analysis": frozenset({"record"}),
    "signals/filters/block-processing": frozenset({"audio_blocks", "audio_stream"}),
    "signals/levels/time-weighting": frozenset({"audio_blocks"}),
    "signals/spectra/spectral-analysis": frozenset({"record"}),
}


def _route(path: pathlib.Path) -> str:
    """The registry key of a page: tree-relative, ``es/`` stripped, no suffix.

    :param path: A documentation page.
    :type path: pathlib.Path
    :return: The route both language twins share.
    :rtype: str
    """
    if path.is_relative_to(CONTENT):
        relative = path.relative_to(CONTENT)
    elif path.is_relative_to(DOCS):
        relative = pathlib.Path("docs") / path.relative_to(DOCS)
    else:
        # A page outside the tree (the tests build them in a temporary
        # directory): its own name is its route.
        relative = pathlib.Path(path.name)
    parts = relative.parts
    if parts and parts[0] == "es":
        parts = parts[1:]
    return str(pathlib.PurePosixPath(*parts).with_suffix(""))


def _defined(tree: ast.AST) -> set[str]:
    """Every name a fence binds, at any depth.

    Comprehension targets and function arguments are scoped tighter than the
    module in real Python; counting them as page-wide definitions is
    deliberately permissive, because this check hunts reading-order breaks,
    not scope leaks.

    :param tree: A parsed fence.
    :type tree: ast.AST
    :return: The bound names.
    :rtype: set[str]
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for entry in node.names:
                names.add((entry.asname or entry.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs)) and node.name:
            names.add(node.name)
    return names


def _used(tree: ast.AST) -> set[str]:
    """Every bare name a fence reads."""
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _pages() -> list[pathlib.Path]:
    """Every documentation page the rule applies to, in a fixed order."""
    found: list[pathlib.Path] = []
    for path in sorted(CONTENT.rglob("*.md*")):
        if "reference/api" in path.as_posix():
            continue
        found.append(path)
    for path in sorted(DOCS.rglob("*.md")):
        if "superpowers" in path.parts or "reference/api" in path.as_posix():
            continue
        found.append(path)
    return found


def check_page(path: pathlib.Path) -> list[str]:
    """The reading-order offences of one page.

    :param path: The page to check.
    :type path: pathlib.Path
    :return: One message per offending fence, empty when the page reads in
        order.
    :rtype: list[str]
    """
    text = path.read_text(encoding="utf8")
    blocks = _FENCE.findall(text)
    if not blocks:
        return []
    trees: list[ast.AST | None] = []
    problems: list[str] = []
    relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    for index, block in enumerate(blocks, start=1):
        try:
            trees.append(ast.parse(block))
        except SyntaxError as error:
            trees.append(None)
            problems.append(
                f"{relative}: fence {index} does not parse as Python "
                f"(line {error.lineno}: {error.msg}). A fragment that cannot "
                "be read cannot be checked; make it parse or drop the "
                "``python`` tag."
            )
    allowed = PLACEHOLDERS.get(_route(path), frozenset())
    definitions = [_defined(tree) if tree else set() for tree in trees]
    seen: set[str] = set()
    for index, tree in enumerate(trees):
        if tree is None:
            seen |= definitions[index]
            continue
        seen |= definitions[index]
        missing = _used(tree) - seen - _BUILTINS - allowed
        if not missing:
            continue
        later: set[str] = set()
        for later_definitions in definitions[index + 1 :]:
            later |= later_definitions
        forward = sorted(missing & later)
        nowhere = sorted(missing - later)
        if forward:
            problems.append(
                f"{relative}: fence {index + 1} uses "
                f"{', '.join(forward)} before the page defines "
                "them. The fences of a page read top to bottom; move the "
                "definition above this fence or define the name here."
            )
        if nowhere:
            problems.append(
                f"{relative}: fence {index + 1} uses "
                f"{', '.join(nowhere)}, which no fence of the page defines. "
                "Define it in an earlier fence, or declare it a "
                "reader-owned placeholder in PLACEHOLDERS at the top of "
                "scripts/check_fence_names.py."
            )
    return problems


def _stale_placeholder_routes() -> list[str]:
    """Registry lines whose route no longer has a page, so the line is dead."""
    routes = {_route(path) for path in _pages()}
    return sorted(set(PLACEHOLDERS) - routes)


def main() -> int:
    """Check every page and report.

    :return: ``0`` when clean, ``1`` otherwise.
    :rtype: int
    """
    problems: list[str] = []
    for path in _pages():
        problems.extend(check_page(path))
    problems.extend(
        f"scripts/check_fence_names.py: PLACEHOLDERS entry {route!r} "
        "matches no page; delete the line."
        for route in _stale_placeholder_routes()
    )
    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    for problem in problems:
        target, _, message = problem.partition(": ")
        print(f"::error file={target}::{message}" if in_ci else problem)
    if not problems:
        pages = _pages()
        fences = sum(len(_FENCE.findall(p.read_text(encoding="utf8"))) for p in pages)
        print(
            f"{fences} fences on {len(pages)} pages read in order; "
            f"{len(PLACEHOLDERS)} pages carry declared placeholders."
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
