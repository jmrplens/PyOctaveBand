#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Coverage gate for the curated API quick table (``docs/reference/api/index.md``).

The hand-written table in ``docs/reference/api/index.md`` is the quick reference for
the GitHub/PyPI audience; the authoritative, generated reference lives on the
site (``make api-docs``). Being curated, the table can silently miss a newly
exported name. This gate closes that gap: it parses every backticked name in
the first column of the file's tables and fails when a ``phonometry.__all__``
name has no row.

Extra rows are fine and expected: methods (``OctaveFilterBank.spectrogram``),
namespace subpackages (``phonometry.metrology``) and convention entries
(``.plot()``) document things ``__all__`` does not export. Only the reverse
direction fails, printing the missing names.

Row *content* is otherwise outside this gate, with one exception: the
``__version__`` row shows the package version as a literal
(``phonometry.__version__  # '3.3.0'``). Nothing else re-reads that literal
after a release bump, so it is checked against the installed
``phonometry.__version__`` here.

Usage::

    python scripts/check_api_reference.py

Exit status 0 when every public name has a row and the version literal is
current, 1 otherwise.
"""

from __future__ import annotations

import pathlib
import re
import sys

#: A backticked name inside a table cell, e.g. ```leq``` or ```.plot()```.
_BACKTICKED = re.compile(r"`([^`]+)`")

#: The version literal in the ``__version__`` row's usage cell, e.g.
#: ``phonometry.__version__  # '3.3.0'``.
_VERSION_LITERAL = re.compile(r"phonometry\.__version__\s*#\s*['\"]([^'\"]*)['\"]")


def table_names(markdown: str) -> set[str]:
    """Backticked names found in the first column of ``markdown`` tables.

    A cell may document several names at once (``| `a` / `b` | ...``); every
    backticked token in the first cell counts.

    :param markdown: Markdown source containing zero or more pipe tables.
    :return: The set of names, backticks stripped (e.g. ``{"leq", ".plot()"}``).
    """
    names: set[str] = set()
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Split on unescaped pipes only (``\|`` is a literal pipe in a cell).
        first_cell = re.split(r"(?<!\\)\|", stripped)[1]
        names.update(_BACKTICKED.findall(first_cell))
    return names


def missing_names(markdown: str, public: list[str]) -> list[str]:
    """Public names without a table row, in ``__all__`` order.

    :param markdown: The ``docs/reference/api/index.md`` source.
    :param public: ``phonometry.__all__``.
    """
    documented = table_names(markdown)
    return [name for name in public if name not in documented]


def stale_versions(markdown: str, version: str) -> list[str]:
    """Version literals in ``markdown`` that disagree with ``version``.

    The ``__version__`` row spells the package version out as a literal, and
    a release bump has no reason to touch this file, so the literal goes
    stale silently.

    :param markdown: The ``docs/reference/api/index.md`` source.
    :param version: ``phonometry.__version__``.
    :return: The literals that do not match, in order of appearance. Empty
        when they all match, and also when the file shows none at all.
    """
    return [found for found in _VERSION_LITERAL.findall(markdown) if found != version]


def main() -> int:
    """Run the gate against the working tree. Returns the exit status."""
    import phonometry

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "docs" / "reference" / "api" / "index.md"
    )
    markdown = path.read_text(encoding="utf-8")
    missing = missing_names(markdown, list(phonometry.__all__))
    if missing:
        print(
            f"docs/reference/api/index.md is missing {len(missing)} public "
            "name(s) from phonometry.__all__:"
        )
        for name in missing:
            print(f"  - {name}")
        print("Add a table row for each name (see the file's existing style).")
        return 1
    stale = stale_versions(markdown, phonometry.__version__)
    if stale:
        print(
            "docs/reference/api/index.md shows a stale version literal "
            f"(phonometry.__version__ is {phonometry.__version__!r}):"
        )
        for found in stale:
            print(f"  - {found!r}")
        print("Update the __version__ row's example to the current version.")
        return 1
    print(
        "docs/reference/api/index.md covers all "
        f"{len(phonometry.__all__)} phonometry.__all__ names "
        f"and shows version {phonometry.__version__}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
