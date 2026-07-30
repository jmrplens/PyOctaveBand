#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Coverage gate for the curated API quick table (``docs/api-reference.md``).

The hand-written table in ``docs/api-reference.md`` is the quick reference for
the GitHub/PyPI audience; the authoritative, generated reference lives on the
site (``make api-docs``). Being curated, the table can silently miss a newly
exported name. This gate closes that gap: it parses every backticked name in
the first column of the file's tables and fails when a ``phonometry.__all__``
name has no row.

Extra rows are fine and expected: methods (``OctaveFilterBank.spectrogram``),
namespace subpackages (``phonometry.metrology``) and convention entries
(``.plot()``) document things ``__all__`` does not export. Only the reverse
direction fails, printing the missing names.

Usage::

    python scripts/check_api_reference.py

Exit status 0 when every public name has a row, 1 otherwise.
"""

from __future__ import annotations

import pathlib
import re
import sys

#: A backticked name inside a table cell, e.g. ```leq``` or ```.plot()```.
_BACKTICKED = re.compile(r"`([^`]+)`")


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

    :param markdown: The ``docs/api-reference.md`` source.
    :param public: ``phonometry.__all__``.
    """
    documented = table_names(markdown)
    return [name for name in public if name not in documented]


def main() -> int:
    """Run the gate against the working tree. Returns the exit status."""
    import phonometry

    path = pathlib.Path(__file__).resolve().parent.parent / "docs" / "api-reference.md"
    missing = missing_names(path.read_text(encoding="utf-8"), list(phonometry.__all__))
    if missing:
        print(
            f"docs/api-reference.md is missing {len(missing)} public "
            "name(s) from phonometry.__all__:"
        )
        for name in missing:
            print(f"  - {name}")
        print("Add a table row for each name (see the file's existing style).")
        return 1
    print(
        "docs/api-reference.md covers all "
        f"{len(phonometry.__all__)} phonometry.__all__ names."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
