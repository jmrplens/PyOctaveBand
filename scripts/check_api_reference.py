#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Coverage gate for the curated API quick table (``docs/reference/api/index.md``).

The hand-written table in ``docs/reference/api/index.md`` is the quick reference for
the GitHub/PyPI audience; the authoritative, generated reference lives on the
site (``make api-docs``). Being curated, the table can silently miss a newly
exported name. This gate closes that gap: it parses every backticked name in
the first column of the file's tables and fails when a public name has no row.
Public means what the library publishes since 4.0: the union of the domain
packages' ``__all__`` plus the four names the top level holds itself. Reading
``phonometry.__all__`` instead would see the twenty-four names the top level
publishes and call the other thirteen hundred private.

Extra rows are fine and expected: methods (``OctaveFilterBank.spectrogram``),
namespace subpackages (``phonometry.metrology``) and convention entries
(``.plot()``) document things ``__all__`` does not export. Only the reverse
direction fails, printing the missing names.

Two things a row *says* are checked, because both go stale in silence and
both send a reader the wrong way.

The **kind column** says what each name is, and a row calling a mapping a
``function`` tells the reader to call it. Twenty-one rows did: eight
dictionaries and a tuple and an array were ``function``, seven result
dataclasses were ``function`` too, and two functions were ``dataclass``. The
vocabulary is deliberately wider than Python's: ``constant`` is a better word
for a module-level float than ``float``, ``mapping`` than ``dict``, and a row
may name the concrete type (``Fluid``, ``Material``). So the check is not
equality: :func:`acceptable_kinds` works out every word that is true of the
object, and the row passes when it uses one of them. A cell naming two
(``dataclass`` / ``constant``, for a class and an instance of it sharing a
row) passes when either is true.

The **version row** shows the package version as a literal
(``phonometry.__version__  # '3.3.0'``). Nothing else re-reads that literal
after a release bump, so it is checked against the installed
``phonometry.__version__``. A page showing no literal fails too, so the check
cannot pass by finding nothing.

Usage::

    python scripts/check_api_reference.py

Exit status 0 when every public name has a row that says what it is, and the
page shows the current version; 1 otherwise.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import pathlib
import re
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

from api_taxonomy import public_names

if TYPE_CHECKING:
    from types import ModuleType

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
    :param public: Every public name (see :func:`api_taxonomy.public_names`).
    """
    documented = table_names(markdown)
    return [name for name in public if name not in documented]


#: Words the table uses for a value rather than for something to call. Any
#: of them is true of any module-level constant, so they are accepted together
#: and the concrete type name is added alongside.
_VALUE_WORDS = frozenset({"constant", "value"})

#: Words that are true of any callable the table lists.
_CALLABLE_WORDS = frozenset({"function", "method", "classmethod"})


def acceptable_kinds(obj: object) -> frozenset[str]:
    """Every word in the table's vocabulary that is true of *obj*.

    A row passes on any one of them, so this is deliberately generous about
    wording and strict about the distinction that matters: something a reader
    calls, versus something a reader reads.

    :param obj: The object the row's name resolves to.
    :return: The acceptable spellings of its kind, lowercased.
    """
    words: set[str] = set()
    if isinstance(obj, type):
        words.add("class")
        if issubclass(obj, Warning):
            words |= {"warning class", "warning"}
        if issubclass(obj, BaseException):
            words |= {"exception", "error"}
        if issubclass(obj, enum.Enum):
            words.add("enum")
        if dataclasses.is_dataclass(obj):
            words.add("dataclass")
        return frozenset(word.lower() for word in words)
    if inspect.isroutine(obj) or isinstance(obj, (staticmethod, classmethod)):
        return frozenset(_CALLABLE_WORDS)
    # Everything else is a value the reader reads: a constant, a table, a
    # prepared instance. The concrete type name is already in, so a row may
    # also say "dict", "tuple", "ndarray", "Fluid" or "Material".
    words |= _VALUE_WORDS
    words.add(type(obj).__name__)
    if isinstance(obj, Mapping):
        words |= {"dict", "mapping", "table"}
    if isinstance(obj, tuple):
        words.add("tuple")
    if hasattr(obj, "dtype") and hasattr(obj, "shape"):
        words.add("array")
    if dataclasses.is_dataclass(obj):
        words.add("dataclass")
    if isinstance(obj, enum.Enum):
        words.add("enum")
    return frozenset(word.lower() for word in words)


def _row_kinds(markdown: str) -> list[tuple[str, frozenset[str]]]:
    """``(name, declared kinds)`` for every table row, in reading order.

    A first cell may hold several names and a second cell several words; the
    row is read as every name being any of the words, which is how the shared
    rows are written.
    """
    rows: list[tuple[str, frozenset[str]]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = re.split(r"(?<!\\)\|", stripped)
        if len(cells) < 3:
            continue
        declared = frozenset(
            word.strip().lower() for word in _BACKTICKED.findall(cells[2])
        )
        if not declared:
            continue
        rows.extend((name, declared) for name in _BACKTICKED.findall(cells[1]))
    return rows


def kind_problems(markdown: str, owners: dict[str, ModuleType]) -> list[str]:
    """Rows whose kind column is not true of the object they name.

    :param markdown: The ``docs/reference/api/index.md`` source.
    :param owners: Public name to the package that publishes it.
    :return: One message per wrong row, in reading order.
    """
    problems: list[str] = []
    for name, declared in _row_kinds(markdown):
        package = owners.get(name)
        if package is None:
            continue  # A method, a subpackage or a convention entry.
        obj = getattr(package, name, None)
        if obj is None:  # pragma: no cover - __all__ and the module agree.
            continue
        allowed = acceptable_kinds(obj)
        if not declared & allowed:
            said = " / ".join(sorted(declared))
            problems.append(f"{name}: says {said}, is {'/'.join(sorted(allowed))}")
    return problems


def version_problems(markdown: str, version: str) -> list[str]:
    """Everything wrong with the version literals in ``markdown``.

    The ``__version__`` row spells the package version out as a literal, and
    a release bump has no reason to touch this file, so the literal goes
    stale silently. Showing none at all is the other half of the same
    defect: a check that finds nothing would pass while reporting a version
    the page does not actually show.

    :param markdown: The ``docs/reference/api/index.md`` source.
    :param version: ``phonometry.__version__``.
    :return: One message per problem, in order of appearance. Empty when the
        page shows the version and every literal agrees with it.
    """
    found = _VERSION_LITERAL.findall(markdown)
    if not found:
        return ["no `phonometry.__version__  # '...'` example to check"]
    return [f"shows {literal!r}" for literal in found if literal != version]


def main() -> int:
    """Run the gate against the working tree. Returns the exit status."""
    import phonometry

    public = sorted(public_names())
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "docs"
        / "reference"
        / "api"
        / "index.md"
    )
    markdown = path.read_text(encoding="utf-8")
    missing = missing_names(markdown, public)
    if missing:
        print(f"docs/reference/api/index.md is missing {len(missing)} public name(s):")
        for name in missing:
            print(f"  - {name}")
        print("Add a table row for each name (see the file's existing style).")
        return 1
    wrong = kind_problems(markdown, public_names())
    if wrong:
        print(
            f"docs/reference/api/index.md misdescribes {len(wrong)} name(s); "
            "the kind column is what a reader acts on:"
        )
        for problem in wrong:
            print(f"  - {problem}")
        return 1
    problems = version_problems(markdown, phonometry.__version__)
    if problems:
        print(
            "docs/reference/api/index.md does not show the current version "
            f"(phonometry.__version__ is {phonometry.__version__!r}):"
        )
        for problem in problems:
            print(f"  - {problem}")
        print("Update the __version__ row's example to the current version.")
        return 1
    print(
        f"docs/reference/api/index.md covers all {len(public)} public names, "
        f"says what each of them is, and shows version {phonometry.__version__}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
