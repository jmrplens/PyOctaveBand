#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fail on a private module-level constant nothing reads.

A constant with a leading underscore and an upper-case name is a number or a
table lifted out of a standard so that the code that uses it can say where it
came from. One that nothing reads is either a leftover, or the trace of a
check the docstring above it still promises, and the second is the reason this
guard exists rather than a periodic sweep: the sweep it came from turned up a
recommended specimen velocity the code carried and never checked, and a
docstring that promised a default the code did not take.

The scan is one AST pass over ``src``, ``tests`` and ``scripts``. A name
counts as read when it appears as a loaded :class:`ast.Name`, as an attribute,
as an import alias, or inside a string, which covers ``getattr`` and
``__all__``; that is deliberately generous, because a false alarm here costs a
reader more than a missed leftover.

:data:`KEPT` is the escape hatch, and an entry is a decision with a reason
attached rather than a bare line.
"""

from __future__ import annotations

import argparse
import ast
import collections
import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: Where a constant may be defined, and everything that may read one.
SOURCE = ROOT / "src" / "phonometry"
SEARCHED = ("src", "tests", "scripts")

#: Private constants that stay although nothing reads them, and why. Keyed by
#: module and name, like every other mapping here: a bare name would silence
#: the check in every module that defines it, and this tree has four modules
#: defining ``_MIN_POLYLINE_NODES``. Nothing belongs here that could instead be
#: used: a constant kept for documentation is documentation, and belongs in a
#: docstring where a reader will find it.
KEPT: dict[tuple[str, str], str] = {}


def _module_of(path: pathlib.Path) -> str | None:
    """The dotted module a file under ``src`` is, or ``None`` outside it."""
    try:
        relative = path.relative_to(ROOT / "src")
    except ValueError:
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(path: pathlib.Path, module: str | None) -> str:
    """The package a relative import in that file climbs from.

    An ``__init__.py`` *is* its package, so its module name is already the
    answer; every other file is a module inside its package and one component
    has to come off. Stripping unconditionally puts every relative import
    written in an ``__init__.py`` one package too high, which both raises a
    false alarm on the constant it reads and clears a genuinely dead one in
    whatever module the wrong target names.
    """
    if module is None:
        return ""
    if path.name == "__init__.py":
        return module
    return module.rsplit(".", 1)[0] if "." in module else ""


#: Statements that hold module-level code without leaving module scope. A
#: constant under ``if TYPE_CHECKING`` or in the fallback arm of a ``try`` is a
#: module-level constant, and reading only ``tree.body`` never sees it.
_NESTING = (ast.If, ast.Try, ast.With)


def _module_level(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Every statement that runs at module scope, through if, try and with."""
    for node in body:
        if isinstance(node, _NESTING):
            branches = [node.body, getattr(node, "orelse", [])]
            if isinstance(node, ast.Try):
                branches += [handler.body for handler in node.handlers]
                branches.append(node.finalbody)
            for branch in branches:
                yield from _module_level(branch)
        else:
            yield node


def _assigned_names(target: ast.expr) -> Iterator[ast.Name]:
    """The names one assignment target binds, unpacking tuples and lists."""
    if isinstance(target, ast.Name):
        yield target
    elif isinstance(target, ast.Tuple | ast.List):
        for element in target.elts:
            yield from _assigned_names(element)


def defined_constants() -> dict[tuple[str, str], tuple[pathlib.Path, int]]:
    """Every ``_UPPER_CASE`` assigned at module level, keyed by module and name.

    Keyed by the pair rather than by the name alone: two modules may define
    the same private name, and one of them being read says nothing about the
    other. That is not hypothetical, it is how a duplicated absolute zero and
    a duplicated model list both stayed in the tree.
    """
    found: dict[tuple[str, str], tuple[pathlib.Path, int]] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        module = _module_of(path)
        if module is None:  # pragma: no cover - SOURCE is inside src
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _module_level(tree.body):
            targets: list[ast.Name] = []
            if isinstance(node, ast.Assign):
                targets = [
                    element
                    for target in node.targets
                    for element in _assigned_names(target)
                ]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            for target in targets:
                name = target.id
                private = name.startswith("_") and not name.startswith("__")
                if private and name.upper() == name:
                    found.setdefault((module, name), (path, node.lineno))
    return found


def _imported_from(tree: ast.AST, package: str) -> set[tuple[str, str]]:
    """``(module, name)`` pairs a file imports by name, relative ones resolved."""
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            # A relative import climbs from the importing file's own package.
            base = package.split(".")
            climbed = base[: len(base) - (node.level - 1)] if node.level > 1 else base
            target = ".".join([*climbed, node.module] if node.module else climbed)
        else:
            target = node.module or ""
        for alias in node.names:
            pairs.add((target, alias.name))
    return pairs


def names_read() -> tuple[dict[str, set[str]], set[tuple[str, str]], set[str]]:
    """What reads what: bare names per file, imported pairs, and loose names.

    The three are not interchangeable. A bare name can only reach a private
    module-level constant from inside the module that defines it, so it is
    kept per file. An ``from x import _Y`` names both sides, so it is kept as
    a pair. An attribute or a string could be either, so those are kept as
    loose names and credited to every definition of that name, which is the
    generous half of the scan.
    """
    bare: dict[str, set[str]] = collections.defaultdict(set)
    imported: set[tuple[str, str]] = set()
    loose: set[str] = set()
    for directory in SEARCHED:
        for path in sorted((ROOT / directory).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - a file being edited
                continue
            imported |= _imported_from(tree, _package_of(path, _module_of(path)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    bare[str(path)].add(node.id)
                elif isinstance(node, ast.Attribute):
                    loose.add(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    loose.add(node.value)
    return bare, imported, loose


def classify(
    defined: dict[tuple[str, str], tuple[pathlib.Path, int]],
    read: tuple[dict[str, set[str]], set[tuple[str, str]], set[str]],
    kept: dict[tuple[str, str], str],
) -> tuple[dict[tuple[str, str], tuple[pathlib.Path, int]], list[str]]:
    """Split the defined constants into the dead ones and the stale ``kept`` entries.

    Deadness is settled first and ``kept`` applied to the answer, not folded
    into the test. Folding it in makes the hatch report itself: a kept name
    would be excluded from the dead set, land in the live one by subtraction,
    and be named stale on every run, so no entry could ever be added.

    :param defined: Every private constant, keyed by module and name.
    :param read: The three readings of :func:`names_read`.
    :param kept: The escape hatch, name to reason.
    :return: The dead constants, and the sorted ``kept`` names that no longer
        describe an unread constant, either because something now reads it or
        because it is gone.
    """
    bare, imported, loose = read
    unread = {
        (module, name): place
        for (module, name), place in sorted(defined.items())
        if name not in bare.get(str(place[0]), frozenset())
        and (module, name) not in imported
        and name not in loose
    }
    dead = {key: place for key, place in unread.items() if key not in kept}
    return dead, sorted(
        f"{module}.{name}" for module, name in kept if (module, name) not in unread
    )


def main() -> int:
    """Report every private constant no file reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    defined = defined_constants()
    dead, stale = classify(defined, names_read(), KEPT)
    if not dead and not stale:
        print(f"No unread private constant among the {len(defined)} defined in src.")
        return 0
    if dead:
        print("::error::a private constant nothing reads - see below")
        print(f"{len(dead)} private constant(s) are defined and never read:")
        for (_module, name), (path, line) in dead.items():
            print(f"  {name}  <- {path.relative_to(ROOT)}:{line}")
        print(
            "  -> use it, delete it, or, if it must stay, add it to KEPT at the "
            "top of scripts/check_dead_constants.py, keyed by (module, name), "
            "with the reason."
        )
    for name in stale:
        print(f"::error::KEPT lists {name}, which is now read or no longer exists")
    return 1


if __name__ == "__main__":
    sys.exit(main())
