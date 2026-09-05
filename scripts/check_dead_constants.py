#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fail on a private module-level constant nothing reads.

A constant with a leading underscore and an upper-case name is a number or a
table lifted out of a standard so that the code that uses it can say where it
came from. One that nothing reads is either a leftover, or the trace of a
check that was planned and never written, and the second is the reason this
guard exists rather than a periodic sweep: the last audit turned up a
frequency-range warning that was never added and a docstring that promised a
default the code did not take.

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

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: Where a constant may be defined, and everything that may read one.
SOURCE = ROOT / "src" / "phonometry"
SEARCHED = ("src", "tests", "scripts")

#: Private constants that stay although nothing reads them, and why. Nothing
#: belongs here that could instead be used: a constant kept for documentation
#: is documentation, and belongs in a docstring where a reader will find it.
KEPT: dict[str, str] = {}


def defined_constants() -> dict[str, list[tuple[pathlib.Path, int]]]:
    """Every ``_UPPER_CASE`` assigned at module level under ``src``."""
    found: dict[str, list[tuple[pathlib.Path, int]]] = collections.defaultdict(list)
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            targets: list[ast.Name] = []
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            for target in targets:
                name = target.id
                private = name.startswith("_") and not name.startswith("__")
                if private and name.upper() == name:
                    found[name].append((path, node.lineno))
    return found


def names_read() -> collections.Counter[str]:
    """Every identifier read anywhere in the tree, however it is reached."""
    seen: collections.Counter[str] = collections.Counter()
    for directory in SEARCHED:
        for path in sorted((ROOT / directory).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - a file being edited
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    seen[node.id] += 1
                elif isinstance(node, ast.Attribute):
                    seen[node.attr] += 1
                elif isinstance(node, ast.alias):
                    seen[node.name.split(".")[-1]] += 1
                    if node.asname:
                        seen[node.asname] += 1
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    seen[node.value] += 1
    return seen


def main() -> int:
    """Report every private constant no file reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    defined = defined_constants()
    read = names_read()
    dead = {
        name: places
        for name, places in sorted(defined.items())
        if read[name] == 0 and name not in KEPT
    }
    stale = sorted(name for name in KEPT if read[name] > 0 or name not in defined)
    if not dead and not stale:
        print(f"No unread private constant among the {len(defined)} defined in src.")
        return 0
    if dead:
        print("::error::a private constant nothing reads - see below")
        print(f"{len(dead)} private constant(s) are defined and never read:")
        for name, places in dead.items():
            path, line = places[0]
            print(f"  {name}  <- {path.relative_to(ROOT)}:{line}")
        print(
            "  -> use it, delete it, or, if it must stay, add it to KEPT at the "
            "top of scripts/check_dead_constants.py with the reason."
        )
    for name in stale:
        print(f"::error::KEPT lists {name}, which is now read or no longer exists")
    return 1


if __name__ == "__main__":
    sys.exit(main())
