#!/usr/bin/env python3
#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fail on a conformance row that certifies the library without running it.

A row in ``scripts/conformance/domains/`` states what a standard prints and
what this library computes, and passes when the two agree. A row that computes
the expected value *itself*, from the same constants, is fixed arithmetic: it
compares a formula with itself and reports Pass whatever the library does. Three
of them were found that way, for VDI 2081 Equations (16), (17) and (34), and
``flow_noise_straight_duct`` could have been given a wrong sign and all three
would still have reported Pass against the printed page.

Counting them is not as simple as grepping for ``ph.``. A row usually delegates:
to a helper beside it, to one imported from ``..shared``, or to the library
under a name that is not ``ph`` (``from phonometry import filters``). Resolving
those three takes the count on this tree from 102 to 7, and the 102 is what a
naive scan reports, so the naive scan is worse than nothing -- it buries the
seven that matter under ninety-five that do not.

Not every row can run the library, and those that cannot are not defects:

* a row that pins the **geometry of a test case**, like the 194,16 m ground
  projection ISO/TR 17534-3 prints for its seven cases, checks that this
  project read the coordinates right. The library does not own the distance
  between two points;
* a row that checks the **standard against itself**, like a printed total
  against the printed parts it is the sum of, is testing the oracle before
  the oracle is trusted;
* a row that pins a **dependency's** published constant is deliberately not
  about this library at all.

:data:`ORACLE_ONLY` is where those are declared, and an entry carries the
reason. A row that is neither reachable nor declared fails, and a declared row
that has since learned to call the library fails too, so the list cannot rot.

Usage::

    python scripts/check_conformance_rows.py

Exit status 0 when every row either runs the library or says why it does not.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "scripts" / "conformance"

#: Rows that certify something other than this library's own computation, and
#: what each of them is for. Nothing belongs here that could instead call the
#: library: the point of a row is that a defect can fail it.
ORACLE_ONLY: dict[str, str] = {
    "iso17534._chk_dp": (
        "the ground projection of the shared test-case geometry, 194,16 m, "
        "against the coordinates of Table 1: this pins the reading of the "
        "case, not a law the library owns"
    ),
    "iso17534._chk_d3": "the straight-line length of the same geometry, likewise",
    "iso17534._chk_d3_sloped": "the same length once the ground rises, likewise",
    "iso17534._chk_q": (
        "q = 1 - 30 (hs + hr) / dp against the printed 0,23. The library "
        "computes q inside ground_attenuation and does not publish it, and "
        "the rows that do exercise that function are the A_gr ones; this one "
        "pins which of the two footnotes of ISO 9613-2 Table 3 the printed "
        "value comes from, which is the errata entry's evidence"
    ),
    "aircraft._chk_doc29_event_assembly": (
        "the printed B-1 total against the energy sum of the 29 printed "
        "segment SELs: the oracle checked for self-consistency before it is "
        "used, since no worked example exists for the per-event quantities"
    ),
    "signal_analysis._chk_multitaper_dpss_eigenvalue": (
        "SciPy's DPSS eigenvalue against the published constant: a check of "
        "the dependency the multitaper estimator is built on, deliberately "
        "not of this library"
    ),
}


def _module_key(path: pathlib.Path) -> str:
    """``domains.vdi2081`` or ``shared``, the way an import names it."""
    stem = path.stem
    return f"domains.{stem}" if path.parent.name == "domains" else stem


def _parse_package() -> dict[str, ast.Module]:
    """Every module of the conformance package, keyed as an import names it."""
    paths = [*PACKAGE.glob("*.py"), *(PACKAGE / "domains").glob("*.py")]
    return {
        _module_key(path): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(paths)
    }


def library_names(tree: ast.Module) -> set[str]:
    """Every local name in *tree* that is this library, however imported.

    ``import phonometry as ph``, ``from phonometry import filters`` and
    ``from phonometry.signals import levels`` all bind the library under a
    different name, and a row reaches it through any of them.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
                if alias.name == "phonometry" or alias.name.startswith("phonometry.")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "phonometry" or module.startswith("phonometry."):
                names.update(alias.asname or alias.name for alias in node.names)
    return names


def sibling_helpers(tree: ast.Module, known: set[str]) -> dict[str, tuple[str, str]]:
    """Local name to ``(module key, defined name)``, for package-local helpers.

    Both halves are needed. ``from ..shared import band as run`` binds ``run``
    here and defines nothing of that name over there, so following the call
    with the local name finds no function and reports a row that does reach
    the library as one that does not.
    """
    found: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.level):
            continue
        target = (node.module or "").lstrip(".")
        key = target if target in known else f"domains.{target}"
        if key in known:
            found.update(
                {alias.asname or alias.name: (key, alias.name) for alias in node.names}
            )
    return found


def _own_scope(function: ast.FunctionDef) -> Iterator[ast.AST]:
    """Every node of *function*'s own body, not descending into a nested def.

    ``ast.walk`` walks into a nested function, whose body only runs if
    something calls it. A row could then define an unused helper that touches
    the library and pass while computing its own answer, which is the whole
    defect this gate exists for, written one level down.
    """
    stack: list[ast.AST] = list(function.body) + list(function.decorator_list)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


class Package:
    """The conformance package, ready to answer what a row reaches."""

    def __init__(self, modules: dict[str, ast.Module]) -> None:
        # ``ast.walk``, so a function defined inside another is in the table
        # under its own name: :meth:`reaches_library` does not walk into it,
        # and reaches it the same way it reaches any helper, by the call.
        self.functions = {
            key: {
                node.name: node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            }
            for key, tree in modules.items()
        }
        self.library = {key: library_names(tree) for key, tree in modules.items()}
        self.helpers = {
            key: sibling_helpers(tree, set(modules)) for key, tree in modules.items()
        }

    def reaches_library(
        self, module: str, name: str, seen: set[tuple[str, str]] | None = None
    ) -> bool:
        """Whether *name* touches the library, directly or through a helper.

        A function defined *inside* this one is not walked as part of it: a row
        that defines a helper it never calls does not run that helper, and
        counting its body would let a row buy itself a clean verdict with dead
        code. It is reachable the same way any other helper is, by being
        called, and the four rows that do define one all call it.
        """
        seen = seen if seen is not None else set()
        if (module, name) in seen or name not in self.functions.get(module, {}):
            return False
        seen.add((module, name))
        for node in _own_scope(self.functions[module][name]):
            if isinstance(node, ast.Name) and node.id in self.library[module]:
                return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callee = node.func.id
                if self.reaches_library(module, callee, seen):
                    return True
                where = self.helpers[module].get(callee)
                if where and self.reaches_library(where[0], where[1], seen):
                    return True
        return False

    def rows(self) -> Iterator[tuple[str, str]]:
        """Every ``@register``-decorated function, as module key and name."""
        for module, functions in self.functions.items():
            if not module.startswith("domains."):
                continue
            for name, node in functions.items():
                if any(
                    isinstance(dec, ast.Call)
                    and getattr(dec.func, "id", "") == "register"
                    for dec in node.decorator_list
                ):
                    yield module, name


def classify(
    package: Package, declared: dict[str, str]
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split the rows into the undeclared silent ones and the stale entries.

    :param package: The parsed conformance package.
    :param declared: The escape hatch, ``domain._chk_name`` to reason.
    :return: The rows that reach nothing and are not declared, and the entries
        that no longer earn their place, each with what became of its row.
    """
    silent: list[str] = []
    reaching: set[str] = set()
    for module, name in sorted(package.rows()):
        label = f"{module.removeprefix('domains.')}.{name}"
        if package.reaches_library(module, name):
            reaching.add(label)
        elif label not in declared:
            silent.append(label)
    known = _all(package)
    stale = [
        (label, "now runs the library" if label in reaching else "is no longer a row")
        for label in sorted(declared)
        if label in reaching or label not in known
    ]
    return silent, stale


def _all(package: Package) -> set[str]:
    """Every row's label, whether or not it reaches anything."""
    return {
        f"{module.removeprefix('domains.')}.{name}" for module, name in package.rows()
    }


def main(argv: list[str] | None = None) -> int:
    """Report every row that certifies the library without running it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    package = Package(_parse_package())
    total = len(_all(package))
    silent, stale = classify(package, ORACLE_ONLY)
    if not silent and not stale:
        print(
            f"All {total} conformance rows run the library, "
            f"except the {len(ORACLE_ONLY)} that say why they cannot."
        )
        return 0
    if silent:
        print("::error::a conformance row that never runs the library - see below")
        for label in silent:
            print(f"  {label}")
        print(
            "  -> call the library so a defect in it can fail the row, or, if "
            "the row is about the oracle or a dependency rather than about "
            "this library, add it to ORACLE_ONLY at the top of "
            "scripts/check_conformance_rows.py with the reason."
        )
    for label, why in stale:
        print(f"::error::ORACLE_ONLY lists {label}, which {why}: drop the entry")
    return 1


if __name__ == "__main__":
    sys.exit(main())
