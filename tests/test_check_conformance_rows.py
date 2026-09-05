#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The gate against a conformance row that never runs the library.

``scripts/check_conformance_rows.py`` walks every ``@register``-decorated row
in ``scripts/conformance/domains/`` and asks whether it reaches this library at
all. A row that does not is comparing a printed formula with a second copy of
itself, and reports Pass whatever the library does; three of those were found
in the VDI 2081 domain.

Answering that question naively -- grep for ``ph.`` inside the function body --
reports a hundred rows on this tree, of which ninety-five are false. A row
delegates: to a helper beside it, to one imported from ``..shared``, and to the
library under whatever name the module's imports gave it. The tests below fix
each of those three resolutions, and the two ways the escape hatch can rot.
"""

from __future__ import annotations

import ast
import pathlib
import sys

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_conformance_rows as ccr


def _package(**sources: str) -> ccr.Package:
    """A synthetic conformance package from ``module key -> source``."""
    return ccr.Package({key: ast.parse(src) for key, src in sources.items()})


def _example(body: str, *, imports: str = "import phonometry as ph") -> ccr.Package:
    """One domain module holding a single registered row with *body*."""
    return _package(
        **{
            "domains.example": f"{imports}\n\n"
            "@register(_D, 'title', 'detail')\n"
            f"def _chk_row():\n{body}\n"
        }
    )


def test_a_row_that_calls_the_library_reaches_it() -> None:
    """The ordinary shape: the row asks the library and judges the answer."""
    package = _example("    return numeric(1.0, ph.filters.thing(2.0), 1e-9)")
    assert package.reaches_library("domains.example", "_chk_row")


def test_a_row_that_computes_the_answer_itself_reaches_nothing() -> None:
    """The defect the gate exists for: a formula compared with itself."""
    package = _example(
        "    return numeric(9.2 + 0.765 * 150.0, 9.2 + 0.765 * 150.0, 0.01)"
    )
    assert not package.reaches_library("domains.example", "_chk_row")


def test_a_row_reaches_the_library_through_a_helper_beside_it() -> None:
    """Delegation within the module. Grepping the row's own body misses this."""
    package = _package(
        **{
            "domains.example": "import phonometry as ph\n\n"
            "def _run(x):\n    return ph.filters.thing(x)\n\n"
            "@register(_D, 't', 'd')\n"
            "def _chk_row():\n    return numeric(1.0, _run(2.0), 1e-9)\n"
        }
    )
    assert package.reaches_library("domains.example", "_chk_row")


def test_a_row_reaches_the_library_through_a_shared_helper() -> None:
    """Delegation across the package, which is where ``..shared`` lives."""
    package = _package(
        shared="import phonometry as ph\n\ndef band(x):\n    return ph.bands.of(x)\n",
        **{
            "domains.example": "from ..shared import band\n\n"
            "@register(_D, 't', 'd')\n"
            "def _chk_row():\n    return numeric(1.0, band(2.0), 1e-9)\n"
        },
    )
    assert package.reaches_library("domains.example", "_chk_row")


def test_a_shared_helper_imported_under_another_name_is_followed() -> None:
    """``from ..shared import band as run`` defines nothing called ``run``.

    Following the call with the local name finds no function over there and
    reports a row that does reach the library as one that does not, which
    fails CI on working code.
    """
    package = _package(
        shared="import phonometry as ph\n\ndef band(x):\n    return ph.bands.of(x)\n",
        **{
            "domains.example": "from ..shared import band as run\n\n"
            "@register(_D, 't', 'd')\n"
            "def _chk_row():\n    return numeric(1.0, run(2.0), 1e-9)\n"
        },
    )
    assert package.reaches_library("domains.example", "_chk_row")


def test_a_nested_helper_the_row_never_calls_does_not_count() -> None:
    """Dead code inside the row must not buy it a clean verdict.

    A nested function's body only runs if something calls it, so a row could
    otherwise define a helper that touches the library, never call it, and go
    on computing its own expected value: the defect this gate exists for,
    written one level down.
    """
    package = _example(
        "    def unused():\n"
        "        return ph.filters.thing(2.0)\n"
        "    return numeric(9.2, 9.2, 0.01)"
    )
    assert not package.reaches_library("domains.example", "_chk_row")


def test_a_nested_helper_the_row_does_call_still_counts() -> None:
    """Four rows on this tree define one and call it; they are not silent."""
    package = _example(
        "    def band():\n"
        "        return ph.filters.thing(2.0)\n"
        "    return numeric(1.0, band(), 1e-9)"
    )
    assert package.reaches_library("domains.example", "_chk_row")


def test_a_helper_that_reaches_nothing_does_not_launder_the_row() -> None:
    """Delegation is followed, not assumed: an inert helper stays inert."""
    package = _package(
        shared="def twice(x):\n    return 2.0 * x\n",
        **{
            "domains.example": "from ..shared import twice\n\n"
            "@register(_D, 't', 'd')\n"
            "def _chk_row():\n    return numeric(4.0, twice(2.0), 1e-9)\n"
        },
    )
    assert not package.reaches_library("domains.example", "_chk_row")


def test_mutually_recursive_helpers_terminate() -> None:
    """Two helpers that call each other must not hang the walk."""
    package = _package(
        **{
            "domains.example": "def a(x):\n    return b(x)\n\n"
            "def b(x):\n    return a(x)\n\n"
            "@register(_D, 't', 'd')\n"
            "def _chk_row():\n    return numeric(1.0, a(2.0), 1e-9)\n"
        }
    )
    assert not package.reaches_library("domains.example", "_chk_row")


def test_the_library_is_found_under_any_import_alias() -> None:
    """``ph`` is a convention, not a rule; three other spellings are in use."""
    for imports, call in (
        ("from phonometry import filters", "filters.thing(2.0)"),
        ("from phonometry.signals import levels", "levels.thing(2.0)"),
        ("import phonometry.aircraft as air", "air.thing(2.0)"),
        ("import phonometry", "phonometry.thing(2.0)"),
    ):
        package = _example(f"    return numeric(1.0, {call}, 1e-9)", imports=imports)
        assert package.reaches_library("domains.example", "_chk_row"), imports


def test_an_unrelated_import_is_not_the_library() -> None:
    """numpy and math are how a row does its own arithmetic, not the library."""
    package = _example(
        "    return numeric(1.0, math.hypot(190.0, 40.0), 1e-9)", imports="import math"
    )
    assert not package.reaches_library("domains.example", "_chk_row")


def test_only_registered_functions_are_rows() -> None:
    """A helper is not a row, however inert; only ``@register`` makes one."""
    package = _package(
        **{
            "domains.example": "def _helper():\n    return 1.0\n\n"
            "@register(_D, 't', 'd')\n"
            "def _chk_row():\n    return numeric(1.0, 1.0, 1e-9)\n"
        }
    )
    assert sorted(package.rows()) == [("domains.example", "_chk_row")]


def test_shared_module_rows_are_not_counted() -> None:
    """Only the domain modules register; a stray decorator elsewhere is not a row."""
    package = _package(
        shared="@register(_D, 't', 'd')\ndef _chk_elsewhere():\n    return None\n",
        **{
            "domains.example": "@register(_D, 't', 'd')\ndef _chk_row():\n    return None\n"
        },
    )
    assert sorted(package.rows()) == [("domains.example", "_chk_row")]


def test_a_silent_undeclared_row_is_reported() -> None:
    """The gate's verdict, with an empty hatch."""
    package = _example("    return numeric(1.0, 1.0, 1e-9)")
    silent, stale = ccr.classify(package, {})
    assert silent == ["example._chk_row"]
    assert stale == []


def test_a_silent_declared_row_is_allowed() -> None:
    """The hatch, for a row about the oracle or a dependency rather than us."""
    package = _example("    return numeric(1.0, 1.0, 1e-9)")
    silent, stale = ccr.classify(package, {"example._chk_row": "why it cannot"})
    assert silent == []
    assert stale == []


def test_a_declared_row_that_now_runs_the_library_is_stale() -> None:
    """The hatch cannot rot: an entry that stopped being needed says so."""
    package = _example("    return numeric(1.0, ph.filters.thing(2.0), 1e-9)")
    silent, stale = ccr.classify(package, {"example._chk_row": "why it cannot"})
    assert silent == []
    assert stale == [("example._chk_row", "now runs the library")]


def test_a_declared_row_that_no_longer_exists_is_stale() -> None:
    """The other way the hatch rots, and it reads differently in the report."""
    package = _example("    return numeric(1.0, ph.filters.thing(2.0), 1e-9)")
    silent, stale = ccr.classify(package, {"example._chk_gone": "a row that was"})
    assert silent == []
    assert stale == [("example._chk_gone", "is no longer a row")]


def test_the_real_package_passes() -> None:
    """The tree itself, which is the gate's actual job."""
    assert ccr.main([]) == 0


def test_every_declared_row_carries_a_reason() -> None:
    """An entry without prose is an entry nobody can review."""
    for label, reason in ccr.ORACLE_ONLY.items():
        assert len(reason) > 40, label
