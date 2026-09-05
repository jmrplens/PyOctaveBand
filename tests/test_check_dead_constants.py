#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The dead-constant gate, and the escape hatch it has to leave usable.

``scripts/check_dead_constants.py`` reports a private module-level constant
nothing reads, and :data:`~check_dead_constants.KEPT` is how a constant that
must stay says so. A first version folded the hatch into the deadness test
itself, which made every entry report as stale the moment it was added: the
name was excluded from the dead set, so the subtraction that built the live
set handed it back. The hatch was therefore unusable, and silently so, since
it was empty. These tests fix the four answers the split has to give.
"""

from __future__ import annotations

import ast
import pathlib
import sys

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_dead_constants as cdc

_PLACE = (pathlib.Path("src/phonometry/thing.py"), 12)
_KEY = ("phonometry.thing", "_UNREAD")
_DEFINED = {_KEY: _PLACE}
#: Nothing reads anything: no bare name, no import, no attribute or string.
_NOTHING_READ: tuple[dict[str, set[str]], set[tuple[str, str]], set[str]] = (
    {},
    set(),
    set(),
)


def test_unread_and_not_kept_is_dead() -> None:
    """The defect the gate exists for."""
    dead, stale = cdc.classify(_DEFINED, _NOTHING_READ, {})
    assert dead == _DEFINED
    assert stale == []


def test_unread_and_kept_is_neither() -> None:
    """The whole point of the hatch: an entry with a reason silences one name."""
    dead, stale = cdc.classify(_DEFINED, _NOTHING_READ, {_KEY: "why it stays"})
    assert dead == {}
    assert stale == []


def test_the_hatch_silences_one_module_and_not_another() -> None:
    """Keyed by the pair: two modules may define the same private name."""
    other = ("phonometry.elsewhere", "_UNREAD")
    defined = {**_DEFINED, other: (pathlib.Path("src/phonometry/elsewhere.py"), 3)}
    dead, stale = cdc.classify(defined, _NOTHING_READ, {_KEY: "why it stays"})
    assert dead == {other: defined[other]}
    assert stale == []


def test_kept_name_that_something_now_reads_is_stale() -> None:
    """A constant that grew a reader no longer needs the hatch."""
    read = ({}, set(), {"_UNREAD"})
    dead, stale = cdc.classify(_DEFINED, read, {_KEY: "why it stays"})
    assert dead == {}
    assert stale == ["phonometry.thing._UNREAD"]


def test_kept_name_that_no_longer_exists_is_stale() -> None:
    """A constant that was deleted leaves its entry behind."""
    gone = ("phonometry.thing", "_GONE")
    dead, stale = cdc.classify({}, _NOTHING_READ, {gone: "why it stayed"})
    assert dead == {}
    assert stale == ["phonometry.thing._GONE"]


def test_a_constant_under_if_or_try_is_still_at_module_scope() -> None:
    """Module scope is not ``tree.body``: an if, a try or a with does not leave it."""
    tree = ast.parse(
        "import sys\n"
        "if sys.version_info >= (3, 14):\n"
        "    _GUARDED = 1\n"
        "try:\n"
        "    _FIRST, _SECOND = 1, 2\n"
        "except ImportError:\n"
        "    _FALLBACK = 3\n"
    )
    bound = {
        name.id
        for node in cdc._module_level(tree.body)
        if isinstance(node, ast.Assign)
        for target in node.targets
        for name in cdc._assigned_names(target)
    }
    assert bound == {"_GUARDED", "_FIRST", "_SECOND", "_FALLBACK"}


def test_a_package_init_is_its_own_package() -> None:
    """A relative import in an ``__init__.py`` climbs from the package it is."""
    init = cdc.ROOT / "src" / "phonometry" / "vibration" / "__init__.py"
    assert cdc._package_of(init, "phonometry.vibration") == "phonometry.vibration"
    module = cdc.ROOT / "src" / "phonometry" / "vibration" / "evaluation.py"
    assert (
        cdc._package_of(module, "phonometry.vibration.evaluation")
        == "phonometry.vibration"
    )


def test_a_bare_read_is_credited_to_its_own_file_only() -> None:
    """The per-file scope of a bare name, which is what makes the scan sound."""
    read = ({"src/phonometry/other.py": {"_UNREAD"}}, set(), set())
    dead, _stale = cdc.classify(_DEFINED, read, {})
    assert dead == _DEFINED
    read_here = ({str(_PLACE[0]): {"_UNREAD"}}, set(), set())
    dead_here, _ = cdc.classify(_DEFINED, read_here, {})
    assert dead_here == {}
