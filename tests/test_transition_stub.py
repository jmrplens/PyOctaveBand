#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
The PyOctaveBand transition stub that lives under ``stub/``.

It is the one package in the tree nothing else looks at. It is not installed
by the test environment, it is not imported by the library, `mypy` was pointed
at ``src`` and ``scripts``, and the publish workflow builds it straight from
these files on a manual dispatch. So its README ships as the rendered PyPI
long description of PyOctaveBand without anything ever comparing it to the
shim it describes, and its dependency line ships as the resolver's only
instruction without anything ever asking which phonometry releases it admits.
Both were wrong: the README named a warning class the shim does not raise, and
the unbounded pin would have carried the next major into a shim whose promise
that release breaks.

These tests execute the shipped shim and hold the published prose and the
published pin to what it actually does.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import tomllib
import types
import warnings

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import phonometry

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_STUB = _ROOT / "stub"
_SHIM = _STUB / "src" / "pyoctaveband" / "__init__.py"
_README = _STUB / "README.md"
_PYPROJECT = _STUB / "pyproject.toml"

# Any CamelCase name ending in "Warning", which is how the README refers to the
# class the shim raises. Deliberately not a search for one hard-coded name:
# the point is to read whatever the prose claims and compare it with reality.
_WARNING_CLASS = re.compile(r"\b([A-Z][A-Za-z]*Warning)\b")

# The migration notice, matched on its stable half so the assertion survives
# an edit to the wording.
_NOTICE = "renamed to 'phonometry'"


def _stub_project() -> dict[str, object]:
    with _PYPROJECT.open("rb") as handle:
        project: dict[str, object] = tomllib.load(handle)["project"]
    return project


def _phonometry_requirement() -> Requirement:
    dependencies = _stub_project()["dependencies"]
    assert isinstance(dependencies, list)
    requirements = [Requirement(str(item)) for item in dependencies]
    named = [req for req in requirements if req.name == "phonometry"]
    assert len(named) == 1, "the stub must depend on phonometry exactly once"
    return named[0]


def _import_shim() -> tuple[types.ModuleType, list[warnings.WarningMessage]]:
    """Execute the shipped shim and return it together with what it warned.

    Loaded from its path under a private name rather than imported by name:
    the stub is not installed in the test environment, and a real import would
    be cached in ``sys.modules``, so the module body would run once per session
    and only the first test to ask would see the warning.
    """
    spec = importlib.util.spec_from_file_location(
        "_pyoctaveband_transition_shim", _SHIM, submodule_search_locations=[]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec.loader.exec_module(module)
    return module, list(caught)


def test_shim_raises_one_migration_notice_and_it_is_a_futurewarning() -> None:
    """The class matters: DeprecationWarning is hidden from users by default."""
    _, caught = _import_shim()
    notices = [w for w in caught if _NOTICE in str(w.message)]
    assert len(notices) == 1, "importing the shim must warn exactly once"
    assert notices[0].category is FutureWarning


def test_readme_names_the_warning_class_the_shim_actually_raises() -> None:
    """The published page must not tell a user to filter the wrong class.

    ``stub/pyproject.toml`` makes this README the long description, so a reader
    who acts on it types ``warnings.simplefilter("ignore", <class>)``. Named
    wrongly, that call is a silent no-op.
    """
    _, caught = _import_shim()
    raised = {w.category.__name__ for w in caught if _NOTICE in str(w.message)}
    named = set(_WARNING_CLASS.findall(_README.read_text(encoding="utf-8")))
    assert named, "the README must name the warning class the shim raises"
    assert named == raised, (
        f"stub/README.md names {sorted(named)}, the shim raises {sorted(raised)}"
    )


def test_stub_readme_is_the_published_long_description() -> None:
    """What makes the assertion above load-bearing rather than decorative."""
    assert _stub_project()["readme"] == "README.md"


def test_shim_reexports_every_name_phonometry_publishes() -> None:
    """The promise the stub is built on: the old name still resolves them all."""
    module, _ = _import_shim()
    missing = [name for name in phonometry.__all__ if not hasattr(module, name)]
    assert not missing, f"the shim does not re-export {missing}"
    assert module.__version__ == phonometry.__version__


def test_stub_pin_excludes_the_major_that_may_break_its_promise() -> None:
    """The stub is a 3.x transition package and its pin has to say so.

    Its README and the 3.0.0 changelog entry promise that ``pip install -U
    PyOctaveBand`` keeps existing code working and that renaming the import is
    a complete migration. That promise holds only for as long as the resolved
    phonometry still exports the names the last release under the old name
    exported, and a major release is exactly where those go: 4.0 removes the
    ``octavefilter``, ``getansifrequencies``, ``normalizedfreq`` and
    ``calculate_sensitivity`` aliases that are in the 3.x ``__all__``. Without
    a ceiling the resolver would hand a 2.x stub a phonometry that raises
    ``AttributeError`` for them, with no diagnostic naming the cause.
    """
    specifier = _phonometry_requirement().specifier
    assert specifier.contains(Version("3.0.0")), "the stub needs the 3.0 API"
    assert specifier.contains(Version("3.99.0")), "every 3.x must stay eligible"
    assert not specifier.contains(Version("4.0.0")), (
        "stub/pyproject.toml must cap phonometry below the next major"
    )
    assert not specifier.contains(Version("4.0.0rc1")), (
        "a pre-release of the next major must be excluded too"
    )


def test_tree_version_still_falls_inside_the_pin_the_stub_publishes() -> None:
    """A forcing gate for the day the ceiling above starts to bite.

    When this tree becomes the major the stub excludes, the published
    PyOctaveBand stops tracking the library and its README has to be rewritten
    to say so, or the stub has to be republished against the new line. Neither
    is a decision to discover from a user's traceback.
    """
    specifier: SpecifierSet = _phonometry_requirement().specifier
    assert specifier.contains(Version(phonometry.__version__)), (
        f"phonometry {phonometry.__version__} is outside the transition stub's "
        f"pin '{specifier}'. Decide what PyOctaveBand should now install and "
        f"say it in stub/README.md before releasing."
    )
