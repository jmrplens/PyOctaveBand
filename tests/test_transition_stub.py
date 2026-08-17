#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
The PyOctaveBand transition stub that lives under ``stub/``.

Almost nothing in the tree looks at it. It is not installed by the test
environment, it is not imported by the library, `mypy` was pointed at ``src``
and ``scripts``, and the publish workflow builds it straight from these files
on a manual dispatch. The lint step is the exception: it walks the tree from
the root, so it has always read the shim, though not its README or its pin,
which are prose and metadata rather than Python. So its README ships as the
rendered PyPI long description of PyOctaveBand without anything ever comparing
it to the shim it describes, and its dependency line ships as the resolver's
only instruction without anything ever asking which phonometry releases it
admits.
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
import shutil
import subprocess
import tomllib
import types
import warnings

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import phonometry

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_STUB = _ROOT / "stub"
_SHIM = _STUB / "src" / "pyoctaveband" / "__init__.py"
_README = _STUB / "README.md"
_PYPROJECT = _STUB / "pyproject.toml"
_WORKFLOW = _ROOT / ".github" / "workflows" / "python-app.yml"

# A claim that some gate never reached the shim, made in the same sentence as
# linting. The lint step has covered `stub/` since the shim was committed, so
# any such sentence is false; a sentence that says linting *does* cover it is
# not matched, and neither is a claim about the gates that really did miss it.
_UNLINTED_CLAIM = re.compile(r"\b(?:nothing|never|no)\b[^.;]*\blint", re.IGNORECASE)

# Any CamelCase name ending in "Warning", which is how the README refers to the
# class the shim raises. Deliberately not a search for one hard-coded name:
# the point is to read whatever the prose claims and compare it with reality.
_WARNING_CLASS = re.compile(r"\b([A-Z][A-Za-z]*Warning)\b")

# The migration notice, matched on its stable half so the assertion survives
# an edit to the wording.
_NOTICE = "renamed to 'phonometry'"


def _gate(tool: str) -> tuple[str, str]:
    """The CI command that invokes *tool*, and the comment introducing its step.

    Read as text rather than parsed as YAML: the comments are half of what the
    tests below check, and a YAML parser throws them away. The comment of a
    step is the run of ``#`` lines above it, whether it sits above the step's
    ``- name:`` or between that name and its ``run:``.
    """
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("run:"):
            continue
        command = stripped.removeprefix("run:").strip()
        if not command.startswith(tool):
            continue
        comment: list[str] = []
        for previous in reversed(lines[:index]):
            text = previous.strip()
            if text.startswith("#"):
                comment.append(text.lstrip("#").strip())
            elif not text.startswith("- name:"):
                break
        return command, " ".join(reversed(comment))
    raise AssertionError(f"no step in {_WORKFLOW.name} runs {tool}")


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
    assert spec is not None
    assert spec.loader is not None
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


def test_the_lint_gate_reaches_the_shim() -> None:
    """The one gate the shim was never outside of, asked rather than assumed.

    ``ruff check .`` discovers its own files by walking the tree, so the shim
    is covered only for as long as the root stays the argument and nothing
    excludes ``stub/``. Either could change without anyone noticing that the
    shim had quietly dropped out of the only gate that reads it. Measured by
    asking ruff which files the gate's own command visits.
    """
    command, _ = _gate("ruff")
    assert command == "ruff check .", (
        f"the lint gate is now `{command}`; confirm it still reaches the shim"
    )
    executable = shutil.which("ruff")
    if executable is None:  # pragma: no cover - ruff is a dev dependency
        pytest.skip("ruff is not installed")
    listing = subprocess.run(
        [executable, "check", ".", "--show-files"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    visited = {pathlib.Path(p).resolve() for p in listing.stdout.splitlines() if p}
    assert _SHIM.resolve() in visited, (
        f"`{command}` no longer checks {_SHIM.relative_to(_ROOT)}"
    )


def test_the_type_check_step_does_not_claim_the_shim_went_unlinted() -> None:
    """The comment explaining the mypy path list must not overstate the gap.

    It was written as `nothing here ever imported, linted or type checked it`,
    three lines under the ``ruff check .`` that had been linting the shim since
    the day it was committed. The gap is real for the other two, and naming a
    third gate that never had it makes the comment argue for the change on a
    false premise.
    """
    command, comment = _gate("mypy")
    assert "stub/src" in command, "the type check must still cover the shim"
    assert "stub/src" in comment, "the comment must still explain why it does"
    overstated = _UNLINTED_CLAIM.search(comment)
    assert overstated is None, (
        f"{_WORKFLOW.name} says '{overstated.group(0) if overstated else ''}', "
        f"but `{_gate('ruff')[0]}` above it already checks the shim"
    )
