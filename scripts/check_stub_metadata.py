#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Hold the built PyOctaveBand stub to the pin its README promises.

``stub/`` builds the final ``PyOctaveBand`` release: a shim that installs
``phonometry`` and re-exports it under the old ``pyoctaveband`` module name.
Its whole promise is that ``pip install -U PyOctaveBand`` keeps existing code
working and that renaming the import is the entire migration. That holds only
while the resolved ``phonometry`` still exports the names the last release
under the old name exported, and a major release is exactly where those go, so
the requirement has to be capped below the next major.

``tests/test_transition_stub.py`` asserts the cap in ``stub/pyproject.toml``.
This asserts it in the artifact, because ``pyproject.toml`` is not what a
resolver reads: pip reads ``Requires-Dist`` out of the built metadata, and a
build backend or a packaging config can put something else there. The two
checks are deliberately at opposite ends of the build.

Run after ``python -m build stub/``::

    python scripts/check_stub_metadata.py

CI runs it in the ``transition-stub`` job, right after ``twine check``.
"""

from __future__ import annotations

import pathlib
import sys
import tarfile
import zipfile
from email import message_from_string
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.version import Version

if TYPE_CHECKING:
    from email.message import Message

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DIST = _ROOT / "stub" / "dist"

#: The first version the stub must refuse. Everything below it is a release of
#: the line whose API the shim re-exports; this one may have retired a name.
_NEXT_MAJOR = Version("4.0.0")

#: Excluded as well, because pip will install a pre-release of the excluded
#: major when a specifier's upper bound admits one.
_NEXT_MAJOR_PRERELEASE = Version("4.0.0rc1")

#: The oldest release that carries the API the shim re-exports.
_OLDEST_SUPPORTED = Version("3.0.0")

#: A late 3.x, to catch a cap that shuts the door on the line it should keep
#: open (``==3.0.*``, say, or a stray ``<3.1``).
_LATEST_SUPPORTED_LINE = Version("3.99.0")


def _metadata_of_wheel(path: pathlib.Path) -> Message:
    with zipfile.ZipFile(path) as archive:
        names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise SystemExit(f"{path.name}: expected one METADATA, found {names}")
        return message_from_string(archive.read(names[0]).decode("utf-8"))


def _metadata_of_sdist(path: pathlib.Path) -> Message:
    with tarfile.open(path) as archive:
        names = [name for name in archive.getnames() if name.endswith("/PKG-INFO")]
        # The top-level PKG-INFO, not one belonging to a vendored tree.
        names = [name for name in names if name.count("/") == 1]
        if len(names) != 1:
            raise SystemExit(f"{path.name}: expected one PKG-INFO, found {names}")
        handle = archive.extractfile(names[0])
        if handle is None:
            raise SystemExit(f"{path.name}: {names[0]} is not a regular file")
        return message_from_string(handle.read().decode("utf-8"))


def _phonometry_requirement(metadata: Message, label: str) -> Requirement:
    declared = [
        Requirement(str(value)) for value in metadata.get_all("Requires-Dist", [])
    ]
    named = [item for item in declared if item.name == "phonometry"]
    if len(named) != 1:
        raise SystemExit(
            f"{label}: expected exactly one phonometry requirement, "
            f"found {[str(item) for item in named]}"
        )
    return named[0]


def _problems(requirement: Requirement, label: str) -> list[str]:
    specifier = requirement.specifier
    found: list[str] = []
    if not specifier.contains(_OLDEST_SUPPORTED):
        found.append(
            f"{label}: '{requirement}' excludes phonometry {_OLDEST_SUPPORTED}, "
            f"the oldest release that carries the API the shim re-exports."
        )
    if not specifier.contains(_LATEST_SUPPORTED_LINE):
        found.append(
            f"{label}: '{requirement}' excludes phonometry "
            f"{_LATEST_SUPPORTED_LINE}; every release of that line must stay "
            f"eligible."
        )
    for rejected, what in (
        (_NEXT_MAJOR, "the next major"),
        (_NEXT_MAJOR_PRERELEASE, "a pre-release of the next major"),
    ):
        if specifier.contains(rejected, prereleases=True):
            found.append(
                f"{label}: '{requirement}' admits phonometry {rejected} ({what}). "
                f"The stub would then re-export a phonometry that may have "
                f"retired names 'import pyoctaveband' promises, and user code "
                f"would raise AttributeError with nothing naming the cause. "
                f"Cap the requirement in stub/pyproject.toml."
            )
    return found


def main() -> int:
    artifacts = sorted(_DIST.glob("*.whl")) + sorted(_DIST.glob("*.tar.gz"))
    if not artifacts:
        raise SystemExit(
            f"no built artifacts in {_DIST.relative_to(_ROOT)}; "
            f"run 'python -m build stub/' first"
        )

    problems: list[str] = []
    for artifact in artifacts:
        label = artifact.name
        metadata = (
            _metadata_of_wheel(artifact)
            if artifact.suffix == ".whl"
            else _metadata_of_sdist(artifact)
        )
        requirement = _phonometry_requirement(metadata, label)
        print(f"{label}: Requires-Dist: {requirement}")
        problems.extend(_problems(requirement, label))

    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
