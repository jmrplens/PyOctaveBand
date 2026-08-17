#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Packaging guarantees: PEP 561 typing marker shipped with the package, and
the generated PyPI long description in sync with the GitHub README.

The long description also has a property the README does not: PyPI freezes it
at upload and never lets it be edited again, so a link it carries to ``main``
goes stale the first time the tree moves under it. The description published
with 3.3.0 was measured with 34 of its 38 repository links dead, all of them
guides that had since been filed into subdirectories. The generator pins those
links to the release tag instead, and the tests below hold that pin: no ``main``
ref may reach the page, every ref must be the tag of the version in ``VERSION``,
and every path behind one must exist in this checkout, which is the tree the tag
will be cut from.
"""

import pathlib
import re
import sys
import tomllib

import phonometry

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import generate_pypi_readme

#: A URL into this repository on the PyPI page, whatever ref it names. Group
#: ``ref`` is the branch or tag, group ``path`` the repo-relative path, with
#: any anchor or query left off.
_REPO_URL = re.compile(
    r"https://(?:github\.com/jmrplens/phonometry/blob"
    r"|raw\.githubusercontent\.com/jmrplens/phonometry)"
    r"/(?P<ref>[^/]+)/(?P<path>[^)\"'\s#?]+)"
)


def _committed_pypi_readme() -> str:
    return (_ROOT / "README_PYPI.md").read_text(encoding="utf-8")


def test_py_typed_marker_is_shipped() -> None:
    pkg_dir = pathlib.Path(phonometry.__file__).parent
    assert (pkg_dir / "py.typed").exists(), "PEP 561 marker missing from package"


def test_pyproject_readme_is_the_generated_pypi_variant() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["project"]["readme"] == "README_PYPI.md"


def test_pypi_readme_matches_generator() -> None:
    """README_PYPI.md is exactly what the generator derives from README.md.

    The generator now reads ``VERSION`` too, so this is also the gate that
    makes a version bump regenerate the page: bump without regenerating and
    every pinned link still names the previous tag, and this fails.
    """
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    tag = f"v{generate_pypi_readme.version()}"
    assert _committed_pypi_readme() == generate_pypi_readme.pypi_readme(readme, tag), (
        "README_PYPI.md is stale; regenerate it with `make pypi-readme`"
    )


def test_pypi_readme_has_no_stripped_markup() -> None:
    """PyPI's renderer drops <picture>/<source>; none may reach the page."""
    committed = _committed_pypi_readme()
    assert "<picture>" not in committed
    assert "<source" not in committed
    assert ".gif" not in committed, "animated GIFs stay out of the PyPI page"


def test_pypi_readme_pins_every_repository_link_to_the_release_tag() -> None:
    """No link on the frozen page may resolve against a moving branch.

    A ``blob/main`` URL is right on the day of upload and wrong forever after,
    because the description cannot be re-edited while ``main`` can. Every one
    of them has to name the tag of the version being published.
    """
    tag = f"v{generate_pypi_readme.version()}"
    refs = {match["ref"] for match in _REPO_URL.finditer(_committed_pypi_readme())}
    assert refs, "no repository links found; the pattern stopped matching"
    assert refs == {tag}, (
        f"README_PYPI.md resolves repository links against {sorted(refs - {tag})}; "
        f"a published description is frozen, so every one must name {tag}"
    )


def test_pinned_pypi_links_resolve_in_this_checkout() -> None:
    """Every pinned path exists in the tree the tag will be cut from.

    The tag itself does not exist until the release workflow creates it, so
    there is a window on ``main`` where these URLs 404 over HTTP by design.
    Checking the working tree instead verifies the same thing without waiting
    for the release: what the tag will point at is right here.
    """
    dangling = sorted(
        match["path"]
        for match in _REPO_URL.finditer(_committed_pypi_readme())
        if not (_ROOT / match["path"]).exists()
    )
    assert not dangling, (
        "README_PYPI.md links to paths that are not in this checkout, so they "
        f"will not be in the release tag either: {dangling}"
    )


def test_pinning_leaves_unrelated_urls_alone() -> None:
    """Only the ref moves: other hosts, other refs and the path are untouched."""
    text = (
        "[guide](https://github.com/jmrplens/phonometry/blob/main/docs/main/x.md) "
        "![img](https://raw.githubusercontent.com/jmrplens/phonometry/main/a.svg) "
        "[ci](https://github.com/jmrplens/phonometry/actions/workflows/main.yml) "
        "[badge](https://img.shields.io/pypi/v/phonometry?logo=main) "
        "[tagged](https://github.com/jmrplens/phonometry/blob/v1.0.0/LICENSE)"
    )
    assert generate_pypi_readme.pin_to_tag(text, "v9.9.9") == (
        "[guide](https://github.com/jmrplens/phonometry/blob/v9.9.9/docs/main/x.md) "
        "![img](https://raw.githubusercontent.com/jmrplens/phonometry/v9.9.9/a.svg) "
        "[ci](https://github.com/jmrplens/phonometry/actions/workflows/main.yml) "
        "[badge](https://img.shields.io/pypi/v/phonometry?logo=main) "
        "[tagged](https://github.com/jmrplens/phonometry/blob/v1.0.0/LICENSE)"
    )
