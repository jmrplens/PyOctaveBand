#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Packaging guarantees: PEP 561 typing marker shipped with the package, and
the generated PyPI long description in sync with the GitHub README.
"""

import pathlib
import sys
import tomllib

import phonometry

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import generate_pypi_readme


def test_py_typed_marker_is_shipped() -> None:
    pkg_dir = pathlib.Path(phonometry.__file__).parent
    assert (pkg_dir / "py.typed").exists(), "PEP 561 marker missing from package"


def test_pyproject_readme_is_the_generated_pypi_variant() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    assert pyproject["project"]["readme"] == "README_PYPI.md"


def test_pypi_readme_matches_generator() -> None:
    """README_PYPI.md is exactly what the generator derives from README.md."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    committed = (_ROOT / "README_PYPI.md").read_text(encoding="utf-8")
    assert committed == generate_pypi_readme.pypi_readme(readme), (
        "README_PYPI.md is stale; regenerate it with `make pypi-readme`"
    )


def test_pypi_readme_has_no_stripped_markup() -> None:
    """PyPI's renderer drops <picture>/<source>; none may reach the page."""
    committed = (_ROOT / "README_PYPI.md").read_text(encoding="utf-8")
    assert "<picture>" not in committed
    assert "<source" not in committed
    assert ".gif" not in committed, "animated GIFs stay out of the PyPI page"
