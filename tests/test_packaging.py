#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Packaging guarantees: PEP 561 typing marker shipped with the package, and
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

Those three hold the page PyPI serves, not the copy committed here. Between
releases ``VERSION`` still names the last release, so the committed copy pins
to a tag that already exists and whose tree predates the README it was
generated from: 86 of its 95 paths are absent from ``v3.3.0``. That state
cannot be gated without giving up the property that matters, so the page says
so in its own header and a test below holds the sentence there.

The last test guards the other thing the install prose quoted from outside
this repository: the NumPy ceiling numba declares. The pages said numba
declares ``numpy<2.5``, so ``phonometry[full]`` "resolves NumPy below 2.5
while a plain install gets the newest release", and advised installing
``phonometry[plot,report]`` to keep NumPy current. numba 0.67.0 declares
``numpy<2.6``, NumPy is at 2.5.2, and ``pip install --dry-run
phonometry[full]`` resolves numba 0.67.0 with NumPy 2.5.2, so the ceiling,
the consequence and the workaround were all wrong at once. The number came
from another project's metadata, which nothing here regenerates and nothing
here can gate, so the prose keeps the mechanism and drops the number.
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

#: Every authored copy of the ``[full]`` NumPy caveat, in both languages, plus
#: the generator that writes it into the llms artifacts. README_PYPI.md,
#: llms.txt, llms-full.txt and site/public/llms/ are derived from these, so
#: holding the sources holds all nine copies.
_INSTALL_CAVEAT_PAGES = (
    "README.md",
    "docs/start/getting-started.md",
    "site/src/content/docs/start/getting-started.mdx",
    "site/src/content/docs/es/start/getting-started.mdx",
    "scripts/generate_llms.py",
)

#: The opening of that caveat in each language, so the test cannot pass by the
#: paragraph having been deleted.
_CAVEAT_OPENINGS = ("One caveat about `[full]`", "Un matiz sobre `[full]`")

#: A NumPy version bound stated in prose: a requirement specifier, or the same
#: thing spelled out in English or Spanish.
_NUMPY_BOUND = re.compile(
    r"numpy\s*(?:[<>]=?|[=!~]=)\s*\d"
    r"|numpy\s+(?:below|under|above|over|por debajo de|por encima de)\s+\d",
    re.IGNORECASE,
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

    This is the release-time guarantee and nothing wider. At the release commit
    the tag does not exist yet, so checking the working tree verifies what the
    tag will point at without waiting for the workflow to create it; for the
    rest of the cycle the tag does exist, and this says nothing about it, since
    the tree it names is the *previous* release and is not what the next page
    will be published from. Resolving these paths against the pinned tag would
    therefore fail on ``main`` almost always while telling us nothing about the
    page being published.
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


def test_pypi_readme_header_warns_that_its_links_are_dead_on_main() -> None:
    """The committed page tells its reader the pinned links do not resolve.

    The pin is right on the page PyPI serves and wrong in the repository for
    the whole stretch between releases: ``VERSION`` still names the last
    release, so the tag exists and its tree predates the README the page was
    generated from. Resolving the committed page against ``git ls-tree -r
    v3.3.0`` puts 86 of its 95 paths outside that tree, the banner at the top
    of the page included, and the checks above cannot see it because they ask
    about the tree the *next* tag is cut from. So the page carries the warning
    itself, and this holds it there.
    """
    header = _committed_pypi_readme().split("-->", 1)[0]
    assert "do not resolve from main" in header, (
        "README_PYPI.md must warn that its links are pinned to the release tag "
        "and not to main, or a reader follows one from the repository and gets "
        "a 404"
    )


def test_release_procedure_regenerates_the_pypi_page() -> None:
    """The documented release steps include the regeneration a bump needs.

    ``VERSION`` is an input to this page now, so a release PR that bumps it and
    stops there leaves every pinned link naming the previous tag. Nothing
    closes that gap on its own: no workflow calls the generator, the pre-commit
    hook rewrites only ``docs/CONFORMANCE.md``, and the pull request template
    asks for ``make pypi-readme`` "after editing README.md", which a
    version-only bump does not do. It fails safe, because the two tests above
    turn the release PR red, but the checklist a maintainer follows has to name
    the step rather than leave it to be discovered from a failure.
    """
    contributing = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    releasing = contributing.split("## 📦 Releasing", 1)[-1].split("\n## ", 1)[0]
    assert "make pypi-readme" in releasing, (
        "CONTRIBUTING.md's release procedure must ask for `make pypi-readme`; "
        "bumping VERSION without it pins the published page to the previous tag"
    )


def test_the_full_extra_caveat_quotes_no_numpy_version() -> None:
    """The install caveat states the mechanism, never a version number.

    Any number here is a copy of numba's declared NumPy ceiling. This
    repository does not own it, does not regenerate it and has no gate that
    could notice it moving, so it goes stale silently and takes the advice
    around it with it: the ceiling numba declared moved to ``<2.6``, and the
    pages went on telling readers to give up numba to keep NumPy current when
    ``[full]`` was already resolving the newest NumPy there is.
    """
    for name in _INSTALL_CAVEAT_PAGES:
        text = (_ROOT / name).read_text(encoding="utf-8")
        assert any(opening in text for opening in _CAVEAT_OPENINGS), (
            f"{name} no longer carries the `[full]` caveat; the mirrored copies "
            "must stay in step"
        )
        quoted = _NUMPY_BOUND.search(text)
        assert quoted is None, (
            f"{name} quotes a NumPy bound ({quoted.group(0)!r}) that belongs to "
            "another project's metadata; nothing here regenerates it, so state "
            "the mechanism instead of the number"
        )


def test_pinning_leaves_unrelated_urls_alone() -> None:
    """Only the ref moves: other hosts, other refs and the path are untouched."""
    text = (
        "[guide](https://github.com/jmrplens/phonometry/blob/main/docs/main/x.md) "
        "[dir](https://github.com/jmrplens/phonometry/tree/main/docs/) "
        "![img](https://raw.githubusercontent.com/jmrplens/phonometry/main/a.svg) "
        "[ci](https://github.com/jmrplens/phonometry/actions/workflows/main.yml) "
        "[badge](https://img.shields.io/pypi/v/phonometry?logo=main) "
        "[tagged](https://github.com/jmrplens/phonometry/blob/v1.0.0/LICENSE)"
    )
    assert generate_pypi_readme.pin_to_tag(text, "v9.9.9") == (
        "[guide](https://github.com/jmrplens/phonometry/blob/v9.9.9/docs/main/x.md) "
        "[dir](https://github.com/jmrplens/phonometry/tree/v9.9.9/docs/) "
        "![img](https://raw.githubusercontent.com/jmrplens/phonometry/v9.9.9/a.svg) "
        "[ci](https://github.com/jmrplens/phonometry/actions/workflows/main.yml) "
        "[badge](https://img.shields.io/pypi/v/phonometry?logo=main) "
        "[tagged](https://github.com/jmrplens/phonometry/blob/v1.0.0/LICENSE)"
    )


def test_pinning_leaves_other_repositories_alone() -> None:
    """A third party's ``main`` link is theirs, and our tag does not exist there.

    Rewriting it would invent a ref in somebody else's repository and publish
    a 404, which is the exact failure the pinning exists to prevent in ours.
    """
    text = (
        "[toolbox](https://github.com/Universite-Gustave-Eiffel/acoustic-toolbox"
        "/blob/main/README.md) "
        "[numpy](https://github.com/numpy/numpy/tree/main/doc/) "
        "![logo](https://raw.githubusercontent.com/scipy/scipy/main/doc/logo.svg) "
        "[ours](https://github.com/jmrplens/phonometry/blob/main/LICENSE)"
    )
    pinned = generate_pypi_readme.pin_to_tag(text, "v9.9.9")
    assert "acoustic-toolbox/blob/main/README.md" in pinned
    assert "numpy/numpy/tree/main/doc/" in pinned
    assert "scipy/scipy/main/doc/logo.svg" in pinned
    assert "jmrplens/phonometry/blob/v9.9.9/LICENSE" in pinned
