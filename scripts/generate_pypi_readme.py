#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Generate README_PYPI.md, the PyPI long description, from README.md.

Two rewrites, for two things a package page does that a repository page
does not.

**Images.** PyPI's readme renderer strips ``<picture>``/``<source>``
elements, so the GitHub README's theme-aware images would degrade there.
Each ``<picture>`` element collapses down to its plain light-theme ``<img>``
fallback, and animated GIFs are swapped for their static ``_poster.jpg``
stills so the package page stays light to load.

**Links.** A PyPI description is frozen at upload and can never be edited
again, while ``main`` keeps moving under it. Every ``blob/main`` and
``raw.githubusercontent.com/.../main`` URL the README carries therefore
resolves against a tree that no longer matches the release the reader is
looking at, and the whole guide map rots the first time the documentation
is reorganised: the description published with 3.3.0 pointed at a flat
``docs/`` layout and 34 of its 38 repository links were dead within one
release. So the links are pinned here, at generation time, to ``blob/v``
plus the contents of the repository-root ``VERSION`` file: the published
page points at the tree it was published from, and stays right forever
because that tree never changes again.

The tag is created by the release workflow, from the commit that lands the
``VERSION`` bump on ``main``, so between the bump and the release there is a
window in which ``README_PYPI.md`` names a tag that does not exist yet. That
is deliberate, and it is why the packaging tests resolve the pinned paths
against **this checkout** rather than over the network: the working tree is
what the tag will point at, so the paths can be verified on ``main`` without
waiting for the release. Anything that checks these URLs over HTTP has to
tolerate the unreleased tag the same way.

Run through ``make pypi-readme``; the packaging tests fail if the committed
file drifts from the README or from ``VERSION``.
"""

from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HEADER = (
    "<!-- AUTO-GENERATED FILE - DO NOT EDIT BY HAND.\n"
    "  PyPI long description derived from README.md by\n"
    "  scripts/generate_pypi_readme.py (make pypi-readme): every picture\n"
    "  element collapses to its light-theme img fallback and animated\n"
    "  clips to their poster stills, because PyPI's readme renderer\n"
    "  strips the picture and source tags, and every link into this\n"
    "  repository is pinned to the release tag, because a published\n"
    "  description is frozen while main keeps moving. Edit README.md\n"
    "  instead. -->\n\n"
)

_PICTURE = re.compile(r"<picture>.*?</picture>", re.DOTALL)
_IMG = re.compile(r"<img\b[^>]*>", re.DOTALL)

#: A URL into this repository at the ``main`` branch, in either of the two
#: forms the README uses: ``github.com/<owner>/<repo>/blob/main/...`` for
#: files rendered on GitHub and ``raw.githubusercontent.com/<owner>/<repo>/
#: main/...`` for images served raw. Group 1 is everything up to and
#: including the ref position, so substituting the ref leaves the rest of
#: the URL untouched.
_MAIN_REF = re.compile(
    r"(https://github\.com/[\w.-]+/[\w.-]+/blob/"
    r"|https://raw\.githubusercontent\.com/[\w.-]+/[\w.-]+/)main/"
)


def version(root: pathlib.Path = _ROOT) -> str:
    """The version being published, from the repository-root VERSION file."""
    return (root / "VERSION").read_text(encoding="utf-8").strip()


def pin_to_tag(text: str, tag: str) -> str:
    """Point every ``main`` URL into this repository at *tag* instead.

    :param text: Markdown carrying the README's absolute repository links.
    :param tag: The git tag to resolve them against, for example ``v3.3.0``.
    :return: The same markdown with the branch ref replaced by the tag.
    """
    return _MAIN_REF.sub(rf"\g<1>{tag}/", text)


def pypi_readme(readme: str, tag: str) -> str:
    """The PyPI long description derived from *readme* (README.md text).

    :param readme: The GitHub README's text.
    :param tag: Tag the repository links are pinned to, for example
        ``v3.3.0``.
    """

    def collapse(match: re.Match[str]) -> str:
        imgs = _IMG.findall(match.group(0))
        if len(imgs) != 1:
            raise ValueError(
                "expected exactly one <img> fallback inside each <picture> "
                "element of README.md"
            )
        img = imgs[0].replace(' loading="lazy"', "")
        swapped = re.sub(r'(src="[^"]+)\.gif"', r'\1_poster.jpg"', img)
        if swapped != img:
            # The still is no longer an animation; fix the alt prefix.
            swapped = re.sub(
                r'alt="Animation: (.)',
                lambda m: 'alt="' + m.group(1).upper(), swapped)
        return swapped

    return pin_to_tag(_HEADER + _PICTURE.sub(collapse, readme), tag)


def main() -> None:
    """Rewrite README_PYPI.md next to the repository README."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    tag = f"v{version()}"
    out = _ROOT / "README_PYPI.md"
    out.write_text(pypi_readme(readme, tag), encoding="utf-8")
    print(f"wrote {out} (repository links pinned to {tag})")


if __name__ == "__main__":
    main()
