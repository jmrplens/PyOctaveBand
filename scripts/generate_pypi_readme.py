#  Copyright (c) 2026. Jose M. Requena-Plens
"""Generate README_PYPI.md, the PyPI long description, from README.md.

PyPI's readme renderer strips ``<picture>``/``<source>`` elements, so the
GitHub README's theme-aware images would degrade there. This script rewrites
each ``<picture>`` element down to its plain light-theme ``<img>`` fallback,
and swaps animated GIFs for their static ``_poster.jpg`` stills so the
package page stays light to load. Run through ``make pypi-readme``; the
packaging tests fail if the committed file drifts from the README.
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
    "  strips the picture and source tags. Edit README.md instead. -->\n\n"
)

_PICTURE = re.compile(r"<picture>.*?</picture>", re.DOTALL)
_IMG = re.compile(r"<img\b[^>]*>", re.DOTALL)


def pypi_readme(readme: str) -> str:
    """The PyPI long description derived from *readme* (README.md text)."""

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

    return _HEADER + _PICTURE.sub(collapse, readme)


def main() -> None:
    """Rewrite README_PYPI.md next to the repository README."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    out = _ROOT / "README_PYPI.md"
    out.write_text(pypi_readme(readme), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
