#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What a verdict indicator is called, where it is served from, what it says.

:mod:`conformance.badges` draws the indicators; this module is everything a
document needs in order to *cite* one. The split is not tidiness, it is a
constraint: the ``pr-comment`` job in ``.github/workflows/python-app.yml``
installs nothing at all. It reads two committed JSON files on the standard
library alone, deliberately, because it used to spend 45 s rebuilding a report
the ``conformance`` job had already proven current. Drawing a badge needs
matplotlib, to turn a committed font into glyph outlines; citing one needs a
filename and a URL. Only the citing half may be imported from that job, so
only the citing half is here.

Three documents cite these names and none of them writes one down: the
Markdown report through :mod:`conformance.render`, the sticky pull-request
comment through ``.github/scripts/comment_pr.py``, and the README by hand.
``tests/test_conformance_badges.py`` scans the whole tracked tree for a
citation of a file nothing generates, which is what turns a rename into a
failing test rather than a broken image nobody notices.

**Why a raw URL and not a relative path.** ``docs/CONFORMANCE.md`` is read on
github.com, mirrored into the documentation site, and quoted in a comment on a
different page of a different host; a relative path resolves in the first case
and in neither of the others. Measured against the live repository, GitHub
serves ``raw.githubusercontent.com`` for this repository untouched, while it
rewrites ``img.shields.io`` and the other third-party badge hosts through
``camo.githubusercontent.com`` - so there is no proxy cache between a reader
and these files, and a badge regenerated on ``main`` is live within the five
minutes raw's ``cache-control`` asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any

from .registry import Verdict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Where the indicators are served from. The repository's own raw host, which
#: GitHub does not proxy.
RAW_BASE = "https://raw.githubusercontent.com/jmrplens/phonometry"

#: Path of the badge directory inside the repository, as a URL fragment. Not
#: ``.github/images``: the ``graphs`` target empties that of SVGs before every
#: figure run, so a badge parked beside the figures would vanish on the next.
RAW_PATH = ".github/badges"

#: Appended before the extension to name a theme variant. The README's
#: ``<picture>`` blocks and ``site/src/components/ThemeImage.astro`` both pair
#: on exactly this substitution, so it is written here once.
DARK_SUFFIX = "_dark"


@dataclass(frozen=True)
class Mark:
    """One verdict's indicator: the file, the word, and how to cite it.

    :param verdict: The verdict string the artefact stores.
    :param filename: The committed file, under :data:`RAW_PATH`.
    :param alt: What a reader gets when the image does not arrive. It is the
        verdict in words, never the name of a picture. It is also the
        accessible name of the link GitHub wraps every Markdown image in, so
        it may not be empty: 566 links with no name is a worse defect than the
        one duplicated announcement that a visible word beside the mark costs.
    :param label: The Markdown link-reference label the tables use. Reference
        style rather than inline, because a definition resolves document-wide:
        the per-row cost is the 16-character label instead of a 117-character
        URL, which is what keeps 566 illustrated rows affordable.
    """

    verdict: str
    filename: str
    alt: str
    label: str


#: The four marks, in the order a legend should read them.
MARKS: tuple[Mark, ...] = (
    Mark(str(Verdict.PASS), "verdict-pass.svg", "Pass", "cv-pass"),
    Mark(str(Verdict.FAIL), "verdict-fail.svg", "Fail", "cv-fail"),
    Mark(str(Verdict.BY_DESIGN), "verdict-by-design.svg", "By design", "cv-by-design"),
    Mark(str(Verdict.NOT_APPLICABLE), "verdict-not-applicable.svg", "n/a", "cv-na"),
)

#: The marks by the verdict they stand for, for a caller holding a row.
MARK_OF: dict[str, Mark] = {mark.verdict: mark for mark in MARKS}

#: The summary banner's light variant.
BANNER = "conformance-summary.svg"


def dark_variant(name: str) -> str:
    """The file that carries the dark-theme twin of ``name``."""
    return name.replace(".svg", f"{DARK_SUFFIX}.svg")


#: The dark twin of :data:`BANNER`, by that rule.
BANNER_DARK = dark_variant(BANNER)


def asset_url(name: str, ref: str) -> str:
    """Absolute URL of one committed indicator at one git ref.

    :param name: A filename from :data:`MARKS` or :data:`BANNER`.
    :param ref: What to pin to. ``main`` for a document that lives on ``main``
        and is read there, and a full commit SHA for a document that must show
        the tree it was generated from - a pull-request comment above all,
        where a link to ``main`` would quietly report ``main``'s numbers on
        every branch and look live while doing it.
    """
    return f"{RAW_BASE}/{ref}/{RAW_PATH}/{name}"


def mark_reference(verdict: str) -> str:
    """One verdict as a reference-style Markdown image.

    Pairs with :func:`mark_definitions`, which has to be emitted into the same
    document or the reference ships as literal text.
    """
    mark = MARK_OF[verdict]
    return f"![{mark.alt}][{mark.label}]"


def mark_definitions(verdicts: Sequence[str], ref: str) -> list[str]:
    """The link-reference definitions for ``verdicts``, in manifest order.

    Only the marks a document actually uses: an unused definition renders as
    nothing either way, but it is a URL claiming a file the page never shows.
    """
    wanted = set(verdicts)
    return [
        f"[{mark.label}]: {asset_url(mark.filename, ref)}"
        for mark in MARKS
        if mark.verdict in wanted
    ]


def mark_image(verdict: str, ref: str) -> str:
    """One verdict as a self-contained inline Markdown image.

    For a document too short to amortise a definition block, which is the
    pull-request comment: a handful of marks there, against 566 in the report.
    """
    mark = MARK_OF[verdict]
    return f"![{mark.alt}]({asset_url(mark.filename, ref)})"


def mark_html(verdict: str, ref: str) -> str:
    """One verdict as a raw ``<img>``, for the inside of an HTML block.

    Markdown is not parsed inside one. A ``<summary>`` on the line after its
    ``<details>`` is part of that block, so ``![Pass](…)`` written there ships
    to the reader as those literal characters - verified against GitHub's own
    renderer, which returned the source text unchanged for the Markdown form
    and an ``<img>`` for this one. It is the same reason the report's own
    summaries have always written ``<b>`` rather than ``**``.
    """
    mark = MARK_OF[verdict]
    alt = escape(mark.alt, quote=True)
    return f'<img src="{asset_url(mark.filename, ref)}" alt="{alt}">'


def outcome_mark(ok: bool, ref: str, *, html: bool = False) -> str:
    """The pass or the fail mark, for a yes/no result that is not a verdict.

    The test table in the pull-request comment is the case: a Python version
    either passed its suite or did not, which is not a conformance verdict but
    is exactly the same two shapes, so it reads as the same vocabulary.

    :param ok: Whether the thing being reported succeeded.
    :param ref: The commit to pin the image to.
    :param html: Emit a raw ``<img>`` instead of Markdown, for a caller
        writing inside an HTML block (see :func:`mark_html`).
    """
    verdict = str(Verdict.PASS if ok else Verdict.FAIL)
    return mark_html(verdict, ref) if html else mark_image(verdict, ref)


def plural(count: int, singular: str) -> str:
    """``singular`` agreeing with ``count``.

    Both nouns the banner names take a plain ``-s``. The sentence is read by
    people, and a fixture with one domain in it should not be captioned
    "1 domains".
    """
    return singular if count == 1 else f"{singular}s"


def banner_alt(counts: Mapping[str, Any]) -> str:
    """What the summary banner says, in words.

    The banner is a picture of four numbers; this is the same four numbers as
    a sentence. It is the ``<title>`` inside the file, the ``alt`` on every
    embedding of it, and what a reader gets when the image does not arrive, so
    the three cannot drift apart.
    """
    passed, total = counts["passing"], counts["checks"]
    failing = total - passed
    verdict = (
        f"{passed} of {total} conformance checks pass"
        if failing
        else f"All {total} conformance checks pass"
    )
    tail = f", {failing} failing" if failing else ""
    domains, standards = counts["domains"], counts["standards"]
    return (
        f"{verdict}{tail}, across {domains} {plural(domains, 'domain')} "
        f"and {standards} {plural(standards, 'standard')}"
    )


def banner_picture(counts: Mapping[str, Any], ref: str) -> str:
    """The summary banner as a theme-aware Markdown embedding.

    A ``<picture>`` whose ``<source>`` carries the dark variant is the
    mechanism GitHub itself documents and the one the README already uses; it
    follows the reader's GitHub theme toggle. The alternative, a ``<style>``
    with ``prefers-color-scheme`` inside the SVG, resolves against the
    reader's operating system instead, because GitHub loads the file through
    an ``<img>`` - so a reader on a light GitHub page with a dark desktop
    would get the dark palette on white.

    Emitted on one line: the pair sits inside a paragraph, and a hard-wrapped
    element puts a ``<`` at the start of a line, which the Markdown-hazard
    gate reads as the start of a block.

    No link around it. The README wraps its own copy in one, because there the
    banner is an invitation to go and read the report; in the report itself,
    and in a comment that already links the report twice in its footer, a link
    would only lead back to where the reader is.

    :param counts: The artefact's ``counts`` block.
    :param ref: The git ref to pin both variants to (see :func:`asset_url`).
    """
    alt = escape(banner_alt(counts), quote=True)
    return (
        f'<picture><source media="(prefers-color-scheme: dark)" '
        f'srcset="{asset_url(BANNER_DARK, ref)}">'
        f'<img src="{asset_url(BANNER, ref)}" alt="{alt}"></picture>'
    )
