#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tolerance-aware staleness check for the committed ``.report()`` fiches.

The ``Example report fiches up to date`` CI job regenerates ``.github/reports``
with ``make reports`` and must confirm the result still matches what is
committed. Nothing used to check this, and the fiches drifted: two of them sat
a plot-styling release behind the library for weeks, and one was registered in
the generator without its rendered files ever being committed. The
documentation links to these PDFs as worked examples of what the library
prints, so a stale one is a wrong answer published under the maintainer's name.

**Why not a byte diff.** Not for the usual PDF reason: the renderer asks
reportlab for an invariant document, so the creation date and document id are
fixed and two runs of unchanged code on one machine do produce identical
bytes. The reason is the one that made the figures job flaky before
:mod:`scripts.check_figures` replaced its byte diff with a tolerance. Every
fiche embeds its plot as vector geometry, and GitHub's runner fleet is
hardware-heterogeneous, so the same pinned stack computes a handful of those
coordinates ~1 ULP apart depending on which CPU microarchitecture the run
lands on. A byte gate would eventually fail on a difference no reader can see,
and a gate that cries wolf gets switched off.

**What is compared instead**, per fiche:

* **Extracted text** (``<name>.pdf``) -- page count and the text of every page
  must match exactly. This is the part of a fiche a reader quotes: the
  measured values, the rating, the verdict, the standard clause. It carries no
  floating-point noise, because every printed number is already rounded to the
  decimals the standard reports. It does *not* cover the embedded plot, whose
  labels are drawn as paths (``svg.fonttype='path'``), so it cannot stand
  alone.
* **Rendered pixels** (``<name>.webp``) -- the committed preview is the first
  page of the committed PDF rasterized at a fixed width, so comparing
  previews compares the rendered page itself, sidestepping the PDF
  metadata entirely. Every registered fiche is a single page, so this covers
  the whole document, plot included. The comparison is the same two-criteria
  raster tolerance the figures use (see
  :class:`~generated_assets.RasterTolerance`), recalibrated in
  :data:`RASTER_TOL` for a document page.

Added or removed files always fail: a newly registered fiche must be
committed, and one that is no longer generated must be removed from the tree.

Calibrating :data:`RASTER_TOL`
------------------------------

matplotlib writes SVG coordinates through ``%f``, so one unit in the last
emitted decimal (1e-6 user units) is the largest text change a last-bit
coordinate difference can produce; anything smaller does not reach the file at
all. Displacing *every* plotted path coordinate by that amount, in a random
direction, is therefore an upper bound on what a coordinate difference could
do, and far worse than reality, where at most a handful of coordinates drift.

Measured that way across all 67 fiches, the result is not a clean separation.
Most sit near zero, but five reach 235 to 1424 pixels and RMS 0.68 to 1.43,
above the thresholds below. Those are fiches whose plots have vertical bar
edges landing on a pixel boundary, where moving the whole edge by one unit in
the last decimal flips entire columns at once.

The thresholds are set anyway, for a reason worth stating plainly: that upper
bound is not what happens. Displacing a *single* coordinate in the worst of
those five changed no pixels at all across sixty trials, because a coordinate
only re-renders differently if its float lands within about one unit in the
last place of a rounding boundary six decimals down, and the coordinates that
flip a bar edge come from exact layout arithmetic rather than from libm or
BLAS. If the SVG text does not change, the PDF and its raster are identical.
No cross-machine evidence exists either way: everything here was measured by
simulating displacement on one machine, not by rendering on two.

What the thresholds do catch is the drift this check was written for: 28710
and 27629 pixels on the two stale fiches, and RMS 0.79 to 1.33 on the subtler
restyled fills that came with them. Raising the RMS bound far enough to clear
the simulated worst case would stop catching those fills, which is the failure
this check exists to prevent, so the bound stays where it is.

The pixel count is the weaker of the two criteria. One character changed in a
printed field moves 53 to 102 pixels, so most single-character edits pass it
and only the RMS sees them, by a margin as small as 7 percent. That is
acceptable because the PDF text is compared exactly: every printed value,
verdict and margin is covered whatever the raster does, and the raster
criteria only have to catch a changed plot.

They are tighter than the figures' (100 pixels, RMS 2.0) on purpose: a fiche
preview is a flat, opaque document page rasterized from a PDF by a pinned
binary rasterizer, which is a markedly quieter thing to compare than a plot
raster. Keeping the figures' looser RMS here would let a restyled fill through
-- which is precisely how the two stale fiches went unnoticed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generated_assets import (
    RasterTolerance,
    committed_bytes,
    raster_problem,
    tracked_files,
)

REPORT_DIR = ".github/reports"

#: Raster tolerance for the WebP previews. ``level`` matches the figures': a
#: per-channel difference of 12 (of 255) is far above the level or two that
#: anti-aliasing can wobble by, and far below a moved line or a restyled fill.
#: ``max_sig_pixels`` and ``rms`` are calibrated in the module docstring.
RASTER_TOL = RasterTolerance(level=12, max_sig_pixels=64, rms=0.5)


def pdf_pages_text(data: bytes) -> list[str]:
    """Return the extracted text of every page of the PDF in ``data``.

    Uses the same rasterizer the generator previews with, so the check needs
    no PDF library the fiche pipeline does not already pull in.
    """
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(data)
    try:
        return [page.get_textpage().get_text_bounded() for page in document]
    finally:
        document.close()


def pdf_problem(old: bytes, new: bytes) -> str | None:
    """Describe how two fiche PDFs differ in text, or ``None`` if they do not."""
    try:
        old_pages = pdf_pages_text(old)
        new_pages = pdf_pages_text(new)
    except Exception as exc:  # noqa: BLE001 - a corrupt fiche is a failure, not a crash
        return f"could not extract text ({exc})"
    if len(old_pages) != len(new_pages):
        return f"page count changed {len(old_pages)} != {len(new_pages)}"
    for number, (old_text, new_text) in enumerate(
        zip(old_pages, new_pages, strict=True), start=1
    ):
        if old_text != new_text:
            return f"text of page {number} changed"
    return None


def _problems() -> list[str]:
    """Collect every way the committed fiches differ from the regenerated ones."""
    disk = {str(p) for p in Path(REPORT_DIR).rglob("*") if p.is_file()}
    tracked = tracked_files(REPORT_DIR)
    problems: list[str] = []

    for path in sorted(disk - tracked):
        problems.append(f"new fiche not committed: {path}")
    for path in sorted(tracked - disk):
        problems.append(f"committed fiche no longer generated: {path}")

    for path in sorted(disk & tracked):
        new = Path(path).read_bytes()
        old = committed_bytes(path)
        if old is None:
            problems.append(f"cannot read committed {path}")
            continue
        if path.endswith(".webp"):
            if old == new:
                continue  # byte-identical: fast path, no tolerance needed
            reason = raster_problem(old, new, RASTER_TOL)
            if reason is not None:
                problems.append(f"preview changed beyond tolerance ({reason}): {path}")
        elif path.endswith(".pdf"):
            # Never byte-compared: a cross-CPU coordinate wobble in the
            # embedded plot rewrites the page stream for no visible reason.
            reason = pdf_problem(old, new)
            if reason is not None:
                problems.append(f"fiche changed ({reason}): {path}")
        else:
            problems.append(f"unexpected file in {REPORT_DIR}: {path}")
    return problems


#: Where the guides live, and the shape of the component that embeds a fiche.
DOCS_DIR = "site/src/content/docs"
EMBEDDED = re.compile(r"<ReportPreview\s[^>]*name=\"([a-z0-9_]+)\"", re.DOTALL)


def _shown_in(language: str) -> set[str]:
    """Fiche names embedded on the pages of one language."""
    root = Path(DOCS_DIR)
    pages = (
        [
            page
            for page in root.rglob("*.mdx")
            if page.relative_to(root).parts[0] == "es"
        ]
        if language == "es"
        else [
            page
            for page in root.rglob("*.mdx")
            if page.relative_to(root).parts[0] != "es"
        ]
    )
    return {
        name
        for page in pages
        for name in EMBEDDED.findall(page.read_text(encoding="utf-8"))
    }


def _unshown() -> list[str]:
    """Fiches no reader can see, and previews of fiches that do not exist.

    A fiche is generated so that a guide can show what the library prints. The
    generator and the staleness check above close the loop from the code to the
    committed file; nothing closed the one from the file to a reader. Seven
    fiches were generated, committed, shipped in the built site and embedded on
    no page in either language, one of them the worked example of the very
    snippet its guide prints.

    Both languages are checked separately, because a fiche shown on the English
    page and missing from the Spanish one is invisible to every other gate: the
    EN/ES parity check compares the two file trees with each other, and a page
    that exists in both is a page it passes.
    """
    generated = {path.stem for path in Path(REPORT_DIR).glob("*.pdf")}
    problems: list[str] = []
    for language, label in (("en", "English"), ("es", "Spanish")):
        shown = _shown_in(language)
        problems += [
            f"fiche generated but shown on no {label} page: {name} "
            '(embed it with <ReportPreview name="..."> in the guide that '
            "documents its .report())"
            for name in sorted(generated - shown)
        ]
        problems += [
            f"{label} page embeds a fiche that is not generated: {name} "
            "(register it in scripts/generate_reports.py and run 'make reports')"
            for name in sorted(shown - generated)
        ]
    return problems


def main() -> int:
    """Report every stale fiche; return 1 if there is one."""
    problems = _problems() + _unshown()
    if problems:
        print(
            f"::error::{REPORT_DIR} is out of date - "
            "run 'make reports' and commit the result."
        )
        for problem in problems:
            print(f"  - {problem}")
        return 1

    fiches = len(list(Path(REPORT_DIR).glob("*.pdf")))
    print(
        f"All {fiches} committed fiches match within tolerance, "
        "and every one of them is shown on a page."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
