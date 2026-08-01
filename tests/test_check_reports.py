#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The fiche staleness gate catches real drift and ignores encoding noise.

``scripts/check_reports.py`` is the gate that would have caught the two fiches
that sat a plot-styling release behind the library. It has to fail on the
changes that made them stale (a restyled fill, a moved plot line, a reworded
table) and pass on the sub-pixel wobble a different runner CPU produces, so
these tests drive both sides of every criterion with synthetic pages rather
than trusting the thresholds by inspection.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("PIL")
# The PDF half of the gate reads its pages through pypdfium2. Without it,
# ``pdf_problem`` reports an extraction failure for every input, which is a
# legitimate answer for a corrupt fiche but turns the text-comparison tests
# into failures rather than skips.
pytest.importorskip("pypdfium2")
from PIL import Image

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import check_reports as cr
from generated_assets import raster_problem

#: A stand-in for a fiche preview: a white page with a shaded band and a line,
#: at the aspect ratio the generator writes (A4 at 1000 px wide).
_PAGE = (1415, 1000, 4)


def _page(fill: tuple[int, int, int] = (255, 236, 219)) -> np.ndarray:
    """A white page carrying one shaded band and one dark plotted line."""
    page = np.full(_PAGE, 255, dtype=np.uint8)
    page[500:800, 150:900, :3] = fill
    page[640:644, 150:900, :3] = 40
    return page


def _encode(page: np.ndarray) -> bytes:
    """Encode a page as the lossless WebP the generator commits."""
    import io

    buffer = io.BytesIO()
    Image.fromarray(page, "RGBA").convert("RGB").save(
        buffer, "WEBP", lossless=True, quality=100, method=6
    )
    return buffer.getvalue()


def test_identical_previews_match() -> None:
    data = _encode(_page())
    assert raster_problem(data, data, cr.RASTER_TOL) is None


def test_sub_pixel_wobble_passes() -> None:
    """A handful of edge pixels off by an anti-aliasing level or two is noise.

    This is what a different runner CPU produces: the plotted geometry differs
    in the last bit, so a few pixels along one edge land on a slightly
    different coverage. Nothing a reader can see, and the gate must not fail
    on it or it will be switched off.
    """
    old = _page()
    new = old.copy()
    new[640, 150:200, :3] = 30  # 50 edge pixels, ten levels darker
    problem = raster_problem(_encode(old), _encode(new), cr.RASTER_TOL)
    assert problem is None


def test_restyled_fill_fails() -> None:
    """The drift that went unnoticed: a shaded band changing tint.

    Every pixel of the band moves by well under the per-pixel level threshold,
    so the pixel count alone lets it through; the RMS bound is what catches
    it. Three of the fiches that had drifted were found this way, the other
    two having moved enough pixels for the count to see them as well.
    """
    old = _page()
    new = _page(fill=(255, 231, 210))
    problem = raster_problem(_encode(old), _encode(new), cr.RASTER_TOL)
    assert problem is not None and "RMS" in problem


def test_moved_line_fails() -> None:
    """A localised change: the plotted line moves by four pixels."""
    old = _page()
    new = old.copy()
    new[640:644, 150:900, :3] = 255
    new[660:664, 150:900, :3] = 40
    problem = raster_problem(_encode(old), _encode(new), cr.RASTER_TOL)
    assert problem is not None and "pixels changed" in problem


def test_resized_preview_fails() -> None:
    old = _page()
    problem = raster_problem(_encode(old), _encode(old[:-1]), cr.RASTER_TOL)
    assert problem is not None and "dimensions changed" in problem


def _fiche_pdf(text: str) -> bytes:
    """A one-page PDF carrying ``text``, standing in for a rendered fiche."""
    import io

    pdfgen_canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    buffer = io.BytesIO()
    page = pdfgen_canvas.Canvas(buffer, invariant=1)
    page.drawString(72, 720, text)
    page.save()
    return buffer.getvalue()


def test_unchanged_text_matches() -> None:
    assert cr.pdf_problem(_fiche_pdf("Rw = 34 dB"), _fiche_pdf("Rw = 34 dB")) is None


def test_changed_rating_fails() -> None:
    """The value a reader quotes off the fiche moved: always a failure."""
    problem = cr.pdf_problem(_fiche_pdf("Rw = 34 dB"), _fiche_pdf("Rw = 35 dB"))
    assert problem == "text of page 1 changed"


def test_unreadable_fiche_fails() -> None:
    """A truncated or corrupt PDF is a failure, not a traceback."""
    problem = cr.pdf_problem(_fiche_pdf("Rw = 34 dB"), b"not a pdf at all")
    assert problem is not None and "could not extract text" in problem
