#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What a fiche test asserts about the file the report wrote.

Fifty-four fiche tests each carried a private copy of this check, in eight
spellings that differed only in wording, in where the imports sat, and in
whether they also asserted a non-zero file size. That last assertion could not
fail. Reading the first four bytes and finding ``%PDF`` already proves the file
holds at least four bytes, so the size check was only ever reached when it was
bound to pass. What survives is the pair that carries weight, written once.

``pypdf`` is imported inside the functions on purpose: the fiche tests skip at
import time when ``reportlab`` is missing, and a module-level import here would
make the reader a hard dependency of every test that merely imports this file.
"""

from __future__ import annotations

from pathlib import Path

#: The four bytes a PDF opens with (ISO 32000-1, 7.5.2, "File Header").
PDF_MAGIC = b"%PDF"


def assert_pdf(path: str | Path) -> None:
    """Assert the file at *path* opens as a PDF.

    :param path: The file the report wrote.
    :raises AssertionError: if the first four bytes are not ``%PDF``, which
        covers an empty file, a truncated write and a wrong format alike.
    """
    with Path(path).open("rb") as handle:
        assert handle.read(4) == PDF_MAGIC


def assert_one_page(path: str | Path) -> None:
    """Assert the file at *path* is a PDF of exactly one page.

    A fiche that spills onto a second page is a layout regression rather than a
    crash, so nothing else in the suite would catch it.

    :param path: The file the report wrote.
    :raises AssertionError: if the file is not a PDF, or does not parse as one
        page exactly.
    """
    from pypdf import PdfReader

    assert_pdf(path)
    assert len(PdfReader(str(path)).pages) == 1
