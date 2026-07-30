#  Copyright (c) 2026. Jose M. Requena-Plens
"""Report what a PDF's text layer is silently deleting.

A contributor tool, not a CI gate. Run it over a source document *before*
quoting its text anywhere, and certainly before filing an errata entry against
it.

The premise is simple: a document full of mathematics that emits **no** square
root sign over its whole text layer is not a document without radicals, it is a
document whose radicals are invisible to extraction. ``f_T/sqrt(2)`` then reads
back as ``f_T/2``, which is a different and perfectly plausible formula, and
nothing in the extracted text says a glyph went missing. The same holds for the
true minus sign U+2212: a text layer with none of those but plenty of ASCII
hyphens has re-encoded its minus signs at best, and dropped them at worst.

Two more counters catch the encodings that mangle rather than delete. C0
control characters in the output mean the ToUnicode map is partial (a BSI
standard in this project's corpus emits sixteen thousand of them, and its whole
body text comes back Caesar-shifted), and a pile of ``þ ð ¼`` means a maths
font whose ligature slots have been mapped to Latin-1 (one textbook emits four
thousand, where ``þ`` is its plus sign).

Usage::

    python scripts/glyph_census.py plan/some-standard.pdf ...
    python scripts/glyph_census.py --quiet plan/*.pdf     # only the bad ones

Extraction uses ``pypdfium2`` if it is installed and falls back to
``pdftotext``. Exit status is 0 unless a file could not be read at all; the
report is advisory. See ``CONTRIBUTING.md``, "Filing an errata entry".
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess  # nosec B404 - pdftotext is an explicit, argument-free fallback
import unicodedata
from dataclasses import dataclass

#: The glyphs whose absence makes a maths-bearing text layer untrustworthy.
RADICAL = "√"
MINUS = "−"

#: Latin-1 code points a mis-mapped maths font tends to emit. In one textbook
#: of this project's corpus every "+" arrives as "þ".
LIGATURE_TELLS = "þðÿ¼½¾"

#: Below this many extracted characters the document is a scan, and the census
#: says nothing about its glyphs.
_TEXT_LAYER_FLOOR = 200


@dataclass(frozen=True)
class Census:
    """What one document's text layer emits."""

    path: pathlib.Path
    characters: int
    radicals: int
    minus_signs: int
    hyphens: int
    controls: int
    ligature_tells: int

    @property
    def is_scan(self) -> bool:
        return self.characters < _TEXT_LAYER_FLOOR

    @property
    def warnings(self) -> list[str]:
        """The findings that make this text layer unsafe to quote."""
        if self.is_scan:
            return [
                (
                    f"only {self.characters} characters extracted: this is a "
                    "scan or an image-only PDF, so there is no text layer to trust"
                )
            ]
        found: list[str] = []
        if self.radicals == 0:
            found.append(
                "no U+221A anywhere: every radical in this document extracts as "
                "if it were absent, so 'sqrt(2)' reads back as '2'"
            )
        if self.minus_signs == 0 and self.hyphens > 0:
            found.append(
                f"no U+2212 but {self.hyphens} ASCII hyphens: minus signs are "
                "re-encoded at best and dropped at worst"
            )
        if self.controls > 0:
            found.append(
                f"{self.controls} C0 control characters: the ToUnicode map is "
                "partial, so some glyphs decode to nothing readable"
            )
        if self.ligature_tells > 0:
            found.append(
                f"{self.ligature_tells} of '{LIGATURE_TELLS}': a maths font "
                "mapped into Latin-1, where those stand in for operators"
            )
        return found


def _extract_pypdfium2(path: pathlib.Path) -> str | None:
    try:
        # Optional dependency, probed at call time rather than at import.
        import pypdfium2
    except ImportError:
        return None
    try:
        document = pypdfium2.PdfDocument(str(path))
    except pypdfium2.PdfiumError:  # pragma: no cover - damaged or not a PDF
        # Fall through to the next extractor rather than crash the census.
        return None
    try:
        return "".join(page.get_textpage().get_text_range() for page in document)
    finally:
        document.close()


def _extract_pdftotext(path: pathlib.Path) -> str | None:
    binary = shutil.which("pdftotext")
    if binary is None:
        return None
    completed = subprocess.run(  # nosec B603 - fixed binary, path is the only input
        [binary, "-layout", str(path), "-"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace")


def extract(path: pathlib.Path) -> str | None:
    """The document's text layer, by whichever extractor is available."""
    for extractor in (_extract_pypdfium2, _extract_pdftotext):
        text = extractor(path)
        if text is not None:
            return text
    return None


def census(path: pathlib.Path) -> Census | None:
    """Count the tell-tale glyphs of one document, or ``None`` if unreadable."""
    text = extract(path)
    if text is None:
        return None
    controls = sum(
        1 for c in text if unicodedata.category(c) == "Cc" and c not in "\t\n\r\f"
    )
    return Census(
        path=path,
        characters=len(text),
        radicals=text.count(RADICAL),
        minus_signs=text.count(MINUS),
        hyphens=text.count("-"),
        controls=controls,
        ligature_tells=sum(text.count(c) for c in LIGATURE_TELLS),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdfs", nargs="+", type=pathlib.Path, help="documents to read")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the documents whose text layer is unsafe to quote",
    )
    args = parser.parse_args(argv)

    unreadable = 0
    flagged = 0
    for path in args.pdfs:
        result = census(path)
        if result is None:
            print(f"{path}: could not extract any text (no extractor available?)")
            unreadable += 1
            continue
        warnings = result.warnings
        if warnings:
            flagged += 1
        elif args.quiet:
            continue
        print(f"{path}")
        print(
            f"    {result.characters} chars | U+221A {result.radicals} | "
            f"U+2212 {result.minus_signs} | '-' {result.hyphens} | "
            f"C0 {result.controls} | ligature tells {result.ligature_tells}"
        )
        for warning in warnings:
            print(f"    ! {warning}")
        if not warnings:
            print("    ok: radicals and minus signs survive extraction")
    print(
        f"\n{len(args.pdfs)} document(s): {flagged} unsafe to quote from the "
        f"text layer, {unreadable} unreadable."
    )
    return 1 if unreadable else 0


if __name__ == "__main__":
    raise SystemExit(main())
