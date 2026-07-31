#  Copyright (c) 2026. Jose M. Requena-Plens
"""The glyph census verdicts, exercised without needing a PDF.

``scripts/glyph_census.py`` tells a contributor whether a source document's
text layer can be quoted at all. The counting is trivial; what matters is the
verdict, so these tests drive :class:`Census` directly with the counts real
documents produce. The numbers below are measured, not invented: they come from
the corpus this project's errata registry cites.
"""

from __future__ import annotations

import pathlib
import sys

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import glyph_census as gc


def _census(**overrides: object) -> gc.Census:
    """A clean text layer, with the given counters replaced."""
    fields: dict[str, object] = {
        "path": pathlib.Path("probe.pdf"),
        "characters": 100_000,
        "radicals": 12,
        "minus_signs": 300,
        "hyphens": 400,
        "controls": 0,
        "ligature_tells": 0,
    }
    fields.update(overrides)
    return gc.Census(**fields)  # type: ignore[arg-type]


def test_a_clean_text_layer_raises_nothing() -> None:
    assert _census().warnings == []


def test_zero_radicals_is_the_headline_finding() -> None:
    """The failure that produced a retracted errata entry."""
    warnings = _census(radicals=0).warnings
    assert len(warnings) == 1
    assert "U+221A" in warnings[0]


def test_hyphens_without_a_true_minus_are_flagged() -> None:
    warnings = _census(minus_signs=0).warnings
    assert any("U+2212" in w for w in warnings)


def test_a_document_with_neither_hyphens_nor_minus_signs_is_not_flagged() -> None:
    """No hyphens either means the document simply has no dashes to lose."""
    assert _census(minus_signs=0, hyphens=0).warnings == []


def test_the_bsi_encoding_signature() -> None:
    """A BS EN print in the corpus: 16 008 control characters, no radicals."""
    warnings = _census(radicals=0, minus_signs=0, controls=16_008).warnings
    assert any("16008 C0 control characters" in w for w in warnings)
    assert len(warnings) == 3


def test_the_textbook_ligature_signature() -> None:
    """Long, Architectural Acoustics 2e: 4 016 Latin-1 tells, "þ" being "+"."""
    warnings = _census(
        radicals=0, minus_signs=0, controls=1_236, ligature_tells=4_016
    ).warnings
    assert any("4016 of" in w for w in warnings)
    assert len(warnings) == 4


def test_a_scan_reports_only_that_it_is_a_scan() -> None:
    """An image-only PDF says nothing about glyphs, so it makes no claims."""
    warnings = _census(characters=12, radicals=0, minus_signs=0).warnings
    assert len(warnings) == 1
    assert "scan" in warnings[0]


def test_extract_returns_none_when_the_file_is_not_a_pdf(
    tmp_path: pathlib.Path,
) -> None:
    not_a_pdf = tmp_path / "not-a.pdf"
    not_a_pdf.write_text("plain text", encoding="utf-8")
    assert gc.census(not_a_pdf) is None
