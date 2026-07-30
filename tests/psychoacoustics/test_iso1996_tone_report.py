#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the tonal audibility assessment report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written for a detected-tone result, unknown engines
and languages are rejected, XML specials in metadata do not break reportlab, the
verdict renders both ways, and the boxed decisive audibility ``ΔL_ta``, the
tonal adjustment ``K`` (ISO 1996-2:2017 Table J.1), the decisive tone frequency
and the metadata appear in the extracted text. The tone-audibility algorithm
itself is validated against the ISO/PAS 20065 Annex E oracle elsewhere
(tests/psychoacoustics/test_tone_audibility.py); this fiche test reuses the same
Annex E spectrum so its numbers are documented.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

import reference_data as ref

from phonometry import (
    ReportMetadata,
    analyze_spectrum,
    tonal_adjustment_from_mean_audibility,
)

_PDF_MAGIC = b"%PDF"


def _result():
    """The ISO/PAS 20065 Annex E spectrum-1 detection (three tones + FG)."""
    return analyze_spectrum(
        ref.ISO20065_E1_LEVELS, ref.ISO20065_E1_FREQUENCIES, ref.ISO20065_LINE_SPACING
    )


def _assert_one_page(path: str) -> None:
    import os

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    from pypdf import PdfReader

    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_report_writes_one_page_pdf(tmp_path) -> None:
    """A detected-tone result renders a one-page PDF fiche."""
    result = _result()
    out = tmp_path / "tone.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="xx")


def test_report_states_audibility_adjustment_and_frequency(tmp_path) -> None:
    """The fiche states ΔL_ta, the derived K and the decisive tone frequency.

    The decisive audibility is the FG entry near the 137.3 Hz critical band; its
    tonal adjustment follows ISO 1996-2:2017 Table J.1. The extracted text must
    carry the boxed audibility, the adjustment value and the decisive frequency.
    """
    result = _result()
    out = tmp_path / "tone.pdf"
    result.report(str(out))
    text = _extract_text(str(out)).replace("\n", " ")

    delta = result.decisive_audibility
    k = tonal_adjustment_from_mean_audibility(delta)
    assert k == 5  # 9 < ΔL_ta <= 12  (Annex E spectrum 1, FG group)
    assert "Tonal audibility" in text
    assert f"= {delta:.1f} dB" in text  # boxed ΔL_ta
    assert f"K = {k} dB" in text
    assert f"{result.decisive_frequency:.1f} Hz" in text
    assert "Table J.1" in text
    # Table J.1 is defined on the mean audibility of the J assessed spectra;
    # a single-spectrum assessment says so rather than implying otherwise.
    assert "mean audibility" in text
    assert "Clause 5.3.9, Formula (20)" in text
    assert "single spectrum" in text


def test_metadata_appears_and_one_page(tmp_path) -> None:
    """A populated ReportMetadata renders one page and prints its fields."""
    md = ReportMetadata(
        specimen="Combustion engine, steady operation",
        client="Acoustic Test Client Ltd.",
        test_room="Free field, 3 m from source",
        instrumentation="Class 1 analyser, 2.7 Hz FFT lines",
        measurement_standard="ISO 1996-2",
        test_date="2026-07-21",
        laboratory="Phonometry Reference Laboratory",
        operator="J. M. Requena-Plens",
        report_id="PHN-2026-1996",
    )
    out = tmp_path / "meta.pdf"
    _result().report(str(out), metadata=md, verbose=True)
    _assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "Combustion engine, steady operation" in text
    assert "Free field, 3 m from source" in text
    assert "PHN-2026-1996" in text
    assert "ISO 1996-2" in text


def test_requirement_pass_and_fail_both_render(tmp_path) -> None:
    """A PASS and a FAIL audibility limit both render one page."""
    result = _result()
    delta = result.decisive_audibility
    passing = tmp_path / "pass.pdf"
    failing = tmp_path / "fail.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=delta + 2.0))
    result.report(str(failing), metadata=ReportMetadata(requirement=delta - 2.0))
    _assert_one_page(str(passing))
    _assert_one_page(str(failing))
    assert "PASS" in _extract_text(str(passing)).replace("\n", " ")
    assert "FAIL" in _extract_text(str(failing)).replace("\n", " ")


def test_report_escapes_xml_specials_in_metadata(tmp_path) -> None:
    """Metadata with XML specials (& < >) renders without crashing reportlab."""
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="engine <A> & exhaust",
        test_room="pos <1> & <2>",
        laboratory="Lab & Sons",
        operator="A <B>",
        report_id="R&D-1996",
        measurement_standard="ISO 1996-2 & Annex J",
    )
    out = tmp_path / "xml.pdf"
    _result().report(str(out), metadata=md)
    _assert_one_page(str(out))


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """``language="es"`` renders a one-page Spanish fiche with comma decimals."""
    import re

    out = tmp_path / "tone_es.pdf"
    _result().report(
        str(out),
        metadata=ReportMetadata(specimen="motor de combustión"),
        language="es",
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Evaluación de la audibilidad tonal" in text
    assert "ajuste tonal" in text
    assert "audibilidad media" in text  # the Clause 5.3.9 caveat
    assert "un único espectro" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator


def test_absent_tone_reports_zero_adjustment(tmp_path) -> None:
    """A weakly-audible decisive tone still renders and states its adjustment.

    A synthetic near-threshold tone keeps K low; the prominence note and boxed
    K stay consistent with the decisive audibility.
    """
    from phonometry import assess_tones

    # One tone barely above the masking threshold: LT chosen so ΔL_ta is small.
    result = assess_tones([500.0], [40.0], [30.0], 2.0)
    out = tmp_path / "weak.pdf"
    result.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    # The tone sits below the masking threshold, so no adjustment applies.
    assert result.decisive_audibility < 0.0
    k = tonal_adjustment_from_mean_audibility(result.decisive_audibility)
    assert k == 0
    assert "K = 0 dB" in text
    assert "No prominent tone is present" in text
