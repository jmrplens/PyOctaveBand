#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 11654 absorption-rating report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural
facts: a valid single-page PDF is written for a weighted absorption rating,
the two normative Annex A worked examples are the ones the fiche renders,
unknown engines are rejected, XML specials in metadata do not break reportlab,
and the requirement verdict renders. Pixel or layout content is never inspected.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

import numpy as np
from reference_data import (
    ISO11654_ANNEX_A1_ALPHA_P as _A1_ALPHA_P,
)
from reference_data import (
    ISO11654_ANNEX_A1_ALPHA_W as _A1_ALPHA_W,
)
from reference_data import (
    ISO11654_ANNEX_A1_CLASS as _A1_CLASS,
)
from reference_data import (
    ISO11654_ANNEX_A1_INDICATOR as _A1_INDICATOR,
)
from reference_data import (
    ISO11654_ANNEX_A2_ALPHA_P as _A2_ALPHA_P,
)
from reference_data import (
    ISO11654_ANNEX_A2_ALPHA_W as _A2_ALPHA_W,
)
from reference_data import (
    ISO11654_ANNEX_A2_INDICATOR as _A2_INDICATOR,
)

from phonometry import ReportMetadata
from phonometry.materials import (
    weighted_absorption,
    weighted_absorption_from_third_octave,
)

_PDF_MAGIC = b"%PDF"

# Fifteen one-third-octave alpha_s (200 Hz to 5000 Hz) whose octave means are
# the Annex A.2 practical coefficients (0.35, 1.00, 0.65, 0.60, 0.55).
_THIRD_OCTAVE_ALPHA_S = (
    0.30,
    0.35,
    0.40,
    1.00,
    1.00,
    1.00,
    0.62,
    0.66,
    0.67,
    0.58,
    0.60,
    0.62,
    0.53,
    0.55,
    0.57,
)


def _assert_pdf(path: str) -> None:
    """A written report is a non-empty file beginning with ``%PDF``."""
    import os

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0


def _assert_one_page(path: str) -> None:
    """The fiche is a single-page PDF beginning with ``%PDF``."""
    _assert_pdf(path)
    from pypdf import PdfReader

    assert len(PdfReader(path).pages) == 1


def test_absorption_report_writes_pdf(tmp_path) -> None:
    """A weighted absorption rating renders a PDF fiche."""
    result = weighted_absorption(_A1_ALPHA_P)
    out = tmp_path / "absorption.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    _assert_pdf(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = weighted_absorption(_A1_ALPHA_P)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_fiche_reproduces_iso11654_annex_a1(tmp_path) -> None:
    """The fiche renders the ISO 11654 Annex A.1 example: alpha_w = 0.60, class C."""
    result = weighted_absorption(_A1_ALPHA_P)
    assert result.alpha_w == pytest.approx(_A1_ALPHA_W)
    assert result.absorption_class == _A1_CLASS
    assert result.shape_indicator == _A1_INDICATOR
    _assert_one_page(str(result.report(str(tmp_path / "a1.pdf"))))


def test_fiche_reproduces_iso11654_annex_a2(tmp_path) -> None:
    """The fiche renders the Annex A.2 example: alpha_w = 0.60(M) shape indicator."""
    result = weighted_absorption(_A2_ALPHA_P)
    assert result.alpha_w == pytest.approx(_A2_ALPHA_W)
    assert result.shape_indicator == _A2_INDICATOR
    _assert_one_page(str(result.report(str(tmp_path / "a2.pdf"))))


def test_third_octave_fiche_renders_and_round_trips(tmp_path) -> None:
    """A rating built from one-third-octave alpha_s renders the full-table fiche.

    The retained alpha_s round-trips on the result, and the accredited
    one-third-octave table renders a valid one-page PDF.
    """
    result = weighted_absorption_from_third_octave(_THIRD_OCTAVE_ALPHA_S)
    np.testing.assert_allclose(result.third_octave_alpha_s, _THIRD_OCTAVE_ALPHA_S)
    np.testing.assert_allclose(
        result.third_octave_bands,
        [
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
        ],
    )
    out = tmp_path / "third_octave.pdf"
    result.report(str(out))
    _assert_one_page(str(out))


def test_third_octave_fiche_with_metadata(tmp_path) -> None:
    """The one-third-octave fiche renders one page with a full metadata header."""
    result = weighted_absorption_from_third_octave(_THIRD_OCTAVE_ALPHA_S)
    out = tmp_path / "third_octave_meta.pdf"
    result.report(str(out), metadata=_full_metadata(requirement=0.55))
    _assert_one_page(str(out))


def test_statement_writes_shape_indicator_without_space(tmp_path) -> None:
    """The boxed rating is written ``0.60(M)``, the ISO 11654 5.3 style.

    The clause 5.3 example prints the shape indicator immediately after the
    value ("0,70(MH)"), matching ``rating_label``; where an indicator
    applies, the 5.3 NOTE recommendation (use alpha_w together with the
    complete curve) is printed as a footnote.
    """
    from pypdf import PdfReader

    result = weighted_absorption_from_third_octave(_THIRD_OCTAVE_ALPHA_S)
    assert result.rating_label == "0.60(M)"
    out = tmp_path / "statement.pdf"
    result.report(str(out))
    text = "\n".join(page.extract_text() for page in PdfReader(str(out)).pages).replace(
        "\n", " "
    )
    assert "0.60(M)" in text
    assert "0.60 (M)" not in text
    assert "5.3 NOTE" in text  # shape-indicator recommendation footnote


def test_plain_rating_without_alpha_s_still_renders(tmp_path) -> None:
    """A plain weighted_absorption result (alpha_s None) falls back and renders."""
    result = weighted_absorption(_A2_ALPHA_P)
    assert result.third_octave_alpha_s is None
    out = tmp_path / "plain.pdf"
    result.report(str(out))
    _assert_one_page(str(out))


def test_verbose_renders_evaluation_table(tmp_path) -> None:
    """``verbose=True`` renders the ISO 11654 evaluation-column one-pager."""
    result = weighted_absorption(_A2_ALPHA_P)
    out = tmp_path / "verbose.pdf"
    result.report(str(out), verbose=True)
    _assert_one_page(str(out))


def _full_metadata(**overrides) -> ReportMetadata:
    base = {
        "specimen": "50 mm porous absorber over a 100 mm air gap",
        "client": "Acoustic Test Client Ltd.",
        "manufacturer": "Acoustics Works Inc.",
        "area": 10.8,
        "mounting": "Type A (against a rigid wall)",
        "test_room": "Reverberation room R1",
        "measurement_standard": "ISO 354",
        "test_date": "2026-07-20",
        "temperature": 21.4,
        "relative_humidity": 54.0,
        "pressure": 101.0,
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-11654",
    }
    base.update(overrides)
    return ReportMetadata(**base)


def test_full_metadata_renders_one_page(tmp_path) -> None:
    """A full ReportMetadata renders a one-page accredited absorption fiche."""
    result = weighted_absorption(_A2_ALPHA_P)
    out = tmp_path / "meta.pdf"
    result.report(str(out), metadata=_full_metadata())
    _assert_one_page(str(out))


def test_requirement_pass_and_fail_both_render(tmp_path) -> None:
    """A PASS and a FAIL alpha_w requirement both render a one-page fiche."""
    result = weighted_absorption(_A1_ALPHA_P)  # alpha_w = 0.60
    passing = tmp_path / "pass.pdf"
    failing = tmp_path / "fail.pdf"
    result.report(str(passing), metadata=_full_metadata(requirement=0.55))
    result.report(str(failing), metadata=_full_metadata(requirement=0.80))
    _assert_one_page(str(passing))
    _assert_one_page(str(failing))


def test_report_escapes_xml_specials_in_metadata(tmp_path) -> None:
    """Metadata with XML specials (& < >) renders without crashing reportlab."""
    result = weighted_absorption(_A1_ALPHA_P)
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="absorber <A> & baffle",
        laboratory="Lab & Sons",
        operator="A <B>",
        report_id="R&D-011",
        measurement_standard="ISO 354 & Annex",
    )
    out = tmp_path / "xml.pdf"
    result.report(str(out), metadata=md)
    _assert_one_page(str(out))


def _extract_text(path: str) -> str:
    """The concatenated text of every page (for language assertions)."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """``language="es"`` renders a one-page Spanish fiche with comma decimals."""
    import re

    result = weighted_absorption_from_third_octave(_THIRD_OCTAVE_ALPHA_S)
    out = tmp_path / "absorption_es.pdf"
    result.report(
        str(out),
        metadata=ReportMetadata(requirement=0.55, temperature=21.4),
        language="es",
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Índice de absorción acústica" in text
    assert "CUMPLE" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = weighted_absorption_from_third_octave(_THIRD_OCTAVE_ALPHA_S)
    with pytest.raises(ValueError, match="language"):
        result.report(str(tmp_path / "bad.pdf"), language="xx")
