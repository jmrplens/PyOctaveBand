#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the IEC 61260-1 filter-class-compliance report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural facts:
``filter_class_compliance`` carries the bank data and agrees with
``verify_filter_class``; a class-1 bank renders a valid single-page COMPLIES
fiche; unknown engines are rejected; the required-class verdict renders both
ways; and a non-compliant bank renders its non-compliance fiche. The class
verification itself is validated against the standard's Table 1 elsewhere
(tests/filters/test_compliance.py).
"""

from __future__ import annotations

import os
import pickle

import pytest

pytest.importorskip("reportlab")

from phonometry import (
    OctaveFilterBank,
    ReportMetadata,
    filter_class_compliance,
    verify_filter_class,
)

_PDF_MAGIC = b"%PDF"


def _class1_bank() -> OctaveFilterBank:
    """A default Butterworth octave bank that meets IEC 61260-1:2014 class 1."""
    return OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[125, 4000])


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def test_filter_class_compliance_carries_bank_data() -> None:
    """The result packages the bank data and agrees with verify_filter_class."""
    bank = OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[500, 2000])
    result = filter_class_compliance(bank)
    verdict = verify_filter_class(bank)

    assert result.overall_class == verdict["overall_class"]
    assert len(result.bands) == bank.num_bands
    assert len(result.sos) == bank.num_bands
    assert result.band_frequencies.shape == (bank.num_bands,)
    assert result.factors == tuple(int(f) for f in bank.factor)
    assert result.fs == float(bank.fs)
    assert result.fraction == int(bank.fraction)
    assert result.edition == "2014"
    # The per-band margins survive the packaging unchanged.
    for stored, computed in zip(result.bands, verdict["bands"], strict=True):
        assert stored["margin_class1_db"] == pytest.approx(
            computed["margin_class1_db"]
        )
    # Frozen and picklable (like the other result dataclasses).
    pickle.loads(pickle.dumps(result))


def test_class1_bank_reports_complies(tmp_path) -> None:
    """A class-1 bank renders a valid one-page COMPLIES fiche."""
    result = filter_class_compliance(_class1_bank())
    assert result.overall_class == 1
    out = tmp_path / "iec.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_1995_edition_reports_class0(tmp_path) -> None:
    """The 1995 edition keeps class 0; a high-order bank renders a Class 0 fiche."""
    bank = OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[250, 4000])
    result = filter_class_compliance(bank, edition="1995")
    assert result.edition == "1995"
    assert result.overall_class == 0
    assert 0 in result.available_classes()  # class 0 only exists in the 1995 mask
    out = tmp_path / "iec1995.pdf"
    result.report(str(out), metadata=ReportMetadata(
        measurement_standard="IEC 61260:1995", required_class=0))
    _assert_one_page(str(out))


def test_full_metadata_renders_one_page(tmp_path) -> None:
    """A populated ReportMetadata renders a one-page filter-compliance fiche."""
    result = filter_class_compliance(_class1_bank())
    md = ReportMetadata(
        specimen="1/1-octave filter bank",
        client="Acoustic Test Client Ltd.",
        manufacturer="Instrument Works Inc.",
        test_room="Electroacoustics laboratory H1",
        measurement_standard="IEC 61260-1:2014",
        test_date="2026-07-20",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-61260",
        required_class=1,
    )
    out = tmp_path / "meta.pdf"
    result.report(str(out), metadata=md)
    _assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = filter_class_compliance(_class1_bank())  # hoisted out of raises (S5778)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_required_class_pass_and_fail_both_render(tmp_path) -> None:
    """A PASS (class 1 meets required 1) and a FAIL (meets no class) both render."""
    result = filter_class_compliance(_class1_bank())
    assert result.overall_class == 1
    passing = tmp_path / "pass.pdf"
    result.report(str(passing), metadata=ReportMetadata(required_class=1))
    _assert_one_page(str(passing))
    # FAIL case: a low-order bank meets no class of the edition.
    failing_result = filter_class_compliance(
        OctaveFilterBank(fs=48000, fraction=1, order=1, limits=[500, 2000])
    )
    assert failing_result.overall_class is None
    failing = tmp_path / "fail.pdf"
    failing_result.report(str(failing), metadata=ReportMetadata(required_class=1))
    _assert_one_page(str(failing))


def test_required_class_missing_from_edition_rejected(tmp_path) -> None:
    """Class 0 against a 2014-edition result raises: the class does not exist.

    IEC 61260-1:2014 defines only classes 1 and 2; a required class 0 verdict
    against a 2014 verification would silently render a meaningless FAIL, so
    it is rejected with a pointer to the 1995 edition.
    """
    result = filter_class_compliance(_class1_bank())
    out = str(tmp_path / "class0.pdf")
    meta = ReportMetadata(required_class=0)
    with pytest.raises(ValueError, match="edition"):
        result.report(out, metadata=meta)


def test_fiche_labels_bands_with_nominal_frequencies(tmp_path) -> None:
    """The per-band table uses the nominal mid-band frequencies.

    Both editions identify the filters by their nominal frequencies
    (2014 5.5 / 1995 4.2): the fiche must print 125 Hz and 4 kHz, never the
    exact base-ten 125.89.. / 3981.. Hz behind them.
    """
    result = filter_class_compliance(_class1_bank())
    out = tmp_path / "nominal.pdf"
    result.report(str(out))
    text = _extract_text(str(out))
    assert "125 Hz" in text
    assert "4 kHz" in text
    assert "126 Hz" not in text
    assert "3981" not in text


def test_range_limited_verdict_prints_qualifying_note(tmp_path) -> None:
    """A range-limited COMPLIES is qualified on the fiche.

    The multirate verification cannot exercise the stop-band mask beyond each
    band's processing Nyquist, so the result carries ``range_limited`` and
    the fiche prints the qualification next to the stated class.
    """
    result = filter_class_compliance(_class1_bank())
    assert result.range_limited is True
    for band in result.bands:
        assert band["checked_to_omega"] > 0.0
    out = tmp_path / "qualified.pdf"
    result.report(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "COMPLIES" in text
    assert "processing Nyquist frequency" in text
    assert "not demonstrated" in text


def test_non_compliant_bank_renders(tmp_path) -> None:
    """A low-order bank that meets no class renders its non-compliance fiche."""
    bank = OctaveFilterBank(fs=48000, fraction=1, order=1, limits=[500, 2000])
    result = filter_class_compliance(bank)
    assert result.overall_class is None
    out = tmp_path / "noncompliant.pdf"
    result.report(str(out))
    _assert_one_page(str(out))


def test_empty_bands_result_is_graceful() -> None:
    """A zero-band result reports no classes and fails clearly, not with IndexError."""
    import numpy as np

    from phonometry.filters.compliance import FilterComplianceResult

    empty = FilterComplianceResult(
        overall_class=None, bands=(), fraction=1, edition="2014",
        sos=(), band_frequencies=np.asarray([], dtype=float), factors=(),
        fs=48000.0, num_points=2048,
    )
    assert empty.available_classes() == []
    with pytest.raises(ValueError, match="no bands"):
        empty.reference_class()
    with pytest.raises(ValueError, match="no bands"):
        empty.report("/dev/null")


def _extract_text(path: str) -> str:
    """The concatenated text of every page (for language assertions)."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """``language="es"`` renders a one-page Spanish fiche with comma decimals."""
    import re

    result = filter_class_compliance(_class1_bank())
    out = tmp_path / "filter_es.pdf"
    result.report(str(out), metadata=ReportMetadata(required_class=1), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Conformidad de clase de filtro" in text
    assert "CUMPLE" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal margins
    assert "margen" in text


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = filter_class_compliance(_class1_bank())
    with pytest.raises(ValueError, match="language"):
        result.report(str(tmp_path / "bad.pdf"), language="xx")
