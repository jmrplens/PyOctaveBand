#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the enclosed-space absorption/reverberation report (``.report()``).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written for an EN 12354-6 enclosed-space result, the
per-band absorption area A and reverberation time T appear in the rendered
text, the fiche is a characterisation (no PASS/FAIL verdict, only a target
reference line when a target is supplied), unknown engines/languages are
rejected, XML specials in metadata do not break reportlab, and the Spanish
fiche is translated with comma decimals. Pixel or layout content is never
inspected.

The displayed A and T are produced by the tested EN 12354-6 Formula 1 /
Formula 5 primitives, so the oracle is re-derived here through the public API
rather than hardcoded.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("svglib")
pytest.importorskip("pypdf")

from phonometry import (
    ReportMetadata,
    ReverberationResult,
    enclosed_space_reverberation,
    hard_object_absorption,
    object_fraction,
)

_PDF_MAGIC = b"%PDF"
_VOLUME = 50.0
_SURFACES = [
    (20.0, [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55]),
    (20.0, [0.20, 0.40, 0.65, 0.75, 0.80, 0.80, 0.75]),
    (45.0, [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05]),
]
_OBJECT_VOLUMES = [0.5, 0.8, 0.3]


def _result() -> ReverberationResult:
    objects = hard_object_absorption(_OBJECT_VOLUMES)
    psi = object_fraction(_OBJECT_VOLUMES, _VOLUME)
    return enclosed_space_reverberation(
        _SURFACES, _VOLUME, objects=objects, object_fraction=psi,
        air_condition="20C_50-70",
    )


def _metadata(**overrides) -> ReportMetadata:
    base = {
        "specimen": "Meeting room, furnished",
        "client": "Acoustic Test Client Ltd.",
        "test_room": "Meeting room M2",
        "measurement_standard": "EN 12354-6",
        "temperature": 20.0,
        "relative_humidity": 55.0,
        "pressure": 101.3,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-EN12354-6",
    }
    base.update(overrides)
    return ReportMetadata(**base)


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert len(PdfReader(path).pages) == 1


def _text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(
        page.extract_text() for page in PdfReader(path).pages
    ).replace("\n", " ")


def test_report_writes_one_page_pdf(tmp_path) -> None:
    out = tmp_path / "enclosed.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_report_with_metadata_one_page(tmp_path) -> None:
    out = tmp_path / "enclosed_meta.pdf"
    _result().report(str(out), metadata=_metadata())
    _assert_one_page(str(out))


def test_no_metadata_still_renders(tmp_path) -> None:
    out = tmp_path / "enclosed_bare.pdf"
    _result().report(str(out), metadata=None)
    _assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    res = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    res = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="Unknown language"):
        res.report(out, language="xx")


def test_band_labels_and_area_and_time_render(tmp_path) -> None:
    """Nominal band labels and the closed-form A and T (2 decimals) render."""
    res = _result()
    out = tmp_path / "values.pdf"
    res.report(str(out), metadata=_metadata())
    text = _text(str(out))
    for label in ("125", "1000", "8000"):
        assert label in text
    # The 1000 Hz absorption area and reverberation time, two decimals.
    assert f"{res.absorption_area[3]:.2f}" in text
    assert f"{res.reverberation_time[3]:.2f}" in text


def test_mid_frequency_descriptor_renders(tmp_path) -> None:
    """The boxed mid-frequency reverberation time appears in the text."""
    res = _result()
    out = tmp_path / "mid.pdf"
    res.report(str(out), metadata=_metadata())
    text = _text(str(out))
    t_mid = 0.5 * (res.reverberation_time[2] + res.reverberation_time[3])
    assert f"{t_mid:.2f}" in text


def test_characterisation_has_no_pass_fail_verdict(tmp_path) -> None:
    """The fiche never invents a PASS/FAIL verdict, even with a target."""
    out = tmp_path / "noverdict.pdf"
    _result().report(str(out), metadata=_metadata(requirement=0.6))
    text = _text(str(out))
    assert "PASS" not in text and "FAIL" not in text
    assert "Target reverberation time" in text


def test_metadata_xml_specials_do_not_break(tmp_path) -> None:
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="room <A> & <B>",
        test_room="Room & Stage",
        laboratory="Lab & Sons",
        report_id="R&D-EN12354",
        measurement_standard="EN 12354-6 & Annex",
    )
    out = tmp_path / "xml.pdf"
    _result().report(str(out), metadata=md)
    _assert_one_page(str(out))


def test_spanish_fiche_uses_comma_decimal(tmp_path) -> None:
    import re

    out = tmp_path / "enclosed_es.pdf"
    _result().report(str(out), metadata=_metadata(requirement=0.6), language="es")
    _assert_one_page(str(out))
    text = _text(str(out))
    assert "Absorción acústica en un recinto" in text
    assert "Tiempo de reverberación objetivo" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator
    assert "PASS" not in text and "FAIL" not in text
