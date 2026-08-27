#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the reverberation-time prediction report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written for a five-model prediction, the five model
names and the closed-form per-model values appear in the rendered text, the
fiche reads as a prediction (no PASS/FAIL verdict, only a target reference line
when a target is supplied), unknown engines/languages are rejected, XML
specials in metadata do not break reportlab, and the Spanish fiche is
translated with comma decimals. Pixel or layout content is never inspected.

The displayed values are produced by the classical closed-form models
themselves (each anchored by the reverberation_prediction module tests), so the
oracle is re-derived here through the public API rather than hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("svglib")
pytest.importorskip("pypdf")

import numpy as np
from report_assertions import assert_one_page

from phonometry import ReportMetadata, room

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

_MODELS = ("Sabine", "Eyring", "Millington-Sette", "Fitzroy", "Arau-Puchades")
# A shoebox 8 x 5 x 3 m (V = 120 m3, S = 158 m2) with an anisotropic
# absorption distribution over the octave bands (one treated wall pair).
_FREQS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
_TREATED = [0.10, 0.15, 0.30, 0.45, 0.55, 0.60]
_SIDE = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
_FLOOR = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]


def _result() -> room.ReverberationModelResult:
    return room.reverberation_time_models(
        (8.0, 5.0, 3.0), (_TREATED, _SIDE, _FLOOR), frequencies=_FREQS
    )


def _metadata(**overrides: float) -> ReportMetadata:
    base = {
        "specimen": "Classroom, one wall lined with a broadband absorber",
        "client": "Acoustic Test Client Ltd.",
        "test_room": "Classroom C1",
        "temperature": 20.0,
        "relative_humidity": 50.0,
        "pressure": 101.3,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-REVERB",
    }
    base.update(overrides)
    return ReportMetadata(**base)


def _text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages).replace(
        "\n", " "
    )


def test_report_writes_one_page_pdf(tmp_path: Path) -> None:
    out = tmp_path / "reverb.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_report_with_metadata_one_page(tmp_path: Path) -> None:
    out = tmp_path / "reverb_meta.pdf"
    _result().report(str(out), metadata=_metadata())
    assert_one_page(str(out))


def test_no_metadata_still_renders(tmp_path: Path) -> None:
    out = tmp_path / "reverb_bare.pdf"
    _result().report(str(out), metadata=None)
    assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    res = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine 'weasyprint'"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    res = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="Unknown language"):
        res.report(out, language="xx")


def test_all_model_names_render(tmp_path: Path) -> None:
    """The five model column headers appear in the rendered fiche text."""
    out = tmp_path / "models.pdf"
    _result().report(str(out), metadata=_metadata())
    text = _text(str(out))
    for name in _MODELS:
        assert name in text


def test_band_labels_and_per_model_values_render(tmp_path: Path) -> None:
    """Nominal band labels and closed-form per-model T (2 decimals) render."""
    res = _result()
    out = tmp_path / "values.pdf"
    res.report(str(out), metadata=_metadata())
    text = _text(str(out))
    for label in ("125", "500", "1000", "4000"):
        assert label in text
    # The 500 Hz Arau-Puchades and Sabine times, formatted to two decimals.
    arau_500 = f"{res.arau_puchades[2]:.2f}"
    sabine_500 = f"{res.sabine[2]:.2f}"
    assert arau_500 in text
    assert sabine_500 in text


def test_mid_frequency_prediction_descriptor_renders(tmp_path: Path) -> None:
    """The boxed Arau-Puchades mid-frequency prediction appears in the text."""
    res = _result()
    out = tmp_path / "mid.pdf"
    res.report(str(out), metadata=_metadata())
    text = _text(str(out))
    t_mid = 0.5 * (res.arau_puchades[2] + res.arau_puchades[3])
    assert f"{t_mid:.2f}" in text
    # It is a prediction, so it is labelled as such (Arau-Puchades descriptor).
    assert "Predicted" in text
    assert "Arau-Puchades" in text


def test_prediction_has_no_pass_fail_verdict(tmp_path: Path) -> None:
    """A prediction never invents a PASS/FAIL verdict, even with a target."""
    out = tmp_path / "noverdict.pdf"
    _result().report(str(out), metadata=_metadata(requirement=0.8))
    text = _text(str(out))
    assert "PASS" not in text
    assert "FAIL" not in text
    # The target is instead shown as a reference line.
    assert "Target reverberation time" in text


def test_metadata_xml_specials_do_not_break(tmp_path: Path) -> None:
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="hall <A> & foyer",
        test_room="Room & Stage",
        laboratory="Lab & Sons",
        report_id="R&D-REVERB",
    )
    out = tmp_path / "xml.pdf"
    _result().report(str(out), metadata=md)
    assert_one_page(str(out))


def test_spanish_fiche_uses_comma_decimal(tmp_path: Path) -> None:
    import re

    out = tmp_path / "reverb_es.pdf"
    _result().report(str(out), metadata=_metadata(requirement=0.8), language="es")
    assert_one_page(str(out))
    text = _text(str(out))
    assert "Predicción del tiempo de reverberación" in text
    assert "Tiempo de reverberación objetivo" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator
    assert "PASS" not in text
    assert "FAIL" not in text
