#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the RD 1367/2007 activity inspection fiche (``.report()`` -> PDF).

The fiche is a rendering feature, so these tests assert structural facts only:
a valid single-page PDF is written, the fiche defaults to Spanish (the language
of the regulation it applies) and translates its limit row with it, the phase
and per-period tables carry the numbers the assessment computed, the verdict
renders both ways, XML specials in metadata survive reportlab, and a fiche with
all three evaluation periods still fits one page. The assessment maths itself is
validated against the regulation and the published worked examples elsewhere
(tests/environment/assessment/test_spain.py).

The case rendered here is the worked example of Aviles Lopez & Perera Martin,
Manual de acustica ambiental y arquitectonica, Ejemplos 3.1 to 3.3: an activity
on residential land whose day period reaches LKeq,d = 57 dB and LK,d = 56 dB
against a 55 dB limit, so a new activity does not comply.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

from phonometry import ReportMetadata
from phonometry.environment.assessment import spain as rd

_PDF_MAGIC = b"%PDF"


def _measurements() -> dict[str, list[rd.NoisePhase]]:
    """The day and evening phases of Manual Ejemplo 3.1."""
    return {
        "day": [
            rd.NoisePhase(2.0, 0.0, label="Actividad cerrada"),
            rd.NoisePhase(6.0, 50.0, kt=6.0, kf=3.0, label="Maquina ruidosa"),
            rd.NoisePhase(4.0, 48.0, kt=3.0, kf=3.0, label="Resto de fuentes"),
        ],
        "evening": [
            rd.NoisePhase(2.0, 48.0, kt=3.0, kf=3.0, label="Resto de fuentes"),
            rd.NoisePhase(2.0, 0.0, label="Actividad cerrada"),
        ],
    }


def _result(**kwargs: object) -> rd.ActivityAssessment:
    """The Ejemplo 3.3 assessment against the area type a limits."""
    return rd.assess_activity(
        _measurements(),
        rd.activity_limits("a"),
        operating_days=303,
        **kwargs,  # type: ignore[arg-type]
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
    """An activity assessment renders a one-page PDF fiche."""
    out = tmp_path / "acta.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="xx")


def test_fiche_defaults_to_spanish(tmp_path) -> None:
    """The fiche renders in Spanish by default, the language of the regulation.

    Its limit row is carried on the result in English, the library's API
    language, so the renderer must translate that too rather than leaving an
    English line in a Spanish acta.
    """
    out = tmp_path / "acta_es.pdf"
    _result().report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "Valoración acústica de actividad (RD 1367/2007)" in text
    assert "Fases de ruido" in text
    assert "Día" in text
    assert "Tarde" in text
    assert "anexo III tabla B1" in text
    # No English leaks from the limit row or the fixed strings.
    assert "Annex III" not in text
    assert "Area type" not in text
    assert "Noise phase" not in text


def test_english_fiche_renders_one_page(tmp_path) -> None:
    """``language="en"`` renders the same fiche in English on one page."""
    out = tmp_path / "acta_en.pdf"
    _result().report(str(out), language="en")
    _assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "Activity noise assessment (RD 1367/2007)" in text
    assert "Noise phases" in text
    assert "Day" in text


def test_report_states_the_phase_and_period_numbers(tmp_path) -> None:
    """The fiche prints the corrections, the phase levels and the period result.

    Ejemplo 3.1 gives LKeq,Ti of 59 dB and 54 dB from LAeq,Ti of 50 dB and
    48 dB with Kt 6/3 and Kf 3/3; the period integration gives LKeq,d = 57 dB
    and the annual average LK,d = 56 dB against the 55 dB of Table B1.
    """
    result = _result()
    out = tmp_path / "numbers.pdf"
    result.report(str(out), language="en")
    text = _extract_text(str(out)).replace("\n", " ")

    day = result.periods[0]
    assert day.reported_level == 57
    assert day.reported_long_term == 56
    assert day.max_phase_level == 59.0

    assert "59.0" in text  # the governing phase LKeq,Ti
    assert "54.0" in text  # the quieter phase
    assert "57 / 58" in text  # daily LKeq,x against limit + 3 dB
    assert "56 / 55" in text  # annual LK,x against the table limit
    assert "59 / 60" in text  # measured phase level against limit + 5 dB


def test_verdict_fails_for_a_new_activity_and_passes_for_an_existing_one(
    tmp_path,
) -> None:
    """Manual Ejemplo 3.3: the same activity fails as new and passes as existing.

    Article 25.1 b applies the annual criterion to a new activity, which
    LK,d = 56 dB over a 55 dB limit fails; Article 25.2 subjects an activity
    already in operation only to the daily and phase criteria.
    """
    new = tmp_path / "new.pdf"
    existing = tmp_path / "existing.pdf"
    _result().report(str(new), language="en")
    _result(new_activity=False).report(str(existing), language="en")
    _assert_one_page(str(new))
    _assert_one_page(str(existing))

    new_text = _extract_text(str(new)).replace("\n", " ")
    existing_text = _extract_text(str(existing)).replace("\n", " ")
    assert "FAIL" in new_text
    assert "exceeds" in new_text
    assert "Article 25.1 b" in new_text
    assert "FAIL" not in existing_text
    assert "PASS" in existing_text
    assert "Article 25.2" in existing_text


def test_spanish_verdict_labels(tmp_path) -> None:
    """The Spanish fiche renders the verdict as CUMPLE / NO CUMPLE."""
    out = tmp_path / "verdict_es.pdf"
    _result().report(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "NO CUMPLE" in text
    assert "supera" in text


def test_metadata_appears_and_stays_one_page(tmp_path) -> None:
    """A populated ReportMetadata renders one page and prints its fields."""
    md = ReportMetadata(
        specimen="Taller mecanico, horario 9 h a 21 h",
        client="Titular de la actividad",
        test_room="Ambiente exterior, punto de evaluacion mas desfavorable",
        instrumentation="Sonometro integrador-promediador clase 1",
        calibration="Verificacion antes y despues, desviacion 0,1 dB",
        test_date="2026-07-29",
        laboratory="Laboratorio de ensayos acusticos",
        operator="Tecnico responsable",
        report_id="EXP-2026-000123",
    )
    out = tmp_path / "meta.pdf"
    _result().report(str(out), metadata=md, verbose=True)
    _assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "Taller mecanico, horario 9 h a 21 h" in text
    assert "Sonometro integrador-promediador clase 1" in text
    assert "EXP-2026-000123" in text


def test_report_escapes_xml_specials_in_metadata_and_phase_labels(tmp_path) -> None:
    """XML specials (& < >) in metadata and phase labels survive reportlab.

    The phase label is free text that reaches the table as a paragraph, so it
    must be escaped on the same path as the metadata fields.
    """
    measurements = _measurements()
    measurements["day"][1] = rd.NoisePhase(
        6.0, 50.0, kt=6.0, kf=3.0, label="Compresor <A> & <B>"
    )
    result = rd.assess_activity(
        measurements, rd.activity_limits("a"), operating_days=303
    )
    md = ReportMetadata(
        specimen="Taller <A> & <B>",
        client="Ac & Co <Ltd>",
        test_room="pos <1> & <2>",
        report_id="R&D-1367",
    )
    out = tmp_path / "xml.pdf"
    result.report(str(out), metadata=md, language="en")
    _assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "Taller <A> & <B>" in text
    assert "Ac & Co <Ltd>" in text
    assert "Compresor <A> & <B>" in text
    assert "R&D-1367" in text


def test_three_period_fiche_stays_one_page(tmp_path) -> None:
    """A fiche carrying all three evaluation periods still fits one page.

    The phase table grows with the number of noise phases, so the assessment
    figure shrinks to keep the fiche on a single page.
    """
    measurements = _measurements()
    measurements["night"] = [
        rd.NoisePhase(4.0, 40.0, kt=3.0, label="Ventilacion"),
        rd.NoisePhase(4.0, 38.0, label="Parada"),
    ]
    result = rd.assess_activity(
        measurements, rd.activity_limits("a"), operating_days=303
    )
    md = ReportMetadata(
        specimen="Taller mecanico",
        client="Titular",
        test_room="Fachada, 1,5 m",
        report_id="EXP-2026-000123",
    )
    for language in ("es", "en"):
        for verbose in (False, True):
            out = tmp_path / f"three_{language}_{verbose}.pdf"
            result.report(
                str(out), metadata=md, language=language, verbose=verbose
            )
            _assert_one_page(str(out))


def test_adjacent_premises_limit_row_renders(tmp_path) -> None:
    """A fiche assessed against the Table B2 adjacent-premises row renders.

    Its limit row description names the premises and the room type, which the
    Spanish fiche must translate along with the table reference.
    """
    out = tmp_path / "adjacent.pdf"
    result = rd.assess_activity(
        _measurements(),
        rd.adjacent_premises_limits("residential", "bedrooms"),
        operating_days=303,
    )
    result.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "anexo III tabla B2" in text
    assert "dormitorios" in text
