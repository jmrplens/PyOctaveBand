#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the IEC 61043 class-verification report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural facts:
a compliant chain renders a valid single-page COMPLIES fiche, the Spanish
fiche renders too, unknown engines and unknown required classes are rejected,
the required-class verdict renders both ways, a non-compliant chain renders its
non-conformity fiche, and a partial band set adds the range-limited note. The
class verification itself is validated against the standard's Table 2 in
tests/metrology/test_intensity_compliance.py.
"""

from __future__ import annotations

import os

import matplotlib
import pytest

matplotlib.use("Agg")

pytest.importorskip("reportlab")
pytest.importorskip("svglib")

import reference_data as ref

from phonometry import ReportMetadata, emission

_PDF_MAGIC = b"%PDF"

_BANDS = [row[0] for row in ref.IEC61043_TABLE2]


def _class1_chain(offset: float = 1.5, spacing: float = 0.025):
    """A complete instrument that clears the class 1 minima by ``offset`` dB."""
    freqs, class1, _ = emission.residual_index_limits("instrument", spacing=spacing)
    return emission.intensity_class_compliance(class1 + offset, freqs, spacing=spacing)


def _metadata(**kwargs: object) -> ReportMetadata:
    base: dict[str, object] = {
        "specimen": "p-p intensity probe and analyser",
        "client": "Example client",
        "manufacturer": "Example instruments",
        "test_room": "Electroacoustics laboratory (example)",
        "measurement_standard": "IEC 61043:1993",
        "test_date": "2026-07-29",
        "laboratory": "Phonometry reference example",
        "operator": "phonometry",
        "report_id": "EXAMPLE-61043",
    }
    base.update(kwargs)
    return ReportMetadata(**base)  # type: ignore[arg-type]


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def test_compliant_chain_renders_one_page(tmp_path) -> None:
    result = _class1_chain()
    path = result.report(str(tmp_path / "iec61043.pdf"), metadata=_metadata())
    _assert_one_page(path)


def test_bare_fiche_without_metadata(tmp_path) -> None:
    """A fiche without metadata still renders body, result and disclaimer."""
    result = _class1_chain()
    _assert_one_page(result.report(str(tmp_path / "bare.pdf")))


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def test_spanish_fiche_renders(tmp_path) -> None:
    result = _class1_chain()
    path = result.report(str(tmp_path / "es.pdf"), metadata=_metadata(), language="es")
    _assert_one_page(path)


def test_spanish_fiche_translates_every_fixed_string(tmp_path) -> None:
    """No English template survives into the Spanish fiche.

    ``_i18n.t`` falls back to the English text when a key is missing, so a
    typo in the translation table shows up as a stray English sentence rather
    than as an error. Read the rendered page back and assert the Spanish
    wording of each block: title, basis line, measurement-basis strip with its
    scope limiter, table headers, boxed result and verdict row.
    """
    result = _class1_chain()
    path = result.report(
        str(tmp_path / "es_text.pdf"),
        metadata=_metadata(required_class=1),
        language="es",
    )
    text = _extract_text(path)
    for spanish in (
        "Verificación de clase de instrumento de intensidad acústica",
        "Conformidad con IEC 61043:1993, tabla 2",
        "Índice presión-intensidad residual",
        "Evaluado frente a los mínimos de la tabla 2 para instrumento completo",
        "de la nota 1. Es uno de los requisitos",
        "Margen",
        "Clase",
        "CUMPLE",
        "Verificado como: instrumento completo",
        "Separación entre micrófonos",
        "Desfase equivalente",
        "Clase obtenida 1, clase exigida 1",
    ):
        assert spanish in text, spanish
    for english in (
        "Pressure-residual intensity index",
        "Judged against the Table 2 minima",
        "It is one IEC 61043 requirement",
        "Verified as:",
        "Microphone separation",
        "Equivalent phase mismatch",
        "COMPLIES",
        "Achieved class",
    ):
        assert english not in text, english


def test_required_class_verdict_both_ways(tmp_path) -> None:
    """The verdict row renders whether the chain meets the requirement or not."""
    passing = _class1_chain()
    _assert_one_page(
        passing.report(str(tmp_path / "pass.pdf"), metadata=_metadata(required_class=1))
    )
    # 1 dB short of class 1 everywhere, so the chain lands on class 2.
    freqs, class1, _ = emission.residual_index_limits("instrument")
    failing = emission.intensity_class_compliance(class1 - 1.0, freqs)
    assert failing.overall_class == 2
    _assert_one_page(
        failing.report(str(tmp_path / "fail.pdf"), metadata=_metadata(required_class=1))
    )


def test_non_compliant_chain_renders(tmp_path) -> None:
    """A chain meeting neither class boxes a non-conformity statement."""
    freqs, _, class2 = emission.residual_index_limits("probe")
    result = emission.intensity_class_compliance(class2 - 2.0, freqs, device="probe")
    assert result.overall_class is None
    _assert_one_page(result.report(str(tmp_path / "none.pdf"), metadata=_metadata()))


def test_range_limited_note_renders(tmp_path) -> None:
    """A partial band set adds the clause 6.1 qualifying note."""
    bands = _BANDS[6:14]
    freqs, class1, _ = emission.residual_index_limits("processor", frequencies=bands)
    result = emission.intensity_class_compliance(class1, freqs, device="processor")
    assert result.range_limited is True
    _assert_one_page(result.report(str(tmp_path / "partial.pdf"), metadata=_metadata()))


def test_range_limited_note_drops_the_class_wording_when_no_class_is_met(
    tmp_path,
) -> None:
    """With no class reached there is no "stated class" for the note to qualify.

    ``verify_intensity_class`` grants full-range status over the 7 octave bands
    only when the chain reaches class 2, so a chain that meets no class over
    exactly those bands is ``range_limited``. The note then sits directly under
    the "Does not comply" box, where a sentence about what "the stated class
    attests" has no referent.
    """
    octaves = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    freqs, _, class2 = emission.residual_index_limits("processor", frequencies=octaves)
    result = emission.intensity_class_compliance(
        class2 - 2.0, freqs, device="processor"
    )
    assert result.overall_class is None
    assert result.range_limited is True

    path = result.report(str(tmp_path / "noclass.pdf"), metadata=_metadata())
    _assert_one_page(path)
    text = _extract_text(path)
    assert "this verification covers the bands listed above" in text
    assert "the stated class attests" not in text

    es_path = result.report(
        str(tmp_path / "noclass_es.pdf"), metadata=_metadata(), language="es"
    )
    es_text = _extract_text(es_path)
    assert "esta verificación abarca las bandas listadas" in es_text
    assert "la clase indicada acredita" not in es_text


def test_unknown_engine_is_rejected(tmp_path) -> None:
    result = _class1_chain()
    target = str(tmp_path / "engine.pdf")
    with pytest.raises(ValueError, match="Unknown report engine"):
        result.report(target, engine="weasyprint")


def test_unknown_required_class_is_rejected(tmp_path) -> None:
    result = _class1_chain()
    target = str(tmp_path / "class0.pdf")
    metadata = _metadata(required_class=0)
    with pytest.raises(ValueError, match="not an IEC 61043"):
        result.report(target, metadata=metadata)


def test_unknown_language_is_rejected(tmp_path) -> None:
    result = _class1_chain()
    target = str(tmp_path / "xx.pdf")
    with pytest.raises(ValueError, match="Unknown language"):
        result.report(target, language="xx")
