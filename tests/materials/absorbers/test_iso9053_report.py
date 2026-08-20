#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 9053-1:2018 static airflow-resistance report (``.report()``).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written, the displayed key quantities match the
clean-room oracle derived from the standard's own clause 7.5 procedure (the
through-origin fit ``dp = a*u + b*u**2`` read at ``u = 0.5 mm/s`` gives
``R_s = a + b*u``, the airflow resistance ``R = R_s/A`` and the resistivity
``sigma = R_s/d``), the metric labels and metadata appear, unknown
engines/languages are rejected, XML specials in metadata do not break
reportlab, and the Spanish fiche uses a decimal comma. Pixel or layout content
is never inspected.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("reportlab")

from phonometry import (
    ReportMetadata,
    materials,
)

_PDF_MAGIC = b"%PDF"

# The committed clean-room example: a 50 mm porous absorber in a 100 mm diameter
# cell (A = pi*0.05^2 m2), stepped to 12 mm/s (below the 15 mm/s clause-7.5
# limit), with the pressure difference exactly on the through-origin quadratic
# dp = a*u + b*u^2, a = 16000 Pa*s/m and b = 400000 Pa*s^2/m^2.
# See scripts/generate_reports.py.
_A_COEFF = 1.6e4
_B_COEFF = 4.0e5
_AREA = math.pi * 0.05**2
_THICKNESS = 0.05
_U_EVAL = 0.5e-3

# Clean-room oracle, independent of the library implementation.
_RS = _A_COEFF + _B_COEFF * _U_EVAL  # 16200 Pa*s/m
_R = _RS / _AREA  # ~2062645 Pa*s/m^3
_SIGMA = _RS / _THICKNESS  # 324000 Pa*s/m^2


def _result():
    u = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0]) * 1e-3
    dp = _A_COEFF * u + _B_COEFF * u**2
    return materials.static_airflow_resistance(u, dp, area=_AREA, thickness=_THICKNESS)


def _metadata(**overrides) -> ReportMetadata:
    base = {
        "specimen": "50 mm porous absorber (open-cell)",
        "client": "Acoustic Test Client Ltd.",
        "manufacturer": "Acoustics Works Inc.",
        "thickness": 0.050,
        "test_room": "Static airflow rig, 100 mm cell",
        "measurement_standard": "ISO 9053-1",
        "temperature": 23.0,
        "relative_humidity": 50.0,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-9053",
    }
    base.update(overrides)
    return ReportMetadata(**base)


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert len(PdfReader(path).pages) == 1


def _raw_text(path: str) -> str:
    """The extracted PDF text with its line breaks kept (for row anchoring)."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def _text(path: str) -> str:
    """The extracted PDF text flattened onto one line (for single-line boxes)."""
    return _raw_text(path).replace("\n", " ")


def test_writes_one_page_pdf(tmp_path) -> None:
    out = tmp_path / "airflow.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_displayed_values_match_oracle(tmp_path) -> None:
    """The fiche prints R_s (boxed), R, sigma and the evaluation velocity.

    The oracle is derived from the standard's clause 7.5 procedure, independent
    of the library: R_s = a + b*u_eval, R = R_s/A, sigma = R_s/d, all evaluated
    at the reference velocity u = 0.5 mm/s.
    """
    assert round(_RS) == 16200
    assert round(_SIGMA) == 324000
    assert round(_R) == 2062648

    out = tmp_path / "airflow.pdf"
    _result().report(str(out), metadata=_metadata())
    text = _text(str(out))
    # Each value is bound to its own row/box statement so it cannot pass on a
    # coincidental chart tick or another number.
    assert "Specific airflow resistance" in text
    assert f"= {round(_RS)} Pa" in text  # boxed R_s
    assert f"Airflow resistance R = {round(_R)} Pa" in text
    assert f"= {round(_SIGMA)} Pa" in text  # resistivity sigma
    assert "Evaluated at u = 0.5 mm/s" in text
    assert "Acoustic Test Client Ltd." in text  # metadata


def test_metadata_and_fit_rows_appear(tmp_path) -> None:
    """The specimen thickness and the through-origin fit coefficients appear."""
    out = tmp_path / "airflow.pdf"
    _result().report(str(out), metadata=_metadata())
    text = _text(str(out))
    # Thickness d = 0.050 m shown in millimetres.
    assert "50" in text
    assert "Zero-velocity resistance" in text
    assert "16000" in text  # linear coefficient a
    assert "400000" in text  # quadratic coefficient b


def test_no_metadata_still_renders(tmp_path) -> None:
    out = tmp_path / "airflow_bare.pdf"
    _result().report(str(out))
    _assert_one_page(str(out))
    assert "ISO 9053-1:2018" in _text(str(out))  # standard-basis line


def test_no_thickness_omits_resistivity(tmp_path) -> None:
    """Without a thickness the fiche prints no airflow-resistivity row/term."""
    u = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0]) * 1e-3
    dp = _A_COEFF * u + _B_COEFF * u**2
    result = materials.static_airflow_resistance(u, dp, area=_AREA)
    out = tmp_path / "airflow_nod.pdf"
    result.report(str(out))
    _assert_one_page(str(out))
    assert "Airflow resistivity" not in _text(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="Unknown language"):
        result.report(out, language="xx")


def test_metadata_xml_specials_do_not_break(tmp_path) -> None:
    out = tmp_path / "airflow_xml.pdf"
    _result().report(str(out), metadata=_metadata(specimen='Foam <A> & <B> "edge"'))
    _assert_one_page(str(out))


def test_spanish_uses_comma_decimal(tmp_path) -> None:
    out = tmp_path / "airflow_es.pdf"
    _result().report(str(out), metadata=_metadata(), language="es")
    _assert_one_page(str(out))
    text = _text(str(out))
    # Spanish fixed strings and the decimal comma on the evaluation velocity.
    assert "Resistencia especifica al flujo de aire" in text.replace("í", "i")
    assert "Evaluado a u = 0,5 mm/s" in text
    assert "16200" in text
