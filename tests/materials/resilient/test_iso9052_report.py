#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the EN 29052-1 / ISO 9052-1 dynamic-stiffness report (``.report()``).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written, the displayed key quantities match the
clean-room oracle derived from the standard's own formulae (Formula 4 for the
apparent dynamic stiffness, Clause 8.2 NOTE for the enclosed-gas term, Formula 6
for the installed stiffness and Formula 2 for the natural frequency), the metric
labels and metadata appear, unknown engines/languages are rejected, XML specials
in metadata do not break reportlab, and the Spanish fiche uses a decimal comma.
Pixel or layout content is never inspected.
"""

from __future__ import annotations

import math
import re

import pytest

pytest.importorskip("reportlab")

from typing import TYPE_CHECKING

from report_assertions import assert_one_page

from phonometry import (
    ReportMetadata,
    materials,
)

if TYPE_CHECKING:
    from pathlib import Path

# The committed clean-room example: the standard 8 kg load plate over the
# 0.04 m2 specimen (m't = 200 kg/m2, Clauses 5-6), a measured resonance of
# fr = 45.0 Hz, a 20 mm layer of porosity 0.9 at an intermediate airflow
# resistivity r = 50 kPa.s/m2, installed under a 110 kg/m2 floating screed.
# See scripts/generate_reports.py.
_FR = 45.0
_MT = 200.0
_MFLOOR = 110.0
_THICKNESS = 0.020
_POROSITY = 0.9
_RESISTIVITY = 50.0


def _result() -> materials.DynamicStiffnessResult:
    return materials.floating_floor_resonance(
        resonant_frequency=_FR,
        total_mass_per_area=_MT,
        floor_mass_per_area=_MFLOOR,
        airflow_resistivity=_RESISTIVITY,
        thickness=_THICKNESS,
        porosity=_POROSITY,
    )


def _metadata(**overrides: object) -> ReportMetadata:
    base: dict[str, object] = {
        "specimen": "20 mm mineral-wool resilient layer",
        "client": "Acoustic Test Client Ltd.",
        "manufacturer": "Acoustics Works Inc.",
        "mass_per_area": 200.0,
        "thickness": 0.020,
        "test_room": "Dynamic-stiffness rig, 8 kg load plate",
        "measurement_standard": "EN 29052-1",
        "temperature": 21.0,
        "relative_humidity": 50.0,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-29052",
    }
    base.update(overrides)
    return ReportMetadata(**base)  # type: ignore[arg-type]


def _raw_text(path: str) -> str:
    """The extracted PDF text with its line breaks kept (for row anchoring)."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def _text(path: str) -> str:
    """The extracted PDF text flattened onto one line (for single-line boxes)."""
    return _raw_text(path).replace("\n", " ")


# The prime (U+2032) reportlab renders for the ``s'`` symbols, as it extracts.
_PRIME = "′"


def test_writes_one_page_pdf(tmp_path: Path) -> None:
    out = tmp_path / "dyn.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_displayed_values_match_oracle(tmp_path: Path) -> None:
    """The fiche prints s't (Formula 4), s'a, s', fr and f0, and the metadata.

    The oracle is derived from the standard's formulae, independent of the
    library: s't = 4*pi^2 * m't * fr^2, s'a = p0/(d*eps), s' = s't + s'a
    (Formula 6, intermediate airflow resistivity), f0 = (1/2pi) sqrt(s'/m').
    Clause 9 rounds the stiffnesses to the nearest MN/m3.
    """
    s_t = 4.0 * math.pi**2 * _MT * _FR**2 / 1e6
    s_a = 1.0e5 / (_THICKNESS * _POROSITY) / 1e6
    s_installed = s_t + s_a
    f0 = math.sqrt(s_installed * 1e6 / _MFLOOR) / (2.0 * math.pi)
    assert round(s_t) == 16
    assert round(s_a) == 6
    assert round(s_installed) == 22
    assert round(f0, 1) == 70.4

    out = tmp_path / "dyn.pdf"
    _result().report(str(out), metadata=_metadata())
    text = _text(str(out))
    raw = _raw_text(str(out))
    # Each value is bound to its own row/box statement so it cannot pass on a
    # coincidental chart tick or another number.
    assert f"Apparent dynamic stiffness s{_PRIME}t = 16 MN/m" in text
    assert f"Dynamic stiffness s{_PRIME} = 22 MN/m" in text
    assert "Resonant frequency fr = 45.0 Hz" in text
    assert "Natural frequency f0 = 70.4 Hz" in text
    # The enclosed-gas term is a table row (label above, value below).
    assert re.search(rf"s{_PRIME}a \[MN/m3\]\s+6\b", raw)
    assert "Acoustic Test Client Ltd." in text  # metadata


def test_metadata_fields_appear(tmp_path: Path) -> None:
    """The mass per area m't and the loaded thickness d appear in the header."""
    out = tmp_path / "dyn.pdf"
    _result().report(str(out), metadata=_metadata())
    raw = _raw_text(str(out))
    assert re.search(rf"Total mass per area m{_PRIME}t\s*\[kg/m2\]:\s*200\b", raw)
    # Thickness d = 0.020 m is shown in millimetres.
    assert re.search(r"Thickness under load d\s*\[mm\]:\s*20\b", raw)
    assert re.search(rf"m{_PRIME} \[kg/m2\]\s+110\b", raw)  # supported-floor mass


def test_no_metadata_still_renders(tmp_path: Path) -> None:
    out = tmp_path / "dyn_bare.pdf"
    _result().report(str(out))
    assert_one_page(str(out))
    assert "EN 29052-1" in _text(str(out))  # standard-basis line


def test_high_resistivity_omits_gas_term(tmp_path: Path) -> None:
    """For r >= 100 kPa.s/m2 the installed s' equals s't and no s'a row shows."""
    out = tmp_path / "dyn_hi.pdf"
    result = materials.floating_floor_resonance(
        resonant_frequency=_FR,
        total_mass_per_area=_MT,
        floor_mass_per_area=_MFLOOR,
    )
    result.report(str(out), metadata=_metadata())
    assert_one_page(str(out))
    assert "Enclosed-gas stiffness" not in _text(str(out))


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="Unknown language"):
        result.report(out, language="xx")


def test_metadata_xml_specials_do_not_break(tmp_path: Path) -> None:
    out = tmp_path / "dyn_xml.pdf"
    _result().report(str(out), metadata=_metadata(specimen='Layer <A> & <B> "edge"'))
    assert_one_page(str(out))


def test_spanish_uses_comma_decimal(tmp_path: Path) -> None:
    out = tmp_path / "dyn_es.pdf"
    _result().report(str(out), metadata=_metadata(), language="es")
    assert_one_page(str(out))
    text = _text(str(out))
    # Each Spanish decimal comma is anchored to its labelled box statement.
    assert "Frecuencia de resonancia fr = 45,0 Hz" in text
    assert "Frecuencia natural f0 = 70,4 Hz" in text
    assert f"Rigidez dinámica aparente s{_PRIME}t = 16 MN/m" in text
