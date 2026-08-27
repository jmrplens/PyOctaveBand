#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the wind-turbine tonal audibility report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written, unknown engines and languages are rejected,
XML specials in metadata do not break reportlab, the verdict renders both ways,
and the boxed tonal audibility ``ΔL_a``, the tone frequency and the audibility
decision appear in the extracted text. The tonality algorithm itself is
validated against the IEC 61400-11 oracle elsewhere
(tests/environment/sources/test_wind_turbine.py); this fiche test reuses the
same hand-derived synthetic tone so its numbers are documented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("reportlab")

from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.environment.sources.wind_turbine import (
    WindTurbineTonalityResult,
    wind_turbine_tonality,
)

if TYPE_CHECKING:
    from pathlib import Path


def _synthetic_tone() -> tuple[np.ndarray, np.ndarray]:
    """A clean 500 Hz tone 30 dB above a flat 30 dB floor (df = 2 Hz).

    Hand-derived in the module tests: ΔL_a = 16.38 dB, audible, at 500 Hz.
    """
    df = 2.0
    freqs = np.arange(440.0, 560.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[int(np.argmin(np.abs(freqs - 500.0)))] = 60.0
    return levels, freqs


def _result() -> WindTurbineTonalityResult:
    levels, freqs = _synthetic_tone()
    return wind_turbine_tonality(levels, freqs)


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_report_writes_one_page_pdf(tmp_path: Path) -> None:
    """An audible-tone result renders a one-page PDF fiche."""
    result = _result()
    out = tmp_path / "wt.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match=r"Unknown language"):
        result.report(out, language="xx")


def test_report_states_audibility_frequency_and_decision(tmp_path: Path) -> None:
    """The fiche states ΔL_a, the tone frequency and the audibility decision."""
    result = _result()
    out = tmp_path / "wt.pdf"
    result.report(str(out))
    text = _extract_text(str(out)).replace("\n", " ")

    assert result.is_audible is True
    assert "Tonal audibility" in text
    assert f"= {result.tonal_audibility:.1f} dB" in text  # boxed ΔL_a
    assert f"{result.tone_frequency:.1f} Hz" in text
    assert "audible" in text
    assert "IEC 61400-11" in text


def test_metadata_appears_and_one_page(tmp_path: Path) -> None:
    """A populated ReportMetadata renders one page and prints its fields."""
    md = ReportMetadata(
        specimen="Horizontal-axis wind turbine, gearbox tone",
        client="Acoustic Test Client Ltd.",
        test_room="Ground board, downwind reference position",
        instrumentation="Class 1 analyser, 2 Hz FFT lines",
        measurement_standard="IEC 61400-11",
        test_date="2026-07-21",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-61400",
    )
    out = tmp_path / "meta.pdf"
    _result().report(str(out), metadata=md, verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "Horizontal-axis wind turbine, gearbox tone" in text
    assert "Ground board, downwind reference position" in text
    assert "PHN-2026-61400" in text
    assert "IEC 61400-11" in text


def test_requirement_pass_and_fail_both_render(tmp_path: Path) -> None:
    """A PASS and a FAIL tonal-audibility limit both render one page."""
    result = _result()
    delta = result.tonal_audibility
    passing = tmp_path / "pass.pdf"
    failing = tmp_path / "fail.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=delta + 2.0))
    result.report(str(failing), metadata=ReportMetadata(requirement=delta - 2.0))
    assert_one_page(str(passing))
    assert_one_page(str(failing))
    assert "PASS" in _extract_text(str(passing)).replace("\n", " ")
    assert "FAIL" in _extract_text(str(failing)).replace("\n", " ")


def test_report_escapes_xml_specials_in_metadata(tmp_path: Path) -> None:
    """Metadata with XML specials (& < >) renders without crashing reportlab."""
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="turbine <A> & gearbox",
        test_room="pos <1> & <2>",
        laboratory="Lab & Sons",
        operator="A <B>",
        report_id="R&D-61400",
        measurement_standard="IEC 61400-11 & A1",
    )
    out = tmp_path / "xml.pdf"
    _result().report(str(out), metadata=md)
    assert_one_page(str(out))


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """``language="es"`` renders a one-page Spanish fiche with comma decimals."""
    import re

    out = tmp_path / "wt_es.pdf"
    _result().report(
        str(out),
        metadata=ReportMetadata(specimen="aerogenerador de eje horizontal"),
        language="es",
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Evaluación de la audibilidad tonal de aerogenerador" in text
    assert "Decisión" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator


def test_decision_matches_displayed_audibility_at_boundary(tmp_path: Path) -> None:
    """The audibility decision reads the displayed rounded ΔL_a, not the raw flag.

    A raw tonal audibility of 0.03 dB is audible (> 0), but it rounds to the
    displayed 0.0 dB; the fiche must state the tone is *not* audible so the
    decision text cannot contradict the number printed in the box.
    """
    import dataclasses

    result = dataclasses.replace(_result(), tonal_audibility=0.03, is_audible=True)
    assert result.is_audible is True  # raw flag is audible
    out = tmp_path / "boundary.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "The tone is not audible" in text
    assert "0.0 dB" in text


def test_no_identified_tone_reports_exclusion(tmp_path: Path) -> None:
    """A spectrum with no classified tone renders and states the exclusion.

    A broad 33 dB bump on a 30 dB floor produces no tone line (subclause 9.5.4),
    so the fiche states that no tone was identified and the spectrum is excluded
    from the bin averaging, rather than fabricating an audibility decision.
    """
    df = 2.0
    freqs = np.arange(380.0, 620.0 + df, df)
    levels = np.full(freqs.size, 30.0)
    levels[np.abs(freqs - 500.0) <= 20.0] = 33.0
    result = wind_turbine_tonality(levels, freqs, tone_frequency=500.0)
    out = tmp_path / "notone.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert result.has_identified_tone is False
    assert result.is_audible is False
    assert "No tone was identified" in text
