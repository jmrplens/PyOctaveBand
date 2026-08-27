#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ICAO Annex 16 EPNL report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written for a computed flyover, unknown engines are
rejected, a passing and a failing certification limit both render, and a
no-metadata prediction fiche renders. The EPNL algorithm itself is validated
against the ICAO Doc 9501 ETM worked examples elsewhere
(tests/aircraft/test_certification.py); here a small self-contained synthetic
flyover keeps the fiche test independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

pytest.importorskip("reportlab")

from report_assertions import assert_one_page

from phonometry import ReportMetadata, aircraft

if TYPE_CHECKING:
    from pathlib import Path


def _flyover() -> aircraft.EPNLResult:
    """A deterministic synthetic flyover EPNLResult (Gaussian temporal peak)."""
    k, dt = 24, 0.5
    idx = np.arange(k)
    from phonometry.aircraft import NOY_BANDS

    shape = 15.0 * np.exp(-((np.log10(NOY_BANDS) - np.log10(400.0)) ** 2) / 0.5)
    gain = 24.0 * np.exp(-((idx - 12.0) ** 2) / (2 * 3.5**2)) - 3.0
    spectra = (46.0 + shape)[None, :] + gain[:, None]
    spectra[:, 17] += 10.0 * np.exp(-((idx - 12.0) ** 2) / (2 * 4.0**2))
    return aircraft.effective_perceived_noise_level(spectra, dt)


def test_report_writes_one_page_pdf(tmp_path: Path) -> None:
    """A computed flyover renders a one-page certification fiche."""
    result = _flyover()
    assert np.isfinite(result.epnl)
    out = tmp_path / "epnl.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = _flyover()  # constructed outside pytest.raises (Sonar S5778)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine"):
        result.report(out, engine="weasyprint")


def test_passing_requirement_renders(tmp_path: Path) -> None:
    """A certification limit at or above the EPNL renders a passing fiche."""
    result = _flyover()
    limit = float(result.epnl) + 3.0
    out = tmp_path / "pass.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=limit))
    assert_one_page(str(out))


def test_failing_requirement_renders(tmp_path: Path) -> None:
    """A certification limit below the EPNL renders a failing fiche."""
    result = _flyover()
    limit = float(result.epnl) - 3.0
    out = tmp_path / "fail.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=limit))
    assert_one_page(str(out))


def test_full_metadata_renders_one_page(tmp_path: Path) -> None:
    """A populated ReportMetadata renders a one-page fiche."""
    md = ReportMetadata(
        specimen="Example twin-turbofan transport",
        manufacturer="Example Aircraft Company",
        client="Example Aircraft Company",
        test_room="Flyover reference point",
        measurement_standard="ICAO Annex 16 Vol I Amendment 14 Chapter 4",
        test_date="2026-07-20",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-EPNL",
        requirement=101.0,
    )
    out = tmp_path / "meta.pdf"
    _flyover().report(str(out), metadata=md)
    assert_one_page(str(out))


def test_no_metadata_prediction_fiche(tmp_path: Path) -> None:
    """With ``metadata=None`` a prediction fiche renders (no verdict row)."""
    out = tmp_path / "prediction.pdf"
    _flyover().report(str(out))
    assert_one_page(str(out))


def _extract_text(path: str) -> str:
    """The concatenated text of every page (for language assertions)."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_verdict_follows_one_decimal_epnl_at_the_limit(tmp_path: Path) -> None:
    """The verdict compares the EPNL determined to one decimal place.

    Annex 16 determines the EPNL "to one decimal place": an unrounded
    101.04 EPNdB against a 101 EPNdB limit is 101.0 EPNdB and PASSes with a
    zero margin (previously the unrounded comparison printed "101.0",
    "margin -0.0" and FAIL, contradicting its own displayed numbers).
    """
    import dataclasses

    result = dataclasses.replace(_flyover(), epnl=101.04)
    out = tmp_path / "boundary.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=101.0))
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "EPNL = 101.0 EPNdB" in text
    assert "margin +0.0" in text
    assert "PASS" in text
    assert "FAIL" not in text
    # Just past the tenth boundary the same limit fails.
    failing = dataclasses.replace(_flyover(), epnl=101.06)
    out2 = tmp_path / "boundary_fail.pdf"
    failing.report(str(out2), metadata=ReportMetadata(requirement=101.0))
    text2 = _extract_text(str(out2)).replace("\n", " ")
    assert "EPNL = 101.1 EPNdB" in text2
    assert "FAIL" in text2


def test_reference_conditions_cite_part_ii(tmp_path: Path) -> None:
    """The reference-conditions strip states the Part II 3.6.1.5 conditions."""
    out = tmp_path / "refcond.pdf"
    _flyover().report(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "25 °C (ISA + 10 °C)" in text
    assert "1013.25 hPa" in text
    assert "Part II, 3.6.1.5" in text


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """``language="es"`` renders a one-page Spanish fiche with comma decimals."""
    import re

    result = _flyover()
    out = tmp_path / "epnl_es.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=101.0), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Certificación de ruido de aeronaves" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = _flyover()
    with pytest.raises(ValueError, match=r"Unknown language"):
        result.report(str(tmp_path / "bad.pdf"), language="xx")
