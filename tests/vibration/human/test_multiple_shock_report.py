#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the whole-body multiple-shock ``.report()`` fiche (ISO 2631-5).

The rendered values are checked against the standard's own Annex C worked
example: five 40 m/s2 spinal-response shocks in the measured day for an 82 kg
male exposed from age 20 for 20 years at 120 days/year give the daily
acceleration dose Dzd = 55.97 m/s2 (Formula 3), the daily compressive stress
Sd = 1.62 MPa (Formula C.1), the cumulative stress variable R = 1.22
(Formula C.3) and the probability of lumbar injury Pi = 0.37 (Formula C.5). The
Table C.2 male stress variables (R = 0.72 / 1.42 / 2.17 at 10 / 50 / 90 % risk)
place R = 1.22 in the moderate band, the standard's own conclusion. Values are
read back from the PDF via pypdf text extraction; structural facts (one page,
rejected engines/languages) complete the rendering contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.vibration.human.multiple_shock import (
    MZ_FEMALE,
    MZ_MALE,
    RISK_THRESHOLDS_FEMALE,
    RISK_THRESHOLDS_MALE,
    MultipleShockResult,
    compression_dose,
    dose_from_peaks,
    injury_probability,
    injury_risk,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _result_from_peaks(
    peaks: ArrayLike,
    *,
    start_age: float = 20.0,
    years: int = 20,
    days_per_year: float = 120.0,
    sex: Literal["male", "female"] = "male",
) -> MultipleShockResult:
    """Build a result directly from the Annex C worked-example response peaks.

    The standard states the worked example in terms of the spinal response peaks
    (five 40 m/s2 shocks), so the result is built from them via the module's own
    dose/risk formulae rather than by inverting the seat-to-spine transfer
    function of an artificial time history.
    """
    mz = MZ_MALE if sex == "male" else MZ_FEMALE
    thresholds = RISK_THRESHOLDS_MALE if sex == "male" else RISK_THRESHOLDS_FEMALE
    peaks = np.asarray(peaks, dtype=float)
    dz = dose_from_peaks(peaks)
    sd = compression_dose(dz, mz=mz)
    r = injury_risk(
        sd, start_age=start_age, years=years, days_per_year=days_per_year, sex=sex
    )
    return MultipleShockResult(
        sex=sex,
        acceleration_dose=dz,
        daily_dose=dz,
        compression_dose=sd,
        risk=r,
        probability=float(injury_probability(r, sex=sex)),
        start_age=start_age,
        years=years,
        days_per_year=days_per_year,
        peaks=peaks,
        risk_thresholds=thresholds,
    )


def _annex_c_male() -> MultipleShockResult:
    """The ISO 2631-5:2018 Annex C worked example (82 kg male)."""
    return _result_from_peaks([40.0] * 5)


# --- published oracle (ISO 2631-5 Annex C worked example) ---------------------


def test_report_renders_annex_c_numbers(tmp_path: Path) -> None:
    """The fiche prints the Annex C dose, stress, R and injury-probability values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_c_male()
    out = tmp_path / "annex_c.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    # Acceleration dose Dz = daily dose Dzd = 55.97 m/s2 (Formula 3).
    assert "55.97" in text
    # Daily compressive stress Sd = 1.62 MPa (Formula C.1).
    assert "1.62" in text
    # Cumulative stress variable R = 1.22 (Formula C.3).
    assert "1.22" in text
    # Probability of lumbar injury Pi = 37 % (Formula C.5).
    assert "37 %" in text
    # Units and basis.
    assert "MPa" in text
    assert "ISO 2631-5:2018" in text
    # Table C.2 male thresholds appear in the classification table.
    assert "0.72" in text
    assert "1.42" in text
    assert "2.17" in text


def test_report_classifies_worked_example_as_moderate(tmp_path: Path) -> None:
    """R = 1.22 is classified in the moderate band (the standard's conclusion)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_c_male()
    out = tmp_path / "moderate.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert "moderate probability of an adverse health effect" in text
    # Exactly one band is marked as this assessment's classification.
    assert text.count("This assessment") == 1


# --- risk-band boundaries on the displayed value ------------------------------


@pytest.mark.parametrize(
    ("peak", "band"),
    [
        (20.0, "low probability of an adverse health effect"),
        (40.0, "moderate probability of an adverse health effect"),
        (48.0, "high probability of an adverse health effect"),
        (75.0, "very high probability of an adverse health effect"),
    ],
)
def test_risk_band_tracks_table_c2(tmp_path: Path, peak: float, band: str) -> None:
    """The boxed classification follows the Table C.2 stress-variable bands."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _result_from_peaks([peak] * 5)
    out = tmp_path / f"band_{peak}.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert band in text


# --- metadata -----------------------------------------------------------------


def test_metadata_header_renders(tmp_path: Path) -> None:
    """Supplied metadata renders the client, subject, workplace and traceability."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_c_male()
    metadata = ReportMetadata(
        client="Example transport operator",
        specimen="82 kg male operator (seated)",
        test_room="Off-road vehicle, driver's seat",
        instrumentation="Seat-pad accelerometer, s/n 0117",
        report_id="MS-2631-5",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example transport operator" in text
    assert "82 kg male operator (seated)" in text
    assert "Off-road vehicle, driver's seat" in text
    assert "Seat-pad accelerometer, s/n 0117" in text
    assert "MS-2631-5" in text
    # The exposure scenario is always shown (from the result, not the metadata).
    assert "120" in text  # days per year N


# --- Spanish fiche ------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """language="es" renders the multiple-shock vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_c_male()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Riesgo para la salud por choques múltiples de cuerpo completo" in text
    assert "probabilidad moderada de un efecto adverso para la salud" in text
    assert "1,22" in text  # comma decimal separator
    assert "55,97" in text
    # The clause references translate the word and keep the number verbatim.
    assert "Fórmula (C.3)" in text
    assert "Formula (C.3)" not in text


# --- rendering contract -------------------------------------------------------


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _annex_c_male()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _annex_c_male()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match=r"Unknown language"):
        res.report(out, language="xx")
