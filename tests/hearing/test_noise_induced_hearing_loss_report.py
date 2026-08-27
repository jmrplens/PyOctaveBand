#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 1999:2013 hearing-loss prediction ``.report()`` fiches.

The two occupational-hearing-loss result types render one-page statistical
prediction fiches: NIPTS (noise-induced permanent threshold shift, clause 6.3)
and HTLAN (hearing threshold level associated with age and noise, clause 6.1).
The rendered values are checked against the ISO 1999:2013 Annex D worked
example (Table D.2, L_EX,8h = 90 dB, 20 years): the median 4 kHz shift is
N50 = 12.9 dB and the fractile value at Q = 0.90 is 17.8 dB. Values are read
back from the PDF via pypdf text extraction; structural facts (one page,
rejected engines/languages) complete the rendering contract, and both the
English and Spanish fiches are exercised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.hearing import NoiseInducedHearingLossWarning, htlan, nipts

if TYPE_CHECKING:
    from pathlib import Path


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


# --- NIPTS fiche --------------------------------------------------------------


def test_nipts_report_renders_annex_d_values(tmp_path: Path) -> None:
    """The NIPTS fiche prints the Annex D N50 and fractile shift and one page."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = nipts(90.0, 20.0, 0.9)
    out = tmp_path / "nipts.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "12.9" in text  # median N50 at 4 kHz (Table D.2)
    assert "17.8" in text  # NIPTS at Q = 0.90, 4 kHz (worst tenth)
    assert "4000" in text  # an audiometric frequency
    assert "13.9" in text  # representative shift averaged over 2/3/4 kHz
    assert "clause 6.3" in text
    assert "statistical prediction" in text
    assert "not a clinical diagnosis" in text


def test_nipts_verbose_adds_spread_columns(tmp_path: Path) -> None:
    """verbose=True adds the du/dl spread columns to the NIPTS table."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = nipts(95.0, 20.0, 0.9)
    out = tmp_path / "nipts_v.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    flat = "".join(_extract_text(str(out)).split())
    assert "du[dB]" in flat
    assert "dl[dB]" in flat


def test_nipts_verdict_against_requirement(tmp_path: Path) -> None:
    """A metadata requirement adds a PASS/FAIL verdict on the representative NIPTS."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = nipts(90.0, 20.0, 0.9)  # representative 2/3/4 kHz shift is 13.9 dB
    out_pass = tmp_path / "pass.pdf"
    res.report(str(out_pass), metadata=ReportMetadata(requirement=15.0))
    text = _extract_text(str(out_pass))
    assert "PASS" in text
    assert "FAIL" not in text
    out_fail = tmp_path / "fail.pdf"
    res.report(str(out_fail), metadata=ReportMetadata(requirement=10.0))
    assert "FAIL" in _extract_text(str(out_fail))


def test_nipts_states_iso_q_and_scope_caveat(tmp_path: Path) -> None:
    """The fiche prints ISO 1999's own Q and the Scope NOTE 1 caveat."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = nipts(90.0, 20.0, 0.9)  # the most-susceptible tenth: ISO's Q = 10 %
    out = tmp_path / "q.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert "Q = 10 %" in text
    assert "fraction with worse hearing" in text
    assert "6.3.2, Formulae (4) and (5)" in text
    assert "does not specify frequencies" in text  # Scope NOTE 1
    assert "tested specimen" not in text  # a population, not a specimen
    assert "do not describe any individual person" in text


def test_nipts_outside_domain_prints_extrapolation_caveat(tmp_path: Path) -> None:
    """Conditions beyond the validated domain carry the caveat on the fiche."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    with pytest.warns(NoiseInducedHearingLossWarning):
        res = nipts(130.0, 60.0, 0.99)
    out = tmp_path / "extrapolated.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "outside the validated domain" in text
    assert "extrapolation" in text
    # A prediction inside the domain does not carry it.
    inside = tmp_path / "inside.pdf"
    nipts(90.0, 20.0, 0.9).report(str(inside))
    assert "outside the validated domain" not in _extract_text(str(inside))


def test_nipts_subset_boxes_peak_shift(tmp_path: Path) -> None:
    """Without the full 2/3/4 kHz set the fiche boxes the peak shift instead."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = nipts(95.0, 20.0, 0.9, frequencies=[500.0, 6000.0])
    out = tmp_path / "sub.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    assert "peak NIPTS" in _extract_text(str(out))


# --- HTLAN fiche --------------------------------------------------------------


def test_htlan_report_renders_components(tmp_path: Path) -> None:
    """The HTLAN fiche prints the combined threshold and one page."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = htlan(60, "male", 95.0, 30.0, 0.5)
    out = tmp_path / "htlan.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    # Regression guards, not normative oracles: the ISO 1999 Annex C example
    # pins the NIPTS chain (see tests/hearing/test_noise_induced_hearing_loss.py
    # and the conformance report), while these two numbers are the library's
    # own HTLAN output with its ISO 7029:2017 age component, kept here so a
    # change in the composition cannot pass unnoticed.
    assert "40.8" in text  # combined H' at 4 kHz
    assert "4000" in text  # an audiometric frequency
    assert "33.0" in text  # representative threshold averaged over 2/3/4 kHz
    assert "clause 6.1" in text
    assert "age and noise" in text
    # The age database the H column comes from is disclosed on the fiche.
    assert "ISO 7029:2017" in text


def test_htlan_verbose_adds_compression_term(tmp_path: Path) -> None:
    """verbose=True adds the H*N/120 compression column to the HTLAN table."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = htlan(60, "male", 95.0, 30.0, 0.5)
    out = tmp_path / "htlan_v.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    flat = "".join(_extract_text(str(out)).split())
    assert "N/120" in flat


def test_htlan_metadata_header_renders(tmp_path: Path) -> None:
    """Supplied metadata renders the header grid identity."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = htlan(60, "male", 95.0, 30.0, 0.5)
    metadata = ReportMetadata(
        client="Example works",
        specimen="Machine operator",
        test_room="Assembly hall",
        report_id="H-1",
    )
    out = tmp_path / "htlan_meta.pdf"
    res.report(str(out), metadata=metadata)
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Machine operator" in text


# --- Spanish fiche ------------------------------------------------------------


def test_nipts_spanish_report(tmp_path: Path) -> None:
    """language="es" renders the Spanish NIPTS vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = nipts(90.0, 20.0, 0.9)
    out = tmp_path / "nipts_es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Predicción de la pérdida auditiva inducida por ruido" in text
    assert "13,9" in text  # comma decimal separator
    # ISO 1999's own Q (the fraction with worse hearing) for fractile 0.9.
    assert "Fractil poblacional Q = 10 % (fracción con peor audición)" in text


def test_htlan_spanish_report(tmp_path: Path) -> None:
    """language="es" renders the Spanish HTLAN vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    pytest.importorskip("svglib")
    res = htlan(60, "male", 95.0, 30.0, 0.5)
    out = tmp_path / "htlan_es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "edad y ruido" in text
    assert "hombre" in text
    assert "33,0" in text  # regression guard on the library's own output
    assert "Fractil poblacional Q = 50 % (mediana)" in text
    assert "mayor (peor) que el indicado" in text
    assert "ISO 7029:2017" in text
    assert "ninguna persona concreta" in text
    assert "probeta ensayada" not in text


# --- rendering contract -------------------------------------------------------


def test_nipts_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = nipts(90.0, 20.0, 0.9)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine 'weasyprint'"):
        res.report(out, engine="weasyprint")


def test_htlan_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ValueError."""
    res = htlan(60, "male", 95.0, 30.0, 0.5)
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match=r"Unknown language 'xx'"):
        res.report(out, language="xx")
