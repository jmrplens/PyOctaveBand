#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 3382-3:2012 open-plan office acoustics report (``.report()``).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written for an open-plan result, the four
single-number quantities carry the closed-form values of a documented synthetic
measurement line, unknown engines/languages are rejected, XML specials in
metadata do not break reportlab, the optional target-D2,S verdict renders both
ways, a degenerate result (no in-range positions) still renders, and the
Spanish fiche is translated with comma decimals. Pixel or layout content is
never inspected.

Oracle. The A-weighted speech level is collinear in the logarithmic distance
axis, Lp,A,S(r) = 62.0 - 7.0*log2(r) dB, so the ISO 3382-3:2012 Clause 6.2
least-squares fit recovers D2,S = 7.0 dB per distance doubling and
Lp,A,S,4m = 62 - 7*log2(4) = 48.0 dB exactly. The STI is linear in distance,
STI(r) = 0.65 - 0.03*r, so the Clause 6.3 STI-vs-distance regression crosses
0.50 at rD = (0.50 - 0.65)/(-0.03) = 5.0 m and 0.20 at
rP = (0.20 - 0.65)/(-0.03) = 15.0 m. Seven positions span the 2 m to 16 m
range (in the 6 to 10 preferred by Clause 5.2.2).
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("svglib")
pytest.importorskip("pypdf")

from typing import TYPE_CHECKING

import numpy as np

from phonometry import (
    ReportMetadata,
    room,
)

if TYPE_CHECKING:
    from phonometry.room import OpenPlanResult

_PDF_MAGIC = b"%PDF"

#: Documented measurement line and its exact closed-form single-number results.
_POSITIONS = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0])
_SPL = 62.0 - 7.0 * np.log2(_POSITIONS)
_STI = 0.65 - 0.03 * _POSITIONS
_D2S = 7.0
_LP4M = 48.0
_RD = 5.0
_RP = 15.0


def _result() -> OpenPlanResult:
    return room.open_plan_metrics(_POSITIONS, _SPL, _STI)


def _assert_one_page(path: str) -> None:
    """The fiche is a single-page PDF beginning with ``%PDF``."""
    import os

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    from pypdf import PdfReader

    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    """The concatenated, single-line text of every page."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages).replace(
        "\n", " "
    )


def _full_metadata(**overrides) -> ReportMetadata:
    base = {
        "specimen": "Furnished, unoccupied, background noise present",
        "client": "Acoustic Test Client Ltd.",
        "test_room": "Open-plan office B",
        "area": 420.0,
        "source_positions": 2,
        "receiver_positions": 7,
        "instrumentation": "Omnidirectional source + class 1 SLM",
        "measurement_standard": "ISO 3382-3",
        "temperature": 22.0,
        "relative_humidity": 45.0,
        "pressure": 101.1,
        "test_date": "2026-07-20",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-3382-3",
    }
    base.update(overrides)
    return ReportMetadata(**base)


# --------------------------------------------------------------------------
# Oracle: the synthetic line reproduces its closed-form single-number values.
# --------------------------------------------------------------------------
def test_synthetic_line_matches_closed_form() -> None:
    """D2,S, Lp,A,S,4m, rD and rP match the closed-form oracle exactly."""
    res = _result()
    assert res.d2s == pytest.approx(_D2S)
    assert res.lp_as_4m == pytest.approx(_LP4M)
    assert res.rd == pytest.approx(_RD)
    assert res.rp == pytest.approx(_RP)


# --------------------------------------------------------------------------
# Structural rendering
# --------------------------------------------------------------------------
def test_report_writes_pdf(tmp_path) -> None:
    """An open-plan result renders a PDF fiche and returns its path."""
    out = tmp_path / "openplan.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_report_with_full_metadata_one_page(tmp_path) -> None:
    """A full ReportMetadata renders a one-page accredited open-plan fiche."""
    out = tmp_path / "openplan_meta.pdf"
    _result().report(str(out), metadata=_full_metadata())
    _assert_one_page(str(out))


def test_report_bare_without_metadata(tmp_path) -> None:
    """metadata=None renders a bare characterisation fiche (no header)."""
    out = tmp_path / "bare.pdf"
    _result().report(str(out), metadata=None)
    _assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    res = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    res = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")


# --------------------------------------------------------------------------
# Displayed values (the oracle appears in the rendered text)
# --------------------------------------------------------------------------
def test_single_number_quantities_render(tmp_path) -> None:
    """The four single-number quantities appear in the rendered text."""
    out = tmp_path / "values.pdf"
    _result().report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "7.0" in text  # D2,S
    assert "48.0" in text  # Lp,A,S,4m
    assert "5.0" in text  # rD
    assert "15.0" in text  # rP
    assert "per distance doubling" in text
    # The metrics-table row labels and units are present.
    assert "Spatial decay rate" in text
    assert "Distraction distance" in text


def test_verdict_renders_both_ways(tmp_path) -> None:
    """A target D2,S renders a PASS when met and a FAIL when not.

    The requirement is the minimum acceptable D2,S: the oracle 7.0 dB passes a
    7.0 dB target (at or above) and fails an 8.0 dB target.
    """
    passing = tmp_path / "pass.pdf"
    failing = tmp_path / "fail.pdf"
    _result().report(str(passing), metadata=_full_metadata(requirement=7.0))
    _result().report(str(failing), metadata=_full_metadata(requirement=8.0))
    _assert_one_page(str(passing))
    _assert_one_page(str(failing))
    assert "PASS" in _extract_text(str(passing))
    assert "FAIL" in _extract_text(str(failing))


def test_report_without_requirement_has_no_verdict(tmp_path) -> None:
    """With no target the characterisation fiche omits the verdict row."""
    out = tmp_path / "noverdict.pdf"
    _result().report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "PASS" not in text
    assert "FAIL" not in text


# --------------------------------------------------------------------------
# Robustness and localisation
# --------------------------------------------------------------------------
def test_degenerate_result_renders_without_plot(tmp_path) -> None:
    """A result with no positions in 2-16 m (NaN D2,S) still renders one page.

    The spatial-decay regression is undefined, so its plot cannot be drawn; the
    fiche omits the figure and shows the metrics table with the em-dashed
    unavailable quantities instead of crashing.
    """
    near = np.array([0.5, 1.0, 1.5, 1.8])
    res = room.open_plan_metrics(near, 60.0 - 5.0 * near, 0.7 - 0.1 * near)
    assert not np.isfinite(res.d2s)
    out = tmp_path / "degenerate.pdf"
    res.report(str(out), metadata=_full_metadata())
    _assert_one_page(str(out))
    assert "—" in _extract_text(str(out))  # em-dash empty-cell symbol


def test_report_escapes_xml_specials_in_metadata(tmp_path) -> None:
    """Metadata with XML specials (& < >) renders without crashing reportlab."""
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="office <A> & annex",
        test_room="Zone & Wing",
        laboratory="Lab & Sons",
        operator="A <B>",
        report_id="R&D-3382-3",
        measurement_standard="ISO 3382-3 & Annex",
    )
    out = tmp_path / "xml.pdf"
    _result().report(str(out), metadata=md)
    _assert_one_page(str(out))


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders a one-page Spanish fiche with comma decimals."""
    import re

    out = tmp_path / "openplan_es.pdf"
    _result().report(str(out), metadata=_full_metadata(requirement=7.0), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Acústica de oficinas diáfanas" in text
    assert "CUMPLE" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator
