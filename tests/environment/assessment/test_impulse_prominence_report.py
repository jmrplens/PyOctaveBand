#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the impulsive-sound prominence report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written for an impulse set, unknown engines and
languages are rejected, XML specials in metadata do not break reportlab, the
verdict renders both ways, and the boxed governing prominence ``P``, the derived
``LAeq`` adjustment ``KI`` and the metadata appear in the extracted text. The
prominence and adjustment maths itself is validated against the NT ACOU 112:2002
formulae elsewhere (tests/environment/assessment/test_impulse_prominence.py); this fiche
test anchors its numbers to the documented three-impulse pile-driving set, whose
governing prominence and adjustment are derived from Formula 1 and Formula 2.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("reportlab")

from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.environment.assessment import impulsive_sound as nt

if TYPE_CHECKING:
    from pathlib import Path

# The documented three-impulse pile-driving set: (onset rate dB/s, level
# difference dB). All three qualify (onset rate > 10 dB/s); the first governs.
_ONSET_RATES = [1200.0, 300.0, 60.0]
_LEVEL_DIFFERENCES = [32.0, 18.0, 11.0]

# Governing prominence P = 3*lg(1200) + 2*lg(32) and its adjustment KI (Formula
# 2), derived by hand from the formulae so the fiche numbers are documented.
_P_GOVERNING = 3.0 * math.log10(1200.0) + 2.0 * math.log10(32.0)  # 12.2478...
_KI_GOVERNING = 1.8 * (_P_GOVERNING - 5.0)  # 13.046...


def _result() -> nt.ImpulseProminenceResult:
    """The documented three-impulse pile-driving prominence result."""
    return nt.impulse_prominence(_ONSET_RATES, _LEVEL_DIFFERENCES)


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_report_writes_one_page_pdf(tmp_path: Path) -> None:
    """An impulse set renders a one-page PDF fiche."""
    result = _result()
    out = tmp_path / "impulse.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="xx")


def test_report_states_prominence_and_adjustment(tmp_path: Path) -> None:
    """The fiche states the governing P, the derived KI and the formula basis.

    The governing prominence is the first (highest-P) impulse; its LAeq
    adjustment follows Formula 2. The extracted text must carry the boxed
    governing prominence, the adjustment value and the standard basis.
    """
    result = _result()
    out = tmp_path / "impulse.pdf"
    result.report(str(out))
    text = _extract_text(str(out)).replace("\n", " ")

    assert result.prominence == pytest.approx(_P_GOVERNING, abs=1e-6)
    assert result.adjustment == pytest.approx(_KI_GOVERNING, abs=1e-6)
    assert "Governing prominence" in text
    assert f"{_P_GOVERNING:.2f}" in text  # boxed governing P (12.25)
    assert f"{_KI_GOVERNING:.1f} dB" in text  # derived KI (13.0 dB)
    assert "NT ACOU 112" in text
    assert "Formula 2" in text


def test_metadata_appears_and_one_page(tmp_path: Path) -> None:
    """A populated ReportMetadata renders one page and prints its fields."""
    md = ReportMetadata(
        specimen="Pile-driving site, intermittent hammering",
        client="Acoustic Test Client Ltd.",
        test_room="Free field, 25 m from source",
        instrumentation="Class 1 SLM (IEC 61672-1)",
        measurement_standard="ISO 1996-2",
        test_date="2026-07-21",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-NTACOU112",
    )
    out = tmp_path / "meta.pdf"
    _result().report(str(out), metadata=md, verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "Pile-driving site, intermittent hammering" in text
    assert "Free field, 25 m from source" in text
    assert "PHN-2026-NTACOU112" in text
    assert "ISO 1996-2" in text
    assert "Assessment period" in text
    assert "30 min" in text


def test_requirement_pass_and_fail_both_render(tmp_path: Path) -> None:
    """A PASS and a FAIL prominence limit both render one page."""
    result = _result()
    p = result.prominence
    passing = tmp_path / "pass.pdf"
    failing = tmp_path / "fail.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=p + 2.0))
    result.report(str(failing), metadata=ReportMetadata(requirement=p - 2.0))
    assert_one_page(str(passing))
    assert_one_page(str(failing))
    assert "PASS" in _extract_text(str(passing)).replace("\n", " ")
    assert "FAIL" in _extract_text(str(failing)).replace("\n", " ")


def test_report_escapes_xml_specials_in_metadata(tmp_path: Path) -> None:
    """Metadata with XML specials (& < >) is escaped, rendered and not dropped.

    The specials must survive reportlab's XML parser (which would otherwise
    reject a bare ``&`` or ``<``) *and* appear intact in the rendered PDF text,
    so the fiche cannot silently omit or mangle the client's metadata.
    """
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="hammer <A> & pile",
        test_room="pos <1> & <2>",
        laboratory="Lab & Sons",
        operator="A <B>",
        report_id="R&D-112",
        measurement_standard="ISO 1996-2 & NT ACOU 112",
    )
    out = tmp_path / "xml.pdf"
    _result().report(str(out), metadata=md)
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    # The escaped values render back to their literal glyphs in the PDF text.
    assert "hammer <A> & pile" in text  # source/situation (specimen)
    assert "Ac & Co <Ltd>" in text  # client
    assert "pos <1> & <2>" in text  # measurement position
    assert "R&D-112" in text  # footer report number


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """``language="es"`` renders a one-page Spanish fiche with comma decimals."""
    import re

    out = tmp_path / "impulse_es.pdf"
    _result().report(
        str(out),
        metadata=ReportMetadata(specimen="hincado de pilotes"),
        language="es",
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Evaluación de la prominencia de sonidos impulsivos" in text
    assert "ajuste de L" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator


def test_verdict_compares_unrounded_prominence(tmp_path: Path) -> None:
    """A prominence just above the requirement FAILs, not rounded to a PASS.

    The governing prominence 12.2478 rounds to 12.25 for display; a requirement
    of 12.247 is just below it, so the assessment must FAIL even though the
    displayed P would round to the same two decimals as a passing value would.
    """
    result = _result()
    out = tmp_path / "boundary.pdf"
    requirement = result.prominence - 1e-3  # just below the unrounded P
    result.report(str(out), metadata=ReportMetadata(requirement=requirement))
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert result.prominence > requirement
    assert "FAIL" in text
    assert "PASS" not in text


def test_verdict_passes_at_the_requirement(tmp_path: Path) -> None:
    """A governing prominence at the requirement passes (``<=``)."""
    result = _result()
    out = tmp_path / "atlimit.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=result.prominence))
    assert_one_page(str(out))
    assert "PASS" in _extract_text(str(out)).replace("\n", " ")


def test_oversized_impulse_set_stays_one_page(tmp_path: Path) -> None:
    """A large valid impulse set caps the table and stays exactly one page.

    Forty qualifying impulses exceed the table row cap; the fiche must keep the
    highest-prominence rows (including the governing impulse), add an explicit
    ``... plus N more`` note and still render as a single A4 page.
    """
    import warnings

    import numpy as np

    rng = np.random.default_rng(0)
    onset_rates = rng.uniform(20.0, 2000.0, 40)
    level_differences = rng.uniform(5.0, 40.0, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = nt.impulse_prominence(onset_rates, level_differences)
    out = tmp_path / "big.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "more impulses of lower prominence" in text
    assert f"{result.prominence:.2f}" in text  # boxed governing P still present

    # The whole point of the cap is that the governing impulse is never dropped:
    # its row must survive the truncation to the highest-prominence impulses.
    from phonometry._report.iso1996_impulse import (
        _MAX_TABLE_ROWS,
        _governing_index,
        _select_rows,
    )

    per = np.asarray(result.per_impulse, dtype=np.float64)
    qualifies = np.asarray(result.qualifies, dtype=bool)
    governing = _governing_index(per, qualifies)
    shown, dropped = _select_rows(per, governing)
    assert len(shown) == _MAX_TABLE_ROWS  # the table is actually capped
    assert dropped == per.size - _MAX_TABLE_ROWS
    assert governing in shown  # the governing impulse's row is retained
    # Its 1-based input index and its prominence appear in the rendered table.
    assert f"{per[governing]:.2f}" in text


def test_row_cap_force_includes_a_low_prominence_governing_impulse() -> None:
    """The cap keeps the governing impulse even when it is not a top-P row.

    A non-qualifying event can carry a higher raw prominence than the governing
    (highest *qualifying*) impulse; a plain top-N-by-prominence cut would drop
    the governing row. ``_select_rows`` must force it in and drop a top row.
    """
    import numpy as np

    from phonometry._report.iso1996_impulse import _MAX_TABLE_ROWS, _select_rows

    # The governing impulse is the lowest-prominence entry, so a naive top-N cut
    # would exclude it; every other entry outranks it.
    n = _MAX_TABLE_ROWS + 3
    per = np.arange(n, dtype=np.float64) + 10.0  # strictly increasing
    governing = 0  # the smallest prominence
    shown, dropped = _select_rows(per, governing)
    assert len(shown) == _MAX_TABLE_ROWS
    assert dropped == n - _MAX_TABLE_ROWS
    assert governing in shown  # forced in despite its low prominence


def test_non_prominent_impulse_reports_zero_adjustment(tmp_path: Path) -> None:
    """A qualifying but weak impulse (P <= 5) renders with a zero adjustment.

    A qualifying onset (rate above 10 dB/s) with a low prominence keeps KI at
    zero; the prominence note and boxed KI stay consistent.
    """
    # onset rate 15 dB/s, level difference 5 dB: P = 3*lg(15) + 2*lg(5) = 4.925.
    result = nt.impulse_prominence([15.0], [5.0])
    out = tmp_path / "weak.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert result.prominence <= 5.0
    assert result.adjustment == 0.0
    assert "No prominent impulse is present" in text
    assert "K = 0" in text or "0 dB" in text


def test_non_qualifying_set_justifies_on_the_onset_gate(tmp_path: Path) -> None:
    """With no qualifying onset the note cites the onset gate, not P <= 5.

    Both gates can withhold the adjustment. Here every onset rate is at or
    below 10 dB/s (clauses 4.5/8), so no adjustment can arise whatever the
    arithmetic prominence is; the governing P shown is informational and
    exceeds 5, so a "P <= 5" justification would contradict the fiche's own
    boxed number.
    """
    with pytest.warns(nt.ImpulseProminenceWarning):
        result = nt.impulse_prominence([10.0, 8.0], [40.0, 30.0])
    assert result.adjustment == 0.0
    assert result.prominence > 5.0  # informational only
    out = tmp_path / "no_qualifying.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "No level rise qualifies as an impulse" in text
    assert "informational only" in text
    assert "No prominent impulse is present" not in text


def test_assessment_period_reflects_the_analysed_interval(tmp_path: Path) -> None:
    """The header prints the analysed interval, defaulting to the 30 min default."""
    default_out = tmp_path / "default.pdf"
    _result().report(str(default_out))
    assert "30 min" in _extract_text(str(default_out))

    other = nt.impulse_prominence(
        _ONSET_RATES, _LEVEL_DIFFERENCES, assessment_period_min=15.0
    )
    other_out = tmp_path / "quarter_hour.pdf"
    other.report(str(other_out))
    text = _extract_text(str(other_out))
    assert "15 min" in text
    assert "30 min" not in text
