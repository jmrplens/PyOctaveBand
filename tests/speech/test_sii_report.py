#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ANSI S3.5-1997 speech-intelligibility-index ``.report()`` fiche.

The rendered index is checked against oracles independent of the renderer: the
standard normal-effort speech spectrum in quiet with normal hearing rates to
SII = 0.996 (ANSI S3.5-1997 clause 6), and the R CRAN "SII" package worked
Example C.2 (an independent implementation) rates to SII = 0.851. Both are pure
arithmetic (no filtering), so the boxed values are stable across platforms. The
Table 3 band-importance value, the verdict direction (a higher SII passes), the
EN/ES parity and the rendering contract complete the checks. Values are read
back from the PDF via pypdf text extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.speech import SIIResult, speech_intelligibility_index

if TYPE_CHECKING:
    from pathlib import Path


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _example_c2() -> SIIResult:
    """R CRAN "SII" Example C.2: SII = 0.851 (independent oracle)."""
    return speech_intelligibility_index(
        np.full(18, 54.0),
        np.array([40.0, 30.0, 20.0] + [0.0] * 15),
        threshold=np.zeros(18),
    )


# --- exact oracle --------------------------------------------------------------


def test_example_c2_renders_index_and_band_importance(tmp_path: Path) -> None:
    """The Example C.2 fiche prints SII = 0.851 and a Table 3 Ii value."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _example_c2()
    assert res.sii == pytest.approx(0.851375, abs=1e-5)
    out = tmp_path / "sii.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "SII = 0.851" in text
    # Band-importance function Ii at 2000 Hz (Table 3, four decimals).
    assert "0.0898" in text


def test_standard_speech_in_quiet_renders(tmp_path: Path) -> None:
    """The standard normal spectrum in quiet rates to SII = 0.996."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = speech_intelligibility_index("normal")
    assert res.sii == pytest.approx(0.995825, abs=1e-5)
    out = tmp_path / "quiet.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    assert "SII = 0.996" in _extract_text(str(out))


# --- verbose adds the disturbance column --------------------------------------


def test_verbose_adds_disturbance_column(tmp_path: Path) -> None:
    """verbose=True adds the equivalent disturbance spectrum level Di column."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _example_c2()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    flat = "".join(_extract_text(str(out)).split())
    assert "Di[dB]" in flat


# --- verdict direction (higher is better) -------------------------------------


def test_verdict_passes_at_or_above_requirement(tmp_path: Path) -> None:
    """SII = 0.851 passes a 0.75 minimum and fails a 0.90 minimum."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _example_c2()
    out_pass = tmp_path / "pass.pdf"
    res.report(str(out_pass), metadata=ReportMetadata(requirement=0.75))
    assert "PASS" in _extract_text(str(out_pass))
    out_fail = tmp_path / "fail.pdf"
    res.report(str(out_fail), metadata=ReportMetadata(requirement=0.90))
    assert "FAIL" in _extract_text(str(out_fail))


def test_verdict_prints_the_requirement_at_full_precision(tmp_path: Path) -> None:
    """The requirement is printed at the SII's own three decimals.

    A one-decimal requirement would print a 0.75 minimum as "0.8", above the
    SII that passed it.
    """
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _example_c2()
    out = tmp_path / "req075.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=0.75))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "SII = 0.851" in text
    assert "0.750" in text
    # The entity renders as a glyph in the PDF, so the negative assertion has
    # to look for what a reader would actually see.
    assert "required ≥ 0.8" not in text
    assert "required ≥ 0.750" in text
    assert "PASS" in text


def test_verdict_boundary_at_the_displayed_precision(tmp_path: Path) -> None:
    """A requirement equal to the displayed SII passes; one digit above fails."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _example_c2()  # SII = 0.851
    out_eq = tmp_path / "eq.pdf"
    res.report(str(out_eq), metadata=ReportMetadata(requirement=0.851))
    assert "PASS" in _extract_text(str(out_eq))
    out_over = tmp_path / "over.pdf"
    res.report(str(out_over), metadata=ReportMetadata(requirement=0.852))
    assert "FAIL" in _extract_text(str(out_over))


def test_verdict_decides_on_the_requirement_it_prints(tmp_path: Path) -> None:
    """A requirement with hidden digits is judged on the value shown.

    0.85104 prints as "0.851"; deciding on the unrounded number would fail an
    SII of 0.851 while printing "SII = 0.851, required >= 0.851".
    """
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _example_c2()  # SII = 0.851
    out = tmp_path / "hidden.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=0.85104))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "SII = 0.851" in text
    assert "PASS" in text


# --- metadata ------------------------------------------------------------------


def test_metadata_header_renders(tmp_path: Path) -> None:
    """Supplied metadata renders the header grid; no requirement, no verdict."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _example_c2()
    out = tmp_path / "meta.pdf"
    res.report(
        str(out),
        metadata=ReportMetadata(
            client="Example works",
            specimen="Speech in office noise",
            laboratory="Reference lab",
            report_id="SII-1",
        ),
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Speech in office noise" in text
    assert "Result vs requirement" not in text


# --- Spanish fiche -------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """language="es" renders the SII vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _example_c2()
    out = tmp_path / "sii_es.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=0.75), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Índice de inteligibilidad del habla" in text
    assert "SII = 0,851" in text
    assert "CUMPLE" in text


# --- rendering contract --------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "basis", "caption"),
    [
        ("critical-band", "critical-band method", "Critical band audibility"),
        (
            "equally-contributing",
            "equally-contributing critical-band method",
            "Equally-contributing critical band audibility",
        ),
        ("octave", "octave-band method", "Octave band audibility"),
    ],
)
def test_non_default_procedure_fiche_names_its_procedure(
    tmp_path: Path, method: str, basis: str, caption: str
) -> None:
    """A fiche for a non-default procedure names it and tables its own bands.

    The guide claims the fiche follows whichever of the four band procedures
    produced the result; without this the claim rested on the default alone.
    """
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    from phonometry.speech import sii_procedure

    proc = sii_procedure(method)
    n = proc.frequencies.size
    res = speech_intelligibility_index("normal", np.full(n, 25.0), method=method)
    out = tmp_path / f"sii_{method}.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert basis in text
    assert caption in text
    # The band column carries this procedure's own centre frequencies, not the
    # one-third-octave ones: check the lowest and highest, which are unique to
    # each table.
    assert f"{round(proc.frequencies[0])}" in text
    assert f"{round(proc.frequencies[-1])}" in text


def test_non_default_procedure_fiche_renders_in_spanish(tmp_path: Path) -> None:
    """The Spanish fiche translates the procedure name in the basis line."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = speech_intelligibility_index("normal", np.full(6, 25.0), method="octave")
    out = tmp_path / "sii_octave_es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "método de bandas de octava" in text
    assert "Audibilidad por bandas de octava" in text


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _example_c2()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _example_c2()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
