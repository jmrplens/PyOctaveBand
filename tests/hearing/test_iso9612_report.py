#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 9612:2009 occupational noise-exposure ``.report()`` fiche.

The rendered values are checked against a hand-derivable oracle independent of
the library: two 4 h tasks at 85 dB and 75 dB give
``LEX,8h = 75 + 10 lg 5.5 = 82.4 dB`` and, with single-sample tasks
(``u1a = 0``) measured by a personal exposure meter (``u2 = 1.5``,
``u3 = 1.0``), the Annex C budget collapses to the closed form
``u^2 = (c1a,1^2 + c1a,2^2)(u2^2 + u3^2) = (101/121) * 3.25`` so
``U = 1.65 u = 2.7 dB``. The Directive 2003/10/EC assessment is exercised at
the 80/85/87 dB(A) boundaries on the displayed (one-decimal) value, and the
Spanish fiche is checked for the occupational vocabulary and the comma
decimal separator. Values are read back from the PDF via pypdf text
extraction; structural facts (one page, rejected engines/languages) complete
the rendering contract.
"""

from __future__ import annotations

import math
import os

import pytest

from phonometry import ReportMetadata
from phonometry.hearing import (
    Task,
    full_day_exposure,
    job_based_exposure,
    task_based_exposure,
)

_PDF_MAGIC = b"%PDF"


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _two_task_result():
    """Hand-derivable oracle: 4 h at 85 dB plus 4 h at 75 dB."""
    tasks = [
        Task(samples=(85.0,), duration_hours=4.0, label="machining"),
        Task(samples=(75.0,), duration_hours=4.0, label="assembly"),
    ]
    return task_based_exposure(tasks, warn=False)


def _single_task_result(level: float):
    """A whole nominal day at one level: LEX,8h equals the level exactly."""
    return task_based_exposure(
        [Task(samples=(level,), duration_hours=8.0, label="task")], warn=False
    )


# --- hand-computed oracle -----------------------------------------------------


def test_two_task_hand_oracle_values() -> None:
    """4h@85 + 4h@75 gives LEX,8h = 75 + 10 lg 5.5 and the closed-form U.

    Energy sum: (4/8) 10^8.5 + (4/8) 10^7.5 = 10^7.5 * 5.5. The sensitivity
    coefficients are c1a,1 = 10/11 and c1a,2 = 1/11 (Formula C.4), u1a = 0 for
    single-sample tasks, so u^2 = (101/121)(1.5^2 + 1.0^2) (Formula C.3) and
    U = 1.65 u (Clause 14).
    """
    res = _two_task_result()
    assert res.lex_8h == pytest.approx(75.0 + 10.0 * math.log10(5.5), abs=1e-9)
    expected_u = math.sqrt((101.0 / 121.0) * (1.5**2 + 1.0**2))
    assert res.combined_standard_uncertainty == pytest.approx(expected_u, abs=1e-9)
    assert res.expanded_uncertainty == pytest.approx(1.65 * expected_u, abs=1e-9)


def test_report_renders_hand_computed_values(tmp_path) -> None:
    """The fiche prints the hand-derived LEX,8h = 82.4 dB and U = 2.7 dB."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _two_task_result()
    out = tmp_path / "iso9612.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "82.4" in text  # LEX,8h to one decimal (Clause 15 e)
    assert "2.7" in text  # U to one decimal, stated separately
    assert "task-based measurement (Clause 9)" in text
    assert "machining" in text
    assert "assembly" in text
    # Personal exposure meter fallback (Clause 15 c) without metadata.
    assert "IEC 61252" in text


def test_annex_d_example_renders_pinned_numbers(tmp_path) -> None:
    """The ISO 9612 Annex D welders' day prints 84.3 dB and U = 3.2 dB."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    tasks = [
        Task(samples=(70.0,), duration_hours=1.5, label="planning"),
        Task(
            samples=(80.1, 82.2, 79.6),
            duration_hours=5.0,
            duration_range=(4.0, 6.0),
            label="welding",
        ),
        Task(
            samples=(86.5, 92.4, 89.3, 93.2, 87.8, 86.2),
            duration_hours=1.5,
            duration_range=(1.0, 2.0),
            label="grinding",
        ),
    ]
    res = task_based_exposure(tasks, warn=False)
    out = tmp_path / "annex_d.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "84.3" in text
    assert "3.2" in text


# --- Directive 2003/10/EC assessment boundaries -------------------------------


@pytest.mark.parametrize(
    ("level", "n_exceeded", "verdict"),
    [
        (80.0, 0, "PASS"),  # at the lower action value: not exceeded
        (80.06, 1, "PASS"),  # displays 80.1: lower action value exceeded
        (85.0, 1, "PASS"),  # at the upper action value: not exceeded
        (85.06, 2, "PASS"),  # displays 85.1: both action values exceeded
        (87.0, 2, "PASS"),  # at the limit value: verdict passes
        (87.06, 3, "FAIL"),  # displays 87.1: limit value exceeded
    ],
)
def test_directive_boundaries_on_displayed_value(
    tmp_path, level: float, n_exceeded: int, verdict: str
) -> None:
    """The 80/85/87 dB(A) rows flip on the one-decimal displayed value."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _single_task_result(level)
    out = tmp_path / f"boundary_{level}.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    # "Exceeded" (capital E) marks only the exceeded rows; the "Not exceeded"
    # status spells it lowercase, so the count isolates the exceeded ones.
    assert text.count("Exceeded") == n_exceeded
    assert text.count("Not exceeded") == 3 - n_exceeded
    assert verdict in text


# --- job-based and full-day layouts -------------------------------------------


def test_job_based_report_renders_sampling_summary(tmp_path) -> None:
    """A job-based result renders the sampling summary (Annex E numbers)."""
    pytest.importorskip("reportlab")
    res = job_based_exposure(
        [88.1, 86.1, 89.7, 86.5, 91.1, 86.7], effective_duration_hours=7.5
    )
    out = tmp_path / "job.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "job-based measurement (Clause 10)" in text
    assert "Sampling summary" in text
    assert "Table C.4" in text
    assert "88.2" in text  # LEX,8h (unrounded intermediates; see module notes)
    assert "3.8" in text  # U


def test_full_day_report_renders(tmp_path) -> None:
    """A full-day result renders the Clause 11 basis and one page."""
    pytest.importorskip("reportlab")
    res = full_day_exposure(
        [88.0, 91.9, 87.6, 90.4, 89.0, 88.4], effective_duration_hours=9.25
    )
    out = tmp_path / "full_day.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "full-day measurement (Clause 11)" in text
    assert "90.1" in text
    assert "FAIL" in text  # 90.1 dB exceeds the 87 dB(A) limit value


# --- metadata, verbose and instrumentation ------------------------------------


def test_metadata_header_and_instrumentation_override(tmp_path) -> None:
    """Supplied metadata renders the header grid; instrumentation overrides."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _two_task_result()
    metadata = ReportMetadata(
        client="Example works",
        specimen="Lathe operators",
        test_room="Machining hall",
        instrumentation="SLM X1, s/n 123",
        calibration="Calibrator C2, verified 2026-01-15",
        laboratory="Prevention service",
        report_id="EXP-1",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Lathe operators" in text
    assert "SLM X1, s/n 123" in text
    assert "Calibrator C2" in text
    # The free-text instrumentation replaces the instrument-class fallback.
    assert "IEC 61252" not in text


def test_verbose_adds_annex_c_columns(tmp_path) -> None:
    """verbose=True adds the per-task u1a/u1b/u2 columns to the task table."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _two_task_result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    _assert_one_page(str(out))
    flat = "".join(_extract_text(str(out)).split())
    assert "u1a,m" in flat
    assert "u1b,m" in flat


def test_instrument_class_is_threaded_to_the_result() -> None:
    """The strategy functions record the instrument class for Clause 15 c."""
    assert _two_task_result().instrument == "personal_exposimeter"
    job = job_based_exposure(
        [80.0, 81.0], effective_duration_hours=8.0, instrument="class1", warn=False
    )
    assert job.instrument == "class1"


def test_class1_fallback_names_the_sound_level_meter(tmp_path) -> None:
    """Without metadata, the class-1 instrument prints its IEC 61672-1 name."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = task_based_exposure(
        [Task(samples=(75.0,), duration_hours=8.0, label="task")],
        instrument="class1",
        warn=False,
    )
    out = tmp_path / "class1.pdf"
    res.report(str(out))
    assert "IEC 61672-1 class 1" in _extract_text(str(out))


# --- Spanish fiche -------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the occupational vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _two_task_result()
    out = tmp_path / "iso9612_es.pdf"
    res.report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Determinación de la exposición al ruido en el trabajo" in text
    assert "Nivel de exposición diario equivalente" in text
    assert "82,4" in text
    assert "Valor límite de exposición" in text
    assert "CUMPLE" in text


# --- rendering contract --------------------------------------------------------


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _two_task_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _two_task_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
