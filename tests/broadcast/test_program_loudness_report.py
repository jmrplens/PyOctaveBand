#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the EBU R 128 programme-loudness report (``.report()`` -> PDF).

These tests assert both structural facts (a valid single-page PDF is written
for a measured programme, unknown engines/tolerances are rejected, a compliant
and a non-compliant programme both render, and the informational rows - LRA,
momentary/short-term maxima - never change the verdict) and the displayed
content: the R 128 tolerance rule applied by the verdict (+-0.2 LU QC per
item i by default, +-1.0 LU live per item h on request), the 0.1 LU
display-rounding consistency at the tolerance boundary, and the exact numbers
the fiche prints (extracted with pypdf). The loudness algorithm itself is
validated against the EBU Tech 3341/3342 test signals elsewhere
(tests/broadcast/test_program_loudness.py); here a small self-contained
synthetic signal keeps the fiche test independent.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("reportlab")

from phonometry import ReportMetadata
from phonometry._report.broadcast import _verdict
from phonometry.broadcast import program_loudness

FS = 48000
_PDF_MAGIC = b"%PDF"


def _sine(level_dbfs: float, duration: float, freq: float = 1000.0) -> np.ndarray:
    """A 1 kHz sine with per-sample peak level ``level_dbfs`` (dB re full scale)."""
    t = np.arange(round(duration * FS)) / FS
    return 10.0 ** (level_dbfs / 20.0) * np.sin(2.0 * np.pi * freq * t)


def _stereo(x: np.ndarray) -> np.ndarray:
    """The signal applied in phase to both channels."""
    return np.vstack([x, x])


def _steps(segments: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Concatenated stereo 1 kHz tone steps (level dBFS, duration s)."""
    return _stereo(np.concatenate([_sine(lvl, dur) for lvl, dur in segments]))


def _case1_result():
    """Tech 3342 Case 1 shaped steps, trimmed 0.4 dB onto the -23.0 target.

    The -20.4/-30.4 dBFS stereo steps keep the Case 1 loudness range near
    10 LU while landing the integrated loudness at -23.0 LUFS, inside the
    default +-0.2 LU QC tolerance of EBU R 128 item i).
    """
    return program_loudness(_steps(((-20.4, 20.0), (-30.4, 20.0))), FS)


def _assert_pdf(path: str) -> None:
    import os

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0


def _assert_one_page(path: str) -> None:
    _assert_pdf(path)
    from pypdf import PdfReader

    assert len(PdfReader(path).pages) == 1


def test_report_writes_one_page_pdf(tmp_path) -> None:
    """A measured programme renders a one-page compliance fiche."""
    result = _case1_result()
    assert np.isfinite(result.integrated)
    out = tmp_path / "loudness.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = _case1_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_full_metadata_renders_one_page(tmp_path) -> None:
    """A populated ReportMetadata renders a one-page fiche."""
    md = ReportMetadata(
        specimen="Reference tone sequence",
        client="Broadcast Client Ltd.",
        manufacturer="Post-production House",
        test_room="Reference monitoring room R1",
        measurement_standard="EBU R 128",
        test_date="2026-07-20",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-R128",
        requirement=-23.0,
    )
    out = tmp_path / "meta.pdf"
    _case1_result().report(str(out), metadata=md)
    _assert_one_page(str(out))


def test_compliant_programme_passes(tmp_path) -> None:
    """The trimmed signal is compliant (I within -23.0 +/-0.2 LU, TP <= -1 dBTP)."""
    result = _case1_result()
    assert abs(result.integrated - (-23.0)) <= 0.2
    _text, passed = _verdict(result, -23.0)
    assert passed is True
    out = tmp_path / "pass.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=-23.0))
    _assert_one_page(str(out))


def test_non_compliant_true_peak_fails(tmp_path) -> None:
    """A 0 dBFS tone exceeds the -1 dBTP true-peak ceiling and fails."""
    result = program_loudness(_stereo(_sine(0.0, 10.0)), FS)
    assert result.true_peak > -1.0
    _text, passed = _verdict(result, -23.0)
    assert passed is False
    out = tmp_path / "fail.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=-23.0))
    _assert_one_page(str(out))


def test_informational_rows_do_not_change_verdict(tmp_path) -> None:
    """LRA and the momentary/short-term maxima never affect the PASS verdict."""
    result = _case1_result()
    _text, passed = _verdict(result, -23.0)
    assert passed is True
    # Force extreme informational values; the verdict must stay PASS because it
    # is driven only by the integrated loudness and the maximum true peak.
    tampered = dataclasses.replace(
        result, loudness_range=999.0, max_momentary=0.0, max_short_term=0.0
    )
    _text2, passed2 = _verdict(tampered, -23.0)
    assert passed2 is True
    out = tmp_path / "info.pdf"
    tampered.report(str(out), metadata=ReportMetadata(requirement=-23.0))
    _assert_one_page(str(out))


def _extract_text(path: str) -> str:
    """The concatenated text of every page (for language assertions)."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_default_tolerance_is_qc_02_lu() -> None:
    """The default verdict applies the +-0.2 LU QC rule of R 128 item i).

    A non-live programme 0.5 LU off target (compliant under the pre-2020
    +-0.5 LU rule that R 128 dropped in its 2020 revision) must FAIL the
    default QC tolerance.
    """
    result = dataclasses.replace(_case1_result(), integrated=-23.5)
    _text, passed = _verdict(result, -23.0)
    assert passed is False
    _text2, passed2 = _verdict(dataclasses.replace(result, integrated=-23.2), -23.0)
    assert passed2 is True


def test_live_tolerance_applies_item_h_1_lu(tmp_path) -> None:
    """``tolerance="live"`` applies the +-1.0 LU rule of R 128 item h)."""
    result = dataclasses.replace(_case1_result(), integrated=-23.8)
    _t1, qc_passed = _verdict(result, -23.0, "qc")
    _t2, live_passed = _verdict(result, -23.0, "live")
    assert qc_passed is False
    assert live_passed is True
    out = tmp_path / "live.pdf"
    result.report(
        str(out), metadata=ReportMetadata(requirement=-23.0), tolerance="live"
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "±1.0 LU" in text
    assert "item h" in text
    assert "live programmes" in text


def test_unknown_tolerance_rejected(tmp_path) -> None:
    """An unknown tolerance rule raises ``ValueError``."""
    result = _case1_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="tolerance"):
        result.report(out, tolerance="broadcast")


def test_verdict_follows_displayed_rounding_at_boundary() -> None:
    """Pass/fail is evaluated on the 0.1 LU displayed rounding (Tech 3341).

    -23.249 LUFS displays as -23.2 (delta -0.2 LU) and must PASS the
    +-0.2 LU rule; -23.251 displays as -23.3 and must FAIL; the exact half
    -23.25 rounds away from zero to -23.3 and also FAILs. The printed
    numbers can then never contradict the verdict.
    """
    base = _case1_result()
    for integrated, expected in ((-23.249, True), (-23.251, False), (-23.25, False)):
        result = dataclasses.replace(base, integrated=integrated)
        _text, passed = _verdict(result, -23.0)
        assert passed is expected, integrated


def test_fiche_pins_displayed_numbers(tmp_path) -> None:
    """The rendered fiche shows the QC tolerance, its R 128 item and the
    0.1 LU rounded measured loudness."""
    result = _case1_result()
    out = tmp_path / "pins.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=-23.0))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    # Independent expectation: the trimmed Case 1 steps land at -23.0 LUFS
    # (the -20/-30 dBFS analytic pair shifted by -0.4 dB). The printed sign is
    # the typographic minus every fiche reading carries.
    assert "−23.0 LUFS ±0.2 LU" in text
    assert "(EBU R 128 item i)" in text
    assert "I = −23.0 LUFS" in text
    assert "Quality Control (EBU R 128 item i)." in text


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """``language="es"`` renders a one-page Spanish fiche with comma decimals."""
    import re

    result = _case1_result()
    out = tmp_path / "program_es.pdf"
    result.report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Conformidad de sonoridad de programa" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = _case1_result()
    with pytest.raises(ValueError, match="language"):
        result.report(str(tmp_path / "bad.pdf"), language="xx")
