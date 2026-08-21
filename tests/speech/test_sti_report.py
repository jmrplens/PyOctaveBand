#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the IEC 60268-16 speech-transmission-index ``.report()`` fiche.

The rendered index is checked against an oracle independent of the renderer: a
uniform modulation transfer value m = 0.5 across every band and modulation
frequency maps to STI = 0.5 exactly (IEC 60268-16:2020 A.3.1.2), which is pure
arithmetic (no filtering), so the boxed value is stable across platforms. The
Annex F qualification band, the verdict direction (a higher STI passes), the
EN/ES parity and the rendering contract complete the checks. Values are read
back from the PDF via pypdf text extraction; structural facts (one page,
rejected engines/languages) complete the rendering contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.speech import sti_from_impulse_response
from phonometry.speech.sti import _MOD_FREQS, _NUM_BANDS, _sti_from_mtf


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _uniform_result(m: float):
    """An STIResult from a uniform MTF (pure arithmetic, platform-stable)."""
    return _sti_from_mtf(np.full((_NUM_BANDS, _MOD_FREQS.size), m))


def _decay_result(t60: float, fs: int = 48000, seed: int = 0):
    """A deterministic full-STI indirect result from an exponential decay IR."""
    rng = np.random.default_rng(seed)
    n = int(2.0 * t60 * fs)
    t = np.arange(n) / fs
    ir = rng.standard_normal(n) * np.exp(-3.0 * np.log(10.0) * t / t60)
    return sti_from_impulse_response(ir, fs)


# --- exact oracle --------------------------------------------------------------


def test_uniform_mtf_half_renders_sti_half(tmp_path) -> None:
    """m = 0.5 in every band maps to STI = 0.50 (A.3.1.2), band G."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _uniform_result(0.5)
    assert res.sti == pytest.approx(0.5, abs=1e-9)
    assert res.rating == "G"
    out = tmp_path / "sti.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "STI = 0.50" in text
    assert "Annex F): G" in text
    # The full indirect method carries the fourteen modulation frequencies.
    assert "full STI, indirect method" in text


def test_stipa_method_is_named(tmp_path) -> None:
    """A two-column MTF (STIPA) names the direct method in the basis line."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _sti_from_mtf(np.full((_NUM_BANDS, 2), 0.5))
    out = tmp_path / "stipa.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    assert "STIPA, direct method" in _extract_text(str(out))


# --- verdict direction (higher is better) -------------------------------------


def test_verdict_passes_at_or_above_requirement(tmp_path) -> None:
    """STI = 0.50 passes a 0.45 minimum and fails a 0.55 minimum."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _uniform_result(0.5)
    out_pass = tmp_path / "pass.pdf"
    res.report(str(out_pass), metadata=ReportMetadata(requirement=0.45))
    assert "PASS" in _extract_text(str(out_pass))
    out_fail = tmp_path / "fail.pdf"
    res.report(str(out_fail), metadata=ReportMetadata(requirement=0.55))
    assert "FAIL" in _extract_text(str(out_fail))


def test_verdict_passes_exactly_at_requirement(tmp_path) -> None:
    """A minimum equal to the displayed STI passes (>= boundary)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _uniform_result(0.5)
    out = tmp_path / "eq.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=0.5))
    assert "PASS" in _extract_text(str(out))


def test_verdict_prints_the_requirement_at_full_precision(tmp_path) -> None:
    """The requirement is printed at the STI's own two decimals.

    A one-decimal requirement would round 0.52 to "0.5" and read as a
    self-contradiction beside a failing STI = 0.50, and would print a plain
    "0.9" for a 0.90 minimum.
    """
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _uniform_result(0.5)
    out_fail = tmp_path / "req052.pdf"
    res.report(str(out_fail), metadata=ReportMetadata(requirement=0.52))
    text = _extract_text(str(out_fail)).replace("\n", " ")
    assert "STI = 0.50" in text
    assert "0.52" in text
    assert "FAIL" in text

    out_round = tmp_path / "req090.pdf"
    res.report(str(out_round), metadata=ReportMetadata(requirement=0.90))
    assert "0.90" in _extract_text(str(out_round))


def test_verdict_decides_on_the_requirement_it_prints(tmp_path) -> None:
    """A requirement with hidden digits is judged on the value shown.

    0.5049 prints as "0.50"; deciding on the unrounded number would fail an
    STI of 0.50 while printing "STI = 0.50, required >= 0.50".
    """
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _uniform_result(0.5)
    out = tmp_path / "hidden.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=0.5049))
    text = _extract_text(str(out)).replace("\n", " ")
    assert "STI = 0.50" in text
    assert "0.50" in text
    assert "PASS" in text


# --- realistic indirect measurement -------------------------------------------


def test_decay_measurement_renders_its_own_value(tmp_path) -> None:
    """A full-STI indirect fiche prints its STI, rating and a band MTI value."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    from phonometry._report._i18n import format_number

    res = _decay_result(0.8)
    out = tmp_path / "decay.pdf"
    res.report(
        str(out),
        metadata=ReportMetadata(
            specimen="Voice-alarm loudspeaker line",
            measurement_standard="IEC 60268-16",
            requirement=0.5,
        ),
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"STI = {format_number(res.sti, 'en', decimals=2)}" in text
    assert f"Annex F): {res.rating}" in text
    assert "Voice-alarm loudspeaker line" in text
    assert "PASS" in text  # STI ~ 0.64 >= 0.50


# --- metadata ------------------------------------------------------------------


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the header grid; no requirement, no verdict."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _uniform_result(0.5)
    out = tmp_path / "meta.pdf"
    res.report(
        str(out),
        metadata=ReportMetadata(
            client="Example works",
            specimen="Public-address system",
            laboratory="Reference lab",
            report_id="STI-1",
        ),
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Public-address system" in text
    assert "Result vs requirement" not in text


# --- Spanish fiche -------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the STI vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _uniform_result(0.5)
    out = tmp_path / "sti_es.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=0.45), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Índice de transmisión del habla" in text
    assert "STI = 0,50" in text
    assert "Banda de calificación" in text
    assert "CUMPLE" in text


# --- rendering contract --------------------------------------------------------


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _uniform_result(0.5)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _uniform_result(0.5)
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
