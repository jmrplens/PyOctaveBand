#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 7626 mechanical-mobility ``.report()`` fiche.

The rendered values are checked against the module's closed-form oracle
(ISO 7626-1:2011 Table 1 / 3.1.2): the driving-point mobility of a single-
degree-of-freedom resonator (m = 2 kg, k = 8000 N/m, c = 5 N.s/m) peaks at the
undamped natural frequency f0 = (1/2pi) sqrt(k/m) = 10.07 Hz, where it is purely
real and equal to 1/c = 0.2 m/(N.s). Including f0 in the axis lands the peak
exactly there. Values are read back from the PDF via pypdf text extraction;
structural facts (one page, rejected engines/languages) complete the rendering
contract.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from phonometry import ReportMetadata
from phonometry.vibration import sdof_mobility_result
from phonometry.vibration.structural.mechanical_mobility import MobilityResult

_PDF_MAGIC = b"%PDF"

_M, _K, _C = 2.0, 8000.0, 5.0
_F0 = math.sqrt(_K / _M) / (2.0 * math.pi)  # ~10.066 Hz


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


def _driving_point_result() -> MobilityResult:
    """The SDOF driving-point mobility with the resonance sampled exactly."""
    freqs = np.unique(np.append(np.logspace(0.0, np.log10(200.0), 300), _F0))
    return sdof_mobility_result(freqs, _M, _K, _C)


def test_result_peak_oracle() -> None:
    """The peak mobility is 1/c at the natural frequency (module oracle)."""
    res = _driving_point_result()
    assert float(res.magnitude.max()) == pytest.approx(1.0 / _C, rel=1e-9)
    peak_f = float(res.frequencies[int(np.argmax(res.magnitude))])
    assert peak_f == pytest.approx(_F0, abs=1e-6)


def test_report_renders_peak_and_basis(tmp_path) -> None:
    """The fiche prints the peak mobility, the resonance frequency and ISO 7626."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _driving_point_result()
    out = tmp_path / "mobility.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Mechanical mobility measurement" in text
    assert "ISO 7626-1:2011" in text
    assert "ISO 7626-2:2015" in text
    # Peak mobility 1/c = 0.2 m/(N.s) at f0 = 10.1 Hz (three sig figs / 0.1 Hz).
    assert "0.2" in text
    assert "10.1" in text
    assert "driving-point" in text
    assert "Peak mobility" in text


def test_transfer_frf_labels_transfer(tmp_path) -> None:
    """A transfer FRF (driving_point=False) is named a transfer mobility."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    base = _driving_point_result()
    res = MobilityResult(
        frequencies=base.frequencies,
        mobility=base.mobility,
        driving_point=False,
    )
    out = tmp_path / "transfer.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "transfer mechanical mobility" in text
    assert "driving-point" not in text


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the client, the specimen and the instrumentation."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _driving_point_result()
    metadata = ReportMetadata(
        specimen="Machine support bracket",
        client="Example structures client",
        test_room="Modal-analysis rig",
        instrumentation="Impact hammer + accelerometer",
        laboratory="Vibration laboratory",
        report_id="MOB-7626",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example structures client" in text
    assert "Machine support bracket" in text
    assert "Impact hammer + accelerometer" in text
    assert "MOB-7626" in text


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the mobility vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _driving_point_result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Medición de la movilidad mecánica" in text
    assert "en punto de excitación" in text
    assert "0,2" in text


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _driving_point_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _driving_point_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
