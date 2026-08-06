#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the reactive-silencer transmission-loss ``.report()`` fiche.

The rendered values are checked against a clean-room oracle derived from the
simple-expansion-chamber closed form (Bies, Hansen & Howard, Engineering Noise
Control 5th ed., Eq. (8.111)), independent of the library's four-pole path:
``TL = 10 lg[1 + (1/4)(m - 1/m)^2 sin^2(kL)]`` with the area ratio
``m = S_exp / S_duct`` and ``k = 2 pi f / c``. The tests recompute the mean and
peak transmission loss and a couple of band values from this closed form and
assert they appear in the PDF, along with the nominal band labels, the
four-pole method basis prose and the device kind. Values are read back via pypdf
text extraction; structural facts (one page, rejected engines/languages)
complete the rendering contract.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from phonometry.noise_control import silencers as sl

_PDF_MAGIC = b"%PDF"

_FREQS = np.array([63, 125, 250, 500, 1000, 2000, 4000], dtype=float)
_LENGTH = 0.5
_S_EXP = 0.08
_S_DUCT = 0.01
_C = 343.0


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _result():
    return sl.expansion_chamber(_FREQS, _LENGTH, _S_EXP, _S_DUCT)


def _oracle_tl() -> np.ndarray:
    """Closed-form expansion-chamber TL (Bies Eq. (8.111))."""
    m = _S_EXP / _S_DUCT
    k = 2.0 * np.pi * _FREQS / _C
    return 10.0 * np.log10(1.0 + 0.25 * (m - 1.0 / m) ** 2 * np.sin(k * _LENGTH) ** 2)


def test_hand_oracle_matches_library() -> None:
    """The library TL equals the closed-form expansion-chamber values."""
    res = _result()
    np.testing.assert_allclose(res.transmission_loss, _oracle_tl(), atol=1e-9)


def test_report_renders_oracle_values(tmp_path) -> None:
    """The fiche prints the mean and peak TL and a couple of band TL values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "silencer.pdf"
    returned = res.report(str(out), metadata=None)
    assert returned == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))

    tl = _oracle_tl()
    assert f"{float(np.mean(tl)):.1f}" in text  # mean TL (boxed)
    assert f"{float(np.max(tl)):.1f}" in text  # peak TL (extended)
    # Two band TL values (125 Hz and 500 Hz).
    assert f"{tl[1]:.1f}" in text
    assert f"{tl[3]:.1f}" in text
    # Nominal band labels and caption.
    assert "63" in text
    assert "4000" in text
    assert "Octave-band transmission loss" in text
    # Method basis prose and the device kind.
    assert "Munjal" in text
    assert "four-pole" in text
    assert "expansion chamber" in text
    # The fiche reports a model prediction, not a bench measurement.
    assert "Predicted transmission loss" in text
    assert "not a measurement" in text
    assert "Predicted (estimated) result" in text
    assert "tested specimen" not in text
    assert "prediction computed from the stated inputs" in text


def test_insertion_loss_column_when_impedances_given(tmp_path) -> None:
    """A result with end impedances renders the insertion-loss column."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    z = 1.206 * _C / _S_DUCT  # anechoic reference -> IL equals TL
    res = sl.expansion_chamber(
        _FREQS, _LENGTH, _S_EXP, _S_DUCT,
        source_impedance=z, radiation_impedance=z,
    )
    out = tmp_path / "with_il.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "IL [dB]" in text
    # For the anechoic reference the insertion loss equals the transmission loss.
    assert "Mean insertion loss IL" in text


def test_resonator_reports_resonances(tmp_path) -> None:
    """A tuned resonator lists its resonance frequencies in the boxed terms."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    f = np.array([50, 63, 80, 100, 125, 160, 200], dtype=float)
    res = sl.helmholtz_resonator(f, 0.01, 1e-4, 0.02, 1e-3)
    out = tmp_path / "helmholtz.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Helmholtz resonator" in text
    assert "Resonance frequencies" in text


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(6.0, "PASS"), (12.0, "FAIL")],
)
def test_verdict_against_declared_minimum(tmp_path, limit: float, verdict: str) -> None:
    """A declared minimum yields a PASS/FAIL verdict (more TL is better)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    from phonometry import ReportMetadata

    res = _result()
    out = tmp_path / f"verdict_{limit}.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=limit))
    text = _extract_text(str(out))
    assert verdict in text
    assert "required" in text
    assert (float(np.mean(_oracle_tl())) >= limit) == (verdict == "PASS")


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the silencer vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Pérdida por transmisión de silenciador reactivo" in text
    assert "Pérdida por transmisión media" in text
    assert f"{float(np.mean(_oracle_tl())):.1f}".replace(".", ",") in text
    # The device kind is result data but a fixed vocabulary word, so it is
    # translated rather than printed in English.
    assert "Dispositivo: cámara de expansión" in text
    assert "expansion chamber" not in text
    assert "Se trata de una predicción" in text
    assert "no son una medición de una probeta" in text


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
