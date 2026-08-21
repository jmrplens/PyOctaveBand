#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the machine-enclosure insertion-loss ``.report()`` fiche.

The rendered values are checked against a clean-room oracle derived from the
Bies, Hansen & Howard model (Engineering Noise Control 5th ed., section 7.4.2),
independent of the library's own combination path: the interior room constant
``R_i = S_i alpha_i / (1 - alpha_i)`` (Eq. (7.103)), the build-up correction
``C = 10 lg(0.3 + S_E / R_i)`` (Eq. (7.111)) and the net insertion loss
``IL = R - C``. The tests recompute the mean insertion loss and a couple of band
values from these closed forms and assert they appear in the PDF, along with the
nominal band labels, the method-basis prose and the surface areas. Values are
read back via pypdf text extraction; structural facts (one page, rejected
engines/languages) complete the rendering contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata, noise_control

_FREQS = np.array([63, 125, 250, 500, 1000, 2000, 4000], dtype=float)
_PANEL_R = np.array([18, 22, 28, 33, 38, 42, 45], dtype=float)
_S_E = 24.0  # external surface area, m^2
_S_I = 30.0  # internal surface area, m^2
_ALPHA = 0.30  # mean interior absorption


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _result():
    return noise_control.enclosure_insertion_loss(
        _PANEL_R, _S_E, _S_I, _ALPHA, frequencies=_FREQS
    )


def _oracle_correction() -> np.ndarray:
    """Closed-form build-up correction C = 10 lg(0.3 + S_E / R_i) (Eq. (7.111))."""
    r_i = _S_I * _ALPHA / (1.0 - _ALPHA)
    return 10.0 * np.log10(0.3 + _S_E / r_i) * np.ones_like(_PANEL_R)


def _oracle_il() -> np.ndarray:
    """Closed-form net insertion loss IL = R - C."""
    return _PANEL_R - _oracle_correction()


def _oracle_mean_il() -> float:
    return float(np.mean(_oracle_il()))


def test_hand_oracle_matches_library() -> None:
    """The library IL and C equal the closed-form Bies values."""
    res = _result()
    np.testing.assert_allclose(res.correction, _oracle_correction(), atol=1e-9)
    np.testing.assert_allclose(res.insertion_loss, _oracle_il(), atol=1e-9)


def test_report_renders_oracle_values(tmp_path) -> None:
    """The fiche prints the mean IL and a couple of band R/C/IL values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "enclosure.pdf"
    returned = res.report(str(out), metadata=ReportMetadata(requirement=20.0))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))

    # Mean insertion loss (boxed) to one decimal.
    assert f"{_oracle_mean_il():.1f}" in text
    # Two band IL values (250 Hz and 1 kHz) to one decimal.
    il = _oracle_il()
    assert f"{il[2]:.1f}" in text
    assert f"{il[4]:.1f}" in text
    # The supplied panel R heads its column.
    assert f"{_PANEL_R[2]:.1f}" in text
    # The build-up correction C to one decimal.
    assert f"{_oracle_correction()[0]:.1f}" in text
    # Nominal band labels and the caption.
    assert "63" in text
    assert "4000" in text
    assert "Octave-band insertion loss" in text
    # Method basis prose and the surface areas in the boxed result.
    assert "Bies" in text
    assert "IL = R - C" in text
    assert "24.00" in text  # external surface area S_E
    assert "30.00" in text  # internal surface area S_i
    # The fiche reports a model prediction, so it says so and does not carry
    # the measurement fiches' specimen-scoped footer.
    assert "Predicted insertion loss" in text
    assert "not a measurement" in text
    assert "Predicted (estimated) result" in text
    assert "tested specimen" not in text
    assert "prediction computed from the stated inputs" in text


def test_verbose_adds_room_constant_column(tmp_path) -> None:
    """verbose=True adds the interior room constant R_i column."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    r_i = _S_I * _ALPHA / (1.0 - _ALPHA)  # = 12.857... -> 12.9
    assert f"{r_i:.1f}" in text


def test_third_octave_labels_and_caption(tmp_path) -> None:
    """A one-third-octave set is labelled by nominal centres and captioned."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630], dtype=float)
    r = np.linspace(20.0, 40.0, freqs.size)
    res = noise_control.enclosure_insertion_loss(
        r, _S_E, _S_I, _ALPHA, frequencies=freqs
    )
    out = tmp_path / "third.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "One-third-octave-band insertion loss" in text
    for label in ("100", "125", "160", "630"):
        assert label in text


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(20.0, "PASS"), (40.0, "FAIL")],
)
def test_verdict_against_declared_minimum(tmp_path, limit: float, verdict: str) -> None:
    """A declared minimum yields a PASS/FAIL verdict (more insertion loss is better)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / f"verdict_{limit}.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=limit))
    text = _extract_text(str(out))
    assert verdict in text
    assert "required" in text
    # Mean IL (28.9) passes a 20 dB minimum, fails a 40 dB one.
    assert (_oracle_mean_il() >= limit) == (verdict == "PASS")
    assert math.isfinite(_oracle_mean_il())


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the machine, environment and identity fields."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    metadata = ReportMetadata(
        client="Example works",
        specimen="Reciprocating compressor",
        test_room="Machine hall, line 3",
        instrumentation="Class 1 SLM, octave bank",
        laboratory="Acoustics lab",
        report_id="ENC-01",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    for token in (
        "Example works",
        "Reciprocating compressor",
        "Machine hall, line 3",
        "Class 1 SLM, octave bank",
        "Acoustics lab",
    ):
        assert token in text


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the enclosure vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Pérdida por inserción de encapsulado de máquina" in text
    assert "Pérdida por inserción media" in text
    # Comma decimal separator on the mean insertion loss.
    assert f"{_oracle_mean_il():.1f}".replace(".", ",") in text


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
