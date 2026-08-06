#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the HVAC duct-noise-spectrum ``.report()`` fiche.

The rendered values are checked against a clean-room oracle independent of the
library's own path. For a regenerated-noise spectrum the band sound power level
follows the VDI 2081-1 straight-duct formula
``L_W = 7 + 50 lg U + 10 lg S - 2 - 26 lg(1.14 + 0.02 f / U)`` (Bies, Hansen &
Howard, Engineering Noise Control 5th ed., Eq. (8.251)); the A-weighted total
combines the bands with the published octave A-weighting corrections (IEC
61672-1 / ISO 3744 Annex E Table E.2). The tests recompute the A-weighted and
overall levels and a couple of band values and assert they appear in the PDF,
along with the nominal band labels, the method-basis prose and the reported
quantity. An attenuation spectrum exercises the more-is-better direction.
Values are read back via pypdf text extraction; structural facts complete the
rendering contract.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from phonometry import ReportMetadata
from phonometry.noise_control import hvac

_PDF_MAGIC = b"%PDF"

_FREQS = np.array([63, 125, 250, 500, 1000, 2000, 4000], dtype=float)
_U = 12.0  # flow velocity, m/s
_S = 0.04  # duct area, m^2

#: Published octave-band A-weighting corrections (IEC 61672-1 / ISO 3744
#: Annex E Table E.2), the clean-room oracle for the A-weighted total.
_A_WEIGHT = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0])


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
    return hvac.flow_noise_straight_duct(_FREQS, _U, _S)


def _oracle_lw() -> np.ndarray:
    """Closed-form VDI 2081-1 straight-duct band sound power level."""
    return (
        7.0 + 50.0 * np.log10(_U) + 10.0 * np.log10(_S) - 2.0
        - 26.0 * np.log10(1.14 + 0.02 * _FREQS / _U)
    )


def _oracle_overall() -> float:
    lw = _oracle_lw()
    return float(10.0 * np.log10(np.sum(10.0 ** (lw / 10.0))))


def _oracle_lwa() -> float:
    lw = _oracle_lw() + _A_WEIGHT
    return float(10.0 * np.log10(np.sum(10.0 ** (lw / 10.0))))


def test_hand_oracle_matches_library() -> None:
    """The library band L_W equals the closed-form VDI values."""
    res = _result()
    np.testing.assert_allclose(res.values, _oracle_lw(), atol=1e-9)


def test_report_renders_oracle_values(tmp_path) -> None:
    """The fiche prints the A-weighted total, the overall L_W and band values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "hvac.pdf"
    returned = res.report(str(out), metadata=ReportMetadata(requirement=45.0))
    assert returned == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))

    assert f"{_oracle_lwa():.1f}" in text  # A-weighted total (boxed)
    assert "dB(A)" in text
    assert f"{_oracle_overall():.1f}" in text  # overall L_W (extended)
    assert "re 1 pW" in text
    # Two band L_W values (125 Hz and 1 kHz).
    lw = _oracle_lw()
    assert f"{lw[1]:.1f}" in text
    assert f"{lw[4]:.1f}" in text
    # Nominal band labels, caption and method basis.
    assert "63" in text
    assert "4000" in text
    assert "Octave-band regenerated sound power levels" in text
    assert "VDI 2081-1" in text
    # The fiche reports a model prediction, not an in-duct measurement.
    assert "Predicted flow-generated" in text
    assert "not a measurement" in text
    assert "Predicted (estimated) result" in text
    assert "tested specimen" not in text
    assert "prediction computed from the stated inputs" in text


def test_verbose_adds_a_weighting_columns(tmp_path) -> None:
    """verbose=True adds the A-weighting correction and A-weighted band columns."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    # The 63 Hz and 2 kHz A-weighting corrections to one decimal.
    assert "-26.2" in text
    assert "1.2" in text


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(45.0, "PASS"), (30.0, "FAIL")],
)
def test_power_verdict_lower_is_better(tmp_path, limit: float, verdict: str) -> None:
    """A regenerated-noise limit passes at or below the A-weighted level."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / f"verdict_{limit}.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=limit))
    text = _extract_text(str(out))
    assert verdict in text
    assert "required" in text
    assert (_oracle_lwa() <= limit) == (verdict == "PASS")
    assert math.isfinite(_oracle_lwa())


def test_attenuation_report_mean_and_direction(tmp_path) -> None:
    """An attenuation spectrum boxes the mean and passes when more is better."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    bands = np.array([63, 125, 250, 500, 1000, 2000], dtype=float)
    res = hvac.end_reflection_loss(bands, 0.3, termination="flush")
    mean_att = float(np.mean(res.values))
    out = tmp_path / "atten.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=1.0))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Octave-band attenuation" in text
    assert "Mean attenuation D" in text
    assert f"{mean_att:.1f}" in text
    # More attenuation is better: mean (>1 dB) clears the 1 dB minimum.
    assert "PASS" in text
    assert mean_att >= 1.0


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the HVAC vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Espectro de ruido de conducto de climatización" in text
    assert "Nivel de potencia acústica ponderado A" in text
    assert f"{_oracle_lwa():.1f}".replace(".", ",") in text
    assert "Se trata de una predicción" in text
    assert "no son una medición de una probeta" in text


def test_spanish_report_translates_the_element_label(tmp_path) -> None:
    """The element label's descriptive words are translated, its numbers kept."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    bands = np.array([63, 125, 250, 500, 1000, 2000], dtype=float)
    res = hvac.elbow_insertion_loss(bands, 0.3, bend_type="square", lined=True)
    out = tmp_path / "label_es.pdf"
    res.report(str(out), language="es")
    text = _extract_text(str(out))
    assert "Codo (rectangular, con absorbente, W = 300 mm)" in text
    assert "Elbow" not in text


@pytest.mark.parametrize(
    ("value", "printed"),
    [(0.25, "0.3"), (0.35, "0.4"), (-0.25, "-0.3")],
)
def test_verdict_prints_the_value_its_verdict_used(value: float, printed: str) -> None:
    """A half-way value prints the display rounding the verdict was decided on.

    Python's own formatting rounds halves to even (0.25 -> "0.2"), which would
    contradict a verdict decided on the half-away-from-zero display rounding
    the fiches use throughout.
    """
    from phonometry._report._noise_control_fiche import performance_verdict

    text, _ = performance_verdict(
        value, value, "D", higher_is_better=True, language="en"
    )
    assert f"D = {printed} dB" in text
    assert f"&#8805; {printed} dB" in text


def test_verdict_boundary_passes_on_the_displayed_value() -> None:
    """A value that displays as its requirement passes, in both directions."""
    from phonometry._report._noise_control_fiche import performance_verdict

    _, higher = performance_verdict(
        0.25, 0.3, "D", higher_is_better=True, language="en"
    )
    _, lower = performance_verdict(
        0.25, 0.3, "L", higher_is_better=False, language="en"
    )
    assert higher
    assert lower


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
