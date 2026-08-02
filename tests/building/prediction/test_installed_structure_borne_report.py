#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the EN 12354-5 installed structure-borne prediction ``.report()``.

The fiche is a prediction, not a measurement, and the sheet says so. The
rendered values are checked against a clean-room oracle derived from the
standard chain (independent of the library combination path): the installed
power ``L_Ws,inst = L_Ws,c - D_C`` (Formula 18b), the per-path normalised SPL
``L_n,s,ij = L_Ws,inst - D_sa - R_ij,ref - 10 lg(S_i/S0) - 10 lg(A0/4)``
(Formula 18a, ``S0 = A0 = 10 m2``) and the energetic path sum
``L_n,s = 10 lg(sum 10^(0.1 L_n,s,ij))`` (Formula 17). The tests recompute a
couple of path and total band values and assert they appear in the PDF, with
the prediction wording, the nominal band labels and the basis prose. Values are
read back via pypdf text extraction; structural facts (one page, rejected
engines/languages) complete the rendering contract.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from phonometry import ReportMetadata, installed_source_prediction

_PDF_MAGIC = b"%PDF"
_S0 = 10.0

_BANDS = np.array([63, 125, 250, 500, 1000, 2000], dtype=float)
# EN 12354-5:2009 Annex I.3 wall source element (Table I.8 characteristic
# power, Table I.9 force-source coupling term and flanking reduction indices).
_LWSC = np.array([84.4, 82.5, 69.9, 67.6, 61.6, 49.9])
_DC = 16.2
_DSA = np.array([-13.6, -17.3, -17.4, -20.0, -26.9, -32.9])
_R_WALL_FLOOR = np.array([43.0, 46.0, 50.2, 54.7, 64.6, 73.0])
_R_WALL_WALL = np.array([37.0, 41.2, 35.9, 37.7, 49.0, 57.8])
_S_WALL = 12.8


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


def _paths():
    return [
        {
            "adjustment_term": _DSA,
            "flanking_reduction_index": _R_WALL_FLOOR,
            "element_area": _S_WALL,
        },
        {
            "adjustment_term": _DSA,
            "flanking_reduction_index": _R_WALL_WALL,
            "element_area": _S_WALL,
        },
    ]


def _result():
    return installed_source_prediction(_LWSC, _DC, _paths(), frequencies=_BANDS)


def _oracle_path(rij: np.ndarray) -> np.ndarray:
    """Closed-form path L_n,s,ij (Formula 18a)."""
    lws_inst = _LWSC - _DC
    return (
        lws_inst
        - _DSA
        - rij
        - 10.0 * np.log10(_S_WALL / _S0)
        - 10.0 * np.log10(_S0 / 4.0)
    )


def _oracle_total() -> np.ndarray:
    """Energetic path sum L_n,s (Formula 17)."""
    stacked = np.vstack([_oracle_path(_R_WALL_FLOOR), _oracle_path(_R_WALL_WALL)])
    return 10.0 * np.log10(np.sum(10.0 ** (0.1 * stacked), axis=0))


def _oracle_overall() -> float:
    """Band-summed overall L_n,s."""
    total = _oracle_total()
    return float(10.0 * np.log10(np.sum(10.0 ** (0.1 * total))))


# --- clean-room oracle --------------------------------------------------------


def test_hand_oracle_matches_library() -> None:
    """The library path/total levels equal the closed-form EN 12354-5 values."""
    res = _result()
    np.testing.assert_allclose(
        res.path_levels[0], _oracle_path(_R_WALL_FLOOR), atol=1e-9
    )
    np.testing.assert_allclose(
        res.path_levels[1], _oracle_path(_R_WALL_WALL), atol=1e-9
    )
    np.testing.assert_allclose(res.total_level, _oracle_total(), atol=1e-9)
    assert res.overall_level == pytest.approx(_oracle_overall(), abs=1e-9)


def test_report_reads_as_prediction(tmp_path) -> None:
    """The sheet is explicitly a prediction, never a measurement."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "en12354_5.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Predicted" in text
    assert "not a measurement" in text
    assert "EN 12354-5:2009" in text
    # The footer disclaimer is the prediction variant (no tested specimen).
    assert "modelled configuration" in text
    assert "tested specimen" not in text.replace("not to a tested specimen", "")


def test_report_renders_oracle_values(tmp_path) -> None:
    """The verbose fiche prints the overall, total and per-path band values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "values.pdf"
    res.report(str(out), verbose=True)
    _assert_one_page(str(out))
    text = _extract_text(str(out))

    # Overall band-summed L_n,s to one decimal.
    assert f"{_oracle_overall():.1f}" in text
    # A couple of path band values (63 Hz) to one decimal.
    assert f"{_oracle_path(_R_WALL_FLOOR)[0]:.1f}" in text
    assert f"{_oracle_path(_R_WALL_WALL)[0]:.1f}" in text
    # A total band value (63 Hz).
    assert f"{_oracle_total()[0]:.1f}" in text
    # An installed power band value L_Ws,inst = L_Ws,c - D_C at 63 Hz.
    assert f"{_LWSC[0] - _DC:.1f}" in text
    # Nominal band labels and the by-path caption.
    assert "63" in text and "2000" in text
    assert "Octave-band normalised structure-borne SPL by path" in text
    assert "Transmission paths: 2" in text


def test_nonverbose_hides_path_columns(tmp_path) -> None:
    """verbose=False keeps only the installed power and the combined total."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "compact.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    # The total is still shown; the wall-wall path column value need not appear.
    assert f"{_oracle_total()[0]:.1f}" in text


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(45.0, "PASS"), (30.0, "FAIL")],
)
def test_verdict_against_declared_limit(tmp_path, limit: float, verdict: str) -> None:
    """A declared limit yields a PASS/FAIL verdict (lower is better)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / f"verdict_{limit}.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=limit))
    text = _extract_text(str(out))
    assert verdict in text
    assert "declared limit" in text


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the source, receiving room and identity."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    metadata = ReportMetadata(
        client="Example dwelling",
        specimen="WC flushing cistern",
        test_room="Adjacent bedroom",
        laboratory="Building acoustics office",
        report_id="IS-12354-5",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example dwelling" in text
    assert "WC flushing cistern" in text
    assert "Adjacent bedroom" in text
    assert "Building acoustics office" in text


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the prediction vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Sonido estructural instalado previsto" in text
    assert "no una medición" in text
    # Comma decimal separator on the overall level.
    assert f"{_oracle_overall():.1f}".replace(".", ",") in text


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


def test_scalar_source_level_fiche_renders(tmp_path) -> None:
    """A single-number L_Ws,c with per-band paths renders (regression).

    The table prints L_Ws,inst per band, so a scalar source level used to
    crash the fiche with an IndexError once the paths carried band axes.
    """
    result = installed_source_prediction(80.0, 10.0, _paths(), frequencies=_BANDS)
    assert result.installed_power_level.shape == _BANDS.shape
    out = tmp_path / "scalar.pdf"
    result.report(str(out), verbose=True)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "70.0" in text  # the broadcast installed power level
