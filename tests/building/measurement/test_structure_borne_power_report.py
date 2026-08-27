#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the EN 15657 structure-borne sound power ``.report()`` fiche.

The rendered values are checked against a clean-room oracle derived from the
standard, independent of the library's own combination path. A source on a
reception plate of mass per area ``m`` and area ``S`` with structural
reverberation time ``Ts`` gives the plate loss factor ``eta = 2,2/(f*Ts)``
(Formula 13) and the injected structure-borne sound power level
``L_Ws = 10*lg(2*pi*f*eta*m*S) + Lv - 60`` dB re 1 pW (Formula 14). The tests
recompute ``L_Ws`` and the band-summed total from these closed forms and assert
they appear in the PDF, along with the nominal band labels, the method basis
prose and the plate-specific note. Values are read back via pypdf text
extraction; structural facts (one page, rejected engines/languages) complete
the rendering contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata, building

if TYPE_CHECKING:
    from pathlib import Path

_FREQS = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
# Spatial mean plate velocity level Lv (dB re 1e-9 m/s) per octave band.
_LV = np.array([88.0, 90.0, 86.0, 82.0, 78.0, 73.0])
_MASS = 25.0  # plate mass per area m, kg/m^2
_AREA = 1.2  # reception-plate area S, m^2
_TS = 0.3  # structural reverberation time Ts, s


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _result() -> building.StructureBornePowerResult:
    return building.reception_plate_power(
        _LV, _FREQS, mass_per_area=_MASS, area=_AREA, reverberation_time=_TS
    )


def _oracle_eta() -> np.ndarray:
    """Closed-form plate loss factor eta = 2,2/(f*Ts) (Formula 13)."""
    return 2.2 / (_FREQS * _TS)


def _oracle_lws() -> np.ndarray:
    """Closed-form band L_Ws = 10 lg(2 pi f eta m S) + Lv - 60 dB (Formula 14)."""
    eta = _oracle_eta()
    return 10.0 * np.log10(2.0 * np.pi * _FREQS * eta * _MASS * _AREA) + _LV - 60.0


def _oracle_total() -> float:
    """Band-summed total L_Ws, 10 lg(sum 10^(0.1 L_Ws))."""
    lws = _oracle_lws()
    return float(10.0 * np.log10(np.sum(10.0 ** (0.1 * lws))))


# --- clean-room oracle --------------------------------------------------------


def test_hand_oracle_matches_library() -> None:
    """The library L_Ws and total equal the closed-form EN 15657 values."""
    res = _result()
    np.testing.assert_allclose(res.power_level, _oracle_lws(), atol=1e-9)
    assert res.total_level == pytest.approx(_oracle_total(), abs=1e-9)


def test_report_renders_oracle_values(tmp_path: Path) -> None:
    """The fiche prints the total L_Ws and a couple of band L_Ws values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "en15657.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))

    # Band-summed total to one decimal, boxed with its reference power.
    assert f"{_oracle_total():.1f}" in text
    assert "re 1 pW" in text
    # Two band L_Ws values (250 Hz and 1 kHz) to one decimal.
    lws = _oracle_lws()
    assert f"{lws[1]:.1f}" in text
    assert f"{lws[3]:.1f}" in text
    # The plate velocity levels head the Lv column.
    assert f"{_LV[1]:.1f}" in text
    # Nominal band labels head the table / axis.
    assert "125" in text
    assert "4000" in text
    # Basis prose and the plate parameters in the boxed result.
    assert "EN 15657:2018" in text
    assert "reception-plate method" in text
    assert "Octave-band structure-borne sound power levels" in text
    assert "25.0" in text  # plate mass per area m
    assert "1.20" in text  # reception-plate area S
    # The plate-specific caveat is stated (Formula 14 is not a source descriptor).
    assert "Plate-specific level" in text


def test_verbose_adds_loss_factor_column(tmp_path: Path) -> None:
    """verbose=True adds the plate loss factor eta column."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    # The 1 kHz loss factor eta = 2,2/(1000*0,3) = 0,0073 (four decimals).
    eta = _oracle_eta()
    assert f"{eta[3]:.4f}" in text


def test_third_octave_labels_and_caption(tmp_path: Path) -> None:
    """A one-third-octave set is labelled by nominal centres and captioned."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800], dtype=float)
    lv = np.linspace(80.0, 92.0, freqs.size)
    res = building.reception_plate_power(
        lv, freqs, mass_per_area=_MASS, area=_AREA, loss_factor=0.01
    )
    out = tmp_path / "third.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "One-third-octave-band structure-borne sound power levels" in text
    for label in ("100", "125", "160", "800"):
        assert label in text


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(100.0, "PASS"), (40.0, "FAIL")],
)
def test_verdict_against_declared_limit(
    tmp_path: Path, limit: float, verdict: str
) -> None:
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


def test_metadata_header_renders(tmp_path: Path) -> None:
    """Supplied metadata renders the source, environment and identity fields."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    metadata = ReportMetadata(
        client="Example works",
        specimen="Circulation pump",
        test_room="Reception-plate rig",
        instrumentation="Accelerometer array, s/n 7",
        laboratory="Acoustics lab",
        report_id="SB-15657",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Circulation pump" in text
    assert "Reception-plate rig" in text
    assert "Accelerometer array, s/n 7" in text
    assert "Acoustics lab" in text


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """language="es" renders the structure-borne vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Determinación de la potencia acústica estructural" in text
    assert "método de la placa de recepción" in text
    # Comma decimal separator on the total.
    assert f"{_oracle_total():.1f}".replace(".", ",") in text


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match=r"Unknown language"):
        res.report(out, language="xx")
