#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 3741 reverberation-room sound-power ``.report()`` fiche.

The rendered values are checked against a clean-room oracle derived from the
standard, independent of the library's own computation path. A steady source
is measured by the direct method in a reverberation room of volume ``V`` and
surface ``S`` with a uniform reverberation time ``T60``; the band sound-power
level follows ISO 3741:2010 Eq. (20),

    LW = Lp + 10*lg(A/A0) + 4,34*(A/S) + 10*lg(1 + S*c/(8*V*f)) + C1 + C2 - 6,

with the Sabine absorption area ``A = (55,26/c)*(V/T60)``, the speed of sound
``c = 20,05*sqrt(273 + theta)`` and the meteorological corrections ``C1``/``C2``
(clause 9.1.4). The A-weighted total is
``LWA = 10*lg(sum 10^((LW + Ck)/10))`` with the standard octave A-weighting
corrections ``Ck`` (Annex F Eq. F.2). The tests recompute LW and LWA from these
closed forms and assert they appear in the PDF, along with the nominal band
labels, the precision-grade statement and the basis prose. Values are read back
via pypdf text extraction; structural facts (one page, rejected engines and
languages) complete the rendering contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.emission import (
    sound_power_comparison,
    sound_power_reverberation,
)

if TYPE_CHECKING:
    from pathlib import Path

    from phonometry.emission import ReverberationSoundPowerResult

# Standard octave-band A-weighting corrections Ck (dB), IEC 61672 / ISO 3744
# Annex E Table E.2 (reused by ISO 3741 Annex F), at the example band centres.
_CK_OCTAVE = {
    125: -16.1,
    250: -8.6,
    500: -3.2,
    1000: 0.0,
    2000: 1.2,
    4000: 1.0,
    8000: -1.1,
}

_FREQS = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
# Documented mean corrected room level Lp(ST) per octave band (Eq. 16), in dB.
_LP = np.array([80.0, 83.0, 85.0, 84.0, 80.0, 75.0, 68.0])
_VOLUME = 200.0
_SURFACE = 240.0
_T60 = 2.0
_THETA = 20.0
_PS = 101.325


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _result() -> ReverberationSoundPowerResult:
    """The direct-method determination whose LW and LWA are hand-derivable."""
    return sound_power_reverberation(
        _LP,
        _T60,
        volume=_VOLUME,
        surface_area=_SURFACE,
        frequencies=_FREQS,
        temperature=_THETA,
        static_pressure=_PS,
    )


def _oracle_lw() -> np.ndarray:
    """Closed-form band LW from ISO 3741:2010 Eq. (20), clause 9.1.4."""
    c = 20.05 * np.sqrt(273.0 + _THETA)
    area = (55.26 / c) * (_VOLUME / _T60)
    waterhouse = 10.0 * np.log10(1.0 + _SURFACE * c / (8.0 * _VOLUME * _FREQS))
    c1 = -10.0 * np.log10(_PS / 101.325) + 5.0 * np.log10((273.15 + _THETA) / 314.0)
    c2 = -10.0 * np.log10(_PS / 101.325) + 15.0 * np.log10((273.15 + _THETA) / 296.0)
    return (
        _LP
        + 10.0 * np.log10(area / 1.0)
        + 4.34 * (area / _SURFACE)
        + waterhouse
        + c1
        + c2
        - 6.0
    )


def _oracle_lwa() -> float:
    """Closed-form LWA via the octave A-weighting corrections (Annex F Eq. F.2)."""
    lw = _oracle_lw()
    ck = np.array([_CK_OCTAVE[int(f)] for f in _FREQS])
    return float(10.0 * np.log10(np.sum(10.0 ** ((lw + ck) / 10.0))))


# --- clean-room oracle --------------------------------------------------------


def test_hand_oracle_matches_library() -> None:
    """The library LW/LWA equal the closed-form ISO 3741 Eq. 20 / Annex F values."""
    res = _result()
    np.testing.assert_allclose(res.sound_power_level, _oracle_lw(), atol=1e-9)
    assert res.sound_power_level_a == pytest.approx(_oracle_lwa(), abs=1e-9)


def test_report_renders_oracle_values(tmp_path: Path) -> None:
    """The fiche prints the hand-derived LWA and a couple of band LW values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "iso3741.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))

    # A-weighted total to one decimal, boxed with its reference power.
    assert f"{_oracle_lwa():.1f}" in text
    assert "re 1 pW" in text
    # Two band LW values (250 Hz and 500 Hz) to one decimal.
    lw = _oracle_lw()
    assert f"{lw[1]:.1f}" in text
    assert f"{lw[2]:.1f}" in text
    # Nominal band labels head the table / axis (not the exact base-ten centre).
    assert "125" in text
    assert "8000" in text
    # Standard, precision grade and method prose.
    assert "ISO 3741:2010" in text
    assert "precision method, accuracy grade 1" in text
    assert "direct method" in text
    assert "Octave-band sound power levels" in text
    # The correction model and A-weighting basis prose.
    assert "Eq. 20" in text
    assert "Annex F" in text


def test_third_octave_labels_and_grouping(tmp_path: Path) -> None:
    """A one-third-octave set is labelled by nominal centres and captioned."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800], dtype=float)
    lp = np.linspace(82.0, 74.0, freqs.size)
    res = sound_power_reverberation(
        lp,
        2.0,
        volume=200.0,
        surface_area=240.0,
        frequencies=freqs,
    )
    out = tmp_path / "third.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert "One-third-octave-band sound power levels" in text
    for label in ("100", "125", "160", "800"):
        assert label in text


# --- verbose (reverberation-specific columns) ---------------------------------


def test_verbose_adds_absorption_and_waterhouse_columns(tmp_path: Path) -> None:
    """verbose=True adds the K1, absorption area A and Waterhouse Cw columns."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    flat = "".join(_extract_text(str(out)).split())
    assert "Cw" in flat
    text = _extract_text(str(out))
    # The constant Sabine absorption area A = 16.1 m2 appears as a column value.
    assert "16.1" in text


# --- comparison method --------------------------------------------------------


def test_comparison_method_reports_eq21(tmp_path: Path) -> None:
    """The comparison-method fiche cites Eq. 21 and the reference sound source."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    lp_rss = _LP - 3.0
    lw_rss = np.full(_FREQS.size, 85.0)
    res = sound_power_comparison(
        _LP,
        lp_rss,
        lw_rss,
        frequencies=_FREQS,
        temperature=_THETA,
        static_pressure=_PS,
    )
    out = tmp_path / "comparison.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "comparison method" in text
    assert "Eq. 21" in text
    assert "reference sound source" in text
    # Closed-form Eq. 21: LW = LW(RSS) + (Lp(ST) - Lp(RSS) + C2).
    c2 = -10.0 * np.log10(_PS / 101.325) + 15.0 * np.log10((273.15 + _THETA) / 296.0)
    lw = lw_rss + (_LP - lp_rss + c2)
    assert f"{lw[2]:.1f}" in text


# --- verdict against a declared limit -----------------------------------------


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(110.0, "PASS"), (80.0, "FAIL")],
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


# --- metadata header ----------------------------------------------------------


def test_metadata_header_renders(tmp_path: Path) -> None:
    """Supplied metadata renders the source, environment and identity fields."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    metadata = ReportMetadata(
        client="Example works",
        specimen="Air compressor",
        test_room="Reverberation room, V = 200 m3",
        instrumentation="Class 1 sound level meter, s/n 7",
        laboratory="Acoustics lab",
        report_id="SP-3741",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Air compressor" in text
    assert "Reverberation room, V = 200 m3" in text
    assert "Class 1 sound level meter, s/n 7" in text
    assert "Acoustics lab" in text


# --- Spanish fiche ------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """language="es" renders the sound-power vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Determinación de la potencia acústica" in text
    assert "cámara reverberante cualificada" in text
    assert "método directo" in text
    # Comma decimal separator on the A-weighted total.
    assert f"{_oracle_lwa():.1f}".replace(".", ",") in text


# --- rendering contract -------------------------------------------------------


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
