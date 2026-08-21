#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO/TS 7849 sound-power-from-vibration ``.report()`` fiche.

The rendered values are checked against a clean-room oracle derived from the
standard, independent of the library's own combination path. A machine casing
of radiating area ``S`` is surveyed over octave bands; the surface-averaged
vibratory velocity level ``Lv`` (ISO/TS 7849-1:2009 Eq. 3) and the band-wise
radiation factor ``epsilon`` (Eq. 8) give the radiated band sound-power level
``LW = Lv + 10*lg(S/S0) + 10*lg(epsilon) + 10*lg(411/400)``, ``S0 = 1 m^2``
(Eq. 12/15), and the A-weighted total is
``LWA = 10*lg(sum 10^((LW + Ck)/10))`` with the standard octave A-weighting
corrections Ck. The tests recompute LW and LWA from these closed forms and
assert they appear in the PDF, along with the nominal band labels, the method
part and the basis prose. Values are read back via pypdf text extraction;
structural facts (one page, rejected engines/languages) complete the rendering
contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.emission import sound_power_from_vibration

# Standard octave-band A-weighting corrections Ck (dB), IEC 61672 / ISO 3744
# Annex E Table E.2, at the example band centres.
_CK_OCTAVE = {125: -16.1, 250: -8.6, 500: -3.2, 1000: 0.0, 2000: 1.2, 4000: 1.0}

_FREQS = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
# Surface-averaged vibratory velocity level Lv (dB re 5e-8 m/s) per octave band.
_LV = np.array([78.0, 82.0, 85.0, 83.0, 79.0, 74.0])
# Band-wise radiation factor epsilon (Eq. 8), dimensionless.
_EPS = np.array([0.20, 0.45, 0.75, 0.95, 1.00, 1.00])
_AREA = 1.6  # radiating area S, m^2

# The fixed terms of Eq. 12/15: 10 lg(S/S0) and the impedance term 10 lg(411/400).
_S_TERM = 10.0 * np.log10(_AREA / 1.0)
_IMP_TERM = 10.0 * np.log10(411.0 / 400.0)


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _engineering_result():
    """A Part 2 (engineering) determination with a measured radiation factor."""
    return sound_power_from_vibration(
        _LV, area=_AREA, radiation_factor=_EPS, frequencies=_FREQS
    )


def _survey_result():
    """A Part 1 (survey) determination with the fixed radiation factor eps = 1."""
    return sound_power_from_vibration(_LV, area=_AREA, frequencies=_FREQS)


def _oracle_lw(eps: np.ndarray) -> np.ndarray:
    """Closed-form band LW = Lv + 10 lg(S/S0) + 10 lg(eps) + 10 lg(411/400)."""
    return _LV + _S_TERM + 10.0 * np.log10(eps) + _IMP_TERM


def _oracle_lwa(eps: np.ndarray) -> float:
    """Closed-form LWA via the octave A-weighting corrections."""
    lw = _oracle_lw(eps)
    ck = np.array([_CK_OCTAVE[int(f)] for f in _FREQS])
    return float(10.0 * np.log10(np.sum(10.0 ** ((lw + ck) / 10.0))))


# --- clean-room oracle --------------------------------------------------------


def test_hand_oracle_matches_library() -> None:
    """The library LW/LWA equal the closed-form ISO/TS 7849 Eq. 3/8/12 values."""
    res = _engineering_result()
    np.testing.assert_allclose(res.sound_power_level, _oracle_lw(_EPS), atol=1e-9)
    assert res.sound_power_level_a == pytest.approx(_oracle_lwa(_EPS), abs=1e-9)


def test_report_renders_oracle_values(tmp_path) -> None:
    """The fiche prints the hand-derived LWA and a couple of band LW values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _engineering_result()
    out = tmp_path / "iso7849.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))

    # A-weighted total to one decimal, boxed with its reference power.
    lwa = f"{_oracle_lwa(_EPS):.1f}"
    assert lwa in text
    assert "re 1 pW" in text
    # Two band LW values (500 Hz and 1 kHz) to one decimal.
    lw = _oracle_lw(_EPS)
    assert f"{lw[2]:.1f}" in text
    assert f"{lw[3]:.1f}" in text
    # The surface velocity levels head the Lv column.
    assert f"{_LV[2]:.1f}" in text
    # Nominal band labels head the table / axis (not the exact base-ten centre).
    assert "125" in text
    assert "4000" in text
    # Method part and basis prose.
    assert "ISO/TS 7849-2:2009" in text
    assert "engineering method" in text
    assert "Octave-band sound power levels" in text
    # The radiating area S = 1.60 m2 is reported in the boxed result.
    assert "1.60" in text


def test_survey_part1_basis_and_method(tmp_path) -> None:
    """The survey method (eps = 1) names Part 1 and the upper-limit method."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _survey_result()
    out = tmp_path / "survey.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "ISO/TS 7849-1:2009" in text
    assert "survey method" in text
    assert "upper limit" in text
    # The survey LW is the Part 1 upper bound and equals eps = 1 closed form.
    lw = _oracle_lw(np.ones_like(_EPS))
    assert f"{lw[2]:.1f}" in text


def test_broadband_result_has_no_a_weighted_claim(tmp_path) -> None:
    """A broadband result (no frequencies) cannot be A-weighted, so no LWA."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    # A single directly measured broadband level, without band centres.
    res = sound_power_from_vibration(85.0, area=_AREA)
    assert np.isnan(res.sound_power_level_a)
    out = tmp_path / "broadband.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=80.0))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    # The unweighted total LW is boxed; nothing is labelled A-weighted.
    lw = float(85.0 + _S_TERM + _IMP_TERM)
    assert f"{lw:.1f}" in text
    assert "dB(A)" not in text
    assert "L WA" not in text
    assert "LWA" not in text.replace(" ", "")
    # The verdict compares the unweighted LW, not an invented A-weighted value.
    assert "declared limit" in text


def test_third_octave_labels_and_grouping(tmp_path) -> None:
    """A one-third-octave set is labelled by nominal centres and captioned."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800], dtype=float)
    lv = np.linspace(70.0, 85.0, freqs.size)
    res = sound_power_from_vibration(lv, area=_AREA, frequencies=freqs)
    out = tmp_path / "third.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "One-third-octave-band sound power levels" in text
    for label in ("100", "125", "160", "800"):
        assert label in text


# --- verbose (radiation-factor column) ----------------------------------------


def test_verbose_adds_radiation_factor_column(tmp_path) -> None:
    """verbose=True adds the radiation factor epsilon column."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _engineering_result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    # The 500 Hz radiation factor 0.750 appears to three decimals.
    assert "0.750" in text


# --- verdict against a declared limit -----------------------------------------


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(95.0, "PASS"), (80.0, "FAIL")],
)
def test_verdict_against_declared_limit(tmp_path, limit: float, verdict: str) -> None:
    """A declared limit yields a PASS/FAIL verdict (lower is better)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _engineering_result()
    out = tmp_path / f"verdict_{limit}.pdf"
    metadata = ReportMetadata(requirement=limit)
    res.report(str(out), metadata=metadata)
    text = _extract_text(str(out))
    assert verdict in text
    assert "declared limit" in text


# --- metadata header ----------------------------------------------------------


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the source, environment and identity fields."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _engineering_result()
    metadata = ReportMetadata(
        client="Example works",
        specimen="Gearbox casing",
        test_room="Machine hall",
        instrumentation="Class 1 accelerometer, s/n 7",
        laboratory="Acoustics lab",
        report_id="SP-7849",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Gearbox casing" in text
    assert "Machine hall" in text
    assert "Class 1 accelerometer, s/n 7" in text
    assert "Acoustics lab" in text


# --- Spanish fiche ------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the sound-power vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _engineering_result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Determinación de la potencia acústica" in text
    assert "método de ingeniería" in text
    assert "Área radiante" in text
    # Comma decimal separator on the A-weighted total.
    assert f"{_oracle_lwa(_EPS):.1f}".replace(".", ",") in text


# --- rendering contract -------------------------------------------------------


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _engineering_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _engineering_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
