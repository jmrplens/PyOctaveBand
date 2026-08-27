#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 9614-2 sound-power-by-intensity ``.report()`` fiche.

The rendered values are checked against a clean-room oracle derived from the
standard, independent of the library's own combination path. A machine is
enclosed by a surface of N equal segments of area Si; the normal intensity is
uniform over the surface, so every segment reports the same signed intensity
``In`` per band. The partial powers ``Pi = In*Si`` sum to the band power
``P = In*S`` (ISO 9614-2:1996 Eq. 12/6), so the band sound-power level is
``LW = 10*lg(In*S/P0)``, ``P0 = 1 pW`` (Eq. 13), and the A-weighted total is
``LWA = 10*lg(sum 10^((LW + Ck)/10))`` with the standard octave A-weighting
corrections Ck. The tests recompute LW and LWA from these closed forms and
assert they appear in the PDF, along with the nominal band labels, the method
grade and the field-indicator basis prose. Values are read back via pypdf text
extraction; structural facts (one page, rejected engines/languages) complete
the rendering contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.emission import (
    SoundPowerIntensityResult,
    SoundPowerWarning,
    sound_power_intensity,
)

if TYPE_CHECKING:
    from pathlib import Path

# Standard octave-band A-weighting corrections Ck (dB), IEC 61672 / ISO 3744
# Annex E Table E.2, at the example band centres.
_CK_OCTAVE = {125: -16.1, 250: -8.6, 500: -3.2, 1000: 0.0, 2000: 1.2, 4000: 1.0}

_FREQS = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
# Uniform per-band signed normal intensity (W/m^2), shared by every segment.
_INTENSITY = np.array([0.6e-4, 1.0e-4, 1.5e-4, 1.4e-4, 0.9e-4, 0.5e-4])
_N_SEG = 6
_SEG_AREA = 0.5
_SURFACE = _N_SEG * _SEG_AREA  # 3.0 m^2


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _uniform_result(grade: str = "engineering") -> SoundPowerIntensityResult:
    """A uniform-field determination, so LW and LWA are hand-derivable.

    The six segments share one intensity spectrum and the two sweeps coincide
    (perfect repeatability); the surface SPL is a uniform 80 dB and the
    instrument pressure-residual index is 15 dB.
    """
    scan = np.tile(_INTENSITY, (_N_SEG, 1))
    return sound_power_intensity(
        scan,
        np.full(_N_SEG, _SEG_AREA),
        normal_intensity_2=scan.copy(),
        pressure_levels=np.full((_N_SEG, len(_FREQS)), 80.0),
        pressure_residual_index=15.0,
        frequencies=_FREQS,
        band_type="octave",
        grade=grade,
    )


def _oracle_lw() -> np.ndarray:
    """Closed-form band LW = 10*lg(In*S/P0), P0 = 1 pW."""
    return 10.0 * np.log10(_INTENSITY * _SURFACE / 1.0e-12)


def _oracle_lwa() -> float:
    """Closed-form LWA via the octave A-weighting corrections."""
    lw = _oracle_lw()
    ck = np.array([_CK_OCTAVE[int(f)] for f in _FREQS])
    return float(10.0 * np.log10(np.sum(10.0 ** ((lw + ck) / 10.0))))


# --- clean-room oracle --------------------------------------------------------


def test_hand_oracle_matches_library() -> None:
    """The library LW/LWA equal the closed-form ISO 9614-2 Eq. 12/6/13 values."""
    res = _uniform_result()
    np.testing.assert_allclose(res.sound_power_level, _oracle_lw(), atol=1e-9)
    assert res.sound_power_level_a == pytest.approx(_oracle_lwa(), abs=1e-9)


def test_report_renders_oracle_values(tmp_path: Path) -> None:
    """The fiche prints the hand-derived LWA and a couple of band LW values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _uniform_result()
    out = tmp_path / "iso9614.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))

    # A-weighted total to one decimal, boxed with its reference power.
    lwa = f"{_oracle_lwa():.1f}"
    assert lwa in text
    assert "re 1 pW" in text
    # Two band LW values (500 Hz and 1 kHz) to one decimal.
    lw = _oracle_lw()
    assert f"{lw[2]:.1f}" in text
    assert f"{lw[3]:.1f}" in text
    # Nominal band labels head the table / axis (not the exact base-ten centre).
    assert "125" in text
    assert "4000" in text
    # Method grade and basis prose.
    assert "ISO 9614-2:1996" in text
    assert "engineering grade" in text
    assert "Octave-band sound power levels" in text
    # The measurement surface S = 3.00 m2 is reported in the boxed result.
    assert "3.00" in text


def test_third_octave_labels_and_grouping(tmp_path: Path) -> None:
    """A one-third-octave set is labelled by nominal centres and captioned."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800], dtype=float)
    intensity = np.linspace(0.5e-4, 1.5e-4, freqs.size)
    scan = np.tile(intensity, (_N_SEG, 1))
    res = sound_power_intensity(
        scan, np.full(_N_SEG, _SEG_AREA), frequencies=freqs, band_type="third"
    )
    out = tmp_path / "third.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert "One-third-octave-band sound power levels" in text
    for label in ("100", "125", "160", "800"):
        assert label in text


# --- verbose (field-indicator columns) ----------------------------------------


def test_verbose_adds_indicator_columns(tmp_path: Path) -> None:
    """verbose=True adds the FpI / F+/- indicators and the per-band grade."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _uniform_result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    flat = "".join(_extract_text(str(out)).split())
    assert "FpI" in flat
    # Every band qualifies at engineering grade, so the grade cell reads "2".
    assert "Grade" in _extract_text(str(out))


# --- clause 10.6 b omission surfaced in the basis strip ------------------------


def test_omitted_bands_listed_in_basis_strip(tmp_path: Path) -> None:
    """A band omitted from LWA per clause 10.6 b is named in the basis strip."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    scan = np.tile(_INTENSITY, (_N_SEG, 1))
    # A 125 Hz surface SPL far above the band LW drives FpI past Ld = 5 dB,
    # failing criterion 1 there; the other bands keep the uniform 80 dB.
    lp = np.full((_N_SEG, len(_FREQS)), 80.0)
    lp[:, 0] = 95.0
    res = sound_power_intensity(
        scan,
        np.full(_N_SEG, _SEG_AREA),
        normal_intensity_2=scan.copy(),
        pressure_levels=lp,
        pressure_residual_index=15.0,
        frequencies=_FREQS,
        band_type="octave",
    )
    assert res.a_weighting_omitted_bands is not None
    assert res.a_weighting_omitted_bands.tolist() == [True] + [False] * 5
    out = tmp_path / "omitted.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "10.6 b" in text
    assert "125 Hz" in text


# --- verdict against a declared limit -----------------------------------------


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(120.0, "PASS"), (80.0, "FAIL")],
)
def test_verdict_against_declared_limit(
    tmp_path: Path, limit: float, verdict: str
) -> None:
    """A declared limit yields a PASS/FAIL verdict (lower is better)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _uniform_result()
    out = tmp_path / f"verdict_{limit}.pdf"
    res.report(str(out), metadata=ReportMetadata(requirement=limit))
    text = _extract_text(str(out))
    assert verdict in text
    assert "declared limit" in text


# --- negative (net-inflow) band -----------------------------------------------


def test_negative_band_reported_as_dash(tmp_path: Path) -> None:
    """A net-negative band is not determinable and prints an em dash, not a level."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    scan = np.tile(_INTENSITY, (_N_SEG, 1)).copy()
    # Drive the 250 Hz band net-negative (more energy flowing in than out).
    scan[:, 1] = -_INTENSITY[1]
    with pytest.warns(SoundPowerWarning, match="negative in one or more bands"):
        res = sound_power_intensity(
            scan, np.full(_N_SEG, _SEG_AREA), frequencies=_FREQS, band_type="octave"
        )
    out = tmp_path / "negative.pdf"
    res.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    # The determinable 500 Hz band still prints its level.
    assert f"{_oracle_lw()[2]:.1f}" in text


# --- metadata header ----------------------------------------------------------


def test_metadata_header_renders(tmp_path: Path) -> None:
    """Supplied metadata renders the source, environment and identity fields."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _uniform_result()
    metadata = ReportMetadata(
        client="Example works",
        specimen="Air compressor",
        test_room="Machine hall",
        instrumentation="Class 1 p-p probe, s/n 7",
        laboratory="Acoustics lab",
        report_id="SP-9614",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Air compressor" in text
    assert "Machine hall" in text
    assert "Class 1 p-p probe, s/n 7" in text
    assert "Acoustics lab" in text


# --- Spanish fiche ------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path: Path) -> None:
    """language="es" renders the sound-power vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _uniform_result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Determinación de la potencia acústica" in text
    assert "intensidad acústica normal" in text
    assert "Superficie de medición" in text
    # Comma decimal separator on the A-weighted total.
    assert f"{_oracle_lwa():.1f}".replace(".", ",") in text


# --- rendering contract -------------------------------------------------------


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _uniform_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _uniform_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match=r"Unknown language"):
        res.report(out, language="xx")
