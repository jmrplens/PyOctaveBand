#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 3744 / ISO 3745 sound-power determination ``.report()`` fiche.

The rendered values are checked against a clean-room oracle derived from the
standard, independent of the library's own combination path. For a uniform
free-field surface-pressure spectrum measured over a hemisphere of radius r
(area S = 2*pi*r^2, ISO 3744:2010 clause 7.2.3), with the background noise a
uniform 10 dB below (so K1 = -10*lg(1 - 10^(-1,0)) dB, Eq. 16) and no room
absorption (K2 = 0), the band sound-power level is
LW = Lp - K1 + 10*lg(S/S0) (Eqs. 17-18) and the A-weighted total is
LWA = 10*lg(sum 10^((LW + Ck)/10)) with the Annex E octave corrections Ck
(Table E.2, Eq. E.1). The tests recompute LW and LWA from these closed forms
and assert they appear in the PDF, along with the nominal band labels and the
method/basis prose. Values are read back via pypdf text extraction; structural
facts (one page, rejected engines/languages) complete the rendering contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.emission import sound_power_anechoic, sound_power_pressure

# ISO 3744:2010 Annex E, Table E.2 octave-band A-weighting corrections Ck (dB).
_CK_OCTAVE = {
    63: -26.2,
    125: -16.1,
    250: -8.6,
    500: -3.2,
    1000: 0.0,
    2000: 1.2,
    4000: 1.0,
    8000: -1.1,
}

_FREQS = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
_SURFACE_LP = np.array([70.0, 74, 78, 80, 79, 76, 71, 64])


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _uniform_result(radius: float = 2.0):
    """A hemisphere determination with a uniform free-field surface spectrum.

    Ten identical position spectra (so the energy average equals ``_SURFACE_LP``
    exactly) with the background a uniform 10 dB below and no room absorption
    (K2 = 0). This makes LW and LWA hand-derivable.
    """
    positions = np.tile(_SURFACE_LP, (10, 1))
    return sound_power_pressure(
        positions,
        "hemisphere",
        radius=radius,
        reflecting_planes=1,
        background_levels=np.tile(_SURFACE_LP - 10.0, (10, 1)),
        frequencies=_FREQS,
        grade="engineering",
    )


def _oracle_lw(radius: float = 2.0) -> np.ndarray:
    """Closed-form band LW = Lp - K1 + 10*lg(S/S0) (K2 = 0)."""
    k1 = -10.0 * math.log10(1.0 - 10.0 ** (-1.0))
    area = 2.0 * math.pi * radius**2
    return _SURFACE_LP - k1 + 10.0 * math.log10(area)


def _oracle_lwa(radius: float = 2.0) -> float:
    """Closed-form LWA via the Annex E octave corrections (Eq. E.1)."""
    lw = _oracle_lw(radius)
    ck = np.array([_CK_OCTAVE[int(f)] for f in _FREQS])
    return float(10.0 * np.log10(np.sum(10.0 ** ((lw + ck) / 10.0))))


# --- clean-room oracle --------------------------------------------------------


def test_hand_oracle_matches_library() -> None:
    """The library LW/LWA equal the closed-form ISO 3744 Eq. 17/18/E.1 values."""
    res = _uniform_result()
    np.testing.assert_allclose(res.sound_power_level, _oracle_lw(), atol=1e-9)
    assert res.sound_power_level_a == pytest.approx(_oracle_lwa(), abs=1e-9)


def test_report_renders_oracle_values(tmp_path) -> None:
    """The fiche prints the hand-derived LWA and a couple of band LW values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _uniform_result()
    out = tmp_path / "iso3744.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))

    # A-weighted total to one decimal, boxed with its reference power.
    lwa = f"{_oracle_lwa():.1f}"
    assert lwa in text
    assert "re 1 pW" in text
    # Two band LW values (63 Hz and 500 Hz) to one decimal.
    lw = _oracle_lw()
    assert f"{lw[0]:.1f}" in text
    assert f"{lw[3]:.1f}" in text
    # Nominal band labels head the table / axis (not the exact base-ten centre).
    assert "63" in text
    assert "8000" in text
    # Method and basis prose.
    assert "ISO 3744:2010" in text
    assert "engineering method" in text
    assert "Octave-band sound power levels" in text


def test_third_octave_labels_and_grouping(tmp_path) -> None:
    """A one-third-octave set is labelled by nominal centres and captioned."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800], dtype=float)
    lp = np.linspace(70.0, 82.0, freqs.size)
    res = sound_power_pressure(
        np.tile(lp, (10, 1)),
        "hemisphere",
        radius=2.0,
        frequencies=freqs,
        grade="engineering",
    )
    out = tmp_path / "third.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert "One-third-octave-band sound power levels" in text
    for label in ("100", "125", "160", "800"):
        assert label in text


# --- verbose (K1/K2 columns) --------------------------------------------------


def test_verbose_adds_correction_columns(tmp_path) -> None:
    """verbose=True adds the mean level and the K1/K2 correction columns."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _uniform_result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    flat = "".join(_extract_text(str(out)).split())
    assert "K1" in flat
    assert "K2" in flat
    # K1 = -10 lg(1 - 10^-1) = 0.5 dB (one decimal), shown in the strip and column.
    assert "0.5" in _extract_text(str(out))


# --- verdict against a declared limit -----------------------------------------


@pytest.mark.parametrize(
    ("limit", "verdict"),
    [(120.0, "PASS"), (80.0, "FAIL")],
)
def test_verdict_against_declared_limit(tmp_path, limit: float, verdict: str) -> None:
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


# --- metadata header ----------------------------------------------------------


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the source, environment and identity fields."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = _uniform_result()
    metadata = ReportMetadata(
        client="Example works",
        specimen="Air compressor",
        test_room="Reverberation-free test hall",
        instrumentation="SLM X1, s/n 7",
        laboratory="Acoustics lab",
        report_id="SP-9",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example works" in text
    assert "Air compressor" in text
    assert "Reverberation-free test hall" in text
    assert "SLM X1, s/n 7" in text
    assert "Acoustics lab" in text


# --- ISO 3745 precision result ------------------------------------------------


def test_precision_report_names_iso3745(tmp_path) -> None:
    """A precision result renders the ISO 3745 basis and the C1/C2/C3 strip."""
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")
    res = sound_power_anechoic(
        np.tile(_SURFACE_LP, (20, 1)), "hemisphere", radius=2.0, frequencies=_FREQS
    )
    out = tmp_path / "precision.pdf"
    res.report(str(out), verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "ISO 3745:2012" in text
    assert "precision method" in text
    assert "C1 =" in text
    assert "C3 =" in text
    # The A-weighted total is boxed re 1 pW.
    assert "re 1 pW" in text
    # The A-weighting is combined per ISO 3745:2012 Annex C (Eq. C.1), not the
    # ISO 3744:2010 Annex E of the surface method: the precision fiche cites
    # neither ISO 3744 nor its Annex E anywhere.
    assert "Eq. C.1" in text
    assert "Annex E" not in text
    assert "ISO 3744" not in text


# --- Spanish fiche ------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
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
    assert "método de ingeniería" in text
    assert "Superficie de medición" in text
    # Comma decimal separator on the A-weighted total.
    assert f"{_oracle_lwa():.1f}".replace(".", ",") in text


# --- rendering contract -------------------------------------------------------


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _uniform_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _uniform_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")
