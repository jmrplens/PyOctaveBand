#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 17497-1/-2 scattering and diffusion reports (``.report()``).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written for each fiche (scattering ``s(f)``, the
diffusion spectrum ``d(f)`` and the single-source polar response), the displayed
coefficients match the documented clean-room oracle, the band/angle labels and
metadata appear, unknown engines/languages are rejected, XML specials in
metadata do not break reportlab, and the Spanish fiche uses a decimal comma.
Pixel or layout content is never inspected.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

import numpy as np

from phonometry import (
    ReportMetadata,
    materials,
)

_PDF_MAGIC = b"%PDF"

# The committed clean-room example: V = 200 m3, S = 10 m2, 20 degC (c = 343.2),
# m = 0, symmetrical base plate (T1 = T3). See scripts/generate_reports.py.
_FREQS = np.array(
    [100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
     1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000],
    dtype=float,
)
_T1 = np.array([8.0, 7.9, 7.8, 7.6, 7.4, 7.2, 7.0, 6.7, 6.4, 6.0,
                5.6, 5.2, 4.8, 4.4, 4.0, 3.6, 3.2, 2.9])
_ANGLES = np.arange(-90.0, 90.5, 10.0)


def _scattering():
    volume, area, c = 200.0, 10.0, 343.2
    t2 = _T1 * 0.90
    t4 = t2 * (1.0 - np.linspace(0.02, 0.28, _FREQS.size))
    alpha_s = materials.random_incidence_absorption(
        volume, area, c1=c, T1=_T1, c2=c, T2=t2
    )
    alpha_spec = materials.specular_absorption_coefficient(
        volume, area, c3=c, T3=_T1, c4=c, T4=t4
    )
    return materials.scattering_coefficient_spectrum(_FREQS, alpha_spec, alpha_s)


_SOURCES = np.array([0.0, 30.0, -30.0, 60.0, -60.0])


def _polar_energy(angles, width, peak, specular=0.0):
    return (
        10.0 * np.log10(1.0 + peak * np.exp(-(((angles - specular) / width) ** 2)))
        + 60.0
    )


def _diffusion_spectrum():
    """The committed random-incidence spectrum: source-averaged per band (8.4)."""
    widths = np.linspace(15.0, 70.0, _FREQS.size)
    peaks = np.linspace(30.0, 3.0, _FREQS.size)
    weights = np.array(materials.TWO_DIMENSIONAL_SOURCE_WEIGHTS, dtype=float)
    d = np.empty(_FREQS.size)
    d_n = np.empty(_FREQS.size)
    for k in range(_FREQS.size):
        d_theta, d_theta_n = [], []
        for source in _SOURCES:
            spec = -source
            d_s = materials.directional_diffusion_coefficient(
                _polar_energy(_ANGLES, widths[k], peaks[k], spec)
            )
            d_ref = materials.directional_diffusion_coefficient(
                _polar_energy(_ANGLES, 0.5 * widths[k], 60.0, spec)
            )
            d_theta.append(d_s)
            d_theta_n.append(
                float(materials.normalized_diffusion_coefficient(d_s, d_ref))
            )
        d[k] = materials.random_incidence_diffusion(d_theta, weights=weights)
        d_n[k] = materials.random_incidence_diffusion(d_theta_n, weights=weights)
    return materials.diffusion_spectrum(_FREQS, d, normalized=d_n)


def _polar():
    band = int(np.argmin(np.abs(_FREQS - 1000.0)))
    widths = np.linspace(15.0, 70.0, _FREQS.size)
    peaks = np.linspace(30.0, 3.0, _FREQS.size)
    return materials.directional_diffusion(
        _ANGLES, _polar_energy(_ANGLES, widths[band], peaks[band])
    )


def _metadata(**overrides) -> ReportMetadata:
    base = {
        "specimen": "Quadratic-residue diffuser (N = 7)",
        "client": "Acoustic Test Client Ltd.",
        "manufacturer": "Acoustics Works Inc.",
        "area": 10.0,
        "room_volume": 200.0,
        "mounting": "Circular sample on the rotating turntable",
        "test_room": "Reverberation room R1",
        "measurement_standard": "ISO 17497-1",
        "temperature": 20.0,
        "relative_humidity": 54.0,
        "pressure": 101.0,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "J. M. Requena-Plens",
        "report_id": "PHN-2026-17497",
    }
    base.update(overrides)
    return ReportMetadata(**base)


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert len(PdfReader(path).pages) == 1


def _text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(
        page.extract_text() for page in PdfReader(path).pages
    ).replace("\n", " ")


# --- ISO 17497-1 scattering ------------------------------------------------
def test_scattering_writes_one_page_pdf(tmp_path) -> None:
    out = tmp_path / "scat.pdf"
    returned = _scattering().report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_scattering_displayed_values_match_oracle(tmp_path) -> None:
    """The fiche prints the closed-form s (Eq. (5)) and the band labels."""
    out = tmp_path / "scat.pdf"
    _scattering().report(str(out), metadata=_metadata())
    text = _text(str(out))
    assert "0.08" in text  # s(500 Hz) = 0.082 -> 0.08
    assert "0.45" in text  # s(4000 Hz) = 0.454 -> 0.45
    assert "500" in text and "4000" in text  # band labels
    assert "Acoustic Test Client Ltd." in text  # metadata
    # The reverberation-room fields belong to the ISO 17497-1 scattering fiche.
    assert "Sample area" in text and "Room volume" in text


def test_scattering_verbose_shows_alpha_spec_one_page(tmp_path) -> None:
    out = tmp_path / "scat_v.pdf"
    _scattering().report(str(out), metadata=_metadata(), verbose=True)
    _assert_one_page(str(out))
    assert "0.13" in _text(str(out))  # alpha_spec(500 Hz) = 0.131 -> 0.13


def test_scattering_unknown_engine_rejected(tmp_path) -> None:
    result = _scattering()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_scattering_unknown_language_rejected(tmp_path) -> None:
    result = _scattering()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="Unknown language"):
        result.report(out, language="xx")


def test_scattering_metadata_xml_specials_do_not_break(tmp_path) -> None:
    out = tmp_path / "scat_xml.pdf"
    _scattering().report(
        str(out), metadata=_metadata(specimen='Panel <A> & <B> "edge"')
    )
    _assert_one_page(str(out))


def test_scattering_no_metadata_still_renders(tmp_path) -> None:
    out = tmp_path / "scat_bare.pdf"
    _scattering().report(str(out))
    _assert_one_page(str(out))
    assert "100 to 5000" in _text(str(out))  # measured frequency range


def test_scattering_spanish_uses_comma_decimal(tmp_path) -> None:
    out = tmp_path / "scat_es.pdf"
    _scattering().report(str(out), metadata=_metadata(), language="es")
    _assert_one_page(str(out))
    assert "0,45" in _text(str(out))  # Spanish decimal comma


# --- ISO 17497-2 diffusion spectrum ---------------------------------------
def test_diffusion_writes_one_page_pdf(tmp_path) -> None:
    out = tmp_path / "diff.pdf"
    _diffusion_spectrum().report(str(out), metadata=_metadata())
    _assert_one_page(str(out))


def test_diffusion_displayed_values_match_oracle(tmp_path) -> None:
    """The fiche prints the per-band random-incidence d (Clause 8.4)."""
    out = tmp_path / "diff.pdf"
    _diffusion_spectrum().report(str(out), metadata=_metadata())
    text = _text(str(out))
    assert "0.51" in text  # d(500 Hz) = 0.506 -> 0.51
    assert "0.81" in text  # d(4000 Hz) = 0.807 -> 0.81
    assert "500" in text and "4000" in text  # band labels


def test_diffusion_omits_room_fields(tmp_path) -> None:
    """The free-field diffusion fiche must not show sample area / room volume."""
    out = tmp_path / "diff.pdf"
    _diffusion_spectrum().report(str(out), metadata=_metadata())
    text = _text(str(out))
    assert "Sample area" not in text
    assert "Room volume" not in text


def test_diffusion_verbose_shows_normalized_one_page(tmp_path) -> None:
    out = tmp_path / "diff_v.pdf"
    _diffusion_spectrum().report(str(out), metadata=_metadata(), verbose=True)
    _assert_one_page(str(out))
    assert "0.35" in _text(str(out))  # d_n(500 Hz) = 0.347 -> 0.35


def test_diffusion_unknown_engine_rejected(tmp_path) -> None:
    result = _diffusion_spectrum()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_diffusion_spanish_uses_comma_decimal(tmp_path) -> None:
    out = tmp_path / "diff_es.pdf"
    _diffusion_spectrum().report(str(out), metadata=_metadata(), language="es")
    _assert_one_page(str(out))
    assert "0,81" in _text(str(out))


# --- ISO 17497-2 polar response -------------------------------------------
def test_polar_writes_one_page_pdf(tmp_path) -> None:
    out = tmp_path / "polar.pdf"
    _polar().report(str(out), metadata=_metadata())
    _assert_one_page(str(out))


def test_polar_displays_coefficient_and_angles(tmp_path) -> None:
    out = tmp_path / "polar.pdf"
    _polar().report(str(out), metadata=_metadata())
    text = _text(str(out))
    assert "0.67" in text  # boxed directional diffusion coefficient d
    assert "90.0" in text  # receiver angle label


def test_polar_unknown_engine_rejected(tmp_path) -> None:
    result = _polar()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_polar_spanish_uses_comma_decimal(tmp_path) -> None:
    out = tmp_path / "polar_es.pdf"
    _polar().report(str(out), metadata=_metadata(), language="es")
    _assert_one_page(str(out))
    assert "0,67" in _text(str(out))
