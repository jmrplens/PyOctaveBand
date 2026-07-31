#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 16283-3 field facade report (``.report()`` -> PDF).

The fiche's numbers are verified against the standard's own definitions using
hand-computable syntheses: with ``T = T0 = 0,5 s`` the standardization term
vanishes exactly (``D2m,nT = D2m``), so feeding the ISO 717-1 Annex C
worked-example curve as the facade level difference pins the fiche's rating to
the published ``30 (-2; -3)`` value; ``R'45`` is checked against the element
method with the ``-1,5 dB`` loudspeaker correction. Rendering assertions are
structural (a one-page ``%PDF``) plus pypdf text-extraction checks of the boxed
rating, the standard-basis line and the mandatory field-method statement.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("reportlab")

from phonometry import (
    FacadeInsulationResult,
    ReportMetadata,
    facade_insulation,
)

_PDF_MAGIC = b"%PDF"

#: ISO 717-1 Annex C Table C.1 measured curve; rated 30 (-2; -3) dB.
_ANNEX_C_R = np.array(
    [20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
     28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5]
)

#: A receiving-room reverberation time equal to T0 = 0,5 s in every band, so
#: D2m,nT = D2m and the fiche's rating is hand-computable.
_T_AT_T0 = np.full(16, 0.5)


def _assert_one_page(path: str) -> None:
    """A written report is a non-empty single-page PDF."""
    import os

    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    """The concatenated, whitespace-normalised text of every page."""
    from pypdf import PdfReader

    return " ".join(
        "\n".join(page.extract_text() for page in PdfReader(path).pages).split()
    )


def _facade_dnt() -> FacadeInsulationResult:
    """A facade result whose D2m,nT equals the ISO 717-1 Annex C curve."""
    return facade_insulation(
        _ANNEX_C_R + 40.0, np.full(16, 40.0), _T_AT_T0, volume=62.5
    )


def _facade_r_prime() -> FacadeInsulationResult:
    """A facade result whose R'45 equals the Annex C curve (S = A, -1,5 dB)."""
    # A = 0,16 x 62,5 / 1 = 10 = S, so 10 lg(S/A) = 0 and R'45 = L1,s - L2 - 1,5.
    surf = _ANNEX_C_R + 1.5
    return facade_insulation(
        np.full(16, 50.0), np.full(16, 0.0), np.full(16, 1.0),
        area=10.0, volume=62.5, surface_level=surf,
    )


def test_dnt_reduces_to_d2m_at_reference_time() -> None:
    """With T = T0 the standardization term vanishes: D2m,nT = D2m exactly."""
    result = _facade_dnt()
    np.testing.assert_allclose(result.d_2m_nt, result.d_2m)
    np.testing.assert_allclose(result.d_2m, _ANNEX_C_R)


def test_dnt_fiche_rating_pinned_to_iso717_1_annex_c(tmp_path) -> None:
    """The D2m,nT fiche's rating is the published ISO 717-1 Annex C 30 (-2; -3)."""
    out = tmp_path / "d2mnt.pdf"
    _facade_dnt().report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "30 (-2; -3) dB" in text
    assert "ISO 16283-3:2016" in text
    assert "ISO 717-1:2020" in text
    assert "engineering method" in text
    assert "D2m,nT" in text


def test_r_prime_fiche(tmp_path) -> None:
    """``quantity='r_prime'`` reports the apparent sound reduction index R'45."""
    result = _facade_r_prime()
    assert result.r_prime is not None
    assert result.method == "loudspeaker"
    np.testing.assert_allclose(result.r_prime, _ANNEX_C_R, atol=1e-9)
    out = tmp_path / "rprime.pdf"
    result.report(str(out), quantity="r_prime")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Apparent sound reduction index" in text
    assert "loudspeaker method" in text
    assert "30 (-2; -3) dB" in text


def test_r_prime_road_traffic_fiche_labelled_rtrs(tmp_path) -> None:
    """A road-traffic result is labelled R'tr,s (Clause 3.13), never R'45."""
    # A = 0,16 x 62,5 / 1 = 10 = S, so 10 lg(S/A) = 0 and R'tr,s = L1,s - L2 - 3.
    surf = _ANNEX_C_R + 3.0
    result = facade_insulation(
        np.full(16, 50.0), np.full(16, 0.0), np.full(16, 1.0),
        area=10.0, volume=62.5, surface_level=surf, method="road_traffic",
    )
    assert result.method == "road_traffic"
    assert result.r_prime is not None
    np.testing.assert_allclose(result.r_prime, _ANNEX_C_R, atol=1e-9)
    out = tmp_path / "rtrs.pdf"
    result.report(str(out), quantity="r_prime")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "road traffic method" in text
    assert "tr,s" in text
    assert "loudspeaker" not in text  # the R'45 basis line is absent
    assert "30 (-2; -3) dB" in text


def test_d_2m_n_fiche(tmp_path) -> None:
    """``quantity='d_2m_n'`` reports the normalized facade level difference."""
    result = _facade_dnt()
    assert result.d_2m_n is not None
    out = tmp_path / "d2mn.pdf"
    result.report(str(out), quantity="d_2m_n")
    _assert_one_page(str(out))
    assert "Normalized facade level difference" in _extract_text(str(out))


def test_full_metadata_and_verbose_render_one_page(tmp_path) -> None:
    """A fully populated verbose facade fiche is one page and passes its target."""
    metadata = ReportMetadata(
        specimen="Dwelling facade, loudspeaker method",
        client="Acoustic Test Client Ltd.",
        receiving_volume=62.5,
        temperature=19.8,
        relative_humidity=55.0,
        test_room="Dwelling living room facing a main road",
        test_date="2026-07-22",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-0311",
        requirement=30.0,
    )
    out = tmp_path / "verbose.pdf"
    _facade_dnt().report(str(out), metadata=metadata, verbose=True)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "PASS" in text  # D2m,nT,w = 30 dB >= 30 dB (higher is better)
    assert "Unfav. dev." in text


def test_requirement_verdict_fail(tmp_path) -> None:
    """A facade level difference below the requirement fails."""
    out = tmp_path / "fail.pdf"
    _facade_dnt().report(str(out), metadata=ReportMetadata(requirement=45.0))
    assert "FAIL" in _extract_text(str(out))


def test_spanish_fiche(tmp_path) -> None:
    """``language='es'`` renders the Spanish facade fiche with comma decimals."""
    import re

    out = tmp_path / "es.pdf"
    _facade_dnt().report(
        str(out), metadata=ReportMetadata(requirement=30.0, laboratory="Ejemplo"),
        language="es",
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Aislamiento acústico de fachada in situ" in text
    assert "CUMPLE" in text
    assert re.search(r"\d+,\d", text)


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = _facade_dnt()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_quantity_rejected(tmp_path) -> None:
    """An unknown facade quantity raises ``ValueError``."""
    result = _facade_dnt()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="quantity"):
        result.report(out, quantity="dnt")


def test_missing_quantity_rejected(tmp_path) -> None:
    """Requesting d_2m_n / r_prime without their inputs raises ``ValueError``."""
    bare = facade_insulation(
        _ANNEX_C_R + 40.0, np.full(16, 40.0), _T_AT_T0
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="d_2m_n"):
        bare.report(out, quantity="d_2m_n")
    with pytest.raises(ValueError, match="r_prime"):
        bare.report(out, quantity="r_prime")


def test_non_core_band_count_rejected(tmp_path) -> None:
    """The facade field fiche needs the 16 core one-third-octave bands."""
    result = facade_insulation(
        np.full(21, 70.0), np.full(21, 30.0), np.full(21, 0.5)
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="16 core"):
        result.report(out)
