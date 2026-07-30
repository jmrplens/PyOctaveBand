#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Tests for the ISO 12354-1/-2:2017 detailed prediction fiches (``.report()``).

Both fiches are pinned to the per-band worked example the two parts share
(ISO 12354-1 Annex L / ISO 12354-2 Annex G, the heavy homogeneous building
built by :mod:`iso12354_building`), run through the tested detailed-model code:
predicted ``R'w`` = 57 dB and ``L'n,w`` = 41 dB. The rendering assertions are
structural (a one-page ``%PDF``) plus pypdf text-extraction checks of the boxed
single number, the path table, the mandatory "prediction / not a measurement"
wording and the detailed model's own accuracy statement.
"""

from __future__ import annotations

import iso12354_building as bld
import numpy as np
import pytest
import reference_data as ref

pytest.importorskip("reportlab")

from phonometry import (
    ReportMetadata,
    detailed_airborne_prediction,
    detailed_impact_prediction,
    direct_impact_level,
    direct_reduction_index,
    floating_floor_improvement,
    in_situ_element,
)

_PDF_MAGIC = b"%PDF"
_BANDS = np.asarray(ref.ISO12354_ANNEX_L_BANDS, dtype=np.float64)


def _situ() -> tuple[dict, np.ndarray]:
    """The Annex L building in situ, plus the floating floor's improvement."""
    situ = {k: in_situ_element(e, _BANDS) for k, e in bld.elements().items()}
    delta = floating_floor_improvement(
        _BANDS, resonance_frequency=bld.floating_floor_resonance()
    )
    return situ, delta


def _annex_l_airborne():
    """The Annex L detailed airborne prediction (R'w = 57 dB)."""
    situ, delta = _situ()
    return detailed_airborne_prediction(
        _BANDS,
        direct_index=direct_reduction_index(
            situ["floor"].sound_reduction_index, delta_r_source=delta),
        flanking_paths=bld.airborne_paths(situ, delta),
    )


def _annex_g_impact():
    """The Annex G detailed impact prediction (L'n,w = 41 dB)."""
    situ, delta = _situ()
    return detailed_impact_prediction(
        _BANDS,
        direct_level=direct_impact_level(situ["floor"].impact_level, delta_l=delta),
        flanking_paths=bld.impact_paths(situ, delta),
    )


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


def test_detailed_airborne_fiche_boxes_the_annex_l_rating(tmp_path) -> None:
    """The detailed airborne fiche boxes R'w = 57 dB and names Clause 4.2."""
    out = tmp_path / "air.pdf"
    assert _annex_l_airborne().report(str(out)) == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{ref.ISO12354_ANNEX_L1_R_PRIME_W} dB" in text
    assert "ISO 12354-1:2017" in text
    assert "ISO 717-1" in text
    assert "frequency band" in text
    assert "prediction" in text
    assert "not a measurement" in text
    # The detailed model's own accuracy statement, with the decimal separator
    # of the fiche language (a period in English, a comma in Spanish).
    assert "1.5 dB to 2.5 dB" in text
    # The two paths that dominate the spectrum appear in the path table.
    assert "Dd" in text
    assert "2d" in text


def test_detailed_impact_fiche_boxes_the_annex_g_rating(tmp_path) -> None:
    """The detailed impact fiche boxes L'n,w = 41 dB and verdicts a limit."""
    out = tmp_path / "imp.pdf"
    result = _annex_g_impact()
    assert result.report(
        str(out),
        metadata=ReportMetadata(
            specimen="220 mm concrete floor with floating screed",
            area=20.0, receiving_volume=55.0, requirement=50.0,
        ),
        verbose=True,
    ) == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{ref.ISO12354_ANNEX_G1_L_PRIME_N_W} dB" in text
    assert "ISO 12354-2:2017" in text
    assert "ISO 717-2" in text
    assert "PASS" in text          # 41 dB against the 50 dB requirement
    assert "not a measurement" in text
    assert "Df1" in text


def test_spanish_detailed_fiche_renders(tmp_path) -> None:
    """The fiches translate to Spanish like every other report."""
    out = tmp_path / "air_es.pdf"
    _annex_l_airborne().report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "ISO 12354-1:2017" in text
    assert "57 dB" in text
    # The basis line, the path caption and the accuracy statement are
    # translated, not left in English.
    assert "banda de frecuencia" in text
    assert "Vías de transmisión" in text
    assert "1,5 dB a 2,5 dB" in text
    assert "not a measurement" not in text


def test_detailed_fiche_rejects_an_unrated_spectrum(tmp_path) -> None:
    """Without the ISO 717 band range there is no single number to box."""
    partial = detailed_airborne_prediction(
        _BANDS[:5], direct_index=np.full(5, 55.0)
    )
    assert partial.rating is None
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="ISO 717"):
        partial.report(out)


def test_detailed_fiche_rejects_an_unknown_engine(tmp_path) -> None:
    """Only the reportlab back end exists."""
    result = _annex_g_impact()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")
