#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 15186-1 element-normalized intensity report (``.report()``).

The element-normalized level difference ``DI,n,e`` (Clause 3.9, Formula (8)) is
a level difference rated by the same ISO 717-1 machinery as the pressure ``R``,
so the fiche is anchored to a documented ``DI,n,e(f)`` spectrum: the
ISO 717-1:2020 Annex C worked-example curve (``Rw = 30 (-2; -3) dB``). Feeding
the receiving-side intensity levels ``LIn`` that make Formula (8) return that
curve (with a source-room level ``Lp1``, a measurement surface ``Sm`` and a
single element unit ``N``) reproduces the published curve exactly and pins the
fiche's rating to 30 (-2; -3) dB through the intensity path. Rendering
assertions are structural (a one-page ``%PDF``) plus pypdf text-extraction
checks of the boxed rating, the standard-basis line and a couple of per-band
values, like the sibling report tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

pytest.importorskip("reportlab")

from phonometry import ReportMetadata, building
from phonometry.building.measurement.intensity_insulation import (
    IntensityElementNormalizedResult,
)

_PDF_MAGIC = b"%PDF"

_A0 = 10.0


def _levels_for_target_dine(dine, lp1, sm, n):
    """Receiving-side LIn that make the corrected Formula (8) return ``dine``."""
    return (
        lp1
        - 6.0
        - 10.0 * np.log10(sm / _A0)
        + 10.0 * np.log10(float(n))
        - np.asarray(dine, dtype=np.float64)
    )


def _element_result() -> IntensityElementNormalizedResult:
    """A result whose DI,n,e equals the ISO 717-1 Annex C curve (30 dB)."""
    lp1, sm, n = 85.0, 12.0, 1
    l_in = _levels_for_target_dine(ref.ISO717_1_ANNEX_C_R, lp1, sm, n)
    return building.intensity_element_normalized_difference(
        np.full(16, lp1), l_in, measurement_area=sm, n=n
    )


def _octave_result() -> IntensityElementNormalizedResult:
    """A five-octave-band element result with a valid ISO 717-1 rating."""
    dine = np.array([30.0, 40.0, 48.0, 52.0, 55.0])
    l_in = _levels_for_target_dine(dine, 85.0, 12.0, 1)
    return building.intensity_element_normalized_difference(
        np.full(5, 85.0), l_in, measurement_area=12.0, n=1
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
    """The concatenated text of every page."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_dine_equals_annex_c_curve() -> None:
    """The synthesised levels reproduce the ISO 717-1 Annex C curve exactly."""
    result = _element_result()
    np.testing.assert_allclose(
        result.d_i_n_e, ref.ISO717_1_ANNEX_C_R, atol=1e-9
    )


def test_fiche_rating_pinned_to_iso717_1_annex_c(tmp_path) -> None:
    """The DI,n,e fiche's rating is the published 30 (-2; -3) dB."""
    out = tmp_path / "dine.pdf"
    _element_result().report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "30 (-2; -3) dB" in text
    assert "ISO 15186-1:2000" in text
    assert "ISO 717-1:2020" in text
    # Nominal band labels and a couple of DI,n,e values from the table.
    assert "100" in text
    assert "3150" in text
    assert "26.6" in text  # the 500 Hz band value
    assert "20.4" in text  # the 100 Hz band value


def test_full_metadata_and_verbose_render_one_page(tmp_path) -> None:
    """A fully populated verbose fiche is one page and shows the evaluation."""
    metadata = ReportMetadata(
        specimen="Trickle ventilator in a 100 mm masonry wall",
        client="Acoustic Test Client Ltd.",
        manufacturer="Example ventilators",
        area=0.02,
        test_room="Transmission suite (example)",
        mounting="Small-element mounting (ISO 10140-1 Annex F)",
        measurement_standard="ISO 15186-1",
        test_date="2026-07-21",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-0151",
        notes="Measurement surface Sm = 12 m2, N = 1 element unit.",
        requirement=25.0,
    )
    out = tmp_path / "verbose.pdf"
    _element_result().report(str(out), metadata=metadata, verbose=True)
    _assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "PASS" in text  # DI,n,e,w = 30 dB >= 25 dB
    assert "Trickle ventilator" in text
    assert "sound-intensity method" in text
    assert "ISO 717 evaluation" in text  # the verbose caption


def test_requirement_verdicts_pass_and_fail(tmp_path) -> None:
    """Element insulation passes at or above the requirement."""
    result = _element_result()  # DI,n,e,w = 30 dB
    failing = tmp_path / "fail.pdf"
    result.report(str(failing), metadata=ReportMetadata(requirement=50.0))
    assert "FAIL" in _extract_text(str(failing))

    passing = tmp_path / "pass.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=30.0))
    assert "PASS" in _extract_text(str(passing))


def test_octave_band_fiche_renders(tmp_path) -> None:
    """A five-octave-band element result also yields a one-page fiche."""
    result = _octave_result()
    assert result.rating is not None
    out = tmp_path / "octave.pdf"
    result.report(str(out))
    _assert_one_page(str(out))
    assert "Octave-band" in _extract_text(str(out))


def test_spanish_fiche_renders_translated(tmp_path) -> None:
    """``language="es"`` renders the Spanish fiche with comma decimals."""
    import re

    out = tmp_path / "es.pdf"
    _element_result().report(
        str(out),
        metadata=ReportMetadata(requirement=25.0, laboratory="Ejemplo"),
        language="es",
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "CUMPLE" in text
    assert "intensidad acústica" in " ".join(text.split())
    assert re.search(r"\d+,\d", text)  # comma decimal separator


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    out = str(tmp_path / "x.pdf")
    result = _element_result()
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unsupported language raises ``ValueError`` (shared validation path)."""
    out = str(tmp_path / "x.pdf")
    result = _element_result()
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="fr")


def test_missing_rating_rejected(tmp_path) -> None:
    """A result without the ISO 717 rating (non-core band count) is rejected."""
    l_in = _levels_for_target_dine(np.full(8, 40.0), 85.0, 12.0, 1)
    result = building.intensity_element_normalized_difference(
        np.full(8, 85.0), l_in, measurement_area=12.0, n=1
    )
    assert result.rating is None
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="rating"):
        result.report(out)


def test_non_iso_band_count_rejected(tmp_path) -> None:
    """A manually built 8-band result with matching rating arrays is rejected.

    The public API promises only the 16 one-third-octave or 5 octave bands
    ISO 717-1 rates; a hand-crafted rating whose per-band arrays all match an
    8-band curve would otherwise satisfy the shared renderer's shape checks.
    """

    centers = np.array([100, 125, 160, 200, 250, 315, 400, 500], dtype=float)
    curve = np.linspace(20.0, 40.0, 8)
    rating = building.WeightedRatingResult(
        rating=30, c=-2, ctr=-3, unfavourable_sum=0.0,
        band_centers=centers, measured=curve, shifted_reference=curve,
    )
    result = IntensityElementNormalizedResult(d_i_n_e=curve, rating=rating)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="16 one-third-octave"):
        result.report(out)


def test_rating_without_per_band_data_rejected(tmp_path) -> None:
    """A manually built rating lacking the per-band arrays is rejected."""

    bare_rating = building.WeightedRatingResult(
        rating=30, c=-2, ctr=-3, unfavourable_sum=0.0
    )
    result = IntensityElementNormalizedResult(
        d_i_n_e=np.asarray(ref.ISO717_1_ANNEX_C_R, dtype=np.float64),
        rating=bare_rating,
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="per-band"):
        result.report(out)
