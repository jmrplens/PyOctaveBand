#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 15186-1 intensity sound-insulation report (``.report()``).

The intensity sound reduction index ``RI`` is an ordinary sound reduction
index rated by the same ISO 717-1 machinery as the pressure ``R``, so the
fiche is anchored to a documented ``RI(f)`` spectrum: the ISO 717-1:2020
Annex C worked-example reduction index (``Rw = 30 (-2; -3) dB``). Feeding the
receiving-side intensity levels ``LIn`` that make Formula (7) return that curve
(with a source-room level ``Lp1``, a measurement surface ``Sm`` and a specimen
``S``) reproduces the published curve exactly and pins the fiche's rating to
30 (-2; -3) dB through the intensity path. Rendering assertions are structural
(a one-page ``%PDF``) plus pypdf text-extraction checks of the boxed rating,
the standard-basis line and a couple of per-band values, like the sibling
report tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import reference_data as ref

pytest.importorskip("reportlab")

from report_assertions import assert_one_page

from phonometry import ReportMetadata, building
from phonometry.building.measurement.intensity_insulation import (
    IntensityReductionResult,
)

if TYPE_CHECKING:
    from pathlib import Path

_RATING_FREQS = np.array(
    [
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
    ],
    dtype=float,
)


def _levels_for_target_ri(
    ri: list[float] | np.ndarray, lp1: float, sm: float, s: float
) -> np.ndarray:
    """Receiving-side LIn that make Formula (7) return exactly ``ri``."""
    return lp1 - 6.0 - 10.0 * np.log10(sm / s) - np.asarray(ri, dtype=np.float64)


def _intensity_result(*, with_kc: bool = False) -> IntensityReductionResult:
    """A result whose RI equals the ISO 717-1 Annex C curve (RI,w = 30 dB)."""
    lp1 = ref.ISO15186_1_REF_LP1
    sm = ref.ISO15186_1_REF_SM
    s = ref.ISO15186_1_REF_S
    l_in = _levels_for_target_ri(ref.ISO15186_1_REF_RI, lp1, sm, s)
    kc = building.adaptation_term_kc(_RATING_FREQS) if with_kc else None
    return building.intensity_sound_reduction(
        np.full(16, lp1), l_in, measurement_area=sm, area=s, kc=kc
    )


def _extract_text(path: str) -> str:
    """The concatenated text of every page."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_ri_equals_annex_c_curve() -> None:
    """The synthesised levels reproduce the ISO 717-1 Annex C curve exactly."""
    result = _intensity_result()
    np.testing.assert_allclose(result.r_i, ref.ISO15186_1_REF_RI, atol=1e-9)


def test_fiche_rating_pinned_to_iso717_1_annex_c(tmp_path: Path) -> None:
    """The RI fiche's rating is the published ISO 717-1 Annex C 30 (-2; -3) dB."""
    out = tmp_path / "ri.pdf"
    _intensity_result().report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{ref.ISO15186_1_REF_RIW} (-2; -3) dB" in text
    assert "ISO 15186-1:2000" in text
    assert "ISO 717-1:2020" in text
    # Nominal band labels and a couple of RI values from the table.
    assert "100" in text
    assert "3150" in text
    assert "26.6" in text  # the 500 Hz band value
    assert "20.4" in text  # the 100 Hz band value


def test_full_metadata_and_verbose_render_one_page(tmp_path: Path) -> None:
    """A fully populated fiche with the RI,M column is one page."""
    metadata = ReportMetadata(
        specimen="100 mm autoclaved aerated concrete block wall",
        client="Acoustic Test Client Ltd.",
        manufacturer="Example blockworks",
        area=10.0,
        mass_per_area=75.0,
        source_volume=53.0,
        receiving_volume=51.0,
        receiving_temperature=20.8,
        receiving_relative_humidity=46.0,
        pressure=101.3,
        test_room="Transmission suite (example)",
        mounting="Type A mounting (ISO 10140-1)",
        measurement_standard="ISO 15186-1",
        test_date="2026-07-21",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-0150",
        requirement=25.0,
    )
    out = tmp_path / "verbose.pdf"
    _intensity_result(with_kc=True).report(str(out), metadata=metadata, verbose=True)
    assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "PASS" in text  # RI,w = 30 dB >= 25 dB
    assert "100 mm autoclaved aerated concrete" in text
    assert "sound-intensity method" in text
    assert "Kc-modified" in text  # the verbose column caption


def test_requirement_verdicts_pass_and_fail(tmp_path: Path) -> None:
    """Intensity insulation passes at or above the requirement."""
    result = _intensity_result()  # RI,w = 30 dB
    failing = tmp_path / "fail.pdf"
    result.report(str(failing), metadata=ReportMetadata(requirement=50.0))
    assert "FAIL" in _extract_text(str(failing))

    passing = tmp_path / "pass.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=30.0))
    assert "PASS" in _extract_text(str(passing))


def _octave_result(*, with_kc: bool = False) -> IntensityReductionResult:
    """A five-octave-band intensity result with a valid ISO 717-1 rating."""
    ri = np.array([30.0, 40.0, 48.0, 52.0, 55.0])
    l_in = _levels_for_target_ri(ri, 85.0, 12.0, 10.0)
    octave_freqs = np.array([125, 250, 500, 1000, 2000], dtype=float)
    kc = building.adaptation_term_kc(octave_freqs) if with_kc else None
    return building.intensity_sound_reduction(
        np.full(5, 85.0), l_in, measurement_area=12.0, area=10.0, kc=kc
    )


def test_octave_band_fiche_renders(tmp_path: Path) -> None:
    """A five-octave-band intensity result also yields a one-page fiche."""
    result = _octave_result()
    assert result.rating is not None
    out = tmp_path / "octave.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    assert "Octave-band" in _extract_text(str(out))


def test_octave_band_verbose_declares_band_resolution(tmp_path: Path) -> None:
    """The verbose RI,M caption still declares the octave-band resolution."""
    out = tmp_path / "octave_verbose.pdf"
    _octave_result(with_kc=True).report(str(out), verbose=True)
    assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "Octave-band" in text  # band resolution stated (ISO 717-1 Clause 5.3)
    assert "Kc-modified" in text  # the RI,M column is present


def test_one_third_octave_verbose_declares_band_resolution(tmp_path: Path) -> None:
    """The verbose RI,M caption declares the one-third-octave resolution."""
    out = tmp_path / "verbose_res.pdf"
    _intensity_result(with_kc=True).report(str(out), verbose=True)
    assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "One-third-octave" in text
    assert "Kc-modified" in text


def test_spanish_fiche_renders_translated(tmp_path: Path) -> None:
    """``language="es"`` renders the Spanish fiche with comma decimals."""
    import re

    out = tmp_path / "es.pdf"
    _intensity_result(with_kc=True).report(
        str(out),
        metadata=ReportMetadata(requirement=25.0, laboratory="Ejemplo"),
        verbose=True,
        language="es",
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "CUMPLE" in text
    assert "intensidad acústica" in " ".join(text.split())
    assert re.search(r"\d+,\d", text)  # comma decimal separator


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    out = str(tmp_path / "x.pdf")
    result = _intensity_result()
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    """An unsupported language raises ``ValueError`` (shared validation path)."""
    out = str(tmp_path / "x.pdf")
    result = _intensity_result()
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="fr")


def test_missing_rating_rejected(tmp_path: Path) -> None:
    """A result without the ISO 717 rating (non-core band count) is rejected."""
    l_in = _levels_for_target_ri(np.full(8, 40.0), 85.0, 12.0, 10.0)
    result = building.intensity_sound_reduction(
        np.full(8, 85.0), l_in, measurement_area=12.0, area=10.0
    )
    assert result.rating is None
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="rating"):
        result.report(out)


def test_verbose_without_kc_renders_plain_table(tmp_path: Path) -> None:
    """``verbose=True`` with no RI,M falls back to the two-column table."""
    out = tmp_path / "plain.pdf"
    _intensity_result(with_kc=False).report(str(out), verbose=True)
    assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "One-third-octave" in text
    assert "Kc-modified" not in text


def test_non_iso_band_count_rejected(tmp_path: Path) -> None:
    """A manually built 8-band result with matching rating arrays is rejected.

    The public API promises only the 16 one-third-octave or 5 octave bands
    ISO 717-1 rates; a hand-crafted rating whose per-band arrays all match an
    8-band curve would otherwise satisfy the shared renderer's shape checks.
    """
    centers = np.array([100, 125, 160, 200, 250, 315, 400, 500], dtype=float)
    curve = np.linspace(20.0, 40.0, 8)
    rating = building.WeightedRatingResult(
        rating=30,
        c=-2,
        ctr=-3,
        unfavourable_sum=0.0,
        band_centers=centers,
        measured=curve,
        shifted_reference=curve,
    )
    result = IntensityReductionResult(
        r_i=curve, r_i_modified=None, rating=rating, rating_modified=None
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="16 one-third-octave"):
        result.report(out)


def test_rating_without_per_band_data_rejected(tmp_path: Path) -> None:
    """A manually built rating lacking the per-band arrays is rejected."""
    bare_rating = building.WeightedRatingResult(
        rating=30, c=-2, ctr=-3, unfavourable_sum=0.0
    )
    result = IntensityReductionResult(
        r_i=np.asarray(ref.ISO15186_1_REF_RI, dtype=np.float64),
        r_i_modified=None,
        rating=bare_rating,
        rating_modified=None,
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="per-band"):
        result.report(out)


def test_clause8_areas_stated_in_statement(tmp_path: Path) -> None:
    """The statement carries S and Sm when the result holds them (Clause 8 g)."""
    out = tmp_path / "areas.pdf"
    _intensity_result().report(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "S = 10" in text
    assert "= 12" in text  # Sm = 12 m2
    assert "Clause 8" in text


def test_clause8_fpi_and_residual_columns(tmp_path: Path) -> None:
    """Supplied FpI and dpI0 spectra are annexed as table columns (Clause 8 i)."""
    result = _intensity_result()
    fpi = np.full(16, 4.0)
    residual = np.full(16, 18.0)
    out = tmp_path / "fpi.pdf"
    result.report(str(out), fpi=fpi, residual_index=residual)
    assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "4.0" in text  # the FpI column values
    assert "18.0" in text  # the dpI0 column values
    # Verbose keeps the qualification columns beside the RI,M column too.
    out2 = tmp_path / "fpi_verbose.pdf"
    _intensity_result(with_kc=True).report(
        str(out2), verbose=True, fpi=fpi, residual_index=residual
    )
    assert_one_page(str(out2))


def test_clause8_fpi_wrong_band_count_rejected(tmp_path: Path) -> None:
    """An FpI spectrum not matching the reported bands is rejected."""
    result = _intensity_result()
    out = str(tmp_path / "x.pdf")
    fpi_5_bands = np.full(5, 4.0)
    residual_3_bands = np.full(3, 18.0)
    with pytest.raises(ValueError, match="fpi"):
        result.report(out, fpi=fpi_5_bands)
    with pytest.raises(ValueError, match="residual_index"):
        result.report(out, residual_index=residual_3_bands)
