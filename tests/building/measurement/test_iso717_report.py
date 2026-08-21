#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 717 Annex C rating report (``.report()`` -> PDF).

The report is a rendering feature, so these tests assert only structural
facts: a non-empty file that starts with the ``%PDF`` magic bytes is written
for both an airborne (ISO 717-1) and an impact (ISO 717-2) rating, unknown
engines and results lacking the per-band data are rejected, and the
convenience wrapper on the panel prediction result also renders. Pixel or
layout content is never inspected.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

import numpy as np
from reference_data import (
    ISO717_1_ANNEX_C_R as _AIRBORNE_R,
)
from reference_data import (
    ISO717_2_ANNEX_C1_EXPECTED as _IMPACT_EXPECTED,
)
from reference_data import (
    ISO717_2_ANNEX_C1_LN as _IMPACT_LN,
)
from report_assertions import assert_one_page, assert_pdf

from phonometry import ReportMetadata, building


def test_airborne_report_writes_pdf(tmp_path) -> None:
    """An ISO 717-1 airborne rating renders a PDF fiche."""
    result = building.weighted_rating(_AIRBORNE_R)
    out = tmp_path / "airborne.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    assert_pdf(str(out))


def test_impact_report_writes_pdf(tmp_path) -> None:
    """An ISO 717-2 impact rating renders a PDF fiche."""
    result = building.weighted_impact_rating(_IMPACT_LN)
    assert result.quantity == "impact"
    out = tmp_path / "impact.pdf"
    returned = result.report(str(out))
    assert returned == str(out)
    assert_pdf(str(out))


def test_panel_result_report_convenience(tmp_path) -> None:
    """``SoundReductionResult.report()`` rates ``R(f)`` and writes its fiche."""
    freqs = [
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
    ]
    res = building.single_panel_transmission_loss(
        freqs, 15.0, critical_frequency=2000.0, loss_factor=0.02
    )
    out = tmp_path / "panel.pdf"
    res.report(str(out))
    assert_pdf(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = building.weighted_rating(_AIRBORNE_R)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_missing_band_data_rejected(tmp_path) -> None:
    """A rating built without the per-band curves cannot be reported."""
    bare = building.WeightedRatingResult(rating=52, c=-1, ctr=-4, unfavourable_sum=30.0)
    assert bare.band_centers is None
    out = str(tmp_path / "bare.pdf")
    with pytest.raises(ValueError, match="per-band data"):
        bare.report(out)


def test_airborne_fiche_reproduces_iso717_1_annex_c1(tmp_path) -> None:
    """The airborne fiche reproduces the ISO 717-1:2020 Annex C Table C.1 example.

    The values printed in the standard's worked example are exactly the ones the
    fiche shows: Rw(C;Ctr) = 30(-2;-3) dB, unfavourable-deviation sum 31,8 dB, the
    reference curve shifted by -22 dB, and the per-band unfavourable deviations.
    """
    result = building.weighted_rating(_AIRBORNE_R)
    assert (result.rating, result.c, result.ctr) == (30, -2, -3)
    assert result.unfavourable_sum == pytest.approx(31.8, abs=0.05)
    # Reference values shifted by -22 dB (Table C.1, column 3).
    shifted = np.array(
        [11, 14, 17, 20, 23, 26, 29, 30, 31, 32, 33, 34, 34, 34, 34, 34], float
    )
    assert np.allclose(result.shifted_reference, shifted)
    # Unfavourable deviations (Table C.1, column 4; "-" printed as 0).
    deviations = np.maximum(result.shifted_reference - result.measured, 0.0)
    expected = np.array(
        [0, 0, 0, 0, 0.6, 3.3, 4.2, 3.4, 3.0, 1.5, 1.2, 1.5, 0.6, 1.0, 3.0, 8.5], float
    )
    assert np.allclose(deviations, expected, atol=0.05)
    assert_pdf(str(result.report(str(tmp_path / "airborne_c1.pdf"))))


def test_impact_fiche_reproduces_iso717_2_annex_c1(tmp_path) -> None:
    """The impact fiche reproduces the ISO 717-2 Annex C Table C.1 example.

    Ln,w = 79 dB, CI = -11 dB, unfavourable-deviation sum 28,0 dB (see the note
    on the 2020 reprint's CI in reference_data).
    """
    result = building.weighted_impact_rating(_IMPACT_LN)
    assert result.rating == _IMPACT_EXPECTED["ln_w"]
    assert result.ci == _IMPACT_EXPECTED["ci"]
    assert result.unfavourable_sum == pytest.approx(
        _IMPACT_EXPECTED["unfavourable_sum"], abs=0.05
    )
    assert_pdf(str(result.report(str(tmp_path / "impact_c1.pdf"))))


def _full_metadata(**overrides) -> ReportMetadata:
    """A fully populated :class:`ReportMetadata` for the accredited fiche."""
    base = {
        "specimen": "200 mm reinforced-concrete wall",
        "client": "Acoustic Test Client Ltd.",
        "mounted_by": "Test laboratory staff",
        "manufacturer": "Concrete Works Inc.",
        "area": 10.0,
        "mass_per_area": 460.0,
        "source_volume": 53.0,
        "receiving_volume": 51.0,
        "temperature": 21.5,
        "relative_humidity": 45.0,
        "pressure": 101.3,
        "test_room": "Transmission suite T1",
        "mounting": "Rigid, mortar-sealed perimeter",
        "measurement_standard": "ISO 10140-2",
        "test_date": "2026-07-18",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-0042",
        "notes": "Engineering method, one-third-octave bands.",
    }
    base.update(overrides)
    return ReportMetadata(**base)


def test_metadata_allows_non_positive_temperature() -> None:
    """Test temperatures of 0 C or below are valid (cold field conditions)."""
    md = ReportMetadata(
        temperature=-5.0, source_temperature=0.0, receiving_temperature=-12.3
    )
    assert md.temperature == -5.0


def test_metadata_rejects_out_of_range_humidity() -> None:
    """Relative humidity outside 0..100 % is rejected."""
    with pytest.raises(ValueError, match="humidity"):
        ReportMetadata(relative_humidity=150.0)


def test_report_escapes_xml_specials_in_metadata(tmp_path) -> None:
    """Metadata with XML specials (& < >) renders without crashing reportlab."""
    result = building.weighted_rating(_AIRBORNE_R)
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="wall <A> & partition",
        laboratory="Lab & Sons",
        operator="A <B>",
        report_id="R&D-001",
        measurement_standard="ISO 10140-2 & Annex",
    )
    out = tmp_path / "xml.pdf"
    result.report(str(out), metadata=md)
    assert_one_page(str(out))


def test_full_metadata_renders_one_page(tmp_path) -> None:
    """A full ReportMetadata renders a one-page accredited fiche."""
    result = building.weighted_rating(_AIRBORNE_R)
    out = tmp_path / "airborne_meta.pdf"
    result.report(str(out), metadata=_full_metadata())
    assert_one_page(str(out))


def test_verbose_renders_annex_c_table(tmp_path) -> None:
    """``verbose=True`` renders the Annex C evaluation table one-pager."""
    result = building.weighted_rating(_AIRBORNE_R)
    out = tmp_path / "airborne_verbose.pdf"
    result.report(str(out), metadata=_full_metadata(), verbose=True)
    assert_one_page(str(out))


def test_requirement_pass_and_fail_both_render(tmp_path) -> None:
    """A PASS and a FAIL requirement both render a one-page fiche."""
    result = building.weighted_rating(_AIRBORNE_R)  # Rw = 30 dB
    passing = tmp_path / "pass.pdf"
    failing = tmp_path / "fail.pdf"
    result.report(str(passing), metadata=_full_metadata(requirement=25.0))
    result.report(str(failing), metadata=_full_metadata(requirement=52.0))
    assert_one_page(str(passing))
    assert_one_page(str(failing))


def test_impact_requirement_verdict_renders(tmp_path) -> None:
    """An impact fiche with a requirement (lower is better) renders."""
    result = building.weighted_impact_rating(_IMPACT_LN)
    out = tmp_path / "impact_meta.pdf"
    result.report(
        str(out),
        metadata=ReportMetadata(
            specimen="150 mm slab",
            measurement_standard="ISO 16283-2",
            requirement=60.0,
            laboratory="Phonometry Reference Laboratory",
        ),
    )
    assert_one_page(str(out))


def test_metadata_rejects_negative_area() -> None:
    """``ReportMetadata`` rejects a non-positive numeric field."""
    with pytest.raises(ValueError, match="area"):
        ReportMetadata(area=-5.0)


def _extract_text(path: str) -> str:
    """The concatenated text of every page (for language assertions)."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_airborne_fiche_pins_displayed_rating(tmp_path) -> None:
    """The fiche prints exactly the Annex C Table C.1 numbers.

    Independent oracle: ISO 717-1:2020 Annex C states Rw (C; Ctr) =
    30 (-2; -3) dB with an unfavourable-deviation sum of 31,8 dB. The
    adaptation terms carry a sign only when negative, as the standard's own
    examples write them (e.g. "41 (0; -5) dB" in clause 5.2).
    """
    result = building.weighted_rating(_AIRBORNE_R)
    out = tmp_path / "pins.pdf"
    result.report(str(out), verbose=True)
    text = _extract_text(str(out))
    assert "Rw (C; Ctr) = 30 (-2; -3) dB" in text
    assert "One-third-octave R [dB]" in text
    assert "31.8" in text  # Annex C unfavourable-deviation sum


def test_octave_band_fiche_declares_octave_bands(tmp_path) -> None:
    """A 5-band octave rating is captioned as octave bands (ISO 717-2 4.4).

    Clause 4.4 requires stating whether the rating came from one-third-octave
    or octave measurements; a 5-row octave fiche must not claim
    "One-third-octave".
    """
    result = building.weighted_impact_rating([65.0, 68.0, 67.0, 63.0, 58.0])
    assert result.band_centers is not None
    assert len(result.band_centers) == 5
    out = tmp_path / "octave.pdf"
    result.report(str(out))
    text = _extract_text(str(out))
    assert "Octave-band Ln [dB]" in text
    assert "One-third-octave" not in text


def test_symbol_labels_field_quantity(tmp_path) -> None:
    """``symbol="DnT,w"`` relabels the boxed result, table and verdict.

    A field measurement rated to a standardized level difference must not be
    reported as the laboratory ``Rw`` (ISO 717-1 Tables 1-2 distinguish the
    quantities).
    """
    result = building.weighted_rating(_AIRBORNE_R)
    out = tmp_path / "symbol.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=25.0), symbol="DnT,w")
    text = _extract_text(str(out))
    assert "DnT,w (C; Ctr) = 30 (-2; -3) dB" in text
    assert "DnT,w = 30 dB" in text  # verdict row
    assert "Rw" not in text


def test_invalid_symbol_rejected(tmp_path) -> None:
    """A malformed quantity symbol raises ``ValueError``."""
    result = building.weighted_rating(_AIRBORNE_R)
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="symbol"):
        result.report(out, symbol="not a symbol")


def test_metadata_area_is_not_display_rounded(tmp_path) -> None:
    """A supplied area of 1.23 m^2 is reprinted verbatim, not reduced to 1.2."""
    result = building.weighted_rating(_AIRBORNE_R)
    out = tmp_path / "area.pdf"
    result.report(str(out), metadata=ReportMetadata(area=1.23))
    text = _extract_text(str(out))
    assert "1.23" in text
    assert "1.2 " not in text


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """``language="es"`` renders a one-page Spanish fiche with comma decimals."""
    import re

    result = building.weighted_rating(_AIRBORNE_R)  # Rw = 30 dB, passes a 25 dB minimum
    out = tmp_path / "airborne_es.pdf"
    result.report(str(out), metadata=_full_metadata(requirement=25.0), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Índice de aislamiento acústico a ruido aéreo" in text
    assert "CUMPLE" in text
    assert re.search(r"\d,\d", text) is not None  # comma decimal separator


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ``ValueError``."""
    result = building.weighted_rating(_AIRBORNE_R)
    with pytest.raises(ValueError, match="language"):
        result.report(str(tmp_path / "bad.pdf"), language="xx")
