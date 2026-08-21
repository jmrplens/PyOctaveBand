#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the room-noise rating reports (``.report()``, ANSI/ASA S12.2-2019).

The report is a rendering feature, so these tests assert only structural
facts: a valid single-page PDF is written for an NC and an RC result, the
octave-band level table and the boxed rating carry the oracle values of the
module's own tangency/Annex-D self-consistency spectra, unknown
engines/languages are rejected, XML specials in metadata do not break
reportlab, the optional target-rating verdict renders both ways, and the
Spanish fiche is translated with comma decimals. Pixel or layout content is
never inspected.

Oracle. Both example spectra are exact: the NC spectrum is the ANSI/ASA
S12.2-2019 Table 1 NC-40 contour with every band depressed 5 dB except the
250 Hz octave (left on the NC-40 curve, so the tangency rating is NC-40 with
the 250 Hz band governing); the RC spectrum is the Annex D RC-35 Mark II curve
with the 250 Hz octave raised 8 dB (the 500/1000/2000 Hz mid bands unchanged,
so LMF = 35 dB and the rating is RC-35, and the raised low band tags it rumble,
RC-35(R)).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("reportlab")
pytest.importorskip("svglib")
pytest.importorskip("pypdf")

from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.room import noise_criteria as rn


def _nc_result() -> rn.NCResult:
    """An NC-40 spectrum governed by the 250 Hz band (oracle from Table 1)."""
    contour = rn.nc_curve(40.0)
    levels = contour - 5.0
    levels[4] = contour[4]  # 250 Hz sits on the NC-40 curve and governs.
    return rn.noise_criterion(levels)


def _rc_result() -> rn.RCResult:
    """An RC-35(R) rumble spectrum (oracle from Annex D, LMF unchanged)."""
    levels = rn.rc_curve(35.0)
    levels[4] += 8.0  # 250 Hz rumble.
    return rn.room_criterion(levels)


def _extract_text(path: str) -> str:
    """The concatenated, single-line text of every page."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages).replace(
        "\n", " "
    )


def _full_metadata(**overrides) -> ReportMetadata:
    base = {
        "specimen": "Open-plan office, air handling at nominal flow",
        "client": "Acoustic Test Client Ltd.",
        "test_room": "Office A",
        "room_volume": 180.0,
        "area": 60.0,
        "instrumentation": "Class 1 sound level meter + octave filter set",
        "measurement_standard": "ANSI/ASA S12.2",
        "temperature": 22.0,
        "relative_humidity": 42.0,
        "pressure": 101.2,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-S122",
    }
    base.update(overrides)
    return ReportMetadata(**base)


# --------------------------------------------------------------------------
# Oracle: the example spectra reproduce their tabulated ratings.
# --------------------------------------------------------------------------
def test_example_spectra_match_standard() -> None:
    """NC-40 governed by 250 Hz; RC-35(R) with LMF = 35 dB."""
    nc = _nc_result()
    assert nc.rating == pytest.approx(40.0, abs=1e-9)
    assert nc.governing_frequency == 250.0
    rc = _rc_result()
    assert rc.rating == 35
    assert rc.lmf == pytest.approx(35.0)
    assert rc.classification == "R"


# --------------------------------------------------------------------------
# Structural rendering
# --------------------------------------------------------------------------
def test_nc_report_writes_pdf(tmp_path) -> None:
    """An NC result renders a PDF fiche and returns its path."""
    out = tmp_path / "nc.pdf"
    returned = _nc_result().report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_rc_report_writes_pdf(tmp_path) -> None:
    """An RC result renders a PDF fiche and returns its path."""
    out = tmp_path / "rc.pdf"
    returned = _rc_result().report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))


def test_reports_with_full_metadata_one_page(tmp_path) -> None:
    """A full ReportMetadata renders a one-page fiche for both ratings."""
    _nc_result().report(str(tmp_path / "nc_meta.pdf"), metadata=_full_metadata())
    _rc_result().report(str(tmp_path / "rc_meta.pdf"), metadata=_full_metadata())
    assert_one_page(str(tmp_path / "nc_meta.pdf"))
    assert_one_page(str(tmp_path / "rc_meta.pdf"))


def test_reports_bare_without_metadata(tmp_path) -> None:
    """metadata=None renders a bare assessment fiche (no header)."""
    _nc_result().report(str(tmp_path / "nc_bare.pdf"), metadata=None)
    _rc_result().report(str(tmp_path / "rc_bare.pdf"), metadata=None)
    assert_one_page(str(tmp_path / "nc_bare.pdf"))
    assert_one_page(str(tmp_path / "rc_bare.pdf"))


def test_verbose_reports_one_page(tmp_path) -> None:
    """verbose=True (NC contour / RC reference + deviation) still fits one page."""
    _nc_result().report(
        str(tmp_path / "nc_v.pdf"), metadata=_full_metadata(), verbose=True
    )
    _rc_result().report(
        str(tmp_path / "rc_v.pdf"), metadata=_full_metadata(), verbose=True
    )
    assert_one_page(str(tmp_path / "nc_v.pdf"))
    assert_one_page(str(tmp_path / "rc_v.pdf"))


@pytest.mark.parametrize("factory", [_nc_result, _rc_result])
def test_unknown_engine_rejected(tmp_path, factory) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    out = str(tmp_path / "x.pdf")
    result = factory()
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


@pytest.mark.parametrize("factory", [_nc_result, _rc_result])
def test_unknown_language_rejected(tmp_path, factory) -> None:
    """An unknown fiche language raises ``ValueError``."""
    out = str(tmp_path / "bad.pdf")
    result = factory()
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="xx")


# --------------------------------------------------------------------------
# Displayed values (the oracle appears in the rendered text)
# --------------------------------------------------------------------------
def test_nc_rating_and_levels_render(tmp_path) -> None:
    """The NC-40 rating, governing band and a couple band levels appear."""
    out = tmp_path / "nc_values.pdf"
    _nc_result().report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "NC-40" in text
    assert "250" in text  # governing band label
    # Octave-band levels of the example spectrum (one decimal).
    assert "50.0" in text  # 250 Hz level (on the NC-40 curve)
    assert "79.0" in text  # 16 Hz level (depressed contour)


def test_rc_rating_and_levels_render(tmp_path) -> None:
    """The RC-35(R) rating, the mid-frequency average and a level appear."""
    out = tmp_path / "rc_values.pdf"
    _rc_result().report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "RC-35(R)" in text
    assert "35.0" in text  # LMF
    assert "53.0" in text  # 250 Hz rumble level (45 + 8)


def test_nc_verbose_shows_contour_column(tmp_path) -> None:
    """The verbose NC table adds the per-band NC contour column."""
    out = tmp_path / "nc_contour.pdf"
    _nc_result().report(str(out), metadata=_full_metadata(), verbose=True)
    text = _extract_text(str(out))
    assert "NC" in text
    # The governing 250 Hz band's contour equals the rating, 40.0.
    assert "40.0" in text


def test_verdict_both_ways(tmp_path) -> None:
    """A target rating renders PASS at or above the rating and FAIL below it."""
    passing = tmp_path / "pass.pdf"
    failing = tmp_path / "fail.pdf"
    _nc_result().report(str(passing), metadata=_full_metadata(requirement=40.0))
    _nc_result().report(str(failing), metadata=_full_metadata(requirement=35.0))
    assert "PASS" in _extract_text(str(passing))
    assert "FAIL" in _extract_text(str(failing))


def test_verdict_uses_display_rounded_values(tmp_path) -> None:
    """A rating that displays as the target must PASS at the boundary.

    The fiche prints the NC rating to one decimal, so the verdict compares
    the display-rounded values: a tangency rating of 40.0125 (250 Hz level
    0.01 dB above the NC-40 curve) prints as 40.0 and must PASS a target of
    40 rather than failing on the invisible third decimal.
    """
    contour = rn.nc_curve(40.0)
    levels = contour - 5.0
    levels[4] = contour[4] + 0.01  # 250 Hz a hair above the NC-40 curve.
    result = rn.noise_criterion(levels)
    assert 40.0 < result.rating < 40.05
    out = tmp_path / "boundary.pdf"
    result.report(str(out), metadata=_full_metadata(requirement=40.0))
    text = _extract_text(str(out))
    assert "PASS" in text
    assert "FAIL" not in text


def test_nc_sil_designation_renders(tmp_path) -> None:
    """A SIL-designated spectrum (clause 5.2.2) boxes NC-(SIL), no band."""
    result = rn.noise_criterion(rn.nc_curve(40.0))
    assert result.method == "SIL"
    out = tmp_path / "nc_sil.pdf"
    result.report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "NC-40" in text
    assert "SIL = 40.5" in text
    assert "Governing band" not in text


def test_nc_out_of_range_above_renders(tmp_path) -> None:
    """A spectrum above NC-70 renders >NC-70 and fails any in-family target."""
    import numpy as np

    result = rn.noise_criterion(np.full(10, 110.0))
    out = tmp_path / "nc_above.pdf"
    result.report(str(out), metadata=_full_metadata(requirement=70.0), verbose=True)
    text = _extract_text(str(out))
    assert ">NC-70" in text
    assert "NC-71" not in text  # the pre-fix fabricated designation
    assert ">70" in text  # verbose contour column and verdict value
    assert "FAIL" in text


def test_nc_out_of_range_below_renders(tmp_path) -> None:
    """A spectrum below NC-15 renders <NC-15 and passes any in-family target."""
    import numpy as np

    result = rn.noise_criterion(np.full(10, 5.0))
    out = tmp_path / "nc_below.pdf"
    result.report(str(out), metadata=_full_metadata(requirement=15.0), verbose=True)
    text = _extract_text(str(out))
    assert "<NC-15" in text
    assert "NC-14" not in text  # the pre-fix fabricated designation
    assert "PASS" in text


def test_rc_out_of_family_note_renders(tmp_path) -> None:
    """An RC rating outside RC-25 to RC-50 renders the extrapolation note."""
    result = rn.room_criterion(rn.rc_curve(55.0))
    assert result.out_of_family
    out = tmp_path / "rc_oof.pdf"
    result.report(str(out), metadata=_full_metadata())
    assert "Outside the tabulated RC-25 to RC-50 family" in _extract_text(str(out))


def test_report_without_requirement_has_no_verdict(tmp_path) -> None:
    """With no target rating the assessment fiche omits the verdict row."""
    out = tmp_path / "noverdict.pdf"
    _rc_result().report(str(out), metadata=_full_metadata())
    text = _extract_text(str(out))
    assert "PASS" not in text
    assert "FAIL" not in text


# --------------------------------------------------------------------------
# Robustness and localisation
# --------------------------------------------------------------------------
def test_report_escapes_xml_specials_in_metadata(tmp_path) -> None:
    """Metadata with XML specials (& < >) renders without crashing reportlab."""
    md = ReportMetadata(
        client="Ac & Co <Ltd>",
        specimen="office <A> & foyer",
        test_room="Room & Stage",
        laboratory="Lab & Sons",
        operator="A <B>",
        report_id="R&D-S122",
        measurement_standard="ANSI/ASA S12.2 & Annex",
    )
    out = tmp_path / "xml.pdf"
    _nc_result().report(str(out), metadata=md)
    assert_one_page(str(out))


def test_spanish_reports_render_translated(tmp_path) -> None:
    """language="es" renders one-page Spanish fiches with comma decimals."""
    import re

    nc_out = tmp_path / "nc_es.pdf"
    rc_out = tmp_path / "rc_es.pdf"
    _nc_result().report(
        str(nc_out), metadata=_full_metadata(requirement=40.0), language="es"
    )
    _rc_result().report(
        str(rc_out), metadata=_full_metadata(requirement=30.0), language="es"
    )
    assert_one_page(str(nc_out))
    assert_one_page(str(rc_out))
    nc_text = _extract_text(str(nc_out))
    assert "Calificación del ruido de salas" in nc_text
    assert "CUMPLE" in nc_text
    assert re.search(r"\d,\d", nc_text) is not None  # comma decimal separator
    rc_text = _extract_text(str(rc_out))
    assert "NO CUMPLE" in rc_text  # RC-35 fails a target of 30
    assert "retumbe" in rc_text  # rumble spectral quality


def test_subset_spectrum_renders_missing_bands(tmp_path) -> None:
    """A subset of octave bands renders, showing an em dash for absent bands."""
    freqs = [500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    levels = [40.0, 35.0, 30.0, 27.0, 22.0]
    out = tmp_path / "subset.pdf"
    rn.noise_criterion(levels, freqs).report(str(out), metadata=_full_metadata())
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "—" in text  # the unmeasured low bands are shown as an em dash


# --------------------------------------------------------------------------
# Hand-built results with mismatched bands
# --------------------------------------------------------------------------
def test_rc_result_refuses_a_reference_curve_of_another_length() -> None:
    """A reference curve of another length cannot be paired with the levels.

    ``RCResult`` is exported from a public package, so a hand-built one used
    to be able to carry a curve of any length; only the verbose columns read
    the two band by band, so the plain fiche rendered a silently short table.
    Refusing the pairing at construction covers both fiches and the plot.
    """
    from dataclasses import replace

    result = _rc_result()
    with pytest.raises(ValueError, match="'reference_curve'"):
        replace(result, reference_curve=result.reference_curve[:-1])


def test_rc_result_refuses_a_reference_curve_with_an_extra_axis() -> None:
    """A curve of shape ``(n, 1)`` is refused for the axis, not the length.

    It carries n entries along its first axis, so every count agrees and only
    the rank is wrong. Counting alone would let it through to the verbose
    column, which pairs the two band by band and would format a one-element
    array into every cell.
    """
    from dataclasses import replace

    result = _rc_result()
    two_dimensional = np.asarray(result.reference_curve).reshape(-1, 1)
    with pytest.raises(ValueError, match="'reference_curve' must have one axis"):
        replace(result, reference_curve=two_dimensional)
