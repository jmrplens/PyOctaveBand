#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 16251-1 floor-covering impact-improvement report (``.report()``).

The report is a rendering feature, so these tests assert only structural facts:
a valid single-page PDF is written for an impact-improvement measurement, the
displayed weighted improvement, band labels and per-band ``delta-L`` match the
documented oracle, the verbose octave-band table and the Spanish fiche stay one
page, unknown engines/languages are rejected, and XML specials in metadata do
not break reportlab. Pixel or layout content is never inspected.

The clean-room oracle is the documented soft-carpet improvement spectrum
delta-L = [0, 0, 1, 2, 4, 7, 11, 15, 18, 21, 23, 25, 27, 28, 29, 30] dB over the
16 one-third-octave bands 100 Hz to 3150 Hz (ISO 16251-1 has no filled worked
example). Its weighted improvement follows ISO 717-2:2020 Clause 5: applied to
the heavyweight reference floor L_n,r,0 (Table 4, rated 78 dB / CI = -11 dB),
L_n,r = L_n,r,0 - delta-L rates to L_n,r,w = 59 dB, so delta-Lw = 78 - 59 = 19 dB
and CI,delta = -11 - 0 = -11 dB.
"""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

import numpy as np

from phonometry import ReportMetadata, building

_PDF_MAGIC = b"%PDF"

_FREQS = np.array(
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
_DELTA_L = np.array(
    [0, 0, 1, 2, 4, 7, 11, 15, 18, 21, 23, 25, 27, 28, 29, 30], dtype=float
)


def _result():
    bare = np.full(16, 78.0)
    return building.impact_improvement(bare, bare - _DELTA_L, _FREQS)


def _metadata(**overrides) -> ReportMetadata:
    base = {
        "specimen": "Textile floor covering (carpet), 6 mm pile",
        "client": "Acoustic Test Client Ltd.",
        "manufacturer": "Floors Works Inc.",
        "mass_per_area": 2.4,
        "mounting": "Laid loose on the mock-up plate (ISO 10140-1 category I)",
        "test_room": "Small-mock-up impact rig R1",
        "measurement_standard": "ISO 16251-1",
        "temperature": 21.0,
        "pressure": 101.2,
        "test_date": "2026-07-21",
        "laboratory": "Phonometry Reference Laboratory",
        "operator": "Jose Manuel Requena Plens",
        "report_id": "PHN-2026-16251",
        "requirement": 17.0,
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

    return "\n".join(page.extract_text() for page in PdfReader(path).pages).replace(
        "\n", " "
    )


def test_report_writes_one_page_pdf(tmp_path) -> None:
    out = tmp_path / "iso16251.pdf"
    returned = _result().report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))


def test_report_with_metadata_one_page(tmp_path) -> None:
    out = tmp_path / "iso16251_meta.pdf"
    _result().report(str(out), metadata=_metadata())
    _assert_one_page(str(out))


def test_unknown_engine_rejected(tmp_path) -> None:
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    result = _result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="Unknown language"):
        result.report(out, language="xx")


def test_displayed_values_match_oracle(tmp_path) -> None:
    """The fiche prints the weighted improvement, band labels and delta-L."""
    out = tmp_path / "iso16251.pdf"
    _result().report(str(out), metadata=_metadata())
    text = _text(str(out))
    # Boxed single number delta-Lw (CI,delta) = 19 (-11) dB.
    assert "19 (-11) dB" in text
    # Band labels and a couple of the per-band delta-L values (one decimal).
    assert "100" in text  # first band centre
    assert "3150" in text  # last band centre
    assert "15.0" in text  # delta-L(500 Hz)
    assert "30.0" in text  # delta-L(3150 Hz)
    assert "100 to 3150" in text  # measured frequency range


def test_verbose_shows_reference_floor_column_one_page(tmp_path) -> None:
    """verbose=True adds the reference-floor L_n,r column and stays one page."""
    out = tmp_path / "iso16251_verbose.pdf"
    _result().report(str(out), metadata=_metadata(), verbose=True)
    _assert_one_page(str(out))
    text = _text(str(out))
    # L_n,r(100 Hz) = 67 - 0 = 67.0 and L_n,r(500 Hz) = 70.5 - 15 = 55.5.
    assert "67.0" in text
    assert "55.5" in text


def test_requirement_pass_verdict(tmp_path) -> None:
    """A higher weighted improvement passes the requirement."""
    out = tmp_path / "iso16251_pass.pdf"
    _result().report(str(out), metadata=_metadata(requirement=17.0))
    assert "PASS" in _text(str(out))


def test_requirement_boundary_verdict(tmp_path) -> None:
    """At equality (requirement == delta_lw) the verdict passes (>= rule)."""
    out = tmp_path / "iso16251_boundary.pdf"
    result = _result()
    assert result.delta_lw == 19
    result.report(str(out), metadata=_metadata(requirement=19.0))
    text = _text(str(out))
    assert "PASS" in text
    assert "FAIL" not in text


def test_requirement_fail_verdict(tmp_path) -> None:
    """A weighted improvement below the requirement fails."""
    out = tmp_path / "iso16251_fail.pdf"
    _result().report(str(out), metadata=_metadata(requirement=25.0))
    assert "FAIL" in _text(str(out))


def test_limited_band_uses_literal_greater_than(tmp_path) -> None:
    """A band at the 1,3 dB limit renders a literal '>' marker, not '&gt;'."""
    bare = np.full(16, 50.0)
    covering = np.full(16, 49.0)
    background = np.full(16, 48.0)  # margin 2 dB < 6 dB -> every band limited
    result = building.impact_improvement(bare, covering, _FREQS, background=background)
    assert bool(np.any(result.limited))
    out = tmp_path / "iso16251_limited.pdf"
    result.report(str(out))
    text = _text(str(out))
    assert "&gt;" not in text  # the XML entity must not leak into a plain cell
    assert ">" in text
    # The single number is qualified as a lower bound (Formula (2) limit).
    assert "lower bound" in text
    assert "≥" in text  # the >= prefix on the boxed delta-Lw


def test_unlimited_result_not_qualified_as_lower_bound(tmp_path) -> None:
    """A result with no background-limited band prints a plain delta-Lw."""
    out = tmp_path / "iso16251_plain.pdf"
    _result().report(str(out))
    text = _text(str(out))
    assert "lower bound" not in text
    assert "≥" not in text


def test_manual_result_without_ci_delta_omits_adaptation_term(tmp_path) -> None:
    """A hand-built result lacking CI,delta must not print a fabricated (+0)."""
    from phonometry.building.measurement.floor_covering_improvement import (
        FloorCoveringImprovementResult,
    )

    result = FloorCoveringImprovementResult(
        frequencies=_FREQS,
        improvement=_DELTA_L,
        limited=np.zeros(16, dtype=bool),
        delta_lw=15,
        ci_delta=None,
    )
    out = tmp_path / "iso16251_noci.pdf"
    result.report(str(out))
    _assert_one_page(str(out))
    text = _text(str(out))
    assert "(+0)" not in text
    assert "15 dB" in text


def test_metadata_xml_specials_do_not_break(tmp_path) -> None:
    out = tmp_path / "iso16251_xml.pdf"
    _result().report(str(out), metadata=_metadata(specimen='Carpet <A> & <B> "edge"'))
    _assert_one_page(str(out))


def test_no_metadata_still_renders(tmp_path) -> None:
    """Without metadata the header still shows the measured frequency range."""
    out = tmp_path / "iso16251_bare.pdf"
    _result().report(str(out))
    _assert_one_page(str(out))
    assert "100 to 3150" in _text(str(out))


def test_spanish_fiche_uses_comma_decimal(tmp_path) -> None:
    out = tmp_path / "iso16251_es.pdf"
    _result().report(str(out), metadata=_metadata(), language="es")
    _assert_one_page(str(out))
    assert "15,0" in _text(str(out))  # Spanish decimal comma


def test_characterisation_headline_without_rating_bands(tmp_path) -> None:
    """A spectrum without the 16 rating bands boxes a characterisation, no number."""
    result = building.impact_improvement(
        [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [500.0, 1000.0, 2000.0]
    )
    assert result.delta_lw is None
    out = tmp_path / "iso16251_norate.pdf"
    result.report(str(out))
    _assert_one_page(str(out))
