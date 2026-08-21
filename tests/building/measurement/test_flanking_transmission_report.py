#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 10848 flanking-transmission fiches (``.report()`` -> PDF).

ISO 10848 carries no worked numeric example, so the fiches are exercised with
the closed-form syntheses the module tests use: a rigid-junction ``Kij`` built
from Formula (13), and the two overall descriptors built so their ISO 717
single number is fixed and hand-checkable. Rendering assertions are structural
(a one-page ``%PDF``) plus pypdf text-extraction checks of the boxed single
number, the standard-basis line and the measurement statement, like the sibling
report tests.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("reportlab")

from report_assertions import assert_one_page

from phonometry import ReportMetadata, building

#: The ISO 10848 mandatory one-third-octave range (100 Hz to 5000 Hz, 18 bands).
_FREQS = [
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
    4000,
    5000,
]

#: A rigid-junction direction-averaged velocity level difference (dB), rising
#: with frequency (the illustrative case the docs and generator use).
_DV = np.array(
    [
        4.5,
        4.8,
        5.2,
        5.6,
        6.0,
        6.5,
        7.0,
        7.6,
        8.1,
        8.7,
        9.2,
        9.8,
        10.3,
        10.9,
        11.4,
        11.9,
        12.3,
        12.7,
    ]
)


def _extract_text(path: str) -> str:
    """The concatenated text of every page."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def _kij_result() -> building.VibrationReductionResult:
    """A rigid-junction Kij with the three lowest bands bracketed."""
    modal_overlap = np.full(len(_FREQS), 1.0)
    modal_overlap[:3] = 0.1
    return building.vibration_reduction_index(
        _DV,
        junction_length=4.0,
        area_i=12.0,
        area_j=10.0,
        frequency=_FREQS,
        structural_reverberation_time_i=0.35,
        structural_reverberation_time_j=0.40,
        modal_overlap=modal_overlap,
    )


def _dnf_result() -> building.FlankingLevelDifferenceResult:
    """A 16-band Dn,f whose ISO 717-1 single number is hand-checkable."""
    dn_f = np.array(
        [48, 49, 50, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65],
        dtype=float,
    )
    source = np.full(16, 80.0)
    return building.normalized_flanking_level_difference(
        source, source - dn_f, absorption_area=np.full(16, 10.0)
    )


def _lnf_result() -> building.FlankingImpactLevelResult:
    """A 16-band Ln,f whose ISO 717-2 single number is hand-checkable."""
    receive = np.array(
        [58, 57, 56, 55, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32],
        dtype=float,
    )
    return building.normalized_flanking_impact_level(
        receive, absorption_area=np.full(16, 10.0)
    )


# --------------------------------------------------------------------------- #
# Vibration reduction index Kij (ISO 10848-1)
# --------------------------------------------------------------------------- #
def test_kij_fiche_renders_single_number_and_basis(tmp_path) -> None:
    """The Kij fiche boxes the Annex A mean and names ISO 10848-1."""
    result = _kij_result()
    out = tmp_path / "kij.pdf"
    result.report(str(out), metadata=ReportMetadata(specimen="Rigid junction"))
    assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert result.single_number is not None
    assert f"{result.single_number:.1f} dB" in text
    assert "ISO 10848-1:2006" in text
    assert "Junction vibration reduction index" in text
    # A bracketed low band prints its value in brackets and a note explains it.
    assert f"[{result.k_ij[0]:.1f}]" in text
    assert "excluded from the single number" in text


def test_kij_fiche_verbose_adds_membership_column(tmp_path) -> None:
    """``verbose=True`` adds the single-number-membership column."""
    out = tmp_path / "kij_verbose.pdf"
    _kij_result().report(str(out), verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "In mean" in text


def test_kij_mean_membership_excludes_out_of_range_bands() -> None:
    """Single-number membership needs the Annex A range, not just no bracket.

    The example spans 100 Hz to 5000 Hz, so the bands above 1250 Hz are not
    bracketed yet must stay out of the mean; only the 200 Hz to 1250 Hz,
    non-bracketed bands count.
    """
    from phonometry._report.iso10848 import _kij_mean_membership

    frequencies = np.asarray(_FREQS, dtype=np.float64)
    bracketed = np.zeros(frequencies.size, dtype=np.bool_)
    bracketed[:3] = True  # the three lowest bands (also below 200 Hz)
    in_mean, low, high = _kij_mean_membership(frequencies, bracketed)
    assert (low, high) == (200.0, 1250.0)
    # 200, 250, 315, 400, 500, 630, 800, 1000, 1250 Hz -> nine bands.
    assert int(in_mean.sum()) == 9
    # A non-bracketed band above the range (1600 Hz) is excluded.
    assert not bool(in_mean[list(_FREQS).index(1600)])
    # A non-bracketed band inside the range (250 Hz) is included.
    assert bool(in_mean[list(_FREQS).index(250)])


def test_kij_fiche_distinguishes_all_bracketed_from_out_of_range(tmp_path) -> None:
    """An empty mean states why: every in-range band bracketed (M < 0,25)."""
    result = building.vibration_reduction_index(
        _DV,
        junction_length=4.0,
        area_i=12.0,
        area_j=10.0,
        frequency=_FREQS,
        modal_overlap=0.1,  # every band bracketed -> no single number
    )
    assert result.single_number is None
    out = tmp_path / "kij_all_bracketed.pdf"
    result.report(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "all in-range bands bracketed" in text
    assert "no bands in the Annex A range" not in text
    out_es = tmp_path / "kij_all_bracketed_es.pdf"
    result.report(str(out_es), language="es")
    text_es = " ".join(_extract_text(str(out_es)).split())
    assert "todas las bandas del intervalo entre corchetes" in text_es


def test_kij_fiche_states_no_bands_in_annex_a_range(tmp_path) -> None:
    """A spectrum entirely outside the Annex A range keeps the range message."""
    high_freqs = [1600.0, 2000.0, 2500.0]
    result = building.vibration_reduction_index(
        [10.0, 11.0, 12.0],
        junction_length=4.0,
        area_i=12.0,
        area_j=10.0,
        frequency=high_freqs,
    )
    assert result.single_number is None
    out = tmp_path / "kij_out_of_range.pdf"
    result.report(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "no bands in the Annex A range" in text
    assert "all in-range bands bracketed" not in text
    out_es = tmp_path / "kij_out_of_range_es.pdf"
    result.report(str(out_es), language="es")
    text_es = " ".join(_extract_text(str(out_es)).split())
    assert "no hay bandas en el intervalo del anexo A" in text_es


def test_kij_fiche_without_frequencies_rejected(tmp_path) -> None:
    """A Kij result with no band frequencies cannot be reported."""
    result = building.vibration_reduction_index([5.0, 6.0, 7.0], 2.0, 4.0, 4.0)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="band centre frequencies"):
        result.report(out)


def test_kij_fiche_spanish(tmp_path) -> None:
    """``language="es"`` renders the Spanish Kij fiche with comma decimals."""
    import re

    out = tmp_path / "kij_es.pdf"
    _kij_result().report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Índice de reducción vibracional de la unión" in text
    assert re.search(r"\d+,\d", text)


# --------------------------------------------------------------------------- #
# Normalized flanking level difference Dn,f (ISO 10848-2, airborne)
# --------------------------------------------------------------------------- #
def test_dnf_fiche_rating_and_basis(tmp_path) -> None:
    """The Dn,f fiche boxes the ISO 717-1 rating and names ISO 10848-2."""
    result = _dnf_result()
    rating = result.rating
    assert rating is not None
    out = tmp_path / "dnf.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    expected = f"{rating.rating} ({rating.c:+d}; {rating.ctr:+d}) dB"
    assert expected in text
    assert "ISO 10848-2:2006" in text
    assert "ISO 717-1:2020" in text


def test_dnf_fiche_verbose_and_verdict(tmp_path) -> None:
    """Verbose annexes the ISO 717 evaluation; the requirement passes."""
    out = tmp_path / "dnf_verbose.pdf"
    _dnf_result().report(
        str(out), metadata=ReportMetadata(requirement=55.0), verbose=True
    )
    assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "PASS" in text  # Dn,f,w = 60 dB >= 55 dB
    assert "Unfav. dev." in text


def test_dnf_fiche_part_selects_basis_designation(tmp_path) -> None:
    """``part`` selects the governing ISO 10848 part named in the basis line."""
    result = _dnf_result()
    for part, designation in ((3, "ISO 10848-3:2006"), (4, "ISO 10848-4:2010")):
        out = tmp_path / f"dnf_part{part}.pdf"
        result.report(str(out), part=part)
        assert_one_page(str(out))
        text = _extract_text(str(out))
        assert designation in text
        assert "ISO 10848-2:2006" not in text


def test_dnf_fiche_unknown_part_rejected(tmp_path) -> None:
    """A part outside 2/3/4 is rejected."""
    out = str(tmp_path / "x.pdf")
    dnf, lnf = _dnf_result(), _lnf_result()
    with pytest.raises(ValueError, match="'part' must be 2, 3 or 4"):
        dnf.report(out, part=1)
    with pytest.raises(ValueError, match="'part' must be 2, 3 or 4"):
        lnf.report(out, part=5)


def test_dnf_fiche_without_rating_rejected(tmp_path) -> None:
    """A band count that yields no ISO 717 rating cannot be reported."""
    result = building.normalized_flanking_level_difference(
        [80.0] * 7, [40.0] * 7, [10.0] * 7
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="single-number rating"):
        result.report(out)


# --------------------------------------------------------------------------- #
# Normalized flanking impact level Ln,f (ISO 10848-2, tapping machine)
# --------------------------------------------------------------------------- #
def test_lnf_fiche_rating_and_basis(tmp_path) -> None:
    """The Ln,f fiche boxes the ISO 717-2 rating and names ISO 10848-2."""
    result = _lnf_result()
    rating = result.rating
    assert rating is not None
    out = tmp_path / "lnf.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=55.0))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{rating.rating} ({rating.ci:+d}) dB" in text
    assert "ISO 10848-2:2006" in text
    assert "ISO 717-2:2020" in text
    assert "PASS" in text  # Ln,f,w = 49 dB <= 55 dB


def test_lnf_fiche_part_selects_basis_designation(tmp_path) -> None:
    """``part=4`` names ISO 10848-4:2010 on the Ln,f basis line."""
    out = tmp_path / "lnf_part4.pdf"
    _lnf_result().report(str(out), part=4)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "ISO 10848-4:2010" in text
    assert "ISO 10848-2:2006" not in text


def test_lnf_fiche_spanish(tmp_path) -> None:
    """``language="es"`` renders the Spanish Ln,f fiche."""
    out = tmp_path / "lnf_es.pdf"
    _lnf_result().report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Aislamiento acústico por flancos a ruido de impactos" in text
    assert "ISO 10848-2:2006" in text  # the {standard} template is filled


# --------------------------------------------------------------------------- #
# Rendering contract
# --------------------------------------------------------------------------- #
def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError`` on every fiche."""
    out = str(tmp_path / "x.pdf")
    kij, dnf, lnf = _kij_result(), _dnf_result(), _lnf_result()
    with pytest.raises(ValueError, match="engine"):
        kij.report(out, engine="weasyprint")
    with pytest.raises(ValueError, match="engine"):
        dnf.report(out, engine="weasyprint")
    with pytest.raises(ValueError, match="engine"):
        lnf.report(out, engine="weasyprint")
