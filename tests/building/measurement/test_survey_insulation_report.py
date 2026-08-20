#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the ISO 10052 survey-method field reports (``.report()`` -> PDF).

ISO 10052 carries no worked numeric example, so the fiches are exercised with
closed-form syntheses like the sibling report tests: with the reverberation
index ``k = 0`` (a receiving-room reverberation time equal to ``T0 = 0,5 s`` in
every band) the standardized quantities equal the raw ones exactly
(``DnT = D``, ``L'nT = Li``, ``D2m,nT = D2m``), so the fiche's rating is the
already-verified ISO 717 rating of the fed octave-band curve. Rendering
assertions are structural (a one-page ``%PDF``) plus pypdf text-extraction
checks of the boxed single number, the standard-basis line and the
survey-method statement.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("reportlab")

from phonometry import ReportMetadata, building

_PDF_MAGIC = b"%PDF"

#: Five octave bands (125 Hz to 2000 Hz), the ISO 10052 survey range.
_OCTAVE = 5

#: A reverberation index of 0 dB in every band (T = T0), so the standardized
#: quantities equal the raw ones and the rating is hand-checkable.
_K0 = np.zeros(_OCTAVE)


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


def _airborne() -> building.survey_airborne_insulation:  # type: ignore[valid-type]
    """A survey airborne result whose DnT equals a known octave curve."""
    l1 = np.full(_OCTAVE, 80.0)
    d = np.array([33.0, 36.0, 40.0, 44.0, 48.0])
    return building.survey_airborne_insulation(l1, l1 - d, _K0, volume=50.0, area=12.0)


def _impact() -> building.survey_impact_insulation:  # type: ignore[valid-type]
    """A survey impact result whose L'nT equals a known octave curve."""
    li = np.array([62.0, 64.0, 63.0, 60.0, 55.0])
    return building.survey_impact_insulation(li, _K0, volume=50.0)


def _facade() -> building.survey_facade_insulation:  # type: ignore[valid-type]
    """A survey facade result whose D2m,nT equals a known octave curve."""
    l1_2m = np.full(_OCTAVE, 75.0)
    d2m = np.array([31.0, 34.0, 37.0, 40.0, 43.0])
    return building.survey_facade_insulation(l1_2m, l1_2m - d2m, _K0, volume=40.0)


# --------------------------------------------------------------------------- #
# Airborne DnT / R' (ISO 10052, ISO 717-1)
# --------------------------------------------------------------------------- #
def test_airborne_fiche_rating_and_basis(tmp_path) -> None:
    """The DnT fiche boxes the ISO 717-1 rating and names ISO 10052."""
    result = _airborne()
    rating = result.rating
    assert rating is not None
    out = tmp_path / "dnt.pdf"
    result.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{rating.rating} ({rating.c:+d}; {rating.ctr:+d}) dB" in text
    assert "ISO 10052:2021" in text
    assert "ISO 717-1:2020" in text
    assert "survey (control) method" in text
    assert "33.0" in text  # the 125 Hz DnT band value (D + k with k = 0)


def test_airborne_r_prime_quantity(tmp_path) -> None:
    """``quantity='r_prime'`` reports the apparent sound reduction index R'."""
    result = _airborne()
    assert result.r_prime_rating is not None
    out = tmp_path / "rprime.pdf"
    result.report(str(out), quantity="r_prime")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Apparent sound reduction index" in text


def test_airborne_one_third_octave_caption(tmp_path) -> None:
    """A 16-band one-third-octave survey report captions its own band set.

    The survey method also accepts the ISO 717 one-third-octave set (16 bands,
    100 Hz to 3150 Hz); the default table caption must follow the reported band
    set, not hard-code "Octave-band".
    """
    n = 16
    l1 = np.full(n, 80.0)
    d = np.linspace(33.0, 52.0, n)
    result = building.survey_airborne_insulation(
        l1, l1 - d, np.zeros(n), volume=50.0, area=12.0
    )
    out = tmp_path / "dnt_third.pdf"
    result.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "One-third-octave" in text
    assert "Octave-band" not in text
    # The basis line also declares the actual band set, not "octave bands".
    assert "one-third-octave bands" in text
    assert "octave bands)" not in text.replace("one-third-octave bands", "")


def test_airborne_verbose_and_verdict_pass(tmp_path) -> None:
    """Verbose annexes the ISO 717 evaluation; a level difference passes above."""
    out = tmp_path / "verbose.pdf"
    _airborne().report(
        str(out), metadata=ReportMetadata(requirement=40.0), verbose=True
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "PASS" in text  # DnT,w = 44 dB >= 40 dB
    assert "Unfav. dev." in text


def test_airborne_verdict_fail(tmp_path) -> None:
    """A level difference below the requirement fails (higher is better)."""
    out = tmp_path / "fail.pdf"
    _airborne().report(str(out), metadata=ReportMetadata(requirement=60.0))
    assert "FAIL" in _extract_text(str(out))


def test_airborne_spanish(tmp_path) -> None:
    """``language='es'`` renders the Spanish survey fiche with comma decimals."""
    import re

    out = tmp_path / "es.pdf"
    _airborne().report(
        str(out),
        metadata=ReportMetadata(requirement=40.0, laboratory="Ejemplo"),
        verbose=True,
        language="es",
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "método de control" in text
    assert "CUMPLE" in text
    assert re.search(r"\d+,\d", text)


# --------------------------------------------------------------------------- #
# Impact L'nT (ISO 10052, ISO 717-2)
# --------------------------------------------------------------------------- #
def test_impact_fiche_rating_and_basis(tmp_path) -> None:
    """The L'nT fiche boxes the ISO 717-2 rating and names the tapping machine."""
    result = _impact()
    rating = result.rating
    assert rating is not None
    out = tmp_path / "lnt.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=62.0))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{rating.rating} ({rating.ci:+d}) dB" in text
    assert "ISO 10052:2021" in text
    assert "ISO 717-2:2020" in text
    assert "tapping machine" in text
    assert "PASS" in text  # L'nT,w = 58 dB <= 62 dB (a lower level is better)


def test_impact_verdict_fail(tmp_path) -> None:
    """An impact level above the requirement fails (lower is better)."""
    out = tmp_path / "fail.pdf"
    _impact().report(str(out), metadata=ReportMetadata(requirement=50.0))
    assert "FAIL" in _extract_text(str(out))


# --------------------------------------------------------------------------- #
# Facade D2m,nT (ISO 10052, ISO 717-1)
# --------------------------------------------------------------------------- #
def test_facade_fiche_rating_and_basis(tmp_path) -> None:
    """The D2m,nT fiche boxes the ISO 717-1 rating and names ISO 10052."""
    result = _facade()
    rating = result.rating
    assert rating is not None
    out = tmp_path / "d2mnt.pdf"
    result.report(str(out), metadata=ReportMetadata(requirement=33.0))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{rating.rating} ({rating.c:+d}; {rating.ctr:+d}) dB" in text
    assert "ISO 10052:2021" in text
    assert "D2m,nT" in text
    assert "PASS" in text  # D2m,nT,w >= 33 dB


def test_facade_spanish(tmp_path) -> None:
    """``language='es'`` renders the Spanish survey facade fiche."""
    out = tmp_path / "es.pdf"
    _facade().report(str(out), language="es")
    _assert_one_page(str(out))
    assert "Aislamiento acústico de fachada in situ" in _extract_text(str(out))


# --------------------------------------------------------------------------- #
# Rendering contract
# --------------------------------------------------------------------------- #
def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError`` on every survey fiche."""
    out = str(tmp_path / "x.pdf")
    airborne, impact, facade = _airborne(), _impact(), _facade()
    with pytest.raises(ValueError, match="engine"):
        airborne.report(out, engine="weasyprint")
    with pytest.raises(ValueError, match="engine"):
        impact.report(out, engine="weasyprint")
    with pytest.raises(ValueError, match="engine"):
        facade.report(out, engine="weasyprint")


def test_unknown_airborne_quantity_rejected(tmp_path) -> None:
    """An unknown airborne quantity raises ``ValueError``."""
    airborne = _airborne()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="quantity"):
        airborne.report(out, quantity="dn")


def test_missing_r_prime_rating_rejected(tmp_path) -> None:
    """Requesting R' without area/volume (no R' rating) raises ``ValueError``."""
    result = building.survey_airborne_insulation(
        np.full(_OCTAVE, 80.0), np.full(_OCTAVE, 45.0), _K0
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="rating"):
        result.report(out, quantity="r_prime")


def test_missing_rating_off_band_count_rejected(tmp_path) -> None:
    """A band count that yields no ISO 717 rating cannot be reported."""
    airborne = building.survey_airborne_insulation(
        np.full(4, 80.0), np.full(4, 45.0), np.zeros(4)
    )
    impact = building.survey_impact_insulation(np.full(4, 60.0), np.zeros(4))
    facade = building.survey_facade_insulation(
        np.full(4, 70.0), np.full(4, 40.0), np.zeros(4)
    )
    out = str(tmp_path / "x.pdf")
    for result in (airborne, impact, facade):
        with pytest.raises(ValueError, match="rating"):
            result.report(out)
