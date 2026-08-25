#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 16283 field test report (``.report()`` -> PDF).

The fiche's numbers are verified against the standards' own definitions
using hand-computable syntheses: with ``T = T0 = 0,5 s`` the reverberation
correction vanishes exactly (``DnT = D``, ``L'nT = Li``), so feeding the
ISO 717 Annex C worked-example curves as the level difference (or impact
level) pins the fiche's rating to the published ``30 (-2; -3)`` / ``79
(-11)`` values; ``R'`` is checked against ``D + 10 lg(S T / (0,16 V))``.
Rendering assertions are structural (a one-page ``%PDF``) plus pypdf
text-extraction checks of the boxed rating, the standard-basis line and
the mandatory field-method statement, like the sibling report tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("reportlab")

from reference_data import (
    ISO717_1_ANNEX_C_R as _AIRBORNE_R,
)
from reference_data import (
    ISO717_2_ANNEX_C1_EXPECTED as _IMPACT_EXPECTED,
)
from reference_data import (
    ISO717_2_ANNEX_C1_LN as _IMPACT_LN,
)
from report_assertions import assert_one_page

from phonometry import ReportMetadata, building

if TYPE_CHECKING:
    from pathlib import Path

#: A receiving-room reverberation time equal to T0 = 0,5 s in every band:
#: the standardized quantities then equal the raw ones exactly
#: (10 lg(T/T0) = 0), making the fiche's rating hand-computable.
_T_AT_T0 = np.full(16, 0.5)


def _airborne_result(**kwargs: float) -> building.AirborneInsulationResult:
    """A field airborne result whose ``DnT`` equals the ISO 717-1 Annex C curve."""
    l1 = np.full(16, 90.0)
    l2 = l1 - np.asarray(_AIRBORNE_R, dtype=np.float64)
    return building.airborne_insulation(l1, l2, _T_AT_T0, **kwargs)


def _extract_text(path: str) -> str:
    """The concatenated text of every page."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_dnt_equals_d_at_reference_time() -> None:
    """With T = T0 the reverberation correction vanishes: DnT = D exactly."""
    result = _airborne_result()
    assert np.allclose(result.dnt, result.d)
    assert np.allclose(result.d, _AIRBORNE_R)


def test_dnt_reverberation_correction_hand_computed() -> None:
    """DnT = D + 10 lg(T/T0) (ISO 16283-1 Formula (2)) for T = 1,0 s."""
    l1 = np.full(16, 90.0)
    l2 = l1 - np.asarray(_AIRBORNE_R, dtype=np.float64)
    result = building.airborne_insulation(l1, l2, np.full(16, 1.0))
    assert np.allclose(result.dnt, result.d + 10.0 * np.log10(2.0))


def test_airborne_fiche_rating_pinned_to_iso717_1_annex_c(tmp_path: Path) -> None:
    """The DnT fiche's rating is the published ISO 717-1 Annex C 30 (-2; -3) dB.

    ``DnT = D`` here by construction, so the fiche must print the exact
    worked-example rating triplet in its boxed result.
    """
    out = tmp_path / "dnt.pdf"
    _airborne_result().report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "30 (-2; -3) dB" in text
    assert "ISO 16283-1:2014" in text
    assert "ISO 717-1:2020" in text
    # The mandatory field-method statement (Clause 14 / Annex B form).
    assert "engineering method" in text


def test_r_prime_fiche_hand_computed(tmp_path: Path) -> None:
    """R' = D + 10 lg(S T / (0,16 V)) (Formulae (4)/(5)) drives the R' fiche.

    With S = 10 m2, V = 50 m3 and T = 0,5 s the absorption area is
    A = 0,16 x 50 / 0,5 = 16 m2, so R' = D + 10 lg(10/16) = D - 2,041 dB
    in every band.
    """
    result = _airborne_result(area=10.0, volume=50.0)
    assert result.r_prime is not None
    expected = result.d + 10.0 * np.log10(10.0 / 16.0)
    assert np.allclose(result.r_prime, expected)
    out = tmp_path / "r_prime.pdf"
    result.report(str(out), quantity="r_prime")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Apparent sound reduction index" in text


def test_impact_lnt_equals_li_at_reference_time() -> None:
    """With T = T0 the impact correction vanishes: L'nT = Li exactly."""
    result = building.impact_insulation(_IMPACT_LN, _T_AT_T0)
    assert np.allclose(result.l_n_t, _IMPACT_LN)


def test_impact_fiche_rating_pinned_to_iso717_2_annex_c(tmp_path: Path) -> None:
    """The L'nT fiche's rating is the published ISO 717-2 Annex C 79 (-11) dB."""
    result = building.impact_insulation(_IMPACT_LN, _T_AT_T0, volume=31.25)
    out = tmp_path / "lnt.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    expected = f"{_IMPACT_EXPECTED['ln_w']} ({_IMPACT_EXPECTED['ci']:+d}) dB"
    assert expected in text
    assert "ISO 16283-2:2020" in text
    assert "ISO 717-2:2020" in text
    # pypdf may wrap the extracted basis line mid-phrase; normalise first.
    assert "tapping machine" in " ".join(text.split())


def test_impact_l_n_fiche_hand_computed(tmp_path: Path) -> None:
    """L'n = Li + 10 lg(A/A0) with A = 0,16 V/T (Formulae (2)/(6)).

    V = 31,25 m3 and T = 0,5 s give A = 10 m2 = A0, so L'n = Li and the
    normalized fiche prints the same Annex C rating as the standardized one.
    """
    result = building.impact_insulation(_IMPACT_LN, _T_AT_T0, volume=31.25)
    assert result.l_n is not None
    assert np.allclose(result.l_n, _IMPACT_LN)
    out = tmp_path / "ln.pdf"
    result.report(str(out), quantity="l_n")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{_IMPACT_EXPECTED['ln_w']} ({_IMPACT_EXPECTED['ci']:+d}) dB" in text
    assert "Normalized impact sound pressure level" in text


def test_full_metadata_and_verbose_render_one_page(tmp_path: Path) -> None:
    """A fully populated field fiche with the measurement chain is one page."""
    result = _airborne_result(area=12.5, volume=30.4)
    metadata = ReportMetadata(
        specimen="Separating wall, 240 mm brick with independent lining",
        client="Acoustic Test Client Ltd.",
        area=12.5,
        source_volume=32.1,
        receiving_volume=30.4,
        temperature=20.4,
        relative_humidity=52.0,
        test_room="Dwelling A living room to dwelling B living room",
        test_date="2026-07-20",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-0143",
        requirement=25.0,
    )
    out = tmp_path / "verbose.pdf"
    result.report(str(out), metadata=metadata, verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "PASS" in text  # DnT,w = 30 dB >= 25 dB


def test_requirement_verdicts_pass_and_fail(tmp_path: Path) -> None:
    """Airborne passes at/above the requirement; impact at/below it."""
    airborne = _airborne_result()  # DnT,w = 30 dB
    failing = tmp_path / "fail.pdf"
    airborne.report(str(failing), metadata=ReportMetadata(requirement=50.0))
    assert "FAIL" in _extract_text(str(failing))

    impact = building.impact_insulation(_IMPACT_LN, _T_AT_T0)  # L'nT,w = 79 dB
    passing = tmp_path / "impact_pass.pdf"
    impact.report(str(passing), metadata=ReportMetadata(requirement=80.0))
    assert "PASS" in _extract_text(str(passing))


def test_spanish_fiche_renders_translated(tmp_path: Path) -> None:
    """``language="es"`` renders the Spanish field fiche with comma decimals."""
    import re

    result = _airborne_result()
    out = tmp_path / "es.pdf"
    result.report(
        str(out),
        metadata=ReportMetadata(requirement=25.0, laboratory="Ejemplo"),
        verbose=True,
        language="es",
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "in situ" in text
    assert "CUMPLE" in text
    assert re.search(r"\d+,\d", text)  # comma decimal separator


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    result = _airborne_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_quantity_rejected(tmp_path: Path) -> None:
    """An unknown field quantity raises ``ValueError``."""
    airborne = _airborne_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="quantity"):
        airborne.report(out, quantity="dn")
    impact = building.impact_insulation(_IMPACT_LN, _T_AT_T0)
    with pytest.raises(ValueError, match="quantity"):
        impact.report(out, quantity="dnt")


def test_missing_r_prime_rejected(tmp_path: Path) -> None:
    """Requesting the R' fiche without area/volume raises ``ValueError``."""
    result = _airborne_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="r_prime"):
        result.report(out, quantity="r_prime")


def test_non_core_band_count_rejected(tmp_path: Path) -> None:
    """The field fiche needs the 16 core one-third-octave bands."""
    result = building.airborne_insulation(
        np.full(5, 90.0), np.full(5, 50.0), np.full(5, 0.5)
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="16 core"):
        result.report(out)


def test_verbose_needs_measurement_chain(tmp_path: Path) -> None:
    """``verbose=True`` on a manually built result (no chain) is rejected."""
    curve = np.asarray(_AIRBORNE_R, dtype=np.float64)
    bare = building.AirborneInsulationResult(d=curve, dnt=curve, r_prime=None)
    rejected = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="measurement chain"):
        bare.report(rejected, verbose=True)
    # The non-verbose form still renders from the quantity alone.
    out = tmp_path / "bare.pdf"
    bare.report(str(out))
    assert_one_page(str(out))


def test_manual_impact_result_renders_without_chain(tmp_path: Path) -> None:
    """A backward-compatibly built impact result reports its L'nT fiche."""
    curve = np.asarray(_IMPACT_LN, dtype=np.float64)
    bare = building.ImpactInsulationResult(l_n_t=curve, l_n=None)
    out = tmp_path / "bare_impact.pdf"
    bare.report(str(out))
    assert_one_page(str(out))
    rejected = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="measurement chain"):
        bare.report(rejected, verbose=True)
