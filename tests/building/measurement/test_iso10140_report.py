#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 10140 laboratory test report (``.report()`` -> PDF).

The fiche's numbers are verified against the standards' own definitions
using hand-computable syntheses. For the airborne fiche, choosing the free
test opening area S = 10 m2 equal to the receiving-room absorption area
A = 0,16 V / T (V = 31,25 m3, T = 0,5 s give A = 10 m2) makes the term
10 lg(S/A) vanish, so R = L1 - L2 reproduces the ISO 717-1 Annex C worked
example exactly and pins the fiche's rating to the published 30 (-2; -3) dB.
For the impact fiche, the same V and T give A = A0 = 10 m2, so the term
10 lg(A/A0) vanishes, Ln = Li reproduces the ISO 717-2 Annex C worked example
and the fiche prints 79 (-11) dB. Rendering assertions are structural (a
one-page ``%PDF``) plus pypdf text-extraction checks of the boxed rating,
the standard-basis line and a couple of per-band values, like the sibling
report tests.
"""

from __future__ import annotations

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

#: Receiving-room reverberation time and volume that make the equivalent
#: absorption area A = 0,16 V / T equal to 10 m2 in every band, so both the
#: airborne 10 lg(S/A) (with S = 10 m2) and the impact 10 lg(A/A0) (with
#: A0 = 10 m2) vanish and the reported quantity equals its raw input.
_T_UNITY = np.full(16, 0.5)
_V_UNITY = 31.25


def _airborne_result(**kwargs) -> building.LabAirborneInsulationResult:
    """A lab airborne result whose R equals the ISO 717-1 Annex C curve."""
    l1 = np.full(16, 90.0)
    l2 = l1 - np.asarray(_AIRBORNE_R, dtype=np.float64)
    params = {"area": 10.0, "volume": _V_UNITY, **kwargs}
    return building.lab_airborne_insulation(l1, l2, _T_UNITY, **params)


def _impact_result(**kwargs) -> building.LabImpactInsulationResult:
    """A lab impact result whose Ln equals the ISO 717-2 Annex C curve."""
    li = np.asarray(_IMPACT_LN, dtype=np.float64)
    params = {"volume": _V_UNITY, **kwargs}
    return building.lab_impact_insulation(li, _T_UNITY, **params)


def _extract_text(path: str) -> str:
    """The concatenated text of every page."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


def test_r_equals_level_difference_at_unity_absorption() -> None:
    """With S = A the 10 lg(S/A) term vanishes: R = L1 - L2 exactly."""
    result = _airborne_result()
    assert np.allclose(result.r, _AIRBORNE_R)
    assert np.allclose(result.absorption, 10.0)


def test_airborne_fiche_rating_pinned_to_iso717_1_annex_c(tmp_path) -> None:
    """The R fiche's rating is the published ISO 717-1 Annex C 30 (-2; -3) dB."""
    out = tmp_path / "r.pdf"
    _airborne_result().report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "30 (-2; -3) dB" in text
    assert "ISO 10140-2:2010" in text
    assert "ISO 717-1:2020" in text
    # Nominal band labels and a couple of R values from the table.
    assert "100" in text
    assert "3150" in text
    assert "26.6" in text  # the 500 Hz band value


def test_ln_equals_impact_level_at_reference_absorption() -> None:
    """With A = A0 the 10 lg(A/A0) term vanishes: Ln = Li exactly."""
    result = _impact_result()
    assert np.allclose(result.l_n, _IMPACT_LN)
    assert np.allclose(result.absorption, 10.0)


def test_impact_fiche_rating_pinned_to_iso717_2_annex_c(tmp_path) -> None:
    """The Ln fiche's rating is the published ISO 717-2 Annex C 79 (-11) dB."""
    out = tmp_path / "ln.pdf"
    _impact_result().report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    expected = f"{_IMPACT_EXPECTED['ln_w']} ({_IMPACT_EXPECTED['ci']:+d}) dB"
    assert expected in text
    assert "ISO 10140-3:2010" in text
    assert "ISO 717-2:2020" in text
    assert "73.1" in text  # the 500 Hz band value
    # pypdf may wrap the extracted basis line mid-phrase; normalise first.
    assert "tapping machine" in " ".join(text.split())


def test_full_metadata_and_verbose_render_one_page(tmp_path) -> None:
    """A fully populated lab fiche with the absorption column is one page."""
    metadata = ReportMetadata(
        specimen="100 mm autoclaved aerated concrete block wall",
        client="Acoustic Test Client Ltd.",
        manufacturer="Example blockworks",
        mounted_by="Example laboratory",
        area=10.0,
        mass_per_area=75.0,
        source_volume=53.0,
        receiving_volume=51.0,
        source_temperature=21.4,
        source_relative_humidity=44.0,
        receiving_temperature=20.8,
        receiving_relative_humidity=46.0,
        pressure=101.3,
        test_room="Transmission suite (example)",
        mounting="Type A mounting (ISO 10140-1)",
        measurement_standard="ISO 10140-2",
        test_date="2026-07-20",
        laboratory="Phonometry Reference Laboratory",
        operator="Jose Manuel Requena Plens",
        report_id="PHN-2026-0143",
        requirement=25.0,
    )
    out = tmp_path / "verbose.pdf"
    _airborne_result().report(str(out), metadata=metadata, verbose=True)
    assert_one_page(str(out))
    text = " ".join(_extract_text(str(out)).split())
    assert "PASS" in text  # Rw = 30 dB >= 25 dB
    assert "100 mm autoclaved aerated concrete" in text
    assert "precision method" in text


def test_requirement_verdicts_pass_and_fail(tmp_path) -> None:
    """Airborne passes at/above the requirement; impact at/below it."""
    airborne = _airborne_result()  # Rw = 30 dB
    failing = tmp_path / "fail.pdf"
    airborne.report(str(failing), metadata=ReportMetadata(requirement=50.0))
    assert "FAIL" in _extract_text(str(failing))

    impact = _impact_result()  # Ln,w = 79 dB
    passing = tmp_path / "impact_pass.pdf"
    impact.report(str(passing), metadata=ReportMetadata(requirement=80.0))
    assert "PASS" in _extract_text(str(passing))


def test_octave_band_fiche_renders(tmp_path) -> None:
    """A five-octave-band laboratory result also yields a one-page fiche."""
    l1 = np.full(5, 90.0)
    r = np.array([30.0, 40.0, 48.0, 52.0, 55.0])
    result = building.lab_airborne_insulation(
        l1, l1 - r, np.full(5, 0.5), area=10.0, volume=_V_UNITY
    )
    assert result.rating is not None
    out = tmp_path / "octave.pdf"
    result.report(str(out))
    assert_one_page(str(out))
    assert "Octave-band" in _extract_text(str(out))


def test_spanish_fiche_renders_translated(tmp_path) -> None:
    """``language="es"`` renders the Spanish lab fiche with comma decimals."""
    import re

    out = tmp_path / "es.pdf"
    _airborne_result().report(
        str(out),
        metadata=ReportMetadata(requirement=25.0, laboratory="Ejemplo"),
        verbose=True,
        language="es",
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "CUMPLE" in text
    assert re.search(r"\d+,\d", text)  # comma decimal separator


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    out = str(tmp_path / "x.pdf")
    airborne = _airborne_result()
    impact = _impact_result()
    with pytest.raises(ValueError, match="engine"):
        airborne.report(out, engine="weasyprint")
    with pytest.raises(ValueError, match="engine"):
        impact.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unsupported language raises ``ValueError`` (shared validation path)."""
    out = str(tmp_path / "x.pdf")
    airborne = _airborne_result()
    impact = _impact_result()
    with pytest.raises(ValueError, match="language"):
        airborne.report(out, language="fr")
    with pytest.raises(ValueError, match="language"):
        impact.report(out, language="fr")


def test_missing_rating_rejected(tmp_path) -> None:
    """A result without the ISO 717 rating (non-core band count) is rejected."""
    l1 = np.full(8, 90.0)
    result = building.lab_airborne_insulation(
        l1, np.full(8, 50.0), np.full(8, 0.5), area=10.0, volume=_V_UNITY
    )
    assert result.rating is None
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="rating"):
        result.report(out)


def test_rating_without_per_band_data_rejected(tmp_path) -> None:
    """A manually built rating lacking the per-band arrays is rejected."""
    # A backward-compatibly constructed rating (band_centers / measured /
    # shifted_reference default to None) would otherwise crash the table.
    bare_rating = building.WeightedRatingResult(
        rating=30, c=-2, ctr=-3, unfavourable_sum=0.0
    )
    result = building.LabAirborneInsulationResult(
        r=np.asarray(_AIRBORNE_R, dtype=np.float64),
        absorption=np.full(16, 10.0),
        rating=bare_rating,
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="per-band"):
        result.report(out)


def test_manual_impact_result_renders(tmp_path) -> None:
    """A manually built lab impact result reports its Ln fiche."""
    result = _impact_result()
    bare = building.LabImpactInsulationResult(
        l_n=result.l_n, absorption=result.absorption, rating=result.rating
    )
    out = tmp_path / "bare.pdf"
    bare.report(str(out))
    assert_one_page(str(out))


def test_report_rejects_band_count_mismatch(tmp_path) -> None:
    """A rating whose per-band arrays are shorter than the curve raises a
    ValueError naming each array and its shape (not an uncaught IndexError).
    """
    import dataclasses

    res = _airborne_result()
    assert res.rating is not None
    short = dataclasses.replace(
        res.rating,
        band_centers=res.rating.band_centers[:-1],
        measured=res.rating.measured[:-1],
        shifted_reference=res.rating.shifted_reference[:-1],
    )
    bad = dataclasses.replace(res, rating=short)
    out = str(tmp_path / "x.pdf")
    with pytest.raises(
        ValueError,
        match=r"LabAirborneInsulationResult\.report: .*'rating\.band_centers' "
        r".* must all have the same shape, one value per band",
    ):
        bad.report(out)
