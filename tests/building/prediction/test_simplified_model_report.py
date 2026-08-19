#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the EN/ISO 12354-1/-2 predicted-insulation reports (``.report()``).

The prediction fiches are pinned to the standards' own worked examples, run
through the tested prediction code: EN 12354-1 Annex H.3 (airborne, a
separating wall with four flanking elements, predicted ``R'w`` = 52 dB) and EN
12354-2 Annex E.3 (impact, a concrete floor with a floating floor, predicted
``L'n,w`` = 45 dB). The rendering assertions are structural (a one-page
``%PDF``) plus pypdf text-extraction checks of the boxed single number, the
model-term labels, the mandatory "prediction / not a measurement" wording and a
couple of the model values, like the sibling report tests.
"""

from __future__ import annotations

import pytest
import reference_data as ref

pytest.importorskip("reportlab")

from phonometry import ReportMetadata, building
from phonometry._i18n import fmt_minus
from phonometry.building.prediction.facade import (
    FacadePredictionResult,
)
from phonometry.building.prediction.simplified_model import (
    AirbornePredictionResult,
    ImpactPredictionResult,
)

_PDF_MAGIC = b"%PDF"


def _annex_h3_airborne() -> AirbornePredictionResult:
    """The EN 12354-1 Annex H.3 airborne prediction (R'w = 52 dB)."""
    ss = ref.EN12354_1_ANNEX_H3_SEPARATING_AREA
    r_direct = ref.EN12354_1_ANNEX_H3_R_DIRECT
    paths = []
    for label, rw, kff, kfd, lf in ref.EN12354_1_ANNEX_H3_ELEMENTS:
        ff, df, fd = building.flanking_element(
            label=label, r_flanking=rw, r_separating=r_direct,
            k_ff=kff, k_fd=kfd, k_df=kfd, separating_area=ss, coupling_length=lf,
        )
        paths += [ff, df, fd]
    return building.predicted_airborne_insulation(
        r_direct=r_direct, flanking_paths=paths
    )


def _annex_e3_impact() -> ImpactPredictionResult:
    """The EN 12354-2 Annex E.3 impact prediction (L'n,w = 45 dB)."""
    ln_w_eq = building.equivalent_impact_level(ref.EN12354_2_ANNEX_E3_MASS)
    k = building.impact_flanking_correction(
        ref.EN12354_2_ANNEX_E3_MASS, ref.EN12354_2_ANNEX_E3_FLANKING_MEAN_MASS
    )
    return building.predicted_impact_insulation(
        ln_w_eq=ln_w_eq, delta_l_w=ref.EN12354_2_ANNEX_E3_DELTA_LW, k_correction=k
    )


def _annex_f_facade() -> FacadePredictionResult:
    """The EN 12354-3 Annex F facade prediction (D2m,nT,w = 33 dB)."""
    elements = [
        building.FacadeElement(name=name, area=area, r=r)
        for name, area, r in ref.EN12354_3_ANNEX_F_ELEMENTS
    ]
    elements.append(
        building.FacadeElement(
            name="air inlet", dn_e=ref.EN12354_3_ANNEX_F_INLET_DNE
        )
    )
    return building.facade_sound_reduction(
        elements,
        area=ref.EN12354_3_ANNEX_F_AREA,
        volume=ref.EN12354_3_ANNEX_F_VOLUME,
        frequencies=ref.EN12354_3_ANNEX_F_BANDS,
        bands="octave",
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


def test_airborne_prediction_rating_pinned_to_annex_h3(tmp_path) -> None:
    """The airborne fiche boxes the Annex H.3 predicted R'w = 52 dB."""
    out = tmp_path / "air.pdf"
    # report() returns the written path (part of the public contract).
    assert _annex_h3_airborne().report(str(out)) == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{ref.EN12354_1_ANNEX_H3_RPRIME_W} dB" in text  # boxed R'w = 52 dB
    assert "EN 12354-1:2000" in text
    assert "ISO 717-1" in text
    # Explicitly a prediction, not a measurement, down to the footer
    # disclaimer (no tested specimen on a prediction sheet).
    assert "prediction" in text
    assert "not a measurement" in text
    assert "modelled configuration" in text
    assert "tested specimen" not in text.replace("not to a tested specimen", "")
    # The direct path and a couple of flanking-path indices from the table.
    assert "Dd" in text
    assert "65.5" in text  # floor-Ff Rij,w
    assert "73" in text  # intwall-Ff Rij,w


def test_impact_prediction_rating_pinned_to_annex_e3(tmp_path) -> None:
    """The impact fiche boxes the Annex E.3 predicted L'n,w = 45 dB."""
    out = tmp_path / "imp.pdf"
    assert _annex_e3_impact().report(str(out)) == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{ref.EN12354_2_ANNEX_E3_LPRIME_N_W} dB" in text  # boxed L'n,w = 45
    assert "EN 12354-2:2000" in text
    assert "ISO 717-2" in text
    assert "prediction" in text
    assert "not a measurement" in text
    # Formula (21) terms: the bare-floor level and the covering improvement.
    assert "76.2" in text  # Ln,w,eq = 164 - 35 lg(322)
    assert "33" in text  # covering improvement DLw
    assert "Formula (21)" in text


def test_airborne_verbose_adds_energy_share(tmp_path) -> None:
    """``verbose=True`` annexes each path's share of the transmitted energy."""
    plain = tmp_path / "plain.pdf"
    _annex_h3_airborne().report(str(plain))
    assert "%" not in _extract_text(str(plain))

    verbose = tmp_path / "verbose.pdf"
    _annex_h3_airborne().report(str(verbose), verbose=True)
    text = _extract_text(str(verbose))
    assert "%" in text  # per-path energy share column
    assert "energy share" in text  # the verbose caption


def test_airborne_requirement_verdict(tmp_path) -> None:
    """Airborne insulation passes at or above the requirement."""
    result = _annex_h3_airborne()  # R'w = 52 dB
    passing = tmp_path / "pass.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=50.0))
    assert "PASS" in _extract_text(str(passing))

    failing = tmp_path / "fail.pdf"
    result.report(str(failing), metadata=ReportMetadata(requirement=55.0))
    assert "FAIL" in _extract_text(str(failing))


def test_impact_requirement_verdict_lower_is_better(tmp_path) -> None:
    """Impact level passes at or below the requirement (lower is better)."""
    result = _annex_e3_impact()  # L'n,w = 45 dB
    passing = tmp_path / "pass.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=53.0))
    assert "PASS" in _extract_text(str(passing))

    failing = tmp_path / "fail.pdf"
    result.report(str(failing), metadata=ReportMetadata(requirement=40.0))
    assert "FAIL" in _extract_text(str(failing))


def test_metadata_header_renders(tmp_path) -> None:
    """A populated metadata header prints on the fiche."""
    metadata = ReportMetadata(
        specimen="Separating wall Rs,w = 57 dB",
        client="Acoustic Consultants Ltd.",
        area=11.5,
        source_volume=53.0,
        receiving_volume=50.0,
        laboratory="Phonometry Reference Laboratory",
        report_id="PHN-2026-0200",
        notes="Flanking: floor/ceiling/facade/internal wall.",
    )
    out = tmp_path / "meta.pdf"
    _annex_h3_airborne().report(str(out), metadata=metadata)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Separating wall Rs,w = 57 dB" in text
    assert "Phonometry Reference Laboratory" in text


def test_lightweight_fiche_without_metadata(tmp_path) -> None:
    """A fiche with no metadata is still a valid one-page prediction report."""
    out = tmp_path / "bare.pdf"
    _annex_e3_impact().report(str(out))
    _assert_one_page(str(out))
    assert "prediction" in _extract_text(str(out))


def test_spanish_fiche_renders_translated(tmp_path) -> None:
    """``language="es"`` renders the Spanish fiche with comma decimals."""
    import re

    out = tmp_path / "es.pdf"
    _annex_h3_airborne().report(
        str(out),
        metadata=ReportMetadata(requirement=50.0, laboratory="Ejemplo"),
        verbose=True,
        language="es",
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "CUMPLE" in text  # R'w = 52 dB >= 50 dB
    assert "previsto" in text  # "predicted"
    assert "no una medici" in text  # "not a measurement"
    assert re.search(r"\d+,\d", text)  # comma decimal separator


def test_impact_spanish_fiche_renders_translated(tmp_path) -> None:
    """The impact fiche also renders in Spanish."""
    out = tmp_path / "es_imp.pdf"
    _annex_e3_impact().report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "previsto" in text
    assert "rmula (21)" in text  # "fórmula (21)"


def test_facade_prediction_rating_pinned_to_annex_f(tmp_path) -> None:
    """The facade fiche boxes the Annex F predicted D2m,nT,w = 33 dB."""
    out = tmp_path / "facade.pdf"
    assert _annex_f_facade().report(str(out)) == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{ref.EN12354_3_ANNEX_F_D2MNT_W} dB" in text  # boxed D2m,nT,w = 33 dB
    assert "D2m,nT,w" in text
    assert "EN 12354-3:2000" in text
    assert "ISO 717-1" in text
    # Explicitly a prediction, not a measurement.
    assert "prediction" in text
    assert "not a measurement" in text
    # Key model terms: the standardized level difference and the elements' R'.
    assert "D2m,nT" in text
    assert "Formula 13" in text
    # The apparent-index values are stated, not just their labels (Annex F).
    assert f"{ref.EN12354_3_ANNEX_F_RTRS_W} dB" in text  # R'tr,s,w = 31 dB
    # Signed through the same helper the fiche prints with, so the expectation
    # cannot drift from the sign the document carries.
    assert f"{fmt_minus(ref.EN12354_3_ANNEX_F_CTR)} dB" in text  # Ctr = −3 dB
    # A couple of the per-element weighted partial indices Rp,w from the table.
    assert "wall" in text
    assert "59" in text  # wall Rp,w
    assert "37" in text  # window Rp,w


def test_facade_verbose_adds_energy_share(tmp_path) -> None:
    """``verbose=True`` annexes each element's share of the transmitted energy."""
    plain = tmp_path / "plain.pdf"
    _annex_f_facade().report(str(plain))
    assert "%" not in _extract_text(str(plain))

    verbose = tmp_path / "verbose.pdf"
    _annex_f_facade().report(str(verbose), verbose=True)
    text = _extract_text(str(verbose))
    assert "%" in text  # per-element energy share column
    assert "energy share" in text  # the verbose caption


def test_facade_requirement_verdict(tmp_path) -> None:
    """Facade insulation passes at or above the requirement."""
    result = _annex_f_facade()  # D2m,nT,w = 33 dB
    passing = tmp_path / "pass.pdf"
    result.report(str(passing), metadata=ReportMetadata(requirement=30.0))
    assert "PASS" in _extract_text(str(passing))

    failing = tmp_path / "fail.pdf"
    result.report(str(failing), metadata=ReportMetadata(requirement=40.0))
    assert "FAIL" in _extract_text(str(failing))


def test_facade_lightweight_fiche_without_metadata(tmp_path) -> None:
    """A facade fiche with no metadata is still a valid one-page prediction."""
    out = tmp_path / "bare.pdf"
    _annex_f_facade().report(str(out))
    _assert_one_page(str(out))
    assert "prediction" in _extract_text(str(out))


def test_facade_spanish_fiche_renders_translated(tmp_path) -> None:
    """``language="es"`` renders the Spanish facade fiche."""
    out = tmp_path / "es.pdf"
    _annex_f_facade().report(
        str(out),
        metadata=ReportMetadata(requirement=30.0, laboratory="Ejemplo"),
        verbose=True,
        language="es",
    )
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "CUMPLE" in text  # D2m,nT,w = 33 dB >= 30 dB
    assert "previsto" in text  # "predicted"
    assert "no una medici" in text  # "not a measurement"
    assert "fachada" in text  # "facade"


def test_facade_report_requires_single_number_ratings(tmp_path) -> None:
    """A facade result without the ISO 717-1 ratings cannot be reported."""
    # Three bands: not the 5 octave / 16 one-third-octave set, so no rating.
    result = building.facade_sound_reduction(
        [building.FacadeElement(name="wall", area=10.0, r=[40.0, 41.0, 42.0])],
        area=10.0, volume=50.0,
    )
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="single-number"):
        result.report(out)


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ``ValueError``."""
    out = str(tmp_path / "x.pdf")
    result = _annex_h3_airborne()
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unsupported language raises ``ValueError`` (shared validation path)."""
    out = str(tmp_path / "x.pdf")
    result = _annex_e3_impact()
    with pytest.raises(ValueError, match="language"):
        result.report(out, language="fr")
