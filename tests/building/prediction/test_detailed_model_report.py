#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 12354-1/-2:2017 detailed prediction fiches (``.report()``).

Both fiches are pinned to the per-band worked example the two parts share
(ISO 12354-1 Annex L / ISO 12354-2 Annex G, the heavy homogeneous building
built by :mod:`iso12354_building`), run through the tested detailed-model code:
predicted ``R'w`` = 57 dB and ``L'n,w`` = 41 dB. The rendering assertions are
structural (a one-page ``%PDF``) plus pypdf text-extraction checks of the boxed
single number, the path table, the mandatory "prediction / not a measurement"
wording and the detailed model's own accuracy statement.
"""

from __future__ import annotations

import iso12354_building as bld
import numpy as np
import pytest
import reference_data as ref

pytest.importorskip("reportlab")

from report_assertions import assert_one_page

from phonometry import ReportMetadata, building
from phonometry._report.iso12354 import (
    render_iso12354_detailed_airborne_report,
    render_iso12354_detailed_impact_report,
)

_BANDS = np.asarray(ref.ISO12354_ANNEX_L_BANDS, dtype=np.float64)


def _situ() -> tuple[dict, np.ndarray]:
    """The Annex L building in situ, plus the floating floor's improvement."""
    situ = {k: building.in_situ_element(e, _BANDS) for k, e in bld.elements().items()}
    delta = building.floating_floor_improvement(
        _BANDS, resonance_frequency=bld.floating_floor_resonance()
    )
    return situ, delta


def _annex_l_airborne():
    """The Annex L detailed airborne prediction (R'w = 57 dB)."""
    situ, delta = _situ()
    return building.detailed_airborne_prediction(
        _BANDS,
        direct_index=building.direct_reduction_index(
            situ["floor"].sound_reduction_index, delta_r_source=delta
        ),
        flanking_paths=bld.airborne_paths(situ, delta),
    )


def _annex_g_impact():
    """The Annex G detailed impact prediction (L'n,w = 41 dB)."""
    situ, delta = _situ()
    return building.detailed_impact_prediction(
        _BANDS,
        direct_level=building.direct_impact_level(
            situ["floor"].impact_level, delta_l=delta
        ),
        flanking_paths=bld.impact_paths(situ, delta),
    )


def _extract_text(path: str) -> str:
    """The concatenated, whitespace-normalised text of every page."""
    from pypdf import PdfReader

    return " ".join(
        "\n".join(page.extract_text() for page in PdfReader(path).pages).split()
    )


def test_detailed_airborne_fiche_boxes_the_annex_l_rating(tmp_path) -> None:
    """The detailed airborne fiche boxes R'w = 57 dB and names Clause 4.2."""
    out = tmp_path / "air.pdf"
    assert _annex_l_airborne().report(str(out)) == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{ref.ISO12354_ANNEX_L1_R_PRIME_W} dB" in text
    assert "ISO 12354-1:2017" in text
    assert "ISO 717-1" in text
    assert "frequency band" in text
    assert "prediction" in text
    assert "not a measurement" in text
    # The detailed model's own accuracy statement, with the decimal separator
    # of the fiche language (a period in English, a comma in Spanish).
    assert "1.5 dB to 2.5 dB" in text
    # The two paths that dominate the spectrum appear in the path table.
    assert "Dd" in text
    assert "2d" in text


def test_detailed_impact_fiche_boxes_the_annex_g_rating(tmp_path) -> None:
    """The detailed impact fiche boxes L'n,w = 41 dB and verdicts a limit."""
    out = tmp_path / "imp.pdf"
    result = _annex_g_impact()
    assert result.report(
        str(out),
        metadata=ReportMetadata(
            specimen="220 mm concrete floor with floating screed",
            area=20.0,
            receiving_volume=55.0,
            requirement=50.0,
        ),
        verbose=True,
    ) == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert f"{ref.ISO12354_ANNEX_G1_L_PRIME_N_W} dB" in text
    assert "ISO 12354-2:2017" in text
    assert "ISO 717-2" in text
    assert "PASS" in text  # 41 dB against the 50 dB requirement
    assert "not a measurement" in text
    assert "Df1" in text


def test_spanish_detailed_fiche_renders(tmp_path) -> None:
    """The fiches translate to Spanish like every other report."""
    out = tmp_path / "air_es.pdf"
    _annex_l_airborne().report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "ISO 12354-1:2017" in text
    assert "57 dB" in text
    # The basis line, the path caption and the accuracy statement are
    # translated, not left in English.
    assert "banda de frecuencia" in text
    assert "Vías de transmisión" in text
    assert "1,5 dB a 2,5 dB" in text
    assert "not a measurement" not in text


def test_detailed_fiche_rejects_an_unrated_spectrum(tmp_path) -> None:
    """Without the ISO 717 band range there is no single number to box."""
    partial = building.detailed_airborne_prediction(
        _BANDS[:5], direct_index=np.full(5, 55.0)
    )
    assert partial.rating is None
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="ISO 717"):
        partial.report(out)


def test_detailed_fiche_rejects_an_unknown_engine(tmp_path) -> None:
    """Only the reportlab back end exists."""
    result = _annex_g_impact()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        result.report(out, engine="weasyprint")


# --------------------------------------------------------------------------
# The legend the plot builds is the legend the fiche prints
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("build", "render", "rated"),
    [
        (_annex_l_airborne, render_iso12354_detailed_airborne_report, r"R^{\prime}"),
        (_annex_g_impact, render_iso12354_detailed_impact_report, r"L^{\prime}"),
    ],
    ids=["airborne", "impact"],
)
def test_the_fiche_legend_keeps_the_curve_it_rates(
    tmp_path, monkeypatch, build, render, rated
) -> None:
    """The rated curve is drawn on a twin axis and must survive the rebuild.

    The fiche moves the legend above the axes, which means removing the one
    the plot built and making it again. Collecting the handles from the
    primary axes alone dropped the twin's half, so the sheet printed a legend
    without the very quantity the whole page rates.
    """
    from matplotlib.axes import Axes

    legends: list[list[str]] = []
    original = Axes.legend

    def record(self, *args, **kwargs):
        drawn = original(self, *args, **kwargs)
        legends.append([text.get_text() for text in drawn.get_texts()])
        return drawn

    monkeypatch.setattr(Axes, "legend", record)
    render(build(), str(tmp_path / "legend.pdf"))

    assert len(legends) == 2, "the plot builds one legend and the fiche rebuilds it"
    assert any(rated in entry for entry in legends[0]), "the plot labels its curve"
    assert legends[1] == legends[0], "the rebuilt legend says what the plot said"
