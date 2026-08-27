#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the duct-borne noise path ``.report()`` fiche.

The sheet is rendered from the published supply path of Long, *Architectural
Acoustics* 2nd ed., Table 14.9, and the printed cells are checked against that
sheet's own numbers: the fan row, an element attenuation printed with the
worksheet's negative sign, the regenerated-noise row of the silencer and the
received spectrum, plus the NC 30 criterion curve of ANSI/ASA S12.2-2019
Table 1. Values are read back via pypdf text extraction; structural facts
complete the rendering contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata
from phonometry.noise_control.duct_path import DuctElement, DuctPathResult, duct_path
from phonometry.noise_control.hvac import OCTAVE_BANDS

if TYPE_CHECKING:
    from pathlib import Path


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _supply() -> DuctPathResult:
    """The supply path of Long Table 14.9, from its published element rows."""
    return duct_path(
        OCTAVE_BANDS,
        [90.0, 86.0, 82.0, 79.0, 77.0, 75.0, 71.0, 61.0],
        [
            DuctElement(
                "Elbow, 36 x 24 in, unlined",
                [0, 1, 2, 3, 3, 3, 3, 3],
                [41, 39, 36, 29, 20, 6, 0, 0],
                code="2",
            ),
            DuctElement(
                "Silencer, standard pressure drop, 3 ft",
                [7, 12, 16, 28, 35, 35, 28, 17],
                [49, 43, 44, 42, 42, 45, 35, 24],
                code="3",
            ),
            DuctElement(
                "Duct, 36 x 24 in, 5 ft, 1 in lining",
                [2, 2, 3, 7, 15, 12, 11, 9],
                code="4",
            ),
            DuctElement("Branch split, 25 per cent", 6.0, code="5"),
            DuctElement(
                "Duct, 18 x 12 in, 6 ft, 1 in lining",
                [3, 3, 5, 11, 25, 22, 16, 13],
                code="6",
            ),
            DuctElement(
                "Flexible duct, 12 in diameter, 6 ft",
                [14, 14, 16, 15, 17, 22, 16, 13],
                code="7",
            ),
            DuctElement(
                "Rectangular diffuser, 312 cfm",
                0.0,
                [33, 32, 29, 23, 15, 4, 0, 0],
                code="8",
            ),
        ],
        room_effect=[6, 6, 5, 5, 6, 7, 6, 6],
        source_label="Fan, centrifugal, forward-curved, 5000 cfm",
        criterion="NC",
        target=30.0,
        label="Supply path",
    )


def _reportlab() -> None:
    pytest.importorskip("reportlab")
    pytest.importorskip("svglib")
    pytest.importorskip("matplotlib")


def test_report_renders_the_published_sheet(tmp_path: Path) -> None:
    _reportlab()
    res = _supply()
    out = tmp_path / "duct_path.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))

    assert "Duct-borne noise path calculation" in text
    assert "Octave-band path calculation, dB" in text
    # Column headings: the octave bands, 1 kHz and above abbreviated.
    for label in ("63", "125", "250", "500", "1k", "2k", "4k", "8k"):
        assert label in text
    # The fan row of Table 14.9 and the worksheet-signed silencer attenuation.
    assert "90 86 82 79 77 75 71 61" in text
    # Signed with the typographic minus, as every reading a fiche prints is.
    assert "−7 −12 −16 −28 −35 −35 −28 −17" in text
    # The silencer's regenerated noise keeps its own row.
    assert "49 43 44 42 42 45 35 24" in text
    # The received spectrum and the ANSI/ASA S12.2-2019 NC 30 curve.
    assert "52 42 30 18 9 −2 −3 −1" in text
    assert "57 48 41 35 32 29 28 27" in text
    assert "NC 30" in text
    # The boxed designation is quoted at the sheet's one-decimal rounding, not
    # at every digit an interpolated tangency rating carries.
    assert "Room criterion NC-22.6 (125 Hz)" in text
    assert "Predicted (estimated) result" in text
    assert "not a measurement" in text


def test_non_verbose_sheet_drops_the_intermediate_rows(tmp_path: Path) -> None:
    _reportlab()
    res = _supply()
    plain = tmp_path / "plain.pdf"
    verbose = tmp_path / "verbose.pdf"
    res.report(str(plain))
    res.report(str(verbose), verbose=True)
    assert_one_page(str(plain))
    assert_one_page(str(verbose))
    plain_text = _extract_text(str(plain))
    verbose_text = _extract_text(str(verbose))
    assert "Sum" not in plain_text
    assert "Sum" in verbose_text
    assert "Combined" in verbose_text
    # Only the three elements that regenerate noise (the elbow, the silencer
    # and the diffuser) keep a self-noise row; the four that sit on the 0 dB
    # floor lose theirs in both modes.
    assert plain_text.count("Self-noise") == 3
    assert verbose_text.count("Self-noise") == 3


def test_metadata_header_and_requirement_override(tmp_path: Path) -> None:
    _reportlab()
    res = _supply()
    out = tmp_path / "meta.pdf"
    res.report(
        str(out),
        metadata=ReportMetadata(
            specimen="Supply air path",
            client="Example client",
            report_id="EXAMPLE-DUCT",
            requirement=20.0,
        ),
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example client" in text
    assert "EXAMPLE-DUCT" in text
    # The metadata requirement overrides the result's own NC 30 target.
    assert "NC 20" in text
    assert "exceeded by" in text


def test_verdict_passes_against_nc_30(tmp_path: Path) -> None:
    _reportlab()
    res = _supply()
    out = tmp_path / "verdict.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert res.meets_target is True
    assert "no band exceeds NC 30" in text


def test_spanish_sheet(tmp_path: Path) -> None:
    _reportlab()
    res = _supply()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es", verbose=True)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Cálculo del trayecto de ruido por conductos" in text
    assert "Elemento" in text
    assert "Suma" in text
    assert "Ruido propio" in text
    assert "Efecto de sala" in text
    assert "Nivel recibido" in text
    assert "Criterio de recinto" in text
    assert "Duct-borne noise path calculation" not in text


def test_combined_paths_sheet_lists_its_contributions(tmp_path: Path) -> None:
    _reportlab()
    from phonometry.noise_control.duct_path import combine_duct_paths

    supply = _supply()
    other = duct_path(
        OCTAVE_BANDS,
        [90.0, 86.0, 82.0, 79.0, 77.0, 75.0, 71.0, 61.0],
        [DuctElement("Return plenum", 30.0, code="R")],
        room_effect=8.0,
        criterion="NC",
        target=30.0,
        label="Return path",
    )
    both = combine_duct_paths([supply, other], label="Supply and return")
    out = tmp_path / "combined.pdf"
    both.report(str(out))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Supply path" in text
    assert "Return path" in text


@pytest.mark.parametrize("verbose", [False, True])
def test_a_long_path_still_renders_on_one_page(tmp_path: Path, verbose: bool) -> None:
    """A fiche is one page: a path too long for it elides its middle rows."""
    _reportlab()
    res = duct_path(
        OCTAVE_BANDS,
        [90.0] * 8,
        [
            DuctElement(f"Duct element {i}", 3.0, [40.0 - i] * 8, code=str(i))
            for i in range(1, 26)
        ],
        room_effect=6.0,
        criterion="NC",
        target=35.0,
        label="Long path",
    )
    out = tmp_path / f"long_{verbose}.pdf"
    res.report(
        str(out),
        verbose=verbose,
        metadata=ReportMetadata(
            specimen="A very long duct path",
            client="Example client",
            report_id="EXAMPLE-LONG",
            laboratory="Phonometry",
            operator="phonometry",
        ),
    )
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "elements omitted" in text
    # The head, the foot and the criterion survive the elision.
    assert "Received level" in text
    assert "Room effect" in text
    assert "NC 35" in text
    # The complete sheet is always available from the result itself.
    assert len(res.table()) > 25


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    res = _supply()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown report engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path: Path) -> None:
    res = _supply()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match=r"Unknown language"):
        res.report(out, language="xx")


def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib.axes import Axes

    res = _supply()
    ax = res.plot()
    assert isinstance(ax, Axes)
    # The received level and the criterion curve are both drawn.
    drawn = [np.asarray(line.get_ydata()) for line in ax.lines]
    assert any(np.allclose(d, res.received_level) for d in drawn)
    assert any(np.allclose(d, res.criterion_curve) for d in drawn)


def test_plot_language_is_validated() -> None:
    pytest.importorskip("matplotlib")
    res = _supply()
    with pytest.raises(ValueError, match=r"Unknown language"):
        res.plot(language="xx")
