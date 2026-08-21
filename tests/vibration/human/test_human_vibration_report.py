#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the daily vibration exposure ``.report()`` fiche (ISO 5349 / ISO 2631).

The rendered values are checked against a published, library-independent oracle:
the ISO 5349-2:2001 Annex E.3 forestry worker's day (brush-saw 2 h at
4.6 m/s2, chain-saw felling 1 h at 6.0 m/s2, branch stripping 2 h at 3.6 m/s2)
gives partial exposures A_i(8) = 2.3, 2.1 and 1.8 m/s2 and a combined daily
exposure A(8) = 3.6 m/s2 (Eqs. (E.6)-(E.9)). The Directive 2002/44/EC hand-arm
assessment (EAV 2.5, ELV 5 m/s2) and the whole-body assessment (EAV 0.5, ELV
1.15 m/s2) are exercised at the action/limit boundaries on the displayed value.
Values are read back from the PDF via pypdf text extraction; structural facts
(one page, rejected engines/languages) complete the rendering contract.
"""

from __future__ import annotations

import dataclasses
import math
import os

import pytest

from phonometry import ReportMetadata
from phonometry.vibration import daily_vibration_exposure

_PDF_MAGIC = b"%PDF"


def _assert_one_page(path: str) -> None:
    from pypdf import PdfReader

    with open(path, "rb") as handle:
        assert handle.read(4) == _PDF_MAGIC
    assert os.path.getsize(path) > 0
    assert len(PdfReader(path).pages) == 1


def _extract_text(path: str) -> str:
    """Whitespace-normalized page text (PDF line wraps fold to single spaces)."""
    from pypdf import PdfReader

    raw = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(raw.split())


def _annex_e3_result():
    """The ISO 5349-2:2001 Annex E.3 forestry worker's hand-arm day."""
    return daily_vibration_exposure(
        [4.6, 6.0, 3.6],
        [2 * 3600.0, 1 * 3600.0, 2 * 3600.0],
        kind="hav",
        labels=["brush-saw", "felling", "stripping"],
    )


# --- published oracle (ISO 5349-2 Annex E.3) ----------------------------------


def test_annex_e3_hand_oracle_values() -> None:
    """The E.3 partials and combined A(8) match the standard's printed values."""
    res = _annex_e3_result()
    # A_i(8) = a_hv sqrt(T/T0): 4.6*0.5, 6*sqrt(1/8), 3.6*0.5.
    assert float(res.partials[0]) == pytest.approx(2.3, abs=5e-3)
    assert float(res.partials[1]) == pytest.approx(2.1, abs=5e-2)
    assert float(res.partials[2]) == pytest.approx(1.8, abs=5e-3)
    assert res.a8 == pytest.approx(3.6, abs=5e-2)
    assert res.assessment.zone == "action"


def test_report_renders_annex_e3_numbers(tmp_path) -> None:
    """The fiche prints the E.3 magnitudes, partials and combined A(8)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_e3_result()
    out = tmp_path / "e3.pdf"
    returned = res.report(str(out))
    assert returned == str(out)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    # Vibration total values (a_hv) per operation.
    assert "4.60" in text
    assert "6.00" in text
    # Partial exposures A_i(8) to two decimals.
    assert "2.30" in text
    assert "2.12" in text
    assert "1.80" in text
    # Combined daily exposure A(8) = 3.61 m/s2 (2 decimals of the 3.6097 value).
    assert "3.61" in text
    # EAV / ELV thresholds and the basis.
    assert "2.50" in text
    assert "5.00" in text
    assert "ISO 5349-1:2001 / ISO 5349-2:2001" in text
    assert "brush-saw" in text
    # Hand-arm assessment: EAV exceeded, ELV not; overall PASS on the limit.
    assert text.count("Exceeded") == 1
    assert text.count("Not exceeded") == 1
    assert "PASS" in text


def test_verbose_adds_energy_share_column(tmp_path) -> None:
    """verbose=True adds each operation's share of the daily vibration energy."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_e3_result()
    out = tmp_path / "verbose.pdf"
    res.report(str(out), verbose=True)
    _assert_one_page(str(out))
    flat = "".join(_extract_text(str(out)).split())
    # Shares: 2.30^2/3.61^2 = 41 %, 2.12^2 = 35 %, 1.80^2 = 25 % (rounded).
    assert "41%" in flat
    assert "35%" in flat
    assert "25%" in flat


# --- Directive 2002/44/EC assessment boundaries -------------------------------


@pytest.mark.parametrize(
    ("a_hv", "n_exceeded", "verdict"),
    [
        (2.5, 1, "PASS"),  # at the EAV: reaches it -> action exceeded
        (2.49, 0, "PASS"),  # displays 2.49: below the action value
        (5.0, 2, "FAIL"),  # at the ELV: reaches it -> limit exceeded, FAIL
        (4.996, 2, "FAIL"),  # displays 5.00: at the limit on the shown value
        (1.0, 0, "PASS"),  # well below the action value
    ],
)
def test_hav_directive_boundaries_on_displayed_value(
    tmp_path, a_hv: float, n_exceeded: int, verdict: str
) -> None:
    """The EAV/ELV rows flip on the two-decimal displayed value (8 h exposure)."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    # A whole 8-h day at one magnitude gives A(8) equal to that magnitude.
    res = daily_vibration_exposure([a_hv], [8 * 3600.0], kind="hav", labels=["op"])
    assert res.a8 == pytest.approx(a_hv, abs=1e-9)
    out = tmp_path / f"hav_{a_hv}.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert text.count("Exceeded") == n_exceeded
    assert text.count("Not exceeded") == 2 - n_exceeded
    assert verdict in text


@pytest.mark.parametrize(
    ("a_hv", "box_phrase", "n_exceeded", "verdict"),
    [
        # Displays 2.50: the box, the EAV row and the verdict all read the
        # displayed value, so the box says "at or above the action value"
        # while the EAV row says Exceeded and the ELV verdict stays PASS.
        (2.495, "at or above the exposure action value", 1, "PASS"),
        # Displays 5.00: the ELV row says Exceeded and the verdict FAIL, so
        # the boxed zone must say "at or above the limit value" too (the raw
        # value 4.997 would place it one zone lower).
        (4.997, "at or above the exposure limit value", 2, "FAIL"),
    ],
)
def test_boxed_zone_agrees_with_displayed_rows_and_verdict(
    tmp_path, a_hv: float, box_phrase: str, n_exceeded: int, verdict: str
) -> None:
    """The boxed zone phrase derives from the same displayed-rounded compare."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = daily_vibration_exposure([a_hv], [8 * 3600.0], kind="hav", labels=["op"])
    out = tmp_path / f"box_{a_hv}.pdf"
    res.report(str(out))
    text = _extract_text(str(out))
    assert box_phrase in text
    assert "below the exposure action value" not in text
    assert text.count("Exceeded") == n_exceeded
    assert text.count("Not exceeded") == 2 - n_exceeded
    assert verdict in text


def test_verdict_sentence_states_the_actual_relation(tmp_path) -> None:
    """The verdict sentence agrees with its PASS/FAIL banner, both directions."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    passing = daily_vibration_exposure([3.0], [8 * 3600.0], kind="hav", labels=["op"])
    out = tmp_path / "pass.pdf"
    passing.report(str(out))
    text = _extract_text(str(out))
    assert "below the exposure limit value" in text
    assert "reaches or exceeds the exposure limit value" not in text
    failing = daily_vibration_exposure([6.0], [8 * 3600.0], kind="hav", labels=["op"])
    out = tmp_path / "fail.pdf"
    failing.report(str(out))
    text = _extract_text(str(out))
    assert "reaches or exceeds the exposure limit value" in text
    assert "below the exposure limit value" not in text
    assert "FAIL" in text


def test_whole_body_thresholds_and_basis(tmp_path) -> None:
    """A whole-body result names ISO 2631-1 and the 0.5 / 1.15 m/s2 values."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = daily_vibration_exposure(
        [0.8], [8 * 3600.0], kind="wbv", labels=["tractor driving"]
    )
    out = tmp_path / "wbv.pdf"
    res.report(str(out))
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "ISO 2631-1:1997" in text
    assert "0.50" in text  # EAV
    assert "1.15" in text  # ELV, resolved by the two-decimal display
    assert "0.80" in text  # A(8)
    # 0.80 is in the action zone (>= 0.5, < 1.15): EAV exceeded, ELV not.
    assert text.count("Exceeded") == 1
    assert "PASS" in text
    # The fiche labels the whole-body basis per Directive 2002/44/EC Annex
    # Part B point 1 (the highest axis value), not the ISO 2631-1 vector total.
    assert "highest frequency-weighted axis value" in text
    assert "Annex Part B point 1" in text


def test_whole_body_spanish_states_part_b_basis(tmp_path) -> None:
    """The Spanish whole-body fiche carries the translated Part B basis note."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = daily_vibration_exposure(
        [0.8], [8 * 3600.0], kind="wbv", labels=["conducción de tractor"]
    )
    out = tmp_path / "wbv_es.pdf"
    res.report(str(out), language="es")
    text = _extract_text(str(out))
    assert "el mayor valor por eje ponderado en frecuencia" in text
    assert "parte B, punto 1" in text


# --- metadata -----------------------------------------------------------------


def test_metadata_header_renders(tmp_path) -> None:
    """Supplied metadata renders the company, worker, workplace and traceability."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_e3_result()
    metadata = ReportMetadata(
        client="Example forestry contractor",
        specimen="Forestry worker (right hand)",
        test_room="Managed woodland, plot 12",
        instrumentation="HAV meter X1, s/n 0042",
        calibration="Field calibrator verified 2026-01-15",
        laboratory="Prevention service",
        report_id="EXP-5349",
    )
    out = tmp_path / "meta.pdf"
    res.report(str(out), metadata=metadata)
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Example forestry contractor" in text
    assert "Forestry worker (right hand)" in text
    assert "Managed woodland, plot 12" in text
    assert "HAV meter X1, s/n 0042" in text
    assert "EXP-5349" in text


# --- Spanish fiche ------------------------------------------------------------


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders the vibration vocabulary and comma decimals."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_e3_result()
    out = tmp_path / "es.pdf"
    res.report(str(out), language="es")
    _assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Evaluación de la exposición diaria a vibraciones" in text
    assert "Valor límite de exposición (ELV)" in text
    assert "3,61" in text
    assert "Superado" in text
    assert "CUMPLE" in text


# --- rendering contract -------------------------------------------------------


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    res = _annex_e3_result()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        res.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    res = _annex_e3_result()
    out = str(tmp_path / "bad.pdf")
    with pytest.raises(ValueError, match="language"):
        res.report(out, language="xx")


@pytest.mark.parametrize("field", ["total_values", "durations_s", "partials"])
def test_report_rejects_per_operation_array_short_by_one(tmp_path, field: str) -> None:
    """A per-operation array one entry short is named in a ValueError.

    ``DailyVibrationExposure`` is a frozen dataclass with no ``__post_init__``,
    so a hand-built result can carry arrays that disagree with ``labels``. The
    operations table used to run short in silence instead of complaining.
    """
    pytest.importorskip("reportlab")
    res = _annex_e3_result()
    short = dataclasses.replace(res, **{field: getattr(res, field)[:-1]})
    out = tmp_path / f"short_{field}.pdf"
    with pytest.raises(ValueError, match=field):
        short.report(str(out))
    # The guard fires before the table and the chart, so nothing is written.
    assert not out.exists()


def test_report_rejects_fewer_labels_than_operations(tmp_path) -> None:
    """``labels`` sets the expected count, so dropping one is a mismatch too.

    The message names the array that disagreed rather than ``labels`` itself,
    because ``labels`` is what the count is read from: matching both halves
    pins that reading, where matching ``labels`` alone would pass against any
    of the four messages this guard can raise.
    """
    pytest.importorskip("reportlab")
    res = _annex_e3_result()
    short = dataclasses.replace(res, labels=res.labels[:-1])
    out = tmp_path / "short_labels.pdf"
    with pytest.raises(ValueError, match=r"'total_values'.*\(2 in 'labels'\)"):
        short.report(str(out))
    assert not out.exists()


def test_report_accepts_matching_per_operation_arrays(tmp_path) -> None:
    """Dropping the same operation from every array passes the guard."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    res = _annex_e3_result()
    two_ops = dataclasses.replace(
        res,
        labels=res.labels[:-1],
        total_values=res.total_values[:-1],
        durations_s=res.durations_s[:-1],
        partials=res.partials[:-1],
    )
    out = tmp_path / "two_ops.pdf"
    two_ops.report(str(out))
    _assert_one_page(str(out))
    assert "stripping" not in _extract_text(str(out))


def test_partial_exposure_closed_form() -> None:
    """A single-operation A(8) equals a_hv sqrt(T/T0) (ISO 5349-1 Eq. (2))."""
    res = daily_vibration_exposure([6.0], [3600.0], kind="hav")
    assert res.a8 == pytest.approx(6.0 * math.sqrt(1.0 / 8.0), abs=1e-9)
