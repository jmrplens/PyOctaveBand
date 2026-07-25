#  Copyright (c) 2026. Jose M. Requena-Plens
"""Daily vibration exposure assessment fiche (reportlab renderer).

Renders a
:class:`~phonometry.vibration.human_vibration.DailyVibrationExposure` to a
one-page PDF laid out like a hand-arm / whole-body vibration exposure
assessment sheet (the layout the HSE exposure calculators and the EU Good
Practice Guides use): each operation's vibration magnitude and daily
exposure time combine into the 8-hour daily exposure ``A(8)``, which is then
assessed against Directive 2002/44/EC (Article 3):

* a title and the standard-basis line naming the applied ISO method
  (ISO 5349-1/-2:2001 for hand-transmitted vibration, ISO 2631-1:1997 for
  whole-body vibration) and the Directive it is assessed against;
* an optional metadata header grid: company, operator/worker, workplace, date
  and the instrumentation and calibration free text;
* the exposure analysis: a full-width per-operation table (operation, the
  vibration magnitude - the vector total ``a_hv`` (hand-arm, Directive
  2002/44/EC Annex Part A) or the dominant-axis value ``a_w,max`` (whole-body,
  Annex Part B) - the daily
  exposure time ``T_i`` and the partial exposure ``A_i(8)`` of ISO 5349-1/-2
  Eq. (2)) closed by the daily-total row (total exposure time and the combined
  ``A(8)`` of Eq. (3)); ``verbose=True`` adds the per-operation share of the
  daily vibration energy;
* the per-operation contribution chart drawn by the result's own
  ``plot(ax=...)``;
* the boxed daily exposure ``A(8)`` with the exposure zone (below action /
  action / limit);
* an assessment table against the Directive 2002/44/EC exposure action value
  (EAV) and exposure limit value (ELV) for the vibration kind, each marked
  exceeded / not exceeded on the value rounded exactly as displayed, plus a
  PASS/FAIL verdict row against the limit value; and
* a footer identity/disclaimer block, preceded by the note that the ISO
  standards define no safe exposure limit and that the Directive action value
  triggers the employer's control and health-surveillance obligations while
  the limit value shall not be exceeded.

Like :mod:`.iso9612` this uses a stacked table layout rather than the narrow
two-panel one: the exposure-analysis table needs the full content width and the
contribution chart is landscape. The quantity-independent skeleton lives in
:mod:`._layout`; this module only holds the vibration specifics. reportlab,
matplotlib and svglib are soft dependencies imported lazily (reportlab and
svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import decimal_comma, t
from ._layout import (
    _ACCENT_HEX,
    _LIGHT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    analysis_cell_styles,
    build_document,
    display_round,
    document_styles,
    exceedance_markup,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    stacked_table,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..vibration.human_vibration import DailyVibrationExposure

#: Number of decimal places the fiche displays vibration magnitudes at. Two
#: decimals resolve the whole-body ELV (1.15 m/s2) and the small whole-body
#: magnitudes, and match the precision of the HSE exposure sheets; the
#: exceedance comparisons are evaluated on the value rounded to this precision.
_ACC_DECIMALS = 2

#: Per-kind labels: the standard, the per-operation magnitude symbol and the
#: exposure-response note. ``a_hv`` for hand-arm (the ISO 5349-1 Eq. (1)
#: vector total, Directive 2002/44/EC Annex Part A point 1); ``a_w,max`` for
#: whole-body (the highest frequency-weighted axis value
#: ``max(1,4*a_wx, 1,4*a_wy, a_wz)``, Annex Part B point 1 - not the
#: ISO 2631-1 Eq. (10) vector total ``a_v``).
_KIND_STANDARD: dict[str, str] = {
    "hav": "ISO 5349-1:2001 / ISO 5349-2:2001",
    "wbv": "ISO 2631-1:1997",
}
_KIND_QUANTITY: dict[str, str] = {"hav": "hand-arm", "wbv": "whole-body"}
_KIND_TOTAL_SYMBOL: dict[str, str] = {
    "hav": "a<sub>hv</sub>",
    "wbv": "a<sub>w,max</sub>",
}


def _unit(metric: str, language: str = "en") -> str:
    """The exposure unit: ``m/s2`` for ``A(8)``, ``m/s^1,75`` for the VDV."""
    if metric == "vdv":
        return "m/s<sup>" + decimal_comma("1.75", language) + "</sup>"
    return "m/s<sup>2</sup>"


def _fmt_acc(value: float, language: str = "en") -> str:
    """A vibration magnitude rounded to the fiche's display precision."""
    from ._i18n import format_number

    return format_number(
        display_round(float(value), _ACC_DECIMALS),
        language,
        decimals=_ACC_DECIMALS,
    )


def _fmt_hours(seconds: float, language: str = "en") -> str:
    """A daily exposure time in hours, up to two decimals (trailing zeros cut)."""
    from ._i18n import format_number

    return format_number(float(seconds) / 3600.0, language, decimals=2, trim=True)


def _basis(result: DailyVibrationExposure, language: str = "en") -> str:
    """The standard-basis line: the ISO method plus the assessment Directive."""
    kind = str(result.assessment.kind)
    quantity = t(_KIND_QUANTITY.get(kind, kind), language)
    standard = _KIND_STANDARD.get(kind, "")
    return t(
        "Determination of daily {quantity} vibration exposure A(8) per "
        "{standard}; assessed against Directive 2002/44/EC (Article 3).",
        language,
    ).format(quantity=quantity, standard=standard)


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _metadata_pairs(
    metadata: ReportMetadata | None, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the header grid.

    A vibration exposure sheet identifies the client company, the operator or
    worker whose exposure was determined and the workplace, plus the
    instrumentation and calibration traceability. Only supplied fields are
    returned.
    """
    if metadata is None:
        return []
    specs: list[tuple[str, str | None]] = [
        (t("Company", language), _esc(metadata.client)),
        (t("Operator / worker", language), _esc(metadata.specimen)),
        (t("Workplace", language), _esc(metadata.test_room)),
        (t("Date of assessment", language), _esc(metadata.test_date)),
        (t("Instrumentation", language), _esc(metadata.instrumentation)),
        (t("Calibration", language), _esc(metadata.calibration)),
    ]
    return [(label, value) for label, value in specs if value]


def _operations_table(
    result: DailyVibrationExposure, verbose: bool = False, language: str = "en"
) -> Any:
    """The per-operation exposure-analysis table (ISO 5349-1/-2 Eqs. (2)/(3)).

    One row per operation (label, the vibration total value, the daily exposure
    time ``T_i`` and the partial exposure ``A_i(8)``), closed by the daily-total
    row (total exposure time and the combined ``A(8)``). ``verbose`` adds each
    operation's share of the daily vibration energy ``A_i(8)^2 / A(8)^2``.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    header_style, label_style, value_style = analysis_cell_styles("humanvib")
    kind = str(result.assessment.kind)
    unit = _unit("a8", language)
    total_symbol = _KIND_TOTAL_SYMBOL.get(kind, "a<sub>w,max</sub>")

    headers = [
        t("Operation", language),
        f"{total_symbol} [{unit}]",
        "T<sub>i</sub> [h]",
        f"A<sub>i</sub>(8) [{unit}]",
    ]
    widths = [78.0, 34.0, 26.0, 36.0]
    if verbose:
        headers.append(t("Share of A(8)&#178;", language))
        widths = [66.0, 30.0, 24.0, 30.0, 24.0]

    a8_energy = float(result.a8) ** 2
    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for label, total_value, duration_s, partial in zip(
        result.labels, result.total_values, result.durations_s, result.partials
    ):
        row = [
            Paragraph(html.escape(str(label)), label_style),
            Paragraph(_fmt_acc(float(total_value), language), value_style),
            Paragraph(_fmt_hours(float(duration_s), language), value_style),
            Paragraph(_fmt_acc(float(partial), language), value_style),
        ]
        if verbose:
            from ._i18n import format_number

            share = 100.0 * float(partial) ** 2 / a8_energy if a8_energy > 0.0 else 0.0
            row.append(
                Paragraph(
                    format_number(share, language, decimals=0) + " %", value_style
                )
            )
        data.append(row)

    total_hours = float(sum(float(d) for d in result.durations_s))
    total_row = [
        Paragraph(f"<b>{t('Daily total', language)}</b>", label_style),
        Paragraph("", value_style),
        Paragraph(f"<b>{_fmt_hours(total_hours, language)}</b>", value_style),
        Paragraph(f"<b>{_fmt_acc(float(result.a8), language)}</b>", value_style),
    ]
    if verbose:
        total_row.append(Paragraph("", value_style))
    data.append(total_row)

    table = stacked_table(data, [w * mm for w in widths])
    # The totals row keeps the light fill regardless of the zebra parity.
    table.setStyle(
        [
            ("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1),
             colors.HexColor(_LIGHT_HEX)),
            ("LINEABOVE", (0, len(data) - 1), (-1, len(data) - 1),
             0.5, colors.HexColor(_ACCENT_HEX)),
        ]
    )
    return table


def _assessment_rows(
    result: DailyVibrationExposure, language: str = "en"
) -> list[tuple[str, str, str, bool]]:
    """The Directive 2002/44/EC assessment rows.

    Each row is ``(label, threshold, measured, exceeded)``. The comparison is
    evaluated on ``A(8)`` rounded exactly as the fiche displays it, so the
    printed numbers can never contradict their own status at a threshold
    boundary; the ``>=`` (reaches or exceeds) direction matches the
    :class:`~phonometry.vibration.human_vibration.ExposureAssessment`.
    """
    assessment = result.assessment
    metric = str(assessment.metric)
    unit = _unit(metric, language)
    displayed = display_round(float(assessment.value), _ACC_DECIMALS)
    measured = f"A(8) = {_fmt_acc(float(assessment.value), language)} {unit}"
    return [
        (
            t("Exposure action value (EAV)", language),
            f"{_fmt_acc(float(assessment.action_value), language)} {unit}",
            measured,
            displayed >= display_round(float(assessment.action_value), _ACC_DECIMALS),
        ),
        (
            t("Exposure limit value (ELV)", language),
            f"{_fmt_acc(float(assessment.limit_value), language)} {unit}",
            measured,
            displayed >= display_round(float(assessment.limit_value), _ACC_DECIMALS),
        ),
    ]


def _assessment_table(result: DailyVibrationExposure, language: str = "en") -> Any:
    """The Directive 2002/44/EC assessment table (exceeded / not exceeded)."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    header_style, label_style, value_style = analysis_cell_styles("humanvib")

    data: list[list[Any]] = [
        [
            Paragraph(t("Directive 2002/44/EC value", language), header_style),
            Paragraph(t("Threshold", language), header_style),
            Paragraph(t("Measured", language), header_style),
            Paragraph(t("Result", language), header_style),
        ]
    ]
    for label, threshold, measured, exceeded in _assessment_rows(result, language):
        data.append(
            [
                Paragraph(label, label_style),
                Paragraph(threshold, value_style),
                Paragraph(measured, value_style),
                Paragraph(exceedance_markup(exceeded, language), label_style),
            ]
        )
    return stacked_table(data, [66 * mm, 32 * mm, 42 * mm, 34 * mm])


#: The exposure-zone phrase for the boxed result, keyed by the zone derived
#: from the displayed-rounded comparisons (:func:`_displayed_zone`).
_ZONE_LABELS: dict[str, str] = {
    "below action": "below the exposure action value",
    "action": "at or above the exposure action value",
    "limit": "at or above the exposure limit value",
}


def _displayed_zone(assessment: Any) -> str:
    """The exposure zone evaluated on the value rounded as displayed.

    The assessment rows and the verdict compare ``A(8)`` rounded exactly as
    the fiche prints it, so the boxed zone phrase derives from the same
    displayed-rounded comparisons: an ``A(8)`` that prints equal to a
    threshold can never sit in the box one zone below the row beside it.
    """
    displayed = display_round(float(assessment.value), _ACC_DECIMALS)
    if displayed >= display_round(float(assessment.limit_value), _ACC_DECIMALS):
        return "limit"
    if displayed >= display_round(float(assessment.action_value), _ACC_DECIMALS):
        return "action"
    return "below action"


def _statement(result: DailyVibrationExposure, language: str = "en") -> str:
    """The boxed result statement: the daily exposure ``A(8)`` and its zone."""
    assessment = result.assessment
    unit = _unit(str(assessment.metric), language)
    zone_key = _displayed_zone(assessment)
    zone = t(_ZONE_LABELS.get(zone_key, zone_key), language)
    return t(
        "Daily vibration exposure A(8) = <b>{a8} {unit}</b> &nbsp; ({zone})",
        language,
    ).format(a8=_fmt_acc(float(result.a8), language), unit=unit, zone=zone)


def _verdict(result: DailyVibrationExposure, language: str = "en") -> tuple[str, bool]:
    """Verdict text and PASS flag against the Directive 2002/44/EC limit value.

    The exposure limit value is the daily exposure a worker may in no
    circumstances reach (Article 3), so it carries the fiche's verdict; the
    exposure action value is an obligation trigger reported in the assessment
    table. The comparison uses the displayed value and the same reaches-or-
    exceeds (``>=``) direction as the assessment rows and the
    :class:`~phonometry.vibration.human_vibration.ExposureAssessment`, so the
    verdict never contradicts the ELV row it sits beside.
    """
    assessment = result.assessment
    unit = _unit(str(assessment.metric), language)
    displayed = display_round(float(assessment.value), _ACC_DECIMALS)
    limit = display_round(float(assessment.limit_value), _ACC_DECIMALS)
    passed = displayed < limit
    # The sentence states the actual relation, so a FAIL never reads as "below
    # the exposure limit value" while its banner says FAIL.
    if passed:
        template = "A(8) = {a8} {unit} below the exposure limit value {elv} {unit}"
    else:
        template = (
            "A(8) = {a8} {unit} reaches or exceeds the exposure limit value "
            "{elv} {unit}"
        )
    text = t(template, language).format(
        a8=_fmt_acc(float(result.a8), language),
        elv=_fmt_acc(float(assessment.limit_value), language),
        unit=unit,
    )
    return text, passed


def render_human_vibration_report(
    result: DailyVibrationExposure,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a daily vibration exposure assessment fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.vibration.human_vibration.DailyVibrationExposure`
        (from :func:`~phonometry.vibration.human_vibration.daily_vibration_exposure`).
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity (``client`` is the company, ``specimen`` the operator/worker,
        ``test_room`` the workplace), the ``instrumentation``/``calibration``
        free text and the footer identity.
    :param verbose: When True, the operations table adds each operation's share
        of the daily vibration energy ``A_i(8)^2 / A(8)^2``.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab or matplotlib is not installed. The
        fiche always embeds the contribution chart, so both are required
        (``pip install "phonometry[report,plot]"``).
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Daily vibration exposure assessment", language)

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(_basis(result, language), basis_style),
    ]

    header_pairs = _metadata_pairs(metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph(t("Exposure analysis", language), caption_style))
    flow.append(_operations_table(result, verbose, language))
    flow.append(Spacer(1, 10))
    # Full-width, landscape per-operation contribution chart (self-scaling).
    flow.append(
        render_figure_drawing(
            result.plot, 174 * mm, y_top=None, figsize=(9.2, 3.3),
            language=language,
        )
    )
    flow.append(Spacer(1, 8))

    flow.append(result_box(_statement(result, language), styles, accent))
    flow.append(Spacer(1, 6))

    flow.append(
        Paragraph(
            t("Assessment against Directive 2002/44/EC exposure values", language),
            caption_style,
        )
    )
    flow.append(_assessment_table(result, language))
    text, passed = _verdict(result, language)
    flow.extend(verdict_flow(text, passed, styles, language))

    note_style = ParagraphStyle(
        "humanvib_notes", parent=getSampleStyleSheet()["Normal"],
        fontSize=7.5, leading=10, textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=6,
    )
    flow.append(
        Paragraph(
            t(
                "The exposure action value (EAV) and exposure limit value (ELV) "
                "are daily exposures A(8) (Directive 2002/44/EC, Article 3). "
                "Reaching the EAV triggers the employer's programme of control "
                "measures and the workers' entitlement to health surveillance "
                "(Articles 5 and 8); the ELV shall not be exceeded (Article 5).",
                language,
            ),
            note_style,
        )
    )
    kind = str(result.assessment.kind)
    if kind == "hav":
        exposure_note = (
            "ISO 5349-1:2001 defines no safe exposure limit; the daily exposure "
            "A(8) is the recommended basis for any exposure criterion, and its "
            "informative Annex C relates A(8) to the lifetime exposure that "
            "produces vibration-white-finger in 10 % of an exposed group."
        )
    else:
        flow.append(
            Paragraph(
                t(
                    "The whole-body exposure basis a<sub>w,max</sub> is the "
                    "highest frequency-weighted axis value "
                    "max(1.4&#183;a<sub>wx</sub>; 1.4&#183;a<sub>wy</sub>; "
                    "a<sub>wz</sub>) for a seated or standing worker "
                    "(Directive 2002/44/EC, Annex Part B point 1), not the "
                    "ISO 2631-1 vector total a<sub>v</sub>.",
                    language,
                ),
                note_style,
            )
        )
        exposure_note = (
            "ISO 2631-1:1997 defines no exposure limit; its informative "
            "Annexes B and C give the health-guidance caution zone as the range "
            "of magnitudes for which health effects are potentially present."
        )
    flow.append(Paragraph(t(exposure_note, language), note_style))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
