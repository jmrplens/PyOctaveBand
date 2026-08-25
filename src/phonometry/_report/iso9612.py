#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 9612:2009 occupational noise-exposure fiche (reportlab renderer).

Renders an
:class:`~phonometry.hearing.occupational_exposure.ExposureResult` to a one-page
PDF laid out like the noise-exposure measurement report of an occupational
prevention service, carrying the information ISO 9612:2009 Clause 15 asks for:

* a title and the standard-basis line naming the applied measurement strategy
  (task-based, job-based or full-day; Clause 15 a);
* an optional metadata header grid: company, worker(s)/job, workplace, date,
  and the instrumentation and calibration-traceability free text of
  Clause 15 c;
* the work analysis (Clause 15 b/d): for a task-based result the full-width
  per-task table (task, mean duration ``T_m``, sample count ``I``, the task
  level ``Lp,A,eqT,m`` of Formula (7) and its contribution ``LEX,8h,m`` of
  Formula (8)) closed by the nominal-day totals row; for a job-based or
  full-day result the sampling summary (``N``, ``Lp,A,eqTe``, ``T_e`` and the
  Formula (C.9) budget terms). ``verbose=True`` adds the Annex C per-task
  uncertainty columns (``u1a``, ``u1b``, ``u2``);
* for a task-based result, the per-task contribution chart drawn by the
  result's own ``plot(ax=...)``;
* the boxed daily noise exposure level ``LEX,8h`` with the expanded
  uncertainty ``U`` stated as a separate value (Clause 15 e), the coverage
  factor ``k`` and the one-sided 95 % upper limit ``LEX,8h + U``;
* an assessment table against the exposure action values (80/85 dB(A)) and
  the exposure limit value (87 dB(A)) of Directive 2003/10/EC Article 3, each
  marked exceeded / not exceeded on the value rounded exactly as displayed,
  plus a PASS/FAIL verdict row against the limit value; and
* a footer identity/disclaimer block, preceded by the Clause 14 uncertainty
  note and the Article 3(2) hearing-protector note (the limit value applies to
  the effective exposure with the worn protectors' attenuation taken into
  account, which the measured ``LEX,8h`` does not include).

Like :mod:`.broadcast` and :mod:`.iso4871` this uses a stacked table layout
rather than the narrow two-panel one: the work-analysis table needs the full
content width and the contribution chart is landscape. The
quantity-independent skeleton lives in :mod:`._layout`; this module only holds
the ISO 9612 specifics. reportlab, matplotlib and svglib are soft dependencies
imported lazily (reportlab and svglib ship in the ``phonometry[report]``
extra, matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _LIGHT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    _VERDICT_BAD_HEX,
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

if TYPE_CHECKING:
    from reportlab.platypus import Table

    from ..hearing.occupational_exposure import ExposureResult
    from .metadata import ReportMetadata

#: Directive 2003/10/EC Article 3 daily exposure values, dB(A) re LEX,8h.
_LOWER_ACTION_DBA = 80.0
_UPPER_ACTION_DBA = 85.0
_LIMIT_DBA = 87.0

#: The strategy line of the basis, keyed by ``ExposureResult.strategy``.
_STRATEGY_LABELS: dict[str, str] = {
    "task": "task-based measurement (Clause 9)",
    "job": "job-based measurement (Clause 10)",
    "full_day": "full-day measurement (Clause 11)",
}

#: Display names of the Table C.5 instrument classes (Clause 15 c fallback).
_INSTRUMENT_LABELS: dict[str, str] = {
    "class1": "Sound level meter IEC 61672-1 class 1",
    "class2": "Sound level meter IEC 61672-1 class 2",
    "personal_exposimeter": "Personal sound exposure meter (IEC 61252)",
}


def _fmt_db(value: float, language: str = "en") -> str:
    """A level or uncertainty rounded to one decimal place (Clause 15 e)."""
    return format_number(display_round(float(value)), language, decimals=1)


def _fmt_hours(value: float, language: str = "en") -> str:
    """A duration in hours with up to one decimal (trailing ``.0`` kept)."""
    return format_number(float(value), language, decimals=1)


def _basis(result: ExposureResult, language: str = "en") -> str:
    """The standard-basis line: ISO 9612:2009 plus the applied strategy."""
    strategy = t(_STRATEGY_LABELS[result.strategy], language)
    return t(
        "Determination of occupational noise exposure per ISO 9612:2009 "
        "(engineering method); {strategy}.",
        language,
    ).format(strategy=strategy)


def _instrumentation_text(
    result: ExposureResult,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> str | None:
    """The Clause 15 c instrumentation line: user text, or the class fallback."""
    if metadata is not None and metadata.instrumentation:
        return html.escape(metadata.instrumentation)
    if result.instrument is not None:
        return t(_INSTRUMENT_LABELS[result.instrument], language)
    return None


def _metadata_pairs(
    result: ExposureResult,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the header grid.

    An exposure fiche identifies the client company, the worker(s) or group
    whose exposure was determined and the workplace (Clause 15 a/b), plus the
    instrumentation and calibration traceability (Clause 15 c). Only supplied
    fields are returned; the instrumentation falls back to the result's own
    instrument class.
    """
    specs: list[tuple[str, str | None]] = []
    if metadata is not None:
        specs += [
            (t("Company", language), _esc(metadata.client)),
            (t("Worker(s) / job", language), _esc(metadata.specimen)),
            (t("Workplace", language), _esc(metadata.test_room)),
            (t("Date of test", language), _esc(metadata.test_date)),
        ]
    specs.append(
        (
            t("Instrumentation", language),
            _instrumentation_text(result, metadata, language),
        )
    )
    if metadata is not None:
        specs.append((t("Calibration", language), _esc(metadata.calibration)))
    return [(label, value) for label, value in specs if value]


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _task_table(
    result: ExposureResult, verbose: bool = False, language: str = "en"
) -> Table:
    """The Clause 15 b/d work-analysis table of a task-based result.

    One row per task (label, mean duration ``T_m``, sample count ``I``, the
    Formula (7) task level and the Formula (8) contribution), closed by the
    nominal-day totals row (``T_e = sum T_m`` and the Formula (9) ``LEX,8h``).
    ``verbose`` adds the Annex C per-task uncertainty columns.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso9612")

    headers = [
        t("Task", language),
        "T<sub>m</sub> [h]",
        t("Samples I", language),
        "L<sub>p,A,eqT,m</sub> [dB]",
        t("Contribution L<sub>EX,8h,m</sub> [dB]", language),
    ]
    widths = [62.0, 22.0, 18.0, 34.0, 38.0]
    if verbose:
        headers += [
            "u<sub>1a,m</sub> [dB]",
            "u<sub>1b,m</sub> [h]",
            "u<sub>2,m</sub> [dB]",
        ]
        widths = [42.0, 16.0, 16.0, 26.0, 28.0, 16.0, 16.0, 14.0]

    data: list[list[Any]] = [[fiche_paragraph(h, header_style) for h in headers]]
    total_hours = 0.0
    for task in result.tasks:
        total_hours += task.duration_hours
        row = [
            fiche_paragraph(html.escape(task.label), label_style),
            fiche_paragraph(_fmt_hours(task.duration_hours, language), value_style),
            fiche_paragraph(str(task.n_samples), value_style),
            fiche_paragraph(_fmt_db(task.lp_aeqt, language), value_style),
            fiche_paragraph(_fmt_db(task.lex_8h_contribution, language), value_style),
        ]
        if verbose:
            row += [
                fiche_paragraph(_fmt_db(task.u1a, language), value_style),
                fiche_paragraph(_fmt_hours(task.u1b, language), value_style),
                fiche_paragraph(_fmt_db(task.u2, language), value_style),
            ]
        data.append(row)

    total_row = [
        fiche_paragraph(f"<b>{t('Nominal day', language)}</b>", label_style),
        fiche_paragraph(f"<b>{_fmt_hours(total_hours, language)}</b>", value_style),
        fiche_paragraph("", value_style),
        fiche_paragraph("", value_style),
        fiche_paragraph(f"<b>{_fmt_db(result.lex_8h, language)}</b>", value_style),
    ]
    if verbose:
        total_row += [fiche_paragraph("", value_style)] * 3
    data.append(total_row)

    table = stacked_table(data, [w * mm for w in widths])
    # The totals row keeps the light fill regardless of the zebra parity.
    table.setStyle(
        [
            (
                "BACKGROUND",
                (0, len(data) - 1),
                (-1, len(data) - 1),
                colors.HexColor(_LIGHT_HEX),
            ),
            (
                "LINEABOVE",
                (0, len(data) - 1),
                (-1, len(data) - 1),
                0.5,
                colors.HexColor(_ACCENT_HEX),
            ),
        ]
    )
    return table


def _sampling_table(result: ExposureResult, language: str = "en") -> Table:
    """The sampling summary of a job-based/full-day result (Formula (C.9) budget)."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso9612")

    rows: list[tuple[str, str]] = [
        (t("Number of samples N", language), str(result.n_samples)),
        (
            t("Energy-average level L<sub>p,A,eqTe</sub> [dB]", language),
            _fmt_db(result.lp_aeqte or 0.0, language),
        ),
        (
            t("Effective day duration T<sub>e</sub> [h]", language),
            _fmt_hours(result.effective_duration_hours or 0.0, language),
        ),
        (
            t("Sampling standard uncertainty u<sub>1</sub> [dB]", language),
            _fmt_db(result.u1 or 0.0, language),
        ),
        (
            t(
                "Sampling contribution c<sub>1</sub>u<sub>1</sub> (Table C.4) [dB]",
                language,
            ),
            _fmt_db(result.c1u1 or 0.0, language),
        ),
        (
            t("Instrument u<sub>2</sub> (Table C.5) [dB]", language),
            _fmt_db(result.u2 or 0.0, language),
        ),
        (
            t("Microphone position u<sub>3</sub> [dB]", language),
            _fmt_db(result.u3 or 0.0, language),
        ),
    ]
    data: list[list[Any]] = [
        [
            fiche_paragraph(t("Metric", language), header_style),
            fiche_paragraph(t("Measured", language), header_style),
        ]
    ]
    for label, value in rows:
        data.append(
            [fiche_paragraph(label, label_style), fiche_paragraph(value, value_style)]
        )
    return stacked_table(data, [120 * mm, 54 * mm])


def _assessment_rows(
    result: ExposureResult, language: str = "en"
) -> list[tuple[str, str, str, bool | None]]:
    """The Directive 2003/10/EC assessment rows.

    Each row is ``(label, threshold, measured, exceeded)``; the informational
    upper-limit row carries ``exceeded=None``. The comparison is evaluated on
    the LEX,8h rounded to one decimal exactly as the fiche displays it
    (Clause 15 e), so the printed numbers can never contradict their own
    status at a threshold boundary.
    """
    displayed = display_round(float(result.lex_8h))
    measured = f"L<sub>EX,8h</sub> = {_fmt_db(result.lex_8h, language)} dB"
    rows: list[tuple[str, str, str, bool | None]] = [
        (
            t("Lower exposure action value", language),
            f"{format_number(_LOWER_ACTION_DBA, language, decimals=0)} dB(A)",
            measured,
            displayed > _LOWER_ACTION_DBA,
        ),
        (
            t("Upper exposure action value", language),
            f"{format_number(_UPPER_ACTION_DBA, language, decimals=0)} dB(A)",
            measured,
            displayed > _UPPER_ACTION_DBA,
        ),
        (
            t("Exposure limit value", language),
            f"{format_number(_LIMIT_DBA, language, decimals=0)} dB(A)",
            measured,
            displayed > _LIMIT_DBA,
        ),
        (
            t("One-sided 95 % upper limit L<sub>EX,8h</sub> + U", language),
            "&#8211;",
            f"{_fmt_db(result.upper_limit, language)} dB",
            None,
        ),
    ]
    return rows


def _assessment_table(result: ExposureResult, language: str = "en") -> Table:
    """The Directive 2003/10/EC assessment table (exceeded / not exceeded)."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso9612")

    data: list[list[Any]] = [
        [
            fiche_paragraph(t("Directive 2003/10/EC value", language), header_style),
            fiche_paragraph(t("Threshold", language), header_style),
            fiche_paragraph(t("Measured", language), header_style),
            fiche_paragraph(t("Result", language), header_style),
        ]
    ]
    for label, threshold, measured, exceeded in _assessment_rows(result, language):
        data.append(
            [
                fiche_paragraph(label, label_style),
                fiche_paragraph(threshold, value_style),
                fiche_paragraph(measured, value_style),
                fiche_paragraph(exceedance_markup(exceeded, language), label_style),
            ]
        )
    return stacked_table(data, [74 * mm, 24 * mm, 42 * mm, 34 * mm])


def _statement(result: ExposureResult, language: str = "en") -> tuple[str, list[str]]:
    """The boxed result statement and its extended terms (Clause 15 e)."""
    statement = t(
        "Daily noise exposure level L<sub>EX,8h</sub> = <b>{lex} dB</b> "
        "&nbsp; U&nbsp;=&nbsp;{u}&nbsp;dB",
        language,
    ).format(
        lex=_fmt_db(result.lex_8h, language),
        u=_fmt_db(result.expanded_uncertainty, language),
    )
    extended = [
        t("Coverage factor k = {k} (one-sided 95 %)", language).format(
            k=format_number(result.coverage_factor, language, decimals=2)
        ),
        t("Combined standard uncertainty u = {value} dB", language).format(
            value=_fmt_db(result.combined_standard_uncertainty, language)
        ),
        t("Upper limit L<sub>EX,8h</sub> + U = {value} dB", language).format(
            value=_fmt_db(result.upper_limit, language)
        ),
    ]
    return statement, extended


def _verdict(result: ExposureResult, language: str = "en") -> tuple[str, bool]:
    """Verdict text and PASS flag against the Directive 2003/10/EC limit value.

    The exposure limit value of 87 dB(A) (Article 3.1) is the only Directive
    value a worker may in no circumstances exceed, so it carries the fiche's
    verdict; the action values are obligation triggers reported in the
    assessment table. The comparison uses the displayed (one-decimal) value,
    like the assessment rows.
    """
    passed = display_round(float(result.lex_8h)) <= _LIMIT_DBA
    text = t(
        "L<sub>EX,8h</sub> = {lex} dB, exposure limit value &#8804; {limit} dB(A)",
        language,
    ).format(
        lex=_fmt_db(result.lex_8h, language),
        limit=format_number(_LIMIT_DBA, language, decimals=0),
    )
    return text, passed


def render_iso9612_report(
    result: ExposureResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 9612:2009 occupational noise-exposure fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.hearing.occupational_exposure.ExposureResult` from
        any of the three ISO 9612 strategies.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity (``client`` is the company, ``specimen`` the worker(s)/job,
        ``test_room`` the workplace), the ``instrumentation``/``calibration``
        free text of Clause 15 c and the footer identity.
    :param verbose: When True, the task-based work-analysis table adds the
        Annex C per-task uncertainty columns (``u1a``, ``u1b``, ``u2``).
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for a task-based result's chart,
        matplotlib) is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Occupational noise exposure determination", language)

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(_basis(result, language), basis_style),
    ]

    header_pairs = _metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    if result.tasks:
        flow.append(fiche_paragraph(t("Work analysis", language), caption_style))
        flow.append(_task_table(result, verbose, language))
        flow.append(Spacer(1, 6))
        # Full-width, landscape per-task contribution chart (self-scaling axis).
        flow.append(
            render_figure_drawing(
                result.plot,
                174 * mm,
                y_top=None,
                figsize=(9.2, 2.55),
                language=language,
            )
        )
    else:
        flow.append(fiche_paragraph(t("Sampling summary", language), caption_style))
        flow.append(_sampling_table(result, language))
    flow.append(Spacer(1, 8))

    statement, extended = _statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))
    flow.append(Spacer(1, 6))

    flow.append(
        fiche_paragraph(
            t("Assessment against Directive 2003/10/EC exposure values", language),
            caption_style,
        )
    )
    flow.append(_assessment_table(result, language))
    text, passed = _verdict(result, language)
    flow.extend(verdict_flow(text, passed, styles, language))

    note_style = ParagraphStyle(
        "iso9612_notes",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=6,
    )
    if result.sampling_advisory:
        advisory_style = ParagraphStyle(
            "iso9612_advisory",
            parent=note_style,
            textColor=colors.HexColor(_VERDICT_BAD_HEX),
        )
        flow.append(
            fiche_paragraph(
                t(
                    "The ISO 9612 sampling rules recommend additional "
                    "measurements (3 dB spread rule of Clauses 9.3/11.3, "
                    "Table C.4 threshold of Clause 10.4, or the Table 1 "
                    "minimum cumulative duration).",
                    language,
                ),
                advisory_style,
            )
        )
    flow.append(
        fiche_paragraph(
            t(
                "Noise exposure level and measurement uncertainty are stated "
                "as separate values (ISO 9612:2009 Clause 15); U = k&#183;u "
                "with k = 1.65 gives the one-sided 95 % confidence interval "
                "(Clause 14, Annex C).",
                language,
            ),
            note_style,
        )
    )
    flow.append(
        fiche_paragraph(
            t(
                "When applying the exposure limit value, Directive 2003/10/EC "
                "(Article 3) takes account of the attenuation of the "
                "individual hearing protectors worn; the measured "
                "L<sub>EX,8h</sub> stated above does not include it.",
                language,
            ),
            note_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
