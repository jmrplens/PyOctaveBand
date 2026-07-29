#  Copyright (c) 2026. Jose M. Requena-Plens
"""Activity noise inspection fiche (reportlab renderer, RD 1367/2007).

Renders a
:class:`~phonometry.environmental.spanish_regulation.ActivityAssessment` to a
one-page PDF laid out like a Spanish *informe de ensayo acustico* / *acta de
inspeccion acustica* of an activity.

The layout follows the format that real Spanish documents converge on, rather
than an invented one. Two families were used. The **legally prescribed content
lists** of the municipal noise ordinances fix which fields must appear: Annex
VIII of the Sevilla ordinance and the equivalent Annex VIII of the Almeria one
("Contenido de los informes sobre ensayos acusticos") enumerate the inspector,
the activity data (expediente, holder, location), the area under study and its
acoustic-area type, the reference standards, the applicable limits and the rule
that governs them, the instrumentation with its calibration check, the
methodology and ambient conditions, the results, and the conclusions; article
50 of the Madrid ordinance prescribes the same minimum content for the *acta de
inspeccion*. The **per-point one-page fiche** of ENAC-accredited laboratory
reports supplies the geometry: an identification strip, a *fases de ruido*
result table carrying the ``Kt``/``Kf``/``Ki`` corrections next to the
``LKeq,Ti`` they produce, and a separate conformity table in which each
criterion of Article 25 is shown against its own allowance (real reports print
the limit column literally as "55+5" and "55+3"), closed by a *dictamen*.

The resulting flow is:

* a title and the standard-basis line (the ``LKeq,T`` definition of Annex I and
  the Annex IV assessment procedure);
* an optional metadata header grid, rendered only for the fields supplied on
  the :class:`ReportMetadata`: the activity (``specimen``), the holder
  (``client``), the assessment point (``test_room``), the instrumentation and
  its calibration check, and the date, plus the applicable limit row read from
  the assessment itself;
* the *fases de ruido* table: one row per noise phase with its duration ``Ti``,
  the background-corrected ``LAeq,Ti``, the ``Kt``/``Kf``/``Ki`` corrections and
  the resulting ``LKeq,Ti``;
* the per-period results table with the three criteria of Article 25.1 b, each
  against its own allowance (the measured phase level against ``limit + 5`` dB,
  the daily ``LKeq,x`` against ``limit + 3`` dB and the annual ``LK,x`` against
  the limit itself);
* the assessment figure (the result's own :meth:`plot`-style vector figure);
* a boxed conclusion and a PASS/FAIL verdict row; and
* a footer identity/disclaimer block whose scope sentence ties the result to
  the operating conditions described, as real Spanish reports do.

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the RD 1367/2007 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    _VERDICT_BAD_HEX,
    _VERDICT_OK_HEX,
    analysis_cell_styles,
    build_document,
    document_styles,
    footer_flow,
    grid_table,
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    stacked_table,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..environmental.spanish_regulation import (
        ActivityAssessment,
        PeriodAssessment,
    )

#: Localised names of the three evaluation periods.
_PERIOD_LABELS = {"day": "Day", "evening": "Evening", "night": "Night"}


def _fmt(value: float, language: str, decimals: int = 1) -> str:
    """A quantity rounded to ``decimals`` decimals, localised separator."""
    return format_number(float(value), language, decimals=decimals)


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _metadata_pairs(
    result: ActivityAssessment,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> list[tuple[str, str]]:
    """Ordered (label, value) pairs of the inspection-report header grid.

    A Spanish noise inspection report identifies the activity, its holder, the
    assessment point, the instrumentation and its calibration check and the
    date of the measurement; only supplied fields are returned. The applicable
    limit row is always shown, read from the assessment rather than the
    metadata.
    """
    specs: list[tuple[str, str | None]] = []
    if metadata is not None:
        specs += [
            (t("Activity", language), _esc(metadata.specimen)),
            (t("Holder", language), _esc(metadata.client)),
            (t("Assessment point", language), _esc(metadata.test_room)),
            (t("Instrumentation", language), _esc(metadata.instrumentation)),
            (t("Calibration check", language), _esc(metadata.calibration)),
            (t("Date of measurement", language), _esc(metadata.test_date)),
        ]
    # The limit row carries its reference and description in English, the
    # library's API language; the fiche renders them in its own language.
    specs.append(
        (
            t("Applicable limit values", language),
            html.escape(t(result.limits.reference, language)),
        )
    )
    specs.append(
        (
            t("Acoustic area / premises", language),
            html.escape(t(result.limits.description, language)),
        )
    )
    return [(label, value) for label, value in specs if value]


def _phase_table(
    result: ActivityAssessment, verbose: bool = False, language: str = "en"
) -> Any:
    """The full-width *fases de ruido* table.

    One row per noise phase of every assessed period: the phase label, its
    duration ``Ti``, the background-corrected ``LAeq,Ti``, the three
    corrections and the resulting ``LKeq,Ti``. ``verbose`` adds the summed
    correction ``K`` column, which the regulation caps at 9 dB.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    header_style, label_style, value_style = analysis_cell_styles("rd1367phase")

    headers = [
        t("Period", language),
        t("Noise phase", language),
        "T<sub>i</sub> [h]",
        "L<sub>Aeq,Ti</sub> [dB]",
        "K<sub>t</sub>",
        "K<sub>f</sub>",
        "K<sub>i</sub>",
        "L<sub>Keq,Ti</sub> [dB]",
    ]
    widths = [22.0, 40.0, 16.0, 26.0, 14.0, 14.0, 14.0, 28.0]
    if verbose:
        headers.insert(7, "K [dB]")
        widths = [20.0, 34.0, 15.0, 25.0, 13.0, 13.0, 13.0, 16.0, 25.0]

    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for period in result.periods:
        period_label = t(_PERIOD_LABELS[period.period], language)
        for index, phase in enumerate(period.phases, start=1):
            name = phase.label or t("Phase {n}", language).format(n=index)
            row = [
                Paragraph(period_label if index == 1 else "", label_style),
                Paragraph(html.escape(name), label_style),
                Paragraph(_fmt(phase.hours, language), value_style),
                Paragraph(_fmt(phase.laeq, language), value_style),
                Paragraph(_fmt(phase.kt, language, decimals=0), value_style),
                Paragraph(_fmt(phase.kf, language, decimals=0), value_style),
                Paragraph(_fmt(phase.ki, language, decimals=0), value_style),
                Paragraph(
                    f"<b>{_fmt(phase.lkeq, language)}</b>", value_style
                ),
            ]
            if verbose:
                row.insert(
                    7,
                    Paragraph(
                        _fmt(phase.correction, language, decimals=0), value_style
                    ),
                )
            data.append(row)
    return stacked_table(data, [w * mm for w in widths])


def _status_markup(passed: bool | None, language: str = "en") -> str:
    """Inline PASS / FAIL / not-assessed markup for a criterion cell."""
    if passed is None:
        return f"<font color='{_MUTED_HEX}'>&#8211;</font>"
    colour = _VERDICT_OK_HEX if passed else _VERDICT_BAD_HEX
    label = t("PASS", language) if passed else t("FAIL", language)
    return f"<font color='{colour}'>&#9679; {label}</font>"


def _criterion_cell(
    value: float | None, limit: float, passed: bool | None, language: str
) -> str:
    """``value / limit`` markup for one Article 25 criterion."""
    if value is None:
        return f"<font color='{_MUTED_HEX}'>&#8211;</font>"
    colour = _VERDICT_BAD_HEX if passed is False else None
    text = f"{_fmt(value, language, decimals=0)} / {_fmt(limit, language, decimals=0)}"
    if colour is not None:
        return f"<font color='{colour}'><b>{text}</b></font>"
    return text


def _results_table(result: ActivityAssessment, language: str = "en") -> Any:
    """The per-period results table against the Article 25.1 b criteria.

    One row per assessed period, each cell showing the assessed index against
    its own allowance: the largest measured ``LKeq,Ti`` against ``limit + 5``
    dB, the daily ``LKeq,x`` against ``limit + 3`` dB and the annual ``LK,x``
    against the limit itself.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    header_style, label_style, value_style = analysis_cell_styles("rd1367result")

    headers = [
        t("Period", language),
        t("Limit [dB]", language),
        "L<sub>Keq,Ti</sub> / +5",
        "L<sub>Keq,x</sub> / +3",
        t("L<sub>K,x</sub> / limit", language),
        t("Result", language),
    ]
    widths = [24.0, 24.0, 32.0, 32.0, 32.0, 30.0]
    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for period in result.periods:
        data.append(
            [
                Paragraph(t(_PERIOD_LABELS[period.period], language), label_style),
                Paragraph(_fmt(period.limit, language, decimals=0), value_style),
                Paragraph(
                    _criterion_cell(
                        period.max_phase_level,
                        period.phase_limit,
                        period.phase_pass,
                        language,
                    ),
                    value_style,
                ),
                Paragraph(
                    _criterion_cell(
                        float(period.reported_level),
                        period.daily_limit,
                        period.daily_pass,
                        language,
                    ),
                    value_style,
                ),
                Paragraph(
                    _criterion_cell(
                        None
                        if period.reported_long_term is None
                        else float(period.reported_long_term),
                        period.limit,
                        period.long_term_pass,
                        language,
                    ),
                    value_style,
                ),
                Paragraph(
                    _status_markup(period.complies, language), value_style
                ),
            ]
        )
    return stacked_table(data, [w * mm for w in widths])


def _statement(result: ActivityAssessment, language: str = "en") -> tuple[str, list[str]]:
    """The boxed conclusion: the governing period and the overall verdict."""
    governing: PeriodAssessment = max(
        result.periods, key=lambda p: p.reported_level - p.limit
    )
    if result.complies:
        statement = t(
            "The activity <b>meets</b> the immission limit values of "
            "RD 1367/2007",
            language,
        )
    else:
        statement = t(
            "The activity <b>exceeds</b> the immission limit values of "
            "RD 1367/2007",
            language,
        )
    extended = [
        t("Governing period: {period}", language).format(
            period=t(_PERIOD_LABELS[governing.period], language)
        ),
        t("L<sub>Keq,x</sub> = {value} dB, limit {limit} dB", language).format(
            value=_fmt(float(governing.reported_level), language, decimals=0),
            limit=_fmt(governing.limit, language, decimals=0),
        ),
    ]
    if governing.reported_long_term is not None:
        extended.append(
            t("L<sub>K,x</sub> = {value} dB", language).format(
                value=_fmt(
                    float(governing.reported_long_term), language, decimals=0
                )
            )
        )
    return statement, extended


def _verdict(result: ActivityAssessment, language: str = "en") -> tuple[str, bool]:
    """Verdict text and PASS flag against the Article 25 criteria."""
    failed = [p for p in result.periods if not p.complies]
    if not failed:
        text = t(
            "every assessed index is within the applicable limit values "
            "(Article 25 of RD 1367/2007)",
            language,
        )
    else:
        names = ", ".join(
            t(_PERIOD_LABELS[p.period], language) for p in failed
        )
        text = t(
            "the applicable limit values are exceeded in: {periods}", language
        ).format(periods=names)
    return text, not failed


def render_activity_report(
    result: ActivityAssessment,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "es",
) -> str:
    """Render an RD 1367/2007 activity inspection fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.environmental.spanish_regulation.ActivityAssessment`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity (``specimen`` the activity, ``client`` its holder,
        ``test_room`` the assessment point, ``instrumentation``,
        ``calibration`` and ``test_date``) and the footer identity.
    :param verbose: Add the summed-correction ``K`` column to the phase table.
    :param language: Fiche language: ``"es"`` (default, the language of the
        regulation) or ``"en"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Activity noise assessment (RD 1367/2007)", language)

    article = (
        "Article 25.1 b" if result.new_activity else "Article 25.2"
    )
    basis = t(
        "Corrected equivalent continuous level L<sub>Keq,T</sub> = "
        "L<sub>Aeq,T</sub> + K<sub>t</sub> + K<sub>f</sub> + K<sub>i</sub> "
        "(Annex I A.2 c), determined by the reference procedures of Annex IV; "
        "compliance assessed per {article} of Real Decreto 1367/2007.",
        language,
    ).format(article=t(article, language))

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    header_pairs = _metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph(t("Noise phases", language), caption_style))
    flow.append(_phase_table(result, verbose, language))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph(t("Assessment by period", language), caption_style))
    flow.append(_results_table(result, language))
    flow.append(Spacer(1, 6))

    def _assessment_plot(ax: Any = None, language: str = "en", **kwargs: Any) -> Any:
        from .._plot.environmental import plot_activity_assessment

        return plot_activity_assessment(result, ax=ax, language=language, **kwargs)

    # The fiche must stay on a single page whatever the case carries, and the
    # phase table grows with the number of noise phases (up to three periods
    # split into several phases each). The figure is the elastic element, so
    # its height shrinks as the table grows.
    rows = sum(len(period.phases) for period in result.periods)
    figure_height = 2.7 if rows <= 5 else max(1.7, 2.7 - 0.22 * (rows - 5))
    flow.append(
        render_figure_drawing(
            _assessment_plot, 174 * mm, y_top=None,
            figsize=(9.2, figure_height), language=language,
        )
    )
    flow.append(Spacer(1, 5))

    statement, extended = _statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))

    text, passed = _verdict(result, language)
    flow.extend(verdict_flow(text, passed, styles, language))

    basis_strip_style = measurement_basis_style()
    flow.append(
        Paragraph(
            t(
                "K<sub>t</sub> follows from the one-third-octave difference "
                "L<sub>t</sub> = L<sub>f</sub> &#8722; L<sub>s</sub> against "
                "the mean of the adjacent bands, K<sub>f</sub> from "
                "L<sub>Ceq,Ti</sub> &#8722; L<sub>Aeq,Ti</sub> and "
                "K<sub>i</sub> from L<sub>AIeq,Ti</sub> &#8722; "
                "L<sub>Aeq,Ti</sub>; each is 0, 3 or 6 dB and their sum never "
                "exceeds 9 dB (Annex IV A.3.3).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.append(
        Paragraph(
            t(
                "The period level is the duration-weighted energy mean of the "
                "phase levels, rounded by adding 0,5 dB and taking the integer "
                "part (Annex IV A.3.4.2). The limit values are respected when "
                "no annual L<sub>K,x</sub> exceeds the table value, no daily "
                "L<sub>Keq,x</sub> exceeds it by more than 3 dB and no measured "
                "L<sub>Keq,Ti</sub> exceeds it by more than 5 dB.",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(
        footer_flow(
            metadata,
            language,
            disclaimer=(
                "The results relate only to the noise phases measured at the "
                "assessment point and remain valid while the operating and "
                "environmental conditions described here are maintained."
            ),
        )
    )

    return build_document(path, flow, title)
