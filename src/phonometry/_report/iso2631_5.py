#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Whole-body multiple-shock health-risk fiche (reportlab renderer).

Renders a
:class:`~phonometry.vibration.human.multiple_shock.MultipleShockResult` to a
one-page PDF laid out like a whole-body multiple-shock health-risk assessment
sheet (ISO 2631-5:2018, the normative Clause 5 spinal-response dose and the
Annex C assessment of adverse health effects):

* a title and the standard-basis line naming ISO 2631-5:2018 (Clause 5 +
  Annex C);
* an optional metadata header grid: client, subject, workplace/vehicle,
  instrumentation, calibration and date;
* an exposure-scenario grid (always shown) with the subject sex, the age ``b``
  at which the exposure started, the number of exposure years ``n``, the number
  of exposure days per year ``N`` and the number of counted response shocks;
* the dose-and-stress analysis: a full-width table of the acceleration dose
  ``Dz`` (Formula 3), the daily acceleration dose ``Dzd`` (Formula 4), the daily
  compressive stress ``Sd`` (Formula C.1), the cumulative stress variable ``R``
  (Formula C.3) and the probability of lumbar injury ``P`` (Formula C.5), each
  with its unit and clause reference;
* the injury-probability curve drawn by the result's own ``plot(ax=...)`` (the
  Weibull ``P(R)`` with the 10 / 50 / 90 % risk levels of Table C.2 and this
  assessment's ``R`` marked);
* the boxed result: the stress variable ``R`` and the injury probability ``P``
  with the Annex C risk classification (low / moderate / high / very high
  probability of an adverse health effect);
* a classification table against the Table C.2 risk levels, the band this
  assessment falls in marked, plus a zone row stating the classification; and
* a footer note that ISO 2631-5:2018 defines no exposure limit and that the
  model is vertical-axis only, followed by the footer identity/disclaimer block.

Like :mod:`.human_vibration` this uses a stacked table layout rather than the
narrow two-panel one: the analysis table needs the full content width and the
injury-probability chart is landscape. The quantity-independent skeleton lives
in :mod:`._layout`; this module only holds the ISO 2631-5 specifics. reportlab,
matplotlib and svglib are soft dependencies imported lazily (reportlab and
svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    analysis_cell_styles,
    build_document,
    display_round,
    document_styles,
    escaped_pairs,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    stacked_table,
)

if TYPE_CHECKING:
    from ..vibration.human.multiple_shock import MultipleShockResult
    from .metadata import ReportMetadata

#: Display precision of the acceleration dose and compressive stress (two
#: decimals resolve the Annex C worked example Dz = 55.97 m/s2 and Sd = 1.62
#: MPa).
_DOSE_DECIMALS = 2
#: Display precision of the cumulative stress variable R (two decimals resolve
#: the Annex C worked example R = 1.22 and the Table C.2 thresholds).
_R_DECIMALS = 2

#: Risk-band phrases keyed by band index (0 = low ... 3 = very high). The bands
#: partition the stress variable R by the Table C.2 risk levels; the moderate
#: band matches the Annex C worked example wording ("a moderate adverse health
#: effect (10 % < risk of injury < 50 %)").
_BAND_LABELS: tuple[str, ...] = (
    "low probability of an adverse health effect",
    "moderate probability of an adverse health effect",
    "high probability of an adverse health effect",
    "very high probability of an adverse health effect",
)


def _fmt(value: float, decimals: int, language: str = "en") -> str:
    """A quantity rounded to a display precision, with the locale separator."""
    from ._i18n import format_number

    return format_number(
        display_round(float(value), decimals), language, decimals=decimals
    )


def _fmt_percent(probability: float, language: str = "en") -> str:
    """The injury probability as a whole-number percentage (e.g. ``37 %``)."""
    from ._i18n import format_number

    return format_number(100.0 * float(probability), language, decimals=0) + " %"


def _band_index(result: MultipleShockResult) -> int:
    """The Annex C risk band (0..3) the assessment's ``R`` falls in.

    The comparison is made on ``R`` and the Table C.2 thresholds rounded exactly
    as the fiche displays them (two decimals, halves away from zero), so the
    boxed classification can never contradict the printed numbers at a threshold
    boundary. The bands are ``R < R10`` (low), ``R10 <= R < R50`` (moderate),
    ``R50 <= R < R90`` (high) and ``R >= R90`` (very high), where ``R10/R50/R90``
    are the Table C.2 stress variables for 10 / 50 / 90 % risk of injury.
    """
    r = display_round(float(result.risk), _R_DECIMALS)
    thresholds = [display_round(float(x), _R_DECIMALS) for x in result.risk_thresholds]
    band = 0
    for threshold in thresholds:
        if r >= threshold:
            band += 1
    return band


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _metadata_pairs(
    metadata: ReportMetadata | None, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the header grid.

    A multiple-shock health-risk sheet identifies the client, the subject whose
    exposure was assessed and the workplace or vehicle, plus the instrumentation
    and calibration traceability. Only supplied fields are returned.
    """
    if metadata is None:
        return []
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), _esc(metadata.client)),
        (t("Subject", language), _esc(metadata.specimen)),
        (t("Workplace / vehicle", language), _esc(metadata.test_room)),
        (t("Date of assessment", language), _esc(metadata.test_date)),
        (t("Instrumentation", language), _esc(metadata.instrumentation)),
        (t("Calibration", language), _esc(metadata.calibration)),
    ]
    return [(label, value) for label, value in specs if value]


def _scenario_pairs(
    result: MultipleShockResult, language: str = "en"
) -> list[tuple[str, str]]:
    """The exposure-scenario grid pairs (subject sex and exposure period).

    These come from the result rather than the metadata, so they are always
    shown: the subject sex (which sets the stress conversion ``mz`` and the
    Weibull coefficients), the age ``b`` at which the exposure started, the
    number of exposure years ``n``, the number of exposure days per year ``N``
    and the number of positive response shocks the dose counted.
    """
    from ._i18n import format_number

    specs: list[tuple[str, str | None]] = [
        (t("Subject sex", language), t(str(result.sex), language)),
        (
            t("Age at start b [years]", language),
            format_number(float(result.start_age), language, decimals=0, trim=True),
        ),
        (t("Exposure duration n [years]", language), str(int(result.years))),
        (
            t("Exposure days per year N", language),
            format_number(float(result.days_per_year), language, decimals=0, trim=True),
        ),
        (t("Counted response shocks", language), str(len(result.peaks))),
    ]
    return escaped_pairs(specs)


def _unit_ms2() -> str:
    """The acceleration-dose unit ``m/s^2`` as reportlab markup."""
    return "m/s<sup>2</sup>"


def _analysis_table(result: MultipleShockResult, language: str = "en") -> Any:
    """The dose-and-stress analysis table (Clause 5 dose and Annex C stress).

    One row per quantity (quantity, symbol, value with unit, clause reference):
    the acceleration dose ``Dz`` (Formula 3), the daily acceleration dose
    ``Dzd`` (Formula 4), the daily compressive stress ``Sd`` (Formula C.1), the
    cumulative stress variable ``R`` (Formula C.3) and the probability of lumbar
    injury ``P`` (Formula C.5).
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso26315")
    ms2 = _unit_ms2()
    # The clause references keep their number verbatim; only the word
    # "Formula" is localised.
    formula = t("Formula", language)

    rows: list[tuple[str, str, str, str]] = [
        (
            t("Acceleration dose", language),
            "D<sub>z</sub>",
            f"{_fmt(result.acceleration_dose, _DOSE_DECIMALS, language)} {ms2}",
            f"{formula} (3)",
        ),
        (
            t("Daily acceleration dose", language),
            "D<sub>zd</sub>",
            f"{_fmt(result.daily_dose, _DOSE_DECIMALS, language)} {ms2}",
            f"{formula} (4)",
        ),
        (
            t("Daily compressive stress", language),
            "S<sub>d</sub>",
            f"{_fmt(result.compression_dose, _DOSE_DECIMALS, language)} MPa",
            f"{formula} (C.1)",
        ),
        (
            t("Cumulative stress variable", language),
            "R",
            _fmt(result.risk, _R_DECIMALS, language),
            f"{formula} (C.3)",
        ),
        (
            t("Probability of lumbar injury", language),
            "&#928;",
            _fmt_percent(result.probability, language),
            f"{formula} (C.5)",
        ),
    ]

    data: list[list[Any]] = [
        [
            fiche_paragraph(t("Quantity", language), header_style),
            fiche_paragraph(t("Symbol", language), header_style),
            fiche_paragraph(t("Value", language), header_style),
            fiche_paragraph(t("Reference", language), header_style),
        ]
    ]
    for quantity, symbol, value, reference in rows:
        data.append(
            [
                fiche_paragraph(quantity, label_style),
                fiche_paragraph(symbol, value_style),
                fiche_paragraph(value, value_style),
                fiche_paragraph(reference, value_style),
            ]
        )
    table = stacked_table(data, [70 * mm, 26 * mm, 42 * mm, 36 * mm])
    table.setStyle(
        [
            ("TOPPADDING", (0, 0), (-1, -1), 2.0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
        ]
    )
    return table


def _classification_table(result: MultipleShockResult, language: str = "en") -> Any:
    """The Annex C risk-classification table against the Table C.2 levels.

    Four rows partition the stress variable ``R`` by the Table C.2 risk levels
    (low / moderate / high / very high); the band this assessment falls in is
    marked with a filled accent dot in the last column.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso26315cls")
    r10, r50, r90 = (_fmt(x, _R_DECIMALS, language) for x in result.risk_thresholds)
    active = _band_index(result)

    bands: list[tuple[str, str, str]] = [
        (
            t("Low", language),
            t("&lt; 10 %", language),
            t("R &lt; {r10}", language).format(r10=r10),
        ),
        (
            t("Moderate", language),
            t("10 % to 50 %", language),
            t("{r10} &#8804; R &lt; {r50}", language).format(r10=r10, r50=r50),
        ),
        (
            t("High", language),
            t("50 % to 90 %", language),
            t("{r50} &#8804; R &lt; {r90}", language).format(r50=r50, r90=r90),
        ),
        (
            t("Very high", language),
            t("&#8805; 90 %", language),
            t("R &#8805; {r90}", language).format(r90=r90),
        ),
    ]

    def _mark(is_active: bool) -> str:
        if is_active:
            return (
                f"<font color='{_ACCENT_HEX}'>&#9679; "
                f"{t('This assessment', language)}</font>"
            )
        return f"<font color='{_MUTED_HEX}'>&#8211;</font>"

    data: list[list[Any]] = [
        [
            fiche_paragraph(t("Risk band", language), header_style),
            fiche_paragraph(t("Risk of lumbar injury", language), header_style),
            fiche_paragraph(t("Stress variable R (Table C.2)", language), header_style),
            fiche_paragraph(t("Classification", language), header_style),
        ]
    ]
    for index, (band, probability, r_range) in enumerate(bands):
        data.append(
            [
                fiche_paragraph(band, label_style),
                fiche_paragraph(probability, value_style),
                fiche_paragraph(r_range, value_style),
                fiche_paragraph(_mark(index == active), label_style),
            ]
        )
    table = stacked_table(data, [40 * mm, 36 * mm, 62 * mm, 36 * mm])
    # Highlight the active band's row so the classification reads at a glance,
    # and tighten the row padding so the sheet stays on one page.
    table.setStyle(
        [
            (
                "BACKGROUND",
                (0, active + 1),
                (-1, active + 1),
                colors.HexColor("#dfe8f2"),
            ),
            ("TOPPADDING", (0, 0), (-1, -1), 2.0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
        ]
    )
    return table


def _statement(result: MultipleShockResult, language: str = "en") -> str:
    """The boxed result statement: R, the injury probability and the risk band."""
    band = t(_BAND_LABELS[_band_index(result)], language)
    return t("R = <b>{r}</b>, &#928; = <b>{p}</b> &nbsp; ({band})", language).format(
        r=_fmt(result.risk, _R_DECIMALS, language),
        p=_fmt_percent(result.probability, language),
        band=band,
    )


def _zone_row(result: MultipleShockResult, language: str = "en") -> Any:
    """The classification zone row stating the Annex C risk band for ``R``."""
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    from ._layout import fiche_paragraph

    band = t(_BAND_LABELS[_band_index(result)], language)
    zone_style = ParagraphStyle(
        "iso26315_zone",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=10,
        leading=14,
        spaceBefore=4,
    )
    lead = t("Health-risk classification", language)
    return fiche_paragraph(
        f"{lead}: {t('stress variable R = {r} &#8594; {band}', language).format(r=_fmt(result.risk, _R_DECIMALS, language), band=band)}",
        zone_style,
    )


def render_iso2631_5_report(
    result: MultipleShockResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a whole-body multiple-shock health-risk fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.vibration.human.multiple_shock.MultipleShockResult`
        (from
        :func:`~phonometry.vibration.human.multiple_shock.multiple_shock_assessment`).
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity (``client``, ``specimen`` the subject, ``test_room`` the
        workplace/vehicle), the ``instrumentation``/``calibration`` free text
        and the footer identity.
    :param verbose: Accepted for a uniform ``.report()`` signature; the fiche
        has a single stacked body layout, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab or matplotlib is not installed. The
        fiche always embeds the injury-probability chart, so both are required
        (``pip install "phonometry[report,plot]"``).
    """
    del verbose  # uniform signature; the fiche has one stacked body layout
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
    title = t("Whole-body multiple-shock health risk", language)

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(
            t(
                "Assessment of adverse health effects from whole-body vibration "
                "containing multiple shocks per ISO 2631-5:2018 (Clause 5 spinal "
                "response and Annex C risk model).",
                language,
            ),
            basis_style,
        ),
    ]

    # The identity metadata (if any) and the always-present exposure scenario
    # share one header grid, so the sheet stays on a single page.
    header_pairs = _metadata_pairs(metadata, language) + _scenario_pairs(
        result, language
    )
    flow.append(Spacer(1, 3))
    flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 6))

    flow.append(fiche_paragraph(t("Dose and stress analysis", language), caption_style))
    flow.append(_analysis_table(result, language))
    flow.append(Spacer(1, 4))
    # Full-width, landscape injury-probability curve (self-scaling 0-100 % axis).
    flow.append(
        render_figure_drawing(
            result.plot,
            174 * mm,
            y_top=None,
            figsize=(9.2, 2.5),
            language=language,
        )
    )
    flow.append(Spacer(1, 4))

    flow.append(result_box(_statement(result, language), styles, accent))
    flow.append(Spacer(1, 4))

    flow.append(
        fiche_paragraph(
            t("Annex C risk classification (Table C.2)", language), caption_style
        )
    )
    flow.append(_classification_table(result, language))
    flow.append(_zone_row(result, language))

    note_style = ParagraphStyle(
        "iso26315_notes",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7,
        leading=9,
        textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=5,
    )
    flow.append(
        fiche_paragraph(
            t(
                "ISO 2631-5:2018 defines no exposure limit; the stress variable "
                "R and the probability of lumbar injury P are the informative "
                "Annex C basis for assessing the potential of an adverse health "
                "effect. The risk bands follow the Table C.2 stress variables "
                "for 10 / 50 / 90 % risk of injury; the moderate band matches the "
                "Annex C worked example (a moderate adverse health effect, "
                "10 % to 50 % risk of injury). The Clause 5 model is "
                "vertical-axis only (clause 4 of the 2018 edition neglects the "
                "horizontal contributions to spinal compression); assess "
                "horizontal whole-body exposure with the ISO 2631-1 metrics "
                "instead, and the Annex A / Annex E finite-element model is out "
                "of scope.",
                language,
            ),
            note_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
