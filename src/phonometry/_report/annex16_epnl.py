#  Copyright (c) 2026. Jose M. Requena-Plens
"""ICAO Annex 16 EPNL aircraft-noise-certification fiche (reportlab renderer).

Renders an :class:`~phonometry.aircraft.aircraft_noise.EPNLResult` to a
one-page PDF laid out like an aircraft-noise-certification data sheet:

* a title and the standard-basis line (measurement standard + ICAO Annex 16
  Vol. I Appendix 2);
* an optional TCDSN-style metadata header block (aircraft, manufacturer / type
  certificate holder, applicant, measurement point ...), rendered only for the
  fields supplied on the :class:`ReportMetadata`;
* a compact metrics table of the informational intermediate quantities (the
  peak PNLTM, the duration correction ``D``, the 10 dB-down record window and,
  when it is non-zero, the bandsharing adjustment);
* a full-width, landscape PNLT-versus-time plot drawn by the result's own
  ``plot(ax=...)`` (the PNL and PNLT traces, the marked PNLTM and the shaded
  10 dB-down integration window);
* a boxed single-number result ``EPNL = X EPNdB``;
* a Level | Limit | Margin verdict row when a certification limit is supplied
  (``metadata.requirement``); the EPNL passes at or below it;
* a static reference-conditions strip (25 C i.e. ISA + 10 C, 70 % RH,
  1013.25 hPa sea-level pressure, zero wind; Annex 16 Vol I Part II,
  3.6.1.5); and
* a footer identity/disclaimer block, plus a line stating the fiche is a
  computational result and not an official State noise certificate.

Like :mod:`.broadcast` this uses a stacked layout rather than the narrow
two-panel one: the PNLT-versus-time trace is landscape and reads best full
width. The quantity-independent skeleton lives in :mod:`._layout`; this module
only holds the Annex 16 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import decimal_comma, format_number, t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    build_document,
    compliance_table,
    display_round,
    document_styles,
    fmt_num,
    footer_flow,
    grid_table,
    metrics_table,
    render_figure_drawing,
    result_box,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..aircraft.aircraft_noise import EPNLResult


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the certification header grid.

    Only fields that are set are returned. EPNL is an aircraft-flyover metric,
    so the room/climate fields of the insulation fiche do not apply; the
    generic identity fields are relabelled to the certification vocabulary
    (aircraft, type-certificate holder, applicant, measurement point).
    """
    specs: list[tuple[str, str | None]] = [
        (t("Aircraft", language), metadata.specimen),
        (t("Manufacturer / TC holder", language), metadata.manufacturer),
        (t("Applicant", language), metadata.client),
        (t("Measurement point", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
    ]
    return [
        (label, html.escape(str(value)))
        for label, value in specs
        if value is not None
    ]


def _metric_rows(
    result: EPNLResult, language: str = "en"
) -> list[tuple[str, str]]:
    """The intermediate EPNL quantities shown in the left-hand metrics table.

    These are informational: PNLTM, the duration correction ``D``, the 10
    dB-down integration window (records, 1-based for display) and, only when it
    is non-zero, the bandsharing adjustment.
    """
    k_first, k_last = result.band_limits
    rows = [
        (t("PNLTM [PNdB]", language), fmt_num(result.pnltm, language)),
        (
            t("Duration correction D [dB]", language),
            fmt_num(result.duration_correction, language),
        ),
        (
            t("10 dB-down window (records)", language),
            f"{k_first + 1}&#8211;{k_last + 1}",
        ),
    ]
    if abs(result.bandsharing_adjustment) > 1e-9:
        rows.append(
            (
                t("Bandsharing adjustment [dB]", language),
                fmt_num(result.bandsharing_adjustment, language),
            )
        )
    return rows


def _statement(result: EPNLResult, language: str = "en") -> str:
    """The boxed single-number statement ``EPNL = X EPNdB``.

    Annex 16 determines the EPNL to one decimal place, so the displayed value
    is the model-rounded tenth (halves away from zero), the same value the
    verdict compares.
    """
    epnl = format_number(display_round(result.epnl), language, decimals=1)
    return f"EPNL = <b>{epnl} EPNdB</b>"


def _verdict_rows(
    result: EPNLResult, limit: float, language: str = "en"
) -> list[tuple[str, str, str, str]]:
    """The Level | Limit | Margin compliance row for a supplied EPNL limit.

    The EPNL passes at or below the certification limit; the limit cell also
    carries the signed margin ``limit - EPNL`` (positive when compliant).
    Annex 16 determines the EPNL "to one decimal place", so the level is
    rounded to a tenth once (halves away from zero) and that same value
    drives the displayed level, the margin and the pass/fail status; the
    printed numbers can then never contradict the verdict at the limit.
    """
    epnl = display_round(result.epnl)
    margin = limit - epnl
    status = "pass" if epnl <= limit + 1e-9 else "fail"
    return [
        (
            t("EPNL [EPNdB]", language),
            format_number(epnl, language, decimals=1),
            t("&#8804; {limit} (margin {margin})", language).format(
                limit=fmt_num(limit, language),
                margin=decimal_comma(f"{margin:+.1f}", language),
            ),
            status,
        )
    ]


def render_annex16_epnl_report(
    result: EPNLResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ICAO Annex 16 EPNL certification fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.aircraft.aircraft_noise.EPNLResult` carrying the
        PNL/PNLT time histories, the peak PNLTM, the duration correction, the
        bandsharing adjustment, the EPNL and the 10 dB-down window.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        prediction fiche (metrics + plot + result, no header and no verdict). A
        supplied ``requirement`` is read as the certification EPNL limit in
        EPNdB (the EPNL passes at or below it).
    :param verbose: Accepted for a uniform ``.report()`` signature; the EPNL
        fiche has a single stacked body layout, so it has no effect.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # uniform signature; the fiche has one stacked body layout
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
    title = t("Aircraft noise certification - EPNL result", language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t("{standard}. Effective Perceived Noise Level per ICAO Annex 16, Volume I, Appendix 2.", language).format(
            standard=html.escape(measurement_standard)
        )
    else:
        basis = t("Effective Perceived Noise Level (EPNL) per ICAO Annex 16, Volume I, Appendix 2.", language)

    muted_strip_style = ParagraphStyle(
        "fiche_reference_conditions", parent=getSampleStyleSheet()["Normal"],
        fontSize=7.5, leading=10, textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=2,
    )

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    if metadata is not None and not metadata.is_empty():
        header_pairs = _metadata_pairs(metadata, language)
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(
        Paragraph(t("Reference conditions: 25 &#176;C (ISA + 10 &#176;C), 70% relative humidity, 1013.25 hPa sea-level pressure, zero wind (ICAO Annex 16 Vol I Part II, 3.6.1.5).", language), muted_strip_style)
    )
    flow.append(Spacer(1, 8))

    flow.append(
        Paragraph(t("Intermediate quantities", language), caption_style)
    )
    flow.append(
        metrics_table(
            _metric_rows(result, language), col_widths=[48 * mm, 26 * mm]
        )
    )
    flow.append(Spacer(1, 8))

    # Full-width, landscape PNLT-vs-time plot (self-scaling axis).
    plot_drawing = render_figure_drawing(
        result.plot, 174 * mm, y_top=None, figsize=(9.2, 4.2), language=language
    )
    flow.append(plot_drawing)
    flow.append(Spacer(1, 8))

    flow.append(result_box(_statement(result, language), styles, accent))

    limit = (
        float(metadata.requirement)
        if metadata is not None and metadata.requirement is not None
        else None
    )
    if limit is not None:
        flow.append(Spacer(1, 6))
        flow.append(
            Paragraph(t("Certification limit", language), caption_style)
        )
        flow.append(
            compliance_table(
                _verdict_rows(result, limit, language), language=language
            )
        )

    flow.extend(footer_flow(metadata, language))

    disclaimer_style = ParagraphStyle(
        "fiche_certificate_disclaimer", parent=getSampleStyleSheet()["Normal"],
        fontSize=7.5, leading=10, textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=2,
    )
    flow.append(
        Paragraph(t("This is a computational EPNL result per ICAO Annex 16 Vol I Appendix 2; it is not an official State noise certificate (e.g. EASA Form 45).", language), disclaimer_style)
    )

    return build_document(path, flow, title)
