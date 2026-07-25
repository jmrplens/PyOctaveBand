#  Copyright (c) 2026. Jose M. Requena-Plens
"""Open-plan office acoustics fiche (reportlab renderer, ISO 3382-3:2012).

Renders a :class:`~phonometry.room.open_plan.OpenPlanResult` to a one-page PDF
laid out like an open-plan-office speech-privacy measurement report
(ISO 3382-3:2012, measured along a line of workstations by the method of
Clause 5.2):

* a title and the standard-basis line (measurement standard + ISO 3382-3:2012);
* an optional metadata header block (client, office/zone, description, floor
  area, source and measurement positions, climate ...), rendered only for the
  fields supplied on the :class:`ReportMetadata`;
* a compact metrics table of the four single-number quantities of Clause 4:
  the spatial decay rate of the A-weighted SPL of speech ``D2,S``, the nominal
  A-weighted speech level at 4 m ``Lp,A,S,4m``, the distraction distance ``rD``
  (STI = 0.50) and the privacy distance ``rP`` (STI = 0.20);
* a full-width, landscape spatial-decay plot drawn by the result's own
  ``plot(ax=...)`` (the Clause 6.2 regression of the A-weighted speech level on
  the logarithmic distance axis, with the 4 m read-off marked and the ``rD`` /
  ``rP`` distance crossings shown), the Figure 3 style spatial-decay curve;
* a boxed single-number result ``D2,S`` with the remaining quantities alongside;
* an optional verdict row when a target spatial decay rate is supplied
  (``metadata.requirement``, read as the minimum acceptable ``D2,S``; a larger
  spatial decay is better, so the room passes at or above it), reflecting the
  informative quality ranges of Annex A; and
* a short measurement-basis strip and a footer identity/disclaimer block.

Like :mod:`.annex16_epnl` this uses a stacked layout rather than the narrow
two-panel one: the spatial-decay trace is landscape and reads best full width.
The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the ISO 3382-3 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING, Any

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _REPORTLAB_HINT,
    build_document,
    display_round,
    document_styles,
    escaped_pairs,
    fmt_meta,
    footer_flow,
    grid_table,
    measurement_basis_style,
    metrics_table,
    render_figure_drawing,
    result_box,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..room.open_plan import OpenPlanResult


def _num(value: float, language: str, *, decimals: int = 1) -> str:
    """Format a single-number quantity, or an em dash when it is not finite.

    A non-finite quantity (``D2,S`` / ``Lp,A,S,4m`` when fewer than two
    positions fall in the 2 m to 16 m range, or ``rD`` / ``rP`` when the STI
    does not decrease with distance) is shown as ``"—"``, the accredited
    empty-cell symbol, rather than a spurious number.
    """
    v = float(value)
    if not math.isfinite(v):
        return "—"
    return format_number(v, language, decimals=decimals)


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the open-plan header grid.

    Only fields that are set are returned, so empty rows never appear. An
    open-plan office is a single furnished space measured along a line of
    workstations, so the floor ``area``, the number of source positions and the
    number of measurement positions (``receiver_positions``) are used rather
    than a source/receiving room pair.
    """

    def num(value: float | None) -> str | None:
        return fmt_meta(value, language) if value is not None else None

    def count(value: int | None) -> str | None:
        return str(int(value)) if value is not None else None

    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Office / zone", language), metadata.test_room),
        (t("Description", language), metadata.specimen),
        (t("Floor area S [m<super>2</super>]", language), num(metadata.area)),
        (t("Source positions", language), count(metadata.source_positions)),
        (t("Measurement positions", language), count(metadata.receiver_positions)),
        (t("Instrumentation", language), metadata.instrumentation),
        (t("Temperature [&#176;C]", language), num(metadata.temperature)),
        (t("Relative humidity [%]", language), num(metadata.relative_humidity)),
        (t("Ambient pressure [kPa]", language), num(metadata.pressure)),
        (t("Date of test", language), metadata.test_date),
    ]
    return escaped_pairs(specs)


def _metric_rows(
    result: OpenPlanResult, language: str = "en"
) -> list[tuple[str, str]]:
    """The four single-number quantities shown in the left-hand metrics table.

    ``D2,S`` (dB per distance doubling), ``Lp,A,S,4m`` (dB), ``rD`` (m) and
    ``rP`` (m) of ISO 3382-3:2012 Clause 4; a non-finite quantity is rendered
    as an em dash.
    """
    return [
        (
            t("Spatial decay rate D<sub>2,S</sub> [dB]", language),
            _num(result.d2s, language),
        ),
        (
            t("Speech level L<sub>p,A,S,4m</sub> [dB]", language),
            _num(result.lp_as_4m, language),
        ),
        (
            t("Distraction distance r<sub>D</sub> [m]", language),
            _num(result.rd, language),
        ),
        (
            t("Privacy distance r<sub>P</sub> [m]", language),
            _num(result.rp, language),
        ),
    ]


def _statement(
    result: OpenPlanResult, language: str = "en"
) -> tuple[str, list[str]]:
    """The boxed spatial-decay-rate statement and its remaining quantities.

    ``D2,S`` is boxed as the headline descriptor of the standard's title
    quantity (the spatial decay of speech); ``Lp,A,S,4m``, ``rD`` and ``rP``
    follow alongside.
    """
    statement = t(
        "D<sub>2,S</sub> = <b>{value} dB</b> per distance doubling", language
    ).format(value=_num(result.d2s, language))
    extended = [
        t("L<sub>p,A,S,4m</sub> = {value} dB", language).format(
            value=_num(result.lp_as_4m, language)
        ),
        t("r<sub>D</sub> = {value} m", language).format(
            value=_num(result.rd, language)
        ),
        t("r<sub>P</sub> = {value} m", language).format(
            value=_num(result.rp, language)
        ),
    ]
    return statement, extended


def _verdict(
    result: OpenPlanResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag for a supplied target spatial decay rate.

    ISO 3382-3:2012 Annex A (informative) gives quality ranges for the spatial
    decay of speech in which a larger ``D2,S`` is better, so the requirement is
    read as the minimum acceptable ``D2,S`` and the room passes when its
    measured value is at or above it. The comparison is made on the value
    rounded exactly as the fiche displays it (one decimal, halves away from
    zero), so the printed number can never contradict the verdict.
    """
    d2s = display_round(result.d2s, 1)
    passed = math.isfinite(d2s) and d2s >= display_round(requirement, 1)
    text = t(
        "D<sub>2,S</sub> = {value} dB, required &#8805; {req} dB", language
    ).format(
        value=_num(result.d2s, language),
        req=format_number(requirement, language, decimals=1),
    )
    return text, passed


def render_iso3382_3_report(
    result: OpenPlanResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an open-plan office acoustics fiche to a PDF at ``path``.

    :param result: An :class:`~phonometry.room.open_plan.OpenPlanResult`
        carrying the four single-number quantities ``d2s``, ``lp_as_4m``,
        ``rd`` and ``rp``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        characterisation fiche (metrics + plot + result, no header and no
        verdict). A supplied ``requirement`` is read as the minimum acceptable
        spatial decay rate ``D2,S`` in dB (the room passes at or above it).
    :param verbose: Accepted for a uniform ``.report()`` signature; the fiche
        has a single stacked body layout, so it has no effect.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab or, for the embedded spatial-decay chart,
        matplotlib is not installed; both are required whenever the regression
        is defined (``pip install "phonometry[report,plot]"``).
    """
    del verbose  # uniform signature; the fiche has one stacked body layout
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, Spacer
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Open-plan office acoustics", language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t(
            "{standard} measurement of open-plan office speech-level spatial "
            "decay and speech privacy per ISO 3382-3:2012.",
            language,
        ).format(standard=html.escape(measurement_standard))
    else:
        basis = t(
            "Open-plan office speech-level spatial decay and speech privacy "
            "per ISO 3382-3:2012.",
            language,
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
    flow.append(Spacer(1, 8))

    flow.append(
        Paragraph(t("Single-number quantities", language), caption_style)
    )
    flow.append(
        metrics_table(
            _metric_rows(result, language), col_widths=[62 * mm, 26 * mm]
        )
    )
    flow.append(Spacer(1, 8))

    # Full-width, landscape spatial-decay plot (self-scaling level axis). The
    # regression line is defined only when D2,S / Lp,A,S,4m are finite (at least
    # two positions in the 2 m to 16 m range); otherwise the result's plot()
    # would raise, so the figure is omitted and the metrics table carries the
    # em-dashed quantities on its own.
    if math.isfinite(result.d2s) and math.isfinite(result.lp_as_4m):
        plot_drawing = render_figure_drawing(
            result.plot, 174 * mm, y_top=None, figsize=(9.2, 4.4),
            language=language,
        )
        flow.append(plot_drawing)
        flow.append(Spacer(1, 8))

    statement, extended = _statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))
    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))

    basis_strip_style = measurement_basis_style()
    flow.append(
        Paragraph(
            t(
                "Spatial decay rate D<sub>2,S</sub> and the nominal 4 m speech "
                "level L<sub>p,A,S,4m</sub> from a least-squares fit of the "
                "A-weighted speech level against lg(r/r<sub>0</sub>) "
                "(r<sub>0</sub> = 1 m) over the 2 m to 16 m positions "
                "(ISO 3382-3:2012, 6.2, Equation (5)); D<sub>2,S</sub> is the "
                "per-distance-doubling decay.",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.append(
        Paragraph(
            t(
                "Distraction distance r<sub>D</sub> (STI = 0.50) and privacy "
                "distance r<sub>P</sub> (STI = 0.20) from a linear regression "
                "of the speech transmission index against distance (full "
                "IEC 60268-16 method, spatially averaged background noise; "
                "ISO 3382-3:2012, 6.3, Figure 3 b).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
