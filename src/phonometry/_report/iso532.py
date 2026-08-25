#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 532-1 Zwicker loudness fiche (reportlab renderer).

Renders a
:class:`~phonometry.psychoacoustics.loudness.zwicker.ZwickerLoudness` to a
one-page PDF laid out like an accredited loudness report:

* a title and the standard-basis line (measurement standard + ISO 532-1),
  stating the method used (stationary sounds, clause 5, or time-varying
  sounds, clause 6) and the sound field (free F / diffuse D) as clause 7 c)
  and d) require;
* an optional metadata header block (client, signal, laboratory ...), rendered
  only for the fields supplied on the :class:`ReportMetadata`;
* a two-panel body with a compact metrics table on the left (total loudness N,
  or maximum loudness Nmax with the N5/N10 percentiles for a time-varying
  result, and the loudness level LN) and the specific-loudness pattern N'(z)
  on the right, drawn by the result's own ``plot(ax=...)`` so the curve is
  native to the library;
* for a time-varying result, a full-width landscape loudness-versus-time
  panel N(t) (the mandatory clause 7 f) "loudness time function in sones");
* a boxed single-number result ``N = X sone (LN = Y phon)``;
* an optional verdict row when a maximum-loudness requirement is supplied;
* a footer identity/disclaimer block.

This is a non-band fiche: the plot self-scales (``y_top=None``) rather than
using the fixed dB axis of the insulation fiches. The quantity-independent
skeleton lives in :mod:`._layout`; this module only holds the ISO 532-1
specifics. reportlab, matplotlib and svglib are soft dependencies imported
lazily (reportlab and svglib ship in the ``phonometry[report]`` extra,
matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _REPORTLAB_HINT,
    build_document,
    display_round,
    document_styles,
    fmt_num,
    footer_flow,
    grid_table,
    metrics_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..psychoacoustics.loudness.zwicker import ZwickerLoudness
    from .metadata import ReportMetadata


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the loudness header grid.

    Only fields that are set are returned. Loudness is a signal metric, so the
    room/climate fields of the insulation fiche do not apply; only the generic
    identity fields are used.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Signal", language), metadata.specimen),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Test room", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
    ]
    return [
        (label, html.escape(str(value))) for label, value in specs if value is not None
    ]


def _is_time_varying(result: ZwickerLoudness) -> bool:
    """Whether the result came from the time-varying method (clause 6)."""
    return result.time is not None and result.loudness_vs_time is not None


def _loudness_symbol(result: ZwickerLoudness) -> str:
    """The reported loudness symbol: ``Nmax`` for time-varying, else ``N``.

    Clause 7 f) calls the reported maximum of a time-varying sound the
    loudness ``Nmax``; the stationary total loudness is plain ``N``.
    """
    return "N<sub>max</sub>" if _is_time_varying(result) else "N"


def _metric_rows(
    result: ZwickerLoudness, language: str = "en"
) -> list[tuple[str, str]]:
    """The scalar loudness results shown in the left-hand metrics table.

    Every value is printed to one decimal (the precision of the standard's
    own examples), matching the boxed result.
    """

    def d1(value: float) -> str:
        return format_number(value, language, decimals=1)

    loudness_label = (
        t("Maximum loudness N<sub>max</sub> [sone]", language)
        if _is_time_varying(result)
        else t("Total loudness N [sone]", language)
    )
    rows = [
        (loudness_label, d1(display_round(float(result.loudness)))),
        (
            t("Loudness level L<sub>N</sub> [phon]", language),
            d1(result.loudness_level),
        ),
    ]
    if result.n5 is not None:
        rows.append((t("N<sub>5</sub> [sone]", language), d1(result.n5)))
    if result.n10 is not None:
        rows.append((t("N<sub>10</sub> [sone]", language), d1(result.n10)))
    return rows


def _statement(result: ZwickerLoudness, language: str = "en") -> str:
    """The boxed single-number statement ``N = X sone (LN = Y phon)``."""
    loudness = format_number(
        display_round(float(result.loudness)), language, decimals=1
    )
    level = format_number(result.loudness_level, language, decimals=1)
    return (
        f"{_loudness_symbol(result)} = <b>{loudness} sone</b> &nbsp; "
        f"(L<sub>N</sub> = {level} phon)"
    )


def _verdict(
    result: ZwickerLoudness, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag: loudness passes at or below the requirement.

    The requirement is read as a maximum permitted loudness in sone (a lower
    loudness is better), so the sign rule is the opposite of the airborne one.
    The comparison is evaluated on the loudness rounded to the displayed one
    decimal, so the printed value can never contradict the verdict.
    """
    passed = display_round(float(result.loudness)) <= requirement + 1e-9
    text = t("{sym} = {value} sone, required &#8804; {req} sone", language).format(
        sym=_loudness_symbol(result),
        value=format_number(
            display_round(float(result.loudness)), language, decimals=1
        ),
        req=fmt_num(requirement, language),
    )
    return text, passed


def _basis_line(
    result: ZwickerLoudness,
    metadata: ReportMetadata | None,
    language: str,
) -> str:
    """The standard-basis line, with the clause 7 c)/d) mandatory items.

    States the method used (stationary sounds, clause 5, or time-varying
    sounds, clause 6) and the sound field the calculation assumed (free F or
    diffuse D, when the result carries it).
    """
    method = (
        t("method for time-varying sounds (clause 6)", language)
        if _is_time_varying(result)
        else t("method for stationary sounds (clause 5)", language)
    )
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t(
            "{standard} loudness. Rating per ISO 532-1:2017 (Zwicker method), {method}.",
            language,
        ).format(standard=html.escape(measurement_standard), method=method)
    else:
        basis = t(
            "Loudness rating per ISO 532-1:2017 (Zwicker method), {method}.", language
        ).format(method=method)
    if result.field in ("free", "diffuse"):
        field_text = (
            t("free (F)", language)
            if result.field == "free"
            else t("diffuse (D)", language)
        )
        basis += " " + t("Sound field: {field}.", language).format(field=field_text)
    return basis


def render_iso532_report(
    result: ZwickerLoudness,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 532-1 Zwicker loudness fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.psychoacoustics.loudness.zwicker.ZwickerLoudness`
        carrying the total loudness, loudness level and specific-loudness
        pattern (and the N5/N10 percentiles for a time-varying result).
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        prediction fiche (body + result + disclaimer, no test metadata). A
        supplied ``requirement`` is read as the maximum permitted loudness in
        sone.
    :param verbose: Accepted for a uniform ``.report()`` signature; the
        loudness fiche has a single body layout, so it has no effect.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # uniform signature; the loudness fiche has one body layout
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Loudness rating", language)
    basis = _basis_line(result, metadata, language)

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(basis, basis_style),
    ]

    if metadata is not None and not metadata.is_empty():
        header_pairs = _metadata_pairs(metadata, language)
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    left_cell = [
        fiche_paragraph(t("Loudness results", language), caption_style),
        metrics_table(_metric_rows(result, language)),
    ]
    # Non-band plot (specific loudness N' over Bark): self-scaling axis. The
    # time-varying fiche shortens this panel to make room for the mandatory
    # N(t) trace below while keeping the one-page layout.
    plot_drawing = render_figure_drawing(
        result.plot,
        116 * mm,
        y_top=None,
        figsize=(5.8, 4.4) if _is_time_varying(result) else None,
        language=language,
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    if _is_time_varying(result):
        # Clause 7 f): a time-varying loudness report states "the loudness
        # time function in sones". Full-width landscape N(t) trace with the
        # N5/N10 percentile levels marked.
        def _time_plot(
            ax: Axes | None = None, language: str = "en", **kwargs: Any
        ) -> Axes:
            from .._plot.psychoacoustics import plot_zwicker_loudness_time

            return plot_zwicker_loudness_time(
                result, ax=ax, language=language, **kwargs
            )

        flow.append(
            fiche_paragraph(
                t("Loudness versus time N(t) (clause 6.5)", language),
                caption_style,
            )
        )
        flow.append(
            render_figure_drawing(
                _time_plot,
                174 * mm,
                y_top=None,
                figsize=(9.2, 3.4),
                language=language,
            )
        )
        flow.append(Spacer(1, 8))

    flow.append(result_box(_statement(result, language), styles, accent))
    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
