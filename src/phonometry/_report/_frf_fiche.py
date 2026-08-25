#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared reportlab body for the structural-vibration FRF fiches.

The two structural-vibration frequency-response fiches, the ISO 7626 mechanical
mobility (:mod:`.iso7626`) and the ISO 10846 dynamic transfer stiffness
(:mod:`.iso10846`), lay out the same skeleton: a title and standard-basis line,
an optional metadata header grid, a two-panel body with a compact table of the
FRF's characteristic points beside the result's own spectrum plot, a boxed
representative single value and a footer identity/disclaimer block. Both are
continuous frequency-response functions over a fine frequency axis rather than
octave bands, so neither carries a per-band table (that would misrepresent the
quantity); the honest presentation is the spectrum plot plus a small table of
characteristic points. Both are characterisations, so neither carries a
pass/fail verdict.

This module holds those shared pieces so each renderer only writes the parts
that are genuinely specific (the title, the basis line, the characteristic-point
rows and the boxed value). reportlab, matplotlib and svglib are soft
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
    _REPORTLAB_HINT,
    build_document,
    document_styles,
    fmt_meta,
    footer_flow,
    grid_table,
    metrics_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
)

if TYPE_CHECKING:
    import numpy as np

    from ..vibration.structural.mechanical_mobility import MobilityResult
    from ..vibration.structural.transfer_stiffness import TransferStiffnessResult
    from .metadata import ReportMetadata


def frequency_range(frequencies: np.ndarray, language: str = "en") -> str:
    """The ``fmin - fmax`` frequency-range string, in hertz (compact form)."""
    lo = format_number(float(frequencies.min()), language, decimals=1, trim=True)
    hi = format_number(float(frequencies.max()), language, decimals=1, trim=True)
    return t("{lo} to {hi}", language).format(lo=lo, hi=hi)


def frf_metadata_pairs(
    metadata: ReportMetadata,
    extra: list[tuple[str, str | None]],
    language: str = "en",
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the FRF fiche header grid.

    The generic identity fields an FRF characterisation report carries (client,
    manufacturer, the tested element/specimen, the test facility, the
    instrumentation, the date and the test temperature) plus the fiche-specific
    ``extra`` pairs (already formatted, inserted after the specimen). Only
    supplied fields are returned; the free-text values are XML-escaped so a stray
    ``&`` or ``<`` cannot break reportlab's ``Paragraph`` parser.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Description", language), metadata.specimen),
        *extra,
        (t("Test facility", language), metadata.test_room),
        (t("Instrumentation", language), metadata.instrumentation),
        (t("Date of test", language), metadata.test_date),
        (
            t("Temperature [&#176;C]", language),
            fmt_meta(metadata.temperature, language)
            if metadata.temperature is not None
            else None,
        ),
    ]
    return [
        (label, html.escape(str(value))) for label, value in specs if value is not None
    ]


def render_frf_fiche(
    result: MobilityResult | TransferStiffnessResult,
    path: str,
    *,
    title: str,
    basis: str,
    caption: str,
    header_pairs: list[tuple[str, str]],
    metric_rows: list[tuple[str, str]],
    statement: str,
    extended: list[str],
    metadata: ReportMetadata | None,
    language: str = "en",
) -> str:
    """Assemble a structural-vibration FRF fiche into a one-page PDF at ``path``.

    :param result: The FRF result, whose ``plot(ax=...)`` draws the spectrum.
    :param title: The fiche title.
    :param basis: The standard-basis line.
    :param caption: The left-panel caption above the characteristic-point table.
    :param header_pairs: The metadata header grid pairs (see
        :func:`frf_metadata_pairs`); empty for a bare fiche.
    :param metric_rows: The ``(label, value)`` characteristic-point rows.
    :param statement: The boxed representative-value statement (markup).
    :param extended: The extended terms shown beside the boxed value.
    :param metadata: Optional :class:`ReportMetadata` for the footer identity.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(basis, basis_style),
    ]

    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    left_cell = [
        fiche_paragraph(caption, caption_style),
        metrics_table(metric_rows),
    ]
    # The FRF is a continuous spectrum, not a band quantity: the plot self-scales
    # (y_top=None) rather than sharing a fixed band axis.
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=None, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    flow.append(result_box(statement, styles, accent, extended=extended))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
