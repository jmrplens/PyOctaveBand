#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared scaffold for the ISO 905x material-test fiches (reportlab renderer).

The dynamic-stiffness (EN 29052-1 / ISO 9052-1, :mod:`.iso9052`) and the static
airflow-resistance (ISO 9053-1, :mod:`.iso9053`) fiches share the same one-page
shape: a title and standard-basis line, an optional metadata identity grid, a
two-panel body with a compact metrics table beside the result's own
self-scaling plot, a boxed single-number result with extended terms alongside,
and the shared footer. Both are material characterisations with no pass/fail
verdict. This module holds that common scaffold so each fiche only supplies its
own labels, metric rows and boxed statement; the quantity-independent pieces
live in :mod:`._layout`.

reportlab is a soft dependency imported lazily by :func:`render_material_fiche`
(it ships in the ``phonometry[report]`` extra); the import is guarded with an
actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._i18n import t
from ._layout import escaped_pairs, fmt_meta
from .metadata import ReportMetadata


@dataclass(frozen=True)
class MaterialFicheContent:
    """The fiche-specific content the shared material scaffold renders.

    :ivar title: The fiche title (also the PDF document title).
    :ivar basis_line: The standard-basis line under the title.
    :ivar caption: The left-panel caption above the metrics table.
    :ivar metadata_pairs: The escaped ``(label, value)`` identity-grid rows;
        an empty list renders no grid.
    :ivar metric_rows: The ``(label, value)`` rows of the left metrics table.
    :ivar statement: The boxed single-number result statement.
    :ivar extended: The extended terms shown beside the boxed statement.
    """

    title: str
    basis_line: str
    caption: str
    metadata_pairs: list[tuple[str, str]]
    metric_rows: list[tuple[str, str]]
    statement: str
    extended: list[str]


def standard_basis_line(
    with_standard: str,
    without_standard: str,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> str:
    """The standard-basis line, naming the measurement standard when supplied.

    ``with_standard`` carries a ``{standard}`` placeholder for the measurement
    standard read from the metadata; ``without_standard`` is used when none was
    supplied. Both are English keys translated through :func:`t`.
    """
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        return t(with_standard, language).format(
            standard=html.escape(measurement_standard)
        )
    return t(without_standard, language)


def material_metadata_pairs(
    metadata: ReportMetadata,
    language: str,
    middle: list[tuple[str, str | None]],
) -> list[tuple[str, str]]:
    """Build the common material identity grid with fiche-specific middle rows.

    The identity grid opens with the client, manufacturer and specimen
    description, then the ``middle`` rows specific to the fiche (a mass or a
    thickness), and closes with the test facility, date and climate. Unset
    fields are dropped and the free-text values XML-escaped by
    :func:`._layout.escaped_pairs`.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Description", language), metadata.specimen),
        *middle,
        (t("Test facility", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
        (
            t("Temperature [&#176;C]", language),
            fmt_meta(metadata.temperature, language)
            if metadata.temperature is not None
            else None,
        ),
        (
            t("Relative humidity [%]", language),
            fmt_meta(metadata.relative_humidity, language)
            if metadata.relative_humidity is not None
            else None,
        ),
    ]
    return escaped_pairs(specs)


def render_material_fiche(
    plot_fn: Callable[..., Any],
    path: str,
    content: MaterialFicheContent,
    metadata: ReportMetadata | None,
    *,
    language: str = "en",
) -> str:
    """Render a material-test fiche from its ``content`` to a PDF at ``path``.

    Lays out the shared skeleton (title, basis line, optional metadata grid, the
    metrics table beside the result's own self-scaling plot, the boxed result
    with its extended terms, and the footer) and writes it as a one-page A4 PDF.
    ``plot_fn`` is the result's own ``plot`` method, drawn as a vector drawing so
    the curve is native to the library.

    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed. The fiche always embeds the result's plot, so both are
        required (``pip install "phonometry[report,plot]"``).
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        from ._layout import _REPORTLAB_HINT

        raise ImportError(_REPORTLAB_HINT) from exc
    from ._layout import (
        _ACCENT_HEX,
        build_document,
        document_styles,
        footer_flow,
        grid_table,
        metrics_table,
        render_figure_drawing,
        result_box,
        two_panel_body,
    )

    accent = colors.HexColor(_ACCENT_HEX)
    styles, title_style, basis_style, caption_style = document_styles(accent)

    flow: list[Any] = [
        Paragraph(content.title, title_style),
        Paragraph(content.basis_line, basis_style),
    ]

    if content.metadata_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(content.metadata_pairs))
    flow.append(Spacer(1, 8))

    left_cell = [
        Paragraph(content.caption, caption_style),
        metrics_table(content.metric_rows),
    ]
    # Non-band plot (a self-scaling design curve): no fixed top on the y-axis.
    plot_drawing = render_figure_drawing(
        plot_fn, 116 * mm, y_top=None, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    flow.append(
        result_box(content.statement, styles, accent, extended=content.extended)
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, content.title)
