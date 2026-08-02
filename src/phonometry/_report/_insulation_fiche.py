#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared renderer for the sound-insulation test-report fiches.

The field (ISO 16283, :mod:`.iso16283`) and laboratory (ISO 10140,
:mod:`.iso10140`) sound-insulation reports are the same one-page sheet: a title
and standard-basis line, an optional metadata header, a two-panel body with the
per-band table on the left and the measured-versus-shifted-ISO 717-reference
curve on the right, a boxed single-number rating, a method statement, an
optional requirement verdict and a footer. Only the fixed text (titles, basis
lines, symbols, the method statement) and the left-hand table content differ
between the two. This module holds the common skeleton so both renderers call
it, parameterised by their differences; the quantity-independent flowable
helpers still live in :mod:`._layout`.

reportlab, matplotlib and svglib are soft dependencies imported lazily
(reportlab and svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    band_table,
    band_table_header_style,
    build_document,
    document_styles,
    fmt_num,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from .iso717 import _Y_TOP_AIRBORNE, _Y_TOP_IMPACT, _metadata_pairs
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..building.measurement.insulation import (
        ImpactRatingResult,
        WeightedRatingResult,
    )

#: A per-band table column: header markup, values and decimal places.
Column = tuple[str, np.ndarray, int]

#: Builder of the left-hand table content for one report: given the reported
#: quantity's value header (already translated), its per-band curve, the
#: verbose flag and the language, it returns the ordered columns following the
#: frequency column, the panel caption and either explicit column widths (for
#: the multi-column verbose table) or ``None`` (the two-column ``f | value``
#: form, whose widths are fixed here).
ColumnsBuilder = Callable[
    [str, np.ndarray, bool, str], tuple[Sequence[Column], str, Any]
]


def single_number_statement(
    rating: WeightedRatingResult | ImpactRatingResult, rating_symbol: str
) -> str:
    """The boxed single-number statement with its adaptation terms.

    Airborne ratings print ``Xw (C; Ctr)``; impact ratings ``Xw (CI)``. The
    adaptation terms carry an explicit sign (the style of the accredited
    reports the fiche mirrors).
    """
    if rating.quantity == "impact":
        impact = cast("ImpactRatingResult", rating)
        return (
            f"{rating_symbol} (C<sub>I</sub>) = "
            f"<b>{impact.rating} ({impact.ci:+d}) dB</b>"
        )
    airborne = cast("WeightedRatingResult", rating)
    return (
        f"{rating_symbol} (C; C<sub>tr</sub>) = "
        f"<b>{airborne.rating} ({airborne.c:+d}; {airborne.ctr:+d}) dB</b>"
    )


def requirement_verdict(
    rating: WeightedRatingResult | ImpactRatingResult,
    rating_symbol: str,
    requirement: float,
    language: str,
) -> tuple[str, bool]:
    """Return the verdict text and PASS flag for a supplied requirement.

    Airborne ratings pass at or above the requirement; impact ratings pass at
    or below it (a lower impact level is better).
    """
    value = float(rating.rating)
    req_text = fmt_num(requirement, language)
    if rating.quantity == "impact":
        passed = value <= requirement
        text = t("{sym} = {rating} dB, required &#8804; {req} dB", language)
    else:
        passed = value >= requirement
        text = t("{sym} = {rating} dB, required &#8805; {req} dB", language)
    return (
        text.format(sym=rating_symbol, rating=rating.rating, req=req_text),
        passed,
    )


def band_value_table(
    centers: np.ndarray,
    columns: Sequence[Column],
    language: str,
    col_widths: Any = None,
) -> Any:
    """Build the left-hand per-band table.

    A single column following the frequency column is the recommended-form
    ``f | value`` table (fixed 28 mm columns, a ``Frequency f [Hz]`` heading);
    more columns are the verbose table (a compact ``f [Hz]`` heading and the
    ``col_widths`` supplied by the caller). Called only after the renderer has
    imported reportlab.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    head_style = band_table_header_style()

    if len(columns) == 1:
        header = [
            Paragraph(t("Frequency f [Hz]", language), head_style),
            Paragraph(columns[0][0], head_style),
        ]
        widths = [28 * mm, 28 * mm]
    else:
        header = [Paragraph(t("f [Hz]", language), head_style)] + [
            Paragraph(markup, head_style) for markup, _, _ in columns
        ]
        widths = col_widths

    rows: list[list[Any]] = [header]
    for k, fk in enumerate(centers):
        row: list[Any] = [f"{round(fk)}"]
        for _, values, decimals in columns:
            row.append(format_number(float(values[k]), language, decimals=decimals))
        rows.append(row)

    return band_table(rows, widths, len(centers))


def iso717_columns_builder(
    rating: WeightedRatingResult | ImpactRatingResult,
    is_impact: bool,
    symbol: str,
    *,
    band_set: str = "One-third-octave",
) -> ColumnsBuilder:
    """Build the left-table callback shared by the ISO 717-rated fiches.

    The default table is the two-column ``f | value`` form (caption
    ``{band_set} {symbol} [dB]``, either ``One-third-octave`` or
    ``Octave-band``); ``verbose`` shows the ISO 717 evaluation per band (the
    reported quantity, the shifted Table 3 reference and the unfavourable
    deviation), read from the rating so no extra per-band data is needed on the
    result. The unfavourable deviation is the reference above the measurement
    for a level difference or reduction index (more is better) and the
    measurement above the reference for an impact level (less is better),
    exactly as the ISO 717 rating forms it.
    """

    def build(
        value_header: str, curve: np.ndarray, verbose: bool, language: str
    ) -> tuple[Sequence[Column], str, Any]:
        from reportlab.lib.units import mm

        if not verbose:
            caption = t(f"{band_set} {{vh}} [dB]", language).format(vh=symbol)
            return [(value_header, curve, 1)], caption, None

        measured = np.asarray(rating.measured, dtype=np.float64)
        shifted = np.asarray(rating.shifted_reference, dtype=np.float64)
        if is_impact:
            deviation = np.maximum(measured - shifted, 0.0)
        else:
            deviation = np.maximum(shifted - measured, 0.0)
        columns: list[Column] = [
            (value_header, curve, 1),
            (t("Shifted ref. [dB]", language), shifted, 1),
            (t("Unfav. dev. [dB]", language), deviation, 1),
        ]
        col_widths = [10 * mm, 16 * mm, 15 * mm, 15 * mm]
        caption = t("ISO 717 evaluation per band", language)
        return columns, caption, col_widths

    return build


def render_insulation_fiche(
    result: Any,
    rating: WeightedRatingResult | ImpactRatingResult,
    path: str,
    *,
    spec: dict[str, str],
    is_impact: bool,
    curve_attr: str,
    build_columns: ColumnsBuilder,
    metadata: ReportMetadata | None,
    verbose: bool,
    language: str,
) -> str:
    """Render a sound-insulation test-report fiche to a PDF at ``path``.

    :param result: The per-band result carrying the reported curve
        (``curve_attr``) and, for the verbose table, the report-specific
        per-band content the ``build_columns`` callback reads.
    :param rating: The ISO 717 rating of the reported quantity; its ``plot``
        draws the fiche curve and it must carry the per-band ``band_centers``,
        ``measured`` and ``shifted_reference`` arrays.
    :param path: Destination path of the PDF file.
    :param spec: Fixed English labels for the reported quantity, with keys
        ``title``, ``basis``, ``symbol``, ``rating_symbol``, ``ylabel`` and
        ``statement`` (the natural-language ones are translated here).
    :param is_impact: ``True`` for an impact quantity (ISO 717-2), ``False``
        for an airborne one (ISO 717-1); it selects the plot's y-axis top and
        is checked against ``rating.quantity``.
    :param curve_attr: Attribute name of the reported per-band curve on
        ``result`` (e.g. ``"dnt"``, ``"r"``, ``"l_n"``).
    :param build_columns: Callback that builds the left-hand table content.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table uses the report-specific
        verbose columns instead of the two-column ``f | value`` form.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``rating.quantity`` does not match ``is_impact`` or
        the rating is missing its per-band ``band_centers`` / ``measured`` /
        ``shifted_reference`` arrays (a manually constructed rating).
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    # Guard a manually constructed rating: its quantity must match the reported
    # result, and it must carry the per-band arrays the table and plot consume
    # (np.asarray(None) would otherwise yield a 0-d array and crash the table).
    if rating.quantity != ("impact" if is_impact else "airborne"):
        raise ValueError(
            "The ISO 717 rating quantity does not match the reported result."
        )
    if (
        rating.band_centers is None
        or rating.measured is None
        or rating.shifted_reference is None
    ):
        raise ValueError(
            "The report needs the ISO 717 per-band rating data ('band_centers', "
            "'measured' and 'shifted_reference') on the rating."
        )

    rating_symbol = spec["rating_symbol"]
    curve = np.asarray(getattr(result, curve_attr), dtype=np.float64)
    centers = np.asarray(rating.band_centers, dtype=np.float64)
    measured = np.asarray(rating.measured, dtype=np.float64)
    shifted = np.asarray(rating.shifted_reference, dtype=np.float64)
    if not (curve.shape == centers.shape == measured.shape == shifted.shape):
        raise ValueError(
            "The report needs matching per-band lengths: the rating's "
            "'band_centers', 'measured' and 'shifted_reference' and the "
            "result's per-band curve must all have the same length."
        )

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title_text = t(spec["title"], language)
    flow: list[Any] = [
        Paragraph(title_text, title_style),
        Paragraph(t(spec["basis"], language), basis_style),
    ]

    # Metadata header block (only the supplied fields; the same grid the two
    # sound-insulation families share, both describing rooms and a specimen).
    if metadata is not None and not metadata.is_empty():
        identity, conditions = _metadata_pairs(metadata, language)
        header_pairs = identity + conditions
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    # Left panel: the report-specific table content; right panel: the rating's
    # own measured-versus-shifted-reference curve.
    value_header = t("{vh} [dB]", language).format(vh=spec["symbol"])
    columns, caption, col_widths = build_columns(
        value_header, curve, verbose, language
    )
    value_table = band_value_table(centers, columns, language, col_widths)
    left_cell = [Paragraph(caption, caption_style), value_table]

    def _plot(ax: Any = None, language: str = language) -> Any:
        axes = rating.plot(ax=ax, language=language)
        axes.set_ylabel(t(spec["ylabel"], language))
        return axes

    y_top = _Y_TOP_IMPACT if is_impact else _Y_TOP_AIRBORNE
    plot_drawing = render_figure_drawing(
        _plot, 116 * mm, y_top=y_top, expand_step=10.0, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    # Boxed single-number rating, the method statement, optional verdict row
    # and the footer.
    flow.append(
        result_box(single_number_statement(rating, rating_symbol), styles, accent)
    )
    statement_style = ParagraphStyle(
        "insulation_statement", parent=styles["Normal"], fontSize=8.5,
        textColor=colors.HexColor(_MUTED_HEX), spaceBefore=4,
    )
    flow.append(Paragraph(t(spec["statement"], language), statement_style))
    if metadata is not None and metadata.requirement is not None:
        text, passed = requirement_verdict(
            rating, rating_symbol, metadata.requirement, language
        )
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title_text)
