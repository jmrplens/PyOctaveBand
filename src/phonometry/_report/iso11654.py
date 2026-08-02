#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 11654 sound-absorption rating fiche (reportlab renderer).

Renders a
:class:`~phonometry.materials.absorbers.rating.AbsorptionRatingResult` to a
one-page PDF laid out like an accredited absorption test report (a reverberation
room measurement per ISO 354 rated per ISO 11654):

* a title and the standard-basis line (measurement standard + ISO 11654 rating);
* an optional metadata header block (client, specimen, sample area, mounting,
  room conditions ...), rendered only for the fields supplied on the
  :class:`ReportMetadata`;
* a two-panel body with an absorption table on the left and the
  practical-versus-shifted-reference curve on the right, drawn by the result's
  own ``plot(ax=...)`` so the curve is native to the library. When the rating
  retained its one-third-octave input (from
  :func:`~phonometry.materials.weighted_absorption_from_third_octave`), the left
  table is the full ISO 354 one-third-octave ``alpha_s`` table with ``alpha_p``
  on the octave rows, as accredited certificates print it; otherwise it is the
  octave-band ``alpha_p`` table;
* a boxed single-number result ``alpha_w = X (shape) dB`` with the absorption
  class and applied shift alongside;
* an optional verdict row when a minimum ``alpha_w`` requirement is supplied;
* a footer identity/disclaimer block.

With ``verbose=True`` the table adds the ISO 11654 evaluation columns (practical
coefficient, shifted reference, unfavourable deviation).

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the ISO 11654 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _LIGHT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    build_document,
    document_styles,
    fmt_meta,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..materials.absorbers.rating import AbsorptionRatingResult

#: Threshold below which an unfavourable deviation is shown as an em dash.
_DEVIATION_EPS = 0.005


def _d2(value: float, language: str = "en") -> str:
    """Two decimals, locale-aware separator (period for English, comma for ES)."""
    return format_number(value, language, decimals=2)


def _thead_style() -> Any:
    """The shared accent table-header paragraph style (white, centred, 7.2 pt)."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    styles = getSampleStyleSheet()
    return ParagraphStyle(
        "iso11654_thead", parent=styles["Normal"], fontSize=7.2,
        textColor=colors.white, alignment=1, leading=8.5,
    )


def _base_table_style(accent: Any, light: Any, n_data: int) -> list[Any]:
    """The accredited-table command list shared by every left-panel table.

    Accent header row with a rule below it, zebra striping over the ``n_data``
    body rows, centred cells and the light box, matching the validated fiche.
    """
    from reportlab.lib import colors

    return [
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, n_data), [colors.white, light]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
        ("BOX", (0, 0), (-1, -1), 0.5, accent),
    ]


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the absorption header grid.

    Only fields that are set are returned, so empty rows never appear. The
    reverberation-room climate is reported with the single-value temperature,
    humidity and pressure fields; the insulation-only room-pair fields do not
    apply to an absorption measurement.
    """

    def num(value: float | None) -> str | None:
        # Round-trip formatting: the header grid reprints client-supplied
        # values and must not silently reduce them (10.8 m^2, 101.32 kPa).
        return fmt_meta(value, language) if value is not None else None

    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Description", language), metadata.specimen),
        (t("Sample area S [m<super>2</super>]", language), num(metadata.area)),
        (t("Mounted by", language), metadata.mounted_by),
        (t("Mounting", language), metadata.mounting),
        (t("Test room", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
        (t("Temperature [&#176;C]", language), num(metadata.temperature)),
        (t("Relative humidity [%]", language), num(metadata.relative_humidity)),
        (t("Ambient pressure [kPa]", language), num(metadata.pressure)),
    ]
    # Values are user-supplied free text; escape XML specials so a '&' or '<'
    # cannot break reportlab's Paragraph parser. Labels carry intentional markup.
    return [
        (label, html.escape(str(value)))
        for label, value in specs
        if value is not None
    ]


def _value_table(
    centers: np.ndarray,
    measured: np.ndarray,
    shifted: np.ndarray,
    deviations: np.ndarray,
    verbose: bool,
    language: str = "en",
) -> Any:
    """Build the left-hand octave-band absorption table (accredited or evaluation).

    Called only after :func:`render_iso11654_report` has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    from ._layout import fiche_paragraph as Paragraph

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    head_style = _thead_style()

    if verbose:
        header = [
            Paragraph(t("f [Hz]", language), head_style),
            Paragraph(t("Practical &#945;<sub>p</sub>", language), head_style),
            Paragraph(t("Shifted ref.", language), head_style),
            Paragraph(t("Unfav. dev.", language), head_style),
        ]
        col_widths = [15 * mm, 20 * mm, 17 * mm, 18 * mm]
    else:
        header = [
            Paragraph(t("Frequency f [Hz]", language), head_style),
            Paragraph("&#945;<sub>p</sub>", head_style),
        ]
        col_widths = [28 * mm, 28 * mm]

    rows: list[list[Any]] = [header]
    for fk, m, r_, d in zip(centers, measured, shifted, deviations):
        if verbose:
            rows.append(
                [
                    f"{round(fk)}",
                    _d2(m, language),
                    _d2(r_, language),
                    _d2(d, language) if d > _DEVIATION_EPS else "—",
                ]
            )
        else:
            rows.append([f"{round(fk)}", _d2(m, language)])

    style_cmds = _base_table_style(accent, light, len(centers))
    if verbose:
        rows.append(
            ["", "", t("sum", language), _d2(float(deviations.sum()), language)]
        )
        style_cmds += [
            ("LINEABOVE", (0, -1), (-1, -1), 0.6, accent),
            ("FONTNAME", (2, -1), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, -1), (-1, -1), 7.5),
        ]

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle(style_cmds))
    return table


def _third_octave_table(
    bands: np.ndarray,
    alpha_s: np.ndarray,
    measured: np.ndarray,
    language: str = "en",
) -> Any:
    """Build the combined one-third-octave ``f | alpha_s | alpha_p`` table.

    The accredited ISO 354 convention (Fellert, Mueller-BBM, Peutz): every
    one-third-octave band 200 Hz to 5000 Hz carries its measured ``alpha_s``,
    and the octave practical coefficient ``alpha_p`` is printed only on the
    matching octave-centre row (250, 500, 1000, 2000, 4000 Hz), the middle of
    each one-third-octave triple. Called only after :func:`render_iso11654_report`
    has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    from ._layout import fiche_paragraph as Paragraph

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    head_style = _thead_style()

    header = [
        Paragraph(t("Frequency f [Hz]", language), head_style),
        Paragraph("&#945;<sub>s</sub>", head_style),
        Paragraph("&#945;<sub>p</sub>", head_style),
    ]
    col_widths = [24 * mm, 16 * mm, 16 * mm]

    rows: list[list[Any]] = [header]
    for j, (fk, a_s) in enumerate(zip(bands, alpha_s)):
        # The middle of each triple is the octave centre carrying alpha_p.
        octave_cell = _d2(measured[j // 3], language) if j % 3 == 1 else ""
        rows.append([f"{round(fk)}", _d2(a_s, language), octave_cell])

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle(_base_table_style(accent, light, len(bands))))
    return table


def _statement(result: AbsorptionRatingResult, language: str = "en") -> str:
    """The boxed single-number statement ``alpha_w = X(shape)``.

    The shape indicator follows the value without a space, exactly as the
    ISO 11654 clause 5.3 example (``0,70(MH)``) and
    :attr:`~phonometry.materials.absorbers.rating.AbsorptionRatingResult.rating_label`
    write it.
    """
    value = format_number(result.alpha_w, language, decimals=2)
    if result.shape_indicator:
        value = f"{value}({result.shape_indicator})"
    return f"&#945;<sub>w</sub> = <b>{value}</b>"


def _extended_terms(
    result: AbsorptionRatingResult, language: str = "en"
) -> list[str]:
    """The absorption class, shape indicator and applied shift shown by the box."""
    terms = [
        t("Absorption class: {value}", language).format(
            value=html.escape(result.absorption_class)
        )
    ]
    if result.shape_indicator:
        terms.append(
            t("Shape indicator: {value}", language).format(
                value=result.shape_indicator
            )
        )
    terms.append(
        t("Applied shift: {value}", language).format(
            value=format_number(result.shift, language, decimals=2)
        )
    )
    return terms


def _verdict(
    result: AbsorptionRatingResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag: absorption passes at or above the requirement."""
    passed = float(result.alpha_w) >= requirement
    text = t("&#945;<sub>w</sub> = {value}, required &#8805; {req}", language).format(
        value=format_number(result.alpha_w, language, decimals=2),
        req=format_number(requirement, language, decimals=2),
    )
    return text, passed


def render_iso11654_report(
    result: AbsorptionRatingResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 11654 absorption-rating fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.materials.absorbers.rating.AbsorptionRatingResult`
        carrying the octave-band ``band_centers``, ``measured`` practical
        coefficients and the fitted ``shifted_reference`` curve.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        prediction fiche (body + result + disclaimer, no test metadata). A
        supplied ``requirement`` is read as the minimum ``alpha_w``.
    :param verbose: When ``True``, the left table adds the ISO 11654 evaluation
        columns (practical coefficient, shifted reference, unfavourable
        deviation); otherwise the accredited two-column ``f | alpha_p`` table.
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

    centers = np.asarray(result.band_centers, dtype=np.float64)
    measured = np.asarray(result.measured, dtype=np.float64)
    shifted = np.asarray(result.shifted_reference, dtype=np.float64)
    if not centers.shape == measured.shape == shifted.shape:
        raise ValueError(
            "render_iso11654_report() needs 'band_centers', 'measured' and "
            "'shifted_reference' of equal length."
        )
    # Unfavourable deviation: measured practical coefficient below the curve.
    deviations = np.maximum(shifted - measured, 0.0)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Sound absorption rating", language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t("{standard} laboratory measurement of sound absorption. Rating per ISO 11654:1997.", language).format(
            standard=html.escape(measurement_standard)
        )
    else:
        basis = t("Sound absorption rating per ISO 11654:1997.", language)

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

    # The accredited default (Fellert et al.) shows the full one-third-octave
    # alpha_s table with alpha_p on the octave rows when the input alpha_s was
    # retained; verbose keeps the octave evaluation columns, and without alpha_s
    # the plain octave alpha_p table is used.
    alpha_s = result.third_octave_alpha_s
    bands = result.third_octave_bands
    if not verbose and alpha_s is not None and bands is not None:
        value_table = _third_octave_table(
            np.asarray(bands, dtype=np.float64),
            np.asarray(alpha_s, dtype=np.float64),
            measured,
            language,
        )
        caption = t("One-third-octave &#945;<sub>s</sub>, octave &#945;<sub>p</sub>", language)
    else:
        value_table = _value_table(
            centers, measured, shifted, deviations, verbose, language
        )
        caption = t("Octave-band &#945;<sub>p</sub>", language)
    left_cell = [
        Paragraph(caption, caption_style),
        value_table,
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=1.0, expand_step=None, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    flow.append(
        result_box(
            _statement(result, language), styles, accent,
            _extended_terms(result, language),
        )
    )
    if result.shape_indicator:
        # ISO 11654 clause 5.3 NOTE: where a shape indicator applies, the
        # weighted coefficient should be used together with the complete
        # absorption-coefficient curve, which this fiche shows above.
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

        note_style = ParagraphStyle(
            "fiche_iso11654_note", parent=getSampleStyleSheet()["Normal"],
            fontSize=7.5, leading=10, textColor=colors.HexColor(_MUTED_HEX),
            spaceBefore=4,
        )
        flow.append(
            Paragraph(
                t("A shape indicator applies: ISO 11654 (5.3 NOTE) recommends using &#945;<sub>w</sub> in combination with the complete sound absorption coefficient curve, shown above.", language),
                note_style,
            )
        )
    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
