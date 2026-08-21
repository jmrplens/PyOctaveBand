#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Room-noise rating fiche (reportlab renderer, ANSI/ASA S12.2-2019).

Renders the two room-noise ratings of ANSI/ASA S12.2-2019 to a one-page PDF
laid out like a room-noise assessment report:

* a :class:`~phonometry.room.noise_criteria.NCResult` (Noise Criteria, clause
  5.2.2): the NC designation (NC-(SIL), or the tangency rating with its
  governing octave band; ``">NC-70"`` / ``"<NC-15"`` outside the Table 1
  family); and
* a :class:`~phonometry.room.noise_criteria.RCResult` (Room Criteria Mark II,
  Annex D): the RC rating and its spectral-quality tag (neutral / rumble /
  hiss).

Both fiches share the same skeleton:

* a title and the standard-basis line (measurement standard + the ANSI/ASA
  S12.2-2019 rating method);
* an optional metadata header block (client, room, description, room volume,
  floor area, instrumentation, climate ...), rendered only for the fields
  supplied on the :class:`ReportMetadata`;
* a two-panel body with the measured octave-band levels on the left and the
  measured spectrum against the NC/RC curve family on the right, drawn by the
  result's own ``plot(ax=...)`` so the curve is native to the library;
* a boxed single-number rating (``NC-nn`` with the governing band, or
  ``RC-nn(tag)`` with the mid-frequency average and the spectral quality);
* an optional verdict row when a target rating is supplied (a lower rating is
  better, so the room passes at or below the target);
* a footer identity/disclaimer block.

With ``verbose=True`` the left table gains the evaluation columns: the per-band
NC contour value that the tangency method reads (NC fiche), or the reference
RC Mark II curve and the measured deviation from it (RC fiche).

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the ANSI/ASA S12.2 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _LIGHT_HEX,
    _REPORTLAB_HINT,
    build_document,
    display_round,
    document_styles,
    escaped_pairs,
    fmt_meta,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)

if TYPE_CHECKING:
    from ..room.noise_criteria import NCResult, RCResult
    from .metadata import ReportMetadata

#: Shared title for both room-noise rating fiches.
_TITLE = "Room noise rating"


def _thead_style() -> Any:
    """The shared white-on-accent table-header paragraph style (7.4 pt)."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    return ParagraphStyle(
        "ansis122_thead",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7.4,
        textColor=colors.white,
        alignment=1,
        leading=8.6,
    )


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the room-noise header grid.

    Only fields that are set are returned, so empty rows never appear. A
    room-noise assessment characterises a single enclosure, so the single-room
    ``room_volume`` and floor ``area`` fields are used rather than the
    source/receiving transmission pair.
    """

    def num(value: float | None) -> str | None:
        return fmt_meta(value, language) if value is not None else None

    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Room", language), metadata.test_room),
        (t("Description", language), metadata.specimen),
        (t("Room volume V [m<super>3</super>]", language), num(metadata.room_volume)),
        (t("Floor area S [m<super>2</super>]", language), num(metadata.area)),
        (t("Instrumentation", language), metadata.instrumentation),
        (t("Temperature [&#176;C]", language), num(metadata.temperature)),
        (t("Relative humidity [%]", language), num(metadata.relative_humidity)),
        (t("Ambient pressure [kPa]", language), num(metadata.pressure)),
        (t("Date of test", language), metadata.test_date),
    ]
    return escaped_pairs(specs)


def _band_labels() -> list[str]:
    """Nominal octave-band labels (16 ... 8000 Hz), aligned with the levels."""
    from ..room.noise_criteria import OCTAVE_BANDS

    return [f"{f:g}" for f in np.asarray(OCTAVE_BANDS, dtype=np.float64)]


def _level_cell(level: float, language: str, decimals: int = 1) -> str:
    """One octave-band level cell, or an em dash when the band is absent."""
    if not math.isfinite(level):
        return "—"
    return format_number(float(level), language, decimals=decimals)


def _value_table(
    columns: list[tuple[str, list[str]]],
    col_widths: list[Any],
) -> Any:
    """Assemble the left-hand octave-band table with the accredited styling.

    ``columns`` is an ordered list of ``(header_markup, cell_strings)`` pairs;
    every cell list has one entry per octave band. Accent header row with a
    rule below it, zebra body striping, centred cells and the light box, the
    look shared with the other fiches. Called only after the renderer has
    imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    from ._layout import fiche_paragraph

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    head = _thead_style()

    n_data = len(columns[0][1])
    rows: list[list[Any]] = [[fiche_paragraph(header, head) for header, _ in columns]]
    rows.extend([cells[i] for _, cells in columns] for i in range(n_data))

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("FONTSIZE", (0, 1), (-1, -1), 7.6),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
            ]
        )
    )
    return table


def _nc_contour_column(levels: np.ndarray, language: str) -> list[str]:
    """Per-band NC contour value read by the tangency method (verbose NC).

    For each band the NC index whose Table 1 curve passes through the measured
    level is found by interpolation, exactly as
    :func:`~phonometry.room.noise_criteria.noise_criterion` does; the tangency
    rating is the maximum of these values. A band with no measured level shows
    an em dash; a band outside the Table 1 family shows ``"<15"`` / ``">70"``
    (no NC contour passes through it), never a fabricated number.
    """
    from ..room.noise_criteria import NC_CURVES, NC_INDICES

    cells: list[str] = []
    for k, level in enumerate(levels):
        if not math.isfinite(level):
            cells.append("—")
        elif level > NC_CURVES[-1, k]:
            cells.append(">70")
        elif level < NC_CURVES[0, k]:
            cells.append("<15")
        else:
            contour = float(np.interp(level, NC_CURVES[:, k], NC_INDICES))
            cells.append(format_number(contour, language, decimals=1))
    return cells


def _base_columns(levels: np.ndarray, language: str) -> list[tuple[str, list[str]]]:
    """The two columns every room-noise table opens with (frequency, level)."""
    return [
        (t("Frequency f [Hz]", language), _band_labels()),
        ("L<sub>p</sub> [dB]", [_level_cell(v, language) for v in levels]),
    ]


def _assemble_left(
    columns: list[tuple[str, list[str]]],
    col_widths: list[Any],
    widths: tuple[float, float],
    caption_style: Any,
    language: str,
) -> tuple[list[Any], list[Any]]:
    """Wrap the caption and value table into a two-panel left cell."""
    from ._layout import fiche_paragraph

    caption = t("Octave-band sound pressure levels", language)
    left_cell = [
        fiche_paragraph(caption, caption_style),
        _value_table(columns, col_widths),
    ]
    return left_cell, [widths[0], widths[1]]


def _nc_left_cell(
    result: NCResult, verbose: bool, caption_style: Any, language: str
) -> tuple[list[Any], list[Any]]:
    """The NC fiche left cell (caption + table) and its two-panel column widths."""
    from reportlab.lib.units import mm

    levels = np.asarray(result.levels, dtype=np.float64)
    columns = _base_columns(levels, language)
    if verbose:
        columns.append(("NC", _nc_contour_column(levels, language)))
        return _assemble_left(
            columns,
            [17 * mm, 18 * mm, 17 * mm],
            (56.0, 118.0),
            caption_style,
            language,
        )
    return _assemble_left(
        columns, [22 * mm, 22 * mm], (50.0, 124.0), caption_style, language
    )


def _rc_left_cell(
    result: RCResult, verbose: bool, caption_style: Any, language: str
) -> tuple[list[Any], list[Any]]:
    """The RC fiche left cell (caption + table) and its two-panel column widths."""
    from reportlab.lib.units import mm

    levels = np.asarray(result.levels, dtype=np.float64)
    reference = np.asarray(result.reference_curve, dtype=np.float64)
    # Guard a manually constructed result: RCResult is a plain frozen dataclass,
    # so a hand-built one can carry a reference curve of a different length than
    # its levels, and only the verbose columns pair the two band by band.
    if verbose and levels.shape != reference.shape:
        msg = (
            "render_rc_report(verbose=True) needs 'levels' and "
            "'reference_curve' of equal length."
        )
        raise ValueError(msg)
    columns = _base_columns(levels, language)
    if verbose:
        columns.append(
            (
                t("Ref. [dB]", language),
                [_level_cell(v, language, decimals=0) for v in reference],
            )
        )
        columns.append(
            (
                t("Dev. [dB]", language),
                [
                    _level_cell(lvl - ref, language)
                    for lvl, ref in zip(levels, reference, strict=True)
                ],
            )
        )
        return _assemble_left(
            columns,
            [13 * mm, 15 * mm, 15 * mm, 15 * mm],
            (62.0, 112.0),
            caption_style,
            language,
        )
    return _assemble_left(
        columns, [22 * mm, 22 * mm], (50.0, 124.0), caption_style, language
    )


def _rating_label(value: float) -> str:
    """Format a rating value as an integer when whole, else one decimal."""
    rounded = round(value, 1)
    return f"{int(rounded)}" if float(rounded).is_integer() else f"{rounded:g}"


def _nc_statement_terms(result: NCResult, language: str) -> tuple[str, list[str]]:
    """The boxed NC designation and its extended terms (clause 5.2.2).

    A SIL-designated spectrum boxes ``NC-nn`` with the SIL value; a tangency
    rating adds the governing band; a spectrum outside the Table 1 family
    boxes ``>NC-70`` / ``<NC-15`` and states that no NC rating is defined.
    """
    extended: list[str] = []
    if math.isfinite(result.sil):
        extended.append(
            t("Speech interference level SIL = {value} dB", language).format(
                value=format_number(result.sil, language, decimals=1)
            )
        )
    if result.out_of_range == "above":
        statement = "&gt;NC-<b>70</b>"
        extended.append(
            t("Governing band: {value} Hz", language).format(
                value=f"{float(result.governing_frequency):g}"
            )
        )
        extended.append(
            t(
                "Above the highest tabulated curve NC-70; no NC rating is "
                "defined (ANSI/ASA S12.2-2019, Table 1).",
                language,
            )
        )
        return statement, extended
    if result.out_of_range == "below":
        statement = "&lt;NC-<b>15</b>"
        extended.append(
            t(
                "Below the lowest tabulated curve NC-15; no NC rating is "
                "defined (ANSI/ASA S12.2-2019, Table 1).",
                language,
            )
        )
        return statement, extended
    value = _rating_label(float(result.rating))
    statement = f"NC-<b>{value}</b>"
    if result.method == "SIL":
        extended.append(
            t(
                "Designated NC-(SIL): no octave band exceeds the NC-(SIL) "
                "curve (ANSI/ASA S12.2-2019, 5.2.2).",
                language,
            )
        )
    else:
        extended.append(
            t("Governing band: {value} Hz", language).format(
                value=f"{float(result.governing_frequency):g}"
            )
        )
        extended.append(
            t(
                "Rated by the tangency method: the spectrum exceeds the "
                "NC-(SIL) curve (ANSI/ASA S12.2-2019, 5.2.2 and 5.2.3).",
                language,
            )
            if math.isfinite(result.sil)
            else t("Rated by the tangency method (ANSI/ASA S12.2-2019).", language)
        )
    return statement, extended


#: Spectral-quality descriptor for each RC Mark II tag (clause D.3).
_RC_QUALITY = {
    "N": "neutral",
    "R": "rumble",
    "H": "hiss",
    "RH": "rumble and hiss",
}


def _rc_statement_terms(result: RCResult, language: str) -> tuple[str, list[str]]:
    """The boxed ``RC-nn(tag)`` statement and its RC extended terms.

    A rating outside the tabulated RC-25 to RC-50 family (Table D.1) adds a
    note that the reference curve is an extrapolation; the combined ``RH``
    tag is labelled as the library's diagnostic extension of clause D.3.
    """
    statement = f"RC-<b>{result.rating}</b>({result.classification})"
    quality = _RC_QUALITY.get(result.classification, "neutral")
    extended = [
        t("Mid-frequency average L<sub>MF</sub> = {value} dB", language).format(
            value=format_number(result.lmf, language, decimals=1)
        ),
        t("Spectral quality: {value}", language).format(value=t(quality, language)),
    ]
    if result.classification == "RH":
        extended.append(
            t(
                "Combined rumble-and-hiss tag: an extension beyond the "
                "clause D.3.5 letters (N, R, H or RV).",
                language,
            )
        )
    if result.out_of_family:
        extended.append(
            t(
                "Outside the tabulated RC-25 to RC-50 family; the reference "
                "curve is extrapolated (ANSI/ASA S12.2-2019, Table D.1).",
                language,
            )
        )
    return statement, extended


def _verdict(
    symbol: str,
    rating: float,
    requirement: float,
    language: str,
    out_of_range: str | None = None,
) -> tuple[str, bool]:
    """Verdict text and PASS flag: a room-noise rating passes at or below target.

    A lower NC or RC rating is quieter, so the room complies when its rating is
    at or below the supplied target. The comparison uses the values rounded
    exactly as the fiche displays them (rating to one decimal, target to the
    nearest integer), so the printed numbers can never contradict the verdict.
    An NC spectrum outside the Table 1 family has no rating: above the family
    it fails any target, below it it passes (quieter than NC-15, the lowest
    target the family can express).
    """
    if out_of_range == "above":
        passed = False
        value = "&gt;70"
    elif out_of_range == "below":
        passed = True
        value = "&lt;15"
    else:
        passed = display_round(float(rating), 1) <= display_round(requirement, 0)
        value = _rating_label(float(rating))
    text = t("{symbol} {value}, required &#8804; {req}", language).format(
        symbol=symbol,
        value=value,
        req=format_number(requirement, language, decimals=0),
    )
    return text, passed


def _render_room_noise(
    result: NCResult | RCResult,
    path: str,
    *,
    basis_template: str,
    left_builder: Any,
    statement_builder: Any,
    symbol: str,
    metadata: ReportMetadata | None,
    verbose: bool,
    language: str,
) -> str:
    """Render a room-noise rating fiche, shared by the NC and RC renderers."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t(_TITLE, language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t(
            "{standard} octave-band measurement of room noise. " + basis_template,
            language,
        ).format(standard=html.escape(measurement_standard))
    else:
        basis = t(basis_template, language)

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

    left_cell, (left_width, plot_width) = left_builder(
        result, verbose, caption_style, language
    )
    plot_drawing = render_figure_drawing(
        result.plot, (plot_width - 2.0) * mm, y_top=None, language=language
    )
    flow.append(
        two_panel_body(
            left_cell,
            plot_drawing,
            left_width_mm=left_width,
            plot_width_mm=plot_width,
        )
    )
    flow.append(Spacer(1, 8))

    statement, extended = statement_builder(result, language)
    flow.append(result_box(statement, styles, accent, extended))

    if metadata is not None and metadata.requirement is not None:
        # Only NCResult carries the textual out_of_range flag; RCResult's
        # out_of_family annotation does not alter the numeric verdict.
        oor = getattr(result, "out_of_range", None)
        text, passed = _verdict(
            symbol,
            float(result.rating),
            metadata.requirement,
            language,
            out_of_range=oor if isinstance(oor, str) else None,
        )
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)


def render_nc_report(
    result: NCResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a Noise Criteria (NC) assessment fiche to a PDF at ``path``.

    :param result: An :class:`~phonometry.room.noise_criteria.NCResult`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        assessment fiche (body + result + disclaimer, no header). A supplied
        ``requirement`` is read as the maximum acceptable NC rating.
    :param verbose: When ``True``, the left table adds the per-band NC contour
        value read by the tangency method.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    return _render_room_noise(
        result,
        path,
        basis_template=(
            "Noise Criteria (NC) rating: NC-(SIL) designation, or the "
            "tangency method when the spectrum exceeds the NC-(SIL) curve "
            "(ANSI/ASA S12.2-2019, 5.2.2)."
        ),
        left_builder=_nc_left_cell,
        statement_builder=_nc_statement_terms,
        symbol="NC",
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


def render_rc_report(
    result: RCResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a Room Criteria Mark II (RC) assessment fiche to a PDF at ``path``.

    :param result: An :class:`~phonometry.room.noise_criteria.RCResult`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        assessment fiche (body + result + disclaimer, no header). A supplied
        ``requirement`` is read as the maximum acceptable RC rating.
    :param verbose: When ``True``, the left table adds the reference RC Mark II
        curve and the measured deviation from it.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    return _render_room_noise(
        result,
        path,
        basis_template=(
            "Room Criteria Mark II (RC) rating (ANSI/ASA S12.2-2019, Annex D)."
        ),
        left_builder=_rc_left_cell,
        statement_builder=_rc_statement_terms,
        symbol="RC",
        metadata=metadata,
        verbose=verbose,
        language=language,
    )
