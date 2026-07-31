#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Duct-borne noise path calculation sheet (reportlab renderer).

Renders a :class:`~phonometry.noise_control.duct_path.DuctPathResult` to a
one-page duct-path calculation sheet in the layout the published procedures
use:

* AHRI Standard 885, *Procedure for Estimating Occupied Space Sound Levels in
  the Application of Air Terminals and Air Outlets*, Table 8 -- one row per
  physical element, each carrying a short worksheet code and a plain-language
  name, with the octave bands across the columns, attenuations entered as
  negative numbers and summed down the column, regenerated (self) noise given
  its own positive row and combined logarithmically, and a single cumulative
  received-level row at the foot rather than a subtotal after every row;
* Long, *Architectural Acoustics* 2nd ed., Table 14.9 -- the same sheet for a
  supply and a return path, closing with the criterion curve printed under the
  received spectrum;
* AHRI 885 Figure 10 / ANSI/ASA S12.2-2019 -- the received spectrum plotted
  against the criterion curve, which is how a duct-path sheet makes its
  pass/fail visible.

The sheet is: title and method-basis line, optional metadata header, the
full-width element table, then a two-panel foot with the boxed room-criterion
rating and the verdict beside the cascade chart, and the shared footer.

reportlab, matplotlib and svglib are soft dependencies imported lazily
(reportlab and svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import t
from ._layout import (
    _ACCENT_HEX,
    _LIGHT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    build_document,
    display_round,
    document_styles,
    fmt_num,
    footer_flow,
    grid_table,
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from ._noise_control_fiche import PREDICTION_DISCLAIMER
from ._sound_power_fiche import metadata_pairs
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..noise_control.duct_path import DuctPathResult

#: Rows whose values are a level rather than a correction, shaded to separate
#: them from the attenuation rows exactly as the published sheets do.
_LEVEL_KINDS = (
    "source", "sum", "self_noise", "level", "contribution", "received",
    "criterion",
)

#: The longest element description printed in full before it is elided.
_LABEL_CHARS = 46


def _band_header(frequencies: np.ndarray, language: str) -> list[str]:
    """Column headings: the row-code and element columns, then the band centres."""
    return [t("Ref.", language), t("Element", language)] + [
        f"{f:g}" if f < 1000 else f"{f / 1000:g}k" for f in frequencies
    ]


def _cell(value: float, language: str) -> str:
    """One band cell: the value at whole-decibel display rounding."""
    if not np.isfinite(value):
        return "-"
    return fmt_num(display_round(value, 0), language)


def _regenerates(row: dict[str, Any]) -> bool:
    """Whether a self-noise row carries anything above the 0 dB floor."""
    return bool(np.any(np.asarray(row["values"]) > 0.0))


def _visible_rows(
    rows: list[dict[str, Any]], verbose: bool
) -> list[dict[str, Any]]:
    """Select the rows a sheet prints at the requested level of detail.

    Every row the cascade produces is meaningful, but a fiche is one page, so
    the redundant ones go first:

    * a self-noise row sitting entirely on the 0 dB floor says only that the
      element regenerates nothing, and is always dropped;
    * a *Sum* row is the level before that element's regenerated noise is added
      back, so when the element regenerates nothing it repeats the *Combined*
      row below it and is dropped even in verbose mode;
    * the running *Sum* and *Combined* rows are dropped entirely unless
      ``verbose`` is set, leaving one attenuation row per element in the
      compact sheet.
    """
    return [
        row
        for position, row in enumerate(rows)
        if _prints(row, rows[position + 1 :], verbose)
    ]


def _prints(
    row: dict[str, Any], rest: list[dict[str, Any]], verbose: bool
) -> bool:
    """Whether one sheet row survives the selection of :func:`_visible_rows`."""
    kind = row["kind"]
    if kind == "self_noise":
        return _regenerates(row)
    if kind not in ("sum", "level"):
        return True
    if not verbose:
        return False
    if kind == "level":
        return True
    following = rest[0] if rest else None
    return not (
        following is not None
        and following["kind"] == "self_noise"
        and not _regenerates(following)
    )


#: A4 minus the fiche margins of :func:`~phonometry._report._layout.build_document`
#: (18 mm left/right, 15 mm top, 14 mm bottom) and minus reportlab's own 6 pt
#: frame padding on each side, in points: the usable area of the single page.
_FRAME_WIDTH = 481.2
_FRAME_HEIGHT = 747.7


def _flow_height(flow: list[Any]) -> float:
    """Height the flowables ask for, including their inter-flowable spacing.

    reportlab's frame lays a flowable out at the height ``wrap`` reports and
    separates it from the previous one by the larger of that one's
    ``spaceAfter`` and its own ``spaceBefore``, which is what is summed here.
    """
    total = 0.0
    previous_gap = 0.0
    for item in flow:
        total += max(previous_gap, float(item.getSpaceBefore()))
        total += float(item.wrap(_FRAME_WIDTH, _FRAME_HEIGHT)[1])
        previous_gap = float(item.getSpaceAfter())
    return total + previous_gap


def _elide(
    rows: list[dict[str, Any]], budget: int, language: str
) -> list[dict[str, Any]]:
    """Trim a path too long to print to ``budget`` rows, eliding the middle.

    The sheet keeps what a reader needs to place the path: the source at the
    head and the room effect, the received spectrum and the criterion curve at
    the foot, with as many leading elements as fit and a single row saying how
    many were left out. A path long enough to trigger this is one whose sheet
    belongs in :meth:`~phonometry.noise_control.duct_path.DuctPathResult.table`
    rather than on a one-page fiche.
    """
    tail_kinds = ("room_effect", "received", "criterion")
    head = rows[:1]
    tail = [row for row in rows if row["kind"] in tail_kinds]
    middle = rows[len(head) : len(rows) - len(tail)]
    room = max(budget - len(head) - len(tail) - 1, 0)
    omitted = len(middle) - room
    if omitted <= 0:
        return rows
    return [
        *head,
        *middle[:room],
        {
            "code": "",
            "label": t("{count} further elements omitted", language).format(
                count=omitted
            ),
            "kind": "elided",
            "values": np.full(len(rows[0]["values"]), np.nan),
        },
        *tail,
    ]


def _element_label(row: dict[str, Any], language: str) -> str:
    """The printed element description, translated for the fixed row names."""
    label = str(row["label"])
    if row["kind"] in ("sum", "self_noise", "level", "room_effect", "received"):
        label = t(label, language)
    if len(label) > _LABEL_CHARS:
        label = label[: _LABEL_CHARS - 1] + "…"
    return label


def _sheet_table(
    result: DuctPathResult,
    verbose: bool,
    language: str,
    rows: list[dict[str, Any]] | None = None,
) -> tuple[Any, int]:
    """Build the full-width element table and report how many rows it has.

    Called only after the renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    frequencies = np.asarray(result.frequencies, dtype=np.float64)
    if rows is None:
        rows = _visible_rows(result.table(), verbose)
    data: list[list[Any]] = [_band_header(frequencies, language)]
    for row in rows:
        data.append(
            [row["code"], _element_label(row, language)]
            + [_cell(v, language) for v in np.asarray(row["values"])]
        )

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    band_width = (174.0 - 12.0 - 62.0) / max(frequencies.size, 1)
    table = Table(
        data,
        colWidths=[12.0 * mm, 62.0 * mm] + [band_width * mm] * frequencies.size,
        repeatRows=1,
    )
    style: list[Any] = [
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 6.6 if verbose else 7.0),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.8 if verbose else 1.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8 if verbose else 1.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("BOX", (0, 0), (-1, -1), 0.5, accent),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
    ]
    for i, row in enumerate(rows, start=1):
        if row["kind"] in _LEVEL_KINDS:
            style.append(("BACKGROUND", (0, i), (-1, i), light))
        if row["kind"] in ("received", "criterion"):
            style.append(("FONTNAME", (0, i), (-1, i), "Helvetica-Bold"))
        if row["kind"] == "received":
            style.append(("LINEABOVE", (0, i), (-1, i), 0.6, accent))
    table.setStyle(TableStyle(style))
    return table, len(rows)


def _rating_designation(rating: Any) -> str:
    """The room-criterion designation, at the display rounding of a fiche.

    The rating objects print their designation with ``:g``, which is right for
    an integer tangency rating but writes an interpolated one out to every digit
    it has (``NC-22.6402``). A sheet quotes one decimal, so an NC designation
    inside the Table 1 family is rebuilt here at that rounding; every other case
    (an out-of-range NC spectrum, an RC designation with its spectral tag) keeps
    the object's own label.
    """
    if getattr(rating, "out_of_range", "") is None and hasattr(rating, "sil"):
        band = getattr(rating, "governing_frequency", float("nan"))
        designation = f"NC-{fmt_num(display_round(float(rating.rating), 1))}"
        if np.isfinite(band):
            return f"{designation} ({band:g} Hz)"
        return designation
    return str(getattr(rating, "label", ""))


def _rating_statement(result: DuctPathResult, language: str) -> tuple[str, list[str]]:
    """The boxed rating headline and the terms printed alongside it."""
    rating = result.rating
    label = _rating_designation(rating)
    statement = t("Room criterion <b>{label}</b>", language).format(label=label)
    extended = [
        t("Received level from {source}", language).format(source=result.source_label)
    ]
    if result.target is not None:
        extended.append(
            t("Design criterion {criterion} {target}", language).format(
                criterion=result.criterion, target=f"{result.target:g}"
            )
        )
    return statement, extended


def _verdict(result: DuctPathResult, language: str) -> tuple[str, bool] | None:
    """The band-by-band verdict against the design criterion curve, if declared."""
    passed = result.meets_target
    if passed is None:
        return None
    excess = np.asarray(result.exceedance, dtype=np.float64)
    worst = int(np.argmax(excess))
    margin = display_round(float(excess[worst]))
    band = f"{result.frequencies[worst]:g}"
    if passed:
        text = t(
            "no band exceeds {criterion} {target}; smallest margin "
            "{margin} dB at {band} Hz", language,
        ).format(
            criterion=result.criterion, target=f"{result.target:g}",
            margin=fmt_num(-margin, language), band=band,
        )
    else:
        text = t(
            "{criterion} {target} exceeded by {margin} dB at {band} Hz",
            language,
        ).format(
            criterion=result.criterion, target=f"{result.target:g}",
            margin=fmt_num(margin, language), band=band,
        )
    return text, passed


def _basis_strips(result: DuctPathResult, language: str) -> list[str]:
    """The method-basis strips explaining the sheet's arithmetic."""
    strips = [
        t(
            "Attenuations are printed as negative level changes and summed "
            "band by band; a regenerated (self) noise row is a sound power "
            "level and is combined with the running level on an energy basis. "
            "The room effect converts the sound power reaching the terminal "
            "device into a sound pressure level in the room.",
            language,
        )
    ]
    if result.target is not None:
        strips.append(
            t(
                "The verdict compares the received octave-band levels with "
                "the {criterion} {target} curve of ANSI/ASA S12.2-2019 band by "
                "band; the boxed designation is the standard's own rating of "
                "the received spectrum.",
                language,
            ).format(criterion=result.criterion, target=f"{result.target:g}")
        )
    return strips


def render_duct_path_report(
    result: DuctPathResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a duct-borne noise path calculation sheet to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.noise_control.duct_path.DuctPathResult` carrying
        the element cascade, the received spectrum and the design criterion.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        sheet (table, result and basis, no header). A supplied ``requirement``
        overrides the result's own design criterion for the verdict.
    :param verbose: When ``True`` the sheet also prints the running level after
        every element and the self-noise rows sitting on the 0 dB floor.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
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

    if metadata is not None and metadata.requirement is not None:
        from dataclasses import replace

        result = replace(result, target=float(metadata.requirement))

    accent = colors.HexColor(_ACCENT_HEX)
    styles, title_style, basis_style, caption_style = document_styles(accent)

    flow: list[Any] = [
        Paragraph(t("Duct-borne noise path calculation", language), title_style),
        Paragraph(
            t(
                "Octave-band sound level in an occupied space from an air "
                "distribution system, estimated element by element along the "
                "duct path (AHRI Standard 885; Long, Architectural Acoustics "
                "2nd ed., Chapters 13-14). Room criterion per "
                "ANSI/ASA S12.2-2019. This is a prediction from design data, "
                "not a measurement.",
                language,
            ),
            basis_style,
        ),
    ]
    if metadata is not None and not metadata.is_empty():
        header_pairs = metadata_pairs(metadata, language)
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 7))

    rows = _visible_rows(result.table(), verbose)
    table, _count = _sheet_table(result, verbose, language, rows)
    flow.append(
        Paragraph(
            t("Octave-band path calculation, dB", language), caption_style
        )
    )
    table_index = len(flow)
    flow.append(table)
    flow.append(Spacer(1, 7))

    statement, extended = _rating_statement(result, language)
    left_cell: list[Any] = [result_box(statement, styles, accent, extended)]
    verdict = _verdict(result, language)
    if verdict is not None:
        left_cell.extend(verdict_flow(verdict[0], verdict[1], styles, language))
    # The verbose sheet prints the running Sum and Combined rows, so the chart
    # gives up the height the extra rows need and the fiche stays on one page.
    plot_width = 88.0 if verbose else 96.0
    plot_drawing = render_figure_drawing(
        result.plot, plot_width * mm, y_top=None,
        figsize=(6.2, 3.1 if verbose else 4.0), language=language,
    )
    flow.append(
        two_panel_body(
            left_cell, plot_drawing, left_width_mm=84.0 if verbose else 76.0,
            plot_width_mm=90.0 if verbose else 98.0,
        )
    )
    flow.append(
        Paragraph(
            t(
                "Predicted (estimated) result computed from the declared duct "
                "geometry, element data and room condition; it is not a "
                "measurement. A real installation also depends on the balancing "
                "of the system, on breakout along the run and on flanking paths "
                "the sheet does not carry.",
                language,
            ),
            ParagraphStyle(
                "duct_path_prediction", parent=styles["Normal"], fontSize=8.0,
                textColor=colors.HexColor(_MUTED_HEX), spaceBefore=3,
            ),
        )
    )

    basis_style_strip = measurement_basis_style()
    for strip in _basis_strips(result, language):
        flow.append(Paragraph(strip, basis_style_strip))
    flow.extend(
        footer_flow(metadata, language, disclaimer=PREDICTION_DISCLAIMER)
    )

    _fit_to_one_page(flow, table_index, result, rows, verbose, language)
    return build_document(
        path, flow, t("Duct-borne noise path calculation", language)
    )


def _fit_to_one_page(
    flow: list[Any],
    table_index: int,
    result: DuctPathResult,
    rows: list[dict[str, Any]],
    verbose: bool,
    language: str,
) -> None:
    """Shrink the sheet's table in place until the flow fits one page.

    A fiche is one page. This measures what the assembled flow asks for and, as
    long as it overruns the frame, elides the middle element rows and rebuilds
    the table; the complete sheet always stays available through
    :meth:`~phonometry.noise_control.duct_path.DuctPathResult.table`.
    """
    while len(rows) > 4:
        used = _flow_height(flow)
        if used <= _FRAME_HEIGHT:
            return
        row_height = flow[table_index].wrap(_FRAME_WIDTH, _FRAME_HEIGHT)[1] / (
            len(rows) + 1
        )
        drop = max(int((used - _FRAME_HEIGHT) / max(row_height, 1.0)) + 1, 1)
        trimmed = _elide(rows, len(rows) - drop, language)
        if len(trimmed) >= len(rows):
            return
        rows = trimmed
        flow[table_index], _count = _sheet_table(result, verbose, language, rows)
