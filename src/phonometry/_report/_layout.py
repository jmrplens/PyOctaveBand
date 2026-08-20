#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared reportlab building blocks for the accredited report fiches.

The per-standard renderers (:mod:`.iso717`, :mod:`.iso11654`, ...) all lay out
the same accredited-laboratory skeleton: a title and standard-basis line, an
optional metadata header grid, a two-panel body (value table beside the
result's own vector plot), a boxed single-number result, an optional verdict
row and a footer identity/disclaimer block. This module holds the pieces that
do not depend on the rated quantity, so each renderer only writes the parts
that are genuinely specific (the labels, the value table and the verdict rule).

reportlab, matplotlib and svglib are soft dependencies imported lazily by the
helpers that need them (reportlab and svglib ship in the ``phonometry[report]``
extra, matplotlib in ``phonometry[plot]``); each raises an actionable
:class:`ImportError`.
"""

from __future__ import annotations

import html
import math
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t

if TYPE_CHECKING:
    from collections.abc import Callable

    from .metadata import ReportMetadata

#: Installation hints for the three soft dependencies of the report.
_REPORTLAB_HINT = (
    "Rendering a report requires reportlab. Install it with: "
    "pip install phonometry[report]"
)
_MATPLOTLIB_HINT = (
    "Rendering the report figure requires matplotlib. Install it with: "
    "pip install phonometry[plot]"
)
_SVGLIB_HINT = (
    "Embedding the report figure as vector graphics requires svglib. Install "
    "it with: pip install phonometry[report]"
)

#: Accent and light shades used for the header row and the zebra striping,
#: matching the validated accredited fiche prototype.
_ACCENT_HEX = "#1f4e79"
_LIGHT_HEX = "#eef2f7"
_MUTED_HEX = "#555555"
_VERDICT_OK_HEX = "#1b6e2f"
_VERDICT_BAD_HEX = "#a11a1a"

#: Baseline shift of a fiche ``<sub>``/``<super>`` fragment, as a percentage
#: of the enclosing font size. reportlab shifts a script by half the font
#: size by default, yet spaces the lines of a paragraph by the style's fixed
#: ``leading`` alone (its ``autoLeading`` modes only look at per-fragment
#: font sizes, never at the script ``rise``), so a wrapped line ending in
#: deep subscripts collided with the ascenders of the next line. A
#: quarter-em shift is the typographic norm and bounds the worst-case line
#: extent (a deepest-subscript line directly above a tallest-superscript
#: line) at 1.24x the font size, which fits inside the fixed leading of
#: every fiche prose style.
_SCRIPT_RISE = "25%"

#: Bare script openers; the fiche markup never writes explicit rise/size
#: attributes, so only the attribute-less tags need rewriting.
_SCRIPT_TAG_RE = re.compile(r"<(sub|sup|super)>")

#: Row count of the octave-band variant of the sound-insulation band table
#: (one row per octave, the five octave centres 125-2000 Hz); a table this
#: size carries no one-third-octave triplets to rule.
_OCTAVE_TABLE_ROWS = 5


def fiche_paragraph(text: str, style: Any, **kwargs: Any) -> Any:
    """Build a reportlab ``Paragraph`` whose sub/superscripts cannot collide.

    Every fiche paragraph is created through this factory (the renderers
    import it as ``Paragraph``): it rewrites the bare ``<sub>``/``<sup>``/
    ``<super>`` openers to carry ``rise="25%"`` so the scripts stay within
    the fixed line leading of the fiche styles (see :data:`_SCRIPT_RISE`).
    The script font-size reduction keeps reportlab's default. Called only
    after the renderer has imported reportlab.
    """
    from reportlab.platypus import Paragraph as _Paragraph

    return _Paragraph(
        _SCRIPT_TAG_RE.sub(rf'<\1 rise="{_SCRIPT_RISE}">', text), style, **kwargs
    )


def escaped_pairs(
    specs: list[tuple[str, str | None]],
) -> list[tuple[str, str]]:
    """Drop unset pairs and XML-escape the user-supplied metadata values.

    Every fiche builds its header grid from an ordered list of
    ``(label, value)`` pairs where a ``None`` value means the field was not
    supplied. This keeps only the supplied pairs and escapes the value (free
    client text) so a stray ``&`` or ``<`` cannot break reportlab's
    ``Paragraph`` parser; the labels are fixed and carry intentional markup, so
    they are left untouched.
    """
    return [
        (label, html.escape(str(value))) for label, value in specs if value is not None
    ]


def measurement_basis_style() -> Any:
    """The muted footnote style for the measurement-basis strip under a fiche.

    The stacked-layout fiches (programme loudness, room acoustics ...) close
    with one or two small grey lines stating the measurement basis; the style
    is identical, so it is built once here. Called only after the renderer has
    imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    return ParagraphStyle(
        "fiche_measurement_basis",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=6,
    )


def fmt_num(value: float, language: str = "en") -> str:
    """Format a number with up to one decimal, dropping a trailing ``.0``.

    Delegates to :func:`phonometry._i18n.format_number`, so English keeps a
    period (matching accredited English lab reports such as the Salford
    reference) and ``language="es"`` uses a comma, as Spanish reports write
    numbers.
    """
    return format_number(value, language, decimals=1, trim=True)


def fmt_meta(value: float, language: str = "en") -> str:
    """Format a user-supplied metadata number without losing precision.

    The header grid prints values the client supplied (a sample area of
    ``1.23`` m^2, a pressure of ``101.32`` kPa); a laboratory fiche must not
    silently alter them, so this round-trips the value (``repr``-shortest via
    ``%g``) instead of forcing the one-decimal display rounding of
    :func:`fmt_num`, and only localises the decimal separator.
    """
    from ._i18n import decimal_comma

    return decimal_comma(f"{float(value):g}", language)


def display_round(value: float, decimals: int = 1) -> float:
    """Round half away from zero to ``decimals``, for display-driven verdicts.

    The fiches print quantities at a fixed display precision, so their
    pass/fail comparisons are evaluated on the value rounded exactly as it is
    displayed (halves away from zero, the reading of the deterministic-rounding
    convention used across the fiche layer); otherwise a result could print
    numbers that contradict its own verdict at the tolerance boundary. A
    result that rounds to zero returns an unsigned ``0.0`` (never ``-0.0``).
    """
    scale = 10.0 ** int(decimals)
    magnitude = math.floor(abs(float(value)) * scale + 0.5) / scale
    if magnitude <= 0.0:  # magnitude is non-negative; <= avoids float equality
        return 0.0
    return -magnitude if value < 0.0 else magnitude


def render_figure_drawing(
    plot_fn: Callable[..., Any],
    target_width: float,
    *,
    y_top: float | None,
    expand_step: float | None = None,
    figsize: tuple[float, float] | None = None,
    subplot_kw: dict[str, Any] | None = None,
    language: str = "en",
) -> Any:
    """Draw a result's plot as a scaled, vector reportlab ``Drawing``.

    ``plot_fn`` is the result's own ``plot`` method; it is called with a fresh
    ``ax`` so the figure is native to the library. The plot is saved to SVG
    with the text rasterised to vector paths (``svg.fonttype='path'``) and
    converted with svglib, so it stays crisp at any zoom/print resolution.

    :param target_width: Target width in points.
    :param y_top: Fixed top of the 0-based y-axis (accredited reports keep a
        fixed axis so band fiches are visually comparable); pass ``None`` to
        leave the plot's own axis untouched (for non-band plots such as a
        specific-loudness pattern or a level-vs-time trace that self-scale).
    :param expand_step: When given, the top is raised to the next multiple of
        this step if the data exceeds ``y_top`` (used for dB axes); ``None``
        keeps ``y_top`` exactly (used for a 0..1 coefficient axis). Ignored
        when ``y_top`` is ``None``.
    :param figsize: Matplotlib figure size ``(width, height)`` in inches;
        ``None`` keeps the default portrait ``(5.8, 6.4)``. A landscape size
        (e.g. a wide, short time plot) is passed for a stacked full-width
        figure.
    :param subplot_kw: Keyword arguments forwarded to ``Figure.subplots`` when
        the plot needs a non-rectangular axes (e.g. ``{"projection": "polar"}``
        for a polar reflected-level response); ``None`` uses the default
        rectangular axes. A ``y_top`` of ``None`` is expected alongside a polar
        axes, since the radial limits are the plot's own.
    :param language: Fiche language, forwarded to ``plot_fn`` so the embedded
        chart's axis labels and legends are localised, and to the tick-label
        decimal separator.
    """
    try:
        import matplotlib as mpl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_MATPLOTLIB_HINT) from exc
    try:
        from svglib.svglib import svg2rlg
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_SVGLIB_HINT) from exc

    svg_fd, svg_path = tempfile.mkstemp(suffix=".svg")
    os.close(svg_fd)
    try:
        fig = Figure(figsize=figsize if figsize is not None else (5.8, 6.4))
        FigureCanvasAgg(fig)
        ax = fig.subplots(subplot_kw=subplot_kw) if subplot_kw else fig.subplots()
        # Forward the fiche language so the embedded chart is localised too
        # (every result ``plot`` accepts ``language``); without it a Spanish
        # fiche would embed English axis labels and legends.
        plot_fn(ax=ax, language=language)
        # The fiche states the rating in the boxed result, so the plot's own
        # title would only duplicate it at a large size.
        ax.set_title("")
        # A fixed 0-based axis keeps band fiches comparable; y_top=None leaves
        # a self-scaling plot (specific loudness, level-vs-time) as its own.
        if y_top is not None:
            top = y_top
            if expand_step is not None:
                _, data_top = ax.get_ylim()
                top = max(top, float(np.ceil(data_top / expand_step) * expand_step))
            ax.set_ylim(0.0, top)
        # Move the legend above the axes, as the reference reports do.
        handles, labels = ax.get_legend_handles_labels()
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()
        if handles:
            # Cap the legend at three columns: a single row of four entries
            # (the impact fiches add the 500 Hz read marker) is wider than the
            # axes, and tight_layout would shrink the plot to fit it, leaving
            # the impact chart narrower than its airborne sibling.
            ax.legend(
                handles,
                labels,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.005),
                ncol=min(len(handles), 3),
                frameon=False,
                fontsize=8,
                handlelength=1.6,
                columnspacing=1.2,
            )
        # Localise the tick-label decimal separator for Spanish (a no-op for
        # English, so English figures are byte-for-byte unchanged). Imported
        # lazily like the matplotlib backend above.
        from .._i18n import localize_axes

        localize_axes(ax, language)
        fig.tight_layout()
        with mpl.rc_context({"svg.fonttype": "path"}):
            fig.savefig(svg_path, format="svg")
        drawing = svg2rlg(svg_path)
    finally:
        if os.path.exists(svg_path):
            os.remove(svg_path)
    if drawing is None or not drawing.width:
        msg = "Could not convert the report plot to vector graphics."
        raise ValueError(msg)
    scale = target_width / drawing.width
    drawing.scale(scale, scale)
    drawing.width = drawing.width * scale
    drawing.height = drawing.height * scale
    drawing.hAlign = "CENTER"
    return drawing


def grid_table(pairs: list[tuple[str, str]]) -> Any:
    """Lay an ordered list of (label, value) pairs into a two-column grid.

    Each grid row holds up to two label/value pairs (four table cells:
    ``label | value | label | value``); a trailing single pair is padded.
    Called only after the renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    styles = getSampleStyleSheet()
    accent = colors.HexColor(_ACCENT_HEX)
    label_style = ParagraphStyle(
        "fiche_meta_label",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor(_MUTED_HEX),
    )
    value_style = ParagraphStyle(
        "fiche_meta_value",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.black,
    )

    rows: list[list[Any]] = []
    for i in range(0, len(pairs), 2):
        left = pairs[i]
        right = pairs[i + 1] if i + 1 < len(pairs) else ("", "")
        rows.append(
            [
                fiche_paragraph(f"{left[0]}:", label_style) if left[0] else "",
                fiche_paragraph(left[1], value_style),
                fiche_paragraph(f"{right[0]}:", label_style) if right[0] else "",
                fiche_paragraph(right[1], value_style) if right[0] else "",
            ]
        )
    table = Table(rows, colWidths=[36 * mm, 51 * mm, 36 * mm, 51 * mm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
                ("LINEABOVE", (0, 0), (-1, 0), 0.5, accent),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, accent),
            ]
        )
    )
    return table


def band_table_header_style() -> Any:
    """The white-on-accent header-cell paragraph style of the band tables.

    Shared by the one-third-octave value tables of the sound-insulation
    fiches. Called only after the renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    return ParagraphStyle(
        "fiche_band_thead",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7.2,
        textColor=colors.white,
        alignment=1,
        leading=8.5,
    )


def band_table(
    rows: list[list[Any]],
    col_widths: list[Any],
    n_data: int,
    extra_styles: list[Any] | None = None,
) -> Any:
    """Assemble a one-third-octave band table with the accredited styling.

    Applies the shared look of the sound-insulation fiches: the accent header
    row, zebra body rows, a box rule and a thin rule after every
    third-octave triplet (grouping the table by octave exactly as the
    accredited reference reports do). ``rows`` holds the header row followed
    by ``n_data`` band rows (plus any trailing summary rows, styled through
    ``extra_styles``). Called only after the renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    thin = colors.HexColor("#c9d4e0")
    style_cmds: list[Any] = [
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, n_data), [colors.white, light]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("BOX", (0, 0), (-1, -1), 0.5, accent),
    ]
    # Octave-band tables (5 rows, one per octave) have no triplets to group.
    if n_data != _OCTAVE_TABLE_ROWS:
        for triplet_end in range(3, n_data, 3):
            style_cmds.append(
                ("LINEBELOW", (0, triplet_end), (-1, triplet_end), 0.4, thin)
            )
    if extra_styles:
        style_cmds += extra_styles
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle(style_cmds))
    return table


def document_styles(accent: Any) -> tuple[Any, Any, Any, Any]:
    """Return ``(stylesheet, title_style, basis_style, caption_style)``.

    The title (accent, 16 pt), the standard-basis line (muted, 9.5 pt) and the
    left-panel caption (accent, 8 pt) are identical across every fiche, so they
    are built once here. The base stylesheet is returned too, as the result box
    and verdict helpers take it. Called only after the renderer has imported
    reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "fiche_title",
        parent=styles["Title"],
        fontSize=16,
        textColor=accent,
        spaceAfter=1,
        alignment=0,
    )
    basis_style = ParagraphStyle(
        "fiche_basis",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=colors.HexColor(_MUTED_HEX),
        spaceAfter=2,
    )
    caption_style = ParagraphStyle(
        "fiche_caption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=accent,
        spaceAfter=3,
    )
    return styles, title_style, basis_style, caption_style


def two_panel_body(
    left_cell: Any,
    plot_drawing: Any,
    *,
    left_width_mm: float = 56.0,
    plot_width_mm: float = 118.0,
) -> Any:
    """Assemble the two-panel body: a left cell beside the result's plot.

    Every fiche puts a table or metrics list on the left and the result's own
    vector plot on the right; the cell alignment is shared and the column
    widths default to the compact ~56 mm left cell used across the fiches. A
    fiche that carries a multi-column detail table (the reverberation times and
    absorption areas, or the reflection factor and the real/imaginary parts of
    the normalised surface impedance) rebalances the split through
    ``left_width_mm`` / ``plot_width_mm`` (they sum to the 174 mm content
    width). Called only after the renderer has imported reportlab.
    """
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    body = Table(
        [[left_cell, plot_drawing]],
        colWidths=[left_width_mm * mm, plot_width_mm * mm],
    )
    body.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (-1, 0), (-1, 0), 0),
            ]
        )
    )
    return body


def metrics_table(rows: list[tuple[str, str]], *, col_widths: Any = None) -> Any:
    """A compact two-column ``metric | value`` table for a non-band fiche.

    The band fiches put a frequency table on the left; the single-metric fiches
    (loudness, EPNL, programme loudness ...) put a short list of scalar results
    there instead. Same accredited styling (accent header rule, zebra rows).
    Called only after the renderer has imported reportlab.

    :param rows: Ordered ``(label, value)`` pairs; labels may carry markup.
    :param col_widths: Optional explicit column widths; defaults to 34/22 mm.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    styles = getSampleStyleSheet()
    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    label_style = ParagraphStyle(
        "fiche_metric_label",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
    )
    value_style = ParagraphStyle(
        "fiche_metric_value",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        alignment=2,
    )
    data = [
        [fiche_paragraph(label, label_style), fiche_paragraph(value, value_style)]
        for label, value in rows
    ]
    table = Table(data, colWidths=col_widths or [34 * mm, 22 * mm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, light]),
                ("LINEABOVE", (0, 0), (-1, 0), 0.6, accent),
                ("LINEBELOW", (0, -1), (-1, -1), 0.6, accent),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def compliance_table(
    rows: list[tuple[str, str, str, str]],
    *,
    col_widths: Any = None,
    language: str = "en",
) -> Any:
    """A full-width 4-column ``metric | measured | target/limit | result`` table.

    The stacked-layout fiches (programme loudness, aircraft noise ...) check a
    result against several limits at once, so they need a wider compliance
    table than the two-column :func:`metrics_table`. Each row is a
    ``(metric_label, measured, limit, status)`` tuple where ``status`` is one
    of ``"pass"``, ``"fail"`` or ``"info"``: a pass renders a filled dot and
    ``PASS`` in green, a fail a dot and ``FAIL`` in red, and an informational
    row an en dash in muted grey (never green/red, as it carries no verdict).
    Same accredited styling as :func:`metrics_table` (accent header row, zebra
    body rows, thin box). Called only after the renderer has imported reportlab.

    :param rows: Ordered ``(metric, measured, limit, status)`` tuples; the
        metric/measured/limit strings may carry markup (they are formatted
        numbers, not user text).
    :param col_widths: Optional explicit column widths; defaults to
        ``[64, 36, 50, 24]`` mm (summing to the 174 mm content width).
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    styles = getSampleStyleSheet()
    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    header_style = ParagraphStyle(
        "fiche_compliance_header",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )
    cell_style = ParagraphStyle(
        "fiche_compliance_cell",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
    )
    result_style = ParagraphStyle(
        "fiche_compliance_result",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
    )

    def _result_markup(status: str) -> str:
        if status == "pass":
            return (
                f"<font color='{_VERDICT_OK_HEX}'>&#9679; {t('PASS', language)}</font>"
            )
        if status == "fail":
            return (
                f"<font color='{_VERDICT_BAD_HEX}'>&#9679; {t('FAIL', language)}</font>"
            )
        return f"<font color='{_MUTED_HEX}'>&#8211;</font>"

    data: list[list[Any]] = [
        [
            fiche_paragraph(t("Metric", language), header_style),
            fiche_paragraph(t("Measured", language), header_style),
            fiche_paragraph(t("Target / Limit", language), header_style),
            fiche_paragraph(t("Result", language), header_style),
        ]
    ]
    for metric, measured, limit, status in rows:
        data.append(
            [
                fiche_paragraph(metric, cell_style),
                fiche_paragraph(measured, cell_style),
                fiche_paragraph(limit, cell_style),
                fiche_paragraph(_result_markup(status), result_style),
            ]
        )
    table = Table(data, colWidths=col_widths or [64 * mm, 36 * mm, 50 * mm, 24 * mm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
                ("LINEABOVE", (0, 0), (-1, 0), 0.6, accent),
                ("LINEBELOW", (0, -1), (-1, -1), 0.6, accent),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def analysis_cell_styles(prefix: str) -> tuple[Any, Any, Any]:
    """Return ``(header, label, value)`` paragraph styles for a stacked table.

    Shared by the exposure fiches (occupational noise, human vibration), whose
    full-width analysis tables use the same cell typography: a white, centred
    accent-header cell, a left-aligned label cell and a centred value cell, all
    at 8.5 pt / 11 pt leading. ``prefix`` namespaces the reportlab style names
    so two fiches rendered in the same process do not collide. Called only after
    the renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        f"{prefix}_thead",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )
    label_style = ParagraphStyle(
        f"{prefix}_label",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
    )
    value_style = ParagraphStyle(
        f"{prefix}_value",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        alignment=1,
    )
    return header_style, label_style, value_style


def exceedance_markup(exceeded: bool | None, language: str = "en") -> str:
    """Inline status markup for an exceeded / not-exceeded assessment row.

    Shared by the exposure fiches (occupational noise, human vibration): a red
    filled dot and ``Exceeded`` when the value reaches or exceeds the threshold,
    a green dot and ``Not exceeded`` when it does not, and a muted en dash for
    an informational row (``exceeded=None``). Returns reportlab paragraph markup
    (a string), so it needs no reportlab import of its own.
    """
    if exceeded is None:
        return f"<font color='{_MUTED_HEX}'>&#8211;</font>"
    if exceeded:
        return (
            f"<font color='{_VERDICT_BAD_HEX}'>&#9679; {t('Exceeded', language)}</font>"
        )
    return (
        f"<font color='{_VERDICT_OK_HEX}'>&#9679; {t('Not exceeded', language)}</font>"
    )


def stacked_table(data: list[list[Any]], col_widths: list[Any]) -> Any:
    """A full-width table with the accredited styling (accent header, zebra rows).

    Shared by the exposure fiches: the accent header row, zebra body rows and a
    thin box rule of the ISO 9612 and human-vibration analysis/assessment
    tables. The returned table accepts further ``setStyle`` calls (for a totals
    row's fill/rule). Called only after the renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
                ("LINEABOVE", (0, 0), (-1, 0), 0.6, accent),
                ("LINEBELOW", (0, -1), (-1, -1), 0.6, accent),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
            ]
        )
    )
    return table


def result_box(
    statement: str,
    styles: Any,
    accent: Any,
    extended: list[str] | None = None,
) -> Any:
    """The boxed single-number result, with optional extended terms alongside.

    Called only after the renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    result_style = ParagraphStyle(
        "fiche_result",
        parent=styles["Normal"],
        fontSize=13,
        leading=17,
        textColor=accent,
    )
    box_cells: list[Any] = [fiche_paragraph(statement, result_style)]
    box_widths = [174 * mm]
    if extended:
        ext_style = ParagraphStyle(
            "fiche_ext",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=12,
        )
        box_cells = [
            fiche_paragraph(statement, result_style),
            fiche_paragraph("<br/>".join(extended), ext_style),
        ]
        box_widths = [104 * mm, 70 * mm]
    box = Table([box_cells], colWidths=box_widths)
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1.0, accent),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(_LIGHT_HEX)),
            ]
        )
    )
    return box


def verdict_flow(
    text: str, passed: bool, styles: Any, language: str = "en"
) -> list[Any]:
    """The PASS/FAIL verdict paragraph for a precomputed comparison.

    Each renderer computes ``text`` (the requirement comparison) and ``passed``
    with its own sign rule; this only renders the shared badge styling.
    Called only after the renderer has imported reportlab.
    """
    from reportlab.lib.styles import ParagraphStyle

    badge = t("PASS", language) if passed else t("FAIL", language)
    badge_hex = _VERDICT_OK_HEX if passed else _VERDICT_BAD_HEX
    verdict_style = ParagraphStyle(
        "fiche_verdict",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceBefore=4,
    )
    lead = t("Result vs requirement", language)
    return [
        fiche_paragraph(
            f"{lead}: {text} &#8594; <b><font color='{badge_hex}'>{badge}</font></b>",
            verdict_style,
        )
    ]


def footer_flow(
    metadata: ReportMetadata | None,
    language: str = "en",
    *,
    disclaimer: str | None = None,
) -> list[Any]:
    """Build the footer identity block plus the always-present disclaimer.

    ``disclaimer`` overrides the default specimen-scoped sentence ("The
    results relate only to the tested specimen."), which fits a laboratory
    measurement on a physical specimen. Prediction fiches and the
    population-statistics fiches pass their own English key (translated via
    :func:`t` before display) so the scope statement matches what the fiche
    actually reports. Called only after the renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Spacer

    styles = getSampleStyleSheet()
    muted = colors.HexColor(_MUTED_HEX)

    ident_style = ParagraphStyle(
        "fiche_footer",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
    )
    sign_style = ParagraphStyle(
        "fiche_sign",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=16,
    )
    disclaimer_style = ParagraphStyle(
        "fiche_disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=muted,
        leading=11,
    )

    signature = t("Signature", language)
    sign_blank = f"{signature}: ______________________________"

    flow: list[Any] = [Spacer(1, 8)]
    lines: list[str] = []
    if metadata is not None:
        if metadata.laboratory:
            lines.append(
                f"<b>{t('Laboratory', language)}:</b> "
                f"{html.escape(metadata.laboratory)}"
            )
        if metadata.report_id:
            lines.append(
                f"<b>{t('Report no.', language)}:</b> {html.escape(metadata.report_id)}"
            )
        if metadata.test_date:
            lines.append(
                f"<b>{t('Date', language)}:</b> {html.escape(metadata.test_date)}"
            )
        if metadata.notes:
            lines.append(
                f"<b>{t('Notes', language)}:</b> {html.escape(metadata.notes)}"
            )
    for line in lines:
        flow.append(fiche_paragraph(line, ident_style))

    operator = metadata.operator if metadata is not None else None
    if operator:
        flow.append(
            fiche_paragraph(
                f"{t('Operator', language)}: {html.escape(operator)} "
                f"&nbsp;&nbsp; {sign_blank}",
                sign_style,
            )
        )
    elif lines:
        flow.append(fiche_paragraph(sign_blank, sign_style))

    if disclaimer is None:
        disclaimer = "The results relate only to the tested specimen."
    flow.append(Spacer(1, 4))
    flow.append(
        fiche_paragraph(
            f"<font color='{_ACCENT_HEX}'>&#9632;</font> {t(disclaimer, language)}",
            disclaimer_style,
        )
    )
    flow.append(
        fiche_paragraph(
            f"<font size=7 color='#888888'>"
            f"{t('Generated by phonometry.', language)}</font>",
            disclaimer_style,
        )
    )
    return flow


def build_document(path: str, flow: list[Any], title: str) -> str:
    """Build the one-page A4 fiche ``flow`` into a reproducible PDF at ``path``.

    Called only after the renderer has imported reportlab.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate

    doc_kwargs = {
        "pagesize": A4,
        "leftMargin": 18 * mm,
        "rightMargin": 18 * mm,
        "topMargin": 15 * mm,
        "bottomMargin": 14 * mm,
        "title": title,
    }
    # invariant=1 drops the embedded timestamp for a reproducible PDF; the
    # guard tolerates reportlab builds that do not accept the keyword.
    try:
        doc = SimpleDocTemplate(path, invariant=1, **doc_kwargs)
    except TypeError:  # pragma: no cover - older reportlab
        doc = SimpleDocTemplate(path, **doc_kwargs)
    doc.build(flow)
    return str(path)
