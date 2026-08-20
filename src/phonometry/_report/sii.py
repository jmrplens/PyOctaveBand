#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ANSI S3.5-1997 speech-intelligibility-index fiche (reportlab renderer).

Renders a :class:`~phonometry.speech.sii.SIIResult` to a one-page PDF laid out
like a speech-audibility report:

* a title and the standard-basis line (measurement standard + ANSI S3.5-1997);
* an optional metadata header block (client, listener/condition ...), rendered
  only for the fields supplied on the :class:`ReportMetadata`;
* a two-panel body with the per-band table on the left (the equivalent speech
  spectrum ``Ei'``, the band-importance function ``Ii`` and the
  band-audibility function ``Ai``, over the bands of whichever of the
  standard's four band procedures the result was computed with) and the
  result's own
  ``plot(ax=...)`` (the audibility and its importance-weighted contribution) on
  the right, so the chart is native to the library;
* a boxed single-number result ``SII = X``;
* an optional verdict row when a minimum-SII requirement is supplied (a higher
  SII is better, so it passes at or above the requirement);
* a footer identity/disclaimer block.

With ``verbose=True`` the table adds the equivalent disturbance spectrum level
``Di`` column (clause 5.6).

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the ANSI S3.5 specifics. reportlab, matplotlib and svglib are soft
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
    _REPORTLAB_HINT,
    build_document,
    display_round,
    document_styles,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..speech.sii import SIIResult

#: The band procedure the fiche describes when the result carries no other.
_DEFAULT_METHOD = "one-third-octave"

#: Name of each ANSI S3.5-1997 band procedure as it reads in the standard-basis
#: line, keyed by the ``method=`` name of the result.
_METHOD_NAMES: dict[str, str] = {
    "critical-band": "critical-band",
    "equally-contributing": "equally-contributing critical-band",
    "one-third-octave": "one-third-octave-band",
    "octave": "octave-band",
}

#: Caption of the left-hand band table, keyed by the ``method=`` name.
_TABLE_CAPTIONS: dict[str, str] = {
    "critical-band": "Critical band audibility",
    "equally-contributing": "Equally-contributing critical band audibility",
    "one-third-octave": "One-third-octave band audibility",
    "octave": "Octave band audibility",
}


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the header grid.

    Only fields that are set are returned. The index is a listener-side
    audibility metric, so the room-pair/climate fields of the insulation fiche
    do not apply; only the generic identity fields are used.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Listening condition", language), metadata.specimen),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Test room", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
    ]
    return [
        (label, html.escape(str(value))) for label, value in specs if value is not None
    ]


def _band_table(result: SIIResult, verbose: bool, language: str = "en") -> Any:
    """Build the left-hand per-band audibility table of the procedure.

    The default table carries the equivalent speech spectrum level ``Ei'``, the
    band-importance function ``Ii`` and the band-audibility function
    ``Ai``; ``verbose`` adds the equivalent disturbance spectrum level ``Di``
    (clause 5.6). Called only after :func:`render_sii_report` has imported
    reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    from ._layout import fiche_paragraph as Paragraph

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    head_style = ParagraphStyle(
        "sii_thead",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7.0,
        textColor=colors.white,
        alignment=1,
        leading=8.2,
    )

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    importance = np.asarray(result.band_importance, dtype=np.float64)
    audibility = np.asarray(result.band_audibility, dtype=np.float64)
    speech = np.asarray(result.speech_spectrum, dtype=np.float64)
    disturbance = np.asarray(result.disturbance, dtype=np.float64)

    header = [
        Paragraph(t("f [Hz]", language), head_style),
        Paragraph("E&#8242;<sub>i</sub> [dB]", head_style),
        Paragraph("I<sub>i</sub>", head_style),
        Paragraph("A<sub>i</sub>", head_style),
    ]
    col_widths = [16 * mm, 20 * mm, 19 * mm, 15 * mm]
    if verbose:
        header.insert(3, Paragraph("D<sub>i</sub> [dB]", head_style))
        col_widths = [15 * mm, 18 * mm, 17 * mm, 18 * mm, 14 * mm]

    def d1(value: float) -> str:
        return format_number(value, language, decimals=1)

    rows: list[list[Any]] = [header]
    for j, fk in enumerate(freqs):
        row = [
            f"{round(fk)}",
            d1(float(speech[j])),
            format_number(float(importance[j]), language, decimals=4),
            format_number(float(audibility[j]), language, decimals=2),
        ]
        if verbose:
            row.insert(3, d1(float(disturbance[j])))
        rows.append(row)

    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("FONTSIZE", (0, 1), (-1, -1), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
            ]
        )
    )
    return table


def _statement(result: SIIResult, language: str = "en") -> str:
    """The boxed single-number statement ``SII = X``."""
    value = format_number(display_round(float(result.sii), 3), language, decimals=3)
    return f"SII = <b>{value}</b>"


def _verdict(
    result: SIIResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag: the SII passes at or above the requirement.

    Speech intelligibility is a "higher is better" quantity (the opposite of
    the emission fiches), so a result passes when it reaches or exceeds the
    minimum required SII. Both the SII and the requirement are compared *and*
    printed at the same three decimals, so the printed pair can never
    contradict its own verdict: a required 0.75 prints as 0.750 rather than as
    a one-decimal 0.8, and a requirement of 0.85104 is decided on the 0.851 it
    prints rather than on a digit the reader cannot see.
    """
    value = display_round(float(result.sii), 3)
    required = display_round(float(requirement), 3)
    passed = value >= required
    text = t("SII = {value}, required &#8805; {req}", language).format(
        value=format_number(value, language, decimals=3),
        req=format_number(required, language, decimals=3),
    )
    return text, passed


def render_sii_report(
    result: SIIResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ANSI S3.5-1997 speech-intelligibility-index fiche to ``path``.

    :param result: A :class:`~phonometry.speech.sii.SIIResult` carrying the
        overall SII, the per-band audibility ``Ai`` and importance ``Ii``, the
        equivalent speech spectrum and the disturbance/masking spectra.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + disclaimer, no test metadata). A supplied
        ``requirement`` is read as the minimum required SII.
    :param verbose: When ``True``, the left table adds the equivalent
        disturbance spectrum level ``Di`` column (clause 5.6).
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
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

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Speech intelligibility index", language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    method = t(_METHOD_NAMES.get(result.method, "one-third-octave-band"), language)
    if measurement_standard:
        basis = t(
            "{standard} speech intelligibility index. Rating per ANSI S3.5-1997 ({method} method).",
            language,
        ).format(standard=html.escape(measurement_standard), method=method)
    else:
        basis = t(
            "Speech intelligibility index per ANSI S3.5-1997 ({method} method).",
            language,
        ).format(method=method)

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

    left_cell = [
        Paragraph(
            t(
                _TABLE_CAPTIONS.get(result.method, _TABLE_CAPTIONS[_DEFAULT_METHOD]),
                language,
            ),
            caption_style,
        ),
        _band_table(result, verbose, language),
    ]
    # The verbose table adds a column, so the two-panel split is rebalanced to
    # keep the wider table beside a still-legible plot on the same page.
    if verbose:
        left_width_mm, plot_width_mm, plot_pt = 84.0, 90.0, 88.0
    else:
        left_width_mm, plot_width_mm, plot_pt = 72.0, 102.0, 100.0
    plot_drawing = render_figure_drawing(
        result.plot, plot_pt * mm, y_top=1.0, expand_step=None, language=language
    )
    flow.append(
        two_panel_body(
            left_cell,
            plot_drawing,
            left_width_mm=left_width_mm,
            plot_width_mm=plot_width_mm,
        )
    )
    flow.append(Spacer(1, 8))

    flow.append(result_box(_statement(result, language), styles, accent))
    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
