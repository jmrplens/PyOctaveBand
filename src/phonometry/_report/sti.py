#  Copyright (c) 2026. Jose Manuel Requena Plens
"""IEC 60268-16 speech-transmission-index fiche (reportlab renderer).

Renders a :class:`~phonometry.speech.sti.STIResult` to a one-page PDF laid out
like a voice-alarm / public-address intelligibility verification report:

* a title and the standard-basis line (measurement standard + IEC 60268-16),
  stating the method the result was obtained by (the full STI indirect method
  from an impulse response, or the direct STIPA method on a recorded signal);
* an optional metadata header block (client, system, room ...), rendered only
  for the fields supplied on the :class:`ReportMetadata`;
* a two-panel body with the per-octave-band modulation transfer index table on
  the left and the result's own ``plot(ax=...)`` (the per-band MTI bars) on the
  right, so the chart is native to the library;
* a boxed single-number result ``STI = X`` with the Annex F qualification band
  alongside;
* an optional verdict row when a minimum-STI requirement is supplied (a higher
  STI is better, so it passes at or above the requirement);
* a footer identity/disclaimer block.

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the IEC 60268-16 specifics. reportlab, matplotlib and svglib are soft
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

if TYPE_CHECKING:
    from ..speech.sti import STIResult
    from .metadata import ReportMetadata

#: Nominal octave-band centre frequencies of the seven STI bands, in hertz
#: (IEC 60268-16 A.5.1: 125 Hz to 8 kHz).
_STI_BAND_CENTERS: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000, 8000)


def _is_stipa(result: STIResult) -> bool:
    """Whether the result came from the direct STIPA method (Annex B).

    STIPA carries two modulation-transfer values per band (the Table B.1
    frequency pair); the full indirect method carries the fourteen modulation
    frequencies of A.2.2.
    """
    mtf = np.asarray(result.mtf)
    return mtf.ndim == 2 and mtf.shape[1] == 2  # noqa: PLR2004


def _method_phrase(result: STIResult, language: str = "en") -> str:
    """The measurement-method phrase used in the basis line."""
    return (
        t("STIPA, direct method", language)
        if _is_stipa(result)
        else t("full STI, indirect method", language)
    )


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the header grid.

    Only fields that are set are returned. Speech intelligibility is a
    transmission-path metric, so the room-pair/climate fields of the insulation
    fiche do not apply; only the generic identity fields are used.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("System", language), metadata.specimen),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Test room", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
    ]
    return [
        (label, html.escape(str(value))) for label, value in specs if value is not None
    ]


def _band_table(result: STIResult, language: str = "en") -> Any:
    """Build the left-hand per-octave-band modulation-transfer-index table.

    Called only after :func:`render_sti_report` has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    from ._layout import fiche_paragraph

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    head_style = ParagraphStyle(
        "sti_thead",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7.2,
        textColor=colors.white,
        alignment=1,
        leading=8.5,
    )

    mti = np.asarray(result.mti, dtype=np.float64)
    header = [
        fiche_paragraph(t("f [Hz]", language), head_style),
        fiche_paragraph("MTI", head_style),
    ]
    rows: list[list[Any]] = [header]
    # Fixed seven-centre table against an 'mti' the frozen dataclass does not
    # validate: a directly built result with fewer bands ends the table early
    # rather than being rejected, and one with more is cut after the seventh.
    for fk, m in zip(_STI_BAND_CENTERS, mti, strict=False):
        rows.append([f"{fk}", format_number(float(m), language, decimals=2)])

    table = Table(rows, colWidths=[28 * mm, 28 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
            ]
        )
    )
    return table


def _statement(result: STIResult, language: str = "en") -> str:
    """The boxed single-number statement ``STI = X``."""
    value = format_number(display_round(float(result.sti), 2), language, decimals=2)
    return f"STI = <b>{value}</b>"


def _extended_terms(result: STIResult, language: str = "en") -> list[str]:
    """The Annex F qualification band and the method shown beside the box."""
    return [
        t("Qualification band (IEC 60268-16 Annex F): {value}", language).format(
            value=html.escape(result.rating)
        ),
        t("Method: {method}", language).format(method=_method_phrase(result, language)),
    ]


def _verdict(
    result: STIResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag: the STI passes at or above the requirement.

    Speech intelligibility is a "higher is better" quantity (the opposite of
    the emission fiches), so a result passes when it reaches or exceeds the
    minimum required STI. Both the STI and the requirement are compared *and*
    printed at the same two decimals, so the printed pair can never contradict
    its own verdict: an STI of 0.50 against a required 0.52 shows both numbers
    at full display precision, and a requirement of 0.5049 is decided on the
    0.50 it prints rather than on a digit the reader cannot see.
    """
    value = display_round(float(result.sti), 2)
    required = display_round(float(requirement), 2)
    passed = value >= required
    text = t("STI = {value}, required &#8805; {req}", language).format(
        value=format_number(value, language, decimals=2),
        req=format_number(required, language, decimals=2),
    )
    return text, passed


def _basis_line(
    result: STIResult, metadata: ReportMetadata | None, language: str
) -> str:
    """The standard-basis line, stating the measurement method."""
    method = _method_phrase(result, language)
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        return t(
            "{standard} speech transmission index. Rating per IEC 60268-16:2020 ({method}).",
            language,
        ).format(standard=html.escape(measurement_standard), method=method)
    return t(
        "Speech transmission index per IEC 60268-16:2020 ({method}).", language
    ).format(method=method)


def render_sti_report(
    result: STIResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an IEC 60268-16 speech-transmission-index fiche to a PDF at ``path``.

    :param result: A :class:`~phonometry.speech.sti.STIResult` carrying the
        overall STI, the per-octave-band modulation transfer index ``mti`` and
        the Annex F qualification letter ``rating``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + disclaimer, no test metadata). A supplied
        ``requirement`` is read as the minimum required STI.
    :param verbose: Accepted for a uniform ``.report()`` signature; the STI
        fiche has a single body layout, so it has no effect.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # uniform signature; the STI fiche has one body layout
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Speech transmission index", language)
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
        fiche_paragraph(
            t("Octave-band modulation transfer index MTI", language), caption_style
        ),
        _band_table(result, language),
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=1.0, expand_step=None, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    flow.append(
        result_box(
            _statement(result, language),
            styles,
            accent,
            _extended_terms(result, language),
        )
    )
    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
