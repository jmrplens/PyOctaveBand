#  Copyright (c) 2026. Jose M. Requena-Plens
"""Reverberation-time fiches (reportlab renderer).

Two related one-page PDF fiches for the octave-band reverberation time of a
room, sharing the accredited-laboratory skeleton of :mod:`._layout`:

* :func:`render_reverberation_models_report` renders a
  :class:`~phonometry.room.reverberation_prediction.ReverberationModelResult`
  as a design-stage **prediction**: the reverberation time estimated by five
  classical statistical-acoustics models (Sabine, Eyring/Norris-Eyring,
  Millington-Sette, Fitzroy and Arau-Puchades) per octave band, laid out as a
  per-band table with one column per model beside the result's own model
  comparison plot. It is explicitly labelled a prediction, not a measurement:
  the five models bracket the reverberation time likely to occur, and
  Arau-Puchades (the recommended model for a non-uniform absorption
  distribution) drives the boxed mid-frequency descriptor.

* :func:`render_enclosed_space_report` renders a
  :class:`~phonometry.room.enclosed_space_absorption.ReverberationResult` as a
  room characterisation: the total equivalent sound absorption area ``A`` and
  the reverberation time ``T`` of an enclosed space per octave band by the
  EN 12354-6:2003 Clause 4 model, laid out as a per-band ``A``/``T`` table
  beside the result's own reverberation-time plot, with the room volume and the
  object fraction in the header.

Neither fiche carries a PASS/FAIL verdict: a room reverberation time is a
target range rather than a simply-higher-or-lower-is-better quantity, so a
supplied target reverberation time is printed as a reference line without an
invented pass direction.

reportlab, matplotlib and svglib are soft dependencies imported lazily by the
shared helpers that need them (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each raises
an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _LIGHT_HEX,
    _REPORTLAB_HINT,
    build_document,
    document_styles,
    escaped_pairs,
    fmt_meta,
    footer_flow,
    grid_table,
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    two_panel_body,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..room.enclosed_space_absorption import ReverberationResult
    from ..room.reverberation_prediction import ReverberationModelResult

#: Octave centres whose mean is the mid-frequency reverberation-time descriptor
#: quoted for rooms (the 500 Hz and 1000 Hz octave bands).
_MID_BANDS = (500.0, 1000.0)

#: Tolerance (relative) for matching a band centre to a nominal frequency.
_BAND_MATCH_REL = 0.06

#: The five prediction models, in the order the table columns print them.
_MODEL_NAMES = ("Sabine", "Eyring", "Millington-Sette", "Fitzroy", "Arau-Puchades")

#: The recommended model for a non-uniform absorption distribution; its
#: mid-frequency reverberation time drives the boxed prediction descriptor.
_PRIMARY_MODEL = "Arau-Puchades"


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _fmt_t(value: float, language: str = "en") -> str:
    """Format a reverberation time to two decimals, or an em dash if not finite."""
    v = float(value)
    if not math.isfinite(v):
        return "—"
    return format_number(v, language, decimals=2)


def _band_value(freqs: np.ndarray, values: np.ndarray, target: float) -> float:
    """Return the value at the band closest to ``target`` Hz, or NaN if none near."""
    if freqs.size == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(freqs - target)))
    if abs(float(freqs[idx]) - target) > _BAND_MATCH_REL * target:
        return float("nan")
    return float(values[idx])


def _mid_or_first(freqs: np.ndarray, values: np.ndarray) -> tuple[float, bool]:
    """Return ``(value, is_mid)`` for the mid-frequency descriptor.

    When both the 500 Hz and 1000 Hz octaves are present and finite, the
    descriptor is their mean (``is_mid`` True); otherwise it is the first
    finite value with ``is_mid`` False, so no false "500-1000 Hz" claim is made
    for a band set that does not span both mid octaves.
    """
    low = _band_value(freqs, values, _MID_BANDS[0])
    high = _band_value(freqs, values, _MID_BANDS[1])
    if math.isfinite(low) and math.isfinite(high):
        return 0.5 * (low + high), True
    finite = values[np.isfinite(values)]
    return (float(finite[0]) if finite.size else float("nan")), False


def _octave_table(
    header_cells: list[Any], rows: list[list[Any]], col_widths: list[Any]
) -> Any:
    """Assemble an octave-band table with the accredited accent/zebra styling.

    Octave-band tables have one row per octave, so (unlike the one-third-octave
    band tables) there is no triplet grouping rule. Called only after the
    renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    data = [header_cells, *rows]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
                ("TOPPADDING", (0, 0), (-1, -1), 2.6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
            ]
        )
    )
    return table


def _band_header_style() -> Any:
    """White-on-accent header-cell paragraph style for the octave tables."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    return ParagraphStyle(
        "reverb_thead", parent=getSampleStyleSheet()["Normal"],
        fontSize=7.2, textColor=colors.white, alignment=1, leading=8.5,
    )


def _target_line(requirement: float, styles: Any, language: str) -> list[Any]:
    """A muted reference line for a supplied target reverberation time.

    A room reverberation time is a target range, not a strictly
    higher/lower-is-better quantity, so the target is reported without an
    invented PASS/FAIL verdict. Called only after the renderer imported
    reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle

    from ._layout import fiche_paragraph as Paragraph

    style = ParagraphStyle(
        "reverb_target", parent=styles["Normal"], fontSize=9.5, leading=13,
        spaceBefore=4, textColor=colors.HexColor("#555555"),
    )
    text = t("Target reverberation time T = {req} s", language).format(
        req=format_number(requirement, language, decimals=2)
    )
    return [Paragraph(f"<font color='{_ACCENT_HEX}'>&#9632;</font> {text}", style)]


def _header_grid(
    volume: float,
    extra_pairs: Sequence[tuple[str, str | None]],
    metadata: ReportMetadata | None,
    language: str,
) -> list[Any]:
    """Build the metadata header grid (volume plus room-identity fields).

    Returns the flowables (a spacer and the grid table) or an empty list when no
    field is supplied. The room volume is authoritative from the result itself;
    the descriptive fields come from the :class:`ReportMetadata` when given.
    """
    from reportlab.platypus import Spacer

    client = metadata.client if metadata is not None else None
    test_room = metadata.test_room if metadata is not None else None
    specimen = metadata.specimen if metadata is not None else None
    test_date = metadata.test_date if metadata is not None else None
    temperature = metadata.temperature if metadata is not None else None
    humidity = metadata.relative_humidity if metadata is not None else None
    pressure = metadata.pressure if metadata is not None else None

    specs: list[tuple[str, str | None]] = [
        (t("Client", language), client),
        (t("Room", language), test_room),
        (t("Description", language), specimen),
        (t("Room volume V [m<super>3</super>]", language),
         fmt_meta(volume, language)),
        *extra_pairs,
        (t("Temperature [&#176;C]", language),
         fmt_meta(temperature, language) if temperature is not None else None),
        (t("Relative humidity [%]", language),
         fmt_meta(humidity, language) if humidity is not None else None),
        (t("Ambient pressure [kPa]", language),
         fmt_meta(pressure, language) if pressure is not None else None),
        (t("Date of prediction", language), test_date),
    ]
    pairs = escaped_pairs(specs)
    if not pairs:
        return []
    return [Spacer(1, 3), grid_table(pairs)]


# ---------------------------------------------------------------------------
# Prediction fiche (five statistical-acoustics models).
# ---------------------------------------------------------------------------


def _models_table(result: ReverberationModelResult, language: str) -> Any:
    """Build the per-band table with one reverberation-time column per model."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    head = _band_header_style()
    header = [Paragraph(t("f [Hz]", language), head)]
    header += [Paragraph(name, head) for name in _MODEL_NAMES]

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    curves = [np.asarray(result.models[name], dtype=np.float64) for name in _MODEL_NAMES]
    rows: list[list[Any]] = []
    for i, fk in enumerate(freqs):
        rows.append(
            [f"{round(float(fk))}", *[_fmt_t(c[i], language) for c in curves]]
        )
    col_widths = [14 * mm, 17.6 * mm, 17.6 * mm, 17.6 * mm, 17.6 * mm, 17.6 * mm]
    return _octave_table(header, rows, col_widths)


def _prediction_statement(
    result: ReverberationModelResult, language: str
) -> tuple[str, list[str]]:
    """The boxed prediction descriptor and the per-model spread beside it."""
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    primary = np.asarray(result.models[_PRIMARY_MODEL], dtype=np.float64)
    value, is_mid = _mid_or_first(freqs, primary)
    if is_mid:
        statement = t(
            "Predicted T<sub>mid</sub> (Arau-Puchades, 500-1000 Hz) = "
            "<b>{value} s</b>",
            language,
        ).format(value=_fmt_t(value, language))
    else:
        statement = t(
            "Predicted T (Arau-Puchades) = <b>{value} s</b>", language
        ).format(value=_fmt_t(value, language))
    extended: list[str] = []
    for name in _MODEL_NAMES:
        model_value, _ = _mid_or_first(
            freqs, np.asarray(result.models[name], dtype=np.float64)
        )
        extended.append(f"{name}: {_fmt_t(model_value, language)} s")
    return statement, extended


def render_reverberation_models_report(
    result: ReverberationModelResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a reverberation-time prediction fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.room.reverberation_prediction.ReverberationModelResult`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` renders a bare
        prediction fiche. A supplied ``requirement`` is printed as a target
        reverberation-time reference line (no PASS/FAIL, since a room target is
        a range).
    :param verbose: Accepted for signature parity with the other fiches; the
        model table already shows every computed value, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # every model column is always shown; kept for signature parity
    try:
        from reportlab.lib import colors
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    flow: list[Any] = [
        Paragraph(t("Reverberation-time prediction", language), title_style),
        Paragraph(
            t(
                "Design-stage prediction of the reverberation time by five "
                "statistical-acoustics models (Sabine, Eyring, "
                "Millington-Sette, Fitzroy and Arau-Puchades).",
                language,
            ),
            basis_style,
        ),
    ]

    extra = [(t("Total surface area S [m<super>2</super>]", language),
              fmt_meta(result.surface_area, language))]
    flow.extend(_header_grid(result.volume, extra, metadata, language))
    flow.append(Spacer(1, 8))

    from reportlab.lib.units import mm

    left_cell = [
        Paragraph(t("Reverberation time by model [s]", language), caption_style),
        _models_table(result, language),
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 70 * mm, y_top=None, language=language
    )
    flow.append(
        two_panel_body(left_cell, plot_drawing, left_width_mm=102.0, plot_width_mm=72.0)
    )
    flow.append(Spacer(1, 8))

    statement, extended = _prediction_statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))
    if metadata is not None and metadata.requirement is not None:
        flow.extend(_target_line(metadata.requirement, styles, language))

    basis_strip = measurement_basis_style()
    flow.append(
        Paragraph(
            t(
                "Design-stage prediction from the room geometry and surface "
                "absorption by the classical statistical-acoustics models; it "
                "is not a measurement. The five models bracket the "
                "reverberation time likely to occur; Arau-Puchades is "
                "recommended for a non-uniform absorption distribution and "
                "drives the boxed descriptor.",
                language,
            ),
            basis_strip,
        )
    )
    flow.append(
        Paragraph(
            t(
                "Sabine T = 24 ln10/c0 * V/(A + 4mV); Eyring replaces A by "
                "-S ln(1 - alpha_bar); Millington-Sette sums -S_i ln(1 - "
                "alpha_i) per surface; Fitzroy and Arau-Puchades are the "
                "area-weighted arithmetic and geometric means of the three "
                "axial Eyring times.",
                language,
            ),
            basis_strip,
        )
    )
    flow.extend(footer_flow(metadata, language))
    return build_document(path, flow, t("Reverberation-time prediction", language))


# ---------------------------------------------------------------------------
# Enclosed-space characterisation fiche (EN 12354-6).
# ---------------------------------------------------------------------------


def _enclosed_table(result: ReverberationResult, language: str) -> Any:
    """Build the per-band ``f | A | T`` table of the enclosed-space fiche."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    head = _band_header_style()
    header = [
        Paragraph(t("f [Hz]", language), head),
        Paragraph("A [m<super>2</super>]", head),
        Paragraph("T [s]", head),
    ]
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    area = np.asarray(result.absorption_area, dtype=np.float64)
    rt = np.asarray(result.reverberation_time, dtype=np.float64)
    rows: list[list[Any]] = []
    for i, fk in enumerate(freqs):
        rows.append(
            [
                f"{round(float(fk))}",
                format_number(float(area[i]), language, decimals=2),
                _fmt_t(rt[i], language),
            ]
        )
    return _octave_table(header, rows, [18 * mm, 19 * mm, 19 * mm])


def _enclosed_statement(
    result: ReverberationResult, language: str
) -> tuple[str, list[str]]:
    """The boxed reverberation-time descriptor and the mid-frequency area."""
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    rt = np.asarray(result.reverberation_time, dtype=np.float64)
    area = np.asarray(result.absorption_area, dtype=np.float64)
    t_value, is_mid = _mid_or_first(freqs, rt)
    if is_mid:
        statement = t(
            "T<sub>mid</sub> (500-1000 Hz) = <b>{value} s</b>", language
        ).format(value=_fmt_t(t_value, language))
    else:
        statement = t("T = <b>{value} s</b>", language).format(
            value=_fmt_t(t_value, language)
        )
    a_value, _ = _mid_or_first(freqs, area)
    extended: list[str] = []
    if math.isfinite(a_value):
        extended.append(
            t("A (500-1000 Hz) = {value} m<super>2</super>", language).format(
                value=format_number(a_value, language, decimals=2)
            )
        )
    return statement, extended


def render_enclosed_space_report(
    result: ReverberationResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an enclosed-space absorption/reverberation fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.room.enclosed_space_absorption.ReverberationResult`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` renders a bare
        characterisation fiche. A supplied ``requirement`` is printed as a
        target reverberation-time reference line (no PASS/FAIL).
    :param verbose: Accepted for signature parity with the other fiches; the
        band table already shows both A and T, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # A and T are always shown; kept for signature parity
    try:
        from reportlab.lib import colors
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t(
            "{standard} estimate of the equivalent sound absorption area and "
            "reverberation time of an enclosed space per EN 12354-6:2003.",
            language,
        ).format(standard=html.escape(measurement_standard))
    else:
        basis = t(
            "Equivalent sound absorption area and reverberation time of an "
            "enclosed space per EN 12354-6:2003.",
            language,
        )
    flow: list[Any] = [
        Paragraph(t("Sound absorption in an enclosed space", language), title_style),
        Paragraph(basis, basis_style),
    ]

    extra = [(t("Object fraction &#968;", language),
              fmt_meta(result.object_fraction, language))]
    flow.extend(_header_grid(result.volume, extra, metadata, language))
    flow.append(Spacer(1, 8))

    from reportlab.lib.units import mm

    left_cell = [
        Paragraph(
            t("Absorption area A and reverberation time T", language), caption_style
        ),
        _enclosed_table(result, language),
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=None, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    statement, extended = _enclosed_statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))
    if metadata is not None and metadata.requirement is not None:
        flow.extend(_target_line(metadata.requirement, styles, language))

    basis_strip = measurement_basis_style()
    flow.append(
        Paragraph(
            t(
                "Estimate of the equivalent sound absorption area A and the "
                "reverberation time T of an enclosed space from its surface, "
                "object and air absorption by the EN 12354-6:2003 Clause 4 "
                "model (A by Formula 1, T = 55.3/c0 * V(1 - psi)/A by "
                "Formula 5, with the object fraction psi = sum Vobj / V). It "
                "is an estimate for a diffuse field, not a measurement.",
                language,
            ),
            basis_strip,
        )
    )
    flow.extend(footer_flow(metadata, language))
    return build_document(
        path, flow, t("Sound absorption in an enclosed space", language)
    )
