#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared renderer for the noise-control performance fiches.

The machine-enclosure insertion loss (:mod:`.enclosure`), the reactive-silencer
transmission loss (:mod:`.silencer`) and the HVAC duct-noise spectrum
(:mod:`.hvac`) print the same one-page performance sheet: a title and
method-basis line, an optional metadata header, a two-panel body with the
per-band table on the left and the result's own ``plot`` on the right, a boxed
single-number performance figure, an optional requirement verdict and a
method-basis strip. Only the fixed text and the per-band table content differ
between them, so this module holds the common skeleton and each renderer
supplies only its specifics (the basis line, the table columns, the boxed
figure and the verdict direction).

The quantity-independent flowable helpers live in :mod:`._layout`; the per-band
table builder, the nominal band labels and the one-decimal cell formatter are
shared with the sound-power fiches (:mod:`._sound_power_fiche`), and the header
grid reuses the sound-power source/environment layout.

reportlab, matplotlib and svglib are soft dependencies imported lazily
(reportlab and svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._i18n import t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    build_document,
    display_round,
    document_styles,
    footer_flow,
    grid_table,
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from ._sound_power_fiche import band_labels, d1, metadata_pairs, power_value_table
from .metadata import ReportMetadata

__all__ = [
    "PREDICTION_DISCLAIMER",
    "band_labels",
    "d1",
    "mean_finite",
    "performance_verdict",
    "power_value_table",
    "render_noise_control_fiche",
]

#: Footer scope statement of the noise-control fiches. These quantities come
#: from a model evaluated on the declared inputs, so the measurement fiches'
#: specimen-scoped sentence does not apply. The wording says "stated inputs"
#: rather than "design data" because some of those inputs may themselves be
#: measured (the enclosure fiche takes a panel transmission loss the caller
#: may have measured); what the fiche reports is still the model's output, not
#: a measurement of the assembled device. English key, translated via
#: :func:`t` at render time.
PREDICTION_DISCLAIMER = (
    "The results are a prediction computed from the stated inputs; they are "
    "not a measurement of a specimen."
)


def mean_finite(values: np.ndarray) -> float:
    """Arithmetic mean of the finite entries of a per-band array (``nan`` if none)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def performance_verdict(
    value: float,
    requirement: float,
    symbol: str,
    *,
    higher_is_better: bool,
    unit: str = "dB",
    language: str = "en",
) -> tuple[str, bool]:
    """Verdict text and PASS flag for a performance figure against a requirement.

    ``higher_is_better`` selects the direction: a predicted insertion or
    transmission loss passes at or above the declared minimum, while a radiated
    noise level passes at or below the declared maximum. Both the value and the
    requirement are compared *and printed* at the same one-decimal display
    rounding (half away from zero, :func:`display_round`), so the printed
    numbers cannot contradict the verdict at the boundary: a value of 0.25
    prints the 0.3 its verdict was decided on, not the 0.2 that Python's
    round-half-to-even formatting would show. ``symbol`` is markup (e.g. ``IL``,
    ``L<sub>WA</sub>``) and ``unit`` its unit symbol (``dB`` or ``dB(A)``),
    neither of which is translated.
    """
    rounded = display_round(value) if math.isfinite(value) else value
    rounded_requirement = display_round(requirement)
    if higher_is_better:
        passed = math.isfinite(value) and rounded >= rounded_requirement
        template = t(
            "{sym} = {value} {unit}, required &#8805; {req} {unit}", language
        )
    else:
        passed = math.isfinite(value) and rounded <= rounded_requirement
        template = t(
            "{sym} = {value} {unit}, required &#8804; {req} {unit}", language
        )
    text = template.format(
        sym=symbol,
        value=d1(rounded, language),
        req=d1(rounded_requirement, language),
        unit=unit,
    )
    return text, passed


def render_noise_control_fiche(
    result: Any,
    path: str,
    *,
    title: str,
    basis: str,
    caption: str,
    value_table: Any,
    statement: str,
    extended: list[str],
    basis_strips: list[str],
    metadata: ReportMetadata | None,
    language: str,
    prediction_statement: str,
    verdict: tuple[str, bool] | None = None,
) -> str:
    """Assemble the shared noise-control fiche flow and build the PDF at ``path``.

    The metadata header grid reuses the sound-power source/environment layout,
    and the requirement verdict is supplied pre-computed by each renderer (its
    quantity symbol and pass direction). The two-panel body puts the per-band
    ``value_table`` on the left (a compact ~64 mm cell that fits the widest
    four-column enclosure/HVAC table) and the result's own ``plot`` on the right.

    :param result: The noise-control result; its ``plot`` draws the right panel.
    :param title: The already-translated fiche title.
    :param basis: The already-translated method-basis line.
    :param caption: The already-translated band-set caption above the table.
    :param value_table: The pre-built per-band reportlab table flowable.
    :param statement: The boxed single-number statement (markup).
    :param extended: The extended terms shown alongside the boxed statement.
    :param basis_strips: The method-basis-strip paragraphs (markup).
    :param metadata: Optional :class:`ReportMetadata` for the header, footer
        identity and (via each renderer) the requirement verdict.
    :param language: ``"en"`` (default) or ``"es"``.
    :param verdict: Optional pre-computed ``(text, passed)`` verdict row.
    :param prediction_statement: The already-translated prediction statement
        printed under the boxed figure. Every quantity these fiches carry is
        computed from a design model rather than measured, so the statement
        (and the prediction-scoped footer) say so, mirroring the EN/ISO 12354
        prediction fiches.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    # Fixed two-panel split, summing to the 174 mm content width: a compact
    # table cell beside the plot, with the embedded figure just inside its cell.
    left_width_mm, plot_width_mm, figure_width_mm = 64.0, 110.0, 108.0
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    if metadata is not None and not metadata.is_empty():
        header_pairs = metadata_pairs(metadata, language)
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    left_cell = [Paragraph(caption, caption_style), value_table]
    plot_drawing = render_figure_drawing(
        result.plot, figure_width_mm * mm, y_top=None, language=language
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

    flow.append(result_box(statement, styles, accent, extended))
    flow.append(
        Paragraph(
            prediction_statement,
            ParagraphStyle(
                "noise_control_prediction", parent=styles["Normal"], fontSize=8.5,
                textColor=colors.HexColor(_MUTED_HEX), spaceBefore=4,
            ),
        )
    )
    if verdict is not None:
        text, passed = verdict
        flow.extend(verdict_flow(text, passed, styles, language))

    basis_style_strip = measurement_basis_style()
    for strip in basis_strips:
        flow.append(Paragraph(strip, basis_style_strip))
    flow.extend(footer_flow(metadata, language, disclaimer=PREDICTION_DISCLAIMER))

    return build_document(path, flow, title)
