#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared renderer for the sound-power determination fiches.

The enveloping-surface pressure method (ISO 3744/3746 and the ISO 3745
precision variant, :mod:`.iso3744`) and the sound-intensity method
(ISO 9614-2 scanning, :mod:`.iso9614`) print the same one-page sound-power
test sheet: a title and standard-basis line, an optional metadata header, a
full-width per-band table, the sound-power spectrum ``LW(f)`` drawn by the
result's own ``plot``, a boxed A-weighted sound power level ``LWA`` (dB re
1 pW), an optional requirement verdict and a measurement-basis strip. Only the
fixed text (the basis line, the table columns and the basis strip) differs
between the methods. This module holds the common skeleton so both renderers
call it, parameterised by their differences; the quantity-independent flowable
helpers still live in :mod:`._layout`.

reportlab, matplotlib and svglib are soft dependencies imported lazily
(reportlab and svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

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
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    verdict_flow,
)
from .metadata import ReportMetadata

#: Reference sound power for the sound power level, 1 pW = 10^-12 W.
POWER_REFERENCE = "1 pW"


def _num(value: float | None, language: str = "en") -> str | None:
    """Round-trip a client-supplied metadata number, or ``None`` when unset."""
    return fmt_meta(value, language) if value is not None else None


def metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the sound-power header grid.

    Only supplied fields are returned; the header is identical for the
    pressure (ISO 3744/3745) and intensity (ISO 9614) methods (both describe a
    noise source and a test environment). ``specimen`` names the noise source
    and ``test_room`` the test environment; the measurement surface area ``S``
    is the result's own computed value, shown in the result box and basis strip
    rather than trusted from the metadata.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Noise source", language), metadata.specimen),
        (t("Test environment", language), metadata.test_room),
        (t("Instrumentation", language), metadata.instrumentation),
        (t("Temperature [&#176;C]", language), _num(metadata.temperature, language)),
        (
            t("Relative humidity [%]", language),
            _num(metadata.relative_humidity, language),
        ),
        (t("Ambient pressure [kPa]", language), _num(metadata.pressure, language)),
        (t("Date of test", language), metadata.test_date),
    ]
    return escaped_pairs(specs)


def d1(value: float, language: str = "en") -> str:
    """One decimal place, locale-aware separator; em dash when not finite."""
    if not math.isfinite(float(value)):
        return "—"
    return format_number(float(value), language, decimals=1)


def band_labels(frequencies: np.ndarray | None, n: int) -> tuple[list[str], int]:
    """Return the nominal band labels and the band fraction (1, 3 or 0).

    A per-band result is labelled by its nominal octave/one-third-octave
    mid-band frequency (IEC 61260), not the exact base-ten centre, matching the
    band axis of the embedded spectrum. A result without band frequencies
    (a directly measured broadband level) is labelled ``Band 1``, ``Band 2``,
    ... and reports fraction ``0`` (no octave grouping).
    """
    if frequencies is None:
        return [f"Band {i + 1}" for i in range(n)], 0
    from ..filters.frequencies import _infer_band_fraction, _nominal_freq_for_band

    freqs = np.asarray(frequencies, dtype=np.float64)
    fraction = _infer_band_fraction(freqs) if freqs.size >= 2 else 1
    labels = [f"{_nominal_freq_for_band(f, float(fraction)):g}" for f in freqs]
    return labels, fraction


def range_str(values: np.ndarray, language: str = "en", decimals: int = 1) -> str:
    """Format a per-band quantity as a single value or an ``a to b`` range.

    ``decimals`` is the precision the quantity is read at: one decimal for a
    level in decibels, two for a dimensionless field indicator judged against a
    threshold of 0,6 or 2 (ISO 9614-3:2002 Annex C). A spread below half the
    last printed digit is one value rather than a range.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "—"
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if abs(hi - lo) < 0.5 * 10.0**-decimals:
        return f"{format_number(lo, language, decimals=decimals)}"
    return t("{lo} to {hi}", language).format(
        lo=format_number(lo, language, decimals=decimals),
        hi=format_number(hi, language, decimals=decimals),
    )


def energy_sum(levels: Any) -> float:
    """Energy sum of a per-band level array, ``10 lg(sum 10^(L/10))`` dB.

    Non-finite bands are skipped: a band the method could not determine
    contributes no energy rather than poisoning the total with ``NaN``. An
    array with no finite band has no total, which is ``NaN``.
    """
    arr = np.asarray(levels, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.sum(10.0 ** (finite / 10.0))))


def total_power_level(result: Any) -> float:
    """Energy sum of the band sound-power levels, ``10 lg(sum 10^(LW/10))`` dB."""
    return energy_sum(result.sound_power_level)


def headline_level(result: Any, level_a: float | None = None) -> float:
    """The single number the verdict compares: ``LWA`` if defined, else total ``LW``.

    ``level_a`` overrides the result's own A-weighted total, for a fiche whose
    standard prescribes a different summation than the one the result computed
    (the ISO 9614-3 A-weighted determination omits the bands its Annex C
    criteria reject, which the result object cannot know).
    """
    lwa = float(result.sound_power_level_a if level_a is None else level_a)
    return lwa if math.isfinite(lwa) else total_power_level(result)


def power_statement(
    result: Any, language: str = "en", *, level_a: float | None = None
) -> tuple[str, list[str]]:
    """The boxed sound-power result and its base extended terms.

    Boxes the A-weighted sound power level ``LWA`` when it is defined (band
    frequencies were supplied); otherwise the energy-summed total ``LW``. The
    base extended terms carry the total ``LW`` (only when ``LWA`` heads the
    box); each renderer appends its own further terms (the expanded uncertainty
    and surface area for the pressure method, the surface area and measurement
    grade for the intensity method).

    ``level_a`` overrides the result's own ``sound_power_level_a`` with the
    A-weighted level the fiche states, for a standard that prescribes its own
    band screening (ISO 9614-3:2002 clause 10 f) 2) omits the bands its Annex C
    criteria reject, a screening the result object cannot perform on its own).
    """
    lwa = float(result.sound_power_level_a if level_a is None else level_a)
    total = total_power_level(result)
    extended: list[str] = []
    if math.isfinite(lwa):
        statement = t(
            "Sound power level L<sub>WA</sub> = <b>{lwa} dB(A)</b> re {ref}",
            language,
        ).format(lwa=d1(lwa, language), ref=POWER_REFERENCE)
        if math.isfinite(total):
            extended.append(
                t("Total L<sub>W</sub> = {value} dB re {ref}", language).format(
                    value=d1(total, language), ref=POWER_REFERENCE
                )
            )
    else:
        statement = t(
            "Sound power level L<sub>W</sub> = <b>{value} dB</b> re {ref}",
            language,
        ).format(value=d1(total, language), ref=POWER_REFERENCE)
    return statement, extended


def power_verdict(
    result: Any,
    requirement: float,
    language: str = "en",
    *,
    level_a: float | None = None,
) -> tuple[str, bool]:
    """Verdict text and PASS flag against a declared sound-power limit.

    A sound-power emission is a quantity where less is better, so the source
    passes when its A-weighted level (or total ``LW`` when no band frequencies
    were supplied) is at or below the declared limit. The comparison uses the
    displayed (one-decimal) value so the printed number cannot contradict the
    verdict at the boundary. ``level_a`` overrides the result's own A-weighted
    total with the one the fiche boxes, so the verdict is taken on the number
    the reader sees (see :func:`power_statement`).
    """
    value = headline_level(result, level_a)
    weighted = math.isfinite(
        float(result.sound_power_level_a if level_a is None else level_a)
    )
    passed = math.isfinite(value) and display_round(value) <= requirement
    if weighted:
        text = t(
            "L<sub>WA</sub> = {value} dB(A), declared limit &#8804; {req} dB(A)",
            language,
        ).format(
            value=d1(value, language),
            req=format_number(requirement, language, decimals=1),
        )
    else:
        text = t(
            "L<sub>W</sub> = {value} dB, declared limit &#8804; {req} dB",
            language,
        ).format(
            value=d1(value, language),
            req=format_number(requirement, language, decimals=1),
        )
    return text, passed


def level_limit_verdict(
    value: float, requirement: float, symbol: str, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag for a level against a declared upper limit.

    A generic lower-is-better comparison (a power or normalised-level emission
    passes at or below the declared limit), stated for the quantity ``symbol``
    (already markup, e.g. ``L<sub>Ws</sub>``). The comparison uses the displayed
    (one-decimal) value so the printed number cannot contradict the verdict at
    the boundary. Shared by the structure-borne fiches (EN 15657 injected power,
    EN 12354-5 installed prediction), whose symbols differ from the airborne
    ``L<sub>WA</sub>``/``L<sub>W</sub>`` handled by :func:`power_verdict`.
    """
    passed = math.isfinite(value) and display_round(value) <= requirement
    text = t(
        "{sym} = {value} dB, declared limit &#8804; {req} dB", language
    ).format(
        sym=symbol,
        value=d1(value, language),
        req=format_number(requirement, language, decimals=1),
    )
    return text, passed


def fraction_caption(result: Any, language: str = "en") -> str:
    """The caption declaring the analysis band set above the table."""
    freqs = getattr(result, "frequencies", None)
    n = np.asarray(result.sound_power_level, dtype=np.float64).size
    if freqs is None:
        return t("Sound power levels per band", language)
    _, fraction = band_labels(freqs, n)
    if fraction == 1:
        return t("Octave-band sound power levels", language)
    return t("One-third-octave-band sound power levels", language)


def power_value_table(
    header: Sequence[str],
    rows_data: Sequence[Sequence[Any]],
    widths: Sequence[float],
    fraction: int,
) -> Any:
    """Build the full-width per-band table with the accredited sound-power style.

    ``header`` holds the (markup) column headings and ``rows_data`` the band
    rows already formatted to strings; ``widths`` are the column widths in
    millimetres and ``fraction`` the band fraction (a one-third-octave set,
    ``fraction == 3``, is grouped by octave with a thin rule after every
    triplet, as accredited sound-power tables print it). Called only after the
    renderer has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    from ._layout import fiche_paragraph as Paragraph

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    thin = colors.HexColor("#c9d4e0")
    styles = getSampleStyleSheet()
    head_style = ParagraphStyle(
        "sound_power_thead", parent=styles["Normal"], fontSize=7.6,
        textColor=colors.white, alignment=1, leading=9.0,
    )

    n = len(rows_data)
    rows: list[list[Any]] = [[Paragraph(h, head_style) for h in header]]
    rows.extend([list(row) for row in rows_data])

    style_cmds: list[Any] = [
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("BOX", (0, 0), (-1, -1), 0.5, accent),
    ]
    # A one-third-octave set groups by octave (a thin rule after every triplet),
    # as accredited sound-power tables print it; an octave set has no triplets.
    if fraction == 3:
        for triplet_end in range(3, n, 3):
            style_cmds.append(
                ("LINEBELOW", (0, triplet_end), (-1, triplet_end), 0.4, thin)
            )
    table = Table(rows, colWidths=[w * mm for w in widths], repeatRows=1)
    table.setStyle(TableStyle(style_cmds))
    table.hAlign = "CENTER"
    return table


def render_sound_power_fiche(
    result: Any,
    path: str,
    *,
    title: str,
    basis: str,
    caption: str,
    value_table: Any,
    statement: str,
    extended: list[str],
    basis_strips: Sequence[str],
    metadata: ReportMetadata | None,
    language: str,
    verdict: tuple[str, bool] | None = None,
    disclaimer: str | None = None,
    figsize: tuple[float, float] = (9.2, 3.4),
) -> str:
    """Assemble the shared sound-power fiche flow and build the PDF at ``path``.

    The metadata header grid and the requirement verdict are identical for the
    pressure and intensity methods, so they are built here from ``metadata``
    (the header via :func:`metadata_pairs`, the verdict via
    :func:`power_verdict`); each renderer only supplies the parts that genuinely
    differ (the basis line, the per-band ``value_table`` and the basis strips).

    :param result: The sound-power result; its ``plot`` draws the band-axis
        ``LW(f)`` spectrum (native to the library) and, with the metadata
        ``requirement``, its A-weighted total drives the verdict.
    :param title: The already-translated fiche title.
    :param basis: The already-translated standard-basis line.
    :param caption: The already-translated band-set caption above the table.
    :param value_table: The pre-built per-band reportlab table flowable.
    :param statement: The boxed single-number statement (markup).
    :param extended: The extended terms shown alongside the boxed statement.
    :param basis_strips: The measurement-basis-strip paragraphs (markup).
    :param metadata: Optional :class:`ReportMetadata` for the header, the
        requirement verdict and the footer identity.
    :param language: ``"en"`` (default) or ``"es"``.
    :param verdict: Optional pre-computed ``(text, passed)`` verdict; when
        given it replaces the default airborne sound-power requirement
        comparison, so a caller with a different quantity symbol (the
        structure-borne fiches) can state its own. ``None`` uses the
        :func:`power_verdict` fallback when a ``requirement`` is supplied.
    :param disclaimer: Footer scope sentence override (an English key
        translated for display); a prediction fiche passes its
        modelled-configuration wording (see :func:`._layout.footer_flow`).
    :param figsize: Size of the embedded spectrum in inches, rendered to the
        full 174 mm text width. The default suits the six or seven bands the
        pressure fiches print; a fiche whose band set fills the page with rows
        (the one-third-octave ISO 9614-3 sheet) asks for a shallower panel so
        that the whole determination still fits on one page.
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

    flow.append(Paragraph(caption, caption_style))
    flow.append(value_table)
    flow.append(Spacer(1, 8))

    # Landscape sound-power spectrum drawn by the result's own single-panel
    # plot(ax=...); the band axis carries nominal labels (not base-ten log).
    flow.append(
        render_figure_drawing(
            result.plot, 174 * mm, y_top=None, figsize=figsize,
            language=language,
        )
    )
    flow.append(Spacer(1, 8))

    flow.append(result_box(statement, styles, accent, extended))
    # A caller may supply its own verdict (its quantity symbol and sign rule);
    # otherwise the airborne sound-power fiches fall back to the L_WA/L_W
    # requirement comparison. The two paths are mutually exclusive.
    if verdict is not None:
        text, passed = verdict
        flow.extend(verdict_flow(text, passed, styles, language))
    elif metadata is not None and metadata.requirement is not None:
        text, passed = power_verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))

    basis_style_strip = measurement_basis_style()
    for strip in basis_strips:
        flow.append(Paragraph(strip, basis_style_strip))
    flow.extend(footer_flow(metadata, language, disclaimer=disclaimer))

    return build_document(path, flow, title)
