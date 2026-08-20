#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tonal audibility assessment fiche (reportlab renderer, ISO 1996-2:2017).

Renders a
:class:`~phonometry.psychoacoustics.quality.tone_audibility.ToneAudibilityResult` to a
one-page PDF laid out like a tonal-assessment report of an environmental-noise
laboratory. ISO 1996-2:2017 Annex J assesses the audibility of prominent tones
by the engineering method of ISO/PAS 20065:2016 (the critical band about a
tone, the critical-band masking-noise level and the audibility
``ΔL_ta = Lpt − Lpn − a_v``) and maps the resulting audibility to the tonal
adjustment ``K`` of Table J.1:

* a title and the standard-basis line (measurement standard + the ISO 1996-2
  Annex J engineering method);
* an optional metadata header grid, rendered only for the fields supplied on
  the :class:`ReportMetadata`: the source/situation (``specimen``), the client,
  the measurement position (``test_room``), the instrumentation and the date,
  with the analysis line spacing ``Δf`` read from the result;
* a full-width table of the key quantities for every detected tone (tone
  frequency ``f_T``, entry type, tone level ``Lpt``, critical-band masking-noise
  level ``Lpn``, critical bandwidth ``Δf_c``, the audibility ``ΔL_ta`` and, in
  verbose mode, the extended uncertainty ``U``), above the level-versus-frequency
  analysis plot with the tones and their critical-band masking noise marked (the
  result's own :meth:`plot`-style vector figure);
* a boxed single-number result: the decisive tonal audibility ``ΔL_ta`` and the
  derived tonal adjustment ``K`` (Table J.1);
* an optional PASS/FAIL verdict row when an audibility limit is supplied via the
  metadata ``requirement`` (read as the maximum acceptable ``ΔL_ta``), plus a
  short prominence note stating whether a prominent tone is present and the
  adjustment applies; and
* a footer identity/disclaimer block.

Like :mod:`.iso9612` this uses a stacked table layout: the per-tone table needs
the full content width and the analysis plot is landscape. The
quantity-independent skeleton lives in :mod:`._layout`; this module only holds
the ISO 1996-2 tone-audibility specifics. reportlab, matplotlib and svglib are
soft dependencies imported lazily (reportlab and svglib ship in the
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
    _MUTED_HEX,
    _REPORTLAB_HINT,
    _VERDICT_OK_HEX,
    analysis_cell_styles,
    build_document,
    display_round,
    document_styles,
    footer_flow,
    grid_table,
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    stacked_table,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..psychoacoustics.quality.tone_audibility import ToneAudibilityResult


def _fmt(value: float, language: str, decimals: int = 1) -> str:
    """A quantity rounded to ``decimals`` decimals, localised separator."""
    return format_number(float(value), language, decimals=decimals)


def _tonal_adjustment(audibility: float) -> int:
    """The ISO 1996-2:2017 Table J.1 tonal adjustment ``K`` for an audibility.

    Imported lazily from the environmental-measurement domain module (the fiche
    layer references domain code only at call time, never at import time).
    """
    from ..environment.assessment.measurement import (
        tonal_adjustment_from_mean_audibility,
    )

    return tonal_adjustment_from_mean_audibility(audibility)


def _metadata_pairs(
    result: ToneAudibilityResult,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the tonal-assessment header grid.

    A tonal assessment identifies the source/situation whose noise was analysed
    (``specimen``), the client, the measurement position (``test_room``) and the
    instrumentation; only supplied fields are returned. The analysis line
    spacing ``Δf`` is always shown, read from the result rather than the
    metadata.
    """
    specs: list[tuple[str, str | None]] = []
    if metadata is not None:
        specs += [
            (t("Source / situation", language), _esc(metadata.specimen)),
            (t("Client", language), _esc(metadata.client)),
            (t("Measurement position", language), _esc(metadata.test_room)),
            (t("Instrumentation", language), _esc(metadata.instrumentation)),
            (t("Date of test", language), _esc(metadata.test_date)),
        ]
    specs.append(
        (
            t("Analysis line spacing &#916;f [Hz]", language),
            _fmt(result.line_spacing, language),
        )
    )
    return [(label, value) for label, value in specs if value]


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _present_markup(present: bool, language: str = "en") -> str:
    """Inline markup for the per-tone present/absent column.

    A tone is *present* when its audibility exceeds the masking threshold
    (``ΔL_ta > 0``, ISO/PAS 20065:2016 Clause 5.3.2); such a row carries a
    filled accent dot and ``Present``, an inaudible one a muted en dash.
    """
    if present:
        return (
            f"<font color='{_VERDICT_OK_HEX}'>&#9679; {t('Present', language)}</font>"
        )
    return f"<font color='{_MUTED_HEX}'>&#8211;</font>"


def _type_label(group_size: int | None, language: str = "en") -> str:
    """The entry-type label: a single tone or a combined ``FG`` group."""
    if group_size is None or group_size <= 1:
        return t("Single", language)
    return t("FG ({n})", language).format(n=int(group_size))


def _key_quantity_table(
    result: ToneAudibilityResult, verbose: bool = False, language: str = "en"
) -> Any:
    """The full-width per-tone key-quantity table (ISO/PAS 20065 Table E.2 style).

    One row per detected tone: frequency ``f_T``, entry type (single tone or a
    combined ``FG`` group), tone level ``Lpt``, critical-band masking-noise
    level ``Lpn``, critical bandwidth ``Δf_c``, the audibility ``ΔL_ta`` and the
    present/absent flag. ``verbose`` adds the extended-uncertainty column when
    the result carries per-tone uncertainties. The decisive (most audible) row
    is emphasised.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso1996tone")

    freqs = np.asarray(result.tone_frequencies, dtype=np.float64)
    lt = np.asarray(result.tone_levels, dtype=np.float64)
    lpn = np.asarray(result.critical_band_levels, dtype=np.float64)
    dfc = np.asarray(result.critical_bandwidths, dtype=np.float64)
    delta = np.asarray(result.audibilities, dtype=np.float64)
    groups = result.group_sizes
    uncertainties = result.extended_uncertainties
    decisive = int(np.argmax(delta))
    order = np.argsort(freqs, kind="stable")

    show_u = verbose and uncertainties is not None
    headers = [
        t("Tone f<sub>T</sub> [Hz]", language),
        t("Type", language),
        "L<sub>pt</sub> [dB]",
        "L<sub>pn</sub> [dB]",
        "&#916;f<sub>c</sub> [Hz]",
        "&#916;L<sub>ta</sub> [dB]",
        t("Present", language),
    ]
    widths = [26.0, 22.0, 24.0, 24.0, 26.0, 28.0, 24.0]
    if show_u:
        headers.insert(6, "U [dB]")
        widths = [23.0, 20.0, 22.0, 22.0, 24.0, 25.0, 18.0, 20.0]

    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for i in order.tolist():
        emph = "<b>{}</b>".format if i == decisive else str
        present = bool(delta[i] > 0.0)
        gs = int(groups[i]) if groups is not None else None
        row = [
            Paragraph(emph(_fmt(freqs[i], language)), label_style),
            Paragraph(_type_label(gs, language), value_style),
            Paragraph(_fmt(lt[i], language), value_style),
            Paragraph(_fmt(lpn[i], language), value_style),
            Paragraph(_fmt(dfc[i], language, decimals=0), value_style),
            Paragraph(emph(_fmt(delta[i], language)), value_style),
            Paragraph(_present_markup(present, language), value_style),
        ]
        if show_u and uncertainties is not None:
            row.insert(6, Paragraph(_fmt(uncertainties[i], language), value_style))
        data.append(row)

    table = stacked_table(data, [w * mm for w in widths])
    # Emphasise the decisive tone's row (its row index in the table is its rank
    # in the frequency order, plus the header row). ``order`` is a permutation,
    # so the decisive tone appears exactly once; a plain list lookup finds it.
    dec_row = order.tolist().index(decisive) + 1
    table.setStyle(
        [
            (
                "LINEBELOW",
                (0, dec_row),
                (-1, dec_row),
                0.5,
                colors.HexColor(_ACCENT_HEX),
            ),
            (
                "LINEABOVE",
                (0, dec_row),
                (-1, dec_row),
                0.5,
                colors.HexColor(_ACCENT_HEX),
            ),
        ]
    )
    return table


def _statement(
    result: ToneAudibilityResult, language: str = "en"
) -> tuple[str, list[str]]:
    """The boxed decisive audibility ``ΔL_ta`` and tonal adjustment ``K``.

    The decisive audibility is the largest over the detected tones (Clause
    5.3.8 Step 4); the adjustment ``K`` follows from ISO 1996-2:2017 Table J.1,
    computed on the audibility rounded exactly as displayed so the box can never
    contradict its own number at a table boundary.
    """
    delta = display_round(result.decisive_audibility)
    adjustment = _tonal_adjustment(delta)
    statement = t(
        "Tonal audibility &#916;L<sub>ta</sub> = <b>{dl} dB</b> &nbsp; "
        "tonal adjustment K = <b>{k} dB</b>",
        language,
    ).format(dl=_fmt(delta, language), k=str(adjustment))
    extended = [
        t("Decisive tone f<sub>T</sub> = {f} Hz", language).format(
            f=_fmt(result.decisive_frequency, language)
        ),
        t("Analysis line spacing &#916;f = {df} Hz", language).format(
            df=_fmt(result.line_spacing, language)
        ),
    ]
    if result.extended_uncertainties is not None:
        i = int(np.argmax(np.asarray(result.audibilities, dtype=np.float64)))
        extended.append(
            t("Extended uncertainty U = {u} dB (90 % coverage)", language).format(
                u=_fmt(result.extended_uncertainties[i], language)
            )
        )
    return statement, extended


def _verdict(
    result: ToneAudibilityResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag against a supplied maximum audibility.

    The ``requirement`` is read as the maximum acceptable decisive audibility
    ``ΔL_ta`` in dB (a lower audibility is better, as a louder tone is more
    intrusive); the assessment passes when the decisive audibility, rounded as
    displayed, is at or below it.
    """
    delta = display_round(result.decisive_audibility)
    passed = delta <= requirement + 1e-9
    text = t(
        "&#916;L<sub>ta</sub> = {dl} dB, required &#8804; {req} dB", language
    ).format(dl=_fmt(delta, language), req=_fmt(requirement, language))
    return text, passed


def render_tone_audibility_report(
    result: ToneAudibilityResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a tonal audibility assessment fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.psychoacoustics.quality.tone_audibility.ToneAudibilityResult`
        carrying the detected tones, their tone/masking levels and audibilities.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity (``specimen`` the source/situation, ``client``, ``test_room``
        the measurement position, ``instrumentation`` and ``test_date``) and the
        footer identity. A supplied ``requirement`` is read as the maximum
        acceptable decisive audibility ``ΔL_ta`` in dB.
    :param verbose: When True, the key-quantity table adds the per-tone extended
        uncertainty column (when the result carries the uncertainties).
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
    title = t("Tonal audibility assessment", language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t(
            "{standard} tonal audibility by the ISO 1996-2:2017 Annex J "
            "engineering method (ISO/PAS 20065:2016); tonal adjustment K per "
            "Table J.1.",
            language,
        ).format(standard=html.escape(measurement_standard))
    else:
        basis = t(
            "Tonal audibility by the ISO 1996-2:2017 Annex J engineering method "
            "(ISO/PAS 20065:2016); tonal adjustment K per Table J.1.",
            language,
        )

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    header_pairs = _metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph(t("Detected tones", language), caption_style))
    flow.append(_key_quantity_table(result, verbose, language))
    flow.append(Spacer(1, 8))

    # Full-width, landscape level-versus-frequency analysis plot: the tone
    # levels above their critical-band masking noise (a self-scaling dB axis).
    def _levels_plot(ax: Any = None, language: str = "en", **kwargs: Any) -> Any:
        from .._plot.psychoacoustics import plot_tone_audibility_levels

        return plot_tone_audibility_levels(result, ax=ax, language=language, **kwargs)

    flow.append(
        render_figure_drawing(
            _levels_plot,
            174 * mm,
            y_top=None,
            figsize=(9.2, 3.1),
            language=language,
        )
    )
    flow.append(Spacer(1, 8))

    statement, extended = _statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))

    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))

    # Prominence note: whether a prominent tone is present and K applies.
    delta = display_round(result.decisive_audibility)
    adjustment = _tonal_adjustment(delta)
    basis_strip_style = measurement_basis_style()
    if delta > 0.0:
        prominence = t(
            "A prominent tone is present (decisive &#916;L<sub>ta</sub> = "
            "{dl} dB &gt; 0); the tonal adjustment K = {k} dB (ISO 1996-2:2017 "
            "Table J.1) applies.",
            language,
        ).format(dl=_fmt(delta, language), k=str(adjustment))
    else:
        prominence = t(
            "No prominent tone is present (decisive &#916;L<sub>ta</sub> = "
            "{dl} dB &#8804; 0); no tonal adjustment applies (K = 0 dB).",
            language,
        ).format(dl=_fmt(delta, language))
    flow.append(Paragraph(prominence, basis_strip_style))
    flow.append(
        Paragraph(
            t(
                "The tonal audibility &#916;L<sub>ta</sub> = L<sub>pt</sub> "
                "&#8722; L<sub>pn</sub> &#8722; a<sub>v</sub> is the amount by "
                "which the tone level rises above the masking threshold of the "
                "surrounding noise (ISO/PAS 20065:2016, Formula (14)); the "
                "decisive value is the largest over the detected tones "
                "(Clause 5.3.8).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.append(
        Paragraph(
            t(
                "L<sub>pn</sub> is the critical-band masking-noise level "
                "(Formula (12)), &#916;f<sub>c</sub> the critical bandwidth "
                "about the tone (Formula (2)); an FG entry combines tones "
                "sharing a critical band (Formula (17)).",
                language,
            ),
            basis_strip_style,
        )
    )
    # Table J.1 is keyed on the mean audibility over the J staggered spectra of
    # ISO/PAS 20065 Clause 5.3.9; this fiche assesses the one spectrum supplied,
    # whose decisive audibility is that mean only for J = 1.
    flow.append(
        Paragraph(
            t(
                "Table J.1 assigns K from the mean audibility &#916;L, the "
                "energy mean of the decisive audibilities of the J assessed "
                "spectra (ISO/PAS 20065:2016, Clause 5.3.9, Formula (20)). "
                "This fiche assesses a single spectrum, so the K shown follows "
                "from its decisive audibility; where several staggered spectra "
                "are assessed, rate their mean audibility instead.",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
