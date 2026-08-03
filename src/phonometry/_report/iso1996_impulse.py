#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Impulsive-sound prominence assessment fiche (reportlab renderer).

Renders a
:class:`~phonometry.environment.assessment.impulsive_sound.ImpulseProminenceResult`
to a one-page PDF laid out like an impulsive-sound assessment report of an
environmental-noise laboratory. NT ACOU 112:2002 (carried into
ISO/PAS 1996-3:2022) measures how *prominent* an impulse is from its onset rate
and level difference (the predicted prominence ``P``) and turns the highest
prominence over a 30-minute period into a graduated adjustment ``KI`` added to
the measured ``LAeq``:

* a title and the standard-basis line (measurement standard + the NT ACOU 112 /
  ISO PAS 1996-3 prominence and adjustment method);
* a metadata header grid carrying whichever fields the :class:`ReportMetadata`
  supplies (the source/situation ``specimen``, the client, the measurement
  position ``test_room``, the instrumentation and the date) always followed by
  the assessment period the result was evaluated over, so the grid is present
  even without metadata;
* a full-width per-impulse table (onset rate, level difference, predicted
  prominence ``P`` and whether the onset qualifies as an impulse), above the
  adjustment-curve plot ``KI(P)`` with the candidate impulses marked (the
  result's own :meth:`plot`);
* a boxed single-number result: the governing prominence ``P`` and the derived
  ``LAeq`` adjustment ``KI`` (Formula 2);
* an optional PASS/FAIL verdict row when a maximum acceptable governing
  prominence is supplied via the metadata ``requirement``, plus a prominence
  note stating whether a prominent impulse is present and the adjustment
  applies; and
* a footer identity/disclaimer block.

Like :mod:`.iso1996_tone` this uses a stacked table layout: the per-impulse
table needs the full content width and the adjustment plot is landscape. The
quantity-independent skeleton lives in :mod:`._layout`; this module only holds
the NT ACOU 112 impulse-prominence specifics. reportlab, matplotlib and svglib
are soft dependencies imported lazily (reportlab and svglib ship in the
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
    from ..environment.assessment.impulsive_sound import ImpulseProminenceResult

#: Minimum onset rate, in dB/s, for a level rise to qualify as an impulse
#: (NT ACOU 112:2002, clause 4.5). Kept here so the renderer does not import
#: the domain module at module level (the render-leaf import rule).
_ONSET_RATE_LIMIT = 10.0

#: Maximum number of impulse rows the per-impulse table shows. A large valid
#: impulse set is capped to the highest-prominence impulses (the governing one
#: always included) so the table cannot push the plot, result and footer off the
#: single A4 page; the dropped rows are reported in an explicit note row rather
#: than silently omitted.
_MAX_TABLE_ROWS = 12


def _fmt(value: float, language: str, decimals: int = 1) -> str:
    """A quantity rounded to ``decimals`` decimals, localised separator."""
    return format_number(float(value), language, decimals=decimals)


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _governing_index(per_impulse: np.ndarray, qualifies: np.ndarray) -> int:
    """Return the index of the governing impulse (the highest qualifying ``P``).

    The governing impulse is the one with the highest predicted prominence among
    those that qualify (onset rate above 10 dB/s, NT ACOU 112 clause 7); when no
    event qualifies the highest overall is used (informational only, matching the
    result's own governing value).
    """
    if np.any(qualifies):
        idx = np.nonzero(qualifies)[0]
        return int(idx[int(np.argmax(per_impulse[idx]))])
    return int(np.argmax(per_impulse))


def _select_rows(
    per_impulse: np.ndarray, governing: int
) -> tuple[list[int], int]:
    """Choose the impulse rows to show and how many are dropped.

    Every impulse is shown (in input order) when the set fits :data:`_MAX_TABLE_ROWS`;
    otherwise the highest-prominence impulses are kept, always including the
    governing impulse (which a non-qualifying event of higher raw prominence
    could otherwise displace), and the remaining count is returned so the fiche
    can state ``... plus N more`` rather than silently dropping rows.
    """
    n = per_impulse.size
    if n <= _MAX_TABLE_ROWS:
        return list(range(n)), 0
    order = np.argsort(per_impulse, kind="stable")[::-1]  # descending prominence
    top = [int(i) for i in order[:_MAX_TABLE_ROWS]]
    if governing not in top:
        top = top[: _MAX_TABLE_ROWS - 1] + [governing]
    return sorted(top), n - len(top)


def _qualifies_markup(qualifies: bool, language: str = "en") -> str:
    """Inline markup for the per-impulse qualifies column.

    An onset qualifies as an impulse when its onset rate exceeds 10 dB/s
    (NT ACOU 112 clause 4.5); such a row carries a filled accent dot and
    ``Yes``, a non-qualifying one a muted en dash.
    """
    if qualifies:
        return (
            f"<font color='{_VERDICT_OK_HEX}'>&#9679; "
            f"{t('Yes', language)}</font>"
        )
    return f"<font color='{_MUTED_HEX}'>&#8211;</font>"


def _metadata_pairs(
    result: ImpulseProminenceResult,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the assessment header grid.

    An impulsive-sound assessment identifies the source/situation whose noise was
    analysed (``specimen``), the client, the measurement position (``test_room``)
    and the instrumentation; only supplied fields are returned. The assessment
    period over which the governing prominence is taken (clause 8) is always
    shown, read from the result: the standard's default is 30 min (Clause 5),
    but another interval may be assessed, so the fiche prints the one analysed.
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
    period = format_number(
        float(result.assessment_period_min), language, decimals=1, trim=True
    )
    specs.append(
        (
            t("Assessment period", language),
            t("{minutes} min", language).format(minutes=period),
        )
    )
    return [(label, value) for label, value in specs if value]


def _per_impulse_table(
    result: ImpulseProminenceResult, language: str = "en"
) -> Any:
    """The full-width per-impulse table (onset rate, level difference, P).

    One row per candidate impulse: the onset rate (dB/s), the level difference
    (dB), the predicted prominence ``P`` (Formula 1) and whether the onset
    qualifies as an impulse (onset rate above 10 dB/s). The governing impulse's
    row is emphasised. A set larger than :data:`_MAX_TABLE_ROWS` is capped to the
    highest-prominence impulses (the governing one always kept) with a trailing
    ``... plus N more`` note row, so the fiche stays one page for any input.
    Called only after :func:`render_impulse_prominence_report` has imported
    reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso1996imp")

    orate = np.asarray(result.onset_rates, dtype=np.float64)
    ld = np.asarray(result.level_differences, dtype=np.float64)
    per = np.asarray(result.per_impulse, dtype=np.float64)
    qualifies = np.asarray(result.qualifies, dtype=bool)
    governing = _governing_index(per, qualifies)
    shown, dropped = _select_rows(per, governing)

    headers = [
        t("Impulse", language),
        "OR [dB/s]",
        "LD [dB]",
        "P",
        t("Onset &gt; 10 dB/s", language),
    ]
    widths = [22.0, 34.0, 34.0, 34.0, 50.0]

    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for i in shown:
        emph = "<b>{}</b>".format if i == governing else str
        data.append(
            [
                Paragraph(emph(str(i + 1)), label_style),
                Paragraph(_fmt(orate[i], language, decimals=0), value_style),
                Paragraph(_fmt(ld[i], language), value_style),
                Paragraph(emph(_fmt(per[i], language, decimals=2)), value_style),
                Paragraph(_qualifies_markup(bool(qualifies[i]), language), value_style),
            ]
        )
    if dropped:
        note = t(
            "&#8230; plus {n} more impulses of lower prominence", language
        ).format(n=dropped)
        data.append([Paragraph(note, label_style), "", "", "", ""])

    table = stacked_table(data, [w * mm for w in widths])
    dec_row = shown.index(governing) + 1  # +1 for the header row
    accent = colors.HexColor(_ACCENT_HEX)
    extra_style: list[Any] = [
        ("LINEBELOW", (0, dec_row), (-1, dec_row), 0.5, accent),
        ("LINEABOVE", (0, dec_row), (-1, dec_row), 0.5, accent),
    ]
    if dropped:
        note_row = len(data) - 1
        extra_style += [
            ("SPAN", (0, note_row), (-1, note_row)),
            ("ALIGN", (0, note_row), (-1, note_row), "LEFT"),
        ]
    table.setStyle(extra_style)
    return table


def _statement(
    result: ImpulseProminenceResult, language: str = "en"
) -> tuple[str, list[str]]:
    """The boxed governing prominence ``P`` and the derived adjustment ``KI``.

    The governing prominence is the highest ``P`` over the qualifying impulses
    (clause 7); the adjustment ``KI = 1.8 (P - 5)`` follows from Formula 2. The
    extended terms name the governing impulse's onset rate and level difference.
    """
    per = np.asarray(result.per_impulse, dtype=np.float64)
    qualifies = np.asarray(result.qualifies, dtype=bool)
    governing = _governing_index(per, qualifies)
    orate = np.asarray(result.onset_rates, dtype=np.float64)[governing]
    ld = np.asarray(result.level_differences, dtype=np.float64)[governing]
    statement = t(
        "Governing prominence P = <b>{p}</b> &nbsp; L<sub>Aeq</sub> adjustment "
        "K<sub>I</sub> = <b>{ki} dB</b>",
        language,
    ).format(
        p=_fmt(result.prominence, language, decimals=2),
        ki=_fmt(result.adjustment, language),
    )
    extended = [
        t("Governing impulse: OR = {orate} dB/s, LD = {ld} dB", language).format(
            orate=_fmt(orate, language, decimals=0),
            ld=_fmt(ld, language),
        )
    ]
    return statement, extended


def _verdict(
    result: ImpulseProminenceResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag against a supplied maximum governing prominence.

    The ``requirement`` is read as the maximum acceptable governing prominence
    ``P`` (a lower prominence is better, as a more prominent impulse is more
    intrusive); the assessment passes when the governing prominence is at or
    below it. The comparison is on the unrounded prominence so a value just
    above the limit (P = 10.004 against a 10.00 maximum) cannot pass on the
    display rounding; only the printed P is rounded for display.
    """
    passed = result.prominence <= requirement
    text = t("P = {p}, required &#8804; {req}", language).format(
        p=_fmt(display_round(result.prominence, 2), language, decimals=2),
        req=_fmt(requirement, language, decimals=2),
    )
    return text, passed


def _basis_line(measurement_standard: str | None, language: str = "en") -> str:
    """The standard-basis line, with the measurement standard when supplied."""
    if measurement_standard:
        return t(
            "{standard} assessment of impulsive-sound prominence and the "
            "L<sub>Aeq</sub> adjustment per NT ACOU 112:2002 (ISO/PAS "
            "1996-3:2022).",
            language,
        ).format(standard=html.escape(measurement_standard))
    return t(
        "Impulsive-sound prominence and the L<sub>Aeq</sub> adjustment per "
        "NT ACOU 112:2002 (ISO/PAS 1996-3:2022).",
        language,
    )


def _prominence_note(
    result: ImpulseProminenceResult, language: str = "en"
) -> str:
    """The prominence-category note: whether a prominent impulse is present.

    Two independent gates can withhold the adjustment, so the note states the
    one that actually governs. No level rise qualifying as an impulse (every
    onset rate at or below 10 dB/s, clauses 4.5/8) rules out an adjustment
    whatever the arithmetic prominence of the events is; only when an impulse
    does qualify does the ``P <= 5`` threshold of Formula 2 decide. Printing
    the ``P <= 5`` justification for a non-qualifying set would contradict the
    fiche's own governing P, which is then informational.
    """
    p = _fmt(result.prominence, language, decimals=2)
    if result.adjustment > 0.0:
        return t(
            "A prominent impulse is present (governing P = {p} &gt; 5); the "
            "L<sub>Aeq</sub> adjustment K<sub>I</sub> = {ki} dB (NT ACOU "
            "112:2002 Formula 2) applies.",
            language,
        ).format(p=p, ki=_fmt(result.adjustment, language))
    if not bool(np.any(np.asarray(result.qualifies, dtype=bool))):
        return t(
            "No level rise qualifies as an impulse (no onset rate above "
            "10 dB/s, NT ACOU 112:2002 clauses 4.5/8), so no adjustment "
            "applies (K<sub>I</sub> = 0 dB); the highest P = {p} is "
            "informational only.",
            language,
        ).format(p=p)
    return t(
        "No prominent impulse is present (governing P = {p} &#8804; 5); no "
        "adjustment applies (K<sub>I</sub> = 0 dB).",
        language,
    ).format(p=p)


def render_impulse_prominence_report(
    result: ImpulseProminenceResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an impulsive-sound prominence assessment fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.environment.assessment.impulsive_sound.ImpulseProminenceResult`
        carrying the per-impulse onset rates, level differences and prominences.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity (``specimen`` the source/situation, ``client``, ``test_room``
        the measurement position, ``instrumentation`` and ``test_date``) and the
        footer identity. A supplied ``requirement`` is read as the maximum
        acceptable governing prominence ``P``.
    :param verbose: Accepted for signature parity with the other fiches; the
        per-impulse table already shows every candidate, so it has no effect.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # every impulse is always shown; kept for signature parity
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Impulsive sound prominence assessment", language)
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(_basis_line(measurement_standard, language), basis_style),
    ]

    header_pairs = _metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph(t("Candidate impulses", language), caption_style))
    flow.append(_per_impulse_table(result, language))
    flow.append(Spacer(1, 8))

    # Full-width, landscape adjustment-curve plot KI(P) with the impulses marked
    # (the result's own self-scaling plot).
    flow.append(
        render_figure_drawing(
            result.plot, 174 * mm, y_top=None, figsize=(9.2, 3.1),
            language=language,
        )
    )
    flow.append(Spacer(1, 8))

    statement, extended = _statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))

    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))

    basis_strip_style = measurement_basis_style()
    flow.append(Paragraph(_prominence_note(result, language), basis_strip_style))
    flow.append(
        Paragraph(
            t(
                "The predicted prominence P = 3 lg(OR) + 2 lg(LD) combines the "
                "onset rate OR (dB/s) and the level difference LD (dB) of each "
                "impulse (NT ACOU 112:2002, clause 7, Formula 1); the governing "
                "value is the highest P over the assessment period among "
                "impulses with an onset rate above 10 dB/s (clauses 4.5/8).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.append(
        Paragraph(
            t(
                "The L<sub>Aeq</sub> adjustment K<sub>I</sub> = 1.8 (P &#8722; "
                "5) dB for P &gt; 5, else 0 dB, is applied to the "
                "L<sub>Aeq</sub> of the assessment period on the basis of the "
                "single impulse with the highest prominence (clause 8, "
                "Formula 2).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
