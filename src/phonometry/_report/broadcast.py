#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EBU R 128 programme-loudness compliance fiche (reportlab renderer).

Renders a
:class:`~phonometry.broadcast.program_loudness.ProgramLoudnessResult` to a
one-page PDF laid out like a broadcast loudness-compliance sheet:

* a title and the standard-basis line (measurement standard + EBU R 128 /
  ITU-R BS.1770-5);
* an optional metadata header block (client, programme, laboratory ...),
  rendered only for the fields supplied on the :class:`ReportMetadata`;
* a full-width compliance table (Metric | Measured | Target / Limit | Result)
  whose verdict is driven only by the integrated loudness and the maximum true
  peak; the loudness range and the momentary/short-term maxima are shown as
  informational rows;
* a full-width, landscape loudness-vs-time plot drawn by the result's own
  ``plot(ax=...)`` (momentary and short-term loudness, the integrated line and
  the LRA band);
* a boxed single-number result ``I = X LUFS (LRA = Y LU, max TP = Z dBTP)``;
* a combined PASS/FAIL verdict row and a short measurement-basis strip; and
* a footer identity/disclaimer block.

Unlike the two-panel non-band fiche (:mod:`.iso532`) this uses a stacked
layout: the four-column compliance table needs the full content width and the
loudness-vs-time trace is landscape. The quantity-independent skeleton lives in
:mod:`._layout`; this module only holds the EBU R 128 specifics. reportlab,
matplotlib and svglib are soft dependencies imported lazily (reportlab and
svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import decimal_comma, format_number, t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    build_document,
    compliance_table,
    display_round,
    document_styles,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    verdict_flow,
)

if TYPE_CHECKING:
    from ..broadcast.program_loudness import ProgramLoudnessResult
    from .metadata import ReportMetadata

#: EBU R 128 target programme loudness, LUFS.
_DEFAULT_TARGET_LUFS = -23.0

#: Programme-loudness tolerance about the target, in LU, per compliance rule
#: (EBU R 128 v4/2020, unchanged in the 2023 revision): ``"qc"`` is the
#: +-0.2 LU allowance of item i) for measurement errors in loudness workflows
#: such as Quality Control; ``"live"`` is the +-1.0 LU tolerance of item h),
#: permitted only where attaining the Target Level is not achievable
#: practically (for example, live programmes). The pre-2020 blanket +-0.5 LU
#: (the June 2014 V3 rule) no longer exists in R 128.
_TOLERANCES_LU: dict[str, float] = {"qc": 0.2, "live": 1.0}

#: The R 128 recommendation item each tolerance rule cites on the fiche.
_TOLERANCE_CLAUSES: dict[str, str] = {"qc": "item i", "live": "item h"}

#: EBU R 128 maximum permitted true-peak level, dBTP (item m).
_MAX_TRUE_PEAK_DBTP = -1.0


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the loudness header grid.

    Only fields that are set are returned. Programme loudness is a signal
    metric, so the room/climate fields of the insulation fiche do not apply;
    the specimen field labels the tested programme.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Programme", language), metadata.specimen),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Test room", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
    ]
    return [
        (label, html.escape(str(value))) for label, value in specs if value is not None
    ]


def _status(
    result: ProgramLoudnessResult, target: float, tolerance_lu: float
) -> tuple[str, str, bool]:
    """The integrated-loudness and true-peak pass states, and their conjunction.

    Both the compliance-table rows and the combined verdict compare against the
    same two thresholds (integrated loudness within ``target`` &#177;
    ``tolerance_lu``; true peak at or below :data:`_MAX_TRUE_PEAK_DBTP`), so
    the comparison is derived once here and reused, keeping the two views in
    lockstep. The loudness comparison is evaluated on the value rounded to
    0.1 LU exactly as the fiche displays it (EBU Tech 3341 section 2 requires
    a display precision of at most one decimal place), so the printed numbers
    can never contradict the verdict at the tolerance boundary. The true peak
    is compared unrounded: item m) is an absolute production ceiling and the
    strict reading is the conservative one.

    :return: ``(i_status, tp_status, passed)`` where each status is ``"pass"``
        or ``"fail"`` and ``passed`` is the conjunction (a programme complies
        only when both pass).
    """
    delta = display_round(float(result.integrated)) - display_round(target)
    i_pass = abs(delta) <= tolerance_lu + 1e-9
    tp_pass = float(result.true_peak) <= _MAX_TRUE_PEAK_DBTP
    return (
        "pass" if i_pass else "fail",
        "pass" if tp_pass else "fail",
        i_pass and tp_pass,
    )


def _compliance_rows(
    result: ProgramLoudnessResult,
    target: float,
    tolerance_lu: float,
    language: str = "en",
) -> list[tuple[str, str, str, str]]:
    """Build the compliance-table rows for the EBU R 128 fiche.

    The verdict is carried only by the integrated loudness and the maximum
    true peak; the loudness range and the momentary/short-term maxima are
    informational (status ``"info"``, no pass/fail). The integrated loudness
    and its delta are displayed from the same 0.1 LU rounding that
    :func:`_status` evaluates, so the printed numbers and the verdict agree.
    """
    integrated = display_round(float(result.integrated))
    true_peak = float(result.true_peak)
    delta = integrated - display_round(target)
    i_status, tp_status, _ = _status(result, target, tolerance_lu)
    tol = decimal_comma(f"{tolerance_lu:g}", language)
    informational = t("informational", language)
    return [
        (
            t("Integrated (Programme) Loudness", language),
            f"{format_number(integrated, language, decimals=1)} LUFS",
            t("{target} LUFS &#177;{tol} LU (&#916; {delta} LU)", language).format(
                target=format_number(target, language, decimals=1),
                tol=tol,
                delta=decimal_comma(f"{delta:+.1f}", language),
            ),
            i_status,
        ),
        (
            t("Maximum True Peak", language),
            f"{format_number(true_peak, language, decimals=1)} dBTP",
            t("&#8804; {limit} dBTP", language).format(
                limit=format_number(_MAX_TRUE_PEAK_DBTP, language, decimals=1)
            ),
            tp_status,
        ),
        (
            t("Loudness Range (LRA)", language),
            f"{format_number(float(result.loudness_range), language, decimals=1)} LU",
            informational,
            "info",
        ),
        (
            t("Max Momentary", language),
            f"{format_number(float(result.max_momentary), language, decimals=1)} LUFS",
            informational,
            "info",
        ),
        (
            t("Max Short-term", language),
            f"{format_number(float(result.max_short_term), language, decimals=1)} LUFS",
            informational,
            "info",
        ),
    ]


def _statement(result: ProgramLoudnessResult, language: str = "en") -> str:
    """The boxed single-number statement ``I = X LUFS (LRA = Y, max TP = Z)``."""
    integrated = format_number(float(result.integrated), language, decimals=1)
    lra = format_number(float(result.loudness_range), language, decimals=1)
    tp = format_number(float(result.true_peak), language, decimals=1)
    return f"I = <b>{integrated} LUFS</b> &nbsp; (LRA = {lra} LU, max TP = {tp} dBTP)"


def _verdict(
    result: ProgramLoudnessResult,
    target: float,
    tolerance: str = "qc",
    language: str = "en",
) -> tuple[str, bool]:
    """Combined verdict text and PASS flag (integrated loudness and true peak).

    A programme complies when the integrated loudness is within the selected
    R 128 tolerance about the target (``"qc"``: +-0.2 LU per item i;
    ``"live"``: +-1.0 LU per item h) and the true peak is at or below the
    permitted ceiling. The verdict sentence cites the applied R 128 item.
    """
    tolerance_lu = _TOLERANCES_LU[tolerance]
    _i_status, _tp_status, passed = _status(result, target, tolerance_lu)
    text = t(
        "Compliant when I is within {target} LUFS &#177;{tol} LU (EBU R 128 {clause}) and true peak &#8804; {tp} dBTP",
        language,
    ).format(
        target=format_number(target, language, decimals=1),
        tol=format_number(tolerance_lu, language, decimals=1),
        clause=_TOLERANCE_CLAUSES[tolerance],
        tp=format_number(_MAX_TRUE_PEAK_DBTP, language, decimals=1),
    )
    return text, passed


def render_program_loudness_report(
    result: ProgramLoudnessResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
    tolerance: str = "qc",
) -> str:
    """Render an EBU R 128 programme-loudness fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.broadcast.program_loudness.ProgramLoudnessResult`
        carrying the integrated loudness, loudness range, true peak and the
        momentary/short-term loudness series.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        measurement fiche (compliance table + plot + verdict, no header). A
        supplied ``requirement`` is read as the target programme loudness in
        LUFS (defaulting to the EBU R 128 -23.0 LUFS).
    :param verbose: Accepted for a uniform ``.report()`` signature; the
        programme-loudness fiche has a single body layout, so it has no effect.
    :param tolerance: Programme-loudness tolerance rule of EBU R 128:
        ``"qc"`` (default) applies the +-0.2 LU measurement-error allowance of
        item i) (loudness workflows such as Quality Control); ``"live"``
        applies the +-1.0 LU tolerance of item h), permitted only where the
        Target Level is not achievable practically (for example, live
        programmes). The applied rule and its R 128 item are printed on the
        fiche.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``tolerance`` is not ``"qc"`` or ``"live"``.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # uniform signature; the fiche has one stacked body layout
    if tolerance not in _TOLERANCES_LU:
        msg = (
            f"Unknown tolerance rule {tolerance!r}; use 'qc' (+-0.2 LU, EBU "
            "R 128 item i) or 'live' (+-1.0 LU, item h)."
        )
        raise ValueError(msg)
    tolerance_lu = _TOLERANCES_LU[tolerance]
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Programme loudness compliance", language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t(
            "{standard} programme loudness. Rating per EBU R 128 / ITU-R BS.1770-5 (K-weighting, gated).",
            language,
        ).format(standard=html.escape(measurement_standard))
    else:
        basis = t(
            "Programme loudness per EBU R 128 / ITU-R BS.1770-5 (K-weighting, gated).",
            language,
        )

    target = (
        float(metadata.requirement)
        if metadata is not None and metadata.requirement is not None
        else _DEFAULT_TARGET_LUFS
    )

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

    flow.append(fiche_paragraph(t("Compliance summary", language), caption_style))
    flow.append(
        compliance_table(
            _compliance_rows(result, target, tolerance_lu, language),
            language=language,
        )
    )
    flow.append(Spacer(1, 8))

    # Full-width, landscape loudness-vs-time plot (self-scaling axis).
    plot_drawing = render_figure_drawing(
        result.plot, 174 * mm, y_top=None, figsize=(9.2, 4.2), language=language
    )
    flow.append(plot_drawing)
    flow.append(Spacer(1, 8))

    flow.append(result_box(_statement(result, language), styles, accent))
    text, passed = _verdict(result, target, tolerance, language)
    flow.extend(verdict_flow(text, passed, styles, language))

    basis_strip_style = ParagraphStyle(
        "fiche_measurement_basis",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=6,
    )
    tolerance_note = (
        t(
            "Loudness tolerance &#177;0.2 LU for measurement errors in loudness workflows, e.g. Quality Control (EBU R 128 item i).",
            language,
        )
        if tolerance == "qc"
        else t(
            "Loudness tolerance &#177;1.0 LU, permitted where the Target Level is not achievable practically, e.g. live programmes (EBU R 128 item h).",
            language,
        )
    )
    flow.append(fiche_paragraph(tolerance_note, basis_strip_style))
    flow.append(
        fiche_paragraph(
            t(
                "Gating -70 LUFS absolute / -10 LU relative (ITU-R BS.1770); 1 LU = 1 dB; true peak per EBU Tech 3341; LRA per EBU Tech 3342 (not recommended for programmes under 60 s).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
